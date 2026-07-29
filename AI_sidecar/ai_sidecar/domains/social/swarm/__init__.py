"""Swarm intelligence system — multi-bot coordination, formations, consensus, and tactics.

The SwarmCoordinator ties together:
  - communication: inter-bot messaging via shared state files
  - formation: party positioning patterns
  - consensus: group decision-making (map voting, retreat, targets)
  - tactics: group combat coordination

Usage:
    from ai_sidecar.domains.social.swarm import SwarmCoordinator

    coordinator = SwarmCoordinator(data_dir="data/swarm")
    coordinator.publish_my_state("kicapmasin", signals)
    coordinator.tick("kicapmasin", signals)
    actions = coordinator.get_actions()
"""

from __future__ import annotations

import logging
import time
from threading import RLock
from typing import Any

from ai_sidecar.actions import HeuristicAction
from ai_sidecar.domains.social.swarm.communication import (
    BotSwarmState,
    SwarmDecision,
    SwarmFileStore,
)
from ai_sidecar.domains.social.swarm.consensus import (
    ConsensusEngine,
    ConsensusResult,
)
from ai_sidecar.domains.social.swarm.formation import (
    FormationManager,
    FormationType,
)
from ai_sidecar.domains.social.swarm.tactics import (
    SwarmTactics,
    TacticalSituation,
    TacticType,
)

logger = logging.getLogger(__name__)


class SwarmCoordinator:
    """Master coordinator for the swarm intelligence system.

    Runs on every bot. Each bot:
      1. Publishes its state to data/swarm_state_{bot_name}.json
      2. Reads all other bots' states
      3. If leader: makes decisions, writes to data/swarm_decision.json
      4. If follower: reads and follows leader's decisions
      5. Generates HeuristicActions based on decisions
    """

    def __init__(
        self,
        data_dir: str = "data/swarm",
        bot_names: list[str] | None = None,
        my_name: str = "",
        my_role: str = "idle",
    ) -> None:
        self._store = SwarmFileStore(data_dir)
        self._formation = FormationManager()
        self._consensus = ConsensusEngine(default_threshold=0.66)
        self._tactics = SwarmTactics()

        self._my_name = my_name
        self._my_role = my_role
        self._bot_names: list[str] = bot_names or []
        self._lock = RLock()

        # Cached state
        self._my_state: BotSwarmState | None = None
        self._all_states: dict[str, BotSwarmState] = {}
        self._current_decision: SwarmDecision | None = None
        self._last_decision_version: int = 0
        self._actions: list[HeuristicAction] = []

        # Cooldowns
        self._last_state_publish: float = 0
        self._last_leader_tick: float = 0
        self._publish_interval: float = 5.0   # Publish state every 5s
        self._leader_tick_interval: float = 10.0  # Leader makes decisions every 10s

        # Party settings
        self._party_auto_share: bool = True
        self._member_range: int = 15
        self._party_composition_aware: bool = True

        logger.info(
            "SwarmCoordinator initialized for %s (role=%s) with %d bots",
            my_name, my_role, len(bot_names or []),
        )

    # ── Configuration ───────────────────────────────────────────

    @property
    def party_auto_share(self) -> bool:
        """Whether experience sharing is enabled."""
        return self._party_auto_share

    @party_auto_share.setter
    def party_auto_share(self, value: bool) -> None:
        self._party_auto_share = value
        logger.info("Party auto-share set to %s", value)

    @property
    def member_range(self) -> int:
        """Maximum distance between party members for shared exp."""
        return self._member_range

    @member_range.setter
    def member_range(self, cells: int) -> None:
        self._member_range = max(5, min(30, cells))

    # ── Core cycle ──────────────────────────────────────────────

    def publish_my_state(self, bot_name: str, signals: dict[str, Any]) -> BotSwarmState:
        """Collect this bot's state from signals and publish to swarm file.

        Should be called every PDCA cycle.
        """
        self._my_name = bot_name
        state = self._store.collect_bot_state_for_leader(bot_name, signals)
        self._my_state = state

        now = time.time()
        if now - self._last_state_publish >= self._publish_interval:
            self._store.write_bot_state(state)
            self._last_state_publish = now
            logger.debug("Published swarm state for %s", bot_name)

        return state

    def tick(
        self,
        bot_name: str,
        signals: dict[str, Any],
    ) -> list[HeuristicAction]:
        """Run one coordination cycle.

        This is the main entry point called from PDCA or domain assess().

        Args:
            bot_name: This bot's name.
            signals: Bridge snapshot signals dict.

        Returns:
            List of HeuristicActions for this bot.
        """
        with self._lock:
            self._actions.clear()
            self._my_name = bot_name

            # 1. Publish my state
            self.publish_my_state(bot_name, signals)

            # 2. Read all bot states
            self._all_states = self._store.read_all_bot_states()

            if not self._all_states:
                logger.debug("No other bot states found yet")
                return list(self._actions)

            # 3. Determine leadership
            leader_name = self._store.get_leader_name()
            am_leader = leader_name == bot_name

            if am_leader:
                # 4a. Leader: make decisions
                self._leader_tick(bot_name, signals)
            else:
                # 4b. Follower: read and follow leader's decision
                self._follower_tick(bot_name)

            return list(self._actions)

    def get_actions(self) -> list[HeuristicAction]:
        """Return the current batch of actions."""
        with self._lock:
            return list(self._actions)

    def get_decision(self) -> SwarmDecision | None:
        """Return the current swarm decision."""
        with self._lock:
            return self._current_decision

    def get_all_states(self) -> dict[str, BotSwarmState]:
        """Return all collected bot states."""
        with self._lock:
            return dict(self._all_states)

    # ── Leader logic ────────────────────────────────────────────

    def _leader_tick(self, bot_name: str, signals: dict[str, Any]) -> None:
        """Party leader makes swarm decisions."""
        now = time.time()
        if now - self._last_leader_tick < self._leader_tick_interval:
            # Use cached decision
            self._process_decision(bot_name)
            return

        self._last_leader_tick = now

        if not self._all_states:
            return

        # Build state references for typed methods
        typed_states: dict[str, BotSwarmState] = {
            n: s for n, s in self._all_states.items()
        }

        # ── Composite snapshots ──
        all_bot_roles = [s.role for s in typed_states.values()]
        party_hp_avg = sum(s.hp_pct for s in typed_states.values()) / max(1, len(typed_states))
        bot_count = len(typed_states)

        # Current map (from leader's perspective)
        current_map = signals.get("map", "")

        # ── 1. Analyze party composition ──
        composition = self._consensus.analyze_composition(typed_states)
        acolytes = composition.get("acolytes", [])
        missing_roles = composition.get("missing_roles", [])

        # ── 2. Decide on formation ──
        threat_level = float(signals.get("threat_level", 0.0))
        aoe_risk = bool(signals.get("aoe_risk", False))
        target_count = int(signals.get("target_count", 1))

        formation = self._formation.select_formation_for_situation(
            threat_level=threat_level,
            team_hp_avg=party_hp_avg,
            target_count=target_count,
            aoe_risk=aoe_risk,
            bot_count=bot_count,
            roles=all_bot_roles,
        )

        # Anchor position (leader's position)
        anchor_x = int(signals.get("x", 0))
        anchor_y = int(signals.get("y", 0))

        # Assign formation positions
        bot_slots = [
            {"name": n, "role": s.role}
            for n, s in typed_states.items()
        ]
        formation_positions = self._formation.assign_positions(
            bot_slots, formation, anchor_x, anchor_y,
        )

        # ── 3. Hunt map consensus ──
        available_maps = signals.get("available_hunt_maps", []) or []
        hunt_map_result = self._consensus.decide_hunt_map(typed_states, available_maps)
        hunt_map = str(hunt_map_result.decision) if hunt_map_result.decision else current_map

        # ── 4. Retreat decision ──
        retreat_result = self._consensus.decide_retreat(typed_states, party_hp_avg)
        should_retreat = bool(retreat_result.decision)

        # ── 5. Target selection ──
        monsters: list[dict[str, Any]] = list(signals.get("monsters", []) or [])
        target_result = self._consensus.decide_hunt_target(
            typed_states, monsters,
            leader_level=int(signals.get("base_level", 1)),
        )
        target_monster = str(target_result.decision)

        # ── 6. Tactics ──
        tactical_situation = self._tactics.assess_situation(
            typed_states, monsters, bot_name,
        )
        tactical_situation.retreat_needed = should_retreat
        tactic = self._tactics.select_tactic(tactical_situation, bot_count)

        # ── 7. Migration order ──
        migration_order: list[str] = []
        if hunt_map and hunt_map != current_map:
            # Bots not yet on hunt map should migrate
            migration_result = self._consensus.decide_migration(
                typed_states, current_map, hunt_map,
            )
            if bool(migration_result.decision):
                # Order: support/healer first, then DPS, tank last
                migration_order = self._order_migration(typed_states)

        # ── 8. Buff targets (acolyte auto-buffs) ──
        buff_targets: list[str] = []
        if acolytes and self._party_composition_aware:
            for name, state in typed_states.items():
                if not state.has_blessing or not state.has_agi:
                    buff_targets.append(name)

        # ── Build decision ──
        decision = SwarmDecision(
            leader_name=bot_name,
            formation=str(formation),
            hunt_map=hunt_map,
            target_monster=target_monster,
            retreat=should_retreat,
            focus_fire_monster_id=target_monster if tactic == TacticType.FOCUS_FIRE else "",
            spread_targets=(tactic == TacticType.SPREAD_TARGETS),
            kite_mode=(tactic == TacticType.KITE),
            formation_positions={
                n: {"x": p["x"], "y": p["y"]}
                for n, p in formation_positions.items()
            },
            member_range=self._member_range,
            party_auto_share=self._party_auto_share,
            acolyte_buffs=bool(acolytes),
            buff_targets=buff_targets,
            migration_order=migration_order,
            reason=(
                f"Formation={formation}, Hunt={hunt_map}, "
                f"Retreat={should_retreat}, Tactic={tactic}, "
                f"Composition={'+'.join(missing_roles) if missing_roles else 'balanced'}"
            ),
        )

        # Write decision
        self._store.write_decision(decision)
        self._current_decision = decision
        self._last_decision_version = decision.version
        logger.info(
            "Swarm decision: %s | formation=%s hunt=%s retreat=%s tactic=%s | %s",
            decision.leader_name, formation, hunt_map, should_retreat, tactic,
            decision.reason,
        )

        # Process decision for leader
        self._process_decision(bot_name)

    def _follower_tick(self, bot_name: str) -> None:
        """Non-leader bot reads and follows the leader's decision."""
        decision = self._store.read_decision()
        if decision is None:
            logger.debug("No swarm decision available yet")
            return

        if decision.version <= self._last_decision_version:
            return  # Already processed this decision

        self._current_decision = decision
        self._last_decision_version = decision.version
        logger.debug(
            "Following decision v%d from %s: %s",
            decision.version, decision.leader_name, decision.reason,
        )

        self._process_decision(bot_name)

    # ── Action generation ───────────────────────────────────────

    def _process_decision(self, bot_name: str) -> None:
        """Process the current decision and generate actions for this bot."""
        decision = self._current_decision
        if decision is None:
            return

        # ── Party sharing commands ──
        if decision.party_auto_share:
            self._actions.append(HeuristicAction(
                kind="command",
                command="party share exp",
                confidence=0.90,
                domain="swarm",
                reason="[SWARM] Enable party experience sharing",
            ))

        # ── Acolyte auto-buffs ──
        if decision.acolyte_buffs and bot_name in decision.buff_targets:
            # This bot needs buffs — request them
            pass  # The acolyte bot will generate buff commands below

        # ── Migration ──
        if decision.migration_order and bot_name in decision.migration_order:
            target_map = decision.hunt_map
            self._actions.append(HeuristicAction(
                kind="command",
                command=f"move {target_map}",
                confidence=0.85,
                domain="swarm",
                reason=f"[SWARM] Migrate to {target_map} per leader order",
            ))

        # ── Formation position ──
        my_pos = decision.formation_positions.get(bot_name)
        if my_pos:
            self._actions.append(HeuristicAction(
                kind="command",
                command=f"move {my_pos['x']} {my_pos['y']}",
                confidence=0.80,
                domain="swarm",
                reason=f"[SWARM] Formation: move to ({my_pos['x']}, {my_pos['y']})",
            ))

        # ── Retreat ──
        if decision.retreat:
            self._actions.append(HeuristicAction(
                kind="command",
                command="ai manual",
                confidence=0.95,
                domain="swarm",
                reason="[SWARM] Retreat signal active — disengage",
            ))

        # ── Combat tactics ──
        if decision.focus_fire_monster_id:
            self._actions.append(HeuristicAction(
                kind="command",
                command=f"attack {decision.focus_fire_monster_id}",
                confidence=0.90,
                domain="swarm",
                reason=f"[SWARM] Focus fire on {decision.focus_fire_monster_id}",
            ))

        if decision.kite_mode:
            self._actions.append(HeuristicAction(
                kind="command",
                command="ai auto",
                confidence=0.75,
                domain="swarm",
                reason="[SWARM] Kite mode active",
            ))

        if decision.member_range:
            self._actions.append(HeuristicAction(
                kind="command",
                command=f"set partyRange {decision.member_range}",
                confidence=0.80,
                domain="swarm",
                reason=f"[SWARM] Set party range to {decision.member_range} cells",
            ))

    # ── Party management helpers ────────────────────────────────

    def generate_party_actions(
        self,
        bot_name: str,
        signals: dict[str, Any],
    ) -> list[HeuristicAction]:
        """Generate party management actions.

        Handles:
          - Party creation (leader)
          - Party sharing settings
          - Member range enforcement
          - Acolyte auto-buffs for party members
        """
        actions: list[HeuristicAction] = []
        is_leader = bool(signals.get("is_leader", False))
        in_party = bool(signals.get("in_party", False))
        all_bots: list[str] = list(signals.get("all_bots", []) or [])
        party_members: list[str] = list(signals.get("party_members", []) or [])

        if not all_bots:
            return actions

        # Get my state for role/skills
        my_state = self._my_state

        # ── Leader: ensure party exists ──
        if is_leader and not in_party:
            ts = int(time.time())
            actions.append(HeuristicAction(
                kind="command",
                command=f"party create SWARM{ts}",
                confidence=0.95,
                domain="swarm",
                reason="[SWARM] Leader creates party",
            ))

            # Request each bot to join
            sorted_bots = sorted(all_bots)
            for other in sorted_bots:
                if other != bot_name and other not in party_members:
                    actions.append(HeuristicAction(
                        kind="command",
                        command=f"party request {other}",
                        confidence=0.90,
                        domain="swarm",
                        reason=f"[SWARM] Request {other} to join",
                    ))

        # ── Non-leader: ensure partyAuto is on ──
        if not is_leader:
            actions.append(HeuristicAction(
                kind="command",
                command="set partyAuto 2",
                confidence=0.90,
                domain="swarm",
                reason="[SWARM] Enable auto-accept party invites",
            ))

        # ── Party share settings ──
        if is_leader and self._party_auto_share:
            actions.append(HeuristicAction(
                kind="command",
                command="party share exp",
                confidence=0.90,
                domain="swarm",
                reason="[SWARM] Share experience in party",
            ))

        # ── Party member range ──
        if self._member_range:
            actions.append(HeuristicAction(
                kind="command",
                command=f"set partyRange {self._member_range}",
                confidence=0.80,
                domain="swarm",
                reason=f"[SWARM] Set party range to {self._member_range} cells",
            ))

        # ── Acolyte auto-buffs ──
        if my_state and my_state.acolyte_can_buff and party_members:
            for member in party_members:
                if member != bot_name:
                    # Blessing
                    actions.append(HeuristicAction(
                        kind="command",
                        command=f"party skill Blessing {member}",
                        confidence=0.85,
                        domain="swarm",
                        reason=f"[SWARM] Acolyte buff: Blessing -> {member}",
                    ))
                    # Increase AGI
                    actions.append(HeuristicAction(
                        kind="command",
                        command=f"party skill Increase AGI {member}",
                        confidence=0.85,
                        domain="swarm",
                        reason=f"[SWARM] Acolyte buff: Increase AGI -> {member}",
                    ))

        return actions

    def generate_acolyte_buff_actions(
        self,
        bot_name: str,
        party_members: list[str],
    ) -> list[HeuristicAction]:
        """Generate buff actions for an acolyte-type bot.

        Only bots with Blessing and Increase AGI skills should
        call this. It generates commands to buff all party members.
        """
        actions: list[HeuristicAction] = []
        for member in party_members:
            if member == bot_name:
                continue
            actions.append(HeuristicAction(
                kind="command",
                command=f"party skill Blessing {member}",
                confidence=0.85,
                domain="swarm",
                reason=f"[SWARM] Buff party: Blessing -> {member}",
            ))
            actions.append(HeuristicAction(
                kind="command",
                command=f"party skill Increase AGI {member}",
                confidence=0.85,
                domain="swarm",
                reason=f"[SWARM] Buff party: Increase AGI -> {member}",
            ))
        return actions

    # ── Utility ─────────────────────────────────────────────────

    def _order_migration(self, states: dict[str, BotSwarmState]) -> list[str]:
        """Order bots for migration: squishy first, tank last."""
        squishy_roles = {"healer", "support", "buffer", "dps_magic", "dps_ranged"}
        tank_roles = {"tank", "dps_melee"}

        first = [n for n, s in states.items() if s.role in squishy_roles]
        middle = [n for n, s in states.items() if s.role not in squishy_roles and s.role not in tank_roles]
        last = [n for n, s in states.items() if s.role in tank_roles]

        return first + middle + last

    def is_leader(self, bot_name: str | None = None) -> bool:
        """Check if the given bot (or this bot) is the swarm leader."""
        name = bot_name or self._my_name
        if not name:
            return False
        leader = self._store.get_leader_name()
        return leader == name

    def get_formation_positions(self) -> dict[str, dict[str, int]]:
        """Get the current formation positions from the latest decision."""
        if self._current_decision:
            return dict(self._current_decision.formation_positions)
        return {}

    def clear_state(self) -> None:
        """Clear all swarm state files."""
        self._store.clear_decision()
        self._current_decision = None
        self._last_decision_version = 0
        self._actions.clear()
        self._all_states.clear()
        logger.info("Swarm state cleared")
