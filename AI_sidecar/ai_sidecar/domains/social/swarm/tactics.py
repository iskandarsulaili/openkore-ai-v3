"""Swarm tactics — group combat coordination.

Provides group combat tactics including:
  - Focus fire: all bots attack the same target
  - Spread targets: each bot attacks a different monster
  - Kite: ranged bots keep distance, melee tanks
  - Coordinated retreat: all bots fall back together
  - Aggro management: taunt rotation, threat control

Integrates with the existing fleet/swarm_ai.py SwarmTacticsEngine
for formation-based positioning during combat.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
#  Tactic types
# ────────────────────────────────────────────────────────────────

class TacticType(StrEnum):
    """Combat tactics the swarm can employ."""
    FOCUS_FIRE = "focus_fire"         # All bots attack one target
    SPREAD_TARGETS = "spread_targets"  # Each bot picks a different target
    KITE = "kite"                      # Ranged kites, melee intercepts
    PINCER = "pincer"                  # Two sides converge
    WITHDRAW = "withdraw"             # Ordered disengagement
    AMBUSH = "ambush"                 # Wait for target to approach
    DEFENSIVE = "defensive"           # Stay close, protect each other
    CHARGE = "charge"                 # All-out rush
    HEAL_ROTATION = "heal_rotation"   # Coordinated healing


# ────────────────────────────────────────────────────────────────
#  Threat assessment
# ────────────────────────────────────────────────────────────────

@dataclass
class ThreatAssessment:
    """Assessment of a single monster's threat level."""
    monster_id: str = ""
    monster_name: str = ""
    level: int = 1
    hp: int = 1
    hp_max: int = 1
    distance: float = 999.0
    damage_taken: float = 0.0     # How much damage this bot has taken from it
    is_attacking_me: bool = False
    is_attacking_teammate: bool = False
    is_mvp: bool = False
    threat_score: float = 0.0     # Computed threat level
    element: str = "neutral"
    size: str = "medium"


@dataclass
class TacticalSituation:
    """Summary of the current tactical situation."""
    party_hp_avg: float = 1.0
    party_sp_avg: float = 1.0
    enemy_count: int = 0
    highest_threat: float = 0.0
    has_mvp: bool = False
    aoe_risk: bool = False
    formation_type: str = "vanguard"
    tactic: TacticType = TacticType.FOCUS_FIRE
    retreat_needed: bool = False
    distance_to_enemy: float = 999.0
    allies_nearby: int = 0
    timestamp: float = field(default_factory=time.time)


# ────────────────────────────────────────────────────────────────
#  SwarmTactics
# ────────────────────────────────────────────────────────────────

class SwarmTactics:
    """Tactical combat coordinator for the swarm.

    Decides which tactic to use and generates HeuristicAction commands
    for each bot based on the current situation.
    """

    def __init__(self) -> None:
        self._current_tactic: TacticType = TacticType.FOCUS_FIRE
        self._focus_target: str = ""
        self._last_tactic_switch: float = 0
        self._tactic_cooldown: float = 5.0  # Minimum seconds between tactic changes
        self._engagement_range: int = 14    # Cells within which bots engage

    # ── Tactic selection ───────────────────────────────────────

    def select_tactic(
        self,
        situation: TacticalSituation,
        party_size: int,
    ) -> TacticType:
        """Select the best tactic for the current situation."""
        now = time.time()
        if now - self._last_tactic_switch < self._tactic_cooldown:
            return self._current_tactic

        # Retreat if critical
        if situation.retreat_needed:
            return TacticType.WITHDRAW

        # MVP: focus fire
        if situation.has_mvp:
            return TacticType.FOCUS_FIRE

        # Outnumbered: defensive
        if situation.enemy_count > party_size * 2:
            return TacticType.DEFENSIVE

        # High AoE risk: spread targets
        if situation.aoe_risk and situation.enemy_count >= 2:
            return TacticType.SPREAD_TARGETS

        # Single enemy: focus fire
        if situation.enemy_count <= 1:
            return TacticType.FOCUS_FIRE

        # Many enemies, good party: charge
        if situation.enemy_count >= 3 and party_size >= 3 and situation.party_hp_avg > 0.7:
            return TacticType.CHARGE

        # Default: spread targets for efficiency
        if situation.enemy_count >= 2:
            return TacticType.SPREAD_TARGETS

        return TacticType.FOCUS_FIRE

    # ── Command generation per tactic ──────────────────────────

    def generate_actions(
        self,
        tactic: TacticType,
        bot_name: str,
        bot_role: str,
        situation: TacticalSituation,
        monsters: list[dict[str, Any]],
        focus_target_id: str = "",
    ) -> list[HeuristicAction]:
        """Generate HeuristicAction commands for a bot based on the tactic.

        Args:
            tactic: The current tactic.
            bot_name: This bot's name.
            bot_role: This bot's role.
            situation: Current tactical situation.
            monsters: List of visible monsters.
            focus_target_id: Monster ID for focus fire.

        Returns:
            List of HeuristicActions to execute.
        """
        self._current_tactic = tactic
        self._last_tactic_switch = time.time()
        self._focus_target = focus_target_id

        if not monsters:
            return []

        tactic_map = {
            TacticType.FOCUS_FIRE: self._focus_fire_actions,
            TacticType.SPREAD_TARGETS: self._spread_targets_actions,
            TacticType.KITE: self._kite_actions,
            TacticType.PINCER: self._pincer_actions,
            TacticType.WITHDRAW: self._withdraw_actions,
            TacticType.AMBUSH: self._ambush_actions,
            TacticType.DEFENSIVE: self._defensive_actions,
            TacticType.CHARGE: self._charge_actions,
            TacticType.HEAL_ROTATION: self._heal_rotation_actions,
        }

        handler = tactic_map.get(tactic, self._focus_fire_actions)
        return handler(bot_name, bot_role, situation, monsters, focus_target_id)

    def _focus_fire_actions(
        self,
        bot_name: str,
        bot_role: str,
        situation: TacticalSituation,
        monsters: list[dict[str, Any]],
        focus_target_id: str,
    ) -> list[HeuristicAction]:
        """All bots attack the same target."""
        actions: list[HeuristicAction] = []

        if focus_target_id:
            actions.append(HeuristicAction(
                kind="command",
                command=f"attack {focus_target_id}",
                confidence=0.95,
                domain="swarm",
                reason=f"[SWARM] Focus fire on {focus_target_id}",
            ))
        elif monsters:
            # Attack the closest/threatening monster
            target = self._pick_priority_target(monsters, bot_role)
            target_id = target.get("id", target.get("name", ""))
            actions.append(HeuristicAction(
                kind="command",
                command=f"attack {target_id}",
                confidence=0.90,
                domain="swarm",
                reason=f"[SWARM] Focus fire on {target_id}",
            ))

        return actions

    def _spread_targets_actions(
        self,
        bot_name: str,
        bot_role: str,
        situation: TacticalSituation,
        monsters: list[dict[str, Any]],
        focus_target_id: str,
    ) -> list[HeuristicAction]:
        """Each bot picks a different target for maximum clear speed.

        Melee takes the closest, ranged takes the farthest,
        support targets the most threatening.
        """
        actions: list[HeuristicAction] = []
        if not monsters:
            return actions

        if bot_role in ("tank", "dps_melee"):
            # Melee: attack closest
            closest = min(monsters, key=lambda m: m.get("distance", 999))
            target_id = closest.get("id", closest.get("name", ""))
            actions.append(HeuristicAction(
                kind="command",
                command=f"attack {target_id}",
                confidence=0.85,
                domain="swarm",
                reason=f"[SWARM] Spread: melee attack closest {target_id}",
            ))

        elif bot_role in ("dps_ranged", "dps_magic"):
            # Ranged/magic: attack from range, prioritize casters
            ranged_targets = [m for m in monsters if m.get("attack_type") in ("magic", "range")]
            if ranged_targets:
                target_id = ranged_targets[0].get("id", ranged_targets[0].get("name", ""))
            else:
                # Farthest monster (not being attacked by melee yet)
                farthest = max(monsters, key=lambda m: m.get("distance", 0))
                target_id = farthest.get("id", farthest.get("name", ""))
            actions.append(HeuristicAction(
                kind="command",
                command=f"attack {target_id}",
                confidence=0.80,
                domain="swarm",
                reason=f"[SWARM] Spread: ranged attack {target_id}",
            ))

        elif bot_role in ("healer", "support", "buffer"):
            # Healer: don't attack, focus on support
            actions.append(HeuristicAction(
                kind="command",
                command="ai manual",
                confidence=0.70,
                domain="swarm",
                reason="[SWARM] Spread: support role, standby",
            ))

        else:
            # Default: attack nearest
            nearest = min(monsters, key=lambda m: m.get("distance", 999))
            target_id = nearest.get("id", nearest.get("name", ""))
            actions.append(HeuristicAction(
                kind="command",
                command=f"attack {target_id}",
                confidence=0.75,
                domain="swarm",
                reason=f"[SWARM] Spread: attack {target_id}",
            ))

        return actions

    def _kite_actions(
        self,
        bot_name: str,
        bot_role: str,
        situation: TacticalSituation,
        monsters: list[dict[str, Any]],
        focus_target_id: str,
    ) -> list[HeuristicAction]:
        """Kite tactic: ranged attacks while moving away, melee intercepts.

        Ranged bots: attack and move back.
        Melee/tank: intercept and hold enemies.
        Healer: stay at max range and heal.
        """
        actions: list[HeuristicAction] = []

        if bot_role in ("dps_ranged", "dps_magic"):
            target = self._pick_priority_target(monsters, bot_role)
            target_name = target.get("name", "")
            # Attack while retreating
            actions.append(HeuristicAction(
                kind="command",
                command=f"attack {target_name}",
                confidence=0.85,
                domain="swarm",
                reason=f"[SWARM] Kite: attack {target_name} while retreating",
            ))
            # Move away from closest enemy
            if monsters:
                closest = min(monsters, key=lambda m: m.get("distance", 999))
                if closest.get("distance", 999) < 8:
                    actions.append(HeuristicAction(
                        kind="command",
                        command="move 5 5",  # Directional retreat
                        confidence=0.75,
                        domain="swarm",
                        reason="[SWARM] Kite: maintain distance",
                        metadata={"kite": True, "direction": "retreat"},
                    ))

        elif bot_role in ("tank", "dps_melee"):
            # Intercept: engage the closest enemy to protect backline
            if monsters:
                closest = min(monsters, key=lambda m: m.get("distance", 999))
                target_name = closest.get("name", "")
                actions.append(HeuristicAction(
                    kind="command",
                    command=f"attack {target_name}",
                    confidence=0.90,
                    domain="swarm",
                    reason=f"[SWARM] Kite: intercept {target_name}",
                ))

        elif bot_role in ("healer", "support"):
            # Healer stays safe, heals when needed
            actions.append(HeuristicAction(
                kind="command",
                command="ai manual",
                confidence=0.80,
                domain="swarm",
                reason="[SWARM] Kite: healer standby",
            ))

        return actions

    def _pincer_actions(
        self,
        bot_name: str,
        bot_role: str,
        situation: TacticalSituation,
        monsters: list[dict[str, Any]],
        focus_target_id: str,
    ) -> list[HeuristicAction]:
        """Pincer: converge on the enemy from two sides."""
        actions: list[HeuristicAction] = []

        if not monsters:
            return actions

        center_x = sum(m.get("x", 0) for m in monsters) // max(1, len(monsters))
        center_y = sum(m.get("y", 0) for m in monsters) // max(1, len(monsters))

        if bot_role in ("tank", "dps_melee"):
            # Melee approaches from one side
            pincer_x = center_x - 3
            pincer_y = center_y - 3
            actions.append(HeuristicAction(
                kind="command",
                command=f"move {pincer_x} {pincer_y}",
                confidence=0.80,
                domain="swarm",
                reason=f"[SWARM] Pincer: approach ({pincer_x}, {pincer_y})",
            ))

        elif bot_role in ("dps_ranged", "dps_magic"):
            # Ranged approaches from opposite side
            pincer_x = center_x + 5
            pincer_y = center_y + 5
            actions.append(HeuristicAction(
                kind="command",
                command=f"move {pincer_x} {pincer_y}",
                confidence=0.80,
                domain="swarm",
                reason=f"[SWARM] Pincer: flank ({pincer_x}, {pincer_y})",
            ))

        return actions

    def _withdraw_actions(
        self,
        bot_name: str,
        bot_role: str,
        situation: TacticalSituation,
        monsters: list[dict[str, Any]],
        focus_target_id: str,
    ) -> list[HeuristicAction]:
        """Ordered withdrawal: support first, then DPS, tank covers.

        The retreat formation is used: healer farthest back, tank last.
        """
        actions: list[HeuristicAction] = []

        if bot_role in ("healer", "support", "buffer", "dps_magic"):
            # These roles retreat first
            actions.append(HeuristicAction(
                kind="command",
                command="move 10 10",
                confidence=0.95,
                domain="swarm",
                reason="[SWARM] Withdraw: fall back",
            ))

        elif bot_role in ("dps_ranged", "dps_melee"):
            # DPS covers while retreating
            actions.append(HeuristicAction(
                kind="command",
                command="ai auto",
                confidence=0.85,
                domain="swarm",
                reason="[SWARM] Withdraw: cover retreat",
            ))
            actions.append(HeuristicAction(
                kind="command",
                command="move 5 5",
                confidence=0.80,
                domain="swarm",
                reason="[SWARM] Withdraw: move back",
            ))

        elif bot_role == "tank":
            # Tank is last to leave
            actions.append(HeuristicAction(
                kind="command",
                command="ai auto",
                confidence=0.90,
                domain="swarm",
                reason="[SWARM] Withdraw: cover rear",
            ))

        return actions

    def _ambush_actions(
        self,
        bot_name: str,
        bot_role: str,
        situation: TacticalSituation,
        monsters: list[dict[str, Any]],
        focus_target_id: str,
    ) -> list[HeuristicAction]:
        """Ambush: stay still, wait for enemy to approach, then strike."""
        actions: list[HeuristicAction] = []

        if situation.distance_to_enemy > self._engagement_range:
            # Stay hidden, don't move
            actions.append(HeuristicAction(
                kind="command",
                command="sit",
                confidence=0.70,
                domain="swarm",
                reason="[SWARM] Ambush: stand by",
            ))
        else:
            # Enemy in range — strike!
            target = self._pick_priority_target(monsters, bot_role)
            target_name = target.get("name", "")
            actions.append(HeuristicAction(
                kind="command",
                command=f"attack {target_name}",
                confidence=0.95,
                domain="swarm",
                reason=f"[SWARM] Ambush: strike {target_name}",
            ))

        return actions

    def _defensive_actions(
        self,
        bot_name: str,
        bot_role: str,
        situation: TacticalSituation,
        monsters: list[dict[str, Any]],
        focus_target_id: str,
    ) -> list[HeuristicAction]:
        """Defensive: focus on survival, stay close to each other."""
        actions: list[HeuristicAction] = []

        if bot_role == "tank":
            # Tank holds the line
            nearest = min(monsters, key=lambda m: m.get("distance", 999))
            target_name = nearest.get("name", "")
            actions.append(HeuristicAction(
                kind="command",
                command=f"attack {target_name}",
                confidence=0.90,
                domain="swarm",
                reason=f"[SWARM] Defensive: hold {target_name}",
            ))

        elif bot_role in ("healer", "support", "buffer"):
            # Stay close to tank/healer center
            actions.append(HeuristicAction(
                kind="command",
                command="ai manual",
                confidence=0.85,
                domain="swarm",
                reason="[SWARM] Defensive: stay safe",
            ))

        else:
            # DPS: attack nearest to reduce pressure
            nearest = min(monsters, key=lambda m: m.get("distance", 999))
            target_name = nearest.get("name", "")
            actions.append(HeuristicAction(
                kind="command",
                command=f"attack {target_name}",
                confidence=0.80,
                domain="swarm",
                reason=f"[SWARM] Defensive: clear {target_name}",
            ))

        return actions

    def _charge_actions(
        self,
        bot_name: str,
        bot_role: str,
        situation: TacticalSituation,
        monsters: list[dict[str, Any]],
        focus_target_id: str,
    ) -> list[HeuristicAction]:
        """Charge: all-out attack, all bots engage aggressively."""
        actions: list[HeuristicAction] = []

        if not monsters:
            return actions

        closest = min(monsters, key=lambda m: m.get("distance", 999))
        target_name = closest.get("name", "")

        if bot_role in ("tank", "dps_melee"):
            actions.append(HeuristicAction(
                kind="command",
                command=f"attack {target_name}",
                confidence=0.95,
                domain="swarm",
                reason=f"[SWARM] Charge: {target_name}",
                metadata={"aggressive": True},
            ))

        elif bot_role in ("dps_ranged", "dps_magic"):
            # Ranged also pushes forward
            actions.append(HeuristicAction(
                kind="command",
                command=f"attack {target_name}",
                confidence=0.90,
                domain="swarm",
                reason=f"[SWARM] Charge: {target_name}",
                metadata={"aggressive": True},
            ))

        elif bot_role in ("healer", "support"):
            # Healer pushes but stays behind
            actions.append(HeuristicAction(
                kind="command",
                command=f"attack {target_name}",
                confidence=0.70,
                domain="swarm",
                reason=f"[SWARM] Charge: push with {target_name}",
                metadata={"aggressive": True},
            ))

        return actions

    def _heal_rotation_actions(
        self,
        bot_name: str,
        bot_role: str,
        situation: TacticalSituation,
        monsters: list[dict[str, Any]],
        focus_target_id: str,
    ) -> list[HeuristicAction]:
        """Coordinated healing rotation."""
        actions: list[HeuristicAction] = []

        if bot_role in ("healer", "support"):
            # Healers focus on keeping party alive
            actions.append(HeuristicAction(
                kind="command",
                command="ai auto",
                confidence=0.85,
                domain="swarm",
                reason="[SWARM] Heal rotation: auto-heal mode",
            ))

        # Non-healers still fight normally
        if bot_role not in ("healer", "support"):
            if monsters:
                nearest = min(monsters, key=lambda m: m.get("distance", 999))
                target_name = nearest.get("name", "")
                actions.append(HeuristicAction(
                    kind="command",
                    command=f"attack {target_name}",
                    confidence=0.80,
                    domain="swarm",
                    reason=f"[SWARM] Heal rotation: dps {target_name}",
                ))

        return actions

    # ── Utility ─────────────────────────────────────────────────

    def assess_situation(
        self,
        bot_states: dict[str, Any],
        monsters: list[dict[str, Any]],
        my_bot_name: str,
    ) -> TacticalSituation:
        """Assess the current tactical situation from available data."""
        total_hp = 0.0
        total_sp = 0.0
        bot_count = 0

        for name, state in bot_states.items():
            if hasattr(state, 'hp_pct'):
                total_hp += state.hp_pct
                total_sp += state.sp_pct
                bot_count += 1

        hp_avg = total_hp / max(1, bot_count)
        sp_avg = total_sp / max(1, bot_count)

        has_mvp = any(m.get("is_mvp", False) for m in monsters)
        enemy_count = len(monsters)
        aoe_risk = any(
            m.get("attack_type") in ("aoe", "splash", "area")
            for m in monsters
        )

        distance = 999.0
        if monsters:
            distance = min(m.get("distance", 999) for m in monsters)

        retreat = hp_avg < 0.25

        return TacticalSituation(
            party_hp_avg=hp_avg,
            party_sp_avg=sp_avg,
            enemy_count=enemy_count,
            has_mvp=has_mvp,
            aoe_risk=aoe_risk,
            retreat_needed=retreat,
            distance_to_enemy=distance,
            allies_nearby=bot_count,
        )

    def _pick_priority_target(
        self,
        monsters: list[dict[str, Any]],
        bot_role: str,
    ) -> dict[str, Any]:
        """Pick the highest-priority monster for this bot's role."""
        if not monsters:
            return {"name": ""}

        if bot_role == "tank":
            # Tank: pick the most dangerous (highest ATK/level)
            return max(monsters, key=lambda m: m.get("level", 1))

        elif bot_role in ("healer", "support"):
            # Healer: pick the closest threat
            return min(monsters, key=lambda m: m.get("distance", 999))

        elif bot_role in ("dps_magic",):
            # Magic: pick the one weakest to magic
            return min(monsters, key=lambda m: m.get("distance", 999))

        # Default: pick the highest-priority target (MVP > high level > close)
        mvp = [m for m in monsters if m.get("is_mvp", False)]
        if mvp:
            return mvp[0]

        return max(monsters, key=lambda m: m.get("level", 1))

    def needs_healing(self, bot_states: dict[str, Any], target_name: str) -> bool:
        """Check if a specific bot needs healing based on swarm state."""
        for name, state in bot_states.items():
            if name == target_name or getattr(state, 'bot_name', '') == target_name:
                hp_pct = getattr(state, 'hp_pct', 1.0)
                return hp_pct < 0.5
        return False

    def get_current_tactic(self) -> TacticType:
        return self._current_tactic
