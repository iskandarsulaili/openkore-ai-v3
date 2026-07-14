"""
Action Execution Engine — maps reflex action strings to actual OpenKore commands.

Sits between the reflex combat layer (reflex_combat.py) and the action queue
(runtime/action_queue.py), translating combat decisions into executable
command strings that the bridge dispatches to OpenKore.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from threading import RLock
from typing import Any

from ai_sidecar.contracts.actions import ActionPriorityTier, ActionProposal

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ActionMapping:
    """Maps a reflex action string to an executable OpenKore command."""

    reflex_action: str
    command_kind: str
    command_template: str
    skill_name: str = ""
    item_name: str = ""
    priority_tier: str = "reflex"
    cooldown_ms: int = 100
    last_executed: float = 0.0


class ActionExecutor:
    """Thread-safe action execution engine that maps reflex actions to commands."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._mappings: dict[str, ActionMapping] = {}
        self._load_default_mappings()

    # ── Default Mappings ──────────────────────────────────────────────

    def _load_default_mappings(self) -> None:
        """Pre-populate all default action mappings (35 total)."""
        mappings: list[ActionMapping] = [
            # ── Elemental Attack Skills (reflex) ──
            ActionMapping(
                reflex_action="use_fire_skill",
                command_kind="attack_skill",
                command_template="skill Fire Bolt 3",
                skill_name="Fire Bolt",
                priority_tier="reflex",
                cooldown_ms=500,
            ),
            ActionMapping(
                reflex_action="use_water_skill",
                command_kind="attack_skill",
                command_template="skill Cold Bolt 5",
                skill_name="Cold Bolt",
                priority_tier="reflex",
                cooldown_ms=500,
            ),
            ActionMapping(
                reflex_action="use_wind_skill",
                command_kind="attack_skill",
                command_template="skill Lightning Bolt 5",
                skill_name="Lightning Bolt",
                priority_tier="reflex",
                cooldown_ms=500,
            ),
            ActionMapping(
                reflex_action="use_earth_skill",
                command_kind="attack_skill",
                command_template="skill Earth Spike 3",
                skill_name="Earth Spike",
                priority_tier="reflex",
                cooldown_ms=500,
            ),
            ActionMapping(
                reflex_action="use_holy_skill",
                command_kind="attack_skill",
                command_template="skill Holy Light 5",
                skill_name="Holy Light",
                priority_tier="reflex",
                cooldown_ms=500,
            ),
            ActionMapping(
                reflex_action="use_ghost_skill",
                command_kind="attack_skill",
                command_template="skill Soul Strike 5",
                skill_name="Soul Strike",
                priority_tier="reflex",
                cooldown_ms=500,
            ),
            # ── Potion / Healing (reflex) ──
            ActionMapping(
                reflex_action="use_potion_or_heal",
                command_kind="use_item",
                command_template="use White Potion",
                item_name="White Potion",
                priority_tier="reflex",
                cooldown_ms=200,
            ),
            # ── Strongest Skills (reflex) ──
            ActionMapping(
                reflex_action="use_strongest_skill",
                command_kind="attack_skill",
                command_template="skill Storm Gust 10",
                skill_name="Storm Gust",
                priority_tier="reflex",
                cooldown_ms=2000,
            ),
            ActionMapping(
                reflex_action="use_strongest_aoe",
                command_kind="attack_skill",
                command_template="skill Meteor Storm 10",
                skill_name="Meteor Storm",
                priority_tier="reflex",
                cooldown_ms=3000,
            ),
            ActionMapping(
                reflex_action="use_aoe_skill",
                command_kind="attack_skill",
                command_template="skill Fire Ball 5",
                skill_name="Fire Ball",
                priority_tier="reflex",
                cooldown_ms=2000,
            ),
            # ── Finisher / Melee / Ranged (reflex) ──
            ActionMapping(
                reflex_action="use_fast_finisher",
                command_kind="attack_skill",
                command_template="skill Sonic Blow 10",
                skill_name="Sonic Blow",
                priority_tier="reflex",
                cooldown_ms=500,
            ),
            ActionMapping(
                reflex_action="use_melee_skill",
                command_kind="attack_skill",
                command_template="skill Bash 10",
                skill_name="Bash",
                priority_tier="reflex",
                cooldown_ms=500,
            ),
            ActionMapping(
                reflex_action="use_ranged_skill",
                command_kind="attack_skill",
                command_template="skill Double Strafe 10",
                skill_name="Double Strafe",
                priority_tier="reflex",
                cooldown_ms=500,
            ),
            # ── Basic Attack (reflex) ──
            ActionMapping(
                reflex_action="use_basic_attack_only",
                command_kind="attack_skill",
                command_template="attack",
                skill_name="",
                priority_tier="reflex",
                cooldown_ms=2000,
            ),
            # ── High SP Spender (reflex) ──
            ActionMapping(
                reflex_action="use_high_sp_skill",
                command_kind="attack_skill",
                command_template="skill Heaven's Drive 5",
                skill_name="Heaven's Drive",
                priority_tier="reflex",
                cooldown_ms=2000,
            ),
            # ── Interrupt / Stun (reflex) ──
            ActionMapping(
                reflex_action="stun_or_silence_target",
                command_kind="attack_skill",
                command_template="skill Bash 10",
                skill_name="Bash",
                priority_tier="reflex",
                cooldown_ms=2000,
            ),
            ActionMapping(
                reflex_action="interrupt_caster",
                command_kind="attack_skill",
                command_template="skill Bash 10",
                skill_name="Bash",
                priority_tier="reflex",
                cooldown_ms=1500,
            ),
            # ── Movement / Teleport (reflex) ──
            ActionMapping(
                reflex_action="flee_to_safe_spot",
                command_kind="ai_manual",
                command_template="ai manual",
                priority_tier="reflex",
                cooldown_ms=500,
            ),
            ActionMapping(
                reflex_action="teleport_away",
                command_kind="use_item",
                command_template="use Butterfly Wing",
                item_name="Butterfly Wing",
                priority_tier="reflex",
                cooldown_ms=1000,
            ),
            ActionMapping(
                reflex_action="retreat_to_safe_spot",
                command_kind="ai_manual",
                command_template="ai manual",
                priority_tier="reflex",
                cooldown_ms=3000,
            ),
            # ── Gear / Equipment (tactical) ──
            ActionMapping(
                reflex_action="swap_to_elemental_weapon",
                command_kind="use_item",
                command_template="eq Fireblend",
                item_name="Fireblend",
                priority_tier="tactical",
                cooldown_ms=2000,
            ),
            ActionMapping(
                reflex_action="equip_tank_set",
                command_kind="use_item",
                command_template="eq Shield",
                item_name="Shield",
                priority_tier="tactical",
                cooldown_ms=3000,
            ),
            # ── Buffs (tactical) ──
            ActionMapping(
                reflex_action="cast_blessing",
                command_kind="attack_skill",
                command_template="skill Blessing 10",
                skill_name="Blessing",
                priority_tier="tactical",
                cooldown_ms=5000,
            ),
            ActionMapping(
                reflex_action="cast_increase_agility",
                command_kind="attack_skill",
                command_template="skill Increase Agility 10",
                skill_name="Increase Agility",
                priority_tier="tactical",
                cooldown_ms=5000,
            ),
            ActionMapping(
                reflex_action="cast_endure",
                command_kind="attack_skill",
                command_template="skill Endure 5",
                skill_name="Endure",
                priority_tier="tactical",
                cooldown_ms=5000,
            ),
            ActionMapping(
                reflex_action="cast_improve_concentration",
                command_kind="attack_skill",
                command_template="skill Improve Concentration 10",
                skill_name="Improve Concentration",
                priority_tier="tactical",
                cooldown_ms=5000,
            ),
            # ── Distance / Movement (reflex) ──
            ActionMapping(
                reflex_action="maintain_distance",
                command_kind="map_move",
                command_template="move",
                priority_tier="reflex",
                cooldown_ms=1000,
            ),
            # ── Pre-buff / Prep (tactical) ──
            ActionMapping(
                reflex_action="pre_drink_potion",
                command_kind="use_item",
                command_template="use White Potion",
                item_name="White Potion",
                priority_tier="tactical",
                cooldown_ms=5000,
            ),
            ActionMapping(
                reflex_action="prepare_for_phase_change",
                command_kind="use_item",
                command_template="use Blue Potion",
                item_name="Blue Potion",
                priority_tier="tactical",
                cooldown_ms=3000,
            ),
            # ── Party Support (tactical) ──
            ActionMapping(
                reflex_action="heal_party_member",
                command_kind="attack_skill",
                command_template="skill Heal 10",
                skill_name="Heal",
                priority_tier="tactical",
                cooldown_ms=2000,
            ),
            ActionMapping(
                reflex_action="buff_party_members",
                command_kind="attack_skill",
                command_template="skill Blessing 10",
                skill_name="Blessing",
                priority_tier="tactical",
                cooldown_ms=10000,
            ),
            # ── Rest / Regen (tactical) ──
            ActionMapping(
                reflex_action="rest_and_regen_sp",
                command_kind="ai_manual",
                command_template="sit",
                priority_tier="tactical",
                cooldown_ms=5000,
            ),
            # ── Explicit Skill Name Variants (reflex) ──
            ActionMapping(
                reflex_action="use_skill_Fire_Bolt",
                command_kind="attack_skill",
                command_template="skill Fire Bolt 3",
                skill_name="Fire Bolt",
                priority_tier="reflex",
                cooldown_ms=500,
            ),
            ActionMapping(
                reflex_action="use_skill_Cold_Bolt",
                command_kind="attack_skill",
                command_template="skill Cold Bolt 5",
                skill_name="Cold Bolt",
                priority_tier="reflex",
                cooldown_ms=500,
            ),
            ActionMapping(
                reflex_action="use_skill_Lightning_Bolt",
                command_kind="attack_skill",
                command_template="skill Lightning Bolt 5",
                skill_name="Lightning Bolt",
                priority_tier="reflex",
                cooldown_ms=500,
            ),
        ]

        for mapping in mappings:
            self._mappings[mapping.reflex_action] = mapping

    # ── Public API ────────────────────────────────────────────────────

    def execute(self, reflex_action: str, bot_id: str, action_queue: Any) -> bool:
        """Execute a reflex action by looking up its mapping and enqueuing it.

        Args:
            reflex_action: The reflex action string (e.g. "use_fire_skill").
            bot_id: The bot identifier to execute for.
            action_queue: An ActionQueue instance to enqueue into.

        Returns:
            True if the action was successfully enqueued, False otherwise.
        """
        mapping = self.get_mapping(reflex_action)
        if mapping is None:
            logger.warning("No mapping found for reflex_action=%r", reflex_action)
            return False

        with self._lock:
            now_ms = time.time() * 1000
            if now_ms - mapping.last_executed * 1000 < mapping.cooldown_ms:
                logger.debug(
                    "Action %r on cooldown (%.0fms remaining)",
                    reflex_action,
                    mapping.cooldown_ms - (now_ms - mapping.last_executed * 1000),
                )
                return False
            mapping.last_executed = time.time()

        command = self.build_command(mapping)
        return self.execute_command(command, bot_id, action_queue)

    def get_mapping(self, reflex_action: str) -> ActionMapping | None:
        """Look up an action mapping by reflex action string."""
        with self._lock:
            return self._mappings.get(reflex_action)

    def register_mapping(self, mapping: ActionMapping) -> None:
        """Register a new action mapping (or overwrite an existing one)."""
        with self._lock:
            self._mappings[mapping.reflex_action] = mapping

    def get_all_mappings(self) -> list[ActionMapping]:
        """Return all registered action mappings."""
        with self._lock:
            return list(self._mappings.values())

    def get_mappings_by_priority(self, priority_tier: str) -> list[ActionMapping]:
        """Return all mappings matching a given priority tier."""
        with self._lock:
            return [m for m in self._mappings.values() if m.priority_tier == priority_tier]

    def get_mappings_by_kind(self, command_kind: str) -> list[ActionMapping]:
        """Return all mappings matching a given command kind."""
        with self._lock:
            return [m for m in self._mappings.values() if m.command_kind == command_kind]

    def build_command(self, mapping: ActionMapping, target_id: str | None = None) -> str:
        """Build the actual command string from a mapping, optionally targeting an entity.

        For attack_skill commands, appends the target_id if provided.
        For map_move commands, returns the template as-is (movement is handled
        separately by the bridge).
        """
        cmd = mapping.command_template
        if target_id and mapping.command_kind == "attack_skill" and cmd != "attack":
            cmd = f"{cmd} {target_id}"
        return cmd

    def execute_skill(
        self,
        skill_name: str,
        target_id: str | None,
        bot_id: str,
        action_queue: Any,
    ) -> bool:
        """Execute a skill by name, looking up the first matching mapping.

        Searches all mappings for one whose skill_name matches. If found,
        builds the command and enqueues it.
        """
        with self._lock:
            mapping = next(
                (m for m in self._mappings.values() if m.skill_name == skill_name),
                None,
            )
        if mapping is None:
            logger.warning("No mapping found for skill_name=%r", skill_name)
            return False
        command = self.build_command(mapping, target_id=target_id)
        return self.execute_command(command, bot_id, action_queue)

    def execute_item(
        self,
        item_name: str,
        bot_id: str,
        action_queue: Any,
    ) -> bool:
        """Execute an item use by name, looking up the first matching mapping.

        Searches all mappings for one whose item_name matches. If found,
        builds the command and enqueues it.
        """
        with self._lock:
            mapping = next(
                (m for m in self._mappings.values() if m.item_name == item_name),
                None,
            )
        if mapping is None:
            logger.warning("No mapping found for item_name=%r", item_name)
            return False
        command = self.build_command(mapping)
        return self.execute_command(command, bot_id, action_queue)

    def execute_command(
        self,
        command_str: str,
        bot_id: str,
        action_queue: Any,
    ) -> bool:
        """Enqueue a raw command string as an ActionProposal into the action queue.

        Determines the command kind and priority tier heuristically from the
        command string when no mapping context is available.

        Args:
            command_str: The command string to enqueue (e.g. "skill Fire Bolt 3").
            bot_id: The bot identifier.
            action_queue: An ActionQueue instance.

        Returns:
            True if the action was successfully enqueued, False otherwise.
        """
        now = datetime.now(UTC)
        action_id = f"exec_{uuid.uuid4().hex[:16]}"
        idempotency_key = f"exec_{command_str}_{int(now.timestamp())}"

        kind, priority_tier = self._infer_command_metadata(command_str)

        proposal = ActionProposal(
            action_id=action_id,
            kind=kind,
            command=command_str,
            priority_tier=ActionPriorityTier(priority_tier),
            conflict_key=f"exec_{command_str}",
            source="reflex",
            created_at=now,
            expires_at=now + timedelta(seconds=30),
            idempotency_key=idempotency_key,
        )

        success, status, returned_action_id, message = action_queue.enqueue(bot_id, proposal)
        if success:
            logger.debug(
                "Enqueued command %r for bot %s (action_id=%s, status=%s)",
                command_str,
                bot_id,
                returned_action_id,
                status,
            )
        else:
            logger.debug(
                "Failed to enqueue command %r for bot %s: %s (status=%s)",
                command_str,
                bot_id,
                message,
                status,
            )
        return success

    def get_execution_summary(self) -> str:
        """Return a human-readable summary of all registered action mappings."""
        with self._lock:
            lines = ["── Action Execution Engine Summary ──"]
            lines.append(f"Total mappings: {len(self._mappings)}")
            by_priority: dict[str, int] = {}
            by_kind: dict[str, int] = {}
            for m in self._mappings.values():
                by_priority[m.priority_tier] = by_priority.get(m.priority_tier, 0) + 1
                by_kind[m.command_kind] = by_kind.get(m.command_kind, 0) + 1
            lines.append("")
            lines.append("By priority tier:")
            for tier in ("reflex", "tactical", "strategic"):
                count = by_priority.get(tier, 0)
                if count:
                    lines.append(f"  {tier}: {count}")
            lines.append("")
            lines.append("By command kind:")
            for kind in sorted(by_kind):
                lines.append(f"  {kind}: {by_kind[kind]}")
            lines.append("")
            lines.append("Registered reflex actions:")
            for name in sorted(self._mappings):
                m = self._mappings[name]
                lines.append(f"  {name} → [{m.command_kind}] {m.command_template}")
            return "\n".join(lines)

    # ── Internal Helpers ─────────────────────────────────────────────

    @staticmethod
    def _infer_command_metadata(command_str: str) -> tuple[str, str]:
        """Infer command kind and priority tier from a command string.

        Returns (kind, priority_tier).
        """
        lower = command_str.strip().lower()

        if lower.startswith("skill "):
            return "attack_skill", "reflex"
        if lower.startswith("use "):
            return "use_item", "reflex"
        if lower.startswith("eq "):
            return "use_item", "tactical"
        if lower == "attack":
            return "attack_skill", "reflex"
        if lower == "sit":
            return "ai_manual", "tactical"
        if lower == "ai manual":
            return "ai_manual", "reflex"
        if lower == "move":
            return "map_move", "reflex"
        if lower.startswith("ai "):
            return "ai_manual", "reflex"
        return "command", "reflex"


# ── Global Singleton ──────────────────────────────────────────────────

_executor: ActionExecutor | None = None
_executor_lock = RLock()


def get_action_executor() -> ActionExecutor:
    """Return the global ActionExecutor singleton."""
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ActionExecutor()
        return _executor
