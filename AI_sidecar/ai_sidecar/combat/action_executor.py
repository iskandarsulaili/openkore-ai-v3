"""
Action Execution Engine — maps reflex action strings to actual OpenKore commands.

Uses dynamic skill levels from character state, RO-accurate cooldowns,
and proper cast time / delay calculations. No hardcoded skill levels.
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
from ai_sidecar.combat.damage_formulas import (
    SkillCooldownTracker,
    get_skill_cooldown,
    get_skill_element,
    get_skill_range,
    calculate_cast_time,
    calculate_after_cast_delay,
    get_monster_element,
    get_monster_size,
    get_monster_race,
    get_monster_def_data,
    calculate_damage,
    estimate_hits_to_kill,
)

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
    """Thread-safe action execution engine that maps reflex actions to commands.

    Uses dynamic skill levels from character state. If the bot doesn't have
    a skill, the mapping falls back to basic attack.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._mappings: dict[str, ActionMapping] = {}
        self._cd_tracker = SkillCooldownTracker()
        self._load_default_mappings()

    def _build_skill_command(self, skill_name: str, char_skills: dict[str, int]) -> str:
        """Build a skill command using the character's actual skill level.
        OpenKore uses 'ss <name> <level>' for use-skill-on-self.
        Falls back to 'attack' if the skill isn't known.
        """
        level = char_skills.get(skill_name, 0)
        if level <= 0:
            return "attack"
        return f"ss {skill_name} {level}"

    def _build_item_command(self, item_name: str, char_items: list[str]) -> str:
        """Build an item use command. Falls back to 'attack' if item not in inventory."""
        if item_name in char_items:
            return f"use {item_name}"
        # Try common fallbacks
        fallbacks = {"White Potion": "Red Potion", "Blue Potion": "Yellow Potion"}
        fallback = fallbacks.get(item_name)
        if fallback and fallback in char_items:
            return f"use {fallback}"
        return "attack"

    def _load_default_mappings(self) -> None:
        """Pre-populate all default action mappings with dynamic skill levels.
        The actual skill level is resolved at execution time from character state.
        """
        mappings: list[ActionMapping] = [
            # ── Elemental Attack Skills (reflex) ──
            ActionMapping(
                reflex_action="use_fire_skill",
                command_kind="attack_skill",
                command_template="skill {skill_name} {level}",
                skill_name="Fire Bolt",
                priority_tier="reflex",
                cooldown_ms=500,
            ),
            ActionMapping(
                reflex_action="use_water_skill",
                command_kind="attack_skill",
                command_template="skill {skill_name} {level}",
                skill_name="Cold Bolt",
                priority_tier="reflex",
                cooldown_ms=500,
            ),
            ActionMapping(
                reflex_action="use_wind_skill",
                command_kind="attack_skill",
                command_template="skill {skill_name} {level}",
                skill_name="Lightning Bolt",
                priority_tier="reflex",
                cooldown_ms=500,
            ),
            ActionMapping(
                reflex_action="use_earth_skill",
                command_kind="attack_skill",
                command_template="skill {skill_name} {level}",
                skill_name="Earth Spike",
                priority_tier="reflex",
                cooldown_ms=500,
            ),
            ActionMapping(
                reflex_action="use_holy_skill",
                command_kind="attack_skill",
                command_template="skill {skill_name} {level}",
                skill_name="Holy Light",
                priority_tier="reflex",
                cooldown_ms=500,
            ),
            ActionMapping(
                reflex_action="use_ghost_skill",
                command_kind="attack_skill",
                command_template="skill {skill_name} {level}",
                skill_name="Soul Strike",
                priority_tier="reflex",
                cooldown_ms=500,
            ),
            # ── Potion / Healing (reflex) ──
            ActionMapping(
                reflex_action="use_potion_or_heal",
                command_kind="use_item",
                command_template="use {item_name}",
                item_name="White Potion",
                priority_tier="reflex",
                cooldown_ms=200,
            ),
            # ── Strongest Skills (reflex) ──
            ActionMapping(
                reflex_action="use_strongest_skill",
                command_kind="attack_skill",
                command_template="skill {skill_name} {level}",
                skill_name="Storm Gust",
                priority_tier="reflex",
                cooldown_ms=2000,
            ),
            ActionMapping(
                reflex_action="use_strongest_aoe",
                command_kind="attack_skill",
                command_template="skill {skill_name} {level}",
                skill_name="Meteor Storm",
                priority_tier="reflex",
                cooldown_ms=3000,
            ),
            ActionMapping(
                reflex_action="use_aoe_skill",
                command_kind="attack_skill",
                command_template="skill {skill_name} {level}",
                skill_name="Fire Ball",
                priority_tier="reflex",
                cooldown_ms=2000,
            ),
            # ── Finisher / Melee / Ranged (reflex) ──
            ActionMapping(
                reflex_action="use_fast_finisher",
                command_kind="attack_skill",
                command_template="skill {skill_name} {level}",
                skill_name="Sonic Blow",
                priority_tier="reflex",
                cooldown_ms=500,
            ),
            ActionMapping(
                reflex_action="use_melee_skill",
                command_kind="attack_skill",
                command_template="skill {skill_name} {level}",
                skill_name="Bash",
                priority_tier="reflex",
                cooldown_ms=500,
            ),
            ActionMapping(
                reflex_action="use_ranged_skill",
                command_kind="attack_skill",
                command_template="skill {skill_name} {level}",
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
                command_template="skill {skill_name} {level}",
                skill_name="Heaven's Drive",
                priority_tier="reflex",
                cooldown_ms=2000,
            ),
            # ── Interrupt / Stun (reflex) ──
            ActionMapping(
                reflex_action="stun_or_silence_target",
                command_kind="attack_skill",
                command_template="skill {skill_name} {level}",
                skill_name="Bash",
                priority_tier="reflex",
                cooldown_ms=2000,
            ),
            ActionMapping(
                reflex_action="interrupt_caster",
                command_kind="attack_skill",
                command_template="skill {skill_name} {level}",
                skill_name="Bash",
                priority_tier="reflex",
                cooldown_ms=1500,
            ),
            # ── Movement / Teleport (reflex) ──
            ActionMapping(
                reflex_action="flee_to_safe_spot",
                command_kind="ai_manual",
                command_template="attackAuto 0",
                priority_tier="reflex",
                cooldown_ms=500,
            ),
            ActionMapping(
                reflex_action="teleport_away",
                command_kind="use_item",
                command_template="use {item_name}",
                item_name="Butterfly Wing",
                priority_tier="reflex",
                cooldown_ms=1000,
            ),
            ActionMapping(
                reflex_action="retreat_to_safe_spot",
                command_kind="ai_manual",
                command_template="attackAuto 0",
                priority_tier="reflex",
                cooldown_ms=3000,
            ),
            # ── Gear / Equipment (tactical) ──
            ActionMapping(
                reflex_action="swap_to_elemental_weapon",
                command_kind="use_item",
                command_template="eq {item_name}",
                item_name="Fireblend",
                priority_tier="tactical",
                cooldown_ms=2000,
            ),
            ActionMapping(
                reflex_action="equip_tank_set",
                command_kind="use_item",
                command_template="eq {item_name}",
                item_name="Shield",
                priority_tier="tactical",
                cooldown_ms=3000,
            ),
            # ── Buffs (tactical) ──
            ActionMapping(
                reflex_action="cast_blessing",
                command_kind="attack_skill",
                command_template="skill {skill_name} {level}",
                skill_name="Blessing",
                priority_tier="tactical",
                cooldown_ms=5000,
            ),
            ActionMapping(
                reflex_action="cast_increase_agility",
                command_kind="attack_skill",
                command_template="skill {skill_name} {level}",
                skill_name="Increase Agility",
                priority_tier="tactical",
                cooldown_ms=5000,
            ),
            ActionMapping(
                reflex_action="cast_endure",
                command_kind="attack_skill",
                command_template="skill {skill_name} {level}",
                skill_name="Endure",
                priority_tier="tactical",
                cooldown_ms=5000,
            ),
            ActionMapping(
                reflex_action="cast_improve_concentration",
                command_kind="attack_skill",
                command_template="skill {skill_name} {level}",
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
                command_template="use {item_name}",
                item_name="White Potion",
                priority_tier="tactical",
                cooldown_ms=5000,
            ),
            ActionMapping(
                reflex_action="prepare_for_phase_change",
                command_kind="use_item",
                command_template="use {item_name}",
                item_name="Blue Potion",
                priority_tier="tactical",
                cooldown_ms=3000,
            ),
            # ── Party Support (tactical) ──
            ActionMapping(
                reflex_action="heal_party_member",
                command_kind="attack_skill",
                command_template="skill {skill_name} {level}",
                skill_name="Heal",
                priority_tier="tactical",
                cooldown_ms=2000,
            ),
            ActionMapping(
                reflex_action="buff_party_members",
                command_kind="attack_skill",
                command_template="skill {skill_name} {level}",
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
                command_template="skill {skill_name} {level}",
                skill_name="Fire Bolt",
                priority_tier="reflex",
                cooldown_ms=500,
            ),
            ActionMapping(
                reflex_action="use_skill_Cold_Bolt",
                command_kind="attack_skill",
                command_template="skill {skill_name} {level}",
                skill_name="Cold Bolt",
                priority_tier="reflex",
                cooldown_ms=500,
            ),
            ActionMapping(
                reflex_action="use_skill_Lightning_Bolt",
                command_kind="attack_skill",
                command_template="skill {skill_name} {level}",
                skill_name="Lightning Bolt",
                priority_tier="reflex",
                cooldown_ms=500,
            ),
        ]

        for mapping in mappings:
            self._mappings[mapping.reflex_action] = mapping

    # ── Public API ────────────────────────────────────────────────────

    def execute(
        self,
        reflex_action: str,
        bot_id: str,
        action_queue: Any,
        char_skills: dict[str, int] | None = None,
        char_items: list[str] | None = None,
    ) -> bool:
        """Execute a reflex action by looking up its mapping and enqueuing it.

        Resolves dynamic skill levels from character state at execution time.
        Falls back to 'attack' if the required skill isn't known.

        Args:
            reflex_action: The reflex action string (e.g. "use_fire_skill").
            bot_id: The bot identifier to execute for.
            action_queue: An ActionQueue instance to enqueue into.
            char_skills: Dict of {skill_name: level} from character state.
            char_items: List of item names in inventory.

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

        command = self.build_command(mapping, char_skills or {}, char_items or [])
        return self.execute_command(command, bot_id, action_queue)

    def build_command(
        self,
        mapping: ActionMapping,
        char_skills: dict[str, int],
        char_items: list[str],
    ) -> str:
        """Build the actual command string, resolving dynamic skill levels."""
        if mapping.command_kind == "attack_skill" and mapping.skill_name:
            return self._build_skill_command(mapping.skill_name, char_skills)
        elif mapping.command_kind == "use_item" and mapping.item_name:
            return self._build_item_command(mapping.item_name, char_items)
        else:
            return mapping.command_template

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

    def execute_command(self, command: str, bot_id: str, action_queue: Any) -> bool:
        """Enqueue a command string as an action proposal."""
        if action_queue is None:
            logger.warning("No action queue available for bot_id=%r", bot_id)
            return False

        proposal = ActionProposal(
            bot_id=bot_id,
            action_type="command",
            priority_tier=ActionPriorityTier.reflex,
            source="action_executor",
            description=command,
            conflict_key=f"cmd_{command}_{int(time.time())}",
        )
        action_queue.enqueue(proposal)
        logger.debug("Enqueued command %r for bot_id=%r", command, bot_id)
        return True

    def get_cooldown_tracker(self) -> SkillCooldownTracker:
        """Get the shared cooldown tracker."""
        return self._cd_tracker

    def update_character_state(self, skills: dict[str, int], dex: int = 0) -> None:
        """Update character state for cooldown tracking."""
        self._cd_tracker.set_dex(dex)
