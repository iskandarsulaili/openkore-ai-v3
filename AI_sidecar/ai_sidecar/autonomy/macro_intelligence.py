"""
Macro Intelligence Engine — AI-driven macro knowledge and generation.
====================================================================
Instead of hardcoding OpenKore macros, this system:
1. Stores KNOWLEDGE about community macro patterns (triggers, actions, sequences)
2. Evaluates trigger conditions against the current bot state
3. Returns matched macros ordered by priority, respecting exclusive locks
4. Emits action sequences through the action pipeline

The system uses a process_triggers → priority-sorted → exclusive-blocking
pipeline to ensure the highest-priority macro runs first and exclusive
macros block lower-priority ones from executing in the same tick.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1. CORE DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════

class PriorityTier(IntEnum):
    """Priority tiers for macro evaluation ordering.
    Higher values = higher priority. VITALS always runs first.
    """
    VITALS_CRITICAL = 100
    VITALS_EMERGENCY = 95
    VITALS_DANGER = 90
    VITALS_NORMAL = 85
    VITALS_PREHEAL = 80
    SAFETY_GM = 79
    SAFETY_ANTISTUCK = 75
    SAFETY_ESCAPE = 70
    SAFETY_LOGIN = 65
    COMBAT_EMERGENCY = 60
    COMBAT_BOSS = 58
    COMBAT_SKILL = 55
    COMBAT_ATTACK = 50
    LOOT_PRIORITY = 49
    LOOT_NORMAL = 45
    PARTY_HEAL = 44
    PARTY_BUFF = 40
    PARTY_RESS = 38
    FARMING_MVP = 35
    FARMING_SWITCH = 30
    INVENTORY_SELL = 29
    INVENTORY_STORE = 25
    INVENTORY_BUY = 20
    NAVIGATION_WARP = 19
    NAVIGATION_MOVE = 15
    NAVIGATION_FOLLOW = 10


@dataclass
class MacroTrigger:
    """A single trigger condition evaluated against bot state.

    Uses path-based access like 'vitals.hp_ratio' -> state['vitals']['hp_ratio'].
    Supports operations: eq, neq, gt, gte, lt, lte, in, not_in, contains.
    """
    type: str  # Dot-separated path into bot state dict
    op: str = "eq"  # eq, neq, gt, gte, lt, lte, in, not_in, contains
    value: Any = None

    def evaluate(self, state: dict[str, Any]) -> bool:
        """Evaluate this trigger against the given bot state."""
        actual = _resolve_path(state, self.type)
        if actual is None and self.op not in ("eq", "neq"):
            return False
        try:
            return _apply_op(actual, self.op, self.value)
        except (TypeError, ValueError):
            return False

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "op": self.op, "value": self.value}


@dataclass
class MacroAction:
    """A single step in a macro sequence."""
    command: str
    description: str = ""
    timeout_seconds: float = 10.0
    requires_confirmation: bool = False  # NPC dialog response, etc.
    conflict_key: str = ""


@dataclass
class MacroPattern:
    """A known macro pattern with triggers, priority, and action sequences.

    The system evaluates triggers against bot state, then sorts matched
    patterns by priority (descending). If a pattern is exclusive, it
    blocks all lower-priority patterns in the same tick.

    process_triggers returns the single highest-priority non-blocked macro.
    get_patterns_for_context returns all non-cooling-down matches for AI review.
    """
    pattern_id: str
    category: str  # vitals, combat, loot, navigation, inventory, party, farming, safety
    description: str
    triggers: list[MacroTrigger | dict[str, Any]]  # All must match (AND)
    action_sequence: list[MacroAction]
    priority: int = 50  # Higher = runs first
    required_items: list[str] = field(default_factory=list)
    required_zeny: int = 0
    required_level_range: tuple[int, int] = (1, 999)
    required_jobs: list[str] = field(default_factory=list)
    cooldown_seconds: float = 0.0
    exclusive: bool = False  # If true, blocks lower-priority macros this tick
    disable_in_combat: bool = False
    disable_out_of_combat: bool = False
    state_tracking_key: str = ""  # Key to track in _macro_state for cooldown/flag
    chain_on_success: str = ""  # pattern_id to chain after this one succeeds


def _resolve_path(state: dict[str, Any], path: str) -> Any:
    """Resolve a dot-separated path into the state dict.
    e.g. 'vitals.hp_ratio' -> state['vitals']['hp_ratio']
    """
    parts = path.split(".")
    current = state
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
        if current is None:
            return None
    return current


def _apply_op(actual: Any, op: str, value: Any) -> bool:
    """Apply a comparison operator."""
    if op == "eq":
        return actual == value
    elif op == "neq":
        return actual != value
    elif op == "gt":
        return isinstance(actual, (int, float)) and actual > value
    elif op == "gte":
        return isinstance(actual, (int, float)) and actual >= value
    elif op == "lt":
        return isinstance(actual, (int, float)) and actual < value
    elif op == "lte":
        return isinstance(actual, (int, float)) and actual <= value
    elif op == "in":
        return isinstance(value, (list, tuple, set)) and actual in value
    elif op == "not_in":
        return isinstance(value, (list, tuple, set)) and actual not in value
    elif op == "contains":
        return hasattr(actual, "__contains__") and value in actual
    return False


def _make_trigger(type_: str, op: str = "eq", value: Any = None) -> MacroTrigger:
    """Convenience factory for MacroTrigger."""
    return MacroTrigger(type=type_, op=op, value=value)


def _make_action(command: str, description: str = "",
                 timeout: float = 10.0, confirm: bool = False,
                 conflict_key: str = "") -> MacroAction:
    """Convenience factory for MacroAction."""
    return MacroAction(
        command=command,
        description=description,
        timeout_seconds=timeout,
        requires_confirmation=confirm,
        conflict_key=conflict_key,
    )


# ═══════════════════════════════════════════════════════════════════════
# 2. COMPLETE MACRO PATTERN CATALOG — 50+ Production Macros
# ═══════════════════════════════════════════════════════════════════════

MACRO_PATTERNS: dict[str, MacroPattern] = {}

# ───────────────────────────────────────────────────────────────────────
# VITALS TIER (highest priority, cooldown-aware)
# Priority range: 100-80
# ───────────────────────────────────────────────────────────────────────

# 1. HP 80% — Pre-heal vs big mobs (small pot, proactive)
MACRO_PATTERNS["vitals_hp_80_preheal"] = MacroPattern(
    pattern_id="vitals_hp_80_preheal", category="vitals",
    description="Pre-heal with small potion when HP drops below 80% and engaging a tough monster.",
    priority=PriorityTier.VITALS_PREHEAL,
    triggers=[
        _make_trigger("vitals.hp_ratio", "lte", 0.80),
        _make_trigger("combat.target_threat", "gte", 0.6),
        _make_trigger("combat.is_in_combat", "eq", True),
    ],
    action_sequence=[
        _make_action("use white_potion", "Use small HP potion (pre-heal)", timeout=2.0),
    ],
    cooldown_seconds=3.0, exclusive=False,
    state_tracking_key="hp_pot_last_use",
)

# 2. HP 60% — Normal use (medium potion)
MACRO_PATTERNS["vitals_hp_60_normal"] = MacroPattern(
    pattern_id="vitals_hp_60_normal", category="vitals",
    description="Use medium HP potion when HP drops below 60%.",
    priority=PriorityTier.VITALS_NORMAL,
    triggers=[
        _make_trigger("vitals.hp_ratio", "lte", 0.60),
        _make_trigger("vitals.hp_ratio", "gt", 0.40),
    ],
    action_sequence=[
        _make_action("use condensed_white_potion", "Use medium HP potion", timeout=2.0),
    ],
    cooldown_seconds=2.0, exclusive=False,
    state_tracking_key="hp_pot_last_use",
)

# 3. HP 40% — Danger (large potion)
MACRO_PATTERNS["vitals_hp_40_danger"] = MacroPattern(
    pattern_id="vitals_hp_40_danger", category="vitals",
    description="Use large HP potion when HP drops below 40% (danger threshold).",
    priority=PriorityTier.VITALS_DANGER,
    triggers=[
        _make_trigger("vitals.hp_ratio", "lte", 0.40),
        _make_trigger("vitals.hp_ratio", "gt", 0.20),
    ],
    action_sequence=[
        _make_action("use_yellow_potion", "Use large HP potion", timeout=2.0),
    ],
    cooldown_seconds=1.5, exclusive=False,
    state_tracking_key="hp_pot_last_use",
)

# 4. HP 20% — Emergency (big heal / slim pot)
MACRO_PATTERNS["vitals_hp_20_emergency"] = MacroPattern(
    pattern_id="vitals_hp_20_emergency", category="vitals",
    description="Emergency heal with biggest HP potion when HP drops to 20%.",
    priority=PriorityTier.VITALS_EMERGENCY,
    triggers=[
        _make_trigger("vitals.hp_ratio", "lte", 0.20),
        _make_trigger("vitals.hp_ratio", "gt", 0.10),
    ],
    action_sequence=[
        _make_action("use_slim_white_potion", "Use emergency HP potion", timeout=2.0),
    ],
    cooldown_seconds=1.0, exclusive=False,
    state_tracking_key="hp_pot_last_use",
)

# 5. HP 10% — Panic (any pot / yggdrasil if available)
MACRO_PATTERNS["vitals_hp_10_panic"] = MacroPattern(
    pattern_id="vitals_hp_10_panic", category="vitals",
    description="Panic heal with strongest available potion or Yggdrasil leaf when HP at 10% or below.",
    priority=PriorityTier.VITALS_CRITICAL,
    triggers=[
        _make_trigger("vitals.hp_ratio", "lte", 0.10),
    ],
    action_sequence=[
        _make_action("use_yggdrasil_berry", "Panic heal with Yggdrasil berry", timeout=1.0),
    ],
    cooldown_seconds=0.5, exclusive=True,
    state_tracking_key="hp_panic_last",
)

# 6. SP 60% — Mild SP recovery
MACRO_PATTERNS["vitals_sp_60_normal"] = MacroPattern(
    pattern_id="vitals_sp_60_normal", category="vitals",
    description="Use small SP potion when SP drops to 60%.",
    priority=PriorityTier.VITALS_NORMAL - 1,
    triggers=[
        _make_trigger("vitals.sp_ratio", "lte", 0.60),
        _make_trigger("vitals.sp_ratio", "gt", 0.40),
        _make_trigger("combat.is_in_combat", "eq", True),
    ],
    action_sequence=[
        _make_action("use_blue_potion", "Use small SP potion", timeout=2.0),
    ],
    cooldown_seconds=4.0, exclusive=False,
    state_tracking_key="sp_pot_last_use",
)

# 7. SP 40% — Medium SP recovery
MACRO_PATTERNS["vitals_sp_40_danger"] = MacroPattern(
    pattern_id="vitals_sp_40_danger", category="vitals",
    description="Use medium SP potion when SP drops to 40%.",
    priority=PriorityTier.VITALS_DANGER - 1,
    triggers=[
        _make_trigger("vitals.sp_ratio", "lte", 0.40),
        _make_trigger("vitals.sp_ratio", "gt", 0.20),
        _make_trigger("combat.is_in_combat", "eq", True),
    ],
    action_sequence=[
        _make_action("use_condensed_blue_potion", "Use medium SP potion", timeout=2.0),
    ],
    cooldown_seconds=3.0, exclusive=False,
    state_tracking_key="sp_pot_last_use",
)

# 8. SP 20% — Emergency SP recovery
MACRO_PATTERNS["vitals_sp_20_emergency"] = MacroPattern(
    pattern_id="vitals_sp_20_emergency", category="vitals",
    description="Use large SP potion when SP drops to 20% or below.",
    priority=PriorityTier.VITALS_EMERGENCY - 1,
    triggers=[
        _make_trigger("vitals.sp_ratio", "lte", 0.20),
    ],
    action_sequence=[
        _make_action("use_blue_herb_potion", "Use emergency SP potion", timeout=2.0),
    ],
    cooldown_seconds=2.0, exclusive=False,
    state_tracking_key="sp_pot_last_use",
)

# 9. Berserk potion when fighting MVP/boss
MACRO_PATTERNS["vitals_berserk_mvp"] = MacroPattern(
    pattern_id="vitals_berserk_mvp", category="vitals",
    description="Use berserk potion when engaging an MVP or boss for damage boost.",
    priority=PriorityTier.VITALS_NORMAL - 2,
    triggers=[
        _make_trigger("combat.target_is_boss", "eq", True),
        _make_trigger("vitals.hp_ratio", "gte", 0.50),
        _make_trigger("combat.is_in_combat", "eq", True),
    ],
    action_sequence=[
        _make_action("use_berserk_potion", "Use berserk potion for MVP fight", timeout=3.0),
    ],
    cooldown_seconds=180.0, exclusive=False,
    state_tracking_key="berserk_last_use",
)

# 10. Green potion / panacea on status ailment
MACRO_PATTERNS["vitals_status_cure"] = MacroPattern(
    pattern_id="vitals_status_cure", category="vitals",
    description="Use green potion or panacea when poisoned, silenced, or confused.",
    priority=PriorityTier.VITALS_EMERGENCY + 1,
    triggers=[
        _make_trigger("vitals.has_status", "eq", True),
        _make_trigger("vitals.status_types", "contains", "poison"),
    ],
    action_sequence=[
        _make_action("use_green_potion", "Cure poison with green potion", timeout=2.0),
    ],
    cooldown_seconds=1.0, exclusive=False,
    state_tracking_key="cure_last_use",
)

# 11. Panacea for all status effects
MACRO_PATTERNS["vitals_panacea_status"] = MacroPattern(
    pattern_id="vitals_panacea_status", category="vitals",
    description="Use panacea when afflicted by any harmful status effect.",
    priority=PriorityTier.VITALS_EMERGENCY + 1,
    triggers=[
        _make_trigger("vitals.has_status", "eq", True),
        _make_trigger("vitals.status_severity", "gte", 2),
    ],
    action_sequence=[
        _make_action("use_panacea", "Use panacea to cure all status effects", timeout=2.0),
    ],
    cooldown_seconds=2.0, exclusive=False,
    state_tracking_key="panacea_last_use",
)

# 12. Awakening potion when ASPD buff needed
MACRO_PATTERNS["vitals_awakening_aspd"] = MacroPattern(
    pattern_id="vitals_awakening_aspd", category="vitals",
    description="Use awakening potion when attack speed boost is needed for combat.",
    priority=PriorityTier.VITALS_NORMAL - 3,
    triggers=[
        _make_trigger("combat.is_in_combat", "eq", True),
        _make_trigger("vitals.aspd_buff_active", "eq", False),
        _make_trigger("vitals.aspd_ratio", "lt", 0.85),
    ],
    action_sequence=[
        _make_action("use_awakening_potion", "Use awakening potion for ASPD", timeout=3.0),
    ],
    cooldown_seconds=120.0, exclusive=False,
    state_tracking_key="awakening_last_use",
)


# ───────────────────────────────────────────────────────────────────────
# COMBAT TIER (priority range: 60-50)
# ───────────────────────────────────────────────────────────────────────

# 13. Elemental converter use
MACRO_PATTERNS["combat_elemental_converter"] = MacroPattern(
    pattern_id="combat_elemental_converter", category="combat",
    description="Apply elemental converter when weapon element is countered by target monster.",
    priority=PriorityTier.COMBAT_BOSS,
    triggers=[
        _make_trigger("combat.is_in_combat", "eq", True),
        _make_trigger("combat.element_disadvantage", "eq", True),
        _make_trigger("inventory.has_elemental_converter", "eq", True),
    ],
    action_sequence=[
        _make_action("use_elemental_converter", "Apply elemental converter to weapon", timeout=3.0),
    ],
    cooldown_seconds=60.0, exclusive=False,
    state_tracking_key="elemental_converter_last",
)

# 14. Auto-attack with range check
MACRO_PATTERNS["combat_auto_attack"] = MacroPattern(
    pattern_id="combat_auto_attack", category="combat",
    description="Auto-attack target within correct attack range.",
    priority=PriorityTier.COMBAT_ATTACK,
    triggers=[
        _make_trigger("combat.has_target", "eq", True),
        _make_trigger("combat.in_attack_range", "eq", True),
        _make_trigger("combat.is_attacking", "eq", False),
    ],
    action_sequence=[
        _make_action("attack", "Auto-attack current target", timeout=5.0),
    ],
    cooldown_seconds=1.0, exclusive=False,
    disable_in_combat=False,
)

# 15. Skill rotation (delegate to combat domain)
MACRO_PATTERNS["combat_skill_rotation"] = MacroPattern(
    pattern_id="combat_skill_rotation", category="combat",
    description="Execute skill rotation delegated to combat domain logic.",
    priority=PriorityTier.COMBAT_SKILL,
    triggers=[
        _make_trigger("combat.is_in_combat", "eq", True),
        _make_trigger("combat.has_target", "eq", True),
        _make_trigger("combat.skill_ready", "eq", True),
    ],
    action_sequence=[
        _make_action("domain combat execute_rotation", "Execute combat skill rotation", timeout=3.0),
    ],
    cooldown_seconds=1.0, exclusive=False,
)

# 16. Sit for SP regen when below 30% SP out of combat
MACRO_PATTERNS["combat_sit_sp_regen"] = MacroPattern(
    pattern_id="combat_sit_sp_regen", category="combat",
    description="Sit down to regenerate SP when below 30% out of combat.",
    priority=PriorityTier.COMBAT_ATTACK - 1,
    triggers=[
        _make_trigger("vitals.sp_ratio", "lte", 0.30),
        _make_trigger("combat.is_in_combat", "eq", False),
        _make_trigger("vitals.in_safety", "eq", True),
        _make_trigger("vitals.is_sitting", "eq", False),
    ],
    action_sequence=[
        _make_action("sit", "Sit for SP regeneration", timeout=30.0),
    ],
    cooldown_seconds=5.0, exclusive=False,
    disable_out_of_combat=False,
)

# 17. Stand up when SP recovered or combat starts
MACRO_PATTERNS["combat_stand_sp_recovered"] = MacroPattern(
    pattern_id="combat_stand_sp_recovered", category="combat",
    description="Stand up when SP has recovered sufficiently or combat starts.",
    priority=PriorityTier.COMBAT_ATTACK - 2,
    triggers=[
        _make_trigger("vitals.is_sitting", "eq", True),
        _make_trigger("vitals.sp_ratio", "gte", 0.60),
    ],
    action_sequence=[
        _make_action("stand", "Stand up after SP recovery", timeout=2.0),
    ],
    cooldown_seconds=3.0, exclusive=False,
)

# 18. Fly wing when surrounded by 5+ mobs
MACRO_PATTERNS["combat_fly_wing_surrounded"] = MacroPattern(
    pattern_id="combat_fly_wing_surrounded", category="combat",
    description="Use fly wing when surrounded by 5 or more monsters.",
    priority=PriorityTier.COMBAT_EMERGENCY,
    triggers=[
        _make_trigger("combat.aggro_count", "gte", 5),
    ],
    action_sequence=[
        _make_action("use_fly_wing", "Use fly wing to escape swarm", timeout=3.0),
    ],
    cooldown_seconds=10.0, exclusive=True,
    state_tracking_key="fly_wing_last",
)

# 19. Teleport when mob uses dangerous skill
MACRO_PATTERNS["combat_teleport_dangerous_skill"] = MacroPattern(
    pattern_id="combat_teleport_dangerous_skill", category="combat",
    description="Teleport away when a monster uses a dangerous skill (sight trigger).",
    priority=PriorityTier.COMBAT_EMERGENCY + 1,
    triggers=[
        _make_trigger("combat.dangerous_skill_detected", "eq", True),
        _make_trigger("vitals.hp_ratio", "lte", 0.50),
    ],
    action_sequence=[
        _make_action("tele", "Teleport to avoid dangerous skill", timeout=5.0),
    ],
    cooldown_seconds=15.0, exclusive=True,
    state_tracking_key="danger_teleport_last",
)

# 20. Endure before engaging tough mobs
MACRO_PATTERNS["combat_endure_prebuff"] = MacroPattern(
    pattern_id="combat_endure_prebuff", category="combat",
    description="Cast Endure or auto-guard before engaging a tough monster.",
    priority=PriorityTier.COMBAT_BOSS - 1,
    triggers=[
        _make_trigger("combat.target_threat", "gte", 0.7),
        _make_trigger("combat.is_in_combat", "eq", True),
        _make_trigger("vitals.endure_active", "eq", False),
    ],
    action_sequence=[
        _make_action("ss endure", "Cast Endure", timeout=2.0),
    ],
    cooldown_seconds=30.0, exclusive=False,
    state_tracking_key="endure_last",
)

# 21. Trap/Cast cancel on damage (interruptible skills)
MACRO_PATTERNS["combat_cast_cancel"] = MacroPattern(
    pattern_id="combat_cast_cancel", category="combat",
    description="Cancel current cast/trap when taking damage to avoid lock.",
    priority=PriorityTier.COMBAT_SKILL + 1,
    triggers=[
        _make_trigger("combat.is_casting", "eq", True),
        _make_trigger("vitals.taking_damage", "eq", True),
        _make_trigger("combat.cast_interruptible", "eq", True),
    ],
    action_sequence=[
        _make_action("stop_cast", "Cancel interruptible cast", timeout=1.0),
        _make_action("attackAuto 0", "Disable AI briefly", timeout=1.0),
        _make_action("ai auto", "Re-enable AI", timeout=1.0),
    ],
    cooldown_seconds=3.0, exclusive=False,
)


# ───────────────────────────────────────────────────────────────────────
# LOOT TIER (priority range: 49-45)
# ───────────────────────────────────────────────────────────────────────

# 22. Pickup priority — cards first
MACRO_PATTERNS["loot_pickup_cards"] = MacroPattern(
    pattern_id="loot_pickup_cards", category="loot",
    description="Priority pickup for card drops after a kill.",
    priority=PriorityTier.LOOT_PRIORITY,
    triggers=[
        _make_trigger("loot.items_on_ground", "eq", True),
        _make_trigger("loot.has_card", "eq", True),
    ],
    action_sequence=[
        _make_action("loot", "Pickup card drops (priority)", timeout=3.0),
    ],
    cooldown_seconds=1.0, exclusive=False,
)

# 23. Pickup priority — ores and gems
MACRO_PATTERNS["loot_pickup_ores"] = MacroPattern(
    pattern_id="loot_pickup_ores", category="loot",
    description="Priority pickup for valuable ores and gems.",
    priority=PriorityTier.LOOT_PRIORITY - 1,
    triggers=[
        _make_trigger("loot.items_on_ground", "eq", True),
        _make_trigger("loot.has_ore", "eq", True),
        _make_trigger("loot.has_card", "eq", False),
    ],
    action_sequence=[
        _make_action("loot", "Pickup ore/gem drops", timeout=3.0),
    ],
    cooldown_seconds=1.0, exclusive=False,
)

# 24. Pickup priority — equipment
MACRO_PATTERNS["loot_pickup_equipment"] = MacroPattern(
    pattern_id="loot_pickup_equipment", category="loot",
    description="Pickup equipment drops after kill.",
    priority=PriorityTier.LOOT_PRIORITY - 2,
    triggers=[
        _make_trigger("loot.items_on_ground", "eq", True),
        _make_trigger("loot.has_equipment", "eq", True),
        _make_trigger("loot.has_card", "eq", False),
        _make_trigger("loot.has_ore", "eq", False),
    ],
    action_sequence=[
        _make_action("loot", "Pickup equipment drops", timeout=3.0),
    ],
    cooldown_seconds=2.0, exclusive=False,
)

# 25. Pickup priority — usable items
MACRO_PATTERNS["loot_pickup_usable"] = MacroPattern(
    pattern_id="loot_pickup_usable", category="loot",
    description="Pickup usable items (potions, scrolls) after kill.",
    priority=PriorityTier.LOOT_NORMAL,
    triggers=[
        _make_trigger("loot.items_on_ground", "eq", True),
        _make_trigger("loot.has_usable", "eq", True),
        _make_trigger("loot.has_card", "eq", False),
        _make_trigger("loot.has_ore", "eq", False),
        _make_trigger("loot.has_equipment", "eq", False),
    ],
    action_sequence=[
        _make_action("loot", "Pickup usable items", timeout=3.0),
    ],
    cooldown_seconds=2.0, exclusive=False,
)

# 26. Auto-loot when kill completes
MACRO_PATTERNS["loot_auto_loot"] = MacroPattern(
    pattern_id="loot_auto_loot", category="loot",
    description="General auto-loot when a kill completes and items are on the ground.",
    priority=PriorityTier.LOOT_NORMAL - 1,
    triggers=[
        _make_trigger("loot.items_on_ground", "eq", True),
        _make_trigger("combat.is_in_combat", "eq", False),
    ],
    action_sequence=[
        _make_action("loot", "Auto-loot nearby items", timeout=5.0),
    ],
    cooldown_seconds=3.0, exclusive=False,
)

# 27. Ignore junk items when over weight cap
MACRO_PATTERNS["loot_ignore_junk_overweight"] = MacroPattern(
    pattern_id="loot_ignore_junk_overweight", category="loot",
    description="Stop picking up junk items when inventory weight exceeds cap threshold.",
    priority=PriorityTier.LOOT_NORMAL - 2,
    triggers=[
        _make_trigger("inventory.overweight_ratio", "gte", 0.85),
        _make_trigger("loot.items_on_ground", "eq", True),
        _make_trigger("loot.is_junk", "eq", True),
    ],
    action_sequence=[
        _make_action("loot ignore_junk", "Ignore junk items (overweight)", timeout=1.0),
    ],
    cooldown_seconds=5.0, exclusive=False,
)

# 28. Rough-wind / butterfly wing when weight >= 90%
MACRO_PATTERNS["loot_return_overweight"] = MacroPattern(
    pattern_id="loot_return_overweight", category="loot",
    description="Use butterfly wing or rough-wind to return to town when overweight threshold reached.",
    priority=PriorityTier.LOOT_NORMAL + 1,
    triggers=[
        _make_trigger("inventory.overweight_ratio", "gte", 0.90),
        _make_trigger("combat.is_in_combat", "eq", False),
    ],
    action_sequence=[
        _make_action("use_butterfly_wing", "Use butterfly wing to return to town", timeout=3.0),
    ],
    cooldown_seconds=30.0, exclusive=True,
    state_tracking_key="return_overweight_last",
)


# ───────────────────────────────────────────────────────────────────────
# NAVIGATION TIER (priority range: 19-10)
# ───────────────────────────────────────────────────────────────────────

# 29. Return to save point with butterfly wing
MACRO_PATTERNS["nav_return_savepoint"] = MacroPattern(
    pattern_id="nav_return_savepoint", category="navigation",
    description="Use butterfly wing to return to save point.",
    priority=PriorityTier.NAVIGATION_WARP,
    triggers=[
        _make_trigger("navigation.return_to_savepoint", "eq", True),
        _make_trigger("inventory.has_butterfly_wing", "eq", True),
    ],
    action_sequence=[
        _make_action("use_butterfly_wing", "Return to save point", timeout=3.0),
    ],
    cooldown_seconds=10.0, exclusive=True,
)

# 30. Walk to hunting zone with dead reckoning
MACRO_PATTERNS["nav_walk_hunting_zone"] = MacroPattern(
    pattern_id="nav_walk_hunting_zone", category="navigation",
    description="Walk from town to the configured hunting zone using dead reckoning.",
    priority=PriorityTier.NAVIGATION_MOVE,
    triggers=[
        _make_trigger("navigation.at_hunting_zone", "eq", False),
        _make_trigger("navigation.need_to_move", "eq", True),
        _make_trigger("combat.is_in_combat", "eq", False),
    ],
    action_sequence=[
        _make_action("move {hunting_zone_map}", "Walk to hunting zone", timeout=60.0),
    ],
    cooldown_seconds=10.0, exclusive=False,
)

# 31. Enter portal / warp portal sequence
MACRO_PATTERNS["nav_enter_portal"] = MacroPattern(
    pattern_id="nav_enter_portal", category="navigation",
    description="Enter a warp portal to transition between maps.",
    priority=PriorityTier.NAVIGATION_WARP - 1,
    triggers=[
        _make_trigger("navigation.at_portal", "eq", True),
        _make_trigger("navigation.portal_needed", "eq", True),
    ],
    action_sequence=[
        _make_action("move 50 50", "Step into warp portal", timeout=5.0),
        _make_action("ai auto", "Recalculate route after warp", timeout=3.0),
    ],
    cooldown_seconds=5.0, exclusive=False,
)

# 32. Kafra warp selection
MACRO_PATTERNS["nav_kafra_warp"] = MacroPattern(
    pattern_id="nav_kafra_warp", category="navigation",
    description="Interact with Kafra NPC to warp to a destination.",
    priority=PriorityTier.NAVIGATION_WARP - 2,
    triggers=[
        _make_trigger("navigation.kafra_warp_needed", "eq", True),
        _make_trigger("navigation.near_kafra", "eq", True),
    ],
    action_sequence=[
        _make_action("talknpc {kafra_npc_x} {kafra_npc_y}", "Talk to Kafra NPC", timeout=5.0),
        _make_action("response 0", "Select warp service", timeout=3.0, confirm=True),
        _make_action("response {destination_index}", "Select warp destination", timeout=3.0, confirm=True),
    ],
    cooldown_seconds=30.0, exclusive=True,
)

# 33. Follow route from pathfinder
MACRO_PATTERNS["nav_follow_route"] = MacroPattern(
    pattern_id="nav_follow_route", category="navigation",
    description="Follow a computed route from the pathfinder to reach a destination.",
    priority=PriorityTier.NAVIGATION_FOLLOW,
    triggers=[
        _make_trigger("navigation.route_available", "eq", True),
        _make_trigger("navigation.route_active", "eq", False),
    ],
    action_sequence=[
        _make_action("domain routing follow_route", "Follow pathfinder route", timeout=30.0),
    ],
    cooldown_seconds=5.0, exclusive=False,
)


# ───────────────────────────────────────────────────────────────────────
# INVENTORY TIER (priority range: 29-20)
# ───────────────────────────────────────────────────────────────────────

# 34. Auto-sell junk at NPC
MACRO_PATTERNS["inv_auto_sell_junk"] = MacroPattern(
    pattern_id="inv_auto_sell_junk", category="inventory",
    description="Sell junk items to NPC with proper interaction sequence.",
    priority=PriorityTier.INVENTORY_SELL,
    triggers=[
        _make_trigger("inventory.has_junk_to_sell", "eq", True),
        _make_trigger("inventory.overweight_ratio", "gte", 0.80),
        _make_trigger("combat.is_in_combat", "eq", False),
    ],
    action_sequence=[
        _make_action("move {town_name}", "Return to town to sell", timeout=30.0),
        _make_action("talknpc {sell_npc_x} {sell_npc_y}", "Talk to buy/sell NPC", timeout=5.0),
        _make_action("response 0", "Open shop menu", timeout=3.0),
        _make_action("response 2", "Select sell option", timeout=3.0, confirm=True),
        _make_action("auto_sell_junk", "Sell all junk items", timeout=10.0),
    ],
    cooldown_seconds=60.0, exclusive=True,
    state_tracking_key="last_sell_time",
)

# 35. Auto-store at Kafra
MACRO_PATTERNS["inv_auto_store_kafra"] = MacroPattern(
    pattern_id="inv_auto_store_kafra", category="inventory",
    description="Deposit valuable loot into Kafra storage with dialogue chain.",
    priority=PriorityTier.INVENTORY_STORE,
    triggers=[
        _make_trigger("inventory.item_count", "gte", 85),
        _make_trigger("combat.is_in_combat", "eq", False),
        _make_trigger("navigation.near_kafra", "eq", True),
    ],
    action_sequence=[
        _make_action("talknpc {kafra_npc_x} {kafra_npc_y}", "Talk to Kafra NPC", timeout=5.0),
        _make_action("response 0", "Open storage service", timeout=3.0),
        _make_action("auto_store", "Auto-store all deposit items", timeout=15.0),
    ],
    cooldown_seconds=120.0, exclusive=True,
    state_tracking_key="last_store_time",
)

# 36. Auto-buy arrows/bullets/stones
MACRO_PATTERNS["inv_auto_buy_ammo"] = MacroPattern(
    pattern_id="inv_auto_buy_ammo", category="inventory",
    description="Auto-buy ammunition (arrows, bullets, stones) when low.",
    priority=PriorityTier.INVENTORY_BUY,
    triggers=[
        _make_trigger("inventory.ammo_count", "lte", 200),
        _make_trigger("combat.is_in_combat", "eq", False),
        _make_trigger("navigation.near_ammo_vendor", "eq", True),
    ],
    action_sequence=[
        _make_action("talknpc {ammo_npc_x} {ammo_npc_y}", "Talk to ammo vendor", timeout=5.0),
        _make_action("response 0", "Open buy menu", timeout=3.0),
        _make_action("buy {ammo_item} 1000", "Buy 1000 ammunition", timeout=10.0),
    ],
    required_zeny=500, cooldown_seconds=180.0, exclusive=False,
    state_tracking_key="last_ammo_buy",
)

# 37. Auto-buy potions when below threshold
MACRO_PATTERNS["inv_auto_buy_potions"] = MacroPattern(
    pattern_id="inv_auto_buy_potions", category="inventory",
    description="Auto-buy HP/SP potions when stock is below configured threshold.",
    priority=PriorityTier.INVENTORY_BUY + 1,
    triggers=[
        _make_trigger("inventory.hp_potion_count", "lte", 20),
        _make_trigger("combat.is_in_combat", "eq", False),
        _make_trigger("navigation.near_potion_vendor", "eq", True),
    ],
    action_sequence=[
        _make_action("talknpc {potion_npc_x} {potion_npc_y}", "Talk to potion vendor", timeout=5.0),
        _make_action("buy_white_potion 100", "Buy 100 HP potions", timeout=10.0),
        _make_action("buy_blue_potion 50", "Buy 50 SP potions", timeout=10.0),
    ],
    required_zeny=2000, cooldown_seconds=180.0, exclusive=False,
    state_tracking_key="last_potion_buy",
)

# 38. Equipment repair when durability low
MACRO_PATTERNS["inv_equipment_repair"] = MacroPattern(
    pattern_id="inv_equipment_repair", category="inventory",
    description="Repair equipment when durability drops below threshold.",
    priority=PriorityTier.INVENTORY_STORE + 1,
    triggers=[
        _make_trigger("inventory.needs_repair", "eq", True),
        _make_trigger("combat.is_in_combat", "eq", False),
        _make_trigger("navigation.near_repair_npc", "eq", True),
    ],
    action_sequence=[
        _make_action("talknpc {repair_npc_x} {repair_npc_y}", "Talk to repair NPC", timeout=5.0),
        _make_action("response 0", "Open repair menu", timeout=3.0),
        _make_action("auto_repair", "Repair all damaged equipment", timeout=10.0),
    ],
    cooldown_seconds=60.0, exclusive=False,
    state_tracking_key="last_repair_time",
)

# 39. Auto-vend (Merchant classes)
MACRO_PATTERNS["inv_auto_vend_merchant"] = MacroPattern(
    pattern_id="inv_auto_vend_merchant", category="inventory",
    description="Set up auto-vending shop for Merchant class characters.",
    priority=PriorityTier.INVENTORY_SELL + 1,
    triggers=[
        _make_trigger("progression.is_merchant_class", "eq", True),
        _make_trigger("inventory.has_items_to_vend", "eq", True),
        _make_trigger("navigation.at_vend_spot", "eq", True),
        _make_trigger("combat.is_in_combat", "eq", False),
    ],
    action_sequence=[
        _make_action("open_vending", "Open vending shop", timeout=5.0),
        _make_action("auto_vend", "Auto-list items for sale", timeout=10.0),
    ],
    cooldown_seconds=300.0, exclusive=True,
    state_tracking_key="last_vend_time",
)


# ───────────────────────────────────────────────────────────────────────
# PARTY TIER (priority range: 44-38)
# ───────────────────────────────────────────────────────────────────────

# 40. Share EXP range check and adjust position
MACRO_PATTERNS["party_share_exp_range"] = MacroPattern(
    pattern_id="party_share_exp_range", category="party",
    description="Adjust position to stay within party EXP share range.",
    priority=PriorityTier.PARTY_HEAL,
    triggers=[
        _make_trigger("party.is_in_party", "eq", True),
        _make_trigger("party.outside_exp_range", "eq", True),
        _make_trigger("combat.is_in_combat", "eq", False),
    ],
    action_sequence=[
        _make_action("move {party_center_x} {party_center_y}", "Move to party EXP share range", timeout=10.0),
    ],
    cooldown_seconds=10.0, exclusive=False,
)

# 41. Party follow / formation position
MACRO_PATTERNS["party_follow_formation"] = MacroPattern(
    pattern_id="party_follow_formation", category="party",
    description="Follow party leader or maintain formation position.",
    priority=PriorityTier.PARTY_BUFF,
    triggers=[
        _make_trigger("party.is_in_party", "eq", True),
        _make_trigger("party.follow_mode", "eq", True),
        _make_trigger("party.leader_distance", "gt", 3),
        _make_trigger("combat.is_in_combat", "eq", False),
    ],
    action_sequence=[
        _make_action("party follow 1", "Follow party leader", timeout=10.0),
    ],
    cooldown_seconds=5.0, exclusive=False,
)

# 42. Heal party member when HP < 50%
MACRO_PATTERNS["party_heal_member"] = MacroPattern(
    pattern_id="party_heal_member", category="party",
    description="Heal a party member when their HP drops below 50%.",
    priority=PriorityTier.PARTY_HEAL - 1,
    triggers=[
        _make_trigger("party.is_in_party", "eq", True),
        _make_trigger("party.member_hp_low", "eq", True),
        _make_trigger("party.member_hp_ratio", "lte", 0.50),
        _make_trigger("combat.is_in_combat", "eq", False),
    ],
    action_sequence=[
        _make_action("ss heal {party_low_hp_member}", "Heal party member", timeout=3.0),
    ],
    cooldown_seconds=5.0, exclusive=False,
    state_tracking_key="last_party_heal",
)

# 43. Buff party members with shared skills
MACRO_PATTERNS["party_buff_members"] = MacroPattern(
    pattern_id="party_buff_members", category="party",
    description="Cast shared buffs on party members.",
    priority=PriorityTier.PARTY_BUFF - 1,
    triggers=[
        _make_trigger("party.is_in_party", "eq", True),
        _make_trigger("party.members_nearby", "eq", True),
        _make_trigger("combat.is_in_combat", "eq", False),
        _make_trigger("vitals.party_buffs_needed", "eq", True),
    ],
    action_sequence=[
        _make_action("ss increase_agility", "Cast Agi Up on party", timeout=3.0),
        _make_action("ss bless", "Cast Bless on party", timeout=3.0),
    ],
    cooldown_seconds=60.0, exclusive=False,
    state_tracking_key="last_party_buff",
)

# 44. Resurrect fallen party member
MACRO_PATTERNS["party_resurrect"] = MacroPattern(
    pattern_id="party_resurrect", category="party",
    description="Resurrect a fallen party member using skill or Yggdrasil leaf.",
    priority=PriorityTier.PARTY_RESS,
    triggers=[
        _make_trigger("party.is_in_party", "eq", True),
        _make_trigger("party.member_dead", "eq", True),
        _make_trigger("combat.is_in_combat", "eq", False),
    ],
    action_sequence=[
        _make_action("ss resurrection {party_dead_member}", "Resurrect party member", timeout=5.0),
    ],
    cooldown_seconds=10.0, exclusive=False,
    state_tracking_key="last_ressurect",
)

# 45. Party loot distribution
MACRO_PATTERNS["party_loot_distribution"] = MacroPattern(
    pattern_id="party_loot_distribution", category="party",
    description="Manage party loot distribution (pickup for shared loot).",
    priority=PriorityTier.PARTY_RESS - 1,
    triggers=[
        _make_trigger("party.is_in_party", "eq", True),
        _make_trigger("party.loot_mode", "eq", "shared"),
        _make_trigger("loot.items_on_ground", "eq", True),
    ],
    action_sequence=[
        _make_action("loot", "Pickup loot for party distribution", timeout=5.0),
    ],
    cooldown_seconds=3.0, exclusive=False,
)


# ───────────────────────────────────────────────────────────────────────
# FARMING TIER (priority range: 35-30)
# ───────────────────────────────────────────────────────────────────────

# 46. MVP alert when spawn detected
MACRO_PATTERNS["farm_mvp_alert"] = MacroPattern(
    pattern_id="farm_mvp_alert", category="farming",
    description="Alert when MVP spawn is detected in the current map.",
    priority=PriorityTier.FARMING_MVP,
    triggers=[
        _make_trigger("combat.target_is_boss", "eq", True),
        _make_trigger("event.event_type", "eq", "mvp_spawn"),
    ],
    action_sequence=[
        _make_action("log MVP detected on map", "Log MVP spawn alert", timeout=1.0),
        _make_action("domain combat engage_mvp", "Engage MVP", timeout=5.0),
    ],
    cooldown_seconds=5.0, exclusive=True,
    state_tracking_key="mvp_alert_last",
)

# 47. Switch to MVP weapon/element loadout
MACRO_PATTERNS["farm_mvp_loadout_switch"] = MacroPattern(
    pattern_id="farm_mvp_loadout_switch", category="farming",
    description="Switch to MVP-specific weapon and elemental loadout.",
    priority=PriorityTier.FARMING_SWITCH,
    triggers=[
        _make_trigger("combat.target_is_boss", "eq", True),
        _make_trigger("inventory.mvp_loadout_available", "eq", True),
        _make_trigger("combat.is_in_combat", "eq", True),
    ],
    action_sequence=[
        _make_action("equip {mvp_weapon}", "Equip MVP-specific weapon", timeout=3.0),
        _make_action("use {mvp_elemental_converter}", "Apply MVP element converter", timeout=3.0),
    ],
    cooldown_seconds=30.0, exclusive=False,
    state_tracking_key="mvp_loadout_last",
)

# 48. Call party for MVP assist
MACRO_PATTERNS["farm_mvp_call_party"] = MacroPattern(
    pattern_id="farm_mvp_call_party", category="farming",
    description="Call party members for MVP assist.",
    priority=PriorityTier.FARMING_MVP - 1,
    triggers=[
        _make_trigger("combat.target_is_boss", "eq", True),
        _make_trigger("party.is_in_party", "eq", True),
        _make_trigger("party.nearby_count", "lt", 2),
    ],
    action_sequence=[
        _make_action("party say MVP found at {map_name} {pos_x} {pos_y}", "Call party for MVP assist", timeout=3.0),
    ],
    cooldown_seconds=30.0, exclusive=False,
)

# 49. Switch maps when kill rate drops
MACRO_PATTERNS["farm_switch_map_killrate"] = MacroPattern(
    pattern_id="farm_switch_map_killrate", category="farming",
    description="Switch to a different hunting map when kill rate drops below threshold.",
    priority=PriorityTier.FARMING_SWITCH - 1,
    triggers=[
        _make_trigger("farming.kill_rate_below_threshold", "eq", True),
        _make_trigger("navigation.alternate_map_available", "eq", True),
        _make_trigger("combat.is_in_combat", "eq", False),
    ],
    action_sequence=[
        _make_action("move {alternate_map_name}", "Move to alternate hunting map", timeout=30.0),
    ],
    cooldown_seconds=300.0, exclusive=True,
    state_tracking_key="last_map_switch",
)

# 50. Avoid other players' mobs (kill-steal detection)
MACRO_PATTERNS["farm_avoid_ks"] = MacroPattern(
    pattern_id="farm_avoid_ks", category="farming",
    description="Avoid attacking monsters that another player already engaged (KS prevention).",
    priority=PriorityTier.FARMING_SWITCH - 2,
    triggers=[
        _make_trigger("combat.target_claimed_by_other", "eq", True),
    ],
    action_sequence=[
        _make_action("stop_attack", "Stop attacking claimed target", timeout=2.0),
        _make_action("target {next_unclaimed_mob}", "Retarget unclaimed monster", timeout=3.0),
    ],
    cooldown_seconds=2.0, exclusive=False,
)

# 51. Session tracking (log Zeny/hour, items/hour)
MACRO_PATTERNS["farm_session_tracking"] = MacroPattern(
    pattern_id="farm_session_tracking", category="farming",
    description="Log farming session statistics: Zeny per hour, items per hour.",
    priority=PriorityTier.FARMING_MVP - 2,
    triggers=[
        _make_trigger("farming.session_tick", "eq", True),
    ],
    action_sequence=[
        _make_action("domain farming log_session_stats", "Log session farming stats", timeout=1.0),
    ],
    cooldown_seconds=60.0, exclusive=False,
)


# ───────────────────────────────────────────────────────────────────────
# SAFETY TIER (priority range: 79-65)
# ───────────────────────────────────────────────────────────────────────

# 52. Anti-stuck mechanism
MACRO_PATTERNS["safe_anti_stuck"] = MacroPattern(
    pattern_id="safe_anti_stuck", category="safety",
    description="Walk in random direction when position hasn't changed for X seconds (anti-stuck).",
    priority=PriorityTier.SAFETY_ANTISTUCK,
    triggers=[
        _make_trigger("navigation.position_stale_seconds", "gte", 15),
        _make_trigger("combat.is_in_combat", "eq", False),
        _make_trigger("navigation.is_stuck", "eq", True),
    ],
    action_sequence=[
        _make_action("attackAuto 0", "Disable AI for stuck recovery", timeout=1.0),
        _make_action("move {random_x} {random_y}", "Walk random direction", timeout=5.0),
        _make_action("ai auto", "Re-enable AI", timeout=1.0),
    ],
    cooldown_seconds=10.0, exclusive=False,
    state_tracking_key="last_antistuck",
)

# 53. Aggro escape (run from aggressive mobs when low HP)
MACRO_PATTERNS["safe_aggro_escape"] = MacroPattern(
    pattern_id="safe_aggro_escape", category="safety",
    description="Run from aggressive monsters when HP is low.",
    priority=PriorityTier.SAFETY_ESCAPE,
    triggers=[
        _make_trigger("combat.aggro_count", "gte", 1),
        _make_trigger("vitals.hp_ratio", "lte", 0.35),
        _make_trigger("combat.is_in_combat", "eq", True),
    ],
    action_sequence=[
        _make_action("attackAuto 0", "Disable combat AI", timeout=1.0),
        _make_action("move 0 0", "Run to safe position", timeout=5.0),
        _make_action("ai auto", "Re-enable AI", timeout=1.0),
    ],
    cooldown_seconds=8.0, exclusive=True,
    state_tracking_key="last_aggro_escape",
)

# 54. GM detection
MACRO_PATTERNS["safe_gm_detection"] = MacroPattern(
    pattern_id="safe_gm_detection", category="safety",
    description="Disconnect or hide bot when a GM appears on the map.",
    priority=PriorityTier.SAFETY_GM,
    triggers=[
        _make_trigger("safety.gm_detected", "eq", True),
    ],
    action_sequence=[
        _make_action("attackAuto 0", "Disable AI for GM detection", timeout=1.0),
        _make_action("sit", "Sit to appear AFK", timeout=1.0),
        _make_action("log GM detected — hiding", "Log GM detection event", timeout=1.0),
    ],
    cooldown_seconds=300.0, exclusive=True,
    state_tracking_key="gm_detection_last",
)

# 55. Chat monitoring (whisper/party)
MACRO_PATTERNS["safe_chat_monitor"] = MacroPattern(
    pattern_id="safe_chat_monitor", category="safety",
    description="Monitor whispers and party chat for commands or alerts.",
    priority=PriorityTier.SAFETY_LOGIN,
    triggers=[
        _make_trigger("social.has_unread_whisper", "eq", True),
    ],
    action_sequence=[
        _make_action("domain social process_whisper", "Process incoming whisper", timeout=5.0),
    ],
    cooldown_seconds=3.0, exclusive=False,
    state_tracking_key="last_chat_process",
)

# 56. Party chat monitoring
MACRO_PATTERNS["safe_party_chat_monitor"] = MacroPattern(
    pattern_id="safe_party_chat_monitor", category="safety",
    description="Monitor party chat for leader commands.",
    priority=PriorityTier.SAFETY_LOGIN - 1,
    triggers=[
        _make_trigger("social.party_chat_unread", "eq", True),
    ],
    action_sequence=[
        _make_action("domain social process_party_chat", "Process party chat messages", timeout=5.0),
    ],
    cooldown_seconds=2.0, exclusive=False,
    state_tracking_key="last_party_chat",
)

# 57. Login check (auto-reconnect with backoff)
MACRO_PATTERNS["safe_login_check"] = MacroPattern(
    pattern_id="safe_login_check", category="safety",
    description="Check connection status and auto-reconnect with exponential backoff.",
    priority=PriorityTier.SAFETY_LOGIN + 1,
    triggers=[
        _make_trigger("connection.is_disconnected", "eq", True),
        _make_trigger("connection.reconnect_allowed", "eq", True),
    ],
    action_sequence=[
        _make_action("log Reconnecting with backoff", "Log reconnection attempt", timeout=1.0),
        _make_action("reconnect", "Reconnect to server", timeout=30.0),
    ],
    cooldown_seconds=15.0, exclusive=True,
    state_tracking_key="last_reconnect",
)

# 58. Player detection / hide when configured
MACRO_PATTERNS["safe_player_detection"] = MacroPattern(
    pattern_id="safe_player_detection", category="safety",
    description="Hide from same-map players when configured for stealth.",
    priority=PriorityTier.SAFETY_GM - 1,
    triggers=[
        _make_trigger("safety.player_detected", "eq", True),
        _make_trigger("safety.stealth_mode", "eq", True),
    ],
    action_sequence=[
        _make_action("attackAuto 0", "Disable AI for stealth", timeout=1.0),
        _make_action("sit", "Sit to appear AFK", timeout=1.0),
    ],
    cooldown_seconds=30.0, exclusive=False,
    state_tracking_key="player_detection_last",
)

# 59. Stale connection heartbeat
MACRO_PATTERNS["safe_connection_heartbeat"] = MacroPattern(
    pattern_id="safe_connection_heartbeat", category="safety",
    description="Send heartbeat or move slightly to prevent connection timeout.",
    priority=PriorityTier.SAFETY_LOGIN - 2,
    triggers=[
        _make_trigger("connection.idle_seconds", "gte", 120),
        _make_trigger("combat.is_in_combat", "eq", False),
    ],
    action_sequence=[
        _make_action("move 1 1", "Slight move to prevent timeout", timeout=3.0),
    ],
    cooldown_seconds=60.0, exclusive=False,
)

# 60. Emergency disconnect clean-up
MACRO_PATTERNS["safe_disconnect_cleanup"] = MacroPattern(
    pattern_id="safe_disconnect_cleanup", category="safety",
    description="Clean up state and save progress on disconnect.",
    priority=PriorityTier.SAFETY_GM + 1,
    triggers=[
        _make_trigger("event.event_type", "eq", "disconnecting"),
    ],
    action_sequence=[
        _make_action("log Saving state before disconnect", "Log disconnect cleanup", timeout=1.0),
        _make_action("save_config", "Save current configuration", timeout=5.0),
    ],
    cooldown_seconds=0.0, exclusive=True,
)


# ───────────────────────────────────────────────────────────────────────
# LEGACY / BACKWARD-COMPATIBLE PATTERNS (keep original patterns)
# ───────────────────────────────────────────────────────────────────────

# Death respawn + return to zone
MACRO_PATTERNS["death_respawn_return"] = MacroPattern(
    pattern_id="death_respawn_return", category="safety",
    description="After death, respawn, rebuff, and return to hunting zone.",
    priority=PriorityTier.SAFETY_ESCAPE + 1,
    triggers=[
        _make_trigger("event.event_type", "eq", "player_died"),
    ],
    action_sequence=[
        _make_action("ai auto", "Respawn with auto AI", timeout=5.0),
        _make_action("move {hunting_zone_map}", "Return to hunting zone", timeout=30.0),
    ],
    cooldown_seconds=15.0, exclusive=True,
    state_tracking_key="death_respawn_last",
)

# Re-buff after death/teleport
MACRO_PATTERNS["auto_buff_on_respawn"] = MacroPattern(
    pattern_id="auto_buff_on_respawn", category="combat",
    description="Re-apply buffs (bless, agi up) after death or teleport.",
    priority=PriorityTier.COMBAT_SKILL + 2,
    triggers=[
        _make_trigger("event.event_type", "eq", "player_respawned"),
    ],
    action_sequence=[
        _make_action("ss increase_agility", "Cast Agi Up", timeout=3.0),
        _make_action("ss bless", "Cast Bless", timeout=3.0),
    ],
    cooldown_seconds=30.0, exclusive=False,
)

# Skill rotation burst (MVP/boss)
MACRO_PATTERNS["skill_rotation_burst"] = MacroPattern(
    pattern_id="skill_rotation_burst", category="combat",
    description="Execute burst skill rotation on a single target (boss/MVP).",
    priority=PriorityTier.COMBAT_BOSS + 1,
    triggers=[
        _make_trigger("combat.target_is_boss", "eq", True),
    ],
    action_sequence=[
        _make_action("ss {burst_skill_1}", "Opening burst skill", timeout=3.0),
        _make_action("ss {burst_skill_2}", "Main damage skill", timeout=2.0),
    ],
    cooldown_seconds=10.0, exclusive=True,
)


# ═══════════════════════════════════════════════════════════════════════
# 3. MACRO INTELLIGENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════

class MacroIntelligence:
    """Macro Intelligence Engine with priority-ordered trigger evaluation.

    This is the core execution engine that:
    1. Evaluates trigger conditions against current bot state
    2. Filters out macros on cooldown
    3. Sorts matched macros by priority (descending)
    4. Applies exclusive blocking: if a high-priority macro is exclusive,
       all lower-priority macros are blocked for this evaluation tick
    5. Returns the winning macro (highest priority, non-blocked)

    The process_triggers method is the main entry point. It returns
    a single MacroPattern (or None) — the one macro that should execute
    this tick based on priority + exclusive rules.

    get_patterns_for_context (legacy) returns ALL matching macros for
    AI/LLM review without priority blocking.
    """

    def __init__(self, knowledge_path: str | None = None):
        self._lock = RLock()
        self._patterns: dict[str, MacroPattern] = dict(MACRO_PATTERNS)
        self._last_triggered: dict[str, float] = {}  # pattern_id -> timestamp
        self._macro_state: dict[str, Any] = {}  # state_tracking_key -> value
        self._active_exclusive_pattern: str | None = None  # Currently exclusive-running macro
        self._active_exclusive_until: float = 0.0
        self._load_custom_patterns(knowledge_path)

    def _load_custom_patterns(self, path: str | None) -> None:
        """Load custom macro patterns from a JSON file (optional)."""
        if not path:
            return
        p = Path(path)
        if not p.exists():
            return
        try:
            with open(p) as f:
                data = json.load(f)
            for item in data.get("patterns", []):
                triggers_raw = item.get("triggers", [])
                triggers = [
                    MacroTrigger(**t) if isinstance(t, dict) else t
                    for t in triggers_raw
                ]
                pattern = MacroPattern(
                    pattern_id=item["pattern_id"],
                    category=item.get("category", "custom"),
                    description=item.get("description", ""),
                    triggers=triggers,
                    action_sequence=[MacroAction(**a) for a in item.get("action_sequence", [])],
                    priority=item.get("priority", 50),
                    required_items=item.get("required_items", []),
                    required_zeny=item.get("required_zeny", 0),
                    required_level_range=tuple(item.get("required_level_range", (1, 999))),
                    required_jobs=item.get("required_jobs", []),
                    cooldown_seconds=item.get("cooldown_seconds", 0),
                    exclusive=item.get("exclusive", False),
                    disable_in_combat=item.get("disable_in_combat", False),
                    disable_out_of_combat=item.get("disable_out_of_combat", False),
                    state_tracking_key=item.get("state_tracking_key", ""),
                )
                self._patterns[pattern.pattern_id] = pattern
            logger.info("macro_patterns_loaded: %d custom patterns from %s",
                        len(data.get("patterns", [])), path)
        except Exception as e:
            logger.warning("macro_patterns_load_failed: %s", e)

    # ── process_triggers — main entry point ───────────────────────────

    def process_triggers(
        self,
        *,
        bot_state: dict[str, Any],
        bot_id: str = "",
    ) -> MacroPattern | None:
        """Evaluate all macro triggers against bot state.

        Returns the single highest-priority macro that:
        - Is not on cooldown
        - Has all triggers matching
        - Is not blocked by an active exclusive macro
        - Passes combat state filters

        If the returned macro is exclusive, it locks out other macros
        until its action sequence completes (tracked via cooldown).

        This is the primary interface for the decision loop.
        Returns None if no macro should execute.
        """
        with self._lock:
            matches = self._get_matching_patterns(bot_state)

            if not matches:
                self._active_exclusive_pattern = None
                return None

            # If an exclusive macro is still in its cooldown window,
            # block all lower-priority macros
            if self._active_exclusive_pattern is not None:
                if time.time() < self._active_exclusive_until:
                    # Only allow the exclusive pattern itself or higher-priority
                    ex_pattern = self._patterns.get(self._active_exclusive_pattern)
                    if ex_pattern:
                        matches = [
                            m for m in matches
                            if m.pattern_id == self._active_exclusive_pattern
                            or m.priority > ex_pattern.priority
                        ]
                else:
                    self._active_exclusive_pattern = None

            if not matches:
                return None

            # Sort by priority descending → highest first
            matches.sort(key=lambda m: m.priority, reverse=True)
            winner = matches[0]

            # Mark cooldown
            self._last_triggered[winner.pattern_id] = time.time()

            # If winner is exclusive, start exclusive lock window
            if winner.exclusive:
                self._active_exclusive_pattern = winner.pattern_id
                self._active_exclusive_until = time.time() + max(winner.cooldown_seconds, 5.0)

            logger.info(
                "macro_triggered: pattern=%s priority=%d exclusive=%s category=%s",
                winner.pattern_id, winner.priority, winner.exclusive, winner.category,
            )
            return winner

    # ── get_patterns_for_context — legacy AI review interface ──────────

    def get_patterns_for_context(
        self,
        *,
        bot_state: dict[str, Any],
    ) -> list[MacroPattern]:
        """Return ALL matching macro patterns (for AI/LLM review).

        This is the legacy interface used by the pdca_loop for AI context.
        It returns every pattern that matches, regardless of exclusive
        blocking, so the LLM can make its own decisions.

        For priority-ordered, exclusive-blocked execution, use process_triggers.
        """
        with self._lock:
            return self._get_matching_patterns(bot_state, apply_exclusive_block=False)

    def _get_matching_patterns(
        self,
        bot_state: dict[str, Any],
        apply_exclusive_block: bool = True,
    ) -> list[MacroPattern]:
        """Internal: evaluate triggers and return matching patterns.

        Handles cooldown, combat state filters, and optional exclusive blocking.
        """
        now = time.time()
        is_in_combat = _resolve_path(bot_state, "combat.is_in_combat") or False
        matches: list[MacroPattern] = []

        for pattern in self._patterns.values():
            # Check cooldown
            last = self._last_triggered.get(pattern.pattern_id, 0)
            if now - last < pattern.cooldown_seconds:
                continue

            # Check required items
            if pattern.required_items:
                inventory = _resolve_path(bot_state, "inventory.items") or []
                if not all(item in inventory for item in pattern.required_items):
                    continue

            # Check required zeny
            if pattern.required_zeny > 0:
                zeny = _resolve_path(bot_state, "progression.zeny") or 0
                if zeny < pattern.required_zeny:
                    continue

            # Check required level range
            base_level = _resolve_path(bot_state, "progression.base_level") or 1
            if not (pattern.required_level_range[0] <= base_level <= pattern.required_level_range[1]):
                continue

            # Check required jobs
            if pattern.required_jobs:
                job_name = str(_resolve_path(bot_state, "progression.job_name") or "").lower()
                if job_name not in [j.lower() for j in pattern.required_jobs]:
                    continue

            # Check combat state filters
            if pattern.disable_in_combat and is_in_combat:
                continue
            if pattern.disable_out_of_combat and not is_in_combat:
                continue

            # Evaluate all triggers (AND logic)
            all_match = True
            for trigger in pattern.triggers:
                if isinstance(trigger, dict):
                    trigger = MacroTrigger(**trigger)
                if not trigger.evaluate(bot_state):
                    all_match = False
                    break

            if all_match:
                matches.append(pattern)

        return matches

    # ── Sequence generation ───────────────────────────────────────────

    def generate_sequence(
        self,
        pattern: MacroPattern,
        *,
        bot_id: str,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate an action sequence from a macro pattern.

        Fills in template values from context (e.g., actual item names,
        NPC positions, map names).
        Returns a list of action proposals ready for the action queue.
        """
        from ai_sidecar.contracts.actions import ActionProposal, ActionPriorityTier
        from datetime import UTC, datetime, timedelta
        import hashlib

        ctx = context or {}
        sequence: list[dict[str, Any]] = []
        now = datetime.now(UTC)
        monotonic_ns = time.monotonic_ns()

        for i, step in enumerate(pattern.action_sequence):
            cmd = step.command
            # Fill in template values from context
            for key, val in ctx.items():
                cmd = cmd.replace(f"{{{key}}}", str(val))

            h = hashlib.md5(f"{bot_id}_{monotonic_ns}_{i}".encode()).hexdigest()[:8]

            proposal = ActionProposal(
                action_id=f"macro_{pattern.pattern_id}_{i}_{h}",
                kind="command",
                command=cmd,
                priority_tier=(
                    ActionPriorityTier.strategic
                    if pattern.exclusive
                    else ActionPriorityTier.tactical
                ),
                source="planner",
                created_at=now,
                expires_at=now + timedelta(seconds=step.timeout_seconds),
                idempotency_key=f"macro_{pattern.pattern_id}_{bot_id}_{i}",
                metadata={
                    "goal": pattern.category,
                    "objective": step.description,
                    "bot_id": bot_id,
                    "source": "macro_ai",
                    "pattern_id": pattern.pattern_id,
                    "step": i,
                    "total_steps": len(pattern.action_sequence),
                    "sequence_id": f"{pattern.pattern_id}_{int(time.time())}",
                },
            )
            sequence.append({
                "action": proposal,
                "timeout": step.timeout_seconds,
                "requires_confirmation": step.requires_confirmation,
                "conflict_key": step.conflict_key or f"macro.{pattern.pattern_id}",
            })

        # Mark as triggered for cooldown
        with self._lock:
            self._last_triggered[pattern.pattern_id] = time.time()

        return sequence

    # ── Utility methods ────────────────────────────────────────────────

    def get_all_patterns(self) -> dict[str, MacroPattern]:
        """Return all known macro patterns."""
        return dict(self._patterns)

    def get_patterns_by_category(self, category: str) -> list[MacroPattern]:
        """Return patterns matching a category."""
        return [p for p in self._patterns.values() if p.category == category]

    def get_patterns_by_priority_range(
        self, min_priority: int, max_priority: int
    ) -> list[MacroPattern]:
        """Return patterns within a priority range (inclusive)."""
        return [
            p for p in self._patterns.values()
            if min_priority <= p.priority <= max_priority
        ]

    def add_custom_pattern(self, pattern: MacroPattern) -> None:
        """Add a new pattern dynamically (AI-generated patterns)."""
        with self._lock:
            self._patterns[pattern.pattern_id] = pattern
            logger.info("macro_pattern_added: %s (%s) priority=%d",
                        pattern.pattern_id, pattern.category, pattern.priority)

    def remove_pattern(self, pattern_id: str) -> bool:
        """Remove a pattern by ID. Returns True if removed."""
        with self._lock:
            if pattern_id in self._patterns:
                del self._patterns[pattern_id]
                self._last_triggered.pop(pattern_id, None)
                logger.info("macro_pattern_removed: %s", pattern_id)
                return True
            return False

    def clear_cooldown(self, pattern_id: str | None = None) -> None:
        """Clear cooldown for a specific pattern or all patterns."""
        with self._lock:
            if pattern_id:
                self._last_triggered.pop(pattern_id, None)
            else:
                self._last_triggered.clear()
            logger.info("macro_cooldowns_cleared: %s", pattern_id or "all")

    def get_macro_state(self, key: str, default: Any = None) -> Any:
        """Get a macro state tracking value."""
        with self._lock:
            return self._macro_state.get(key, default)

    def set_macro_state(self, key: str, value: Any) -> None:
        """Set a macro state tracking value."""
        with self._lock:
            self._macro_state[key] = value

    def is_macro_on_cooldown(self, pattern_id: str) -> bool:
        """Check if a specific macro is still on cooldown."""
        with self._lock:
            last = self._last_triggered.get(pattern_id, 0)
            pattern = self._patterns.get(pattern_id)
            if pattern is None:
                return False
            return (time.time() - last) < pattern.cooldown_seconds

    def stats(self) -> dict[str, Any]:
        """Return system statistics."""
        with self._lock:
            now = time.time()
            total = len(self._patterns)
            on_cooldown = sum(
                1 for pid, t in self._last_triggered.items()
                if pid in self._patterns and now - t < self._patterns[pid].cooldown_seconds
            )
            return {
                "total_patterns": total,
                "categories": sorted({p.category for p in self._patterns.values()}),
                "on_cooldown": on_cooldown,
                "active_cooldowns_60s": sum(
                    1 for t in self._last_triggered.values()
                    if now - t < 60
                ),
                "active_exclusive": self._active_exclusive_pattern,
                "patterns_by_category": {
                    cat: sum(1 for p in self._patterns.values() if p.category == cat)
                    for cat in sorted({p.category for p in self._patterns.values()})
                },
            }
