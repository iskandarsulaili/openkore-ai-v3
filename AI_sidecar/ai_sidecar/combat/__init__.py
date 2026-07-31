"""RO Combat Mechanics — complete elemental, size, race, defense, card, flee, cast, status, refinement system.

Exports all combat formulas for use by the PDCA loop and heuristic service.
"""

from ai_sidecar.combat.ro_mechanics import (
    # Element chart
    ELEMENT_CHART,
    SIZE_MODIFIERS,
    RACE_MODIFIERS,
    DEFENSE_TYPE_MODIFIERS,

    # Core damage calculation
    calculate_damage,

    # Flee / hit
    calculate_flee,
    calculate_hit_chance,

    # Cast time / after-cast delay
    calculate_cast_time,
    calculate_after_cast_delay,

    # Status resistance
    calculate_status_resistance,

    # Card stacking with diminishing returns
    calculate_card_stacking,

    # Refinement
    calculate_refinement_bonus,

    # Best-skill / best-element helpers
    best_skill_for_monster,
    best_element_for_monster,

    # Skill element lookup
    get_skill_element_info,
)

# New exports from element_table.py (Level 1-4 support)
from ai_sidecar.combat.element_table import (
    ElementTable,
    get_element_table,
    element_modifier,
    best_element_against,
)

# New exports from card_db.py (correct stacking formula)
from ai_sidecar.combat.card_db import (
    CardDatabase,
    Card,
    CardSlot,
    CardBonusType,
    get_card_database,
    get_total_card_multiplier,
    calculate_card_damage_bonus,
)

# New exports from damage_formulas.py (correct RO formulas)
from ai_sidecar.combat.damage_formulas import (
    calculate_flee as calc_flee,
    calculate_hit_chance as calc_hit_chance,
    calculate_status_resistance as calc_status_resistance,
    calculate_refinement_bonus as calc_refinement_bonus,
    calculate_aspd,
    calculate_aspd_interval,
    SkillCooldownTracker,
    estimate_hits_to_kill,
    get_monster_element,
    get_monster_size,
    get_monster_race,
    get_monster_def_data,
    get_skill_element,
    get_skill_cooldown,
    get_skill_range,
)

# New exports from mvp_tracker.py
from ai_sidecar.combat.mvp_tracker import (
    MVPTracker,
    MVPRecord,
    MVPHuntTarget,
    get_mvp_tracker,
)

# New exports from woe_combat_ai.py
from ai_sidecar.combat.woe_combat_ai import (
    WoECombatAI,
    WoEPhase,
    get_woe_combat_ai,
)

__all__ = [
    # Legacy
    "ELEMENT_CHART",
    "SIZE_MODIFIERS",
    "RACE_MODIFIERS",
    "DEFENSE_TYPE_MODIFIERS",
    "calculate_damage",
    "calculate_flee",
    "calculate_hit_chance",
    "calculate_cast_time",
    "calculate_after_cast_delay",
    "calculate_status_resistance",
    "calculate_card_stacking",
    "calculate_refinement_bonus",
    "best_skill_for_monster",
    "best_element_for_monster",
    "get_skill_element_info",

    # Element table (Level 1-4)
    "ElementTable",
    "get_element_table",
    "element_modifier",
    "best_element_against",

    # Card DB
    "CardDatabase",
    "Card",
    "CardSlot",
    "CardBonusType",
    "get_card_database",
    "get_total_card_multiplier",
    "calculate_card_damage_bonus",

    # Damage formulas
    "calc_flee",
    "calc_hit_chance",
    "calc_status_resistance",
    "calc_refinement_bonus",
    "calculate_aspd",
    "calculate_aspd_interval",
    "SkillCooldownTracker",
    "estimate_hits_to_kill",
    "get_monster_element",
    "get_monster_size",
    "get_monster_race",
    "get_monster_def_data",
    "get_skill_element",
    "get_skill_cooldown",
    "get_skill_range",

    # MVP tracker
    "MVPTracker",
    "MVPRecord",
    "MVPHuntTarget",
    "get_mvp_tracker",

    # WoE combat AI
    "WoECombatAI",
    "WoEPhase",
    "get_woe_combat_ai",
]
