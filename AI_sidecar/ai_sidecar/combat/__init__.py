"""RO Combat Mechanics — complete elemental, size, race, defense, card, flee, cast, status, refinement system.

Exports all combat formulas for use by the PDCA loop and heuristic service.
"""

# Core RO mechanics from autonomy/ro_mechanics.py
from ai_sidecar.autonomy.ro_mechanics import (
    # Element table (Level 1-4)
    ELEMENT_TABLE,
    # Size penalty table
    SIZE_PENALTY,
    # Core damage calculation
    calculate_damage,
    # Flee / hit
    calculate_flee,
    # ASPD
    calculate_aspd,
    # Renewal defense
    calculate_renewal_defense,
    # Stat helpers
    get_stat_atk,
    get_stat_matk,
    get_stat_max_hp,
    # Monster stats
    get_monster_stats,
    # Skill helpers
    calculate_skill_dps,
    get_skill_element,
    get_skill_sp_cost,
    get_best_skill,
    # Breakpoint helpers
    get_nearest_breakpoint,
    get_scaling_stat_targets,
    # Party helpers
    calculate_party_exp_share,
    # Profit helpers
    calculate_profit_per_kill,
    # MVP helpers
    is_mvp,
    get_mvp_value,
    get_mvp_low_hp_element,
    # Map helpers
    get_optimal_element_for_map,
    # Server mode
    get_server_mode,
    set_server_mode,
    ServerMode,
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
    calculate_cast_time,
    calculate_after_cast_delay,
    calculate_status_resistance,
    calculate_refinement_bonus,
    calculate_aspd as calc_aspd,
    calculate_aspd_interval,
    SkillCooldownTracker,
    estimate_hits_to_kill,
    get_monster_element,
    get_monster_size,
    get_monster_race,
    get_monster_def_data,
    get_skill_element as get_skill_element_info,
    get_skill_cooldown,
    get_skill_range,
    get_element_multiplier,
    get_size_multiplier,
    get_race_multiplier,
    get_level_penalty,
    calculate_damage_pre_renewal,
    calculate_damage_renewal,
    calculate_hard_def,
    calculate_soft_def,
    calculate_mdef,
    calculate_flee as calculate_flee_formula,
    calculate_hit_chance as calculate_hit_chance_formula,
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
    # Core RO mechanics
    "ELEMENT_TABLE",
    "SIZE_PENALTY",
    "calculate_damage",
    "calculate_flee",
    "calculate_aspd",
    "calculate_renewal_defense",
    "get_stat_atk",
    "get_stat_matk",
    "get_stat_max_hp",
    "get_monster_stats",
    "calculate_skill_dps",
    "get_skill_element",
    "get_skill_sp_cost",
    "get_best_skill",
    "get_nearest_breakpoint",
    "get_scaling_stat_targets",
    "calculate_party_exp_share",
    "calculate_profit_per_kill",
    "is_mvp",
    "get_mvp_value",
    "get_mvp_low_hp_element",
    "get_optimal_element_for_map",
    "get_server_mode",
    "set_server_mode",
    "ServerMode",

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
    "calculate_cast_time",
    "calculate_after_cast_delay",
    "calculate_status_resistance",
    "calculate_refinement_bonus",
    "calc_aspd",
    "calculate_aspd_interval",
    "SkillCooldownTracker",
    "estimate_hits_to_kill",
    "get_monster_element",
    "get_monster_size",
    "get_monster_race",
    "get_monster_def_data",
    "get_skill_element_info",
    "get_skill_cooldown",
    "get_skill_range",
    "get_element_multiplier",
    "get_size_multiplier",
    "get_race_multiplier",
    "get_level_penalty",
    "calculate_damage_pre_renewal",
    "calculate_damage_renewal",
    "calculate_hard_def",
    "calculate_soft_def",
    "calculate_mdef",
    "calculate_flee_formula",
    "calculate_hit_chance_formula",

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
