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

__all__ = [
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
]
