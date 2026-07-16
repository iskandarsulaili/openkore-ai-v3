"""Combat intelligence package — threat prediction, tactics, instinct analysis, build management, gear optimization, and card database."""
from ai_sidecar.combat.skill_rotation import SkillRotationSystem, get_skill_rotation_system, Skill, RotationStep, SkillRotation
from ai_sidecar.combat.elemental_matrix import ElementalMatrix, get_elemental_matrix, Element, Size, Race, WeaponType
from ai_sidecar.combat.buff_maintenance import BuffMaintenance, get_buff_maintenance, Buff
from ai_sidecar.combat.gear_swapper import GearSwapper, get_gear_swapper, GearSet
from ai_sidecar.combat.build_manager import BuildManager, get_build_manager, Build, StatAllocation, SkillLearnOrder, EquipmentGoal
from ai_sidecar.combat.combat_loop import CombatLoop, get_combat_loop, CombatState
from ai_sidecar.combat.card_db import CardDatabase, get_card_database, Card, CardSlot, CardBonusType
from ai_sidecar.combat.navigation import Router, MAP_CONNECTIONS

__all__ = [
    "SkillRotationSystem", "get_skill_rotation_system", "Skill", "RotationStep", "SkillRotation",
    "ElementalMatrix", "get_elemental_matrix", "Element", "Size", "Race", "WeaponType",
    "BuffMaintenance", "get_buff_maintenance", "Buff",
    "GearSwapper", "get_gear_swapper", "GearSet",
    "BuildManager", "get_build_manager", "Build", "StatAllocation", "SkillLearnOrder", "EquipmentGoal",
    "CombatLoop", "get_combat_loop", "CombatState",
    "CardDatabase", "get_card_database", "Card", "CardSlot", "CardBonusType",
    "Router", "MAP_CONNECTIONS",
]
