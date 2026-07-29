"""Gear upgrade suggestion based on level and class with cost/benefit analysis."""
from __future__ import annotations

import logging
from typing import Any

from ai_sidecar.domains.equipment.manager import EquipmentManager, UpgradePath

logger = logging.getLogger(__name__)

# Item value thresholds for cost/benefit analysis
_VALUE_TIERS: dict[str, dict[str, int]] = {
    "budget": {"max_cost": 10000, "priority_min": 3},
    "moderate": {"max_cost": 50000, "priority_min": 5},
    "premium": {"max_cost": 200000, "priority_min": 7},
}

# Class gear progression (needed by optimizer for gear score calculation)
_CLASS_GEAR_PROGRESSION: dict[str, list[dict[str, Any]]] = {
    "novice": [],
    "swordman": [
        {"slot": "weapon", "levels": [(1, "Knife[4]"), (20, "Blade[3]"), (40, "Saber[3]"),
                                       (55, "Flamberge[2]"), (70, "Nagan[1]")]},
        {"slot": "shield", "levels": [(1, "Guard[1]"), (30, "Buckler[1]"), (50, "Shield[1]")]},
        {"slot": "armor", "levels": [(1, "Cotton Shirt[1]"), (25, "Chain Mail[1]"),
                                      (45, "Full Plate[1]"), (65, "Plate Armor[1]")]},
    ],
    "mage": [
        {"slot": "weapon", "levels": [(1, "Rod[4]"), (25, "Wand[3]"), (45, "Staff[2]"),
                                       (65, "Arc Wand[1]"), (80, "Lich's Bone Wand[1]")]},
        {"slot": "armor", "levels": [(1, "Cotton Shirt[1]"), (25, "Manteau[1]"),
                                      (50, "Silk Robe[1]"), (70, "Saint's Robe[1]")]},
    ],
    "archer": [
        {"slot": "weapon", "levels": [(1, "Bow[4]"), (20, "Composite Bow[3]"), (40, "Great Bow[2]"),
                                       (60, "Arbalest[1]"), (80, "Hunter Bow[1]")]},
        {"slot": "armor", "levels": [(1, "Cotton Shirt[1]"), (25, "Manteau[1]"),
                                      (50, "Silk Robe[1]")]},
    ],
    "acolyte": [
        {"slot": "weapon", "levels": [(1, "Mace[4]"), (20, "Smashing Mace[3]"), (40, "Chain[2]"),
                                       (60, "Warrior's Mace[1]"), (80, "Grand Cross[1]")]},
        {"slot": "armor", "levels": [(1, "Cotton Shirt[1]"), (25, "Chain Mail[1]"),
                                      (50, "Full Plate[1]")]},
    ],
}


class EquipmentOptimizer:
    """Suggest gear upgrades based on level and class with cost/benefit analysis."""

    def __init__(self, manager: EquipmentManager | None = None, db: Any = None) -> None:
        self.manager = manager or EquipmentManager()
        self._gk_db = db
        if self._gk_db is None:
            from ai_sidecar.game_knowledge_db import GameKnowledgeDB
            self._gk_db = GameKnowledgeDB()

    def prioritize_upgrades(
        self,
        upgrades: list[UpgradePath],
        zeny: int,
        tier: str = "moderate",
    ) -> list[UpgradePath]:
        """Rank upgrades by cost/benefit and affordability.

        Args:
            upgrades: List of potential upgrades
            zeny: Current zeny balance
            tier: Budget tier ('budget', 'moderate', 'premium')

        Returns:
            Ranked list of affordable upgrades
        """
        tier_config = _VALUE_TIERS.get(tier, _VALUE_TIERS["moderate"])
        max_cost = tier_config["max_cost"]
        min_priority = tier_config["priority_min"]

        affordable = [
            u for u in upgrades
            if u.cost_estimate <= max_cost and u.priority >= min_priority
        ]

        # Sort by priority descending, then cost ascending
        affordable.sort(key=lambda u: (-u.priority, u.cost_estimate))
        return affordable

    def suggest_refine(self, item_name: str, current_refine: int, zeny: int) -> bool:
        """Suggest whether to attempt refining an item.

        Refining becomes exponentially riskier at higher levels.
        Returns True if refining is recommended.
        """
        if current_refine >= 10:
            return False
        if current_refine >= 7:
            return zeny > 500000
        if current_refine >= 4:
            return zeny > 100000
        return True

    def get_card_recommendations(
        self,
        job_name: str,
        slot: str,
        base_level: int,
    ) -> list[dict]:
        """Get card recommendations for a given slot."""
        recommendations: list[dict] = []
        if slot == "weapon":
            recommendations.append({
                "card_name": "Vadon Card",
                "benefit": "+20% damage to Water element",
                "estimated_cost": 50000,
            })
            recommendations.append({
                "card_name": "Drainliar Card",
                "benefit": "+20% damage to Earth element",
                "estimated_cost": 45000,
            })
        elif slot == "armor":
            recommendations.append({
                "card_name": "Peco Peco Card",
                "benefit": "Max HP +10%",
                "estimated_cost": 30000,
            })
        elif slot in ("accessory_1", "accessory_2"):
            recommendations.append({
                "card_name": "Zerom Card",
                "benefit": "LUK +2",
                "estimated_cost": 20000,
            })
        return recommendations

    def build_shopping_list(
        self,
        job_name: str,
        base_level: int,
        current_equip: dict,
        zeny: int,
        tier: str = "moderate",
    ) -> list[dict]:
        """Build a prioritized shopping list of gear to buy."""
        shopping_list: list[dict] = []
        upgrades = self.manager._find_upgrades(job_name, base_level, current_equip, [])
        prioritized = self.prioritize_upgrades(upgrades, zeny, tier)
        for upg in prioritized:
            shopping_list.append({
                "item": upg.target_item,
                "reason": upg.benefit,
                "estimated_cost": upg.cost_estimate,
                "priority": upg.priority,
                "type": "buy",
            })
        return shopping_list

    def assess_gear_score(
        self,
        equipped: dict[str, Any],
        base_level: int,
        job_name: str,
    ) -> float:
        """Calculate a 'gear score' from 0.0 to 1.0.

        Compares current gear against ideal progression at this level.
        """
        if not equipped:
            return 0.0

        expected_equips: dict[str, tuple[int, str]] = {}
        progression = _CLASS_GEAR_PROGRESSION.get(job_name, [])
        for slot_info in progression:
            slot_name = slot_info["slot"]
            for lv, item_name in slot_info["levels"]:
                if lv <= base_level:
                    expected_equips[slot_name] = (lv, item_name)

        current_score = 0
        total_score = max(len(expected_equips), 1)

        for slot_name, (lv, item_name) in expected_equips.items():
            equip = equipped.get(slot_name)
            if equip:
                eq_name = (equip.get("name", "") or "").lower()
                target = item_name.split("[")[0].lower()
                if target in eq_name:
                    current_score += 1

        return current_score / total_score
