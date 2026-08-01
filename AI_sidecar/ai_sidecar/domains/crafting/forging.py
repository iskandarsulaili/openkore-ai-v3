"""Weapon and armor smithing recipes and management."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Weapon forging recipes
_FORGING_RECIPES: dict[str, dict[str, Any]] = {
    "blade": {
        "name": "Blade[3]", "item_id": "1104", "slot": "weapon",
        "weapon_type": "sword",
        "ingredients": [("Iron", 30), ("Steel", 10), ("Coal", 5)],
        "skill_required": "BS_SWORD", "skill_level": 1,
        "min_level": 20, "difficulty": 1,
    },
    "saber": {
        "name": "Saber[3]", "item_id": "1106", "slot": "weapon",
        "weapon_type": "sword",
        "ingredients": [("Steel", 25), ("Iron", 50), ("Coal", 10)],
        "skill_required": "BS_SWORD", "skill_level": 3,
        "min_level": 35, "difficulty": 2,
    },
    "flamberge": {
        "name": "Flamberge[2]", "item_id": "1115", "slot": "weapon",
        "weapon_type": "two_hand_sword",
        "ingredients": [("Steel", 60), ("Iron", 80), ("Coal", 20), ("Gold", 5)],
        "skill_required": "BS_SWORD", "skill_level": 5,
        "min_level": 50, "difficulty": 3,
    },
    "dagger": {
        "name": "Dagger[4]", "item_id": "1201", "slot": "weapon",
        "weapon_type": "dagger",
        "ingredients": [("Iron", 10)],
        "skill_required": "BS_DAGGER", "skill_level": 1,
        "min_level": 1, "difficulty": 1,
    },
    "main_gauche": {
        "name": "Main Gauche[3]", "item_id": "1205", "slot": "weapon",
        "weapon_type": "dagger",
        "ingredients": [("Iron", 25), ("Steel", 5)],
        "skill_required": "BS_DAGGER", "skill_level": 3,
        "min_level": 30, "difficulty": 2,
    },
    "mace": {
        "name": "Mace[4]", "item_id": "1301", "slot": "weapon",
        "weapon_type": "mace",
        "ingredients": [("Iron", 20), ("Steel", 5)],
        "skill_required": "BS_MACE", "skill_level": 1,
        "min_level": 1, "difficulty": 1,
    },
    "chain": {
        "name": "Chain[2]", "item_id": "1304", "slot": "weapon",
        "weapon_type": "mace",
        "ingredients": [("Iron", 45), ("Steel", 15)],
        "skill_required": "BS_MACE", "skill_level": 3,
        "min_level": 32, "difficulty": 2,
    },
    "axe": {
        "name": "Axe[4]", "item_id": "1401", "slot": "weapon",
        "weapon_type": "axe",
        "ingredients": [("Iron", 30)],
        "skill_required": "BS_AXE", "skill_level": 1,
        "min_level": 1, "difficulty": 1,
    },
    "spear": {
        "name": "Spear[4]", "item_id": "1401", "slot": "weapon",
        "weapon_type": "spear",
        "ingredients": [("Iron", 20), ("Steel", 5)],
        "skill_required": "BS_SPEAR", "skill_level": 1,
        "min_level": 1, "difficulty": 1,
    },
}


class ForgingCrafting:
    """Manage weapon and armor smithing."""

    def __init__(self, db: Any = None) -> None:
        self._gk_db = db
        if self._gk_db is None:
            from ai_sidecar.game_knowledge_db import GameKnowledgeDB
            self._gk_db = GameKnowledgeDB()

    def assess_forging_opportunities(
        self,
        signals: dict[str, Any],
        bot_id: str,
    ) -> list[dict]:
        """Check if we should forge equipment.

        Returns list of action dicts.
        """
        actions: list[dict] = []
        inventory = signals.get("inventory", []) or []
        base_level = int(signals.get("base_level", 1) or 1)
        job_name = str(signals.get("job_name", "novice") or "novice").lower()
        zeny = int(signals.get("zeny", 0) or 0)

        # Only blacksmiths can forge
        is_blacksmith = "blacksmith" in job_name or "merchant" in job_name
        if not is_blacksmith:
            return actions

        for recipe_id, recipe in _FORGING_RECIPES.items():
            if recipe["min_level"] > base_level:
                continue

            can_craft, missing = self._check_ingredients(recipe, inventory)

            if can_craft:
                actions.append({
                    "type": "forge_item",
                    "priority": 6,
                    "reason": f"Forge {recipe['name']} — have ingredients",
                    "recipe_id": recipe_id,
                    "item": recipe["name"],
                    "slot": recipe["slot"],
                })
            elif zeny > 50000:
                actions.append({
                    "type": "buy_forge_materials",
                    "priority": 4,
                    "reason": f"Buy materials to forge {recipe['name']}",
                    "missing": missing,
                })

        return actions

    def get_forge_command(self, item_name: str, quantity: int = 1) -> str:
        return f"forge {item_name} {quantity}"

    def get_refine_command(self, item_name: str, material: str = "Elunium") -> str:
        return f"refine {item_name} {material}"

    def get_available_forge_recipes(self, base_level: int, known_skills: list[str]) -> list[dict]:
        recipes = []
        for recipe_id, recipe in _FORGING_RECIPES.items():
            if recipe["min_level"] > base_level:
                continue
            if recipe["skill_required"] and recipe["skill_required"] not in known_skills:
                continue
            recipes.append({
                "recipe_id": recipe_id,
                "name": recipe["name"],
                "slot": recipe["slot"],
                "weapon_type": recipe["weapon_type"],
                "ingredients": recipe["ingredients"],
                "difficulty": recipe["difficulty"],
            })
        return recipes

    def should_refine(self, current_refine: int, zeny: int, item_value: int) -> bool:
        if current_refine >= 10:
            return False
        if current_refine >= 7:
            return zeny > item_value * 5
        return True

    def _check_ingredients(self, recipe: dict, inventory: list[dict]) -> tuple[bool, list[str]]:
        missing: list[str] = []
        inventory_lower = {}
        for item in inventory:
            name = (item.get("name", "") or "").lower()
            amount = int(item.get("amount", 0) or 0)
            inventory_lower[name] = inventory_lower.get(name, 0) + amount

        for ing_name, ing_qty in recipe.get("ingredients", []):
            have = inventory_lower.get(ing_name.lower(), 0)
            if have < ing_qty:
                missing.append(f"{ing_name}x{ing_qty}")

        return len(missing) == 0, missing

    def cleanup_bot(self, bot_id: str) -> None:
        """Remove per-bot state on unregistration.

        ForgingCrafting keeps no persistent per-bot dicts today (only the
        shared GameKnowledgeDB), but cleanup must be idempotent and defensive:
        any per-bot tracker attribute present is popped so a re-registered
        bot starts fresh.
        """
        for _attr in ("_active_batches", "_last_craft", "_craft_timers", "_cooldowns", "_states"):
            _holder = getattr(self, _attr, None)
            if isinstance(_holder, dict):
                _holder.pop(bot_id, None)
        logger.debug("[forging] cleanup_bot %s", bot_id)

# Alias for compatibility
Forging = ForgingCrafting
