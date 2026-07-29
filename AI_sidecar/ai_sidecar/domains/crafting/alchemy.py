"""Potion brewing recipes and alchemy management."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Known alchemy recipes: item_id -> {name, ingredients, result_count, skill_required}
_ALCHEMY_RECIPES: dict[str, dict[str, Any]] = {
    "red_potion": {
        "name": "Red Potion",
        "item_id": "501",
        "ingredients": [("Empty Bottle", 1), ("Red Herb", 1)],
        "result_count": 1,
        "skill_required": "",
        "min_level": 1,
    },
    "orange_potion": {
        "name": "Orange Potion",
        "item_id": "502",
        "ingredients": [("Empty Bottle", 1), ("Orange Herb", 1)],
        "result_count": 1,
        "skill_required": "",
        "min_level": 10,
    },
    "yellow_potion": {
        "name": "Yellow Potion",
        "item_id": "503",
        "ingredients": [("Empty Bottle", 1), ("Yellow Herb", 1)],
        "result_count": 1,
        "skill_required": "",
        "min_level": 20,
    },
    "white_potion": {
        "name": "White Potion",
        "item_id": "504",
        "ingredients": [("Empty Bottle", 1), ("White Herb", 1)],
        "result_count": 1,
        "skill_required": "",
        "min_level": 40,
    },
    "blue_potion": {
        "name": "Blue Potion",
        "item_id": "505",
        "ingredients": [("Empty Bottle", 1), ("Blue Herb", 1)],
        "result_count": 1,
        "skill_required": "",
        "min_level": 50,
    },
    "poison_bottle": {
        "name": "Poison Bottle",
        "item_id": "678",
        "ingredients": [("Empty Bottle", 1), ("Poison Herb", 1)],
        "result_count": 1,
        "skill_required": "AM_POTION",
        "min_level": 40,
    },
    "constitution": {
        "name": "Constitution Potion",
        "item_id": "12113",
        "ingredients": [("Empty Bottle", 1), ("White Herb", 3), ("Blue Herb", 1), ("Royal Jelly", 1)],
        "result_count": 1,
        "skill_required": "AM_POTION",
        "min_level": 55,
    },
}


class AlchemyCrafting:
    """Manage potion brewing and alchemy."""

    def __init__(self, db: Any = None) -> None:
        self._gk_db = db
        if self._gk_db is None:
            from ai_sidecar.game_knowledge_db import GameKnowledgeDB
            self._gk_db = GameKnowledgeDB()

    def assess_crafting_opportunities(
        self,
        signals: dict[str, Any],
        bot_id: str,
    ) -> list[dict]:
        """Check if we should brew potions based on ingredients and need.

        Returns list of action dicts.
        """
        actions: list[dict] = []
        inventory = signals.get("inventory", []) or []
        base_level = int(signals.get("base_level", 1) or 1)
        hp_ratio = float(signals.get("hp_ratio", 1.0) or 1.0)
        sp_ratio = float(signals.get("sp_ratio", 1.0) or 1.0)

        need_heal = hp_ratio < 0.5 or sp_ratio < 0.4

        for recipe_id, recipe in _ALCHEMY_RECIPES.items():
            if recipe["min_level"] > base_level:
                continue

            can_craft, missing = self._check_ingredients(recipe, inventory)
            if not can_craft:
                continue

            is_heal = "Potion" in recipe["name"] and not recipe["name"].startswith("Poison")
            is_sp = "Blue" in recipe["name"] or "Constitution" in recipe["name"]

            if need_heal and (is_heal or is_sp):
                actions.append({
                    "type": "craft_item",
                    "priority": 5,
                    "reason": f"Craft {recipe['name']} — have ingredients, need potions",
                    "recipe_id": recipe_id,
                    "item": recipe["name"],
                    "quantity": recipe["result_count"],
                })

            if not need_heal:
                actions.append({
                    "type": "craft_item_for_sale",
                    "priority": 3,
                    "reason": f"Craft {recipe['name']} for sale or storage",
                    "recipe_id": recipe_id,
                    "item": recipe["name"],
                    "quantity": recipe["result_count"],
                })

        return actions

    def get_available_recipes(self, base_level: int) -> list[dict]:
        """Get all recipes available at a given level."""
        recipes = []
        for recipe_id, recipe in _ALCHEMY_RECIPES.items():
            if recipe["min_level"] <= base_level:
                recipes.append({
                    "recipe_id": recipe_id,
                    "name": recipe["name"],
                    "item_id": recipe["item_id"],
                    "ingredients": recipe["ingredients"],
                    "result_count": recipe["result_count"],
                })
        return recipes

    def get_craft_command(self, item_name: str, quantity: int = 1) -> str:
        """Generate a crafting command."""
        return f"craft {item_name} {quantity}"

    def _check_ingredients(
        self,
        recipe: dict,
        inventory: list[dict],
    ) -> tuple[bool, list[str]]:
        """Check if we have the required ingredients.

        Returns (can_craft, missing_ingredients).
        """
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

    def get_shopping_list(
        self,
        recipe_ids: list[str],
        inventory: list[dict],
    ) -> list[dict]:
        """Get list of ingredients needed for target recipes."""
        shopping: list[dict] = []
        for rid in recipe_ids:
            recipe = _ALCHEMY_RECIPES.get(rid)
            if not recipe:
                continue
            can_craft, missing = self._check_ingredients(recipe, inventory)
            if not can_craft:
                for ing in missing:
                    if "x" in ing:
                        parts = ing.rsplit("x", 1)
                        name, qty = parts[0], int(parts[1])
                    else:
                        name, qty = ing, 1
                    shopping.append({"item": name, "quantity": qty, "for_recipe": recipe["name"]})
        return shopping

    def cleanup_bot(self, bot_id: str) -> None:
        pass
