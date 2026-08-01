"""Food buff recipes and cooking management."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Known cooking recipes
_COOKING_RECIPES: dict[str, dict[str, Any]] = {
    "str_food": {
        "name": "Intestines Rice Ball",
        "item_id": "531",
        "ingredients": [("Rice", 1), ("Intestines", 1)],
        "stat": "str",
        "bonus": 4,
        "duration": 1800,
        "min_level": 20,
    },
    "agi_food": {
        "name": "Tentacle Rice Ball",
        "item_id": "532",
        "ingredients": [("Rice", 1), ("Tentacle", 1)],
        "stat": "agi",
        "bonus": 4,
        "duration": 1800,
        "min_level": 20,
    },
    "vit_food": {
        "name": "Skin Rice Ball",
        "item_id": "533",
        "ingredients": [("Rice", 1), ("Skin", 1)],
        "stat": "vit",
        "bonus": 4,
        "duration": 1800,
        "min_level": 20,
    },
    "int_food": {
        "name": "Brain Rice Ball",
        "item_id": "534",
        "ingredients": [("Rice", 1), ("Brain", 1)],
        "stat": "int",
        "bonus": 4,
        "duration": 1800,
        "min_level": 20,
    },
    "dex_food": {
        "name": "Eye Rice Ball",
        "item_id": "535",
        "ingredients": [("Rice", 1), ("Eye", 1)],
        "stat": "dex",
        "bonus": 4,
        "duration": 1800,
        "min_level": 20,
    },
    "luk_food": {
        "name": "Bone Rice Ball",
        "item_id": "536",
        "ingredients": [("Rice", 1), ("Bone", 1)],
        "stat": "luk",
        "bonus": 4,
        "duration": 1800,
        "min_level": 20,
    },
}

# Which stat food to use per class
_CLASS_FOOD_PREFERENCE: dict[str, list[str]] = {
    "swordman": ["str", "vit"],
    "knight": ["str", "vit"],
    "mage": ["int", "dex"],
    "wizard": ["int", "dex"],
    "archer": ["dex", "agi"],
    "hunter": ["dex", "agi"],
    "acolyte": ["int", "dex"],
    "priest": ["int", "dex"],
    "merchant": ["str", "luk"],
    "blacksmith": ["str", "luk"],
    "thief": ["agi", "dex"],
    "assassin": ["str", "agi"],
    "novice": ["str"],
}


class CookingCrafting:
    """Manage food buff preparation and consumption."""

    def __init__(self, db: Any = None) -> None:
        self._gk_db = db
        if self._gk_db is None:
            from ai_sidecar.game_knowledge_db import GameKnowledgeDB
            self._gk_db = GameKnowledgeDB()

    def assess_cooking_opportunities(
        self,
        signals: dict[str, Any],
        bot_id: str,
    ) -> list[dict]:
        """Check if we should cook food for stat buffs.

        Returns list of action dicts.
        """
        actions: list[dict] = []
        inventory = signals.get("inventory", []) or []
        base_level = int(signals.get("base_level", 1) or 1)
        job_name = str(signals.get("job_name", "novice") or "novice").lower()
        active_buffs = signals.get("buffs", []) or signals.get("food_active", []) or []
        zeny = int(signals.get("zeny", 0) or 0)

        if base_level < 20:
            return actions

        # Check if food buff is already active
        food_names = {fb.get("name", "") if isinstance(fb, dict) else str(fb) for fb in active_buffs}
        has_food_buff = any("food" in fname.lower() or "buff" in fname.lower() for fname in food_names)

        if has_food_buff:
            return actions

        pref_stats = _CLASS_FOOD_PREFERENCE.get(job_name, ["str"])

        for pref_stat in pref_stats:
            for recipe_id, recipe in _COOKING_RECIPES.items():
                if recipe["stat"] != pref_stat:
                    continue
                if recipe["min_level"] > base_level:
                    continue

                can_craft, missing = self._check_ingredients(recipe, inventory)
                if can_craft:
                    actions.append({
                        "type": "cook_food",
                        "priority": 5,
                        "reason": f"Cook {recipe['name']} (+{recipe['bonus']} {recipe['stat']}) for buff",
                        "recipe_id": recipe_id,
                        "item": recipe["name"],
                        "stat": recipe["stat"],
                        "bonus": recipe["bonus"],
                    })
                elif zeny > 10000:
                    actions.append({
                        "type": "buy_cooking_ingredients",
                        "priority": 3,
                        "reason": f"Buy ingredients for {recipe['name']}",
                        "missing": missing,
                    })
                break

        return actions

    def get_food_command(self, food_name: str) -> str:
        """Generate command to consume food."""
        return f"use {food_name}"

    def get_preferred_food(self, job_name: str) -> str | None:
        """Get the preferred food item ID for a job class."""
        pref_stats = _CLASS_FOOD_PREFERENCE.get(job_name.lower(), ["str"])
        for stat in pref_stats:
            for recipe_id, recipe in _COOKING_RECIPES.items():
                if recipe["stat"] == stat:
                    return recipe["item_id"]
        return None

    def _check_ingredients(self, recipe: dict, inventory: list[dict]) -> tuple[bool, list[str]]:
        """Check if we have the required cooking ingredients."""
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

        CookingCrafting keeps no persistent per-bot dicts today (only the
        shared GameKnowledgeDB), but cleanup must be idempotent and defensive:
        any per-bot tracker attribute present is popped so a re-registered
        bot starts fresh.
        """
        for _attr in ("_active_batches", "_last_craft", "_craft_timers", "_cooldowns", "_states"):
            _holder = getattr(self, _attr, None)
            if isinstance(_holder, dict):
                _holder.pop(bot_id, None)
        logger.debug("[cooking] cleanup_bot %s", bot_id)

# Alias for compatibility
Cooking = CookingCrafting
