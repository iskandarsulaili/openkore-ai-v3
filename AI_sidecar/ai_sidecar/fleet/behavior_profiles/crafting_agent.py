"""CraftingAgent — forging, potion making, cooking, arrow crafting, elemental weapons."""

from __future__ import annotations

from typing import Any

from ai_sidecar.fleet.behavior_profiles import BehaviorProfile

_FORGE_RECIPES = {
    "saber": {"iron": 30, "coal": 6, "level": 1},
    "blade": {"iron": 50, "coal": 10, "level": 3},
    "bastard_sword": {"iron": 30, "steel": 20, "coal": 10, "level": 3},
    "two_handed_sword": {"iron": 100, "steel": 25, "coal": 10, "level": 5},
    "spear": {"iron": 20, "coal": 5, "level": 2},
    "pike": {"iron": 50, "steel": 10, "coal": 8, "level": 4},
    "battle_axe": {"iron": 60, "steel": 15, "coal": 10, "level": 4},
    "hammer": {"iron": 40, "coal": 8, "level": 3},
}

_POTION_RECIPES = {
    "red_potion": {"empty_bottle": 1, "red_herb": 3},
    "orange_potion": {"empty_bottle": 1, "orange_potent": 3},
    "yellow_potion": {"empty_bottle": 1, "yellow_herb": 3},
    "white_potion": {"empty_bottle": 1, "white_herb": 3},
    "blue_potion": {"empty_bottle": 1, "blue_herb": 2, "ment": 1},
}


class CraftingAgent(BehaviorProfile):
    """Handles RO crafting — Blacksmith forging, Alchemist potions, cooking, arrows."""

    def forge_decision(self, blacksmith_level: int, inventory: dict[str, int],
                       desired_weapon: str = "") -> dict[str, Any]:
        if desired_weapon and desired_weapon in _FORGE_RECIPES:
            recipe = _FORGE_RECIPES[desired_weapon]
            if blacksmith_level < recipe["level"]:
                return {"action": "insufficient_level",
                        "required": recipe["level"], "current": blacksmith_level}
            missing = {mat: qty - inventory.get(mat, 0)
                       for mat, qty in recipe.items() if mat != "level"
                       and qty > inventory.get(mat, 0)}
            if missing:
                return {"action": "gather_materials", "missing": missing,
                        "recipe": desired_weapon}
            return {"action": "forge", "weapon": desired_weapon,
                    "materials_cost": {k: v for k, v in recipe.items() if k != "level"}}
        best, score = self.best_action("craft")
        if best and score > 0.5:
            return {"action": "forge_learned", "weapon": best}
        return {"action": "check_available_recipes",
                "recipes": [w for w, r in _FORGE_RECIPES.items() if r["level"] <= blacksmith_level]}

    def potion_making(self, alchemist_level: int, inventory: dict[str, int],
                      desired_potion: str = "") -> dict[str, Any]:
        if desired_potion and desired_potion in _POTION_RECIPES:
            recipe = _POTION_RECIPES[desired_potion]
            missing = {mat: qty - inventory.get(mat, 0)
                       for mat, qty in recipe.items() if qty > inventory.get(mat, 0)}
            if missing:
                return {"action": "gather_herbs", "missing": missing,
                        "potion": desired_potion}
            success_chance = min(0.9, 0.1 + alchemist_level * 0.05)
            return {"action": "make_potion", "potion": desired_potion,
                    "success_rate": success_chance, "recipe": recipe}
        return {"action": "list_potions",
                "potions": list(_POTION_RECIPES.keys())}

    def cooking_action(self, cooking_level: int, ingredients: dict[str, int],
                       meal: str) -> dict[str, Any]:
        base_recipes = {
            "candy": {"honey": 3, "sugar": 1, "cinnamon": 1},
            "steak": {"meat": 5, "butter": 3, "garlic": 2},
            "sashimi": {"raw_fish": 4, "wasabi": 1, "soy_sauce": 1},
        }
        if meal not in base_recipes:
            return {"action": "unknown_recipe", "meal": meal}
        recipe = base_recipes[meal]
        missing = {i: q - ingredients.get(i, 0) for i, q in recipe.items()
                   if q > ingredients.get(i, 0)}
        if missing:
            return {"action": "get_ingredients", "missing": missing}
        return {"action": "cook", "meal": meal, "cooking_level": cooking_level}

    def arrow_crafting(self, arrow_type: str, inventory: dict[str, int]) -> dict[str, Any]:
        arrow_recipes = {
            "arrow": {"trunk": 1, "feather": 1, "qty": 100},
            "silver_arrow": {"trunk": 1, "feather": 1, "silver": 1, "qty": 50},
            "elemental_arrow": {"trunk": 1, "feather": 1, "elemental_ore": 1, "qty": 30},
            "crystal_arrow": {"trunk": 1, "feather": 1, "crystal": 2, "qty": 30},
        }
        recipe = arrow_recipes.get(arrow_type)
        if not recipe:
            return {"action": "unknown_arrow", "arrow_type": arrow_type}
        missing = {m: q - inventory.get(m, 0) for m, q in recipe.items()
                   if m != "qty" and q > inventory.get(m, 0)}
        if missing:
            return {"action": "gather_arrow_materials", "missing": missing}
        return {"action": "craft_arrows", "arrow_type": arrow_type, "qty": recipe["qty"]}

    def elemental_weapon(self, base_weapon: str, element: str,
                         inventory: dict[str, int]) -> dict[str, Any]:
        elemental_ore = {"fire": "coal", "water": "aquamarine", "earth": "yellow_gemstone",
                         "wind": "wind_of_verdu", "holy": "holy_water", "shadow": "darkness"}
        ore = elemental_ore.get(element)
        if not ore:
            return {"action": "invalid_element", "element": element}
        if inventory.get(ore, 0) < 3:
            return {"action": "need_ore", "ore": ore, "qty_needed": 3}
        return {"action": "enchant_weapon", "weapon": base_weapon,
                "element": element, "cost": f"3x{ore}"}

    def record_outcome(self, action: str, success: bool, item_created: str = "",
                       qty: int = 0) -> None:
        self._record_experience("craft", action, success, reward=float(qty),
                                item_created=item_created, qty=qty)
