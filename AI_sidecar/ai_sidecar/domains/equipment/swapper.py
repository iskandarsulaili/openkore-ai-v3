"""Auto-swap weapons based on monster element and size."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Element -> weapon type mapping for optimal damage
_ELEMENT_WEAPON_MAP: dict[str, dict[str, str]] = {
    "fire": {
        "dagger": "Fire Knife", "sword": "Fire Sword", "bow": "Fire Bow",
        "spear": "Fire Spear", "mace": "Fire Mace", "staff": "Fire Staff",
    },
    "water": {
        "dagger": "Water Knife", "sword": "Water Sword", "bow": "Water Bow",
        "spear": "Water Spear", "mace": "Water Mace", "staff": "Water Staff",
    },
    "wind": {
        "dagger": "Wind Knife", "sword": "Wind Sword", "bow": "Wind Bow",
        "spear": "Wind Spear", "mace": "Wind Mace", "staff": "Wind Staff",
    },
    "earth": {
        "dagger": "Earth Knife", "sword": "Earth Sword", "bow": "Earth Bow",
        "spear": "Earth Spear", "mace": "Earth Mace", "staff": "Earth Staff",
    },
    "holy": {
        "dagger": "Holy Knife", "sword": "Holy Sword", "bow": "Holy Bow",
        "mace": "Holy Mace",
    },
    "shadow": {
        "dagger": "Shadow Knife", "sword": "Shadow Sword",
    },
    "ghost": {
        "sword": "Ghost Sword", "bow": "Ghost Bow",
    },
    "undead": {
        "dagger": "Undead Knife", "sword": "Undead Sword", "mace": "Holy Mace",
    },
}

# Element effectiveness against monster elements
_ELEMENT_ADVANTAGE: dict[str, dict[str, float]] = {
    "fire": {"fire": 0.25, "water": 0.50, "earth": 1.25, "wind": 1.25,
             "poison": 1.00, "holy": 1.00, "shadow": 1.00, "ghost": 0.50, "undead": 1.25},
    "water": {"fire": 1.25, "water": 0.25, "earth": 1.25, "wind": 0.50,
              "poison": 0.75, "holy": 1.00, "shadow": 1.00, "ghost": 0.50, "undead": 1.00},
    "wind": {"fire": 0.75, "water": 1.25, "earth": 0.50, "wind": 0.25,
             "poison": 1.00, "holy": 1.00, "shadow": 1.00, "ghost": 0.50, "undead": 1.00},
    "earth": {"fire": 0.75, "water": 1.25, "earth": 0.25, "wind": 1.25,
              "poison": 0.50, "holy": 1.00, "shadow": 1.00, "ghost": 0.50, "undead": 1.00},
    "holy": {"undead": 2.00, "shadow": 2.00, "fire": 1.00, "water": 1.00,
             "earth": 1.00, "wind": 1.00, "poison": 1.00, "holy": 0.25, "ghost": 1.00},
    "shadow": {"holy": 0.50, "shadow": 0.25, "fire": 1.00, "water": 1.00,
               "earth": 1.00, "wind": 1.00, "poison": 1.00, "ghost": 1.00, "undead": 1.00},
    "ghost": {"ghost": 0.75, "fire": 1.00, "water": 1.00, "earth": 1.00,
              "wind": 1.00, "poison": 1.00, "holy": 1.00, "shadow": 1.00, "undead": 1.00},
    "undead": {"fire": 1.25, "undead": 0.25, "holy": 2.00, "shadow": 1.00,
               "water": 0.75, "earth": 0.75, "wind": 0.75, "poison": 0.50, "ghost": 0.75},
}

_CLASS_WEAPON_MAP = {
    "novice": "dagger", "swordman": "sword", "knight": "spear",
    "mage": "staff", "wizard": "staff", "archer": "bow", "hunter": "bow",
    "acolyte": "mace", "priest": "mace", "merchant": "sword",
    "blacksmith": "sword", "thief": "dagger", "assassin": "katar",
}


class WeaponSwapper:
    """Auto-swap weapons based on monster element for optimal damage."""

    def __init__(self, db: Any = None) -> None:
        self._last_swap: dict[str, str] = {}  # bot_id -> last swapped weapon
        self._gk_db = db
        if self._gk_db is None:
            from ai_sidecar.game_knowledge_db import GameKnowledgeDB
            self._gk_db = GameKnowledgeDB()

    def assess_swap(
        self,
        signals: dict[str, Any],
        bot_id: str,
    ) -> list[dict]:
        """Check if weapon should be swapped based on current target monster.

        Returns list of action dicts.
        """
        actions: list[dict] = []
        current_monster = signals.get("current_monster", {}) or {}
        if not current_monster:
            return actions

        monster_name = (
            current_monster.get("name", "")
            if isinstance(current_monster, dict)
            else str(current_monster)
        )
        monster_element = str(signals.get("monster_element", "Neutral") or "Neutral").lower()
        inventory = signals.get("inventory", []) or []
        job_name = str(signals.get("job_name", "novice") or "novice").lower()

        weapon_type = _CLASS_WEAPON_MAP.get(job_name, "dagger")

        # Find the best element to use against this monster
        best_element = self._find_best_element(monster_element, inventory)
        if not best_element:
            return actions

        # Check if we have that weapon in inventory
        weapon_name = self._get_element_weapon(best_element, weapon_type)
        if not weapon_name:
            return actions

        # Avoid redundant swaps
        last_weapon = self._last_swap.get(bot_id, "")
        if weapon_name == last_weapon:
            return actions

        self._last_swap[bot_id] = weapon_name
        actions.append({
            "type": "swap_weapon",
            "priority": 6,
            "reason": f"Swap to {weapon_name} ({best_element}) vs {monster_element} monster {monster_name}",
            "weapon": weapon_name,
            "element": best_element,
            "target_element": monster_element,
        })

        return actions

    def get_swap_command(self, weapon_name: str) -> str:
        """Generate equip command for the weapon."""
        return f"equip {weapon_name}"

    def _find_best_element(self, target_element: str, inventory: list[dict]) -> str | None:
        """Find the element that does the most damage against the target.

        Checks what elemental weapons the bot actually has.
        """
        target_elem = target_element.lower().replace(" ", "_")
        if target_elem in ("dark", "shadow"):
            target_elem = "shadow"

        best_element: str | None = None
        best_mult = 1.0

        # Check what elements we have available
        available_elements = set()
        inventory_names = [(item.get("name", "") or "").lower() for item in inventory]
        for elem, weapons in _ELEMENT_WEAPON_MAP.items():
            for wtype, wname in weapons.items():
                if wname.lower() in " ".join(inventory_names):
                    available_elements.add(elem)

        if not available_elements:
            return None

        for elem in available_elements:
            adv = _ELEMENT_ADVANTAGE.get(elem, {})
            mult = adv.get(target_elem, 1.0)
            if mult > best_mult:
                best_mult = mult
                best_element = elem

        return best_element

    def _get_element_weapon(self, element: str, weapon_type: str) -> str | None:
        """Get the weapon name for a given element and type."""
        elem_map = _ELEMENT_WEAPON_MAP.get(element, {})
        return elem_map.get(weapon_type)

    def get_optimal_for_map(
        self,
        map_name: str,
        job_name: str,
        inventory: list[dict],
    ) -> str | None:
        """Get the optimal weapon element for a map based on dominant monster element."""
        from ai_sidecar.autonomy.ro_mechanics import get_optimal_element_for_map

        target_element = get_optimal_element_for_map(map_name)
        if not target_element:
            return None

        weapon_type = _CLASS_WEAPON_MAP.get(job_name.lower(), "dagger")
        best_element = self._find_best_element(target_element, inventory)
        if best_element:
            return self._get_element_weapon(best_element, weapon_type)
        return None

    def reset_swap(self, bot_id: str) -> None:
        """Reset the last swap tracking."""
        self._last_swap.pop(bot_id, None)

    def cleanup_bot(self, bot_id: str) -> None:
        """Remove swap state for a bot."""
        self._last_swap.pop(bot_id, None)


# Alias for compatibility
EquipmentSwapper = WeaponSwapper
