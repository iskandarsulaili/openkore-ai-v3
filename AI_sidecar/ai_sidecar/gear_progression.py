"""
Gear progression — per-class weapon/armor upgrade paths with rAthena-accurate data.

Each entry: item name, level requirement, card slots, attack/defense, effects.
The LLM references this for purchase decisions; reflex handles auto-equip.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GearEntry:
    name: str
    item_type: str  # weapon | armor | shield | garment | shoes | accessory | headgear
    equip_level_min: int = 0
    equip_level_max: int = 999
    slots: int = 0
    atk: int = 0
    matk: int = 0
    def_: int = 0
    weight: int = 0
    weapon_level: int = 1
    classes: list[str] = field(default_factory=lambda: ["all"])
    price: int = 0
    cards: list[str] = field(default_factory=list)
    notes: str = ""


# Per-class weapon progression paths (rAthena-accurate)
CLASS_WEAPON_PROGRESSION: dict[str, list[dict[str, Any]]] = {
    "swordman": [
        {"name": "Blade[4]", "level": 1, "atk": 30, "slots": 4, "price": 1000, "notes": "Starter sword, 4 card slots"},
        {"name": "Saber[4]", "level": 20, "atk": 55, "slots": 4, "price": 8000, "notes": "Best farming sword, 4 Drainliar cards"},
        {"name": "Broadsword[2]", "level": 35, "atk": 75, "slots": 2, "price": 20000, "notes": "Mid-tier, 2 slots"},
        {"name": "Claymore", "level": 50, "atk": 120, "slots": 1, "price": 50000, "notes": "High ATK, 1 slot"},
        {"name": "Muramasa", "level": 70, "atk": 150, "slots": 1, "price": 200000, "notes": "End-game, auto-cast"},
        {"name": "Holy Avenger", "level": 85, "atk": 180, "slots": 1, "price": 500000, "notes": "MVP weapon"},
    ],
    "mage": [
        {"name": "Rod[4]", "level": 1, "matk": 15, "slots": 4, "price": 500, "notes": "Starter rod"},
        {"name": "Wand[3]", "level": 15, "matk": 30, "slots": 3, "price": 5000, "notes": "Early magic"},
        {"name": "Staff[2]", "level": 25, "matk": 50, "slots": 2, "price": 15000, "notes": "Mid magic"},
        {"name": "Wizardry Staff", "level": 40, "matk": 80, "slots": 1, "price": 40000, "notes": "Wizard staple"},
        {"name": "Lich's Bone Wand", "level": 60, "matk": 120, "slots": 1, "price": 150000, "notes": "High MATK"},
        {"name": "Crimson Staff", "level": 80, "matk": 160, "slots": 2, "price": 400000, "notes": "End-game"},
    ],
    "archer": [
        {"name": "Bow[4]", "level": 1, "atk": 20, "slots": 4, "price": 1000, "notes": "Starter bow"},
        {"name": "Composite Bow[3]", "level": 15, "atk": 40, "slots": 3, "price": 6000, "notes": "Early archer"},
        {"name": "Crossbow[2]", "level": 30, "atk": 65, "slots": 2, "price": 20000, "notes": "Mid archer"},
        {"name": "Arbalest[1]", "level": 45, "atk": 90, "slots": 1, "price": 50000, "notes": "Hunter staple"},
        {"name": "Hunter Bow[1]", "level": 60, "atk": 120, "slots": 1, "price": 150000, "notes": "High ATK"},
        {"name": "Ixion Wings", "level": 80, "atk": 160, "slots": 1, "price": 500000, "notes": "End-game"},
    ],
    "thief": [
        {"name": "Knife[4]", "level": 1, "atk": 20, "slots": 4, "price": 500, "notes": "Starter dagger"},
        {"name": "Cutter[3]", "level": 15, "atk": 40, "slots": 3, "price": 5000, "notes": "Early thief"},
        {"name": "Main Gauche[3]", "level": 25, "atk": 55, "slots": 3, "price": 15000, "notes": "Mid dagger"},
        {"name": "Damascus[2]", "level": 40, "atk": 80, "slots": 2, "price": 40000, "notes": "Assassin staple"},
        {"name": "Katar[1]", "level": 55, "atk": 120, "slots": 1, "price": 120000, "notes": "Katar, high ATK"},
        {"name": "Bloody Roar", "level": 75, "atk": 160, "slots": 1, "price": 400000, "notes": "End-game katar"},
    ],
    "acolyte": [
        {"name": "Mace[4]", "level": 1, "atk": 25, "slots": 4, "price": 1000, "notes": "Starter mace"},
        {"name": "Smashing Mace[3]", "level": 15, "atk": 45, "slots": 3, "price": 6000, "notes": "Early acolyte"},
        {"name": "Chain[3]", "level": 25, "atk": 60, "slots": 3, "price": 15000, "notes": "Mid mace"},
        {"name": "Sword Mace[2]", "level": 40, "atk": 85, "slots": 2, "price": 40000, "notes": "Priest staple"},
        {"name": "Grand Cross", "level": 60, "atk": 130, "slots": 1, "price": 200000, "notes": "High ATK"},
        {"name": "Mace of Judgement", "level": 80, "atk": 170, "slots": 1, "price": 500000, "notes": "End-game"},
    ],
    "merchant": [
        {"name": "Axe[4]", "level": 1, "atk": 30, "slots": 4, "price": 1000, "notes": "Starter axe"},
        {"name": "Battle Axe[3]", "level": 15, "atk": 55, "slots": 3, "price": 8000, "notes": "Early merchant"},
        {"name": "Hammer[2]", "level": 30, "atk": 80, "slots": 2, "price": 25000, "notes": "Mid axe"},
        {"name": "Two-Handed Axe[1]", "level": 45, "atk": 120, "slots": 1, "price": 60000, "notes": "Blacksmith staple"},
        {"name": "Battle Hammer", "level": 65, "atk": 160, "slots": 1, "price": 200000, "notes": "High ATK"},
        {"name": "Doom Slayer", "level": 85, "atk": 200, "slots": 1, "price": 500000, "notes": "End-game"},
    ],
}

# Recommended cards per farming map
FARM_CARDS: dict[str, list[dict[str, Any]]] = {
    "prt_fild08": [
        {"card": "Drainliar Card", "effect": "5% HP drain", "slot": "weapon", "price": 50000},
        {"card": "Poring Card", "effect": "Crit +1", "slot": "garment", "price": 10000},
    ],
    "moc_fild01": [
        {"card": "Savage Card", "effect": "ATK +5", "slot": "weapon", "price": 30000},
    ],
    "pay_fild01": [
        {"card": "Elder Willow Card", "effect": "INT +2", "slot": "headgear", "price": 40000},
    ],
    "gef_fild01": [
        {"card": "Drainliar Card", "effect": "5% HP drain", "slot": "weapon", "price": 50000},
    ],
}

# Armor progression
ARMOR_PROGRESSION: list[dict[str, Any]] = [
    {"name": "Cotton Shirt[1]", "level": 1, "def": 2, "slots": 1, "price": 500, "classes": ["all"]},
    {"name": "Adventurer's Suit[1]", "level": 10, "def": 4, "slots": 1, "price": 5000, "classes": ["all"]},
    {"name": "Manteau[1]", "level": 20, "def": 6, "slots": 1, "price": 15000, "classes": ["all"]},
    {"name": "Chain Mail[1]", "level": 35, "def": 8, "slots": 1, "price": 40000, "classes": ["swordman", "knight", "merchant"]},
    {"name": "Silk Robe[1]", "level": 35, "def": 5, "slots": 1, "price": 35000, "classes": ["mage", "wizard", "acolyte", "priest"]},
    {"name": "Tights[1]", "level": 35, "def": 6, "slots": 1, "price": 35000, "classes": ["archer", "hunter", "thief", "assassin"]},
    {"name": "Full Plate[1]", "level": 55, "def": 10, "slots": 1, "price": 150000, "classes": ["knight", "paladin"]},
    {"name": "Orleans Gown[1]", "level": 60, "def": 7, "slots": 1, "price": 200000, "classes": ["wizard", "high_wizard"]},
]


@dataclass(slots=True)
class GearProgression:
    """Gear progression planning and recommendations."""
    
    _lock: RLock = field(default_factory=RLock)
    
    def get_weapon_recommendation(self, player_class: str, level: int, zeny: int) -> dict[str, Any] | None:
        """Get the best weapon for a class at a given level."""
        progression = CLASS_WEAPON_PROGRESSION.get(player_class.lower(), [])
        best = None
        for item in progression:
            if item["level"] <= level and item["price"] <= zeny:
                best = item
        return best
    
    def get_armor_recommendation(self, player_class: str, level: int, zeny: int) -> dict[str, Any] | None:
        """Get the best armor for a class at a given level."""
        best = None
        for item in ARMOR_PROGRESSION:
            if item["level"] <= level and item["price"] <= zeny:
                if "all" in item["classes"] or player_class.lower() in item["classes"]:
                    best = item
        return best
    
    def get_card_recommendation(self, map_name: str, zeny: int) -> list[dict[str, Any]]:
        """Get recommended cards for a farming map."""
        cards = FARM_CARDS.get(map_name.lower(), [])
        affordable = [c for c in cards if c["price"] <= zeny]
        return affordable[:3]
    
    def get_upgrade_path(self, player_class: str, current_weapon: str | None = None) -> list[dict[str, Any]]:
        """Get the full upgrade path for a class."""
        progression = CLASS_WEAPON_PROGRESSION.get(player_class.lower(), [])
        if current_weapon:
            # Find current position in path
            found = False
            result = []
            for item in progression:
                if found:
                    result.append(item)
                if item["name"] == current_weapon:
                    found = True
            return result
        return progression
    
    def counters(self) -> dict[str, int]:
        total = sum(len(v) for v in CLASS_WEAPON_PROGRESSION.values())
        return {"weapons": total, "armors": len(ARMOR_PROGRESSION), "cards": sum(len(v) for v in FARM_CARDS.values())}
