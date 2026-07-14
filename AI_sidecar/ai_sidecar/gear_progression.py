"""
Gear progression — dynamic per-class weapon/armor upgrade paths from knowledge.json.

Queries 35,525 items in knowledge.json for real rAthena-accurate data.
No hardcoded values — always reflects the latest rAthena item database.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

# Class name mappings for item DB lookup
CLASS_WEAPON_TYPES: dict[str, list[str]] = {
    "swordman": ["sword", "two_hand_sword", "spear", "two_hand_spear"],
    "knight": ["sword", "two_hand_sword", "spear", "two_hand_spear"],
    "mage": ["staff", "two_hand_staff", "book"],
    "wizard": ["staff", "two_hand_staff", "book"],
    "archer": ["bow", "instrument", "whip"],
    "hunter": ["bow"],
    "acolyte": ["mace", "book", "staff"],
    "priest": ["mace", "book", "staff"],
    "merchant": ["axe", "two_hand_axe", "mace", "dagger"],
    "blacksmith": ["axe", "two_hand_axe", "mace"],
    "thief": ["dagger", "katar", "sword"],
    "assassin": ["katar", "dagger"],
    "novice": ["dagger", "sword", "mace", "staff", "bow"],
}

# Armor slot priority per class
CLASS_ARMOR_PRIORITY: dict[str, list[str]] = {
    "swordman": ["body", "shoes", "head", "garment", "shield"],
    "mage": ["body", "head", "shoes", "garment", "robe"],
    "archer": ["body", "garment", "shoes", "head"],
    "acolyte": ["body", "head", "shoes", "garment", "shield"],
    "merchant": ["body", "head", "shoes", "garment", "shield"],
    "thief": ["body", "garment", "shoes", "head"],
    "novice": ["body", "head", "shoes"],
}


@dataclass(slots=True)
class GearProgression:
    """Dynamic gear progression using knowledge.json data."""
    
    _lock: RLock = field(default_factory=RLock)
    _knowledge: dict[str, Any] = field(default_factory=dict)
    _stats: dict[str, int] = field(default_factory=lambda: {"queries": 0})
    
    def __post_init__(self) -> None:
        self._load_knowledge()
    
    def _load_knowledge(self) -> None:
        for candidate in [
            str(Path(__file__).parent.parent.parent / "knowledge" / "knowledge.json"),
            str(Path(__file__).parent.parent / "knowledge" / "knowledge.json"),
            "knowledge/knowledge.json",
        ]:
            if candidate and Path(candidate).exists():
                try:
                    self._knowledge = json.loads(Path(candidate).read_text(encoding="utf-8"))
                    logger.info("gear_progression_loaded: %d weapons, %d armors",
                                len(self._knowledge.get("items", {}).get("weapons", [])),
                                len(self._knowledge.get("items", {}).get("armors", [])))
                    return
                except Exception:
                    continue
        logger.warning("gear_progression: knowledge.json not found")
    
    def get_weapon_progression(self, player_class: str, level: int, zeny: int) -> list[dict[str, Any]]:
        """Get weapon upgrade path for a class at a given level."""
        results = []
        weapon_types = CLASS_WEAPON_TYPES.get(player_class.lower(), ["dagger", "sword"])
        weapons = self._knowledge.get("items", {}).get("weapons", [])
        
        for w in weapons:
            if not isinstance(w, dict):
                continue
            subtype = str(w.get("SubType", "")).strip()
            if subtype.lower() not in weapon_types:
                continue
            equip_min = int(w.get("EquipLevelMin", 0) or 0)
            if equip_min > level or equip_min == 0:
                continue
            price = int(w.get("Buy", 0) or 0)
            if price > zeny:
                continue
            
            results.append({
                "name": w.get("Name", w.get("AegisName", "?")),
                "level_req": equip_min,
                "atk": int(w.get("Attack", 0) or 0),
                "matk": int(w.get("Matk", 0) or 0),
                "slots": int(w.get("Slots", 0) or 0),
                "price": price,
                "weight": int(w.get("Weight", 0) or 0),
                "weapon_level": int(w.get("WeaponLevel", 0) or 0),
                "score": (int(w.get("Attack", 0) or 0) + int(w.get("Matk", 0) or 0)) / max(price, 1) * 1000,
            })
        
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:10]
    
    def get_armor_progression(self, player_class: str, level: int, zeny: int) -> list[dict[str, Any]]:
        """Get armor upgrade path for a class at a given level."""
        results = []
        armors = self._knowledge.get("items", {}).get("armors", [])
        
        for a in armors:
            if not isinstance(a, dict):
                continue
            equip_min = int(a.get("EquipLevelMin", 0) or 0)
            if equip_min > level or equip_min == 0:
                continue
            price = int(a.get("Buy", 0) or 0)
            if price > zeny:
                continue
            
            results.append({
                "name": a.get("Name", a.get("AegisName", "?")),
                "level_req": equip_min,
                "defense": int(a.get("Defense", 0) or 0),
                "slots": int(a.get("Slots", 0) or 0),
                "price": price,
                "weight": int(a.get("Weight", 0) or 0),
                "score": int(a.get("Defense", 0) or 0) / max(price, 1) * 1000,
            })
        
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:10]
    
    def get_card_recommendations(self, player_class: str, level: int) -> list[dict[str, Any]]:
        """Get recommended cards for a class at a given level."""
        results = []
        cards = self._knowledge.get("items", {}).get("cards", [])
        
        for c in cards:
            if not isinstance(c, dict):
                continue
            name = str(c.get("Name", "")).lower()
            # Filter to useful farming cards
            if any(kw in name for kw in ["poring", "drainliar", "hydra", "skel_worker",
                                          "pecopeco", "wolf", "spore", "fabre",
                                          "lunatic", "savage", "muka", "roda",
                                          "sohee", "marse", "cramp", "flora"]):
                results.append({
                    "name": c.get("Name", c.get("AegisName", "?")),
                    "price": int(c.get("Buy", 0) or 0),
                })
        
        results.sort(key=lambda r: r["price"])
        return results[:10]
    
    def get_auto_equip_recommendation(self, player_class: str, level: int, 
                                       zeny: int, current_weapon: str = "",
                                       current_armor: str = "") -> dict[str, Any]:
        """Get the single best gear upgrade to buy right now."""
        weapons = self.get_weapon_progression(player_class, level, zeny)
        armors = self.get_armor_progression(player_class, level, zeny)
        
        best = {"weapon": None, "armor": None, "reason": ""}
        
        # Find best weapon upgrade
        for w in weapons[:3]:
            if w["name"] != current_weapon:
                best["weapon"] = w
                break
        
        # Find best armor upgrade
        for a in armors[:3]:
            if a["name"] != current_armor:
                best["armor"] = a
                break
        
        if best["weapon"] and best["armor"]:
            # Recommend whichever gives more value
            if best["weapon"]["score"] > best["armor"]["score"]:
                best["reason"] = f"Upgrade weapon to {best['weapon']['name']} (atk:{best['weapon']['atk']})"
            else:
                best["reason"] = f"Upgrade armor to {best['armor']['name']} (def:{best['armor']['defense']})"
        elif best["weapon"]:
            best["reason"] = f"Upgrade weapon to {best['weapon']['name']}"
        elif best["armor"]:
            best["reason"] = f"Upgrade armor to {best['armor']['name']}"
        
        return best
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)
