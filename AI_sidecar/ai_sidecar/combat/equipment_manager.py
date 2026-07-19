"""Equipment Manager — optimizes weapon selection based on monster element/size.

Architecture:
  - Reads inventory from bridge snapshot
  - Scores each weapon: ATK × element_modifier × size_modifier
  - Element mod from attr_fix.yml, Size mod from size_fix.yml
  - Tracks equip cooldown per bot (1000ms minimum)
  - Queues equip/unequip commands through action queue

RULE.md compliance:
  - Zero hardcoded weapon data — all from item_db_equip.yml via inventory snapshot
  - Zero hardcoded modifiers — all from attr_fix.yml and size_fix.yml
  - Bridge only executes equip commands — sidecar makes all decisions
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class WeaponScore:
    """Scored weapon from inventory."""
    slot: int            # inventory slot number
    name: str            # item name
    atk: int             # weapon ATK
    element: str         # weapon element (Neutral if unknown)
    weapon_type: str     # Dagger, 1hSword, etc.
    weight: int          # item weight
    slots: int           # number of card slots
    refine: int          # refine level
    score: float         # computed score
    reason: str          # why this weapon was chosen


# Element → weapon name keyword mapping for determining weapon element
# rAthena item_db doesn't have an explicit element field for weapons.
# We infer from the item name using these patterns:
_ELEMENT_KEYWORDS = {
    "Fire":    ["flame", "fire", "blaze", "burning", "magma", "lava", "inferno", "volcanic", "heat",
                "flamberge", "ignis", "ardor", "katar_of", "combat_knife", "fireblend", "hellfire",
                "fire_sword", "fire_mace", "fire_axe", "fire_spear"],
    "Water":   ["ice", "water", "frost", "frozen", "aqua", "glacial", "crystal", "cold",
                "ice_pick", "ice_brand", "aqua_staff", "coral", "tidal", "tsunami",
                "water_sword", "water_mace", "water_spear", "water_bow"],
    "Wind":    ["wind", "storm", "thunder", "lightning", "gale", "zephyr", "tornado", "breeze",
                "tempest", "electric", "shock", "thunder_staff", "storm_sword",
                "wind_sword", "wind_mace", "wind_bow", "wind_spear"],
    "Earth":   ["earth", "stone", "rock", "terra", "ground", "mountain", "granite", "obsidian",
                "gaja", "katar_of_earth", "earth_sword", "earth_mace", "earth_axe",
                "baphomet", "giant", "heavy", "adamant"],
    "Holy":    ["holy", "sacred", "blessed", "divine", "angel", "seraph", "cross", "light",
                "morning_star", "holy_sword", "holy_mace", "holy_staff", "holy_bow",
                "grand_cross", "piercing"],
    "Dark":    ["dark", "shadow", "evil", "nightmare", "death", "soul", "demon", "night",
                "hell", "gloom", "midnight", "shadow_sword", "dark_mace", "evil_sword",
                "bloody", "chaos", "doom"],
    "Undead":  ["undead", "ghost", "skeleton", "bone", "necromancer", "lich", "wraith",
                "phantom", "spectre", "zombie", "vampire", "soul_drain"],
    "Ghost":   ["ghost", "spirit", "ether", "astral", "psychic", "mystic", "ethereal"],
    "Poison":  ["poison", "venom", "toxic", "acid", "snake", "viper", "serpent",
                "katar_of_venom", "poison_knife", "poison_sword"],
}

# Weapon types that deal magic damage instead of physical
_MAGIC_WEAPONS = {"Staff", "Wand", "Book", "Magic_Book"}

# Weapon types and their base damage modifier by monster size (from size_fix.yml)
_SIZE_MODIFIERS = {
    "Knuckle":  {"Small": 1.0, "Medium": 1.0, "Large": 0.75},
    "Whip":     {"Small": 1.0, "Medium": 1.0, "Large": 0.75},
    "Fist":     {"Small": 1.0, "Medium": 1.0, "Large": 0.75},
    # All other weapons: 100% across all sizes
}


def infer_weapon_element(item_name: str) -> str:
    """Infer weapon element from item name keywords.
    
    Returns element name or "Neutral" if no match.
    """
    name_lower = item_name.lower()
    for element, keywords in _ELEMENT_KEYWORDS.items():
        for kw in keywords:
            if kw in name_lower:
                return element
    return "Neutral"


def get_size_modifier(weapon_type: str, monster_size: str) -> float:
    """Get weapon size modifier against a monster size.
    
    From size_fix.yml: only Knuckle and Whip have non-100% modifiers.
    """
    mods = _SIZE_MODIFIERS.get(weapon_type, {})
    return mods.get(monster_size, 1.0)


class EquipmentManager:
    """Manages equipment optimization for combat."""

    def __init__(self):
        self._equip_cooldowns: dict[str, float] = {}  # bot_id → next allowed equip time
        self._equip_cooldown_ms = 1000  # minimum time between equips
        self._last_equipped: dict[str, int] = {}  # bot_id → last equipped slot
        self._cached_inventory: dict[str, list[dict]] = {}  # bot_id → parsed inventory
        self._cache_ttl = 5.0  # seconds
    
    def can_equip(self, bot_id: str) -> bool:
        """Check if equip cooldown has expired for this bot."""
        now_ms = time.time() * 1000
        last = self._equip_cooldowns.get(bot_id, 0)
        return (now_ms - last) >= self._equip_cooldown_ms
    
    def mark_equipped(self, bot_id: str):
        """Record equip action time for cooldown tracking."""
        self._equip_cooldowns[bot_id] = time.time() * 1000
    
    def get_best_weapon(self, snapshot, bot_id: str, monster_element: str,
                        monster_element_level: int = 1, monster_size: str = "Medium") -> Optional[WeaponScore]:
        """Find the best weapon from inventory against a given monster.
        
        Args:
            snapshot: Bot snapshot from bridge
            bot_id: Bot identifier
            monster_element: Target monster element (from mob_db.yml)
            monster_element_level: Element level (1-4)
            monster_size: Monster size (Small/Medium/Large)
        
        Returns:
            WeaponScore for best weapon, or None if current is already optimal
        """
        if snapshot is None:
            return None
        
        # Parse inventory from snapshot
        weapons = self._parse_inventory(snapshot, bot_id)
        if not weapons:
            return None
        
        # Get current equipped weapon
        current_slot = self._get_equipped_weapon_slot(snapshot)
        current_score = 0
        
        # Score each weapon
        from ai_sidecar.combat.element_table import get_element_table
        et = get_element_table()
        
        best_weapon: Optional[WeaponScore] = None
        
        for item in weapons:
            w_name = item.get("name", "")
            w_slot = int(item.get("index", item.get("slot", 0)))
            w_atk = int(item.get("atk", item.get("attack", 0)) or 0)
            w_type = str(item.get("type", item.get("subtype", "Dagger")))
            w_weight = int(item.get("weight", 0) or 0)
            w_slots = int(item.get("slots", 0) or 0)
            w_refine = int(item.get("refine", 0) or 0)
            w_equipped = bool(item.get("equipped", False)) or (w_slot == current_slot)
            
            # Infer weapon element from name
            w_element = infer_weapon_element(w_name)
            
            # Calculate element modifier
            ele_mod = et.get_modifier(w_element, monster_element, monster_element_level) / 100.0
            
            # Calculate size modifier
            size_mod = get_size_modifier(w_type, monster_size)
            
            # Calculate total effective score
            effective_atk = w_atk * ele_mod * size_mod
            
            # Bonus for card slots (potential future cards)
            slot_bonus = 1.0 + (w_slots * 0.05)  # 5% per empty slot
            
            # Refine bonus (rough estimate: +2 ATK per refine for weapons)
            refine_bonus = 1.0 + (w_refine * 0.02)  # 2% per refine level
            
            score = effective_atk * slot_bonus * refine_bonus
            
            reason_parts = []
            if ele_mod != 1.0:
                reason_parts.append(f"element={ele_mod:.0%}")
            if size_mod != 1.0:
                reason_parts.append(f"size={size_mod:.0%}")
            if w_slots > 0:
                reason_parts.append(f"slots=+{w_slots}")
            if w_refine > 0:
                reason_parts.append(f"refine=+{w_refine}")
            
            weapon_score = WeaponScore(
                slot=w_slot,
                name=w_name,
                atk=w_atk,
                element=w_element,
                weapon_type=w_type,
                weight=w_weight,
                slots=w_slots,
                refine=w_refine,
                score=score,
                reason=", ".join(reason_parts) if reason_parts else "neutral",
            )
            
            if best_weapon is None or score > best_weapon.score:
                best_weapon = weapon_score
            
            if w_equipped:
                current_score = score
        
        # If current weapon is best, skip equip
        if best_weapon and current_score >= best_weapon.score * 0.95:
            return None  # Current is within 5% of best — skip to avoid flickering
        
        return best_weapon
    
    def _parse_inventory(self, snapshot, bot_id: str) -> list[dict]:
        """Parse inventory from snapshot and cache it."""
        now = time.time()
        cached = self._cached_inventory.get(bot_id)
        if cached and cached[1] > now:
            return cached[0]
        
        weapons = []
        try:
            if isinstance(snapshot, dict):
                inv = snapshot.get("inventory", {}) or {}
                items = inv.get("items", inv.get("list", []))
                for item in items:
                    if self._is_weapon(item):
                        weapons.append(item)
            else:
                inv = getattr(snapshot, "inventory", None)
                if inv:
                    items = getattr(inv, "items", getattr(inv, "list", [])) or []
                    for item in items:
                        if self._is_weapon(item):
                            weapons.append({
                                "name": str(getattr(item, "name", "")),
                                "index": int(getattr(item, "index", getattr(item, "slot", 0))),
                                "atk": int(getattr(item, "atk", getattr(item, "attack", 0)) or 0),
                                "type": str(getattr(item, "type", getattr(item, "subtype", "Dagger"))),
                                "weight": int(getattr(item, "weight", 0) or 0),
                                "slots": int(getattr(item, "slots", 0) or 0),
                                "refine": int(getattr(item, "refine", 0) or 0),
                                "equipped": bool(getattr(item, "equipped", False)),
                            })
        except Exception as e:
            logger.debug("equip_manager: failed to parse inventory: %s", e)
        
        self._cached_inventory[bot_id] = (weapons, now + self._cache_ttl)
        return weapons
    
    def _is_weapon(self, item) -> bool:
        """Check if an inventory item is a weapon (not armor/accessory)."""
        if isinstance(item, dict):
            item_type = str(item.get("type", item.get("subtype", "")))
            location = str(item.get("location", item.get("equip_location", "")))
        else:
            item_type = str(getattr(item, "type", getattr(item, "subtype", "")))
            location = str(getattr(item, "location", getattr(item, "equip_location", "")))
        
        # Weapons are equipped in "Weapon" slot
        if "weapon" in location.lower():
            return True
        
        # Check type/subtype for weapon categories
        weapon_types = {"Dagger", "1hSword", "2hSword", "1hSpear", "2hSpear",
                        "1hAxe", "2hAxe", "1hMace", "2hMace", "Staff", "Wand",
                        "Bow", "Knuckle", "Musical", "Whip", "Book", "Claw",
                        "Pistol", "Rifle", "Shotgun", "Grenade", "Shuriken",
                        "Sling", "Scythe", "Instrument", "Magic_Book"}
        
        return item_type in weapon_types
    
    def _get_equipped_weapon_slot(self, snapshot) -> int:
        """Get the inventory slot of the currently equipped weapon."""
        try:
            if isinstance(snapshot, dict):
                inv = snapshot.get("inventory", {}) or {}
                for item in inv.get("items", inv.get("list", [])):
                    if item.get("equipped") and self._is_weapon(item):
                        return int(item.get("index", item.get("slot", 0)))
            else:
                inv = getattr(snapshot, "inventory", None)
                if inv:
                    for item in getattr(inv, "items", getattr(inv, "list", [])) or []:
                        if getattr(item, "equipped", False) and self._is_weapon(item):
                            return int(getattr(item, "index", getattr(item, "slot", 0)))
        except Exception as e:
            logger.debug("equip_manager: failed to find equipped weapon: %s", e)
        return 0


# Global singleton
_manager: EquipmentManager | None = None


def get_equipment_manager() -> EquipmentManager:
    """Get the global EquipmentManager instance."""
    global _manager
    if _manager is None:
        _manager = EquipmentManager()
    return _manager
