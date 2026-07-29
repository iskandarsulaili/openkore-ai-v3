"""Equipment tracking, inventory management, and upgrade paths."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


_EQUIPMENT_SLOTS = [
    "weapon", "shield", "head_top", "head_mid", "head_low",
    "armor", "robe", "garment", "shoes", "accessory_1", "accessory_2",
    "arrow", "costume_top", "costume_mid", "costume_low",
]


@dataclass
class EquipState:
    """Tracks a single piece of equipped gear."""
    slot: str = ""
    item_id: str = ""
    item_name: str = ""
    refine_level: int = 0
    cards: list[str] = field(default_factory=list)
    elemental: str = ""
    broken: bool = False
    durability: int = 100


@dataclass
class UpgradePath:
    """A potential upgrade path."""
    current_item: str = ""
    target_item: str = ""
    target_level: int = 1
    cost_estimate: int = 0
    benefit: str = ""
    priority: int = 5  # 1-10, higher = more important


# Class-specific gear progression
_CLASS_GEAR_PROGRESSION: dict[str, list[dict[str, Any]]] = {
    "novice": [
        {"slot": "weapon", "levels": [(1, "Knife[4]"), (10, "Knife[4]")],
         "upgrade_at": []},
    ],
    "swordman": [
        {"slot": "weapon", "levels": [(1, "Sword[4]"), (20, "Blade[3]"), (40, "Saber[3]"),
                                       (55, "Flamberge[2]"), (70, "Nagan[1]")],
         "upgrade_at": [20, 40, 55, 70]},
        {"slot": "shield", "levels": [(1, "Guard[1]"), (30, "Buckler[1]"), (50, "Shield[1]")],
         "upgrade_at": [30, 50]},
        {"slot": "armor", "levels": [(1, "Cotton Shirt[1]"), (25, "Chain Mail[1]"),
                                      (45, "Full Plate[1]"), (65, "Plate Armor[1]")],
         "upgrade_at": [25, 45, 65]},
    ],
    "mage": [
        {"slot": "weapon", "levels": [(1, "Rod[4]"), (25, "Wand[3]"), (45, "Staff[2]"),
                                       (65, "Arc Wand[1]"), (80, "Lich's Bone Wand[1]")],
         "upgrade_at": [25, 45, 65, 80]},
        {"slot": "armor", "levels": [(1, "Cotton Shirt[1]"), (25, "Manteau[1]"),
                                      (50, "Silk Robe[1]"), (70, "Saint's Robe[1]")],
         "upgrade_at": [25, 50, 70]},
    ],
    "archer": [
        {"slot": "weapon", "levels": [(1, "Bow[4]"), (20, "Composite Bow[3]"), (40, "Great Bow[2]"),
                                       (60, "Arbalest[1]"), (80, "Hunter Bow[1]")],
         "upgrade_at": [20, 40, 60, 80]},
        {"slot": "armor", "levels": [(1, "Cotton Shirt[1]"), (25, "Manteau[1]"),
                                      (50, "Silk Robe[1]")],
         "upgrade_at": [25, 50]},
    ],
    "acolyte": [
        {"slot": "weapon", "levels": [(1, "Mace[4]"), (20, "Smashing Mace[3]"), (40, "Chain[2]"),
                                       (60, "Warrior's Mace[1]"), (80, "Grand Cross[1]")],
         "upgrade_at": [20, 40, 60, 80]},
        {"slot": "armor", "levels": [(1, "Cotton Shirt[1]"), (25, "Chain Mail[1]"),
                                      (50, "Full Plate[1]")],
         "upgrade_at": [25, 50]},
    ],
    "merchant": [
        {"slot": "weapon", "levels": [(1, "Sword[4]"), (20, "Blade[3]"), (40, "Saber[3]"),
                                       (60, "Two-Handed Sword[2]"), (75, "Muramasa[1]")],
         "upgrade_at": [20, 40, 60, 75]},
        {"slot": "shield", "levels": [(1, "Guard[1]"), (30, "Buckler[1]")],
         "upgrade_at": [30]},
        {"slot": "armor", "levels": [(1, "Cotton Shirt[1]"), (25, "Chain Mail[1]"),
                                      (50, "Full Plate[1]"), (70, "Plate Armor[1]")],
         "upgrade_at": [25, 50, 70]},
    ],
    "thief": [
        {"slot": "weapon", "levels": [(1, "Knife[4]"), (20, "Cutter[3]"), (40, "Main Gauche[3]"),
                                       (55, "Gladius[2]"), (70, "Damascus[1]")],
         "upgrade_at": [20, 40, 55, 70]},
        {"slot": "armor", "levels": [(1, "Cotton Shirt[1]"), (25, "Manteau[1]"),
                                      (50, "Thief Clothes[1]")],
         "upgrade_at": [25, 50]},
    ],
}


class EquipmentManager:
    """Track equipped items, inventory, and upgrade paths."""

    def __init__(self, db: Any = None) -> None:
        self._equip_states: dict[str, dict[str, EquipState]] = {}
        self._gk_db = db
        if self._gk_db is None:
            from ai_sidecar.game_knowledge_db import GameKnowledgeDB
            self._gk_db = GameKnowledgeDB()

    def assess_equipment(
        self,
        signals: dict[str, Any],
        bot_id: str,
    ) -> list[dict]:
        """Assess equipment state and identify actions.

        Returns list of action dicts.
        """
        actions: list[dict] = []
        inventory = signals.get("inventory", []) or []
        equipped = signals.get("equipment", []) or signals.get("inventory_equipped", []) or []
        job_name = str(signals.get("job_name", "novice") or "novice").lower()
        base_level = int(signals.get("base_level", 1) or 1)

        # Parse current equipped items
        current_equip = self._parse_equipment(equipped)
        self._equip_states[bot_id] = current_equip

        # Get upgrade path recommendations
        upgrades = self._find_upgrades(job_name, base_level, current_equip, inventory)
        for upg in upgrades:
            actions.append({
                "type": "upgrade_equipment",
                "priority": upg.priority,
                "reason": f"Upgrade {upg.current_item} -> {upg.target_item} (level {upg.target_level})",
                "slot": upg.current_item,
                "target": upg.target_item,
                "cost": upg.cost_estimate,
            })

        # Check for broken equipment
        broken = [eq for eq in current_equip.values() if eq.broken]
        if broken:
            for eq in broken:
                actions.append({
                    "type": "repair_equipment",
                    "priority": 9,
                    "reason": f"{eq.item_name} ({eq.slot}) is broken — needs repair",
                    "slot": eq.slot,
                    "item": eq.item_name,
                })

        return actions

    def _parse_equipment(self, equipped: list[dict]) -> dict[str, EquipState]:
        """Parse equipped items from signals into EquipState dict."""
        result: dict[str, EquipState] = {}
        for item in equipped:
            slot = str(item.get("slot", "") or item.get("type", "") or "").lower().replace(" ", "_")
            if not slot:
                continue
            result[slot] = EquipState(
                slot=slot,
                item_id=str(item.get("id", "") or ""),
                item_name=str(item.get("name", "") or ""),
                refine_level=int(item.get("refine", 0) or 0),
                cards=item.get("cards", []) or [],
                elemental=str(item.get("element", "") or ""),
                broken=bool(item.get("broken", False)),
                durability=int(item.get("durability", 100) or 100),
            )
        return result

    def _find_upgrades(
        self,
        job_name: str,
        base_level: int,
        current_equip: dict[str, EquipState],
        inventory: list[dict],
    ) -> list[UpgradePath]:
        """Find upgrade paths based on class progression."""
        paths: list[UpgradePath] = []
        progression = _CLASS_GEAR_PROGRESSION.get(job_name)
        if not progression:
            return paths

        for slot_info in progression:
            slot_name = slot_info["slot"]
            upgrade_levels = slot_info["upgrade_at"]
            levels = slot_info["levels"]

            # Check if we're at an upgrade threshold
            relevant_bps = [lv for lv in upgrade_levels if lv <= base_level]

            # Determine current item in this slot
            current_item = current_equip.get(slot_name)
            current_name = current_item.item_name if current_item else ""

            for bp in relevant_bps:
                target_item = ""
                for lv, item_name in levels:
                    if lv <= base_level:
                        target_item = item_name
                    if lv > base_level:
                        break

                if target_item and target_item != current_name:
                    # Check if we have the item in inventory already
                    has_in_inventory = any(
                        target_item.split("[")[0].lower() in (inv.get("name", "") or "").lower()
                        for inv in inventory
                    )

                    paths.append(UpgradePath(
                        current_item=current_name or slot_name,
                        target_item=target_item,
                        target_level=bp,
                        cost_estimate=0 if has_in_inventory else 5000,
                        benefit=f"Better {slot_name} for {job_name} at level {bp}",
                        priority=min(8, 3 + bp // 20),
                    ))

        return paths

    def get_equip_command(self, item_name: str) -> str:
        """Generate command to equip an item."""
        return f"equip {item_name}"

    def get_unequip_command(self, item_name: str) -> str:
        """Generate command to unequip an item."""
        return f"unequip {item_name}"

    def is_equipped(self, bot_id: str, item_name: str) -> bool:
        """Check if an item is currently equipped."""
        equip_state = self._equip_states.get(bot_id, {})
        return any(
            item_name.lower() in eq.item_name.lower()
            for eq in equip_state.values()
        )

    def get_free_slots(self, bot_id: str) -> list[str]:
        """Get equipment slots that are currently empty."""
        equip_state = self._equip_states.get(bot_id, {})
        return [slot for slot in _EQUIPMENT_SLOTS if slot not in equip_state]

    def cleanup_bot(self, bot_id: str) -> None:
        """Remove equipment state for a bot."""
        self._equip_states.pop(bot_id, None)
