"""
Gear Progression Planner — tracks current equipment, identifies upgrade
opportunities, budgets zeny for upgrades, schedules refinement sessions,
and automatically purchases upgrades when affordable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class EquipmentSlot:
    """An equipment slot with current and target gear."""
    slot_name: str  # weapon, shield, armor, garment, shoes, accessory1, accessory2, headgear
    current_item: str = ""
    current_refine: int = 0
    current_cards: list[str] = field(default_factory=list)
    target_item: str = ""
    target_refine: int = 0
    target_cards: list[str] = field(default_factory=list)
    upgrade_priority: int = 50
    estimated_cost: int = 0
    is_upgraded: bool = False


@dataclass
class UpgradePlan:
    """A plan to upgrade equipment."""
    slot_name: str
    current_item: str = ""
    target_item: str = ""
    upgrade_type: str = ""  # refine, card, replace
    cost: int = 0
    benefit: str = ""
    priority: int = 50
    is_affordable: bool = False


class GearProgressionPlanner:
    """Plans and executes gear progression."""

    # Known equipment upgrade paths
    UPGRADE_PATHS: dict[str, list[dict]] = {}

    @classmethod
    def _load_upgrade_paths(cls) -> dict[str, list[dict]]:
        """Load upgrade paths from the knowledge database.

        Covers ALL 8 equipment slots (weapon/shield/armor/garment/shoes/
        accessory1/accessory2/headgear) + cards, mapped agnostically from the
        DB's location flags (Locations.Right_Hand / Left_Hand / Armor / Garment /
        Shoes / Right_Accessory / Left_Accessory / Head_Top / Head_Mid / Head_Low).
        No hardcoded item IDs — items are ranked by (stat per zeny) so a different
        server with different prices still picks the best affordable gear.
        """
        paths: dict[str, list[dict]] = {}
        try:
            from ai_sidecar.knowledge_loader import get_weapons, get_armors
            weapons = get_weapons()
            armors = get_armors()

            # ── Weapon paths ──
            weapon_paths = []
            for w in weapons:
                # usability: novice (or all jobs) OR low-level (any job) — agnostic
                novice_ok = bool(w.get("Jobs.Novice")) or bool(w.get("Jobs.All"))
                equip_min = int(w.get("EquipLevelMin", 0) or 0)
                if not novice_ok and equip_min > 25:
                    continue
                if equip_min > 99:
                    continue
                buy_price = int(w.get("Buy", 0) or 0)
                # price floor: 1z items are event/rare rewards NOT buyable from an NPC
                # shop (verified on RAW: Angra Manyu/Ahura Mazdah = cost 1). Real
                # starter gear is 50z+ (Knife 1201 @50z, Sword @100z). Filtering
                # these keeps the plan to genuinely-purchasable gear (agnostic: the
                # floor excludes the event-price class, not any specific item).
                if buy_price < 10:
                    continue
                if not (w.get("Locations.Right_Hand") or w.get("Locations.Both_Hand")):
                    continue
                atk = int(w.get("Attack", 0) or 0)
                matk = int(w.get("MagicAttack", 0) or 0)
                weapon_paths.append({
                    "item": w.get("Name", "") or w.get("AegisName", ""),
                    "refine": min(10, max(4, equip_min // 10)),
                    "cost": buy_price,
                    "level": equip_min,
                    "atk": atk,
                    "matk": matk,
                    # efficiency-aware: ATK(+MATK) per zeny — picks the best-value weapon
                    "score": (atk + matk) / max(buy_price, 1) * 1000.0,
                })
            weapon_paths.sort(key=lambda x: (-x["score"], x["cost"]))
            if weapon_paths:
                paths["weapon"] = weapon_paths

            # ── Armor/gear paths by slot flag (agnostic location mapping) ──
            # slot_flag -> (slot_name, stat_key) : each equipment location flag.
            slot_map = [
                ("Locations.Armor",            "armor",        "Defense"),
                ("Locations.Left_Hand",        "shield",       "Defense"),
                ("Locations.Garment",          "garment",      "Defense"),
                ("Locations.Shoes",            "shoes",        "Defense"),
                ("Locations.Head_Top",         "headgear",     "Defense"),
                ("Locations.Right_Accessory",  "accessory1",   "Defense"),
                ("Locations.Left_Accessory",   "accessory2",   "Defense"),
                ("Locations.Both_Accessory",   "accessory1",   "Defense"),  # both-hands accessory -> slot 1
            ]
            slot_paths: dict[str, list[dict]] = {slot: [] for _, slot, _ in slot_map}
            for a in armors:
                novice_ok = bool(a.get("Jobs.Novice")) or bool(a.get("Jobs.All"))
                equip_min = int(a.get("EquipLevelMin", 0) or 0)
                if not novice_ok and equip_min > 25:
                    continue
                if equip_min > 99:
                    continue
                buy_price = int(a.get("Buy", 0) or 0)
                # price floor — exclude 1z event/rare reward items (not NPC-buyable)
                if buy_price < 10:
                    continue
                defense = int(a.get("Defense", 0) or 0)
                for flag, slot, stat in slot_map:
                    if a.get(flag):
                        slot_paths[slot].append({
                            "item": a.get("Name", "") or a.get("AegisName", ""),
                            "refine": min(10, max(4, equip_min // 10)),
                            "cost": buy_price,
                            "level": equip_min,
                            "def": defense,
                            # efficiency-aware: DEF per zeny
                            "score": defense / max(buy_price, 1) * 1000.0,
                            "slot": slot,
                        })
            for slot, items in slot_paths.items():
                items.sort(key=lambda x: (-x["score"], x["cost"]))
                if items:
                    paths[slot] = items

            logger.info("upgrade_paths_loaded_from_db: %s",
                        {k: len(v) for k, v in paths.items()})
        except Exception as e:
            logger.warning("upgrade_paths_db_load_failed: %s (DB is the source of truth)", e)
        return paths

    # Card upgrade paths
    CARD_PATHS: dict[str, list[dict]] = {}

    @classmethod
    def _load_card_paths(cls) -> dict[str, list[dict]]:
        """Load card upgrade paths from the knowledge database."""
        paths: dict[str, list[dict]] = {}
        try:
            from ai_sidecar.knowledge_loader import get_cards
            cards = get_cards()
            weapon_cards = []
            armor_cards = []
            for c in cards:
                name = c.get("Name", "")
                buy_price = c.get("Buy", 0)
                if buy_price > 0:
                    loc = c.get("Locations", {})
                    if isinstance(loc, dict):
                        is_weapon = loc.get("Weapon", False)
                        is_armor = loc.get("Armor", False) or loc.get("Shield", False)
                    else:
                        is_weapon = "Weapon" in str(loc)
                        is_armor = "Armor" in str(loc) or "Shield" in str(loc)
                    if is_weapon:
                        weapon_cards.append({"card": name, "cost": buy_price, "effect": "", "level": 50})
                    elif is_armor:
                        armor_cards.append({"card": name, "cost": buy_price, "effect": "", "level": 50})
            if weapon_cards:
                paths["weapon_card"] = weapon_cards
            if armor_cards:
                paths["armor_card"] = armor_cards
            logger.info("card_paths_loaded_from_db: %d weapon cards, %d armor cards",
                        len(weapon_cards), len(armor_cards))
        except Exception as e:
            logger.warning("card_paths_db_load_failed: %s (DB is the source of truth)", e)
        return paths

    def __init__(self) -> None:
        self._lock = RLock()
        self._slots: dict[str, EquipmentSlot] = {}
        self._plans: list[UpgradePlan] = []
        self._total_spent: int = 0
        self._enqueue_fn: Callable | None = None
        self._init_slots()
        # Load data from knowledge DB at runtime
        self.UPGRADE_PATHS = self._load_upgrade_paths()
        self.CARD_PATHS = self._load_card_paths()

    def _init_slots(self) -> None:
        for slot in ["weapon", "shield", "armor", "garment", "shoes", "accessory1", "accessory2", "headgear"]:
            self._slots[slot] = EquipmentSlot(slot_name=slot)

    # ── Public API ──

    def update_equipment(self, slot: str, item: str, refine: int = 0, cards: list[str] | None = None) -> None:
        """Update current equipment for a slot."""
        with self._lock:
            if slot in self._slots:
                self._slots[slot].current_item = item
                self._slots[slot].current_refine = refine
                self._slots[slot].current_cards = cards or []

    def generate_plan(self, level: int, budget: int) -> list[UpgradePlan]:
        """Generate an upgrade plan based on current level and budget."""
        with self._lock:
            plans: list[UpgradePlan] = []
            for slot_name, slot in self._slots.items():
                path = self.UPGRADE_PATHS.get(slot_name, [])
                for step in path:
                    if step["level"] <= level and step["cost"] <= budget:
                        if slot.current_item != step["item"] or slot.current_refine < step["refine"]:
                            # efficiency-aware benefit text: show the stat the item gives
                            if slot_name == "weapon":
                                _stat = f"ATK+{step.get('atk', 0)}{'/'+str(step.get('matk',0))+' MATK' if step.get('matk', 0) else ''}"
                            else:
                                _stat = f"DEF+{step.get('def', 0)}"
                            plans.append(UpgradePlan(
                                slot_name=slot_name,
                                current_item=slot.current_item,
                                target_item=step["item"],
                                upgrade_type="replace" if slot.current_item != step["item"] else "refine",
                                cost=step["cost"],
                                benefit=f"{_stat} (score {step.get('score', 0):.0f})",
                                priority=step["level"],
                                is_affordable=step["cost"] <= budget,
                            ))
                            break

            plans.sort(key=lambda p: (-p.priority, p.cost))
            self._plans = plans
            return plans

    def get_best_upgrade(self, level: int, budget: int) -> UpgradePlan | None:
        """Get the best affordable upgrade."""
        plans = self.generate_plan(level, budget)
        affordable = [p for p in plans if p.is_affordable]
        return affordable[0] if affordable else None

    def execute_upgrade(self, plan: UpgradePlan) -> bool:
        """Execute an upgrade."""
        with self._lock:
            if not self._enqueue_fn:
                return False
            if plan.upgrade_type == "refine":
                self._enqueue_fn("self", f"refine {plan.slot_name}")
            elif plan.upgrade_type == "replace":
                self._enqueue_fn("self", f"buy {plan.target_item}")
            elif plan.upgrade_type == "card":
                self._enqueue_fn("self", f"buy {plan.target_item}")
            self._total_spent += plan.cost
            logger.info("upgrade_executed: %s %s (cost=%dz)", plan.slot_name, plan.target_item, plan.cost)
            return True

    def get_gear_summary(self) -> str:
        with self._lock:
            lines = [f"── Gear Progression ──"]
            lines.append(f"Total spent: {self._total_spent:,}z")
            for slot_name, slot in self._slots.items():
                if slot.current_item:
                    cards = f" [{', '.join(slot.current_cards)}]" if slot.current_cards else ""
                    lines.append(f"  {slot_name}: {slot.current_item}+{slot.current_refine}{cards}")
            if self._plans:
                lines.append(f"Next upgrade: {self._plans[0].slot_name} -> {self._plans[0].target_item} ({self._plans[0].cost:,}z)")
            return "\n".join(lines)

    def set_enqueue_fn(self, fn: Callable) -> None:
        with self._lock:
            self._enqueue_fn = fn

    def reset(self) -> None:
        with self._lock:
            self._slots.clear()
            self._plans.clear()
            self._total_spent = 0
            self._init_slots()


# ── Global Singleton ──

_gear_planner: GearProgressionPlanner | None = None
_gear_planner_lock = RLock()


def get_gear_progression_planner() -> GearProgressionPlanner:
    global _gear_planner
    with _gear_planner_lock:
        if _gear_planner is None:
            _gear_planner = GearProgressionPlanner()
        return _gear_planner
