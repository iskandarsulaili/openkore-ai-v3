"""
Gear Swapping System — dynamic gear changes based on target.

A pro player carries 4+ elemental weapons and swaps mid-combat.
This module manages gear sets and recommends the best gear for any target.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GearSet:
    """A complete gear configuration."""
    name: str
    weapon: str = ""
    weapon_element: str = "neutral"
    weapon_type: str = "dagger"
    shield: str | None = None
    armor: str | None = None
    garment: str | None = None
    shoes: str | None = None
    accessory_left: str | None = None
    accessory_right: str | None = None
    headgear_top: str | None = None
    headgear_mid: str | None = None
    headgear_low: str | None = None
    cards: list[str] = field(default_factory=list)
    intended_target: str = "general"
    priority: int = 50
    required_job: str = "novice"


class GearSwapper:
    """Manages gear sets and recommends optimal gear for targets."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._gear_sets: dict[str, GearSet] = {}
        self._load_default_sets()

    def _load_default_sets(self) -> None:
        """Load default gear set templates."""
        # Elemental weapons
        self._gear_sets["fire_weapon"] = GearSet(
            name="fire_weapon",
            weapon="Fire Elemental Weapon",
            weapon_element="fire",
            weapon_type="sword",
            intended_target="element:earth",
            priority=80,
            cards=["Drainliar Card", "Fireblend Card"],
        )
        self._gear_sets["water_weapon"] = GearSet(
            name="water_weapon",
            weapon="Water Elemental Weapon",
            weapon_element="water",
            weapon_type="sword",
            intended_target="element:fire",
            priority=80,
            cards=["Vadon Card", "Marc Card"],
        )
        self._gear_sets["wind_weapon"] = GearSet(
            name="wind_weapon",
            weapon="Wind Elemental Weapon",
            weapon_element="wind",
            weapon_type="sword",
            intended_target="element:water",
            priority=80,
            cards=["Drainliar Card"],
        )
        self._gear_sets["earth_weapon"] = GearSet(
            name="earth_weapon",
            weapon="Earth Elemental Weapon",
            weapon_element="earth",
            weapon_type="sword",
            intended_target="element:wind",
            priority=80,
            cards=["Mantis Card"],
        )
        self._gear_sets["holy_weapon"] = GearSet(
            name="holy_weapon",
            weapon="Holy Elemental Weapon",
            weapon_element="holy",
            weapon_type="sword",
            intended_target="race:undead",
            priority=85,
            cards=["Undead Card", "Holy Card"],
        )
        self._gear_sets["ghost_weapon"] = GearSet(
            name="ghost_weapon",
            weapon="Ghost Elemental Weapon",
            weapon_element="ghost",
            weapon_type="sword",
            intended_target="element:ghost",
            priority=80,
            cards=["Ghost Card"],
        )

        # Size-specific weapons
        self._gear_sets["large_weapon"] = GearSet(
            name="large_weapon",
            weapon="Two-Handed Sword",
            weapon_element="neutral",
            weapon_type="two_handed_sword",
            intended_target="size:large",
            priority=70,
        )
        self._gear_sets["small_weapon"] = GearSet(
            name="small_weapon",
            weapon="Dagger",
            weapon_element="neutral",
            weapon_type="dagger",
            intended_target="size:small",
            priority=70,
        )

        # Race-specific gear
        self._gear_sets["demon_slayer"] = GearSet(
            name="demon_slayer",
            weapon="Demon Slayer",
            weapon_element="neutral",
            weapon_type="sword",
            intended_target="race:demon",
            priority=75,
            cards=["Demon Card", "Hydra Card"],
        )
        self._gear_sets["brute_hunter"] = GearSet(
            name="brute_hunter",
            weapon="Brute Hunter",
            weapon_element="neutral",
            weapon_type="spear",
            intended_target="race:brute",
            priority=70,
            cards=["Brute Card"],
        )
        self._gear_sets["fish_killer"] = GearSet(
            name="fish_killer",
            weapon="Fish Killer",
            weapon_element="neutral",
            weapon_type="spear",
            intended_target="race:fish",
            priority=70,
            cards=["Fish Card"],
        )
        self._gear_sets["insect_swatter"] = GearSet(
            name="insect_swatter",
            weapon="Insect Swatter",
            weapon_element="neutral",
            weapon_type="sword",
            intended_target="race:insect",
            priority=70,
            cards=["Insect Card"],
        )
        self._gear_sets["plant_chopper"] = GearSet(
            name="plant_chopper",
            weapon="Plant Chopper",
            weapon_element="fire",
            weapon_type="sword",
            intended_target="race:plant",
            priority=70,
            cards=["Plant Card"],
        )

        # Role-specific sets
        self._gear_sets["tank_set"] = GearSet(
            name="tank_set",
            weapon="One-Handed Sword",
            weapon_element="neutral",
            weapon_type="sword",
            shield="Shield",
            armor="Heavy Armor",
            garment="Manteau",
            shoes="Greaves",
            intended_target="boss",
            priority=90,
            cards=["Thara Frog Card", "Rybio Card"],
        )
        self._gear_sets["farming_set"] = GearSet(
            name="farming_set",
            weapon="Main Weapon",
            weapon_element="neutral",
            weapon_type="sword",
            armor="Light Armor",
            shoes="Sprint Shoes",
            intended_target="general",
            priority=50,
        )
        self._gear_sets["mage_set"] = GearSet(
            name="mage_set",
            weapon="Staff",
            weapon_element="neutral",
            weapon_type="staff",
            armor="Robe",
            garment="Manteau",
            shoes="Shoes",
            intended_target="general",
            priority=50,
            required_job="mage",
        )
        self._gear_sets["archer_set"] = GearSet(
            name="archer_set",
            weapon="Bow",
            weapon_element="neutral",
            weapon_type="bow",
            armor="Leather Armor",
            garment="Manteau",
            shoes="Boots",
            intended_target="general",
            priority=50,
            required_job="archer",
        )

    # ── Public API ──

    def register_gear_set(self, gear_set: GearSet) -> None:
        with self._lock:
            self._gear_sets[gear_set.name] = gear_set

    def get_gear_set(self, name: str) -> GearSet | None:
        with self._lock:
            return self._gear_sets.get(name)

    def get_best_gear_for_target(
        self,
        target_element: str = "neutral",
        target_size: str = "medium",
        target_race: str = "formless",
        is_boss: bool = False,
        job_class: str = "novice",
    ) -> GearSet | None:
        """Get the best gear set for a specific target."""
        with self._lock:
            candidates: list[tuple[GearSet, float]] = []

            for gs in self._gear_sets.values():
                score = 0.0
                target = gs.intended_target

                # Element match
                if target.startswith("element:") and target.split(":")[1] == target_element:
                    score += 60

                # Size match
                if target.startswith("size:") and target.split(":")[1] == target_size:
                    score += 50

                # Race match
                if target.startswith("race:") and target.split(":")[1] == target_race:
                    score += 50

                # Boss/tank
                if target == "boss" and is_boss:
                    score += 80

                # General purpose
                if target == "general":
                    score += 20

                # Job class bonus
                if gs.required_job == job_class:
                    score += 20

                # Priority bonus
                score += gs.priority * 0.5

                if score > 0:
                    candidates.append((gs, score))

            if not candidates:
                return self._gear_sets.get("farming_set")

            candidates.sort(key=lambda x: -x[1])
            return candidates[0][0]

    def get_gear_for_element(self, target_element: str) -> GearSet | None:
        with self._lock:
            for gs in self._gear_sets.values():
                if gs.intended_target == f"element:{target_element}":
                    return gs
            return None

    def get_gear_for_size(self, target_size: str) -> GearSet | None:
        with self._lock:
            for gs in self._gear_sets.values():
                if gs.intended_target == f"size:{target_size}":
                    return gs
            return None

    def get_gear_for_race(self, target_race: str) -> GearSet | None:
        with self._lock:
            for gs in self._gear_sets.values():
                if gs.intended_target == f"race:{target_race}":
                    return gs
            return None

    def get_tank_set(self) -> GearSet | None:
        return self.get_gear_set("tank_set")

    def get_farming_set(self) -> GearSet | None:
        return self.get_gear_set("farming_set")

    def get_gear_swap_commands(self, current_gear: GearSet | None, target_gear: GearSet) -> list[str]:
        """Get the commands needed to swap from current to target gear."""
        commands: list[str] = []
        if current_gear and current_gear.weapon != target_gear.weapon:
            commands.append(f"eq {target_gear.weapon}")
        if current_gear and current_gear.shield != target_gear.shield and target_gear.shield:
            commands.append(f"eq {target_gear.shield}")
        if current_gear and current_gear.armor != target_gear.armor and target_gear.armor:
            commands.append(f"eq {target_gear.armor}")
        if not commands:
            commands.append(f"eq {target_gear.weapon}")
        return commands

    def get_gear_recommendation(self, target_info: dict[str, Any], job_class: str = "novice") -> str:
        """Get a human-readable gear recommendation."""
        element = target_info.get("element", "neutral")
        size = target_info.get("size", "medium")
        race = target_info.get("race", "formless")
        is_boss = target_info.get("is_boss", False)

        best = self.get_best_gear_for_target(element, size, race, is_boss, job_class)
        if not best:
            return "Use farming set"

        parts = [f"Equip {best.weapon} ({best.weapon_element})"]
        if best.shield:
            parts.append(f"+ {best.shield}")
        if best.cards:
            parts.append(f"[Cards: {', '.join(best.cards)}]")
        return " → ".join(parts)

    def get_all_gear_sets(self) -> list[GearSet]:
        with self._lock:
            return list(self._gear_sets.values())

    def get_gear_sets_for_job(self, job_class: str) -> list[GearSet]:
        with self._lock:
            return [gs for gs in self._gear_sets.values() if gs.required_job == job_class]


# ── Global Singleton ──

_gear_swapper: GearSwapper | None = None
_gear_swapper_lock = RLock()


def get_gear_swapper() -> GearSwapper:
    global _gear_swapper
    with _gear_swapper_lock:
        if _gear_swapper is None:
            _gear_swapper = GearSwapper()
        return _gear_swapper
