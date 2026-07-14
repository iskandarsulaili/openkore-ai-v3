"""
Elemental Combat Matrix — Ragnarok Online full 25×25 elemental advantage table.

Provides thread-safe lookups for elemental, size, and race damage multipliers
using official rAthena values.  Includes a global singleton getter for shared use.
"""

from __future__ import annotations

import enum
import threading
from typing import Final, Optional

# ──────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────


class Element(str, enum.Enum):
    """The 10 Ragnarok Online elements."""

    NEUTRAL = "Neutral"
    WATER = "Water"
    EARTH = "Earth"
    FIRE = "Fire"
    WIND = "Wind"
    POISON = "Poison"
    HOLY = "Holy"
    DARK = "Dark"
    GHOST = "Ghost"
    UNDEAD = "Undead"


class Size(str, enum.Enum):
    """Monster / target sizes."""

    SMALL = "Small"
    MEDIUM = "Medium"
    LARGE = "Large"


class Race(str, enum.Enum):
    """Monster races."""

    FORMLESS = "Formless"
    UNDEAD = "Undead"
    BRUTE = "Brute"
    PLANT = "Plant"
    INSECT = "Insect"
    FISH = "Fish"
    DEMON = "Demon"
    DEMI_HUMAN = "DemiHuman"
    ANGEL = "Angel"
    DRAGON = "Dragon"


class WeaponType(str, enum.Enum):
    """Weapon types for size-modifier lookups."""

    DAGGER = "Dagger"
    SWORD = "Sword"
    TWO_HANDED_SWORD = "Two-Handed Sword"
    SPEAR = "Spear"
    BOW = "Bow"
    STAFF = "Staff"
    MACE = "Mace"
    KNUCKLE = "Knuckle"
    INSTRUMENT = "Instrument"
    WHIP = "Whip"
    BOOK = "Book"
    KATAR = "Katar"
    REVOLVER = "Revolver"
    SHOTGUN = "Shotgun"
    GRENADE = "Grenade"
    FUUMA_SHURIKEN = "Fuuma Shuriken"
    TWO_HANDED_STAFF = "Two-Handed Staff"


# ──────────────────────────────────────────────────────────────────────
# Elemental Combat Matrix
# ──────────────────────────────────────────────────────────────────────

# Official rAthena elemental table.
# Rows  = attack element (attacker's weapon/spell element)
# Cols  = target defense element (monster's element)
# Values are percentages: 100 = 100% damage, 200 = 200%, 0 = 0%, 25 = 25%.
#
# Column order: Neutral, Water, Earth, Fire, Wind, Poison, Holy, Dark, Ghost, Undead

_ELEMENT_ORDER: Final[tuple[Element, ...]] = (
    Element.NEUTRAL,
    Element.WATER,
    Element.EARTH,
    Element.FIRE,
    Element.WIND,
    Element.POISON,
    Element.HOLY,
    Element.DARK,
    Element.GHOST,
    Element.UNDEAD,
)

# Index map for fast lookups
_ELEMENT_INDEX: Final[dict[Element, int]] = {e: i for i, e in enumerate(_ELEMENT_ORDER)}

# fmt: off
_ELEMENTAL_TABLE: Final[list[list[int]]] = [
    #  Neut  Wat  Ear  Fir  Win  Poi  Hol  Dar  Gho  Und
    [ 100, 100, 100, 100, 100, 100, 100, 100,  75, 100 ],  # Neutral
    [ 100,  25,  50, 200, 100, 100, 100, 100, 100, 100 ],  # Water
    [ 100, 100,  25,  50, 200, 100, 100, 100, 100, 100 ],  # Earth
    [ 100,  50, 200,  25, 100, 100, 100, 100, 100, 100 ],  # Fire
    [ 100, 200, 100,  50,  25, 100, 100, 100, 100, 100 ],  # Wind
    [ 100, 100, 100, 100, 100,   0, 100, 100,  75,  50 ],  # Poison
    [ 100, 100, 100, 100, 100, 100,   0, 200, 100, 200 ],  # Holy
    [ 100, 100, 100, 100, 100, 100, 200,   0,  75, 100 ],  # Dark
    [   0, 100, 100, 100, 100, 100, 100, 100, 200, 100 ],  # Ghost
    [ 100, 100, 100, 100, 100,  50, 200, 100, 100,   0 ],  # Undead
]
# fmt: on

assert len(_ELEMENTAL_TABLE) == 10, "Elemental table must have 10 rows"
assert all(len(row) == 10 for row in _ELEMENTAL_TABLE), "Each row must have 10 columns"

# ──────────────────────────────────────────────────────────────────────
# Size modifier table
# ──────────────────────────────────────────────────────────────────────

# Weapon type → { size → percentage }
_SIZE_MODIFIERS: Final[dict[WeaponType, dict[Size, int]]] = {
    WeaponType.DAGGER:            {Size.SMALL: 100, Size.MEDIUM: 75,  Size.LARGE: 50},
    WeaponType.SWORD:             {Size.SMALL: 75,  Size.MEDIUM: 100, Size.LARGE: 75},
    WeaponType.TWO_HANDED_SWORD:  {Size.SMALL: 75,  Size.MEDIUM: 75,  Size.LARGE: 100},
    WeaponType.SPEAR:             {Size.SMALL: 75,  Size.MEDIUM: 75,  Size.LARGE: 100},
    WeaponType.BOW:               {Size.SMALL: 100, Size.MEDIUM: 100, Size.LARGE: 100},
    WeaponType.STAFF:             {Size.SMALL: 100, Size.MEDIUM: 100, Size.LARGE: 100},
    WeaponType.MACE:              {Size.SMALL: 75,  Size.MEDIUM: 100, Size.LARGE: 75},
    WeaponType.KNUCKLE:           {Size.SMALL: 100, Size.MEDIUM: 100, Size.LARGE: 75},
    WeaponType.INSTRUMENT:        {Size.SMALL: 75,  Size.MEDIUM: 100, Size.LARGE: 75},
    WeaponType.WHIP:              {Size.SMALL: 75,  Size.MEDIUM: 100, Size.LARGE: 75},
    WeaponType.BOOK:              {Size.SMALL: 100, Size.MEDIUM: 100, Size.LARGE: 100},
    WeaponType.KATAR:             {Size.SMALL: 100, Size.MEDIUM: 100, Size.LARGE: 75},
    WeaponType.REVOLVER:          {Size.SMALL: 100, Size.MEDIUM: 100, Size.LARGE: 100},
    WeaponType.SHOTGUN:           {Size.SMALL: 100, Size.MEDIUM: 100, Size.LARGE: 100},
    WeaponType.GRENADE:           {Size.SMALL: 100, Size.MEDIUM: 100, Size.LARGE: 100},
    WeaponType.FUUMA_SHURIKEN:   {Size.SMALL: 100, Size.MEDIUM: 100, Size.LARGE: 100},
    WeaponType.TWO_HANDED_STAFF:  {Size.SMALL: 100, Size.MEDIUM: 100, Size.LARGE: 100},
}

# ──────────────────────────────────────────────────────────────────────
# Race modifier table
# ──────────────────────────────────────────────────────────────────────

# Default race modifier is 100% for all weapon types.
# Cards / weapon-specific race bonuses are expressed as a dict of
# { weapon_type: { race: percentage } }.
# Entries not present default to 100.

_RACE_MODIFIERS: Final[dict[WeaponType, dict[Race, int]]] = {
    # All weapon types default to 100% vs all races.
    # Specific race bonuses (e.g. Holy Avenger vs Demon) can be added here.
    wt: {r: 100 for r in Race}
    for wt in WeaponType
}

# ──────────────────────────────────────────────────────────────────────
# ElementalAdvantageDescription
# ──────────────────────────────────────────────────────────────────────

_ELEMENT_DESCRIPTIONS: Final[dict[Element, str]] = {
    Element.NEUTRAL: "Neutral — no inherent advantage or weakness.",
    Element.WATER: "Water — strong vs Fire, weak vs Wind and Water.",
    Element.EARTH: "Earth — strong vs Wind, weak vs Fire and Earth.",
    Element.FIRE: "Fire — strong vs Earth, weak vs Water and Fire.",
    Element.WIND: "Wind — strong vs Water, weak vs Earth and Wind.",
    Element.POISON: "Poison — strong vs nothing, weak vs Poison, Ghost, Undead.",
    Element.HOLY: "Holy — strong vs Dark and Undead, weak vs Holy.",
    Element.DARK: "Dark — strong vs Holy, weak vs Dark and Ghost.",
    Element.GHOST: "Ghost — immune to Neutral, strong vs Ghost, weak vs Ghost.",
    Element.UNDEAD: "Undead — strong vs Holy, weak vs Poison and Undead.",
}

_SIZE_DESCRIPTIONS: Final[dict[Size, str]] = {
    Size.SMALL: "Small — daggers and fast weapons excel.",
    Size.MEDIUM: "Medium — swords and maces are balanced.",
    Size.LARGE: "Large — two-handed swords and spears dominate.",
}

_RACE_DESCRIPTIONS: Final[dict[Race, str]] = {
    Race.FORMLESS: "Formless — amorphous beings, no special weakness.",
    Race.UNDEAD: "Undead — vulnerable to Holy, resistant to Poison.",
    Race.BRUTE: "Brute — animals and beasts.",
    Race.PLANT: "Plant — flora-based monsters.",
    Race.INSECT: "Insect — bug-type monsters.",
    Race.FISH: "Fish — aquatic creatures.",
    Race.DEMON: "Demon — demonic entities, weak to Holy.",
    Race.DEMI_HUMAN: "Demi-Human — humanoid monsters and players.",
    Race.ANGEL: "Angel — holy beings, resistant to Dark.",
    Race.DRAGON: "Dragon — powerful ancient creatures.",
}


# ──────────────────────────────────────────────────────────────────────
# ElementalMatrix class
# ──────────────────────────────────────────────────────────────────────


class ElementalMatrix:
    """Thread-safe container for all RO combat damage multipliers.

    All lookup methods accept strings or enum members for convenience.
    """

    # Expose constants for external use
    ELEMENT_ORDER: Final[tuple[Element, ...]] = _ELEMENT_ORDER
    ELEMENTAL_TABLE: Final[list[list[int]]] = _ELEMENTAL_TABLE
    SIZE_MODIFIERS: Final[dict[WeaponType, dict[Size, int]]] = _SIZE_MODIFIERS
    RACE_MODIFIERS: Final[dict[WeaponType, dict[Race, int]]] = _RACE_MODIFIERS

    def __init__(self) -> None:
        self._lock = threading.RLock()

    # ── helpers ────────────────────────────────────────────────────

    @staticmethod
    def _resolve_element(e: str | Element) -> Element:
        if isinstance(e, Element):
            return e
        return Element(e.capitalize())

    @staticmethod
    def _resolve_size(s: str | Size) -> Size:
        if isinstance(s, Size):
            return s
        return Size(s.capitalize())

    @staticmethod
    def _resolve_race(r: str | Race) -> Race:
        if isinstance(r, Race):
            return r
        return Race(r.capitalize())

    @staticmethod
    def _resolve_weapon(w: str | WeaponType) -> WeaponType:
        if isinstance(w, WeaponType):
            return w
        # Handle hyphenated names like "Two-Handed Sword"
        return WeaponType(w)

    # ── public API ──────────────────────────────────────────────────

    def get_elemental_multiplier(
        self,
        attack_element: str | Element,
        target_element: str | Element,
    ) -> float:
        """Return the elemental damage multiplier as a float (e.g. 2.0 for 200%)."""
        ae = self._resolve_element(attack_element)
        te = self._resolve_element(target_element)
        with self._lock:
            row = _ELEMENT_INDEX[ae]
            col = _ELEMENT_INDEX[te]
            return _ELEMENTAL_TABLE[row][col] / 100.0

    def get_size_multiplier(
        self,
        weapon_type: str | WeaponType,
        target_size: str | Size,
    ) -> float:
        """Return the size damage modifier as a float (e.g. 0.5 for 50%)."""
        wt = self._resolve_weapon(weapon_type)
        ts = self._resolve_size(target_size)
        with self._lock:
            return _SIZE_MODIFIERS[wt][ts] / 100.0

    def get_race_multiplier(
        self,
        weapon_type: str | WeaponType,
        target_race: str | Race,
    ) -> float:
        """Return the race damage modifier as a float (default 1.0)."""
        wt = self._resolve_weapon(weapon_type)
        tr = self._resolve_race(target_race)
        with self._lock:
            return _RACE_MODIFIERS[wt][tr] / 100.0

    def get_best_element_against(
        self,
        target_element: str | Element,
    ) -> str:
        """Return the element name that deals the most damage to *target_element*.

        If multiple elements tie, the first one in element order is returned.
        """
        te = self._resolve_element(target_element)
        with self._lock:
            col = _ELEMENT_INDEX[te]
            best_val = -1
            best_elem = Element.NEUTRAL
            for row_idx, row in enumerate(_ELEMENTAL_TABLE):
                if row[col] > best_val:
                    best_val = row[col]
                    best_elem = _ELEMENT_ORDER[row_idx]
            return best_elem.value

    def get_best_weapon_type(
        self,
        target_size: str | Size,
    ) -> str:
        """Return the weapon type name that deals the most damage to *target_size*.

        If multiple weapon types tie, the first one in weapon-type order is returned.
        """
        ts = self._resolve_size(target_size)
        with self._lock:
            best_val = -1
            best_wt = WeaponType.DAGGER
            for wt, size_map in _SIZE_MODIFIERS.items():
                val = size_map[ts]
                if val > best_val:
                    best_val = val
                    best_wt = wt
            return best_wt.value

    def get_effective_damage_multiplier(
        self,
        attack_element: str | Element,
        weapon_type: str | WeaponType,
        target_element: str | Element,
        target_size: str | Size,
        target_race: str | Race,
    ) -> float:
        """Compute the combined damage multiplier from element × size × race.

        Returns a float where 1.0 = 100% damage.
        """
        elem_mult = self.get_elemental_multiplier(attack_element, target_element)
        size_mult = self.get_size_multiplier(weapon_type, target_size)
        race_mult = self.get_race_multiplier(weapon_type, target_race)
        return elem_mult * size_mult * race_mult

    def get_elemental_advantage_description(
        self,
        target_element: str | Element,
        target_size: str | Size,
        target_race: str | Race,
    ) -> str:
        """Return a human-readable description of the best approach against a target."""
        te = self._resolve_element(target_element)
        ts = self._resolve_size(target_size)
        tr = self._resolve_race(target_race)

        best_elem = self.get_best_element_against(te)
        best_weapon = self.get_best_weapon_type(ts)

        elem_desc = _ELEMENT_DESCRIPTIONS.get(te, "")
        size_desc = _SIZE_DESCRIPTIONS.get(ts, "")
        race_desc = _RACE_DESCRIPTIONS.get(tr, "")

        return (
            f"Target: {te.value} / {ts.value} / {tr.value}\n"
            f"  Best element: {best_elem}\n"
            f"  Best weapon:  {best_weapon}\n"
            f"  Element: {elem_desc}\n"
            f"  Size:    {size_desc}\n"
            f"  Race:    {race_desc}"
        )


# ──────────────────────────────────────────────────────────────────────
# Global singleton
# ──────────────────────────────────────────────────────────────────────

_matrix_instance: Optional[ElementalMatrix] = None
_matrix_lock: Final[threading.RLock] = threading.RLock()


def get_elemental_matrix() -> ElementalMatrix:
    """Return the global ElementalMatrix singleton (thread-safe)."""
    global _matrix_instance  # noqa: PLW0603
    with _matrix_lock:
        if _matrix_instance is None:
            _matrix_instance = ElementalMatrix()
        return _matrix_instance
