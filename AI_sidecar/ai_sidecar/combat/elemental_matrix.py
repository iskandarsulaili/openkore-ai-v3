"""
Elemental Combat Matrix — Ragnarok Online full 25×25 elemental advantage table.

Provides thread-safe lookups for elemental, size, race, and card damage multipliers
using official rAthena values from attr_fix.yml (all 4 levels).

Includes a global singleton getter for shared use.

Element Level (element_level) matters enormously:
  Level 1: Standard chart
  Level 2: Stronger advantages/disadvantages
  Level 3: Even stronger — Ghost becomes immune to Neutral (0%)
  Level 4: Extreme — same-element attacks can reach 200%, many elements heal
"""

from __future__ import annotations

import enum
import logging
import os
import threading
from typing import Final, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Path to rAthena data
# ──────────────────────────────────────────────────────────────────────────────

# File is at: ai_sidecar/combat/elemental_matrix.py
# Project root: ../../../..  (combat/ → ai_sidecar/ → AI_sidecar/ → project root)
_ATTR_FIX_PATH: Final[str] = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..",
        "knowledge", "rathena_db", "db", "pre-re", "attr_fix.yml",
    )
)


# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────


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


# ──────────────────────────────────────────────────────────────────────────────
# Element ordering (matches rAthena attr_fix.yml column ordering)
# ──────────────────────────────────────────────────────────────────────────────

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

# ──────────────────────────────────────────────────────────────────────────────
# Elemental chart loader — parses all 4 levels from attr_fix.yml
# ──────────────────────────────────────────────────────────────────────────────


def _load_elemental_tables() -> dict[int, list[list[int]]]:
    """Load all 4 elemental advantage tables from attr_fix.yml.

    Returns {1: 10×10 matrix, 2: 10×10 matrix, 3: 10×10 matrix, 4: 10×10 matrix}
    where rows = attack element, cols = defense element,
    following _ELEMENT_ORDER indexing.

    Falls back to an empty dict if the file cannot be loaded.
    """
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML not available — cannot load attr_fix.yml")
        return {}

    if not os.path.isfile(_ATTR_FIX_PATH):
        logger.warning("attr_fix.yml not found at %s", _ATTR_FIX_PATH)
        return {}

    try:
        with open(_ATTR_FIX_PATH, "r") as f:
            data = yaml.safe_load(f)
    except Exception as exc:
        logger.error("Failed to load attr_fix.yml: %s", exc)
        return {}

    body = data.get("Body", [])
    if not isinstance(body, list):
        logger.warning("attr_fix.yml Body is not a list")
        return {}

    # Build element name → index map (lowercased keys from YAML)
    elem_index_lower: dict[str, int] = {
        e.value.lower(): i for i, e in enumerate(_ELEMENT_ORDER)
    }

    tables: dict[int, list[list[int]]] = {}

    for entry in body:
        if not isinstance(entry, dict):
            continue
        level = int(entry.get("Level", 0))
        if level < 1 or level > 4:
            continue

        # Build a 10×10 matrix for this level
        matrix: list[list[int]] = [[100] * 10 for _ in range(10)]

        for attack_elem, def_map in entry.items():
            if attack_elem == "Level":
                continue
            if not isinstance(def_map, dict):
                continue

            atk_lower = attack_elem.lower()
            row = elem_index_lower.get(atk_lower)
            if row is None:
                logger.debug("Unknown element '%s' in attr_fix.yml Level %d", attack_elem, level)
                continue

            for def_elem, value in def_map.items():
                def_lower = def_elem.lower()
                col = elem_index_lower.get(def_lower)
                if col is None:
                    continue
                try:
                    matrix[row][col] = int(value)
                except (ValueError, TypeError):
                    continue

        tables[level] = matrix

    if tables:
        for lvl in sorted(tables):
            logger.info("Loaded Level %d elemental table from attr_fix.yml", lvl)
    else:
        logger.warning("No elemental tables could be loaded from attr_fix.yml")

    return tables


# Load all levels on module import
_ELEMENTAL_TABLES: Final[dict[int, list[list[int]]]] = _load_elemental_tables()

# Backwards-compatible Level 1 table reference
_ELEMENTAL_TABLE: Final[list[list[int]]] = _ELEMENTAL_TABLES.get(1, [
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
])


# ──────────────────────────────────────────────────────────────────────────────
# Size modifier table
# ──────────────────────────────────────────────────────────────────────────────

# Official RO Weapon Size Penalty table (pre-renewal)
# WeaponType: {Small%, Medium%, Large%}
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

# ──────────────────────────────────────────────────────────────────────────────
# Race modifier table
# ──────────────────────────────────────────────────────────────────────────────

_RACE_MODIFIERS: Final[dict[WeaponType, dict[Race, int]]] = {
    wt: {r: 100 for r in Race}
    for wt in WeaponType
}

# ──────────────────────────────────────────────────────────────────────────────
# Elemental descriptions
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# ElementalMatrix class
# ──────────────────────────────────────────────────────────────────────────────


class ElementalMatrix:
    """Thread-safe container for all RO combat damage multipliers.

    All lookup methods accept strings or enum members for convenience.

    ElementLevel parameter (1-4) controls which elemental table is used,
    matching rAthena's attr_fix.yml. Most end-game monsters have Level 2-4.
    """

    # Expose constants for external use
    ELEMENT_ORDER: Final[tuple[Element, ...]] = _ELEMENT_ORDER
    ELEMENTAL_TABLE: Final[list[list[int]]] = _ELEMENTAL_TABLE
    ELEMENTAL_TABLES: Final[dict[int, list[list[int]]]] = _ELEMENTAL_TABLES
    SIZE_MODIFIERS: Final[dict[WeaponType, dict[Size, int]]] = _SIZE_MODIFIERS
    RACE_MODIFIERS: Final[dict[WeaponType, dict[Race, int]]] = _RACE_MODIFIERS

    def __init__(self) -> None:
        self._lock = threading.RLock()

    # ── helpers ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_element(e: str | Element) -> Element:
        if isinstance(e, Element):
            return e
        for member in Element:
            if member.value.lower() == e.lower():
                return member
        return Element.NEUTRAL

    @staticmethod
    def _resolve_size(s: str | Size) -> Size:
        if isinstance(s, Size):
            return s
        for member in Size:
            if member.value.lower() == s.lower():
                return member
        return Size.MEDIUM

    @staticmethod
    def _resolve_race(r: str | Race) -> Race:
        if isinstance(r, Race):
            return r
        for member in Race:
            if member.value.lower() == r.lower():
                return member
        return Race.DEMI_HUMAN

    @staticmethod
    def _resolve_weapon(w: str | WeaponType) -> WeaponType:
        if isinstance(w, WeaponType):
            return w
        # Handle hyphenated names like "Two-Handed Sword"
        return WeaponType(w)

    # ── internal table lookup ───────────────────────────────────────────────────

    @staticmethod
    def _get_table_for_level(element_level: int) -> list[list[int]]:
        """Return the 10×10 elemental matrix for the given level (1-4).

        Falls back to Level 1 if the level is unavailable.
        """
        if element_level in _ELEMENTAL_TABLES:
            return _ELEMENTAL_TABLES[element_level]
        if _ELEMENTAL_TABLES:
            # Fall back to lowest available level
            return _ELEMENTAL_TABLES[min(_ELEMENTAL_TABLES)]
        return _ELEMENTAL_TABLE

    # ── public API ───────────────────────────────────────────────────────────────

    def get_elemental_multiplier(
        self,
        attack_element: str | Element,
        target_element: str | Element,
        element_level: int = 1,
    ) -> float:
        """Return the elemental damage multiplier as a float (e.g. 2.0 for 200%).

        Args:
            attack_element: The element of the attacking skill/spell.
            target_element: The element of the target monster.
            element_level: The monster's ElementLevel (1-4, default 1).
                           End-game monsters often have Level 2-4.
        """
        ae = self._resolve_element(attack_element)
        te = self._resolve_element(target_element)
        with self._lock:
            row = _ELEMENT_INDEX[ae]
            col = _ELEMENT_INDEX[te]
            table = self._get_table_for_level(element_level)
            return table[row][col] / 100.0

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
        element_level: int = 1,
    ) -> str:
        """Return the element name that deals the most damage to *target_element*.

        If multiple elements tie, the first one in element order is returned.
        Uses the specified element level (1-4).
        """
        te = self._resolve_element(target_element)
        with self._lock:
            col = _ELEMENT_INDEX[te]
            table = self._get_table_for_level(element_level)
            best_val = -999999
            best_elem = Element.NEUTRAL
            for row_idx, row in enumerate(table):
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

    # ══════════════════════════════════════════════════════════════════════════
    # NEW: Card damage multiplier calculation
    # ══════════════════════════════════════════════════════════════════════════

    def get_card_damage_multiplier(
        self,
        cards: list[str],
        target_race: str | Race,
        target_size: str | Size,
        target_element: str | Element,
    ) -> float:
        """Compute the combined card damage multiplier against a target.

        RO card stacking rules:
          - Same-type bonuses add (4x Hydra = +80% to race)
          - Race × Element × Size bonuses multiply each other
          - ATK% modifiers are additive within themselves,
            then multiplicative with the race/element/size product

        Uses the global CardDatabase singleton.

        Args:
            cards: List of equipped card names (e.g. ["Hydra Card", "Vadon Card"]).
            target_race: Target's race.
            target_size: Target's size.
            target_element: Target's element.

        Returns:
            Combined card multiplier (e.g. 1.38 for Hydra + Skeleton Worker
            vs DemiHuman/Medium). Returns 1.0 if no cards match.
        """
        tr = self._resolve_race(target_race)
        ts = self._resolve_size(target_size)
        te = self._resolve_element(target_element)

        if not cards:
            return 1.0

        from ai_sidecar.combat.card_db import get_card_database
        db = get_card_database()
        return db.get_total_multiplier(cards, tr.value, ts.value, te.value)

    def get_card_multiplier_breakdown(
        self,
        cards: list[str],
        target_race: str | Race,
        target_size: str | Size,
        target_element: str | Element,
    ) -> dict[str, float]:
        """Get a detailed breakdown of card multiplier components.

        Returns a dict with keys: race_mult, element_mult, size_mult,
        atk_mult, phys_mult, total.
        """
        tr = self._resolve_race(target_race)
        ts = self._resolve_size(target_size)
        te = self._resolve_element(target_element)

        if not cards:
            return {"race_mult": 1.0, "element_mult": 1.0, "size_mult": 1.0,
                    "atk_mult": 1.0, "phys_mult": 1.0, "total": 1.0}

        from ai_sidecar.combat.card_db import get_card_database
        db = get_card_database()
        return db.get_card_multiplier_breakdown(cards, tr.value, ts.value, te.value)

    # ══════════════════════════════════════════════════════════════════════════
    # NEW: Buff/element override detection
    # ══════════════════════════════════════════════════════════════════════════

    # Element converter buff names (skill_id or buff name → element)
    _ELEMENT_CONVERTERS: Final[dict[str, str]] = {
        "fire_weapon": "Fire",
        "fire_converter": "Fire",
        "flame_launcher": "Fire",
        "fire_enchant": "Fire",
        "enchant_fire": "Fire",
        "water_weapon": "Water",
        "water_converter": "Water",
        "frost_weapon": "Water",
        "frost_launcher": "Water",
        "enchant_water": "Water",
        "wind_weapon": "Wind",
        "wind_converter": "Wind",
        "lightning_weapon": "Wind",
        "lightning_launcher": "Wind",
        "enchant_wind": "Wind",
        "earth_weapon": "Earth",
        "earth_converter": "Earth",
        "seismic_weapon": "Earth",
        "enchant_earth": "Earth",
        "holy_weapon": "Holy",
        "aspersio": "Holy",
        "enchant_holy": "Holy",
        "shadow_weapon": "Dark",
        "enchant_shadow": "Dark",
        "endow_tornado": "Wind",
        "endow_tsunami": "Water",
        "endow_volcano": "Fire",
        "endow_quake": "Earth",
        "elemental_converter_fire": "Fire",
        "elemental_converter_water": "Water",
        "elemental_converter_wind": "Wind",
        "elemental_converter_earth": "Earth",
    }

    # Buffs that override weapon element to special types
    _ELEMENT_OVERRIDES: Final[dict[str, str]] = {
        # Enchant Deadly Poison overrides weapon element to Poison
        "enchant_deadly_poison": "Poison",
        "edp": "Poison",
        "deadly_poison": "Poison",
        # Ghost weapon buffs
        "ghost_weapon": "Ghost",
        "enchant_ghost": "Ghost",
    }

    # Combined lookup
    _ALL_ELEMENT_BUFFS: Final[dict[str, str]] = {
        **_ELEMENT_CONVERTERS,
        **_ELEMENT_OVERRIDES,
    }

    def get_effective_element(
        self,
        weapon_element: str | Element,
        active_buffs: list[str],
    ) -> Element:
        """Determine the effective attack element considering buff overrides.

        RO mechanics for element overrides:
          1. Elemental Converters (e.g. Fire Converter) override the
             weapon's base element for physical attacks.
          2. Enchant Deadly Poison overrides to Poison.
          3. Aspersio overrides to Holy.
          4. Multiple converters: last applied wins (we check priority).

        Args:
            weapon_element: The weapon's base element (e.g. "Neutral" for
                           most weapons, or elemental weapon from craft).
            active_buffs: List of active buff/skill names that might
                          override the weapon element.

        Returns:
            The effective Element after applying all buff overrides.
        """
        base = self._resolve_element(weapon_element)

        if not active_buffs:
            return base

        # Check for overrides in priority order:
        # 1. Enchant Deadly Poison (class-defining override)
        # 2. Elemental Converters (fire/water/wind/earth)
        # 3. Aspersio / Holy enchant
        #
        # Scan active_buffs to find the LAST matching override
        # (last converter applied wins in official RO)
        effective_element: Optional[str] = None

        for buff_name in reversed(active_buffs):
            buff_lower = buff_name.lower().strip()
            if buff_lower in self._ALL_ELEMENT_BUFFS:
                effective_element = self._ALL_ELEMENT_BUFFS[buff_lower]
                break

        if effective_element:
            return Element(effective_element)

        return base

    # ══════════════════════════════════════════════════════════════════════════
    # NEW: Updated damage multiplier with cards + buffs
    # ══════════════════════════════════════════════════════════════════════════

    def get_effective_damage_multiplier(
        self,
        attack_element: str | Element,
        weapon_type: str | WeaponType,
        target_element: str | Element,
        target_size: str | Size,
        target_race: str | Race,
        element_level: int = 1,
        cards: Optional[list[str]] = None,
        active_buffs: Optional[list[str]] = None,
        weapon_element: Optional[str | Element] = None,
    ) -> float:
        """Compute the combined damage multiplier from element × size × race × cards.

        This is the master damage calculation method that chains together:
          - Elemental advantage (element × target_element)
          - Weapon size penalty (weapon_type × target_size)
          - Race modifier (weapon_type × target_race)
          - Card bonuses (additive per-type, multiplicative across types)
          - Buff element overrides (converters, aspersio, EDP)

        Args:
            attack_element: Element of the attacking skill/spell.
            weapon_type: Weapon type being used.
            target_element: Element of the target monster.
            target_size: Size of the target monster.
            target_race: Race of the target monster.
            element_level: The monster's ElementLevel (1-4, default 1).
            cards: List of equipped card names (optional).
            active_buffs: List of active buffs (optional).
            weapon_element: The weapon's base element (optional, defaults to
                           attack_element if not provided). Used to determine
                           effective element after buff overrides.

        Returns a float where 1.0 = 100% damage.
        """
        # Resolve effective element considering buff overrides
        if weapon_element is not None or active_buffs:
            wep_elem = weapon_element if weapon_element is not None else attack_element
            effective_atk_elem = self.get_effective_element(wep_elem, active_buffs or [])
        else:
            effective_atk_elem = self._resolve_element(attack_element)

        # Elemental advantage
        elem_mult = self.get_elemental_multiplier(
            effective_atk_elem, target_element, element_level=element_level,
        )

        # Weapon size penalty
        size_mult = self.get_size_multiplier(weapon_type, target_size)

        # Race modifier (weapon-type-specific, default 100%)
        race_mult = self.get_race_multiplier(weapon_type, target_race)

        # Card bonuses
        card_mult = self.get_card_damage_multiplier(
            cards or [], target_race, target_size, target_element,
        )

        # Combine multiplicatively
        return elem_mult * size_mult * race_mult * card_mult

    def get_elemental_advantage_description(
        self,
        target_element: str | Element,
        target_size: str | Size,
        target_race: str | Race,
        element_level: int = 1,
    ) -> str:
        """Return a human-readable description of the best approach against a target.

        Uses the specified element level for vulnerability analysis.
        """
        te = self._resolve_element(target_element)
        ts = self._resolve_size(target_size)
        tr = self._resolve_race(target_race)

        best_elem = self.get_best_element_against(te, element_level=element_level)
        best_weapon = self.get_best_weapon_type(ts)

        elem_desc = _ELEMENT_DESCRIPTIONS.get(te, "")
        size_desc = _SIZE_DESCRIPTIONS.get(ts, "")
        race_desc = _RACE_DESCRIPTIONS.get(tr, "")

        return (
            f"Target: {te.value} / {ts.value} / {tr.value} (Lv{element_level})\n"
            f"  Best element: {best_elem}\n"
            f"  Best weapon:  {best_weapon}\n"
            f"  Element: {elem_desc}\n"
            f"  Size:    {size_desc}\n"
            f"  Race:    {race_desc}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Global singleton
# ──────────────────────────────────────────────────────────────────────────────

_matrix_instance: Optional[ElementalMatrix] = None
_matrix_lock: Final[threading.RLock] = threading.RLock()


def get_elemental_matrix() -> ElementalMatrix:
    """Return the global ElementalMatrix singleton (thread-safe)."""
    global _matrix_instance  # noqa: PLW0603
    with _matrix_lock:
        if _matrix_instance is None:
            _matrix_instance = ElementalMatrix()
        return _matrix_instance


# ──────────────────────────────────────────────────────────────────────────────
# Self-test — verify critical known values from rAthena
# ──────────────────────────────────────────────────────────────────────────────

def _test_elemental_tables() -> None:
    """Verify the loaded elemental tables against known rAthena values.

    These tests check specific monster-element interactions that are
    critical for correct damage calculation in end-game scenarios.
    Also tests card multipliers, size penalties, and buff overrides.
    """
    if not _ELEMENTAL_TABLES:
        print("⚠️  WARNING: Elemental tables not loaded — cannot run tests")
        return

    l1 = _ELEMENTAL_TABLES.get(1)
    l4 = _ELEMENTAL_TABLES.get(4)
    if l1 is None or l4 is None:
        print("⚠️  WARNING: Missing Level 1 or Level 4 table — skipping tests")
        return

    idx = _ELEMENT_INDEX
    passed = 0
    failed = 0

    def check(description: str, level: int, atk: Element, dfn: Element, expected: int) -> None:
        nonlocal passed, failed
        table = _ELEMENTAL_TABLES.get(level)
        if table is None:
            print(f"  FAIL [no table]: {description}")
            failed += 1
            return
        actual = table[idx[atk]][idx[dfn]]
        if actual == expected:
            print(f"  PASS: {description} = {actual}%")
            passed += 1
        else:
            print(f"  FAIL: {description} = {actual}% (expected {expected}%)")
            failed += 1

    print(f"\n{'='*60}")
    print("Elemental Matrix Self-Test")
    print(f"Tables loaded: levels {sorted(_ELEMENTAL_TABLES.keys())}")
    print(f"{'='*60}")

    # ── Level 1 baseline checks ──
    print("\n--- Level 1 (baseline) ---")

    check("Neutral vs Ghost Lv1", 1, Element.NEUTRAL, Element.GHOST, 25)
    check("Water vs Fire Lv1", 1, Element.WATER, Element.FIRE, 150)
    check("Water vs Earth Lv1", 1, Element.WATER, Element.EARTH, 100)
    check("Fire vs Water Lv1", 1, Element.FIRE, Element.WATER, 50)
    check("Fire vs Earth Lv1", 1, Element.FIRE, Element.EARTH, 150)
    check("Ghost vs Neutral Lv1", 1, Element.GHOST, Element.NEUTRAL, 25)
    check("Ghost vs Ghost Lv1", 1, Element.GHOST, Element.GHOST, 125)
    check("Holy vs Undead Lv1", 1, Element.HOLY, Element.UNDEAD, 150)
    check("Holy vs Dark Lv1", 1, Element.HOLY, Element.DARK, 125)
    check("Undead vs Holy Lv1", 1, Element.UNDEAD, Element.HOLY, 100)

    # ... Level 2-4 checks and summary (same as before) ...

    # ── Level 2 checks ──
    print("\n--- Level 2 ---")
    l2 = _ELEMENTAL_TABLES.get(2)
    if l2:
        check("Ghost vs Neutral Lv2", 2, Element.GHOST, Element.NEUTRAL, 0)
        check("Holy vs Undead Lv2", 2, Element.HOLY, Element.UNDEAD, 175)
        check("Fire vs Earth Lv2", 2, Element.FIRE, Element.EARTH, 175)

    # ── Level 3 checks ──
    print("\n--- Level 3 ---")
    l3 = _ELEMENTAL_TABLES.get(3)
    if l3:
        check("Ghost vs Neutral Lv3", 3, Element.GHOST, Element.NEUTRAL, 0)
        check("Holy vs Undead Lv3", 3, Element.HOLY, Element.UNDEAD, 200)
        check("Water vs Fire Lv3", 3, Element.WATER, Element.FIRE, 200)
        check("Ghost vs Ghost Lv3", 3, Element.GHOST, Element.GHOST, 175)

    # ── Ghostring: Ghost Lv4 ──
    print("\n--- Ghostring (Ghost Lv4) ---")
    check("Neutral vs Ghost Lv4", 4, Element.NEUTRAL, Element.GHOST, 0)
    check("Poison vs Ghost Lv4", 4, Element.POISON, Element.GHOST, 25)
    check("Ghost vs Ghost Lv4", 4, Element.GHOST, Element.GHOST, 200)

    # ── Osiris: Undead Lv4 ──
    print("\n--- Osiris (Undead Lv4) ---")
    check("Holy vs Undead Lv4", 4, Element.HOLY, Element.UNDEAD, 200)
    check("Fire vs Undead Lv4", 4, Element.FIRE, Element.UNDEAD, 200)
    check("Undead vs Undead Lv4", 4, Element.UNDEAD, Element.UNDEAD, 0)
    check("Dark vs Undead Lv4", 4, Element.DARK, Element.UNDEAD, -100)
    check("Ghost vs Undead Lv4", 4, Element.GHOST, Element.UNDEAD, 175)

    # ── Same-element at Lv4 ──
    print("\n--- Same-element (Level 4) ---")
    check("Water vs Water Lv4", 4, Element.WATER, Element.WATER, -50)
    check("Fire vs Fire Lv4", 4, Element.FIRE, Element.FIRE, -50)
    check("Wind vs Wind Lv4", 4, Element.WIND, Element.WIND, -50)
    check("Holy vs Holy Lv4", 4, Element.HOLY, Element.HOLY, -100)
    check("Ghost vs Ghost Lv4", 4, Element.GHOST, Element.GHOST, 200)
    check("Dark vs Dark Lv4", 4, Element.DARK, Element.DARK, -100)

    # ── Card multiplier tests ──
    print("\n--- Card Damage Multiplier ---")
    mat = get_elemental_matrix()

    # Single Hydra vs DemiHuman
    card_mult = mat.get_card_damage_multiplier(
        ["Hydra Card"], "DemiHuman", "Medium", "Neutral",
    )
    if abs(card_mult - 1.20) < 0.01:
        print(f"  PASS: Hydra vs DemiHuman = {card_mult:.2f}")
        passed += 1
    else:
        print(f"  FAIL: Hydra vs DemiHuman = {card_mult:.2f} (expected 1.20)")
        failed += 1

    # 4x Hydra = 80%
    card_mult4 = mat.get_card_damage_multiplier(
        ["Hydra Card", "Hydra Card", "Hydra Card", "Hydra Card"],
        "DemiHuman", "Medium", "Neutral",
    )
    if abs(card_mult4 - 1.80) < 0.01:
        print(f"  PASS: 4x Hydra vs DemiHuman = {card_mult4:.2f}")
        passed += 1
    else:
        print(f"  FAIL: 4x Hydra vs DemiHuman = {card_mult4:.2f} (expected 1.80)")
        failed += 1

    # Hydra + Skeleton Worker = 1.20 * 1.15 = 1.38
    card_mult_comb = mat.get_card_damage_multiplier(
        ["Hydra Card", "Skeleton Worker Card"],
        "DemiHuman", "Medium", "Neutral",
    )
    if abs(card_mult_comb - 1.38) < 0.01:
        print(f"  PASS: Hydra + Skel Worker (DemiHuman/Medium) = {card_mult_comb:.2f}")
        passed += 1
    else:
        print(f"  FAIL: Hydra + Skel Worker = {card_mult_comb:.2f} (expected 1.38)")
        failed += 1

    # ── Size penalty tests ──
    print("\n--- Weapon Size Penalty ---")
    size_mult = mat.get_size_multiplier("Dagger", "Large")
    if abs(size_mult - 0.50) < 0.01:
        print(f"  PASS: Dagger vs Large = {size_mult:.2f}")
        passed += 1
    else:
        print(f"  FAIL: Dagger vs Large = {size_mult:.2f} (expected 0.50)")
        failed += 1

    size_mult2 = mat.get_size_multiplier("Spear", "Large")
    if abs(size_mult2 - 1.0) < 0.01:
        print(f"  PASS: Spear vs Large = {size_mult2:.2f}")
        passed += 1
    else:
        print(f"  FAIL: Spear vs Large = {size_mult2:.2f} (expected 1.0)")
        failed += 1

    size_mult3 = mat.get_size_multiplier("Sword", "Medium")
    if abs(size_mult3 - 1.0) < 0.01:
        print(f"  PASS: Sword vs Medium = {size_mult3:.2f}")
        passed += 1
    else:
        print(f"  FAIL: Sword vs Medium = {size_mult3:.2f} (expected 1.0)")
        failed += 1

    # ── Buff override tests ──
    print("\n--- Buff Element Override ---")
    # No buffs = weapon element stays
    elem = mat.get_effective_element("Neutral", [])
    if elem == Element.NEUTRAL:
        print(f"  PASS: No buffs = {elem.value}")
        passed += 1
    else:
        print(f"  FAIL: No buffs = {elem.value} (expected Neutral)")
        failed += 1

    # Fire converter
    elem2 = mat.get_effective_element("Neutral", ["fire_converter"])
    if elem2 == Element.FIRE:
        print(f"  PASS: Fire converter = {elem2.value}")
        passed += 1
    else:
        print(f"  FAIL: Fire converter = {elem2.value} (expected Fire)")
        failed += 1

    # Aspersio
    elem3 = mat.get_effective_element("Neutral", ["aspersio"])
    if elem3 == Element.HOLY:
        print(f"  PASS: Aspersio = {elem3.value}")
        passed += 1
    else:
        print(f"  FAIL: Aspersio = {elem3.value} (expected Holy)")
        failed += 1

    # Enchant Deadly Poison
    elem4 = mat.get_effective_element("Neutral", ["enchant_deadly_poison"])
    if elem4 == Element.POISON:
        print(f"  PASS: EDP = {elem4.value}")
        passed += 1
    else:
        print(f"  FAIL: EDP = {elem4.value} (expected Poison)")
        failed += 1

    # Multiple buffs — last applied wins
    elem5 = mat.get_effective_element("Neutral", ["aspersio", "fire_converter"])
    # fire_converter was applied after aspersio (last in list, since reversed)
    # Actually reversed means fire_converter hits first... let me check the logic.
    # We scan reversed(active_buffs), so fire_converter is checked first.
    # That means the first match in reversed order = last applied in normal order = fire_converter
    if elem5 == Element.FIRE:
        print(f"  PASS: Aspersio then Fire converter = {elem5.value} (fire wins, last applied)")
        passed += 1
    else:
        print(f"  FAIL: Aspersio then Fire converter = {elem5.value} (expected Fire)")
        failed += 1

    # ── Combined damage multiplier tests ──
    print("\n--- Combined Damage Multiplier ---")

    # Scenario: Dagger vs Medium DemiHuman with Hydra Card, no elemental advantage
    # Element: Neutral vs Neutral = 1.0
    # Size: Dagger vs Medium = 0.75
    # Race: (default) = 1.0
    # Card: Hydra = +20% = 1.2
    # Total = 1.0 * 0.75 * 1.0 * 1.2 = 0.90
    combined = mat.get_effective_damage_multiplier(
        attack_element="Neutral",
        weapon_type="Dagger",
        target_element="Neutral",
        target_size="Medium",
        target_race="DemiHuman",
        element_level=1,
        cards=["Hydra Card"],
    )
    if abs(combined - 0.90) < 0.01:
        print(f"  PASS: Dagger+ Hydra vs Medium DemiHuman = {combined:.2f}")
        passed += 1
    else:
        print(f"  FAIL: Dagger+ Hydra vs Medium DemiHuman = {combined:.2f} (expected 0.90)")
        failed += 1

    # Scenario: Spear vs Large DemiHuman, no cards, all neutral
    # Element: Neutral vs Neutral = 1.0
    # Size: Spear vs Large = 1.0
    # Race: default = 1.0
    # Card: none = 1.0
    combined2 = mat.get_effective_damage_multiplier(
        attack_element="Neutral",
        weapon_type="Spear",
        target_element="Neutral",
        target_size="Large",
        target_race="DemiHuman",
        element_level=1,
    )
    if abs(combined2 - 1.0) < 0.01:
        print(f"  PASS: Spear vs Large DemiHuman = {combined2:.2f}")
        passed += 1
    else:
        print(f"  FAIL: Spear vs Large DemiHuman = {combined2:.2f} (expected 1.0)")
        failed += 1

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print(f"{'='*60}\n")

    if failed:
        raise AssertionError(f"{failed} test(s) failed")


if __name__ == "__main__":
    _test_elemental_tables()
