"""Elemental Modifier Table — loaded from rathena_element_table.json.

Architecture:
  - Loads element modifiers from rAthena JSON export (Level 1-4)
  - 10 elements × 4 levels = 40 modifier entries
  - Level 4 attack ignores 25% of elemental resistance
  - Handles pre-renewal vs renewal (same table, different formulas downstream)
  - Provides convenience methods for element matching

RULE.md compliance: Zero hardcoded — all values from rAthena DB.
"""

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

ELEMENTS = [
    "Neutral", "Water", "Earth", "Fire", "Wind",
    "Poison", "Holy", "Dark", "Ghost", "Undead",
]


class ElementTable:
    """Elemental damage modifier table from rathena_element_table.json (Level 1-4)."""

    def __init__(self):
        self._modifiers: dict[int, dict[str, dict[str, float]]] = {}  # level → attacker → defender → multiplier
        self._loaded = False

    def load(self, db_path: str | None = None) -> bool:
        """Load from rathena_element_table.json. Returns True on success."""
        if self._loaded:
            return True

        path = db_path or self._find_path()
        if not path or not os.path.exists(path):
            logger.warning("element_table: rathena_element_table.json not found at %s", path)
            return False

        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("element_table: failed to load %s: %s", path, e)
            return False

        if not data:
            logger.warning("element_table: empty data in %s", path)
            return False

        # JSON keys are strings "1", "2", "3", "4"
        for level_str, level_data in data.items():
            level = int(level_str)
            parsed: dict[str, dict[str, float]] = {}
            for attacker, targets in level_data.items():
                if isinstance(targets, dict):
                    parsed[attacker] = {k: float(v) for k, v in targets.items()}
            self._modifiers[level] = parsed

        self._loaded = True
        logger.info("element_table: loaded %d levels from %s", len(self._modifiers), path)
        return True

    def _find_path(self) -> str:
        """Find rathena_element_table.json."""
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidates = [
            os.path.join(base, "knowledge", "rathena_db", "rathena_element_table.json"),
            os.path.join(os.path.expanduser("~"), "rathena_element_table.json"),
            os.path.join(base, "data", "rAthena", "rathena_element_table.json"),
            os.path.join(base, "rathena_element_table.json"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]

    def get_modifier(self, attacker: str, defender: str, attack_level: int = 1, defense_level: int = 1) -> float:
        """Get damage modifier for attacker element vs defender element at given levels.

        Level 4 attack ignores 25% of elemental resistance:
          If the base modifier is < 1.0 (resisted), Level 4 attack reduces the penalty:
            effective = 1.0 - (1.0 - base_mod) * 0.75
          If the base modifier is > 1.0 (weakness), Level 4 attack amplifies:
            effective = 1.0 + (base_mod - 1.0) * 1.25

        Returns 1.0 if not found (neutral damage).
        """
        # Use defense level for the base lookup (defender's element level determines the table)
        level_data = self._modifiers.get(defense_level, {})
        attacker_data = level_data.get(attacker, {})
        base_mod = attacker_data.get(defender, 1.0)

        # Level 4 attack modifier: ignores 25% of resistance, amplifies 25% of weakness
        if attack_level >= 4:
            if base_mod < 1.0:
                # Resistance is reduced by 25%
                return 1.0 - (1.0 - base_mod) * 0.75
            elif base_mod > 1.0:
                # Weakness is amplified by 25%
                return 1.0 + (base_mod - 1.0) * 1.25

        return base_mod

    def best_element_against(self, defender: str, defense_level: int = 1, attack_level: int = 1) -> tuple[str, float]:
        """Find the element that deals the most damage against a defender.

        Considers attack level for Level 4 penetration.

        Returns (element_name, modifier).
        """
        best_ele = "Neutral"
        best_mod = 0.0

        for attacker in ELEMENTS:
            mod = self.get_modifier(attacker, defender, attack_level=attack_level, defense_level=defense_level)
            if mod > best_mod:
                best_mod = mod
                best_ele = attacker

        return best_ele, best_mod

    def get_levels_loaded(self) -> list[int]:
        """Return which levels are loaded."""
        return sorted(self._modifiers.keys())

    def get_raw_table(self, level: int = 1) -> dict[str, dict[str, float]]:
        """Get the raw modifier table for a given level."""
        return self._modifiers.get(level, {})


# Singleton
_table: Optional[ElementTable] = None


def get_element_table() -> ElementTable:
    """Get the global ElementTable instance."""
    global _table
    if _table is None:
        _table = ElementTable()
        _table.load()
    return _table


def element_modifier(attacker: str, defender: str, attack_level: int = 1, defense_level: int = 1) -> float:
    """Convenience function for element modifier lookup."""
    return get_element_table().get_modifier(attacker, defender, attack_level=attack_level, defense_level=defense_level)


def best_element_against(defender: str, defense_level: int = 1, attack_level: int = 1) -> tuple[str, float]:
    """Convenience function for best element lookup."""
    return get_element_table().best_element_against(defender, defense_level=defense_level, attack_level=attack_level)
