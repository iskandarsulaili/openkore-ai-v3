"""Elemental Modifier Table — loaded from attr_fix.yml.

Architecture:
  - Loads element modifiers from rAthena DB (attr_fix.yml)
  - 10 elements × 4 levels = 40 modifier entries
  - Handles pre-renewal vs renewal (same table, different formulas downstream)
  - Provides convenience methods for element matching

RULE.md compliance: Zero hardcoded — all values from rAthena DB.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

ELEMENTS = [
    "Neutral", "Water", "Earth", "Fire", "Wind",
    "Poison", "Holy", "Dark", "Ghost", "Undead",
]


class ElementTable:
    """Elemental damage modifier table from attr_fix.yml."""

    def __init__(self):
        self._modifiers: dict[int, dict[str, dict[str, int]]] = {}  # level → attacker → defender → %
        self._loaded = False
    
    def load(self, db_path: str | None = None) -> bool:
        """Load from attr_fix.yml. Returns True on success."""
        if self._loaded:
            return True
        
        path = db_path or self._find_path()
        if not path or not os.path.exists(path):
            logger.warning("element_table: attr_fix.yml not found")
            return False
        
        try:
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f)
        except Exception as e:
            logger.warning("element_table: failed to load %s: %s", path, e)
            return False
        
        if not data or "Body" not in data:
            logger.warning("element_table: invalid format in %s", path)
            return False
        
        for entry in data["Body"]:
            level = entry.get("Level", 1)
            level_data = {}
            for attacker, targets in entry.items():
                if attacker == "Level":
                    continue
                if isinstance(targets, dict):
                    level_data[attacker] = {k: int(v) for k, v in targets.items()}
            self._modifiers[level] = level_data
        
        self._loaded = True
        logger.info("element_table: loaded %d levels from %s", len(self._modifiers), path)
        return True
    
    def _find_path(self) -> str:
        """Find attr_fix.yml — try pre-re and re versions."""
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # Try knowledge/rathena_db first
        candidates = [
            os.path.join(base, "knowledge", "rathena_db", "db", "re", "attr_fix.yml"),
            os.path.join(base, "knowledge", "rathena_db", "db", "pre-re", "attr_fix.yml"),
            os.path.join(os.path.expanduser("~"), "rathena", "db", "re", "attr_fix.yml"),
            os.path.join(os.path.expanduser("~"), "rathena", "db", "pre-re", "attr_fix.yml"),
            os.path.join(base, "knowledge", "rathena_db", "db", "attr_fix.yml"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]
    
    def get_modifier(self, attacker: str, defender: str, level: int = 1) -> int:
        """Get damage modifier % for attacker element vs defender element at given level.
        
        Returns 100 if not found (neutral damage).
        """
        level_data = self._modifiers.get(level, {})
        attacker_data = level_data.get(attacker, {})
        return attacker_data.get(defender, 100)
    
    def best_element_against(self, defender: str, defender_level: int = 1) -> tuple[str, int]:
        """Find the element that deals the most damage against a defender.
        
        Returns (element_name, modifier_percent).
        """
        best_ele = "Neutral"
        best_mod = 100
        
        for attacker in ELEMENTS:
            mod = self.get_modifier(attacker, defender, defender_level)
            if mod > best_mod:
                best_mod = mod
                best_ele = attacker
        
        return best_ele, best_mod


# Singleton
_table: Optional[ElementTable] = None


def get_element_table() -> ElementTable:
    """Get the global ElementTable instance."""
    global _table
    if _table is None:
        _table = ElementTable()
        _table.load()
    return _table


def element_modifier(attacker: str, defender: str, level: int = 1) -> int:
    """Convenience function for element modifier lookup."""
    return get_element_table().get_modifier(attacker, defender, level)


def best_element_against(defender: str, defender_level: int = 1) -> tuple[str, int]:
    """Convenience function for best element lookup."""
    return get_element_table().best_element_against(defender, defender_level)
