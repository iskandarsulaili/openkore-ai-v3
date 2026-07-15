"""
Monster database loader — reads real rAthena mob_db.yml.

Pro RO Player knowledge is already baked into KNOWN_AGGRESSIVE_MONSTERS in
predictive_aggro.py.  This module *augments* that with the full 2,675-entry
rAthena database so the bot can look up ANY monster it encounters, not just
the ~100 we've hand-picked.

The loader is lazy (on first use) to keep bot startup fast.
"""

from __future__ import annotations

import logging
import os
import time
from threading import RLock
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_MOB_DB_PATH = os.path.join(
    _PROJECT_ROOT,
    "knowledge", "rathena_db", "db", "pre-re", "mob_db.yml",
)
_ATTR_FIX_PATH = os.path.join(
    _PROJECT_ROOT,
    "knowledge", "rathena_db", "db", "pre-re", "attr_fix.yml",
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class MonsterEntry:
    """One monster from mob_db.yml."""

    __slots__ = (
        "id", "aegis_name", "name", "level",
        "hp", "base_exp", "job_exp", "mvp_exp",
        "atk_min", "atk_max",
        "def_", "mdef_",
        "str_val", "agi_val", "vit_val",
        "int_val", "dex_val", "luk_val",
        "atk_range", "skill_range", "chase_range",
        "size", "race", "element", "element_level",
        "is_boss",
    )

    def __init__(self, raw: dict[str, Any]) -> None:
        self.id: int = raw.get("Id", 0)
        self.aegis_name: str = str(raw.get("AegisName", "")).upper()
        self.name: str = str(raw.get("Name", self.aegis_name))
        self.level: int = int(raw.get("Level", 1))
        self.hp: int = int(raw.get("Hp", 1))
        self.base_exp: int = int(raw.get("BaseExp", 0))
        self.job_exp: int = int(raw.get("JobExp", 0))
        self.mvp_exp: int = int(raw.get("MvpExp", 0))

        atk = raw.get("Attack", 0)
        atk2 = raw.get("Attack2", 0)
        self.atk_min: int = int(atk) if atk else 0
        self.atk_max: int = int(atk2) if atk2 else int(atk) if atk else 0

        self.def_: int = int(raw.get("Defense", 0))
        self.mdef_: int = int(raw.get("MagicDefense", 0))

        self.str_val: int = int(raw.get("Str", 1))
        self.agi_val: int = int(raw.get("Agi", 1))
        self.vit_val: int = int(raw.get("Vit", 1))
        self.int_val: int = int(raw.get("Int", 1))
        self.dex_val: int = int(raw.get("Dex", 1))
        self.luk_val: int = int(raw.get("Luk", 1))

        self.atk_range: int = int(raw.get("AttackRange", 0))
        self.skill_range: int = int(raw.get("SkillRange", 0))
        self.chase_range: int = int(raw.get("ChaseRange", 0))

        self.size: str = str(raw.get("Size", "Small")).lower()
        self.race: str = str(raw.get("Race", "Formless")).lower()
        self.element: str = str(raw.get("Element", "Neutral")).lower()
        self.element_level: int = int(raw.get("ElementLevel", 1))

        # Boss detection
        cls_val = str(raw.get("Class", "Normal")).lower()
        modes = raw.get("Modes", {})
        self.is_boss: bool = (
            cls_val in ("boss", "mini")
            or (isinstance(modes, dict) and modes.get("MVP", False))
        )

    def __repr__(self) -> str:
        return f"MonsterEntry(id={self.id}, name={self.name}, lv={self.level})"


# ---------------------------------------------------------------------------
# Element Level → multiplier table (pre-renewal)
# ---------------------------------------------------------------------------
# ElementLevel 1 = standard chart
# ElementLevel 2 = attacker element × 2 (elemental multiplier is doubled)
# ElementLevel 3 = attacker element × 3
# ElementLevel 4 = attacker element × 4 (special: same element = 2×,
#                   neutral = 25% to neutral)
#
# The base chart is in combat_tactics.py ELEMENT_MULT.  This module
# derives the 4-level tables from it.

_BASE_ELEMENT_MULT: dict[str, dict[str, float]] = {
    "neutral":  {"neutral": 1.0, "water": 1.0, "earth": 1.0, "fire": 1.0, "wind": 1.0,
                 "poison": 1.0, "holy": 1.0, "dark": 1.0, "ghost": 0.75, "undead": 1.0},
    "water":    {"neutral": 1.0, "water": 0.25, "earth": 0.75, "fire": 1.0, "wind": 0.75,
                 "poison": 1.0, "holy": 1.0, "dark": 1.0, "ghost": 0.5, "undead": 1.0},
    "earth":    {"neutral": 1.0, "water": 1.0, "earth": 0.25, "fire": 0.75, "wind": 1.0,
                 "poison": 1.0, "holy": 1.0, "dark": 1.0, "ghost": 0.5, "undead": 1.0},
    "fire":     {"neutral": 1.0, "water": 0.5, "earth": 1.0, "fire": 0.25, "wind": 0.75,
                 "poison": 1.0, "holy": 1.0, "dark": 1.0, "ghost": 0.5, "undead": 1.25},
    "wind":     {"neutral": 1.0, "water": 1.0, "earth": 0.5, "fire": 1.0, "wind": 0.25,
                 "poison": 1.0, "holy": 1.0, "dark": 1.0, "ghost": 0.5, "undead": 1.0},
    "poison":   {"neutral": 1.0, "water": 1.0, "earth": 0.75, "fire": 1.0, "wind": 1.0,
                 "poison": 0.25, "holy": 1.0, "dark": 1.0, "ghost": 0.5, "undead": 0.5},
    "holy":     {"neutral": 1.0, "water": 1.0, "earth": 1.0, "fire": 1.0, "wind": 1.0,
                 "poison": 1.0, "holy": 0.25, "dark": 2.0, "ghost": 1.0, "undead": 1.5},
    "dark":     {"neutral": 1.0, "water": 1.0, "earth": 1.0, "fire": 1.0, "wind": 1.0,
                 "poison": 1.0, "holy": 0.5, "dark": 0.25, "ghost": 1.0, "undead": 1.0},
    "ghost":    {"neutral": 0.0, "water": 1.0, "earth": 1.0, "fire": 1.0, "wind": 1.0,
                 "poison": 1.0, "holy": 1.0, "dark": 1.0, "ghost": 0.5, "undead": 1.0},
    "undead":   {"neutral": 1.0, "water": 1.0, "earth": 1.0, "fire": 1.25, "wind": 1.0,
                 "poison": 0.5, "holy": 2.0, "dark": 0.5, "ghost": 1.0, "undead": 0.5},
}


def _build_element_chart(element_level: int) -> dict[str, dict[str, float]]:
    """Build the full 10-element × 10-element chart for a given ElementLevel.

    Pre-renewal rules (rAthena):
      Level 1:  base chart.
      Level 2:  attacker element multiplier × 2 (cap 4.0).
                neutral → neutral: 2.0
      Level 3:  attacker element multiplier × 3 (cap 6.0).
                neutral → neutral: 3.0
      Level 4:  attacker element multiplier × 4 (cap 8.0).
                neutral → neutral: 1.0
                same element: 2.0
                neutral → same elem: 0.25
    """
    charts: dict[str, dict[str, float]] = {}
    base = _BASE_ELEMENT_MULT

    for atk_elem, def_map in base.items():
        charts[atk_elem] = {}
        for def_elem, mult in def_map.items():
            if element_level == 1:
                charts[atk_elem][def_elem] = mult
            elif element_level == 2:
                if atk_elem == def_elem:
                    charts[atk_elem][def_elem] = min(4.0, mult * 2)
                elif atk_elem == "neutral" and def_elem == "neutral":
                    charts[atk_elem][def_elem] = 2.0
                else:
                    charts[atk_elem][def_elem] = min(4.0, mult * 2)
            elif element_level == 3:
                if atk_elem == def_elem:
                    charts[atk_elem][def_elem] = min(6.0, mult * 3)
                elif atk_elem == "neutral" and def_elem == "neutral":
                    charts[atk_elem][def_elem] = 3.0
                else:
                    charts[atk_elem][def_elem] = min(6.0, mult * 3)
            elif element_level >= 4:
                if atk_elem == def_elem:
                    charts[atk_elem][def_elem] = min(8.0, mult * 4)
                elif atk_elem == "neutral" and def_elem == "neutral":
                    charts[atk_elem][def_elem] = 1.0
                else:
                    charts[atk_elem][def_elem] = min(8.0, mult * 4)
    return charts


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_monster_db(path: Optional[str] = None) -> dict[str, MonsterEntry]:
    """Load the full monster database from mob_db.yml.

    Returns a dict keyed by AegisName (uppercase).
    """
    path = path or _MOB_DB_PATH
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not available — monster DB will not load from file")
        return {}

    if not os.path.isfile(path):
        logger.warning("mob_db.yml not found at %s — using built-in data only", path)
        return {}

    start = time.time()
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    body = data.get("Body", []) if isinstance(data, dict) else []
    db: dict[str, MonsterEntry] = {}
    for entry in body:
        if not isinstance(entry, dict):
            continue
        if "AegisName" not in entry:
            continue
        try:
            m = MonsterEntry(entry)
            db[m.aegis_name] = m
        except Exception as e:
            logger.debug("Failed to parse monster entry: %s — %s", entry.get("AegisName"), e)

    logger.info(
        "monster_db loaded: %d monsters in %.2fs from %s",
        len(db), time.time() - start, path,
    )
    return db


def load_element_charts() -> dict[int, dict[str, dict[str, float]]]:
    """Build and return element charts for levels 1-4.

    Returns {1: chart, 2: chart, 3: chart, 4: chart}.
    """
    charts: dict[int, dict[str, dict[str, float]]] = {}
    for lvl in (1, 2, 3, 4):
        charts[lvl] = _build_element_chart(lvl)
    logger.info("element_charts built for levels 1-4")
    return charts


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_monster_db: Optional[dict[str, MonsterEntry]] = None
_element_charts: Optional[dict[int, dict[str, dict[str, float]]]] = None
_db_lock = RLock()


def get_monster_db() -> dict[str, MonsterEntry]:
    """Return the global monster database (lazy-loaded)."""
    global _monster_db
    with _db_lock:
        if _monster_db is None:
            _monster_db = load_monster_db()
        return _monster_db


def get_element_charts() -> dict[int, dict[str, dict[str, float]]]:
    """Return the global element charts for levels 1-4 (lazy-loaded)."""
    global _element_charts
    with _db_lock:
        if _element_charts is None:
            _element_charts = load_element_charts()
        return _element_charts
