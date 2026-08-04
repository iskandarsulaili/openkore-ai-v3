"""Renewal-era gear mechanics knowledge (server-agnostic, loaded from the server's DBs).

Covers the Renewal gear systems the server defines: ENCHANTGRADE (grade chances by
item level + refine), RANDOMOPT (random option effects on gear), and ITEM_REFORM
(upgrade/reform system). The server may configure none, some, or all of these; the
loader gracefully reports what is present so gear scoring can account for it.

This is a KNOWLEDGE loader, not a decision maker — it feeds the GearScorer /
equipment reasoning with the mechanics the server actually has, so the bot's gear
optimization is server-agnostic (a fresh server with different DBs yields different
awareness automatically).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _db_paths() -> list[Path]:
    """Candidate paths to the rathena DB directory (re then pre-re), newest first."""
    base = Path(__file__).resolve().parent.parent.parent.parent  # AI_sidecar/
    candidates = [
        base / "knowledge" / "rathena_db" / "db" / "re",
        base / "knowledge" / "rathena_db" / "db" / "pre-re",
        Path.home() / "rathena" / "db" / "re",
        Path.home() / "rathena-AI-world" / "db" / "re",
    ]
    return [p for p in candidates if p.is_dir()]


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("renewal_gear: failed to load %s: %s", path, exc)
        return {}


class EnchantGrade:
    """Enchant-grade availability per item type + level (Renewal ENCHANTGRADE_DB)."""

    def __init__(self, grades_by_type_level: dict[str, dict[int, list[str]]]) -> None:
        # grades_by_type_level: {"Weapon": {5: ["None","D","C","B"]}, ...}
        self._by_type_level = grades_by_type_level

    def grades_for(self, item_type: str, item_level: int) -> list[str]:
        """Grades available for an item type+level, or [] if the server has none."""
        return list(self._by_type_level.get(item_type, {}).get(item_level, []) or [])

    def has_grade_system(self) -> bool:
        return bool(self._by_type_level)

    def grade_index(self, grade: str, item_type: str, item_level: int) -> int:
        """Rank of a grade in the item's available grade list (higher = better)."""
        g = self.grades_for(item_type, item_level)
        try:
            return g.index(grade)
        except ValueError:
            return 0

    def __bool__(self) -> bool:
        return self.has_grade_system()


class RenewalGearKnowledge:
    """Aggregated Renewal gear mechanics loaded from the server DBs (server-agnostic)."""

    def __init__(self, db_dir: Path | None = None) -> None:
        self._enchantgrade = EnchantGrade({})
        self._has_randomopt = False
        self._randomopt_count = 0
        self._has_item_reform = False
        self._reform_count = 0
        self._load(db_dir)

    def _load(self, db_dir: Path | None) -> None:
        # Resolve the DB dir (explicit, or first available re/pre-re dir).
        if db_dir is None:
            dirs = _db_paths()
            if not dirs:
                logger.warning("renewal_gear: no rathena db dir found")
                return
            db_dir = dirs[0]
        if not db_dir.is_dir():
            logger.warning("renewal_gear: db dir %s not found", db_dir)
            return

        # ENCHANTGRADE_DB
        _eg_path = db_dir / "enchantgrade.yml"
        if _eg_path.is_file():
            _eg = _load_yaml(_eg_path)
            _by: dict[str, dict[int, list[str]]] = {}
            for _entry in _eg.get("Body", []):
                if not isinstance(_entry, dict):
                    continue
                _type = str(_entry.get("Type", ""))
                _levels = _entry.get("Levels", []) or []
                for _lv in _levels:
                    if not isinstance(_lv, dict):
                        continue
                    _lv_num = int(_lv.get("Level", 0) or 0)
                    _grades = [str(_g.get("Grade", "")) for _g in (_lv.get("Grades", []) or []) if isinstance(_g, dict) and _g.get("Grade")]
                    if _lv_num and _grades:
                        _by.setdefault(_type, {})[_lv_num] = _grades
            self._enchantgrade = EnchantGrade(_by)
            logger.info("renewal_gear: enchantgrade loaded %d type/level sets", len(_by))

        # RANDOMOPT_DB
        _ro_path = db_dir / "randomopt_db.yml"
        if _ro_path.is_file():
            _ro = _load_yaml(_ro_path)
            _body = _ro.get("Body", []) or []
            self._has_randomopt = bool(_body)
            self._randomopt_count = len(_body)
            logger.info("renewal_gear: randomopt %s (%d options)",
                        "present" if self._has_randomopt else "absent", self._randomopt_count)

        # ITEM_REFORM_DB
        _rf_path = db_dir / "item_reform.yml"
        if _rf_path.is_file():
            _rf = _load_yaml(_rf_path)
            _body = _rf.get("Body", []) or []
            self._has_item_reform = bool(_body)
            self._reform_count = len(_body)
            logger.info("renewal_gear: item_reform %s (%d entries)",
                        "present" if self._has_item_reform else "absent", self._reform_count)

    @property
    def enchantgrade(self) -> EnchantGrade:
        return self._enchantgrade

    @property
    def has_randomopt(self) -> bool:
        return self._has_randomopt

    @property
    def randomopt_count(self) -> int:
        return self._randomopt_count

    @property
    def has_item_reform(self) -> bool:
        return self._has_item_reform

    @property
    def reform_count(self) -> int:
        return self._reform_count

    def status(self) -> dict[str, Any]:
        """Report which Renewal gear mechanics this server configures (for the tracker/log)."""
        return {
            "enchantgrade": self._enchantgrade.has_grade_system(),
            "randomopt": self._has_randomopt,
            "randomopt_count": self._randomopt_count,
            "item_reform": self._has_item_reform,
            "reform_count": self._reform_count,
        }


# Module-level singleton (loaded once).
_KNOWLEDGE: RenewalGearKnowledge | None = None


def get_renewal_gear_knowledge() -> RenewalGearKnowledge:
    """Return the process-wide Renewal gear knowledge (loads on first call)."""
    global _KNOWLEDGE
    if _KNOWLEDGE is None:
        _KNOWLEDGE = RenewalGearKnowledge()
    return _KNOWLEDGE
