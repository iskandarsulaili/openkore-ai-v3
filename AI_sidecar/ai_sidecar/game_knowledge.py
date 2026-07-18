"""
GameKnowledgeService — data-driven RO knowledge for CrewAI agents.

Replaces hardcoded CLASS_EARLY_GAME and agent-specific dicts with
dynamic lookups from the knowledge database (knowledge.json + OpenKore tables).

All agents (Pro RO Player, Combat, Navigation, Economy) query this service
instead of using hardcoded constants, making the system adaptive to
any job class, server rates, and game version.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_KNOWLEDGE_PATH = Path(__file__).resolve().parent.parent / "knowledge" / "knowledge.json"

# ── Known indoor maps (bots spawn inside buildings) ──
INDOOR_MAPS: set[str] = {
    "prt_in", "morocc_in", "payon_in", "geffen_in", "alberta_in",
    "aldebaran_in", "izlude_in", "comodo_in", "rachel_in", "veins_in",
    "yuno_in", "xmas_in", "um_in", "niflheim_in", "einbroch_in",
    "lighthalzen_in",
}

# ── Estimated exit coordinates for indoor maps (fallback nudges) ──
INDOOR_EXITS: dict[str, tuple[int, int]] = {
    "prt_in": (131, 136),
    "morocc_in": (80, 110),
    "payon_in": (120, 100),
    "geffen_in": (50, 80),
    "alberta_in": (90, 120),
    "izlude_in": (70, 90),
}

# ── Level-based hunting map ladder (server-agnostic defaults) ──
LEVEL_LADDER: list[tuple[int, int, str, str]] = [
    (1, 15, "prt_fild00", "Porings, Lunatics, Fabres — safe for all classes"),
    (15, 30, "payon", "Payon Cave 1: Popolions, Wolves"),
    (25, 40, "payon", "Payon Cave 2: Drainliars, Munaks"),
    (35, 55, "gef_fild04", "Orc Village: Orcs, Orc Warriors"),
    (50, 70, "ein_fild07", "High Orcs, Injustice"),
    (60, 80, "glastheim", "Glastheim: Strouf, Rideword"),
    (70, 90, "thana_boss", "Thanatos Tower: Alarm, Clock"),
    (85, 99, "abyss_03", "Abyss Lake: Mysteltainn, Executioner"),
]


class GameKnowledgeService:
    """Thread-safe, cached service for RO game knowledge.

    Loads knowledge.json once and provides query methods that all
    CrewAI agents use instead of hardcoded class/strategy data.
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._loaded = False
        self._load()

    # ── Loading ────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            p = _KNOWLEDGE_PATH
            if not p.exists():
                logger.warning("game_knowledge: knowledge.json not found at %s", p)
                self._data = {}
                return
            with open(p, encoding="utf-8") as f:
                self._data = json.load(f)
            self._loaded = True
            logger.info(
                "game_knowledge_loaded: jobs=%d mobs=%d skills=%d",
                len(self._data.get("job_stats", {})),
                len(self._data.get("mobs", [])),
                len(self._data.get("skill_trees", [])),
            )
        except Exception as exc:
            logger.exception("game_knowledge_load_failed: %s", exc)
            self._data = {}

    # ── Job / Class helpers ────────────────────────────────────────

    def all_jobs(self) -> list[str]:
        """Return all known job class names (lowercase)."""
        return list(self._data.get("job_stats", {}).keys())

    @lru_cache(maxsize=256)
    def job_stats(self, job_class: str) -> dict[str, Any]:
        """Return stat bonuses, weapons, etc. for a job class.

        Returns empty dict if job is unknown — caller should treat
        as 'novice'-equivalent.
        """
        raw = self._data.get("job_stats", {}).get(job_class.lower(), {})
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, list) and raw:
            return raw[0] if isinstance(raw[0], dict) else {}
        return {}

    def job_weapon_types(self, job_class: str) -> list[str]:
        """Weapon types a job can equip (from knowledge DB)."""
        stats = self.job_stats(job_class)
        return list(stats.get("WeaponTypes", stats.get("weapons", [])))

    @lru_cache(maxsize=256)
    def skill_tree(self, job_class: str) -> list[dict[str, Any]]:
        """Skill tree for a job — skills, max levels, descriptions."""
        trees = self._data.get("skill_trees", [])
        for t in trees:
            job_name = (t.get("Job") or "").lower()
            if job_name == job_class.lower():
                return t.get("Tree", [])
        return []

    def starting_stats(self, job_class: str) -> str:
        """Recommended stat build for a job class (adaptive heuristic)."""
        stats = self.job_stats(job_class)
        bonuses = stats.get("BonusStats", {})
        if isinstance(bonuses, dict):
            # Jobs that get STR bonuses → STR build
            str_bonus = int(bonuses.get("str", bonuses.get("Str", 0)) or 0)
            agi_bonus = int(bonuses.get("agi", bonuses.get("Agi", 0)) or 0)
            dex_bonus = int(bonuses.get("dex", bonuses.get("Dex", 0)) or 0)
            int_bonus = int(bonuses.get("int", bonuses.get("Int", 0)) or 0)

            # Heuristic: pick primary stat based on highest bonus
            scores = {"str": str_bonus, "agi": agi_bonus, "dex": dex_bonus, "int": int_bonus}
            primary = max(scores, key=scores.get)
            if primary == "int":
                return "INT 40 > DEX 30 > rest INT"
            elif primary == "dex":
                return "DEX 40 > AGI 30 > rest DEX"
            elif primary == "agi":
                return "AGI 40 > DEX 30 > rest STR"
            else:
                return "STR 40 > DEX 30 > rest STR"
        return "STR 20 > DEX 20 > rest STR"  # fallback

    # ── Hunting / Leveling ─────────────────────────────────────────

    @lru_cache(maxsize=128)
    def recommended_map(self, level: int, job_class: str = "") -> tuple[str, str]:
        """Best hunting map for given level.

        Returns (map_name, description). Falls back to LEVEL_LADDER.
        """
        for min_lv, max_lv, map_name, desc in LEVEL_LADDER:
            if min_lv <= level <= max_lv:
                return map_name, desc
        # Level too high for ladder — return last entry
        return LEVEL_LADDER[-1][2], LEVEL_LADDER[-1][3]

    @lru_cache(maxsize=256)
    def monsters_for_map(self, map_name: str) -> list[dict[str, Any]]:
        """Monsters that spawn on a given map (from mobs + map_drops)."""
        mobs = self._data.get("mobs", [])
        map_name_lower = map_name.lower()
        results = []
        for m in mobs:
            # Check if monster spawns on this map
            spawns = m.get("Spawns", m.get("spawns", []))
            # spawns might be a list of map names
            if spawns and isinstance(spawns, list):
                for s in spawns:
                    if isinstance(s, str) and map_name_lower in s.lower():
                        results.append(m)
                        break
                    elif isinstance(s, dict) and map_name_lower in str(s.get("map", "")).lower():
                        results.append(m)
                        break
            # Some entries have field 'Maps' or 'maps'
            maps_field = m.get("Maps", m.get("maps", []))
            if maps_field and isinstance(maps_field, list):
                for mf in maps_field:
                    if map_name_lower in (mf.lower() if isinstance(mf, str) else str(mf)):
                        if m not in results:
                            results.append(m)
                            break
        return results[:20]  # cap

    def monsters_by_level_range(self, min_lv: int, max_lv: int) -> list[dict[str, Any]]:
        """Monsters within a level range (for finding appropriate grinding)."""
        mobs = self._data.get("mobs", [])
        results = []
        for m in mobs:
            lv = int(m.get("Level", 0) or 0)
            if min_lv <= lv <= max_lv:
                results.append(m)
        return sorted(results, key=lambda x: int(x.get("Level", 0) or 0))[:30]

    # ── Equipment ──────────────────────────────────────────────────

    @lru_cache(maxsize=128)
    def equipment_by_level(self, level: int, job_class: str = "") -> str:
        """Simple equipment recommendation based on level."""
        if level < 15:
            return "Cotton Shirt, Sandals, Sword[3] + 3x Fabre Card"
        elif level < 30:
            return "Chain Mail, Buckler, Boots, Main Gauche[4] + 4x Fabre Card"
        elif level < 50:
            return "Silver Robe, Manteau, Finger of ROG"
        elif level < 70:
            return "Hood[1], Pantie, Shoes[1]"
        elif level < 90:
            return "Skeleton Armor, Biretta, High Heels"
        else:
            return "Elven Chain, Manteau of Toad, Boots of RNG"

    # ── Economy ────────────────────────────────────────────────────

    @lru_cache(maxsize=256)
    def item_value(self, item_name: str) -> float:
        """Estimated Zeny value of an item (from item DB if available)."""
        item_types = self._data.get("items", {})
        for category, items in item_types.items():
            if isinstance(items, dict):
                for name, info in items.items():
                    if item_name.lower() in name.lower():
                        return float(info.get("value", info.get("sell", 0)) or 0)
        return 0.0

    # ── Utility ─────────────────────────────────────────────────────

    def is_indoor_map(self, map_name: str) -> bool:
        """Check if a map name is an indoor/building map."""
        if not map_name:
            return False
        m = map_name.lower().strip()
        # Remove file extension
        m = m.replace(".gat", "").replace(".rsw", "").strip()
        for indoor in INDOOR_MAPS:
            if m.startswith(indoor):
                return True
        return False

    def indoor_exit(self, map_name: str) -> tuple[int, int] | None:
        """Estimated exit coordinates for an indoor map."""
        if not map_name:
            return None
        m = map_name.lower().strip().replace(".gat", "").replace(".rsw", "").strip()
        return INDOOR_EXITS.get(m)

    def town_prefixes(self) -> list[str]:
        """Return known town map prefixes for server-agnostic town detection."""
        return ["prontera", "morocc", "payon", "geffen", "aldebaran", "yuno", "xmas", "amatsu", "alberta", "izlude", "comodo", "umbala", "niflheim", "rachel", "veins", "moscovia", "einbroch", "lighthalzen", "hugel"]

    def town_for_map(self, map_name: str) -> str:
        """Return town name for a given map by matching known prefixes."""
        if not map_name:
            return "prontera"
        m = map_name.lower().strip().replace(".gat", "").replace(".rsw", "")
        for prefix in self.town_prefixes():
            if m.startswith(prefix) or m.startswith(prefix[:3]):
                return prefix
        if m.startswith("prt_"):
            return "prontera"
        if m.startswith("pay_"):
            return "payon"
        if m.startswith("moc_"):
            return "morocc"
        if m.startswith("gef_"):
            return "geffen"
        return "prontera"

    def safe_hunting_maps(self, level: int) -> list[str]:
        """Maps where monsters are within safe level range."""
        mobs = self._data.get("mobs", [])
        # Count monsters per map within level range
        map_monsters: dict[str, int] = {}
        for m in mobs:
            lv = int(m.get("Level", 0) or 0)
            if lv - 10 <= level <= lv + 5:  # safe range
                spawns = m.get("Spawns", m.get("Maps", m.get("maps", [])))
                if spawns and isinstance(spawns, list):
                    for s in spawns:
                        mn = s if isinstance(s, str) else (s.get("map", "") if isinstance(s, dict) else "")
                        if mn:
                            map_monsters[mn] = map_monsters.get(mn, 0) + 1
        # Sort by abundance
        sorted_maps = sorted(map_monsters, key=lambda x: -map_monsters[x])
        return sorted_maps[:10]


# ── Module-level singleton ──
_gk: GameKnowledgeService | None = None


def game_knowledge() -> GameKnowledgeService:
    global _gk
    if _gk is None:
        _gk = GameKnowledgeService()
    return _gk
