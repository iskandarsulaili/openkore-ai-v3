"""MapIntelligence — Level-appropriate hunting zone recommendations.

Provides:
- ``danger_rating(map_name, character_level) -> float`` — how deadly a map is.
- ``next_hunting_zone(current_level, job) -> dict`` — recommended map with reasoning.
- ``flee_route(current_map, danger_level) -> str`` — nearest safe map to flee to.
- ``record_death(map_name)`` — track death rate per map for learning.
- ``record_kill(map_name, count=1)`` — track kill rate per map for efficiency.

All public methods are thread-safe via ``RLock``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from threading import RLock
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_DEFAULT_START_ZONES = _DATA_DIR / "start_zones.yaml"
_DEFAULT_MAP_ROTATION = _DATA_DIR / "map_rotation.yaml"


def _parse_level_range(key: str) -> tuple[int, int]:
    """Parse a level-range key like ``'level_1_15'`` → ``(1, 15)``."""
    parts = key.split("_")
    if len(parts) >= 3:
        try:
            return int(parts[1]), int(parts[2])
        except (ValueError, IndexError):
            pass
    return (0, 0)


class MapIntelligence:
    """Level-appropriate hunting-zone recommendations and map safety tracking.

    Combines two data sources:
    - ``start_zones.yaml`` — safe starting maps per level band.
    - ``map_rotation.yaml`` — broader rotation zones with danger ratings and
      per-hour expectations.

    Dynamically learns which maps are dangerous (death tracking) and which
    are efficient (kill tracking) for this bot.
    """

    # ── Lifecycle ──

    def __init__(
        self,
        start_zones_path: str | Path | None = None,
        map_rotation_path: str | Path | None = None,
    ) -> None:
        self._lock = RLock()
        self._start_zones_path = Path(start_zones_path or _DEFAULT_START_ZONES)
        self._map_rotation_path = Path(map_rotation_path or _DEFAULT_MAP_ROTATION)

        self._start_zones: dict[str, list[dict[str, Any]]] = {}
        self._rotation_zones: dict[str, list[dict[str, Any]]] = {}

        # Learning data: map_name -> {deaths, kills}
        self._death_counts: dict[str, int] = {}
        self._kill_counts: dict[str, int] = {}

        self._load_data()

    # ── Data Loading ──

    def _load_data(self) -> None:
        """Load both YAML data files into memory."""
        self._start_zones = self._load_yaml(self._start_zones_path, "start_zones") or {}
        self._rotation_zones = self._load_yaml(self._map_rotation_path, "rotation_zones") or {}

        start_count = sum(len(v) for v in self._start_zones.values())
        rot_count = sum(len(v) for v in self._rotation_zones.values())
        logger.info(
            "map_intel_loaded: start_zones=%d entries, rotation_zones=%d entries",
            start_count,
            rot_count,
        )

    def _load_yaml(self, path: Path, top_key: str) -> dict[str, Any] | None:
        """Safely load a YAML file and return the value at *top_key*, or None."""
        if yaml is None:
            logger.warning("map_intel_no_yaml: PyYAML not installed")
            return None
        if not path.exists():
            logger.warning("map_intel_no_file: path=%s", path)
            return None
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            return data.get(top_key)
        except Exception as exc:
            logger.error("map_intel_load_failed: path=%s error=%s", path, exc)
            return None

    def reload(self) -> None:
        """Reload data from disk (useful after config changes)."""
        with self._lock:
            self._load_data()

    # ── Danger Rating ──

    def danger_rating(self, map_name: str, character_level: int | None = None) -> float:
        """Return how dangerous *map_name* is for a character at *character_level*.

        Returns:
            ``0.0`` — completely safe.
            ``1.0`` — deadly (player will die quickly).

        The rating is the highest of:
        1. The map's stored ``danger_rating`` from the rotation data.
        2. A bonus from death-rate learning (observed deaths inflate danger).
        3. A level-mismatch penalty (maps far above the character's level get
           pushed toward deadly).
        """
        with self._lock:
            base = self._get_stored_danger(map_name)
            death_penalty = self._death_penalty(map_name)
            level_penalty = self._level_mismatch_penalty(map_name, character_level)

            rating = min(1.0, base + death_penalty + level_penalty)
            return rating

    def _get_stored_danger(self, map_name: str) -> float:
        """Look up the stored danger rating from rotation data, or 0.5 if unknown."""
        for zones in self._rotation_zones.values():
            for entry in zones:
                if isinstance(entry, dict) and entry.get("map") == map_name:
                    return float(entry.get("danger_rating", entry.get("danger", 0.5)))
        # Also check start zones
        for zones in self._start_zones.values():
            for entry in zones:
                if isinstance(entry, dict) and entry.get("map") == map_name:
                    return float(entry.get("danger", entry.get("danger_rating", 0.5)))
        return 0.5  # Unknown maps get a neutral default

    def _death_penalty(self, map_name: str) -> float:
        """Compute extra danger from observed deaths (learning)."""
        deaths = self._death_counts.get(map_name, 0)
        if deaths <= 0:
            return 0.0
        # Each death adds 0.05, capped at 0.3
        return min(0.3, deaths * 0.05)

    def _level_mismatch_penalty(self, map_name: str, character_level: int | None) -> float:
        """Compute extra danger when the map is above the character's level."""
        if character_level is None:
            return 0.0
        min_lv, max_lv = self._map_level_range(map_name)
        if max_lv <= 0:
            return 0.0
        # If character is way below the minimum level, add penalty
        if character_level < min_lv:
            gap = min_lv - character_level
            return min(0.4, gap * 0.05)
        # If character is way above max level, small penalty (inefficient)
        if character_level > max_lv + 15:
            return 0.1
        return 0.0

    def _map_level_range(self, map_name: str) -> tuple[int, int]:
        """Return (min_level, max_level) for a map from rotation data."""
        for key, zones in self._rotation_zones.items():
            for entry in zones:
                if isinstance(entry, dict) and entry.get("map") == map_name:
                    lo = entry.get("min_level")
                    hi = entry.get("max_level")
                    if lo is not None and hi is not None:
                        return int(lo), int(hi)
                    # Fall back to parsing the level key
                    return _parse_level_range(key)
        # Check start zones
        for key, zones in self._start_zones.items():
            for entry in zones:
                if isinstance(entry, dict) and entry.get("map") == map_name:
                    return _parse_level_range(key)
        return (0, 0)

    # ── Next Hunting Zone ──

    def next_hunting_zone(
        self,
        current_level: int,
        job: str = "",
    ) -> dict[str, Any]:
        """Recommend the best hunting zone for *current_level* and *job*.

        Returns a dict with keys:
          - ``map``: map name (e.g. ``"prt_fild05"``).
          - ``danger``: adjusted danger rating for this level.
          - ``expected_exp_per_hour``: from rotation data or 0.
          - ``expected_zeny_per_hour``: from rotation data or 0.
          - ``mobs``: list of monster names.
          - ``nearest_town``: nearest town for safety.
          - ``reason``: human-readable justification.
          - ``efficiency_score``: a float combining danger, exp, and kills.
        """
        with self._lock:
            candidates = self._find_candidates(current_level)

            if not candidates:
                return {
                    "map": "unknown",
                    "danger": 0.5,
                    "expected_exp_per_hour": 0,
                    "expected_zeny_per_hour": 0,
                    "mobs": [],
                    "nearest_town": "prontera",
                    "reason": "No suitable hunting zone found for this level",
                    "efficiency_score": 0.0,
                }

            best: dict[str, Any] | None = None
            best_score = -1.0

            for cand in candidates:
                kills = self._kill_counts.get(cand["map"], 0)
                deaths = self._death_counts.get(cand["map"], 0)
                danger = min(1.0, cand.get("danger_rating", cand.get("danger", 0.5)))
                exp = cand.get("expected_exp_per_hour", 0)
                zeny = cand.get("expected_zeny_per_hour", 0)

                # Efficiency score: high exp/zeny, low danger, high kills/death ratio
                death_ratio = kills / max(deaths, 1)
                efficiency = (exp + zeny * 2) * death_ratio * (1.0 - danger)

                if efficiency > best_score:
                    best_score = efficiency
                    best = {
                        "map": cand["map"],
                        "danger": danger,
                        "expected_exp_per_hour": int(exp),
                        "expected_zeny_per_hour": int(zeny),
                        "mobs": cand.get("mobs", []),
                        "nearest_town": cand.get("nearest_town", cand.get("npc", "prontera")),
                        "reason": (
                            f"Level {current_level} — "
                            f"danger={danger:.2f}, "
                            f"exp={int(exp):,}/h, "
                            f"zeny={int(zeny):,}/h"
                        ),
                        "efficiency_score": best_score,
                    }

            return best or self.next_hunting_zone(current_level, job)

    def _find_candidates(self, level: int) -> list[dict[str, Any]]:
        """Return all zone entries whose level band contains *level*."""
        candidates: list[dict[str, Any]] = []

        # Check rotation zones (primary)
        for key, zones in self._rotation_zones.items():
            lo, hi = _parse_level_range(key)
            if lo <= level <= hi:
                for entry in zones:
                    if isinstance(entry, dict):
                        candidates.append(entry)

        # Check start zones (secondary, for low-level)
        for key, zones in self._start_zones.items():
            lo, hi = _parse_level_range(key)
            if lo <= level <= hi:
                for entry in zones:
                    if isinstance(entry, dict) and entry["map"] not in {c["map"] for c in candidates}:
                        candidates.append(entry)

        return candidates

    # ── Flee Route ──

    def flee_route(self, current_map: str, danger_threshold: float = 0.6) -> str:
        """Return a safe map to flee to when *current_map* exceeds *danger_threshold*.

        Picks the safest nearby town map from data. Falls back to ``"prontera"``.
        """
        with self._lock:
            # Collect all known maps and find the safest one that is different
            # from current_map and has a town anchor (npc / nearest_town).
            safe_options: list[tuple[float, str]] = []

            for key, zones in {**self._rotation_zones, **self._start_zones}.items():
                for entry in zones:
                    if isinstance(entry, dict):
                        m = entry["map"]
                        if m == current_map:
                            continue
                        danger = float(entry.get("danger_rating", entry.get("danger", 0.5)))
                        if danger < danger_threshold:
                            safe_options.append((danger, m))

            if not safe_options:
                return "prontera"

            safe_options.sort(key=lambda x: x[0])
            return safe_options[0][1]

    # ── Learning: Death & Kill Tracking ──

    def record_death(self, map_name: str) -> None:
        """Record a death on *map_name*. Increases the map's danger rating."""
        with self._lock:
            self._death_counts[map_name] = self._death_counts.get(map_name, 0) + 1
            logger.info("map_intel_death: map=%s total=%d", map_name, self._death_counts[map_name])

    def record_kill(self, map_name: str, count: int = 1) -> None:
        """Record *count* kills on *map_name*.

        Increases the map's efficiency score for future recommendations.
        """
        with self._lock:
            self._kill_counts[map_name] = self._kill_counts.get(map_name, 0) + count

    def death_rate(self, map_name: str) -> float:
        """Return death rate (deaths / max(deaths+kills, 1)) for *map_name*.

        0.0 = no deaths recorded.  1.0 = 100% death rate (all engagement ended
        in death).
        """
        with self._lock:
            deaths = self._death_counts.get(map_name, 0)
            kills = self._kill_counts.get(map_name, 0)
            total = deaths + kills
            if total == 0:
                return 0.0
            return deaths / total

    def kill_rate(self, map_name: str) -> float:
        """Return kill rate (kills / max(deaths+kills, 1)).

        Complement of death_rate.
        """
        return 1.0 - self.death_rate(map_name)

    def get_learning_stats(self) -> dict[str, dict[str, int | float]]:
        """Return per-map learning stats for diagnostics."""
        with self._lock:
            all_maps = set(self._death_counts.keys()) | set(self._kill_counts.keys())
            stats: dict[str, dict[str, int | float]] = {}
            for m in sorted(all_maps):
                deaths = self._death_counts.get(m, 0)
                kills = self._kill_counts.get(m, 0)
                stats[m] = {
                    "deaths": deaths,
                    "kills": kills,
                    "death_rate": round(self.death_rate(m), 3),
                    "kill_rate": round(self.kill_rate(m), 3),
                }
            return stats

    # ── Diagnostics ──

    def get_stats(self) -> dict[str, Any]:
        """Return diagnostic stats about the map intelligence engine."""
        with self._lock:
            start_count = sum(len(v) for v in self._start_zones.values())
            rot_count = sum(len(v) for v in self._rotation_zones.values())
            return {
                "start_zones_path": str(self._start_zones_path),
                "map_rotation_path": str(self._map_rotation_path),
                "start_zones_bands": list(self._start_zones.keys()),
                "rotation_zones_bands": list(self._rotation_zones.keys()),
                "start_zones_entries": start_count,
                "rotation_zones_entries": rot_count,
                "maps_with_deaths": len(self._death_counts),
                "maps_with_kills": len(self._kill_counts),
                "is_yaml_available": yaml is not None,
            }


def create_map_intel(
    data_path: str | Path | None = None,
) -> MapIntelligence:
    """Factory function — create a :class:`MapIntelligence` with defaults.

    If *data_path* is provided, it is used as the base directory for both
    ``start_zones.yaml`` and ``map_rotation.yaml``.
    """
    if data_path is not None:
        bp = Path(data_path)
        return MapIntelligence(
            start_zones_path=bp / "start_zones.yaml",
            map_rotation_path=bp / "map_rotation.yaml",
        )
    return MapIntelligence()