"""
Exploration Driver — identifies unexplored maps, prioritizes them by reward
potential, navigates to new areas, records safe routes, and reports findings.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class MapDiscovery:
    """A discovered map with metadata."""
    map_name: str
    first_visited: float = 0.0
    last_visited: float = 0.0
    visit_count: int = 0
    is_explored: bool = False
    monster_density: int = 0  # 0-10
    danger_level: int = 0  # 0-10
    has_mvp: bool = False
    has_safe_spot: bool = False
    recommended_level: int = 1
    notes: str = ""


class ExplorationDriver:
    """Drives exploration of new maps and content."""

    # Maps to explore, organized by level range
    EXPLORATION_MAPS: dict[str, dict] = {}

    @classmethod
    def _load_exploration_maps(cls) -> dict[str, dict]:
        """Load exploration targets from the knowledge database (mob data)."""
        maps: dict[str, dict] = {}
        try:
            from ai_sidecar.knowledge_loader import get_mobs, get_mvps
            mobs = get_mobs()
            mvps = get_mvps()
            mvp_ids = {m.get("Id") for m in mvps}

            # Group mobs by map to determine level ranges and danger
            map_mobs: dict[str, list[dict]] = {}
            for mob in mobs:
                map_name = mob.get("Map", "")
                if map_name:
                    if map_name not in map_mobs:
                        map_mobs[map_name] = []
                    map_mobs[map_name].append(mob)

            for map_name, map_mob_list in map_mobs.items():
                if not map_mob_list:
                    continue
                avg_level = sum(m.get("Level", 50) for m in map_mob_list) / len(map_mob_list)
                has_mvp = any(m.get("Id") in mvp_ids for m in map_mob_list)
                danger = min(10, max(1, int(avg_level / 10)))
                maps[map_name] = {
                    "level": int(avg_level),
                    "danger": danger,
                    "has_mvp": has_mvp,
                    "desc": f"{map_name} (Lv{int(avg_level)})",
                }

            logger.info("exploration_maps_loaded_from_db: %d maps", len(maps))
        except Exception as e:
            logger.warning("exploration_maps_db_load_failed: %s (DB is the source of truth)", e)
        return maps

    def __init__(self) -> None:
        self._lock = RLock()
        self._discoveries: dict[str, MapDiscovery] = {}
        self._current_exploration: str = ""
        self._exploration_active: bool = False
        self._enqueue_fn: Callable | None = None
        self.EXPLORATION_MAPS = self._load_exploration_maps()

    # ── Public API ──

    def record_visit(self, map_name: str, monster_count: int = 0, is_safe: bool = True) -> None:
        """Record a visit to a map."""
        with self._lock:
            now = time.time()
            if map_name not in self._discoveries:
                info = self.EXPLORATION_MAPS.get(map_name, {"level": 1, "danger": 1, "has_mvp": False, "desc": ""})
                self._discoveries[map_name] = MapDiscovery(
                    map_name=map_name,
                    first_visited=now,
                    last_visited=now,
                    visit_count=1,
                    is_explored=True,
                    monster_density=min(10, monster_count // 5),
                    danger_level=info["danger"],
                    has_mvp=info["has_mvp"],
                    recommended_level=info["level"],
                    notes=info["desc"],
                )
            else:
                d = self._discoveries[map_name]
                d.last_visited = now
                d.visit_count += 1
                d.monster_density = min(10, monster_count // 5)

    def get_next_exploration(self, current_level: int) -> str | None:
        """Get the next map to explore based on current level."""
        with self._lock:
            unexplored: list[tuple[int, str]] = []
            for map_name, info in self.EXPLORATION_MAPS.items():
                if map_name not in self._discoveries and info["level"] <= current_level + 10:
                    unexplored.append((info["level"], map_name))

            if not unexplored:
                return None

            unexplored.sort(key=lambda x: x[0])
            return unexplored[0][1]

    def start_exploration(self, map_name: str) -> bool:
        """Start exploring a new map."""
        with self._lock:
            if map_name not in self.EXPLORATION_MAPS:
                return False
            self._current_exploration = map_name
            self._exploration_active = True
            logger.info("exploration_started: %s", map_name)
            return True

    def get_exploration_target(self) -> str | None:
        """Get the current exploration target."""
        with self._lock:
            if self._exploration_active and self._current_exploration:
                return self._current_exploration
            return None

    def complete_exploration(self) -> None:
        """Mark current exploration as complete."""
        with self._lock:
            if self._current_exploration:
                logger.info("exploration_completed: %s", self._current_exploration)
            self._exploration_active = False
            self._current_exploration = ""

    def get_exploration_summary(self) -> str:
        with self._lock:
            lines = [f"── Exploration ──"]
            lines.append(f"Maps discovered: {len(self._discoveries)}/{len(self.EXPLORATION_MAPS)}")
            if self._exploration_active:
                lines.append(f"Exploring: {self._current_exploration}")
            unexplored = len(self.EXPLORATION_MAPS) - len(self._discoveries)
            if unexplored > 0:
                lines.append(f"Unexplored: {unexplored} maps remaining")
            return "\n".join(lines)

    def set_enqueue_fn(self, fn: Callable) -> None:
        with self._lock:
            self._enqueue_fn = fn

    def reset(self) -> None:
        with self._lock:
            self._discoveries.clear()
            self._current_exploration = ""
            self._exploration_active = False


# ── Global Singleton ──

_exploration: ExplorationDriver | None = None
_exploration_lock = RLock()


def get_exploration_driver() -> ExplorationDriver:
    global _exploration
    with _exploration_lock:
        if _exploration is None:
            _exploration = ExplorationDriver()
        return _exploration
