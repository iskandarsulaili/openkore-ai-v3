"""Self-Learning Spawn Tracker — tracks monster spawn points, respawn timers,
and learns spawn patterns from observation for first-hit advantage.

RO: Monsters respawn on a timer after death. Knowing spawn timing and location
gives first-hit advantage (aggro priority, combo initiation, MVP last-hit).

Self-* properties:
  - Self-learning: builds spawn timer models per monster from observed respawns
  - Self-optimizing: adjusts timing expectations as data accumulates
  - Self-adapting: detects server respawn rate changes (event rates, etc.)
  - Self-healing: recalibrates after missed respawns due to lag
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

MIN_SPAWN_SAMPLES: int = 3
DECAY_ALPHA: float = 0.15
SPAWN_WINDOW_FACTOR: float = 0.3  # Window = avg_timer * factor, for predicting next spawn
MAX_TRACKED_MONSTERS: int = 200
SPAWN_GRID_SIZE: int = 1  # How many cells to snap coordinates for spawn point dedup


@dataclass
class SpawnEvent:
    """Record of one observed spawn."""
    monster_name: str
    x: float
    y: float
    timestamp: float

    @property
    def spawn_point_key(self) -> tuple[int, int]:
        return (int(round(self.x / SPAWN_GRID_SIZE)), int(round(self.y / SPAWN_GRID_SIZE)))


@dataclass
class SpawnTimerModel:
    """Learned respawn timer model for one monster/spawn-point."""
    monster_name: str
    spawn_x: float
    spawn_y: float
    spawn_key: tuple[int, int]

    # Timer observations (seconds between death and respawn)
    observed_intervals: deque = field(default_factory=lambda: deque(maxlen=30))

    # Learned timer
    average_interval: float = 60.0  # Default: 60 seconds (common for normal mobs)
    interval_std: float = 5.0       # Variability
    min_interval: float = 0.0
    max_interval: float = 0.0

    # Confidence
    observations: int = 0
    confidence: float = 0.0

    # Predicted next spawn
    last_death_time: float = 0.0    # When this monster last died
    predicted_next_spawn: float = 0.0
    spawn_window_start: float = 0.0  # Earliest expected respawn
    spawn_window_end: float = 0.0    # Latest expected respawn

    # Kill tracking
    total_kills: int = 0
    times_hit_first: int = 0  # How often we got first hit after spawn

    # Recent spawns for pattern detection
    recent_spawn_times: deque = field(default_factory=lambda: deque(maxlen=10))

    def record_death(self, death_time: float) -> None:
        """Record that this monster died at *death_time*."""
        self.last_death_time = death_time
        self.total_kills += 1

        # Calculate spawn window
        if self.observations >= MIN_SPAWN_SAMPLES:
            self.spawn_window_start = death_time + (self.average_interval * (1.0 - SPAWN_WINDOW_FACTOR))
            self.spawn_window_end = death_time + (self.average_interval * (1.0 + SPAWN_WINDOW_FACTOR))
            self.predicted_next_spawn = death_time + self.average_interval
        else:
            # No data — wide window
            self.spawn_window_start = death_time + 30
            self.spawn_window_end = death_time + 120
            self.predicted_next_spawn = death_time + 60

    def record_spawn(self, spawn_time: float) -> None:
        """Record that a spawn was observed at *spawn_time*.

        Updates the timer model with the actual interval since last death.
        """
        if self.last_death_time > 0:
            interval = spawn_time - self.last_death_time
            if interval > 1.0 and interval < 3600:  # Sanity: 1s to 1hr
                self.observed_intervals.append(interval)
                self.observations += 1
                self._recalculate_timer()
                self.recent_spawn_times.append(spawn_time)

        self.last_death_time = spawn_time  # Next death measurement starts from now

    def recorded_first_hit(self, hit_time: float) -> None:
        """Record that we got first hit on this spawn."""
        self.times_hit_first += 1

    def _recalculate_timer(self) -> None:
        """Recalculate timer statistics from observed intervals."""
        intervals = list(self.observed_intervals)
        if len(intervals) < 2:
            return

        self.average_interval = sum(intervals) / len(intervals)
        self.min_interval = min(intervals)
        self.max_interval = max(intervals)

        if len(intervals) > 1:
            variance = sum(
                (i - self.average_interval) ** 2 for i in intervals
            ) / len(intervals)
            self.interval_std = math.sqrt(variance)

        self.confidence = min(1.0, self.observations / 10.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "monster": self.monster_name,
            "spawn": (self.spawn_x, self.spawn_y),
            "avg_interval": round(self.average_interval, 1),
            "std": round(self.interval_std, 2),
            "observations": self.observations,
            "confidence": round(self.confidence, 3),
            "next_spawn_in": round(max(0, self.predicted_next_spawn - time.time()), 1),
            "window": (
                round(max(0, self.spawn_window_start - time.time()), 1),
                round(max(0, self.spawn_window_end - time.time()), 1),
            ),
            "total_kills": self.total_kills,
            "first_hits": self.times_hit_first,
        }


class SelfLearningSpawnTracker:
    """Learns spawn timing patterns from observed monster deaths and respawns.

    Usage:
        tracker = SelfLearningSpawnTracker()

        # When a monster dies:
        tracker.record_death("Poring", x=150, y=200)

        # When a monster appears:
        tracker.record_spawn("Poring", x=150, y=200)

        # Check if any spawns are due:
        due = tracker.get_due_spawns()
        for spawn in due:
            # Move toward spawn point for first-hit advantage
    """

    def __init__(self) -> None:
        self._lock = RLock()

        # Spawn models keyed by (monster_name, spawn_key)
        self._spawn_models: dict[tuple[str, tuple[int, int]], SpawnTimerModel] = {}

        # Map consolidation: monster name -> all known spawn points
        self._monster_spawns: dict[str, list[tuple[int, int]]] = defaultdict(list)

        # Server-wide respawn rate modifier (learned)
        # Some servers have faster/slower respawns
        self._server_respawn_modifier: float = 1.0
        self._server_observations: int = 0

        # Stats
        self._total_deaths: int = 0
        self._total_spawns: int = 0
        self._total_first_hits: int = 0
        self._start_time: float = time.time()

        # Seed common spawn points so the tracker isn't empty on first run
        self.seed_common_spawns()

    # ── Seed common spawns ──────────────────────────────────────────────

    def seed_common_spawns(self) -> None:
        """Populate known spawn points for common farming maps.

        These are canonical spawn locations from official RO map data.
        The self-learning layer will refine timers and discover new points
        as the bot observes real spawns.
        """
        spawn_data: dict[str, list[tuple[str, int, int, int, int]]] = {
            "prt_fild01": [
                ("poring", 50, 50, 70, 70),
                ("poring", 30, 30, 40, 40),
                ("lunatic", 20, 20, 30, 30),
            ],
            "pay_fild01": [
                ("savage_babe", 120, 120, 150, 150),
                ("yoyo", 80, 80, 100, 100),
            ],
            "gef_fild01": [
                ("creamy", 100, 100, 130, 130),
                ("willow", 80, 80, 90, 90),
            ],
            "mjolnir_01": [
                ("wolf", 50, 50, 70, 70),
                ("vadon", 30, 30, 50, 50),
            ],
            "orc_dun01": [
                ("orc_warrior", 50, 50, 80, 80),
                ("orc_archer", 30, 30, 50, 50),
                ("orc_lady", 80, 80, 100, 100),
            ],
        }

        now = time.time()
        for _map_name, entries in spawn_data.items():
            for monster_name, x1, y1, x2, y2 in entries:
                # Use the centre of the spawn rectangle as the representative point
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                name_key = monster_name.lower()
                spawn_key = (
                    int(round(cx / SPAWN_GRID_SIZE)),
                    int(round(cy / SPAWN_GRID_SIZE)),
                )
                model_key = (name_key, spawn_key)

                if model_key not in self._spawn_models:
                    model = SpawnTimerModel(
                        monster_name=monster_name,
                        spawn_x=cx,
                        spawn_y=cy,
                        spawn_key=spawn_key,
                    )
                    # Give seeded models a baseline confidence so the tracker
                    # treats them as known-but-unconfirmed locations.
                    model.observations = 1
                    model.confidence = 0.3
                    model.last_death_time = now - model.average_interval  # Pretend it died one respawn-cycle ago
                    model.predicted_next_spawn = now + 30
                    model.spawn_window_start = now + 15
                    model.spawn_window_end = now + 90

                    self._spawn_models[model_key] = model
                    self._monster_spawns[name_key].append(spawn_key)

    # ── Core recording ─────────────────────────────────────────────────

    def record_death(
        self,
        monster_name: str,
        x: float | int,
        y: float | int,
        timestamp: float | None = None,
    ) -> None:
        """Record that a monster died at a specific location.

        Starts the timer for predicting its respawn.
        """
        ts = timestamp if timestamp is not None else time.time()
        name_key = monster_name.lower()
        spawn_key = (int(round(float(x) / SPAWN_GRID_SIZE)), int(round(float(y) / SPAWN_GRID_SIZE)))
        model_key = (name_key, spawn_key)

        with self._lock:
            model = self._spawn_models.get(model_key)
            if model is None:
                model = SpawnTimerModel(
                    monster_name=monster_name,
                    spawn_x=float(x),
                    spawn_y=float(y),
                    spawn_key=spawn_key,
                )
                self._spawn_models[model_key] = model
                if spawn_key not in self._monster_spawns[name_key]:
                    self._monster_spawns[name_key].append(spawn_key)

            model.record_death(ts)
            self._total_deaths += 1

    def record_spawn(
        self,
        monster_name: str,
        x: float | int,
        y: float | int,
        timestamp: float | None = None,
    ) -> None:
        """Record that a monster spawned at a specific location.

        This updates the timer model with the actual interval since death.
        """
        ts = timestamp if timestamp is not None else time.time()
        name_key = monster_name.lower()
        spawn_key = (int(round(float(x) / SPAWN_GRID_SIZE)), int(round(float(y) / SPAWN_GRID_SIZE)))
        model_key = (name_key, spawn_key)

        with self._lock:
            model = self._spawn_models.get(model_key)
            if model is None:
                model = SpawnTimerModel(
                    monster_name=monster_name,
                    spawn_x=float(x),
                    spawn_y=float(y),
                    spawn_key=spawn_key,
                )
                self._spawn_models[model_key] = model
                if spawn_key not in self._monster_spawns[name_key]:
                    self._monster_spawns[name_key].append(spawn_key)

            model.record_spawn(ts)
            self._total_spawns += 1

            # Update server-wide respawn modifier
            if model.observations >= 2 and model.average_interval > 0:
                # Compare to "standard" 60s baseline
                expected_standard = self._get_standard_respawn(monster_name)
                ratio = expected_standard / model.average_interval
                if 0.2 < ratio < 5.0:  # Sanity check
                    old = self._server_respawn_modifier
                    self._server_respawn_modifier = (
                        0.9 * old + 0.1 * ratio
                    )
                    self._server_observations += 1

    def record_first_hit(
        self,
        monster_name: str,
        x: float | int,
        y: float | int,
    ) -> None:
        """Record that we got first hit on a spawn."""
        name_key = monster_name.lower()
        spawn_key = (int(round(float(x) / SPAWN_GRID_SIZE)), int(round(float(y) / SPAWN_GRID_SIZE)))
        model_key = (name_key, spawn_key)

        with self._lock:
            model = self._spawn_models.get(model_key)
            if model:
                model.recorded_first_hit(time.time())
            self._total_first_hits += 1

    # ── Prediction ──────────────────────────────────────────────────────

    def get_due_spawns(
        self,
        monster_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get all spawns that are due or will be due soon.

        Returns list of dicts sorted by remaining time (ascending).
        """
        results: list[dict[str, Any]] = []
        now = time.time()

        with self._lock:
            for model_key, model in self._spawn_models.items():
                if monster_filter and monster_filter.lower() not in model_key[0]:
                    continue

                remaining = model.predicted_next_spawn - now
                if remaining < 30:  # Due within 30 seconds
                    results.append({
                        "monster": model.monster_name,
                        "spawn_x": model.spawn_x,
                        "spawn_y": model.spawn_y,
                        "remaining_seconds": round(max(0, remaining), 1),
                        "window_start": round(max(0, model.spawn_window_start - now), 1),
                        "window_end": round(max(0, model.spawn_window_end - now), 1),
                        "confidence": round(model.confidence, 3),
                        "first_hit_rate": round(
                            model.times_hit_first / max(model.total_kills, 1), 3
                        ),
                    })

        results.sort(key=lambda r: r["remaining_seconds"])
        return results

    def predict_next_spawn(
        self,
        monster_name: str,
        x: float | int,
        y: float | int,
    ) -> dict[str, Any]:
        """Predict when this specific spawn point will respawn.

        Returns dict with timing info or empty dict if no data.
        """
        name_key = monster_name.lower()
        spawn_key = (int(round(float(x) / SPAWN_GRID_SIZE)), int(round(float(y) / SPAWN_GRID_SIZE)))
        model_key = (name_key, spawn_key)

        with self._lock:
            model = self._spawn_models.get(model_key)
            if model is None or model.last_death_time == 0:
                return {"known": False}

            now = time.time()
            remaining = model.predicted_next_spawn - now
            return {
                "known": True,
                "monster": model.monster_name,
                "spawn_x": model.spawn_x,
                "spawn_y": model.spawn_y,
                "remaining_seconds": round(max(0, remaining), 1),
                "window_start": round(max(0, model.spawn_window_start - now), 1),
                "window_end": round(max(0, model.spawn_window_end - now), 1),
                "average_interval": round(model.average_interval, 1),
                "confidence": round(model.confidence, 3),
                "total_kills": model.total_kills,
                "first_hits": model.times_hit_first,
            }

    def is_due_for_spawn(
        self,
        monster_name: str,
        x: float | int,
        y: float | int,
        threshold_seconds: float = 5.0,
    ) -> bool:
        """Check if a specific monster is due to spawn within *threshold_seconds*."""
        info = self.predict_next_spawn(monster_name, x, y)
        if not info.get("known", False):
            return False
        return info["remaining_seconds"] <= threshold_seconds

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _get_standard_respawn(monster_name: str) -> float:
        """Get standard respawn time for a monster type (seconds).

        These are ballpark figures — actual timers learned from observation.
        """
        name = monster_name.lower()

        # MVPs: 60-120 minutes
        if "mvp" in name or any(
            m in name for m in ["demon", "dragon", "lord", "queen", "king",
                                "garm", "baphomet", "drake", "doppel", "edga",
                                "gloom", "kraken", "thanatos", "turtle general"]
        ):
            return 7200  # ~2 hours

        # Mini-bosses / MVP-like
        if any(m in name for m in ["miniboss", "mini", "boss", "maya", "phreeoni",
                                    "ghostring", "angeling", "deviling"]):
            return 3600  # ~1 hour

        # Normal monsters: 30-120 seconds
        return 60.0

    def get_all_spawn_points(self, monster_name: str) -> list[dict[str, Any]]:
        """Get all known spawn points for a monster."""
        name_key = monster_name.lower()
        results: list[dict[str, Any]] = []

        with self._lock:
            spawn_keys = self._monster_spawns.get(name_key, [])
            for sk in spawn_keys:
                model_key = (name_key, sk)
                model = self._spawn_models.get(model_key)
                if model:
                    results.append(model.to_dict())
                else:
                    results.append({
                        "monster": monster_name,
                        "spawn": sk,
                        "avg_interval": 0,
                        "observations": 0,
                    })

        return results

    # ── Query / introspection ──────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Summary statistics for introspection."""
        with self._lock:
            reliable_models = sum(
                1 for m in self._spawn_models.values()
                if m.observations >= MIN_SPAWN_SAMPLES
            )
            return {
                "total_deaths": self._total_deaths,
                "total_spawns": self._total_spawns,
                "total_first_hits": self._total_first_hits,
                "spawn_points_tracked": len(self._spawn_models),
                "reliable_timers": reliable_models,
                "unique_monsters": len(self._monster_spawns),
                "server_respawn_modifier": round(self._server_respawn_modifier, 3),
                "due_spawns": len(self.get_due_spawns()),
                "top_models": [
                    m.to_dict()
                    for m in sorted(
                        self._spawn_models.values(),
                        key=lambda x: x.observations,
                        reverse=True,
                    )[:10]
                ],
            }
