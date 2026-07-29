"""Opponent modeling — track monster spawn patterns and predict behavior.

Monster behavior prediction using a simple statistical model:
  - Spawn point tracking (learn where monsters spawn).
  - Aggression prediction (which monsters aggro first).
  - Kill speed estimation (how fast we kill certain monster types).
  - Danger assessment (which monsters/combos are dangerous).

Integrates with ro_mechanics.py monster DB for base stats.
"""

from __future__ import annotations

import logging
import time
import math
from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from ai_sidecar.autonomy.ro_mechanics import (
    get_monster_stats, is_mvp,
    ELEMENT_TABLE,
)

logger = logging.getLogger(__name__)


# ── Data Models ──

@dataclass
class SpawnRecord:
    """Record of a monster spawn observation."""
    monster_name: str
    x: int
    y: int
    timestamp: float
    respawn_time_s: float = 0.0  # Estimated respawn time


@dataclass
class KillRecord:
    """Record of killing a monster."""
    monster_name: str
    time_to_kill_s: float
    damage_taken: int
    skill_used: str
    timestamp: float
    player_hp_before: int = 0
    player_hp_after: int = 0


@dataclass
class MonsterProfile:
    """Learned profile for a monster type."""
    name: str
    encounter_count: int = 0
    avg_kill_time_s: float = 10.0
    avg_damage_taken: float = 50.0
    danger_score: float = 1.0  # 1.0 = baseline, > 1.0 more dangerous
    death_count: int = 0       # How many times we died to this monster
    last_encounter: float = 0.0
    spawn_points: list[tuple[int, int, int]] = field(default_factory=list)
    # element, size, race from DB (cached)
    element: str = "neutral"
    size: str = "medium"
    race: str = "formless"
    is_boss: bool = False

    @property
    def is_dangerous(self) -> bool:
        """Monster is dangerous if danger_score exceeds threshold."""
        return self.danger_score > 1.5 or self.death_count > 0


@dataclass
class BehaviorPrediction:
    """Prediction about monster behavior."""
    monster_name: str
    predicted_action: str  # "attack", "cast", "wander", "patrol"
    confidence: float      # 0.0 to 1.0
    threat_level: float    # 0.0 (harmless) to 1.0 (critical)
    estimated_damage_per_hit: int = 0
    recommended_action: str = ""  # "kill_first", "interrupt", "avoid", "tank"


# ── Opponent Model Engine ──

class OpponentModel:
    """Tracks and predicts monster behavior from combat observations.

    Maintains per-monster profiles with spawn locations, kill speed,
    damage patterns, and danger assessment.

    Thread-safe for concurrent access from combat loop and tactics modules.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        # monster_name -> MonsterProfile
        self._profiles: dict[str, MonsterProfile] = {}
        # Recent spawn observations for heatmap
        self._spawn_history: list[SpawnRecord] = []
        # Recent kill records for performance tracking
        self._kill_history: list[KillRecord] = []
        # Map-level danger cache
        self._map_danger: dict[str, float] = {}
        # Current map
        self._current_map: str = ""

    # ── Public API ──

    def record_spawn(self, monster_name: str, x: int, y: int) -> None:
        """Record a monster spawn observation."""
        with self._lock:
            now = time.time()
            record = SpawnRecord(
                monster_name=monster_name,
                x=x, y=y,
                timestamp=now,
            )
            self._spawn_history.append(record)

            # Limit spawn history
            if len(self._spawn_history) > 1000:
                self._spawn_history = self._spawn_history[-500:]

            # Update profile
            profile = self._get_or_create_profile(monster_name)
            profile.last_encounter = now
            profile.encounter_count += 1

            # Track spawn point clusters
            self._update_spawn_points(profile, x, y)

    def record_kill(self, monster_name: str, time_to_kill_s: float,
                    damage_taken: int, skill_used: str = "",
                    hp_before: int = 0, hp_after: int = 0) -> None:
        """Record killing a monster."""
        with self._lock:
            now = time.time()
            record = KillRecord(
                monster_name=monster_name,
                time_to_kill_s=time_to_kill_s,
                damage_taken=damage_taken,
                skill_used=skill_used,
                timestamp=now,
                player_hp_before=hp_before,
                player_hp_after=hp_after,
            )
            self._kill_history.append(record)

            # Limit kill history
            if len(self._kill_history) > 500:
                self._kill_history = self._kill_history[-250:]

            # Update moving averages
            profile = self._get_or_create_profile(monster_name)
            alpha = 0.3  # EMA smoothing factor

            if profile.encounter_count == 0:
                profile.avg_kill_time_s = time_to_kill_s
                profile.avg_damage_taken = float(damage_taken)
            else:
                profile.avg_kill_time_s = (
                    alpha * time_to_kill_s + (1 - alpha) * profile.avg_kill_time_s
                )
                profile.avg_damage_taken = (
                    alpha * damage_taken + (1 - alpha) * profile.avg_damage_taken
                )

            profile.encounter_count += 1
            profile.last_encounter = now

            # Recalculate danger score
            self._recalc_danger(profile)

    def record_death(self, monster_name: str) -> None:
        """Record a death caused by a monster."""
        with self._lock:
            profile = self._get_or_create_profile(monster_name)
            profile.death_count += 1
            profile.danger_score += 2.0  # Big penalty for killing us
            logger.warning("opponent_model: death recorded to %s (total=%d)",
                           monster_name, profile.death_count)

    def record_aggro(self, monster_name: str) -> None:
        """Record that a monster aggroed on us."""
        with self._lock:
            profile = self._get_or_create_profile(monster_name)
            # Frequent aggro = higher threat
            if profile.encounter_count > 10:
                profile.danger_score = min(5.0, profile.danger_score + 0.1)

    def get_profile(self, monster_name: str) -> MonsterProfile | None:
        """Get the learned profile for a monster."""
        with self._lock:
            return self._profiles.get(monster_name.lower())

    def predict_behavior(self, monster_name: str) -> BehaviorPrediction:
        """Predict what a monster will do based on its profile."""
        with self._lock:
            profile = self._get_or_create_profile(monster_name)
            stats = get_monster_stats(monster_name)

            atk = 0
            if stats:
                atk = int(stats.get("attack", stats.get("atk1", 50)))

            # Simple prediction model
            danger = profile.danger_score
            if danger > 3.0:
                action = "attack"
                confidence = 0.85
                threat = 0.9
                recommended = "avoid" if profile.death_count > 0 else "tank"
            elif danger > 1.5:
                action = "attack"
                confidence = 0.7
                threat = 0.6
                recommended = "kill_first"
            else:
                action = "wander"
                confidence = 0.5
                threat = 0.3
                recommended = ""

            return BehaviorPrediction(
                monster_name=monster_name,
                predicted_action=action,
                confidence=confidence,
                threat_level=threat,
                estimated_damage_per_hit=atk,
                recommended_action=recommended,
            )

    def get_dangerous_monsters(self, threshold: float = 1.5) -> list[MonsterProfile]:
        """Get all monsters above a danger threshold."""
        with self._lock:
            return [
                p for p in self._profiles.values()
                if p.danger_score > threshold or p.death_count > 0
            ]

    def get_kill_speed(self, monster_name: str) -> float:
        """Get estimated kill speed in seconds for a monster type."""
        with self._lock:
            profile = self._profiles.get(monster_name.lower())
            if profile and profile.encounter_count > 0:
                return profile.avg_kill_time_s
            # Estimate from stats
            return 15.0  # Default estimate

    def get_spawn_heatmap(self, monster_name: str | None = None,
                          radius: int = 5) -> dict[tuple[int, int], int]:
        """Get spawn point heatmap (clustered by radius).

        Args:
            monster_name: Optional filter to specific monster.
            radius: Cell radius for clustering spawn points.

        Returns:
            Dict of (x, y) -> spawn count.
        """
        with self._lock:
            points: list[tuple[int, int]] = []
            for record in self._spawn_history:
                if monster_name is None or record.monster_name.lower() == monster_name.lower():
                    points.append((record.x, record.y))

            if not points:
                return {}

            # Cluster by radius
            clusters: dict[tuple[int, int], int] = {}
            for x, y in points:
                found = False
                for (cx, cy) in list(clusters.keys()):
                    if abs(x - cx) <= radius and abs(y - cy) <= radius:
                        clusters[(cx, cy)] += 1
                        found = True
                        break
                if not found:
                    clusters[(x, y)] = 1

            return clusters

    def get_map_danger(self, map_name: str) -> float:
        """Get overall danger score for a map based on all profiles.

        Returns a score: < 1.0 = safe, 1.0-2.0 = moderate, > 2.0 = dangerous.
        """
        with self._lock:
            if map_name in self._map_danger:
                return self._map_danger[map_name]

            # Calculate from monster profiles on this map
            total_danger = 0.0
            count = 0
            for profile in self._profiles.values():
                if profile.encounter_count > 5:
                    total_danger += profile.danger_score
                    count += 1

            if count == 0:
                return 0.5  # Unknown map = slightly dangerous by default

            avg = total_danger / count
            self._map_danger[map_name] = avg
            return avg

    def set_current_map(self, map_name: str) -> None:
        """Update the current map context."""
        with self._lock:
            if map_name != self._current_map:
                self._current_map = map_name
                # Clear map danger cache on map change (new context)
                self._map_danger.clear()

    def get_efficiency_score(self, monster_name: str) -> float:
        """Get efficiency score (exp/damage ratio) for farming decisions.

        Higher is better. Compares kill speed and damage taken to estimate
        whether this monster is worth farming.
        """
        with self._lock:
            profile = self._profiles.get(monster_name.lower())
            if not profile or profile.encounter_count < 3:
                return 1.0  # Unknown = neutral

            if profile.avg_kill_time_s <= 0 or profile.danger_score <= 0:
                return 1.0

            # Efficiency = 1 / (kill_time * danger)
            efficiency = 1.0 / (profile.avg_kill_time_s * profile.danger_score)
            return min(10.0, max(0.1, efficiency * 10.0))  # Normalize to 0.1-10.0

    def get_kill_count(self, monster_name: str) -> int:
        """Get total kill count for a monster type."""
        with self._lock:
            profile = self._profiles.get(monster_name.lower())
            return profile.encounter_count if profile else 0

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all tracked monster profiles."""
        with self._lock:
            summary = {
                "total_monsters_tracked": len(self._profiles),
                "kill_history_count": len(self._kill_history),
                "spawn_history_count": len(self._spawn_history),
                "dangerous_monsters": len(self.get_dangerous_monsters()),
                "current_map": self._current_map,
                "map_danger": self._map_danger.get(self._current_map, 0.5),
            }

            # Top 5 most dangerous
            dangerous = sorted(
                self._profiles.values(),
                key=lambda p: p.danger_score,
                reverse=True,
            )[:5]
            summary["most_dangerous"] = [
                {
                    "name": p.name,
                    "danger_score": round(p.danger_score, 2),
                    "deaths": p.death_count,
                    "encounters": p.encounter_count,
                }
                for p in dangerous
            ]

            # Top 5 most killed
            most_killed = sorted(
                self._profiles.values(),
                key=lambda p: p.encounter_count,
                reverse=True,
            )[:5]
            summary["most_killed"] = [
                {
                    "name": p.name,
                    "kills": p.encounter_count,
                    "avg_kill_time_s": round(p.avg_kill_time_s, 1),
                    "avg_damage": round(p.avg_damage_taken, 0),
                }
                for p in most_killed
            ]

            return summary

    # ── Internal ──

    def _get_or_create_profile(self, monster_name: str) -> MonsterProfile:
        """Get or create a profile for a monster."""
        key = monster_name.lower()
        if key not in self._profiles:
            stats = get_monster_stats(monster_name)
            profile = MonsterProfile(
                name=monster_name,
                element=str(stats.get("element", "neutral")).lower() if stats else "neutral",
                size=str(stats.get("size", "medium")).lower() if stats else "medium",
                race=str(stats.get("race", "formless")).lower() if stats else "formless",
                is_boss=is_mvp(monster_name) if stats else False,
            )
            self._profiles[key] = profile

        return self._profiles[key]

    def _update_spawn_points(self, profile: MonsterProfile, x: int, y: int) -> None:
        """Update spawn point clusters for a profile."""
        # Cluster: check if (x,y) is within 5 cells of an existing point
        for i, (sx, sy, count) in enumerate(profile.spawn_points):
            if abs(x - sx) <= 5 and abs(y - sy) <= 5:
                # Weighted average
                new_x = (sx * count + x) // (count + 1)
                new_y = (sy * count + y) // (count + 1)
                profile.spawn_points[i] = (new_x, new_y, count + 1)
                return

        # New spawn point
        profile.spawn_points.append((x, y, 1))
        # Limit to 10 points per monster
        if len(profile.spawn_points) > 10:
            profile.spawn_points.sort(key=lambda p: -p[2])
            profile.spawn_points = profile.spawn_points[:10]

    def _recalc_danger(self, profile: MonsterProfile) -> None:
        """Recalculate danger score from profile data."""
        if profile.encounter_count < 3:
            return  # Not enough data

        danger = 1.0

        # Factor 1: Deaths (weighted heavily)
        danger += profile.death_count * 2.0

        # Factor 2: Damage taken per encounter (normalized)
        if profile.avg_damage_taken > 0:
            dmg_factor = profile.avg_damage_taken / 100.0  # 100 damage = 1.0
            danger += dmg_factor * 0.5

        # Factor 3: Kill time (longer = more dangerous = harder)
        if profile.avg_kill_time_s > 0:
            time_factor = profile.avg_kill_time_s / 10.0  # 10s = 1.0
            danger += time_factor * 0.3

        # Factor 4: Monster stats from DB
        stats = get_monster_stats(profile.name)
        if stats:
            atk = int(stats.get("attack", stats.get("atk1", 50)))
            danger += (atk / 100.0) * 0.5

        profile.danger_score = max(0.5, min(10.0, danger))


# ── Global Singleton ──

_model: OpponentModel | None = None
_model_lock = RLock()


def get_opponent_model() -> OpponentModel:
    global _model
    with _model_lock:
        if _model is None:
            _model = OpponentModel()
        return _model
OpponentModeling = OpponentModel
