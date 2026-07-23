"""
Hunting Zone Manager — Dynamically discovers optimal hunting zones.

No hardcoded maps. Every recommendation comes from:
- Observed spawn data (what mobs are actually on each map)
- Character level and gear
- Real-time exp/hour tracking
- Danger assessment from death history
- Player competition (how many other bots/players are on the map)
- Server-specific rates (auto-detected from exp gain)

The system learns which maps are good and which are dangerous over time.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from ai_sidecar.combat.damage_formulas import (
    get_monster_element,
    get_monster_size,
    get_monster_race,
    get_monster_def_data,
    calculate_damage,
    estimate_hits_to_kill,
    get_element_multiplier,
)

logger = logging.getLogger(__name__)


@dataclass
class HuntingZone:
    """A hunting zone recommendation with full tactical data."""
    map_name: str
    primary_monster: str
    monster_level: int
    monster_hp: int
    base_exp: int
    job_exp: int
    exp_per_hp: float
    element: str
    race: str
    size: str
    element_efficiency: float
    size_efficiency: float
    race_efficiency: float
    level_penalty: float
    effective_exp: int
    danger_score: float  # 0.0 = safe, 1.0 = deadly
    zeny_per_kill: float  # Expected zeny from drops per kill
    score: float  # Composite score
    reason: str
    spawn_density: float = 1.0  # Relative spawn density (1.0 = normal)
    competition_level: float = 0.0  # 0.0 = empty, 1.0 = crowded
    observed_exp_per_hour: float = 0.0  # Real exp/hour from actual hunting
    death_count: int = 0  # How many times we died on this map
    last_visited: float = 0.0  # Timestamp of last visit


@dataclass
class SpawnObservation:
    """Observed spawn data for a map."""
    map_name: str
    monsters: dict[str, int]  # monster_name -> count observed
    total_spawns: int
    last_observed: float
    avg_respawn_time: float  # seconds between kills
    player_count: int  # other players/bots observed


class HuntingZoneManager:
    """Dynamically discovers and manages hunting zones.

    No hardcoded maps. Learns from:
    - Observed spawn data (what's actually on each map)
    - Real exp/hour from hunting
    - Death history
    - Competition level
    - Server rates (auto-detected)
    """

    def __init__(self) -> None:
        self._lock = RLock()
        # Dynamic data — learned from experience
        self._observed_zones: dict[str, HuntingZone] = {}
        self._spawn_observations: dict[str, SpawnObservation] = {}
        self._death_history: list[dict[str, Any]] = []
        self._exp_history: dict[str, list[float]] = {}  # map_name -> [exp_per_hour_samples]
        self._server_rate_mult: float = 1.0  # Auto-detected from actual exp gain
        self._last_exp_check: float = 0.0
        self._last_exp_value: int = 0
        self._last_job_exp_value: int = 0
        self._known_maps: set[str] = set()  # Maps we've visited
        self._forbidden_maps: set[str] = set()  # Maps that killed us too often

    # ── Dynamic Learning ──────────────────────────────────────────────

    def record_spawn_observation(self, map_name: str, monsters: dict[str, int], player_count: int = 0) -> None:
        """Record observed spawns on a map."""
        with self._lock:
            self._known_maps.add(map_name)
            total = sum(monsters.values())
            now = time.time()
            prev = self._spawn_observations.get(map_name)
            if prev:
                # Update with new data
                for m, c in monsters.items():
                    prev.monsters[m] = prev.monsters.get(m, 0) + c
                prev.total_spawns += total
                prev.last_observed = now
                prev.player_count = max(prev.player_count, player_count)
                # Estimate respawn time from time between observations
                if prev.last_observed > 0:
                    time_diff = now - prev.last_observed
                    if time_diff > 0 and total > 0:
                        prev.avg_respawn_time = (prev.avg_respawn_time + time_diff / max(total, 1)) / 2
            else:
                self._spawn_observations[map_name] = SpawnObservation(
                    map_name=map_name,
                    monsters=monsters,
                    total_spawns=total,
                    last_observed=now,
                    avg_respawn_time=5.0,  # Default: 5s respawn
                    player_count=player_count,
                )

    def record_exp_gain(self, base_exp: int, job_exp: int) -> None:
        """Record exp gain to auto-detect server rates."""
        now = time.time()
        if self._last_exp_check > 0 and self._last_exp_value > 0:
            elapsed = now - self._last_exp_check
            if elapsed > 0:
                base_gain = base_exp - self._last_exp_value
                job_gain = job_exp - self._last_job_exp_value
                if base_gain > 0 and elapsed > 0:
                    # Estimate server rate from gain vs expected
                    # Expected: ~1x rate gives ~1% of level per kill at appropriate level
                    exp_per_sec = base_gain / elapsed
                    # Store for rate detection
                    self._server_rate_mult = max(1.0, exp_per_sec / 10.0)  # Rough estimate
        self._last_exp_check = now
        self._last_exp_value = base_exp
        self._last_job_exp_value = job_exp

    def record_death(self, map_name: str, reason: str, position: tuple[int, int] | None = None) -> None:
        """Record a death for danger analysis."""
        with self._lock:
            self._death_history.append({
                "map": map_name,
                "reason": reason,
                "position": position,
                "time": time.time(),
            })
            # Update danger score for this map
            zone = self._observed_zones.get(map_name)
            if zone:
                zone.death_count += 1
                zone.danger_score = min(1.0, zone.danger_score + 0.1)
            # Track forbidden maps regardless of zone existence
            recent_deaths = sum(1 for d in self._death_history[-10:]
                                if d["map"] == map_name)
            if recent_deaths >= 3:
                self._forbidden_maps.add(map_name)
                logger.warning("hunting_zone_forbidden: %s (3+ recent deaths)", map_name)

    def record_exp_per_hour(self, map_name: str, exp_per_hour: float) -> None:
        """Record observed exp/hour for a map."""
        with self._lock:
            if map_name not in self._exp_history:
                self._exp_history[map_name] = []
            self._exp_history[map_name].append(exp_per_hour)
            # Keep last 10 samples
            if len(self._exp_history[map_name]) > 10:
                self._exp_history[map_name] = self._exp_history[map_name][-10:]
            # Update zone if exists
            zone = self._observed_zones.get(map_name)
            if zone:
                samples = self._exp_history[map_name]
                zone.observed_exp_per_hour = sum(samples) / len(samples) if samples else 0.0

    # ── Zone Recommendation ──────────────────────────────────────────

    def get_best_zone(
        self,
        character_level: int,
        weapon_type: str = "Dagger",
        attacker_element: str = "Neutral",
        raw_damage: float = 50.0,
        is_physical: bool = True,
        current_map: str | None = None,
    ) -> HuntingZone | None:
        """Get the best hunting zone recommendation based on all available data.

        Uses observed data first, falls back to knowledge-based recommendations.
        """
        with self._lock:
            candidates = self._rank_zones(
                character_level, weapon_type, attacker_element,
                raw_damage, is_physical,
            )
            if not candidates:
                return None

            # Prefer current map if it's good (avoid unnecessary travel)
            if current_map:
                for z in candidates:
                    if z.map_name == current_map and z.score > 0:
                        return z

            return candidates[0]

    def _rank_zones(
        self,
        character_level: int,
        weapon_type: str,
        attacker_element: str,
        raw_damage: float,
        is_physical: bool,
    ) -> list[HuntingZone]:
        """Rank all known zones by composite score."""
        scored: list[tuple[float, HuntingZone]] = []

        for map_name, zone in self._observed_zones.items():
            if map_name in self._forbidden_maps:
                continue

            # Calculate score components
            score = 0.0
            reasons = []

            # 1. Level appropriateness (0-30 points)
            level_diff = abs(character_level - zone.monster_level)
            if level_diff <= 10:
                level_score = 30 - level_diff * 2
            elif level_diff <= 20:
                level_score = 10
            else:
                level_score = 0
            score += level_score

            # 2. Element efficiency (0-20 points)
            elem_mult = get_element_multiplier(attacker_element, zone.element)
            elem_score = elem_mult * 20
            score += elem_score
            if elem_mult > 1.0:
                reasons.append(f"element_advantage({elem_mult:.1f}x)")

            # 3. Size efficiency (0-10 points)
            size_mult = 1.0  # Simplified
            size_score = size_mult * 10
            score += size_score

            # 4. Exp efficiency (0-20 points)
            if zone.observed_exp_per_hour > 0:
                exp_score = min(20, zone.observed_exp_per_hour / 10000)
                score += exp_score
                reasons.append(f"observed_exp({zone.observed_exp_per_hour:.0f}/h)")

            # 5. Danger penalty (-50 to 0)
            danger_penalty = -zone.danger_score * 50
            score += danger_penalty
            if zone.danger_score > 0.3:
                reasons.append(f"dangerous({zone.danger_score:.1f})")

            # 6. Competition penalty (0 to -20)
            comp_penalty = -zone.competition_level * 20
            score += comp_penalty
            if zone.competition_level > 0.5:
                reasons.append("crowded")

            # 7. Spawn density bonus (0-10)
            density_bonus = min(10, zone.spawn_density * 5)
            score += density_bonus

            # 8. Recency bonus (0-5)
            if zone.last_visited > 0:
                hours_since = (time.time() - zone.last_visited) / 3600
                recency = max(0, 5 - hours_since)
                score += recency

            zone.score = max(0, score)
            zone.reason = ", ".join(reasons) if reasons else "default"
            scored.append((zone.score, zone))

        # Sort by score descending
        scored.sort(key=lambda x: -x[0])
        return [z for _, z in scored]

    def get_zone_for_map(self, map_name: str) -> HuntingZone | None:
        """Get zone data for a specific map."""
        with self._lock:
            return self._observed_zones.get(map_name)

    def get_all_zones(self) -> list[HuntingZone]:
        """Get all known zones."""
        with self._lock:
            return list(self._observed_zones.values())

    def get_forbidden_maps(self) -> set[str]:
        """Get maps that are too dangerous."""
        with self._lock:
            return set(self._forbidden_maps)

    def get_known_maps(self) -> set[str]:
        """Get all maps we've visited."""
        with self._lock:
            return set(self._known_maps)

    def get_death_analysis(self) -> dict[str, Any]:
        """Get analysis of death patterns."""
        with self._lock:
            if not self._death_history:
                return {"total_deaths": 0, "message": "No deaths recorded"}

            by_map: dict[str, int] = {}
            by_reason: dict[str, int] = {}
            for d in self._death_history:
                by_map[d["map"]] = by_map.get(d["map"], 0) + 1
                by_reason[d["reason"]] = by_reason.get(d["reason"], 0) + 1

            most_dangerous = max(by_map, key=by_map.get) if by_map else "none"
            most_common_cause = max(by_reason, key=by_reason.get) if by_reason else "none"

            return {
                "total_deaths": len(self._death_history),
                "deaths_by_map": by_map,
                "deaths_by_reason": by_reason,
                "most_dangerous_map": most_dangerous,
                "most_common_cause": most_common_cause,
                "forbidden_maps": list(self._forbidden_maps),
            }

    def get_server_rate_mult(self) -> float:
        """Get auto-detected server rate multiplier."""
        return self._server_rate_mult

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for persistence."""
        with self._lock:
            return {
                "observed_zones": {
                    k: {
                        "map_name": v.map_name,
                        "primary_monster": v.primary_monster,
                        "monster_level": v.monster_level,
                        "monster_hp": v.monster_hp,
                        "base_exp": v.base_exp,
                        "job_exp": v.job_exp,
                        "element": v.element,
                        "race": v.race,
                        "size": v.size,
                        "danger_score": v.danger_score,
                        "spawn_density": v.spawn_density,
                        "competition_level": v.competition_level,
                        "observed_exp_per_hour": v.observed_exp_per_hour,
                        "death_count": v.death_count,
                        "last_visited": v.last_visited,
                    }
                    for k, v in self._observed_zones.items()
                },
                "forbidden_maps": list(self._forbidden_maps),
                "known_maps": list(self._known_maps),
                "server_rate_mult": self._server_rate_mult,
                "death_history": self._death_history[-50:],  # Last 50 deaths
            }
