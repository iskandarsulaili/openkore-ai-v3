"""Map Rotation Planner — prevents overdue stamina penalties by rotating hunting maps.

rAthena servers apply overdue time penalties when farming the same map
for extended periods (typically 1-2 hours), reducing drops/XP by ~50%.

This planner:
  - Tracks time spent on each map per session
  - Maintains a rotation list of 2-3 maps suited to current level/job
  - Recommends switching maps when overdue time penalty kicks in
  - Prioritizes maps by highest expected exp/hour after penalty
  - Produces HeuristicAction "rotate_map {next_map}" commands
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

import yaml

from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)

# ── Data file path ──
_DATA_DIR = Path(os.environ.get(
    "RO_MECHANICS_DATA_DIR",
    str(Path(__file__).resolve().parent.parent.parent.parent / "data"),
))
_DEFAULT_YAML_PATH = _DATA_DIR / "map_rotation.yaml"

# ── Overdue stamina penalty curve (rAthena typical) ──
# After penalty_hours on the same map, exp/drop rate multiplier drops
# linearly from 1.0 to 0.5 over the next hour, then stays at 0.5.
# Configurable per-server.
_DEFAULT_PENALTY_HOURS = 1.5       # Hours before penalty starts
_DEFAULT_MIN_MULTIPLIER = 0.50     # Floor for penalty multiplier
_ROTATION_WINDOW_HOURS = 2.5       # Max time before we force a rotation


@dataclass
class ZoneInfo:
    """Recommended hunting zone loaded from YAML."""
    map: str
    min_level: int
    max_level: int
    monster_density: float       # 0-1
    danger_rating: float         # 0-1
    expected_exp_per_hour: int
    expected_zeny_per_hour: int
    mobs: list[str] = field(default_factory=list)
    nearest_town: str = ""
    penalty_hours: float = 1.5


@dataclass
class MapStaminaRecord:
    """Tracks how long we've been on a map and the penalty status."""
    map_name: str
    entered_at: float           # timestamp when we arrived
    total_seconds: float = 0.0  # cumulative time on this map this session
    penalty_multiplier: float = 1.0  # current exp/drop multiplier
    last_update: float = 0.0


@dataclass
class RotationRecommendation:
    """A rotation recommendation produced by the planner."""
    current_map: str
    recommended_map: str
    reason: str
    current_penalty: float       # 0.5-1.0
    expected_exp_after_penalty: int
    confidence: float            # 0-1
    nearest_town: str = ""       # town to route through
    metadata: dict[str, Any] = field(default_factory=dict)


class MapRotationPlanner:
    """Plans map rotation to avoid overdue stamina penalties.

    Tracks time-per-map per session and recommends rotation when
    penalty would significantly reduce efficiency.

    Thread-safe for concurrent access from heuristic loop.
    """

    def __init__(
        self,
        yaml_path: str | Path | None = None,
    ):
        self._lock = RLock()
        self._yaml_path = Path(yaml_path) if yaml_path else _DEFAULT_YAML_PATH

        # Map name -> ZoneInfo (from YAML data)
        self._zones: dict[str, ZoneInfo] = {}

        # Per-bot: bot_id -> MapStaminaRecord for current map
        self._current_map: dict[str, MapStaminaRecord] = {}

        # Per-bot: bot_id -> list of recently visited maps (rotation history)
        # Helps avoid recommending maps we just left
        self._visited_maps: dict[str, list[str]] = {}

        # Load data
        self._load_data()

        # Sever-side penalty config (can be overridden)
        self.penalty_hours: float = _DEFAULT_PENALTY_HOURS
        self.min_multiplier: float = _DEFAULT_MIN_MULTIPLIER
        self.rotation_window: float = _ROTATION_WINDOW_HOURS

    # ── Public API ──

    def set_penalty_config(
        self,
        penalty_hours: float | None = None,
        min_multiplier: float | None = None,
    ) -> None:
        """Override server-side penalty configuration."""
        with self._lock:
            if penalty_hours is not None:
                self.penalty_hours = max(0.5, min(6.0, penalty_hours))
            if min_multiplier is not None:
                self.min_multiplier = max(0.1, min(1.0, min_multiplier))

    def get_zones_for_level(self, level: int) -> list[ZoneInfo]:
        """Get all zones suitable for the given level."""
        with self._lock:
            return [
                z for z in self._zones.values()
                if z.min_level <= level <= z.max_level
            ]

    def on_map_enter(self, bot_id: str, map_name: str) -> None:
        """Record that a bot entered a new map."""
        with self._lock:
            now = time.time()
            self._current_map[bot_id] = MapStaminaRecord(
                map_name=map_name,
                entered_at=now,
                last_update=now,
            )
            # Track visited maps (most recent first, max 10)
            visited = self._visited_maps.setdefault(bot_id, [])
            if map_name in visited:
                visited.remove(map_name)
            visited.insert(0, map_name)
            if len(visited) > 10:
                self._visited_maps[bot_id] = visited[:10]

    def on_tick(self, bot_id: str, current_map: str) -> None:
        """Update elapsed time on the current map (call every assess cycle).

        Args:
            bot_id: Bot identifier.
            current_map: Current map name from signals.
        """
        with self._lock:
            now = time.time()
            record = self._current_map.get(bot_id)

            # Detect map change
            if record is None or record.map_name != current_map:
                self.on_map_enter(bot_id, current_map)
                record = self._current_map.get(bot_id)

            if record is None:
                return

            # Accumulate elapsed time since last update
            if record.last_update > 0:
                elapsed = now - record.last_update
                record.total_seconds += elapsed
            record.last_update = now

            # Calculate penalty multiplier
            hours_on_map = record.total_seconds / 3600.0
            zone = self._zones.get(current_map)
            zone_penalty_hours = zone.penalty_hours if zone else self.penalty_hours

            if hours_on_map <= zone_penalty_hours:
                record.penalty_multiplier = 1.0
            else:
                # Linear drop from 1.0 to min_multiplier over 1 hour
                excess_hours = hours_on_map - zone_penalty_hours
                drop = excess_hours * (1.0 - self.min_multiplier)
                record.penalty_multiplier = max(
                    self.min_multiplier,
                    1.0 - drop,
                )

    def get_penalty_for_map(self, bot_id: str, map_name: str) -> float:
        """Get current penalty multiplier for a specific map for a bot.

        Returns 1.0 if no record exists (no penalty).
        """
        with self._lock:
            record = self._current_map.get(bot_id)
            if record and record.map_name == map_name:
                return record.penalty_multiplier
            return 1.0

    def get_hours_on_map(self, bot_id: str) -> float:
        """Get total hours spent on the current map this session."""
        with self._lock:
            record = self._current_map.get(bot_id)
            if record:
                return record.total_seconds / 3600.0
            return 0.0

    def should_rotate(self, bot_id: str) -> bool:
        """Check if the bot should rotate maps due to penalty."""
        with self._lock:
            record = self._current_map.get(bot_id)
            if not record:
                return False

            hours_on_map = record.total_seconds / 3600.0
            zone = self._zones.get(record.map_name)
            zone_penalty_hours = zone.penalty_hours if zone else self.penalty_hours

            return hours_on_map >= zone_penalty_hours

    def recommend_rotation(
        self,
        bot_id: str,
        level: int,
        current_map: str = "",
    ) -> RotationRecommendation | None:
        """Recommend the best map to rotate to.

        Args:
            bot_id: Bot identifier.
            level: Character's current base level.
            current_map: Current map (if known).

        Returns:
            RotationRecommendation or None if no suitable alternative.
        """
        with self._lock:
            # Ensure tick called to update penalty
            if current_map:
                # Tick was already called via assess, but let's make sure record exists
                if bot_id not in self._current_map:
                    self.on_map_enter(bot_id, current_map)

            record = self._current_map.get(bot_id)
            if not record:
                return None

            current_map = record.map_name
            hours_on_map = record.total_seconds / 3600.0
            current_penalty = record.penalty_multiplier

            # Get candidates for this level
            candidates = self.get_zones_for_level(level)

            # Filter out current map and recently visited maps
            visited = self._visited_maps.get(bot_id, [])
            candidates = [
                z for z in candidates
                if z.map != current_map and z.map not in visited[:3]
            ]

            if not candidates:
                # Fallback: allow any zone including recently visited
                candidates = self.get_zones_for_level(level)
                candidates = [z for z in candidates if z.map != current_map]
                if not candidates:
                    return None

            # Score each candidate: expected exp/hour after penalty
            # Formula: expected_exp * (1 - danger_rating) * monster_density
            def candidate_score(zone: ZoneInfo) -> float:
                effective_exp = zone.expected_exp_per_hour * (1.0 - zone.danger_rating * 0.3)
                effective_exp *= zone.monster_density
                # Slight bonus for towns we know
                return effective_exp

            candidates.sort(key=candidate_score, reverse=True)
            best = candidates[0]

            # Calculate expected exp after penalty on new map (no penalty yet)
            expected_exp = best.expected_exp_per_hour * best.monster_density

            recommendation = RotationRecommendation(
                current_map=current_map,
                recommended_map=best.map,
                reason=(
                    f"Map rotation: {current_map} penalty={current_penalty:.0%} "
                    f"after {hours_on_map:.1f}h → rotating to {best.map} "
                    f"(exp/h={expected_exp:,}, density={best.monster_density:.0%})"
                ),
                current_penalty=current_penalty,
                expected_exp_after_penalty=int(expected_exp),
                confidence=0.85 if current_penalty < 0.8 else 0.95,
                nearest_town=best.nearest_town,
                metadata={
                    "hours_on_map": round(hours_on_map, 2),
                    "penalty": round(current_penalty, 2),
                    "candidates": [z.map for z in candidates[:3]],
                },
            )
            return recommendation

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        bot_id: str,
    ) -> None:
        """Full assessment: tick, check penalty, recommend rotation if needed.

        Designed to be called from the heuristic assess() loop.

        Signal keys used:
          - map: current map name
          - base_level / level: character level
        """
        _bot_id = bot_id or signals.get("bot_id", "default")
        current_map = signals.get("map", "").lower()
        level = signals.get("base_level", signals.get("level", 1))

        if not current_map:
            return

        # Always tick to update time tracking
        self.on_tick(_bot_id, current_map)

        # Only rotate if penalty is active
        if not self.should_rotate(_bot_id):
            return

        record = self._current_map.get(_bot_id)
        if record and record.penalty_multiplier >= 0.85:
            # Mild penalty — log but don't force rotation yet
            return

        recommendation = self.recommend_rotation(
            _bot_id, level, current_map,
        )
        if recommendation is None:
            return

        # Build the action
        actions.append(HeuristicAction(
            kind="command",
            command=f"rotate_map {recommendation.recommended_map}",
            confidence=recommendation.confidence,
            reason=recommendation.reason,
            domain="planning",
            metadata={
                "current_map": recommendation.current_map,
                "recommended_map": recommendation.recommended_map,
                "current_penalty": recommendation.current_penalty,
                "expected_exp_after_penalty": recommendation.expected_exp_after_penalty,
                "nearest_town": recommendation.nearest_town,
                "hours_on_map": recommendation.metadata.get("hours_on_map", 0),
                "subtype": "map_rotation",
            },
        ))

        # If penalty is severe (>20% loss), also suggest a pathfinding route
        if recommendation.current_penalty < 0.8:
            actions.append(HeuristicAction(
                kind="command",
                command=f"move {recommendation.nearest_town}",
                confidence=0.7,
                reason=f"Navigate to {recommendation.nearest_town} for map rotation",
                domain="planning",
                metadata={
                    "subtype": "rotation_town_nav",
                    "target_town": recommendation.nearest_town,
                },
            ))

        logger.info(
            "Rotation: %s -> %s (penalty=%.0f%%, level=%d)",
            recommendation.current_map,
            recommendation.recommended_map,
            recommendation.current_penalty * 100,
            level,
        )

    def get_summary(self, bot_id: str | None = None) -> dict[str, Any]:
        """Get diagnostic summary of all tracked maps."""
        with self._lock:
            if bot_id:
                records = {bot_id: self._current_map.get(bot_id)}
            else:
                records = dict(self._current_map)

            summary = {}
            for bid, record in records.items():
                if record:
                    summary[bid] = {
                        "map": record.map_name,
                        "hours": round(record.total_seconds / 3600.0, 2),
                        "penalty_multiplier": round(record.penalty_multiplier, 2),
                        "should_rotate": self.should_rotate(bid),
                    }
                else:
                    summary[bid] = {"map": "unknown"}
            return summary

    # ── Internal ──

    def _load_data(self) -> None:
        """Load map rotation data from YAML."""
        path = self._yaml_path
        if not path.exists():
            logger.warning(
                "Map rotation YAML not found at %s, using empty data",
                path,
            )
            return

        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
        except (yaml.YAMLError, OSError) as e:
            logger.error("Failed to load map rotation YAML: %s", e)
            return

        raw_zones = data.get("rotation_zones", {})
        if not raw_zones:
            logger.warning("No rotation_zones found in %s", path)
            return

        for level_key, zone_list in raw_zones.items():
            if not isinstance(zone_list, list):
                continue
            for entry in zone_list:
                zone = ZoneInfo(
                    map=entry.get("map", ""),
                    min_level=entry.get("min_level", 1),
                    max_level=entry.get("max_level", 99),
                    monster_density=float(entry.get("monster_density", 0.5)),
                    danger_rating=float(entry.get("danger_rating", 0.5)),
                    expected_exp_per_hour=int(entry.get("expected_exp_per_hour", 0)),
                    expected_zeny_per_hour=int(entry.get("expected_zeny_per_hour", 0)),
                    mobs=entry.get("mobs", []),
                    nearest_town=entry.get("nearest_town", ""),
                    penalty_hours=float(entry.get("penalty_hours", 1.5)),
                )
                if zone.map:
                    self._zones[zone.map] = zone

        logger.info(
            "Map rotation data loaded: %d zones from %s",
            len(self._zones), path,
        )
