"""
Server Population Tracker — tracks player count per map over time,
identifies peak/off-peak hours, knows server event schedules,
and adjusts farming strategy accordingly.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MapPopulation:
    """Population data for a single map."""
    map_name: str
    player_count: int = 0
    bot_count: int = 0
    total_count: int = 0
    timestamp: float = 0.0
    is_crowded: bool = False
    crowding_threshold: int = 10


@dataclass
class PopulationTrend:
    """Population trend for a map over time."""
    map_name: str
    avg_players_peak: float = 0.0
    avg_players_offpeak: float = 0.0
    peak_hours: list[int] = field(default_factory=list)
    offpeak_hours: list[int] = field(default_factory=list)
    best_farming_time: str = ""
    worst_farming_time: str = ""
    sample_count: int = 0


@dataclass
class ServerEvent:
    """A known server event."""
    name: str
    event_type: str  # double_xp, double_drop, woE, maintenance, holiday
    start_hour: int = 0
    end_hour: int = 0
    day_of_week: int = -1  # 0=Mon, 6=Sun, -1=every day
    is_active: bool = False
    description: str = ""


class ServerPopulationTracker:
    """Tracks server population and events."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._map_populations: dict[str, list[MapPopulation]] = defaultdict(list)
        self._trends: dict[str, PopulationTrend] = {}
        self._events: list[ServerEvent] = []
        self._max_history: int = 1000
        self._crowding_threshold: int = 10
        self._load_default_events()

    def _load_default_events(self) -> None:
        """Load known server events."""
        self._events = [
            ServerEvent("Double XP Weekend", "double_xp", start_hour=0, end_hour=24, day_of_week=6, description="Double experience all weekend"),
            ServerEvent("Double XP Weekend", "double_xp", start_hour=0, end_hour=24, day_of_week=0, description="Double experience all weekend"),
            ServerEvent("WoE Wednesday", "woe", start_hour=20, end_hour=22, day_of_week=3, description="War of Emperium"),
            ServerEvent("WoE Saturday", "woe", start_hour=20, end_hour=23, day_of_week=6, description="War of Emperium"),
            ServerEvent("WoE Sunday", "woe", start_hour=20, end_hour=23, day_of_week=0, description="War of Emperium"),
            ServerEvent("Weekly Maintenance", "maintenance", start_hour=9, end_hour=12, day_of_week=2, description="Server maintenance Tuesday morning"),
            ServerEvent("Double Drop Rate", "double_drop", start_hour=14, end_hour=18, day_of_week=4, description="Double drop rate Thursday afternoon"),
        ]

    # ── Public API ──

    def record_population(self, map_name: str, player_count: int, bot_count: int = 0) -> None:
        """Record a population observation."""
        with self._lock:
            now = time.time()
            pop = MapPopulation(
                map_name=map_name,
                player_count=player_count,
                bot_count=bot_count,
                total_count=player_count + bot_count,
                timestamp=now,
                is_crowded=(player_count + bot_count) >= self._crowding_threshold,
                crowding_threshold=self._crowding_threshold,
            )
            self._map_populations[map_name].append(pop)
            if len(self._map_populations[map_name]) > self._max_history:
                self._map_populations[map_name] = self._map_populations[map_name][-self._max_history:]
            self._update_trend(map_name)

    def _update_trend(self, map_name: str) -> None:
        """Update population trend for a map."""
        history = self._map_populations.get(map_name, [])
        if len(history) < 10:
            return

        now = datetime.now()
        current_hour = now.hour

        # Split into peak and off-peak
        peak_hours = list(range(18, 24)) + list(range(0, 2))  # 6 PM - 2 AM
        offpeak_hours = list(range(6, 18))  # 6 AM - 6 PM

        peak_samples = [p for p in history if datetime.fromtimestamp(p.timestamp).hour in peak_hours]
        offpeak_samples = [p for p in history if datetime.fromtimestamp(p.timestamp).hour in offpeak_hours]

        avg_peak = sum(p.total_count for p in peak_samples) / len(peak_samples) if peak_samples else 0
        avg_offpeak = sum(p.total_count for p in offpeak_samples) / len(offpeak_samples) if offpeak_samples else 0

        self._trends[map_name] = PopulationTrend(
            map_name=map_name,
            avg_players_peak=avg_peak,
            avg_players_offpeak=avg_offpeak,
            peak_hours=peak_hours,
            offpeak_hours=offpeak_hours,
            best_farming_time=f"Off-peak ({offpeak_hours[0]}:00-{offpeak_hours[-1]}:00)" if avg_offpeak < avg_peak else f"Peak ({peak_hours[0]}:00-{peak_hours[-1]}:00)",
            worst_farming_time=f"Peak ({peak_hours[0]}:00-{peak_hours[-1]}:00)" if avg_peak > avg_offpeak else f"Off-peak ({offpeak_hours[0]}:00-{offpeak_hours[-1]}:00)",
            sample_count=len(history),
        )

    def is_map_crowded(self, map_name: str) -> bool:
        """Check if a map is currently crowded."""
        with self._lock:
            history = self._map_populations.get(map_name, [])
            if not history:
                return False
            latest = history[-1]
            return latest.is_crowded

    def get_best_farming_time(self, map_name: str) -> str:
        """Get the best time to farm a specific map."""
        with self._lock:
            trend = self._trends.get(map_name)
            if not trend:
                return "Unknown (insufficient data)"
            return trend.best_farming_time

    def get_current_population(self, map_name: str) -> MapPopulation | None:
        """Get the latest population data for a map."""
        with self._lock:
            history = self._map_populations.get(map_name, [])
            if not history:
                return None
            return history[-1]

    def get_least_crowded_maps(self, limit: int = 5) -> list[str]:
        """Get the least crowded maps right now."""
        with self._lock:
            candidates: list[tuple[str, int]] = []
            for map_name, history in self._map_populations.items():
                if history:
                    latest = history[-1]
                    if not latest.is_crowded:
                        candidates.append((map_name, latest.total_count))
            candidates.sort(key=lambda x: x[1])
            return [c[0] for c in candidates[:limit]]

    def get_most_crowded_maps(self, limit: int = 5) -> list[str]:
        """Get the most crowded maps right now."""
        with self._lock:
            candidates: list[tuple[str, int]] = []
            for map_name, history in self._map_populations.items():
                if history:
                    latest = history[-1]
                    candidates.append((map_name, latest.total_count))
            candidates.sort(key=lambda x: -x[1])
            return [c[0] for c in candidates[:limit]]

    def get_active_events(self) -> list[ServerEvent]:
        """Get currently active server events."""
        with self._lock:
            now = datetime.now()
            current_hour = now.hour
            current_day = now.weekday()
            active: list[ServerEvent] = []
            for event in self._events:
                if event.day_of_week == -1 or event.day_of_week == current_day:
                    if event.start_hour <= current_hour < event.end_hour:
                        event.is_active = True
                        active.append(event)
            return active

    def is_double_xp(self) -> bool:
        """Check if double XP is active."""
        return any(e.event_type == "double_xp" and e.is_active for e in self.get_active_events())

    def is_woe_time(self) -> bool:
        """Check if WoE is active."""
        return any(e.event_type == "woe" and e.is_active for e in self.get_active_events())

    def is_maintenance_time(self) -> bool:
        """Check if maintenance is active."""
        return any(e.event_type == "maintenance" and e.is_active for e in self.get_active_events())

    def get_recommended_strategy(self) -> str:
        """Get the recommended farming strategy based on current conditions."""
        with self._lock:
            now = datetime.now()
            current_hour = now.hour
            current_day = now.weekday()

            if self.is_maintenance_time():
                return "log_out"
            if self.is_woe_time():
                return "prepare_for_woe"
            if self.is_double_xp():
                return "maximize_xp_farming"

            # Time of day
            if current_hour < 6:
                return "hardcore_farming"  # Few players, good spawns
            elif current_hour < 12:
                return "efficient_farming"  # Moderate players
            elif current_hour < 18:
                return "normal_farming"  # Peak building
            else:
                return "mvp_hunting"  # Peak hours, crowded maps

    def get_server_summary(self) -> str:
        with self._lock:
            lines = [f"── Server Population Summary ──"]
            now = datetime.now()
            lines.append(f"Time: {now.strftime('%A %H:%M')}")
            lines.append(f"Maps tracked: {len(self._map_populations)}")

            active_events = self.get_active_events()
            if active_events:
                lines.append(f"Active events: {', '.join(e.name for e in active_events)}")
            else:
                lines.append("No active events")

            lines.append(f"Strategy: {self.get_recommended_strategy()}")
            least = self.get_least_crowded_maps(3)
            if least:
                lines.append(f"Least crowded: {', '.join(least)}")
            most = self.get_most_crowded_maps(3)
            if most:
                lines.append(f"Most crowded: {', '.join(most)}")
            return "\n".join(lines)

    def set_crowding_threshold(self, threshold: int) -> None:
        with self._lock:
            self._crowding_threshold = threshold

    def add_event(self, event: ServerEvent) -> None:
        with self._lock:
            self._events.append(event)

    def reset(self) -> None:
        with self._lock:
            self._map_populations.clear()
            self._trends.clear()


# ── Global Singleton ──

_server_tracker: ServerPopulationTracker | None = None
_server_tracker_lock = RLock()


def get_server_tracker() -> ServerPopulationTracker:
    global _server_tracker
    with _server_tracker_lock:
        if _server_tracker is None:
            _server_tracker = ServerPopulationTracker()
        return _server_tracker
