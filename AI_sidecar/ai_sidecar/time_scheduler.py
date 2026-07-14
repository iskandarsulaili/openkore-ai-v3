"""
Time-Aware Scheduler — knows the server's event calendar, switches strategies
based on time of day, prepares for WoE/maintenance/events, and optimizes
around the clock.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class TimeSlot:
    """A time slot with a recommended strategy."""
    hour_start: int
    hour_end: int
    strategy: str = "normal_farming"
    label: str = ""


@dataclass
class ServerEvent:
    """A scheduled server event."""
    name: str
    event_type: str  # woe, double_xp, double_drop, maintenance, holiday
    day_of_week: int = -1  # 0=Mon, 6=Sun, -1=every day
    hour_start: int = 0
    hour_end: int = 24
    is_active: bool = False
    preparation_time_min: int = 30


class TimeAwareScheduler:
    """Switches strategies based on time of day and server events."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._events: list[ServerEvent] = []
        self._time_slots: list[TimeSlot] = []
        self._current_strategy: str = "normal_farming"
        self._enqueue_fn: Callable | None = None
        self._load_defaults()

    def _load_defaults(self) -> None:
        """Load default time slots and events."""
        self._time_slots = [
            TimeSlot(0, 6, "hardcore_farming", "Late night — low population, good spawns"),
            TimeSlot(6, 9, "efficient_farming", "Morning — moderate population"),
            TimeSlot(9, 12, "normal_farming", "Midday — normal population"),
            TimeSlot(12, 14, "efficient_farming", "Early afternoon — moderate"),
            TimeSlot(14, 18, "normal_farming", "Afternoon — building peak"),
            TimeSlot(18, 20, "mvp_hunting", "Evening — peak hours, good for MVPs"),
            TimeSlot(20, 22, "woe_preparation", "WoE time — prepare or participate"),
            TimeSlot(22, 24, "hardcore_farming", "Night — low population"),
        ]

        self._events = [
            ServerEvent("WoE Wednesday", "woe", 3, 20, 22, preparation_time_min=30),
            ServerEvent("WoE Saturday", "woe", 6, 20, 23, preparation_time_min=30),
            ServerEvent("WoE Sunday", "woe", 0, 20, 23, preparation_time_min=30),
            ServerEvent("Double XP Weekend", "double_xp", 6, 0, 24, preparation_time_min=0),
            ServerEvent("Double XP Weekend", "double_xp", 0, 0, 24, preparation_time_min=0),
            ServerEvent("Weekly Maintenance", "maintenance", 2, 9, 12, preparation_time_min=15),
            ServerEvent("Double Drop Thursday", "double_drop", 4, 14, 18, preparation_time_min=0),
        ]

    # ── Public API ──

    def get_current_strategy(self) -> str:
        """Get the recommended strategy for the current time."""
        with self._lock:
            now = datetime.now()
            current_hour = now.hour
            current_day = now.weekday()

            # Check events first (highest priority)
            for event in self._events:
                if event.day_of_week == -1 or event.day_of_week == current_day:
                    if event.hour_start <= current_hour < event.hour_end:
                        event.is_active = True
                        if event.event_type == "woe":
                            self._current_strategy = "woe_combat"
                        elif event.event_type == "double_xp":
                            self._current_strategy = "maximize_xp_farming"
                        elif event.event_type == "double_drop":
                            self._current_strategy = "maximize_drop_farming"
                        elif event.event_type == "maintenance":
                            self._current_strategy = "prepare_for_maintenance"
                        return self._current_strategy
                    else:
                        event.is_active = False

            # Check preparation windows
            for event in self._events:
                if event.day_of_week == -1 or event.day_of_week == current_day:
                    prep_start = event.hour_start - (event.preparation_time_min / 60.0)
                    if prep_start <= current_hour < event.hour_start:
                        if event.event_type == "woe":
                            self._current_strategy = "prepare_for_woe"
                            return self._current_strategy
                        elif event.event_type == "maintenance":
                            self._current_strategy = "prepare_for_maintenance"
                            return self._current_strategy

            # Fall back to time slot
            for slot in self._time_slots:
                if slot.hour_start <= current_hour < slot.hour_end:
                    self._current_strategy = slot.strategy
                    return self._current_strategy

            self._current_strategy = "normal_farming"
            return self._current_strategy

    def get_active_events(self) -> list[ServerEvent]:
        """Get currently active events."""
        with self._lock:
            now = datetime.now()
            current_hour = now.hour
            current_day = now.weekday()
            active: list[ServerEvent] = []
            for event in self._events:
                if event.day_of_week == -1 or event.day_of_week == current_day:
                    if event.hour_start <= current_hour < event.hour_end:
                        event.is_active = True
                        active.append(event)
            return active

    def get_upcoming_events(self, hours: int = 2) -> list[ServerEvent]:
        """Get events starting within the next N hours."""
        with self._lock:
            now = datetime.now()
            current_hour = now.hour
            current_day = now.weekday()
            upcoming: list[ServerEvent] = []
            for event in self._events:
                if event.day_of_week == -1 or event.day_of_week == current_day:
                    if event.hour_start > current_hour and event.hour_start <= current_hour + hours:
                        upcoming.append(event)
            return upcoming

    def get_scheduler_summary(self) -> str:
        with self._lock:
            now = datetime.now()
            lines = [f"── Time-Aware Scheduler ──"]
            lines.append(f"Time: {now.strftime('%A %H:%M')}")
            lines.append(f"Strategy: {self._current_strategy}")
            active = self.get_active_events()
            if active:
                lines.append(f"Active events: {', '.join(e.name for e in active)}")
            upcoming = self.get_upcoming_events()
            if upcoming:
                lines.append(f"Upcoming: {', '.join(f'{e.name} @ {e.hour_start}:00' for e in upcoming)}")
            return "\n".join(lines)

    def set_enqueue_fn(self, fn: Callable) -> None:
        with self._lock:
            self._enqueue_fn = fn

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._time_slots.clear()
            self._current_strategy = "normal_farming"
            self._load_defaults()


# ── Global Singleton ──

_time_scheduler: TimeAwareScheduler | None = None
_time_scheduler_lock = RLock()


def get_time_scheduler() -> TimeAwareScheduler:
    global _time_scheduler
    with _time_scheduler_lock:
        if _time_scheduler is None:
            _time_scheduler = TimeAwareScheduler()
        return _time_scheduler
