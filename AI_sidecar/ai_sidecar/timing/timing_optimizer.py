"""
Timing optimizer — knows the server's schedule and farms at optimal times.

A top player knows when to farm. Early morning when fewer players are online.
Right after maintenance when spawns reset. During WoE when everyone is
distracted. This module tracks server time patterns and optimizes activity.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TimeSlot:
    """A time slot with known characteristics."""
    day_of_week: int  # 0=Monday, 6=Sunday
    hour: int  # 0-23
    player_density: str = "medium"  # low, medium, high, very_high
    gm_activity: str = "low"  # none, low, medium, high
    competition_level: str = "medium"  # low, medium, high
    efficiency_multiplier: float = 1.0  # 0.0-2.0
    notes: str = ""


@dataclass
class ServerEvent:
    """A recurring server event."""
    name: str
    day_of_week: int  # 0=Monday, 6=Sunday
    hour: int
    duration_hours: int = 2
    effect: str = "distraction"  # distraction, opportunity, danger
    description: str = ""


@dataclass(slots=True)
class TimingOptimizer:
    """Optimizes farming schedule based on server timing."""
    
    _lock: RLock = field(default_factory=RLock)
    _slots: dict[str, TimeSlot] = field(default_factory=dict)
    _events: list[ServerEvent] = field(default_factory=list)
    _observations: list[dict] = field(default_factory=list)
    _stats: dict[str, int] = field(default_factory=lambda: {"observations": 0, "optimizations": 0})
    
    def __post_init__(self) -> None:
        """Initialize with default server knowledge."""
        # WoE schedule (most servers have WoE on Wed/Sat/Sun evenings)
        self._events = [
            ServerEvent("WoE", 2, 20, 2, "distraction", "WoE Wednesday — GMs and players busy"),
            ServerEvent("WoE", 5, 20, 2, "distraction", "WoE Saturday — GMs and players busy"),
            ServerEvent("WoE", 6, 20, 2, "distraction", "WoE Sunday — GMs and players busy"),
            ServerEvent("Maintenance", 2, 9, 3, "opportunity", "Weekly maintenance — fresh spawns after"),
            ServerEvent("Peak hours", 0, 19, 4, "danger", "Monday evening peak — many players, GMs active"),
            ServerEvent("Peak hours", 1, 19, 4, "danger", "Tuesday evening peak"),
            ServerEvent("Peak hours", 2, 19, 4, "danger", "Wednesday evening peak"),
            ServerEvent("Peak hours", 3, 19, 4, "danger", "Thursday evening peak"),
            ServerEvent("Peak hours", 4, 19, 4, "danger", "Friday evening peak"),
            ServerEvent("Weekend", 5, 10, 14, "danger", "Saturday — high player count, GMs active"),
            ServerEvent("Weekend", 6, 10, 14, "danger", "Sunday — high player count, GMs active"),
            ServerEvent("Early morning", 0, 3, 4, "opportunity", "Early Monday — very few players, GMs asleep"),
            ServerEvent("Early morning", 1, 3, 4, "opportunity", "Early Tuesday — very few players, GMs asleep"),
            ServerEvent("Early morning", 2, 3, 4, "opportunity", "Early Wednesday — very few players, GMs asleep"),
            ServerEvent("Early morning", 3, 3, 4, "opportunity", "Early Thursday — very few players, GMs asleep"),
            ServerEvent("Early morning", 4, 3, 4, "opportunity", "Early Friday — very few players, GMs asleep"),
            ServerEvent("Early morning", 5, 3, 4, "opportunity", "Early Saturday — very few players"),
            ServerEvent("Early morning", 6, 3, 4, "opportunity", "Early Sunday — very few players"),
        ]
    
    def observe(self, event_type: str, detail: str) -> None:
        """Record a timing observation."""
        with self._lock:
            self._observations.append({
                "type": event_type,
                "detail": detail,
                "timestamp": time.time(),
            })
            self._stats["observations"] += 1
            if len(self._observations) > 100:
                self._observations = self._observations[-100:]
    
    def get_current_slot(self) -> TimeSlot:
        """Get the current time slot with characteristics."""
        now = datetime.now(timezone.utc)
        dow = now.weekday()
        hour = now.hour
        
        key = f"{dow}_{hour}"
        with self._lock:
            slot = self._slots.get(key)
            if slot is None:
                slot = TimeSlot(day_of_week=dow, hour=hour)
                self._slots[key] = slot
            
            # Calculate characteristics based on time
            # Early morning (3-7 AM): low density, low GM, low competition
            if 3 <= hour < 7:
                slot.player_density = "low"
                slot.gm_activity = "none" if hour < 6 else "low"
                slot.competition_level = "low"
                slot.efficiency_multiplier = 1.8
                slot.notes = "Prime farming time — very few players or GMs"
            # Morning (7-12): medium
            elif 7 <= hour < 12:
                slot.player_density = "medium"
                slot.gm_activity = "low"
                slot.competition_level = "medium"
                slot.efficiency_multiplier = 1.2
                slot.notes = "Good farming time"
            # Afternoon (12-17): medium-high
            elif 12 <= hour < 17:
                slot.player_density = "medium"
                slot.gm_activity = "medium"
                slot.competition_level = "medium"
                slot.efficiency_multiplier = 1.0
                slot.notes = "Normal hours"
            # Evening (17-22): high density, high GM
            elif 17 <= hour < 22:
                slot.player_density = "high"
                slot.gm_activity = "high"
                slot.competition_level = "high"
                slot.efficiency_multiplier = 0.6
                slot.notes = "Peak hours — high risk, high competition"
            # Night (22-3): low
            else:
                slot.player_density = "low"
                slot.gm_activity = "low"
                slot.competition_level = "low"
                slot.efficiency_multiplier = 1.5
                slot.notes = "Night farming — few players"
            
            # Weekend adjustment
            if dow >= 5:  # Saturday or Sunday
                if 10 <= hour < 22:
                    slot.player_density = "very_high"
                    slot.gm_activity = "high"
                    slot.competition_level = "very_high"
                    slot.efficiency_multiplier = 0.4
                    slot.notes = "Weekend peak — extremely dangerous"
            
            # WoE adjustment
            for event in self._events:
                if event.day_of_week == dow and event.hour <= hour < event.hour + event.duration_hours:
                    if event.effect == "distraction":
                        slot.player_density = "low"
                        slot.gm_activity = "low"
                        slot.competition_level = "low"
                        slot.efficiency_multiplier = 2.0
                        slot.notes = f"WoE time — GMs and players busy, safe to farm"
                    elif event.effect == "opportunity":
                        slot.efficiency_multiplier = max(slot.efficiency_multiplier, 1.5)
                        slot.notes = f"{event.description}"
                    elif event.effect == "danger":
                        slot.efficiency_multiplier = min(slot.efficiency_multiplier, 0.5)
                        slot.notes = f"{event.description}"
            
            return slot
    
    def should_farm_aggressively(self) -> bool:
        """Should the bot farm aggressively right now?"""
        slot = self.get_current_slot()
        return slot.efficiency_multiplier >= 1.5
    
    def should_lay_low(self) -> bool:
        """Should the bot reduce activity right now?"""
        slot = self.get_current_slot()
        return slot.efficiency_multiplier <= 0.5
    
    def get_timing_context(self) -> str:
        """Get formatted timing context for LLM prompts."""
        slot = self.get_current_slot()
        now = datetime.now(timezone.utc)
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        lines = ["── Server Timing ──"]
        lines.append(f"  Current: {day_names[slot.day_of_week]} {slot.hour}:00 UTC")
        lines.append(f"  Player density: {slot.player_density}")
        lines.append(f"  GM activity: {slot.gm_activity}")
        lines.append(f"  Competition: {slot.competition_level}")
        lines.append(f"  Efficiency multiplier: {slot.efficiency_multiplier:.1f}x")
        lines.append(f"  Recommendation: {'FARM AGGRESSIVELY' if self.should_farm_aggressively() else 'NORMAL' if not self.should_lay_low() else 'LAY LOW'}")
        if slot.notes:
            lines.append(f"  Note: {slot.notes}")
        
        # Upcoming events
        upcoming = [e for e in self._events if e.day_of_week == slot.day_of_week and e.hour > slot.hour][:3]
        if upcoming:
            lines.append("  Upcoming today:")
            for e in upcoming:
                lines.append(f"    {e.hour}:00 — {e.name} ({e.effect})")
        
        return "\n".join(lines)
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# Global instance
_timing: TimingOptimizer | None = None
_timing_lock = RLock()


def get_timing_optimizer() -> TimingOptimizer:
    global _timing
    with _timing_lock:
        if _timing is None:
            _timing = TimingOptimizer()
        return _timing
