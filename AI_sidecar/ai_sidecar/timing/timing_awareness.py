"""
Timing awareness system — understands server rhythm and schedules.

A top player knows the server's heartbeat. They know when GMs patrol,
when the economy peaks, when MVP spawns are due, when to farm vs
when to socialize. This module tracks time-based patterns and makes
timing-aware decisions.

Key capabilities:
- Server time tracking (peak/off-peak hours)
- GM patrol pattern learning
- Economy cycle tracking (weekend peaks, daily resets)
- MVP spawn timing
- Event schedule awareness
- Safe/unsafe time windows
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TimeWindow:
    """A time window with a safety/activity rating."""
    day_of_week: int  # 0=Monday, 6=Sunday
    start_hour: int  # 0-23
    end_hour: int
    safety_rating: float  # 0.0 (dangerous) to 1.0 (safe)
    activity_type: str = "farming"  # farming, social, pvp, trading, risky
    reason: str = ""


@dataclass(slots=True)
class TimingAwareness:
    """Tracks server rhythm and makes timing-aware decisions."""
    
    _lock: RLock = field(default_factory=RLock)
    _windows: list[TimeWindow] = field(default_factory=list)
    _observations: list[dict[str, Any]] = field(default_factory=list)
    _stats: dict[str, int] = field(default_factory=lambda: {"observations": 0, "recommendations": 0})
    
    def __post_init__(self):
        self._init_default_windows()
    
    def _init_default_windows(self) -> None:
        """Initialize default time windows based on common server patterns."""
        defaults = [
            # Weekday off-peak (safe for farming)
            TimeWindow(0, 0, 6, 0.95, "farming", "Off-peak weekday — minimal GM presence"),
            TimeWindow(1, 0, 6, 0.95, "farming", "Off-peak weekday — minimal GM presence"),
            TimeWindow(2, 0, 6, 0.95, "farming", "Off-peak weekday — minimal GM presence"),
            TimeWindow(3, 0, 6, 0.95, "farming", "Off-peak weekday — minimal GM presence"),
            TimeWindow(4, 0, 6, 0.95, "farming", "Off-peak weekday — minimal GM presence"),
            TimeWindow(5, 0, 6, 0.95, "farming", "Off-peak weekday — minimal GM presence"),
            TimeWindow(6, 0, 6, 0.90, "farming", "Off-peak weekend — slightly more players"),
            
            # Weekday morning (moderate safety)
            TimeWindow(0, 6, 9, 0.80, "farming", "Morning — GMs may start patrols"),
            TimeWindow(1, 6, 9, 0.80, "farming", "Morning — GMs may start patrols"),
            TimeWindow(2, 6, 9, 0.80, "farming", "Morning — GMs may start patrols"),
            TimeWindow(3, 6, 9, 0.80, "farming", "Morning — GMs may start patrols"),
            TimeWindow(4, 6, 9, 0.80, "farming", "Morning — GMs may start patrols"),
            TimeWindow(5, 6, 9, 0.75, "farming", "Friday morning — increased activity"),
            TimeWindow(6, 6, 9, 0.70, "farming", "Weekend morning — more players online"),
            
            # Weekday peak (dangerous for botting)
            TimeWindow(0, 9, 17, 0.50, "risky", "Peak hours — GMs active, many players"),
            TimeWindow(1, 9, 17, 0.50, "risky", "Peak hours — GMs active, many players"),
            TimeWindow(2, 9, 17, 0.50, "risky", "Peak hours — GMs active, many players"),
            TimeWindow(3, 9, 17, 0.50, "risky", "Peak hours — GMs active, many players"),
            TimeWindow(4, 9, 17, 0.45, "risky", "Friday peak — highest GM activity"),
            TimeWindow(5, 9, 17, 0.40, "risky", "Weekend peak — maximum players"),
            TimeWindow(6, 9, 17, 0.35, "risky", "Weekend peak — maximum players, GMs"),
            
            # Weekday evening (social/trading time)
            TimeWindow(0, 17, 22, 0.60, "social", "Evening — good for trading, socializing"),
            TimeWindow(1, 17, 22, 0.60, "social", "Evening — good for trading, socializing"),
            TimeWindow(2, 17, 22, 0.60, "social", "Evening — good for trading, socializing"),
            TimeWindow(3, 17, 22, 0.60, "social", "Evening — good for trading, socializing"),
            TimeWindow(4, 17, 22, 0.55, "social", "Friday evening — peak economy activity"),
            TimeWindow(5, 17, 22, 0.45, "social", "Weekend evening — PvP/WoE time"),
            TimeWindow(6, 17, 22, 0.40, "social", "Weekend evening — maximum activity"),
            
            # Night (safe for farming)
            TimeWindow(0, 22, 24, 0.90, "farming", "Night — minimal players, low GM risk"),
            TimeWindow(1, 22, 24, 0.90, "farming", "Night — minimal players, low GM risk"),
            TimeWindow(2, 22, 24, 0.90, "farming", "Night — minimal players, low GM risk"),
            TimeWindow(3, 22, 24, 0.90, "farming", "Night — minimal players, low GM risk"),
            TimeWindow(4, 22, 24, 0.85, "farming", "Thursday night — moderate activity"),
            TimeWindow(5, 22, 24, 0.70, "farming", "Weekend night — still many players"),
            TimeWindow(6, 22, 24, 0.70, "farming", "Weekend night — still many players"),
        ]
        self._windows = defaults
    
    def get_current_window(self) -> TimeWindow | None:
        """Get the time window for the current time."""
        now = datetime.now(timezone.utc)
        day = now.weekday()
        hour = now.hour
        
        for w in self._windows:
            if w.day_of_week == day and w.start_hour <= hour < w.end_hour:
                return w
        return None
    
    def get_safety_rating(self) -> float:
        """Get the current safety rating (0.0-1.0)."""
        window = self.get_current_window()
        return window.safety_rating if window else 0.5
    
    def recommend_activity(self) -> str:
        """Recommend what to do right now based on timing."""
        with self._lock:
            self._stats["recommendations"] += 1
        
        window = self.get_current_window()
        if window is None:
            return "farming"
        
        # Adjust recommendation based on safety
        if window.safety_rating >= 0.8:
            return "farming"
        elif window.safety_rating >= 0.6:
            return "trading"
        elif window.safety_rating >= 0.4:
            return "social"
        else:
            return "stealth"
    
    def record_observation(self, observation_type: str, detail: str, severity: int = 5) -> None:
        """Record a timing observation (e.g., 'GM spotted at 3 PM on Tuesday')."""
        with self._lock:
            self._observations.append({
                "type": observation_type,
                "detail": detail,
                "severity": severity,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "day": datetime.now(timezone.utc).weekday(),
                "hour": datetime.now(timezone.utc).hour,
            })
            self._stats["observations"] += 1
    
    def get_timing_context(self) -> str:
        """Get a formatted timing context string for LLM prompts."""
        window = self.get_current_window()
        now = datetime.now(timezone.utc)
        
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_name = day_names[now.weekday()]
        
        lines = [f"── Timing Context: {day_name} {now.hour}:{now.minute:02d} UTC ──"]
        
        if window:
            lines.append(f"  Safety: {window.safety_rating*100:.0f}% — {window.reason}")
            lines.append(f"  Recommended activity: {self.recommend_activity()}")
        else:
            lines.append("  No time window data available.")
        
        # Recent observations
        recent = [o for o in self._observations[-5:]]
        if recent:
            lines.append("  Recent observations:")
            for o in recent:
                lines.append(f"    {o['type']}: {o['detail']} (day={o['day']} hour={o['hour']})")
        
        return "\n".join(lines)
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# ── Global instance ──

_timing: TimingAwareness | None = None
_timing_lock = RLock()


def get_timing() -> TimingAwareness:
    """Get or create the global timing awareness instance."""
    global _timing
    with _timing_lock:
        if _timing is None:
            _timing = TimingAwareness()
        return _timing
