"""Event Calendar — tracks in-game events, holidays, double-exp, MVP spawns.

Data-driven from configuration. Allows the bot to plan around
server events rather than treating every day the same.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

# Default events that apply to most pre-renewal servers
DEFAULT_EVENTS: list[dict[str, Any]] = [
    # Weekly recurring
    {"name": "Double EXP Weekend", "day_of_week": "saturday,sunday", "multiplier": 2.0, "type": "exp", "recurring": True},
    {"name": "Double Drop Weekend", "day_of_week": "saturday,sunday", "multiplier": 2.0, "type": "drop", "recurring": True},
    # Daily
    {"name": "Server Reset", "hour": 0, "type": "reset", "recurring": True},
    # MVP spawn windows (pre-renewal defaults)
    {"name": "MVP Spawn Window", "hour_range": [18, 23], "type": "mvp", "recurring": True,
     "description": "Prime MVP hunting hours - higher spawn competition but also more parties"},
]


class EventCalendar:
    """Tracks and predicts in-game events for farming optimization.
    
    Features:
    - Knows what events are active right now
    - Predicts upcoming events based on schedules
    - Provides optimization suggestions (e.g., 'farm during double exp')
    """
    
    def __init__(self, data_path: str | None = None) -> None:
        self._lock = RLock()
        self._events: list[dict[str, Any]] = []
        self._load_events()
    
    def _load_events(self) -> None:
        self._events = list(DEFAULT_EVENTS)
    
    # ── Public API ──
    
    def active_events(self) -> list[dict[str, Any]]:
        """Get list of currently active events."""
        with self._lock:
            now = datetime.now()
            weekday = now.strftime("%A").lower()
            hour = now.hour
            active = []
            
            for event in self._events:
                # Check day of week
                dow = event.get("day_of_week", "")
                if dow and weekday not in dow.split(","):
                    continue
                
                # Check hour range
                hr = event.get("hour", None)
                hr_range = event.get("hour_range", None)
                if hr is not None and hour != hr:
                    continue
                if hr_range and not (hr_range[0] <= hour <= hr_range[1]):
                    continue
                
                active.append(event)
            
            return active
    
    def exp_multiplier(self) -> float:
        """Get current EXP multiplier from active events."""
        with self._lock:
            mult = 1.0
            for event in self.active_events():
                if event.get("type") == "exp":
                    mult *= event.get("multiplier", 1.0)
            return mult
    
    def drop_multiplier(self) -> float:
        """Get current drop multiplier from active events."""
        with self._lock:
            mult = 1.0
            for event in self.active_events():
                if event.get("type") == "drop":
                    mult *= event.get("multiplier", 1.0)
            return mult
    
    def is_mvp_hour(self) -> bool:
        """Check if we're in prime MVP hunting hours."""
        with self._lock:
            for event in self.active_events():
                if event.get("type") == "mvp":
                    return True
            return False
    
    def suggestion(self) -> dict[str, Any]:
        """Get farming suggestion based on active events."""
        with self._lock:
            exp = self.exp_multiplier()
            drop = self.drop_multiplier()
            mvp = self.is_mvp_hour()
            
            suggestions = []
            if exp > 1.0:
                suggestions.append(f"Double EXP ({exp}x) — prioritize leveling over farming")
            if drop > 1.0:
                suggestions.append(f"Double Drops ({drop}x) — prioritize farming over leveling")
            if mvp:
                suggestions.append("MVP spawn window active — consider MVP hunting")
            
            return {
                "exp_multiplier": exp,
                "drop_multiplier": drop,
                "mvp_hour": mvp,
                "suggestions": suggestions,
                "active_events": [e["name"] for e in self.active_events()],
            }
    
    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {"events_loaded": len(self._events), "active": len(self.active_events())}


def create_event_calendar(data_path: str | None = None) -> EventCalendar:
    return EventCalendar(data_path=data_path)
