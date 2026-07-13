"""
Server awareness — detects server state, player density, time patterns, events.

A pro player adjusts to server conditions in real time. This module tracks:
- Time of day (peak vs off-peak)
- Player density estimation (how many players nearby)
- Server events (WoE, MVP competitions, holidays)
- Bot detection risk windows
- Lag/performance monitoring
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

# Peak hours by timezone (server-local)
PEAK_HOURS = range(18, 23)  # 6pm-11pm
OFF_PEAK_HOURS = range(2, 7)  # 2am-7am

# Event schedules (approximate, server-dependent)
WOE_SCHEDULE = {
    "wednesday": (20, 22),  # 8pm-10pm
    "saturday": (20, 22),
    "sunday": (20, 22),
}

# Risk windows for bot detection
HIGH_RISK_HOURS = range(0, 6)  # 12am-6am — GMs check overnight
MEDIUM_RISK_HOURS = range(6, 12)  # 6am-12pm


@dataclass(slots=True)
class ServerAwareness:
    """Tracks server state and adjusts behavior accordingly."""
    
    _lock: RLock = field(default_factory=RLock)
    _player_count_estimates: dict[str, int] = field(default_factory=dict)  # map -> estimated players
    _last_lag_check: float = 0.0
    _lag_spikes: list[float] = field(default_factory=list)
    _stats: dict[str, int] = field(default_factory=lambda: {"adjustments": 0, "lag_events": 0})
    
    def get_server_state(self, current_map: str) -> dict[str, Any]:
        """Get current server state for decision-making."""
        now = time.localtime()
        hour = now.tm_hour
        weekday = now.tm_wday  # 0=Monday, 6=Sunday
        weekday_name = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"][weekday]
        
        with self._lock:
            player_density = self._player_count_estimates.get(current_map, 0)
        
        is_peak = hour in PEAK_HOURS
        is_off_peak = hour in OFF_PEAK_HOURS
        is_high_risk = hour in HIGH_RISK_HOURS
        is_medium_risk = hour in MEDIUM_RISK_HOURS
        
        # Check WoE
        woe_schedule = WOE_SCHEDULE.get(weekday_name)
        is_woe = False
        if woe_schedule:
            is_woe = woe_schedule[0] <= hour < woe_schedule[1]
        
        # Risk level
        if is_high_risk:
            risk_level = "high"
        elif is_medium_risk:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Recommendations
        recommendations: list[str] = []
        if is_woe:
            recommendations.append("avoid_pvp_maps")
            recommendations.append("stay_in_town")
        if is_peak and player_density > 5:
            recommendations.append("switch_to_less_crowded_map")
        if is_high_risk:
            recommendations.append("reduce_uptime")
            recommendations.append("avoid_obvious_bot_patterns")
        if player_density > 10:
            recommendations.append("too_crowded_move")
        
        return {
            "hour": hour,
            "weekday": weekday_name,
            "is_peak": is_peak,
            "is_off_peak": is_off_peak,
            "is_woe": is_woe,
            "risk_level": risk_level,
            "player_density": player_density,
            "recommendations": recommendations,
        }
    
    def report_player_sighting(self, map_name: str) -> None:
        """Report seeing another player on a map."""
        with self._lock:
            self._player_count_estimates[map_name] = self._player_count_estimates.get(map_name, 0) + 1
            self._stats["adjustments"] += 1
    
    def report_lag_spike(self, latency_ms: float) -> None:
        """Report a lag spike."""
        with self._lock:
            self._lag_spikes.append(latency_ms)
            self._lag_spikes = self._lag_spikes[-20:]  # Keep last 20
            self._stats["lag_events"] += 1
    
    def is_lagging(self) -> bool:
        """Check if the server is currently lagging."""
        with self._lock:
            if len(self._lag_spikes) < 3:
                return False
            recent = self._lag_spikes[-3:]
            return all(l > 500 for l in recent)  # 3 consecutive spikes >500ms
    
    def get_farm_intensity(self) -> str:
        """Get recommended farming intensity based on risk."""
        state = self.get_server_state("")
        if state["risk_level"] == "high":
            return "conservative"  # Farm slowly, take breaks
        if state["is_peak"]:
            return "normal"
        return "aggressive"  # Off-peak, farm hard
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)
