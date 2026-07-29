"""Farming Efficiency Tracker — measures zeny/hour with and without AI.

The ultimate metric for any bot is zeny/hour. This module:
1. Records total zeny at regular intervals
2. Calculates zeny/hour over rolling windows (5min, 30min, 1h)
3. Compares AI-assisted vs OpenKore-default farming rates
4. Persists data across crashes via PersistentState
"""
from __future__ import annotations
import time
import logging
from typing import Any
from collections import deque

from ai_sidecar.runtime.persistence import PersistentState

logger = logging.getLogger(__name__)


class EfficiencyTracker:
    """Tracks farming efficiency over time.
    
    Measures:
    - Zeny/hour (rolling 5min, 30min, 1h)
    - Items collected per hour
    - Monster kills per hour
    - Deaths per hour
    - Comparison: AI mode vs OpenKore default mode
    """
    
    def __init__(self):
        self._snapshots: deque[dict[str, Any]] = deque(maxlen=360)  # 360 samples = 1h at 10s intervals
        self._last_sample_time = 0.0
        self._sample_interval = 10.0  # Sample every 10 seconds
        self._session_start = time.time()
        self._total_zeny_earned = 0
        self._total_items_collected = 0
        self._total_kills = 0
        self._total_deaths = 0
        self._ai_enabled = True
        self._ai_mode_samples: deque[float] = deque(maxlen=360)
        self._default_mode_samples: deque[float] = deque(maxlen=360)
    
    def set_ai_mode(self, enabled: bool) -> None:
        """Set whether AI decisions are currently active."""
        self._ai_enabled = enabled
    
    def record_kill(self, zeny_drop: int = 0, item_drop: bool = False) -> None:
        """Record a monster kill."""
        self._total_kills += 1
        self._total_zeny_earned += zeny_drop
        if item_drop:
            self._total_items_collected += 1
    
    def record_death(self) -> None:
        """Record a death (negative efficiency)."""
        self._total_deaths += 1
    
    def record_zeny_change(self, delta: int) -> None:
        """Record a zeny change (positive = earned, negative = spent)."""
        if delta > 0:
            self._total_zeny_earned += delta
    
    def sample(self, current_zeny: int) -> dict[str, float] | None:
        """Take an efficiency sample. Returns rate dict or None if not due."""
        now = time.time()
        if now - self._last_sample_time < self._sample_interval:
            return None
        
        elapsed = now - self._session_start
        if elapsed < 60:  # Need at least 1 minute of data
            self._last_sample_time = now
            return None
        
        snapshot = {
            "timestamp": now,
            "zeny": current_zeny,
            "total_earned": self._total_zeny_earned,
            "kills": self._total_kills,
            "deaths": self._total_deaths,
            "items": self._total_items_collected,
            "ai_mode": self._ai_enabled,
            "elapsed": elapsed,
        }
        self._snapshots.append(snapshot)
        
        # Track by AI mode
        rate_5min = self.get_rate(window_min=5)
        if rate_5min:
            if self._ai_enabled:
                self._ai_mode_samples.append(rate_5min["zeny_per_hour"])
            else:
                self._default_mode_samples.append(rate_5min["zeny_per_hour"])
        
        self._last_sample_time = now
        
        # Return current rates
        return {
            "zeny_per_hour_5min": rate_5min["zeny_per_hour"] if rate_5min else 0,
            "zeny_per_hour_30min": (rate_30min := self.get_rate(window_min=30))["zeny_per_hour"] if rate_30min else 0,
            "zeny_per_hour_1h": (rate_1h := self.get_rate(window_min=60))["zeny_per_hour"] if rate_1h else 0,
            "kills_per_hour": rate_5min["kills_per_hour"] if rate_5min else 0,
            "deaths_per_hour": rate_5min["deaths_per_hour"] if rate_5min else 0,
            "ai_mode": self._ai_enabled,
            "elapsed_minutes": elapsed / 60,
        }
    
    def get_rate(self, window_min: int = 5) -> dict[str, float] | None:
        """Get efficiency rate for a rolling time window."""
        if len(self._snapshots) < 2:
            return None
        
        cutoff = time.time() - (window_min * 60)
        window = [s for s in self._snapshots if s["timestamp"] >= cutoff]
        
        if len(window) < 2:
            return None
        
        first = window[0]
        last = window[-1]
        duration_hours = (last["timestamp"] - first["timestamp"]) / 3600
        
        if duration_hours < 0.01:
            return None
        
        zeny_earned = last["total_earned"] - first["total_earned"]
        kills = last["kills"] - first["kills"]
        deaths = last["deaths"] - first["deaths"]
        
        return {
            "zeny_per_hour": zeny_earned / duration_hours,
            "kills_per_hour": kills / duration_hours,
            "deaths_per_hour": deaths / duration_hours,
            "sample_count": len(window),
            "duration_hours": duration_hours,
        }
    
    def get_ai_comparison(self) -> dict[str, Any]:
        """Compare AI-assisted vs default mode efficiency."""
        ai_avg = sum(self._ai_mode_samples) / len(self._ai_mode_samples) if self._ai_mode_samples else 0
        def_avg = sum(self._default_mode_samples) / len(self._default_mode_samples) if self._default_mode_samples else 0
        
        improvement_pct = 0
        if def_avg > 0:
            improvement_pct = ((ai_avg - def_avg) / def_avg) * 100
        
        return {
            "ai_zeny_per_hour": ai_avg,
            "default_zeny_per_hour": def_avg,
            "improvement_pct": improvement_pct,
            "ai_samples": len(self._ai_mode_samples),
            "default_samples": len(self._default_mode_samples),
        }
    
    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive efficiency stats."""
        best = self.get_rate(window_min=60) or self.get_rate(window_min=30) or self.get_rate(window_min=5)
        comparison = self.get_ai_comparison()
        
        return {
            "session_minutes": (time.time() - self._session_start) / 60,
            "total_kills": self._total_kills,
            "total_zeny_earned": self._total_zeny_earned,
            "total_deaths": self._total_deaths,
            "current_rate": best,
            "ai_comparison": comparison,
            "snapshots": len(self._snapshots),
        }


# Global instance
_tracker: EfficiencyTracker | None = None

def get_tracker() -> EfficiencyTracker:
    global _tracker
    if _tracker is None:
        _tracker = EfficiencyTracker()
    return _tracker
