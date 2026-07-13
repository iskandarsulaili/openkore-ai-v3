"""
Server profiler — learns server-specific behavior patterns over time.

Tracks GM patrol patterns, bot detection waves, player economy, events.
Initially empty — learns over days/weeks of runtime.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ServerProfiler:
    """Learns server-specific behavior patterns."""
    
    _lock: RLock = field(default_factory=RLock)
    
    # GM patrol tracking
    _gm_sightings: list[dict[str, Any]] = field(default_factory=list)
    _gm_patrol_hours: dict[int, int] = field(default_factory=lambda: defaultdict(int))  # hour -> sightings
    
    # Bot detection tracking
    _ban_events: list[dict[str, Any]] = field(default_factory=list)
    _suspicious_activity: list[dict[str, Any]] = field(default_factory=list)
    
    # Economy tracking
    _player_vendor_prices: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))  # item -> [prices]
    _npc_price_ratios: dict[str, float] = field(default_factory=dict)  # item -> player_avg / npc_price
    
    # Event tracking
    _server_events: list[dict[str, Any]] = field(default_factory=list)
    
    _stats: dict[str, int] = field(default_factory=lambda: {"gm_sightings": 0, "ban_events": 0, "price_records": 0})
    
    def report_gm_sighting(self, map_name: str, player_name: str) -> None:
        """Report a GM sighting."""
        now = time.localtime()
        with self._lock:
            self._gm_sightings.append({
                "map": map_name,
                "player": player_name,
                "hour": now.tm_hour,
                "weekday": now.tm_wday,
                "timestamp": time.time(),
            })
            self._gm_sightings = self._gm_sightings[-50:]
            self._gm_patrol_hours[now.tm_hour] += 1
            self._stats["gm_sightings"] += 1
    
    def get_gm_risk(self) -> float:
        """Get current GM patrol risk (0.0-1.0)."""
        now = time.localtime()
        hour = now.tm_hour
        with self._lock:
            sightings_this_hour = self._gm_patrol_hours.get(hour, 0)
            total_sightings = sum(self._gm_patrol_hours.values())
            if total_sightings == 0:
                return 0.0
            return min(1.0, sightings_this_hour / max(total_sightings / 24, 1) * 2)
    
    def report_ban_event(self, reason: str, map_name: str = "") -> None:
        """Report a bot ban event."""
        with self._lock:
            self._ban_events.append({
                "reason": reason,
                "map": map_name,
                "timestamp": time.time(),
            })
            self._ban_events = self._ban_events[-20:]
            self._stats["ban_events"] += 1
    
    def get_ban_risk(self) -> float:
        """Get current ban risk based on recent events."""
        with self._lock:
            recent = [e for e in self._ban_events if time.time() - e["timestamp"] < 3600]
            return min(1.0, len(recent) * 0.2)
    
    def record_vendor_price(self, item_name: str, price: int) -> None:
        """Record a player vendor price for an item."""
        with self._lock:
            self._player_vendor_prices[item_name].append(price)
            self._player_vendor_prices[item_name] = self._player_vendor_prices[item_name][-20:]
            self._stats["price_records"] += 1
    
    def get_avg_vendor_price(self, item_name: str) -> float | None:
        """Get the average player vendor price for an item."""
        with self._lock:
            prices = self._player_vendor_prices.get(item_name, [])
            if not prices:
                return None
            return sum(prices) / len(prices)
    
    def get_server_personality(self) -> dict[str, Any]:
        """Get the server's learned personality profile."""
        with self._lock:
            gm_risk = self.get_gm_risk()
            ban_risk = self.get_ban_risk()
            
            if gm_risk > 0.5 or ban_risk > 0.3:
                strictness = "strict"
            elif gm_risk > 0.2:
                strictness = "moderate"
            else:
                strictness = "laissez_faire"
            
            return {
                "strictness": strictness,
                "gm_risk": round(gm_risk, 2),
                "ban_risk": round(ban_risk, 2),
                "gm_patrol_hours": dict(self._gm_patrol_hours),
                "total_gm_sightings": self._stats["gm_sightings"],
                "total_ban_events": self._stats["ban_events"],
                "recommendation": "be_careful" if strictness == "strict" else "normal",
            }
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)
