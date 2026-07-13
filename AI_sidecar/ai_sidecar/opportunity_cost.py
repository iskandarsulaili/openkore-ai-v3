"""
Opportunity cost optimizer — minimizes non-productive time.

A pro player has 95%+ uptime. They minimize every non-farming activity:
- Stock up on potions before leaving town
- Plan route to minimize walking
- Know exactly when to return based on weight
- Never stand still unless necessary
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
class OpportunityCostOptimizer:
    """Minimizes non-productive time by optimizing every action."""
    
    _lock: RLock = field(default_factory=RLock)
    _uptime_tracker: dict[str, dict[str, Any]] = field(default_factory=dict)
    _stats: dict[str, int] = field(default_factory=lambda: {"uptime_checks": 0, "optimizations_applied": 0})
    
    def record_activity(self, bot_id: str, activity: str, duration_s: float) -> None:
        """Record what the bot was doing and for how long."""
        with self._lock:
            if bot_id not in self._uptime_tracker:
                self._uptime_tracker[bot_id] = {
                    "farming": 0.0,
                    "walking": 0.0,
                    "sorting": 0.0,
                    "shopping": 0.0,
                    "dead": 0.0,
                    "idle": 0.0,
                    "total": 0.0,
                    "start_time": time.time(),
                }
            if activity in self._uptime_tracker[bot_id]:
                self._uptime_tracker[bot_id][activity] += duration_s
            self._uptime_tracker[bot_id]["total"] += duration_s
            self._stats["uptime_checks"] += 1
    
    def get_uptime(self, bot_id: str) -> dict[str, Any]:
        """Calculate uptime percentage and identify waste."""
        with self._lock:
            tracker = self._uptime_tracker.get(bot_id)
            if not tracker or tracker["total"] <= 0:
                return {"uptime_pct": 0.0, "waste_pct": 0.0, "recommendations": []}
            
            productive = tracker["farming"]
            waste = tracker["walking"] + tracker["sorting"] + tracker["dead"] + tracker["idle"]
            total = tracker["total"]
            
            uptime_pct = (productive / total) * 100
            waste_pct = (waste / total) * 100
            
            recommendations = []
            if tracker["walking"] / total > 0.2:
                recommendations.append("Too much walking — use Fly Wings or Kafra warps")
            if tracker["sorting"] / total > 0.1:
                recommendations.append("Too much inventory sorting — use auto-storage or sell more frequently")
            if tracker["dead"] / total > 0.05:
                recommendations.append("Too much time dead — improve survivability or avoid dangerous zones")
            if tracker["idle"] / total > 0.1:
                recommendations.append("Too much idle time — check for stuck states or pathfinding issues")
            
            return {
                "uptime_pct": round(uptime_pct, 1),
                "waste_pct": round(waste_pct, 1),
                "farming": round(tracker["farming"], 1),
                "walking": round(tracker["walking"], 1),
                "sorting": round(tracker["sorting"], 1),
                "dead": round(tracker["dead"], 1),
                "idle": round(tracker["idle"], 1),
                "recommendations": recommendations,
            }
    
    def should_return_to_town(self, bot_id: str, weight_pct: float, 
                               potion_count: int, min_potions: int = 10) -> dict[str, Any]:
        """Determine if the bot should return to town based on opportunity cost."""
        with self._lock:
            uptime = self.get_uptime(bot_id)
            result = {"should_return": False, "reason": "", "priority": "low"}
            
            # Weight check
            if weight_pct > 0.8:
                result["should_return"] = True
                result["reason"] = "Inventory nearly full"
                result["priority"] = "high"
            elif weight_pct > 0.6:
                result["should_return"] = True
                result["reason"] = "Inventory getting full"
                result["priority"] = "medium"
            
            # Potion check
            if potion_count < min_potions:
                result["should_return"] = True
                result["reason"] = f"Only {potion_count} potions left"
                result["priority"] = "high" if potion_count < 5 else "medium"
            
            # Uptime check
            if uptime["uptime_pct"] < 50 and uptime["total"] > 300:
                result["should_return"] = True
                result["reason"] = f"Low uptime ({uptime['uptime_pct']}%) — need to fix issues"
                result["priority"] = "high"
            
            return result
    
    def get_optimal_return_weight(self, bot_id: str, max_weight: int) -> int:
        """Calculate the optimal weight to return to town."""
        with self._lock:
            uptime = self.get_uptime(bot_id)
            # If walking time is high, return less often (carry more)
            if uptime["walking"] > uptime["farming"] * 0.3:
                return int(max_weight * 0.85)
            # If farming is efficient, return more often (carry less)
            return int(max_weight * 0.7)
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)
