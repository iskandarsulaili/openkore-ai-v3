"""
Movement optimizer — scores route options (walk vs Fly Wing vs Kafra vs Butterfly Wing).

The LLM decides DESTINATION; the optimizer decides HOW to get there.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

# Estimated travel times (seconds) between major towns
TOWN_TRAVEL_TIMES: dict[str, dict[str, float]] = {
    "prontera": {"morocc": 120, "geffen": 90, "payon": 150, "aldebaran": 180, "yuno": 300},
    "morocc": {"prontera": 120, "geffen": 180, "payon": 200, "aldebaran": 250},
    "geffen": {"prontera": 90, "morocc": 180, "payon": 120, "aldebaran": 150},
    "payon": {"prontera": 150, "morocc": 200, "geffen": 120, "aldebaran": 100},
    "aldebaran": {"prontera": 180, "morocc": 250, "geffen": 150, "payon": 100},
    "yuno": {"prontera": 300, "geffen": 200},
}

# Cost of travel methods
FLY_WING_COST = 500
BUTTERFLY_WING_COST = 2000
KAFRA_FEE = 100


@dataclass(slots=True)
class MovementOptimizer:
    """Optimizes travel between locations."""
    
    _lock: RLock = field(default_factory=RLock)
    _stats: dict[str, int] = field(default_factory=lambda: {"walk": 0, "fly_wing": 0, "butterfly": 0, "kafra": 0})
    
    def get_best_route(self, current_map: str, target_map: str, zeny: int,
                       has_fly_wings: bool, has_butterfly_wings: bool) -> dict[str, Any]:
        """Score route options and return the best one."""
        options = []
        
        # Option 1: Walk
        walk_time = self._estimate_walk_time(current_map, target_map)
        options.append({
            "method": "walk",
            "time_s": walk_time,
            "cost": 0,
            "command": f"move {target_map}",
            "score": 100.0 / max(walk_time, 1),
        })
        
        # Option 2: Fly Wing (random teleport, then walk)
        if has_fly_wings and zeny >= FLY_WING_COST:
            fly_time = walk_time * 0.3  # Fly wing saves ~70% time
            options.append({
                "method": "fly_wing",
                "time_s": fly_time,
                "cost": FLY_WING_COST,
                "command": "ai manual",
                "score": 100.0 / max(fly_time, 1) * 0.8,  # 0.8 penalty for cost
            })
        
        # Option 3: Butterfly Wing (return to save point, then walk)
        if has_butterfly_wings and zeny >= BUTTERFLY_WING_COST:
            bw_time = 5 + self._estimate_walk_time("prontera", target_map)  # 5s to use wing
            options.append({
                "method": "butterfly_wing",
                "time_s": bw_time,
                "cost": BUTTERFLY_WING_COST,
                "command": "ai manual",
                "score": 100.0 / max(bw_time, 1) * 0.6,
            })
        
        # Option 4: Kafra Warp (if both towns have Kafra)
        if zeny >= KAFRA_FEE:
            kafra_time = 15  # Talk to Kafra + warp animation
            options.append({
                "method": "kafra",
                "time_s": kafra_time,
                "cost": KAFRA_FEE,
                "command": "talknpc Kafra",
                "score": 100.0 / max(kafra_time, 1) * 0.9,
            })
        
        options.sort(key=lambda o: o["score"], reverse=True)
        
        with self._lock:
            if options:
                self._stats[options[0]["method"]] += 1
        
        return options[0] if options else {"method": "walk", "command": f"move {target_map}"}
    
    def _estimate_walk_time(self, current: str, target: str) -> float:
        """Estimate walking time between two maps."""
        if current == target:
            return 0
        # Check direct connection
        if current in TOWN_TRAVEL_TIMES and target in TOWN_TRAVEL_TIMES[current]:
            return TOWN_TRAVEL_TIMES[current][target]
        # Estimate based on map name similarity (same field area)
        current_prefix = current.split("_")[0] if "_" in current else current
        target_prefix = target.split("_")[0] if "_" in target else target
        if current_prefix == target_prefix:
            return 60  # Same field area — close
        return 300  # Cross-country — far
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)
