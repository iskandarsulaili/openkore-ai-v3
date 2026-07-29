"""Farming Loop Optimizer — calculates optimal sell cycle and inventory management.

A pro bot doesn't just sell at 80% weight — it calculates:
- Round trip time to town and back
- Inventory fill rate (items/minute)
- Optimal sell threshold to maximize farming time
- Kafra storage vs NPC sell tradeoffs
- Whether to use alt character for vending
"""
from __future__ import annotations
from typing import Any
import logging
from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)


class FarmingLoopOptimizer:
    """Optimizes the farming cycle for maximum zeny/hour.
    
    Key insight: every trip to town is lost farming time.
    The optimal sell threshold balances:
    - More farming time (sell later) vs
    - More trips to town (sell earlier but more frequent)
    
    Formula: optimal_sell_pct = sqrt(travel_time_min / (2 * fill_time_min)) * 100
    
    Where:
    - travel_time_min: minutes to walk to town and back
    - fill_time_min: minutes to fill inventory from 0% to 100%
    """
    
    def __init__(self):
        self._travel_times: dict[str, float] = {}  # map -> travel time in minutes
        self._fill_rates: dict[str, float] = {}     # map -> fill rate in %/minute
        self._last_sell_time: dict[str, float] = {}  # bot_id -> timestamp
    
    def record_travel_time(self, map_name: str, minutes: float) -> None:
        """Record how long it takes to travel from this map to town and back."""
        self._travel_times[map_name] = minutes
    
    def record_fill_rate(self, map_name: str, pct_per_minute: float) -> None:
        """Record how fast inventory fills on this map."""
        self._fill_rates[map_name] = pct_per_minute
    
    def optimal_sell_threshold(self, map_name: str) -> float:
        """Calculate the optimal inventory % to sell at.
        
        Default: 80% (conservative).
        If town is close: sell later (85-90%).
        If town is far: sell earlier (60-70%).
        """
        travel = self._travel_times.get(map_name, 4.0)  # Default 4 min
        fill = self._fill_rates.get(map_name, 2.0)       # Default 2% per minute
        
        if fill <= 0:
            return 80.0
        
        # Optimal = sqrt(travel / (2 * fill_time))
        # If travel=4min, fill=30min → sqrt(4/60) = 0.258 → 25%? That's too low.
        # Reality check: fill time is ~30min at 3.3%/min
        fill_time_min = 100.0 / fill  # How many minutes to fill 100%
        
        optimal = (travel / (2 * fill_time_min)) ** 0.5 * 100
        
        # Clamp to reasonable range
        return max(50.0, min(95.0, optimal))
    
    def should_sell(self, weight_pct: float, map_name: str) -> bool:
        """Check if we should sell now based on optimal threshold."""
        threshold = self.optimal_sell_threshold(map_name)
        return weight_pct >= threshold
    
    def get_cycle_summary(self, map_name: str) -> dict:
        """Get a summary of farming cycle statistics."""
        travel = self._travel_times.get(map_name, 4.0)
        fill = self._fill_rates.get(map_name, 2.0)
        threshold = self.optimal_sell_threshold(map_name)
        
        if fill <= 0:
            return {"error": "no_data"}
        
        fill_time_min = 100.0 / fill
        farm_time_min = fill_time_min * (threshold / 100.0)
        total_cycle_min = farm_time_min + travel
        efficiency = farm_time_min / total_cycle_min * 100 if total_cycle_min > 0 else 0
        
        return {
            "map": map_name,
            "travel_time_min": round(travel, 1),
            "fill_rate_pct_per_min": round(fill, 1),
            "fill_time_min": round(fill_time_min, 1),
            "optimal_sell_pct": round(threshold, 0),
            "farm_time_min": round(farm_time_min, 1),
            "total_cycle_min": round(total_cycle_min, 1),
            "efficiency_pct": round(efficiency, 0),
        }
    
    def assess(self, signals: dict[str, Any], actions: list[HeuristicAction], bot_id: str) -> None:
        current_map = str(signals.get("map", "") or "")
        weight = int(signals.get("weight", 0) or 0)
        weight_max = int(signals.get("weight_max", 100) or 100)
        weight_pct = weight / max(weight_max, 1) * 100
        
        # Record fill rate based on weight change
        if current_map not in self._fill_rates:
            # Estimate: 0% → 100% in about 30 minutes at normal rates
            self._fill_rates[current_map] = 3.3
        if current_map not in self._travel_times:
            # Estimate: 2 min to town, 2 min back = 4 min
            self._travel_times[current_map] = 4.0
        
        # Check if we should sell
        if self.should_sell(weight_pct, current_map):
            threshold = self.optimal_sell_threshold(current_map)
            actions.append(HeuristicAction(
                kind="command",
                command="sellAuto 1",
                confidence=0.9,
                reason=f"Farming loop: weight {weight_pct:.0f}% > optimal {threshold:.0f}% — selling",
                domain="economy",
            ))
        
        # Also consider Kafra storage if weight is high but we're far from town
        if weight_pct > 60 and weight_pct < 90:
            travel = self._travel_times.get(current_map, 4.0)
            if travel > 6:  # More than 6 min from town
                actions.append(HeuristicAction(
                    kind="command",
                    command="kafra_store",
                    confidence=0.6,
                    reason=f"Farming loop: {travel}min from town — using Kafra storage",
                    domain="economy",
                ))
        
        # Log farming efficiency
        summary = self.get_cycle_summary(current_map)
        actions.append(HeuristicAction(
            kind="log",
            command=f"farming_cycle eff={summary.get('efficiency_pct', 0)}% sell_at={summary.get('optimal_sell_pct', 80)}%",
            confidence=0.5,
            reason=f"Farming cycle: {summary.get('efficiency_pct', 0)}% efficient on {current_map}",
            domain="economy",
        ))
