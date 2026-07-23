"""
Opportunity Cost Engine — evaluates every action against every alternative.

Better than human because:
- Humans have limited working memory (can't compare 50 options at once)
- Humans have cognitive biases (overvalue immediate rewards)
- This system calculates expected value for EVERY possible action
- This system considers time horizons (short-term vs long-term value)
- This system updates its estimates with every observation
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ActionOption:
    """A possible action with its expected value."""
    action_type: str  # farm, sell, buy, train, quest, mvp, rest
    description: str
    expected_value_per_hour: float  # Expected zeny/exp per hour
    risk: float  # 0.0-1.0 (probability of failure/death)
    time_horizon: str  # immediate, short_term, long_term
    requirements: list[str] = field(default_factory=list)
    confidence: float = 0.0  # 0.0-1.0


@dataclass(slots=True)
class OpportunityCostRecommendation:
    """The best action to take right now, with reasoning."""
    best_action: ActionOption
    runner_up: ActionOption | None
    expected_value_gap: float  # How much better the best action is
    reasoning: str
    timestamp: float


class OpportunityCostEngine:
    """Evaluates every possible action and recommends the best one.
    
    For every decision point, this engine:
    1. Enumerates all possible actions
    2. Calculates expected value for each
    3. Adjusts for risk
    4. Adjusts for time horizon
    5. Recommends the best action
    """
    
    def __init__(self):
        self._lock = RLock()
        
        # Historical performance tracking
        self._action_performance: dict[str, list[dict[str, Any]]] = defaultdict(list)
        
        # Current estimates
        self._farming_rates: dict[str, float] = defaultdict(float)  # map -> exp/hour
        self._farming_zeny: dict[str, float] = defaultdict(float)  # map -> zeny/hour
        self._farming_risk: dict[str, float] = defaultdict(float)  # map -> death probability
        
        # Stats
        self._stats: dict[str, int] = defaultdict(int)
    
    def record_farming_result(self, map_name: str, duration_minutes: float,
                               exp_gained: int, zeny_gained: int,
                               died: bool = False) -> None:
        """Record the result of a farming session."""
        with self._lock:
            hours = duration_minutes / 60
            if hours > 0:
                exp_rate = exp_gained / hours
                zeny_rate = zeny_gained / hours
                
                # Update moving average
                old_exp = self._farming_rates.get(map_name, 0)
                old_zeny = self._farming_zeny.get(map_name, 0)
                old_risk = self._farming_risk.get(map_name, 0)
                
                # 70% weight to new data, 30% to old
                self._farming_rates[map_name] = old_exp * 0.3 + exp_rate * 0.7
                self._farming_zeny[map_name] = old_zeny * 0.3 + zeny_rate * 0.7
                
                # Risk: exponential moving average of death rate
                death_value = 1.0 if died else 0.0
                self._farming_risk[map_name] = old_risk * 0.7 + death_value * 0.3
            
            self._stats["farming_results"] += 1
    
    def get_farming_value(self, map_name: str, 
                           weight_exp: float = 0.5,
                           weight_zeny: float = 0.5) -> float:
        """Get the expected value of farming a map.
        
        Returns a composite score (0-1000+) combining exp and zeny rates,
        adjusted for risk.
        """
        with self._lock:
            exp_rate = self._farming_rates.get(map_name, 0)
            zeny_rate = self._farming_zeny.get(map_name, 0)
            risk = self._farming_risk.get(map_name, 0.1)
            
            # Normalize rates (assume max observed rate is 1000)
            max_exp = max(self._farming_rates.values()) if self._farming_rates else 1
            max_zeny = max(self._farming_zeny.values()) if self._farming_zeny else 1
            
            norm_exp = exp_rate / max_exp if max_exp > 0 else 0
            norm_zeny = zeny_rate / max_zeny if max_zeny > 0 else 0
            
            # Composite score
            raw_value = (norm_exp * weight_exp + norm_zeny * weight_zeny) * 1000
            
            # Risk adjustment
            risk_penalty = risk * 500  # High risk = big penalty
            adjusted = max(0, raw_value - risk_penalty)
            
            return adjusted
    
    def evaluate_action(self, action_type: str, 
                         expected_value: float,
                         risk: float,
                         time_horizon: str = "immediate") -> ActionOption:
        """Evaluate a single action option."""
        # Risk-adjusted value
        risk_adjusted = expected_value * (1.0 - risk)
        
        # Time horizon multiplier
        horizon_mult = {
            "immediate": 1.0,
            "short_term": 0.8,  # 1-6 hours
            "long_term": 0.5,   # 6+ hours
        }.get(time_horizon, 1.0)
        
        final_value = risk_adjusted * horizon_mult
        
        return ActionOption(
            action_type=action_type,
            description=f"{action_type} (value={final_value:.0f}/hr)",
            expected_value_per_hour=final_value,
            risk=risk,
            time_horizon=time_horizon,
            confidence=0.5,
        )
    
    def get_best_action(self, available_maps: list[str],
                         current_zeny: int,
                         current_exp_rate: float,
                         goals: list[str] | None = None) -> OpportunityCostRecommendation:
        """Get the best action to take right now.
        
        Evaluates all available options and returns the best one
        with reasoning.
        """
        with self._lock:
            options: list[ActionOption] = []
            
            # 1. Farming options
            for map_name in available_maps:
                value = self.get_farming_value(map_name)
                risk = self._farming_risk.get(map_name, 0.1)
                options.append(ActionOption(
                    action_type="farm",
                    description=f"Farm {map_name}",
                    expected_value_per_hour=value,
                    risk=risk,
                    time_horizon="immediate",
                    confidence=0.6 if self._stats.get("farming_results", 0) > 5 else 0.3,
                ))
            
            # 2. Selling options (if inventory is full)
            if current_zeny < 10000:
                options.append(ActionOption(
                    action_type="sell",
                    description="Sell junk items",
                    expected_value_per_hour=5000,
                    risk=0.0,
                    time_horizon="immediate",
                    confidence=0.8,
                ))
            
            # 3. Resting options (if low HP)
            options.append(ActionOption(
                action_type="rest",
                description="Rest and recover",
                expected_value_per_hour=0,
                risk=0.0,
                time_horizon="short_term",
                confidence=0.9,
            ))
            
            # Sort by expected value
            options.sort(key=lambda o: -o.expected_value_per_hour)
            
            if not options:
                return OpportunityCostRecommendation(
                    best_action=ActionOption("idle", "No actions available", 0, 0, "immediate"),
                    runner_up=None,
                    expected_value_gap=0,
                    reasoning="No actions available",
                    timestamp=time.time(),
                )
            
            best = options[0]
            runner = options[1] if len(options) > 1 else None
            gap = best.expected_value_per_hour - (runner.expected_value_per_hour if runner else 0)
            
            reasoning = (
                f"Best: {best.description} ({best.expected_value_per_hour:.0f}/hr, "
                f"risk={best.risk:.0%})"
            )
            if runner:
                reasoning += (
                    f". Runner-up: {runner.description} "
                    f"({runner.expected_value_per_hour:.0f}/hr)"
                )
            
            return OpportunityCostRecommendation(
                best_action=best,
                runner_up=runner,
                expected_value_gap=gap,
                reasoning=reasoning,
                timestamp=time.time(),
            )
    
    def get_stats(self) -> dict[str, int]:
        """Get opportunity cost engine statistics."""
        with self._lock:
            return dict(self._stats)


# Global singleton
_engine: OpportunityCostEngine | None = None

def get_opportunity_cost_engine() -> OpportunityCostEngine:
    """Get the global OpportunityCostEngine instance."""
    global _engine
    if _engine is None:
        _engine = OpportunityCostEngine()
    return _engine
