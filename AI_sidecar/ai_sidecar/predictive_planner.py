"""
Predictive planner — anticipates needs before they arise.

A pro player predicts. They stock up on arrows before they run out.
They pre-buff before entering combat. They plan gear upgrades 10 levels ahead.
This module forecasts future needs and triggers proactive actions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PredictivePlanner:
    """Forecasts future needs and triggers proactive actions."""
    
    _lock: RLock = field(default_factory=RLock)
    _stats: dict[str, int] = field(default_factory=lambda: {"predictions": 0, "preventions": 0})
    
    def predict_potion_needs(self, current_potions: int, level: int, hunting_zone: str) -> dict[str, Any]:
        """Predict how many potions will be needed and when to buy more."""
        # Estimate potion consumption rate
        # Novices: ~1 potion per 2 minutes of combat
        # Mid-level: ~1 potion per 5 minutes
        # High-level: ~1 potion per 10 minutes
        if level < 20:
            consumption_per_min = 0.5
        elif level < 50:
            consumption_per_min = 0.2
        else:
            consumption_per_min = 0.1
        
        minutes_remaining = current_potions / consumption_per_min if consumption_per_min > 0 else 999
        
        with self._lock:
            self._stats["predictions"] += 1
        
        return {
            "current_potions": current_potions,
            "consumption_per_min": consumption_per_min,
            "minutes_remaining": minutes_remaining,
            "should_buy_soon": minutes_remaining < 15,
            "critical": minutes_remaining < 5,
        }
    
    def predict_arrow_needs(self, current_arrows: int, level: int) -> dict[str, Any]:
        """Predict arrow consumption for ranged classes."""
        if level < 20:
            consumption_per_min = 10
        elif level < 50:
            consumption_per_min = 20
        else:
            consumption_per_min = 30
        
        minutes_remaining = current_arrows / consumption_per_min if consumption_per_min > 0 else 999
        
        return {
            "current_arrows": current_arrows,
            "minutes_remaining": minutes_remaining,
            "should_buy_soon": minutes_remaining < 30,
            "critical": minutes_remaining < 10,
        }
    
    def predict_gear_upgrade(self, level: int, current_weapon_atk: int, zeny: int) -> dict[str, Any] | None:
        """Predict when to upgrade gear based on level and zeny."""
        # Every 10 levels, check if gear should be upgraded
        next_milestone = ((level // 10) + 1) * 10
        levels_until_upgrade = next_milestone - level
        
        if levels_until_upgrade <= 3 and zeny > 10000:
            with self._lock:
                self._stats["predictions"] += 1
            return {
                "recommend_level": next_milestone,
                "levels_until": levels_until_upgrade,
                "current_atk": current_weapon_atk,
                "zeny_available": zeny,
                "should_prepare": True,
            }
        return None
    
    def predict_zone_change(self, level: int, current_zone_max_level: int, exp_rate: float) -> dict[str, Any] | None:
        """Predict when to change hunting zones."""
        if level >= current_zone_max_level - 5 and exp_rate < 0.5:
            with self._lock:
                self._stats["predictions"] += 1
            return {
                "reason": "zone_outleveled",
                "current_level": level,
                "zone_max_level": current_zone_max_level,
                "exp_rate": exp_rate,
                "should_move": True,
            }
        return None
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)
