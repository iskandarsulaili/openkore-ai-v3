"""
Meta prediction — anticipates game changes and adapts strategy.

A pro player reads patch notes and immediately knows:
- New skill changes will make Agi Knights viable again
- New dungeon will crash prices of certain cards
- New equipment will obsolete current builds
- New MVP requires completely different strategy
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MetaPrediction:
    """Predicts meta shifts and recommends proactive adaptation."""
    
    _lock: RLock = field(default_factory=RLock)
    _meta_history: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    _stats: dict[str, int] = field(default_factory=lambda: {"predictions_made": 0, "adaptations_applied": 0})
    
    def __post_init__(self) -> None:
        # Known meta cycles (time-based) — initialized here to avoid mutable default issues
        self.META_CYCLES: dict[str, dict[str, Any]] = {
        "woe_season": {
            "description": "WoE season — tanky builds and AoE skills dominate",
            "recommended_classes": ["knight", "wizard", "priest"],
            "recommended_gear": ["valkyrie_armor", "freyja_shield", "woe_potions"],
            "avoid_classes": ["thief", "hunter"],
            "timing": "weekend_evening",
        },
        "leveling_season": {
            "description": "Leveling season — EXP efficiency is king",
            "recommended_classes": ["mage", "archer", "thief"],
            "recommended_gear": ["exp_boost_gear", "efficient_weapons"],
            "avoid_classes": ["merchant"],
            "timing": "weekday",
        },
        "mvp_hunting": {
            "description": "MVP hunting season — boss-killing builds",
            "recommended_classes": ["knight", "assassin", "priest"],
            "recommended_gear": ["mvp_weapons", "elemental_armors"],
            "avoid_classes": ["mage"],
            "timing": "early_morning",
        },
    }
    
    def predict_meta_shift(self, current_time: time.struct_time | None = None) -> dict[str, Any]:
        """Predict what the current meta looks like based on time patterns."""
        self._stats["predictions_made"] += 1
        now = current_time or time.localtime()
        hour = now.tm_hour
        day = now.tm_wday  # 0=Monday, 6=Sunday
        
        # Weekend evening = WoE season
        if day >= 5 and hour >= 18:
            return {
                "meta": "woe_season",
                "details": self.META_CYCLES["woe_season"],
                "confidence": 0.9,
                "action": "switch_to_woe_build",
            }
        
        # Early morning = MVP hunting
        if hour < 8:
            return {
                "meta": "mvp_hunting",
                "details": self.META_CYCLES["mvp_hunting"],
                "confidence": 0.7,
                "action": "prepare_mvp_gear",
            }
        
        # Weekday daytime = leveling
        return {
            "meta": "leveling_season",
            "details": self.META_CYCLES["leveling_season"],
            "confidence": 0.8,
            "action": "optimize_for_exp",
        }
    
    def get_build_recommendation(self, player_class: str, level: int) -> dict[str, Any]:
        """Get build recommendation based on current meta."""
        meta = self.predict_meta_shift()
        meta_details = meta["details"]
        
        is_recommended = player_class in meta_details.get("recommended_classes", [])
        is_avoided = player_class in meta_details.get("avoid_classes", [])
        
        return {
            "meta": meta["meta"],
            "is_recommended": is_recommended,
            "is_avoided": is_avoided,
            "recommended_gear": meta_details.get("recommended_gear", []),
            "action": meta["action"],
            "confidence": meta["confidence"],
        }
    
    def should_adapt_build(self, player_class: str, current_build: str) -> dict[str, Any]:
        """Should the player adapt their build to the current meta?"""
        rec = self.get_build_recommendation(player_class, 1)
        
        if rec["is_avoided"]:
            return {
                "should_adapt": True,
                "reason": f"{player_class} is not meta for {rec['meta']}",
                "suggestion": f"Consider switching to {rec['details']['recommended_classes'][0]}",
                "urgency": "medium",
            }
        
        if rec["is_recommended"]:
            return {
                "should_adapt": False,
                "reason": f"{player_class} is meta for {rec['meta']}",
                "suggestion": f"Optimize gear: {', '.join(rec['recommended_gear'][:3])}",
                "urgency": "low",
            }
        
        return {
            "should_adapt": False,
            "reason": f"{player_class} is neutral for {rec['meta']}",
            "suggestion": "Continue current build",
            "urgency": "none",
        }
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)
