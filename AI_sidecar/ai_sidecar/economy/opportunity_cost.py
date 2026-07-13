"""
Opportunity cost engine — calculates the true cost of every decision.

A top player doesn't just ask "can I survive this fight?" They ask
"is this fight worth my time?" They calculate zeny per hour, experience
per hour, risk of death, risk of ban, and opportunity cost of not doing
something else.

This module integrates with the economy optimizer, risk assessment,
and timing awareness to provide a unified cost-benefit analysis.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Opportunity:
    """A potential activity with calculated value."""
    activity: str
    map_name: str = ""
    estimated_zeny_per_hour: float = 0.0
    estimated_xp_per_hour: float = 0.0
    risk_score: float = 0.0  # 0.0 (safe) to 1.0 (deadly)
    ban_risk: float = 0.0  # 0.0 (safe) to 1.0 (certain ban)
    travel_time_minutes: float = 0.0
    setup_time_minutes: float = 0.0
    competition_level: float = 0.0  # 0.0 (empty) to 1.0 (crowded)
    value_score: float = 0.0  # Computed


@dataclass(slots=True)
class OpportunityCostEngine:
    """Calculates the true cost of every decision."""
    
    _lock: RLock = field(default_factory=RLock)
    _opportunities: dict[str, Opportunity] = field(default_factory=dict)
    _history: list[dict[str, Any]] = field(default_factory=list)
    _stats: dict[str, int] = field(default_factory=lambda: {"evaluations": 0, "recommendations": 0})
    
    def evaluate_opportunity(
        self,
        activity: str,
        *,
        map_name: str = "",
        estimated_zeny_per_hour: float = 0.0,
        estimated_xp_per_hour: float = 0.0,
        risk_score: float = 0.0,
        ban_risk: float = 0.0,
        travel_time_minutes: float = 0.0,
        setup_time_minutes: float = 0.0,
        competition_level: float = 0.0,
    ) -> Opportunity:
        """Evaluate an opportunity and compute its value score."""
        with self._lock:
            self._stats["evaluations"] += 1
        
        # Base value: zeny + XP (converted to zeny equivalent)
        xp_value = estimated_xp_per_hour * 0.01  # 1 XP = 0.01 zeny (rough)
        base_value = estimated_zeny_per_hour + xp_value
        
        # Time penalty: travel + setup
        total_overhead = travel_time_minutes + setup_time_minutes
        time_penalty = total_overhead * (base_value / 60) * 0.5  # 50% of hourly rate during overhead
        
        # Risk penalty
        risk_penalty = base_value * risk_score * 2.0  # Double penalty for high risk
        ban_penalty = base_value * ban_risk * 10.0  # 10x penalty for ban risk
        
        # Competition penalty
        competition_penalty = base_value * competition_level * 0.3
        
        # Final value score
        value_score = base_value - time_penalty - risk_penalty - ban_penalty - competition_penalty
        
        opp = Opportunity(
            activity=activity,
            map_name=map_name,
            estimated_zeny_per_hour=estimated_zeny_per_hour,
            estimated_xp_per_hour=estimated_xp_per_hour,
            risk_score=risk_score,
            ban_risk=ban_risk,
            travel_time_minutes=travel_time_minutes,
            setup_time_minutes=setup_time_minutes,
            competition_level=competition_level,
            value_score=max(0, value_score),
        )
        
        with self._lock:
            self._opportunities[activity] = opp
            self._stats["recommendations"] += 1
        
        return opp
    
    def get_best_opportunity(self, min_value: float = 0.0) -> Opportunity | None:
        """Get the highest-value opportunity."""
        with self._lock:
            valid = [o for o in self._opportunities.values() if o.value_score >= min_value]
            if not valid:
                return None
            return max(valid, key=lambda o: o.value_score)
    
    def record_outcome(self, activity: str, actual_zeny: float, actual_xp: float, died: bool = False) -> None:
        """Record the actual outcome of an activity for learning."""
        with self._lock:
            self._history.append({
                "activity": activity,
                "actual_zeny": actual_zeny,
                "actual_xp": actual_xp,
                "died": died,
                "timestamp": time.time(),
            })
    
    def get_opportunity_context(self) -> str:
        """Get formatted opportunity context for LLM prompts."""
        with self._lock:
            lines = ["── Opportunity Cost Analysis ──"]
            
            opportunities = sorted(
                self._opportunities.values(),
                key=lambda o: -o.value_score
            )[:5]
            
            for opp in opportunities:
                lines.append(
                    f"  {opp.activity} ({opp.map_name}): "
                    f"value={opp.value_score:.0f} "
                    f"zeny/hr={opp.estimated_zeny_per_hour:.0f} "
                    f"risk={opp.risk_score:.1f} "
                    f"ban={opp.ban_risk:.1f}"
                )
            
            return "\n".join(lines)
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# ── Global instance ──

_opportunity: OpportunityCostEngine | None = None
_opportunity_lock = RLock()


def get_opportunity() -> OpportunityCostEngine:
    global _opportunity
    with _opportunity_lock:
        if _opportunity is None:
            _opportunity = OpportunityCostEngine()
        return _opportunity
