"""
Risk assessment — scores every decision by risk/reward ratio.

The LLM sees risk scores and decides: "This MVP has 0.8 risk but 0.9 reward — worth it"
Risk model learns from outcomes: "I died attempting this → increase risk score"
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RiskAssessment:
    """Scores decisions by risk/reward ratio. Learns from outcomes."""
    
    _lock: RLock = field(default_factory=RLock)
    _outcome_history: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    _risk_scores: dict[str, float] = field(default_factory=dict)  # action_key -> learned risk
    _stats: dict[str, int] = field(default_factory=lambda: {"assessments": 0, "outcomes": 0})
    
    def assess(self, action_type: str, context: dict[str, Any]) -> dict[str, Any]:
        """Assess risk/reward for an action. Returns risk_score, reward_score, recommendation."""
        with self._lock:
            self._stats["assessments"] += 1
        
        # Base risk factors
        risk = 0.0
        reward = 0.0
        
        hp_pct = float(context.get("hp_pct", 1.0))
        sp_pct = float(context.get("sp_pct", 1.0))
        level = int(context.get("level", 1))
        target_level = int(context.get("target_level", 1))
        zeny = int(context.get("zeny", 0))
        is_mvp = bool(context.get("is_mvp", False))
        has_escape = bool(context.get("has_escape", True))
        player_density = int(context.get("player_density", 0))
        is_woe = bool(context.get("is_woe", False))
        risk_window = str(context.get("risk_window", "low"))
        
        # Risk factors
        if hp_pct < 0.3:
            risk += 0.3
        if sp_pct < 0.2:
            risk += 0.2
        if target_level > level + 20:
            risk += 0.4
        elif target_level > level + 10:
            risk += 0.2
        if is_mvp:
            risk += 0.3
        if not has_escape:
            risk += 0.2
        if player_density > 5:
            risk += 0.1
        if is_woe:
            risk += 0.3
        if risk_window == "high":
            risk += 0.2
        
        # Learned risk modifier
        learned_risk = self._risk_scores.get(action_type, 0.0)
        risk = min(1.0, risk + learned_risk)
        
        # Reward factors
        if is_mvp:
            reward += 0.4
        if target_level <= level + 5:
            reward += 0.3
        if zeny > 10000:
            reward += 0.1
        reward = min(1.0, reward + 0.3)  # Base reward
        
        # Decision
        risk_reward_ratio = reward / max(risk, 0.01)
        
        if risk_reward_ratio > 2.0:
            recommendation = "strongly_recommend"
        elif risk_reward_ratio > 1.0:
            recommendation = "consider"
        elif risk_reward_ratio > 0.5:
            recommendation = "cautious"
        else:
            recommendation = "avoid"
        
        return {
            "risk_score": round(risk, 2),
            "reward_score": round(reward, 2),
            "risk_reward_ratio": round(risk_reward_ratio, 2),
            "recommendation": recommendation,
            "risk_factors": {
                "low_hp": hp_pct < 0.3,
                "low_sp": sp_pct < 0.2,
                "level_gap": target_level > level + 10,
                "is_mvp": is_mvp,
                "no_escape": not has_escape,
                "crowded": player_density > 5,
                "woe_active": is_woe,
                "high_risk_window": risk_window == "high",
            },
        }
    
    def record_outcome(self, action_type: str, succeeded: bool, context: dict[str, Any]) -> None:
        """Record the outcome of an action to adjust risk scores."""
        with self._lock:
            self._stats["outcomes"] += 1
            if action_type not in self._outcome_history:
                self._outcome_history[action_type] = []
            self._outcome_history[action_type].append({
                "succeeded": succeeded,
                "context": context,
                "timestamp": time.time(),
            })
            self._outcome_history[action_type] = self._outcome_history[action_type][-50:]
            
            # Update learned risk score
            recent = self._outcome_history[action_type]
            if len(recent) >= 5:
                failure_rate = sum(1 for o in recent[-10:] if not o["succeeded"]) / max(len(recent[-10:]), 1)
                self._risk_scores[action_type] = failure_rate * 0.5
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)
