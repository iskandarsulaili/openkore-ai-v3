"""Risk Manager — data-driven risk/reward assessment for bot decisions.

Architecture:
  - Wraps existing RiskAssessment class with per-map survival/kill statistics
  - Collects outcome data: kills, deaths, survival time per map
  - Assesses risk/reward for routing decisions using historical data
  - Provides dynamic map recommendations based on reward/punish learning
  - Follows RULE.md Rule 6: Reward/Punish System

Data flow:
  record_outcome(map, kill/death/levelup) → update stats → assess(map) → recommend
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MapStats:
    """Per-map survival and kill statistics."""
    visits: int = 0
    kills: int = 0
    deaths: int = 0
    level_ups: int = 0
    total_survival_seconds: float = 0.0
    last_visit: float = 0.0
    last_kill: float = 0.0
    last_death: float = 0.0

    @property
    def survival_rate(self) -> float:
        """Probability of surviving a visit (no death)."""
        if self.visits == 0:
            return 0.0
        return 1.0 - (self.deaths / max(self.visits, 1))

    @property
    def kills_per_death(self) -> float:
        """Kill/death ratio. Higher = better."""
        if self.deaths == 0:
            return float(self.kills * 10)  # No deaths = great
        return self.kills / self.deaths

    @property
    def score(self) -> float:
        """Overall map score: kills - deaths * 2 + level_ups * 5."""
        return self.kills - (self.deaths * 2) + (self.level_ups * 5)


class RiskManager:
    """Data-driven risk/reward manager for bot decisions."""

    def __init__(self):
        self._lock = RLock()
        self._map_stats: dict[str, MapStats] = defaultdict(MapStats)
        self._action_history: list[dict] = []
        self._assessment = None  # Lazy-loaded RiskAssessment
    
    def _get_assessment(self):
        """Lazy-load the RiskAssessment instance."""
        if self._assessment is None:
            try:
                from ai_sidecar.risk_assessment import RiskAssessment
                self._assessment = RiskAssessment()
            except Exception as e:
                logger.warning("risk_manager: RiskAssessment init failed: %s", e)
                self._assessment = object()  # dummy
        return self._assessment
    
    def record_visit(self, bot_id: str, map_name: str):
        """Record that a bot visited a map."""
        with self._lock:
            now = time.time()
            stats = self._map_stats[map_name]
            stats.visits += 1
            stats.last_visit = now
    
    def record_kill(self, bot_id: str, map_name: str, xp_gained: int = 0):
        """Record a kill on a map. REWARD signal."""
        with self._lock:
            now = time.time()
            stats = self._map_stats[map_name]
            stats.kills += 1
            stats.last_kill = now
        logger.info("risk_manager: reward bot=%s map=%s (total kills=%d)", bot_id, map_name, stats.kills)
    
    def record_death(self, bot_id: str, map_name: str):
        """Record a death on a map. PUNISH signal."""
        with self._lock:
            now = time.time()
            stats = self._map_stats[map_name]
            stats.deaths += 1
            stats.last_death = now
        logger.info("risk_manager: punish bot=%s map=%s (total deaths=%d)", bot_id, map_name, stats.deaths)
    
    def record_level_up(self, bot_id: str, map_name: str):
        """Record a level up on a map. STRONG REWARD signal."""
        with self._lock:
            stats = self._map_stats[map_name]
            stats.level_ups += 1
        logger.info("risk_manager: reward bot=%s map=%s level_up (total=%d)", bot_id, map_name, stats.level_ups)
    
    def assess_map(self, map_name: str, context: Optional[dict] = None) -> dict:
        """Assess risk/reward for a specific map.
        
        Returns:
            risk_score: 0.0 (safe) to 1.0 (deadly)
            reward_score: 0.0 (nothing) to 1.0 (great)
            kills_per_death: ratio
            visits: how many times bots have been there
        """
        stats = self._map_stats[map_name]
        with self._lock:
            visits = stats.visits
            kills = stats.kills
            deaths = stats.deaths
            kdr = stats.kills_per_death
            survival = stats.survival_rate
        
        # Risk: based on death rate
        if visits == 0:
            risk_score = 0.5  # Unknown = medium risk
        elif deaths == 0:
            risk_score = 0.1  # No deaths = low risk
        else:
            death_rate = deaths / max(visits, 1)
            risk_score = min(1.0, death_rate * 2.0)  # Scale: 50% death rate = 1.0
        
        # Reward: based on kill/death ratio
        if visits == 0:
            reward_score = 0.3  # Unknown = medium reward
        elif kills > 0 and deaths == 0:
            reward_score = 1.0  # Killed without dying = max reward
        elif kills > 0:
            reward_score = min(1.0, kdr * 0.3)  # Scale: 3.0 KDR = 0.9 reward
        else:
            reward_score = 0.1  # No kills = minimal reward
        
        # Level-ups boost reward significantly
        level_bonus = min(0.5, stats.level_ups * 0.2)
        reward_score = min(1.0, reward_score + level_bonus)
        
        rr_ratio = reward_score / max(risk_score, 0.01)
        
        if rr_ratio > 2.0:
            recommendation = "strongly_recommend"
        elif rr_ratio > 1.0:
            recommendation = "recommend"
        elif rr_ratio > 0.5:
            recommendation = "cautious"
        else:
            recommendation = "avoid"
        
        return {
            "map": map_name,
            "risk_score": round(risk_score, 2),
            "reward_score": round(reward_score, 2),
            "kills_per_death": round(kdr, 2),
            "visits": visits,
            "kills": kills,
            "deaths": deaths,
            "level_ups": stats.level_ups,
            "survival_rate": round(survival, 2),
            "rr_ratio": round(rr_ratio, 2),
            "recommendation": recommendation,
        }
    
    def best_map(self, candidates: list[str], context: Optional[dict] = None) -> Optional[str]:
        """Pick the best map from candidates based on risk/reward.
        
        Uses reward/punish data to choose the map with the best outcome history.
        Falls back to the first candidate if no data available.
        """
        if not candidates:
            return None
        
        best = None
        best_score = float('-inf')
        
        for map_name in candidates:
            assessment = self.assess_map(map_name, context)
            # Weight: reward - risk*2 (penalize risky maps heavily)
            weighted = assessment["reward_score"] - (assessment["risk_score"] * 2.0)
            
            if weighted > best_score:
                best_score = weighted
                best = map_name
        
        return best
    
    def worst_map(self, candidates: list[str]) -> Optional[str]:
        """Pick the worst map (highest risk, lowest reward) — for exclusion."""
        if not candidates:
            return None
        
        worst = None
        worst_score = float('inf')
        
        for map_name in candidates:
            assessment = self.assess_map(map_name)
            weighted = assessment["risk_score"] - assessment["reward_score"]
            
            if weighted > worst_score:
                worst_score = weighted
                worst = map_name
        
        return worst
    
    def summary(self) -> dict:
        """Get summary of all tracked map stats."""
        with self._lock:
            return {map_name: {
                "visits": s.visits,
                "kills": s.kills,
                "deaths": s.deaths,
                "kdr": s.kills_per_death,
                "score": s.score,
            } for map_name, s in self._map_stats.items()}


# Global singleton
_manager: Optional[RiskManager] = None


def get_risk_manager() -> RiskManager:
    """Get global RiskManager instance."""
    global _manager
    if _manager is None:
        _manager = RiskManager()
    return _manager
