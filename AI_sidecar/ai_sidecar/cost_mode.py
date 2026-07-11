"""
Cost Mode Manager — 3-tier LLM usage control.
==============================================
saver:    heuristic + game engine for 95% of decisions, LLM only for novel situations
standard: heuristic + game engine + occasional LLM for strategic planning
max:      full LLM for all horizons, game engine as fallback

The key insight: a pro farmer doesn't need an LLM to decide what to hunt.
The game engine + hunting zone manager already knows the optimal answer
from rAthena data. The LLM is only needed for truly novel situations.
"""

from __future__ import annotations

import logging
from threading import RLock
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CostMode(Enum):
    SAVER = "saver"
    STANDARD = "standard"
    MAX = "max"


class CostModeManager:
    """Manages LLM usage based on cost mode.

    Each mode defines:
    - Which horizons use LLM vs heuristic vs game engine
    - Confidence thresholds for heuristic skip
    - Whether to call LLM for novel situations
    - Whether to use game engine for hunting recommendations
    """

    def __init__(self, mode: str = "standard"):
        self.mode = self._parse_mode(mode)
        self._lock = RLock()
        self._novel_situation_count: dict[str, int] = {}  # bot_id -> count
        self._last_llm_call: dict[str, float] = {}  # bot_id -> timestamp

    def _parse_mode(self, mode: str) -> CostMode:
        try:
            return CostMode(mode.lower())
        except ValueError:
            logger.warning("Unknown cost mode '%s', falling back to standard", mode)
            return CostMode.STANDARD

    def should_use_llm(
        self,
        horizon: str,
        heuristic_confidence: float,
        bot_id: str = "default",
        is_novel_situation: bool = False,
    ) -> bool:
        """Decide whether to call the LLM for this horizon.

        Returns True if LLM should be called, False if heuristic/game engine is enough.
        """
        if self.mode == CostMode.MAX:
            # Max mode: always use LLM
            return True

        if self.mode == CostMode.SAVER:
            # Saver mode: only use LLM for truly novel situations
            # Heuristic confidence > 0.5 is enough for all horizons
            if heuristic_confidence >= 0.5:
                return False
            # Only use LLM for novel situations (first time seeing a problem)
            if is_novel_situation:
                bot_novel_count = self._novel_situation_count.get(bot_id, 0)
                if bot_novel_count < 3:  # Learn from first 3 novel situations
                    self._novel_situation_count[bot_id] = bot_novel_count + 1
                    return True
            return False

        # Standard mode: use LLM for strategic, heuristic for tactical
        if horizon in ("long_term", "strategic"):
            return True
        if heuristic_confidence >= 0.7:
            return False
        return True

    def should_use_game_engine(self) -> bool:
        """Game engine is always used in all modes."""
        return True

    def should_use_swarm_ai(self) -> bool:
        """Swarm AI is always used in all modes."""
        return True

    def get_heuristic_threshold(self) -> float:
        """Get the heuristic confidence threshold for this mode."""
        if self.mode == CostMode.SAVER:
            return 0.5  # Lower threshold = more heuristic, less LLM
        elif self.mode == CostMode.STANDARD:
            return 0.7
        else:  # MAX
            return 0.9  # Higher threshold = more LLM

    def get_llm_calls_per_hour_limit(self) -> int:
        """Get the max LLM calls per hour for this mode."""
        if self.mode == CostMode.SAVER:
            return 5  # 5 calls/hour max
        elif self.mode == CostMode.STANDARD:
            return 30
        else:  # MAX
            return 100

    def get_daily_budget_tokens(self) -> int:
        """Get the daily token budget for this mode."""
        if self.mode == CostMode.SAVER:
            return 10000  # ~$0.50/day
        elif self.mode == CostMode.STANDARD:
            return 100000  # ~$5/day
        else:  # MAX
            return 1000000  # ~$50/day

    def get_cost_per_hour_estimate(self) -> str:
        """Get estimated cost per hour for this mode."""
        costs = {
            CostMode.SAVER: "$0.01-0.05/hr",
            CostMode.STANDARD: "$0.05-0.30/hr",
            CostMode.MAX: "$0.30-2.00/hr",
        }
        return costs.get(self.mode, "$0.05-0.30/hr")

    def record_novel_situation(self, bot_id: str) -> None:
        """Record that a novel situation was encountered."""
        self._novel_situation_count[bot_id] = self._novel_situation_count.get(bot_id, 0) + 1

    def get_stats(self) -> dict[str, Any]:
        """Get cost mode statistics."""
        return {
            "mode": self.mode.value,
            "heuristic_threshold": self.get_heuristic_threshold(),
            "llm_calls_per_hour_limit": self.get_llm_calls_per_hour_limit(),
            "daily_budget_tokens": self.get_daily_budget_tokens(),
            "cost_per_hour": self.get_cost_per_hour_estimate(),
            "novel_situations": dict(self._novel_situation_count),
        }
