"""
Opportunity Cost Engine — compares options across multiple dimensions,
calculates long-term value, and makes optimal trade-offs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class DecisionOption:
    """An option to compare."""
    name: str
    zeny_per_hour: float = 0.0
    xp_per_hour: float = 0.0
    duration_hours: float = 1.0
    risk_level: int = 1  # 1-10
    permanent_reward_value: int = 0  # One-time value of permanent rewards
    one_time_reward: int = 0
    total_zeny: float = 0.0
    total_xp: float = 0.0
    score: float = 0.0


class OpportunityCostEngine:
    """Compares options and recommends the best one."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._enqueue_fn: Callable | None = None

    # ── Public API ──

    def compare(self, options: list[DecisionOption], weight_zeny: float = 1.0,
                weight_xp: float = 0.5, weight_risk: float = -2.0,
                weight_permanent: float = 3.0) -> DecisionOption | None:
        """Compare options and return the best one."""
        with self._lock:
            if not options:
                return None

            for opt in options:
                # Base value: zeny + XP (converted to zeny-equivalent)
                base_value = opt.zeny_per_hour * opt.duration_hours
                xp_value = opt.xp_per_hour * opt.duration_hours * 0.01  # 100 XP ≈ 1 zeny
                permanent_value = opt.permanent_reward_value / max(opt.duration_hours, 1)
                one_time = opt.one_time_reward / max(opt.duration_hours, 1)

                # Risk penalty
                risk_penalty = opt.risk_level * 1000 * opt.duration_hours

                # Total score
                opt.total_zeny = base_value + one_time + permanent_value
                opt.total_xp = opt.xp_per_hour * opt.duration_hours
                opt.score = (
                    base_value * weight_zeny +
                    xp_value * weight_xp +
                    permanent_value * weight_permanent +
                    one_time * weight_zeny -
                    risk_penalty * abs(weight_risk)
                )

            options.sort(key=lambda o: -o.score)
            return options[0]

    def compare_farming_vs_quest(self, farming_zeny_per_hour: float,
                                  quest_duration_hours: float,
                                  quest_permanent_reward_value: int,
                                  quest_one_time_reward: int = 0,
                                  quest_xp: float = 0) -> str:
        """Compare farming vs doing a quest."""
        farming = DecisionOption(
            name="Farming",
            zeny_per_hour=farming_zeny_per_hour,
            duration_hours=quest_duration_hours,
            risk_level=2,
        )
        questing = DecisionOption(
            name="Quest",
            zeny_per_hour=0,
            xp_per_hour=quest_xp / quest_duration_hours if quest_duration_hours > 0 else 0,
            duration_hours=quest_duration_hours,
            risk_level=1,
            permanent_reward_value=quest_permanent_reward_value,
            one_time_reward=quest_one_time_reward,
        )

        best = self.compare([farming, questing])
        if best:
            return (
                f"Recommend: {best.name} (score={best.score:.0f})\n"
                f"  Farming: {farming.score:.0f} ({farming_zeny_per_hour:,.0f}z/hr x {quest_duration_hours:.1f}h)\n"
                f"  Quest:   {questing.score:.0f} (permanent={quest_permanent_reward_value:,}z, one-time={quest_one_time_reward:,}z)"
            )
        return "No decision possible"

    def compare_maps(self, current_map: str, current_zeny_per_hour: float,
                     alternative_map: str, alternative_zeny_per_hour: float,
                     travel_time_min: int = 5) -> str:
        """Compare two farming maps."""
        travel_cost = (travel_time_min / 60.0) * current_zeny_per_hour
        current = DecisionOption(
            name=current_map,
            zeny_per_hour=current_zeny_per_hour,
            duration_hours=1,
            risk_level=3,
        )
        alternative = DecisionOption(
            name=alternative_map,
            zeny_per_hour=alternative_zeny_per_hour,
            duration_hours=1 - (travel_time_min / 60.0),
            risk_level=3,
        )

        best = self.compare([current, alternative])
        if best:
            return (
                f"Recommend: {best.name}\n"
                f"  {current_map}: {current.zeny_per_hour:,.0f}z/hr\n"
                f"  {alternative_map}: {alternative.zeny_per_hour:,.0f}z/hr (travel cost: {travel_cost:,.0f}z)"
            )
        return "No decision possible"

    def get_opportunity_summary(self) -> str:
        return "── Opportunity Cost Engine ──\nReady to compare options"

    def set_enqueue_fn(self, fn: Callable) -> None:
        with self._lock:
            self._enqueue_fn = fn

    def reset(self) -> None:
        pass


# ── Global Singleton ──

_opp_cost: OpportunityCostEngine | None = None
_opp_cost_lock = RLock()


def get_opportunity_cost_engine() -> OpportunityCostEngine:
    global _opp_cost
    with _opp_cost_lock:
        if _opp_cost is None:
            _opp_cost = OpportunityCostEngine()
        return _opp_cost
