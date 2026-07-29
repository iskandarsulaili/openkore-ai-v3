"""Learning domain — SQLite-backed experience tracking and strategy adaptation.

Provides:
  - LearningDomain: Domain integration with the PDCA assessment loop
  - ExperienceTracker: SQLite persistence for outcomes (kills, deaths, loot, exp/hour)
  - StrategyAdapter: Weighted scoring for map recommendations
"""
from __future__ import annotations

from typing import Any

from ai_sidecar.actions import HeuristicAction
from ai_sidecar.domains.learning.experience import (
    ExperienceTracker,
    get_experience_tracker,
)
from ai_sidecar.domains.learning.adaptation import (
    StrategyAdapter,
    get_strategy_adapter,
)

# Re-export
__all__ = [
    "LearningDomain",
    "ExperienceTracker",
    "get_experience_tracker",
    "StrategyAdapter",
    "get_strategy_adapter",
]


class LearningDomain:
    """Learning domain — experience tracking and strategy adaptation.

    Collects kill/death/loot/exp data per map and provides
    optimal map recommendations based on historical performance.

    Priority: 20 (runs early — data collection before planning)
    """

    name = "learning"
    priority = 20

    def __init__(self) -> None:
        self._experience = get_experience_tracker()
        self._adapter: StrategyAdapter | None = None

    def initialize(self) -> None:
        """Set up the strategy adapter after tracker is ready."""
        self._adapter = get_strategy_adapter(self._experience)

    def assess(
        self,
        signals: dict,
        actions: list,
        bot_id: str,
    ) -> None:
        """Assess the current state from signals and record experience data.

        Handles signals keys:
          - event: 'kill' | 'death' | 'loot' | 'exp' — record an outcome event
          - map: current map name
          - exp_gained: exp earned in event
          - zeny_gained: zeny/loot value earned
          - force_assess: if True, run adaptation recommendation
        """
        event = signals.get("event", "")
        current_map = signals.get("map", "unknown")
        bot = bot_id or "default"

        if event == "kill":
            self._experience.record_kill(current_map, bot_id=bot)
        elif event == "death":
            self._experience.record_death(current_map, bot_id=bot)
        elif event == "loot":
            zeny = signals.get("zeny_gained", 0)
            self._experience.record_loot(current_map, zeny, bot_id=bot)
        elif event == "exp":
            exp = signals.get("exp_gained", 0)
            self._experience.record_exp(current_map, exp, bot_id=bot)

        # Full assessment requested?
        if signals.get("force_assess") and self._adapter:
            recommendation = self._adapter.recommend_map(
                current_map=current_map,
                bot_id=bot,
            )
            action_kind = "log"
            if recommendation.get("action") == "move":
                action_kind = "command"
            actions.append(HeuristicAction(
                kind=action_kind,
                command=recommendation.get("command", ""),
                confidence=recommendation.get("confidence", 0.5),
                reason=recommendation.get("reason", "learning assessment"),
                domain="learning",
                metadata={"recommendation": recommendation},
            ))

    # ── Public services ──────────────────────────────────────────────

    def get_experience_tracker(self) -> ExperienceTracker:
        """Access the underlying experience tracker."""
        return self._experience

    def get_strategy_adapter(self) -> StrategyAdapter | None:
        """Access the strategy adapter."""
        return self._adapter

    def recommend_map(
        self,
        current_map: str,
        bot_id: str = "default",
    ) -> dict:
        """Get an optimal map recommendation based on learned data."""
        if not self._adapter:
            self._adapter = get_strategy_adapter(self._experience)
        return self._adapter.recommend_map(
            current_map=current_map, bot_id=bot_id,
        )

    def counters(self) -> dict:
        """Return diagnostic counters."""
        stats = self._experience.get_session_stats()
        return {
            "total_kills": stats.get("total_kills", 0),
            "total_deaths": stats.get("total_deaths", 0),
            "maps_tracked": self._experience.get_map_count(),
            "adapter_initialized": self._adapter is not None,
        }

    def __repr__(self) -> str:
        maps = self._experience.get_map_count()
        return f"<LearningDomain: {maps} maps tracked>"
