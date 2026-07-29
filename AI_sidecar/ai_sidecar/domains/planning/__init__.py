"""Planning domain — goal hierarchy and task scheduler.

Provides:
  - PlanningDomain: Domain integration with the PDCA assessment loop
  - GoalManager: Short/medium/long-term goal hierarchy
  - TaskScheduler: Priority queue with time-aware and learning-adaptive scheduling
  - Goal: Individual goal with progress tracking
  - ScheduledTask: Prioritized task for execution
"""
from __future__ import annotations

import logging
from typing import Any

from ai_sidecar.actions import HeuristicAction
from ai_sidecar.domains.planning.goals import (
    Goal,
    GoalManager,
    GoalStatus,
    GoalTier,
)
from ai_sidecar.domains.planning.scheduler import (
    ScheduledTask,
    TaskCategory,
    TaskScheduler,
)
from ai_sidecar.domains.planning.build_planner import (
    BuildPlanner,
)
from ai_sidecar.domains.planning.stat_planner import (
    StatBreakpointPlanner,
)
from ai_sidecar.domains.planning.rotation import (
    MapRotationPlanner,
    RotationRecommendation,
    ZoneInfo,
)

logger = logging.getLogger(__name__)


class PlanningDomain:
    """Planning domain — goal-driven task scheduling.

    Reads experience data from the learning domain to inform
    scheduling decisions. Converts goals into actionable tasks
    in priority order.

    Priority: 30 (runs after learning but before combat)
    """

    name = "planning"
    priority = 30

    def __init__(self) -> None:
        self._goal_manager: GoalManager | None = None
        self._scheduler: TaskScheduler | None = None
        self._experience_tracker = None

    def initialize(self) -> None:
        """Set up goal manager and scheduler."""
        self._goal_manager = GoalManager()
        self._scheduler = TaskScheduler(
            goal_manager=self._goal_manager,
            bot_id="default",
        )

    def wire_experience_tracker(self, tracker: Any) -> None:
        """Connect the learning domain's experience tracker.

        This is called after both domains are initialized to avoid
        circular import issues.
        """
        self._experience_tracker = tracker
        if self._scheduler:
            self._scheduler.set_experience_tracker(tracker)

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        bot_id: str,
    ) -> None:
        """Assess current state and schedule tasks.

        Signal keys used:
          - map: current map name
          - inventory_full: bool
          - low_hp: bool
          - near_level_up: bool
          - level: current character level
        """
        bot = bot_id or "default"

        # Ensure initialized
        if not self._scheduler:
            self.initialize()
        if not self._scheduler or not self._goal_manager:
            return

        # Evaluate goal progress
        completed = self._goal_manager.evaluate_goals()
        for goal in completed:
            logger.info(
                "Goal '%s' completed for bot '%s'",
                goal.id, bot,
            )

        # Handle repeats
        self._goal_manager.process_repeats()

        # Get active goals and schedule tasks
        active_goals = self._goal_manager.get_active_goals()
        if active_goals:
            top = active_goals[0]
            # Log current top goal
            actions.append(HeuristicAction(
                kind="log",
                command="",
                confidence=0.9,
                reason=(
                    f"planning: top goal '{top.id}' "
                    f"({top.tier.value}, {top.progress:.0%} complete)"
                ),
                domain="planning",
                metadata={
                    "goal_id": top.id,
                    "goal_tier": top.tier.value,
                    "goal_category": top.category,
                    "goal_progress": top.progress,
                    "active_goal_count": len(active_goals),
                },
            ))

        # Build task schedule from signals
        schedule = self._scheduler.schedule_from_signals(signals)

        # Emit scheduled tasks as actions
        for task in schedule[:5]:  # Top 5 tasks
            task_actions = self._scheduler.execute_task(task)
            for action in task_actions:
                action.metadata["bot_id"] = bot
                actions.append(action)

    # ── Public services ──────────────────────────────────────────────

    def get_goal_manager(self) -> GoalManager | None:
        """Access the goal manager."""
        return self._goal_manager

    def get_scheduler(self) -> TaskScheduler | None:
        """Access the scheduler."""
        return self._scheduler

    def assess_and_recommend(
        self,
        current_map: str = "unknown",
        bot_id: str = "default",
        inventory_full: bool = False,
        low_hp: bool = False,
        near_level_up: bool = False,
        level: int = 1,
    ) -> list[HeuristicAction]:
        """Convenience: run full planning assessment outside PDCA loop.

        Returns HeuristicActions for the planned tasks.
        """
        signals: dict[str, Any] = {
            "map": current_map,
            "inventory_full": inventory_full,
            "low_hp": low_hp,
            "near_level_up": near_level_up,
            "level": level,
        }
        actions: list[HeuristicAction] = []
        self.assess(signals, actions, bot_id)
        return actions

    def counters(self) -> dict[str, int]:
        """Return diagnostic counters."""
        if self._scheduler:
            return self._scheduler.counters()
        return {"tasks_registered": 0, "queue_depth": 0, "history_size": 0}

    def __repr__(self) -> str:
        gm = self._goal_manager
        sched = self._scheduler
        if gm and sched:
            return (
                f"<PlanningDomain: {len(gm.get_active_goals())} active goals, "
                f"{len(sched.get_all_tasks())} tasks>"
            )
        return "<PlanningDomain: uninitialized>"
