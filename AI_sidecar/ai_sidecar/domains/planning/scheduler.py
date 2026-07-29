"""Task scheduler — priority queue with time awareness and adaptation.

Schedules tasks from the goal hierarchy and adapts based on
learned efficiency data from the learning domain.

Priority ordering:
  survival > combat > economy > progression > social

Time-aware rules:
  - If near level-up → grind more
  - If full inventory → sell first
  - Adapts based on learned efficiency data
"""
from __future__ import annotations

import heapq
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from ai_sidecar.actions import HeuristicAction

if TYPE_CHECKING:
    from ai_sidecar.domains.learning.experience import ExperienceTracker
    from ai_sidecar.domains.planning.goals import Goal, GoalManager

logger = logging.getLogger(__name__)


class TaskCategory(Enum):
    """Task priority categories. Lower value = higher priority."""
    SURVIVAL = 0
    COMBAT = 10
    ECONOMY = 20
    PROGRESSION = 30
    SOCIAL = 40
    IDLE = 99


@dataclass(order=True)
class ScheduledTask:
    """A scheduled task with priority ordering.

    Uses negative priority_score so heapq returns highest-priority first.
    """
    priority_score: int = field(compare=True)
    category: TaskCategory = field(compare=False)
    name: str = field(compare=False)
    description: str = field(compare=False)
    execute: Callable[[], list[HeuristicAction]] | None = field(compare=False, default=None)
    commands: list[str] = field(compare=False, default_factory=list)
    goal_id: str = field(compare=False, default="")
    estimated_duration: float = field(compare=False, default=60.0)  # seconds
    metadata: dict[str, Any] = field(compare=False, default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"<Task:{self.name} cat={self.category.name} "
            f"priority={self.priority_score}>"
        )


# ── Default tasks ───────────────────────────────────────────────────

SURVIVAL_TASKS: list[dict[str, Any]] = [
    {
        "name": "emergency_heal",
        "category": TaskCategory.SURVIVAL,
        "description": "Use emergency healing (pot/first aid)",
        "commands": ["use_emergency_heal"],
        "estimated_duration": 2.0,
    },
    {
        "name": "restock_pots",
        "category": TaskCategory.SURVIVAL,
        "description": "Return to town and restock potions",
        "commands": ["return_town", "buy_pots"],
        "estimated_duration": 120.0,
    },
    {
        "name": "repair_equipment",
        "category": TaskCategory.SURVIVAL,
        "description": "Repair broken or damaged equipment",
        "commands": ["return_town", "repair_gear"],
        "estimated_duration": 60.0,
    },
]

COMBAT_TASKS: list[dict[str, Any]] = [
    {
        "name": "hunt_current_map",
        "category": TaskCategory.COMBAT,
        "description": "Hunt monsters on current map",
        "commands": ["attack"],
        "estimated_duration": 600.0,
    },
    {
        "name": "skill_rotation",
        "category": TaskCategory.COMBAT,
        "description": "Execute combat skill rotation",
        "commands": ["skill_auto"],
        "estimated_duration": 30.0,
    },
]

ECONOMY_TASKS: list[dict[str, Any]] = [
    {
        "name": "sell_loot",
        "category": TaskCategory.ECONOMY,
        "description": "Return to town and sell loot to NPC",
        "commands": ["return_town", "sell_loot"],
        "estimated_duration": 180.0,
    },
    {
        "name": "store_items",
        "category": TaskCategory.ECONOMY,
        "description": "Store items in Kafra storage",
        "commands": ["return_town", "open_storage", "deposit_items"],
        "estimated_duration": 120.0,
    },
    {
        "name": "buy_consumables",
        "category": TaskCategory.ECONOMY,
        "description": "Buy arrows/bullets/traps at NPC",
        "commands": ["return_town", "buy_ammo"],
        "estimated_duration": 60.0,
    },
]

PROGRESSION_TASKS: list[dict[str, Any]] = [
    {
        "name": "grind_levels",
        "category": TaskCategory.PROGRESSION,
        "description": "Grind experience for next level",
        "commands": ["attack"],
        "estimated_duration": 1800.0,
    },
    {
        "name": "complete_quest_objective",
        "category": TaskCategory.PROGRESSION,
        "description": "Complete quest objectives",
        "commands": ["quest_auto"],
        "estimated_duration": 600.0,
    },
    {
        "name": "job_change_prep",
        "category": TaskCategory.PROGRESSION,
        "description": "Prepare for job change (items, levels)",
        "commands": ["npc_talk"],
        "estimated_duration": 1200.0,
    },
]

SOCIAL_TASKS: list[dict[str, Any]] = [
    {
        "name": "party_check",
        "category": TaskCategory.SOCIAL,
        "description": "Check party status and invites",
        "commands": ["party"],
        "estimated_duration": 10.0,
    },
    {
        "name": "guild_check",
        "category": TaskCategory.SOCIAL,
        "description": "Check guild messages and notifications",
        "commands": ["guild"],
        "estimated_duration": 15.0,
    },
]


class TaskScheduler:
    """Priority-based task scheduler for bot actions.

    Converts goals into scheduled tasks, ordered by urgency
    (survival > combat > economy > progression > social > idle),
    and adapts based on learned efficiency data.
    """

    def __init__(
        self,
        goal_manager: GoalManager | None = None,
        experience_tracker: ExperienceTracker | None = None,
        bot_id: str = "default",
    ) -> None:
        self.bot_id = bot_id
        self._goal_manager = goal_manager
        self._tracker = experience_tracker
        self._tasks: dict[str, ScheduledTask] = {}
        self._task_queue: list[ScheduledTask] = []  # heap
        self._history: deque[dict[str, Any]] = deque(maxlen=200)
        self._last_task_time: dict[str, float] = {}
        self._initialize_defaults()

    def set_goal_manager(self, gm: GoalManager) -> None:
        """Set the goal manager for task generation."""
        self._goal_manager = gm

    def set_experience_tracker(self, et: ExperienceTracker) -> None:
        """Set the experience tracker for adaptive scheduling."""
        self._tracker = et

    def _initialize_defaults(self) -> None:
        """Register default tasks."""
        for template in (
            SURVIVAL_TASKS + COMBAT_TASKS + ECONOMY_TASKS
            + PROGRESSION_TASKS + SOCIAL_TASKS
        ):
            task = ScheduledTask(
                priority_score=template["category"].value,
                **template,
            )
            self._tasks[task.name] = task

    # ── Task management ─────────────────────────────────────────────

    def register_task(self, task: ScheduledTask) -> None:
        """Register a custom task."""
        self._tasks[task.name] = task

    def remove_task(self, name: str) -> None:
        """Remove a registered task by name."""
        self._tasks.pop(name, None)

    def get_task(self, name: str) -> ScheduledTask | None:
        """Get a registered task by name."""
        return self._tasks.get(name)

    def get_all_tasks(self) -> list[ScheduledTask]:
        """Get all registered tasks."""
        return list(self._tasks.values())

    # ── Scheduling ──────────────────────────────────────────────────

    def schedule(
        self,
        map_name: str,
        inventory_full: bool = False,
        low_hp: bool = False,
        near_level_up: bool = False,
        current_level: int = 1,
        **context: Any,
    ) -> list[ScheduledTask]:
        """Build a prioritized schedule of tasks based on context.

        Args:
            map_name: Current map the bot is on.
            inventory_full: Whether inventory is full.
            low_hp: Whether HP is critically low.
            near_level_up: Whether bot is close to next level.
            current_level: Current character level.
            **context: Additional context flags.

        Returns:
            List of ScheduledTask in priority order (most urgent first).
        """
        self._task_queue = []
        now = time.time()

        # 1. Survival tasks — always check
        if low_hp:
            self._push_task("emergency_heal", 0)

        # Check if restock is due
        last_restock = self._last_task_time.get("restock_pots", 0)
        if now - last_restock > 900:  # 15 min since last restock
            if self._should_restock(map_name):
                self._push_task("restock_pots", 5)

        # Check repair needs
        last_repair = self._last_task_time.get("repair_equipment", 0)
        if now - last_repair > 3600:  # 1 hour since last repair
            self._push_task("repair_equipment", 10)

        # 2. Economy — if inventory full, sell first
        if inventory_full:
            self._push_task("sell_loot", 15)
            self._push_task("store_items", 20)
        else:
            # Check if it's time for a regular loot sell
            last_sell = self._last_task_time.get("sell_loot", 0)
            if now - last_sell > 600:  # 10 min since last sell
                self._push_task("sell_loot", 25)

        # 3. Combat — if not in immediate danger
        if not low_hp:
            # Check if we should grind more aggressively
            if near_level_up:
                self._push_task("grind_levels", 30)
            else:
                self._push_task("hunt_current_map", 35)

        # 4. Progression
        if near_level_up:
            self._push_task("grind_levels", 40)
        else:
            self._push_task("grind_levels", 50)

        # Check for quest goals
        if self._goal_manager:
            top_goal = self._goal_manager.get_top_goal()
            if top_goal and top_goal.category == "progression":
                if "quest" in top_goal.id:
                    self._push_task("complete_quest_objective", 45)
                if "job" in top_goal.id:
                    self._push_task("job_change_prep", 48)

        # 5. Social — lowest priority
        last_party = self._last_task_time.get("party_check", 0)
        if now - last_party > 300:  # 5 min
            self._push_task("party_check", 60)

        last_guild = self._last_task_time.get("guild_check", 0)
        if now - last_guild > 600:  # 10 min
            self._push_task("guild_check", 70)

        # Adapt schedule based on learned data
        self._apply_learning_adaptations(map_name)

        # Build and return the priority-ordered list
        result: list[ScheduledTask] = []
        seen: set[str] = set()
        while self._task_queue:
            task = heapq.heappop(self._task_queue)
            if task.name not in seen:
                seen.add(task.name)
                result.append(task)

        return result

    def _push_task(self, name: str, base_priority: int) -> None:
        """Push a task onto the priority queue."""
        task = self._tasks.get(name)
        if task is None:
            return
        # Create a copy with the given base priority
        scheduled = ScheduledTask(
            priority_score=task.category.value + base_priority,
            category=task.category,
            name=task.name,
            description=task.description,
            execute=task.execute,
            commands=list(task.commands),
            goal_id=task.goal_id,
            estimated_duration=task.estimated_duration,
            metadata=dict(task.metadata),
        )
        heapq.heappush(self._task_queue, scheduled)

    # ── Adaptive scheduling ─────────────────────────────────────────

    def _apply_learning_adaptations(self, map_name: str) -> None:
        """Adjust schedule priorities based on learned efficiency data.

        Uses ExperienceTracker data to fine-tune task priorities:
        - If death rate is high on this map → boost emergency tasks
        - If exp rate is good → favor grind tasks
        """
        if not self._tracker:
            return

        death_rate = self._tracker.get_death_rate_per_hour(map_name, self.bot_id)
        exp_rate = self._tracker.get_exp_rate_per_hour(map_name, self.bot_id)

        # Rebuild heap with adjusted priorities
        adjusted: list[ScheduledTask] = []
        for task in self._task_queue:
            modified = ScheduledTask(
                priority_score=task.priority_score,
                category=task.category,
                name=task.name,
                description=task.description,
                execute=task.execute,
                commands=list(task.commands),
                goal_id=task.goal_id,
                estimated_duration=task.estimated_duration,
                metadata=dict(task.metadata),
            )
            # High death rate → boost survival tasks
            if death_rate > 3.0 and task.category == TaskCategory.SURVIVAL:
                modified.priority_score -= 20
            # Good exp rate → favor combat/grind
            if exp_rate > 5000.0 and task.category in (
                TaskCategory.COMBAT, TaskCategory.PROGRESSION,
            ):
                modified.priority_score -= 10
            adjusted.append(modified)

        # Re-heapify (rebuild from scratch to restore heap invariant)
        self._task_queue = adjusted
        heapq.heapify(self._task_queue)

        if death_rate > 3.0:
            logger.info(
                "Learning adapt: death rate %.1f/h on %s → boosting survival",
                death_rate, map_name,
            )
        if exp_rate > 5000.0:
            logger.info(
                "Learning adapt: exp rate %.0f/h on %s → favoring combat/grind",
                exp_rate, map_name,
            )

    # ── Signal-based scheduling ──────────────────────────────────────

    def schedule_from_signals(
        self,
        signals: dict[str, Any],
    ) -> list[ScheduledTask]:
        """Convenience: extract scheduling context from signals dict.

        Expected signal keys:
          - map: current map name
          - inventory_full: bool
          - low_hp: bool
          - near_level_up: bool
          - level: current character level
        """
        return self.schedule(
            map_name=signals.get("map", "unknown"),
            inventory_full=signals.get("inventory_full", False),
            low_hp=signals.get("low_hp", False),
            near_level_up=signals.get("near_level_up", False),
            current_level=signals.get("level", 1),
        )

    # ── Task execution ──────────────────────────────────────────────

    def execute_task(
        self,
        task: ScheduledTask,
    ) -> list[HeuristicAction]:
        """Execute a task and produce HeuristicActions.

        If the task has an execute callable, it's called directly.
        Otherwise, commands are converted to HeuristicActions.
        """
        self._last_task_time[task.name] = time.time()
        self._history.append({
            "task": task.name,
            "time": time.time(),
            "category": task.category.name,
        })

        if task.execute:
            return task.execute()

        # Convert commands to actions
        actions: list[HeuristicAction] = []
        for cmd in task.commands:
            actions.append(HeuristicAction(
                kind="command",
                command=cmd,
                confidence=0.85,
                reason=f"scheduler: {task.description}",
                domain="planning",
                metadata={
                    "task": task.name,
                    "category": task.category.name,
                    "goal_id": task.goal_id,
                },
            ))
        return actions

    # ── Query ───────────────────────────────────────────────────────

    def last_run(self, task_name: str) -> float:
        """Get the last execution time of a task (0 if never run)."""
        return self._last_task_time.get(task_name, 0.0)

    def get_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent task execution history."""
        return list(self._history)[-limit:]

    def counters(self) -> dict[str, int]:
        """Return diagnostic counters."""
        return {
            "tasks_registered": len(self._tasks),
            "queue_depth": len(self._task_queue),
            "history_size": len(self._history),
        }

    def __repr__(self) -> str:
        return (
            f"<TaskScheduler: {len(self._tasks)} tasks for '{self.bot_id}'>"
        )

    def _should_restock(self, map_name: str) -> bool:
        """Determine if restock is needed based on context.

        Uses learning data if available: if map has high death rate,
        restock more aggressively.
        """
        if not self._tracker:
            return False  # Let signals trigger restock

        death_rate = self._tracker.get_death_rate_per_hour(map_name, self.bot_id)
        # If dying a lot, restock more often
        if death_rate > 5.0:
            return True
        return False
