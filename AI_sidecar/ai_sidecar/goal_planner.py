"""
Goal-Oriented Planner — sets long-term goals, breaks them into daily tasks,
tracks progress, and adjusts based on results.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class Goal:
    """A long-term goal with progress tracking."""
    name: str
    goal_type: str  # level, zeny, gear, quest, mvp_card, skill
    target_value: int = 0
    current_value: int = 0
    priority: int = 50
    deadline: float = 0.0
    is_complete: bool = False
    is_active: bool = True
    created_at: float = 0.0
    notes: str = ""


@dataclass
class DailyTask:
    """A daily task derived from a goal."""
    goal_name: str
    task_description: str
    estimated_duration_min: int = 30
    priority: int = 50
    is_complete: bool = False
    is_active: bool = True


class GoalPlanner:
    """Sets and tracks long-term goals."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._goals: list[Goal] = []
        self._daily_tasks: list[DailyTask] = []
        self._enqueue_fn: Callable | None = None
        self._load_default_goals()

    def _load_default_goals(self) -> None:
        """Load default goals based on common progression paths."""
        now = time.time()
        self._goals = [
            Goal("Reach Level 99", "level", 99, 1, 90, now + 86400 * 30, notes="Main leveling goal"),
            Goal("Save 10M Zeny", "zeny", 10000000, 0, 80, now + 86400 * 14, notes="For equipment upgrades"),
            Goal("Complete Eden Quests", "quest", 10, 0, 70, now + 86400 * 7, notes="Permanent stat bonuses"),
            Goal("Get +7 Weapon", "gear", 7, 0, 75, now + 86400 * 21, notes="Weapon refinement goal"),
            Goal("Hunt 10 MVPs", "mvp_card", 10, 0, 60, now + 86400 * 30, notes="MVP card hunting"),
        ]

    # ── Public API ──

    def add_goal(self, goal: Goal) -> None:
        with self._lock:
            self._goals.append(goal)

    def update_progress(self, goal_name: str, current_value: int) -> None:
        """Update progress toward a goal."""
        with self._lock:
            for goal in self._goals:
                if goal.name == goal_name and goal.is_active:
                    goal.current_value = current_value
                    if goal.current_value >= goal.target_value:
                        goal.is_complete = True
                        logger.info("goal_completed: %s", goal_name)
                    break

    def get_active_goals(self) -> list[Goal]:
        with self._lock:
            return [g for g in self._goals if g.is_active and not g.is_complete]

    def get_completed_goals(self) -> list[Goal]:
        with self._lock:
            return [g for g in self._goals if g.is_complete]

    def get_next_priority_goal(self) -> Goal | None:
        """Get the highest-priority incomplete goal."""
        with self._lock:
            active = self.get_active_goals()
            if not active:
                return None
            active.sort(key=lambda g: -g.priority)
            return active[0]

    def generate_daily_tasks(self) -> list[DailyTask]:
        """Generate daily tasks from active goals."""
        with self._lock:
            self._daily_tasks.clear()
            for goal in self.get_active_goals()[:3]:
                if goal.goal_type == "level":
                    remaining = goal.target_value - goal.current_value
                    self._daily_tasks.append(DailyTask(
                        goal_name=goal.name,
                        task_description=f"Farm XP: {remaining} levels remaining",
                        estimated_duration_min=120,
                        priority=goal.priority,
                    ))
                elif goal.goal_type == "zeny":
                    remaining = goal.target_value - goal.current_value
                    self._daily_tasks.append(DailyTask(
                        goal_name=goal.name,
                        task_description=f"Farm zeny: {remaining:,}z remaining",
                        estimated_duration_min=60,
                        priority=goal.priority,
                    ))
                elif goal.goal_type == "quest":
                    remaining = goal.target_value - goal.current_value
                    self._daily_tasks.append(DailyTask(
                        goal_name=goal.name,
                        task_description=f"Complete quests: {remaining} remaining",
                        estimated_duration_min=30,
                        priority=goal.priority,
                    ))
            return self._daily_tasks

    def get_planner_summary(self) -> str:
        with self._lock:
            lines = [f"── Goal Planner ──"]
            active = self.get_active_goals()
            completed = self.get_completed_goals()
            lines.append(f"Active goals: {len(active)} | Completed: {len(completed)}")
            for g in active[:5]:
                pct = (g.current_value / g.target_value * 100) if g.target_value > 0 else 0
                lines.append(f"  {g.name}: {g.current_value}/{g.target_value} ({pct:.0f}%)")
            tasks = self.generate_daily_tasks()
            if tasks:
                lines.append(f"Today's tasks:")
                for t in tasks[:3]:
                    lines.append(f"  {t.task_description} (~{t.estimated_duration_min}min)")
            return "\n".join(lines)

    def set_enqueue_fn(self, fn: Callable) -> None:
        with self._lock:
            self._enqueue_fn = fn

    def reset(self) -> None:
        with self._lock:
            self._goals.clear()
            self._daily_tasks.clear()
            self._load_default_goals()


# ── Global Singleton ──

_goal_planner: GoalPlanner | None = None
_goal_planner_lock = RLock()


def get_goal_planner() -> GoalPlanner:
    global _goal_planner
    with _goal_planner_lock:
        if _goal_planner is None:
            _goal_planner = GoalPlanner()
        return _goal_planner
