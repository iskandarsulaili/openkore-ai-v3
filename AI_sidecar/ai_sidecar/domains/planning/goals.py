"""Goal hierarchy — short-term, medium-term, and long-term bot goals.

Each goal has a priority, a repeat interval, and conditions for
when it is satisfied. The scheduler uses these to determine what
the bot should be doing at any point in time.

Goal hierarchy:
  - Short-term (next 5 min): finish current hunt, sell loot, restock pots
  - Medium-term (next hour): reach level target, complete quest, save for gear
  - Long-term (next day): job change, reach level bracket, farm expensive gear
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class GoalTier(Enum):
    """Goal time horizon tiers."""
    SHORT_TERM = "short_term"      # next 5 minutes
    MEDIUM_TERM = "medium_term"    # next hour
    LONG_TERM = "long_term"        # next day+


class GoalStatus(Enum):
    """Status of a goal in its lifecycle."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Priority order: lower number = higher urgency
GOAL_PRIORITY_ORDER: dict[str, int] = {
    "survival": 0,
    "economy": 10,
    "progression": 20,
    "combat": 30,
    "social": 40,
}


@dataclass
class Goal:
    """A single goal in the hierarchy.

    Attributes:
        id: Unique identifier for the goal.
        tier: Time horizon (short/medium/long).
        category: Priority category (survival, economy, combat, etc.).
        description: Human-readable goal description.
        condition: Optional callable that returns True when goal is met.
        target_value: Numeric target (e.g., level 50, 100k zeny).
        current_value: Current progress toward target.
        priority: Urgency within its tier (lower = more urgent).
        status: Current lifecycle status.
        created_at: Timestamp the goal was created.
        expires_at: Optional timestamp when the goal auto-expires.
        repeat_interval: Seconds after completion before re-activating (0 = no repeat).
        metadata: Arbitrary additional data.
    """
    id: str
    tier: GoalTier = GoalTier.SHORT_TERM
    category: str = "survival"
    description: str = ""
    condition: Callable[[], bool] | None = None
    target_value: float = 0.0
    current_value: float = 0.0
    priority: int = 50
    status: GoalStatus = GoalStatus.PENDING
    created_at: float = 0.0
    expires_at: float | None = None
    repeat_interval: float = 0.0  # seconds
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = time.time()

    @property
    def progress(self) -> float:
        """Progress toward goal as 0.0-1.0."""
        if self.target_value <= 0:
            return 1.0 if self.status == GoalStatus.COMPLETED else 0.0
        return min(1.0, self.current_value / self.target_value)

    @property
    def is_expired(self) -> bool:
        """Check if the goal has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    @property
    def tier_priority(self) -> int:
        """Combined priority: tier rank * 100 + category rank + priority.
        Lower = more urgent.
        """
        tier_rank = {
            GoalTier.SHORT_TERM: 0,
            GoalTier.MEDIUM_TERM: 1,
            GoalTier.LONG_TERM: 2,
        }.get(self.tier, 3)
        cat_order = GOAL_PRIORITY_ORDER.get(self.category, 50)
        return tier_rank * 100 + cat_order * 10 + self.priority

    def check_completed(self) -> bool:
        """Evaluate the condition if set, or compare current vs target."""
        if self.status in (GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.CANCELLED):
            return self.status == GoalStatus.COMPLETED
        if self.is_expired:
            self.status = GoalStatus.FAILED
            return False
        if self.condition is not None:
            if self.condition():
                self.status = GoalStatus.COMPLETED
                return True
            return False
        if self.target_value > 0 and self.current_value >= self.target_value:
            self.status = GoalStatus.COMPLETED
            return True
        return False

    def reset_for_repeat(self) -> None:
        """Reset the goal for another cycle if repeatable."""
        if self.repeat_interval > 0 and self.status == GoalStatus.COMPLETED:
            self.status = GoalStatus.PENDING
            self.current_value = 0.0
            self.created_at = time.time()
            if self.expires_at:
                self.expires_at = time.time() + self.repeat_interval

    def __repr__(self) -> str:
        return (
            f"<Goal:{self.id} tier={self.tier.value} "
            f"cat={self.category} status={self.status.value}>"
        )


# ── Goal factory functions ─────────────────────────────────────────

SHORT_TERM_GOALS: list[dict[str, Any]] = [
    {
        "id": "finish_current_hunt",
        "tier": GoalTier.SHORT_TERM,
        "category": "combat",
        "description": "Finish current hunt session on map",
        "priority": 30,
        "repeat_interval": 300.0,  # 5 min
    },
    {
        "id": "sell_loot",
        "tier": GoalTier.SHORT_TERM,
        "category": "economy",
        "description": "Sell accumulated loot to NPC",
        "priority": 20,
        "repeat_interval": 600.0,  # 10 min
    },
    {
        "id": "restock_pots",
        "tier": GoalTier.SHORT_TERM,
        "category": "survival",
        "description": "Restock HP/SP potions",
        "priority": 10,
        "repeat_interval": 900.0,  # 15 min
    },
    {
        "id": "check_equipment",
        "tier": GoalTier.SHORT_TERM,
        "category": "survival",
        "description": "Check and repair equipment durability",
        "priority": 5,
        "repeat_interval": 1800.0,  # 30 min
    },
]

MEDIUM_TERM_GOALS: list[dict[str, Any]] = [
    {
        "id": "reach_level_target",
        "tier": GoalTier.MEDIUM_TERM,
        "category": "progression",
        "description": "Reach next level target",
        "priority": 10,
        "metadata": {"target_level": 0},
    },
    {
        "id": "complete_quest",
        "tier": GoalTier.MEDIUM_TERM,
        "category": "progression",
        "description": "Complete current active quest",
        "priority": 20,
    },
    {
        "id": "save_for_gear",
        "tier": GoalTier.MEDIUM_TERM,
        "category": "economy",
        "description": "Save zeny for next gear upgrade",
        "priority": 15,
        "metadata": {"target_zeny": 0, "item_name": ""},
    },
    {
        "id": "farm_materials",
        "tier": GoalTier.MEDIUM_TERM,
        "category": "economy",
        "description": "Farm crafting materials or consumables",
        "priority": 25,
    },
]

LONG_TERM_GOALS: list[dict[str, Any]] = [
    {
        "id": "job_change",
        "tier": GoalTier.LONG_TERM,
        "category": "progression",
        "description": "Complete job change quest",
        "priority": 5,
    },
    {
        "id": "reach_level_bracket",
        "tier": GoalTier.LONG_TERM,
        "category": "progression",
        "description": "Reach next level bracket (e.g., 50->70)",
        "priority": 10,
        "metadata": {"target_bracket": 0},
    },
    {
        "id": "farm_expensive_gear",
        "tier": GoalTier.LONG_TERM,
        "category": "economy",
        "description": "Farm zeny for expensive gear item",
        "priority": 15,
        "metadata": {"target_zeny": 0, "item_name": ""},
    },
    {
        "id": "max_skills",
        "tier": GoalTier.LONG_TERM,
        "category": "progression",
        "description": "Max out key job skills",
        "priority": 20,
    },
]


# ── GoalManager ─────────────────────────────────────────────────────

class GoalManager:
    """Manages the goal hierarchy for a bot.

    Maintains goals at three tiers, tracks progress, and provides
    the scheduler with the most urgent active goals.
    """

    def __init__(self, bot_id: str = "default") -> None:
        self.bot_id = bot_id
        self._goals: list[Goal] = []
        self._initialize_defaults()

    def _initialize_defaults(self) -> None:
        """Populate default goals from templates."""
        now = time.time()
        for template in SHORT_TERM_GOALS + MEDIUM_TERM_GOALS + LONG_TERM_GOALS:
            goal = Goal(
                **template,
                status=GoalStatus.PENDING,
                created_at=now,
            )
            self._goals.append(goal)

    # ── Goal lifecycle ──────────────────────────────────────────────

    def add_goal(self, goal: Goal) -> None:
        """Add a custom goal."""
        self._goals.append(goal)

    def remove_goal(self, goal_id: str) -> None:
        """Remove a goal by ID."""
        self._goals = [g for g in self._goals if g.id != goal_id]

    def get_goal(self, goal_id: str) -> Goal | None:
        """Get a goal by ID."""
        for g in self._goals:
            if g.id == goal_id:
                return g
        return None

    def update_progress(self, goal_id: str, value: float) -> None:
        """Update current progress toward a goal."""
        goal = self.get_goal(goal_id)
        if goal:
            goal.current_value = value

    def set_goal_target(self, goal_id: str, target: float) -> None:
        """Set a new target value for a goal."""
        goal = self.get_goal(goal_id)
        if goal:
            goal.target_value = target

    def mark_completed(self, goal_id: str) -> None:
        """Mark a goal as completed."""
        goal = self.get_goal(goal_id)
        if goal:
            goal.status = GoalStatus.COMPLETED

    def mark_failed(self, goal_id: str) -> None:
        """Mark a goal as failed."""
        goal = self.get_goal(goal_id)
        if goal:
            goal.status = GoalStatus.FAILED

    # ── Query ───────────────────────────────────────────────────────

    def get_active_goals(
        self,
        tier: GoalTier | None = None,
    ) -> list[Goal]:
        """Get all active (pending or active status) goals, optionally filtered by tier.

        Results sorted by tier_priority (most urgent first).
        """
        active = []
        for g in self._goals:
            if g.status in (GoalStatus.PENDING, GoalStatus.ACTIVE):
                if tier is None or g.tier == tier:
                    active.append(g)
        active.sort(key=lambda g: g.tier_priority)
        return active

    def get_top_goal(self, tier: GoalTier | None = None) -> Goal | None:
        """Get the single most urgent active goal."""
        active = self.get_active_goals(tier)
        return active[0] if active else None

    def get_goals_by_status(self, status: GoalStatus) -> list[Goal]:
        """Get all goals with a specific status."""
        return [g for g in self._goals if g.status == status]

    def evaluate_goals(self) -> list[Goal]:
        """Check all pending goals for completion. Returns newly completed goals."""
        completed: list[Goal] = []
        for g in self._goals:
            if g.status == GoalStatus.PENDING and g.check_completed():
                completed.append(g)
                logger.info(
                    "Goal '%s' completed for bot '%s'",
                    g.id, self.bot_id,
                )
        return completed

    def process_repeats(self) -> None:
        """Reset completed repeatable goals for next cycle."""
        for g in self._goals:
            if g.status == GoalStatus.COMPLETED and g.repeat_interval > 0:
                was_old = time.time() - g.created_at
                if was_old >= g.repeat_interval:
                    g.reset_for_repeat()
                    logger.debug(
                        "Goal '%s' reset for repeat (interval=%ss)",
                        g.id, g.repeat_interval,
                    )

    def get_tier_summary(self) -> dict[str, Any]:
        """Get a summary of goals per tier."""
        tiers: dict[str, dict[str, int]] = {}
        for tier in GoalTier:
            goals = [g for g in self._goals if g.tier == tier]
            total = len(goals)
            completed_count = sum(1 for g in goals if g.status == GoalStatus.COMPLETED)
            active_count = sum(1 for g in goals if g.status in (
                GoalStatus.PENDING, GoalStatus.ACTIVE,
            ))
            tiers[tier.value] = {
                "total": total,
                "completed": completed_count,
                "active": active_count,
            }
        return tiers

    def to_dict(self) -> list[dict[str, Any]]:
        """Serialize all goals to dicts."""
        return [
            {
                "id": g.id,
                "tier": g.tier.value,
                "category": g.category,
                "description": g.description,
                "progress": g.progress,
                "status": g.status.value,
                "target_value": g.target_value,
                "current_value": g.current_value,
            }
            for g in self._goals
        ]

    def __repr__(self) -> str:
        active = self.get_active_goals()
        return f"<GoalManager: {len(active)} active goals for '{self.bot_id}'>"
