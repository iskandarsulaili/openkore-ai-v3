"""
Long-Term Planner — Complete strategy system for the bot fleet.

A pro player has:
- Multi-hour/multi-day planning horizon
- Opportunity cost awareness (value per hour of each activity)
- Goal decomposition (ambition -> weekly goals -> daily tasks)
- Server meta adaptation (configurable per-server mechanics)
- Activity scheduling (farm during off-peak, social during peak)
- Resource allocation (which bot does what, when)

This engine wires into:
- ambition_engine.py (existing ambition system)
- market_engine.py (economic intelligence)
- social_engine.py (social intelligence)
- opportunity_cost.py (value per hour)
"""

from __future__ import annotations

import logging
import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────────────────────


@dataclass
class WeeklyGoal:
    """A weekly goal derived from an ambition."""
    name: str
    ambition_name: str
    category: str  # wealth, pvp, economy, social, collection, mastery
    target_value: float
    current_value: float = 0.0
    deadline: float = 0.0  # timestamp
    status: str = "active"  # active, completed, failed
    daily_tasks: list[str] = field(default_factory=list)
    priority: int = 5  # 1-10


@dataclass
class DailyTask:
    """A daily task derived from a weekly goal."""
    name: str
    weekly_goal: str
    estimated_duration_minutes: int = 60
    estimated_value: float = 0.0
    best_time: str = "any"  # any, peak, off_peak, morning, evening
    status: str = "pending"  # pending, in_progress, completed, skipped
    priority: int = 5
    assigned_bot: str = ""
    created_at: float = 0.0
    completed_at: float = 0.0


@dataclass
class ActivitySchedule:
    """A scheduled activity for a specific time slot."""
    activity: str  # farm, social, rest, shop, craft, quest
    map_name: str = ""
    duration_minutes: int = 60
    value_per_hour: float = 0.0
    best_hour_start: int = 0  # 0-23
    best_hour_end: int = 23
    priority: int = 5
    assigned_bot: str = ""


@dataclass
class ServerMetaConfig:
    """Per-server configuration for meta adaptation."""
    server_name: str = "default"
    exp_rate: float = 1.0
    drop_rate: float = 1.0
    card_rate: float = 1.0
    zeny_rate: float = 1.0
    max_level: int = 99
    has_rebirth: bool = False
    has_transcendent: bool = False
    has_third_class: bool = False
    has_fourth_class: bool = False
    woe_schedule: list[str] = field(default_factory=lambda: ["wed", "sat", "sun"])
    woe_hours: list[int] = field(default_factory=lambda: [20, 21])
    popular_classes: list[str] = field(default_factory=list)
    popular_maps: list[str] = field(default_factory=list)
    economy_type: str = "free"  # free, controlled, stagnant
    pvp_activity: str = "low"  # none, low, medium, high


@dataclass(slots=True)
class LongTermPlanner:
    """
    Complete long-term strategy engine.

    Decomposes ambitions into weekly goals and daily tasks,
    schedules activities optimally, and adapts to server meta.
    """

    _lock: RLock = field(default_factory=RLock)
    _weekly_goals: list[WeeklyGoal] = field(default_factory=list)
    _daily_tasks: list[DailyTask] = field(default_factory=list)
    _activity_schedule: list[ActivitySchedule] = field(default_factory=list)
    _server_meta: ServerMetaConfig = field(default_factory=ServerMetaConfig)
    _completed_tasks: deque = field(default_factory=lambda: deque(maxlen=100))
    _stats: dict[str, int] = field(default_factory=lambda: {
        "goals_set": 0, "tasks_created": 0, "tasks_completed": 0,
        "schedules_generated": 0, "meta_adaptations": 0,
    })
    _ambition_engine: Any = None  # AmbitionEngine instance
    _market_engine: Any = None  # MarketEngine instance
    _social_engine: Any = None  # SocialEngine instance
    _opportunity_cost: Any = None  # OpportunityCostEngine instance
    _last_plan_time: float = 0.0
    _last_schedule_time: float = 0.0
    _last_meta_check: float = 0.0

    # ── Configuration ──

    PLAN_INTERVAL: float = 3600  # Re-plan every hour
    SCHEDULE_INTERVAL: float = 7200  # Re-schedule every 2 hours
    META_CHECK_INTERVAL: float = 86400  # Check meta daily
    OFF_PEAK_HOURS: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 22, 23])
    PEAK_HOURS: list[int] = field(default_factory=lambda: [18, 19, 20, 21])

    # ── Public API ──

    def set_ambition_engine(self, engine: Any) -> None:
        """Wire AmbitionEngine instance."""
        self._ambition_engine = engine

    def set_market_engine(self, engine: Any) -> None:
        """Wire MarketEngine instance."""
        self._market_engine = engine

    def set_social_engine(self, engine: Any) -> None:
        """Wire SocialEngine instance."""
        self._social_engine = engine

    def set_opportunity_cost(self, oc: Any) -> None:
        """Wire OpportunityCostEngine instance."""
        self._opportunity_cost = oc

    # ── Server Meta ──

    def configure_server(self, config: ServerMetaConfig) -> None:
        """Configure server meta for adaptation."""
        with self._lock:
            self._server_meta = config
            self._stats["meta_adaptations"] += 1
            logger.info("server_meta_configured: %s (rate=%.1fx, max_lv=%d)",
                        config.server_name, config.exp_rate, config.max_level)

    def get_server_meta(self) -> ServerMetaConfig:
        with self._lock:
            return self._server_meta

    def adapt_to_meta(self) -> dict[str, Any]:
        """Adapt strategy based on server meta."""
        with self._lock:
            meta = self._server_meta
            adaptations = []

            # Adjust farming targets based on rates
            if meta.drop_rate > 2.0:
                adaptations.append("High drop rate: prioritize item farming over exp")
            if meta.exp_rate > 2.0:
                adaptations.append("High exp rate: leveling is efficient, prioritize exp")
            if meta.zeny_rate > 2.0:
                adaptations.append("High zeny rate: zeny farming is very profitable")

            # Adjust class recommendations
            if meta.popular_classes:
                adaptations.append(f"Popular classes: {', '.join(meta.popular_classes[:3])}")

            # Adjust WOE strategy
            if meta.woe_schedule:
                today = time.strftime("%a").lower()[:3]
                if today in meta.woe_schedule:
                    adaptations.append("WOE today: prepare consumables and gear")

            self._stats["meta_adaptations"] += 1
            return {"adaptations": adaptations, "meta": meta}

    # ── Goal Decomposition ──

    def decompose_ambitions(self) -> list[WeeklyGoal]:
        """Decompose active ambitions into weekly goals."""
        with self._lock:
            goals = []

            if self._ambition_engine is not None:
                try:
                    ambitions = self._ambition_engine._ambitions
                    active = [a for a in ambitions if a.status == "active"]

                    for amb in active:
                        # Create weekly goal from ambition
                        weekly_target = amb.target_value / 4 if amb.target_value > 0 else 1000
                        goal = WeeklyGoal(
                            name=f"Weekly: {amb.name}",
                            ambition_name=amb.name,
                            category=amb.category,
                            target_value=weekly_target,
                            deadline=time.time() + 604800,  # 7 days
                            priority=amb.importance,
                        )

                        # Generate daily tasks based on category
                        if amb.category == "wealth":
                            goal.daily_tasks = [
                                "Farm high-value items for 4 hours",
                                "Check market prices and sell inventory",
                                "Look for arbitrage opportunities",
                            ]
                        elif amb.category == "economy":
                            goal.daily_tasks = [
                                "Run merchant vending for 2 hours",
                                "Buy low, sell high on market",
                                "Craft profitable items",
                            ]
                        elif amb.category == "social":
                            goal.daily_tasks = [
                                "Chat with other players",
                                "Join or form a party",
                                "Participate in guild activities",
                            ]
                        elif amb.category == "pvp":
                            goal.daily_tasks = [
                                "Practice PVP mechanics",
                                "Farm PVP consumables",
                                "Join WOE preparation",
                            ]
                        elif amb.category == "mastery":
                            goal.daily_tasks = [
                                "Level up skills",
                                "Farm exp in optimal zone",
                                "Complete quests for exp",
                            ]
                        else:
                            goal.daily_tasks = [
                                f"Work toward {amb.name}",
                            ]

                        goals.append(goal)
                        self._stats["goals_set"] += 1

                except Exception as e:
                    logger.debug("ambition_decompose_failed: %s", e)

            # If no ambitions, create default goals
            if not goals:
                goals.append(WeeklyGoal(
                    name="Weekly: Level up",
                    ambition_name="Leveling",
                    category="mastery",
                    target_value=10,  # 10 levels per week
                    daily_tasks=["Farm exp for 4 hours", "Sell loot", "Upgrade gear"],
                ))

            self._weekly_goals = goals
            return goals

    def create_daily_tasks(self) -> list[DailyTask]:
        """Create daily tasks from weekly goals."""
        with self._lock:
            tasks = []
            now = time.time()

            for goal in self._weekly_goals:
                if goal.status != "active":
                    continue

                for task_desc in goal.daily_tasks:
                    # Estimate value based on goal category
                    base_value = goal.target_value / max(len(goal.daily_tasks), 1)

                    # Determine best time for this task
                    best_time = "any"
                    if goal.category == "social":
                        best_time = "peak"  # Socialize during peak hours
                    elif goal.category == "economy":
                        best_time = "peak"  # Trade during peak hours
                    elif goal.category == "wealth":
                        best_time = "off_peak"  # Farm during off-peak

                    task = DailyTask(
                        name=task_desc,
                        weekly_goal=goal.name,
                        estimated_duration_minutes=random.randint(30, 120),
                        estimated_value=base_value,
                        best_time=best_time,
                        priority=goal.priority,
                        created_at=now,
                    )
                    tasks.append(task)
                    self._stats["tasks_created"] += 1

            self._daily_tasks = tasks
            return tasks

    # ── Activity Scheduling ──

    def generate_schedule(self, bot_count: int = 1) -> list[ActivitySchedule]:
        """Generate an optimal daily schedule."""
        with self._lock:
            now = time.localtime()
            current_hour = now.tm_hour
            schedule = []

            # Determine if peak or off-peak
            is_peak = current_hour in self.PEAK_HOURS
            is_off_peak = current_hour in self.OFF_PEAK_HOURS

            # Farming (off-peak is best)
            if is_off_peak:
                schedule.append(ActivitySchedule(
                    activity="farm",
                    duration_minutes=120,
                    value_per_hour=10000,
                    best_hour_start=current_hour,
                    best_hour_end=current_hour + 2,
                    priority=8,
                ))
            elif is_peak:
                # During peak, do social activities
                schedule.append(ActivitySchedule(
                    activity="social",
                    duration_minutes=30,
                    value_per_hour=5000,
                    best_hour_start=current_hour,
                    best_hour_end=current_hour + 1,
                    priority=6,
                ))
                schedule.append(ActivitySchedule(
                    activity="farm",
                    duration_minutes=60,
                    value_per_hour=8000,
                    best_hour_start=current_hour + 1,
                    best_hour_end=current_hour + 2,
                    priority=7,
                ))
            else:
                # Normal hours: mix
                schedule.append(ActivitySchedule(
                    activity="farm",
                    duration_minutes=90,
                    value_per_hour=9000,
                    best_hour_start=current_hour,
                    best_hour_end=current_hour + 2,
                    priority=7,
                ))

            # Always include rest breaks
            schedule.append(ActivitySchedule(
                activity="rest",
                duration_minutes=10,
                value_per_hour=0,
                best_hour_start=current_hour + 2,
                best_hour_end=current_hour + 2,
                priority=3,
            ))

            # Shop/sell if inventory is full
            schedule.append(ActivitySchedule(
                activity="shop",
                duration_minutes=15,
                value_per_hour=3000,
                best_hour_start=current_hour + 3,
                best_hour_end=current_hour + 3,
                priority=5,
            ))

            # Assign bots
            for i in range(min(bot_count, len(schedule))):
                schedule[i].assigned_bot = f"bot_{i}"

            self._activity_schedule = schedule
            self._stats["schedules_generated"] += 1
            return schedule

    # ── Opportunity Cost ──

    def calculate_opportunity_cost(self, activity: str, current_value: float) -> float:
        """Calculate the opportunity cost of doing an activity."""
        if self._opportunity_cost is not None:
            try:
                return self._opportunity_cost.calculate(
                    current_activity=activity,
                    current_value=current_value,
                )
            except Exception:
                pass

        # Fallback calculation
        best_alternative = 0
        if activity == "farm":
            best_alternative = current_value * 0.7  # Social is 70% as valuable
        elif activity == "social":
            best_alternative = current_value * 1.5  # Farming is 50% more valuable
        elif activity == "rest":
            best_alternative = current_value * 2.0  # Anything is better than resting
        else:
            best_alternative = current_value * 0.8

        return best_alternative

    # ── Resource Allocation ──

    def allocate_resources(self, bots: list[dict[str, Any]]) -> dict[str, str]:
        """Allocate bots to tasks based on their capabilities."""
        with self._lock:
            allocation: dict[str, str] = {}

            for bot in bots:
                bot_id = bot.get("bot_id", "unknown")
                bot_class = bot.get("class", "novice").lower()
                bot_level = bot.get("level", 1)

                # Find best task for this bot
                best_task = None
                best_score = -1

                for task in self._daily_tasks:
                    if task.status != "pending":
                        continue
                    if task.assigned_bot and task.assigned_bot != bot_id:
                        continue

                    score = self._score_bot_for_task(bot, task)
                    if score > best_score:
                        best_score = score
                        best_task = task

                if best_task:
                    best_task.assigned_bot = bot_id
                    best_task.status = "in_progress"
                    allocation[bot_id] = best_task.name

            return allocation

    def _score_bot_for_task(self, bot: dict[str, Any], task: DailyTask) -> float:
        """Score how well a bot fits a task."""
        score = 0.0
        bot_class = bot.get("class", "novice").lower()

        # Class-task compatibility
        if task.weekly_goal and "wealth" in task.weekly_goal.lower():
            if bot_class in ("merchant", "blacksmith", "alchemist"):
                score += 10
        elif task.weekly_goal and "social" in task.weekly_goal.lower():
            if bot_class in ("bard", "dancer", "minstrel", "wanderer"):
                score += 10
        elif task.weekly_goal and "pvp" in task.weekly_goal.lower():
            if bot_class in ("knight", "assassin", "hunter", "wizard"):
                score += 10

        # Priority bonus
        score += task.priority * 2

        # Random factor for variety
        score += random.uniform(0, 5)

        return score

    # ── Context ──

    def get_planning_context(self) -> str:
        """Get formatted planning context for LLM prompts."""
        with self._lock:
            lines = ["── Long-Term Strategy ──"]

            # Server meta
            meta = self._server_meta
            lines.append(f"Server: {meta.server_name} "
                         f"(exp={meta.exp_rate}x, drop={meta.drop_rate}x, "
                         f"zeny={meta.zeny_rate}x)")

            # Weekly goals
            active_goals = [g for g in self._weekly_goals if g.status == "active"]
            if active_goals:
                lines.append("Weekly goals:")
                for g in active_goals[:3]:
                    progress = f"{g.current_value:.0f}/{g.target_value:.0f}" if g.target_value > 0 else "in progress"
                    lines.append(f"  [{g.priority}] {g.name} ({g.category}) — {progress}")

            # Today's tasks
            pending = [t for t in self._daily_tasks if t.status == "pending"]
            in_progress = [t for t in self._daily_tasks if t.status == "in_progress"]
            if pending or in_progress:
                lines.append("Today's tasks:")
                for t in (in_progress + pending)[:5]:
                    status = "▶" if t.status == "in_progress" else "○"
                    lines.append(f"  {status} {t.name} ({t.estimated_duration_minutes}min)")

            # Schedule
            if self._activity_schedule:
                lines.append("Current schedule:")
                for s in self._activity_schedule[:3]:
                    lines.append(f"  {s.activity} ({s.duration_minutes}min)")

            # Completed today
            today_completed = [t for t in self._completed_tasks
                              if time.time() - t.completed_at < 86400]
            if today_completed:
                lines.append(f"Completed today: {len(today_completed)} tasks")

            return "\n".join(lines)

    # ── Cycle Tick ──

    def tick(self) -> dict[str, Any]:
        """Called every PDCA cycle to update strategy state."""
        now = time.time()
        result = {
            "goals_active": 0,
            "tasks_pending": 0,
            "tasks_in_progress": 0,
            "scheduled_activities": len(self._activity_schedule),
            "meta_adapted": False,
        }

        with self._lock:
            result["goals_active"] = len([g for g in self._weekly_goals if g.status == "active"])
            result["tasks_pending"] = len([t for t in self._daily_tasks if t.status == "pending"])
            result["tasks_in_progress"] = len([t for t in self._daily_tasks if t.status == "in_progress"])

        # Re-plan every hour
        if now - self._last_plan_time > self.PLAN_INTERVAL:
            self.decompose_ambitions()
            self.create_daily_tasks()
            self._last_plan_time = now

        # Re-schedule every 2 hours
        if now - self._last_schedule_time > self.SCHEDULE_INTERVAL:
            self.generate_schedule()
            self._last_schedule_time = now

        # Check meta daily
        if now - self._last_meta_check > self.META_CHECK_INTERVAL:
            adapt = self.adapt_to_meta()
            result["meta_adapted"] = True
            self._last_meta_check = now

        return result

    def complete_task(self, task_name: str) -> None:
        """Mark a task as completed."""
        with self._lock:
            for task in self._daily_tasks:
                if task.name == task_name and task.status == "in_progress":
                    task.status = "completed"
                    task.completed_at = time.time()
                    self._completed_tasks.append(task)
                    self._stats["tasks_completed"] += 1

                    # Update weekly goal progress
                    for goal in self._weekly_goals:
                        if goal.name == task.weekly_goal:
                            goal.current_value += task.estimated_value
                            if goal.target_value > 0 and goal.current_value >= goal.target_value:
                                goal.status = "completed"
                            break
                    break

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# ── Global Singleton ──

_long_term_planner: LongTermPlanner | None = None
_long_term_planner_lock = RLock()


def get_long_term_planner() -> LongTermPlanner:
    global _long_term_planner
    with _long_term_planner_lock:
        if _long_term_planner is None:
            _long_term_planner = LongTermPlanner()
        return _long_term_planner
