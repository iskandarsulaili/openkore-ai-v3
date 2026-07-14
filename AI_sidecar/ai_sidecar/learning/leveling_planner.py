"""
Long-term Strategy / Leveling Planner — plans the optimal leveling path.

A pro player doesn't just farm. They plan: "I need to reach level 99/70
for Transcendent class. That means farming these specific maps from 70-80,
then these from 80-90, then these from 90-99."
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LevelingMilestone:
    """A leveling milestone."""
    level: int
    job_level: int
    total_base_exp: int
    total_job_exp: int
    recommended_maps: list[str] = field(default_factory=list)
    recommended_gear: list[str] = field(default_factory=list)
    new_skills: list[str] = field(default_factory=list)
    class_upgrade: str = ""


@dataclass
class LevelingPlan:
    """A complete leveling plan."""
    current_level: int = 1
    current_job_level: int = 1
    target_level: int = 99
    target_job_level: int = 70
    job_class: str = "novice"
    milestones: list[LevelingMilestone] = field(default_factory=list)
    estimated_hours: float = 0.0
    estimated_zeny_needed: int = 0
    estimated_potion_cost: int = 0
    created_at: datetime = field(default_factory=datetime.now)


class LevelingPlanner:
    """Plans the optimal leveling path."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._plans: dict[str, LevelingPlan] = {}
        self._exp_table: dict[int, int] = self._build_exp_table()
        self._job_exp_table: dict[int, int] = self._build_job_exp_table()
        self._class_upgrades: dict[str, list[tuple[int, str]]] = {
            "novice": [(1, "first_class"), (40, "second_class"), (99, "transcendent")],
            "mage": [(40, "wizard"), (99, "high_wizard")],
            "archer": [(40, "hunter"), (99, "sniper")],
            "swordman": [(40, "knight"), (99, "lord_knight")],
            "thief": [(40, "assassin"), (99, "assassin_cross")],
            "acolyte": [(40, "priest"), (99, "high_priest")],
            "merchant": [(40, "blacksmith"), (99, "mastersmith")],
        }

    def _build_exp_table(self) -> dict[int, int]:
        """Build the RO base experience table (simplified)."""
        table: dict[int, int] = {}
        for level in range(1, 100):
            if level <= 10:
                exp = level * 100
            elif level <= 20:
                exp = level * 500
            elif level <= 30:
                exp = level * 2000
            elif level <= 40:
                exp = level * 5000
            elif level <= 50:
                exp = level * 10000
            elif level <= 60:
                exp = level * 20000
            elif level <= 70:
                exp = level * 40000
            elif level <= 80:
                exp = level * 80000
            elif level <= 90:
                exp = level * 150000
            else:
                exp = level * 300000
            table[level] = exp
        return table

    def _build_job_exp_table(self) -> dict[int, int]:
        """Build the RO job experience table (simplified)."""
        table: dict[int, int] = {}
        for level in range(1, 71):
            if level <= 10:
                exp = level * 50
            elif level <= 20:
                exp = level * 200
            elif level <= 30:
                exp = level * 1000
            elif level <= 40:
                exp = level * 3000
            elif level <= 50:
                exp = level * 8000
            elif level <= 60:
                exp = level * 20000
            else:
                exp = level * 50000
            table[level] = exp
        return table

    # ── Public API ──

    def create_plan(
        self,
        current_level: int = 1,
        current_job_level: int = 1,
        target_level: int = 99,
        target_job_level: int = 70,
        job_class: str = "novice",
        exp_per_hour: int = 50000,
    ) -> LevelingPlan:
        """Create a leveling plan."""
        with self._lock:
            plan = LevelingPlan(
                current_level=current_level,
                current_job_level=current_job_level,
                target_level=target_level,
                target_job_level=target_job_level,
                job_class=job_class,
            )

            # Calculate milestones every 10 levels
            for level in range(1, target_level + 1, 10):
                milestone = LevelingMilestone(
                    level=level,
                    job_level=min(level, target_job_level),
                    total_base_exp=sum(self._exp_table.get(i, 0) for i in range(1, level + 1)),
                    total_job_exp=sum(self._job_exp_table.get(i, 0) for i in range(1, min(level, target_job_level) + 1)),
                )

                # Recommend maps based on level
                if level < 20:
                    milestone.recommended_maps = ["prt_fild01", "prt_fild02", "prt_fild03"]
                    milestone.recommended_gear = ["Novice Gear"]
                elif level < 40:
                    milestone.recommended_maps = ["prt_fild04", "prt_fild05", "pay_dun00", "moc_fild17"]
                    milestone.recommended_gear = ["First Class Gear"]
                elif level < 60:
                    milestone.recommended_maps = ["prt_fild06", "prt_fild07", "pay_dun01", "gef_dun00", "orcsdun01"]
                    milestone.recommended_gear = ["Second Class Gear"]
                elif level < 80:
                    milestone.recommended_maps = ["prt_fild08", "prt_fild09", "pay_dun02", "gef_dun01", "orcsdun02"]
                    milestone.recommended_gear = ["Advanced Gear"]
                else:
                    milestone.recommended_maps = ["prt_fild10", "prt_fild11", "pay_dun03", "pay_dun04", "gef_dun02", "gef_dun03"]
                    milestone.recommended_gear = ["Endgame Gear"]

                # Check for class upgrades
                for cls, upgrades in self._class_upgrades.items():
                    if job_class.lower() == cls:
                        for req_level, upgrade_name in upgrades:
                            if level >= req_level:
                                milestone.class_upgrade = upgrade_name

                plan.milestones.append(milestone)

            # Calculate estimated time
            total_exp_needed = sum(
                self._exp_table.get(i, 0) for i in range(current_level, target_level + 1)
            )
            if exp_per_hour > 0:
                plan.estimated_hours = total_exp_needed / exp_per_hour

            # Estimate costs
            plan.estimated_zeny_needed = int(plan.estimated_hours * 10000)  # rough estimate
            plan.estimated_potion_cost = int(plan.estimated_hours * 5000)

            plan_id = f"plan_{current_level}_{job_class}_{datetime.now().timestamp():.0f}"
            self._plans[plan_id] = plan
            return plan

    def get_plan(self, plan_id: str) -> LevelingPlan | None:
        with self._lock:
            return self._plans.get(plan_id)

    def get_latest_plan(self) -> LevelingPlan | None:
        with self._lock:
            if not self._plans:
                return None
            return max(self._plans.values(), key=lambda p: p.created_at)

    def get_next_milestone(self, current_level: int, job_class: str = "novice") -> LevelingMilestone | None:
        """Get the next leveling milestone."""
        with self._lock:
            plan = self.get_latest_plan()
            if not plan:
                plan = self.create_plan(current_level=current_level, job_class=job_class)
            for milestone in plan.milestones:
                if milestone.level > current_level:
                    return milestone
            return None

    def get_next_class_upgrade(self, current_level: int, job_class: str) -> str | None:
        """Get the next class upgrade available."""
        with self._lock:
            for cls, upgrades in self._class_upgrades.items():
                if job_class.lower() == cls:
                    for req_level, upgrade_name in upgrades:
                        if current_level >= req_level:
                            return upgrade_name
            return None

    def get_recommended_maps_for_level(self, level: int) -> list[str]:
        """Get recommended maps for a given level."""
        with self._lock:
            if level < 20:
                return ["prt_fild01", "prt_fild02", "prt_fild03"]
            elif level < 40:
                return ["prt_fild04", "prt_fild05", "pay_dun00", "moc_fild17"]
            elif level < 60:
                return ["prt_fild06", "prt_fild07", "pay_dun01", "gef_dun00", "orcsdun01"]
            elif level < 80:
                return ["prt_fild08", "prt_fild09", "pay_dun02", "gef_dun01", "orcsdun02"]
            else:
                return ["prt_fild10", "prt_fild11", "pay_dun03", "pay_dun04", "gef_dun02", "gef_dun03"]

    def get_exp_to_level(self, current_level: int, target_level: int) -> int:
        """Get total experience needed to reach target level."""
        with self._lock:
            return sum(self._exp_table.get(i, 0) for i in range(current_level, target_level + 1))

    def get_estimated_time_to_target(
        self, current_level: int, target_level: int, exp_per_hour: int
    ) -> float:
        """Get estimated hours to reach target level."""
        total_exp = self.get_exp_to_level(current_level, target_level)
        if exp_per_hour <= 0:
            return float("inf")
        return total_exp / exp_per_hour

    def get_leveling_summary(self, current_level: int, job_class: str = "novice") -> str:
        """Get a human-readable leveling summary."""
        with self._lock:
            plan = self.get_latest_plan()
            if not plan:
                plan = self.create_plan(current_level=current_level, job_class=job_class)

            lines = [f"── Leveling Plan ──"]
            lines.append(f"Current: Lv.{current_level} {job_class}")
            lines.append(f"Target: Lv.{plan.target_level} / Job {plan.target_job_level}")
            lines.append(f"Estimated time: {plan.estimated_hours:.1f} hours")
            lines.append(f"Estimated zeny needed: {plan.estimated_zeny_needed:,}z")

            next_milestone = self.get_next_milestone(current_level, job_class)
            if next_milestone:
                lines.append(f"\nNext milestone: Lv.{next_milestone.level}")
                lines.append(f"Recommended maps: {', '.join(next_milestone.recommended_maps)}")
                if next_milestone.class_upgrade:
                    lines.append(f"Class upgrade available: {next_milestone.class_upgrade}")

            return "\n".join(lines)

    def get_all_plans(self) -> list[LevelingPlan]:
        with self._lock:
            return list(self._plans.values())

    def clear_plans(self) -> None:
        with self._lock:
            self._plans.clear()


# ── Global Singleton ──

_leveling_planner: LevelingPlanner | None = None
_leveling_planner_lock = RLock()


def get_leveling_planner() -> LevelingPlanner:
    global _leveling_planner
    with _leveling_planner_lock:
        if _leveling_planner is None:
            _leveling_planner = LevelingPlanner()
        return _leveling_planner
