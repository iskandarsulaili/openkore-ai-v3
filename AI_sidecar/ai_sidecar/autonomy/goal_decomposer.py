"""
Goal Decomposition & Priority Sequencer
========================================
Chunks big goals into smaller achievable sub-goals across short, medium,
and long term horizons. Each sub-goal has a clear dependency chain so the
PDCA loop knows what to execute NOW vs what to plan for LATER.

Key design:
- Big goals are decomposed into DAGs (directed acyclic graphs) of sub-goals
- Each sub-goal is assigned to a horizon (short/medium/long)
- Short-term sub-goals are prerequisites for medium-term, which feed long-term
- Priority is adaptive — shifts based on bot state, not fixed ranks
- Multiple bots coordinate to avoid overlapping sub-goals
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from enum import StrEnum
from typing import Any

from ai_sidecar.contracts.autonomy import GoalCategory, GoalDirective, GoalStackState, SituationalAssessment

logger = logging.getLogger(__name__)


class GoalHorizon(StrEnum):
    tactical = "tactical"       # 0-30 seconds: immediate actions
    short_term = "short_term"   # 30s-5min: what to do NOW
    medium_term = "medium_term" # 5-30min: what to do NEXT
    long_term = "long_term"     # 30min+: the BIG goal


@dataclass(slots=True)
class DecomposedGoal:
    """A single chunk of a larger goal."""
    id: str
    parent: GoalCategory
    horizon: GoalHorizon
    objective: str
    prerequisites: list[str]  # IDs of sub-goals that must be done first
    estimated_duration_s: float
    success_metrics: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending | ready | in_progress | completed | blocked
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GoalDecomposition:
    """Full decomposition of a goal into sub-goals across horizons."""
    parent: GoalCategory
    sub_goals: dict[str, DecomposedGoal]  # id -> DecomposedGoal
    dependencies: dict[str, list[str]]  # sub_goal_id -> prerequisite ids
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime = field(default_factory=lambda: datetime.now(UTC) + timedelta(minutes=15))


class GoalDecomposer:
    """Decomposes high-level goals into executable sub-goals.

    The decomposer is the "conscious" layer — it uses the LLM to understand
    the bot's current state and break down big goals into achievable chunks.
    When the LLM is unavailable, deterministic fallback patterns are used.
    """

    def __init__(self):
        self._decompositions: dict[str, GoalDecomposition] = {}  # bot_id -> decomposition
        self._completed: dict[str, set[str]] = defaultdict(set)  # bot_id -> set of completed sub_goal ids

    def decompose(self, *, bot_id: str, assessment: SituationalAssessment,
                  goal_stack: list[GoalDirective], selected: GoalDirective,
                  knowledge: dict[str, Any] | None = None) -> GoalDecomposition:
        """Decompose the selected goal into sub-goals across horizons.

        Uses LLM if available, falls back to deterministic patterns.
        """
        # Check if we already have a fresh decomposition
        existing = self._decompositions.get(bot_id)
        if existing and existing.expires_at > datetime.now(UTC) and existing.parent == selected.goal_key:
            return existing

        decomposition = self._deterministic_decompose(
            bot_id=bot_id, assessment=assessment,
            goal=selected, goal_stack=goal_stack, knowledge=knowledge
        )
        self._decompositions[bot_id] = decomposition
        return decomposition

    def _deterministic_decompose(self, *, bot_id: str, assessment: SituationalAssessment,
                                  goal: GoalDirective, goal_stack: list[GoalDirective],
                                  knowledge: dict[str, Any] | None = None) -> GoalDecomposition:
        """Deterministic fallback decomposition based on goal category."""
        now = datetime.now(UTC)
        sub_goals: dict[str, DecomposedGoal] = {}
        deps: dict[str, list[str]] = {}

        # Extract state from assessment
        map_name = str(assessment.map_name or "unknown")
        hp = assessment.hp_ratio
        has_points = assessment.skill_points > 0 or assessment.stat_points > 0
        base_level = assessment.base_level

        if goal.goal_key == GoalCategory.survival:
            # Survival is always highest priority — quick, immediate actions
            if assessment.is_dead or assessment.is_disconnected:
                sg_id = "reconnect"
                sub_goals[sg_id] = DecomposedGoal(
                    id=sg_id, parent=GoalCategory.survival, horizon=GoalHorizon.tactical,
                    objective=f"reconnect and respawn at savepoint",
                    prerequisites=[], estimated_duration_s=30,
                    success_metrics={"connected": True, "alive": True},
                )
                deps[sg_id] = []
            if hp < 0.3:
                sg_id = "urgent_heal"
                sub_goals[sg_id] = DecomposedGoal(
                    id=sg_id, parent=GoalCategory.survival, horizon=GoalHorizon.tactical,
                    objective=f"urgent HP recovery — sit or use healing items",
                    prerequisites=[], estimated_duration_s=15,
                    success_metrics={"hp_ratio": hp + 0.3},
                )
                deps[sg_id] = []
            if hp < 0.6:
                sg_id = "safe_recover"
                sub_goals[sg_id] = DecomposedGoal(
                    id=sg_id, parent=GoalCategory.survival, horizon=GoalHorizon.short_term,
                    objective=f"recover HP safely on {map_name}",
                    prerequisites=[], estimated_duration_s=30,
                    success_metrics={"hp_ratio": 0.8},
                )
                deps[sg_id] = []

        elif goal.goal_key == GoalCategory.job_advancement:
            # Job advancement is a BIG goal — decompose into steps
            if has_points:
                sg1 = "allocate_stats"
                sub_goals[sg1] = DecomposedGoal(
                    id=sg1, parent=GoalCategory.job_advancement, horizon=GoalHorizon.tactical,
                    objective=f"allocate {assessment.skill_points} skill + {assessment.stat_points} stat points",
                    prerequisites=[], estimated_duration_s=10,
                    success_metrics={"skill_points_allocated": True, "stat_points_allocated": True},
                )
                deps[sg1] = []

            sg2 = "reach_hunting_ground"
            sub_goals[sg2] = DecomposedGoal(
                id=sg2, parent=GoalCategory.job_advancement, horizon=GoalHorizon.short_term,
                objective=f"move from {map_name} to a hunting ground and level up",
                prerequisites=[], estimated_duration_s=120,
                success_metrics={"map_changed": True, "in_combat": True},
            )
            deps_list = []
            if has_points:
                deps_list.append("allocate_stats")
            deps[sg2] = deps_list

            sg3 = "grind_for_exp"
            sub_goals[sg3] = DecomposedGoal(
                id=sg3, parent=GoalCategory.job_advancement, horizon=GoalHorizon.medium_term,
                objective=f"grind until job level requirements met for job change",
                prerequisites=["reach_hunting_ground"], estimated_duration_s=1800,
                success_metrics={"job_exp_ratio": 1.0, "level_gained": True},
            )
            deps[sg3] = ["reach_hunting_ground"]

            sg4 = "execute_job_change"
            sub_goals[sg4] = DecomposedGoal(
                id=sg4, parent=GoalCategory.job_advancement, horizon=GoalHorizon.long_term,
                objective=f"travel to job change NPC and complete the job advancement quest",
                prerequisites=["grind_for_exp"], estimated_duration_s=300,
                success_metrics={"job_changed": True, "new_job_set": True},
            )
            deps[sg4] = ["grind_for_exp"]

        elif goal.goal_key == GoalCategory.leveling:
            # Leveling — decompose into hunt zones based on bot level
            zone = self._recommend_zone(base_level)

            sg1 = "move_to_zone"
            sub_goals[sg1] = DecomposedGoal(
                id=sg1, parent=GoalCategory.leveling, horizon=GoalHorizon.short_term,
                objective=f"move to {zone} hunting zone from {map_name}",
                prerequisites=[], estimated_duration_s=120,
                success_metrics={"arrived_at_zone": True},
            )
            deps[sg1] = []

            sg2 = "hunt_for_exp"
            sub_goals[sg2] = DecomposedGoal(
                id=sg2, parent=GoalCategory.leveling, horizon=GoalHorizon.medium_term,
                objective=f"hunt in {zone} for experience and loot",
                prerequisites=["move_to_zone"], estimated_duration_s=1800,
                success_metrics={"base_exp_gained": True, "items_looted": True},
            )
            deps[sg2] = ["move_to_zone"]

            sg3 = "check_progression"
            sub_goals[sg3] = DecomposedGoal(
                id=sg3, parent=GoalCategory.leveling, horizon=GoalHorizon.long_term,
                objective=f"evaluate leveling progress and decide next zone or job change",
                prerequisites=["hunt_for_exp"], estimated_duration_s=60,
                success_metrics={"next_step_decided": True},
            )
            deps[sg3] = ["hunt_for_exp"]

        elif goal.goal_key == GoalCategory.opportunistic_upgrades:
            sg1 = "assess_economy"
            sub_goals[sg1] = DecomposedGoal(
                id=sg1, parent=GoalCategory.opportunistic_upgrades, horizon=GoalHorizon.short_term,
                objective=f"scan market and inventory for upgrade opportunities",
                prerequisites=[], estimated_duration_s=30,
                success_metrics={"opportunities_scanned": True},
            )
            deps[sg1] = []

            sg2 = "execute_upgrade"
            sub_goals[sg2] = DecomposedGoal(
                id=sg2, parent=GoalCategory.opportunistic_upgrades, horizon=GoalHorizon.medium_term,
                objective=f"execute the best upgrade opportunity found",
                prerequisites=["assess_economy"], estimated_duration_s=120,
                success_metrics={"upgrade_completed": True},
            )
            deps[sg2] = ["assess_economy"]

        # Always add the current objective as a wrapper
        if not sub_goals:
            sg_id = "continue_current"
            sub_goals[sg_id] = DecomposedGoal(
                id=sg_id, parent=goal.goal_key, horizon=GoalHorizon.short_term,
                objective=goal.objective,
                prerequisites=[], estimated_duration_s=60,
                success_metrics={"objective_advanced": True},
            )
            deps[sg_id] = []

        # Compute expiry
        now = datetime.now(UTC)
        expires = max(
            (now + timedelta(seconds=g.estimated_duration_s * 2))
            for g in sub_goals.values()
        )

        return GoalDecomposition(
            parent=goal.goal_key,
            sub_goals=sub_goals,
            dependencies=deps,
            expires_at=expires,
        )

    def _recommend_zone(self, level: int) -> str:
        """Recommend a hunting zone based on bot level."""
        if level <= 10:
            return "prt_fild08"  # Porings, Pops
        elif level <= 20:
            return "prt_fild04"  # Fabre, Peco Peco
        elif level <= 35:
            return "pay_fild08"  # Spores, etc.
        elif level <= 50:
            return "pay_fild04"  # Argiope, etc.
        elif level <= 70:
            return "gef_fild14"  # High orcs, etc.
        elif level <= 90:
            return "moc_fild17"  # Mimics, etc.
        else:
            return "mjolnir_04"

    def next_action(self, *, bot_id: str) -> DecomposedGoal | None:
        """Get the NEXT actionable sub-goal — the first ready one."""
        decomposition = self._decompositions.get(bot_id)
        if not decomposition:
            return None

        completed = self._completed.get(bot_id, set())

        # Find all sub-goals whose prerequisites are met
        ready = []
        for sg_id, sg in decomposition.sub_goals.items():
            if sg_id in completed:
                continue
            prereqs = decomposition.dependencies.get(sg_id, [])
            if all(p in completed for p in prereqs):
                ready.append(sg)

        if not ready:
            return None

        # Return the one with the shortest horizon first (tactical > short > medium > long)
        horizon_order = {GoalHorizon.tactical: 0, GoalHorizon.short_term: 1,
                         GoalHorizon.medium_term: 2, GoalHorizon.long_term: 3}
        ready.sort(key=lambda g: horizon_order.get(g.horizon, 99))
        return ready[0]

    def mark_completed(self, *, bot_id: str, sub_goal_id: str) -> None:
        """Mark a sub-goal as completed."""
        self._completed[bot_id].add(sub_goal_id)

    def progress(self, *, bot_id: str) -> dict[str, Any]:
        """Return progress metrics for the current decomposition."""
        decomposition = self._decompositions.get(bot_id)
        if not decomposition:
            return {"status": "no_decomposition", "progress": 0.0}

        completed = self._completed.get(bot_id, set())
        total = len(decomposition.sub_goals)
        done = len(completed)
        ratio = done / max(total, 1)

        next_sg = self.next_action(bot_id=bot_id)
        return {
            "status": "in_progress" if next_sg else "complete",
            "progress": ratio,
            "completed": done,
            "total": total,
            "next_action": next_sg.objective if next_sg else None,
            "next_horizon": next_sg.horizon.value if next_sg else None,
            "parent_goal": decomposition.parent.value,
        }


class CrossHorizonSynergy:
    """Ensures short-term goals feed into medium-term, which feed into long-term.

    Detects conflicts between horizons and adjusts priorities.
    """

    def detect_conflicts(self, *, decomposition: GoalDecomposition) -> list[str]:
        """Detect conflicts between sub-goals across horizons."""
        conflicts = []
        for sg_id, sg in decomposition.sub_goals.items():
            for other_id, other in decomposition.sub_goals.items():
                if sg_id == other_id:
                    continue
                # Check if two tactical goals compete for the same resource
                if sg.horizon == GoalHorizon.tactical and other.horizon == GoalHorizon.tactical:
                    if "heal" in sg.objective and "heal" not in other.objective:
                        conflicts.append(f"tactical conflict: {sg.objective} vs {other.objective}")
        return list(set(conflicts))

    def adjust_for_horizon(self, *, decomposition: GoalDecomposition,
                           primary_horizon: GoalHorizon) -> list[str]:
        """Return the sub-goal IDs that should be active for the given horizon."""
        active = []
        for sg_id, sg in decomposition.sub_goals.items():
            horizon_order = {GoalHorizon.tactical: 0, GoalHorizon.short_term: 1,
                             GoalHorizon.medium_term: 2, GoalHorizon.long_term: 3}
            h_order = horizon_order.get(sg.horizon, 99)
            p_order = horizon_order.get(primary_horizon, 99)
            # Include goals at or below the primary horizon
            if h_order <= p_order:
                active.append(sg_id)
        return active


class SwarmGoalCoordinator:
    """Coordinates goals across multiple bots to avoid overlap and maximize synergy.

    For a swarm of N bots, distributes hunting zones and roles so they
    don't compete for the same monsters.
    """

    def __init__(self):
        self._bot_assignments: dict[str, dict[str, Any]] = {}  # bot_id -> assignment

    def assign_zone(self, *, bot_id: str, available_zones: list[str],
                    decomposition: GoalDecomposition | None) -> str | None:
        """Assign the best zone for a bot, avoiding overlap with other bots."""
        # Get zones already assigned to other bots
        assigned_zones = set()
        for other_bot, assignment in self._bot_assignments.items():
            if other_bot != bot_id:
                zone = assignment.get("zone")
                if zone:
                    assigned_zones.add(zone)

        # Get the recommended zone from decomposition
        recommended = None
        if decomposition:
            for sg in decomposition.sub_goals.values():
                if "move to" in sg.objective.lower() or "hunt in" in sg.objective.lower():
                    # Extract zone name from objective
                    for zone in available_zones:
                        if zone in sg.objective:
                            recommended = zone
                            break

        if recommended and recommended not in assigned_zones:
            self._bot_assignments[bot_id] = {"zone": recommended}
            return recommended

        # Pick the first unassigned zone
        for zone in available_zones:
            if zone not in assigned_zones:
                self._bot_assignments[bot_id] = {"zone": zone}
                return zone

        # All zones taken — share the best one
        if available_zones:
            self._bot_assignments[bot_id] = {"zone": available_zones[0]}
            return available_zones[0]

        return None

    def clear(self, bot_id: str | None = None) -> None:
        """Clear assignments for a bot or all bots."""
        if bot_id:
            self._bot_assignments.pop(bot_id, None)
        else:
            self._bot_assignments.clear()