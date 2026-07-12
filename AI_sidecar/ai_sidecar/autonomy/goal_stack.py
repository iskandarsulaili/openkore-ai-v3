"""
Goal Stack Computation & Execution Management
==============================================
Manages a per-bot stack of goals as lightweight dicts with auto-prioritization
across short-term, medium-term, and long-term horizons.

Key design:
- Per-bot_id stacks stored as list[dict] with thread-safe RLock
- Auto-prioritize: long_term goals persist, medium_term replaces short_term,
  short_term goals are consumed by execution (popped)
- Integrates with GoalStackState from contracts/autonomy.py
- Computes current best goal, next actionable, completion %, failed detection
"""

from __future__ import annotations

import time
import logging
from copy import deepcopy
from datetime import datetime, timezone
from enum import StrEnum
from threading import RLock
from typing import Any

from ai_sidecar.contracts.autonomy import (
    GoalCategory,
    GoalDirective,
    GoalStackState,
    SituationalAssessment,
)

logger = logging.getLogger(__name__)


class GoalHorizon(StrEnum):
    """Horizon labels for goal lifespan and auto-prioritize logic."""
    SHORT_TERM = "short_term"    # ~5s, consumed by execution
    MEDIUM_TERM = "medium_term"  # ~30s, replaces short_term
    LONG_TERM = "long_term"      # ~120s, persists indefinitely


# Default horizon durations (seconds) — used for staleness detection.
HORIZON_DURATION_S: dict[str, float] = {
    "short_term": 5.0,
    "medium_term": 30.0,
    "long_term": 120.0,
}

# Valid status values for a goal dict.
VALID_STATUSES = frozenset({"pending", "in_progress", "completed", "failed"})
VALID_HORIZONS = frozenset({"short_term", "medium_term", "long_term"})


def _utc_now() -> datetime:
    """Timezone-aware UTC now."""
    return datetime.now(timezone.utc)


def _goal_dict(
    *,
    goal_id: str,
    description: str,
    priority: int = 50,
    horizon: str = "short_term",
    status: str = "pending",
    parent_id: str | None = None,
) -> dict[str, Any]:
    """Build a canonical goal dict with all required fields."""
    assert 0 <= priority <= 100, f"priority {priority} out of [0, 100]"
    assert horizon in VALID_HORIZONS, f"invalid horizon {horizon!r}"
    assert status in VALID_STATUSES, f"invalid status {status!r}"
    return {
        "id": goal_id,
        "description": description,
        "priority": priority,
        "horizon": horizon,
        "status": status,
        "created_at": _utc_now(),
        "parent_id": parent_id,
    }


class GoalStackComputation:
    """Per-bot goal stack manager with auto-prioritize and thread safety.

    Each bot has an ordered list of goal dicts.  Higher-priority entries
    appear first.  Auto-prioritize logic runs on every push().
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._stacks: dict[str, list[dict[str, Any]]] = {}

        # --- backward-compatible attributes set by compute_goal_stack() ---
        self.goal_stack: list[GoalDirective] = []
        self.selected_goal: GoalDirective | None = None

    # ------------------------------------------------------------------
    # Public stack operations
    # ------------------------------------------------------------------

    def push(self, bot_id: str, goal: dict[str, Any]) -> None:
        """Push a goal onto bot_id's stack then auto-prioritize."""
        with self._lock:
            self._ensure_stack(bot_id)
            # Validate / normalise the incoming dict
            g = goal.copy()
            g.setdefault("id", f"g_{time.monotonic_ns()}")
            g.setdefault("description", "")
            g.setdefault("priority", 50)
            g.setdefault("horizon", "short_term")
            g.setdefault("status", "pending")
            g.setdefault("created_at", _utc_now())
            g.setdefault("parent_id", None)
            # Clamp
            g["priority"] = max(0, min(100, int(g.get("priority", 50))))
            if g["horizon"] not in VALID_HORIZONS:
                g["horizon"] = "short_term"
            if g["status"] not in VALID_STATUSES:
                g["status"] = "pending"
            self._stacks[bot_id].append(g)
            self._auto_prioritize(bot_id)

    def pop(self, bot_id: str) -> dict[str, Any] | None:
        """Pop and return the highest-priority actionable goal.

        Only returns goals with status ``pending`` or ``in_progress``.
        Returns None when no actionable goals remain.
        """
        with self._lock:
            stack = self._stacks.get(bot_id)
            if not stack:
                return None
            actionable = [
                g for g in stack
                if g.get("status") in ("pending", "in_progress")
            ]
            if not actionable:
                return None
            best = actionable[0]
            stack.remove(best)
            return best

    def peek(self, bot_id: str) -> dict[str, Any] | None:
        """Return the highest-priority actionable goal without removing it."""
        with self._lock:
            stack = self._stacks.get(bot_id)
            if not stack:
                return None
            for g in stack:
                if g.get("status") in ("pending", "in_progress"):
                    return g
            return None

    def list_goals(self, bot_id: str) -> list[dict[str, Any]]:
        """Return a deep copy of bot_id's entire goal stack."""
        with self._lock:
            return deepcopy(self._stacks.get(bot_id, []))

    def clear(self, bot_id: str) -> None:
        """Remove all goals for bot_id."""
        with self._lock:
            self._stacks.pop(bot_id, None)

    def reorder(self, bot_id: str, priorities: dict[str, int]) -> None:
        """Update priorities for specific goal ids, then re-sort the stack.

        Args:
            bot_id: Bot identifier.
            priorities: Mapping of goal_id -> new priority (0-100).
        """
        with self._lock:
            stack = self._stacks.get(bot_id)
            if not stack:
                return
            for g in stack:
                gid = g.get("id")
                if gid in priorities:
                    g["priority"] = max(0, min(100, int(priorities[gid])))
            stack.sort(key=_goal_sort_key, reverse=True)

    # ------------------------------------------------------------------
    # Computations
    # ------------------------------------------------------------------

    def current_best_goal(self, bot_id: str) -> dict[str, Any] | None:
        """Return the single best goal to execute right now.

        ``best`` == highest-priority pending or in_progress goal.
        """
        with self._lock:
            stack = self._stacks.get(bot_id)
            if not stack:
                return None
            for g in stack:
                if g.get("status") in ("pending", "in_progress"):
                    return g
            return None

    def next_actionable_goal(self, bot_id: str) -> dict[str, Any] | None:
        """Return the next goal that can be acted on (strictly pending)."""
        with self._lock:
            stack = self._stacks.get(bot_id)
            if not stack:
                return None
            for g in stack:
                if g.get("status") == "pending":
                    return g
            return None

    def completion_percentage(self, bot_id: str) -> float:
        """Ratio of completed goals to total goals for bot_id.

        Returns 0.0 if no goals exist.
        """
        with self._lock:
            stack = self._stacks.get(bot_id)
            if not stack:
                return 0.0
            total = len(stack)
            done = sum(1 for g in stack if g.get("status") == "completed")
            return done / total if total > 0 else 0.0

    def detect_failed_goals(self, bot_id: str) -> list[dict[str, Any]]:
        """Return goals that have exceeded their horizon's expected duration.

        A goal is considered ``stuck`` / potentially failed when:
        - status is ``in_progress``
        - age > HORIZON_DURATION_S for its horizon level
        """
        with self._lock:
            stack = self._stacks.get(bot_id)
            if not stack:
                return []
            now = _utc_now()
            stuck: list[dict[str, Any]] = []
            for g in stack:
                if g.get("status") != "in_progress":
                    continue
                created = g.get("created_at")
                if created is None or not isinstance(created, datetime):
                    continue
                horizon = g.get("horizon", "short_term")
                max_age = HORIZON_DURATION_S.get(horizon, 5.0)
                age_s = (now - created).total_seconds()
                if age_s > max_age:
                    stuck.append(g)
            return stuck

    def failed_goal_count(self, bot_id: str) -> int:
        """Number of goals explicitly marked as ``failed``."""
        with self._lock:
            stack = self._stacks.get(bot_id)
            if not stack:
                return 0
            return sum(1 for g in stack if g.get("status") == "failed")

    # ------------------------------------------------------------------
    # GoalStackState integration
    # ------------------------------------------------------------------

    def to_goal_stack_state(
        self,
        bot_id: str,
        assessment: SituationalAssessment | None = None,
        tick_id: str | None = None,
        horizon: str = "tactical",
    ) -> GoalStackState | None:
        """Convert internal goal stack to a GoalStackState contract object.

        If no assessment is provided, a minimal one is synthesised so the
        contract is always valid.
        """
        with self._lock:
            stack = self._stacks.get(bot_id)
            if not stack:
                return None
            directives = [_goal_to_directive(g) for g in stack]
            best = self.current_best_goal(bot_id)
            selected = (
                _goal_to_directive(best) if best else directives[0]
            ) if directives else GoalDirective(
                goal_key=GoalCategory.survival,
                priority_rank=999,
                active=False,
                objective="no goals",
            )
            sa = assessment or SituationalAssessment(bot_id=bot_id)
            return GoalStackState(
                bot_id=bot_id,
                tick_id=tick_id,
                horizon=horizon,
                assessment=sa,
                goal_stack=directives,
                selected_goal=selected,
            )

    @classmethod
    def from_goal_stack_state(cls, state: GoalStackState) -> GoalStackComputation:
        """Create a GoalStackComputation restored from a GoalStackState."""
        comp = cls()
        with comp._lock:
            comp.goal_stack = list(state.goal_stack)
            comp.selected_goal = state.selected_goal
            bot_id = state.bot_id
            stack: list[dict[str, Any]] = []
            for gd in state.goal_stack:
                stack.append(_directive_to_goal(gd))
            if stack:
                comp._stacks[bot_id] = stack
        return comp

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_stack(self, bot_id: str) -> None:
        if bot_id not in self._stacks:
            self._stacks[bot_id] = []

    def _auto_prioritize(self, bot_id: str) -> None:
        """Enforce auto-prioritize rules on bot_id's stack.

        Rules (run after every push):
        1. long_term goals stay (never removed).
        2. medium_term goals replace (remove) short_term goals that have lower
           or equal priority.
        3. short_term goals with status ``in_progress`` or ``pending`` are
           candidates for replacement.
        4. Always re-sort by priority descending.
        """
        stack = self._stacks.get(bot_id)
        if not stack:
            return

        long_term_goals: list[dict] = []
        medium_term_goals: list[dict] = []
        short_term_goals: list[dict] = []

        for g in stack:
            h = g.get("horizon", "short_term")
            if h == "long_term":
                long_term_goals.append(g)
            elif h == "medium_term":
                medium_term_goals.append(g)
            else:
                short_term_goals.append(g)

        # medium_term replaces short_term (lower/equal priority)
        for mt in medium_term_goals:
            mt_prio = mt.get("priority", 50)
            short_term_goals = [
                st for st in short_term_goals
                if st.get("priority", 0) > mt_prio
            ]

        long_term_goals.sort(key=_goal_sort_key, reverse=True)
        medium_term_goals.sort(key=_goal_sort_key, reverse=True)
        short_term_goals.sort(key=_goal_sort_key, reverse=True)

        self._stacks[bot_id] = long_term_goals + medium_term_goals + short_term_goals


# ======================================================================
# Conversion helpers
# ======================================================================


def _goal_sort_key(g: dict[str, Any]) -> tuple:
    """Sort key: higher priority first, then by horizon rank, then created_at."""
    horizon_rank = {"long_term": 0, "medium_term": 1, "short_term": 2}
    h = horizon_rank.get(g.get("horizon", "short_term"), 3)
    created = g.get("created_at", _utc_now())
    return (g.get("priority", 0), -h, created.timestamp())


def _goal_to_directive(g: dict[str, Any]) -> GoalDirective:
    """Convert a goal dict to a GoalDirective contract object."""
    return GoalDirective(
        goal_key=GoalCategory.survival,
        priority_rank=max(1, min(999, 100 - g.get("priority", 50))),
        active=g.get("status") in ("pending", "in_progress"),
        objective=g.get("description", ""),
        rationale=f"auto_stack:horizon={g.get('horizon','short_term')}",
        blockers=[],
        metadata={
            "goal_id": g.get("id", ""),
            "horizon": g.get("horizon", "short_term"),
            "status": g.get("status", "pending"),
            "parent_id": g.get("parent_id"),
            "created_at": str(g.get("created_at", "")),
        },
    )


def _directive_to_goal(gd: GoalDirective) -> dict[str, Any]:
    """Convert a GoalDirective back to a goal dict."""
    md = gd.metadata if isinstance(gd.metadata, dict) else {}
    return {
        "id": md.get("goal_id", gd.goal_key.value),
        "description": gd.objective,
        "priority": max(0, min(100, 100 - gd.priority_rank)),
        "horizon": md.get("horizon", "short_term"),
        "status": md.get("status", "pending"),
        "created_at": md.get("created_at", _utc_now()),
        "parent_id": md.get("parent_id"),
    }


# ======================================================================
# Legacy functions (kept for backward compatibility with PDCA loop)
# ======================================================================


def compute_goal_stack(
    *, assessment: SituationalAssessment, horizon: str
) -> GoalStackComputation:
    """Deterministic goal stack builder from a situational assessment.

    Returns a GoalStackComputation whose instance can also be used as a
    per-bot stack manager.
    """
    map_name = str(assessment.map_name or "unknown")
    progression = (
        assessment.progression_recommendation
        if isinstance(assessment.progression_recommendation, dict)
        else {}
    )
    job_advancement = (
        assessment.job_advancement
        if isinstance(assessment.job_advancement, dict)
        else {}
    )
    opportunistic = (
        assessment.opportunistic_upgrades
        if isinstance(assessment.opportunistic_upgrades, dict)
        else {}
    )

    points_pending = bool(assessment.skill_points > 0 or assessment.stat_points > 0)
    playbook_supported = bool(job_advancement.get("supported"))
    playbook_ready = bool(job_advancement.get("ready"))
    route_id = str(job_advancement.get("route_id") or "")
    target_job = str(job_advancement.get("target_job") or "")
    missing_requirements = [
        str(item)
        for item in job_advancement.get("missing_requirements", [])
        if str(item).strip()
    ]
    unsupported_notes = [
        str(item)
        for item in job_advancement.get("notes", [])
        if str(item).strip()
    ]

    leveling_objective_template = str(
        progression.get("objective_template")
        or "continue deterministic leveling progression safely"
    )
    leveling_target_maps = [
        str(item) for item in progression.get("target_maps", []) if str(item).strip()
    ]

    survival_active = bool(
        assessment.is_dead
        or assessment.is_disconnected
        or assessment.hp_ratio <= 0.35
        or assessment.danger_score >= 0.75
        or assessment.death_risk_score >= 0.75
        or (
            assessment.reconnect_age_s is not None
            and assessment.reconnect_age_s >= 20.0
        )
    )
    job_advancement_active = bool(points_pending or (playbook_supported and playbook_ready))
    opportunistic_active = bool(opportunistic.get("actionable"))
    leveling_active = True

    top_opportunity: dict[str, Any] = {}
    opportunities = (
        opportunistic.get("opportunities")
        if isinstance(opportunistic.get("opportunities"), list)
        else []
    )
    if opportunities and isinstance(opportunities[0], dict):
        top_opportunity = opportunities[0]
    non_actionable_reasons = [
        str(item)
        for item in opportunistic.get("non_actionable_reasons", [])
        if str(item).strip()
    ]

    if playbook_supported and playbook_ready and route_id and target_job:
        job_advancement_objective = (
            f"execute curated job-change playbook {route_id} "
            f"from {map_name} toward {target_job}"
        )
        job_advancement_rationale = (
            "deterministic_priority:job_advancement;knowledge_backed:playbook_ready"
        )
        job_advancement_blockers: list[str] = []
    elif points_pending:
        job_advancement_objective = (
            f"allocate pending progression points safely on {map_name}"
        )
        job_advancement_rationale = (
            "deterministic_priority:job_advancement;knowledge_backed:points_pending"
        )
        job_advancement_blockers = []
    elif playbook_supported and route_id and target_job:
        job_advancement_objective = (
            f"prepare requirements for curated route {route_id} "
            f"toward {target_job} from {map_name}"
        )
        job_advancement_rationale = (
            "deterministic_priority:job_advancement;"
            "knowledge_backed:requirements_pending"
        )
        job_advancement_blockers = list(missing_requirements)
    else:
        job_advancement_objective = (
            f"job advancement route unsupported for current class on {map_name}"
        )
        job_advancement_rationale = (
            "deterministic_priority:job_advancement;knowledge_backed:unsupported_route"
        )
        job_advancement_blockers = list(unsupported_notes)

    if leveling_target_maps:
        leveling_objective = (
            f"{leveling_objective_template} near "
            f"{','.join(leveling_target_maps[:2])}"
        )
    else:
        leveling_objective = f"{leveling_objective_template} on {map_name}"

    if opportunistic_active and top_opportunity:
        candidate_name = str(
            top_opportunity.get("candidate_item_name")
            or top_opportunity.get("candidate_item_id")
            or "upgrade"
        )
        slot_name = str(top_opportunity.get("slot") or "equipment")
        domain_name = str(top_opportunity.get("domain") or "opportunistic_upgrades")
        score_delta = int(top_opportunity.get("score_delta") or 0)
        buy_price = int(top_opportunity.get("buy_price") or 0)
        opportunistic_objective = (
            f"execute curated opportunistic {domain_name} {slot_name} "
            f"upgrade to {candidate_name} "
            f"(score_delta={score_delta}, buy_price={buy_price}) from {map_name}"
        )
        opportunistic_rationale = (
            "deterministic_priority:opportunistic_upgrades;"
            "knowledge_backed:stage4_actionable"
        )
        opportunistic_blockers: list[str] = []
    else:
        opportunistic_objective = (
            "hold opportunistic upgrade posture on "
            f"{map_name} until deterministic evidence is complete"
        )
        opportunistic_rationale = (
            "deterministic_priority:opportunistic_upgrades;"
            "knowledge_backed:stage4_non_actionable"
        )
        opportunistic_blockers = non_actionable_reasons[:8]

    goals: list[GoalDirective] = [
        GoalDirective(
            goal_key=GoalCategory.survival,
            priority_rank=1,
            active=survival_active,
            objective=f"stabilize survival posture safely on {map_name}",
            rationale="deterministic_priority:survival",
            blockers=[],
            metadata={
                "horizon": horizon,
                "hp_ratio": assessment.hp_ratio,
                "danger_score": assessment.danger_score,
                "death_risk_score": assessment.death_risk_score,
            },
        ),
        GoalDirective(
            goal_key=GoalCategory.job_advancement,
            priority_rank=2,
            active=job_advancement_active,
            objective=job_advancement_objective,
            rationale=job_advancement_rationale,
            blockers=job_advancement_blockers,
            metadata={
                "horizon": horizon,
                "skill_points": assessment.skill_points,
                "stat_points": assessment.stat_points,
                "job_exp_ratio": assessment.job_exp_ratio,
                "active_quest_count": assessment.active_quest_count,
                "playbook_supported": playbook_supported,
                "playbook_ready": playbook_ready,
                "route_id": route_id,
                "target_job": target_job,
                "missing_requirements": missing_requirements,
                "job_advancement": dict(job_advancement),
            },
        ),
        GoalDirective(
            goal_key=GoalCategory.opportunistic_upgrades,
            priority_rank=3,
            active=opportunistic_active,
            objective=opportunistic_objective,
            rationale=opportunistic_rationale,
            blockers=opportunistic_blockers,
            metadata={
                "horizon": horizon,
                "knowledge_loaded": bool(opportunistic.get("knowledge_loaded")),
                "supported": bool(opportunistic.get("supported")),
                "status": str(opportunistic.get("status") or "unknown"),
                "actionable": bool(opportunistic.get("actionable")),
                "known_rule_ids": [
                    str(item)
                    for item in opportunistic.get("known_rule_ids", [])
                    if str(item).strip()
                ],
                "overweight_ratio": assessment.overweight_ratio,
                "vendor_exposure": assessment.vendor_exposure,
                "recommended_domain": str(top_opportunity.get("domain") or ""),
                "recommended_opportunity": (
                    dict(top_opportunity)
                    if isinstance(top_opportunity, dict)
                    else {}
                ),
                "non_actionable_reasons": non_actionable_reasons,
            },
        ),
        GoalDirective(
            goal_key=GoalCategory.leveling,
            priority_rank=4,
            active=leveling_active,
            objective=leveling_objective,
            rationale="deterministic_priority:leveling;knowledge_backed:progression_profiles",
            blockers=[],
            metadata={
                "horizon": horizon,
                "base_level": assessment.base_level,
                "job_level": assessment.job_level,
                "base_exp_ratio": assessment.base_exp_ratio,
                "job_exp_ratio": assessment.job_exp_ratio,
                "progression_recommendation": dict(progression),
            },
        ),
    ]

    selected = next((item for item in goals if item.active), goals[0])

    comp = GoalStackComputation()
    comp.goal_stack = goals
    comp.selected_goal = selected
    return comp


def summarize_goal_stack(*, state: GoalStackState) -> dict[str, object]:
    """Produce a compact dict summary of a GoalStackState."""
    return {
        "decision_version": state.decision_version,
        "horizon": state.horizon,
        "selected_goal": state.selected_goal.goal_key.value,
        "selected_objective": state.selected_goal.objective,
        "stack": [
            {
                "goal_key": item.goal_key.value,
                "priority_rank": item.priority_rank,
                "active": item.active,
                "objective": item.objective,
            }
            for item in state.goal_stack
        ],
    }
