"""Progress tracker — monitors plan progress and detects stuck states."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_sidecar.autonomy.pdca_loop import Horizon
from ai_sidecar.contracts.state import BotStateSnapshot
from ai_sidecar.planner.schemas import StrategicPlan, TacticalIntentBundle

logger = logging.getLogger(__name__)


@dataclass
class ProgressEvaluation:
    """Result of evaluating progress against a plan."""

    progress_pct: float = 0.0
    stuck_cycles: int = 0
    completed_objectives: int = 0
    total_objectives: int = 0
    status: str = "unknown"  # "advancing", "stuck", "completed", "idle"
    reasons: list[str] = field(default_factory=list)
    map_dwell_cycles: int = 0
    death_loops: int = 0
    route_churn_cycles: int = 0
    objective_age_cycles: int = 0
    stale_seconds: float = 0.0
    force_replan_hint: bool = False


class ProgressTracker:
    """Tracks progress of active plans and detects stuck states."""

    def __init__(self, runtime_state: Any) -> None:
        self._runtime = runtime_state
        self._previous_snapshots: dict[str, BotStateSnapshot] = {}
        self._stuck_history: dict[str, int] = {}  # plan_id -> consecutive stuck cycles
        self._last_progress: dict[str, float] = {}
        self._last_progress_at: dict[str, float] = {}
        self._map_dwell_history: dict[str, int] = {}
        self._route_churn_history: dict[str, int] = {}
        self._death_loop_history: dict[str, int] = {}
        self._last_death_at: dict[str, float] = {}
        self._objective_age_history: dict[str, int] = {}

    def evaluate(
        self,
        horizon: "Horizon",
        active_plan: StrategicPlan | TacticalIntentBundle | None,
        snapshot: BotStateSnapshot | None,
    ) -> ProgressEvaluation:
        """Evaluate progress of the active plan against the current snapshot."""
        if active_plan is None:
            return ProgressEvaluation(status="idle")

        plan_id = self._get_plan_id(active_plan)
        if plan_id is None:
            return ProgressEvaluation(status="unknown")

        now = time.time()

        # Track snapshot changes
        prev = self._previous_snapshots.get(plan_id)
        self._previous_snapshots[plan_id] = snapshot

        # Compute progress first (includes multi-objective signal scoring).
        progress_pct, completed_objectives, total_objectives = self._calculate_progress(
            active_plan,
            snapshot,
            prev=prev,
        )

        prev_progress = self._last_progress.get(plan_id, progress_pct)
        progress_delta = progress_pct - prev_progress
        self._last_progress[plan_id] = progress_pct

        # Detect if state has changed (advancing) or not (stuck).
        state_changed = self._state_changed(prev, snapshot)

        if state_changed or progress_delta > 0.005:
            self._stuck_history[plan_id] = 0
            self._last_progress_at[plan_id] = now
        elif prev is not None:
            stuck = self._stuck_history.get(plan_id, 0) + 1
            self._stuck_history[plan_id] = stuck
        else:
            self._stuck_history[plan_id] = 0
            self._last_progress_at.setdefault(plan_id, now)

        stuck_cycles = self._stuck_history.get(plan_id, 0)
        stale_seconds = max(0.0, now - self._last_progress_at.get(plan_id, now))

        map_dwell_cycles = self._map_dwell_cycles(plan_id=plan_id, prev=prev, curr=snapshot)
        route_churn_cycles = self._route_churn_cycles(plan_id=plan_id, prev=prev, curr=snapshot)
        death_loops = self._death_loop_cycles(plan_id=plan_id, prev=prev, curr=snapshot, now=now)
        objective_age_cycles = self._objective_age_cycles(
            plan_id=plan_id,
            progress_delta=progress_delta,
            progress_pct=progress_pct,
        )

        reasons: list[str] = []
        stale_threshold_s = self._policy_float("stale_plan_threshold_s", 60.0)
        objective_age_threshold = self._policy_int("objective_max_age_cycles", 6)

        if map_dwell_cycles >= 3:
            reasons.append("map_dwell_no_gain")
        if route_churn_cycles >= 2:
            reasons.append("route_churn_no_position_gain")
        if death_loops >= 2:
            reasons.append("death_loop_detected")
        if stale_seconds >= stale_threshold_s:
            reasons.append("stale_progress")
        if objective_age_cycles >= objective_age_threshold:
            reasons.append("objective_aged_out")
        if stuck_cycles >= 3:
            reasons.append("stuck_cycles")

        force_replan_hint = progress_pct < 1.0 and bool(reasons)
        if progress_pct >= 1.0:
            status = "completed"
        elif force_replan_hint:
            status = "stuck"
        elif state_changed or progress_delta > 0.0:
            status = "advancing"
        else:
            status = "idle"

        return ProgressEvaluation(
            progress_pct=progress_pct,
            stuck_cycles=stuck_cycles,
            completed_objectives=completed_objectives,
            total_objectives=total_objectives,
            status=status,
            reasons=reasons,
            map_dwell_cycles=map_dwell_cycles,
            death_loops=death_loops,
            route_churn_cycles=route_churn_cycles,
            objective_age_cycles=objective_age_cycles,
            stale_seconds=stale_seconds,
            force_replan_hint=force_replan_hint,
        )

    def _get_plan_id(self, plan: Any) -> str | None:
        """Extract a stable plan identifier."""
        for attr in ("plan_id", "bundle_id", "id", "name", "objective"):
            val = getattr(plan, attr, None)
            if val is not None:
                return str(val)
        return None

    def _state_changed(
        self,
        prev: BotStateSnapshot | None,
        curr: BotStateSnapshot | None,
    ) -> bool:
        """Detect if the bot state has meaningfully changed."""
        if prev is None or curr is None:
            return prev != curr

        if getattr(prev.position, "map", None) != getattr(curr.position, "map", None):
            return True
        if (
            getattr(prev.position, "x", None) != getattr(curr.position, "x", None)
            or getattr(prev.position, "y", None) != getattr(curr.position, "y", None)
        ):
            return True
        if (
            getattr(prev.vitals, "hp", None) != getattr(curr.vitals, "hp", None)
            or getattr(prev.vitals, "sp", None) != getattr(curr.vitals, "sp", None)
        ):
            return True
        if (
            getattr(prev.inventory, "zeny", None) != getattr(curr.inventory, "zeny", None)
            or getattr(prev.inventory, "item_count", None) != getattr(curr.inventory, "item_count", None)
        ):
            return True
        if (
            getattr(prev.progression, "base_level", None) != getattr(curr.progression, "base_level", None)
            or getattr(prev.progression, "base_exp", None) != getattr(curr.progression, "base_exp", None)
            or getattr(prev.progression, "job_level", None) != getattr(curr.progression, "job_level", None)
            or getattr(prev.progression, "job_exp", None) != getattr(curr.progression, "job_exp", None)
        ):
            return True
        if (
            getattr(prev.combat, "is_in_combat", None) != getattr(curr.combat, "is_in_combat", None)
            or getattr(prev.combat, "target_id", None) != getattr(curr.combat, "target_id", None)
            or getattr(prev.combat, "ai_sequence", None) != getattr(curr.combat, "ai_sequence", None)
        ):
            return True

        return abs(self._quest_completion_ratio(curr) - self._quest_completion_ratio(prev)) > 1e-6

    def _calculate_progress(
        self,
        plan: Any,
        snapshot: BotStateSnapshot | None,
        *,
        prev: BotStateSnapshot | None,
    ) -> tuple[float, int, int]:
        """Estimate progress toward plan completion (0.0 to 1.0)."""
        if snapshot is None:
            return 0.0, 0, 0

        objectives = self._extract_objectives(plan)
        total = len(objectives)
        completed = sum(1 for obj in objectives if self._objective_completed(obj, snapshot))
        objective_score = (completed / total) if total > 0 else 0.0

        signal_score = self._signal_progress_score(prev=prev, curr=snapshot)

        if total > 0:
            blended = (0.70 * objective_score) + (0.30 * signal_score)
        else:
            blended = signal_score if signal_score > 0.0 else 0.5

        return max(0.0, min(1.0, blended)), completed, total

    def _extract_objectives(self, plan: Any) -> list[str]:
        if plan is None:
            return []

        extracted: list[str] = []
        raw_objectives = getattr(plan, "objectives", None) or getattr(plan, "goals", None) or []
        for item in raw_objectives:
            text = str(getattr(item, "description", None) or getattr(item, "name", None) or item).strip()
            if text:
                extracted.append(text)

        if isinstance(plan, StrategicPlan):
            plan_objective = str(getattr(plan, "objective", "") or "").strip()
            if plan_objective:
                extracted.append(plan_objective)
            for step in list(getattr(plan, "steps", []) or []):
                desc = str(getattr(step, "description", "") or "").strip()
                if desc:
                    extracted.append(desc)
        elif isinstance(plan, TacticalIntentBundle):
            for intent in list(getattr(plan, "intents", []) or []):
                desc = str(getattr(intent, "objective", "") or "").strip()
                if desc:
                    extracted.append(desc)
            for note in list(getattr(plan, "notes", []) or []):
                desc = str(note).strip()
                if desc:
                    extracted.append(desc)

        deduped: list[str] = []
        seen: set[str] = set()
        for item in extracted[:32]:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _signal_progress_score(self, *, prev: BotStateSnapshot | None, curr: BotStateSnapshot) -> float:
        if prev is None:
            return 0.0

        score = 0.0

        zeny_delta = int(curr.inventory.zeny or 0) - int(prev.inventory.zeny or 0)
        if zeny_delta > 0:
            score += 0.25

        base_exp_delta = int(curr.progression.base_exp or 0) - int(prev.progression.base_exp or 0)
        job_exp_delta = int(curr.progression.job_exp or 0) - int(prev.progression.job_exp or 0)
        if base_exp_delta > 0 or job_exp_delta > 0:
            score += 0.35

        quest_delta = self._quest_completion_ratio(curr) - self._quest_completion_ratio(prev)
        if quest_delta > 0.0:
            score += 0.25

        if self._distance(prev, curr) >= 3.0:
            score += 0.15

        return max(0.0, min(1.0, score))

    def _objective_completed(self, objective: Any, snapshot: BotStateSnapshot) -> bool:
        """Check if a single objective is completed based on current state."""
        desc = str(
            getattr(objective, "description", None)
            or getattr(objective, "name", None)
            or objective
        ).lower()

        # Level-based objectives
        if "level" in desc or "lvl" in desc:
            target_level = self._extract_number(desc)
            current_level = max(
                int(getattr(snapshot.progression, "base_level", 0) or 0),
                int(getattr(snapshot.progression, "job_level", 0) or 0),
            )
            if target_level and current_level >= target_level:
                return True

        # Zeny-based objectives
        if "zeny" in desc or "money" in desc or "gold" in desc:
            target_zeny = self._extract_number(desc)
            current_zeny = int(getattr(snapshot.inventory, "zeny", 0) or 0)
            if target_zeny and current_zeny >= target_zeny:
                return True

        # Map-based objectives
        if "map" in desc or "go to" in desc or "reach" in desc:
            target_map = self._extract_map_name(desc)
            current_map = str(getattr(snapshot.position, "map", "") or "")
            if target_map and target_map in current_map:
                return True

        # Quest-based objectives
        if "quest" in desc:
            if self._quest_completion_ratio(snapshot) >= 1.0:
                return True
            for quest in list(getattr(snapshot, "quests", []) or []):
                state = str(getattr(quest, "state", "") or "").lower()
                if state in {"completed", "done", "finished", "success", "turn_in"}:
                    return True

        # Recovery-style objectives.
        if "recover" in desc or "respawn" in desc or "survive" in desc or "heal" in desc:
            hp = float(getattr(snapshot.vitals, "hp", 0) or 0)
            hp_max = float(getattr(snapshot.vitals, "hp_max", 0) or 0)
            if hp_max > 0 and (hp / hp_max) >= 0.85:
                return True

        # Economy pressure objectives.
        if "economy" in desc or "inventory" in desc or "overweight" in desc:
            if self._overweight_ratio(snapshot) < 0.85:
                return True

        return False

    def _quest_completion_ratio(self, snapshot: BotStateSnapshot) -> float:
        quests = list(getattr(snapshot, "quests", []) or [])
        if not quests:
            return 0.0

        units_total = 0
        units_done = 0.0
        completed_quests = 0

        for quest in quests:
            state = str(getattr(quest, "state", "") or "").strip().lower()
            if state in {"completed", "done", "finished", "success", "turn_in"}:
                completed_quests += 1

            objectives = list(getattr(quest, "objectives", []) or [])
            if not objectives:
                continue

            for obj in objectives:
                units_total += 1
                status = str(getattr(obj, "status", "") or "").strip().lower()
                current = getattr(obj, "current", None)
                target = getattr(obj, "target", None)

                if status in {"complete", "completed", "done", "finished", "success"}:
                    units_done += 1.0
                    continue
                if isinstance(current, int) and isinstance(target, int) and target > 0:
                    units_done += max(0.0, min(1.0, float(current) / float(target)))

        if units_total > 0:
            return max(0.0, min(1.0, units_done / float(units_total)))
        return max(0.0, min(1.0, float(completed_quests) / float(len(quests))))

    def _map_dwell_cycles(
        self,
        *,
        plan_id: str,
        prev: BotStateSnapshot | None,
        curr: BotStateSnapshot | None,
    ) -> int:
        if prev is None or curr is None:
            self._map_dwell_history[plan_id] = 0
            return 0

        prev_map = str(getattr(prev.position, "map", "") or "")
        curr_map = str(getattr(curr.position, "map", "") or "")
        if not curr_map or prev_map != curr_map:
            self._map_dwell_history[plan_id] = 0
            return 0

        if self._resource_gain(prev=prev, curr=curr):
            self._map_dwell_history[plan_id] = 0
            return 0

        value = self._map_dwell_history.get(plan_id, 0) + 1
        self._map_dwell_history[plan_id] = value
        return value

    def _route_churn_cycles(
        self,
        *,
        plan_id: str,
        prev: BotStateSnapshot | None,
        curr: BotStateSnapshot | None,
    ) -> int:
        if prev is None or curr is None:
            self._route_churn_history[plan_id] = 0
            return 0

        ai_seq = str(getattr(curr.combat, "ai_sequence", "") or "").strip().lower()
        if not (ai_seq.startswith("route") or ai_seq.startswith("move")):
            self._route_churn_history[plan_id] = 0
            return 0

        same_map = str(getattr(prev.position, "map", "") or "") == str(getattr(curr.position, "map", "") or "")
        if not same_map:
            self._route_churn_history[plan_id] = 0
            return 0

        moved = self._distance(prev, curr)
        if moved >= 2.0:
            self._route_churn_history[plan_id] = 0
            return 0

        value = self._route_churn_history.get(plan_id, 0) + 1
        self._route_churn_history[plan_id] = value
        return value

    def _death_loop_cycles(
        self,
        *,
        plan_id: str,
        prev: BotStateSnapshot | None,
        curr: BotStateSnapshot | None,
        now: float,
    ) -> int:
        if curr is None:
            self._death_loop_history[plan_id] = 0
            return 0

        prev_dead = self._is_dead(prev)
        curr_dead = self._is_dead(curr)
        current_value = self._death_loop_history.get(plan_id, 0)
        recovery_cooldown_s = max(1.0, self._policy_float("death_recovery_cooldown_s", 15.0))

        if curr_dead and not prev_dead:
            last_death_at = self._last_death_at.get(plan_id)
            if last_death_at is not None and (now - last_death_at) <= recovery_cooldown_s:
                current_value += 1
            else:
                current_value = 1
            self._last_death_at[plan_id] = now
            self._death_loop_history[plan_id] = current_value
            return current_value

        last_death_at = self._last_death_at.get(plan_id)
        if last_death_at is not None and (now - last_death_at) > (recovery_cooldown_s * 6.0):
            self._death_loop_history[plan_id] = 0
            return 0

        return current_value

    def _objective_age_cycles(self, *, plan_id: str, progress_delta: float, progress_pct: float) -> int:
        if progress_pct >= 1.0 or progress_delta > 0.01:
            self._objective_age_history[plan_id] = 0
            return 0
        value = self._objective_age_history.get(plan_id, 0) + 1
        self._objective_age_history[plan_id] = value
        return value

    def _resource_gain(self, *, prev: BotStateSnapshot, curr: BotStateSnapshot) -> bool:
        zeny_delta = int(curr.inventory.zeny or 0) - int(prev.inventory.zeny or 0)
        base_exp_delta = int(curr.progression.base_exp or 0) - int(prev.progression.base_exp or 0)
        job_exp_delta = int(curr.progression.job_exp or 0) - int(prev.progression.job_exp or 0)
        quest_delta = self._quest_completion_ratio(curr) - self._quest_completion_ratio(prev)
        return zeny_delta > 0 or base_exp_delta > 0 or job_exp_delta > 0 or quest_delta > 0.0

    def _distance(self, prev: BotStateSnapshot, curr: BotStateSnapshot) -> float:
        x1 = getattr(prev.position, "x", None)
        y1 = getattr(prev.position, "y", None)
        x2 = getattr(curr.position, "x", None)
        y2 = getattr(curr.position, "y", None)
        if not all(isinstance(item, int) for item in (x1, y1, x2, y2)):
            return 0.0
        dx = float(int(x2) - int(x1))
        dy = float(int(y2) - int(y1))
        return (dx * dx + dy * dy) ** 0.5

    def _is_dead(self, snapshot: BotStateSnapshot | None) -> bool:
        if snapshot is None:
            return False

        hp = int(getattr(snapshot.vitals, "hp", 0) or 0)
        if hp <= 0:
            return True

        raw = getattr(snapshot, "raw", {})
        if isinstance(raw, dict):
            if bool(raw.get("is_dead")):
                return True
            status = str(raw.get("status") or raw.get("state") or raw.get("liveness_state") or "").strip().lower()
            if status in {"dead", "death", "died", "fainted"}:
                return True
        return False

    def _overweight_ratio(self, snapshot: BotStateSnapshot) -> float:
        weight = getattr(snapshot.vitals, "weight", None)
        weight_max = getattr(snapshot.vitals, "weight_max", None)
        if not isinstance(weight, int) or not isinstance(weight_max, int) or weight_max <= 0:
            return 0.0
        return max(0.0, min(2.0, float(weight) / float(weight_max)))

    def _policy_int(self, key: str, default: int) -> int:
        policy = getattr(self._runtime, "autonomy_policy", {})
        if isinstance(policy, dict):
            try:
                return int(policy.get(key, default))
            except (TypeError, ValueError):
                return default
        return default

    def _policy_float(self, key: str, default: float) -> float:
        policy = getattr(self._runtime, "autonomy_policy", {})
        if isinstance(policy, dict):
            try:
                return float(policy.get(key, default))
            except (TypeError, ValueError):
                return default
        return default

    def _extract_number(self, text: str) -> int | None:
        """Extract the first number from text."""
        match = re.search(r"(\d+)", text)
        return int(match.group(1)) if match else None

    def _extract_map_name(self, text: str) -> str | None:
        """Extract a potential map name from text."""
        # Prefer RO-like map tokens: prt_fild08, pay_dun01, etc.
        match = re.search(r"\b([a-z]{2,}_[a-z0-9]{2,}|[a-z]{4,}\d{0,2})\b", text)
        return match.group(1) if match else None
