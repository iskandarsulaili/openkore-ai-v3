"""Progress tracker — monitors plan progress and detects stuck states."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ai_sidecar.autonomy.pdca_loop import Horizon
from ai_sidecar.planner.schemas import StrategicPlan, TacticalIntentBundle
from ai_sidecar.contracts.state import BotStateSnapshot

logger = logging.getLogger(__name__)


@dataclass
class ProgressEvaluation:
    """Result of evaluating progress against a plan."""

    progress_pct: float = 0.0
    stuck_cycles: int = 0
    completed_objectives: int = 0
    total_objectives: int = 0
    status: str = "unknown"  # "advancing", "stuck", "completed", "idle"


class ProgressTracker:
    """Tracks progress of active plans and detects stuck states."""

    def __init__(self, runtime_state: Any) -> None:
        self._runtime = runtime_state
        self._previous_snapshots: dict[str, BotStateSnapshot] = {}
        self._stuck_history: dict[str, int] = {}  # plan_id -> consecutive stuck cycles

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

        # Track snapshot changes
        prev = self._previous_snapshots.get(plan_id)
        self._previous_snapshots[plan_id] = snapshot

        # Detect if state has changed (advancing) or not (stuck)
        state_changed = self._state_changed(prev, snapshot)

        if not state_changed and prev is not None:
            stuck = self._stuck_history.get(plan_id, 0) + 1
            self._stuck_history[plan_id] = stuck
        else:
            self._stuck_history[plan_id] = 0

        stuck_cycles = self._stuck_history.get(plan_id, 0)

        # Calculate progress percentage
        progress_pct = self._calculate_progress(active_plan, snapshot)

        # Determine status
        if progress_pct >= 1.0:
            status = "completed"
        elif stuck_cycles >= 3:
            status = "stuck"
        elif state_changed:
            status = "advancing"
        else:
            status = "idle"

        return ProgressEvaluation(
            progress_pct=progress_pct,
            stuck_cycles=stuck_cycles,
            status=status,
        )

    def _get_plan_id(self, plan: Any) -> str | None:
        """Extract a stable plan identifier."""
        for attr in ("plan_id", "id", "name", "objective"):
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

        # Compare key state fields
        checks = []

        # Position change
        if hasattr(prev, "position") and hasattr(curr, "position"):
            checks.append(getattr(prev.position, "map", None) != getattr(curr.position, "map", None))
            checks.append(
                getattr(prev.position, "x", None) != getattr(curr.position, "x", None)
                or getattr(prev.position, "y", None) != getattr(curr.position, "y", None)
            )

        # HP/SP change
        if hasattr(prev, "resources") and hasattr(curr, "resources"):
            checks.append(
                getattr(prev.resources, "hp", None) != getattr(curr.resources, "hp", None)
                or getattr(prev.resources, "sp", None) != getattr(curr.resources, "sp", None)
            )

        # Level change
        if hasattr(prev, "progression") and hasattr(curr, "progression"):
            checks.append(
                getattr(prev.progression, "level", None) != getattr(curr.progression, "level", None)
            )

        # Combat state change
        if hasattr(prev, "combat") and hasattr(curr, "combat"):
            checks.append(
                getattr(prev.combat, "in_combat", None) != getattr(curr.combat, "in_combat", None)
                or getattr(prev.combat, "monster_hp_pct", None) != getattr(curr.combat, "monster_hp_pct", None)
            )

        # Zeny change
        if hasattr(prev, "economy") and hasattr(curr, "economy"):
            checks.append(
                getattr(prev.economy, "zeny", None) != getattr(curr.economy, "zeny", None)
            )

        return any(checks)

    def _calculate_progress(
        self,
        plan: Any,
        snapshot: BotStateSnapshot | None,
    ) -> float:
        """Estimate progress toward plan completion (0.0 to 1.0)."""
        if snapshot is None:
            return 0.0

        # Try to extract target objectives from the plan
        objectives = getattr(plan, "objectives", None) or getattr(plan, "goals", None) or []
        if not objectives:
            return 0.5  # Unknown progress

        completed = 0
        total = len(objectives)

        for obj in objectives:
            if self._objective_completed(obj, snapshot):
                completed += 1

        return completed / max(total, 1)

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
            current_level = getattr(getattr(snapshot, "progression", None), "level", 0)
            if target_level and current_level >= target_level:
                return True

        # Zeny-based objectives
        if "zeny" in desc or "money" in desc or "gold" in desc:
            target_zeny = self._extract_number(desc)
            current_zeny = getattr(getattr(snapshot, "economy", None), "zeny", 0)
            if target_zeny and current_zeny >= target_zeny:
                return True

        # Map-based objectives
        if "map" in desc or "go to" in desc or "reach" in desc:
            target_map = self._extract_map_name(desc)
            current_map = getattr(getattr(snapshot, "position", None), "map", "")
            if target_map and target_map in current_map:
                return True

        # Quest-based objectives
        if "quest" in desc:
            quest_progress = getattr(getattr(snapshot, "progression", None), "quests", {}) or {}
            if any(q.get("completed") for q in quest_progress.values()):
                return True

        return False

    def _extract_number(self, text: str) -> int | None:
        """Extract the first number from text."""
        import re

        match = re.search(r"(\d+)", text)
        return int(match.group(1)) if match else None

    def _extract_map_name(self, text: str) -> str | None:
        """Extract a potential map name from text."""
        import re

        # Map names are typically like "prontera", "morocc", "payon"
        match = re.search(r"\b([a-z]{3,20})\b", text)
        return match.group(1) if match else None
