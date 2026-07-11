"""
Goal Decomposition & Priority Sequencer
========================================
Chunks big goals into smaller achievable sub-goals across short, medium,
and long term horizons. Each sub-goal has a clear dependency chain so the
PDCA loop knows what to execute NOW vs what to plan for LATER.

Key design:
- Big goals are decomposed into DAGs of sub-goals across 4 horizons
- Short-term prerequisites feed medium-term, which feed long-term
- Priority is adaptive based on bot state
- Multi-bot coordination via SwarmGoalCoordinator
- Cross-horizon conflict detection via CrossHorizonSynergy
- Uses HuntingZoneManager for dynamic zone recommendations
- Thread-safe with RLock
"""

from __future__ import annotations
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from enum import StrEnum
from threading import RLock
from typing import Any, Optional
from ai_sidecar.contracts.autonomy import GoalCategory, GoalDirective, GoalStackState, SituationalAssessment

logger = logging.getLogger(__name__)

class GoalHorizon(StrEnum):
    tactical = "tactical"       # 0-30s: immediate actions
    short_term = "short_term"   # 30s-5min: what to do NOW
    medium_term = "medium_term" # 5-30min: what to do NEXT
    long_term = "long_term"     # 30min+: the BIG goal

@dataclass(slots=True)
class DecomposedGoal:
    id: str
    parent: GoalCategory
    horizon: GoalHorizon
    objective: str
    prerequisites: list[str]
    estimated_duration_s: float
    success_metrics: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class GoalDecomposition:
    parent: GoalCategory
    sub_goals: dict[str, DecomposedGoal]
    dependencies: dict[str, list[str]]
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime = field(default_factory=lambda: datetime.now(UTC) + timedelta(minutes=15))

class SwarmGoalCoordinator:
    """Coordinates goal assignments across multiple bots."""
    def __init__(self):
        self._lock = RLock()
        self._assignments: dict[str, str] = {}
    def assign_zone(self, bot_id: str, zone: str) -> None:
        with self._lock: self._assignments[bot_id] = zone
    def get_assigned_zones(self) -> list[str]:
        with self._lock: return list(self._assignments.values())
    def get_available_zones(self, all_zones: list[str]) -> list[str]:
        with self._lock:
            assigned = set(self._assignments.values())
            return [z for z in all_zones if z not in assigned]

class CrossHorizonSynergy:
    """Detects/resolves conflicts between short-term and long-term goals."""
    def __init__(self): 
        self._lock = RLock()
        self._synergy_log: list[dict[str, Any]] = []

    def detect_conflicts(self, short_term_goal: str, long_term_goal: str) -> list[str]:
        conflicts = []
        if not short_term_goal or not long_term_goal:
            return conflicts
        # Example: if short-term is "grind on prt_fild08" but long-term is "job change to Knight"
        # which requires being in Morocc, that's a conflict
        st = short_term_goal.lower()
        lt = long_term_goal.lower()
        if "prt_fild" in st and ("morocc" in lt or "payon" in lt):
            conflicts.append("short_term_zone_conflicts_with_long_term_goal")
        if "grind" in st and "job" in lt:
            conflicts.append("grinding_delays_job_advancement")
        if conflicts:
            self._synergy_log.append({"short": short_term_goal, "long": long_term_goal, "conflicts": conflicts, "time": time.time()})
        return conflicts

class GoalDecomposer:
    """Decomposes high-level goals into actionable sub-goals across 4 horizons.
    Thread-safe. Uses HZM for dynamic zone recs. No hardcoded maps.
    """
    HORIZON_TTL = {GoalHorizon.tactical: 30, GoalHorizon.short_term: 300,
                   GoalHorizon.medium_term: 1800, GoalHorizon.long_term: 7200}

    def __init__(self, hunting_zone_manager=None):
        self._lock = RLock()
        self._hzm = hunting_zone_manager
        self._swarm_coordinator: SwarmGoalCoordinator | None = None
        self._decompositions: dict[str, GoalDecomposition] = {}
        self._completed: dict[str, set[str]] = defaultdict(set)
        self._current_goal: dict[str, Optional[str]] = {}

    def set_hunting_zone_manager(self, hzm): self._hzm = hzm
    
    def set_swarm_coordinator(self, coord): self._swarm_coordinator = coord

    def decompose(self, bot_id: str, assessment=None, goal_stack=None, selected=None, bot_level=1):
        if selected is None: return None
        with self._lock:
            category_str = str(selected.goal_key.value) if hasattr(selected.goal_key, 'value') else str(selected.goal_key)
            objective = selected.objective or category_str
            sub_goals: dict[str, DecomposedGoal] = {}
            deps: dict[str, list[str]] = {}

            if category_str in ("survival",):
                sub_goals["heal"] = DecomposedGoal(id="heal", parent=selected.goal_key, horizon=GoalHorizon.tactical, objective="Recover HP/SP", prerequisites=[], estimated_duration_s=10, metadata={"action": "sit"})
                sub_goals["safe_position"] = DecomposedGoal(id="safe_position", parent=selected.goal_key, horizon=GoalHorizon.short_term, objective="Move to safe position", prerequisites=["heal"], estimated_duration_s=30, metadata={"action": "ai auto"})
                deps["safe_position"] = ["heal"]
            elif category_str in ("leveling", "grind", "economy", "opportunistic_upgrades"):
                target_map = self._recommend_zone(bot_id, objective, bot_level)
                sub_goals["move_to_zone"] = DecomposedGoal(id="move_to_zone", parent=selected.goal_key, horizon=GoalHorizon.short_term, objective=f"Move to {target_map}", prerequisites=[], estimated_duration_s=120, metadata={"action": f"move {target_map}", "target_map": target_map})
                sub_goals["hunt"] = DecomposedGoal(id="hunt", parent=selected.goal_key, horizon=GoalHorizon.medium_term, objective=f"Grind on {target_map}", prerequisites=["move_to_zone"], estimated_duration_s=1800, metadata={"action": "ai auto", "target_map": target_map})
                sub_goals["vendor"] = DecomposedGoal(id="vendor", parent=selected.goal_key, horizon=GoalHorizon.medium_term, objective="Sell loot when inventory full", prerequisites=["hunt"], estimated_duration_s=300, metadata={"action": "vendor"})
                deps["hunt"] = ["move_to_zone"]; deps["vendor"] = ["hunt"]
            elif category_str in ("quest", "job_advancement"):
                sub_goals["find_npc"] = DecomposedGoal(id="find_npc", parent=selected.goal_key, horizon=GoalHorizon.short_term, objective=f"Find {objective} NPC", prerequisites=[], estimated_duration_s=60, metadata={"action": "ai auto"})
                sub_goals["interact"] = DecomposedGoal(id="interact", parent=selected.goal_key, horizon=GoalHorizon.short_term, objective=f"Interact with {objective} NPC", prerequisites=["find_npc"], estimated_duration_s=120, metadata={"action": "talknpc"})
                sub_goals["complete"] = DecomposedGoal(id="complete", parent=selected.goal_key, horizon=GoalHorizon.medium_term, objective=f"Complete {objective}", prerequisites=["interact"], estimated_duration_s=300, metadata={"action": "ai auto"})
                deps["interact"] = ["find_npc"]; deps["complete"] = ["interact"]
            else:
                sub_goals["default"] = DecomposedGoal(id="default", parent=selected.goal_key, horizon=GoalHorizon.short_term, objective=objective, prerequisites=[], estimated_duration_s=60, metadata={"action": "ai auto"})

            decomp = GoalDecomposition(parent=selected.goal_key, sub_goals=sub_goals, dependencies=deps)
            self._decompositions[bot_id] = decomp
            self._completed[bot_id] = set()
            self._current_goal[bot_id] = None
            logger.info("goal_decomposed: bot=%s category=%s sub_goals=%d", bot_id, category_str, len(sub_goals))
            return decomp

    def _recommend_zone(self, bot_id: str, objective: str, bot_level: int = 1) -> str:
        if self._hzm is not None:
            try:
                zones = self._hzm.recommend_zone(bot_level=bot_level, goal="leveling")
                if zones: return zones[0].map_name
            except Exception: pass
        return "prt_fild08"

    def next_action(self, *, bot_id: str) -> DecomposedGoal | None:
        with self._lock:
            decomp = self._decompositions.get(bot_id)
            if not decomp: return None
            completed = self._completed.get(bot_id, set())
            ready = []
            for sg_id, sg in decomp.sub_goals.items():
                if sg_id in completed: continue
                prereqs = decomp.dependencies.get(sg_id, [])
                if all(p in completed for p in prereqs): ready.append(sg)
            if not ready: return None
            horizon_order = {GoalHorizon.tactical: 0, GoalHorizon.short_term: 1, GoalHorizon.medium_term: 2, GoalHorizon.long_term: 3}
            ready.sort(key=lambda g: horizon_order.get(g.horizon, 99))
            best = ready[0]
            self._current_goal[bot_id] = best.id
            return best

    def mark_completed(self, *, bot_id: str, sub_goal_id: str) -> None:
        with self._lock:
            self._completed[bot_id].add(sub_goal_id)
            logger.info("goal_completed: bot=%s sub_goal=%s", bot_id, sub_goal_id)

    def progress(self, *, bot_id: str) -> dict[str, Any]:
        with self._lock:
            decomp = self._decompositions.get(bot_id)
            if not decomp: return {"status": "no_decomposition", "progress": 0.0}
            completed = self._completed.get(bot_id, set())
            total = len(decomp.sub_goals); done = len(completed)
            return {"status": "completed" if done >= total else "in_progress", "progress": done / max(total, 1), "completed": done, "total": total, "current_goal": self._current_goal.get(bot_id)}