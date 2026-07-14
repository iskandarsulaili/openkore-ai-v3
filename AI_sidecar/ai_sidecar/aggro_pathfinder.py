"""
Aggro Pathfinder — Weighted A* pathfinding with dynamic aggro avoidance.

The core problem: bots wander randomly because they don't know where portals are
or how to avoid dangerous areas. This module solves both problems.

Key features:
1. Portal-aware routing: walks directly to portals, not random waypoints
2. Aggro avoidance: prefers paths through safe maps, avoids high-aggro zones
3. Dynamic re-routing: if aggro is detected, recalculates a safer path
4. Bridge integration: produces move commands the bridge can execute
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from ai_sidecar.portal_knowledge import get_portal_knowledge, Portal
from ai_sidecar.pathfinding_engine import (
    get_pathfinding_engine, PathfindingEngine,
    NavigationPlan, NavigationStep,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AggroState:
    """Current aggro state for a bot."""
    bot_id: str
    current_map: str = ""
    current_x: int = 0
    current_y: int = 0
    aggro_count: int = 0
    aggro_threat: float = 0.0  # 0.0-1.0
    last_aggro_time: float = 0.0
    in_combat: bool = False
    hp_ratio: float = 1.0
    sp_ratio: float = 1.0
    is_dead: bool = False
    is_stuck: bool = False
    stuck_since: float = 0.0
    last_position: tuple[int, int] = (0, 0)
    position_unchanged_count: int = 0


@dataclass(slots=True)
class AggroPathResult:
    """Result of an aggro-aware pathfinding request."""
    plan: NavigationPlan | None = None
    commands: list[dict[str, Any]] = field(default_factory=list)
    safe: bool = True
    reason: str = ""
    needs_reroute: bool = False
    emergency_action: str = ""  # flee | sit | teleport | respawn


class AggroPathfinder:
    """Weighted A* pathfinder with dynamic aggro avoidance.

    Thread-safe. Tracks per-bot aggro state and produces safe navigation plans.
    """

    # Danger thresholds
    DANGER_LOW: float = 0.15
    DANGER_MEDIUM: float = 0.30
    DANGER_HIGH: float = 0.50
    DANGER_DEADLY: float = 0.70

    # Aggro thresholds
    AGGR0_SAFE: int = 0
    AGGR0_CAUTION: int = 2
    AGGR0_DANGER: int = 4
    AGGR0_DEADLY: int = 6

    # Stuck detection
    STUCK_THRESHOLD_COUNT: int = 5
    STUCK_THRESHOLD_SECONDS: float = 15.0

    def __init__(self) -> None:
        self._lock = RLock()
        self._portal_knowledge = get_portal_knowledge()
        self._pathfinding = get_pathfinding_engine()
        self._bot_states: dict[str, AggroState] = {}
        # Map danger levels (populated from knowledge or runtime data)
        self._map_danger: dict[str, float] = {
            # Towns are always safe
            "prontera": 0.0, "geffen": 0.0, "payon": 0.0, "morocc": 0.0,
            "aldebaran": 0.0, "yuno": 0.0, "izlude": 0.0, "xmas": 0.0,
            "comodo": 0.0, "amatsu": 0.0,
            # Low-level fields are safe
            "prt_fild01": 0.05, "prt_fild02": 0.05, "prt_fild03": 0.10,
            "prt_fild04": 0.10, "prt_fild05": 0.15,
            # Medium-level fields
            "prt_fild06": 0.20, "prt_fild07": 0.25, "prt_fild08": 0.30,
            "prt_fild09": 0.35, "prt_fild10": 0.40, "prt_fild11": 0.45,
            # Morocc fields
            "moc_fild01": 0.15, "moc_fild02": 0.20, "moc_fild03": 0.25,
            "moc_fild17": 0.20, "moc_fild18": 0.25, "moc_fild19": 0.30,
            "moc_fild20": 0.35, "moc_fild21": 0.40, "moc_fild22": 0.45,
            # Geffen fields
            "gef_fild00": 0.15, "gef_fild01": 0.20, "gef_fild02": 0.25,
            "gef_fild03": 0.30, "gef_fild04": 0.35, "gef_fild05": 0.40,
            # Payon fields
            "pay_fild01": 0.15, "pay_fild02": 0.20, "pay_fild03": 0.25,
            "pay_fild04": 0.30, "pay_fild05": 0.35, "pay_fild06": 0.40,
            "pay_fild07": 0.45, "pay_fild08": 0.50,
            # Dungeons
            "pay_dun00": 0.30, "pay_dun01": 0.40, "pay_dun02": 0.50,
            "pay_dun03": 0.60, "pay_dun04": 0.70,
            "gef_dun00": 0.30, "gef_dun01": 0.40, "gef_dun02": 0.50,
            "gef_dun03": 0.60,
            "moc_dun01": 0.35, "moc_dun02": 0.45, "moc_dun03": 0.55,
            "moc_dun04": 0.65,
            "orcsdun01": 0.40, "orcsdun02": 0.55,
            "iz_dun00": 0.25, "iz_dun01": 0.35, "iz_dun02": 0.45,
            "iz_dun03": 0.55, "iz_dun04": 0.65,
            # High-level
            "mjolnir_04": 0.20,
            "xmas_fild01": 0.30,
            "ice_dun01": 0.40, "ice_dun02": 0.50, "ice_dun03": 0.60,
            "yuno_fild01": 0.30, "yuno_fild02": 0.35, "yuno_fild03": 0.40,
            "ama_fild01": 0.40,
            "comodo_fild01": 0.30,
        }

    # ── Bot State Management ─────────────────────────────────────────

    def register_bot(self, bot_id: str) -> None:
        """Register a bot for aggro tracking."""
        with self._lock:
            if bot_id not in self._bot_states:
                self._bot_states[bot_id] = AggroState(bot_id=bot_id)

    def update_state(self, bot_id: str, **kwargs: Any) -> None:
        """Update a bot's aggro state."""
        with self._lock:
            state = self._bot_states.get(bot_id)
            if state is None:
                state = AggroState(bot_id=bot_id)
                self._bot_states[bot_id] = state
            for key, value in kwargs.items():
                if hasattr(state, key):
                    setattr(state, key, value)

            # Stuck detection
            if "current_x" in kwargs and "current_y" in kwargs:
                new_pos = (kwargs["current_x"], kwargs["current_y"])
                if new_pos == state.last_position:
                    state.position_unchanged_count += 1
                    if state.position_unchanged_count >= self.STUCK_THRESHOLD_COUNT:
                        if not state.is_stuck:
                            state.is_stuck = True
                            state.stuck_since = time.time()
                            logger.warning("aggro_bot_stuck: bot=%s pos=%s count=%d",
                                           bot_id, new_pos, state.position_unchanged_count)
                else:
                    state.position_unchanged_count = 0
                    state.is_stuck = False
                state.last_position = new_pos

    def get_state(self, bot_id: str) -> AggroState | None:
        """Get a bot's aggro state."""
        with self._lock:
            return self._bot_states.get(bot_id)

    # ── Danger Assessment ────────────────────────────────────────────

    def assess_danger(self, bot_id: str) -> float:
        """Assess the current danger level for a bot (0.0-1.0)."""
        with self._lock:
            state = self._bot_states.get(bot_id)
            if state is None:
                return 0.0

            score = 0.0

            # Aggro count
            if state.aggro_count >= self.AGGR0_DEADLY:
                score += 0.5
            elif state.aggro_count >= self.AGGR0_DANGER:
                score += 0.3
            elif state.aggro_count >= self.AGGR0_CAUTION:
                score += 0.15

            # HP ratio
            if state.hp_ratio < 0.2:
                score += 0.3
            elif state.hp_ratio < 0.4:
                score += 0.15
            elif state.hp_ratio < 0.6:
                score += 0.05

            # Map danger
            map_danger = self._map_danger.get(state.current_map, 0.0)
            score += map_danger * 0.3

            # Stuck
            if state.is_stuck:
                score += 0.2

            # Dead
            if state.is_dead:
                score += 0.5

            return min(1.0, score)

    def get_emergency_action(self, bot_id: str) -> str:
        """Determine if an emergency action is needed."""
        state = self.get_state(bot_id)
        if state is None:
            return ""

        if state.is_dead:
            return "respawn"
        if state.hp_ratio < 0.2 and state.aggro_count > 0:
            return "flee"
        if state.is_stuck and state.aggro_count > 0:
            return "teleport"
        if state.aggro_count >= self.AGGR0_DEADLY:
            return "flee"
        return ""

    # ── Pathfinding ──────────────────────────────────────────────────

    def find_safe_path(self, bot_id: str, target_map: str) -> AggroPathResult:
        """Find a safe path to a target map, considering aggro state.

        Returns an AggroPathResult with navigation commands.
        """
        result = AggroPathResult()

        with self._lock:
            state = self._bot_states.get(bot_id)
            if state is None:
                result.reason = "bot_not_registered"
                return result

            current_map = state.current_map
            if not current_map:
                result.reason = "current_map_unknown"
                return result

        # Check for emergency
        emergency = self.get_emergency_action(bot_id)
        if emergency:
            result.emergency_action = emergency
            result.safe = False
            result.reason = f"emergency: {emergency}"
            logger.warning("aggro_emergency: bot=%s action=%s", bot_id, emergency)
            return result

        # Update danger map with current aggro state
        danger = dict(self._map_danger)
        state = self.get_state(bot_id)
        if state and state.aggro_count > 0:
            # Temporarily increase danger on current map
            current_danger = danger.get(current_map, 0.0)
            aggro_penalty = min(0.5, state.aggro_count * 0.1)
            danger[current_map] = min(1.0, current_danger + aggro_penalty)

        self._pathfinding.set_danger_map(danger)

        # Find path
        plan = self._pathfinding.find_path(current_map, target_map)
        if not plan.complete:
            result.reason = plan.error
            return result

        # Convert to bridge commands
        commands = self._pathfinding.to_bridge_commands(plan)

        result.plan = plan
        result.commands = commands
        result.safe = plan.danger_level in ("low", "medium")
        result.reason = f"path found: {plan.total_maps} maps, {plan.total_portals} portals, danger={plan.danger_level}"
        return result

    def find_safe_path_to_zone(self, bot_id: str, target_map: str,
                                current_x: int = 0, current_y: int = 0) -> AggroPathResult:
        """Find a safe path to a hunting zone."""
        with self._lock:
            state = self._bot_states.get(bot_id)
            if state:
                state.current_x = current_x or state.current_x
                state.current_y = current_y or state.current_y

        return self.find_safe_path(bot_id, target_map)

    def check_reroute_needed(self, bot_id: str) -> bool:
        """Check if the bot needs to reroute due to aggro or being stuck."""
        state = self.get_state(bot_id)
        if state is None:
            return False

        # Reroute if aggro is too high
        if state.aggro_count >= self.AGGR0_DANGER:
            return True

        # Reroute if stuck for too long
        if state.is_stuck:
            stuck_duration = time.time() - state.stuck_since
            if stuck_duration > self.STUCK_THRESHOLD_SECONDS:
                return True

        # Reroute if HP is critically low and in combat
        if state.hp_ratio < 0.3 and state.in_combat:
            return True

        return False

    # ── Map Danger Management ────────────────────────────────────────

    def set_map_danger(self, map_name: str, score: float) -> None:
        """Set the base danger level for a map."""
        with self._lock:
            self._map_danger[map_name] = max(0.0, min(1.0, score))

    def get_map_danger(self, map_name: str) -> float:
        """Get the base danger level for a map."""
        with self._lock:
            return self._map_danger.get(map_name, 0.0)

    def get_safe_maps_near(self, map_name: str, radius: int = 2) -> list[str]:
        """Get safe maps within N portal hops of a map."""
        safe_maps: list[str] = []
        visited: set[str] = set()
        queue: list[tuple[str, int]] = [(map_name, 0)]

        while queue:
            current, dist = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            danger = self._map_danger.get(current, 0.0)
            if danger < self.DANGER_MEDIUM and current != map_name:
                safe_maps.append(current)

            if dist < radius:
                for neighbor, _ in self._portal_knowledge.get_neighbors(current):
                    if neighbor not in visited:
                        queue.append((neighbor, dist + 1))

        return safe_maps

    def get_status_summary(self, bot_id: str) -> str:
        """Get a human-readable status summary for a bot."""
        state = self.get_state(bot_id)
        if state is None:
            return f"Bot {bot_id}: not registered"
        danger = self.assess_danger(bot_id)
        emergency = self.get_emergency_action(bot_id)
        lines = [
            f"Bot: {bot_id}",
            f"Map: {state.current_map} ({state.current_x}, {state.current_y})",
            f"Aggro: {state.aggro_count} | Threat: {state.aggro_threat:.1%}",
            f"HP: {state.hp_ratio:.0%} | SP: {state.sp_ratio:.0%}",
            f"Danger: {danger:.0%} | Emergency: {emergency or 'none'}",
            f"Stuck: {state.is_stuck} ({state.position_unchanged_count} ticks)",
            f"Dead: {state.is_dead}",
        ]
        return "\n".join(lines)


# ── Global Singleton ──

_aggro_pathfinder: AggroPathfinder | None = None
_aggro_pathfinder_lock = RLock()


def get_aggro_pathfinder() -> AggroPathfinder:
    global _aggro_pathfinder
    with _aggro_pathfinder_lock:
        if _aggro_pathfinder is None:
            _aggro_pathfinder = AggroPathfinder()
        return _aggro_pathfinder
