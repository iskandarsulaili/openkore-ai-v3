"""
Fly Wing Manager — intelligent fly wing usage for pathfinding.

A pro player uses Fly Wings strategically:
1. When stuck (can't reach destination after N attempts)
2. When surrounded by aggro (teleport to safety)
3. When traveling long distances (save time walking)
4. When map-server boundary crossing is detected

Tracks fly wing inventory and produces bridge-compatible use commands.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FlyWingState:
    """Current fly wing state for a bot."""
    bot_id: str
    fly_wing_count: int = 0
    butterfly_wing_count: int = 0
    last_use_time: float = 0.0
    consecutive_stuck_count: int = 0
    fly_wing_cooldown: float = 3.0  # seconds between uses
    total_uses: int = 0
    successful_escapes: int = 0
    failed_attempts: int = 0


# Bridge action constants
ACTION_USE_FLY_WING = "use_item"
ITEM_FLY_WING = "Fly Wing"
ITEM_BUTTERFLY_WING = "Butterfly Wing"

# Distance thresholds (in portal hops)
SHORT_DISTANCE = 2
MEDIUM_DISTANCE = 5
LONG_DISTANCE = 10


class FlyWingManager:
    """Manages fly wing usage across all bots.

    Thread-safe. Decides when to use fly wings based on stuck detection,
    aggro levels, travel distance, and map-server boundaries.
    """

    # Stuck thresholds
    STUCK_ATTEMPT_LIMIT = 3
    STUCK_TIME_LIMIT = 20.0  # seconds

    # Aggro thresholds
    AGGR0_SURROUNDED = 4  # monsters nearby -> use fly wing
    AGGR0_CRITICAL = 6    # too many -> emergency fly wing

    # Distance thresholds (portal hops)
    FLY_WING_DISTANCE_MIN = 4  # use fly wing if path is this many hops or more

    # Inventory warnings
    LOW_FLY_WING_WARNING = 5
    CRITICAL_FLY_WING = 2

    def __init__(self) -> None:
        self._lock = RLock()
        self._bot_states: dict[str, FlyWingState] = {}

    # ── Bot State Management ─────────────────────────────────────────

    def register_bot(self, bot_id: str) -> None:
        """Register a bot for fly wing tracking."""
        with self._lock:
            if bot_id not in self._bot_states:
                self._bot_states[bot_id] = FlyWingState(bot_id=bot_id)

    def update_inventory(self, bot_id: str, fly_wings: int | None = None,
                         butterfly_wings: int | None = None) -> None:
        """Update fly wing inventory counts."""
        with self._lock:
            state = self._bot_states.get(bot_id)
            if state is None:
                state = FlyWingState(bot_id=bot_id)
                self._bot_states[bot_id] = state
            if fly_wings is not None:
                state.fly_wing_count = fly_wings
            if butterfly_wings is not None:
                state.butterfly_wing_count = butterfly_wings

    def get_state(self, bot_id: str) -> FlyWingState | None:
        with self._lock:
            return self._bot_states.get(bot_id)

    # ── Decision Logic ───────────────────────────────────────────────

    def should_use_fly_wing(self, bot_id: str, aggro_count: int = 0,
                            is_stuck: bool = False, path_hops: int = 0,
                            crossing_map_server: bool = False) -> tuple[bool, str]:
        """Determine if a fly wing should be used.

        Returns (should_use, reason).
        """
        with self._lock:
            state = self._bot_states.get(bot_id)
            if state is None:
                return False, "not_registered"

            # Check cooldown
            if time.time() - state.last_use_time < state.fly_wing_cooldown:
                return False, "cooldown"

            # Check inventory
            if state.fly_wing_count <= 0:
                return False, "no_fly_wings"

            # Emergency: surrounded by aggro
            if aggro_count >= self.AGGR0_CRITICAL:
                return True, "emergency_aggro_surrounded"

            # Emergency: stuck with aggro
            if is_stuck and aggro_count >= self.AGGR0_SURROUNDED:
                return True, "stuck_with_aggro"

            # Stuck detection
            if is_stuck and state.consecutive_stuck_count >= self.STUCK_ATTEMPT_LIMIT:
                return True, "stuck_exceeded_limit"

            # Map-server boundary crossing
            if crossing_map_server:
                return True, "map_server_boundary"

            # Long distance travel
            if path_hops >= self.FLY_WING_DISTANCE_MIN:
                return True, "long_distance_travel"

            return False, "no_trigger"

    def should_use_butterfly_wing(self, bot_id: str, hp_ratio: float = 1.0,
                                   is_dead: bool = False) -> tuple[bool, str]:
        """Determine if a butterfly wing should be used (return to save point)."""
        with self._lock:
            state = self._bot_states.get(bot_id)
            if state is None:
                return False, "not_registered"

            if state.butterfly_wing_count <= 0:
                return False, "no_butterfly_wings"

            if is_dead:
                return False, "already_dead_respawn_instead"

            if hp_ratio < 0.15:
                return True, "critically_low_hp"

            return False, "no_trigger"

    def record_use(self, bot_id: str, wing_type: str = "fly", success: bool = True) -> None:
        """Record a fly wing use attempt."""
        with self._lock:
            state = self._bot_states.get(bot_id)
            if state is None:
                return

            state.last_use_time = time.time()
            state.total_uses += 1

            if wing_type == "fly":
                state.fly_wing_count = max(0, state.fly_wing_count - 1)
            else:
                state.butterfly_wing_count = max(0, state.butterfly_wing_count - 1)

            if success:
                state.successful_escapes += 1
                state.consecutive_stuck_count = 0
            else:
                state.failed_attempts += 1

    def record_stuck_attempt(self, bot_id: str) -> None:
        """Increment the stuck counter for a bot."""
        with self._lock:
            state = self._bot_states.get(bot_id)
            if state is None:
                return
            state.consecutive_stuck_count += 1

    def reset_stuck(self, bot_id: str) -> None:
        """Reset stuck counter (bot moved successfully)."""
        with self._lock:
            state = self._bot_states.get(bot_id)
            if state is None:
                return
            state.consecutive_stuck_count = 0

    # ── Bridge Commands ──────────────────────────────────────────────

    def get_fly_wing_command(self, bot_id: str) -> dict[str, Any]:
        """Get a bridge command to use a fly wing."""
        self.record_use(bot_id, "fly")
        return {
            "action": ACTION_USE_FLY_WING,
            "item": ITEM_FLY_WING,
            "quantity": 1,
            "reason": "navigation",
        }

    def get_butterfly_wing_command(self, bot_id: str) -> dict[str, Any]:
        """Get a bridge command to use a butterfly wing (return to save)."""
        self.record_use(bot_id, "butterfly")
        return {
            "action": ACTION_USE_FLY_WING,
            "item": ITEM_BUTTERFLY_WING,
            "quantity": 1,
            "reason": "emergency_recall",
        }

    def get_inventory_status(self, bot_id: str) -> dict[str, Any]:
        """Get fly wing inventory status for reporting."""
        with self._lock:
            state = self._bot_states.get(bot_id)
            if state is None:
                return {"fly_wings": 0, "butterfly_wings": 0, "status": "unknown"}

            fw = state.fly_wing_count
            bw = state.butterfly_wing_count

            if fw <= 0 and bw <= 0:
                status = "out_of_stock"
            elif fw <= self.CRITICAL_FLY_WING:
                status = "critical"
            elif fw <= self.LOW_FLY_WING_WARNING:
                status = "low"
            else:
                status = "ok"

            return {
                "fly_wings": fw,
                "butterfly_wings": bw,
                "status": status,
            }

    def get_status_summary(self, bot_id: str) -> str:
        """Get a human-readable status summary."""
        state = self.get_state(bot_id)
        if state is None:
            return f"Bot {bot_id}: not registered"
        inv = self.get_inventory_status(bot_id)
        lines = [
            f"Bot: {bot_id}",
            f"Fly Wings: {state.fly_wing_count} | Butterfly Wings: {state.butterfly_wing_count}",
            f"Status: {inv['status']}",
            f"Total Uses: {state.total_uses} | Success: {state.successful_escapes} | Fail: {state.failed_attempts}",
            f"Stuck Count: {state.consecutive_stuck_count}",
        ]
        return "\n".join(lines)


# ── Global Singleton ──

_fly_wing_manager: FlyWingManager | None = None
_fly_wing_manager_lock = RLock()


def get_fly_wing_manager() -> FlyWingManager:
    global _fly_wing_manager
    with _fly_wing_manager_lock:
        if _fly_wing_manager is None:
            _fly_wing_manager = FlyWingManager()
        return _fly_wing_manager
