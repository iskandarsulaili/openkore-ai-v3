"""Tracks committed actions to prevent conflicting commands within a time window.

This is the Python-side counterpart to the bridge's %_committed_actions hash.
The PDCA loop uses this to avoid generating conflicting actions when the bridge
already has a committed action in progress.
"""
from __future__ import annotations

import time
import logging
from threading import RLock

logger = logging.getLogger(__name__)

# Default cooldown for committed move actions (ms)
DEFAULT_MOVE_COOLDOWN_MS = 25000  # 25 seconds (slightly less than bridge's 30s)
DEFAULT_LOCKMAP_COOLDOWN_MS = 25000
DEFAULT_SIT_COOLDOWN_MS = 25000


class CommittedActionTracker:
    """Tracks committed actions to prevent conflicting command generation.

    The PDCA loop checks this before emitting move/lockMap/sit actions.
    If a conflicting action was recently committed, the loop skips that
    emission to avoid flooding the bridge with conflicting commands.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        # action_type:target => timestamp_ms
        self._committed: dict[str, float] = {}
        # action_type => timestamp_ms (for type-level tracking)
        self._type_committed: dict[str, float] = {}

    def record_move(self, target_map: str) -> None:
        """Record that a move to target_map was committed."""
        with self._lock:
            now_ms = time.time() * 1000
            self._committed[f"move:{target_map}"] = now_ms
            self._type_committed["move"] = now_ms
            self._cleanup(now_ms)

    def record_lockmap(self, target_map: str) -> None:
        """Record that a lockMap change to target_map was committed."""
        with self._lock:
            now_ms = time.time() * 1000
            self._committed[f"set_lockmap:{target_map}"] = now_ms
            self._type_committed["set_lockmap"] = now_ms
            self._cleanup(now_ms)

    def record_sit(self) -> None:
        """Record that a sit command was committed."""
        with self._lock:
            now_ms = time.time() * 1000
            self._committed["sit:"] = now_ms
            self._type_committed["sit"] = now_ms
            self._cleanup(now_ms)

    def has_committed_move(self, bot_id: str | None = None, cooldown_ms: int = DEFAULT_MOVE_COOLDOWN_MS) -> bool:
        """Check if a move action was recently committed.

        Args:
            bot_id: Optional bot ID (for future per-bot tracking).
            cooldown_ms: Cooldown window in milliseconds.

        Returns:
            True if a move was committed within the cooldown window.
        """
        return self._has_type_committed("move", cooldown_ms)

    def has_committed_lockmap(self, cooldown_ms: int = DEFAULT_LOCKMAP_COOLDOWN_MS) -> bool:
        """Check if a lockMap change was recently committed."""
        return self._has_type_committed("set_lockmap", cooldown_ms)

    def has_committed_sit(self, cooldown_ms: int = DEFAULT_SIT_COOLDOWN_MS) -> bool:
        """Check if a sit command was recently committed."""
        return self._has_type_committed("sit", cooldown_ms)

    def has_conflict(self, action_type: str, target: str | None = None, cooldown_ms: int = DEFAULT_MOVE_COOLDOWN_MS) -> bool:
        """Check if a proposed action conflicts with a recently committed one.

        Args:
            action_type: 'move', 'set_lockmap', 'sit', etc.
            target: Target map or identifier.
            cooldown_ms: Cooldown window in milliseconds.

        Returns:
            True if a conflicting action was committed within the cooldown window.
        """
        with self._lock:
            now_ms = time.time() * 1000
            self._cleanup(now_ms)

            for key, last_ms in self._committed.items():
                elapsed = now_ms - last_ms
                if elapsed > cooldown_ms:
                    continue

                committed_type, committed_target = key.split(":", 1) if ":" in key else (key, "")

                # Same type, same target -> no conflict (idempotent)
                if committed_type == action_type and committed_target == (target or ""):
                    return False

                # move <map> conflicts with other move <map> commands (different maps)
                if action_type == "move" and committed_type == "move" and committed_target != (target or ""):
                    logger.info(
                        "committed_action_conflict: move '%s' blocked by committed move '%s' (%.0fms ago)",
                        target, committed_target, elapsed,
                    )
                    return True

                # set lockMap conflicts with other set lockMap commands
                if action_type == "set_lockmap" and committed_type == "set_lockmap" and committed_target != (target or ""):
                    logger.info(
                        "committed_action_conflict: set_lockmap '%s' blocked by committed set_lockmap '%s' (%.0fms ago)",
                        target, committed_target, elapsed,
                    )
                    return True

                # sit conflicts with move commands
                if action_type == "sit" and committed_type == "move":
                    logger.info(
                        "committed_action_conflict: sit blocked by committed move '%s' (%.0fms ago)",
                        committed_target, elapsed,
                    )
                    return True
                if action_type == "move" and committed_type == "sit":
                    logger.info(
                        "committed_action_conflict: move '%s' blocked by committed sit (%.0fms ago)",
                        target, elapsed,
                    )
                    return True

            return False

    def clear(self) -> None:
        """Clear all committed actions."""
        with self._lock:
            self._committed.clear()
            self._type_committed.clear()

    def _has_type_committed(self, action_type: str, cooldown_ms: int) -> bool:
        with self._lock:
            now_ms = time.time() * 1000
            last_ms = self._type_committed.get(action_type, 0)
            if last_ms == 0:
                return False
            elapsed = now_ms - last_ms
            if elapsed > cooldown_ms:
                return False
            return True

    def _cleanup(self, now_ms: float) -> None:
        max_age = max(DEFAULT_MOVE_COOLDOWN_MS, DEFAULT_LOCKMAP_COOLDOWN_MS, DEFAULT_SIT_COOLDOWN_MS) * 2
        stale_keys = [k for k, v in self._committed.items() if now_ms - v > max_age]
        for k in stale_keys:
            del self._committed[k]
        stale_types = [k for k, v in self._type_committed.items() if now_ms - v > max_age]
        for k in stale_types:
            del self._type_committed[k]
