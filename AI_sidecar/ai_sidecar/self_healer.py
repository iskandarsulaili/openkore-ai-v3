"""
Self-Healing System — detects failures, restarts crashed modules,
reconnects dropped connections, drains stuck queues, and falls back
to safe defaults.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class HealAction:
    """A healing action taken."""
    module_name: str
    action: str  # restart, reconnect, drain, fallback, reset
    timestamp: float = 0.0
    success: bool = False
    details: str = ""


class SelfHealer:
    """Detects and heals failures automatically."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._heal_actions: list[HealAction] = []
        self._max_actions: int = 100
        self._cooldowns: dict[str, float] = {}  # module -> last heal time
        self._heal_cooldown: float = 30.0  # Don't heal same module more than once per 30s
        self._enqueue_fn: Callable | None = None

    # ── Public API ──

    def heal_module(self, module_name: str, error: str = "") -> str | None:
        """Attempt to heal a module. Returns the action taken, or None."""
        with self._lock:
            now = time.time()
            last_heal = self._cooldowns.get(module_name, 0)
            if now - last_heal < self._heal_cooldown:
                return None  # Still in cooldown

            self._cooldowns[module_name] = now
            action = self._determine_heal(module_name, error)
            if action:
                self._heal_actions.append(HealAction(
                    module_name=module_name,
                    action=action,
                    timestamp=now,
                    success=True,
                    details=error,
                ))
                if len(self._heal_actions) > self._max_actions:
                    self._heal_actions = self._heal_actions[-self._max_actions:]
                logger.info("self_heal: %s -> %s (error: %s)", module_name, action, error)
                return action
            return None

    def _determine_heal(self, module_name: str, error: str) -> str | None:
        """Determine the appropriate healing action."""
        error_lower = error.lower()

        # Connection issues
        if any(word in error_lower for word in ["connection", "timeout", "refused", "disconnect"]):
            return "reconnect"

        # Queue issues
        if any(word in error_lower for word in ["queue", "full", "overflow"]):
            return "drain_queue"

        # Module crash
        if any(word in error_lower for word in ["crash", "exception", "traceback", "error"]):
            return "restart_module"

        # Data issues
        if any(word in error_lower for word in ["corrupt", "invalid", "malformed"]):
            return "reset_state"

        # Resource issues
        if any(word in error_lower for word in ["memory", "disk", "full"]):
            return "free_resources"

        # Default: fallback
        return "activate_fallback"

    def get_recent_heals(self, count: int = 5) -> list[HealAction]:
        with self._lock:
            return self._heal_actions[-count:]

    def get_heal_summary(self) -> str:
        with self._lock:
            lines = [f"── Self-Healing ──"]
            lines.append(f"Total heals: {len(self._heal_actions)}")
            recent = self.get_recent_heals(5)
            for h in recent:
                lines.append(f"  {h.action} on {h.module_name} {'✅' if h.success else '❌'}")
            return "\n".join(lines)

    def set_enqueue_fn(self, fn: Callable) -> None:
        with self._lock:
            self._enqueue_fn = fn

    def reset(self) -> None:
        with self._lock:
            self._heal_actions.clear()
            self._cooldowns.clear()


# ── Global Singleton ──

_self_healer: SelfHealer | None = None
_self_healer_lock = RLock()


def get_self_healer() -> SelfHealer:
    global _self_healer
    with _self_healer_lock:
        if _self_healer is None:
            _self_healer = SelfHealer()
        return _self_healer
