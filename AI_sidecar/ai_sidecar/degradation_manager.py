"""
Graceful Degradation Layer — wraps every external dependency with sensible defaults.
No knowledge DB? Use built-in item prices. No LLM? Use rule-based decisions.
No multi-client? Run solo. No shared DB? Log locally.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ModuleHealth:
    """Health status of a module."""
    module_name: str
    is_healthy: bool = True
    last_check: float = 0.0
    failure_count: int = 0
    last_error: str = ""
    degradation_mode: str = "none"  # none, degraded, failed
    fallback_active: bool = False


class DegradationManager:
    """Manages graceful degradation of all modules."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._modules: dict[str, ModuleHealth] = {}
        self._fallbacks: dict[str, Callable] = {}
        self._max_failures: int = 3
        self._check_interval: float = 30.0

    # ── Public API ──

    def register_module(self, name: str, fallback_fn: Callable | None = None) -> None:
        """Register a module with an optional fallback function."""
        with self._lock:
            self._modules[name] = ModuleHealth(module_name=name)
            if fallback_fn:
                self._fallbacks[name] = fallback_fn
            logger.info("degradation: module registered: %s", name)

    def report_success(self, name: str) -> None:
        """Report a successful operation."""
        with self._lock:
            module = self._modules.get(name)
            if module:
                module.is_healthy = True
                module.last_check = time.time()
                module.failure_count = 0
                module.degradation_mode = "none"
                module.fallback_active = False

    def report_failure(self, name: str, error: str = "") -> None:
        """Report a failure. Triggers degradation after threshold."""
        with self._lock:
            module = self._modules.get(name)
            if not module:
                return
            module.failure_count += 1
            module.last_error = error
            module.last_check = time.time()

            if module.failure_count >= self._max_failures:
                module.is_healthy = False
                module.degradation_mode = "degraded"
                if name in self._fallbacks:
                    module.fallback_active = True
                    module.degradation_mode = "degraded"
                    logger.warning("degradation: %s degraded, fallback active (failures=%d, error=%s)",
                                   name, module.failure_count, error)
                else:
                    module.degradation_mode = "failed"
                    logger.error("degradation: %s FAILED (no fallback, failures=%d, error=%s)",
                                 name, module.failure_count, error)

    def is_healthy(self, name: str) -> bool:
        """Check if a module is healthy."""
        with self._lock:
            module = self._modules.get(name)
            if not module:
                return True  # Unknown modules are assumed healthy
            return module.is_healthy

    def get_degradation_mode(self, name: str) -> str:
        """Get the degradation mode for a module."""
        with self._lock:
            module = self._modules.get(name)
            if not module:
                return "none"
            return module.degradation_mode

    def get_fallback(self, name: str) -> Callable | None:
        """Get the fallback function for a module."""
        with self._lock:
            return self._fallbacks.get(name)

    def get_all_health(self) -> dict[str, ModuleHealth]:
        with self._lock:
            return dict(self._modules)

    def get_health_summary(self) -> str:
        with self._lock:
            lines = [f"── Module Health ──"]
            healthy = sum(1 for m in self._modules.values() if m.is_healthy)
            degraded = sum(1 for m in self._modules.values() if m.degradation_mode == "degraded")
            failed = sum(1 for m in self._modules.values() if m.degradation_mode == "failed")
            lines.append(f"Healthy: {healthy} | Degraded: {degraded} | Failed: {failed}")
            for name, module in sorted(self._modules.items()):
                status = "✅" if module.is_healthy else "⚠️" if module.degradation_mode == "degraded" else "❌"
                fb = " (fallback)" if module.fallback_active else ""
                lines.append(f"  {status} {name}{fb}")
                if not module.is_healthy and module.last_error:
                    lines.append(f"    Last error: {module.last_error[:100]}")
            return "\n".join(lines)

    def reset(self) -> None:
        with self._lock:
            self._modules.clear()
            self._fallbacks.clear()


# ── Global Singleton ──

_degradation: DegradationManager | None = None
_degradation_lock = RLock()


def get_degradation_manager() -> DegradationManager:
    global _degradation
    with _degradation_lock:
        if _degradation is None:
            _degradation = DegradationManager()
        return _degradation
