"""Graceful Degradation — per-module isolation with fallback.

When any domain module fails (YAML parse error, import failure, runtime
exception), ONLY that module's decisions are dropped. The rest of the
system continues normally.

This replaces the current all-or-nothing delegation try block with
per-module try/except handlers. Each module gets:
1. Its own try/except
2. A fallback action set (empty = skip, or survival default)
3. Error logging with module name
4. Runtime metrics so we can track which modules fail most
"""
from __future__ import annotations
import logging
import traceback
from typing import Any, Callable
from collections import defaultdict

from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)


class DegradationRegistry:
    """Tracks which modules are degraded and their failure counts."""
    
    def __init__(self):
        self._failures: dict[str, int] = defaultdict(int)
        self._degraded: set[str] = set()
        self._max_failures = 5  # After 5 failures, mark permanently degraded
    
    def record_failure(self, module_name: str) -> None:
        self._failures[module_name] += 1
        if self._failures[module_name] >= self._max_failures:
            self._degraded.add(module_name)
            logger.warning(f"[degradation] {module_name} permanently degraded after {self._max_failures} failures")
    
    def record_success(self, module_name: str) -> None:
        """Reset failure count on success (module recovered)."""
        if module_name in self._failures:
            del self._failures[module_name]
    
    def is_degraded(self, module_name: str) -> bool:
        return module_name in self._degraded
    
    def get_status(self) -> dict[str, Any]:
        return {
            "degraded": sorted(list(self._degraded)),
            "failures": dict(self._failures),
            "total_degraded": len(self._degraded),
        }


def safe_assess(
    module: Any,
    module_name: str,
    signals: dict[str, Any],
    actions: list[HeuristicAction],
    bot_id: str,
    registry: DegradationRegistry | None = None,
) -> bool:
    """Safely call a domain module's assess() method.
    
    Args:
        module: The domain module instance
        module_name: Human-readable name for logging
        signals: Game state signals
        actions: Action list to append to
        bot_id: Bot identifier
        registry: Optional degradation tracker
    
    Returns:
        True if module executed normally, False if it failed
    """
    if module is None:
        return True  # Module not configured, not a failure
    
    if registry and registry.is_degraded(module_name):
        return False  # Module permanently degraded
    
    if not hasattr(module, 'assess'):
        return True  # No assess method, skip
    
    try:
        module.assess(signals, actions, bot_id)
        if registry:
            registry.record_success(module_name)
        return True
    except Exception as e:
        logger.error(f"[degradation] {module_name}.assess() failed: {type(e).__name__}: {e}")
        logger.debug(traceback.format_exc())
        if registry:
            registry.record_failure(module_name)
        return False


def safe_init(module_factory: Callable[[], Any], module_name: str, registry: DegradationRegistry | None = None) -> Any | None:
    """Safely initialize a domain module.
    
    Args:
        module_factory: Zero-argument callable that creates the module
        module_name: Human-readable name for logging
        registry: Optional degradation tracker
    
    Returns:
        Module instance, or None if initialization failed
    """
    try:
        instance = module_factory()
        logger.debug(f"[degradation] {module_name} initialized successfully")
        return instance
    except Exception as e:
        logger.error(f"[degradation] {module_name} init failed: {type(e).__name__}: {e}")
        logger.debug(traceback.format_exc())
        if registry:
            registry.mark_degraded(module_name)
        return None


def safe_yaml_load(filepath: str, module_name: str) -> dict:
    """Safely load a YAML file. Returns empty dict on failure."""
    try:
        import yaml
        with open(filepath) as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        logger.warning("[degradation] %s: YAML not found: %s", module_name, filepath)
        return {}
    except Exception:
        logger.error("[degradation] %s: YAML error loading %s", module_name, filepath, exc_info=True)
        return {}


def get_registry() -> DegradationRegistry:
    global _registry
    if _registry is None:
        _registry = DegradationRegistry()
    return _registry
