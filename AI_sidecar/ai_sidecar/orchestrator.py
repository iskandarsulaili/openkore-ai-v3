"""BotOrchestrator — top-level simplification layer.

The system grew to 50+ modules across 30 domains. This orchestrator
provides a single entry point for common operations, hiding complexity.
"""
from __future__ import annotations
import logging
from typing import Any

from ai_sidecar.autonomy.heuristic_service import HeuristicService
from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)


class BotOrchestrator:
    """Simplified interface to the full AI system.
    
    Usage:
        bot = BotOrchestrator()
        bot.initialize()
        actions = bot.think(signals)
        bot.act(actions)
    """
    
    def __init__(self):
        self._hs: HeuristicService | None = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize all subsystems."""
        if self._initialized:
            return
        try:
            self._hs = HeuristicService()
            self._initialized = True
            logger.info("BotOrchestrator initialized with all subsystems")
        except Exception as e:
            logger.error(f"BotOrchestrator initialization failed: {e}")
            raise
    
    def think(self, signals: dict[str, Any]) -> list[HeuristicAction]:
        """Process signals and return actions to execute.
        
        This is the main entry point — the bridge calls this once per tick
        with the current game state. Returns a list of HeuristicAction objects.
        """
        if not self._initialized:
            self.initialize()
        
        if not self._hs:
            return []
        
        try:
            assessment = self._hs.assess(signals)
            return assessment.actions
        except Exception as e:
            logger.error(f"think() failed: {e}")
            return []
    
    def act(self, actions: list[HeuristicAction]) -> list[str]:
        """Convert actions to bridge commands.
        
        Each action produces one or more commands for the OpenKore bridge.
        """
        commands = []
        for action in actions:
            if action.kind == "command" and action.command:
                commands.append(action.command)
        return commands
    
    def get_status(self) -> dict[str, Any]:
        """Get orchestrator status summary."""
        return {
            "initialized": self._initialized,
            "healthy": self._hs is not None,
        }
    
    def get_system_stats(self) -> dict[str, Any]:
        """Get comprehensive system statistics."""
        stats: dict[str, Any] = {
            "status": "running",
            "modules": [],
        }
        if self._hs:
            # Check which subsystems are active
            for attr_name in dir(self._hs):
                if attr_name.startswith("_") and not attr_name.startswith("__"):
                    val = getattr(self._hs, attr_name, None)
                    if val is not None and hasattr(val, "assess"):
                        domain = attr_name.lstrip("_")
                        stats["modules"].append(domain)
        return stats


# Global singleton for bridge access
_orchestrator: BotOrchestrator | None = None


def get_orchestrator() -> BotOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = BotOrchestrator()
    return _orchestrator
