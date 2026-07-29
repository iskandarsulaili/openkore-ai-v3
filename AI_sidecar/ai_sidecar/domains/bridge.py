"""Integration between state collectors and domain modules."""
from __future__ import annotations
from typing import Any
import logging

logger = logging.getLogger(__name__)


class StateDomainBridge:
    """Bridge between the structured state system and domain modules.
    
    Each domain receives the raw signals dict from the bridge.
    The state module provides structured GameState for domains that need it.
    This bridge lazily converts signals to GameState on demand.
    """
    
    def __init__(self) -> None:
        self._game_state: Any = None
        self._state_collector: Any = None
    
    def get_game_state(self, signals: dict[str, Any]) -> Any:
        """Get structured GameState from signals. Caches the result."""
        if self._state_collector is None:
            from ai_sidecar.state.collector import StateCollector
            self._state_collector = StateCollector()
        if self._game_state is None:
            self._game_state = self._state_collector.collect(signals)
        return self._game_state
    
    def invalidate(self) -> None:
        """Clear cached game state — call at the start of each cycle."""
        self._game_state = None
