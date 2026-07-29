"""Domain base classes for openkore-ai-v3."""
from __future__ import annotations
from typing import Any, ClassVar
import logging

logger = logging.getLogger(__name__)


class BaseDomain:
    """Base class for all domain modules.
    
    Each domain handles a specific gameplay area (combat, economy, routing, etc.)
    and is registered in the DomainRegistry.
    """
    
    # Domain name — must be unique
    name: ClassVar[str] = "base"
    
    # Priority: lower = runs first. Survival > Combat > Hygiene > Economy > Routing
    priority: ClassVar[int] = 100
    
    def __init__(self) -> None:
        self._initialized = False
    
    def initialize(self) -> None:
        """Called once when the domain is registered. Set up resources here."""
        self._initialized = True
    
    def assess(
        self,
        signals: dict[str, Any],
        actions: list[Any],
        bot_id: str,
    ) -> None:
        """Assess the current state and append actions.
        
        Args:
            signals: Raw signals dict from the bridge snapshot
            actions: List to append HeuristicAction objects to
            bot_id: Normalized bot identifier
        """
        raise NotImplementedError
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}:{self.name}>"
