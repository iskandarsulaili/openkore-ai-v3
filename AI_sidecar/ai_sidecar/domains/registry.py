"""Domain Registry — discovers and manages all domain modules."""
from __future__ import annotations
from typing import Any
import logging
from ai_sidecar.domains import BaseDomain

logger = logging.getLogger(__name__)


class DomainRegistry:
    """Registry of all domain modules.
    
    Domains are registered by name and iterated in priority order.
    Each domain gets a chance to assess the current state and emit actions.
    """
    
    def __init__(self) -> None:
        self._domains: dict[str, BaseDomain] = {}
        self._sorted: list[BaseDomain] = []
    
    def register(self, domain: BaseDomain) -> None:
        """Register a domain module."""
        if domain.name in self._domains:
            logger.warning(f"Domain {domain.name} already registered, overwriting")
        self._domains[domain.name] = domain
        self._sorted = sorted(self._domains.values(), key=lambda d: d.priority)
        logger.debug(f"Registered domain: {domain.name} (priority={domain.priority})")
    
    def register_all(self, *domains: BaseDomain) -> None:
        """Register multiple domains at once."""
        for domain in domains:
            self.register(domain)
    
    def get(self, name: str) -> BaseDomain | None:
        """Get a domain by name."""
        return self._domains.get(name)
    
    @property
    def domains(self) -> list[BaseDomain]:
        """All domains sorted by priority (ascending)."""
        return self._sorted
    
    def assess_all(
        self,
        state: str,
        signals: dict[str, Any],
        actions: list[Any],
        bot_id: str,
    ) -> None:
        """Call assess() on each domain in priority order.
        
        Args:
            state: Current bot state (COLD_START, HUNT, TOWN, etc.)
            signals: Raw signals dict
            actions: List to append actions to
            bot_id: Normalized bot ID
        """
        for domain in self._sorted:
            try:
                domain.assess(signals, actions, bot_id)
            except Exception as e:
                logger.error(f"Domain {domain.name}.assess() failed: {e}", exc_info=True)
    
    def __len__(self) -> int:
        return len(self._domains)
    
    def __repr__(self) -> str:
        return f"<DomainRegistry: {len(self)} domains>"
