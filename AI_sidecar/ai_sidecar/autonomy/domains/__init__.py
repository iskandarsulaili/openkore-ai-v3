"""Domain modules for heuristic service decomposition.

Each domain module encapsulates a focused area of bot decision-making.
The DomainRegistry discovers and manages all domains, calling each
domain's assess() method in priority order.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ai_sidecar.autonomy.heuristic_service import HeuristicAction

logger = logging.getLogger(__name__)


class BaseDomain:
    """Base class for all heuristic domains.

    Subclasses must set:
        name (str): short domain identifier (e.g. 'combat')
        priority (int): execution order (lower = earlier)

    Subclasses must implement:
        assess(signals, actions, service)
            signals:  dict of bot state signals from the bridge
            actions:  list[HeuristicAction] — domain appends to it
            service:  HeuristicService instance for shared helpers
    """

    name: str = "base"
    priority: int = 100

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        service: Any,  # HeuristicService
    ) -> None:
        """Analyze signals and append actions. Override in subclass."""
        raise NotImplementedError


class DomainRegistry:
    """Discovers, registers, and iterates all domain modules.

    Usage:
        registry = DomainRegistry()
        registry.load_all()
        registry.assess(signals, actions, service)
    """

    def __init__(self) -> None:
        self._domains: list[BaseDomain] = []

    def register(self, domain: BaseDomain) -> None:
        """Register a single domain instance."""
        self._domains.append(domain)

    def load_all(self) -> None:
        """Auto-discover and register all domain modules.

        Each domain module is expected to expose a module-level
        function ``create_domain() -> BaseDomain`` or a class
        ``Domain`` that can be instantiated.
        """
        domain_module_names = [
        ]
        for mod_name in domain_module_names:
            try:
                mod = __import__(
                    f"ai_sidecar.autonomy.domains.{mod_name}",
                    fromlist=[mod_name],
                )
                # Try create_domain() function first, then Domain class
                if hasattr(mod, "create_domain"):
                    instance = mod.create_domain()
                elif hasattr(mod, "Domain"):
                    instance = mod.Domain()
                else:
                    logger.warning(
                        "Domain module %s has no create_domain() or Domain class",
                        mod_name,
                    )
                    continue
                self.register(instance)
                logger.debug("Registered domain: %s (priority %d)", instance.name, instance.priority)
            except Exception as exc:
                logger.warning("Failed to load domain module %s: %s", mod_name, exc)

        # Sort by priority so assess() iterates in priority order
        self._domains.sort(key=lambda d: d.priority)

    def assess_all(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        service: Any,
    ) -> None:
        """Call every domain's assess() in priority order."""
        for domain in self._domains:
            try:
                domain.assess(signals, actions, service)
            except Exception as exc:
                logger.error(
                    "Domain %s crashed: %s", domain.name, exc, exc_info=True
                )

    @property
    def domain_names(self) -> list[str]:
        return [d.name for d in self._domains]

    def __len__(self) -> int:
        return len(self._domains)


__all__ = [
    "BaseDomain",
    "DomainRegistry",
]
