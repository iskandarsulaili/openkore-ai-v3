"""Domain modules for heuristic service decomposition.

Each domain module encapsulates a focused area of bot decision-making.
The DomainRegistry discovers and manages all domains, calling each
domain's assess() method in priority order.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ai_sidecar.autonomy.heuristic_service import HeuristicAction

logger = logging.getLogger(__name__)


class BaseDomain(ABC):
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

    @abstractmethod
    def assess(
        self,
        signals: dict[str, Any],
        actions: list[Any],
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
        self._weights: dict[str, float] = {}

    def register(self, domain: BaseDomain) -> None:
        """Register a single domain instance."""
        self._domains.append(domain)

    def set_weights(self, weights: dict[str, float]) -> None:
        """Apply per-domain execution weights (weight > 1.0 runs earlier).

        Effective priority = base_priority / weight, so a domain weighted 2.0
        runs before one weighted 1.0 at the same base priority. Unknown names
        are ignored (weight stays 1.0). Re-sorts the domain list in place.
        """
        sanitized: dict[str, float] = {}
        for name, w in (weights or {}).items():
            try:
                fw = float(w)
            except (TypeError, ValueError):
                continue
            if fw <= 0:
                fw = 1.0
            sanitized[str(name)] = fw
        self._weights.update(sanitized)
        self._domains.sort(
            key=lambda d: d.priority / self._weights.get(d.name, 1.0)
        )

    def load_all(self) -> None:
        """Auto-discover and register all domain modules.

        Each domain module is expected to expose a module-level
        function ``create_domain() -> BaseDomain`` or a class
        ``Domain`` that can be instantiated.
        """
        # ⚠️ DELIBERATELY EMPTY — DO NOT POPULATE.
        # The modules under autonomy/domains/* (combat, social, economy, ...) are
        # a LEGACY parallel domain system. The modern per-manager wiring in
        # heuristic_service._init_new_domains (PartyDomain, EquipmentOptimizer,
        # ColdStartManager, LifecycleManager, ...) supersedes them. Populating
        # this list would DOUBLE-EMIT with those wired managers AND reintroduce
        # the round-2 party spam (social.py emits 'party create AI<ts>' /
        # 'party request <name>' / 'party leave' — the exact commands the bridge
        # party gates and the fleet coordinator now suppress).
        # If a new domain is genuinely needed, add it to _init_new_domains
        # instead of this list. See tests/test_task_scheduler_wiring.py and
        # the bridge party-suppression blocks for the guard rationale.
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
