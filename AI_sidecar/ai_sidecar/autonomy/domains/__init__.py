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

        OBSERVE-ONLY MODE: all legacy autonomy/domains modules are
        registered but every command they emit is converted to an
        observable log intent (kind="log") at assess_all time. Their
        analysis runs live (proving the code is exercised, not dormant)
        but nothing executes — the modern per-manager wiring in
        heuristic_service._init_new_domains owns actual command emission,
        so observe-only prevents double-emission. Party commands and
        ai-mode flips are additionally stripped at the source (social.py
        emits party_* log intents; ai_mode_* log intents replace
        `ai manual`/`ai auto`).
        """
        domain_module_names = [
            "combat",
            "consumables",
            "economy",
            "environment",
            "equipment",
            "learning",
            "mimicry",
            "npc",
            "progression",
            "quests",
            "routing",
            "social",
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
        if self._domains:
            logger.info(
                "legacy_domains_activated: %d modules in observe-only mode (%s)",
                len(self._domains),
                ",".join(d.name for d in self._domains),
            )

    def assess_all(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        service: Any,
    ) -> None:
        """Call every domain's assess() in priority order.

        OBSERVE-ONLY: any kind="command" action a legacy domain emits is
        converted to kind="log" (observable intent, never executed). This
        exercises the modules' analysis without double-emitting with the
        modern wired managers or reintroducing party/ai-mode spam.
        """
        for domain in self._domains:
            try:
                before = len(actions)
                domain.assess(signals, actions, service)
                for _idx in range(before, len(actions)):
                    _a = actions[_idx]
                    if _a.kind == "command":
                        _a.kind = "log"
                        _a.metadata = dict(getattr(_a, "metadata", {}) or {})
                        _a.metadata["observe_only"] = True
                        _a.metadata["source_domain"] = domain.name
                        _a.reason = f"[observe-only {domain.name}] {_a.reason}"
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
