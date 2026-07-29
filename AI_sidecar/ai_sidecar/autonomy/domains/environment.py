"""Environment domain — time, weather, map awareness.

Extracted from heuristic_service.py (minimal standalone logic —
this domain provides hooks for time-of-day routing, weather-based
decisions, and map-specific awareness).
"""

from __future__ import annotations

import logging
from typing import Any

from ai_sidecar.autonomy.domains import BaseDomain
from ai_sidecar.autonomy.heuristic_service import HeuristicAction

logger = logging.getLogger(__name__)


class EnvironmentDomain(BaseDomain):
    name: str = "environment"
    priority: int = 70

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        service: Any,
    ) -> None:
        """Evaluate environment-aware decisions.

        Future: time-based routing (night = safer maps),
        weather-based equipment choices (rain = water element bonus),
        map congestion awareness.
        """
        map_name = str(signals.get("map", "") or "").lower()
        base_level = int(signals.get("base_level", 1) or 1)

        # Map awareness: ensure bot is on an appropriate map for its level
        # (Level-range checking is handled by progression/routing domains)
        pass


def create_domain() -> EnvironmentDomain:
    return EnvironmentDomain()
