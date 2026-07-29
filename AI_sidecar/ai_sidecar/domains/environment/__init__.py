"""Environment domain — time tracking, day/night awareness."""
from __future__ import annotations

from ai_sidecar.domains.environment.time import GameTimeTracker

__all__ = [
    "EnvironmentDomain",
    "GameTimeManager",
]


class EnvironmentDomain:
    """Aggregate domain for environment awareness."""

    name = "environment"
    priority = 15

    def __init__(self) -> None:
        self.time = GameTimeManager()
