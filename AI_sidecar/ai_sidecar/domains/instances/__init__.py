"""Instance dungeon domain."""
from __future__ import annotations

from ai_sidecar.domains.instances.registry import InstanceRegistry
from ai_sidecar.domains.instances.coordinator import InstanceCoordinator

__all__ = [
    "InstanceDomain",
    "InstanceRegistry",
    "InstanceCoordinator",
]


class InstanceDomain:
    """Aggregate domain for instance dungeon management."""

    name = "instances"
    priority = 65

    def __init__(self) -> None:
        self.registry = InstanceRegistry()
        self.coordinator = InstanceCoordinator(registry=self.registry)
