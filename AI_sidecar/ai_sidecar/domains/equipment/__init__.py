"""Equipment management domain."""
from __future__ import annotations

from ai_sidecar.domains.equipment.manager import EquipmentManager
from ai_sidecar.domains.equipment.optimizer import EquipmentOptimizer
from ai_sidecar.domains.equipment.swapper import WeaponSwapper

__all__ = [
    "EquipmentDomain",
    "EquipmentManager",
    "EquipmentOptimizer",
    "WeaponSwapper",
]


class EquipmentDomain:
    """Aggregate domain for all equipment management."""

    name = "equipment"
    priority = 45

    def __init__(self) -> None:
        self.manager = EquipmentManager()
        self.optimizer = EquipmentOptimizer(manager=self.manager)
        self.swapper = WeaponSwapper()
