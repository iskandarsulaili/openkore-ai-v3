"""Consumables management domain."""
from __future__ import annotations

from ai_sidecar.domains.consumables.buffs import AutoBuffManager
from ai_sidecar.domains.consumables.recovery import RecoveryManager

__all__ = [
    "ConsumablesDomain",
    "AutoBuffManager",
    "RecoveryManager",
]


class ConsumablesDomain:
    """Aggregate domain for consumables management."""

    name = "consumables"
    priority = 30

    def __init__(self) -> None:
        self.buffs = AutoBuffManager()
        self.recovery = RecoveryManager()
