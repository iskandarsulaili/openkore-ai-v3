"""Companions domain — pets, homunculus, mercenary."""
from __future__ import annotations

from ai_sidecar.domains.companions.pets import PetManager
from ai_sidecar.domains.companions.homunculus import HomunculusManager
from ai_sidecar.domains.companions.mercenary import MercenaryManager

__all__ = [
    "CompanionsDomain",
    "PetManager",
    "HomunculusManager",
    "MercenaryManager",
]


class CompanionsDomain:
    """Aggregate domain for companion management."""

    name = "companions"
    priority = 40

    def __init__(self) -> None:
        self.pets = PetManager()
        self.homunculus = HomunculusManager()
        self.mercenary = MercenaryManager()
