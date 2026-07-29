"""Crafting domain — alchemy, cooking, forging."""
from __future__ import annotations

from ai_sidecar.domains.crafting.alchemy import AlchemyCrafting
from ai_sidecar.domains.crafting.cooking import CookingCrafting
from ai_sidecar.domains.crafting.forging import ForgingCrafting

__all__ = [
    "CraftingDomain",
    "AlchemyCrafting",
    "CookingCrafting",
    "ForgingCrafting",
]


class CraftingDomain:
    """Aggregate domain for all crafting activities."""

    name = "crafting"
    priority = 55

    def __init__(self) -> None:
        self.alchemy = AlchemyCrafting()
        self.cooking = CookingCrafting()
        self.forging = ForgingCrafting()
