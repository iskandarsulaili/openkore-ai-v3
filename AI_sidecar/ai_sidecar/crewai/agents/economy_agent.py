from __future__ import annotations

from typing import Any
from .base_agent import BehaviorProfile



class EconomyProfile(BehaviorProfile):
    """Vending, buying, item storage."""

    agent_id = "economy"
    role = "Economy Manager"
    goal = "Maximize zeny through smart vending, buying, and inventory management"
    backstory = (
        "A shrewd merchant who never misses a profitable deal. This agent "
        "tracks market prices, manages the shop, and ensures the character "
        "always has enough zeny for essentials."
    )

    def can_handle(self, signals: dict[str, Any]) -> float:
        zeny = signals.get("zeny", 0)
        weight_pct = signals.get("weight", 0) / max(signals.get("max_weight", 1), 1)
        score = 0.0
        if signals.get("vending_active"):
            score += 0.5
        if weight_pct > 0.8:
            score += 0.3
        if zeny < 1000:
            score += 0.2
        return min(score, 1.0)

    def get_action(self, signals: dict[str, Any]) -> dict[str, Any] | None:
        weight_pct = signals.get("weight", 0) / max(signals.get("max_weight", 1), 1)
        if weight_pct > 0.85:
            return {"kind": "store", "command": "storage add all", "confidence": 0.8, "reason": "Inventory near full, storing items"}

        if signals.get("vending_active"):
            return {"kind": "vend", "command": "shop open", "confidence": 0.7, "reason": "Running player shop"}

        zeny = signals.get("zeny", 0)
        if zeny < 500:
            return {"kind": "buy", "command": "buy 0 Arrow 100", "confidence": 0.5, "reason": "Low zeny — conserving funds"}

        return None
