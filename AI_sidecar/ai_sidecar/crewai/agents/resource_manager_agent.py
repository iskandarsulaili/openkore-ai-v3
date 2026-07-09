from __future__ import annotations

from typing import Any
from .base_agent import BehaviorProfile



class ResourceManagerProfile(BehaviorProfile):
    """Inventory, weight, arrows, pots."""

    agent_id = "resource_manager"
    role = "Resource Manager"
    goal = "Ensure essential supplies (arrows, pots, ammo) are always stocked"
    backstory = (
        "The quartermaster who never lets supplies run dry. This agent "
        "tracks every consumable — arrows, potions, traps, ammo — and "
        "reorders before they hit critical lows. Weight management is "
        "second nature."
    )

    def can_handle(self, signals: dict[str, Any]) -> float:
        weight_pct = signals.get("weight", 0) / max(signals.get("max_weight", 1), 1)
        inventory = signals.get("inventory", {})
        consumables_low = any(
            item.get("quantity", 999) < 10
            for item in inventory.values()
            if item.get("type") in ("potion", "arrow", "ammunition")
        )
        score = 0.0
        if weight_pct > 0.75:
            score += 0.3
        if consumables_low:
            score += 0.5
        return min(score, 1.0)

    def get_action(self, signals: dict[str, Any]) -> dict[str, Any] | None:
        inventory = signals.get("inventory", {})
        weight_pct = signals.get("weight", 0) / max(signals.get("max_weight", 1), 1)

        # Reorder low consumables
        for item_id, item in inventory.items():
            if item.get("type") in ("potion", "arrow", "ammunition") and item.get("quantity", 999) < 10:
                return {"kind": "restock", "command": f"buy {item_id} 100", "confidence": 0.85, "reason": f"Restocking {item.get('name', '?')} (only {item.get('quantity', 0)} left)"}

        if weight_pct > 0.85:
            return {"kind": "cleanup", "command": "storage add all", "confidence": 0.7, "reason": "Weight high, storing surplus"}

        return None
