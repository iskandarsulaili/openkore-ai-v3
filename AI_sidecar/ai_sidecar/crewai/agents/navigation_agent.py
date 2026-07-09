from __future__ import annotations

from typing import Any
from .base_agent import BehaviorProfile



class NavigationProfile(BehaviorProfile):
    """Map movement, portal usage, auto-travel."""

    agent_id = "navigation"
    role = "Navigation Specialist"
    goal = "Navigate the world efficiently using portals and safe routes"
    backstory = (
        "A cartographer at heart, this agent knows the fastest paths between "
        "every town, dungeon, and field. It plans routes that avoid dangerous "
        "zones and minimizes travel time."
    )

    def can_handle(self, signals: dict[str, Any]) -> float:
        has_target = bool(signals.get("travel_target"))
        is_lost = signals.get("state") == "lost"
        if has_target:
            return 0.9
        if is_lost:
            return 0.8
        return 0.0

    def get_action(self, signals: dict[str, Any]) -> dict[str, Any] | None:
        target = signals.get("travel_target")
        current_map = signals.get("map", "")
        if target and current_map != target:
            return {"kind": "move", "command": f"move {target}", "confidence": 0.9, "reason": f"Traveling to {target}"}

        if signals.get("state") == "lost":
            return {"kind": "explore", "command": "auto_travel save", "confidence": 0.7, "reason": "Recovering from unknown position"}

        return None
