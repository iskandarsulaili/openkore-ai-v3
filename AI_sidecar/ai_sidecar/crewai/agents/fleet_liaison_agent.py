from __future__ import annotations

from typing import Any
from .base_agent import BehaviorProfile



class FleetLiaisonProfile(BehaviorProfile):
    """Multi-bot coordination, inter-bot messaging."""

    agent_id = "fleet_liaison"
    role = "Fleet Liaison"
    goal = "Coordinate activities across multiple bots in the fleet"
    backstory = (
        "The communications hub for multi-bot operations. This agent "
        "manages inter-bot messaging, shares map positions, coordinates "
        "party roles, and ensures the fleet moves as a cohesive unit "
        "rather than a collection of solo bots."
    )

    def can_handle(self, signals: dict[str, Any]) -> float:
        fleet = signals.get("fleet_members", [])
        fleet_msg = signals.get("fleet_message")
        if fleet_msg:
            return 0.9
        if fleet and len(fleet) > 0:
            return 0.6
        return 0.0

    def get_action(self, signals: dict[str, Any]) -> dict[str, Any] | None:
        message = signals.get("fleet_message")
        if message:
            return {
                "kind": "fleet_relay",
                "command": f"relay {message.get('from', '')} {message.get('text', '')}",
                "confidence": 0.9,
                "reason": f"Relaying fleet message from {message.get('from', '?')}",
            }

        fleet = signals.get("fleet_members", [])
        if fleet:
            pos = signals.get("position", "")
            return {
                "kind": "fleet_broadcast",
                "command": f"broadcast pos {pos}",
                "confidence": 0.6,
                "reason": f"Broadcasting position to {len(fleet)} fleet members",
            }

        return None
