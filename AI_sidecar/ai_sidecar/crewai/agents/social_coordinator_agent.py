from __future__ import annotations

from typing import Any
from .base_agent import BehaviorProfile



class SocialCoordinatorProfile(BehaviorProfile):
    """Party coordination."""

    agent_id = "social_coordinator"
    role = "Party Coordinator"
    goal = "Coordinate party member positions, roles, and support actions"
    backstory = (
        "The party leader when no human is around. This agent assigns "
        "party roles, coordinates formation movement, triggers support "
        "skills, and ensures the party stays together and effective."
    )

    def can_handle(self, signals: dict[str, Any]) -> float:
        party = signals.get("party_members", [])
        if not party:
            return 0.0
        if len(party) >= 2:
            return 0.7
        return 0.3

    def get_action(self, signals: dict[str, Any]) -> dict[str, Any] | None:
        party = signals.get("party_members", [])
        if not party:
            return None

        members_online = [m for m in party if m.get("online")]
        offline = [m for m in party if not m.get("online")]

        if offline:
            return {"kind": "party_check", "command": f"party check", "confidence": 0.5, "reason": f"{len(offline)} party member(s) offline"}

        if len(members_online) >= 2:
            # Check if party needs healing
            for m in members_online:
                hp_pct = m.get("hp", 1) / max(m.get("hp_max", 1), 1)
                if hp_pct < 0.5:
                    return {"kind": "party_heal", "command": f"use_skill 0 0 {m.get('name', '')}", "confidence": 0.75, "reason": f"Healing party member {m.get('name', '?')} (HP {hp_pct:.0%})"}

        return None
