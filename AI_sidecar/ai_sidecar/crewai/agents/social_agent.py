from __future__ import annotations

from typing import Any
from .base_agent import BehaviorProfile



class SocialProfile(BehaviorProfile):
    """Party, guild, chat interactions."""

    agent_id = "social"
    role = "Social Butterfly"
    goal = "Maintain positive relationships with party members and guild"
    backstory = (
        "The face of the operation. This agent handles party invitations, "
        "guild communication, and keeps the social fabric intact so the "
        "bot is never seen as antisocial or unresponsive."
    )

    def can_handle(self, signals: dict[str, Any]) -> float:
        party = signals.get("party_members", [])
        guild = signals.get("guild_members", [])
        pending_invites = signals.get("pending_invites", [])
        score = 0.0
        if pending_invites:
            score += 0.7
        if party and len(party) < 4:
            score += 0.3
        if guild:
            score += 0.2
        return min(score, 1.0)

    def get_action(self, signals: dict[str, Any]) -> dict[str, Any] | None:
        invites = signals.get("pending_invites", [])
        party_invites = [i for i in invites if i.get("type") == "party"]
        if party_invites:
            inv = party_invites[0]
            return {"kind": "party_accept", "command": f"party accept {inv.get('from', '')}", "confidence": 0.8, "reason": f"Accepting party invite from {inv.get('from', '?')}"}

        party = signals.get("party_members", [])
        if len(party) < 4:
            near = signals.get("nearby_players", [])
            if near:
                return {"kind": "party_invite", "command": f"party invite {near[0].get('name', '')}", "confidence": 0.5, "reason": "Party has openings, inviting nearby player"}

        return None
