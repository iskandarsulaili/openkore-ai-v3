from __future__ import annotations

from typing import Any
from .base_agent import BehaviorProfile



class TacticalCommanderProfile(BehaviorProfile):
    """Combat tactics, skill rotation."""

    agent_id = "tactical_commander"
    role = "Tactical Commander"
    goal = "Execute optimal skill rotations and combat tactics for each encounter"
    backstory = (
        "A veteran of a thousand battles, this agent reads every fight "
        "with surgical precision. It knows exactly which skills to chain, "
        "when to buff, when to debuff, and how to position for maximum "
        "effectiveness against any foe."
    )

    def can_handle(self, signals: dict[str, Any]) -> float:
        monsters = signals.get("monsters_around", [])
        target = signals.get("target", {})
        in_combat = bool(target) or bool(monsters)
        if not in_combat:
            return 0.0
        return 0.7 if target else 0.3

    def get_action(self, signals: dict[str, Any]) -> dict[str, Any] | None:
        target = signals.get("target", {})
        if not target:
            return None

        skills = signals.get("skills", [])
        # first offensive skill in rotation
        offense = [s for s in skills if s.get("type") == "offensive"]
        if offense:
            skill = offense[0]
            return {"kind": "skill", "command": f"use_skill {skill.get('id', 0)} {target.get('id', '')}", "confidence": 0.85, "reason": f"Executing skill rotation: {skill.get('name', '?')}"}

        return {"kind": "attack", "command": f"attack {target.get('name', 'monster')}", "confidence": 0.6, "reason": "Auto-attacking target"}
