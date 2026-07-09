from __future__ import annotations

from typing import Any
from .base_agent import BehaviorProfile



class ProgressionPlannerProfile(BehaviorProfile):
    """Equipment upgrades, job change."""

    agent_id = "progression_planner"
    role = "Progression Planner"
    goal = "Plan equipment upgrades and job changes at the right time"
    backstory = (
        "Always looking ahead to the next power spike. This agent knows "
        "the stat requirements for job changes, the equipment sets for "
        "each level bracket, and when to grind for that critical upgrade."
    )

    def can_handle(self, signals: dict[str, Any]) -> float:
        level = signals.get("level", 1)
        job_level = signals.get("job_level", 1)
        can_change = bool(signals.get("job_change_available"))
        score = 0.0
        if can_change:
            score += 0.8
        if level % 10 == 0:  # every 10 levels, check gear
            score += 0.4
        return min(score, 1.0)

    def get_action(self, signals: dict[str, Any]) -> dict[str, Any] | None:
        if signals.get("job_change_available"):
            job_change_npc = signals.get("job_change_npc")
            if job_change_npc:
                return {"kind": "job_change", "command": f"move {job_change_npc}", "confidence": 0.9, "reason": "Job change available, proceeding"}

        equipment = signals.get("equipment", [])
        level = signals.get("level", 1)
        outdated = [e for e in equipment if e.get("required_level", 999) < level - 15]
        if outdated:
            return {"kind": "equipment_check", "command": f"equip_list", "confidence": 0.6, "reason": f"Checking {len(outdated)} outdated equipment pieces"}

        return None
