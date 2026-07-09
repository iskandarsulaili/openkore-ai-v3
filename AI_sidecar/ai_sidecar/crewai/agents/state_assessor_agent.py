from __future__ import annotations

from typing import Any
from .base_agent import BehaviorProfile



class StateAssessorProfile(BehaviorProfile):
    """Evaluates current state quality."""

    agent_id = "state_assessor"
    role = "State Assessor"
    goal = "Evaluate the quality and safety of the current bot state"
    backstory = (
        "The situational awareness module. This agent continuously "
        "evaluates whether the bot is in a good state — healthy, "
        "unstuck, adequately supplied, in a safe zone — and flags "
        "any issues that need attention."
    )

    def can_handle(self, signals: dict[str, Any]) -> float:
        # Always ready to assess
        return 0.2

    def get_action(self, signals: dict[str, Any]) -> dict[str, Any] | None:
        issues = []
        hp_pct = signals.get("hp", 0) / max(signals.get("hp_max", 1), 1)
        weight_pct = signals.get("weight", 0) / max(signals.get("max_weight", 1), 1)

        if hp_pct < 0.3:
            issues.append("low_hp")
        if weight_pct > 0.9:
            issues.append("overweight")
        if signals.get("state") == "stuck":
            issues.append("stuck")
        if signals.get("state") == "lost":
            issues.append("lost")

        if not issues:
            return {"kind": "assess", "command": "", "confidence": 1.0, "reason": "State nominal"}

        return {
            "kind": "assess",
            "command": f"flag {' '.join(issues)}",
            "confidence": 0.8,
            "reason": f"Issues detected: {', '.join(issues)}",
        }
