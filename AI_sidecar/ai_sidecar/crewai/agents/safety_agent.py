from __future__ import annotations

from typing import Any
from .base_agent import BehaviorProfile



class SafetyProfile(BehaviorProfile):
    """HP/SP monitoring, emergency protocols."""

    agent_id = "safety"
    role = "Safety Officer"
    goal = "Prevent character death by monitoring HP/SP and triggering emergency responses"
    backstory = (
        "Always watching the vital bars, this agent is the last line of "
        "defense against death. It triggers healing, teleport escapes, and "
        "emergency potions the moment things get dangerous."
    )

    def can_handle(self, signals: dict[str, Any]) -> float:
        hp = signals.get("hp", 0)
        hp_max = max(signals.get("hp_max", 1), 1)
        sp = signals.get("sp", 0)
        sp_max = max(signals.get("sp_max", 1), 1)
        hp_pct = hp / hp_max
        sp_pct = sp / sp_max
        score = 0.0
        if hp_pct < 0.5:
            score += 0.5 * (1.0 - hp_pct)
        if sp_pct < 0.3:
            score += 0.3 * (1.0 - sp_pct)
        if hp_pct < 0.1:
            score = 1.0  # emergency override
        return min(score, 1.0)

    def get_action(self, signals: dict[str, Any]) -> dict[str, Any] | None:
        hp = signals.get("hp", 0)
        hp_max = max(signals.get("hp_max", 1), 1)
        sp = signals.get("sp", 0)
        sp_max = max(signals.get("sp_max", 1), 1)
        hp_pct = hp / hp_max
        sp_pct = sp / sp_max

        if hp_pct < 0.15:
            return {"kind": "emergency", "command": "teleport auto", "confidence": 1.0, "reason": "CRITICAL HP — emergency teleport"}
        if hp_pct < 0.4:
            return {"kind": "heal", "command": "use_skill 0 0 0", "confidence": 0.9, "reason": f"HP low ({hp_pct:.0%}), healing"}
        if sp_pct < 0.2:
            return {"kind": "recover_sp", "command": "sit auto", "confidence": 0.7, "reason": f"SP low ({sp_pct:.0%}), sitting to recover"}
        return None
