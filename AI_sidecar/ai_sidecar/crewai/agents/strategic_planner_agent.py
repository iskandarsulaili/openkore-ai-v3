from __future__ import annotations

from typing import Any
from .base_agent import BehaviorProfile



class StrategicPlannerProfile(BehaviorProfile):
    """Long-term goals, leveling route."""

    agent_id = "strategic_planner"
    role = "Strategic Planner"
    goal = "Plan the optimal leveling route and long-term progression path"
    backstory = (
        "A grandmaster strategist who sees the big picture. This agent "
        "plans the most efficient leveling routes, identifies the best "
        "hunting grounds for each level range, and adjusts the long-term "
        "plan based on real-world character progression."
    )

    def can_handle(self, signals: dict[str, Any]) -> float:
        level = signals.get("level", 1)
        if level < 10:
            return 0.2  # early game — just go
        return 0.4

    def get_action(self, signals: dict[str, Any]) -> dict[str, Any] | None:
        level = signals.get("level", 1)
        current_map = signals.get("map", "")
        base_exp = signals.get("base_exp", 0)
        job_exp = signals.get("job_exp", 0)
        max_base_exp = max(signals.get("max_base_exp", 1), 1)
        max_job_exp = max(signals.get("max_job_exp", 1), 1)
        base_pct = base_exp / max_base_exp
        job_pct = job_exp / max_job_exp

        if base_pct > 0.95 and job_pct > 0.5:
            return {"kind": "leveling", "command": f"move_route {current_map}", "confidence": 0.6, "reason": "Close to level up, staying on current map"}

        # Recommend next hunting area based on level
        hunting_areas = signals.get("hunting_areas", [])
        if hunting_areas:
            next_area = hunting_areas[0]
            if next_area.get("map") != current_map:
                return {"kind": "travel", "command": f"move {next_area.get('map', '')}", "confidence": 0.5, "reason": f"Moving to recommended hunting area: {next_area.get('name', '?')}"}

        return None
