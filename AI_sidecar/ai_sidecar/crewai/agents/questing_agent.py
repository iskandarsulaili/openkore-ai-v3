from __future__ import annotations

from typing import Any
from .base_agent import BehaviorProfile



class QuestingProfile(BehaviorProfile):
    """Quest tracking, turn-in, accept."""

    agent_id = "questing"
    role = "Quest Specialist"
    goal = "Complete quests efficiently from acceptance through turn-in"
    backstory = (
        "An adventurer who lives for the quest log. This agent tracks every "
        "active quest, knows where to find objectives, and never forgets to "
        "turn in completed tasks for maximum rewards."
    )

    def can_handle(self, signals: dict[str, Any]) -> float:
        quests = signals.get("quests", [])
        if not quests:
            return 0.0
        completable = [q for q in quests if q.get("status") == "complete"]
        active = [q for q in quests if q.get("status") == "active"]
        score = 0.0
        if completable:
            score += 0.6
        if active:
            score += 0.4
        return min(score, 1.0)

    def get_action(self, signals: dict[str, Any]) -> dict[str, Any] | None:
        quests = signals.get("quests", [])
        completable = [q for q in quests if q.get("status") == "complete"]
        if completable:
            q = completable[0]
            return {"kind": "quest_turnin", "command": f"quest turnin {q.get('id', '')}", "confidence": 0.9, "reason": f"Turning in completed quest {q.get('name', '?')}"}

        active = [q for q in quests if q.get("status") == "active"]
        if active:
            q = active[0]
            return {"kind": "quest_progress", "command": f"quest goto {q.get('id', '')}", "confidence": 0.7, "reason": f"Progressing quest {q.get('name', '?')}"}

        return None
