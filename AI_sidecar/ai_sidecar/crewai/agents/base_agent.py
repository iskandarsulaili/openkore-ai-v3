from __future__ import annotations

from typing import Any


class BehaviorProfile:
    """Base class for heuristic behavior profiles. No CrewAI dependency."""

    agent_id: str
    role: str
    goal: str
    backstory: str

    def can_handle(self, signals: dict[str, Any]) -> float:
        """Return a relevance score [0.0, 1.0] for the current game state."""
        return 0.0

    def get_action(self, signals: dict[str, Any]) -> dict[str, Any] | None:
        """Return a heuristic action dict (kind, command, confidence, reason) or None."""
        return None
