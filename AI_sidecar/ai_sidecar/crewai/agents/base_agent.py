from __future__ import annotations

from typing import Any


class BehaviorProfile:
    """Base class for heuristic behavior profiles. No CrewAI dependency."""

    agent_id: str
    role: str
    goal: str
    backstory: str

    def __init__(self) -> None:
        self._outcomes: dict[str, dict[str, int]] = {}  # command -> {attempts, successes, failures}

    def can_handle(self, signals: dict[str, Any]) -> float:
        """Return a relevance score [0.0, 1.0] for the current game state."""
        return 0.0

    def get_action(self, signals: dict[str, Any]) -> dict[str, Any] | None:
        """Return a heuristic action dict (kind, command, confidence, reason) or None."""
        return None

    def record_outcome(self, command: str, success: bool) -> None:
        """Record whether a heuristic action succeeded or failed.
        
        This feeds back into can_handle() via a success-rate penalty.
        Commands with consistently poor outcomes get their confidence penalized.
        """
        if command not in self._outcomes:
            self._outcomes[command] = {"attempts": 0, "successes": 0, "failures": 0}
        self._outcomes[command]["attempts"] += 1
        if success:
            self._outcomes[command]["successes"] += 1
        else:
            self._outcomes[command]["failures"] += 1

    def _success_penalty(self, command: str) -> float:
        """Return a penalty multiplier [0.5, 1.0] based on historical success rate.
        
        A command with 0 failures gets 1.0 (no penalty).
        A command with 80%+ failure rate gets 0.5 (heavy penalty).
        """
        cmd_stats = self._outcomes.get(command)
        if not cmd_stats or cmd_stats["attempts"] < 3:
            return 1.0  # Not enough data, no penalty
        failure_rate = cmd_stats["failures"] / cmd_stats["attempts"]
        return max(0.5, 1.0 - failure_rate)

    def outcome_stats(self) -> dict[str, dict[str, int]]:
        """Return outcome statistics for all commands this profile has executed."""
        return dict(self._outcomes)
