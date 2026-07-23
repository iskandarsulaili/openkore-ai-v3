from __future__ import annotations

from typing import Any

from .base_agent import BehaviorProfile
from .base_agent import BehaviorProfile

class ManagerProfile(BehaviorProfile):
    """Coordinates other agents, picks the best one for the current state."""

    agent_id = "manager"
    role = "Agent Coordinator"
    goal = "Select the most relevant behavior profile for the current game state"
    backstory = (
        "The executive function of the bot. This agent evaluates all "
        "available profiles, weighs their scores, and dispatches control "
        "to the one best suited for the moment — ensuring smooth, "
        "coordinated behavior across all subsystems."
    )

    def can_handle(self, signals: dict[str, Any]) -> float:
        # Manager always scores something — it's the decider
        return 0.5

    def get_action(self, signals: dict[str, Any]) -> dict[str, Any] | None:
        # The manager delegates — it doesn't produce a game action itself.
        # This is handled by the heuristic routing layer above the agent stack.
        return None

    def select_agent(self, profiles: list[BehaviorProfile], signals: dict[str, Any]) -> BehaviorProfile | None:
        """Pick the highest-scoring agent that can handle the current state."""
        best = None
        best_score = -1.0
        for profile in profiles:
            score = profile.can_handle(signals)
            if score > best_score:
                best_score = score
                best = profile
        return best if best_score > 0 else None

def create_manager_agent(*, llm: Any = None, tools: list[Any] | None = None, verbose: bool = False) -> Any:
    """Backward-compatible factory for crew_manager.py. Returns a ManagerProfile."""
    return ManagerProfile()
