"""Character progression domain — lifecycle state machine and job change automation.

Provides:
  - ProgressionDomain: domain integration for the heuristic assessment loop
  - LifecycleStateMachine: tracks NOVICE → ENDGAME with per-phase configs
  - AdvancementDomain: auto-detect job changes and execute NPC interaction
"""
from __future__ import annotations

from typing import Any

from ai_sidecar.actions import HeuristicAction
from ai_sidecar.domains import BaseDomain
from ai_sidecar.domains.progression.lifecycle import (
    LifecycleStateMachine,
    LifecyclePhase,
    PhaseConfig,
)
from ai_sidecar.domains.progression.advancement import (
    AdvancementDomain,
    JobChangePlan,
    JobChangeStep,
)

logger = __import__("logging").getLogger(__name__)

__all__ = [
    "ProgressionDomain",
    "LifecycleStateMachine",
    "LifecyclePhase",
    "PhaseConfig",
    "AdvancementDomain",
    "JobChangePlan",
    "JobChangeStep",
]


class ProgressionDomain(BaseDomain):
    """Domain for character progression — lifecycle state machine and job
    change automation.

    Runs after combat (priority 30) and handles all long-term character
    development decisions: stat allocation, skill training, map preference,
    gear targets, and job advancement.
    """

    name: str = "progression"
    priority: int = 30

    def __init__(self) -> None:
        super().__init__()
        self.lifecycle = LifecycleStateMachine()
        self.advancement = AdvancementDomain()

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        bot_id: str,
    ) -> None:
        """Assess progression state and emit phase-appropriate + advancement actions."""
        # Phase-based lifecycle decisions
        self.lifecycle.assess(signals, actions, bot_id)

        # Job change automation
        self.advancement.assess(signals, actions, bot_id)


# Convenience factory used by DomainRegistry
def create_domain() -> ProgressionDomain:
    return ProgressionDomain()
