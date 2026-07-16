from __future__ import annotations

from typing import Any

from ai_sidecar.crewai.agents.base_agent import BehaviorProfile
from ai_sidecar.crewai.agents.combat_agent import CombatProfile
from ai_sidecar.crewai.agents.navigation_agent import NavigationProfile
from ai_sidecar.crewai.agents.economy_agent import EconomyProfile
from ai_sidecar.crewai.agents.questing_agent import QuestingProfile
from ai_sidecar.crewai.agents.safety_agent import SafetyProfile
from ai_sidecar.crewai.agents.social_agent import SocialProfile
from ai_sidecar.crewai.agents.manager_agent import ManagerProfile
from ai_sidecar.crewai.agents.tactical_commander_agent import TacticalCommanderProfile
from ai_sidecar.crewai.agents.strategic_planner_agent import StrategicPlannerProfile
from ai_sidecar.crewai.agents.progression_planner_agent import ProgressionPlannerProfile
from ai_sidecar.crewai.agents.resource_manager_agent import ResourceManagerProfile
from ai_sidecar.crewai.agents.command_emitter_agent import CommandEmitterProfile
from ai_sidecar.crewai.agents.macro_engineer_agent import MacroEngineerProfile
from ai_sidecar.crewai.agents.opportunistic_trader_agent import OpportunisticTraderProfile
from ai_sidecar.crewai.agents.fleet_liaison_agent import FleetLiaisonProfile
from ai_sidecar.crewai.agents.state_assessor_agent import StateAssessorProfile
from ai_sidecar.crewai.agents.social_coordinator_agent import SocialCoordinatorProfile
from ai_sidecar.crewai.agents.pro_ro_player_agent import ProRoPlayerProfile

# ---- New behavior-profile API (no CrewAI dependency) ----

AGENT_PROFILES: dict[str, type[BehaviorProfile]] = {
    "pro_ro_player": ProRoPlayerProfile,
    "combat": CombatProfile,
    "navigation": NavigationProfile,
    "economy": EconomyProfile,
    "questing": QuestingProfile,
    "safety": SafetyProfile,
    "social": SocialProfile,
    "manager": ManagerProfile,
    "tactical_commander": TacticalCommanderProfile,
    "strategic_planner": StrategicPlannerProfile,
    "progression_planner": ProgressionPlannerProfile,
    "resource_manager": ResourceManagerProfile,
    "command_emitter": CommandEmitterProfile,
    "macro_engineer": MacroEngineerProfile,
    "opportunistic_trader": OpportunisticTraderProfile,
    "fleet_liaison": FleetLiaisonProfile,
    "state_assessor": StateAssessorProfile,
    "social_coordinator": SocialCoordinatorProfile,
}


def get_profile(agent_id: str) -> BehaviorProfile:
    """Instantiate a profile by agent_id."""
    cls = AGENT_PROFILES.get(agent_id)
    if cls is None:
        raise ValueError(f"unknown behavior profile: {agent_id}")
    return cls()


def get_all_profiles() -> list[BehaviorProfile]:
    """Return one instance of every registered profile."""
    return [cls() for cls in AGENT_PROFILES.values()]


def best_profile(signals: dict) -> tuple[str, float]:
    """Score all profiles against the given signals and return the best (agent_id, score)."""
    best_id = ""
    best_score = -1.0
    for agent_id, cls in AGENT_PROFILES.items():
        profile = cls()
        score = profile.can_handle(signals)
        if score > best_score:
            best_score = score
            best_id = agent_id
    return best_id, best_score


# ---- Backward-compatible factory for crew_manager.py (constructs CrewAI Agent) ----


def _normalize_agent_id(agent_id: str) -> str:
    return str(agent_id)


def create_agent_by_id(*, agent_id: str, llm: Any = None, tools: list[Any] | None = None, verbose: bool = False) -> Any:
    """Get a behavior profile by agent ID. Returns the matching BehaviorProfile."""
    return get_profile(agent_id)


__all__ = [
    # Base
    "BehaviorProfile",
    # Profiles
    "ProRoPlayerProfile",
    "CombatProfile",
    "NavigationProfile",
    "EconomyProfile",
    "QuestingProfile",
    "SafetyProfile",
    "SocialProfile",
    "ManagerProfile",
    "TacticalCommanderProfile",
    "StrategicPlannerProfile",
    "ProgressionPlannerProfile",
    "ResourceManagerProfile",
    "CommandEmitterProfile",
    "MacroEngineerProfile",
    "OpportunisticTraderProfile",
    "FleetLiaisonProfile",
    "StateAssessorProfile",
    "SocialCoordinatorProfile",
    # New API
    "AGENT_PROFILES",
    "get_profile",
    "get_all_profiles",
    "best_profile",
    # Backward compat
    "create_agent_by_id",
]
