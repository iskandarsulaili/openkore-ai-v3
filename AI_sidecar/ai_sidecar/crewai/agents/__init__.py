from __future__ import annotations

from typing import Any

from ai_sidecar.crewai.agents.manager_agent import create_manager_agent
from ai_sidecar.crewai.agents.resource_manager_agent import create_resource_manager_agent
from ai_sidecar.crewai.agents.social_coordinator_agent import create_social_coordinator_agent
from ai_sidecar.crewai.agents.strategic_planner_agent import create_strategic_planner_agent
from ai_sidecar.crewai.agents.tactical_commander_agent import create_tactical_commander_agent
from ai_sidecar.crewai.config import LEGACY_AGENT_ROUTING


def _normalize_agent_id(agent_id: str) -> str:
    resolved = LEGACY_AGENT_ROUTING.get(agent_id, agent_id)
    return str(resolved)


def create_agent_by_id(*, agent_id: str, llm: Any, tools: list[Any], verbose: bool) -> Any:
    normalized = _normalize_agent_id(agent_id)
    if normalized == "tactical_commander":
        return create_tactical_commander_agent(llm=llm, tools=tools, verbose=verbose)
    if normalized == "strategic_planner":
        return create_strategic_planner_agent(llm=llm, tools=tools, verbose=verbose)
    if normalized == "resource_manager":
        return create_resource_manager_agent(llm=llm, tools=tools, verbose=verbose)
    if normalized == "social_coordinator":
        return create_social_coordinator_agent(llm=llm, tools=tools, verbose=verbose)
    raise ValueError(f"unknown_agent_id:{agent_id}->{normalized}")


__all__ = [
    "create_agent_by_id",
    "create_manager_agent",
]
