from __future__ import annotations

from typing import Any

from ai_sidecar.crewai.agents.combat_agent import create_combat_agent
from ai_sidecar.crewai.agents.economy_agent import create_economy_agent
from ai_sidecar.crewai.agents.fleet_liaison_agent import create_fleet_liaison_agent
from ai_sidecar.crewai.agents.macro_engineer_agent import create_macro_engineer_agent
from ai_sidecar.crewai.agents.manager_agent import create_manager_agent
from ai_sidecar.crewai.agents.navigation_agent import create_navigation_agent
from ai_sidecar.crewai.agents.questing_agent import create_questing_agent
from ai_sidecar.crewai.agents.safety_agent import create_safety_agent
from ai_sidecar.crewai.agents.social_agent import create_social_agent


def create_agent_by_id(*, agent_id: str, llm: Any, tools: list[Any], verbose: bool) -> Any:
    if agent_id == "combat":
        return create_combat_agent(llm=llm, tools=tools, verbose=verbose)
    if agent_id == "navigation":
        return create_navigation_agent(llm=llm, tools=tools, verbose=verbose)
    if agent_id == "questing":
        return create_questing_agent(llm=llm, tools=tools, verbose=verbose)
    if agent_id == "economy":
        return create_economy_agent(llm=llm, tools=tools, verbose=verbose)
    if agent_id == "social":
        return create_social_agent(llm=llm, tools=tools, verbose=verbose)
    if agent_id == "safety":
        return create_safety_agent(llm=llm, tools=tools, verbose=verbose)
    if agent_id == "fleet_liaison":
        return create_fleet_liaison_agent(llm=llm, tools=tools, verbose=verbose)
    if agent_id == "macro_engineer":
        return create_macro_engineer_agent(llm=llm, tools=tools, verbose=verbose)
    raise ValueError(f"unknown_agent_id:{agent_id}")


__all__ = [
    "create_agent_by_id",
    "create_manager_agent",
]

