from __future__ import annotations

from typing import Any

from ai_sidecar.crewai.agents.base_agent import build_agent
from ai_sidecar.crewai.config import AGENT_PROFILES


def create_social_coordinator_agent(*, llm: Any, tools: list[Any], verbose: bool) -> Any:
    profile = next(item for item in AGENT_PROFILES if item.agent_id == "social_coordinator")
    return build_agent(profile, llm=llm, tools=tools, allow_delegation=False, verbose=verbose)

