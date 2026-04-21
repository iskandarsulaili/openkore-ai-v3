from __future__ import annotations

from typing import Any

from ai_sidecar.crewai.agents.base_agent import build_agent
from ai_sidecar.crewai.config import MANAGER_PROFILE


def create_manager_agent(*, llm: Any, tools: list[Any], verbose: bool) -> Any:
    return build_agent(MANAGER_PROFILE, llm=llm, tools=tools, allow_delegation=True, verbose=verbose)

