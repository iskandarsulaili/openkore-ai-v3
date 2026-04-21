from __future__ import annotations

from typing import Any

from ai_sidecar.crewai.config import AgentProfile


def build_agent(
    profile: AgentProfile,
    *,
    llm: Any,
    tools: list[Any],
    allow_delegation: bool,
    verbose: bool,
) -> Any:
    from crewai import Agent

    return Agent(
        role=profile.role,
        goal=profile.goal,
        backstory=profile.backstory,
        tools=tools,
        llm=llm,
        allow_delegation=allow_delegation,
        verbose=verbose,
    )

