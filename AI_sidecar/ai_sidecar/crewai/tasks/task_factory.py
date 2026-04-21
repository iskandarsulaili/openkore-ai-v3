from __future__ import annotations

from typing import Any

from ai_sidecar.crewai.config import AGENT_PROFILES


def build_collaborative_tasks(
    *,
    objective: str,
    task_hint: str,
    agents_by_id: dict[str, Any],
) -> list[Any]:
    from crewai import Task

    tasks: list[Any] = []
    previous: list[Any] = []
    for profile in AGENT_PROFILES:
        agent = agents_by_id.get(profile.agent_id)
        if agent is None:
            continue

        description = (
            f"Objective: {objective}\n"
            f"Global task hint: {task_hint}\n"
            f"You are {profile.role}. Provide concrete, low-ambiguity recommendations in your specialty."
        )
        expected_output = (
            "Return concise actionable recommendations, explicit risks, and assumptions. "
            "If blocked, clearly state blocker reason and safe fallback."
        )
        tasks.append(
            Task(
                name=f"task_{profile.agent_id}",
                description=description,
                expected_output=expected_output,
                agent=agent,
                context=previous[-2:] if previous else None,
                async_execution=True,
            )
        )
        previous.append(tasks[-1])

    return tasks

