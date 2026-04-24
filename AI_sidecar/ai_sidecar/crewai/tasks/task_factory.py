from __future__ import annotations

from typing import Any
import logging

from ai_sidecar.crewai.config import AGENT_PROFILES

logger = logging.getLogger(__name__)


def build_collaborative_tasks(
    *,
    objective: str,
    task_hint: str,
    agents_by_id: dict[str, Any],
) -> list[Any]:
    from crewai import Task

    tasks: list[Any] = []
    previous: list[Any] = []
    # NOTE:
    # crewAI enforces strict async-task ordering constraints, and newer releases
    # may inject internal tasks (e.g. planning/manager stages) after user tasks.
    # That can make an otherwise-valid "last user task async" configuration fail
    # with `async_task_count` validation errors during Crew construction.
    # Keep collaborative tasks synchronous for deterministic compatibility.
    async_task_names: list[str] = []
    available_profiles = [profile for profile in AGENT_PROFILES if profile.agent_id in agents_by_id]
    last_index = len(available_profiles) - 1
    for index, profile in enumerate(available_profiles):
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
        async_execution = False
        task_name = f"task_{profile.agent_id}"
        tasks.append(
            Task(
                name=task_name,
                description=description,
                expected_output=expected_output,
                agent=agent,
                context=previous[-2:] if previous else None,
                async_execution=async_execution,
            )
        )
        previous.append(tasks[-1])

    if tasks:
        logger.info(
            "crewai_tasks_built",
            extra={
                "event": "crewai_tasks_built",
                "task_count": len(tasks),
                "async_task_count": len(async_task_names),
                "async_task_names": list(async_task_names),
                "async_mode": "disabled_for_sdk_compatibility",
                "agent_ids": [profile.agent_id for profile in available_profiles],
            },
        )
    return tasks
