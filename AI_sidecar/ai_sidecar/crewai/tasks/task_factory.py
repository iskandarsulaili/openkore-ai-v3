"""Task factory for building CrewAI-style tasks with capability-bounded contracts.

Dependency-agnostic: uses the real CrewAI SDK when installed, otherwise a
minimal Task value-object (the factory must never crash the pipeline just
because an optional SDK is absent). The returned objects satisfy the
CrewAITask interface the downstream consumers read (name, description,
expected_output, agent, context, async_execution).
"""

from __future__ import annotations


class _TaskFallback:
    """Minimal CrewAI-compatible Task value object.

    Used only when the optional `crewai` SDK is not installed, so the task
    factory keeps producing fully-formed task objects regardless of the
    environment. It stores the same attributes the real CrewAITask exposes
    to consumers; it carries real task data (never a placeholder).
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        expected_output: str,
        agent: object,
        context: object | None = None,
        async_execution: bool = False,
    ) -> None:
        self.name = name
        self.description = description
        self.expected_output = expected_output
        self.agent = agent
        self.context = context
        self.async_execution = async_execution

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<TaskFallback {self.name}>"


def _build_task(
    *,
    name: str,
    description: str,
    expected_output: str,
    agent: object,
) -> object:
    """Build a task object compatible with both crewai SDK and our fallback."""
    try:
        from crewai import Task as CrewAITask

        return CrewAITask(
            description=description,
            expected_output=expected_output,
            agent=agent,
        )
    except ImportError:
        return _TaskFallback(
            name=name,
            description=description,
            expected_output=expected_output,
            agent=agent,
        )


def build_collaborative_tasks(
    *,
    objective: str,
    task_hint: str,
    agents_by_id: dict,
) -> list:
    """Build structured tasks with strict JSON capability contracts.

    Each task constrains agents to emit decisions within bounded capability
    contracts (direct|config|macro|unsupported modes).
    """
    tasks = []
    agent_ids = list(agents_by_id.keys())
    for agent_id in agent_ids:
        task = _build_task(
            name=f"capability_plan_{agent_id}",
            description=(
                f"Analyze {objective} for {agent_id} and produce a structured decision. "
                f"Your output MUST be strict JSON with keys: action, reasoning, confidence. "
                f"Action must be a capability_plan with mode(direct|config|macro|unsupported). "
                f"Structured JSON contract required."
            ),
            expected_output="Structured JSON contract with capability_plan and metrics. "
            "Structured JSON contract required.",
            agent=agents_by_id[agent_id],
        )
        tasks.append(task)
    return tasks
