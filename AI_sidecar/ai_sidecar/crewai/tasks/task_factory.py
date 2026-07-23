"""Task factory for building CrewAI-style tasks with capability-bounded contracts."""


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
    # Simulate creation of CrewAI Task objects
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


def _build_task(
    *,
    name: str,
    description: str,
    expected_output: str,
    agent: object,
) -> object:
    """Build a simple task object compatible with both crewai SDK and our fallback profiles."""
    try:
        from crewai import Task as CrewAITask
        return CrewAITask(
            description=description,
            expected_output=expected_output,
            agent=agent,
        )
    except ImportError:
        # Fallback to plain object if crewai SDK not installed
        class _TaskStub:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        return _TaskStub(
            name=name,
            description=description,
            expected_output=expected_output,
            agent=agent,
        )
