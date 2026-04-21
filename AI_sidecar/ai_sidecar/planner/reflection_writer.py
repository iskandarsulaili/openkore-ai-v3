from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ReflectionWriter:
    memory_service: object

    def write(
        self,
        *,
        bot_id: str,
        plan_id: str,
        objective: str,
        succeeded: bool,
        rationale: str,
        metadata: dict[str, object] | None = None,
    ) -> None:
        message = (
            f"plan={plan_id} objective={objective} outcome={'success' if succeeded else 'failure'} "
            f"rationale={rationale}"
        )
        payload = dict(metadata or {})
        payload["plan_id"] = plan_id
        payload["succeeded"] = succeeded
        self.memory_service.capture_action(
            bot_id=bot_id,
            action_id=f"reflection:{plan_id}",
            kind="planner_reflection",
            message=message,
            metadata=payload,
        )

