"""Plan executor — converts plans to action queue entries."""
from __future__ import annotations

from datetime import timedelta
import logging
from typing import Any, Optional, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from ai_sidecar.autonomy.pdca_loop import Horizon
from ai_sidecar.contracts.common import utc_now
from ai_sidecar.planner.schemas import StrategicPlan, TacticalIntentBundle
from ai_sidecar.contracts.actions import ActionProposal, ActionPriorityTier

logger = logging.getLogger(__name__)


class PlanExecutor:
    """Converts plan outputs into queued actions."""

    def __init__(self, runtime_state: Any) -> None:
        self._runtime = runtime_state

    async def execute(
        self,
        plan: StrategicPlan | TacticalIntentBundle | None,
        horizon: "Horizon",
        max_actions: int = 5,
    ) -> int:
        """Execute a plan by converting its intents to action queue entries.

        Returns the number of actions successfully queued.
        """
        if plan is None:
            return 0

        actions_queued = 0
        bot_id = str(getattr(plan, "bot_id", None) or "openkoreai")

        try:
            if isinstance(plan, StrategicPlan):
                actions_queued = await self._execute_strategic(plan, bot_id, max_actions)
            elif isinstance(plan, TacticalIntentBundle):
                actions_queued = await self._execute_tactical(plan, bot_id, max_actions)
            else:
                # Generic dict-like plan
                actions_queued = await self._execute_generic(plan, bot_id, max_actions)
        except Exception:
            horizon_value = getattr(horizon, "value", str(horizon))
            logger.exception("PlanExecutor.execute failed for horizon=%s", horizon_value)

        return actions_queued

    async def _execute_strategic(
        self, plan: StrategicPlan, bot_id: str, max_actions: int
    ) -> int:
        """Convert a StrategicPlan into queued actions."""
        count = 0
        recommended_actions = list(getattr(plan, "recommended_actions", []) or [])
        for action in recommended_actions[:max_actions]:
            await self._queue_action(action, bot_id)
            count += 1
        if count:
            return count

        # Strategic plans contain high-level objectives
        objectives = getattr(plan, "steps", []) or []
        for obj in objectives[:max_actions]:
            action = self._objective_to_action(obj)
            if action:
                await self._queue_action(action, bot_id)
                count += 1
        return count

    async def _execute_tactical(
        self, bundle: TacticalIntentBundle, bot_id: str, max_actions: int
    ) -> int:
        """Convert a TacticalIntentBundle into queued actions."""
        count = 0
        ready_actions = list(getattr(bundle, "actions", []) or [])
        for action in ready_actions[:max_actions]:
            await self._queue_action(action, bot_id)
            count += 1
        if count:
            return count

        intents = getattr(bundle, "intents", []) or []
        for intent in intents[:max_actions]:
            action = self._intent_to_action(intent)
            if action:
                await self._queue_action(action, bot_id)
                count += 1
        return count

    async def _execute_generic(
        self, plan: Any, bot_id: str, max_actions: int
    ) -> int:
        """Fallback for dict-like or unknown plan types."""
        count = 0
        # Try common field names
        for field in ("recommended_actions", "actions", "steps", "tasks", "commands", "intents"):
            items = getattr(plan, field, None)
            if isinstance(items, list):
                for item in items[:max_actions]:
                    action = item if isinstance(item, ActionProposal) else self._generic_item_to_action(item)
                    if action:
                        await self._queue_action(action, bot_id)
                        count += 1
                break
        return count

    def _objective_to_action(self, objective: Any) -> ActionProposal | None:
        """Convert a strategic objective to an action proposal."""
        try:
            description = (
                getattr(objective, "description", None)
                or getattr(objective, "name", None)
                or str(objective)
            )
            return self._build_action_proposal(
                command="ai auto",
                priority=ActionPriorityTier.strategic,
                source="pdca_loop_strategic",
                conflict_key="pdca.strategic",
                metadata={"description": description},
            )
        except Exception:
            logger.exception("Failed to convert objective to action")
            return None

    def _intent_to_action(self, intent: Any) -> ActionProposal | None:
        """Convert a tactical intent to an action proposal."""
        try:
            objective = getattr(intent, "objective", None) or getattr(intent, "type", None) or "tactical_action"
            return self._build_action_proposal(
                command="ai auto",
                priority=ActionPriorityTier.tactical,
                source="pdca_loop_tactical",
                conflict_key="pdca.tactical",
                metadata={"objective": str(objective)},
            )
        except Exception:
            logger.exception("Failed to convert intent to action")
            return None

    def _generic_item_to_action(self, item: Any) -> ActionProposal | None:
        """Convert a generic plan item to an action proposal."""
        try:
            if isinstance(item, dict):
                command = str(item.get("command") or "ai auto")
                metadata = {k: v for k, v in item.items() if k != "command"}
            else:
                command = str(getattr(item, "command", None) or "ai auto")
                metadata = {"item": str(getattr(item, "description", None) or getattr(item, "type", None) or item)}
            return self._build_action_proposal(
                command=command,
                priority=ActionPriorityTier.tactical,
                source="pdca_loop_generic",
                metadata=metadata if isinstance(metadata, dict) else {},
            )
        except Exception:
            logger.exception("Failed to convert generic item to action")
            return None

    async def _queue_action(self, action: ActionProposal, bot_id: str) -> None:
        """Queue a single action via the runtime."""
        try:
            self._runtime.queue_action(proposal=action, bot_id=bot_id)
            logger.debug("Queued action: %s [%s]", action.command, action.priority_tier)
        except Exception:
            logger.exception("Failed to queue action: %s", action.command)

    def _build_action_proposal(
        self,
        *,
        command: str,
        priority: ActionPriorityTier,
        source: str,
        conflict_key: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ActionProposal:
        now = utc_now()
        action_id = f"pdca-{uuid4().hex[:20]}"
        return ActionProposal(
            action_id=action_id,
            kind="command",
            command=command[:256],
            priority_tier=priority,
            conflict_key=conflict_key,
            created_at=now,
            expires_at=now + timedelta(seconds=90),
            idempotency_key=f"{source}:{action_id}"[:128],
            metadata={"source": source, **dict(metadata or {})},
        )
