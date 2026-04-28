"""Plan executor — converts plans to action queue entries."""
from __future__ import annotations

from datetime import timedelta
import logging
from typing import Any, TYPE_CHECKING
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
            action = self._objective_to_action(obj, bot_id=bot_id)
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
            action = self._intent_to_action(intent, bot_id=bot_id)
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

    def _fleet_constraints_for_bot(self, *, bot_id: str) -> dict[str, object]:
        runtime = self._runtime

        resolver = getattr(runtime, "_fleet_constraints_for_bot", None)
        if callable(resolver):
            try:
                payload = resolver(bot_id=bot_id)
                if isinstance(payload, dict):
                    return payload
            except Exception:
                logger.debug("PlanExecutor: local fleet constraints unavailable", exc_info=True)

        public_resolver = getattr(runtime, "fleet_constraints", None)
        if callable(public_resolver):
            try:
                payload = public_resolver(bot_id=bot_id)
                if isinstance(payload, dict):
                    constraints = payload.get("constraints")
                    if isinstance(constraints, dict):
                        return constraints
                    return payload
                constraints = getattr(payload, "constraints", None)
                if isinstance(constraints, dict):
                    return constraints
            except Exception:
                logger.debug("PlanExecutor: public fleet constraints unavailable", exc_info=True)

        constraint_state = getattr(runtime, "fleet_constraint_state", None)
        resolver = getattr(constraint_state, "constraints_for_bot", None)
        if callable(resolver):
            try:
                payload = resolver(bot_id=bot_id)
                if isinstance(payload, dict):
                    return payload
            except Exception:
                logger.debug("PlanExecutor: state fleet constraints unavailable", exc_info=True)

        return {}

    def _current_map_for_bot(self, *, bot_id: str) -> str:
        cache = getattr(self._runtime, "snapshot_cache", None)
        getter = getattr(cache, "get", None)
        if not callable(getter):
            return ""
        try:
            snapshot = getter(bot_id)
        except Exception:
            logger.debug("PlanExecutor: snapshot cache lookup failed", exc_info=True)
            return ""

        position = getattr(snapshot, "position", None)
        map_name = str(getattr(position, "map", "") or "").strip()
        if map_name:
            return map_name
        raw = getattr(snapshot, "raw", None)
        if isinstance(raw, dict):
            return str(raw.get("map") or raw.get("map_name") or "").strip()
        return ""

    def _preferred_grind_maps(self, *, bot_id: str) -> list[str]:
        payload = self._fleet_constraints_for_bot(bot_id=bot_id)
        constraints = payload.get("constraints") if isinstance(payload.get("constraints"), dict) else payload
        candidates: list[object] = []

        for key in ("preferred_grind_maps", "preferred_maps"):
            rows = constraints.get(key) if isinstance(constraints, dict) else None
            if isinstance(rows, list):
                candidates.extend(rows)
            rows_root = payload.get(key)
            if isinstance(rows_root, list):
                candidates.extend(rows_root)

        assignment = ""
        if isinstance(payload, dict):
            assignment = str(payload.get("assignment") or "").strip()
        if not assignment and isinstance(constraints, dict):
            assignment = str(constraints.get("assignment") or "").strip()
        if assignment:
            candidates.append(assignment)

        preferred: list[str] = []
        for item in candidates:
            text = str(item or "").strip()
            if not text:
                continue
            for token in text.replace(";", ",").split(","):
                clean = token.strip()
                if clean and clean not in preferred:
                    preferred.append(clean)
        return preferred

    def _preferred_grind_target(self, *, bot_id: str) -> str:
        preferred = self._preferred_grind_maps(bot_id=bot_id)
        current_map = self._current_map_for_bot(bot_id=bot_id)
        for map_name in preferred:
            if map_name != current_map:
                return map_name
        return ""

    def _contains_any(self, text: str, needles: tuple[str, ...]) -> bool:
        lowered = text.lower()
        return any(token in lowered for token in needles)

    def _objective_to_action(self, objective: Any, *, bot_id: str) -> ActionProposal | None:
        """Convert a strategic objective to an action proposal."""
        try:
            description = (
                getattr(objective, "description", None)
                or getattr(objective, "name", None)
                or str(objective)
            )

            preferred_maps = self._preferred_grind_maps(bot_id=bot_id)
            preferred_target = self._preferred_grind_target(bot_id=bot_id)

            if self._contains_any(description, ("death", "dead", "savepoint", "respawn")):
                return self._build_action_proposal(
                    command="respawn",
                    priority=ActionPriorityTier.strategic,
                    source="pdca_loop_strategic",
                    conflict_key="recovery.death",
                    preconditions=["session.in_game"],
                    metadata={
                        "description": description,
                        "fallback_mode": "death_recovery",
                        "target": "savepoint",
                    },
                )

            resume_tokens = ("grind", "farm", "resume", "route", "travel", "move", "map")
            seek_tokens = ("seek", "search", "target", "hunt", "scan")
            rest_tokens = ("rest", "idle", "wait", "hold")

            if preferred_target and self._contains_any(description, resume_tokens):
                return self._build_action_proposal(
                    command=f"move {preferred_target}",
                    priority=ActionPriorityTier.strategic,
                    source="pdca_loop_strategic",
                    conflict_key="nav.resume_grind",
                    preconditions=["navigation.ready"],
                    metadata={
                        "description": description,
                        "fallback_mode": "resume_grind",
                        "target": preferred_target,
                        "preferred_grind_maps": preferred_maps,
                    },
                )

            if self._contains_any(description, seek_tokens):
                return self._build_action_proposal(
                    command="move random_walk_seek",
                    priority=ActionPriorityTier.strategic,
                    source="pdca_loop_strategic",
                    conflict_key="planner.seek.random_walk",
                    preconditions=["navigation.ready", "scan.targets_absent"],
                    metadata={
                        "description": description,
                        "fallback_mode": "seek_targets",
                        "seek_only_random_walk": True,
                        "target_scan_required": True,
                        "target_scan": {
                            "targets_found": False,
                            "source": "plan_executor.objective",
                        },
                    },
                )

            if preferred_target:
                return self._build_action_proposal(
                    command=f"move {preferred_target}",
                    priority=ActionPriorityTier.strategic,
                    source="pdca_loop_strategic",
                    conflict_key="nav.resume_grind",
                    preconditions=["navigation.ready"],
                    metadata={
                        "description": description,
                        "fallback_mode": "resume_grind",
                        "target": preferred_target,
                        "preferred_grind_maps": preferred_maps,
                    },
                )

            if self._contains_any(description, rest_tokens):
                return self._build_action_proposal(
                    command="sit",
                    priority=ActionPriorityTier.strategic,
                    source="pdca_loop_strategic",
                    conflict_key="planner.safe_idle",
                    preconditions=["vitals.safe_to_rest"],
                    metadata={
                        "description": description,
                        "fallback_mode": "safe_idle",
                    },
                )

            return self._build_action_proposal(
                command="sit",
                priority=ActionPriorityTier.strategic,
                source="pdca_loop_strategic",
                conflict_key="planner.safe_idle",
                preconditions=["vitals.safe_to_rest"],
                metadata={"description": description, "fallback_mode": "safe_idle"},
            )
        except Exception:
            logger.exception("Failed to convert objective to action")
            return None

    def _intent_to_action(self, intent: Any, *, bot_id: str) -> ActionProposal | None:
        """Convert a tactical intent to an action proposal."""
        try:
            objective = getattr(intent, "objective", None) or getattr(intent, "type", None) or "tactical_action"
            objective_text = str(objective)
            constraints = [str(item).strip().lower() for item in list(getattr(intent, "constraints", []) or [])]
            step_kind = ""
            for item in constraints:
                if item.startswith("step_kind="):
                    step_kind = item.split("=", 1)[1]
                    break

            preferred_maps = self._preferred_grind_maps(bot_id=bot_id)
            preferred_target = self._preferred_grind_target(bot_id=bot_id)

            seek_tokens = ("seek", "search", "target", "hunt", "scan")
            resume_tokens = ("grind", "farm", "resume", "route", "travel", "move", "map")
            rest_tokens = ("rest", "idle", "wait", "hold")

            if self._contains_any(objective_text, seek_tokens):
                return self._build_action_proposal(
                    command="move random_walk_seek",
                    priority=ActionPriorityTier.tactical,
                    source="pdca_loop_tactical",
                    conflict_key="planner.seek.random_walk",
                    preconditions=["navigation.ready", "scan.targets_absent"],
                    metadata={
                        "objective": objective_text,
                        "fallback_mode": "seek_targets",
                        "seek_only_random_walk": True,
                        "target_scan_required": True,
                        "target_scan": {
                            "targets_found": False,
                            "source": "plan_executor.intent",
                        },
                    },
                )

            if preferred_target and (
                self._contains_any(objective_text, resume_tokens)
                or step_kind in {"travel", "task", "combat"}
            ):
                return self._build_action_proposal(
                    command=f"move {preferred_target}",
                    priority=ActionPriorityTier.tactical,
                    source="pdca_loop_tactical",
                    conflict_key="nav.resume_grind",
                    preconditions=["navigation.ready"],
                    metadata={
                        "objective": objective_text,
                        "fallback_mode": "resume_grind",
                        "target": preferred_target,
                        "preferred_grind_maps": preferred_maps,
                    },
                )

            if self._contains_any(objective_text, rest_tokens):
                return self._build_action_proposal(
                    command="sit",
                    priority=ActionPriorityTier.tactical,
                    source="pdca_loop_tactical",
                    conflict_key="planner.safe_idle",
                    preconditions=["vitals.safe_to_rest"],
                    metadata={
                        "objective": objective_text,
                        "fallback_mode": "safe_idle",
                    },
                )

            if preferred_target:
                return self._build_action_proposal(
                    command=f"move {preferred_target}",
                    priority=ActionPriorityTier.tactical,
                    source="pdca_loop_tactical",
                    conflict_key="nav.resume_grind",
                    preconditions=["navigation.ready"],
                    metadata={
                        "objective": objective_text,
                        "fallback_mode": "resume_grind",
                        "target": preferred_target,
                        "preferred_grind_maps": preferred_maps,
                    },
                )

            return self._build_action_proposal(
                command="sit",
                priority=ActionPriorityTier.tactical,
                source="pdca_loop_tactical",
                conflict_key="planner.safe_idle",
                preconditions=["vitals.safe_to_rest"],
                metadata={"objective": objective_text, "fallback_mode": "safe_idle"},
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
        preconditions: list[str] | None = None,
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
            preconditions=list(preconditions or []),
            created_at=now,
            expires_at=now + timedelta(seconds=90),
            idempotency_key=f"{source}:{action_id}"[:128],
            metadata={"source": source, **dict(metadata or {})},
        )
