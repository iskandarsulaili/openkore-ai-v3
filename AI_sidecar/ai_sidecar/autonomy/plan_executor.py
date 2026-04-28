"""Plan executor — converts plans to action queue entries."""
from __future__ import annotations

from datetime import timedelta
import logging
from typing import Any, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from ai_sidecar.autonomy.pdca_loop import Horizon
from ai_sidecar.contracts.autonomy import GoalStackState
from ai_sidecar.contracts.control_domain import ControlApplyRequest, ControlPlanRequest
from ai_sidecar.contracts.common import utc_now
from ai_sidecar.contracts.macros import EventAutomacro, MacroPublishRequest, MacroRoutine
from ai_sidecar.planner.schemas import StrategicPlan, TacticalIntentBundle
from ai_sidecar.contracts.actions import ActionProposal, ActionPriorityTier

logger = logging.getLogger(__name__)


class PlanExecutor:
    """Converts plan outputs into queued actions."""

    def __init__(self, runtime_state: Any) -> None:
        self._runtime = runtime_state
        self._fallback_escalation_state: dict[str, int] = {}

    async def execute(
        self,
        plan: StrategicPlan | TacticalIntentBundle | None,
        horizon: "Horizon",
        max_actions: int = 5,
        goal_state: GoalStackState | None = None,
    ) -> int:
        """Execute a plan by converting its intents to action queue entries.

        Returns the number of actions successfully queued.
        """
        if plan is None and goal_state is None:
            return 0

        actions_queued = 0
        bot_id = str(
            getattr(plan, "bot_id", None)
            or getattr(goal_state, "bot_id", None)
            or "openkoreai"
        )

        try:
            if goal_state is not None:
                actions_queued = await self._execute_deterministic_hints(
                    goal_state=goal_state,
                    bot_id=bot_id,
                    max_actions=max_actions,
                )
                if actions_queued > 0:
                    return actions_queued

            if plan is None:
                return 0

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

    async def _execute_deterministic_hints(
        self,
        *,
        goal_state: GoalStackState,
        bot_id: str,
        max_actions: int,
    ) -> int:
        hints = self._execution_hints(goal_state=goal_state)
        if not hints:
            return 0

        queued = 0
        for hint in hints:
            if queued >= max_actions:
                break
            if not isinstance(hint, dict):
                continue
            mode = str(hint.get("execution_mode") or "").strip().lower()
            if mode == "direct":
                queued += await self._execute_direct_hint(
                    hint=hint,
                    bot_id=bot_id,
                    max_actions=max_actions - queued,
                )
            elif mode == "config":
                queued += self._execute_config_hint(hint=hint, bot_id=bot_id)
            elif mode == "macro":
                queued += self._execute_macro_hint(hint=hint, bot_id=bot_id)
            else:
                logger.debug("PlanExecutor deterministic hint ignored: unsupported mode=%s", mode or "unknown")

        return queued

    def _execution_hints(self, *, goal_state: GoalStackState) -> list[dict[str, object]]:
        selected_metadata = goal_state.selected_goal.metadata if isinstance(goal_state.selected_goal.metadata, dict) else {}
        selected_hints = selected_metadata.get("execution_hints")
        if isinstance(selected_hints, list):
            hints = [dict(item) for item in selected_hints if isinstance(item, dict)]
            if hints:
                return hints

        opportunistic = (
            goal_state.assessment.opportunistic_upgrades
            if isinstance(goal_state.assessment.opportunistic_upgrades, dict)
            else {}
        )
        fallback_hints = opportunistic.get("execution_hints")
        if isinstance(fallback_hints, list):
            return [dict(item) for item in fallback_hints if isinstance(item, dict)]
        return []

    async def _execute_direct_hint(self, *, hint: dict[str, object], bot_id: str, max_actions: int) -> int:
        intents = hint.get("intents") if isinstance(hint.get("intents"), list) else []
        rule_id = str(hint.get("rule_id") or "").strip()
        queued = 0

        for idx, intent in enumerate(intents):
            if queued >= max_actions:
                break
            if not isinstance(intent, dict):
                continue
            command = str(intent.get("command") or intent.get("action") or intent.get("intent") or "").strip()
            if not command:
                continue
            if command.lower() == "ai auto":
                logger.info(
                    "PlanExecutor deterministic direct hint skipped unsafe command=%s rule_id=%s",
                    command,
                    rule_id or "unknown",
                )
                continue

            priority_raw = str(intent.get("priority_tier") or intent.get("priority") or "tactical").strip().lower()
            try:
                priority_tier = ActionPriorityTier(priority_raw)
            except Exception:
                priority_tier = ActionPriorityTier.tactical

            now = utc_now()
            ttl_seconds = int(intent.get("expires_in_seconds") or 90)
            ttl_seconds = max(5, min(ttl_seconds, 1800))
            action_id = str(intent.get("action_id") or f"pdca-{uuid4().hex[:20]}")
            idempotency_key = str(
                intent.get("idempotency_key")
                or f"autonomy_hint:{bot_id}:{rule_id or 'rule_unknown'}:{idx}:{command}"[:128]
            )
            conflict_key = intent.get("conflict_key")
            metadata = dict(intent.get("metadata") or {})
            metadata.setdefault("source", "autonomy_stage4")
            if rule_id:
                metadata.setdefault("rule_id", rule_id)
            metadata.setdefault("execution_mode", "direct")

            raw_preconditions = intent.get("preconditions")
            if not isinstance(raw_preconditions, list):
                fallback = metadata.get("preconditions")
                raw_preconditions = fallback if isinstance(fallback, list) else []
            preconditions = [str(item).strip() for item in raw_preconditions if str(item).strip()]

            action = ActionProposal(
                action_id=action_id,
                kind=str(intent.get("kind") or "command")[:64],
                command=command[:256],
                priority_tier=priority_tier,
                conflict_key=None if conflict_key is None else str(conflict_key)[:128],
                preconditions=preconditions,
                created_at=now,
                expires_at=now + timedelta(seconds=ttl_seconds),
                idempotency_key=idempotency_key[:128],
                metadata=metadata,
            )
            await self._queue_action(action, bot_id)
            queued += 1

        return queued

    def _execute_config_hint(self, *, hint: dict[str, object], bot_id: str) -> int:
        request = hint.get("request") if isinstance(hint.get("request"), dict) else {}
        rule_id = str(hint.get("rule_id") or "").strip()
        try:
            plan_payload = ControlPlanRequest.model_validate(
                {
                    "meta": {
                        "contract_version": "v1",
                        "source": "autonomy_plan_executor",
                        "bot_id": bot_id,
                    },
                    "bot_id": str(request.get("bot_id") or bot_id),
                    "profile": request.get("profile"),
                    "artifact_type": request.get("artifact_type"),
                    "name": request.get("name"),
                    "target_path": request.get("target_path"),
                    "desired": dict(request.get("desired") or {}),
                    "source": str(request.get("source") or "autonomy"),
                }
            )
        except Exception:
            logger.exception("PlanExecutor deterministic config hint invalid")
            return 0

        try:
            plan_response = self._runtime.control_plan(plan_payload)
            plan_ok = bool(getattr(plan_response, "ok", True))
            plan = getattr(plan_response, "plan", None)
            plan_id = str(getattr(plan, "plan_id", "") or "").strip()
            if not plan_ok or not plan_id:
                logger.warning(
                    "PlanExecutor deterministic config plan failed ok=%s plan_id=%s",
                    plan_ok,
                    plan_id,
                )
                return 0

            apply_payload = ControlApplyRequest(
                meta=plan_payload.meta,
                plan_id=plan_id,
                dry_run=bool(request.get("dry_run", False)),
            )
            apply_response = self._runtime.control_apply(apply_payload)
            apply_ok = bool(getattr(apply_response, "ok", True))
            if not apply_ok:
                logger.warning("PlanExecutor deterministic config apply failed plan_id=%s", plan_id)
                return 0
            desired = request.get("desired") if isinstance(request.get("desired"), dict) else {}
            desired_keys = [str(key).strip() for key in sorted(desired.keys()) if str(key).strip()]
            logger.info(
                "autonomy_deterministic_config_hint_applied",
                extra={
                    "event": "autonomy_deterministic_config_hint_applied",
                    "bot_id": bot_id,
                    "rule_id": rule_id,
                    "plan_id": plan_id,
                    "target_path": str(request.get("target_path") or ""),
                    "artifact_type": str(request.get("artifact_type") or "config"),
                    "desired_keys": desired_keys,
                },
            )
            return 1
        except Exception:
            logger.exception("PlanExecutor deterministic config hint execution failed")
            return 0

    def _execute_macro_hint(self, *, hint: dict[str, object], bot_id: str) -> int:
        bundle = hint.get("macro_bundle") if isinstance(hint.get("macro_bundle"), dict) else {}
        rule_id = str(hint.get("rule_id") or "").strip()
        try:
            macros_raw = bundle.get("macros") if isinstance(bundle.get("macros"), list) else []
            event_raw = bundle.get("event_macros") if isinstance(bundle.get("event_macros"), list) else []
            automacros_raw = bundle.get("automacros") if isinstance(bundle.get("automacros"), list) else []

            macros = [
                item if isinstance(item, MacroRoutine) else MacroRoutine.model_validate(item)
                for item in macros_raw
            ]
            event_macros = [
                item if isinstance(item, MacroRoutine) else MacroRoutine.model_validate(item)
                for item in event_raw
            ]
            automacros = [
                item if isinstance(item, EventAutomacro) else EventAutomacro.model_validate(item)
                for item in automacros_raw
            ]

            request = MacroPublishRequest(
                meta={
                    "contract_version": "v1",
                    "source": "autonomy_plan_executor",
                    "bot_id": bot_id,
                },
                target_bot_id=str(bundle.get("target_bot_id") or bot_id),
                macros=macros,
                event_macros=event_macros,
                automacros=automacros,
                enqueue_reload=bool(bundle.get("enqueue_reload", True)),
                reload_conflict_key=str(bundle.get("reload_conflict_key") or "macro_reload"),
                macro_plugin=None if bundle.get("macro_plugin") is None else str(bundle.get("macro_plugin")),
                event_macro_plugin=None
                if bundle.get("event_macro_plugin") is None
                else str(bundle.get("event_macro_plugin")),
            )
        except Exception:
            logger.exception("PlanExecutor deterministic macro hint invalid")
            return 0

        try:
            response = self._runtime.publish_macros(request)
            if isinstance(response, tuple):
                ok = bool(response[0])
            else:
                ok = bool(getattr(response, "ok", False))

            logger.info(
                "autonomy_deterministic_macro_hint_publish_result",
                extra={
                    "event": "autonomy_deterministic_macro_hint_publish_result",
                    "bot_id": bot_id,
                    "rule_id": rule_id,
                    "ok": ok,
                    "macro_count": len(macros),
                    "event_macro_count": len(event_macros),
                    "automacro_count": len(automacros),
                    "enqueue_reload": bool(bundle.get("enqueue_reload", True)),
                    "reload_conflict_key": str(bundle.get("reload_conflict_key") or "macro_reload"),
                },
            )
            return 1 if ok else 0
        except Exception:
            logger.exception("PlanExecutor deterministic macro hint execution failed")
            return 0

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

    def _targets_present_for_bot(self, *, bot_id: str) -> bool:
        cache = getattr(self._runtime, "snapshot_cache", None)
        getter = getattr(cache, "get", None)
        if not callable(getter):
            return False
        try:
            snapshot = getter(bot_id)
        except Exception:
            logger.debug("PlanExecutor: snapshot cache lookup failed for targets", exc_info=True)
            return False

        combat = getattr(snapshot, "combat", None)
        target_id = str(getattr(combat, "target_id", "") or "").strip()
        if target_id:
            return True

        raw = getattr(snapshot, "raw", None)
        if isinstance(raw, dict):
            raw_target = str(raw.get("target_id") or raw.get("attack_target") or "").strip()
            if raw_target:
                return True
            try:
                return int(raw.get("nearby_hostiles") or 0) > 0
            except Exception:
                return False
        return False

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

    def _fallback_signature(self, *, bot_id: str, channel: str, subject_text: str) -> str:
        clean = " ".join(str(subject_text or "").strip().lower().split())
        return f"{bot_id}:{channel}:{clean[:120]}"

    def _next_fallback_stage(self, *, signature: str, escalation_active: bool) -> int:
        if not escalation_active:
            self._fallback_escalation_state.pop(signature, None)
            return 0
        previous = int(self._fallback_escalation_state.get(signature, -1))
        stage = (previous + 1) % 4
        self._fallback_escalation_state[signature] = stage
        return stage

    def _select_escalated_fallback_action(
        self,
        *,
        bot_id: str,
        channel: str,
        subject_text: str,
        priority: ActionPriorityTier,
        source: str,
        preferred_target: str,
        preferred_maps: list[str],
        current_map: str,
        targets_present: bool,
        escalation_reason: str,
        metadata_field: str,
    ) -> ActionProposal:
        signature = self._fallback_signature(bot_id=bot_id, channel=channel, subject_text=subject_text)
        stage = self._next_fallback_stage(signature=signature, escalation_active=bool(escalation_reason))

        if stage == 0:
            if preferred_target:
                command = f"move {preferred_target}"
                conflict_key = "nav.resume_grind"
                preconditions = ["navigation.ready"]
                fallback_mode = "resume_grind"
            else:
                command = "move random_walk_seek"
                conflict_key = "planner.seek.random_walk"
                preconditions = ["navigation.ready", "scan.targets_absent"]
                fallback_mode = "seek_targets"
        elif stage == 1:
            command = "ai clear"
            conflict_key = "planner.recovery.ai_clear"
            preconditions = ["session.in_game"]
            fallback_mode = "ai_queue_reset"
        elif stage == 2:
            if current_map:
                command = f"move {current_map}"
                conflict_key = "planner.recovery.map_refresh"
                preconditions = ["navigation.ready"]
                fallback_mode = "map_refresh"
            else:
                command = "move random_walk_seek"
                conflict_key = "planner.recovery.seek_refresh"
                preconditions = ["navigation.ready"]
                fallback_mode = "seek_refresh"
        else:
            command = "sit"
            conflict_key = "planner.safe_idle"
            preconditions = ["vitals.safe_to_rest"]
            fallback_mode = "safe_idle"

        logger.info(
            "autonomy_fallback_escalation_selected",
            extra={
                "event": "autonomy_fallback_escalation_selected",
                "bot_id": bot_id,
                "channel": channel,
                "stage": stage,
                "reason": escalation_reason,
                "selected_command": command,
                "selected_mode": fallback_mode,
                "preferred_target": preferred_target,
                "current_map": current_map,
                "targets_present": targets_present,
            },
        )
        return self._build_action_proposal(
            command=command,
            priority=priority,
            source=source,
            conflict_key=conflict_key,
            preconditions=preconditions,
            metadata={
                metadata_field: subject_text,
                "fallback_mode": fallback_mode,
                "escalation_stage": stage,
                "escalation_reason": escalation_reason,
                "preferred_target": preferred_target,
                "current_map": current_map,
                "targets_present": targets_present,
                "preferred_grind_maps": preferred_maps,
            },
        )

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
            current_map = self._current_map_for_bot(bot_id=bot_id)
            targets_present = self._targets_present_for_bot(bot_id=bot_id)

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
            stalled_tokens = (
                "stale",
                "stall",
                "stuck",
                "blocked",
                "no target",
                "targetless",
                "no enemy",
                "sparse",
            )

            stalled_state = self._contains_any(description, stalled_tokens)
            sparse_state = (not current_map) or (not preferred_target and not targets_present)
            escalation_reason = "|".join(
                reason
                for reason, enabled in (
                    ("stalled", stalled_state),
                    ("sparse_state", sparse_state),
                )
                if enabled
            )

            if preferred_target and self._contains_any(description, resume_tokens) and not escalation_reason:
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

            if preferred_target and not escalation_reason:
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
                if escalation_reason:
                    return self._select_escalated_fallback_action(
                        bot_id=bot_id,
                        channel="strategic",
                        subject_text=description,
                        priority=ActionPriorityTier.strategic,
                        source="pdca_loop_strategic",
                        preferred_target=preferred_target,
                        preferred_maps=preferred_maps,
                        current_map=current_map,
                        targets_present=targets_present,
                        escalation_reason=escalation_reason,
                        metadata_field="description",
                    )
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

            if escalation_reason:
                return self._select_escalated_fallback_action(
                    bot_id=bot_id,
                    channel="strategic",
                    subject_text=description,
                    priority=ActionPriorityTier.strategic,
                    source="pdca_loop_strategic",
                    preferred_target=preferred_target,
                    preferred_maps=preferred_maps,
                    current_map=current_map,
                    targets_present=targets_present,
                    escalation_reason=escalation_reason,
                    metadata_field="description",
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
            current_map = self._current_map_for_bot(bot_id=bot_id)
            targets_present = self._targets_present_for_bot(bot_id=bot_id)

            seek_tokens = ("seek", "search", "target", "hunt", "scan")
            resume_tokens = ("grind", "farm", "resume", "route", "travel", "move", "map")
            rest_tokens = ("rest", "idle", "wait", "hold")
            stalled_tokens = (
                "stale",
                "stall",
                "stuck",
                "blocked",
                "no target",
                "targetless",
                "no enemy",
                "sparse",
            )

            stalled_state = self._contains_any(objective_text, stalled_tokens)
            sparse_state = (not current_map) or (not preferred_target and not targets_present)
            escalation_reason = "|".join(
                reason
                for reason, enabled in (
                    ("stalled", stalled_state),
                    ("sparse_state", sparse_state),
                )
                if enabled
            )

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
            ) and not escalation_reason:
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
                if escalation_reason:
                    return self._select_escalated_fallback_action(
                        bot_id=bot_id,
                        channel="tactical",
                        subject_text=objective_text,
                        priority=ActionPriorityTier.tactical,
                        source="pdca_loop_tactical",
                        preferred_target=preferred_target,
                        preferred_maps=preferred_maps,
                        current_map=current_map,
                        targets_present=targets_present,
                        escalation_reason=escalation_reason,
                        metadata_field="objective",
                    )
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

            if preferred_target and not escalation_reason:
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

            if escalation_reason:
                return self._select_escalated_fallback_action(
                    bot_id=bot_id,
                    channel="tactical",
                    subject_text=objective_text,
                    priority=ActionPriorityTier.tactical,
                    source="pdca_loop_tactical",
                    preferred_target=preferred_target,
                    preferred_maps=preferred_maps,
                    current_map=current_map,
                    targets_present=targets_present,
                    escalation_reason=escalation_reason,
                    metadata_field="objective",
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
