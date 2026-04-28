from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import logging
from typing import Any
from uuid import uuid4

from ai_sidecar.contracts.actions import ActionPriorityTier, ActionProposal
from ai_sidecar.contracts.control_domain import ControlChangePlan, ControlChangeState, ControlOwnerScope
from ai_sidecar.domain.control_policy import ControlPolicy
from ai_sidecar.domain.control_storage import ControlStorage


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ControlExecutionResult:
    state: ControlChangeState
    reload_action_id: str | None
    reload_status: str | None
    reload_reason: str
    operation_id: str | None = None


@dataclass(slots=True)
class ControlExecutor:
    runtime: Any
    storage: ControlStorage

    def apply(
        self,
        *,
        plan: ControlChangePlan,
        owner: ControlOwnerScope,
        policy: ControlPolicy,
        dry_run: bool,
        operation_id: str | None = None,
    ) -> ControlExecutionResult:
        path = plan.annotations.get("path")
        if not path:
            return ControlExecutionResult(ControlChangeState.failed, None, None, "missing_path", operation_id)
        target_path = self.storage.workspace_root / str(path)
        current_text = self.storage.read_text(target_path)
        current_checksum = self.storage.checksum(current_text)
        desired_checksum = str(plan.checksum_after)

        if operation_id is None:
            operation_id = self._begin_operation(
                plan=plan,
                dry_run=dry_run,
                base_checksum=current_checksum,
                desired_checksum=desired_checksum,
            )
        parsed = self.storage.parser.parse(current_text)
        next_values = dict(parsed.values)
        denied = False

        self._transition_operation(
            operation_id=operation_id,
            status="applying",
            status_reason="control_apply_started",
            observed_checksum=current_checksum,
            increment_attempt=True,
        )

        if current_checksum == desired_checksum:
            self._transition_operation(
                operation_id=operation_id,
                status="completed",
                status_reason="control_artifact_already_current",
                observed_checksum=current_checksum,
                mark_reconciled=True,
            )
            logger.info(
                "control_apply_duplicate_suppressed",
                extra={
                    "event": "control_apply_duplicate_suppressed",
                    "bot_id": plan.bot_id,
                    "plan_id": plan.plan_id,
                    "path": str(path),
                    "checksum": current_checksum,
                    "operation_id": operation_id,
                },
            )
            return ControlExecutionResult(
                ControlChangeState.applied,
                None,
                None,
                "already_current",
                operation_id,
            )

        for change in plan.changes:
            ok, _reason = policy.allow_write(change.key, owner)
            if not ok:
                denied = True
                continue
            if change.previous != change.updated:
                next_values[change.key] = change.updated

        if denied:
            if dry_run:
                self._transition_operation(
                    operation_id=operation_id,
                    status="completed",
                    status_reason="control_apply_dry_run_policy_denied",
                    observed_checksum=current_checksum,
                    mark_reconciled=True,
                )
                return ControlExecutionResult(ControlChangeState.applied, None, None, "dry_run_policy_denied", operation_id)
            self._transition_operation(
                operation_id=operation_id,
                status="retry_pending",
                status_reason="control_apply_policy_denied",
                observed_checksum=current_checksum,
                error_message="policy_denied",
            )
            return ControlExecutionResult(ControlChangeState.failed, None, None, "policy_denied", operation_id)
        if dry_run:
            self._transition_operation(
                operation_id=operation_id,
                status="completed",
                status_reason="control_apply_dry_run",
                observed_checksum=current_checksum,
                mark_reconciled=True,
            )
            return ControlExecutionResult(ControlChangeState.applied, None, None, "dry_run", operation_id)

        self.storage.write_config(path=target_path, values=next_values)
        observed_checksum = self.storage.checksum(self.storage.read_text(target_path))
        if observed_checksum != desired_checksum:
            self._transition_operation(
                operation_id=operation_id,
                status="retry_pending",
                status_reason="control_apply_drift_detected",
                observed_checksum=observed_checksum,
                error_message="post_write_checksum_mismatch",
            )
            logger.warning(
                "control_apply_drift_detected",
                extra={
                    "event": "control_apply_drift_detected",
                    "bot_id": plan.bot_id,
                    "plan_id": plan.plan_id,
                    "path": str(path),
                    "expected_checksum": desired_checksum,
                    "observed_checksum": observed_checksum,
                    "operation_id": operation_id,
                },
            )
            return ControlExecutionResult(ControlChangeState.failed, None, None, "post_write_checksum_mismatch", operation_id)

        self._transition_operation(
            operation_id=operation_id,
            status="applied",
            status_reason="control_apply_persisted",
            observed_checksum=observed_checksum,
        )

        reload_action_id, reload_status, reload_reason = self._queue_reload(
            plan=plan,
            operation_id=operation_id,
            reload_idempotency_key=f"config-reload:{plan.bot_id}:{plan.artifact.path}:{desired_checksum}",
        )
        return ControlExecutionResult(ControlChangeState.applied, reload_action_id, reload_status, reload_reason, operation_id)

    def _queue_reload(
        self,
        *,
        plan: ControlChangePlan,
        operation_id: str | None,
        reload_idempotency_key: str,
    ) -> tuple[str | None, str | None, str]:
        now = datetime.now(UTC)
        action_id = f"config-reload-{uuid4().hex[:18]}"
        metadata: dict[str, object] = {
            "target": plan.artifact.name,
            "plan_id": plan.plan_id,
        }
        if operation_id:
            metadata["sidecar_operation_id"] = operation_id
            metadata["operation_kind"] = "control_apply"

        proposal = ActionProposal(
            action_id=action_id,
            kind="config_reload",
            command="",
            priority_tier=ActionPriorityTier.macro_management,
            conflict_key="config_reload",
            created_at=now,
            expires_at=now + timedelta(seconds=60),
            idempotency_key=reload_idempotency_key,
            metadata=metadata,
        )
        accepted, status, action_id, reason = self.runtime.queue_action(proposal=proposal, bot_id=plan.bot_id)
        if accepted:
            self._transition_operation(
                operation_id=operation_id,
                status="reload_pending",
                status_reason="control_reload_queued",
                linked_action_id=action_id,
            )
        elif reason == "idempotent_duplicate":
            self._transition_operation(
                operation_id=operation_id,
                status="reload_pending",
                status_reason="control_reload_duplicate_suppressed",
                linked_action_id=action_id,
            )
        else:
            self._transition_operation(
                operation_id=operation_id,
                status="retry_pending",
                status_reason="control_reload_queue_failed",
                linked_action_id=action_id,
                error_message=str(reason),
            )
        if not accepted:
            return action_id, str(status), reason
        return action_id, str(status), "queued"

    def _begin_operation(
        self,
        *,
        plan: ControlChangePlan,
        dry_run: bool,
        base_checksum: str,
        desired_checksum: str,
    ) -> str | None:
        runtime = self.runtime
        begin_fn = getattr(runtime, "begin_sidecar_operation", None)
        if begin_fn is None:
            return None
        payload: dict[str, object] = {
            "plan_id": plan.plan_id,
            "dry_run": bool(dry_run),
            "policy_version": plan.policy_version,
            "change_count": len(plan.changes),
            "artifact_name": plan.artifact.name,
        }
        try:
            return begin_fn(
                bot_id=plan.bot_id,
                operation_kind="control_apply",
                artifact_kind=str(plan.artifact.artifact_type),
                artifact_path=str(plan.artifact.path),
                idempotency_key=f"control-apply:{plan.bot_id}:{plan.artifact.path}:{desired_checksum}",
                payload=payload,
                base_checksum=base_checksum,
                desired_checksum=desired_checksum,
                status="planned",
                status_reason="control_operation_planned",
            )
        except Exception:
            logger.exception(
                "control_operation_begin_failed",
                extra={
                    "event": "control_operation_begin_failed",
                    "bot_id": plan.bot_id,
                    "plan_id": plan.plan_id,
                    "artifact_path": str(plan.artifact.path),
                },
            )
            return None

    def _transition_operation(
        self,
        *,
        operation_id: str | None,
        status: str,
        status_reason: str,
        observed_checksum: str | None = None,
        linked_action_id: str | None = None,
        error_message: str | None = None,
        increment_attempt: bool = False,
        mark_reconciled: bool = False,
    ) -> None:
        if not operation_id:
            return
        runtime = self.runtime
        transition_fn = getattr(runtime, "transition_sidecar_operation", None)
        if transition_fn is None:
            return
        try:
            transition_fn(
                operation_id=operation_id,
                status=status,
                status_reason=status_reason,
                observed_checksum=observed_checksum,
                linked_action_id=linked_action_id,
                error_message=error_message,
                increment_attempt=increment_attempt,
                mark_reconciled=mark_reconciled,
            )
        except Exception:
            logger.exception(
                "control_operation_transition_failed",
                extra={
                    "event": "control_operation_transition_failed",
                    "operation_id": operation_id,
                    "status": status,
                    "status_reason": status_reason,
                },
            )
