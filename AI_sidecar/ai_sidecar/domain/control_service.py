from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import logging

from ai_sidecar.contracts.control_domain import (
    ControlApplyRequest,
    ControlApplyResponse,
    ControlArtifactsResponse,
    ControlChangeState,
    ControlPlanRequest,
    ControlPlanResponse,
    ControlRollbackRequest,
    ControlRollbackResponse,
    ControlValidateRequest,
    ControlValidateResponse,
    ControlOwnerScope,
)
from ai_sidecar.domain.control_executor import ControlExecutor
from ai_sidecar.domain.control_planner import ControlPlanner
from ai_sidecar.domain.control_policy import ControlPolicy
from ai_sidecar.domain.control_registry import ControlRegistry
from ai_sidecar.domain.control_state import ControlPlanState, ControlStateStore
from ai_sidecar.domain.control_storage import ControlStorage
from ai_sidecar.domain.control_validator import ControlValidator


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ControlDomainService:
    storage: ControlStorage
    policy: ControlPolicy
    registry: ControlRegistry
    planner: ControlPlanner
    executor: ControlExecutor
    validator: ControlValidator
    state: ControlStateStore

    def plan(self, payload: ControlPlanRequest) -> ControlPlanResponse:
        owner = ControlOwnerScope.sidecar if payload.source != "operator" else ControlOwnerScope.operator
        plan = self.planner.plan_config(
            bot_id=payload.bot_id,
            profile=payload.profile,
            target_path=payload.target_path,
            desired=payload.desired,
            policy=self.policy,
            owner=owner,
            policy_version=self.policy.policy.version,
        )
        expected: dict[str, str] = {}
        for change in plan.changes:
            if change.previous != change.updated and self.policy.allow_write(change.key, owner)[0]:
                expected[change.key] = change.updated
        self.state.upsert(
            ControlPlanState(plan=plan, state=ControlChangeState.planned, expected=expected, lineage=[plan.plan_id])
        )
        logger.info(
            "control_plan_created",
            extra={
                "event": "control_plan_created",
                "bot_id": payload.bot_id,
                "plan_id": plan.plan_id,
                "changes": len(plan.changes),
            },
        )
        return ControlPlanResponse(ok=True, message="planned", plan=plan)

    def apply(self, payload: ControlApplyRequest) -> ControlApplyResponse:
        stored = self.state.get(payload.plan_id)
        if stored is None:
            raise ValueError("plan_not_found")
        owner = ControlOwnerScope.sidecar
        result = self.executor.apply(
            plan=stored.plan,
            owner=owner,
            policy=self.policy,
            dry_run=payload.dry_run,
        )
        stored.state = result.state
        self.state.upsert(stored)
        return ControlApplyResponse(
            ok=result.state != ControlChangeState.failed,
            message=str(result.state),
            state=result.state,
            plan=stored.plan,
            reload_action_id=result.reload_action_id,
            reload_status=result.reload_status,
            reload_reason=result.reload_reason,
        )

    def validate(self, payload: ControlValidateRequest) -> ControlValidateResponse:
        stored = self.state.get(payload.plan_id)
        if stored is None:
            raise ValueError("plan_not_found")
        path = stored.plan.annotations.get("path")
        drift = self.validator.drift(path=str(path), expected=stored.expected)
        new_state = ControlChangeState.validated if not drift else ControlChangeState.failed
        stored.state = new_state
        self.state.upsert(stored)
        return ControlValidateResponse(
            ok=not drift,
            message="validated" if not drift else "drift_detected",
            state=new_state,
            plan=stored.plan,
            drift=drift,
        )

    def rollback(self, payload: ControlRollbackRequest) -> ControlRollbackResponse:
        stored = self.state.get(payload.plan_id)
        if stored is None:
            raise ValueError("plan_not_found")
        stored.state = ControlChangeState.rolled_back
        stored.lineage.append(f"rollback:{datetime.now(UTC).isoformat()}")
        self.state.upsert(stored)
        return ControlRollbackResponse(
            ok=True,
            message="rolled_back",
            state=stored.state,
            plan=stored.plan,
            rollback_lineage=list(stored.lineage),
        )

    def artifacts(self, *, bot_id: str) -> ControlArtifactsResponse:
        artifacts = self.registry.list_for_bot(bot_id=bot_id)
        return ControlArtifactsResponse(ok=True, artifacts=artifacts)

