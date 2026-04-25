from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from ai_sidecar.contracts.actions import ActionPriorityTier, ActionProposal
from ai_sidecar.contracts.control_domain import ControlChangePlan, ControlChangeState, ControlOwnerScope
from ai_sidecar.domain.control_policy import ControlPolicy
from ai_sidecar.domain.control_storage import ControlStorage


@dataclass(slots=True)
class ControlExecutionResult:
    state: ControlChangeState
    reload_action_id: str | None
    reload_status: str | None
    reload_reason: str


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
    ) -> ControlExecutionResult:
        path = plan.annotations.get("path")
        if not path:
            return ControlExecutionResult(ControlChangeState.failed, None, None, "missing_path")
        target_path = self.storage.workspace_root / str(path)
        current_text = self.storage.read_text(target_path)
        parsed = self.storage.parser.parse(current_text)
        next_values = dict(parsed.values)
        denied = False
        for change in plan.changes:
            ok, _reason = policy.allow_write(change.key, owner)
            if not ok:
                denied = True
                continue
            if change.previous != change.updated:
                next_values[change.key] = change.updated

        if denied:
            if dry_run:
                return ControlExecutionResult(ControlChangeState.applied, None, None, "dry_run_policy_denied")
            return ControlExecutionResult(ControlChangeState.failed, None, None, "policy_denied")
        if dry_run:
            return ControlExecutionResult(ControlChangeState.applied, None, None, "dry_run")

        self.storage.write_config(path=target_path, values=next_values)
        reload_action_id, reload_status, reload_reason = self._queue_reload(plan=plan)
        return ControlExecutionResult(ControlChangeState.applied, reload_action_id, reload_status, reload_reason)

    def _queue_reload(self, *, plan: ControlChangePlan) -> tuple[str | None, str | None, str]:
        now = datetime.now(UTC)
        action_id = f"config-reload-{uuid4().hex[:18]}"
        proposal = ActionProposal(
            action_id=action_id,
            kind="config_reload",
            command="",
            priority_tier=ActionPriorityTier.macro_management,
            conflict_key="config_reload",
            created_at=now,
            expires_at=now + timedelta(seconds=60),
            idempotency_key=f"config-reload:{plan.plan_id}",
            metadata={
                "target": plan.artifact.name,
                "plan_id": plan.plan_id,
            },
        )
        accepted, status, action_id, reason = self.runtime.queue_action(proposal=proposal, bot_id=plan.bot_id)
        if not accepted:
            return action_id, str(status), reason
        return action_id, str(status), "queued"
