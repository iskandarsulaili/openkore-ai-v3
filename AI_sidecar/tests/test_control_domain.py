from __future__ import annotations

from pathlib import Path

from ai_sidecar.contracts.common import ContractMeta
from ai_sidecar.contracts.control_domain import (
    ControlApplyRequest,
    ControlPlanRequest,
    ControlValidateRequest,
    ControlArtifactsResponse,
)
from ai_sidecar.domain.control_executor import ControlExecutor
from ai_sidecar.domain.control_parser import ControlParser
from ai_sidecar.domain.control_planner import ControlPlanner
from ai_sidecar.domain.control_policy import default_control_policy
from ai_sidecar.domain.control_registry import ControlRegistry
from ai_sidecar.domain.control_service import ControlDomainService
from ai_sidecar.domain.control_state import ControlStateStore
from ai_sidecar.domain.control_storage import ControlStorage
from ai_sidecar.domain.control_validator import ControlValidator


class _Runtime:
    def __init__(self) -> None:
        self.actions: list[object] = []

    def queue_action(self, proposal, *, bot_id: str):  # type: ignore[no-untyped-def]
        self.actions.append((proposal, bot_id))
        return True, proposal.priority_tier, proposal.action_id, "queued"


class _RuntimeState:
    def __init__(self, service: ControlDomainService) -> None:
        self._service = service

    def control_plan(self, payload: ControlPlanRequest):  # type: ignore[no-untyped-def]
        return self._service.plan(payload)

    def control_apply(self, payload: ControlApplyRequest):  # type: ignore[no-untyped-def]
        return self._service.apply(payload)

    def control_validate(self, payload: ControlValidateRequest):  # type: ignore[no-untyped-def]
        return self._service.validate(payload)

    def control_artifacts(self, *, bot_id: str) -> ControlArtifactsResponse:  # type: ignore[no-untyped-def]
        return self._service.artifacts(bot_id=bot_id)


def _service(tmp_path: Path) -> ControlDomainService:
    parser = ControlParser()
    storage = ControlStorage(workspace_root=tmp_path, parser=parser)
    runtime = _Runtime()
    executor = ControlExecutor(runtime=runtime, storage=storage)
    return ControlDomainService(
        storage=storage,
        policy=default_control_policy(),
        registry=ControlRegistry(),
        planner=ControlPlanner(storage=storage),
        executor=executor,
        validator=ControlValidator(storage=storage),
        state=ControlStateStore(),
    )


def test_control_domain_plan_and_apply_blocks_protected_keys(tmp_path: Path) -> None:
    service = _service(tmp_path)
    meta = ContractMeta(contract_version="v1", source="pytest", bot_id="bot:cd", trace_id="trace-1")
    plan_req = ControlPlanRequest(
        meta=meta,
        bot_id="bot:cd",
        profile=None,
        artifact_type="config",
        name="config.txt",
        target_path="control/config.txt",
        desired={"username": "bad", "aiSidecar_enable": "1"},
        source="crewai",
    )
    plan_resp = service.plan(plan_req)
    assert plan_resp.ok is True
    assert any(change.key == "username" and change.reason == "protected_key" for change in plan_resp.plan.changes)

    apply_resp = service.apply(ControlApplyRequest(meta=meta, plan_id=plan_resp.plan.plan_id, dry_run=True))
    assert apply_resp.ok is True
    assert apply_resp.state.value == "applied"

    validate_resp = service.validate(ControlValidateRequest(meta=meta, plan_id=plan_resp.plan.plan_id))
    assert validate_resp.ok is False
    assert validate_resp.state.value == "failed"


def test_control_domain_router_plan(tmp_path: Path) -> None:
    from ai_sidecar.api.routers.control_domain import plan as plan_route

    service = _service(tmp_path)
    runtime = _RuntimeState(service)
    meta = ContractMeta(contract_version="v1", source="pytest", bot_id="bot:cd", trace_id="trace-2")
    payload = ControlPlanRequest(
        meta=meta,
        bot_id="bot:cd",
        profile=None,
        artifact_type="config",
        name="config.txt",
        target_path="control/config.txt",
        desired={"aiSidecar_enable": "1"},
        source="manual",
    )
    response = plan_route(payload, runtime)  # type: ignore[arg-type]
    assert response.ok is True
    assert response.plan.bot_id == "bot:cd"
