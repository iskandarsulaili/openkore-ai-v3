from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from ai_sidecar.api.deps import get_runtime
from ai_sidecar.contracts.control_domain import (
    ControlApplyRequest,
    ControlApplyResponse,
    ControlArtifactsResponse,
    ControlPlanRequest,
    ControlPlanResponse,
    ControlRollbackRequest,
    ControlRollbackResponse,
    ControlValidateRequest,
    ControlValidateResponse,
)
from ai_sidecar.lifecycle import RuntimeState


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v2/control", tags=["control-domain"])


@router.post("/plan", response_model=ControlPlanResponse)
def plan(
    payload: ControlPlanRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> ControlPlanResponse:
    try:
        result = runtime.control_plan(payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    logger.info(
        "control_plan_requested",
        extra={"event": "control_plan_requested", "bot_id": payload.bot_id, "plan_id": result.plan.plan_id},
    )
    return result


@router.post("/apply", response_model=ControlApplyResponse)
def apply(
    payload: ControlApplyRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> ControlApplyResponse:
    try:
        result = runtime.control_apply(payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    logger.info(
        "control_apply_requested",
        extra={"event": "control_apply_requested", "plan_id": payload.plan_id, "state": result.state.value},
    )
    return result


@router.post("/validate", response_model=ControlValidateResponse)
def validate(
    payload: ControlValidateRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> ControlValidateResponse:
    try:
        result = runtime.control_validate(payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    logger.info(
        "control_validate_requested",
        extra={"event": "control_validate_requested", "plan_id": payload.plan_id, "ok": result.ok},
    )
    return result


@router.post("/rollback", response_model=ControlRollbackResponse)
def rollback(
    payload: ControlRollbackRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> ControlRollbackResponse:
    try:
        result = runtime.control_rollback(payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    logger.warning(
        "control_rollback_requested",
        extra={"event": "control_rollback_requested", "plan_id": payload.plan_id},
    )
    return result


@router.get("/artifacts/{bot_id}", response_model=ControlArtifactsResponse)
def artifacts(
    bot_id: str,
    runtime: RuntimeState = Depends(get_runtime),
) -> ControlArtifactsResponse:
    return runtime.control_artifacts(bot_id=bot_id)

