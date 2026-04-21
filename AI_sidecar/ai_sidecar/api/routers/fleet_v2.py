from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from ai_sidecar.api.deps import get_runtime
from ai_sidecar.contracts.fleet_v2 import (
    FleetBlackboardLocalResponse,
    FleetClaimRequestV2,
    FleetClaimResponseV2,
    FleetConstraintResponse,
    FleetOutcomeReportRequest,
    FleetOutcomeReportResponse,
    FleetRoleResponse,
    FleetSyncRequest,
    FleetSyncResponse,
)
from ai_sidecar.lifecycle import RuntimeState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v2/fleet", tags=["fleet-v2"])


@router.post("/sync", response_model=FleetSyncResponse)
def sync(payload: FleetSyncRequest, runtime: RuntimeState = Depends(get_runtime)) -> FleetSyncResponse:
    return runtime.fleet_sync(payload)


@router.get("/constraints", response_model=FleetConstraintResponse)
def constraints(bot_id: str, runtime: RuntimeState = Depends(get_runtime)) -> FleetConstraintResponse:
    return runtime.fleet_constraints(bot_id=bot_id)


@router.post("/report-outcome", response_model=FleetOutcomeReportResponse)
def report_outcome(payload: FleetOutcomeReportRequest, runtime: RuntimeState = Depends(get_runtime)) -> FleetOutcomeReportResponse:
    result = runtime.fleet_report_outcome(payload)
    logger.info(
        "fleet_outcome_reported",
        extra={
            "event": "fleet_outcome_reported",
            "bot_id": payload.meta.bot_id,
            "event_type": payload.event_type,
            "ok": result.ok,
            "central_available": result.central_available,
            "queued_for_retry": result.queued_for_retry,
        },
    )
    return result


@router.get("/role", response_model=FleetRoleResponse)
def role(bot_id: str, runtime: RuntimeState = Depends(get_runtime)) -> FleetRoleResponse:
    return runtime.fleet_role(bot_id=bot_id)


@router.post("/claim", response_model=FleetClaimResponseV2)
def claim(payload: FleetClaimRequestV2, runtime: RuntimeState = Depends(get_runtime)) -> FleetClaimResponseV2:
    return runtime.fleet_claim(payload)


@router.get("/blackboard", response_model=FleetBlackboardLocalResponse)
def blackboard(bot_id: str, runtime: RuntimeState = Depends(get_runtime)) -> FleetBlackboardLocalResponse:
    return runtime.fleet_blackboard(bot_id=bot_id)

