from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from ai_sidecar.api.deps import get_runtime
from ai_sidecar.contracts.crewai import (
    CrewAgentsResponse,
    CrewCoordinateRequest,
    CrewCoordinateResponse,
    CrewStatusResponse,
    CrewStrategizeRequest,
    CrewStrategizeResponse,
    CrewToolExecuteRequest,
    CrewToolExecuteResponse,
)
from ai_sidecar.lifecycle import RuntimeState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v2/crewai", tags=["crewai-v2"])


@router.post("/strategize", response_model=CrewStrategizeResponse)
async def strategize(
    payload: CrewStrategizeRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> CrewStrategizeResponse:
    result = await runtime.crewai_strategize(payload)
    logger.info(
        "crewai_strategize_completed",
        extra={
            "event": "crewai_strategize_completed",
            "bot_id": payload.meta.bot_id,
            "trace_id": payload.meta.trace_id,
            "ok": result.ok,
            "error_count": len(result.errors),
        },
    )
    return result


@router.post("/coordinate", response_model=CrewCoordinateResponse)
async def coordinate(
    payload: CrewCoordinateRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> CrewCoordinateResponse:
    result = await runtime.crewai_coordinate(payload)
    logger.info(
        "crewai_coordinate_completed",
        extra={
            "event": "crewai_coordinate_completed",
            "bot_id": payload.meta.bot_id,
            "trace_id": payload.meta.trace_id,
            "ok": result.ok,
            "task": payload.task,
            "error_count": len(result.errors),
        },
    )
    return result


@router.get("/agents", response_model=CrewAgentsResponse)
def agents(runtime: RuntimeState = Depends(get_runtime)) -> CrewAgentsResponse:
    return runtime.crewai_agents()


@router.get("/status", response_model=CrewStatusResponse)
def status(runtime: RuntimeState = Depends(get_runtime)) -> CrewStatusResponse:
    return runtime.crewai_status()


@router.post("/tools/execute", response_model=CrewToolExecuteResponse)
def execute_tool(
    payload: CrewToolExecuteRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> CrewToolExecuteResponse:
    return runtime.crewai_execute_tool(payload)

