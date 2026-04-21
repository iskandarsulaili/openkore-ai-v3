from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from ai_sidecar.api.deps import get_runtime
from ai_sidecar.lifecycle import RuntimeState
from ai_sidecar.planner.schemas import (
    PlannerExplainRequest,
    PlannerMacroPromoteRequest,
    PlannerPlanRequest,
    PlannerResponse,
    PlannerStatusResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v2/planner", tags=["planner-v2"])


@router.post("/plan", response_model=PlannerResponse)
async def plan(
    payload: PlannerPlanRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> PlannerResponse:
    result = await runtime.planner_plan(payload)
    logger.info(
        "planner_plan_generated",
        extra={
            "event": "planner_plan_generated",
            "bot_id": payload.meta.bot_id,
            "ok": result.ok,
            "provider": result.provider,
            "model": result.model,
        },
    )
    return result


@router.post("/replan", response_model=PlannerResponse)
async def replan(
    payload: PlannerPlanRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> PlannerResponse:
    result = await runtime.planner_replan(payload)
    logger.info(
        "planner_replan_generated",
        extra={
            "event": "planner_replan_generated",
            "bot_id": payload.meta.bot_id,
            "ok": result.ok,
            "provider": result.provider,
            "model": result.model,
        },
    )
    return result


@router.post("/explain")
def explain(
    payload: PlannerExplainRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, object]:
    return runtime.planner_explain(payload)


@router.post("/promote-macro", response_model=PlannerResponse)
async def promote_macro(
    payload: PlannerMacroPromoteRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> PlannerResponse:
    result = await runtime.planner_promote_macro(payload)
    logger.info(
        "planner_macro_promotion",
        extra={
            "event": "planner_macro_promotion",
            "bot_id": payload.meta.bot_id,
            "ok": result.ok,
            "has_macro": result.macro_proposal is not None,
        },
    )
    return result


@router.get("/status/{bot_id}", response_model=PlannerStatusResponse)
def status(
    bot_id: str,
    runtime: RuntimeState = Depends(get_runtime),
) -> PlannerStatusResponse:
    return runtime.planner_status(bot_id=bot_id)

