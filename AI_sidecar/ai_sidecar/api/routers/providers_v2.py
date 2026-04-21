from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Query

from ai_sidecar.api.deps import get_runtime
from ai_sidecar.lifecycle import RuntimeState
from ai_sidecar.planner.schemas import ProviderPolicyUpdateRequest, ProviderRouteRequest, ProviderRouteResponse

router = APIRouter(prefix="/v2/providers", tags=["providers-v2"])


@router.get("/health")
async def health(
    bot_id: str = Query(default="fleet", min_length=1, max_length=128),
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, object]:
    rows = await runtime.providers_health(bot_id=bot_id)
    return {
        "ok": True,
        "bot_id": bot_id,
        "count": len(rows),
        "providers": rows,
        "generated_at": datetime.now(UTC),
    }


@router.post("/route", response_model=ProviderRouteResponse)
def route(
    payload: ProviderRouteRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> ProviderRouteResponse:
    return runtime.provider_route(payload)


@router.get("/policy")
def policy(runtime: RuntimeState = Depends(get_runtime)) -> dict[str, object]:
    return runtime.provider_policy()


@router.put("/policy")
def update_policy(
    payload: ProviderPolicyUpdateRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, object]:
    return runtime.update_provider_policy(payload)

