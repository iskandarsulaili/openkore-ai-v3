from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from ai_sidecar.api.deps import get_runtime
from ai_sidecar.contracts.actions import ActionStatus
from ai_sidecar.contracts.fleet import (
    AssignmentUpdateResponse,
    BotAssignmentUpdateRequest,
    BotListResponse,
    BotStatusResponse,
    BotStatusView,
    FleetStatusResponse,
)
from ai_sidecar.lifecycle import RuntimeState

router = APIRouter(prefix="/v1/fleet", tags=["fleet"])


def _to_bot_status_view(item: dict[str, object]) -> BotStatusView:
    return BotStatusView(
        bot_id=str(item["bot_id"]),
        bot_name=item.get("bot_name"),
        role=item.get("role"),
        assignment=item.get("assignment"),
        capabilities=list(item.get("capabilities") or []),
        attributes=dict(item.get("attributes") or {}),
        first_seen_at=item.get("first_seen_at"),
        last_seen_at=item.get("last_seen_at"),
        last_tick_id=item.get("last_tick_id"),
        liveness_state=str(item.get("liveness_state") or "unknown"),
        pending_actions=int(item.get("pending_actions") or 0),
        latest_snapshot_at=item.get("latest_snapshot_at"),
        telemetry_events=int(item.get("telemetry_events") or 0),
    )


@router.get("/bots", response_model=BotListResponse)
def list_bots(runtime: RuntimeState = Depends(get_runtime)) -> BotListResponse:
    rows = runtime.list_bots()
    return BotListResponse(ok=True, total_bots=len(rows), bots=[_to_bot_status_view(item) for item in rows])


@router.get("/bots/{bot_id}", response_model=BotStatusResponse)
def bot_status(
    bot_id: str,
    action_limit: int = Query(default=25, ge=1, le=200),
    snapshot_limit: int = Query(default=10, ge=1, le=100),
    audit_limit: int = Query(default=25, ge=1, le=200),
    runtime: RuntimeState = Depends(get_runtime),
) -> BotStatusResponse:
    bot = runtime.bot_status(bot_id)
    if bot is None:
        raise HTTPException(status_code=404, detail=f"bot not found: {bot_id}")

    return BotStatusResponse(
        ok=True,
        bot=_to_bot_status_view(bot),
        recent_actions=runtime.recent_actions(bot_id=bot_id, limit=action_limit),
        recent_snapshots=runtime.recent_snapshots(bot_id=bot_id, limit=snapshot_limit),
        latest_macro_publication=runtime.latest_macro_publication(bot_id=bot_id),
        recent_audit=runtime.recent_audit(limit=audit_limit, bot_id=bot_id),
    )


@router.put("/bots/{bot_id}/assignment", response_model=AssignmentUpdateResponse)
def update_bot_assignment(
    bot_id: str,
    payload: BotAssignmentUpdateRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> AssignmentUpdateResponse:
    updated = runtime.update_assignment(
        bot_id=bot_id,
        role=payload.role,
        assignment=payload.assignment,
        attributes=payload.attributes,
    )
    if updated is None:
        return AssignmentUpdateResponse(ok=True, updated=False, bot=None, message=f"bot not found: {bot_id}")
    return AssignmentUpdateResponse(ok=True, updated=True, bot=_to_bot_status_view(updated), message="assignment updated")


@router.get("/status", response_model=FleetStatusResponse)
def fleet_status(runtime: RuntimeState = Depends(get_runtime)) -> FleetStatusResponse:
    data = runtime.fleet_status()
    action_status_totals = {
        status.value: int(data.get("action_status_totals", {}).get(status.value, 0))
        for status in ActionStatus
    }
    return FleetStatusResponse(
        ok=True,
        generated_at=data["generated_at"],
        total_bots=int(data.get("total_bots", 0)),
        online_bots=int(data.get("online_bots", 0)),
        total_pending_actions=int(data.get("total_pending_actions", 0)),
        action_status_totals=action_status_totals,
        telemetry_window=dict(data.get("telemetry_window", {})),
        counters=dict(data.get("counters", {})),
    )


@router.get("/audit")
def recent_audit(
    bot_id: str | None = Query(default=None),
    event_type: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, object]:
    rows = runtime.recent_audit(limit=limit, bot_id=bot_id, event_type=event_type)
    return {"ok": True, "count": len(rows), "events": rows}


@router.get("/memory/{bot_id}")
def memory_context(
    bot_id: str,
    query: str = Query(min_length=1, max_length=512),
    limit: int = Query(default=5, ge=1, le=50),
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, object]:
    return {
        "ok": True,
        "bot_id": bot_id,
        "query": query,
        "matches": runtime.memory_context(bot_id=bot_id, query=query, limit=limit),
        "episodes": runtime.memory_recent_episodes(bot_id=bot_id, limit=min(limit, 20)),
        "stats": runtime.memory_stats(bot_id=bot_id),
    }

