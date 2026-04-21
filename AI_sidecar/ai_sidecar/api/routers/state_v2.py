from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ai_sidecar.api.deps import get_runtime
from ai_sidecar.lifecycle import RuntimeState

router = APIRouter(prefix="/v2/state", tags=["state-v2"])


@router.get("/{bot_id}")
def enriched_state(
    bot_id: str,
    include_events: bool = Query(default=True),
    event_limit: int = Query(default=50, ge=1, le=500),
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, object]:
    state = runtime.enriched_state(bot_id=bot_id)
    payload = state.model_dump(mode="json")
    if include_events:
        payload["recent_events"] = runtime.recent_ingest_events(bot_id=bot_id, limit=event_limit)
    return {"ok": True, "state": payload}


@router.get("/{bot_id}/graph")
def normalized_state_graph(
    bot_id: str,
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, object]:
    return {"ok": True, "graph": runtime.normalized_state_graph(bot_id=bot_id)}

