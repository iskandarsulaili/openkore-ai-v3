from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends

from ai_sidecar.api.deps import get_runtime
from ai_sidecar.lifecycle import RuntimeState

router = APIRouter(prefix="/v1/health", tags=["health"])


@router.get("/live")
def live(runtime: RuntimeState = Depends(get_runtime)) -> dict[str, object]:
    return {
        "ok": True,
        "status": "live",
        "started_at": runtime.started_at.isoformat(),
        "now": datetime.now(UTC).isoformat(),
    }


@router.get("/ready")
def ready(runtime: RuntimeState = Depends(get_runtime)) -> dict[str, object]:
    return {
        "ok": True,
        "status": "ready",
        "bots_registered": runtime.bot_registry.count(),
        "snapshots_cached": runtime.snapshot_cache.count(),
        "telemetry_events": runtime.telemetry_store.count(),
        "latency_avg_ms": round(runtime.latency_router.average_ms(), 3),
        "counters": runtime.counter_snapshot(),
    }

