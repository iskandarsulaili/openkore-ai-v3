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
    persisted_bots = runtime.repositories.bots.count() if runtime.repositories is not None else runtime.bot_registry.count()
    telemetry_backlog = runtime.telemetry_store.backlog_size()
    return {
        "ok": True,
        "status": "ready",
        "bots_registered": runtime.bot_registry.count(),
        "bots_persisted": persisted_bots,
        "snapshots_cached": runtime.snapshot_cache.count(),
        "snapshots_persisted": runtime.repositories.snapshots.count() if runtime.repositories is not None else 0,
        "telemetry_events": runtime.telemetry_store.count(),
        "telemetry_backlog": telemetry_backlog,
        "persistence_enabled": runtime.repositories is not None,
        "persistence_degraded": runtime.persistence_degraded,
        "sqlite_path": str(runtime.sqlite_path) if runtime.sqlite_path is not None else None,
        "latency_avg_ms": round(runtime.latency_router.average_ms(), 3),
        "counters": runtime.counter_snapshot(),
    }
