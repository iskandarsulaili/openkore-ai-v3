from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends

from ai_sidecar.api.deps import get_runtime
from ai_sidecar.lifecycle import RuntimeState

router = APIRouter(tags=["health"])


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
    readiness = runtime.readiness_indicators()
    fleet_central_enabled = bool(readiness.get("fleet_central_enabled", True))
    readiness_degraded = bool(
        readiness.get("planner_stale")
        or (fleet_central_enabled and readiness.get("fleet_central_stale"))
        or readiness.get("objective_scheduler_degraded")
    )
    return {
        "ok": True,
        "status": "ready",
        "readiness_degraded": readiness_degraded,
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
        "planner_stale": bool(readiness.get("planner_stale", True)),
        "planner_stale_seconds": readiness.get("planner_stale_seconds"),
        "planner_stale_threshold_s": readiness.get("planner_stale_threshold_s"),
        "planner_last_updated_at": readiness.get("planner_last_updated_at"),
        "fleet_central_enabled": fleet_central_enabled,
        "fleet_central_stale": bool(readiness.get("fleet_central_stale", True)),
        "fleet_central_available": bool(readiness.get("fleet_central_available", False)),
        "fleet_mode": readiness.get("fleet_mode"),
        "fleet_last_sync_at": readiness.get("fleet_last_sync_at"),
        "objective_scheduler_degraded": bool(readiness.get("objective_scheduler_degraded", True)),
        "objective_scheduler_degraded_reason": readiness.get("objective_scheduler_degraded_reason"),
        "pdca_running": bool(readiness.get("pdca_running", False)),
        "pdca_circuit_breaker_tripped": readiness.get("pdca_circuit_breaker_tripped"),
        "autonomy_policy": dict(runtime.autonomy_policy),
        "counters": runtime.counter_snapshot(),
    }
