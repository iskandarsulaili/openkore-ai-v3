from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query

from ai_sidecar.api.deps import get_runtime
from ai_sidecar.contracts.telemetry import (
    TelemetryIncident,
    TelemetryIngestRequest,
    TelemetryIngestResponse,
    TelemetrySummaryResponse,
)
from ai_sidecar.lifecycle import RuntimeState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/telemetry", tags=["telemetry"])


@router.post("/ingest", response_model=TelemetryIngestResponse)
def ingest_telemetry(
    payload: TelemetryIngestRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> TelemetryIngestResponse:
    result = runtime.ingest_telemetry(payload.meta.bot_id, payload.events)
    logger.info(
        "telemetry_ingested",
        extra={
            "event": "telemetry_ingested",
            "bot_id": payload.meta.bot_id,
            "accepted": result.accepted,
            "dropped": result.dropped,
            "queued_for_retry": result.queued_for_retry,
        },
    )
    return result


@router.get("/summary", response_model=TelemetrySummaryResponse)
def telemetry_summary(
    bot_id: str | None = Query(default=None),
    runtime: RuntimeState = Depends(get_runtime),
) -> TelemetrySummaryResponse:
    data = runtime.telemetry_operational_summary(bot_id=bot_id)
    incidents = [TelemetryIncident.model_validate(item) for item in data.get("recent_incidents", [])]
    return TelemetrySummaryResponse(
        ok=True,
        window_minutes=int(data.get("window_minutes", 0)),
        window_since=data["window_since"],
        total_events=int(data.get("total_events", 0)),
        levels=dict(data.get("levels", {})),
        top_events=list(data.get("top_events", [])),
        recent_incidents=incidents,
    )


@router.get("/incidents", response_model=list[TelemetryIncident])
def recent_incidents(
    bot_id: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=500),
    runtime: RuntimeState = Depends(get_runtime),
) -> list[TelemetryIncident]:
    data = runtime.telemetry_operational_summary(bot_id=bot_id)
    rows = list(data.get("recent_incidents", []))[:limit]
    return [TelemetryIncident.model_validate(item) for item in rows]
