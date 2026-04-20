from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from ai_sidecar.api.deps import get_runtime
from ai_sidecar.contracts.telemetry import TelemetryIngestRequest, TelemetryIngestResponse
from ai_sidecar.lifecycle import RuntimeState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/telemetry", tags=["telemetry"])


@router.post("/ingest", response_model=TelemetryIngestResponse)
def ingest_telemetry(
    payload: TelemetryIngestRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> TelemetryIngestResponse:
    result = runtime.telemetry_store.push(payload.meta.bot_id, payload.events)
    runtime.incr("telemetry_ingested", n=result.accepted)
    logger.info(
        "telemetry_ingested",
        extra={"event": "telemetry_ingested", "bot_id": payload.meta.bot_id},
    )
    return result

