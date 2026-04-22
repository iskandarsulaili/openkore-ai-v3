from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from ai_sidecar.api.deps import get_runtime
from ai_sidecar.contracts.state import (
    BotRegistrationRequest,
    BotRegistrationResponse,
    BotStateSnapshot,
    SnapshotIngestResponse,
)
from ai_sidecar.lifecycle import RuntimeState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/ingest", tags=["ingest"])


@router.post("/register", response_model=BotRegistrationResponse)
def register_bot(
    payload: BotRegistrationRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> BotRegistrationResponse:
    registration = runtime.register_bot(payload)
    logger.info(
        "bot_registered",
        extra={"event": "bot_registered", "bot_id": payload.meta.bot_id},
    )
    return BotRegistrationResponse(
        ok=True,
        registered=True,
        bot_id=str(registration.get("bot_id") or payload.meta.bot_id),
        seen_at=registration["seen_at"],
        role=registration.get("role"),
        assignment=registration.get("assignment"),
        liveness_state=str(registration.get("liveness_state") or "online"),
    )


@router.post("/snapshot", response_model=SnapshotIngestResponse)
def ingest_snapshot(
    payload: BotStateSnapshot,
    runtime: RuntimeState = Depends(get_runtime),
) -> SnapshotIngestResponse:
    runtime.ingest_snapshot(payload)
    logger.info(
        "snapshot_ingested",
        extra={"event": "snapshot_ingested", "bot_id": payload.meta.bot_id},
    )
    return SnapshotIngestResponse(
        ok=True,
        accepted=True,
        message="snapshot accepted",
        bot_id=payload.meta.bot_id,
        tick_id=payload.tick_id,
    )
