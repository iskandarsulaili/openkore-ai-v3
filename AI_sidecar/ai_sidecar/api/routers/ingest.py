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
    reg_bot_id = str(registration.get("bot_id") or payload.meta.bot_id)
    if "Asgards" in reg_bot_id or "Glory" in reg_bot_id or "asgardsglory" in reg_bot_id.lower():
        logger.warning(
            "bot_id_asgards_origins requested=%s registered=%s bot_name=%s",
            payload.meta.bot_id, reg_bot_id, payload.bot_name,
        )
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
    # Override observed_at to now() — the bridge sends game event timestamps
    # which can be older than the TTL, causing immediate cache expiry.
    from datetime import UTC, datetime
    payload.observed_at = datetime.now(UTC)
    # map_known must reflect REAL in-game state, not the act of sending a
    # snapshot. Char-select / disconnected snapshots have no position data
    # (map empty, raw.in_game false). Forcing True here made the heuristic's
    # not-in-game guard (heuristic_service `_assess_impl`) believe every bot
    # knows its map, so phantom actions were emitted against logged-out
    # sessions ("You must be logged in" spam). Derive from actual data.
    try:
        _raw = getattr(payload, "raw", None) or {}
        if not isinstance(_raw, dict):
            _raw = {}
        _in_game = _raw.get("in_game", False)
        _pos = getattr(payload, "position", None)
        _has_map = bool(_pos and getattr(_pos, "map", None))
        payload.map_known = bool(_in_game and _has_map)
    except Exception:
        payload.map_known = False
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
