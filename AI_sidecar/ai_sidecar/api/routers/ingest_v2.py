from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from ai_sidecar.api.deps import get_runtime
from ai_sidecar.contracts.events import (
    ActorDeltaPushRequest,
    ChatStreamIngestRequest,
    ConfigDoctrineFingerprintRequest,
    EventBatchIngestRequest,
    IngestAcceptedResponse,
    QuestTransitionRequest,
)
from ai_sidecar.lifecycle import RuntimeState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v2/ingest", tags=["ingest-v2"])


@router.post("/event", response_model=IngestAcceptedResponse)
def ingest_event_batch(
    payload: EventBatchIngestRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> IngestAcceptedResponse:
    try:
        result = runtime.ingest_event_batch(payload)
        logger.info(
            "v2_event_ingested",
            extra={
                "event": "v2_event_ingested",
                "bot_id": payload.meta.bot_id,
                "accepted": result.accepted,
                "dropped": result.dropped,
            },
        )
        return result
    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "v2_event_ingest_failed",
            extra={"event": "v2_event_ingest_failed", "bot_id": payload.meta.bot_id, "trace_id": payload.meta.trace_id},
        )
        raise HTTPException(status_code=500, detail="v2_event_ingest_failed") from None


@router.post("/actors", response_model=IngestAcceptedResponse)
def ingest_actor_delta(
    payload: ActorDeltaPushRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> IngestAcceptedResponse:
    try:
        observed_count = len(payload.actors)
        removed_count = len(payload.removed_actor_ids)
        hostile_count = sum(
            1
            for item in payload.actors
            if (str(item.relation or "").strip().lower() in {"hostile", "enemy", "monster"})
            or (str(item.actor_type or "").strip().lower() == "monster")
        )
        logger.info(
            "v2_actor_delta_received",
            extra={
                "event": "v2_actor_delta_received",
                "bot_id": payload.meta.bot_id,
                "revision": payload.revision,
                "observed_count": observed_count,
                "removed_count": removed_count,
                "hostile_count": hostile_count,
            },
        )
        result = runtime.ingest_actor_delta(payload)
        logger.info(
            "v2_actors_ingested",
            extra={
                "event": "v2_actors_ingested",
                "bot_id": payload.meta.bot_id,
                "revision": payload.revision,
                "observed_count": observed_count,
                "removed_count": removed_count,
                "hostile_count": hostile_count,
                "accepted": result.accepted,
                "dropped": result.dropped,
                "ingest_message": result.message,
            },
        )
        return result
    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "v2_actor_ingest_failed",
            extra={"event": "v2_actor_ingest_failed", "bot_id": payload.meta.bot_id, "trace_id": payload.meta.trace_id},
        )
        raise HTTPException(status_code=500, detail="v2_actor_ingest_failed") from None


@router.post("/chat", response_model=IngestAcceptedResponse)
def ingest_chat_stream(
    payload: ChatStreamIngestRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> IngestAcceptedResponse:
    try:
        result = runtime.ingest_chat_stream(payload)
        logger.info(
            "v2_chat_ingested",
            extra={"event": "v2_chat_ingested", "bot_id": payload.meta.bot_id, "accepted": result.accepted},
        )
        return result
    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "v2_chat_ingest_failed",
            extra={"event": "v2_chat_ingest_failed", "bot_id": payload.meta.bot_id, "trace_id": payload.meta.trace_id},
        )
        raise HTTPException(status_code=500, detail="v2_chat_ingest_failed") from None


@router.post("/config", response_model=IngestAcceptedResponse)
def ingest_config_update(
    payload: ConfigDoctrineFingerprintRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> IngestAcceptedResponse:
    try:
        result = runtime.ingest_config_update(payload)
        logger.info(
            "v2_config_ingested",
            extra={"event": "v2_config_ingested", "bot_id": payload.meta.bot_id, "accepted": result.accepted},
        )
        return result
    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "v2_config_ingest_failed",
            extra={"event": "v2_config_ingest_failed", "bot_id": payload.meta.bot_id, "trace_id": payload.meta.trace_id},
        )
        raise HTTPException(status_code=500, detail="v2_config_ingest_failed") from None


@router.post("/quest", response_model=IngestAcceptedResponse)
def ingest_quest_transition(
    payload: QuestTransitionRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> IngestAcceptedResponse:
    try:
        result = runtime.ingest_quest_transition(payload)
        logger.info(
            "v2_quest_ingested",
            extra={"event": "v2_quest_ingested", "bot_id": payload.meta.bot_id, "accepted": result.accepted},
        )
        return result
    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "v2_quest_ingest_failed",
            extra={"event": "v2_quest_ingest_failed", "bot_id": payload.meta.bot_id, "trace_id": payload.meta.trace_id},
        )
        raise HTTPException(status_code=500, detail="v2_quest_ingest_failed") from None
