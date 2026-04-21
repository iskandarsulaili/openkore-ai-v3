from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

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


@router.post("/actors", response_model=IngestAcceptedResponse)
def ingest_actor_delta(
    payload: ActorDeltaPushRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> IngestAcceptedResponse:
    result = runtime.ingest_actor_delta(payload)
    logger.info(
        "v2_actors_ingested",
        extra={"event": "v2_actors_ingested", "bot_id": payload.meta.bot_id, "accepted": result.accepted},
    )
    return result


@router.post("/chat", response_model=IngestAcceptedResponse)
def ingest_chat_stream(
    payload: ChatStreamIngestRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> IngestAcceptedResponse:
    result = runtime.ingest_chat_stream(payload)
    logger.info(
        "v2_chat_ingested",
        extra={"event": "v2_chat_ingested", "bot_id": payload.meta.bot_id, "accepted": result.accepted},
    )
    return result


@router.post("/config", response_model=IngestAcceptedResponse)
def ingest_config_update(
    payload: ConfigDoctrineFingerprintRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> IngestAcceptedResponse:
    result = runtime.ingest_config_update(payload)
    logger.info(
        "v2_config_ingested",
        extra={"event": "v2_config_ingested", "bot_id": payload.meta.bot_id, "accepted": result.accepted},
    )
    return result


@router.post("/quest", response_model=IngestAcceptedResponse)
def ingest_quest_transition(
    payload: QuestTransitionRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> IngestAcceptedResponse:
    result = runtime.ingest_quest_transition(payload)
    logger.info(
        "v2_quest_ingested",
        extra={"event": "v2_quest_ingested", "bot_id": payload.meta.bot_id, "accepted": result.accepted},
    )
    return result

