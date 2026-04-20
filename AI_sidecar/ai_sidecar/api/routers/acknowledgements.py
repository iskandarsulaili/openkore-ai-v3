from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from ai_sidecar.api.deps import get_runtime
from ai_sidecar.contracts.actions import ActionAckRequest, ActionAckResponse
from ai_sidecar.lifecycle import RuntimeState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/acknowledgements", tags=["acknowledgements"])


@router.post("/action", response_model=ActionAckResponse)
def acknowledge_action(
    payload: ActionAckRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> ActionAckResponse:
    acknowledged, status = runtime.acknowledge(payload)
    logger.info(
        "action_acknowledged",
        extra={"event": "action_acknowledged", "bot_id": payload.meta.bot_id},
    )
    return ActionAckResponse(
        ok=True,
        acknowledged=acknowledged,
        action_id=payload.action_id,
        status=status,
    )

