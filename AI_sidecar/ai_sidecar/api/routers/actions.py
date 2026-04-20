from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends

from ai_sidecar.api.deps import get_runtime
from ai_sidecar.config import settings
from ai_sidecar.contracts.actions import (
    ActionProposal,
    NextActionRequest,
    NextActionResponse,
    QueueActionRequest,
    QueueActionResponse,
)
from ai_sidecar.contracts.common import NoopActionPayload
from ai_sidecar.lifecycle import RuntimeState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/actions", tags=["actions"])


@router.post("/queue", response_model=QueueActionResponse)
def queue_action(
    payload: QueueActionRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> QueueActionResponse:
    proposal = payload.proposal
    if proposal.expires_at <= proposal.created_at:
        proposal = ActionProposal(
            **proposal.model_dump(),
            expires_at=proposal.created_at + timedelta(seconds=settings.action_default_ttl_seconds),
        )

    accepted, status, action_id, reason = runtime.queue_action(proposal=proposal, bot_id=payload.meta.bot_id)
    logger.info(
        "action_queue_result",
        extra={
            "event": "action_queue_result",
            "bot_id": payload.meta.bot_id,
            "accepted": accepted,
            "status": str(status),
            "reason": reason,
        },
    )

    if accepted:
        message = "action queued"
    else:
        reason_map = {
            "idempotent_duplicate": "idempotent duplicate",
            "conflict_key_blocked": "conflict key blocked by higher-priority action",
            "action_already_expired": "action already expired",
            "queue_full": "queue full",
            "queue_full_lower_priority": "queue full and action priority too low",
        }
        message = reason_map.get(reason, reason)

    return QueueActionResponse(
        ok=True,
        accepted=accepted,
        message=message,
        bot_id=payload.meta.bot_id,
        action_id=action_id,
        status=status,
    )


@router.post("/next", response_model=NextActionResponse)
def next_action(
    payload: NextActionRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> NextActionResponse:
    started = runtime.latency_router.begin()
    action = runtime.next_action(payload.meta.bot_id, poll_id=payload.poll_id)
    elapsed_ms = runtime.latency_router.end("actions.next", started)

    if action is not None and not runtime.latency_router.within_budget(elapsed_ms):
        runtime.rollback_action_dispatch(action.action_id)
        action = None

    if action is None:
        reason = "latency_budget_exceeded" if not runtime.latency_router.within_budget(elapsed_ms) else "no_action_available"
        return NextActionResponse(
            ok=True,
            bot_id=payload.meta.bot_id,
            poll_id=payload.poll_id,
            has_action=False,
            action=NoopActionPayload(
                action_id="noop",
                kind="noop",
                command="",
                conflict_key=None,
                expires_at=datetime.now(UTC) + timedelta(seconds=1),
            ),
            reason=reason,
        )

    return NextActionResponse(
        ok=True,
        bot_id=payload.meta.bot_id,
        poll_id=payload.poll_id,
        has_action=True,
        action=action,
        reason="action_ready",
    )
