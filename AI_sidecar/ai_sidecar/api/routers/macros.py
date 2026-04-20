from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from ai_sidecar.api.deps import get_runtime
from ai_sidecar.contracts.actions import ActionStatus
from ai_sidecar.contracts.macros import (
    MacroArtifactPaths,
    MacroPublication,
    MacroPublishRequest,
    MacroPublishResponse,
)
from ai_sidecar.lifecycle import RuntimeState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/macros", tags=["macros"])


@router.post("/publish", response_model=MacroPublishResponse)
def publish_macros(
    payload: MacroPublishRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> MacroPublishResponse:
    ok, data, message = runtime.publish_macros(payload)
    target_bot_id = payload.target_bot_id or payload.meta.bot_id

    if not ok or data is None:
        return MacroPublishResponse(
            ok=False,
            published=False,
            message=message,
            target_bot_id=target_bot_id,
            reload_queued=False,
            reload_reason="publish_failed",
        )

    publication = MacroPublication(
        publication_id=str(data["publication_id"]),
        version=str(data["version"]),
        content_sha256=str(data["content_sha256"]),
        published_at=data["published_at"],
        paths=MacroArtifactPaths(
            macro_file=str(data["macro_file"]),
            event_macro_file=str(data["event_macro_file"]),
            catalog_file=str(data["catalog_file"]),
            manifest_file=str(data["manifest_file"]),
        ),
    )

    reload_status = data.get("reload_status")
    if isinstance(reload_status, ActionStatus):
        status_value = reload_status
    elif isinstance(reload_status, str):
        try:
            status_value = ActionStatus(reload_status)
        except ValueError:
            status_value = None
    else:
        status_value = None

    logger.info(
        "macro_publish_result",
        extra={
            "event": "macro_publish_result",
            "bot_id": target_bot_id,
            "published": True,
            "reload_queued": bool(data.get("reload_queued", False)),
        },
    )

    return MacroPublishResponse(
        ok=True,
        published=True,
        message=message,
        publication=publication,
        target_bot_id=target_bot_id,
        reload_queued=bool(data.get("reload_queued", False)),
        reload_action_id=data.get("reload_action_id"),
        reload_status=status_value,
        reload_reason=str(data.get("reload_reason") or ""),
    )
