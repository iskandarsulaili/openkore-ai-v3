from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from ai_sidecar.api.deps import get_runtime
from ai_sidecar.config import settings
from ai_sidecar.lifecycle import RuntimeState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v2/observability", tags=["observability-v2"])


class _IncidentMutationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assignee: str = Field(default="", max_length=128)


class _DoctrinePublishRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = Field(min_length=1, max_length=128)
    policy: dict[str, object] = Field(default_factory=dict)
    canary_percentage: float = Field(default=100.0, ge=0.0, le=100.0)
    activate: bool = True
    author: str = Field(default="", max_length=128)
    trace_id: str | None = Field(default=None, min_length=1, max_length=256)


class _DoctrineRollbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_version: str | None = Field(default=None, min_length=1, max_length=128)
    author: str = Field(default="", max_length=128)
    trace_id: str | None = Field(default=None, min_length=1, max_length=256)


@router.get("/metrics", response_class=PlainTextResponse)
def metrics(runtime: RuntimeState = Depends(get_runtime)) -> PlainTextResponse:
    text = runtime.observability_metrics_text()
    if not text and not settings.observability_enable_metrics:
        text = "# observability metrics disabled\n"
    return PlainTextResponse(content=text, media_type="text/plain; version=0.0.4; charset=utf-8")


@router.get("/traces")
def traces(
    limit: int = Query(default=50, ge=1, le=500),
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, object]:
    rows = runtime.observability_recent_traces(limit=limit)
    return {"ok": True, "count": len(rows), "traces": rows}


@router.get("/traces/{trace_id}")
def trace(
    trace_id: str,
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, object]:
    rows = runtime.observability_trace(trace_id=trace_id)
    return {"ok": True, "trace_id": trace_id, "count": len(rows), "events": rows}


@router.get("/incidents")
def incidents(
    include_closed: bool = Query(default=False),
    limit: int = Query(default=100, ge=1, le=2000),
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, object]:
    rows = runtime.observability_incidents(include_closed=include_closed, limit=limit)
    return {"ok": True, "count": len(rows), "incidents": rows}


@router.post("/incidents/{incident_id}/ack")
def ack_incident(
    incident_id: str,
    payload: _IncidentMutationRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, object]:
    result = runtime.observability_ack_incident(incident_id=incident_id, assignee=payload.assignee)
    return {"ok": bool(result.get("ok")), "result": result}


@router.post("/incidents/{incident_id}/escalate")
def escalate_incident(
    incident_id: str,
    payload: _IncidentMutationRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, object]:
    result = runtime.observability_escalate_incident(incident_id=incident_id, assignee=payload.assignee)
    return {"ok": bool(result.get("ok")), "result": result}


@router.get("/explainability")
def explainability(
    kind: str | None = Query(default=None),
    bot_id: str | None = Query(default=None),
    trace_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=2000),
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, object]:
    rows = runtime.observability_explainability(kind=kind, bot_id=bot_id, trace_id=trace_id, limit=limit)
    return {"ok": True, "count": len(rows), "records": rows}


@router.get("/security/violations")
def security_violations(
    limit: int = Query(default=100, ge=1, le=2000),
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, object]:
    rows = runtime.observability_security_violations(limit=limit)
    return {"ok": True, "count": len(rows), "violations": rows}


@router.get("/doctrine/active")
def doctrine_active(runtime: RuntimeState = Depends(get_runtime)) -> dict[str, object]:
    return runtime.observability_doctrine_active()


@router.get("/doctrine/versions")
def doctrine_versions(
    limit: int = Query(default=50, ge=1, le=200),
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, object]:
    return runtime.observability_doctrine_versions(limit=limit)


@router.post("/doctrine/publish")
def doctrine_publish(
    payload: _DoctrinePublishRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, object]:
    result = runtime.observability_doctrine_publish(
        version=payload.version,
        policy=payload.policy,
        canary_percentage=payload.canary_percentage,
        activate=payload.activate,
        author=payload.author,
        trace_id=payload.trace_id,
    )
    logger.warning(
        "doctrine_publish_requested",
        extra={
            "event": "doctrine_publish_requested",
            "version": payload.version,
            "activate": payload.activate,
            "ok": bool(result.get("ok")),
        },
    )
    return result


@router.post("/doctrine/rollback")
def doctrine_rollback(
    payload: _DoctrineRollbackRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, object]:
    result = runtime.observability_doctrine_rollback(
        target_version=payload.target_version,
        author=payload.author,
        trace_id=payload.trace_id,
    )
    logger.warning(
        "doctrine_rollback_requested",
        extra={
            "event": "doctrine_rollback_requested",
            "target_version": payload.target_version,
            "ok": bool(result.get("ok")),
        },
    )
    return result

