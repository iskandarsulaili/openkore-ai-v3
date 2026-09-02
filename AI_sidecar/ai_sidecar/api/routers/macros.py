from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

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
            "publication_id": str(data.get("publication_id") or ""),
            "version": str(data.get("version") or ""),
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


# ── CRUD (2026-09-02): the macros router was publish-only. Add list/get/delete
# ── so the macro-agent's shared registry is fully manageable (Create/Read/Update/
# ── Delete). These operate on the MacroAgent registry (committed macros/ dir).


@router.get("/registry")
def list_registry(runtime: RuntimeState = Depends(get_runtime)) -> dict:
    """List all macros in the shared registry (reusable by other users)."""
    agent = getattr(runtime, "macro_agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="macro_agent_unavailable")
    return {"ok": True, "macros": agent.registry()}


@router.get("/registry/{name}")
def get_registry_macro(name: str, runtime: RuntimeState = Depends(get_runtime)) -> dict:
    """Get a single registry macro by name."""
    agent = getattr(runtime, "macro_agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="macro_agent_unavailable")
    for item in agent.registry():
        if item["name"] == name:
            return {"ok": True, "macro": item}
    raise HTTPException(status_code=404, detail="macro_not_found")


@router.delete("/registry/{name}")
def delete_registry_macro(name: str, runtime: RuntimeState = Depends(get_runtime)) -> dict:
    """Delete a macro from the registry (demote a repeatedly-failing one)."""
    agent = getattr(runtime, "macro_agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="macro_agent_unavailable")
    removed = agent.demote(name)
    if not removed:
        raise HTTPException(status_code=404, detail="macro_not_found")
    return {"ok": True, "removed": name}


@router.post("/generate")
def generate_macro(
    payload: dict,
    runtime: RuntimeState = Depends(get_runtime),
) -> dict:
    """AI macro-agent: generate + verify a macro for a specific case.

    Body: {"case": "...", "context": {...}, "bot_id": "..."}
    The agent asks the LLM for a macro, verifies it (parse+security+dry-run+
    outcome), and returns the result. If verified, it is registered as a
    skill-set pattern in the MacroIntelligence engine.
    """
    agent = getattr(runtime, "macro_agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="macro_agent_unavailable")
    case = str(payload.get("case") or "").strip()
    if not case:
        raise HTTPException(status_code=422, detail="case_required")
    context = payload.get("context") or {}
    bot_id = str(payload.get("bot_id") or "default")
    macro = agent.generate(case=case, context=context, bot_id=bot_id)
    if macro is None:
        return {"ok": False, "message": "generation_failed", "case": case}
    if macro.verified:
        agent.register(macro, getattr(runtime, "macro_intelligence", None))
    return {
        "ok": True,
        "verified": macro.verified,
        "name": macro.name,
        "case": macro.case,
        "lines": macro.lines,
        "errors": macro.verification.errors if macro.verification else [],
        "warnings": macro.verification.warnings if macro.verification else [],
        "registered": macro.verified,
    }


@router.post("/reward")
def reward_macro(payload: dict, runtime: RuntimeState = Depends(get_runtime)) -> dict:
    """Reward a macro for a successful outcome (self-improving)."""
    agent = getattr(runtime, "macro_agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="macro_agent_unavailable")
    name = str(payload.get("name") or "").strip()
    bot_id = str(payload.get("bot_id") or "default")
    detail = str(payload.get("detail") or "")
    if not name:
        raise HTTPException(status_code=422, detail="name_required")
    agent.reward(name, bot_id=bot_id, detail=detail)
    return {"ok": True, "rewarded": name}


@router.post("/punish")
def punish_macro(payload: dict, runtime: RuntimeState = Depends(get_runtime)) -> dict:
    """Punish a macro for a failed outcome (self-improving)."""
    agent = getattr(runtime, "macro_agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="macro_agent_unavailable")
    name = str(payload.get("name") or "").strip()
    bot_id = str(payload.get("bot_id") or "default")
    detail = str(payload.get("detail") or "")
    if not name:
        raise HTTPException(status_code=422, detail="name_required")
    agent.punish(name, bot_id=bot_id, detail=detail)
    return {"ok": True, "punished": name}
