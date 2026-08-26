"""Peer-host (bot serves maps) API — observable status + manual start/stop.

The bot's peer-host capacity supervisor (ai_sidecar/peer_host.py) exposes its
state here so the conscious tier / operator can see + control it. Complementarity
rule: it reuses the launcher's linux-host-map-server binary and refuses to spawn
on a box that already owns the map port (single-writer EVE).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ai_sidecar.api.deps import get_runtime
from ai_sidecar.lifecycle import RuntimeState

router = APIRouter(prefix="/v1/peer-host", tags=["peer-host"])


class PeerHostAction(BaseModel):
    action: str  # "start" | "stop"


def _sup(runtime: RuntimeState) -> Any:
    s = getattr(runtime, "peer_host", None)
    if s is None:
        raise HTTPException(status_code=404, detail="peer-host supervisor not initialized")
    return s


@router.get("/status")
def status(runtime: RuntimeState = Depends(get_runtime)) -> dict[str, Any]:
    s = getattr(runtime, "peer_host", None)
    if s is None:
        return {"initialized": False}
    return {"initialized": True, **s.status()}


@router.post("/start")
def start(runtime: RuntimeState = Depends(get_runtime)) -> dict[str, Any]:
    s = _sup(runtime)
    ok, detail = s.start()
    if not ok:
        raise HTTPException(status_code=400, detail=detail)
    return {"ok": True, "detail": detail}


@router.post("/stop")
def stop(runtime: RuntimeState = Depends(get_runtime)) -> dict[str, Any]:
    s = _sup(runtime)
    s.stop()
    return {"ok": True}
