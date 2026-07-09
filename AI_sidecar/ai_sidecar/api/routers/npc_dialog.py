"""NPC dialog endpoints — bridge plugin uses these for LLM-powered NPC conversations."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from ai_sidecar.api.deps import get_runtime
from ai_sidecar.lifecycle import RuntimeState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/npc", tags=["npc-dialog"])


@router.post("/talk/{npc_id}")
def talk_to_npc(npc_id: str, bot_id: str = "default", npc_name: str = "", npc_type: str = "generic",
                runtime: RuntimeState = Depends(get_runtime)) -> dict:
    """Start a conversation with an NPC. Returns the command to execute."""
    if runtime.npc_dialog is None:
        return {"ok": False, "error": "npc_dialog_unavailable"}
    cmd = runtime.npc_dialog.start_dialog(bot_id, npc_id, npc_name, npc_type)
    return {"ok": True, "command": cmd}


@router.post("/respond/{bot_id}")
def respond_to_npc(bot_id: str, npc_text: str = "", options: list[dict] | None = None,
                   runtime: RuntimeState = Depends(get_runtime)) -> dict:
    """Process NPC response and return the next command. Returns None when conversation ends."""
    if runtime.npc_dialog is None:
        return {"ok": False, "error": "npc_dialog_unavailable"}
    cmd = runtime.npc_dialog.process_response(bot_id, npc_text, options or [])
    state = runtime.npc_dialog.get_state(bot_id)
    return {
        "ok": True,
        "command": cmd,
        "is_complete": state.is_complete if state else True,
        "npc_type": state.npc_type if state else "unknown",
    }


@router.get("/state/{bot_id}")
def npc_dialog_state(bot_id: str, runtime: RuntimeState = Depends(get_runtime)) -> dict:
    """Get the current NPC dialog state for a bot."""
    if runtime.npc_dialog is None:
        return {"ok": False, "error": "npc_dialog_unavailable"}
    state = runtime.npc_dialog.get_state(bot_id)
    if state is None:
        return {"ok": True, "active": False}
    return {
        "ok": True,
        "active": True,
        "npc_id": state.npc_id,
        "npc_name": state.npc_name,
        "npc_type": state.npc_type,
        "turn_count": len(state.dialog_history),
        "is_complete": state.is_complete,
    }


@router.post("/end/{bot_id}")
def end_npc_dialog(bot_id: str, runtime: RuntimeState = Depends(get_runtime)) -> dict:
    """End an NPC conversation."""
    if runtime.npc_dialog is None:
        return {"ok": False, "error": "npc_dialog_unavailable"}
    runtime.npc_dialog.end_dialog(bot_id)
    return {"ok": True, "message": "dialog_ended"}
