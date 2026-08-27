"""
Party API endpoints — follow positioning and leader coordination for the bridge.
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, Depends

from ai_sidecar.api.deps import get_runtime
from ai_sidecar.lifecycle import RuntimeState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v2/party", tags=["party"])


@router.post("/follow/position")
def get_follow_position(
    runtime: RuntimeState = Depends(get_runtime),
):
    """Get the optimal follow position based on party composition."""
    try:
        follow = getattr(runtime, "party_follow", None)
        if follow is None:
            return {"ok": False, "error": "party_follow_not_initialized"}
        
        position = follow.get_position(
            my_role="melee_dps",
            party_roles=["tank", "healer", "ranged_dps"],
        )
        return {"ok": True, "position": position}
    except Exception as e:
        logger.warning("party_follow_failed: %s", e)
        return {"ok": False, "error": str(e)}


@router.post("/leader/status")
def get_leader_status(
    runtime: RuntimeState = Depends(get_runtime),
):
    """Get party leader coordination status."""
    try:
        leader = getattr(runtime, "party_leader", None)
        if leader is None:
            return {"ok": False, "error": "party_leader_not_initialized"}
        
        status = leader.should_wait()
        return {"ok": True, "status": status}
    except Exception as e:
        logger.warning("party_leader_failed: %s", e)
        return {"ok": False, "error": str(e)}


@router.post("/status")
def party_status(
    payload: dict,
    runtime: RuntimeState = Depends(get_runtime),
):
    """Ingest periodic party-status relay from the bridge.

    The bridge POSTs /v2/party/status every 30s (and on death/respawn/
    map-change/reconnect) with the bot's party membership + leader state. This
    endpoint records it so the fleet coordinator / party systems have a live
    view even when the bot is not mid-encounter. Previously 404 — the router
    only had /follow/position + /leader/status, so every relay silently failed.
    """
    try:
        bot_id = str(payload.get("bot_id") or "")
        in_party = bool(payload.get("in_party"))
        members = payload.get("party_members") or []
        is_leader = bool(payload.get("is_party_leader"))
        map_name = str(payload.get("map") or "")
        if bot_id:
            runtime.party_status_store[bot_id] = {
                "in_party": in_party,
                "members": members,
                "is_party_leader": is_leader,
                "map": map_name,
                "reason": str(payload.get("reason") or "periodic"),
                "received_at": __import__("time").time(),
            }
        return {"ok": True, "bot_id": bot_id, "in_party": in_party}
    except Exception as e:
        logger.warning("party_status_failed: %s", e)
        return {"ok": False, "error": str(e)}
