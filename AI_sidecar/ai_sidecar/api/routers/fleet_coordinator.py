"""FleetCoordinator API router — endpoints for real-time multi-bot coordination."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from ai_sidecar.api.deps import get_runtime
from ai_sidecar.lifecycle import RuntimeState

router = APIRouter(prefix="/v1/fleet", tags=["fleet_coordinator"])


def _require_coordinator(runtime: RuntimeState):
    """Get the FleetCoordinator or raise 503 if not initialized."""
    coord = getattr(runtime, "fleet_coordinator", None)
    if coord is None:
        raise HTTPException(
            status_code=503,
            detail="FleetCoordinator not initialized",
        )
    return coord


@router.post("/register")
def register_bot(
    body: dict[str, Any],
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, Any]:
    """Register a bot with the fleet coordinator.

    Body:
        bot_id (str, required): Unique bot identifier.
        capabilities (list[str], optional): List of roles this bot can perform.

    Returns:
        Dict with assigned role and bot state.
    """
    coord = _require_coordinator(runtime)
    bot_id = body.get("bot_id", "").strip()
    if not bot_id:
        raise HTTPException(status_code=400, detail="bot_id is required")
    capabilities = body.get("capabilities") or None
    result = coord.register_bot(bot_id, capabilities=capabilities)
    return {
        "ok": True,
        **result,
    }


@router.post("/heartbeat")
def bot_heartbeat(
    body: dict[str, Any],
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, Any]:
    """Bot sends its live state as a heartbeat.

    Body:
        bot_id (str, required): Unique bot identifier.
        position (list[int], optional): [x, y] coordinates.
        map_name (str, optional): Current map name.
        hp (int, optional): Current HP.
        hp_max (int, optional): Maximum HP.
        sp (int, optional): Current SP.
        sp_max (int, optional): Maximum SP.
        level (int, optional): Bot level.
        job_level (int, optional): Job level.
        zeny (int, optional): Current zeny.
        weight (int, optional): Current weight.
        max_weight (int, optional): Maximum weight.
        status_message (str, optional): Status text.
        active_objective (str, optional): Current objective.

    Returns:
        Updated bot state and any pending messages.
    """
    coord = _require_coordinator(runtime)
    bot_id = body.get("bot_id", "").strip()
    if not bot_id:
        raise HTTPException(status_code=400, detail="bot_id is required")

    state = coord.update_bot_state(bot_id, body)
    if state is None:
        # Auto-register on first heartbeat
        capabilities = body.get("capabilities") or None
        coord.register_bot(bot_id, capabilities=capabilities)
        state = coord.update_bot_state(bot_id, body)

    # Check for auto-reassign if enough time has passed
    pending_messages = coord.get_messages(bot_id, since=0)

    return {
        "ok": True,
        "bot": state,
        "pending_messages": pending_messages,
    }


@router.get("/status")
def fleet_status(
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, Any]:
    """Get status of the entire fleet.

    Returns:
        Full fleet status including bots, parties, goals, and shared knowledge.
    """
    coord = _require_coordinator(runtime)
    return {
        "ok": True,
        **coord.status(),
    }


@router.get("/state/{bot_id}")
def bot_state(
    bot_id: str,
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, Any]:
    """Get a specific bot's state.

    Args:
        bot_id: Bot identifier.

    Returns:
        Bot state dict.
    """
    coord = _require_coordinator(runtime)
    # FleetCoordinator has NO get_bot_state (the old call 500'd every time).
    # The live per-bot state lives in the snapshot cache + charstatus reader;
    # merge them for a complete view (this is what the fleet actually knows).
    state: dict[str, Any] = {}
    if runtime.snapshot_cache is not None:
        snap = runtime.snapshot_cache.get(bot_id)
        if snap is not None:
            state["snapshot"] = snap
    reader = getattr(runtime, "charstatus_reader", None)
    if reader is not None:
        cs = reader.get(bot_id)
        if cs is not None:
            state["charstatus"] = cs
    registered = coord.get_bot(bot_id) if hasattr(coord, "get_bot") else None
    if registered is not None:
        state["registration"] = vars(registered) if hasattr(registered, "__dict__") else registered
    if not state:
        raise HTTPException(status_code=404, detail=f"bot not found: {bot_id}")
    return {
        "ok": True,
        "bot": state,
    }


@router.get("/charstatus/{bot_id}")
def bot_charstatus(
    bot_id: str,
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, Any]:
    """Get the complete real-time charstatus contract for a bot.

    This is the authoritative INPUT for all three brains (Conscious/LLM,
    Subconscious/ML, Reflex). It prefers the durable charstatus.json file the
    bridge writes (full enriched contract: identity, vitals, position,
    inventory, stats/skills, combat, environment, party, economy, AI
    internals, telemetry) and falls back to the in-memory snapshot cache.
    Read-only for brains.

    Args:
        bot_id: Bot identifier.

    Returns:
        Full charstatus contract dict.
    """
    reader = getattr(runtime, "charstatus_reader", None)
    if reader is not None:
        data = reader.get(bot_id)
        if data is not None:
            return {"ok": True, "bot_id": bot_id, "source": "charstatus.json", **data}
    snap = None
    if runtime.snapshot_cache is not None:
        snap = runtime.snapshot_cache.get(bot_id)
    if snap is None:
        raise HTTPException(status_code=404, detail=f"no charstatus for bot: {bot_id}")
    return {
        "ok": True,
        "bot_id": bot_id,
        "source": "snapshot_cache",
        "schema_version": 1,
        "snapshot": snap.model_dump(mode="json"),
    }


@router.post("/claim/{role}")
def claim_role(
    role: str,
    body: dict[str, Any],
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, Any]:
    """Claim a role for a bot.

    Args:
        role: The role to claim.

    Body:
        bot_id (str, required): Bot identifier.

    Returns:
        Dict with assigned role and reasoning.
    """
    coord = _require_coordinator(runtime)
    bot_id = body.get("bot_id", "").strip()
    if not bot_id:
        raise HTTPException(status_code=400, detail="bot_id is required")
    result = coord.claim_role(bot_id, role)
    return {
        "ok": True,
        **result,
    }


@router.post("/relay")
def relay_message(
    body: dict[str, Any],
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, Any]:
    """Send a message to other bots in the fleet.

    Body:
        sender_id (str, required): Sender bot ID.
        recipient_id (str, required): Recipient bot ID (use "*" for broadcast).
        message_type (str, required): Type of message.
        payload (dict, optional): Message payload.
    """
    coord = _require_coordinator(runtime)
    sender_id = body.get("sender_id", "").strip()
    recipient_id = body.get("recipient_id", "").strip()
    message_type = body.get("message_type", "").strip()

    if not sender_id:
        raise HTTPException(status_code=400, detail="sender_id is required")
    if not recipient_id:
        raise HTTPException(status_code=400, detail="recipient_id is required")
    if not message_type:
        raise HTTPException(status_code=400, detail="message_type is required")

    result = coord.send_message(
        sender_id=sender_id,
        recipient_id=recipient_id,
        message_type=message_type,
        payload=body.get("payload"),
    )
    return {
        "ok": True,
        **result,
    }


@router.post("/goal")
def set_goal(
    body: dict[str, Any],
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, Any]:
    """Set a team goal for the fleet.

    Body:
        goal_type (str, required): Type of goal (hunt, mvp, quest, level, farm, trade, pvp, gvg).
        params (dict, optional): Goal parameters (target, location, quantity, etc.).
        params.priority (int, optional): Priority from 1-10, default 5.

    Returns:
        The created goal.
    """
    coord = _require_coordinator(runtime)
    goal_type = body.get("goal_type", "").strip()
    if not goal_type:
        raise HTTPException(status_code=400, detail="goal_type is required")
    params = body.get("params") or {}
    result = coord.set_goal(goal_type, params=params)
    return {
        "ok": True,
        **result,
    }


@router.get("/goals")
def list_goals(
    status: str | None = Query(default="active", description="Filter by status: active, completed, failed, cancelled, or omit for all"),
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, Any]:
    """List fleet goals."""
    coord = _require_coordinator(runtime)
    goals = coord.get_goals(status=status)
    return {
        "ok": True,
        "goals": goals,
        "total": len(goals),
    }


@router.get("/knowledge")
def shared_knowledge(
    knowledge_type: str | None = Query(default=None, description="Filter by type: hunting_spot, danger_zone, mvp_spawn, etc."),
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, Any]:
    """Get shared knowledge across the fleet — learned patterns from ExperienceDatabase."""
    coord = _require_coordinator(runtime)
    knowledge = coord.get_shared_knowledge(knowledge_type=knowledge_type)
    return {
        "ok": True,
        "knowledge": knowledge,
        "total": len(knowledge),
    }


@router.get("/messages/{bot_id}")
def bot_messages(
    bot_id: str,
    since: float = Query(default=0.0, description="Only messages after this timestamp"),
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, Any]:
    """Get pending messages for a bot."""
    coord = _require_coordinator(runtime)
    messages = coord.get_messages(bot_id, since=since)
    return {
        "ok": True,
        "bot_id": bot_id,
        "messages": messages,
        "total": len(messages),
    }


@router.post("/reassign")
def trigger_reassign(
    body: dict[str, Any],
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, Any]:
    """Trigger role reassignment for underperforming bots.

    Body:
        bot_id (str, optional): Specific bot to reassign. If omitted, checks all bots.
    """
    coord = _require_coordinator(runtime)
    bot_id = body.get("bot_id", "").strip() or None
    actions = coord.auto_reassign(bot_id=bot_id)
    return {
        "ok": True,
        "reassignments": actions,
        "total": len(actions),
    }


@router.post("/outcome")
def record_outcome(
    body: dict[str, Any],
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, Any]:
    """Record an action outcome for cross-bot learning.

    Body:
        bot_id (str, required): Bot that performed the action.
        context_type (str, required): Context (combat, economy, survival, quest, craft, trade, refine, pvp, gvg, mvp).
        action_taken (str, required): What action was taken.
        success (bool, required): Whether the action succeeded.
        reward (float, optional): XP gained or zeny earned.
        map_name (str, optional): Map where it happened.
        monster_name (str, optional): Monster involved.
        role (str, optional): Role the bot was in.
        damage (float, optional): Damage dealt.
        healing (float, optional): Healing done.
        zeny (float, optional): Zeny earned.
        xp (float, optional): XP gained.
        death (bool, optional): Whether the bot died.
        details (dict, optional): Additional details.
    """
    coord = _require_coordinator(runtime)
    bot_id = body.get("bot_id", "").strip()
    context_type = body.get("context_type", "").strip()
    action_taken = body.get("action_taken", "").strip()
    success = bool(body.get("success", False))

    if not bot_id:
        raise HTTPException(status_code=400, detail="bot_id is required")
    if not context_type:
        raise HTTPException(status_code=400, detail="context_type is required")
    if not action_taken:
        raise HTTPException(status_code=400, detail="action_taken is required")

    coord.record_outcome(
        bot_id=bot_id,
        context_type=context_type,
        action_taken=action_taken,
        success=success,
        reward=float(body.get("reward", 0.0)),
        map_name=body.get("map_name", ""),
        monster_name=body.get("monster_name", ""),
        role=body.get("role", ""),
        details=body.get("details"),
        damage=float(body.get("damage", 0.0)),
        healing=float(body.get("healing", 0.0)),
        zeny=float(body.get("zeny", 0.0)),
        xp=float(body.get("xp", 0.0)),
        death=bool(body.get("death", False)),
        response_time_s=float(body.get("response_time_s", 0.0)),
    )
    return {
        "ok": True,
        "recorded": True,
    }


@router.post("/knowledge/add")
def add_knowledge(
    body: dict[str, Any],
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, Any]:
    """Add or update shared knowledge.

    Body:
        knowledge_type (str, required): Type (hunting_spot, danger_zone, mvp_spawn, etc.).
        key (str, required): Unique key within the type.
        value (dict, required): Knowledge data.
        reported_by (str, optional): Bot ID that discovered this.
        confidence (float, optional): Confidence score (0.0 to 1.0).
    """
    coord = _require_coordinator(runtime)
    knowledge_type = body.get("knowledge_type", "").strip()
    key = body.get("key", "").strip()
    value = body.get("value")
    if not knowledge_type:
        raise HTTPException(status_code=400, detail="knowledge_type is required")
    if not key:
        raise HTTPException(status_code=400, detail="key is required")
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail="value must be a dict")

    coord.add_shared_knowledge(
        knowledge_type=knowledge_type,
        key=key,
        value=value,
        reported_by=body.get("reported_by", ""),
        confidence=float(body.get("confidence", 1.0)),
    )
    return {
        "ok": True,
        "added": True,
    }
