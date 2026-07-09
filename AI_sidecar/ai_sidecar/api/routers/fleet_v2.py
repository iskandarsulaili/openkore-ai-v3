"""Enhanced fleet router with coordination, status, relay, and blackboard endpoints."""

from __future__ import annotations

import logging
import time
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from ai_sidecar.api.deps import get_runtime
from ai_sidecar.contracts.fleet_v2 import (
    FleetBlackboardLocalResponse,
    FleetClaimRequestV2,
    FleetClaimResponseV2,
    FleetConstraintResponse,
    FleetOutcomeReportRequest,
    FleetOutcomeReportResponse,
    FleetRoleResponse,
    FleetSyncRequest,
    FleetSyncResponse,
)
from ai_sidecar.fleet import FleetCoordinatorService, FleetMessage
from ai_sidecar.lifecycle import RuntimeState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v2/fleet", tags=["fleet-v2"])


def _get_coordinator(runtime: RuntimeState) -> FleetCoordinatorService:
    if runtime.fleet_coordinator is None:
        # Lazily create if not already initialized
        from ai_sidecar.fleet.coordinator import FleetCoordinatorService as FCS
        runtime.fleet_coordinator = FCS()
    return runtime.fleet_coordinator


# ── Existing V2 endpoints (unchanged) ──────────────────────────────────────


@router.post("/sync", response_model=FleetSyncResponse)
def sync(payload: FleetSyncRequest, runtime: RuntimeState = Depends(get_runtime)) -> FleetSyncResponse:
    return runtime.fleet_sync(payload)


@router.get("/constraints", response_model=FleetConstraintResponse)
def constraints(bot_id: str, runtime: RuntimeState = Depends(get_runtime)) -> FleetConstraintResponse:
    return runtime.fleet_constraints(bot_id=bot_id)


@router.post("/report-outcome", response_model=FleetOutcomeReportResponse)
def report_outcome(payload: FleetOutcomeReportRequest, runtime: RuntimeState = Depends(get_runtime)) -> FleetOutcomeReportResponse:
    result = runtime.fleet_report_outcome(payload)
    logger.info(
        "fleet_outcome_reported",
        extra={"event": "fleet_outcome_reported", "bot_id": payload.meta.bot_id,
               "event_type": payload.event_type, "ok": result.ok},
    )
    return result


@router.get("/role", response_model=FleetRoleResponse)
def role(bot_id: str, runtime: RuntimeState = Depends(get_runtime)) -> FleetRoleResponse:
    return runtime.fleet_role(bot_id=bot_id)


@router.post("/claim", response_model=FleetClaimResponseV2)
def claim(payload: FleetClaimRequestV2, runtime: RuntimeState = Depends(get_runtime)) -> FleetClaimResponseV2:
    coordinator = _get_coordinator(runtime)
    bot_id = payload.meta.bot_id
    coordinator.register_bot(bot_id)
    assigned_role = coordinator.assign_role(bot_id, payload.claim_type, reason=payload.meta.source)
    return FleetClaimResponseV2(
        ok=True, accepted=True, mode="local",
        claim={"bot_id": bot_id, "role": assigned_role or payload.claim_type},
    )


@router.get("/blackboard", response_model=FleetBlackboardLocalResponse)
def blackboard(bot_id: str, runtime: RuntimeState = Depends(get_runtime)) -> FleetBlackboardLocalResponse:
    coordinator = _get_coordinator(runtime)
    coordinator.register_bot(bot_id)
    bb = coordinator.blackboard_snapshot()
    return FleetBlackboardLocalResponse(
        ok=True, bot_id=bot_id, mode="local",
        blackboard=bb, local_summary={"bots_online": len(coordinator.list_bots())},
    )


# ── NEW: Fleet Status ─────────────────────────────────────────────────────


@router.get("/status")
def fleet_status(bot_id: str = "", runtime: RuntimeState = Depends(get_runtime)):
    """GET /v2/fleet/status — all bots' status and fleet overview."""
    coordinator = _get_coordinator(runtime)
    if bot_id:
        coordinator.register_bot(bot_id)
    status = coordinator.fleet_status()
    return {"ok": True, "mode": "local", **status}


# ── NEW: Fleet Relay (cross-bot messaging) ────────────────────────────────


@router.post("/relay")
def fleet_relay(
    sender_id: str, recipient_id: str = "*", message_type: str = "info",
    payload: dict = {}, runtime: RuntimeState = Depends(get_runtime),
):
    """POST /v2/fleet/relay — send a message to other bots in the fleet."""
    coordinator = _get_coordinator(runtime)
    msg = FleetMessage(
        message_id=str(uuid4()),
        sender_id=sender_id,
        recipient_id=recipient_id,
        message_type=message_type,
        payload=payload,
        sent_at=time.time(),
        ttl_seconds=60,
    )
    coordinator.send_message(msg)
    return {"ok": True, "message_id": msg.message_id, "relayed": True}


@router.get("/messages")
def fleet_messages(bot_id: str, since: float = 0.0, runtime: RuntimeState = Depends(get_runtime)):
    """GET /v2/fleet/messages — retrieve pending messages for a bot."""
    coordinator = _get_coordinator(runtime)
    messages = coordinator.get_messages_for(bot_id, since=since)
    return {
        "ok": True,
        "bot_id": bot_id,
        "messages": [
            {"id": m.message_id, "from": m.sender_id, "type": m.message_type,
             "payload": m.payload, "sent_at": m.sent_at}
            for m in messages
        ],
        "count": len(messages),
    }


# ── NEW: Fleet Coordinate ─────────────────────────────────────────────────


@router.post("/coordinate")
def fleet_coordinate(
    bot_id: str, objective: str = "farming",
    runtime: RuntimeState = Depends(get_runtime),
):
    """POST /v2/fleet/coordinate — request coordination for an objective."""
    coordinator = _get_coordinator(runtime)
    composition = coordinator.suggest_party_composition(objective)
    return {"ok": True, **composition}


@router.post("/register-bot")
def register_bot(
    bot_id: str, available_roles: list[str] = [],
    runtime: RuntimeState = Depends(get_runtime),
):
    """POST /v2/fleet/register-bot — register a bot with the fleet coordinator."""
    coordinator = _get_coordinator(runtime)
    state = coordinator.register_bot(bot_id, available_roles=available_roles or None)
    return {
        "ok": True,
        "bot_id": state.bot_id,
        "current_role": state.current_role,
        "available_roles": state.available_roles,
    }


@router.post("/update-bot")
def update_bot_state(
    bot_id: str, runtime: RuntimeState = Depends(get_runtime),
    **kwargs,
):
    """POST /v2/fleet/update-bot — update bot state in the fleet coordinator."""
    coordinator = _get_coordinator(runtime)
    state = coordinator.update_bot_state(bot_id, **kwargs)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Bot {bot_id} not registered")
    return {"ok": True, "bot_id": bot_id, "current_role": state.current_role}


@router.get("/role-performance")
def role_performance(bot_id: str, runtime: RuntimeState = Depends(get_runtime)):
    """GET /v2/fleet/role-performance — get role performance metrics for a bot."""
    coordinator = _get_coordinator(runtime)
    bot = coordinator.get_bot(bot_id)
    if bot is None:
        raise HTTPException(status_code=404, detail=f"Bot {bot_id} not found")
    return {
        "ok": True,
        "bot_id": bot_id,
        "current_role": bot.current_role,
        "role_scores": {r: m.compute_score() for r, m in bot.role_metrics.items()},
        "role_metrics": {
            r: {"assignments": m.total_assignments, "success_rate": m.success_rate(),
                "score": m.score, "deaths": m.deaths, "damage": m.total_damage_dealt,
                "healing": m.total_healing_done, "zeny": m.total_zeny_earned}
            for r, m in bot.role_metrics.items()
        },
    }


@router.post("/recommend-role")
def recommend_role(bot_id: str, runtime: RuntimeState = Depends(get_runtime)):
    """POST /v2/fleet/recommend-role — get a role change recommendation."""
    coordinator = _get_coordinator(runtime)
    recommendation = coordinator.recommend_role_change(bot_id)
    return {"ok": True, "bot_id": bot_id, **recommendation}


@router.post("/mvp-report")
def report_mvp(
    bot_id: str, mvp_name: str, map_name: str, x: int = 0, y: int = 0,
    hp: int = 0, hp_max: int = 0,
    runtime: RuntimeState = Depends(get_runtime),
):
    """POST /v2/fleet/mvp-report — report an MVP spawn to the fleet."""
    coordinator = _get_coordinator(runtime)
    coordinator.report_mvp_spawn(mvp_name, map_name, (x, y), hp, hp_max, reported_by=bot_id)
    return {"ok": True, "mvp_name": mvp_name, "map_name": map_name, "shared": True}


@router.get("/mvp-active")
def active_mvps(runtime: RuntimeState = Depends(get_runtime)):
    """GET /v2/fleet/mvp-active — list active MVP spawns."""
    coordinator = _get_coordinator(runtime)
    return {"ok": True, "mvps": coordinator.get_active_mvps()}


# ── Party management ──────────────────────────────────────────────────────


@router.post("/party/create")
def create_party(
    party_id: str, leader_id: str, member_ids: list[str] = [],
    runtime: RuntimeState = Depends(get_runtime),
):
    coordinator = _get_coordinator(runtime)
    ok = coordinator.create_party(party_id, leader_id, member_ids or None)
    return {"ok": ok, "party_id": party_id, "created": ok}


@router.post("/party/disband")
def disband_party(party_id: str, runtime: RuntimeState = Depends(get_runtime)):
    coordinator = _get_coordinator(runtime)
    coordinator.disband_party(party_id)
    return {"ok": True, "party_id": party_id, "disbanded": True}


@router.get("/party/members")
def party_members(party_id: str, runtime: RuntimeState = Depends(get_runtime)):
    coordinator = _get_coordinator(runtime)
    members = coordinator.party_members(party_id)
    return {
        "ok": True,
        "party_id": party_id,
        "members": [m.bot_id for m in members],
        "count": len(members),
    }


@router.post("/party/suggest")
def suggest_party(objective: str = "farming", bot_ids: list[str] = [],
                  runtime: RuntimeState = Depends(get_runtime)):
    coordinator = _get_coordinator(runtime)
    composition = coordinator.suggest_party_composition(objective, available_bots=bot_ids or None)
    return {"ok": True, **composition}


# ── Resource sharing ──────────────────────────────────────────────────────


@router.post("/shared-zeny/add")
def add_shared_zeny(bot_id: str, amount: int, runtime: RuntimeState = Depends(get_runtime)):
    coordinator = _get_coordinator(runtime)
    new_balance = coordinator.add_to_shared_zeny(amount)
    return {"ok": True, "added": amount, "new_balance": new_balance}


@router.post("/shared-zeny/take")
def take_shared_zeny(bot_id: str, amount: int, runtime: RuntimeState = Depends(get_runtime)):
    coordinator = _get_coordinator(runtime)
    taken = coordinator.take_from_shared_zeny(amount)
    return {"ok": True, "taken": taken}


@router.get("/shared-zeny")
def shared_zeny_balance(runtime: RuntimeState = Depends(get_runtime)):
    coordinator = _get_coordinator(runtime)
    return {"ok": True, "balance": coordinator.shared_zeny_balance()}


@router.post("/shared-item/add")
def add_shared_item(item_name: str, quantity: int, runtime: RuntimeState = Depends(get_runtime)):
    coordinator = _get_coordinator(runtime)
    new_qty = coordinator.add_to_shared_inventory(item_name, quantity)
    return {"ok": True, "item": item_name, "quantity": new_qty}


@router.post("/shared-item/take")
def take_shared_item(item_name: str, quantity: int, runtime: RuntimeState = Depends(get_runtime)):
    coordinator = _get_coordinator(runtime)
    taken = coordinator.take_from_shared_inventory(item_name, quantity)
    return {"ok": True, "item": item_name, "taken": taken}
