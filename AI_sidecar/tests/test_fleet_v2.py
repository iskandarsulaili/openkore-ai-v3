from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_sidecar.api.deps import get_runtime
from ai_sidecar.api.routers import fleet_v2
from ai_sidecar.contracts.common import ContractMeta
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


class _FleetRouterRuntime:
    def fleet_sync(self, payload: FleetSyncRequest) -> FleetSyncResponse:
        if payload.meta.bot_id == "bot:central":
            return FleetSyncResponse(
                ok=True,
                mode="central",
                central_available=True,
                doctrine_version="doctrine-v2",
                constraints={"avoid": [], "required": [{"keep": "quest.alpha", "type": "quest_collision"}]},
                blackboard={"doctrine": {"version": "doctrine-v2"}},
                message="synced",
            )
        return FleetSyncResponse(
            ok=True,
            mode="local",
            central_available=False,
            doctrine_version="local",
            constraints={"avoid": [{"conflict_key": "quest.alpha", "type": "quest_collision"}]},
            blackboard={},
            message="central_unavailable:timeout",
        )

    def fleet_constraints(self, *, bot_id: str) -> FleetConstraintResponse:
        if bot_id == "bot:central":
            return FleetConstraintResponse(
                ok=True,
                bot_id=bot_id,
                mode="central",
                doctrine_version="doctrine-v2",
                constraints={
                    "avoid": [],
                    "required": [{"keep": "route.prt_fild08", "type": "territory"}],
                    "sources": ["conflict_resolution"],
                    "policy": {
                        "step_1_detect_conflict": True,
                        "step_2_compare_priority_and_lease": True,
                        "step_3_apply_doctrine": True,
                        "step_4_emit_constraints": True,
                        "step_5_rearbitrate_pending_strategic": True,
                    },
                },
            )
        return FleetConstraintResponse(
            ok=True,
            bot_id=bot_id,
            mode="local",
            doctrine_version="local",
            constraints={
                "avoid": [{"conflict_key": "quest.alpha", "type": "quest_collision"}],
                "required": [],
                "sources": ["local_default"],
                "policy": {
                    "step_1_detect_conflict": True,
                    "step_2_compare_priority_and_lease": True,
                    "step_3_apply_doctrine": True,
                    "step_4_emit_constraints": True,
                    "step_5_rearbitrate_pending_strategic": True,
                },
            },
        )

    def fleet_report_outcome(self, payload: FleetOutcomeReportRequest) -> FleetOutcomeReportResponse:
        if payload.event_type == "quest.step.done":
            return FleetOutcomeReportResponse(
                ok=True,
                accepted=True,
                central_available=True,
                queued_for_retry=False,
                mode="central",
                result={"event_id": 101, "status": "recorded"},
            )
        return FleetOutcomeReportResponse(
            ok=True,
            accepted=True,
            central_available=False,
            queued_for_retry=True,
            mode="local",
            result={},
        )

    def fleet_role(self, *, bot_id: str) -> FleetRoleResponse:
        return FleetRoleResponse(
            ok=True,
            bot_id=bot_id,
            role="support",
            confidence=0.86,
            expires_at=datetime.now(UTC) + timedelta(minutes=3),
            source="fleet-central",
            mode="central",
        )

    def fleet_claim(self, payload: FleetClaimRequestV2) -> FleetClaimResponseV2:
        conflict_key = str(payload.metadata.get("conflict_key") or "")
        conflicts = []
        accepted = True
        reason = "accepted"
        if conflict_key == "quest.alpha":
            accepted = False
            reason = "conflict_detected_local"
            conflicts = [{"type": "quest_collision", "key": "quest.alpha"}]
        return FleetClaimResponseV2(
            ok=True,
            accepted=accepted,
            central_available=False,
            mode="local",
            reason=reason,
            claim={
                "bot_id": payload.meta.bot_id,
                "claim_type": payload.claim_type,
                "map_name": payload.map_name,
                "channel": payload.channel,
            },
            conflicts=conflicts,
        )

    def fleet_blackboard(self, *, bot_id: str) -> FleetBlackboardLocalResponse:
        return FleetBlackboardLocalResponse(
            ok=True,
            bot_id=bot_id,
            mode="local",
            constraints={"avoid": [{"conflict_key": "quest.alpha"}], "required": [], "sources": ["local_default"]},
            blackboard={"doctrine": {"version": "local"}, "patterns": {"quest_swarm": {"slice_count": 2}}},
            local_summary={"queue_depth": 1, "outcome_backlog": 2},
        )


def test_fleet_v2_router_endpoints() -> None:
    runtime = _FleetRouterRuntime()
    app = FastAPI()
    app.include_router(fleet_v2.router)
    app.dependency_overrides[get_runtime] = lambda: runtime

    try:
        with TestClient(app) as client:
            sync_resp = client.post(
                "/v2/fleet/sync",
                json={
                    "meta": {"contract_version": "v1", "source": "pytest", "bot_id": "bot:central", "trace_id": "trace-sync"},
                    "include_blackboard": True,
                },
            )
            assert sync_resp.status_code == 200
            assert sync_resp.json()["mode"] == "central"
            assert sync_resp.json()["doctrine_version"] == "doctrine-v2"

            constraints_resp = client.get("/v2/fleet/constraints", params={"bot_id": "bot:local"})
            assert constraints_resp.status_code == 200
            assert constraints_resp.json()["mode"] == "local"
            assert constraints_resp.json()["constraints"]["avoid"][0]["conflict_key"] == "quest.alpha"

            outcome_ok = client.post(
                "/v2/fleet/report-outcome",
                json={
                    "meta": {
                        "contract_version": "v1",
                        "source": "pytest",
                        "bot_id": "bot:central",
                        "trace_id": "trace-outcome-1",
                    },
                    "event_type": "quest.step.done",
                    "priority_class": 100,
                    "lease_owner": "fleet-central",
                    "conflict_key": "quest.alpha",
                    "payload": {"quest": "alpha"},
                },
            )
            assert outcome_ok.status_code == 200
            assert outcome_ok.json()["central_available"] is True
            assert outcome_ok.json()["queued_for_retry"] is False

            outcome_retry = client.post(
                "/v2/fleet/report-outcome",
                json={
                    "meta": {
                        "contract_version": "v1",
                        "source": "pytest",
                        "bot_id": "bot:local",
                        "trace_id": "trace-outcome-2",
                    },
                    "event_type": "route.failed",
                    "priority_class": 200,
                    "lease_owner": "",
                    "conflict_key": "route.prt_fild08",
                    "payload": {"reason": "stuck"},
                },
            )
            assert outcome_retry.status_code == 200
            assert outcome_retry.json()["central_available"] is False
            assert outcome_retry.json()["queued_for_retry"] is True

            role_resp = client.get("/v2/fleet/role", params={"bot_id": "bot:central"})
            assert role_resp.status_code == 200
            assert role_resp.json()["role"] == "support"
            assert role_resp.json()["mode"] == "central"

            claim_resp = client.post(
                "/v2/fleet/claim",
                json={
                    "meta": {"contract_version": "v1", "source": "pytest", "bot_id": "bot:local", "trace_id": "trace-claim"},
                    "claim_type": "territory",
                    "map_name": "prt_fild08",
                    "channel": "0",
                    "quantity": 1,
                    "ttl_seconds": 120,
                    "priority": 100,
                    "metadata": {"conflict_key": "quest.alpha"},
                },
            )
            assert claim_resp.status_code == 200
            assert claim_resp.json()["accepted"] is False
            assert claim_resp.json()["conflicts"][0]["type"] == "quest_collision"

            blackboard_resp = client.get("/v2/fleet/blackboard", params={"bot_id": "bot:local"})
            assert blackboard_resp.status_code == 200
            assert blackboard_resp.json()["mode"] == "local"
            assert blackboard_resp.json()["local_summary"]["outcome_backlog"] == 2
    finally:
        app.dependency_overrides = {}

