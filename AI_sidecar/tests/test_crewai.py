from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_sidecar.api.deps import get_runtime
from ai_sidecar.api.routers import crewai_v2
from ai_sidecar.contracts.actions import ActionProposal, ActionStatus
from ai_sidecar.contracts.common import ContractMeta
from ai_sidecar.contracts.crewai import (
    CrewAgentDescriptor,
    CrewAgentsResponse,
    CrewCoordinateRequest,
    CrewCoordinateResponse,
    CrewStatusResponse,
    CrewStrategizeRequest,
    CrewStrategizeResponse,
    CrewToolExecuteResponse,
)
from ai_sidecar.crewai.crew_manager import CrewManager
from ai_sidecar.planner.schemas import PlanHorizon, PlannerPlanRequest, PlannerResponse


@dataclass(slots=True)
class _DummyState:
    bot_id: str

    def model_dump(self, mode: str = "json") -> dict[str, object]:
        del mode
        return {
            "bot_id": self.bot_id,
            "operational": {
                "hp": 100,
                "hp_max": 100,
            },
        }


@dataclass(slots=True)
class _DummyQueue:
    depth: int = 0

    def count(self, _bot_id: str) -> int:
        return self.depth


class _ManagerRuntime:
    def __init__(self) -> None:
        self.action_queue = _DummyQueue(depth=1)
        self.planner_service = None

    def enriched_state(self, *, bot_id: str) -> _DummyState:
        return _DummyState(bot_id=bot_id)

    def memory_context(self, *, bot_id: str, query: str, limit: int) -> list[dict[str, object]]:
        return [{"bot_id": bot_id, "query": query, "limit": limit}]

    def memory_recent_episodes(self, *, bot_id: str, limit: int) -> list[dict[str, object]]:
        return [{"bot_id": bot_id, "episode": "recent", "limit": limit}]

    def memory_stats(self, *, bot_id: str) -> dict[str, int]:
        del bot_id
        return {"episodes": 1, "semantic_records": 1}

    def list_reflex_rules(self, *, bot_id: str) -> list[object]:
        del bot_id
        return []

    def queue_action(self, proposal: ActionProposal, bot_id: str) -> tuple[bool, ActionStatus, str, str]:
        return True, ActionStatus.pending, proposal.action_id, f"queued:{bot_id}"

    async def planner_plan(self, payload: PlannerPlanRequest) -> PlannerResponse:
        return PlannerResponse(
            ok=True,
            message="ok",
            trace_id=payload.meta.trace_id,
            provider="test",
            model="test-model",
            latency_ms=0.1,
            route={"source": "test"},
        )


def test_crew_manager_strategize_disabled_fallback() -> None:
    runtime = _ManagerRuntime()
    manager = CrewManager(runtime=runtime, model_router=None, enabled=False, verbose=False)
    payload = CrewStrategizeRequest(
        meta=ContractMeta(contract_version="v1", source="pytest", bot_id="bot:crew", trace_id="trace-crew-strategize"),
        objective="farm zeny with low risk",
        horizon=PlanHorizon.strategic,
        force_replan=False,
        max_steps=8,
    )

    result = asyncio.run(manager.strategize(payload))

    assert result.ok is False
    assert "crewai_disabled" in result.errors
    assert result.objective == payload.objective
    assert result.planner_response is not None and result.planner_response.ok is True
    assert len(result.agent_outputs) >= 1


def test_crew_manager_coordinate_disabled_and_tool_dispatch() -> None:
    runtime = _ManagerRuntime()
    manager = CrewManager(runtime=runtime, model_router=None, enabled=False, verbose=False)
    payload = CrewCoordinateRequest(
        meta=ContractMeta(contract_version="v1", source="pytest", bot_id="bot:crew", trace_id="trace-crew-coordinate"),
        task="assist party tanking cycle",
        objective="maintain formation",
        target_bots=["bot:a", "bot:b"],
        required_agents=["fleet_liaison", "safety"],
    )

    result = asyncio.run(manager.coordinate(payload))
    assert result.ok is False
    assert "crewai_disabled" in result.errors
    assert result.planner_response is not None and result.planner_response.ok is True

    known_tool = manager.execute_tool(bot_id="bot:crew", tool_name="get_bot_state", arguments={})
    assert known_tool.get("ok") is True
    assert known_tool.get("queue_depth") == 1

    unknown_tool = manager.execute_tool(bot_id="bot:crew", tool_name="missing_tool", arguments={})
    assert unknown_tool.get("ok") is False


class _RouterRuntime:
    async def crewai_strategize(self, payload: CrewStrategizeRequest) -> CrewStrategizeResponse:
        return CrewStrategizeResponse(
            ok=True,
            message="strategized",
            trace_id=payload.meta.trace_id,
            bot_id=payload.meta.bot_id,
            objective=payload.objective,
            planner_response=PlannerResponse(
                ok=True,
                message="ok",
                trace_id=payload.meta.trace_id,
                provider="router",
                model="local",
                latency_ms=1.0,
            ),
            agent_outputs=[{"agent": "Manager", "summary": "ok"}],
            consolidated_output="ok",
        )

    async def crewai_coordinate(self, payload: CrewCoordinateRequest) -> CrewCoordinateResponse:
        return CrewCoordinateResponse(
            ok=True,
            message="coordinated",
            trace_id=payload.meta.trace_id,
            bot_id=payload.meta.bot_id,
            task=payload.task,
            planner_response=PlannerResponse(
                ok=True,
                message="ok",
                trace_id=payload.meta.trace_id,
                provider="router",
                model="local",
                latency_ms=1.0,
            ),
            agent_outputs=[{"agent": "Fleet Liaison", "summary": "ok"}],
            consolidated_output="ok",
        )

    def crewai_agents(self) -> CrewAgentsResponse:
        return CrewAgentsResponse(
            ok=True,
            total_agents=1,
            agents=[
                CrewAgentDescriptor(
                    agent_id="combat",
                    role="Combat Strategist",
                    goal="Handle combat priorities",
                    tools=["get_bot_state"],
                    enabled=True,
                )
            ],
        )

    def crewai_status(self) -> CrewStatusResponse:
        return CrewStatusResponse(
            ok=True,
            crew_available=True,
            crewai_enabled=True,
            active_runs=0,
            counters={"strategize_calls": 1},
            agents=self.crewai_agents().agents,
        )

    def crewai_execute_tool(self, payload) -> CrewToolExecuteResponse:
        return CrewToolExecuteResponse(
            ok=True,
            message="ok",
            trace_id=payload.meta.trace_id,
            bot_id=payload.meta.bot_id,
            tool_name=payload.tool_name,
            result={"ok": True, "echo": payload.arguments},
        )


def test_crewai_v2_router_endpoints() -> None:
    runtime = _RouterRuntime()
    app = FastAPI()
    app.include_router(crewai_v2.router)
    app.dependency_overrides[get_runtime] = lambda: runtime

    with TestClient(app) as client:
        strategize_payload = {
            "meta": {"contract_version": "v1", "source": "pytest", "bot_id": "bot:r1", "trace_id": "trace-r1"},
            "objective": "run strategic planning",
            "horizon": "strategic",
            "force_replan": False,
            "max_steps": 6,
        }
        strategize_resp = client.post("/v2/crewai/strategize", json=strategize_payload)
        assert strategize_resp.status_code == 200
        assert strategize_resp.json()["ok"] is True

        coordinate_payload = {
            "meta": {"contract_version": "v1", "source": "pytest", "bot_id": "bot:r1", "trace_id": "trace-r2"},
            "task": "coordinate map routing",
            "objective": "move safely",
            "target_bots": ["bot:a", "bot:b"],
            "required_agents": ["navigation", "safety"],
            "constraints": [],
            "metadata": {},
        }
        coordinate_resp = client.post("/v2/crewai/coordinate", json=coordinate_payload)
        assert coordinate_resp.status_code == 200
        assert coordinate_resp.json()["ok"] is True

        agents_resp = client.get("/v2/crewai/agents")
        assert agents_resp.status_code == 200
        assert agents_resp.json()["total_agents"] == 1

        status_resp = client.get("/v2/crewai/status")
        assert status_resp.status_code == 200
        assert status_resp.json()["crew_available"] is True

        tool_resp = client.post(
            "/v2/crewai/tools/execute",
            json={
                "meta": {"contract_version": "v1", "source": "pytest", "bot_id": "bot:r1", "trace_id": "trace-r3"},
                "tool_name": "query_memory",
                "arguments": {"query": "spawn map"},
            },
        )
        assert tool_resp.status_code == 200
        assert tool_resp.json()["ok"] is True
        assert tool_resp.json()["result"]["echo"]["query"] == "spawn map"
