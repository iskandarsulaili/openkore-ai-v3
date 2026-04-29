from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_sidecar.api.deps import get_runtime
from ai_sidecar.api.routers import crewai_v2
from ai_sidecar.contracts.autonomy import GoalCategory, GoalDirective, SituationalAssessment
from ai_sidecar.contracts.actions import ActionProposal, ActionStatus
from ai_sidecar.contracts.common import ContractMeta
from ai_sidecar.contracts.crewai import (
    CrewAutonomyDecisionContext,
    CrewAutonomyDecisionOutput,
    CrewAutonomyRefinementRequest,
    CrewAutonomyRefinementResponse,
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
        return True, ActionStatus.queued, proposal.action_id, f"queued:{bot_id}"

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

    def control_plan(self, payload):
        del payload
        return {
            "ok": True,
            "message": "planned",
            "plan": {"plan_id": "ctrl-1"},
        }


class _CaptureCrew:
    def __init__(self, **kwargs) -> None:
        self.kwargs = dict(kwargs)


def _autonomy_assessment(*, bot_id: str = "bot:crew") -> SituationalAssessment:
    return SituationalAssessment(
        bot_id=bot_id,
        tick_id="tick-autonomy",
        map_name="prt_fild08",
        hp_ratio=0.85,
        danger_score=0.10,
        death_risk_score=0.12,
        reconnect_age_s=0.0,
        is_dead=False,
        is_disconnected=False,
        skill_points=2,
        stat_points=0,
        base_level=45,
        job_level=20,
        base_exp_ratio=0.50,
        job_exp_ratio=0.90,
        active_quest_count=1,
        objective_completion_ratio=0.20,
        overweight_ratio=0.12,
        item_count=41,
        zeny=5000,
        vendor_exposure=0,
        replan_reasons=["stale_progress"],
    )


def _autonomy_goal() -> GoalDirective:
    return GoalDirective(
        goal_key=GoalCategory.job_advancement,
        priority_rank=2,
        active=True,
        objective="advance job progression deterministically from prt_fild08",
        rationale="deterministic_priority:job_advancement",
        blockers=[],
        metadata={"horizon": "short_term"},
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
    assert result.agent_outputs == []
    assert result.consolidated_output == "crewai_disabled"
    assert result.planner_response is not None and result.planner_response.ok is True


def test_crew_manager_disabled_warning_throttled(monkeypatch, caplog) -> None:
    runtime = _ManagerRuntime()
    manager = CrewManager(runtime=runtime, model_router=None, enabled=False, verbose=False)
    payload = CrewStrategizeRequest(
        meta=ContractMeta(contract_version="v1", source="pytest", bot_id="bot:crew", trace_id="trace-crew-throttle"),
        objective="farm zeny with low risk",
        horizon=PlanHorizon.strategic,
        force_replan=False,
        max_steps=8,
    )

    times = iter([0.0, 0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 31.0, 31.1])
    state = {"last": 31.1}

    def _fake_perf_counter() -> float:
        try:
            state["last"] = next(times)
        except StopIteration:
            state["last"] = float(state["last"]) + 0.1
        return float(state["last"])

    monkeypatch.setattr("ai_sidecar.crewai.crew_manager.perf_counter", _fake_perf_counter)
    caplog.set_level(logging.DEBUG, logger="ai_sidecar.crewai.crew_manager")

    first = asyncio.run(manager.strategize(payload))
    second = asyncio.run(manager.strategize(payload))
    third = asyncio.run(manager.strategize(payload))

    assert first.ok is False
    assert second.ok is False
    assert third.ok is False

    warning_records = [
        item for item in caplog.records if item.name == "ai_sidecar.crewai.crew_manager" and item.levelno == logging.WARNING
    ]
    warning_events = [getattr(item, "event", "") for item in warning_records]
    assert warning_events.count("crewai_pipeline_disabled") == 2

    debug_events = [
        getattr(item, "event", "")
        for item in caplog.records
        if item.name == "ai_sidecar.crewai.crew_manager" and item.levelno == logging.DEBUG
    ]
    assert "crewai_pipeline_disabled_throttled" in debug_events


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
    assert result.agent_outputs == []
    assert result.consolidated_output == "crewai_disabled"
    assert result.planner_response is not None and result.planner_response.ok is True

    known_tool = manager.execute_tool(bot_id="bot:crew", tool_name="get_bot_state", arguments={})
    assert known_tool.get("ok") is True
    assert known_tool.get("queue_depth") == 1

    unknown_tool = manager.execute_tool(bot_id="bot:crew", tool_name="missing_tool", arguments={})
    assert unknown_tool.get("ok") is False


def test_crew_manager_build_crew_memory_defaults_disabled() -> None:
    runtime = _ManagerRuntime()
    manager = CrewManager(runtime=runtime, model_router=None, enabled=False, verbose=False)

    crew = manager._build_crew(
        Crew=_CaptureCrew,
        Process=object,
        bot_id="bot:crew-memory-default",
        agents_by_id={"agent": object()},
        tasks=[object()],
        manager=object(),
        planning_llm=object(),
        include_manager=False,
        process="sequential",
        planning=False,
    )

    assert crew.kwargs["memory"] is False
    assert "before_kickoff_callbacks" not in crew.kwargs
    assert "after_kickoff_callbacks" not in crew.kwargs


def test_crew_manager_build_crew_memory_honors_explicit_enable() -> None:
    runtime = _ManagerRuntime()
    manager = CrewManager(runtime=runtime, model_router=None, enabled=False, verbose=False, memory_enabled=True)

    crew = manager._build_crew(
        Crew=_CaptureCrew,
        Process=object,
        bot_id="bot:crew-memory-enabled",
        agents_by_id={"agent": object()},
        tasks=[object()],
        manager=object(),
        planning_llm=object(),
        include_manager=False,
        process="sequential",
        planning=False,
    )

    assert crew.kwargs["memory"] is True
    assert "before_kickoff_callbacks" not in crew.kwargs
    assert "after_kickoff_callbacks" not in crew.kwargs


def test_crew_manager_autonomy_task_hint_selects_stage2_roster() -> None:
    runtime = _ManagerRuntime()
    manager = CrewManager(runtime=runtime, model_router=None, enabled=False, verbose=False)

    resolved = manager._resolve_required_agents(task_hint="autonomous_decision_intelligence", required_agents=[])

    assert resolved == [
        "state_assessor",
        "progression_planner",
        "opportunistic_trader",
        "command_emitter",
    ]


def test_crewai_autonomy_contract_roundtrip_validation() -> None:
    assessment = _autonomy_assessment(bot_id="bot:contract")
    selected_goal = _autonomy_goal()
    context = CrewAutonomyDecisionContext(
        horizon="short_term",
        assessment=assessment,
        selected_goal=selected_goal,
        goal_stack=[selected_goal],
        deterministic_priority_order=[
            "survival",
            "job_advancement",
            "opportunistic_upgrades",
            "leveling",
        ],
        replan_reasons=["stale_progress"],
        task_hint="autonomous_decision_intelligence",
        required_agents=[
            "state_assessor",
            "progression_planner",
            "opportunistic_trader",
            "command_emitter",
        ],
    )
    request = CrewAutonomyRefinementRequest(
        meta=ContractMeta(contract_version="v1", source="pytest", bot_id="bot:contract", trace_id="trace-contract"),
        task_hint="autonomous_decision_intelligence",
        required_agents=list(context.required_agents),
        decision_context=context,
        objective=selected_goal.objective,
    )

    payload = request.model_dump(mode="json")
    rebuilt = CrewAutonomyRefinementRequest.model_validate(payload)

    assert rebuilt.task_hint == "autonomous_decision_intelligence"
    assert rebuilt.decision_context.selected_goal.goal_key == GoalCategory.job_advancement
    assert rebuilt.decision_context.required_agents == [
        "state_assessor",
        "progression_planner",
        "opportunistic_trader",
        "command_emitter",
    ]


def test_crew_manager_autonomy_refinement_integration_response_wiring() -> None:
    runtime = _ManagerRuntime()
    manager = CrewManager(runtime=runtime, model_router=None, enabled=True, verbose=False)
    context = CrewAutonomyDecisionContext(
        horizon="short_term",
        assessment=_autonomy_assessment(bot_id="bot:crew"),
        selected_goal=_autonomy_goal(),
        goal_stack=[_autonomy_goal()],
        deterministic_priority_order=["survival", "job_advancement", "opportunistic_upgrades", "leveling"],
        replan_reasons=["stale_progress"],
        task_hint="autonomous_decision_intelligence",
        required_agents=["state_assessor", "progression_planner", "opportunistic_trader", "command_emitter"],
    )
    payload = CrewAutonomyRefinementRequest(
        meta=ContractMeta(contract_version="v1", source="pytest", bot_id="bot:crew", trace_id="trace-crew-autonomy"),
        task_hint="autonomous_decision_intelligence",
        required_agents=list(context.required_agents),
        decision_context=context,
        objective=context.selected_goal.objective,
    )

    async def _fake_pipeline(
        self,
        *,
        bot_id: str,
        trace_id: str,
        objective: str,
        task_hint: str,
        required_agents: list[str],
        decision_context: CrewAutonomyDecisionContext | None,
    ):
        del self
        assert bot_id == "bot:crew"
        assert trace_id == "trace-crew-autonomy"
        assert objective == payload.objective
        assert task_hint == "autonomous_decision_intelligence"
        assert decision_context is not None
        assert required_agents == ["state_assessor", "progression_planner", "opportunistic_trader", "command_emitter"]
        return (
            [{"agent": "state_assessor", "summary": "ok", "json": {}}],
            "autonomy refined",
            {
                "flow": {
                    "task_hint": "autonomous_decision_intelligence",
                    "required_agents": [
                        "state_assessor",
                        "progression_planner",
                        "opportunistic_trader",
                        "command_emitter",
                    ],
                }
            },
            [],
            CrewAutonomyDecisionOutput(
                selected_goal_key="job_advancement",
                refined_objective="refine objective safely",
                situational_report="stable posture",
                execution_translation=["cmd1", "cmd2"],
                rationale="stage2 refinement",
                confidence=0.82,
                annotations={"source": "pytest"},
            ),
        )

    original = CrewManager._run_crew_pipeline
    try:
        CrewManager._run_crew_pipeline = _fake_pipeline  # type: ignore[assignment]
        result = asyncio.run(manager.autonomy_refine_decision(payload))
    finally:
        CrewManager._run_crew_pipeline = original  # type: ignore[assignment]

    assert result.ok is True
    assert result.task_hint == "autonomous_decision_intelligence"
    assert result.required_agents == [
        "state_assessor",
        "progression_planner",
        "opportunistic_trader",
        "command_emitter",
    ]
    assert result.decision_output is not None
    assert result.decision_output.refined_objective == "refine objective safely"
    assert result.decision_output.execution_translation == ["cmd1", "cmd2"]


def test_crew_manager_derives_stage4_execution_translation_from_context_direct_config_macro() -> None:
    runtime = _ManagerRuntime()
    manager = CrewManager(runtime=runtime, model_router=None, enabled=False, verbose=False)

    def _context_with_hint(hint: dict[str, object]) -> CrewAutonomyDecisionContext:
        selected_goal = GoalDirective(
            goal_key=GoalCategory.opportunistic_upgrades,
            priority_rank=3,
            active=True,
            objective="execute curated opportunistic upgrade",
            rationale="deterministic_priority:opportunistic_upgrades;knowledge_backed:stage4_actionable",
            blockers=[],
            metadata={"execution_hints": [hint]},
        )
        return CrewAutonomyDecisionContext(
            horizon="short_term",
            assessment=_autonomy_assessment(bot_id="bot:crew-stage4"),
            selected_goal=selected_goal,
            goal_stack=[selected_goal],
            deterministic_priority_order=[
                "survival",
                "job_advancement",
                "opportunistic_upgrades",
                "leveling",
            ],
            replan_reasons=["stale_progress"],
            task_hint="autonomous_decision_intelligence",
            required_agents=["state_assessor", "progression_planner", "opportunistic_trader", "command_emitter"],
        )

    direct_context = _context_with_hint(
        {
            "execution_mode": "direct",
            "tool": "propose_actions",
            "intents": [{"kind": "command", "command": "ai auto"}],
        }
    )
    config_context = _context_with_hint(
        {
            "execution_mode": "config",
            "tool": "plan_control_change",
            "request": {"target_path": "control/config.txt"},
        }
    )
    macro_context = _context_with_hint(
        {
            "execution_mode": "macro",
            "tool": "publish_macro",
            "macro_bundle": {"macros": [{"name": "crew_stage4_upgrade_posture", "lines": ["do ai auto"]}]},
        }
    )

    assert manager._derive_execution_translation_from_context(direct_context) == ["propose_actions:ai auto"]
    assert manager._derive_execution_translation_from_context(config_context) == ["plan_control_change:control/config.txt"]
    assert manager._derive_execution_translation_from_context(macro_context) == ["publish_macro:crew_stage4_upgrade_posture"]


def test_crew_manager_strategic_task_hint_resolves_specialized_roster() -> None:
    runtime = _ManagerRuntime()
    manager = CrewManager(runtime=runtime, model_router=None, enabled=False, verbose=False)

    resolved = manager._resolve_required_agents(task_hint="strategic_planning", required_agents=[])

    assert resolved == [
        "strategic_planner",
        "resource_manager",
        "social_coordinator",
        "tactical_commander",
    ]


def test_task_factory_builds_structured_capability_bounded_contracts() -> None:
    from ai_sidecar.crewai.tasks.task_factory import build_collaborative_tasks

    class _TaskStub:
        def __init__(self, **kwargs) -> None:
            self.name = kwargs.get("name")
            self.description = kwargs.get("description")
            self.expected_output = kwargs.get("expected_output")
            self.agent = kwargs.get("agent")
            self.context = kwargs.get("context")
            self.async_execution = kwargs.get("async_execution")

    import sys
    import types

    module_name = "crewai"
    original = sys.modules.get(module_name)
    sys.modules[module_name] = types.SimpleNamespace(Task=_TaskStub)
    try:
        tasks = build_collaborative_tasks(
            objective="refine deterministic objective",
            task_hint="autonomous_decision_intelligence",
            agents_by_id={
                "state_assessor": object(),
                "progression_planner": object(),
            },
        )
    finally:
        if original is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original

    assert tasks
    first = tasks[0]
    assert "strict JSON" in str(first.description)
    assert "capability_plan" in str(first.description)
    assert "mode(direct|config|macro|unsupported)" in str(first.description)
    assert "Structured JSON contract" in str(first.expected_output)


def test_crewai_tool_dispatch_propose_actions_rejects_unsupported_direct_roots() -> None:
    runtime = _ManagerRuntime()
    manager = CrewManager(runtime=runtime, model_router=None, enabled=False, verbose=False)

    result = manager.execute_tool(
        bot_id="bot:crew",
        tool_name="propose_actions",
        arguments={
            "intents": [
                {"kind": "command", "command": "dropall"},
                {"kind": "command", "command": "move prt_fild08"},
            ]
        },
    )

    assert result["ok"] is True
    assert result["accepted"] == 1
    assert result["rejected"] == 1
    reasons = [row.get("reason") for row in result["results"]]
    assert "unsupported_direct_command_root" in reasons
    assert result.get("execution_mode") == "direct"
    assert result.get("tool") == "propose_actions"


def test_crewai_tool_dispatch_plan_control_change_includes_capability_metadata() -> None:
    runtime = _ManagerRuntime()
    manager = CrewManager(runtime=runtime, model_router=None, enabled=False, verbose=False)

    result = manager.execute_tool(
        bot_id="bot:crew",
        tool_name="plan_control_change",
        arguments={
            "request": {
                "artifact_type": "config",
                "name": "config.txt",
                "target_path": "control/config.txt",
                "desired": {"aiSidecar_enable": "1"},
            }
        },
    )

    assert result["ok"] is True
    assert result.get("execution_mode") == "config"
    assert result.get("tool") == "plan_control_change"
    capability = result.get("capability")
    assert isinstance(capability, dict)
    assert capability.get("config", {}).get("tool") == "plan_control_change"


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
