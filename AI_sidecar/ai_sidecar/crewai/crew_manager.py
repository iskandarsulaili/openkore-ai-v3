from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
import logging
from threading import RLock
from time import perf_counter
from typing import Any

from ai_sidecar.contracts.crewai import (
    CrewAgentDescriptor,
    CrewAgentsResponse,
    CrewCoordinateRequest,
    CrewCoordinateResponse,
    CrewStatusResponse,
    CrewStrategizeRequest,
    CrewStrategizeResponse,
)
from ai_sidecar.crewai.agents import create_agent_by_id
from ai_sidecar.crewai.agents.manager_agent import create_manager_agent
from ai_sidecar.crewai.config import AGENT_OPERATING_MODEL, AGENT_PROFILES, CREW_TOOL_NAMES
from ai_sidecar.crewai.llm_adapter import ProviderBackedCrewLLM
from ai_sidecar.crewai.tasks import build_collaborative_tasks
from ai_sidecar.crewai.tools import CrewToolFacade, build_crewai_tools
from ai_sidecar.planner.schemas import PlanHorizon, PlannerPlanRequest

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CrewManager:
    runtime: Any
    model_router: Any
    enabled: bool = True
    verbose: bool = False
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)
    _active_runs: int = field(default=0, init=False, repr=False)
    _counters: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _crewai_available: bool = field(default=False, init=False, repr=False)
    _init_error: str = field(default="", init=False, repr=False)
    _tool_facade: CrewToolFacade = field(init=False, repr=False)
    _tool_map: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _listener_events: list[dict[str, object]] = field(default_factory=list, init=False, repr=False)
    _run_locks: dict[str, asyncio.Lock] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._counters = {
            "strategize_calls": 0,
            "coordinate_calls": 0,
            "success": 0,
            "failures": 0,
        }
        self._tool_facade = CrewToolFacade(runtime=self.runtime)
        try:
            import crewai  # noqa: F401

            self._crewai_available = True
            self._tool_map = build_crewai_tools(facade=self._tool_facade)
            logger.info("crewai_sdk_initialized", extra={"event": "crewai_sdk_initialized", "tools": sorted(self._tool_map.keys())})
        except Exception as exc:
            self._crewai_available = False
            self._init_error = str(exc)
            self._tool_map = {}
            logger.exception(
                "crewai_sdk_init_failed",
                extra={"event": "crewai_sdk_init_failed", "error": type(exc).__name__},
            )

    def agents(self) -> CrewAgentsResponse:
        operating_map = {item.agent_id: item for item in AGENT_OPERATING_MODEL}
        rows = [
            CrewAgentDescriptor(
                agent_id=item.agent_id,
                role=item.role,
                goal=item.goal,
                tools=list(item.tools),
                operating_model=item.operating_model,
                responsibilities=list(operating_map.get(item.agent_id, None).responsibilities)
                if item.agent_id in operating_map
                else [],
                handoff_inputs=list(operating_map.get(item.agent_id, None).handoff_inputs)
                if item.agent_id in operating_map
                else [],
                handoff_outputs=list(operating_map.get(item.agent_id, None).handoff_outputs)
                if item.agent_id in operating_map
                else [],
                enabled=self.enabled,
            )
            for item in AGENT_PROFILES
        ]
        return CrewAgentsResponse(ok=True, total_agents=len(rows), agents=rows)

    def status(self) -> CrewStatusResponse:
        with self._lock:
            active_runs = self._active_runs
            counters = dict(self._counters)
        data = self.agents()
        return CrewStatusResponse(
            ok=True,
            generated_at=datetime.now(UTC),
            crew_available=self._crewai_available,
            crewai_enabled=self.enabled,
            active_runs=active_runs,
            counters=counters,
            agents=data.agents,
        )

    def execute_tool(self, *, bot_id: str, tool_name: str, arguments: dict[str, object]) -> dict[str, object]:
        return self._tool_facade.execute(bot_id=bot_id, tool_name=tool_name, arguments=arguments)

    async def strategize(self, payload: CrewStrategizeRequest) -> CrewStrategizeResponse:
        started = perf_counter()
        with self._lock:
            self._active_runs += 1
            self._counters["strategize_calls"] += 1
        run_lock = self._get_run_lock(payload.meta.bot_id)
        try:
            async with run_lock:
                agent_outputs, consolidated_output, orchestrator, errors = await self._run_crew_pipeline(
                    bot_id=payload.meta.bot_id,
                    trace_id=payload.meta.trace_id,
                    objective=payload.objective,
                    task_hint="strategic_planning",
                    required_agents=[],
                )
                planner_response = await self.runtime.planner_plan(
                    PlannerPlanRequest(
                        meta=payload.meta,
                        objective=payload.objective,
                        horizon=payload.horizon,
                        force_replan=payload.force_replan,
                        max_steps=payload.max_steps,
                    )
                )
            ok = planner_response.ok and not errors
            message = "strategized" if ok else "strategize_degraded"
            with self._lock:
                self._counters["success" if ok else "failures"] += 1
            return CrewStrategizeResponse(
                ok=ok,
                message=message,
                trace_id=payload.meta.trace_id,
                bot_id=payload.meta.bot_id,
                objective=payload.objective,
                agent_outputs=agent_outputs,
                consolidated_output=consolidated_output,
                planner_response=planner_response,
                orchestrator=orchestrator,
                duration_ms=(perf_counter() - started) * 1000.0,
                errors=errors,
            )
        finally:
            with self._lock:
                self._active_runs = max(0, self._active_runs - 1)

    async def coordinate(self, payload: CrewCoordinateRequest) -> CrewCoordinateResponse:
        started = perf_counter()
        with self._lock:
            self._active_runs += 1
            self._counters["coordinate_calls"] += 1
        run_lock = self._get_run_lock(payload.meta.bot_id)
        try:
            objective = payload.objective or payload.task
            async with run_lock:
                agent_outputs, consolidated_output, orchestrator, errors = await self._run_crew_pipeline(
                    bot_id=payload.meta.bot_id,
                    trace_id=payload.meta.trace_id,
                    objective=objective,
                    task_hint=payload.task,
                    required_agents=list(payload.required_agents),
                )
                planner_response = await self.runtime.planner_plan(
                    PlannerPlanRequest(
                        meta=payload.meta,
                        objective=objective,
                        horizon=PlanHorizon.tactical,
                        force_replan=False,
                        max_steps=12,
                    )
                )
            ok = planner_response.ok and not errors
            message = "coordinated" if ok else "coordinate_degraded"
            with self._lock:
                self._counters["success" if ok else "failures"] += 1
            return CrewCoordinateResponse(
                ok=ok,
                message=message,
                trace_id=payload.meta.trace_id,
                bot_id=payload.meta.bot_id,
                task=payload.task,
                agent_outputs=agent_outputs,
                consolidated_output=consolidated_output,
                planner_response=planner_response,
                orchestrator=orchestrator,
                duration_ms=(perf_counter() - started) * 1000.0,
                errors=errors,
            )
        finally:
            with self._lock:
                self._active_runs = max(0, self._active_runs - 1)

    async def _run_crew_pipeline(
        self,
        *,
        bot_id: str,
        trace_id: str,
        objective: str,
        task_hint: str,
        required_agents: list[str],
    ) -> tuple[list[dict[str, object]], str, dict[str, object], list[str]]:
        if not self.enabled:
            logger.warning(
                "crewai_pipeline_disabled",
                extra={"event": "crewai_pipeline_disabled", "bot_id": bot_id, "trace_id": trace_id},
            )
            return [], "crewai_disabled", {}, ["crewai_disabled"]
        if not self._crewai_available:
            err = self._init_error or "crewai_not_installed"
            logger.error(
                "crewai_pipeline_unavailable",
                extra={"event": "crewai_pipeline_unavailable", "bot_id": bot_id, "trace_id": trace_id, "error": err},
            )
            return [], "crewai_unavailable", {}, [err]

        if self.model_router is None:
            logger.error(
                "crewai_pipeline_missing_model_router",
                extra={"event": "crewai_pipeline_missing_model_router", "bot_id": bot_id, "trace_id": trace_id},
            )
            return [], "crewai_model_router_unavailable", {}, ["crewai_model_router_unavailable"]

        try:
            from crewai import Crew, Process

            llm_workload = self._resolve_llm_workload(task_hint=task_hint)
            llm = ProviderBackedCrewLLM(
                model="router",
                model_router=self.model_router,
                workload=llm_workload,
                timeout_seconds=float(getattr(self.runtime, "planner_service", None) and 45.0 or 45.0),
                max_retries=1,
                bot_id=bot_id,
                trace_id=trace_id,
            )
            allowed = set(required_agents) if required_agents else {item.agent_id for item in AGENT_PROFILES}
            selected_profiles = [item for item in AGENT_PROFILES if item.agent_id in allowed]
            if not selected_profiles:
                selected_profiles = list(AGENT_PROFILES)

            agents_by_id: dict[str, Any] = {}
            for profile in selected_profiles:
                tools = [self._tool_map[name] for name in profile.tools if name in self._tool_map]
                agents_by_id[profile.agent_id] = create_agent_by_id(
                    agent_id=profile.agent_id,
                    llm=llm,
                    tools=tools,
                    verbose=self.verbose,
                )

            manager = create_manager_agent(llm=llm, tools=[], verbose=self.verbose)

            tasks = build_collaborative_tasks(objective=objective, task_hint=task_hint, agents_by_id=agents_by_id)
            listener_events: list[dict[str, object]] = []

            def _before_kickoff(inputs: dict[str, object]) -> dict[str, object]:
                listener_events.append(
                    {
                        "event": "crew_before_kickoff",
                        "timestamp": datetime.now(UTC).isoformat(),
                        "bot_id": bot_id,
                        "trace_id": trace_id,
                    }
                )
                return dict(inputs or {})

            def _after_kickoff(result: Any) -> Any:
                listener_events.append(
                    {
                        "event": "crew_after_kickoff",
                        "timestamp": datetime.now(UTC).isoformat(),
                        "bot_id": bot_id,
                        "trace_id": trace_id,
                    }
                )
                return result

            execution_attempts: list[dict[str, object]] = [
                {
                    "profile": "hierarchical_planning",
                    "process": Process.hierarchical,
                    "planning": True,
                    "with_manager": True,
                },
                {
                    "profile": "sequential_planning",
                    "process": Process.sequential,
                    "planning": True,
                    "with_manager": True,
                },
                {
                    "profile": "sequential_no_planning",
                    "process": Process.sequential,
                    "planning": False,
                    "with_manager": True,
                },
                {
                    "profile": "sequential_no_planning_no_manager",
                    "process": Process.sequential,
                    "planning": False,
                    "with_manager": False,
                },
            ]

            result: Any | None = None
            execution_profile = ""
            execution_process_label = "sequential"
            execution_planning = False
            execution_with_manager = False
            last_async_error: Exception | None = None

            for attempt in execution_attempts:
                attempt_profile = str(attempt["profile"])
                attempt_process = attempt["process"]
                attempt_planning = bool(attempt["planning"])
                attempt_with_manager = bool(attempt["with_manager"])

                try:
                    crew = self._build_crew(
                        Crew=Crew,
                        Process=Process,
                        bot_id=bot_id,
                        agents_by_id=agents_by_id,
                        tasks=tasks,
                        manager=manager,
                        planning_llm=llm,
                        include_manager=attempt_with_manager,
                        before_kickoff=_before_kickoff,
                        after_kickoff=_after_kickoff,
                        process=attempt_process,
                        planning=attempt_planning,
                    )
                except Exception as build_exc:
                    if self._is_async_order_validation_error(build_exc):
                        last_async_error = build_exc
                        logger.warning(
                            "crewai_async_order_retry",
                            extra={
                                "event": "crewai_async_order_retry",
                                "bot_id": bot_id,
                                "trace_id": trace_id,
                                "retry_profile": attempt_profile,
                                "retry_stage": "build",
                                "error": str(build_exc),
                            },
                        )
                        continue
                    raise

                try:
                    result = await crew.akickoff(inputs={"bot_id": bot_id, "objective": objective, "trace_id": trace_id})
                    execution_profile = attempt_profile
                    execution_process_label = "hierarchical" if attempt_process == Process.hierarchical else "sequential"
                    execution_planning = attempt_planning
                    execution_with_manager = attempt_with_manager
                    break
                except Exception as kickoff_exc:
                    if self._is_async_order_validation_error(kickoff_exc):
                        last_async_error = kickoff_exc
                        logger.warning(
                            "crewai_async_order_retry",
                            extra={
                                "event": "crewai_async_order_retry",
                                "bot_id": bot_id,
                                "trace_id": trace_id,
                                "retry_profile": attempt_profile,
                                "retry_stage": "kickoff",
                                "error": str(kickoff_exc),
                            },
                        )
                        continue
                    raise

            if result is None:
                if last_async_error is not None:
                    raise last_async_error
                raise RuntimeError("crewai_no_execution_profile_succeeded")
            consolidated = str(getattr(result, "raw", "") or str(result))
            task_outputs = list(getattr(result, "tasks_output", []) or [])
            rows: list[dict[str, object]] = []
            for item in task_outputs:
                rows.append(
                    {
                        "task_name": str(getattr(item, "name", "") or "task"),
                        "agent": str(getattr(item, "agent", "")),
                        "summary": str(getattr(item, "summary", "") or ""),
                        "raw": str(getattr(item, "raw", "")),
                        "json": dict(getattr(item, "json_dict", {}) or {}),
                    }
                )
            with self._lock:
                self._listener_events.extend(listener_events[-10:])

            if not rows and consolidated:
                rows.append(
                    {
                        "task_name": "crew_result",
                        "agent": "crew_manager",
                        "summary": "Crew execution completed without explicit task outputs",
                        "raw": consolidated,
                        "json": {},
                    }
                )

            orchestrator = {
                "crew": {
                    "process": execution_process_label,
                    "planning": execution_planning,
                    "memory": True,
                    "manager_agent": "manager" if execution_with_manager else "",
                    "execution_profile": execution_profile,
                },
                "flow": {
                    "objective": objective,
                    "task_hint": task_hint,
                    "required_agents": list(required_agents),
                },
                "agents": [profile.agent_id for profile in selected_profiles],
                "tasks": [str(getattr(item, "name", "task")) for item in tasks],
                "llm_workload": llm_workload,
                "listener_events": list(listener_events),
            }

            logger.info(
                "crewai_pipeline_completed",
                extra={
                    "event": "crewai_pipeline_completed",
                    "bot_id": bot_id,
                    "trace_id": trace_id,
                    "task_hint": task_hint,
                    "llm_workload": llm_workload,
                    "agent_count": len(agents_by_id),
                    "task_count": len(tasks),
                    "output_rows": len(rows),
                    "listener_events": len(listener_events),
                },
            )
            return rows, consolidated, orchestrator, []
        except Exception as exc:
            logger.exception(
                "crewai_pipeline_failed",
                extra={
                    "event": "crewai_pipeline_failed",
                    "bot_id": bot_id,
                    "trace_id": trace_id,
                    "task_hint": task_hint,
                },
            )
            return [], "crewai_execution_failed", {}, [f"{type(exc).__name__}:{exc}"]

    def _resolve_llm_workload(self, *, task_hint: str) -> str:
        candidate = (task_hint or "").strip().lower()
        if candidate in {"strategic_planning", "tactical_short_reasoning", "long_reflection", "reflex_explain"}:
            return candidate
        return "tactical_short_reasoning"

    def _get_run_lock(self, bot_id: str) -> asyncio.Lock:
        with self._lock:
            lock = self._run_locks.get(bot_id)
            if lock is None:
                lock = asyncio.Lock()
                self._run_locks[bot_id] = lock
            return lock

    def _is_async_order_validation_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            "async_task_count" in message
            or "async task" in message
            or "asynchronous task" in message
            or "at most one asynchronous task" in message
        )

    def _build_crew(
        self,
        *,
        Crew: Any,
        Process: Any,
        bot_id: str,
        agents_by_id: dict[str, Any],
        tasks: list[Any],
        manager: Any,
        planning_llm: Any,
        include_manager: bool,
        before_kickoff: Any,
        after_kickoff: Any,
        process: Any,
        planning: bool,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "name": f"sidecar-{bot_id}",
            "agents": list(agents_by_id.values()),
            "tasks": tasks,
            "process": process,
            "planning": planning,
            "memory": True,
            "before_kickoff_callbacks": [before_kickoff],
            "after_kickoff_callbacks": [after_kickoff],
            "verbose": self.verbose,
        }
        if planning:
            kwargs["planning_llm"] = planning_llm
        if include_manager:
            kwargs["manager_agent"] = manager
        return Crew(
            **kwargs,
        )
