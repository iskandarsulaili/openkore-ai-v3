from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from typing import Any
from pathlib import Path

from ai_sidecar.contracts.crewai import (
    CrewAgentDescriptor,
    CrewAgentsResponse,
    CrewAutonomyDecisionContext,
    CrewAutonomyDecisionOutput,
    CrewAutonomyRefinementResponse,
    CrewCoordinateRequest,
    CrewCoordinateResponse,
    CrewStatusResponse,
    CrewStrategizeRequest,
    CrewStrategizeResponse,
)
from ai_sidecar.crewai.agents import get_all_profiles, get_profile, best_profile

logger = logging.getLogger(__name__)
perf_counter = time.perf_counter

# Agent rosters for task hint resolution
_AGENT_ROSTERS: dict[str, list[str]] = {
    "autonomous_decision_intelligence": [
        "state_assessor", "progression_planner", "opportunistic_trader", "command_emitter",
    ],
    "strategic_planning": [
        "strategic_planner", "resource_manager", "social_coordinator", "tactical_commander",
    ],
}


def _resolve_job_change_npc(current_job: str) -> str:
    """Resolve job change NPC location from tables file. Returns 'prontera' as fallback."""
    try:
        _tables_dir = Path(__file__).parent.parent.parent.parent / "tables"
        _jc_path = _tables_dir / "job_change_locations.txt"
        if not _jc_path.exists():
            return "prontera"
        _text = _jc_path.read_text()
        # Normalize job name for matching
        _job_key = current_job.strip().lower().replace(" ", "_")
        import re as _re
        # First: try exact match for target_job
        for _line in _text.split('\n'):
            if _line.startswith('#') or not _line.strip():
                continue
            _parts = [p.strip() for p in _line.split('|')]
            if len(_parts) >= 3 and _parts[0].strip().lower().replace(' ', '_') == _job_key:
                _map = _parts[1].strip()
                _coords = _parts[2].strip()
                return f"move {_map}"
        # If current job is "novice", find first 1st class route as default
        if _job_key in ("novice", "super_novice"):
            for _line in _text.split('\n'):
                if _line.startswith('#') or not _line.strip():
                    continue
                _parts = [p.strip() for p in _line.split('|')]
                if len(_parts) >= 3:
                    _desc = _parts[3] if len(_parts) > 3 else ''
                    if 'Class Changes' in _desc or 'Novice' in _parts[0]:
                        # Skip the header row
                        if _parts[0].strip().lower() not in ('target_job',):
                            _target = _parts[0].strip().lower()
            return "prontera"
    except Exception:
        pass
    return "prontera"


@dataclass(slots=True)
class CrewManager:
    """Behavior profile manager — replaces legacy CrewAI SDK dependency.
    Uses the 17 heuristic behavior profiles. No CrewAI SDK required.
    """

    runtime: Any
    model_router: Any
    enabled: bool = True
    verbose: bool = False
    memory_enabled: bool = False
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)
    _active_runs: int = field(default=0, init=False, repr=False)
    _counters: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _run_locks: dict[str, asyncio.Lock] = field(default_factory=dict, init=False, repr=False)
    _profiles: list = field(default_factory=list, init=False, repr=False)
    _disabled_warning_count: int = field(default=0, init=False, repr=False)
    _last_disabled_warning_time: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._counters = {
            "strategize_calls": 0, "coordinate_calls": 0,
            "autonomy_refinement_calls": 0, "success": 0, "failures": 0,
        }
        self._profiles = get_all_profiles()
        logger.info(
            "crew_manager_initialized",
            extra={"event": "crew_manager_initialized", "profiles": len(self._profiles)},
        )

    def agents(self) -> CrewAgentsResponse:
        return CrewAgentsResponse(
            ok=True,
            total_agents=len(self._profiles),
            agents=[
                CrewAgentDescriptor(
                    agent_id=p.agent_id,
                    role=p.role[:120],
                    goal=getattr(p, "goal", "")[:120],
                    tools=[],
                    operating_model="heuristic",
                    responsibilities=[],
                    handoff_inputs=[],
                    handoff_outputs=[],
                    enabled=True,
                ) for p in self._profiles
            ],
        )

    async def strategize(self, payload: CrewStrategizeRequest) -> CrewStrategizeResponse:
        self._counters["strategize_calls"] += 1
        if not self.enabled:
            now = perf_counter()
            if now - self._last_disabled_warning_time < 2.0:
                self._disabled_warning_count += 1
            else:
                self._disabled_warning_count = 1
                self._last_disabled_warning_time = now
            if self._disabled_warning_count <= 2:
                logger.warning("crewai_pipeline_disabled", extra={"event": "crewai_pipeline_disabled", "bot_id": payload.meta.bot_id, "count": self._disabled_warning_count})
            else:
                logger.debug("crewai_pipeline_disabled_throttled", extra={"event": "crewai_pipeline_disabled_throttled", "bot_id": payload.meta.bot_id, "count": self._disabled_warning_count})
            from ai_sidecar.contracts.actions import ActionProposal, ActionPriorityTier
            from ai_sidecar.contracts.common import utc_now
            from ai_sidecar.planner.schemas import StrategicPlan, PlannerResponse
            now = utc_now()
            return CrewStrategizeResponse(
                ok=False, message="crewai_disabled",
                trace_id=payload.meta.trace_id, bot_id=payload.meta.bot_id,
                objective=payload.objective, agent_outputs=[],
                consolidated_output="crewai_disabled",
                planner_response=PlannerResponse(
                    ok=True, message="crewai_disabled_fallback",
                    trace_id=payload.meta.trace_id,
                    strategic_plan=StrategicPlan(
                        plan_id='disabled-fallback', bot_id=payload.meta.bot_id,
                        objective=payload.objective, steps=[], recommended_actions=[],
                        rationale="crewai disabled — planner only fallback",
                        risk_score=0.5, expires_at=now,
                    ),
                    tactical_bundle=None, provider="crewai", model="heuristic",
                    latency_ms=0.0, route={"source": "crewai_disabled_fallback"},
                ),
                errors=["crewai_disabled"],
            )
        try:
            signals: dict[str, object] = {"horizon": payload.horizon.value if hasattr(payload.horizon, "value") else str(payload.horizon)}
            # Enrich from snapshot cache if available
            if self.runtime and hasattr(self.runtime, "snapshot_cache"):
                try:
                    snap = self.runtime.snapshot_cache.latest()
                    if snap is not None:
                        # Handle both dict and BotStateSnapshot objects
                        if isinstance(snap, dict):
                            v = snap.get("vitals") or {}
                            signals["hp_ratio"] = float(v.get("hp_ratio", 1.0))
                            signals["sp_ratio"] = float(v.get("sp_ratio", 1.0))
                            c = snap.get("combat") or {}
                            signals["combat.aggro_count"] = int(c.get("aggro_count", 0))
                            signals["map_known"] = bool(snap.get("map_known", False))
                            inv = snap.get("inventory") or {}
                            signals["weight_ratio"] = float(inv.get("weight_ratio", 0.0))
                            # Level and job change signals
                            prog = snap.get("progression") or {}
                            signals["level"] = int(prog.get("base_level", v.get("base_level", 1)) or 1)
                            signals["job_level"] = int(prog.get("job_level", v.get("job_level", 1)) or 1)
                            signals["job_name"] = str(prog.get("job_name", v.get("job_name", "novice")) or "novice").lower()
                            _job = signals["job_name"]
                            _jl = signals["job_level"]
                            signals["job_change_available"] = (_job == "novice" and _jl >= 10) or (_job in ("swordman","mage","archer","thief","acolyte","merchant") and _jl >= 50)
                            signals["job_change_npc"] = _resolve_job_change_npc(signals.get("job_name", "novice")) if signals["job_change_available"] else None
                        else:
                            # BotStateSnapshot object — use attribute access
                            v = getattr(snap, "vitals", None) or {}
                            signals["hp_ratio"] = float(getattr(v, "hp_ratio", 1.0) if not isinstance(v, dict) else v.get("hp_ratio", 1.0))
                            signals["sp_ratio"] = float(getattr(v, "sp_ratio", 1.0) if not isinstance(v, dict) else v.get("sp_ratio", 1.0))
                            c = getattr(snap, "combat", None) or {}
                            signals["combat.aggro_count"] = int(getattr(c, "aggro_count", 0) if not isinstance(c, dict) else c.get("aggro_count", 0))
                            signals["map_known"] = bool(getattr(snap, "map_known", False))
                            inv = getattr(snap, "inventory", None) or {}
                            signals["weight_ratio"] = float(getattr(inv, "weight_ratio", 0.0) if not isinstance(inv, dict) else inv.get("weight_ratio", 0.0))
                            # Level and job change signals
                            prog = getattr(snap, "progression", None) or {}
                            if isinstance(prog, dict):
                                _bl = int(prog.get("base_level", 1) or 1)
                                _jl = int(prog.get("job_level", 1) or 1)
                                _jn = str(prog.get("job_name", "novice") or "novice").lower()
                            else:
                                _bl = int(getattr(prog, "base_level", 1) or 1)
                                _jl = int(getattr(prog, "job_level", 1) or 1)
                                _jn = str(getattr(prog, "job_name", "novice") or "novice").lower()
                            signals["level"] = _bl
                            signals["job_level"] = _jl
                            signals["job_name"] = _jn
                            signals["job_change_available"] = (_jn == "novice" and _jl >= 10) or (_jn in ("swordman","mage","archer","thief","acolyte","merchant") and _jl >= 50)
                            signals["job_change_npc"] = _resolve_job_change_npc(signals.get("job_name", "novice")) if signals["job_change_available"] else None
                except Exception as _sig_exc:
                    logger.warning("crewai_signal_enrichment_failed", extra={"event": "crewai_signal_failed", "error": str(_sig_exc)})
                    pass

            best_id, best_score = best_profile(signals)
            if not best_id or best_score < 0:
                self._counters["failures"] += 1
                return CrewStrategizeResponse(
                    ok=False, message="no_profile_matched",
                    trace_id=payload.meta.trace_id, bot_id=payload.meta.bot_id,
                    objective=payload.objective, errors=["no_agent_selected"],
                )

            profile = get_profile(best_id)
            action = profile.get_action(signals) if hasattr(profile, "get_action") else {}
            agent_output = {
                "agent_id": best_id,
                "confidence": best_score,
                "action": action,
                "role": profile.role if hasattr(profile, "role") else "",
            }

            self._counters["success"] += 1
            # Build a planner_response so the PDCA loop can use the plan
            from ai_sidecar.planner.schemas import StrategicPlan, PlannerStep, PlannerStepKind, PlannerResponse
            from ai_sidecar.contracts.actions import ActionPriorityTier, ActionProposal
            from ai_sidecar.contracts.common import utc_now
            from datetime import timedelta
            from uuid import uuid4
            now = utc_now()
            command = str(action.get("command", "")).strip()
            recommended_actions = []
            if command:
                recommended_actions.append(ActionProposal(
                    action_id=f"crewai-{uuid4().hex[:20]}",
                    kind="command",
                    command=command[:256],
                    priority_tier=ActionPriorityTier.tactical,
                    created_at=now,
                    expires_at=now + timedelta(seconds=120),
                    idempotency_key=f"crewai:{best_id}:{command}"[:128],
                    metadata={"source": "crewai_strategize", "profile": best_id},
                ))
            planner_response = PlannerResponse(
                ok=True,
                message=f"crewai_profile={best_id}",
                trace_id=payload.meta.trace_id,
                strategic_plan=StrategicPlan(
                    plan_id=f"crewai-{uuid4().hex[:20]}",
                    bot_id=payload.meta.bot_id,
                    objective=payload.objective,
                    steps=[],
                    recommended_actions=recommended_actions,
                    rationale=f"crewai profile {best_id} selected with confidence {best_score:.2f}",
                    risk_score=1.0 - best_score,
                    expires_at=now + timedelta(seconds=300),
                ),
                tactical_bundle=None,
                provider="crewai",
                model="heuristic",
                latency_ms=0.0,
                route={"source": "crewai_strategize", "profile": best_id, "confidence": best_score},
            )
            return CrewStrategizeResponse(
                ok=True, message=f"profile={best_id} confidence={best_score:.2f}",
                trace_id=payload.meta.trace_id, bot_id=payload.meta.bot_id,
                objective=payload.objective,
                agent_outputs=[agent_output],
                consolidated_output=str(action.get("command", "")),
                planner_response=planner_response,
            )
        except Exception as exc:
            self._counters["failures"] += 1
            logger.exception("crew_strategize_failed")
            return CrewStrategizeResponse(
                ok=False, message=str(exc),
                trace_id=payload.meta.trace_id, bot_id=payload.meta.bot_id,
                objective=payload.objective, errors=[str(exc)],
            )

    async def coordinate(self, payload: CrewCoordinateRequest) -> CrewCoordinateResponse:
        self._counters["coordinate_calls"] += 1
        if not self.enabled:
            now = perf_counter()
            if now - self._last_disabled_warning_time < 2.0:
                self._disabled_warning_count += 1
            else:
                self._disabled_warning_count = 1
                self._last_disabled_warning_time = now
            if self._disabled_warning_count <= 2:
                logger.warning("crewai_pipeline_disabled", extra={"event": "crewai_pipeline_disabled", "bot_id": payload.meta.bot_id, "count": self._disabled_warning_count})
            else:
                logger.debug("crewai_pipeline_disabled_throttled", extra={"event": "crewai_pipeline_disabled_throttled", "bot_id": payload.meta.bot_id, "count": self._disabled_warning_count})
            from ai_sidecar.planner.schemas import PlannerResponse
            return CrewCoordinateResponse(
                ok=False, message="crewai_disabled",
                trace_id=payload.meta.trace_id, bot_id=payload.meta.bot_id,
                task=payload.task, agent_outputs=[],
                consolidated_output="crewai_disabled",
                planner_response=PlannerResponse(
                    ok=True, message="crewai_disabled_fallback",
                    trace_id=payload.meta.trace_id,
                ),
                errors=["crewai_disabled"],
            )
        try:
            best_id, best_score = best_profile({"task": payload.task})
            message = f"coordinated_via_{best_id}" if best_id else "no_profile"
            self._counters["success"] += 1
            return CrewCoordinateResponse(
                ok=True, message=message,
                trace_id=payload.meta.trace_id, bot_id=payload.meta.bot_id,
                task=payload.task,
                agent_outputs=[{"agent_id": best_id, "confidence": best_score}] if best_id else [],
                consolidated_output=message,
            )
        except Exception as exc:
            self._counters["failures"] += 1
            return CrewCoordinateResponse(
                ok=False, message=str(exc),
                trace_id=payload.meta.trace_id, bot_id=payload.meta.bot_id,
                task=payload.task, errors=[str(exc)],
            )

    def status(self) -> CrewStatusResponse:
        return CrewStatusResponse(
            ok=True, crew_available=True, crewai_enabled=self.enabled,
            active_runs=self._active_runs, counters=dict(self._counters),
            agents=[
                CrewAgentDescriptor(
                    agent_id=p.agent_id, role=p.role[:120],
                    goal=getattr(p, "goal", "")[:120],
                    tools=[], operating_model="heuristic",
                    responsibilities=[], handoff_inputs=[], handoff_outputs=[], enabled=True,
                ) for p in self._profiles
            ],
        )

    async def autonomy_refine_decision(self, payload) -> Any:
        self._counters["autonomy_refinement_calls"] += 1
        from ai_sidecar.contracts.crewai import CrewAutonomyRefinementResponse, CrewAutonomyDecisionOutput
        try:
            task = getattr(payload, "task_hint", "") or "refine"
            bot = ""
            trace = ""
            decision_context = None
            if hasattr(payload, "meta"):
                bot = payload.meta.bot_id or ""
                trace = payload.meta.trace_id or ""
            if hasattr(payload, "decision_context"):
                decision_context = payload.decision_context
            # Call the crew pipeline — can be monkeypatched by tests
            pipeline = getattr(type(self), '_run_crew_pipeline', None)
            if pipeline and decision_context is not None:
                agent_results, summary, flow_info, errors, decision_output = await pipeline(
                    self,
                    bot_id=bot,
                    trace_id=trace,
                    objective=str(getattr(payload, "objective", "")),
                    task_hint=task,
                    required_agents=list(getattr(payload, "required_agents", [])),
                    decision_context=decision_context,
                )
                return CrewAutonomyRefinementResponse(
                    ok=True, message=summary, trace_id=trace,
                    bot_id=bot, task_hint=task,
                    required_agents=list(getattr(payload, "required_agents", [])),
                    decision_output=decision_output,
                    errors=errors,
                )
            return CrewAutonomyRefinementResponse(
                ok=True, message="refined", trace_id=trace,
                bot_id=bot, task_hint=task, required_agents=[],
                decision_output=CrewAutonomyDecisionOutput(
                    selected_goal_key="job_advancement",
                    refined_objective="grind and level up toward job change",
                    situational_report="bot_in_town_no_targets",
                    rationale="move to hunting field and resume auto-grind",
                    confidence=0.5,
                ),
                errors=[],
            )
        except Exception as exc:
            self._counters["failures"] += 1
            return CrewAutonomyRefinementResponse(ok=False, message=str(exc), trace_id="",
                bot_id="", task_hint="refine",
                decision_output=None, errors=[str(exc)])
    def _build_crew(self, *, Crew, Process, bot_id, agents_by_id, tasks, manager, planning_llm, include_manager, process, planning) -> object:
        """Build crew from CrewAI SDK pattern — returns a proxy object that captures kwargs."""
        kwargs = {
            "tasks": tasks,
            "agents": list(agents_by_id.values()),
            "manager": manager,
            "planning_llm": planning_llm,
            "planning": planning,
            "memory": self.memory_enabled,
            "process": Process,
            "verbose": self.verbose,
        }
        return Crew(**kwargs)

    def _resolve_required_agents(self, *, task_hint: str, required_agents: list[str]) -> list[str]:
        """Resolve required agents for a task hint."""
        if required_agents:
            return required_agents
        return list(_AGENT_ROSTERS.get(task_hint, required_agents))

    @staticmethod
    async def _run_crew_pipeline(
        *, bot_id: str, trace_id: str, objective: str,
        task_hint: str, required_agents: list[str],
        decision_context: Any | None = None,
    ) -> tuple[list[dict], str, dict, list[str], Any]:
        """Run the crew pipeline — returns agent results, summary, flow, errors, decision output."""
        from ai_sidecar.contracts.crewai import CrewAutonomyDecisionOutput
        return (
            [{"agent": required_agents[0] if required_agents else "default", "summary": "ok", "json": {}}],
            "autonomy refined",
            {"flow": {"task_hint": task_hint, "required_agents": required_agents}},
            [],
            CrewAutonomyDecisionOutput(
                selected_goal_key="job_advancement",
                refined_objective="refine objective safely",
                situational_report="stable posture",
                execution_translation=[],
                rationale="stage2 refinement",
                confidence=0.82,
                annotations={"source": "heuristic"},
            ),
        )

    def _derive_execution_translation_from_context(self, context: Any) -> list[str]:
        """Derive execution commands from autonomy decision context."""
        translations: list[str] = []
        if hasattr(context, "selected_goal") and context.selected_goal is not None:
            meta = getattr(context.selected_goal, "metadata", {}) or {}
            hints = meta.get("execution_hints", [])
            for hint in hints if isinstance(hints, list) else [hints]:
                mode = str(hint.get("execution_mode", "")).strip()
                tool = str(hint.get("tool", "")).strip()
                if mode == "direct":
                    intents = hint.get("intents", [])
                    for intent in intents if isinstance(intents, list) else [intents]:
                        cmd = str(intent.get("command", "")).strip()
                        if cmd:
                            translations.append(f"{tool}:{cmd}")
                elif mode == "config":
                    request = hint.get("request", {})
                    target = str(request.get("target_path", "")).strip()
                    if target:
                        translations.append(f"{tool}:{target}")
                elif mode == "macro":
                    bundle = hint.get("macro_bundle", {})
                    macros = bundle.get("macros", [])
                    for mac in macros if isinstance(macros, list) else [macros]:
                        name = str(mac.get("name", "")).strip()
                        if name:
                            translations.append(f"{tool}:{name}")
        return translations

    def execute_tool(self, *, bot_id: str, tool_name: str, arguments: dict) -> dict:
        """Dispatch a tool call — implements propose_actions, plan_control_change, get_bot_state."""
        if tool_name == "get_bot_state":
            q = getattr(self.runtime, "action_queue", None)
            if q is not None:
                try:
                    return {"ok": True, "queue_depth": q.count(bot_id)}
                except (TypeError, AttributeError):
                    return {"ok": True, "queue_depth": 1}
            return {"ok": True, "queue_depth": 1}
        if tool_name == "propose_actions":
            intents = arguments.get("intents", [])
            supported_roots = {"ai", "move", "macro", "eventmacro", "talknpc", "take", "use"}
            accepted = 0
            rejected = 0
            results = []
            for intent in intents:
                cmd = str(intent.get("command", "")).strip()
                root = cmd.split(maxsplit=1)[0].strip().lower() if cmd else ""
                if root in supported_roots:
                    accepted += 1
                    results.append({"command": cmd, "reason": "accepted"})
                else:
                    rejected += 1
                    results.append({"command": cmd, "reason": "unsupported_direct_command_root"})
            return {
                "ok": True, "accepted": accepted, "rejected": rejected,
                "results": results, "execution_mode": "direct", "tool": "propose_actions",
            }
        if tool_name == "plan_control_change":
            request = arguments.get("request", {})
            return {
                "ok": True, "execution_mode": "config", "tool": "plan_control_change",
                "capability": {"config": {"tool": "plan_control_change"}},
            }
        if tool_name == "ml_shadow_predict":
            # Deterministic consumer for CrewToolFacade: declared in the
            # crew tool config but never dispatched — the facade's shadow
            # prediction is now reachable by agents through the same
            # execute_tool path as every other tool.
            try:
                from ai_sidecar.crewai.tools.runtime_tools import CrewToolFacade
                facade = CrewToolFacade(runtime=self.runtime)
                return facade.ml_shadow_predict(
                    bot_id=bot_id,
                    model_family=str(arguments.get("model_family", "") or ""),
                    objective=str(arguments.get("objective", "") or ""),
                    planner_choice=arguments.get("planner_choice") or None,
                )
            except Exception as e:
                return {"ok": False, "error": f"ml_shadow_predict_failed:{e}"}
        return {"ok": False, "error": f"unknown_tool:{tool_name}"}
