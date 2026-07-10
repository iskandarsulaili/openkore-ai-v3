from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
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
from ai_sidecar.crewai.agents import get_all_profiles, get_profile, best_profile

logger = logging.getLogger(__name__)


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
                except Exception:
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
            if hasattr(payload, "meta"):
                bot = payload.meta.bot_id or ""
                trace = payload.meta.trace_id or ""
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