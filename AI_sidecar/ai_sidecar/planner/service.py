from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from typing import Any

from ai_sidecar.planner.context_assembler import PlannerContextAssembler
from ai_sidecar.planner.intent_synthesizer import IntentSynthesizer
from ai_sidecar.planner.macro_synthesizer import MacroSynthesizer
from ai_sidecar.planner.plan_generator import PlanGenerator
from ai_sidecar.planner.reflection_writer import ReflectionWriter
from ai_sidecar.planner.schemas import (
    EscalationNotice,
    LearningLabel,
    MacroSynthesisProposal,
    MemoryWriteback,
    PlanHorizon,
    PlannerExplainRequest,
    PlannerMacroPromoteRequest,
    PlannerPlanRequest,
    PlannerResponse,
    PlannerStatusResponse,
)
from ai_sidecar.planner.self_critic import SelfCritic


@dataclass(slots=True)
class _PlannerBotState:
    current_objective: str | None = None
    last_plan_id: str | None = None
    last_provider: str | None = None
    last_model: str | None = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_rationale: str = ""


@dataclass(slots=True)
class PlannerService:
    runtime: Any
    context_assembler: PlannerContextAssembler
    intent_synthesizer: IntentSynthesizer
    plan_generator: PlanGenerator
    self_critic: SelfCritic
    macro_synthesizer: MacroSynthesizer
    reflection_writer: ReflectionWriter
    _lock: RLock = field(init=False, repr=False)
    _state: dict[str, _PlannerBotState] = field(init=False, repr=False)
    _counters: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._lock = RLock()
        self._state: dict[str, _PlannerBotState] = {}
        self._counters: dict[str, int] = {
            "planner_requests": 0,
            "planner_replans": 0,
            "planner_failures": 0,
            "planner_success": 0,
            "planner_macro_promotions": 0,
        }

    async def plan(self, payload: PlannerPlanRequest) -> PlannerResponse:
        with self._lock:
            self._counters["planner_requests"] += 1
            if payload.force_replan:
                self._counters["planner_replans"] += 1

        context = self.context_assembler.assemble(meta=payload.meta, objective=payload.objective, horizon=payload.horizon)
        _ = self.intent_synthesizer.synthesize(context=context)

        plan, tactical_bundle, route, provider, model, latency_ms = await self.plan_generator.generate(
            bot_id=payload.meta.bot_id,
            trace_id=payload.meta.trace_id,
            context=context,
            max_steps=payload.max_steps,
        )

        verdict = self.self_critic.evaluate(plan=plan)
        if not verdict.ok:
            with self._lock:
                self._counters["planner_failures"] += 1
                bot_state = self._state.setdefault(payload.meta.bot_id, _PlannerBotState())
                bot_state.current_objective = payload.objective
                bot_state.last_plan_id = plan.plan_id
                bot_state.last_provider = provider
                bot_state.last_model = model
                bot_state.last_rationale = ";".join(verdict.issues)
                bot_state.updated_at = datetime.now(UTC)

            escalation = EscalationNotice(
                bot_id=payload.meta.bot_id,
                severity="warning",
                reason=f"self_critic_rejected:{';'.join(verdict.issues)}",
                recommended_action="fallback_safe_mode",
            )
            self.reflection_writer.write(
                bot_id=payload.meta.bot_id,
                plan_id=plan.plan_id,
                objective=payload.objective,
                succeeded=False,
                rationale=escalation.reason,
                metadata={"trace_id": payload.meta.trace_id, "route": route},
            )
            return PlannerResponse(
                ok=False,
                message="plan_rejected_by_self_critic",
                trace_id=payload.meta.trace_id,
                strategic_plan=plan,
                tactical_bundle=tactical_bundle,
                macro_proposal=None,
                memory_writeback=MemoryWriteback(
                    bot_id=payload.meta.bot_id,
                    summary=f"Plan rejected: {escalation.reason}",
                    semantic_tags=["planner", "critic", "rejected"],
                    metadata={"issues": verdict.issues},
                ),
                learning_label=LearningLabel(
                    bot_id=payload.meta.bot_id,
                    label="planner_rejected",
                    reward=-0.25,
                    details={"issues": verdict.issues},
                ),
                escalation=escalation,
                provider=provider,
                model=model,
                latency_ms=latency_ms,
                route=route,
            )

        macro_proposal = self.macro_synthesizer.synthesize(plan=plan, min_repeat=3)
        if macro_proposal is not None:
            with self._lock:
                self._counters["planner_macro_promotions"] += 1

        with self._lock:
            self._counters["planner_success"] += 1
            bot_state = self._state.setdefault(payload.meta.bot_id, _PlannerBotState())
            bot_state.current_objective = payload.objective
            bot_state.last_plan_id = plan.plan_id
            bot_state.last_provider = provider
            bot_state.last_model = model
            bot_state.last_rationale = plan.rationale
            bot_state.updated_at = datetime.now(UTC)

        self.reflection_writer.write(
            bot_id=payload.meta.bot_id,
            plan_id=plan.plan_id,
            objective=payload.objective,
            succeeded=True,
            rationale=plan.rationale,
            metadata={"trace_id": payload.meta.trace_id, "route": route},
        )

        return PlannerResponse(
            ok=True,
            message="planned",
            trace_id=payload.meta.trace_id,
            strategic_plan=plan,
            tactical_bundle=tactical_bundle,
            macro_proposal=macro_proposal,
            memory_writeback=MemoryWriteback(
                bot_id=payload.meta.bot_id,
                summary=f"Plan generated for objective: {payload.objective}",
                semantic_tags=["planner", "success", payload.horizon.value],
                metadata={"plan_id": plan.plan_id, "risk_score": plan.risk_score},
            ),
            learning_label=LearningLabel(
                bot_id=payload.meta.bot_id,
                label="planner_generated",
                reward=max(-0.2, 1.0 - plan.risk_score),
                details={"plan_id": plan.plan_id, "horizon": payload.horizon.value},
            ),
            escalation=None,
            provider=provider,
            model=model,
            latency_ms=latency_ms,
            route=route,
        )

    async def promote_macro(self, payload: PlannerMacroPromoteRequest) -> PlannerResponse:
        context = self.context_assembler.assemble(meta=payload.meta, objective=payload.objective, horizon=PlanHorizon.tactical)
        plan, tactical_bundle, route, provider, model, latency_ms = await self.plan_generator.generate(
            bot_id=payload.meta.bot_id,
            trace_id=payload.meta.trace_id,
            context=context,
            max_steps=12,
        )
        macro_proposal = self.macro_synthesizer.synthesize(plan=plan, min_repeat=payload.min_repeat)
        if macro_proposal is None:
            return PlannerResponse(
                ok=True,
                message="no_macro_candidate",
                trace_id=payload.meta.trace_id,
                strategic_plan=plan,
                tactical_bundle=tactical_bundle,
                provider=provider,
                model=model,
                latency_ms=latency_ms,
                route=route,
            )
        with self._lock:
            self._counters["planner_macro_promotions"] += 1
        return PlannerResponse(
            ok=True,
            message="macro_candidate_ready",
            trace_id=payload.meta.trace_id,
            strategic_plan=plan,
            tactical_bundle=tactical_bundle,
            macro_proposal=macro_proposal,
            provider=provider,
            model=model,
            latency_ms=latency_ms,
            route=route,
        )

    def explain(self, payload: PlannerExplainRequest) -> dict[str, object]:
        with self._lock:
            state = self._state.get(payload.meta.bot_id)
            if state is None:
                return {
                    "ok": True,
                    "bot_id": payload.meta.bot_id,
                    "trace_id": payload.meta.trace_id,
                    "message": "no_planner_history",
                    "rationale": "",
                }
            rationale = state.last_rationale
        return {
            "ok": True,
            "bot_id": payload.meta.bot_id,
            "trace_id": payload.meta.trace_id,
            "plan_id": payload.plan_id or state.last_plan_id,
            "action_id": payload.action_id,
            "query": payload.query,
            "rationale": rationale,
        }

    def status(self, *, bot_id: str) -> PlannerStatusResponse:
        with self._lock:
            state = self._state.get(bot_id)
            counters = dict(self._counters)
        if state is None:
            return PlannerStatusResponse(
                ok=True,
                bot_id=bot_id,
                planner_healthy=True,
                current_objective=None,
                last_plan_id=None,
                last_provider=None,
                last_model=None,
                counters=counters,
            )
        return PlannerStatusResponse(
            ok=True,
            bot_id=bot_id,
            planner_healthy=True,
            current_objective=state.current_objective,
            last_plan_id=state.last_plan_id,
            last_provider=state.last_provider,
            last_model=state.last_model,
            updated_at=state.updated_at,
            counters=counters,
        )

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)
