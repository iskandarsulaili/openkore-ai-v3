from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from ai_sidecar.contracts.actions import ActionPriorityTier, ActionProposal
from ai_sidecar.planner.schemas import (
    PlanHorizon,
    PlannerContext,
    PlannerStep,
    StrategicPlan,
    TacticalIntent,
    TacticalIntentBundle,
)
from ai_sidecar.providers.base import PlannerModelRequest


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _plan_schema() -> dict[str, object]:
    return {
        "type": "object",
        "required": ["objective", "steps", "risk_score", "assumptions", "constraints", "rationale"],
        "properties": {
            "objective": {"type": "string"},
            "assumptions": {"type": "array", "items": {"type": "string"}},
            "constraints": {"type": "array", "items": {"type": "string"}},
            "hypotheses": {"type": "array", "items": {"type": "string"}},
            "policies": {"type": "array", "items": {"type": "string"}},
            "risk_score": {"type": "number"},
            "rationale": {"type": "string"},
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["step_id", "kind", "description"],
                    "properties": {
                        "step_id": {"type": "string"},
                        "kind": {"type": "string"},
                        "target": {"type": "string"},
                        "description": {"type": "string"},
                        "success_predicates": {"type": "array", "items": {"type": "string"}},
                        "fallbacks": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        },
    }


@dataclass(slots=True)
class PlanGenerator:
    model_router: object
    planner_timeout_seconds: float
    planner_retries: int

    async def generate(
        self,
        *,
        bot_id: str,
        trace_id: str,
        context: PlannerContext,
        max_steps: int,
    ) -> tuple[StrategicPlan, TacticalIntentBundle, dict[str, object], str, str, float]:
        request = PlannerModelRequest(
            bot_id=bot_id,
            trace_id=trace_id,
            task="strategic_planning" if context.horizon == PlanHorizon.strategic else "tactical_short_reasoning",
            model="",
            system_prompt=self._system_prompt(context=context),
            user_prompt=self._user_prompt(context=context, max_steps=max_steps),
            schema=_plan_schema(),
            timeout_seconds=self.planner_timeout_seconds,
            max_retries=self.planner_retries,
            metadata={"horizon": context.horizon.value},
        )
        response, route_decision = await self.model_router.generate_with_fallback(request=request)
        route = {
            "workload": route_decision.workload,
            "provider_order": route_decision.provider_order,
            "selected_provider": route_decision.selected_provider,
            "selected_model": route_decision.selected_model,
            "fallback_chain": route_decision.fallback_chain,
            "policy_version": route_decision.policy_version,
        }
        if not response.ok or not response.content:
            return self._fallback(bot_id=bot_id, context=context, max_steps=max_steps), self._bundle_fallback(bot_id=bot_id, context=context), route, response.provider, response.model, response.latency_ms

        content = response.content
        plan = self._to_plan(bot_id=bot_id, context=context, content=content, max_steps=max_steps)
        bundle = self._to_tactical_bundle(bot_id=bot_id, context=context, plan=plan)
        return plan, bundle, route, response.provider, response.model, response.latency_ms

    def _to_plan(self, *, bot_id: str, context: PlannerContext, content: dict[str, object], max_steps: int) -> StrategicPlan:
        now = _now_utc()
        raw_steps = content.get("steps") if isinstance(content.get("steps"), list) else []
        steps: list[PlannerStep] = []
        for idx, item in enumerate(raw_steps[:max_steps]):
            if not isinstance(item, dict):
                continue
            step = PlannerStep(
                step_id=str(item.get("step_id") or f"s{idx + 1}"),
                kind=str(item.get("kind") or "task"),
                target=str(item.get("target") or "") or None,
                description=str(item.get("description") or ""),
                success_predicates=[str(x) for x in list(item.get("success_predicates") or [])],
                fallbacks=[str(x) for x in list(item.get("fallbacks") or [])],
            )
            steps.append(step)

        assumptions = [str(x) for x in list(content.get("assumptions") or [])]
        constraints = [str(x) for x in list(content.get("constraints") or [])]
        hypotheses = [str(x) for x in list(content.get("hypotheses") or [])]
        policies = [str(x) for x in list(content.get("policies") or [])]
        risk_score = float(content.get("risk_score") or 0.35)
        horizon = context.horizon
        objective = str(content.get("objective") or context.objective)

        actions = self._actions_from_steps(bot_id=bot_id, steps=steps, horizon=horizon)
        return StrategicPlan(
            plan_id=f"plan-{uuid4().hex[:20]}",
            bot_id=bot_id,
            objective=objective,
            horizon=horizon,
            assumptions=assumptions,
            constraints=constraints,
            hypotheses=hypotheses,
            policies=policies,
            steps=steps,
            recommended_actions=actions,
            recommended_macros=[],
            risk_score=max(0.0, min(1.0, risk_score)),
            requires_fleet_coordination=bool(content.get("requires_fleet_coordination") or False),
            rationale=str(content.get("rationale") or ""),
            expires_at=now + timedelta(seconds=120 if horizon == PlanHorizon.tactical else 1200),
        )

    def _actions_from_steps(self, *, bot_id: str, steps: list[PlannerStep], horizon: PlanHorizon) -> list[ActionProposal]:
        now = _now_utc()
        ttl = timedelta(seconds=120 if horizon == PlanHorizon.tactical else 900)
        priority = ActionPriorityTier.tactical if horizon == PlanHorizon.tactical else ActionPriorityTier.strategic
        actions: list[ActionProposal] = []
        for idx, step in enumerate(steps[:10]):
            kind = (step.kind or "").lower()
            command = ""
            conflict_key = None
            if kind in {"travel", "move", "route"}:
                command = f"move {step.target or ''}".strip()
                conflict_key = "nav.route"
            elif kind in {"combat", "fight", "attack"}:
                command = "attack"
                conflict_key = "combat.primary"
            elif kind in {"loot", "collect"}:
                command = "take"
                conflict_key = "loot.collect"
            elif kind in {"recover", "heal"}:
                command = "sit"
                conflict_key = "recovery"
            if not command:
                continue
            actions.append(
                ActionProposal(
                    action_id=f"planact-{uuid4().hex[:20]}",
                    kind="command",
                    command=command[:256],
                    priority_tier=priority,
                    conflict_key=conflict_key,
                    created_at=now,
                    expires_at=now + ttl,
                    idempotency_key=f"plan:{bot_id}:{idx}:{command}"[:128],
                    metadata={
                        "source": "planner",
                        "step_id": step.step_id,
                        "step_kind": step.kind,
                        "target": step.target,
                    },
                )
            )
        return actions

    def _to_tactical_bundle(self, *, bot_id: str, context: PlannerContext, plan: StrategicPlan) -> TacticalIntentBundle:
        intents: list[TacticalIntent] = []
        for idx, step in enumerate(plan.steps[:8]):
            intents.append(
                TacticalIntent(
                    intent_id=f"intent-{plan.plan_id[:8]}-{idx + 1}",
                    objective=step.description or step.kind,
                    priority=max(1, 100 - idx * 5),
                    constraints=list(step.success_predicates),
                    expected_latency_ms=1500 if context.horizon == PlanHorizon.tactical else 5000,
                )
            )
        return TacticalIntentBundle(
            bundle_id=f"bundle-{uuid4().hex[:20]}",
            bot_id=bot_id,
            intents=intents,
            actions=plan.recommended_actions,
            notes=[f"derived_from_plan:{plan.plan_id}", f"horizon:{context.horizon.value}"],
        )

    def _fallback(self, *, bot_id: str, context: PlannerContext, max_steps: int) -> StrategicPlan:
        now = _now_utc()
        steps = [
            PlannerStep(
                step_id="s1",
                kind="observe",
                target=None,
                description="Gather fresh state and hold safe tactical posture",
                success_predicates=["state_refreshed"],
                fallbacks=["safe_idle"],
            )
        ]
        actions = [
            ActionProposal(
                action_id=f"fallback-{uuid4().hex[:20]}",
                kind="command",
                command="ai auto",
                priority_tier=ActionPriorityTier.tactical,
                conflict_key="planner.safe_idle",
                created_at=now,
                expires_at=now + timedelta(seconds=90),
                idempotency_key=f"fallback:{bot_id}:safe_idle"[:128],
                metadata={"source": "planner_fallback", "objective": context.objective},
            )
        ]
        return StrategicPlan(
            plan_id=f"fallback-plan-{uuid4().hex[:16]}",
            bot_id=bot_id,
            objective=context.objective,
            horizon=context.horizon,
            assumptions=["provider_unavailable_or_schema_invalid"],
            constraints=["keep_safe_posture"],
            hypotheses=[],
            policies=["fallback_safe_mode"],
            steps=steps[:max_steps],
            recommended_actions=actions,
            recommended_macros=[],
            risk_score=0.2,
            requires_fleet_coordination=False,
            rationale="Fallback plan generated due to model call failure.",
            expires_at=now + timedelta(seconds=180),
        )

    def _bundle_fallback(self, *, bot_id: str, context: PlannerContext) -> TacticalIntentBundle:
        return TacticalIntentBundle(
            bundle_id=f"fallback-bundle-{uuid4().hex[:16]}",
            bot_id=bot_id,
            intents=[
                TacticalIntent(
                    intent_id="fallback-intent-1",
                    objective="Hold safe idle while waiting for replan",
                    priority=50,
                    constraints=["no risky actions"],
                    expected_latency_ms=500,
                )
            ],
            actions=[],
            notes=["fallback_bundle"],
        )

    def _system_prompt(self, *, context: PlannerContext) -> str:
        return (
            "You are the local sidecar conscious planner for Ragnarok Online bots. "
            "Output only strict JSON following schema. "
            "Do not emit free-form commands outside schema. "
            "Respect doctrine, safety constraints, and latency tiers."
        )

    def _user_prompt(self, *, context: PlannerContext, max_steps: int) -> str:
        payload = {
            "bot_id": context.bot_id,
            "objective": context.objective,
            "horizon": context.horizon.value,
            "max_steps": max_steps,
            "state": context.state,
            "fleet_constraints": context.fleet_constraints,
            "doctrine": context.doctrine,
            "queue": context.queue,
            "recent_events": context.recent_events[:30],
            "memory_matches": context.memory_matches[:10],
            "episodes": context.episodes[:10],
            "macros": context.macros,
        }
        return json.dumps(payload, ensure_ascii=False)

