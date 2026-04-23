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
    PlannerStepKind,
    StrategicPlan,
    TacticalIntent,
    TacticalIntentBundle,
)
from ai_sidecar.providers.base import PlannerModelRequest


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _json_default(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _plan_schema() -> dict[str, object]:
    step_kinds = [item.value for item in PlannerStepKind]
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
                        "kind": {"type": "string", "enum": step_kinds},
                        "target": {"type": "string"},
                        "description": {"type": "string"},
                        "priority": {"type": "integer", "minimum": 0, "maximum": 1000},
                        "success_predicates": {"type": "array", "items": {"type": "string"}},
                        "fallbacks": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "requires_fleet_coordination": {"type": "boolean"},
        },
    }


@dataclass(slots=True)
class PlanGenerator:
    model_router: object
    planner_timeout_seconds: float
    planner_retries: int
    max_user_prompt_chars: int = 18000

    async def generate(
        self,
        *,
        bot_id: str,
        trace_id: str,
        context: PlannerContext,
        max_steps: int,
    ) -> tuple[StrategicPlan, TacticalIntentBundle, dict[str, object], str, str, float]:
        user_prompt, prompt_meta = self._user_prompt(context=context, max_steps=max_steps)
        request = PlannerModelRequest(
            bot_id=bot_id,
            trace_id=trace_id,
            task="strategic_planning" if context.horizon == PlanHorizon.strategic else "tactical_short_reasoning",
            model="",
            system_prompt=self._system_prompt(context=context),
            user_prompt=user_prompt,
            schema=_plan_schema(),
            timeout_seconds=self.planner_timeout_seconds,
            max_retries=self.planner_retries,
            metadata={
                "horizon": context.horizon.value,
                **prompt_meta,
            },
        )
        response, route_decision = await self.model_router.generate_with_fallback(request=request)
        route = {
            "workload": route_decision.workload,
            "provider_order": route_decision.provider_order,
            "selected_provider": route_decision.selected_provider,
            "selected_model": route_decision.selected_model,
            "fallback_chain": route_decision.fallback_chain,
            "policy_version": route_decision.policy_version,
            "planned_provider": route_decision.planned_provider,
            "planned_model": route_decision.planned_model,
            "attempted_providers": route_decision.attempted_providers,
            "attempted_models": route_decision.attempted_models,
            "fallback_used": route_decision.fallback_used,
            **prompt_meta,
        }
        if not response.ok or not response.content:
            return (
                self._fallback(bot_id=bot_id, context=context, max_steps=max_steps),
                self._bundle_fallback(bot_id=bot_id, context=context),
                route,
                response.provider,
                response.model,
                response.latency_ms,
            )

        content = response.content
        plan = self._to_plan(bot_id=bot_id, context=context, content=content, max_steps=max_steps)
        bundle = self._to_tactical_bundle(bot_id=bot_id, context=context, plan=plan)
        return plan, bundle, route, response.provider, response.model, response.latency_ms

    def _to_plan(self, *, bot_id: str, context: PlannerContext, content: dict[str, object], max_steps: int) -> StrategicPlan:
        now = _now_utc()
        raw_content = content
        if isinstance(content.get("plan"), dict):
            raw_content = content["plan"]

        raw_steps = raw_content.get("steps") if isinstance(raw_content.get("steps"), list) else []
        steps: list[PlannerStep] = []
        for idx, item in enumerate(raw_steps[:max_steps]):
            if not isinstance(item, dict):
                continue
            step = PlannerStep(
                step_id=str(item.get("step_id") or f"s{idx + 1}"),
                kind=self._normalize_step_kind(item.get("kind")),
                target=str(item.get("target") or "") or None,
                description=str(item.get("description") or f"{item.get('kind') or 'task'}"),
                priority=_safe_int(item.get("priority"), max(1, 100 - idx * 5)),
                success_predicates=[str(x) for x in list(item.get("success_predicates") or [])],
                fallbacks=[str(x) for x in list(item.get("fallbacks") or [])],
            )
            steps.append(step)

        assumptions = [str(x) for x in list(raw_content.get("assumptions") or [])]
        constraints = [str(x) for x in list(raw_content.get("constraints") or [])]
        hypotheses = [str(x) for x in list(raw_content.get("hypotheses") or [])]
        policies = [str(x) for x in list(raw_content.get("policies") or [])]
        risk_score = _safe_float(raw_content.get("risk_score") or 0.35, 0.35)
        horizon = context.horizon
        objective = str(raw_content.get("objective") or context.objective)

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
            requires_fleet_coordination=bool(raw_content.get("requires_fleet_coordination") or False),
            rationale=str(raw_content.get("rationale") or ""),
            expires_at=now + timedelta(seconds=120 if horizon == PlanHorizon.tactical else 1200),
        )

    def _actions_from_steps(self, *, bot_id: str, steps: list[PlannerStep], horizon: PlanHorizon) -> list[ActionProposal]:
        now = _now_utc()
        ttl = timedelta(seconds=120 if horizon == PlanHorizon.tactical else 900)
        priority = ActionPriorityTier.tactical if horizon == PlanHorizon.tactical else ActionPriorityTier.strategic
        actions: list[ActionProposal] = []
        for idx, step in enumerate(steps[:10]):
            kind = str(step.kind or "").lower()
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
            elif kind in {"npc", "quest"}:
                command = "talknpc"
                conflict_key = "npc.interact"
            elif kind in {"econ"}:
                command = "storage"
                conflict_key = "economy.loop"
            elif kind in {"skill_up"}:
                command = "skills"
                conflict_key = "progression.skill"
            elif kind in {"rest"}:
                command = "sit"
                conflict_key = "recovery.rest"
            elif kind in {"social"}:
                command = "chat"
                conflict_key = "social.message"
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
                        "step_kind": str(step.kind),
                        "step_priority": int(step.priority),
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
                    priority=max(1, int(step.priority)),
                    constraints=[*list(step.success_predicates), f"step_kind={step.kind}"],
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

    def build_tactical_bundle(self, *, bot_id: str, context: PlannerContext, plan: StrategicPlan) -> TacticalIntentBundle:
        return self._to_tactical_bundle(bot_id=bot_id, context=context, plan=plan)

    def _fallback(self, *, bot_id: str, context: PlannerContext, max_steps: int) -> StrategicPlan:
        now = _now_utc()
        steps = [
            PlannerStep(
                step_id="s1",
                kind=PlannerStepKind.observe,
                target=None,
                description="Gather fresh state and hold safe tactical posture",
                priority=25,
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
        allowed_step_kinds = ", ".join(item.value for item in PlannerStepKind)
        horizon_budget_ms = 2000 if context.horizon == PlanHorizon.tactical else 10000
        return (
            "You are the local sidecar conscious planner for Ragnarok Online bots. "
            "Output only strict JSON following schema. "
            "Do not emit free-form commands outside schema. "
            f"Allowed step kinds: [{allowed_step_kinds}]. "
            f"Current horizon={context.horizon.value} with latency budget under {horizon_budget_ms} ms. "
            "Respect doctrine, safety constraints, and latency tiers."
        )

    def _user_prompt(self, *, context: PlannerContext, max_steps: int) -> tuple[str, dict[str, object]]:
        horizon_budget_ms = 2000 if context.horizon == PlanHorizon.tactical else 10000
        payload = {
            "bot_id": context.bot_id,
            "objective": context.objective,
            "horizon": context.horizon.value,
            "max_steps": max_steps,
            "latency_budget_ms": horizon_budget_ms,
            "latency_headroom": context.latency_headroom,
            "state": context.state,
            "job_progression": context.job_progression,
            "economy_context": context.economy_context,
            "quest_context": context.quest_context,
            "npc_context": context.npc_context,
            "fleet_constraints": context.fleet_constraints,
            "doctrine": context.doctrine,
            "queue": context.queue,
            "recent_events": context.recent_events[:30],
            "memory_matches": context.memory_matches[:10],
            "episodes": context.episodes[:10],
            "macros": context.macros,
            "reflex": context.reflex,
        }

        payload_working = json.loads(json.dumps(payload, ensure_ascii=False, default=_json_default))
        initial_chars = len(json.dumps(payload_working, ensure_ascii=False, default=_json_default))
        reductions: list[str] = []

        def _trim_list(path: tuple[str, ...], new_len: int, label: str) -> bool:
            node: object = payload_working
            for key in path[:-1]:
                if not isinstance(node, dict):
                    return False
                node = node.get(key)
            if not isinstance(node, dict):
                return False
            leaf = node.get(path[-1])
            if not isinstance(leaf, list):
                return False
            if len(leaf) <= new_len:
                return False
            node[path[-1]] = leaf[:new_len]
            reductions.append(f"{label}:{len(leaf)}->{new_len}")
            return True

        def _trim_dict(path: tuple[str, ...], new_len: int, label: str) -> bool:
            node: object = payload_working
            for key in path[:-1]:
                if not isinstance(node, dict):
                    return False
                node = node.get(key)
            if not isinstance(node, dict):
                return False
            leaf = node.get(path[-1])
            if not isinstance(leaf, dict):
                return False
            keys = list(leaf.keys())
            if len(keys) <= new_len:
                return False
            node[path[-1]] = {k: leaf[k] for k in keys[:new_len]}
            reductions.append(f"{label}:{len(keys)}->{new_len}")
            return True

        def _size() -> int:
            return len(json.dumps(payload_working, ensure_ascii=False, default=_json_default))

        if initial_chars > self.max_user_prompt_chars:
            _trim_list(("recent_events",), 20, "recent_events")
            _trim_list(("memory_matches",), 8, "memory_matches")
            _trim_list(("episodes",), 6, "episodes")
            _trim_list(("state", "entities"), 16, "state.entities")
            _trim_dict(("state", "features", "values"), 96, "state.features.values")
            _trim_dict(("job_progression", "skills"), 12, "job_progression.skills")
            _trim_list(("economy_context", "market_listings"), 8, "economy_context.market_listings")
            _trim_list(("quest_context", "active_quests"), 12, "quest_context.active_quests")
            _trim_list(("quest_context", "completed_quests"), 12, "quest_context.completed_quests")
            _trim_list(("npc_context", "relationships"), 8, "npc_context.relationships")
            _trim_list(("state", "recent_event_ids"), 16, "state.recent_event_ids")
            if _size() > self.max_user_prompt_chars:
                _trim_dict(("state", "fleet_intent", "constraints"), 16, "state.fleet_intent.constraints")
                _trim_dict(("fleet_constraints", "constraints"), 16, "fleet_constraints.constraints")
                _trim_dict(("doctrine", "constraints"), 16, "doctrine.constraints")

            if _size() > self.max_user_prompt_chars:
                _trim_list(("recent_events",), 8, "recent_events.tight")
                _trim_list(("memory_matches",), 4, "memory_matches.tight")
                _trim_list(("episodes",), 4, "episodes.tight")
                _trim_list(("state", "entities"), 8, "state.entities.tight")
                _trim_dict(("state", "features", "values"), 48, "state.features.values.tight")
                _trim_list(("economy_context", "market_listings"), 4, "economy_context.market_listings.tight")
                _trim_list(("quest_context", "active_quests"), 8, "quest_context.active_quests.tight")
                _trim_list(("quest_context", "completed_quests"), 8, "quest_context.completed_quests.tight")
                _trim_list(("npc_context", "relationships"), 4, "npc_context.relationships.tight")

            if _size() > self.max_user_prompt_chars:
                payload_working["recent_events"] = []
                payload_working["memory_matches"] = []
                payload_working["episodes"] = []
                reductions.append("drop:recent_events,memory_matches,episodes")

            if _size() > self.max_user_prompt_chars:
                state_block = payload_working.get("state") if isinstance(payload_working.get("state"), dict) else {}
                compact_state = {
                    "generated_at": state_block.get("generated_at"),
                    "operational": state_block.get("operational", {}),
                    "encounter": state_block.get("encounter", {}),
                    "navigation": state_block.get("navigation", {}),
                    "inventory": state_block.get("inventory", {}),
                    "economy": state_block.get("economy", {}),
                    "quest": {
                        "active_objective_count": (
                            state_block.get("quest", {}).get("active_objective_count")
                            if isinstance(state_block.get("quest"), dict)
                            else None
                        ),
                        "objective_completion_ratio": (
                            state_block.get("quest", {}).get("objective_completion_ratio")
                            if isinstance(state_block.get("quest"), dict)
                            else None
                        ),
                    },
                    "risk": state_block.get("risk", {}),
                    "fleet_intent": state_block.get("fleet_intent", {}),
                }
                payload_working["state"] = compact_state
                payload_working["job_progression"] = {
                    "job_name": context.job_progression.get("job_name"),
                    "base_level": context.job_progression.get("base_level"),
                    "job_level": context.job_progression.get("job_level"),
                    "skill_points": context.job_progression.get("skill_points"),
                }
                payload_working["economy_context"] = {
                    "zeny": context.economy_context.get("zeny"),
                    "overweight_ratio": context.economy_context.get("overweight_ratio"),
                }
                payload_working["quest_context"] = {
                    "active_objective_count": context.quest_context.get("active_objective_count"),
                    "objective_completion_ratio": context.quest_context.get("objective_completion_ratio"),
                    "last_npc": context.quest_context.get("last_npc"),
                }
                payload_working["npc_context"] = {
                    "last_interacted_npc": context.npc_context.get("last_interacted_npc"),
                }
                reductions.append("collapse:state_and_domain_context")

            if _size() > self.max_user_prompt_chars:
                payload_working["doctrine"] = {
                    "doctrine_version": context.doctrine.get("doctrine_version"),
                }
                payload_working["fleet_constraints"] = {
                    "role": context.fleet_constraints.get("role"),
                    "assignment": context.fleet_constraints.get("assignment"),
                    "objective": context.fleet_constraints.get("objective"),
                }
                reductions.append("collapse:doctrine_and_fleet_constraints")

        final_chars = _size()
        prompt = json.dumps(payload_working, ensure_ascii=False, default=_json_default)
        return prompt, {
            "prompt_chars_initial": initial_chars,
            "prompt_chars_final": final_chars,
            "prompt_chars_limit": self.max_user_prompt_chars,
            "prompt_reductions": reductions,
        }

    def _normalize_step_kind(self, raw_kind: object) -> PlannerStepKind:
        value = str(raw_kind or "").strip().lower()
        aliases = {
            "move": PlannerStepKind.travel,
            "route": PlannerStepKind.travel,
            "fight": PlannerStepKind.combat,
            "attack": PlannerStepKind.combat,
            "collect": PlannerStepKind.loot,
            "recover": PlannerStepKind.rest,
            "heal": PlannerStepKind.rest,
            "economy": PlannerStepKind.econ,
            "skill": PlannerStepKind.skill_up,
        }
        if value in aliases:
            return aliases[value]
        for item in PlannerStepKind:
            if value == item.value:
                return item
        return PlannerStepKind.task
