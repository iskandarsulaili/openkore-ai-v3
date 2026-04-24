from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from ai_sidecar.contracts.actions import ActionPriorityTier, ActionProposal
from ai_sidecar.contracts.common import ContractMeta
from ai_sidecar.planner.plan_generator import PlanGenerator
from ai_sidecar.planner.schemas import (
    PlanHorizon,
    PlannerContext,
    PlannerPlanRequest,
    PlannerStep,
    PlannerStepKind,
    StrategicPlan,
    TacticalIntentBundle,
)
from ai_sidecar.planner.self_critic import SelfCritic
from ai_sidecar.planner.service import PlannerService
from ai_sidecar.planner.validator import PlanValidationVerdict, PlanValidator
from ai_sidecar.providers.prompt_guard import PromptGuard


def _meta() -> ContractMeta:
    return ContractMeta(contract_version="v1", source="pytest", bot_id="bot:ws-c", trace_id="trace-ws-c")


def _now() -> datetime:
    return datetime.now(UTC)


def _plan(*, horizon: PlanHorizon = PlanHorizon.tactical) -> StrategicPlan:
    now = _now()
    step = PlannerStep(
        step_id="s1",
        kind=PlannerStepKind.travel,
        target="prt_fild08",
        description="Move to farm map",
        priority=80,
        success_predicates=["arrived"],
        fallbacks=["safe_idle"],
    )
    action = ActionProposal(
        action_id="a1",
        kind="command",
        command="move prt_fild08",
        priority_tier=ActionPriorityTier.tactical,
        conflict_key="nav.route",
        created_at=now,
        expires_at=now + timedelta(seconds=120),
        idempotency_key="plan:bot:1",
        metadata={"source": "planner"},
    )
    return StrategicPlan(
        plan_id="plan-1",
        bot_id="bot:ws-c",
        objective="farm safely",
        horizon=horizon,
        assumptions=["state_is_fresh"],
        constraints=["avoid_pvp"],
        hypotheses=[],
        policies=["doctrine-v3"],
        steps=[step],
        recommended_actions=[action],
        recommended_macros=[],
        risk_score=0.2,
        requires_fleet_coordination=False,
        rationale="bounded tactical movement",
        expires_at=now + timedelta(seconds=120),
    )


def test_prompt_guard_normalizes_deepseek_style_payload_before_validation() -> None:
    guard = PromptGuard()
    schema = {
        "type": "object",
        "required": ["objective", "steps", "risk_score", "assumptions", "constraints", "rationale"],
        "properties": {
            "objective": {"type": "string"},
            "assumptions": {"type": "array", "items": {"type": "string"}},
            "constraints": {"type": "array", "items": {"type": "string"}},
            "risk_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "rationale": {"type": "string"},
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["step_id", "kind", "description"],
                    "properties": {
                        "step_id": {"type": "string"},
                        "kind": {"type": "string", "enum": ["travel", "combat"]},
                        "description": {"type": "string"},
                    },
                },
            },
        },
    }

    raw = """```json
    {
      "objective": "farm",
      "assumptions": [],
      "constraints": [],
      "risk_score": "0.21",
      "rationale": "ok",
      "steps": [{"step_id": "s1", "kind": "TRAVEL", "description": "move"}]
    }
    ```"""
    parsed = guard.parse_json_object(raw)
    assert parsed is not None

    normalized = guard.normalize_for_schema(parsed, schema)
    guard.validate_schema(normalized, schema)
    assert normalized["risk_score"] == 0.21
    assert normalized["steps"][0]["kind"] == "travel"


def test_plan_generator_compacts_prompt_and_emits_metadata() -> None:
    generator = PlanGenerator(
        model_router=object(),
        planner_timeout_seconds=10.0,
        planner_retries=0,
        max_user_prompt_chars=2400,
    )
    huge_state = {
        "entities": [{"id": f"e{i}", "name": "mob", "noise": "x" * 120} for i in range(200)],
        "features": {"values": {f"f{i}": i for i in range(1000)}},
        "recent_event_ids": [f"evt{i}" for i in range(120)],
        "fleet_intent": {"constraints": {f"k{i}": i for i in range(200)}},
    }
    context = PlannerContext(
        bot_id="bot:ws-c",
        objective="farm safely",
        horizon=PlanHorizon.strategic,
        state=huge_state,
        recent_events=[{"text": "evt", "payload": {"big": "x" * 200}} for _ in range(80)],
        memory_matches=[{"text": "memory"} for _ in range(20)],
        episodes=[{"text": "episode"} for _ in range(20)],
        doctrine={"constraints": {f"d{i}": i for i in range(100)}},
        fleet_constraints={"constraints": {f"c{i}": i for i in range(100)}},
        queue={"pending_actions": 10},
        macros={"latest_publication": {"version": "v1"}},
        reflex={"active_rules": 1},
        job_progression={"skills": {f"s{i}": i for i in range(100)}},
        economy_context={"market_listings": [{"item": i, "blob": "x" * 120} for i in range(100)]},
        quest_context={"active_quests": [f"q{i}" for i in range(80)], "completed_quests": [f"qc{i}" for i in range(80)]},
        npc_context={"relationships": [{"npc": i, "blob": "x" * 80} for i in range(80)]},
        latency_headroom={"horizon_budget_ms": 10000},
    )

    prompt, meta = generator._user_prompt(context=context, max_steps=8)
    payload = json.loads(prompt)

    assert len(prompt) <= 2400
    assert int(meta["prompt_chars_initial"]) > int(meta["prompt_chars_final"])
    assert int(meta["prompt_chars_final"]) <= int(meta["prompt_chars_limit"])
    assert isinstance(meta["prompt_reductions"], list)
    assert payload["latency_budget_ms"] == 30000


@dataclass(slots=True)
class _Assembler:
    context: PlannerContext

    def assemble(self, **_: object) -> PlannerContext:
        return self.context


@dataclass(slots=True)
class _Generator:
    plan: StrategicPlan
    bundle: TacticalIntentBundle
    latency_ms: float

    async def generate(self, **_: object):
        return self.plan, self.bundle, {"seed": "ok"}, "deepseek", "deepseek-chat", self.latency_ms

    def build_tactical_bundle(self, **_: object) -> TacticalIntentBundle:
        return self.bundle


class _IntentSynth:
    def synthesize(self, **_: object) -> list[object]:
        return []


class _ReflectionWriter:
    def write(self, **_: object) -> None:
        return None


class _MacroSynth:
    def synthesize(self, **_: object):
        return None


def test_planner_service_rejects_plan_when_latency_budget_exceeded() -> None:
    now = _now()
    context = PlannerContext(bot_id="bot:ws-c", objective="farm safely", horizon=PlanHorizon.tactical)
    bundle = TacticalIntentBundle(bundle_id="bundle-1", bot_id="bot:ws-c", intents=[], actions=[], notes=[])
    service = PlannerService(
        runtime=object(),
        context_assembler=_Assembler(context=context),
        intent_synthesizer=_IntentSynth(),
        plan_generator=_Generator(plan=_plan(horizon=PlanHorizon.tactical), bundle=bundle, latency_ms=2500.0),
        plan_validator=PlanValidator(tactical_budget_ms=2000, strategic_budget_ms=10000),
        self_critic=SelfCritic(tactical_budget_ms=2000, strategic_budget_ms=10000),
        macro_synthesizer=_MacroSynth(),
        reflection_writer=_ReflectionWriter(),
    )

    response = asyncio.run(
        service.plan(
            PlannerPlanRequest(
                meta=ContractMeta(contract_version="v1", source="pytest", bot_id="bot:ws-c", trace_id="trace-ws-c"),
                objective="farm safely",
                horizon=PlanHorizon.tactical,
                force_replan=False,
                max_steps=8,
            )
        )
    )

    assert response.ok is True
    assert response.message == "planned_with_fallback"
    assert response.provider == "deepseek"
    assert response.route["execution_flow"] == [
        "context_assembly",
        "plan_generation",
        "plan_validation",
        "self_critic",
        "output",
    ]
    assert response.route["latency_budget"]["within_budget"] is False
    assert "latency_budget_exceeded" in ";".join(response.route["validation"]["issues"])
    assert response.route["fallback"]["used"] is True
    assert response.strategic_plan is not None
    assert response.tactical_bundle is not None


def test_plan_validator_accepts_strategic_plan_within_budget() -> None:
    validator = PlanValidator(tactical_budget_ms=2000, strategic_budget_ms=10000)
    verdict = validator.validate(plan=_plan(horizon=PlanHorizon.strategic), latency_ms=1200.0)
    assert verdict.ok is True
    assert verdict.issues == []
    assert verdict.normalized.steps[0].kind == PlannerStepKind.travel
