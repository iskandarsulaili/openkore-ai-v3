from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from ai_sidecar.contracts.actions import ActionPriorityTier, ActionProposal
from ai_sidecar.contracts.common import ContractMeta
from ai_sidecar.lifecycle import RuntimeState
from ai_sidecar.planner.plan_generator import PlanGenerator
from ai_sidecar.planner.schemas import (
    PlanHorizon,
    PlannerContext,
    PlannerPlanRequest,
    PlannerStatusResponse,
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


def test_plan_generator_system_prompt_includes_phase5_invariants_and_capability_truth() -> None:
    generator = PlanGenerator(model_router=object(), planner_timeout_seconds=5.0, planner_retries=0)
    context = PlannerContext(
        bot_id="bot:ws-c",
        objective="farm safely",
        horizon=PlanHorizon.tactical,
        invariants={
            "reasoning_protocol": [
                "observe",
                "verify",
                "risk",
                "options",
                "capability_check",
                "plan",
                "fallback",
                "output",
            ],
            "rathena_axioms": ["evidence_only", "abstain_on_unknowns"],
            "capability_truth": {
                "direct": {"tool": "propose_actions", "allowed_roots": ["ai", "move", "talknpc"]},
                "config": {"tool": "plan_control_change"},
                "macro": {"tool": "publish_macro"},
            },
            "known_upgrade_rule_ids": ["rule-a", "rule-b"],
        },
    )

    prompt = generator._system_prompt(context=context)

    assert "Mandatory internal reasoning protocol" in prompt
    assert "Capability truth:" in prompt
    assert "direct:propose_actions" in prompt
    assert "Known upgrade rule ids" in prompt


def test_plan_generator_user_prompt_includes_phase5_context_surfaces() -> None:
    generator = PlanGenerator(model_router=object(), planner_timeout_seconds=5.0, planner_retries=0, max_user_prompt_chars=6000)
    context = PlannerContext(
        bot_id="bot:ws-c",
        objective="farm safely",
        horizon=PlanHorizon.strategic,
        state={"operational": {"map": "prt_fild08", "hp": 900}},
        invariants={"reasoning_protocol": ["observe", "verify"], "knowledge_version": "k-v1"},
        runtime_facts={"queue_pending_actions": 2, "planner_state": "healthy"},
        knowledge_summary={"knowledge_version": "k-v1", "known_upgrade_rules": ["rule-a"]},
    )

    prompt, _meta = generator._user_prompt(context=context, max_steps=8)
    payload = json.loads(prompt)

    assert payload.get("invariants", {}).get("knowledge_version") == "k-v1"
    assert payload.get("runtime_facts", {}).get("queue_pending_actions") == 2
    assert payload.get("knowledge_summary", {}).get("known_upgrade_rules") == ["rule-a"]


def test_plan_generator_fallback_emits_context_aware_actions_without_ai_auto() -> None:
    generator = PlanGenerator(model_router=object(), planner_timeout_seconds=5.0, planner_retries=0)
    context = PlannerContext(
        bot_id="bot:ws-c",
        objective="farm safely",
        horizon=PlanHorizon.tactical,
        state={
            "operational": {"map": "prt_fild01", "hp": 300},
            "navigation": {"map": "prt_fild01"},
            "encounter": {"nearby_hostiles": 0, "target_id": None},
            "inventory": {"item_count": 99, "overweight_ratio": 0.92},
            "risk": {"anomaly_flags": ["planner.stale_loop"]},
            "features": {"values": {"reconnect_age_s": 12.0}},
        },
        fleet_constraints={"assignment": "prt_fild08", "constraints": {"preferred_grind_maps": ["prt_fild08"]}},
        quest_context={"active_objective_count": 4},
        economy_context={"overweight_ratio": 0.92},
    )

    fallback = generator._fallback(bot_id="bot:ws-c", context=context, max_steps=8)

    assert fallback.recommended_actions
    assert all(item.command != "ai auto" for item in fallback.recommended_actions)
    modes = {str(item.metadata.get("fallback_mode") or "") for item in fallback.recommended_actions}
    assert "economy_relief" in modes
    assert "resume_grind" in modes
    assert "seek_targets" in modes
    seek_action = next(item for item in fallback.recommended_actions if item.metadata.get("fallback_mode") == "seek_targets")
    assert "scan.targets_absent" in seek_action.preconditions
    assert seek_action.metadata.get("target_scan_required") is True
    assert seek_action.metadata.get("seek_only_random_walk") is True


def test_plan_generator_fallback_safe_idle_only_when_no_other_safe_action_exists() -> None:
    generator = PlanGenerator(model_router=object(), planner_timeout_seconds=5.0, planner_retries=0)
    context = PlannerContext(
        bot_id="bot:ws-c",
        objective="farm safely",
        horizon=PlanHorizon.tactical,
        state={
            "operational": {"map": "prt_fild08", "hp": 500},
            "navigation": {"map": "prt_fild08"},
            "encounter": {"nearby_hostiles": 2, "target_id": "mob:poring"},
            "inventory": {"item_count": 6, "overweight_ratio": 0.1},
            "risk": {"anomaly_flags": []},
        },
        fleet_constraints={"assignment": "prt_fild08", "constraints": {"preferred_grind_maps": ["prt_fild08"]}},
        economy_context={"overweight_ratio": 0.1},
        quest_context={"active_objective_count": 1},
    )

    fallback = generator._fallback(bot_id="bot:ws-c", context=context, max_steps=8)

    assert len(fallback.recommended_actions) == 1
    action = fallback.recommended_actions[0]
    assert action.command == "sit"
    assert action.metadata.get("fallback_mode") == "safe_idle"


def test_plan_generator_fallback_death_recovery_uses_respawn_command() -> None:
    generator = PlanGenerator(model_router=object(), planner_timeout_seconds=5.0, planner_retries=0)
    context = PlannerContext(
        bot_id="bot:ws-c",
        objective="recover from death",
        horizon=PlanHorizon.tactical,
        state={
            "operational": {"map": "prt_fild08", "hp": 0},
            "navigation": {"map": "prt_fild08"},
            "encounter": {"nearby_hostiles": 2, "target_id": "mob:poring"},
            "inventory": {"item_count": 6, "overweight_ratio": 0.1},
            "risk": {"anomaly_flags": []},
        },
        fleet_constraints={"assignment": "prt_fild08", "constraints": {"preferred_grind_maps": ["prt_fild08"]}},
        economy_context={"overweight_ratio": 0.1},
        quest_context={"active_objective_count": 1},
    )

    fallback = generator._fallback(bot_id="bot:ws-c", context=context, max_steps=8)

    death_action = next(item for item in fallback.recommended_actions if item.metadata.get("fallback_mode") == "death_recovery")
    assert death_action.command == "respawn"
    assert death_action.conflict_key == "recovery.death"
    assert death_action.metadata.get("target") == "savepoint"


def test_plan_generator_actions_from_steps_bridge_compatible_for_residual_kinds() -> None:
    generator = PlanGenerator(model_router=object(), planner_timeout_seconds=5.0, planner_retries=0)
    steps = [
        PlannerStep(step_id="s1", kind=PlannerStepKind.combat, description="engage target", priority=100),
        PlannerStep(step_id="s2", kind=PlannerStepKind.rest, description="recover", priority=95),
        PlannerStep(step_id="s3", kind=PlannerStepKind.econ, description="storage loop", priority=90),
        PlannerStep(step_id="s4", kind=PlannerStepKind.skill_up, description="allocate skills", priority=85),
        PlannerStep(step_id="s5", kind=PlannerStepKind.equip, description="swap gear", priority=80),
        PlannerStep(step_id="s6", kind=PlannerStepKind.party, description="party sync", priority=75),
        PlannerStep(step_id="s7", kind=PlannerStepKind.chat, description="announce", priority=70),
        PlannerStep(step_id="s8", kind=PlannerStepKind.social, description="social ping", priority=65),
        PlannerStep(step_id="s9", kind=PlannerStepKind.vending, description="open vending", priority=60),
        PlannerStep(step_id="s10", kind=PlannerStepKind.craft, description="craft supplies", priority=55),
    ]

    actions = generator._actions_from_steps(bot_id="bot:ws-c", steps=steps, horizon=PlanHorizon.tactical)

    assert len(actions) == 2
    commands = {item.command for item in actions}
    assert "ai auto" in commands
    assert "ai manual" in commands

    allowed_roots = {"ai", "move", "macro", "eventmacro", "talknpc", "take"}
    for action in actions:
        root = action.command.split(maxsplit=1)[0].strip().lower()
        assert root in allowed_roots
        compat = dict(action.metadata).get("bridge_compat")
        assert isinstance(compat, dict)
        assert compat.get("status") == "rewritten"


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


def test_planner_service_status_no_state_is_unhealthy_and_without_timestamp() -> None:
    context = PlannerContext(bot_id="bot:ws-c", objective="farm safely", horizon=PlanHorizon.tactical)
    bundle = TacticalIntentBundle(bundle_id="bundle-1", bot_id="bot:ws-c", intents=[], actions=[], notes=[])
    service = PlannerService(
        runtime=object(),
        context_assembler=_Assembler(context=context),
        intent_synthesizer=_IntentSynth(),
        plan_generator=_Generator(plan=_plan(horizon=PlanHorizon.tactical), bundle=bundle, latency_ms=10.0),
        plan_validator=PlanValidator(tactical_budget_ms=2000, strategic_budget_ms=10000),
        self_critic=SelfCritic(tactical_budget_ms=2000, strategic_budget_ms=10000),
        macro_synthesizer=_MacroSynth(),
        reflection_writer=_ReflectionWriter(),
    )

    status = service.status(bot_id="bot:no-state")

    assert status.ok is True
    assert status.bot_id == "bot:no-state"
    assert status.planner_healthy is False
    assert status.updated_at is None
    assert status.current_objective is None
    assert status.last_plan_id is None


def test_readiness_bot_selection_prefers_bot_with_real_planner_state() -> None:
    now = datetime.now(UTC)

    class _Runtime:
        def list_bots(self) -> list[dict[str, object]]:
            return [
                {
                    "bot_id": "bot:idle",
                    "last_seen_at": now - timedelta(seconds=2),
                    "latest_snapshot_at": now - timedelta(seconds=2),
                    "pending_actions": 1,
                    "telemetry_events": 10,
                    "liveness_state": "online",
                },
                {
                    "bot_id": "bot:active",
                    "last_seen_at": now - timedelta(seconds=8),
                    "latest_snapshot_at": now - timedelta(seconds=8),
                    "pending_actions": 0,
                    "telemetry_events": 1,
                    "liveness_state": "online",
                },
            ]

        def planner_status(self, *, bot_id: str) -> PlannerStatusResponse:
            if bot_id == "bot:active":
                return PlannerStatusResponse(
                    ok=True,
                    bot_id=bot_id,
                    planner_healthy=True,
                    current_objective="farm safely",
                    last_plan_id="plan-active",
                    last_provider="deepseek",
                    last_model="deepseek-chat",
                    updated_at=now - timedelta(seconds=1),
                    counters={},
                )
            return PlannerStatusResponse(
                ok=True,
                bot_id=bot_id,
                planner_healthy=False,
                current_objective=None,
                last_plan_id=None,
                last_provider=None,
                last_model=None,
                updated_at=None,
                counters={},
            )

    assert RuntimeState._readiness_bot_id(_Runtime()) == "bot:active"


def test_readiness_indicators_absent_state_is_stale_and_real_recent_state_is_not() -> None:
    now = datetime.now(UTC)

    class _Runtime:
        planner_stale_threshold_s = 60.0
        autonomy_scheduler_degraded = False
        autonomy_scheduler_degraded_reason = ""
        pdca_loop = SimpleNamespace(running=True, _circuit_breaker_tripped=lambda: False)

        def __init__(self, planner: PlannerStatusResponse) -> None:
            self._planner = planner

        def _readiness_bot_id(self) -> str:
            return self._planner.bot_id

        def planner_status(self, *, bot_id: str) -> PlannerStatusResponse:
            assert bot_id == self._planner.bot_id
            return self._planner

        def _fleet_status(self) -> dict[str, object]:
            return {
                "mode": "central",
                "central_available": True,
                "stale": False,
                "last_sync_at": now,
            }

    absent = PlannerStatusResponse(
        ok=True,
        bot_id="bot:none",
        planner_healthy=False,
        current_objective=None,
        last_plan_id=None,
        last_provider=None,
        last_model=None,
        updated_at=None,
        counters={},
    )
    absent_indicators = RuntimeState.readiness_indicators(_Runtime(absent))
    assert absent_indicators["planner_stale"] is True
    assert absent_indicators["planner_stale_seconds"] is None
    assert absent_indicators["planner_last_updated_at"] is None

    recent = PlannerStatusResponse(
        ok=True,
        bot_id="bot:recent",
        planner_healthy=True,
        current_objective="farm safely",
        last_plan_id="plan-recent",
        last_provider="deepseek",
        last_model="deepseek-chat",
        updated_at=now - timedelta(seconds=5),
        counters={},
    )
    recent_indicators = RuntimeState.readiness_indicators(_Runtime(recent))
    assert recent_indicators["planner_stale"] is False
    assert isinstance(recent_indicators["planner_stale_seconds"], float)
    assert 0.0 <= float(recent_indicators["planner_stale_seconds"]) <= 60.0


def test_readiness_indicators_fleet_disabled_not_stale() -> None:
    now = datetime.now(UTC)

    class _Runtime:
        planner_stale_threshold_s = 60.0
        autonomy_scheduler_degraded = False
        autonomy_scheduler_degraded_reason = ""
        pdca_loop = SimpleNamespace(running=True, _circuit_breaker_tripped=lambda: False)

        def _readiness_bot_id(self) -> str:
            return "bot:fleet-disabled"

        def planner_status(self, *, bot_id: str) -> PlannerStatusResponse:
            assert bot_id == "bot:fleet-disabled"
            return PlannerStatusResponse(
                ok=True,
                bot_id=bot_id,
                planner_healthy=True,
                current_objective="farm safely",
                last_plan_id="plan-disabled",
                last_provider="deepseek",
                last_model="deepseek-chat",
                updated_at=now - timedelta(seconds=5),
                counters={},
            )

        def _fleet_status(self) -> dict[str, object]:
            return {
                "mode": "local",
                "central_enabled": False,
                "central_available": False,
                "stale": False,
                "last_sync_at": None,
            }

    indicators = RuntimeState.readiness_indicators(_Runtime())
    assert indicators["fleet_mode"] == "local"
    assert indicators["fleet_central_enabled"] is False
    assert indicators["fleet_central_available"] is False
    assert indicators["fleet_central_stale"] is False
