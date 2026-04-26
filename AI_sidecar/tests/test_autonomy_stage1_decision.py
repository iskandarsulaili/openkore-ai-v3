from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime

from ai_sidecar.autonomy.decision_service import DecisionService
from ai_sidecar.autonomy.goal_stack import compute_goal_stack
from ai_sidecar.autonomy.pdca_loop import Horizon, PDCALoop
from ai_sidecar.contracts.autonomy import GoalCategory, GoalStackState, SituationalAssessment
from ai_sidecar.contracts.common import ContractMeta
from ai_sidecar.contracts.crewai import (
    CrewAutonomyDecisionContext,
    CrewAutonomyDecisionOutput,
    CrewAutonomyRefinementResponse,
)
from ai_sidecar.contracts.state import BotStateSnapshot, CombatState, InventoryDigest, Position, ProgressionDigest, Vitals
from ai_sidecar.persistence.db import SQLiteDB
from ai_sidecar.persistence.repositories import create_repositories
from ai_sidecar.planner.schemas import PlannerResponse, TacticalIntentBundle


def _snapshot(
    *,
    bot_id: str = "bot:stage1",
    tick_id: str = "tick-1",
    hp: int = 900,
    hp_max: int = 1000,
    map_name: str = "prt_fild08",
    skill_points: int = 0,
    stat_points: int = 0,
    base_exp: int = 100,
    base_exp_max: int = 1000,
    job_exp: int = 200,
    job_exp_max: int = 1000,
    base_level: int = 45,
    job_level: int = 20,
    job_name: str | None = None,
    weight: int = 1200,
    weight_max: int = 8000,
) -> BotStateSnapshot:
    return BotStateSnapshot(
        meta=ContractMeta(contract_version="v1", source="pytest", bot_id=bot_id, trace_id=f"trace-{tick_id}"),
        tick_id=tick_id,
        observed_at=datetime.now(UTC),
        position=Position(map=map_name, x=100, y=100),
        vitals=Vitals(hp=hp, hp_max=hp_max, sp=100, sp_max=200, weight=weight, weight_max=weight_max),
        combat=CombatState(ai_sequence="route", target_id=None, is_in_combat=False),
        inventory=InventoryDigest(zeny=5000, item_count=42),
        progression=ProgressionDigest(
            job_id=None,
            job_name=job_name,
            base_level=base_level,
            job_level=job_level,
            base_exp=base_exp,
            base_exp_max=base_exp_max,
            job_exp=job_exp,
            job_exp_max=job_exp_max,
            skill_points=skill_points,
            stat_points=stat_points,
        ),
        raw={},
    )


@dataclass(slots=True)
class _SnapshotCache:
    snapshot: BotStateSnapshot

    def get(self, _bot_id: str) -> BotStateSnapshot:
        return self.snapshot


class _DecisionRuntime:
    def __init__(self, snapshot: BotStateSnapshot, enriched: dict[str, object]) -> None:
        self.snapshot_cache = _SnapshotCache(snapshot=snapshot)
        self._enriched = enriched
        self.persisted: list[GoalStackState] = []
        self.audits: list[dict[str, object]] = []
        self.runtime_events: list[dict[str, object]] = []
        self.autonomy_refine_calls: list[object] = []
        self.autonomy_refine_result: CrewAutonomyRefinementResponse | Exception | None = None

    def enriched_state(self, *, bot_id: str) -> dict[str, object]:
        del bot_id
        return self._enriched

    def persist_goal_state(self, *, bot_id: str, state: GoalStackState) -> None:
        del bot_id
        self.persisted.append(state)

    def _audit(self, *, level: str, event_type: str, summary: str, bot_id: str | None, payload: dict[str, object]) -> None:
        self.audits.append(
            {
                "level": level,
                "event_type": event_type,
                "summary": summary,
                "bot_id": bot_id,
                "payload": payload,
            }
        )

    def _emit_runtime_event(self, **kwargs: object) -> None:
        self.runtime_events.append(dict(kwargs))

    async def crewai_autonomy_refine_decision(self, payload):
        self.autonomy_refine_calls.append(payload)
        if isinstance(self.autonomy_refine_result, Exception):
            raise self.autonomy_refine_result
        return self.autonomy_refine_result


def test_decision_service_deterministic_priority_survival_first() -> None:
    snapshot = _snapshot(hp=150, hp_max=1000, skill_points=3, stat_points=2)
    enriched = {
        "operational": {
            "map": "prt_fild08",
            "in_combat": False,
            "base_level": 45,
            "job_level": 20,
            "skill_points": 3,
            "stat_points": 2,
            "base_exp": 300,
            "base_exp_max": 1000,
            "job_exp": 850,
            "job_exp_max": 1000,
        },
        "risk": {"danger_score": 0.81, "death_risk_score": 0.78},
        "quest": {"active_objective_count": 2, "objective_completion_ratio": 0.20},
        "inventory": {"overweight_ratio": 0.10, "item_count": 35, "zeny": 5000},
        "economy": {"vendor_exposure": 0},
    }
    runtime = _DecisionRuntime(snapshot=snapshot, enriched=enriched)
    service = DecisionService(runtime=runtime)

    state = service.decide(
        meta=ContractMeta(contract_version="v1", source="pytest", bot_id="bot:stage1", trace_id="trace-stage1"),
        horizon="short_term",
        replan_reasons=["stale_progress"],
    )

    assert state.selected_goal.goal_key == GoalCategory.survival
    assert [item.goal_key for item in state.goal_stack] == [
        GoalCategory.survival,
        GoalCategory.job_advancement,
        GoalCategory.opportunistic_upgrades,
        GoalCategory.leveling,
    ]
    assert runtime.persisted
    assert runtime.audits and runtime.audits[-1]["event_type"] == "autonomy_goal_decision"
    assert runtime.runtime_events


def test_decision_service_applies_crewai_refinement_without_overriding_goal() -> None:
    snapshot = _snapshot(hp=900, hp_max=1000, skill_points=3, stat_points=0)
    enriched = {
        "operational": {
            "map": "prt_fild08",
            "in_combat": False,
            "base_level": 45,
            "job_level": 20,
            "skill_points": 3,
            "stat_points": 0,
            "base_exp": 300,
            "base_exp_max": 1000,
            "job_exp": 900,
            "job_exp_max": 1000,
        },
        "risk": {"danger_score": 0.2, "death_risk_score": 0.2},
        "quest": {"active_objective_count": 1, "objective_completion_ratio": 0.2},
        "inventory": {"overweight_ratio": 0.10, "item_count": 35, "zeny": 5000},
        "economy": {"vendor_exposure": 0},
    }
    runtime = _DecisionRuntime(snapshot=snapshot, enriched=enriched)
    service_assessment = DecisionService(runtime=runtime)._build_assessment(
        bot_id="bot:stage2",
        snapshot=snapshot,
        enriched=enriched,
        replan_reasons=["stale_progress"],
    )
    computed = compute_goal_stack(
        assessment=service_assessment,
        horizon="short_term",
    )
    runtime.autonomy_refine_result = CrewAutonomyRefinementResponse(
        ok=True,
        message="autonomy_refined",
        trace_id="trace-stage2-refine",
        bot_id="bot:stage2",
        task_hint="autonomous_decision_intelligence",
        required_agents=["state_assessor", "progression_planner", "opportunistic_trader", "command_emitter"],
        decision_context=CrewAutonomyDecisionContext(
            horizon="short_term",
            assessment=service_assessment,
            selected_goal=computed.selected_goal,
            goal_stack=computed.goal_stack,
            deterministic_priority_order=[
                "survival",
                "job_advancement",
                "opportunistic_upgrades",
                "leveling",
            ],
            replan_reasons=["stale_progress"],
            required_agents=["state_assessor", "progression_planner", "opportunistic_trader", "command_emitter"],
        ),
        decision_output=CrewAutonomyDecisionOutput(
            selected_goal_key="survival",
            refined_objective="apply skill point upgrades safely at prt_fild08",
            situational_report="stable",
            execution_translation=["skills add", "stats add"],
            rationale="crew refinement",
            confidence=0.9,
            annotations={"note": "keep deterministic goal"},
        ),
    )
    service = DecisionService(runtime=runtime)

    state = service.decide(
        meta=ContractMeta(contract_version="v1", source="pytest", bot_id="bot:stage2", trace_id="trace-stage2-refine"),
        horizon="short_term",
        replan_reasons=["stale_progress"],
    )

    assert state.selected_goal.goal_key == GoalCategory.job_advancement
    assert state.selected_goal.objective == "apply skill point upgrades safely at prt_fild08"
    assert state.decision_version == "stage2-autonomy-crewai-refinement-v1"
    assert runtime.autonomy_refine_calls


def test_decision_service_fallback_when_crewai_refinement_unusable() -> None:
    snapshot = _snapshot(hp=900, hp_max=1000, skill_points=2, stat_points=0)
    enriched = {
        "operational": {
            "map": "prt_fild08",
            "in_combat": False,
            "base_level": 45,
            "job_level": 20,
            "skill_points": 2,
            "stat_points": 0,
            "base_exp": 300,
            "base_exp_max": 1000,
            "job_exp": 900,
            "job_exp_max": 1000,
        },
        "risk": {"danger_score": 0.1, "death_risk_score": 0.1},
        "quest": {"active_objective_count": 0, "objective_completion_ratio": 0.0},
        "inventory": {"overweight_ratio": 0.10, "item_count": 35, "zeny": 5000},
        "economy": {"vendor_exposure": 0},
    }
    runtime = _DecisionRuntime(snapshot=snapshot, enriched=enriched)
    runtime.autonomy_refine_result = RuntimeError("crewai unavailable")
    service = DecisionService(runtime=runtime)

    state = service.decide(
        meta=ContractMeta(contract_version="v1", source="pytest", bot_id="bot:stage2", trace_id="trace-stage2-fallback"),
        horizon="short_term",
        replan_reasons=["stale_progress"],
    )

    assert state.selected_goal.goal_key == GoalCategory.job_advancement
    assert state.decision_version == "stage1-deterministic-v1"


def test_decision_service_stage3_curated_job_advancement_ready() -> None:
    snapshot = _snapshot(
        hp=900,
        hp_max=1000,
        skill_points=0,
        stat_points=0,
        base_level=10,
        job_level=10,
        job_name="Novice",
    )
    enriched = {
        "operational": {
            "map": "prt_fild08",
            "in_combat": False,
            "job_name": "Novice",
            "base_level": 10,
            "job_level": 10,
            "skill_points": 0,
            "stat_points": 0,
            "base_exp": 300,
            "base_exp_max": 1000,
            "job_exp": 900,
            "job_exp_max": 1000,
        },
        "risk": {"danger_score": 0.1, "death_risk_score": 0.1},
        "quest": {"active_objective_count": 0, "objective_completion_ratio": 0.0},
        "inventory": {"overweight_ratio": 0.10, "item_count": 35, "zeny": 5000},
        "economy": {"vendor_exposure": 0},
    }
    runtime = _DecisionRuntime(snapshot=snapshot, enriched=enriched)
    service = DecisionService(runtime=runtime)

    state = service.decide(
        meta=ContractMeta(contract_version="v1", source="pytest", bot_id="bot:stage3-ready", trace_id="trace-stage3-ready"),
        horizon="short_term",
        replan_reasons=["stale_progress"],
    )

    assert state.selected_goal.goal_key == GoalCategory.job_advancement
    assert state.assessment.job_advancement.get("supported") is True
    assert state.assessment.job_advancement.get("ready") is True
    assert state.assessment.job_advancement.get("route_id") == "novice_to_swordman"
    assert "novice_to_swordman" in state.selected_goal.objective


def test_decision_service_stage3_fallback_to_leveling_when_route_unsupported() -> None:
    snapshot = _snapshot(
        hp=900,
        hp_max=1000,
        skill_points=0,
        stat_points=0,
        base_level=20,
        job_level=20,
        job_name="Thief",
    )
    enriched = {
        "operational": {
            "map": "moc_fild10",
            "in_combat": False,
            "job_name": "Thief",
            "base_level": 20,
            "job_level": 20,
            "skill_points": 0,
            "stat_points": 0,
            "base_exp": 300,
            "base_exp_max": 1000,
            "job_exp": 300,
            "job_exp_max": 1000,
        },
        "risk": {"danger_score": 0.1, "death_risk_score": 0.1},
        "quest": {"active_objective_count": 0, "objective_completion_ratio": 0.0},
        "inventory": {"overweight_ratio": 0.10, "item_count": 35, "zeny": 5000},
        "economy": {"vendor_exposure": 0},
    }
    runtime = _DecisionRuntime(snapshot=snapshot, enriched=enriched)
    service = DecisionService(runtime=runtime)

    state = service.decide(
        meta=ContractMeta(contract_version="v1", source="pytest", bot_id="bot:stage3-unsupported", trace_id="trace-stage3-unsupported"),
        horizon="short_term",
        replan_reasons=["stale_progress"],
    )

    assert state.selected_goal.goal_key == GoalCategory.leveling
    assert state.assessment.job_advancement.get("status") == "unsupported_job_route"
    job_goal = next(item for item in state.goal_stack if item.goal_key == GoalCategory.job_advancement)
    assert job_goal.active is False
    assert "unsupported_route" in job_goal.rationale


def test_decision_service_stage4_selects_opportunistic_when_actionable_and_higher_goals_inactive() -> None:
    snapshot = _snapshot(
        hp=920,
        hp_max=1000,
        skill_points=0,
        stat_points=0,
        base_level=9,
        job_level=9,
        job_name="Novice",
    )
    snapshot.inventory = InventoryDigest(zeny=8000, item_count=22)
    snapshot.inventory_items = [
        {
            "item_id": "sword_2",
            "name": "Sword [2]",
            "equipped": True,
            "category": "weapon",
            "metadata": {"slot": "weapon"},
        }
    ]

    enriched = {
        "operational": {
            "map": "prt_fild08",
            "in_combat": False,
            "job_name": "Novice",
            "base_level": 9,
            "job_level": 9,
            "skill_points": 0,
            "stat_points": 0,
            "base_exp": 300,
            "base_exp_max": 1000,
            "job_exp": 300,
            "job_exp_max": 1000,
        },
        "risk": {"danger_score": 0.1, "death_risk_score": 0.1},
        "quest": {"active_objective_count": 0, "objective_completion_ratio": 0.0},
        "inventory": {"overweight_ratio": 0.10, "item_count": 22, "zeny": 8000},
        "economy": {
            "vendor_exposure": 0,
            "market_listings": [
                {
                    "item_id": "sword_3",
                    "item_name": "Sword [3]",
                    "buy_price": 5500,
                    "source": "npc_shop",
                }
            ],
        },
    }
    runtime = _DecisionRuntime(snapshot=snapshot, enriched=enriched)
    service = DecisionService(runtime=runtime)

    state = service.decide(
        meta=ContractMeta(
            contract_version="v1",
            source="pytest",
            bot_id="bot:stage4-opportunistic",
            trace_id="trace-stage4-opportunistic",
        ),
        horizon="short_term",
        replan_reasons=["stale_progress"],
    )

    assert state.selected_goal.goal_key == GoalCategory.opportunistic_upgrades
    stage4 = state.assessment.opportunistic_upgrades
    assert stage4.get("actionable") is True
    assert stage4.get("status") == "actionable"
    assert stage4.get("recommended_opportunity", {}).get("rule_id") == "novice_weapon_sword_2_to_3"

    hints = state.selected_goal.metadata.get("execution_hints") if isinstance(state.selected_goal.metadata, dict) else []
    assert isinstance(hints, list) and hints
    assert isinstance(hints[0], dict)
    assert hints[0].get("execution_mode") == "direct"
    assert hints[0].get("tool") == "propose_actions"


def test_decision_service_wave_a1_exploration_domain_emits_safe_direct_hint() -> None:
    snapshot = _snapshot(
        hp=920,
        hp_max=1000,
        skill_points=0,
        stat_points=0,
        base_level=20,
        job_level=20,
        job_name="Thief",
    )
    snapshot.inventory = InventoryDigest(zeny=3000, item_count=16)

    enriched = {
        "operational": {
            "map": "prt_fild08",
            "in_combat": False,
                "job_name": "Thief",
            "base_level": 20,
            "job_level": 20,
            "skill_points": 0,
            "stat_points": 0,
            "base_exp": 300,
            "base_exp_max": 1000,
            "job_exp": 300,
            "job_exp_max": 1000,
        },
        "encounter": {"nearby_hostiles": 1},
        "risk": {"danger_score": 0.1, "death_risk_score": 0.1},
        "quest": {"active_objective_count": 0, "objective_completion_ratio": 0.0},
        "inventory": {"overweight_ratio": 0.10, "item_count": 16, "zeny": 3000},
        "economy": {"vendor_exposure": 0, "market_listings": []},
    }
    runtime = _DecisionRuntime(snapshot=snapshot, enriched=enriched)
    service = DecisionService(runtime=runtime)

    state = service.decide(
        meta=ContractMeta(
            contract_version="v1",
            source="pytest",
            bot_id="bot:wave-a1-exploration",
            trace_id="trace-wave-a1-exploration",
        ),
        horizon="short_term",
        replan_reasons=["explore_unseen_route"],
    )

    assert state.selected_goal.goal_key == GoalCategory.opportunistic_upgrades
    stage4 = state.assessment.opportunistic_upgrades
    assert stage4.get("actionable") is True
    assert stage4.get("recommended_opportunity", {}).get("domain") == "exploration"

    hints = state.selected_goal.metadata.get("execution_hints") if isinstance(state.selected_goal.metadata, dict) else []
    assert isinstance(hints, list) and hints
    assert hints[0].get("execution_mode") == "direct"
    assert hints[0].get("tool") == "propose_actions"


def test_decision_service_wave_a1_card_gear_domain_emits_safe_direct_hint() -> None:
    snapshot = _snapshot(
        hp=920,
        hp_max=1000,
        skill_points=0,
        stat_points=0,
        base_level=20,
        job_level=20,
        job_name="Thief",
    )
    snapshot.inventory = InventoryDigest(zeny=3000, item_count=22)

    enriched = {
        "operational": {
            "map": "prt_fild08",
            "in_combat": False,
                "job_name": "Thief",
            "base_level": 20,
            "job_level": 20,
            "skill_points": 0,
            "stat_points": 0,
            "base_exp": 300,
            "base_exp_max": 1000,
            "job_exp": 300,
            "job_exp_max": 1000,
        },
        "encounter": {"nearby_hostiles": 3},
        "risk": {"danger_score": 0.1, "death_risk_score": 0.1},
        "quest": {"active_objective_count": 0, "objective_completion_ratio": 0.0},
        "inventory": {"overweight_ratio": 0.10, "item_count": 22, "zeny": 3000},
        "economy": {
            "vendor_exposure": 0,
            "market_listings": [
                {
                    "item_id": "red_potion",
                    "item_name": "Red Potion",
                    "buy_price": 45,
                    "source": "npc_shop",
                }
            ],
        },
    }
    runtime = _DecisionRuntime(snapshot=snapshot, enriched=enriched)
    service = DecisionService(runtime=runtime)

    state = service.decide(
        meta=ContractMeta(
            contract_version="v1",
            source="pytest",
            bot_id="bot:wave-a1-card-gear",
            trace_id="trace-wave-a1-card-gear",
        ),
        horizon="short_term",
        replan_reasons=["card_farm_rotation"],
    )

    assert state.selected_goal.goal_key == GoalCategory.opportunistic_upgrades
    stage4 = state.assessment.opportunistic_upgrades
    assert stage4.get("actionable") is True
    assert stage4.get("recommended_opportunity", {}).get("domain") == "card_gear_farming"

    hints = state.selected_goal.metadata.get("execution_hints") if isinstance(state.selected_goal.metadata, dict) else []
    assert isinstance(hints, list) and hints
    assert hints[0].get("execution_mode") == "direct"
    assert hints[0].get("tool") == "propose_actions"


def test_decision_service_wave_a1_mercenary_homunculus_domain_emits_macro_hint() -> None:
    snapshot = _snapshot(
        hp=920,
        hp_max=1000,
        skill_points=0,
        stat_points=0,
        base_level=20,
        job_level=20,
        job_name="Thief",
    )
    snapshot.inventory = InventoryDigest(zeny=3000, item_count=20)
    snapshot.raw = {"homunculus_active": True}

    enriched = {
        "operational": {
            "map": "prt_fild08",
            "in_combat": False,
                "job_name": "Thief",
            "base_level": 20,
            "job_level": 20,
            "skill_points": 0,
            "stat_points": 0,
            "base_exp": 300,
            "base_exp_max": 1000,
            "job_exp": 300,
            "job_exp_max": 1000,
        },
        "encounter": {"nearby_hostiles": 0},
        "risk": {"danger_score": 0.1, "death_risk_score": 0.1},
        "quest": {"active_objective_count": 0, "objective_completion_ratio": 0.0},
        "inventory": {"overweight_ratio": 0.10, "item_count": 20, "zeny": 3000},
        "economy": {"vendor_exposure": 0, "market_listings": []},
    }
    runtime = _DecisionRuntime(snapshot=snapshot, enriched=enriched)
    service = DecisionService(runtime=runtime)

    state = service.decide(
        meta=ContractMeta(
            contract_version="v1",
            source="pytest",
            bot_id="bot:wave-a1-merc-homu",
            trace_id="trace-wave-a1-merc-homu",
        ),
        horizon="short_term",
        replan_reasons=["homunculus_support_window"],
    )

    assert state.selected_goal.goal_key == GoalCategory.opportunistic_upgrades
    stage4 = state.assessment.opportunistic_upgrades
    assert stage4.get("actionable") is True
    assert stage4.get("recommended_opportunity", {}).get("domain") == "mercenary_homunculus"

    hints = state.selected_goal.metadata.get("execution_hints") if isinstance(state.selected_goal.metadata, dict) else []
    assert isinstance(hints, list) and hints
    assert hints[0].get("execution_mode") == "macro"
    assert hints[0].get("tool") == "publish_macro"


def test_decision_service_wave_a1_vending_domain_emits_config_hint() -> None:
    snapshot = _snapshot(
        hp=920,
        hp_max=1000,
        skill_points=0,
        stat_points=0,
        base_level=20,
        job_level=20,
        job_name="Thief",
    )
    snapshot.inventory = InventoryDigest(zeny=3000, item_count=24)

    enriched = {
        "operational": {
            "map": "prt_fild08",
            "in_combat": False,
                "job_name": "Thief",
            "base_level": 20,
            "job_level": 20,
            "skill_points": 0,
            "stat_points": 0,
            "base_exp": 300,
            "base_exp_max": 1000,
            "job_exp": 300,
            "job_exp_max": 1000,
        },
        "encounter": {"nearby_hostiles": 0},
        "risk": {"danger_score": 0.1, "death_risk_score": 0.1},
        "quest": {"active_objective_count": 0, "objective_completion_ratio": 0.0},
        "inventory": {"overweight_ratio": 0.50, "item_count": 24, "zeny": 3000},
        "economy": {"vendor_exposure": 2, "market_listings": []},
    }
    runtime = _DecisionRuntime(snapshot=snapshot, enriched=enriched)
    service = DecisionService(runtime=runtime)

    state = service.decide(
        meta=ContractMeta(
            contract_version="v1",
            source="pytest",
            bot_id="bot:wave-a1-vending",
            trace_id="trace-wave-a1-vending",
        ),
        horizon="short_term",
        replan_reasons=["vending_cycle"],
    )

    assert state.selected_goal.goal_key == GoalCategory.opportunistic_upgrades
    stage4 = state.assessment.opportunistic_upgrades
    assert stage4.get("actionable") is True
    assert stage4.get("recommended_opportunity", {}).get("domain") == "vending"

    hints = state.selected_goal.metadata.get("execution_hints") if isinstance(state.selected_goal.metadata, dict) else []
    assert isinstance(hints, list) and hints
    assert hints[0].get("execution_mode") == "config"
    assert hints[0].get("tool") == "plan_control_change"


def test_goal_state_persistence_roundtrip(tmp_path) -> None:
    db = SQLiteDB(path=tmp_path / "sidecar-stage1.sqlite", busy_timeout_ms=250)
    db.initialize()
    repos = create_repositories(
        db=db,
        snapshot_history_per_bot=32,
        telemetry_max_per_bot=256,
        telemetry_operational_window_minutes=15,
        audit_history=128,
    )

    assessment = SituationalAssessment(
        bot_id="bot:persist",
        tick_id="tick-persist",
        map_name="prt_fild08",
        hp_ratio=0.95,
        danger_score=0.10,
        death_risk_score=0.10,
        skill_points=2,
        stat_points=1,
        base_level=45,
        job_level=20,
        base_exp_ratio=0.30,
        job_exp_ratio=0.88,
        active_quest_count=1,
        objective_completion_ratio=0.2,
        overweight_ratio=0.15,
        item_count=40,
        zeny=5000,
        vendor_exposure=0,
    )
    computed = compute_goal_stack(assessment=assessment, horizon="medium_term")
    state = GoalStackState(
        bot_id="bot:persist",
        tick_id="tick-persist",
        horizon="medium_term",
        assessment=assessment,
        goal_stack=computed.goal_stack,
        selected_goal=computed.selected_goal,
    )

    repos.autonomy_goals.upsert(state)
    loaded = repos.autonomy_goals.latest(bot_id="bot:persist")

    assert loaded is not None
    assert loaded.bot_id == "bot:persist"
    assert loaded.tick_id == "tick-persist"
    assert loaded.selected_goal.goal_key == GoalCategory.job_advancement
    assert len(loaded.goal_stack) == 4


class _PDCAStage1Runtime:
    def __init__(self, selected_goal_state: GoalStackState) -> None:
        self.snapshot_cache = _SnapshotCache(
            snapshot=_snapshot(bot_id="bot:pdca", tick_id="tick-pdca", hp=900, hp_max=1000),
        )
        self.autonomy_policy = {
            "ranked_objectives": ["grind", "recovery", "economy", "quest"],
            "objective_rotation_cooldown_s": 0.0,
            "reconnect_grace_s": 20.0,
        }
        self.fleet_constraint_state = type(
            "FleetState",
            (),
            {
                "status": lambda self: {
                    "mode": "local",
                    "central_enabled": False,
                    "central_available": False,
                    "stale": False,
                    "last_sync_at": None,
                    "doctrine_version": "local",
                    "last_error": "disabled",
                }
            },
        )()
        self._goal_state = selected_goal_state
        self._latest_goal_state = selected_goal_state
        self.raise_autonomy_error = False
        self.return_autonomy_none = False
        self.autonomy_calls: list[dict[str, object]] = []
        self.latest_goal_state_calls: list[str] = []
        self.planner_calls: list[object] = []

    def list_bots(self) -> list[dict[str, object]]:
        return [{"bot_id": "bot:pdca"}]

    def autonomy_decide(self, *, meta: ContractMeta, horizon: str, replan_reasons: list[str] | None = None) -> GoalStackState:
        self.autonomy_calls.append(
            {
                "bot_id": meta.bot_id,
                "horizon": horizon,
                "replan_reasons": list(replan_reasons or []),
            }
        )
        if self.raise_autonomy_error:
            raise RuntimeError("forced_autonomy_decision_failure")
        if self.return_autonomy_none:
            return None  # type: ignore[return-value]
        return self._goal_state

    def latest_goal_state(self, *, bot_id: str) -> GoalStackState | None:
        self.latest_goal_state_calls.append(bot_id)
        return self._latest_goal_state

    async def planner_plan(self, payload) -> PlannerResponse:
        self.planner_calls.append(payload)
        return PlannerResponse(
            ok=True,
            message="ok",
            trace_id=payload.meta.trace_id,
            tactical_bundle=TacticalIntentBundle(
                bundle_id="bundle-stage1",
                bot_id=payload.meta.bot_id,
                intents=[],
                actions=[],
                notes=[],
            ),
            provider="pytest",
            model="local",
            latency_ms=1.0,
        )

    def queue_action(self, proposal, _bot_id: str):
        from ai_sidecar.contracts.actions import ActionStatus

        return True, ActionStatus.queued, proposal.action_id, "queued"


def test_pdca_uses_runtime_autonomy_decision_boundary() -> None:
    assessment = SituationalAssessment(
        bot_id="bot:pdca",
        tick_id="tick-pdca",
        map_name="prt_fild08",
        hp_ratio=0.20,
        danger_score=0.8,
        death_risk_score=0.8,
        skill_points=0,
        stat_points=0,
        base_level=45,
        job_level=20,
        base_exp_ratio=0.3,
        job_exp_ratio=0.3,
        active_quest_count=0,
        objective_completion_ratio=0.0,
        overweight_ratio=0.1,
        item_count=35,
        zeny=5000,
        vendor_exposure=0,
    )
    computed = compute_goal_stack(assessment=assessment, horizon="short_term")
    goal_state = GoalStackState(
        bot_id="bot:pdca",
        tick_id="tick-pdca",
        horizon="short_term",
        assessment=assessment,
        goal_stack=computed.goal_stack,
        selected_goal=computed.selected_goal,
    )
    runtime = _PDCAStage1Runtime(selected_goal_state=goal_state)
    pdca = PDCALoop(runtime_state=runtime)

    result = asyncio.run(pdca._run_one_cycle(Horizon.SHORT_TERM))

    assert runtime.autonomy_calls
    assert runtime.autonomy_calls[-1]["horizon"] == "short_term"
    assert runtime.planner_calls
    assert runtime.planner_calls[-1].objective == goal_state.selected_goal.objective
    assert result.selected_goal == goal_state.selected_goal.goal_key.value
    assert result.objective == goal_state.selected_goal.objective


def test_pdca_falls_back_to_latest_goal_state_when_autonomy_decision_fails() -> None:
    assessment = SituationalAssessment(
        bot_id="bot:pdca-fallback",
        tick_id="tick-pdca-fallback",
        map_name="prt_fild08",
        hp_ratio=0.85,
        danger_score=0.2,
        death_risk_score=0.2,
        skill_points=1,
        stat_points=0,
        base_level=45,
        job_level=20,
        base_exp_ratio=0.3,
        job_exp_ratio=0.3,
        active_quest_count=0,
        objective_completion_ratio=0.0,
        overweight_ratio=0.1,
        item_count=35,
        zeny=5000,
        vendor_exposure=0,
    )
    computed = compute_goal_stack(assessment=assessment, horizon="short_term")
    goal_state = GoalStackState(
        bot_id="bot:pdca",
        tick_id="tick-pdca-fallback",
        horizon="short_term",
        assessment=assessment,
        goal_stack=computed.goal_stack,
        selected_goal=computed.selected_goal,
    )

    runtime = _PDCAStage1Runtime(selected_goal_state=goal_state)
    runtime.raise_autonomy_error = True
    pdca = PDCALoop(runtime_state=runtime)

    result = asyncio.run(pdca._run_one_cycle(Horizon.SHORT_TERM))

    assert runtime.autonomy_calls
    assert runtime.latest_goal_state_calls == ["bot:pdca"]
    assert runtime.planner_calls
    assert runtime.planner_calls[-1].objective == goal_state.selected_goal.objective
    assert result.selected_goal == goal_state.selected_goal.goal_key.value
    assert result.objective == goal_state.selected_goal.objective
