from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable

import pytest

from ai_sidecar.autonomy.pdca_loop import Horizon, PDCALoop
from ai_sidecar.config import settings
from ai_sidecar.contracts.autonomy import GoalStackState
from ai_sidecar.contracts.common import ContractMeta
from ai_sidecar.contracts.state import (
    BotRegistrationRequest,
    BotStateSnapshot,
    CombatState,
    InventoryDigest,
    Position,
    ProgressionDigest,
    Vitals,
)
from ai_sidecar.lifecycle import create_runtime
from ai_sidecar.lifecycle import RuntimeState
from ai_sidecar.planner.context_assembler import PlannerContextAssembler
from ai_sidecar.planner.schemas import PlanHorizon


def _configure_isolated_runtime(monkeypatch, tmp_path, *, crewai_enabled: bool) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(settings, "sqlite_path", str(tmp_path / "sidecar.sqlite"))
    monkeypatch.setattr(settings, "memory_openmemory_path", str(tmp_path / "openmemory.sqlite"))
    monkeypatch.setattr(settings, "provider_ollama_enabled", False)
    monkeypatch.setattr(settings, "provider_openai_enabled", False)
    monkeypatch.setattr(settings, "provider_deepseek_enabled", False)
    monkeypatch.setattr(settings, "provider_policy_json", "")
    monkeypatch.setattr(settings, "fleet_central_enabled", False)
    monkeypatch.setattr(settings, "crewai_enabled", crewai_enabled)
    monkeypatch.setattr(settings, "crewai_memory_enabled", False)


def _snapshot(
    *,
    bot_id: str,
    tick_id: str,
    hp: int = 900,
    hp_max: int = 1000,
    sp: int = 90,
    sp_max: int = 200,
    map_name: str = "prt_fild08",
    x: int = 120,
    y: int = 88,
    ai_sequence: str = "route",
    zeny: int = 1200,
    item_count: int = 12,
    base_level: int = 30,
    job_level: int = 12,
    base_exp: int = 100,
    base_exp_max: int = 1000,
    job_exp: int = 200,
    job_exp_max: int = 1000,
    skill_points: int = 0,
    stat_points: int = 0,
    job_name: str | None = None,
    inventory_items: list[dict[str, object]] | None = None,
    market_listings: list[dict[str, object]] | None = None,
    raw: dict[str, object] | None = None,
) -> BotStateSnapshot:
    return BotStateSnapshot(
        meta=ContractMeta(contract_version="v1", source="pytest", bot_id=bot_id, trace_id=f"trace-{tick_id}"),
        tick_id=tick_id,
        observed_at=datetime.now(UTC),
        position=Position(map=map_name, x=x, y=y),
        vitals=Vitals(hp=hp, hp_max=hp_max, sp=sp, sp_max=sp_max, weight=1200, weight_max=8000),
        combat=CombatState(ai_sequence=ai_sequence, target_id=None, is_in_combat=False),
        inventory=InventoryDigest(zeny=zeny, item_count=item_count),
        inventory_items=list(inventory_items or []),
        progression=ProgressionDigest(
            base_level=base_level,
            job_level=job_level,
            base_exp=base_exp,
            base_exp_max=base_exp_max,
            job_exp=job_exp,
            job_exp_max=job_exp_max,
            skill_points=skill_points,
            stat_points=stat_points,
            job_name=job_name,
        ),
        market={"listings": list(market_listings or [])},
        raw=dict(raw or {}),
    )


@dataclass(slots=True)
class _StartupScenario:
    name: str
    snapshot_kwargs: dict[str, object]
    expected_goal: str
    expected_commands: tuple[str, ...]
    expected_conflict_keys: tuple[str, ...]
    expected_fallback_modes: tuple[str, ...]
    objective_contains: tuple[str, ...] = ()
    assert_state: Callable[[GoalStackState], None] | None = None


def _assert_stage3_job_advancement_ready(goal_state: GoalStackState) -> None:
    advancement = goal_state.assessment.job_advancement
    assert advancement.get("supported") is True
    assert advancement.get("ready") is True
    assert advancement.get("route_id") == "novice_to_swordman"
    assert "novice_to_swordman" in goal_state.selected_goal.objective


def _assert_stage4_opportunistic_actionable(goal_state: GoalStackState) -> None:
    stage4 = goal_state.assessment.opportunistic_upgrades
    assert stage4.get("supported") is True
    assert stage4.get("actionable") is True
    assert stage4.get("status") == "actionable"
    assert stage4.get("recommended_opportunity", {}).get("rule_id") == "novice_weapon_sword_2_to_3"

    metadata = goal_state.selected_goal.metadata if isinstance(goal_state.selected_goal.metadata, dict) else {}
    execution_hints = metadata.get("execution_hints") if isinstance(metadata.get("execution_hints"), list) else []
    assert execution_hints
    assert isinstance(execution_hints[0], dict)
    assert execution_hints[0].get("tool") == "propose_actions"
    assert execution_hints[0].get("execution_mode") == "direct"


_STARTUP_SCENARIOS: list[_StartupScenario] = [
    _StartupScenario(
        name="idle_safe_town",
        snapshot_kwargs={
            "map_name": "prontera",
            "base_level": 45,
            "job_level": 20,
            "job_name": "Swordsman",
        },
        expected_goal="leveling",
        expected_commands=("move random_walk_seek",),
        expected_conflict_keys=("planner.seek.random_walk",),
        expected_fallback_modes=("seek_targets",),
        objective_contains=("grind",),
    ),
    _StartupScenario(
        name="field_grind_continuation",
        snapshot_kwargs={
            "map_name": "prt_fild08",
            "base_level": 45,
            "job_level": 20,
            "job_name": "Swordsman",
        },
        expected_goal="leveling",
        expected_commands=("move random_walk_seek",),
        expected_conflict_keys=("planner.seek.random_walk",),
        expected_fallback_modes=("seek_targets",),
        objective_contains=("grind",),
    ),
    _StartupScenario(
        name="dead_recovery",
        snapshot_kwargs={
            "hp": 0,
            "hp_max": 1000,
            "map_name": "prt_fild08",
            "base_level": 45,
            "job_level": 20,
            "job_name": "Swordsman",
            "raw": {"death_count": 1, "respawn_state": "dead"},
        },
        expected_goal="survival",
        expected_commands=("respawn",),
        expected_conflict_keys=("recovery.death",),
        expected_fallback_modes=("death_recovery",),
        objective_contains=("stabilize survival posture",),
    ),
    _StartupScenario(
        name="job_advancement_ready",
        snapshot_kwargs={
            "map_name": "izlude_in",
            "base_level": 10,
            "job_level": 10,
            "job_name": "Novice",
            "skill_points": 0,
            "stat_points": 0,
        },
        expected_goal="job_advancement",
        expected_commands=("move prt_fild08", "move random_walk_seek"),
        expected_conflict_keys=("nav.resume_grind", "planner.seek.random_walk"),
        expected_fallback_modes=("resume_grind", "seek_targets"),
        objective_contains=("novice_to_swordman",),
        assert_state=_assert_stage3_job_advancement_ready,
    ),
    _StartupScenario(
        name="opportunistic_upgrade_actionable",
        snapshot_kwargs={
            "map_name": "prt_fild08",
            "base_level": 9,
            "job_level": 9,
            "job_name": "Novice",
            "skill_points": 0,
            "stat_points": 0,
            "zeny": 8000,
            "item_count": 22,
            "inventory_items": [
                {
                    "item_id": "sword_2",
                    "name": "Sword [2]",
                    "equipped": True,
                    "category": "weapon",
                    "metadata": {"slot": "weapon"},
                }
            ],
            "market_listings": [
                {
                    "item_id": "sword_3",
                    "item_name": "Sword [3]",
                    "buy_price": 5500,
                    "source": "npc_shop",
                }
            ],
        },
        expected_goal="opportunistic_upgrades",
        expected_commands=("move random_walk_seek",),
        expected_conflict_keys=("planner.seek.random_walk",),
        expected_fallback_modes=("seek_targets",),
        objective_contains=("curated opportunistic",),
        assert_state=_assert_stage4_opportunistic_actionable,
    ),
]


def test_lifespan_cold_start_pdca_autonomy_reaches_dispatch(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _configure_isolated_runtime(monkeypatch, tmp_path, crewai_enabled=False)
    from ai_sidecar import app as app_module

    planner_payloads: list[object] = []
    crewai_payloads: list[object] = []
    assembled_contexts: list[object] = []

    async def _exercise() -> None:
        app = app_module.create_app()
        async with app.router.lifespan_context(app):
            runtime = app.state.runtime
            pdca = runtime.pdca_loop
            assert pdca is not None
            assert pdca.running is True
            await pdca.stop()
            assert pdca.running is False

            original_assemble = PlannerContextAssembler.assemble

            def _capture_assemble(self, *, meta, objective, horizon, event_limit=64, memory_limit=8):  # type: ignore[no-untyped-def]
                context = original_assemble(
                    self,
                    meta=meta,
                    objective=objective,
                    horizon=horizon,
                    event_limit=event_limit,
                    memory_limit=memory_limit,
                )
                assembled_contexts.append(context)
                return context

            monkeypatch.setattr(PlannerContextAssembler, "assemble", _capture_assemble)

            original_planner_plan = RuntimeState.planner_plan

            async def _capture_planner(self, payload):  # type: ignore[no-untyped-def]
                planner_payloads.append(payload)
                return await original_planner_plan(self, payload)

            monkeypatch.setattr(RuntimeState, "planner_plan", _capture_planner)

            original_crewai_strategize = RuntimeState.crewai_strategize

            async def _capture_crewai(self, payload):  # type: ignore[no-untyped-def]
                crewai_payloads.append(payload)
                return await original_crewai_strategize(self, payload)

            monkeypatch.setattr(RuntimeState, "crewai_strategize", _capture_crewai)

            bot_id = "botcoldstart"
            runtime.register_bot(
                BotRegistrationRequest(
                    meta=ContractMeta(contract_version="v1", source="pytest", bot_id=bot_id, trace_id="trace-register"),
                )
            )
            runtime.ingest_snapshot(_snapshot(bot_id=bot_id, tick_id="cold-start-snap-1", map_name="prt_fild08"))

            enriched = runtime.enriched_state(bot_id=bot_id)
            deadline = time.monotonic() + 2.0
            while enriched.navigation.map != "prt_fild08" and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
                enriched = runtime.enriched_state(bot_id=bot_id)
            assert enriched.navigation.map == "prt_fild08"

            for horizon in Horizon:
                pdca._active_plan[horizon] = None
                pdca._last_plan_time[horizon] = 0.0

            short_result = await pdca._run_one_cycle(Horizon.SHORT_TERM)
            long_result = await pdca._run_one_cycle(Horizon.LONG_TERM)

            assert short_result.re_planned is True
            assert short_result.actions_queued >= 1
            assert long_result.error is None

            assert any(getattr(item.meta, "bot_id", "") == bot_id for item in crewai_payloads)
            assert any(getattr(item.meta, "bot_id", "") == bot_id for item in planner_payloads)
            assert any(
                getattr(item.meta, "bot_id", "") == bot_id and getattr(item, "horizon", None) == PlanHorizon.strategic
                for item in planner_payloads
            )
            assert any(
                getattr(ctx, "bot_id", "") == bot_id
                and isinstance(getattr(ctx, "state", None), dict)
                and (
                    str(ctx.state.get("navigation", {}).get("map") or "") == "prt_fild08"
                    or str(ctx.state.get("operational", {}).get("map") or "") == "prt_fild08"
                )
                for ctx in assembled_contexts
            )

            assert runtime.action_queue.count(bot_id) > 0
            dispatched = runtime.next_action(bot_id, poll_id="poll-cold-start")
            assert dispatched is not None
            assert dispatched.command != ""

    asyncio.run(_exercise())


@pytest.mark.parametrize("scenario", _STARTUP_SCENARIOS, ids=lambda item: item.name)
def test_pdca_startup_state_matrix_end_to_end(monkeypatch, tmp_path, scenario: _StartupScenario) -> None:  # type: ignore[no-untyped-def]
    _configure_isolated_runtime(monkeypatch, tmp_path, crewai_enabled=False)
    runtime = create_runtime()

    planner_payloads: list[object] = []
    assembled_contexts: list[object] = []
    autonomy_goal_states: list[GoalStackState] = []

    try:
        original_assemble = PlannerContextAssembler.assemble

        def _capture_assemble(self, *, meta, objective, horizon, event_limit=64, memory_limit=8):  # type: ignore[no-untyped-def]
            context = original_assemble(
                self,
                meta=meta,
                objective=objective,
                horizon=horizon,
                event_limit=event_limit,
                memory_limit=memory_limit,
            )
            assembled_contexts.append(context)
            return context

        monkeypatch.setattr(PlannerContextAssembler, "assemble", _capture_assemble)

        original_autonomy_decide = RuntimeState.autonomy_decide

        def _capture_autonomy(self, *, meta, horizon, replan_reasons=None):  # type: ignore[no-untyped-def]
            state = original_autonomy_decide(
                self,
                meta=meta,
                horizon=horizon,
                replan_reasons=replan_reasons,
            )
            if state is not None:
                autonomy_goal_states.append(state)
            return state

        monkeypatch.setattr(RuntimeState, "autonomy_decide", _capture_autonomy)

        original_planner_plan = RuntimeState.planner_plan

        async def _capture_planner(self, payload):  # type: ignore[no-untyped-def]
            planner_payloads.append(payload)
            return await original_planner_plan(self, payload)

        monkeypatch.setattr(RuntimeState, "planner_plan", _capture_planner)

        bot_id = f"botstartup-{scenario.name}"
        runtime.register_bot(
            BotRegistrationRequest(
                meta=ContractMeta(contract_version="v1", source="pytest", bot_id=bot_id, trace_id=f"trace-register-{scenario.name}"),
            )
        )

        runtime.ingest_snapshot(
            _snapshot(
                bot_id=bot_id,
                tick_id=f"{scenario.name}-snap-1",
                **scenario.snapshot_kwargs,
            )
        )

        expected_map = str(scenario.snapshot_kwargs.get("map_name") or "")
        enriched = runtime.enriched_state(bot_id=bot_id)
        deadline = time.monotonic() + 2.0
        while str(enriched.navigation.map or "") != expected_map and time.monotonic() < deadline:
            time.sleep(0.01)
            enriched = runtime.enriched_state(bot_id=bot_id)
        assert str(enriched.navigation.map or "") == expected_map

        pdca = PDCALoop(runtime_state=runtime)
        result = asyncio.run(pdca._run_one_cycle(Horizon.SHORT_TERM))

        assert result.re_planned is True
        assert result.actions_queued >= 1
        assert result.selected_goal == scenario.expected_goal

        assert autonomy_goal_states
        decided = autonomy_goal_states[-1]
        assert decided.selected_goal.goal_key.value == scenario.expected_goal
        assert result.objective == decided.selected_goal.objective
        for token in scenario.objective_contains:
            assert token in decided.selected_goal.objective
        if scenario.assert_state is not None:
            scenario.assert_state(decided)

        assert planner_payloads
        planner_payload = planner_payloads[-1]
        assert planner_payload.meta.bot_id == bot_id
        assert planner_payload.objective == decided.selected_goal.objective

        assert any(
            getattr(ctx, "bot_id", "") == bot_id
            and isinstance(getattr(ctx, "selected_goal", None), dict)
            and str(ctx.selected_goal.get("goal_key") or "") == scenario.expected_goal
            and str(ctx.selected_goal.get("objective") or "") == decided.selected_goal.objective
            for ctx in assembled_contexts
        )

        assert runtime.action_queue.count(bot_id) > 0
        action = runtime.next_action(bot_id, poll_id=f"poll-{scenario.name}")
        assert action is not None
        assert action.command in scenario.expected_commands
        assert action.conflict_key in scenario.expected_conflict_keys
        assert action.metadata.get("fallback_mode") in scenario.expected_fallback_modes
    finally:
        asyncio.run(runtime.shutdown())


def test_pdca_cold_start_dead_snapshot_queues_respawn(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _configure_isolated_runtime(monkeypatch, tmp_path, crewai_enabled=False)
    runtime = create_runtime()

    try:
        bot_id = "botcoldstartdead"
        runtime.register_bot(
            BotRegistrationRequest(
                meta=ContractMeta(contract_version="v1", source="pytest", bot_id=bot_id, trace_id="trace-register-dead"),
            )
        )

        runtime.ingest_snapshot(
            _snapshot(
                bot_id=bot_id,
                tick_id="dead-snap-1",
                hp=0,
                hp_max=1000,
                map_name="prt_fild08",
                raw={"death_count": 1, "respawn_state": "dead"},
            )
        )

        enriched = runtime.enriched_state(bot_id=bot_id)
        deadline = time.monotonic() + 2.0
        while enriched.operational.hp != 0 and time.monotonic() < deadline:
            time.sleep(0.01)
            enriched = runtime.enriched_state(bot_id=bot_id)
        assert enriched.operational.hp == 0

        pdca = PDCALoop(runtime_state=runtime)
        result = asyncio.run(pdca._run_one_cycle(Horizon.SHORT_TERM))

        assert result.re_planned is True
        assert result.actions_queued >= 1

        action = runtime.next_action(bot_id, poll_id="poll-death-recovery")
        assert action is not None
        assert action.command == "respawn"
        assert action.conflict_key == "recovery.death"
        assert action.metadata.get("fallback_mode") == "death_recovery"
    finally:
        asyncio.run(runtime.shutdown())


def test_create_runtime_restores_persisted_goal_state_on_restart(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _configure_isolated_runtime(monkeypatch, tmp_path, crewai_enabled=False)

    bot_id = "bot-persist-restart"
    runtime1 = create_runtime()
    try:
        runtime1.register_bot(
            BotRegistrationRequest(
                meta=ContractMeta(contract_version="v1", source="pytest", bot_id=bot_id, trace_id="trace-register-restart"),
            )
        )

        decided = runtime1.autonomy_decide(
            meta=ContractMeta(contract_version="v1", source="pytest", bot_id=bot_id, trace_id="trace-decide-restart"),
            horizon="short_term",
            replan_reasons=["cold_start_bootstrap"],
        )
        assert decided is not None
    finally:
        asyncio.run(runtime1.shutdown())

    runtime2 = create_runtime()
    try:
        restored = runtime2.latest_goal_state(bot_id=bot_id)
        assert restored is not None
        assert restored.bot_id == bot_id
        assert restored.selected_goal.goal_key.value == decided.selected_goal.goal_key.value
        assert restored.selected_goal.objective == decided.selected_goal.objective

        # Validate registration-path restore still works when cache is cold.
        runtime2._last_goal_state_by_bot.clear()
        runtime2.register_bot(
            BotRegistrationRequest(
                meta=ContractMeta(contract_version="v1", source="pytest", bot_id=bot_id, trace_id="trace-register-restart-2"),
            )
        )
        assert bot_id in runtime2._last_goal_state_by_bot
    finally:
        asyncio.run(runtime2.shutdown())


def test_persistence_degraded_self_heals_after_subsequent_success(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _configure_isolated_runtime(monkeypatch, tmp_path, crewai_enabled=False)
    runtime = create_runtime()

    try:
        assert runtime.repositories is not None
        original_upsert = runtime.repositories.bots.upsert_registration
        calls = {"n": 0}

        def _fail_once(*args, **kwargs):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("forced_persistence_failure")
            return original_upsert(*args, **kwargs)

        monkeypatch.setattr(runtime.repositories.bots, "upsert_registration", _fail_once)

        runtime.register_bot(
            BotRegistrationRequest(
                meta=ContractMeta(contract_version="v1", source="pytest", bot_id="bot-heal-1", trace_id="trace-heal-1"),
            )
        )

        # First persist call fails, later safe persistence calls recover and clear degraded flag.
        assert calls["n"] >= 1
        assert runtime.persistence_degraded is False
    finally:
        asyncio.run(runtime.shutdown())
