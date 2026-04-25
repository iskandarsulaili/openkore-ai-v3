from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from ai_sidecar.autonomy.pdca_loop import Horizon, PDCALoop
from ai_sidecar.autonomy.progress_tracker import ProgressEvaluation, ProgressTracker
from ai_sidecar.contracts.actions import ActionStatus
from ai_sidecar.contracts.common import ContractMeta
from ai_sidecar.contracts.crewai import CrewStrategizeResponse
from ai_sidecar.contracts.state import BotStateSnapshot, CombatState, InventoryDigest, Position, ProgressionDigest, Vitals
from ai_sidecar.planner.schemas import PlanHorizon, PlannerResponse, StrategicPlan, TacticalIntentBundle


def _snapshot(
    *,
    bot_id: str = "bot:autonomy",
    tick_id: str,
    hp: int = 900,
    hp_max: int = 1000,
    map_name: str = "prt_fild08",
    x: int = 100,
    y: int = 100,
    ai_sequence: str = "route",
    zeny: int = 5000,
    base_exp: int = 0,
    job_exp: int = 0,
    weight: int = 1000,
    weight_max: int = 8000,
    raw: dict[str, object] | None = None,
) -> BotStateSnapshot:
    return BotStateSnapshot(
        meta=ContractMeta(contract_version="v1", source="pytest", bot_id=bot_id, trace_id=f"trace-{tick_id}"),
        tick_id=tick_id,
        observed_at=datetime.now(UTC),
        position=Position(map=map_name, x=x, y=y),
        vitals=Vitals(hp=hp, hp_max=hp_max, sp=120, sp_max=200, weight=weight, weight_max=weight_max),
        combat=CombatState(ai_sequence=ai_sequence, target_id=None, is_in_combat=False),
        inventory=InventoryDigest(zeny=zeny, item_count=42),
        progression=ProgressionDigest(base_level=45, job_level=20, base_exp=base_exp, job_exp=job_exp),
        raw=dict(raw or {}),
    )


@dataclass(slots=True)
class _ProgressRuntime:
    autonomy_policy: dict[str, object] = field(
        default_factory=lambda: {
            "stale_plan_threshold_s": 60.0,
            "objective_max_age_cycles": 6,
            "death_recovery_cooldown_s": 15.0,
        }
    )


def test_progress_tracker_detects_map_dwell_and_route_churn() -> None:
    tracker = ProgressTracker(_ProgressRuntime())
    plan = TacticalIntentBundle(bundle_id="bundle-a", bot_id="bot:autonomy", intents=[], actions=[], notes=[])

    result = ProgressEvaluation()
    for idx in range(1, 5):
        result = tracker.evaluate(
            horizon=Horizon.SHORT_TERM,
            active_plan=plan,
            snapshot=_snapshot(tick_id=f"tick-{idx}", x=100, y=100, ai_sequence="route"),
        )

    assert result.route_churn_cycles >= 2
    assert result.map_dwell_cycles >= 3
    assert "route_churn_no_position_gain" in result.reasons
    assert "map_dwell_no_gain" in result.reasons
    assert result.force_replan_hint is True


def test_progress_tracker_detects_death_loop() -> None:
    tracker = ProgressTracker(_ProgressRuntime())
    plan = TacticalIntentBundle(bundle_id="bundle-b", bot_id="bot:autonomy", intents=[], actions=[], notes=[])

    tracker.evaluate(horizon=Horizon.SHORT_TERM, active_plan=plan, snapshot=_snapshot(tick_id="alive-1", hp=900))
    tracker.evaluate(horizon=Horizon.SHORT_TERM, active_plan=plan, snapshot=_snapshot(tick_id="dead-1", hp=0))
    tracker.evaluate(horizon=Horizon.SHORT_TERM, active_plan=plan, snapshot=_snapshot(tick_id="alive-2", hp=850))
    result = tracker.evaluate(horizon=Horizon.SHORT_TERM, active_plan=plan, snapshot=_snapshot(tick_id="dead-2", hp=0))

    assert result.death_loops >= 2
    assert "death_loop_detected" in result.reasons
    assert result.force_replan_hint is True


@dataclass(slots=True)
class _FleetState:
    stale: bool = False
    central_available: bool = True

    def status(self) -> dict[str, object]:
        return {
            "mode": "central" if self.central_available and not self.stale else "local",
            "central_available": self.central_available,
            "stale": self.stale,
            "last_sync_at": datetime.now(UTC),
            "doctrine_version": "doctrine-v1",
            "last_error": "",
        }


@dataclass(slots=True)
class _SnapshotCache:
    snapshot: BotStateSnapshot

    def get(self, _bot_id: str) -> BotStateSnapshot:
        return self.snapshot


class _PDCAStubRuntime:
    def __init__(self) -> None:
        self.autonomy_policy = {
            "ranked_objectives": ["grind", "recovery", "economy", "quest"],
            "objective_rotation_cooldown_s": 0.0,
            "reconnect_grace_s": 20.0,
        }
        self.snapshot_cache = _SnapshotCache(snapshot=_snapshot(tick_id="pdca-snap"))
        self.fleet_constraint_state = _FleetState(stale=False, central_available=True)
        self.planner_calls: list[object] = []
        self.crewai_calls: list[object] = []

    def list_bots(self) -> list[dict[str, object]]:
        return [{"bot_id": "bot:autonomy"}]

    async def planner_plan(self, payload) -> PlannerResponse:
        self.planner_calls.append(payload)
        return PlannerResponse(
            ok=True,
            message="ok",
            trace_id=payload.meta.trace_id,
            tactical_bundle=TacticalIntentBundle(
                bundle_id=f"bundle-{len(self.planner_calls)}",
                bot_id=payload.meta.bot_id,
                intents=[],
                actions=[],
                notes=[],
            ),
            provider="pytest",
            model="local",
            latency_ms=1.0,
        )

    async def crewai_strategize(self, payload) -> CrewStrategizeResponse:
        self.crewai_calls.append(payload)
        strategic = StrategicPlan(
            plan_id=f"plan-{len(self.crewai_calls)}",
            bot_id=payload.meta.bot_id,
            objective=payload.objective,
            horizon=PlanHorizon.strategic,
            steps=[],
            recommended_actions=[],
            expires_at=datetime.now(UTC) + timedelta(minutes=10),
        )
        return CrewStrategizeResponse(
            ok=True,
            message="ok",
            trace_id=payload.meta.trace_id,
            bot_id=payload.meta.bot_id,
            objective=payload.objective,
            planner_response=PlannerResponse(
                ok=True,
                message="ok",
                trace_id=payload.meta.trace_id,
                strategic_plan=strategic,
                provider="pytest",
                model="local",
                latency_ms=1.0,
            ),
            consolidated_output="ok",
            agent_outputs=[],
        )

    def queue_action(self, proposal, _bot_id: str):
        return True, ActionStatus.queued, proposal.action_id, "queued"


def test_pdca_force_replan_propagates_to_planner_and_crewai(monkeypatch) -> None:
    runtime = _PDCAStubRuntime()
    pdca = PDCALoop(runtime_state=runtime)

    pdca._active_plan[Horizon.SHORT_TERM] = TacticalIntentBundle(
        bundle_id="active-short",
        bot_id="bot:autonomy",
        intents=[],
        actions=[],
        notes=[],
    )
    pdca._active_plan[Horizon.LONG_TERM] = StrategicPlan(
        plan_id="active-long",
        bot_id="bot:autonomy",
        objective="advance progression",
        horizon=PlanHorizon.strategic,
        steps=[],
        recommended_actions=[],
        expires_at=datetime.now(UTC) + timedelta(minutes=10),
    )

    forced = ProgressEvaluation(
        progress_pct=0.1,
        stuck_cycles=0,
        status="stuck",
        reasons=["stale_progress"],
        force_replan_hint=True,
    )
    monkeypatch.setattr(pdca._progress_tracker, "evaluate", lambda **_: forced)

    short_result = asyncio.run(pdca._run_one_cycle(Horizon.SHORT_TERM))
    long_result = asyncio.run(pdca._run_one_cycle(Horizon.LONG_TERM))

    assert short_result.force_replan is True
    assert short_result.re_planned is True
    assert runtime.planner_calls and bool(runtime.planner_calls[-1].force_replan) is True

    assert long_result.force_replan is True
    assert long_result.re_planned is True
    assert runtime.crewai_calls and bool(runtime.crewai_calls[-1].force_replan) is True


def test_pdca_short_term_objective_rotation_on_replan() -> None:
    runtime = _PDCAStubRuntime()
    pdca = PDCALoop(runtime_state=runtime)

    snap = _snapshot(tick_id="rotate-1")
    first = pdca._select_objective(
        horizon=Horizon.SHORT_TERM,
        snapshot=snap,
        replan_reasons=["fleet_central_stale"],
    )
    second = pdca._select_objective(
        horizon=Horizon.SHORT_TERM,
        snapshot=snap,
        replan_reasons=["fleet_central_stale"],
    )

    assert first is not None
    assert second is not None
    assert first != second
