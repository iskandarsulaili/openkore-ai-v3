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
    central_enabled: bool = True
    stale: bool = False
    central_available: bool = True

    def status(self) -> dict[str, object]:
        return {
            "mode": "central" if self.central_available and not self.stale else "local",
            "central_enabled": self.central_enabled,
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


def test_pdca_collect_replan_reasons_ignores_central_failure_when_disabled() -> None:
    runtime = _PDCAStubRuntime()
    runtime.fleet_constraint_state = _FleetState(central_enabled=False, stale=True, central_available=False)
    pdca = PDCALoop(runtime_state=runtime)

    reasons = pdca._collect_replan_reasons(
        horizon=Horizon.SHORT_TERM,
        progress=ProgressEvaluation(),
        snapshot=_snapshot(tick_id="disabled-central"),
    )

    assert "fleet_central_stale" not in reasons
    assert "fleet_central_unavailable" not in reasons


def test_pdca_collect_replan_reasons_keeps_central_failure_when_enabled_unavailable() -> None:
    runtime = _PDCAStubRuntime()
    runtime.fleet_constraint_state = _FleetState(central_enabled=True, stale=True, central_available=False)
    pdca = PDCALoop(runtime_state=runtime)

    reasons = pdca._collect_replan_reasons(
        horizon=Horizon.SHORT_TERM,
        progress=ProgressEvaluation(),
        snapshot=_snapshot(tick_id="enabled-unavailable"),
    )

    assert "fleet_central_stale" in reasons
    assert "fleet_central_unavailable" in reasons


def test_pdca_long_term_falls_back_to_planner_when_crewai_unusable() -> None:
    class _Runtime(_PDCAStubRuntime):
        async def crewai_strategize(self, payload) -> CrewStrategizeResponse:
            self.crewai_calls.append(payload)
            return CrewStrategizeResponse(
                ok=False,
                message="crewai_disabled",
                trace_id=payload.meta.trace_id,
                bot_id=payload.meta.bot_id,
                objective=payload.objective,
                planner_response=None,
                consolidated_output="crewai_disabled",
                agent_outputs=[],
                errors=["crewai_disabled"],
            )

        async def planner_plan(self, payload) -> PlannerResponse:
            self.planner_calls.append(payload)
            strategic = StrategicPlan(
                plan_id="plan-fallback",
                bot_id=payload.meta.bot_id,
                objective=payload.objective,
                horizon=PlanHorizon.strategic,
                steps=[],
                recommended_actions=[],
                expires_at=datetime.now(UTC) + timedelta(minutes=10),
            )
            return PlannerResponse(
                ok=True,
                message="ok",
                trace_id=payload.meta.trace_id,
                strategic_plan=strategic,
                provider="pytest",
                model="local",
                latency_ms=1.0,
            )

    runtime = _Runtime()
    pdca = PDCALoop(runtime_state=runtime)

    result = asyncio.run(pdca._run_one_cycle(Horizon.LONG_TERM))

    assert result.error is None
    assert result.re_planned is True
    assert runtime.crewai_calls
    assert runtime.planner_calls
    assert runtime.planner_calls[-1].horizon == PlanHorizon.strategic


def test_pdca_startup_gate_warmup_blocks_dispatch_until_minimum_live_state() -> None:
    class _Runtime(_PDCAStubRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.snapshot_cache = type("NullSnapshotCache", (), {"get": staticmethod(lambda _bot_id: None)})()
            self._startup_gate_by_bot: dict[str, dict[str, object]] = {}

        def startup_gate_status(self, *, bot_id: str) -> dict[str, object]:
            cached = self._startup_gate_by_bot.get(bot_id)
            if cached is not None:
                return dict(cached)
            return {
                "bot_id": bot_id,
                "gate_open": False,
                "mode": "warmup",
                "reason": "startup_gate_initializing",
                "failure_count": 0,
                "last_error": "",
                "elapsed_s": 0.0,
                "grace_s": 20.0,
                "min_events": 2,
                "snapshot_ready": False,
                "history_ready": False,
                "continuity_goal_state_present": False,
                "recent_event_count": 0,
            }

        def update_startup_gate(
            self,
            *,
            bot_id: str,
            gate_open: bool,
            mode: str,
            reason: str,
            failure_count: int,
            last_error: str,
            grace_s: float | None = None,
            min_events: int | None = None,
            major_reasons: list[str] | None = None,
        ) -> dict[str, object]:
            del major_reasons
            self._startup_gate_by_bot[bot_id] = {
                "bot_id": bot_id,
                "gate_open": bool(gate_open),
                "mode": str(mode),
                "reason": str(reason),
                "failure_count": int(failure_count),
                "last_error": str(last_error),
                "elapsed_s": 0.0,
                "grace_s": float(20.0 if grace_s is None else grace_s),
                "min_events": int(2 if min_events is None else min_events),
                "snapshot_ready": False,
                "history_ready": False,
                "continuity_goal_state_present": False,
                "recent_event_count": 0,
            }
            return self.startup_gate_status(bot_id=bot_id)

    runtime = _Runtime()
    pdca = PDCALoop(runtime_state=runtime)

    result = asyncio.run(pdca._run_one_cycle(Horizon.SHORT_TERM))

    assert result.error is None
    assert result.re_planned is False
    assert result.actions_queued == 0
    assert runtime.planner_calls == []
    assert runtime.crewai_calls == []

    status = runtime.startup_gate_status(bot_id="bot:autonomy")
    assert status["gate_open"] is False
    assert status["mode"] == "warmup"
    assert str(status["reason"]).startswith("startup_gate_waiting_minimum_live_state")


def test_pdca_startup_gate_opens_in_degraded_mode_when_optional_subsystems_are_unavailable() -> None:
    class _Runtime(_PDCAStubRuntime):
        def __init__(self) -> None:
            super().__init__()
            self._startup_gate_by_bot: dict[str, dict[str, object]] = {}
            self.fleet_constraint_state = _FleetState(
                central_enabled=True,
                stale=True,
                central_available=False,
            )

        def startup_gate_status(self, *, bot_id: str) -> dict[str, object]:
            cached = self._startup_gate_by_bot.get(bot_id)
            if cached is not None:
                return dict(cached)
            return {
                "bot_id": bot_id,
                "gate_open": False,
                "mode": "warmup",
                "reason": "startup_gate_initializing",
                "failure_count": 0,
                "last_error": "",
                "elapsed_s": 0.0,
                "grace_s": 20.0,
                "min_events": 2,
                "snapshot_ready": False,
                "history_ready": True,
                "continuity_goal_state_present": True,
                "recent_event_count": 2,
            }

        def update_startup_gate(
            self,
            *,
            bot_id: str,
            gate_open: bool,
            mode: str,
            reason: str,
            failure_count: int,
            last_error: str,
            grace_s: float | None = None,
            min_events: int | None = None,
            major_reasons: list[str] | None = None,
        ) -> dict[str, object]:
            del major_reasons
            self._startup_gate_by_bot[bot_id] = {
                "bot_id": bot_id,
                "gate_open": bool(gate_open),
                "mode": str(mode),
                "reason": str(reason),
                "failure_count": int(failure_count),
                "last_error": str(last_error),
                "elapsed_s": 0.0,
                "grace_s": float(20.0 if grace_s is None else grace_s),
                "min_events": int(2 if min_events is None else min_events),
                "snapshot_ready": True,
                "history_ready": True,
                "continuity_goal_state_present": True,
                "recent_event_count": 2,
            }
            return self.startup_gate_status(bot_id=bot_id)

        def planner_status(self, *, bot_id: str):
            del bot_id

            class _Planner:
                planner_healthy = True
                updated_at = datetime.now(UTC) - timedelta(minutes=10)

            return _Planner()

        def crewai_status(self):
            class _Crew:
                crew_available = False
                crewai_enabled = True

            return _Crew()

    runtime = _Runtime()
    pdca = PDCALoop(runtime_state=runtime)

    result = asyncio.run(pdca._run_one_cycle(Horizon.SHORT_TERM))

    assert result.error is None
    assert runtime.planner_calls
    status = runtime.startup_gate_status(bot_id="bot:autonomy")
    assert status["gate_open"] is True
    assert status["mode"] == "degraded"
    reason = str(status["reason"])
    assert reason.startswith("startup_gate_open_degraded_optional_subsystems")
    assert "fleet_central_stale" in reason
    assert "fleet_central_unavailable" in reason
    assert ("planner_stale" in reason) or ("planner_status_unavailable" in reason)
    assert "crewai_unavailable" in reason
