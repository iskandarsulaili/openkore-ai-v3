from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ai_sidecar.autonomy.plan_executor import PlanExecutor
from ai_sidecar.contracts.actions import ActionStatus
from ai_sidecar.contracts.common import ContractMeta
from ai_sidecar.contracts.state import BotStateSnapshot, CombatState, Position, Vitals
from ai_sidecar.planner.schemas import PlanHorizon, PlannerStep, PlannerStepKind, StrategicPlan, TacticalIntent, TacticalIntentBundle


class _SnapshotCache:
    def __init__(self) -> None:
        self._snapshots: dict[str, BotStateSnapshot] = {}

    def set(self, bot_id: str, snapshot: BotStateSnapshot) -> None:
        self._snapshots[bot_id] = snapshot

    def get(self, bot_id: str):
        return self._snapshots.get(bot_id)


class _Runtime:
    def __init__(self) -> None:
        self.snapshot_cache = _SnapshotCache()
        self.queued: list[tuple[str, object]] = []

    def _fleet_constraints_for_bot(self, *, bot_id: str) -> dict[str, object]:
        return {
            "assignment": "",
            "preferred_grind_maps": ["prt_fild08"],
            "preferred_maps": [],
        }

    def queue_action(self, proposal, bot_id: str):
        self.queued.append((bot_id, proposal))
        return True, ActionStatus.queued, proposal.action_id, "queued"


def _snapshot(*, bot_id: str, map_name: str = "prt_fild01") -> BotStateSnapshot:
    return BotStateSnapshot(
        meta=ContractMeta(contract_version="v1", source="pytest", bot_id=bot_id, trace_id=f"trace-{bot_id}"),
        tick_id="tick-1",
        observed_at=datetime.now(UTC),
        position=Position(map=map_name, x=100, y=100),
        vitals=Vitals(hp=900, hp_max=1000, sp=100, sp_max=200, weight=1000, weight_max=8000),
        combat=CombatState(ai_sequence="route", target_id=None, is_in_combat=False),
        raw={},
    )


def test_plan_executor_strategic_step_prefers_resume_grind_map() -> None:
    runtime = _Runtime()
    runtime.snapshot_cache.set("bot:exec", _snapshot(bot_id="bot:exec", map_name="prt_fild01"))
    executor = PlanExecutor(runtime_state=runtime)

    plan = StrategicPlan(
        plan_id="plan-1",
        bot_id="bot:exec",
        objective="farm safely",
        horizon=PlanHorizon.strategic,
        steps=[
            PlannerStep(
                step_id="s1",
                kind=PlannerStepKind.travel,
                description="Resume grind posture and route to preferred map",
            )
        ],
        recommended_actions=[],
        expires_at=datetime.now(UTC) + timedelta(minutes=2),
    )

    queued = __import__("asyncio").run(executor.execute(plan=plan, horizon=__import__("types").SimpleNamespace(value="long_term"), max_actions=2))

    assert queued == 1
    assert len(runtime.queued) == 1
    action = runtime.queued[0][1]
    assert action.command == "move prt_fild08"
    assert action.conflict_key == "nav.resume_grind"
    assert "navigation.ready" in action.preconditions
    assert action.metadata.get("fallback_mode") == "resume_grind"
    assert action.metadata.get("target") == "prt_fild08"


def test_plan_executor_tactical_intent_seek_targets_uses_scan_guard() -> None:
    runtime = _Runtime()
    runtime.snapshot_cache.set("bot:exec", _snapshot(bot_id="bot:exec", map_name="prt_fild08"))
    executor = PlanExecutor(runtime_state=runtime)

    bundle = TacticalIntentBundle(
        bundle_id="bundle-1",
        bot_id="bot:exec",
        intents=[
            TacticalIntent(
                intent_id="i1",
                objective="Seek target safely when no enemies are visible",
                constraints=["step_kind=travel"],
            )
        ],
        actions=[],
        notes=[],
    )

    queued = __import__("asyncio").run(executor.execute(plan=bundle, horizon=__import__("types").SimpleNamespace(value="short_term"), max_actions=2))

    assert queued == 1
    assert len(runtime.queued) == 1
    action = runtime.queued[0][1]
    assert action.command == "move random_walk_seek"
    assert action.conflict_key == "planner.seek.random_walk"
    assert "navigation.ready" in action.preconditions
    assert "scan.targets_absent" in action.preconditions
    assert bool(action.metadata.get("seek_only_random_walk")) is True
    assert bool(action.metadata.get("target_scan_required")) is True
    target_scan = action.metadata.get("target_scan")
    assert isinstance(target_scan, dict)
    assert target_scan.get("targets_found") is False


def test_plan_executor_tactical_intent_safe_idle_no_ai_auto() -> None:
    runtime = _Runtime()
    runtime.snapshot_cache.set("bot:exec", _snapshot(bot_id="bot:exec", map_name="prt_fild08"))
    executor = PlanExecutor(runtime_state=runtime)

    bundle = TacticalIntentBundle(
        bundle_id="bundle-2",
        bot_id="bot:exec",
        intents=[
            TacticalIntent(
                intent_id="i2",
                objective="Hold rest posture while waiting for fresh planner state",
                constraints=[],
            )
        ],
        actions=[],
        notes=[],
    )

    queued = __import__("asyncio").run(executor.execute(plan=bundle, horizon=__import__("types").SimpleNamespace(value="short_term"), max_actions=2))

    assert queued == 1
    assert len(runtime.queued) == 1
    action = runtime.queued[0][1]
    assert action.command == "sit"
    assert action.conflict_key == "planner.safe_idle"
    assert "vitals.safe_to_rest" in action.preconditions
    assert action.metadata.get("fallback_mode") == "safe_idle"
    assert action.command != "ai auto"


def test_plan_executor_strategic_death_recovery_uses_respawn_command() -> None:
    runtime = _Runtime()
    runtime.snapshot_cache.set("bot:exec", _snapshot(bot_id="bot:exec", map_name="prt_fild08"))
    executor = PlanExecutor(runtime_state=runtime)

    plan = StrategicPlan(
        plan_id="plan-death",
        bot_id="bot:exec",
        objective="recover safely",
        horizon=PlanHorizon.strategic,
        steps=[
            PlannerStep(
                step_id="s1",
                kind=PlannerStepKind.rest,
                description="Recover from death at savepoint before route resume",
            )
        ],
        recommended_actions=[],
        expires_at=datetime.now(UTC) + timedelta(minutes=2),
    )

    queued = __import__("asyncio").run(executor.execute(plan=plan, horizon=__import__("types").SimpleNamespace(value="long_term"), max_actions=2))

    assert queued == 1
    assert len(runtime.queued) == 1
    action = runtime.queued[0][1]
    assert action.command == "respawn"
    assert action.conflict_key == "recovery.death"
    assert action.metadata.get("fallback_mode") == "death_recovery"
    assert action.metadata.get("target") == "savepoint"
