from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from ai_sidecar.autonomy.plan_executor import PlanExecutor
from ai_sidecar.contracts.actions import ActionStatus
from ai_sidecar.contracts.autonomy import GoalCategory, GoalDirective, GoalStackState, SituationalAssessment
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
        self.control_plan_calls: list[object] = []
        self.control_apply_calls: list[object] = []
        self.published_macros: list[object] = []

    def _fleet_constraints_for_bot(self, *, bot_id: str) -> dict[str, object]:
        return {
            "assignment": "",
            "preferred_grind_maps": ["prt_fild08"],
            "preferred_maps": [],
        }

    def queue_action(self, proposal, bot_id: str):
        self.queued.append((bot_id, proposal))
        return True, ActionStatus.queued, proposal.action_id, "queued"

    def control_plan(self, payload):
        self.control_plan_calls.append(payload)
        return SimpleNamespace(ok=True, plan=SimpleNamespace(plan_id="ctrl-plan-1"))

    def control_apply(self, payload):
        self.control_apply_calls.append(payload)
        return SimpleNamespace(ok=True)

    def publish_macros(self, request):
        self.published_macros.append(request)
        return True, {"publication_id": "pub-1"}, "published"


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


def _goal_state_with_hint(*, bot_id: str, hint: dict[str, object]) -> GoalStackState:
    assessment = SituationalAssessment(
        bot_id=bot_id,
        opportunistic_upgrades={"execution_hints": [dict(hint)]},
    )
    selected = GoalDirective(
        goal_key=GoalCategory.opportunistic_upgrades,
        priority_rank=1,
        active=True,
        objective="execute deterministic stage4 hint",
        rationale="pytest",
        metadata={"execution_hints": [dict(hint)]},
    )
    return GoalStackState(
        bot_id=bot_id,
        horizon="short_term",
        assessment=assessment,
        goal_stack=[selected],
        selected_goal=selected,
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


def test_plan_executor_tactical_intent_sparse_state_escalates_before_safe_idle() -> None:
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
    assert action.command == "move random_walk_seek"
    assert action.conflict_key == "planner.seek.random_walk"
    assert "navigation.ready" in action.preconditions
    assert action.metadata.get("fallback_mode") == "seek_targets"
    assert action.metadata.get("escalation_stage") == 0
    assert "sparse_state" in str(action.metadata.get("escalation_reason") or "")
    assert action.command != "ai auto"


def test_plan_executor_tactical_sparse_state_escalation_cycles_through_recovery_steps() -> None:
    runtime = _Runtime()
    runtime.snapshot_cache.set("bot:exec", _snapshot(bot_id="bot:exec", map_name="prt_fild08"))
    executor = PlanExecutor(runtime_state=runtime)

    bundle = TacticalIntentBundle(
        bundle_id="bundle-escalate",
        bot_id="bot:exec",
        intents=[
            TacticalIntent(
                intent_id="i-escalate",
                objective="Hold rest posture while waiting for fresh planner state",
                constraints=[],
            )
        ],
        actions=[],
        notes=[],
    )

    for _ in range(4):
        __import__("asyncio").run(executor.execute(plan=bundle, horizon=__import__("types").SimpleNamespace(value="short_term"), max_actions=1))

    commands = [entry[1].command for entry in runtime.queued]
    assert commands == ["move random_walk_seek", "ai clear", "move prt_fild08", "sit"]

    stages = [entry[1].metadata.get("escalation_stage") for entry in runtime.queued]
    assert stages == [0, 1, 2, 3]

    modes = [entry[1].metadata.get("fallback_mode") for entry in runtime.queued]
    assert modes == ["seek_targets", "ai_queue_reset", "map_refresh", "safe_idle"]

    reasons = [str(entry[1].metadata.get("escalation_reason") or "") for entry in runtime.queued]
    assert all("sparse_state" in reason for reason in reasons)

    conflict_keys = [entry[1].conflict_key for entry in runtime.queued]
    assert conflict_keys == [
        "planner.seek.random_walk",
        "planner.recovery.ai_clear",
        "planner.recovery.map_refresh",
        "planner.safe_idle",
    ]


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
    assert "session.in_game" in action.preconditions
    assert action.metadata.get("fallback_mode") == "death_recovery"
    assert action.metadata.get("target") == "savepoint"


def test_plan_executor_deterministic_direct_hint_queues_intent() -> None:
    runtime = _Runtime()
    runtime.snapshot_cache.set("bot:exec", _snapshot(bot_id="bot:exec", map_name="prt_fild08"))
    executor = PlanExecutor(runtime_state=runtime)

    bundle = TacticalIntentBundle(bundle_id="bundle-direct", bot_id="bot:exec", intents=[], actions=[], notes=[])
    hint = {
        "execution_mode": "direct",
        "tool": "propose_actions",
        "rule_id": "rule-direct-1",
        "intents": [
            {
                "kind": "command",
                "command": "move prt_fild09",
                "conflict_key": "opportunistic.explore",
                "priority_tier": "tactical",
                "metadata": {"source": "autonomy_stage4", "preconditions": ["navigation.ready"]},
            }
        ],
    }
    goal_state = _goal_state_with_hint(bot_id="bot:exec", hint=hint)

    queued = __import__("asyncio").run(
        executor.execute(
            plan=bundle,
            horizon=__import__("types").SimpleNamespace(value="short_term"),
            max_actions=2,
            goal_state=goal_state,
        )
    )

    assert queued == 1
    assert len(runtime.queued) == 1
    action = runtime.queued[0][1]
    assert action.command == "move prt_fild09"
    assert action.conflict_key == "opportunistic.explore"
    assert "navigation.ready" in action.preconditions
    assert action.metadata.get("rule_id") == "rule-direct-1"
    assert action.metadata.get("execution_mode") == "direct"
    assert action.metadata.get("source") == "autonomy_stage4"
    assert action.idempotency_key.startswith("autonomy_hint:bot:exec:rule-direct-1:0:move prt_fild09")


def test_plan_executor_deterministic_config_hint_routes_control_plan_and_apply() -> None:
    runtime = _Runtime()
    runtime.snapshot_cache.set("bot:exec", _snapshot(bot_id="bot:exec", map_name="prt_fild08"))
    executor = PlanExecutor(runtime_state=runtime)

    plan = StrategicPlan(
        plan_id="plan-config",
        bot_id="bot:exec",
        objective="apply deterministic config upgrade",
        horizon=PlanHorizon.strategic,
        steps=[],
        recommended_actions=[],
        expires_at=datetime.now(UTC) + timedelta(minutes=2),
    )
    hint = {
        "execution_mode": "config",
        "tool": "plan_control_change",
        "rule_id": "rule-config-1",
        "request": {
            "artifact_type": "config",
            "name": "config.txt",
            "target_path": "control/config.txt",
            "desired": {
                "aiSidecar_enable": "1",
                "aiSidecar_profile": "vending_cycle",
                "lockMap": "prontera",
                "itemsTakeAuto": "0",
            },
            "source": "crewai",
        },
    }
    goal_state = _goal_state_with_hint(bot_id="bot:exec", hint=hint)

    queued = __import__("asyncio").run(
        executor.execute(
            plan=plan,
            horizon=__import__("types").SimpleNamespace(value="long_term"),
            max_actions=2,
            goal_state=goal_state,
        )
    )

    assert queued == 1
    assert runtime.control_plan_calls
    assert runtime.control_apply_calls
    assert len(runtime.queued) == 0
    assert runtime.control_apply_calls[0].plan_id == "ctrl-plan-1"

    planned = runtime.control_plan_calls[0]
    assert planned.meta.source == "autonomy_plan_executor"
    assert planned.bot_id == "bot:exec"
    assert planned.target_path == "control/config.txt"
    assert planned.source == "crewai"
    assert planned.desired.get("aiSidecar_enable") == "1"
    assert planned.desired.get("aiSidecar_profile") == "vending_cycle"
    assert planned.desired.get("lockMap") == "prontera"
    assert planned.desired.get("itemsTakeAuto") == "0"

    applied = runtime.control_apply_calls[0]
    assert applied.dry_run is False


def test_plan_executor_deterministic_macro_hint_routes_publish_macros() -> None:
    runtime = _Runtime()
    runtime.snapshot_cache.set("bot:exec", _snapshot(bot_id="bot:exec", map_name="prt_fild08"))
    executor = PlanExecutor(runtime_state=runtime)

    plan = StrategicPlan(
        plan_id="plan-macro",
        bot_id="bot:exec",
        objective="publish deterministic macro upgrade",
        horizon=PlanHorizon.strategic,
        steps=[],
        recommended_actions=[],
        expires_at=datetime.now(UTC) + timedelta(minutes=2),
    )
    hint = {
        "execution_mode": "macro",
        "tool": "publish_macro",
        "rule_id": "rule-macro-1",
        "macro_bundle": {
            "macros": [
                {
                    "name": "crew_stage4_upgrade_posture",
                    "lines": ["do ai auto"],
                }
            ],
            "event_macros": [
                {
                    "name": "crew_wave_a1_companion_regroup",
                    "lines": ["call crew_stage4_upgrade_posture"],
                }
            ],
            "automacros": [
                {
                    "name": "on_crew_wave_a1_companion_regroup",
                    "conditions": ["OnCharLogIn"],
                    "call": "crew_stage4_upgrade_posture",
                    "parameters": {"source": "wave_a1"},
                }
            ],
            "reload_conflict_key": "macro_reload.wave_a1",
            "macro_plugin": "macro",
            "event_macro_plugin": "eventMacro",
            },
    }
    goal_state = _goal_state_with_hint(bot_id="bot:exec", hint=hint)

    queued = __import__("asyncio").run(
        executor.execute(
            plan=plan,
            horizon=__import__("types").SimpleNamespace(value="long_term"),
            max_actions=2,
            goal_state=goal_state,
        )
    )

    assert queued == 1
    assert runtime.published_macros
    assert len(runtime.queued) == 0
    published = runtime.published_macros[0]
    assert published.meta.source == "autonomy_plan_executor"
    assert published.target_bot_id == "bot:exec"
    assert published.enqueue_reload is True
    assert published.reload_conflict_key == "macro_reload.wave_a1"
    assert published.macro_plugin == "macro"
    assert published.event_macro_plugin == "eventMacro"
    assert published.macros[0].name == "crew_stage4_upgrade_posture"
    assert published.event_macros[0].name == "crew_wave_a1_companion_regroup"
    assert published.automacros[0].call == "crew_stage4_upgrade_posture"
    assert published.automacros[0].parameters.get("source") == "wave_a1"
