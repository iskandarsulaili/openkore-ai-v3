"""Tests for the TaskScheduler wiring + TaskCommandTranslator.

The TaskScheduler execution path was dormant: heuristic_service instantiated
it but never called schedule_from_signals/execute_task. Now wired through
TaskCommandTranslator, which maps semantic task commands to safe real
commands or observable log intents. Key invariants:
  - bare 'party' / 'guild' / 'attack' pseudo-commands NEVER become real commands
  - return_town only fires with a real inventory_full need, level >= 6, on a fild
  - buy_pots only fires in town
  - emission is throttled per (bot, task)
"""

from __future__ import annotations

from ai_sidecar.domains.planning.goals import GoalManager
from ai_sidecar.domains.planning.scheduler import ScheduledTask, TaskCategory, TaskScheduler
from ai_sidecar.domains.planning.translator import TaskCommandTranslator


def _scheduler() -> TaskScheduler:
    ts = TaskScheduler()
    ts.set_goal_manager(GoalManager(bot_id="default"))
    return ts


def _real_commands(actions) -> list[str]:
    return [a.command for a in actions if a.kind == "command"]


def test_scheduler_is_wired_through_heuristic() -> None:
    from ai_sidecar.autonomy.heuristic_service import HeuristicService

    h = HeuristicService()
    h._init_new_domains()
    assert h._task_scheduler is not None
    assert h._task_translator is not None
    # goal manager wired into scheduler (quest/job tasks can fire)
    assert h._task_scheduler._goal_manager is not None


def test_party_and_guild_tasks_never_emit_real_commands() -> None:
    ts = _scheduler()
    tr = TaskCommandTranslator()
    sig = {"map": "prt_fild05", "base_level": 8, "job": "swordman", "lockMap": "prt_fild05"}
    # Force the SOCIAL_TASKS through the translator directly
    for task_name, cmd in [("party_check", "party"), ("guild_check", "guild")]:
        task = ScheduledTask(
            priority_score=60, category=TaskCategory.SOCIAL,
            name=task_name, description=f"Check {task_name}", commands=[cmd],
        )
        acts = tr.translate(task, sig, "bot:x")
        real = _real_commands(acts)
        assert real == [], f"{task_name} must never emit: {real}"
        assert all(a.kind == "log" for a in acts)


def test_return_town_gated_on_inventory_full_fild_and_level() -> None:
    ts = _scheduler()
    task = ScheduledTask(
        priority_score=5, category=TaskCategory.ECONOMY, name="sell_loot",
        description="Return to town and sell loot", commands=["return_town", "sell_loot"],
    )

    # Level 8 on fild WITHOUT inventory_full -> no move (timer is not a need)
    tr = TaskCommandTranslator()
    acts = tr.translate(task, {"map": "prt_fild05", "base_level": 8}, "bot:x")
    assert "move prontera" not in _real_commands(acts)

    # WITH inventory_full -> move prontera fires
    tr2 = TaskCommandTranslator()
    acts2 = tr2.translate(task, {"map": "prt_fild05", "base_level": 8, "inventory_full": True}, "bot:x")
    assert "move prontera" in _real_commands(acts2)

    # Level 4 academy bot even with inventory_full -> no move (cold-start owns)
    tr3 = TaskCommandTranslator()
    acts3 = tr3.translate(task, {"map": "izlude", "base_level": 4, "inventory_full": True}, "bot:x")
    assert "move prontera" not in _real_commands(acts3)

    # In town already -> no move
    tr4 = TaskCommandTranslator()
    acts4 = tr4.translate(task, {"map": "prontera", "base_level": 8, "inventory_full": True}, "bot:x")
    assert "move prontera" not in _real_commands(acts4)


def test_buy_pots_only_in_town() -> None:
    task = ScheduledTask(
        priority_score=5, category=TaskCategory.ECONOMY, name="restock_pots",
        description="Return to town and restock potions", commands=["buy_pots"],
    )
    tr = TaskCommandTranslator()
    in_town = tr.translate(task, {"map": "prontera", "base_level": 8}, "bot:x")
    assert "buy potion 30" in _real_commands(in_town)

    tr2 = TaskCommandTranslator()
    on_fild = tr2.translate(task, {"map": "prt_fild05", "base_level": 8}, "bot:x")
    assert "buy potion 30" not in _real_commands(on_fild)


def test_emission_throttled_per_bot_task() -> None:
    task = ScheduledTask(
        priority_score=35, category=TaskCategory.COMBAT, name="hunt_current_map",
        description="Hunt monsters on current map", commands=["attack"],
    )
    tr = TaskCommandTranslator()
    sig = {"map": "prt_fild05", "base_level": 8}
    first = tr.translate(task, sig, "bot:x")
    second = tr.translate(task, sig, "bot:x")  # within cooldown
    assert first, "first emission must produce actions"
    assert second == [], "second emission within cooldown must be throttled"


def test_grind_and_hunt_are_observed_only() -> None:
    ts = _scheduler()
    tr = TaskCommandTranslator()
    sig = {"map": "prt_fild05", "base_level": 8}
    sched = ts.schedule_from_signals(sig)
    for t in sched[:3]:
        acts = tr.translate(t, sig, "bot:x")
        for a in acts:
            if "attack" in getattr(t, "commands", []) or "attack" in a.command:
                assert a.kind == "log", f"attack intent must be observed-only: {a}"


def test_restock_triggered_by_depletion_score() -> None:
    # The depletion score (enriched world projection) must drive restock
    # BEFORE pots run out — previously _should_restock returned False
    # whenever no learning tracker was attached.
    ts = _scheduler()
    drained = ts.schedule_from_signals(
        {"map": "prt_fild05", "base_level": 8, "consumable_depletion_score": 0.85}
    )
    restock = [t for t in drained if t.name == "restock_pots"]
    assert restock, "depletion >= 0.75 must schedule restock_pots"

    ts2 = _scheduler()
    full = ts2.schedule_from_signals(
        {"map": "prt_fild05", "base_level": 8, "consumable_depletion_score": 0.2}
    )
    assert not [t for t in full if t.name == "restock_pots"], \
        "low depletion must NOT schedule restock"


def test_depletion_score_reads_nested_inventory_shape() -> None:
    # Enriched-state fallback shape: {"inventory": {"consumable_depletion_score": ...}}
    ts = _scheduler()
    drained = ts.schedule_from_signals({
        "map": "prt_fild05", "base_level": 8,
        "inventory": {"consumable_depletion_score": 0.9},
    })
    assert [t for t in drained if t.name == "restock_pots"], \
        "nested enriched shape must also trigger restock"
