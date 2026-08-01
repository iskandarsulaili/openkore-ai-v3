"""Regression test: recurring-failure config adjustment actually APPLIES.

The failure_reasoning auto-adjust loop was dormant: _apply_config_adjustment
returned suggestions that were only logged/persisted, never applied, and
get_recurring_failures_check had ZERO callers. Now wired into the pdca loop
(time-gated 300s) and enqueues `set <key> <value>` commands via runtime.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ai_sidecar.contracts.actions import ActionProposal
from ai_sidecar.learning.failure_reasoning import FailureReasoningEngine
from ai_sidecar.learning.failure_wiring import get_recurring_failures_check


class _FakeQueue:
    def __init__(self) -> None:
        self.enqueued: list[ActionProposal] = []

    def enqueue(self, bot_id: str, proposal: ActionProposal) -> tuple[bool, Any, str, str]:
        self.enqueued.append(proposal)
        return True, None, proposal.action_id, "ok"


class _FakeRuntime:
    def __init__(self, failures: list[dict[str, Any]]) -> None:
        self.failure_reasoning_engine = FailureReasoningEngine()
        self.failure_reasoning_engine.get_recurring_failures = (  # type: ignore[method-assign]
            lambda server_id=None, min_count=3, limit=10: failures
        )
        self.action_queue = _FakeQueue()


def _failure(cat: str, bot_id: str = "kicapmasin4") -> dict[str, Any]:
    return {
        "id": f"rec-{cat}",
        "server_id": "default",
        "bot_id": bot_id,
        "category": cat,
        "subcategory": None,
        "timestamp": 1785500000.0,
        "recurrence_count": 4,
    }


def test_recurring_failure_enqueues_set_command() -> None:
    rt = _FakeRuntime([_failure("overweight")])
    check = get_recurring_failures_check(rt)
    assert check is not None
    result = check()
    assert len(result) == 1
    assert len(rt.action_queue.enqueued) >= 1
    cmd = rt.action_queue.enqueued[0].command
    assert cmd.startswith("set ")
    # overweight -> sellAuto 1 / storageAuto 1
    assert any("sellAuto" in p.command for p in rt.action_queue.enqueued)
    assert any("storageAuto" in p.command for p in rt.action_queue.enqueued)


def test_no_failures_no_enqueue() -> None:
    rt = _FakeRuntime([])
    check = get_recurring_failures_check(rt)
    result = check()
    assert result == []
    assert rt.action_queue.enqueued == []


def test_no_bot_id_skips_enqueue_but_returns_applied() -> None:
    f = _failure("stuck")
    f["bot_id"] = ""
    rt = _FakeRuntime([f])
    check = get_recurring_failures_check(rt)
    result = check()
    # still reported as processed (the record is returned)
    assert len(result) == 1
    # but nothing enqueued (no bot target)
    assert rt.action_queue.enqueued == []


def test_partial_bad_config_change_skipped() -> None:
    # "party_ghost" -> ["partyAuto 0"] — valid change, bot targeted.
    rt = _FakeRuntime([_failure("party_ghost")])
    check = get_recurring_failures_check(rt)
    check()
    assert len(rt.action_queue.enqueued) == 1
    assert rt.action_queue.enqueued[0].command == "set partyAuto 0"
