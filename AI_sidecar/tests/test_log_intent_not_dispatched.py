"""Regression test: kind=log observability intents must never be
dispatched to the bridge as executable commands.

History: party_leave_requested was emitted as a kind="log" intent but was
still reaching the bridge, which treated the literal string as a command
and spammed unknown-command errors (fighting the party system). The
next_action drain must skip kind != "command" intents.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from functools import partial
from typing import Any

from ai_sidecar.api.routers.actions import next_action
from ai_sidecar.contracts.actions import (
    ActionPriorityTier,
    ActionProposal,
    NextActionRequest,
    NextActionResponse,
)
from ai_sidecar.contracts.common import ContractMeta


class _FakeQueue:
    def __init__(self, items: list[ActionProposal]) -> None:
        self._items = list(items)

    def fetch_next(self, bot_id: str) -> ActionProposal | None:
        if self._items:
            return self._items.pop(0)
        return None


class _FakeLatency:
    def begin(self):
        return 0.0

    def end(self, name, started):
        return 1.0

    def within_budget(self, elapsed_ms):
        return True


class _FakeRuntime:
    def __init__(self, items: list[ActionProposal]) -> None:
        self.action_queue = _FakeQueue(items)
        self.latency_router = _FakeLatency()

    def next_action(self, bot_id: str, poll_id: str) -> ActionProposal | None:
        return self.action_queue.fetch_next(bot_id)


def _proposal(cmd: str, kind: str = "command", i: int = 0) -> ActionProposal:
    now = datetime.now(UTC)
    return ActionProposal(
        action_id=f"t-{i}",
        bot_id="bot:x",
        kind=kind,
        command=cmd,
        priority_tier=ActionPriorityTier.tactical,
        source="heuristic",
        conflict_key=None,
        created_at=now,
        expires_at=now + timedelta(seconds=30),
        idempotency_key=f"t-{i}-{kind}",
    )


def _request() -> NextActionRequest:
    return NextActionRequest(
        meta=ContractMeta(
            contract_version="v1",
            bot_id="Local rAthena AI World:kicapmasin4",
        ),
        poll_id="p1",
        max_actions=5,
    )


def test_log_intent_is_skipped_from_dispatch() -> None:
    """A queue of [log, command] must dispatch ONLY the command."""
    rt = _FakeRuntime([
        _proposal("party_leave_requested", kind="log", i=1),
        _proposal("ai auto", kind="command", i=2),
    ])
    resp: NextActionResponse = next_action(_request(), rt)
    assert resp.has_action is True
    assert resp.action is not None
    # Only the command reachable — the log intent is drained and skipped.
    assert resp.action.command == "ai auto"


def test_all_log_intents_yield_no_action() -> None:
    """A queue of only log intents must yield has_action=False (noop)."""
    rt = _FakeRuntime([
        _proposal("party_leave_requested", kind="log", i=1),
        _proposal("party_share_pending", kind="log", i=2),
    ])
    resp: NextActionResponse = next_action(_request(), rt)
    assert resp.has_action is False
    assert resp.action is not None
    assert resp.action.kind == "noop"


def test_command_actions_still_dispatched() -> None:
    """A queue of real commands dispatches normally."""
    rt = _FakeRuntime([
        _proposal("buy 1201 1", kind="command", i=1),
        _proposal("stand", kind="command", i=2),
    ])
    resp: NextActionResponse = next_action(_request(), rt)
    assert resp.has_action is True
    assert resp.action.command == "buy 1201 1"
    assert len(resp.actions or []) == 2
