"""Regression test: /v1/actions/next drains up to max_actions per poll.

The bridge sends max_actions in every poll; the endpoint previously
returned ONE action regardless, so a 5-capacity poll got 1 action and the
fleet's queued actions drained at 1/5th the intended rate. The endpoint now
drains up to max_actions (clamped to 10) and returns the batch in `actions`
while keeping `action` (first) for backward-compatible single-action
clients. The bridge already consumes the array.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from ai_sidecar.contracts.actions import (
    ActionPriorityTier,
    ActionProposal,
    NextActionRequest,
    NextActionResponse,
)
from ai_sidecar.contracts.common import ContractMeta


def _proposal(i: int) -> ActionProposal:
    return ActionProposal(
        action_id=f"batch-{i}", kind="command", command=f"move prontera {i}",
        conflict_key="", priority_tier=ActionPriorityTier.tactical, source="test",
        created_at=datetime.now(UTC), expires_at=datetime.now(UTC) + timedelta(seconds=60),
        idempotency_key=f"bk{i}",
    )


def _request(bot_id: str, poll_id: str, max_actions: int = 1) -> NextActionRequest:
    return NextActionRequest(
        meta=ContractMeta(contract_version="v1", source="bridge", bot_id=bot_id, trace_id="t"),
        poll_id=poll_id,
        max_actions=max_actions,
    )


class _FakeQueue:
    def __init__(self, proposals: list[ActionProposal]) -> None:
        self._items = list(proposals)

    def fetch_next(self, bot_id: str) -> ActionProposal | None:
        del bot_id
        return self._items.pop(0) if self._items else None


class _FakeLatency:
    def begin(self):
        return 0.0

    def end(self, name, started):
        del name, started
        return 1.0

    def within_budget(self, elapsed_ms):
        del elapsed_ms
        return True


class _FakeRuntime:
    def __init__(self, proposals: list[ActionProposal]) -> None:
        self.action_queue = _FakeQueue(proposals)
        self.latency_router = _FakeLatency()
        self._fetched: list[str] = []

    def next_action(self, bot_id: str, poll_id: str | None = None):
        del poll_id
        p = self.action_queue.fetch_next(bot_id)
        if p:
            self._fetched.append(p.action_id)
        return p


def _call_next(payload: NextActionRequest, runtime: Any) -> NextActionResponse:
    # Call the handler directly — Depends(get_runtime) only binds during
    # FastAPI routing; the handler accepts runtime as a plain parameter.
    from ai_sidecar.api.routers.actions import next_action as _na

    return _na(payload, runtime=runtime)


def test_single_action_backward_compatible() -> None:
    rt = _FakeRuntime([_proposal(1)])
    resp = _call_next(_request("bot:x", "p1", max_actions=1), rt)
    assert resp.has_action is True
    assert resp.action is not None and resp.action.action_id == "batch-1"
    assert resp.actions == [resp.action], "single-action batch must equal action"


def test_batch_drains_up_to_max_actions() -> None:
    rt = _FakeRuntime([_proposal(1), _proposal(2), _proposal(3), _proposal(4)])
    resp = _call_next(_request("bot:x", "p1", max_actions=3), rt)
    assert resp.has_action is True
    assert resp.action is not None and resp.action.action_id == "batch-1"
    assert [a.action_id for a in (resp.actions or [])] == ["batch-1", "batch-2", "batch-3"]
    assert len(rt._fetched) == 3, "must drain exactly max_actions"


def test_batch_stops_when_queue_empty() -> None:
    rt = _FakeRuntime([_proposal(1)])
    resp = _call_next(_request("bot:x", "p1", max_actions=5), rt)
    assert resp.has_action is True
    assert len(resp.actions or []) == 1, "must stop when queue is empty"


def test_max_actions_schema_caps_at_10() -> None:
    # Schema Field(le=10) rejects >10; at the valid max of 10 with 20 queued
    # actions, exactly 10 are drained.
    rt = _FakeRuntime([_proposal(i) for i in range(20)])
    resp = _call_next(_request("bot:x", "p1", max_actions=10), rt)
    assert len(resp.actions or []) == 10, "batch must be capped at 10"


def test_no_action_when_queue_empty() -> None:
    rt = _FakeRuntime([])
    resp = _call_next(_request("bot:x", "p1", max_actions=3), rt)
    assert resp.has_action is False
    assert resp.action is not None and resp.action.kind == "noop"
