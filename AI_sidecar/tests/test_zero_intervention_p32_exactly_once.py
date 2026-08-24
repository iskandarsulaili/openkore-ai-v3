"""Regression tests for ZERO_INTERVENTION P3.2: exactly-once across restart.

A DISPATCHED action was handed to the bridge; on sidecar restart we cannot know
whether it executed (the bridge's in-memory command dedup is also gone). Re-queuing
dispatched actions (old behavior) re-executes an action that may already have run ->
double-exec. Lock: rehydrate() must NOT re-queue dispatched actions (drop them), while
still re-queuing genuinely-queued (never-sent) ones.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ai_sidecar.contracts.actions import ActionPriorityTier, ActionProposal, ActionStatus
from ai_sidecar.runtime.action_queue import ActionQueue, QueuedAction


def _proposal(cmd: str, idem: str) -> ActionProposal:
    now = datetime.now(UTC)
    return ActionProposal(
        action_id=f"a-{idem}",
        kind="command",
        command=cmd,
        conflict_key="",
        priority_tier=ActionPriorityTier.tactical,
        source="planner",
        created_at=now,
        expires_at=now + timedelta(seconds=60),
        idempotency_key=idem,
    )


def test_rehydrate_does_not_requeue_dispatched() -> None:
    q = ActionQueue(max_per_bot=10)
    # A dispatched action (was fetched by bridge) + a queued action (never sent).
    dispatched = QueuedAction(proposal=_proposal("use 501 1", "idem-dispatch"),
                              status=ActionStatus.dispatched, enqueue_seq=1)
    queued = QueuedAction(proposal=_proposal("move 100 200", "idem-queue"),
                          status=ActionStatus.queued, enqueue_seq=2)
    restored = q.rehydrate("botA", [dispatched, queued])
    # Only the queued action survives.
    assert restored == 1
    snap = q.snapshot().get("botA", [])
    assert len(snap) == 1
    assert snap[0].proposal.idempotency_key == "idem-queue"
    assert snap[0].status == ActionStatus.queued
    # Dispatched was dropped, not re-executable.
    assert q.fetch_next("botA") is not None  # the queued one
    assert q.fetch_next("botA") is None  # dispatched must not come back


def test_rehydrate_keeps_only_queued() -> None:
    q = ActionQueue(max_per_bot=10)
    dispatched = QueuedAction(proposal=_proposal("use 602", "d1"),
                              status=ActionStatus.dispatched, enqueue_seq=1)
    acknowledged = QueuedAction(proposal=_proposal("skill 8", "a1"),
                                status=ActionStatus.acknowledged, enqueue_seq=2)
    restored = q.rehydrate("botB", [dispatched, acknowledged])
    assert restored == 0
    assert q.snapshot().get("botB") == []
