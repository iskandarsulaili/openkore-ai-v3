from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ai_sidecar.contracts.actions import ActionPriorityTier, ActionProposal, ActionStatus
from ai_sidecar.runtime import action_arbiter as arbiter_module
from ai_sidecar.runtime.action_arbiter import ActionArbiter
from ai_sidecar.runtime.action_queue import ActionQueue
from ai_sidecar.runtime.snapshot_cache import SnapshotCache
from ai_sidecar.contracts.common import ContractMeta
from ai_sidecar.contracts.state import BotStateSnapshot


class _FleetClient:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled


def _make_proposal(
    action_id: str,
    *,
    created_at: datetime | None = None,
    priority: ActionPriorityTier = ActionPriorityTier.tactical,
    conflict_key: str | None = None,
    source: str | None = None,
    preconditions: list[str] | None = None,
    latency_budget_ms: int | None = None,
    lease_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> ActionProposal:
    now = created_at or datetime.now(UTC)
    payload: dict[str, object] = {
        "action_id": action_id,
        "kind": "command",
        "command": "sit",
        "priority_tier": priority,
        "conflict_key": conflict_key,
        "created_at": now,
        "expires_at": now + timedelta(seconds=30),
        "idempotency_key": f"idem:{action_id}",
        "metadata": dict(metadata or {}),
    }
    if source is not None:
        payload["source"] = source
    if preconditions is not None:
        payload["preconditions"] = preconditions
    if latency_budget_ms is not None:
        payload["latency_budget_ms"] = latency_budget_ms
    if lease_id is not None:
        payload["lease_id"] = lease_id
    return ActionProposal(**payload)


def test_action_arbiter_basic_admission() -> None:
    queue = ActionQueue(max_per_bot=8)
    arbiter = ActionArbiter(queue=queue, fleet_client=None)
    proposal = _make_proposal("action.basic")

    result = arbiter.admit_sync(proposal, bot_id="bot:basic")

    assert result.admitted is True
    assert result.status == ActionStatus.queued
    assert queue.count("bot:basic") == 1


def test_action_arbiter_precondition_rejection() -> None:
    queue = ActionQueue(max_per_bot=8)
    snapshot_cache = SnapshotCache(ttl_seconds=60)
    arbiter = ActionArbiter(queue=queue, fleet_client=None, snapshot_cache=snapshot_cache)
    proposal = _make_proposal("action.precond", preconditions=["navigation.ready"])

    result = arbiter.admit_sync(proposal, bot_id="bot:precond")

    assert result.admitted is True
    assert result.status == ActionStatus.queued
    assert queue.count("bot:precond") == 1


def test_action_arbiter_precondition_snapshot_missing_admits() -> None:
    queue = ActionQueue(max_per_bot=8)
    snapshot_cache = SnapshotCache(ttl_seconds=60)
    arbiter = ActionArbiter(queue=queue, fleet_client=None, snapshot_cache=snapshot_cache)
    proposal = _make_proposal("action.precond.missing", preconditions=["navigation.ready"])

    result = arbiter.admit_sync(proposal, bot_id="bot:precond.missing")

    assert result.admitted is True
    assert result.status == ActionStatus.queued


def test_action_arbiter_precondition_passes_with_snapshot() -> None:
    queue = ActionQueue(max_per_bot=8)
    snapshot_cache = SnapshotCache(ttl_seconds=60)
    arbiter = ActionArbiter(queue=queue, fleet_client=None, snapshot_cache=snapshot_cache)
    proposal = _make_proposal("action.precond.pass", preconditions=["navigation.ready"])

    snapshot_cache.set(
        BotStateSnapshot(
            meta=ContractMeta(bot_id="bot:precond.pass"),
            tick_id="t-1",
            observed_at=datetime.now(UTC),
            position={"map": "prt_fild01", "x": 5, "y": 9},
            vitals={"hp": 100, "hp_max": 100},
            raw={"status": "alive"},
        )
    )

    result = arbiter.admit_sync(proposal, bot_id="bot:precond.pass")

    assert result.admitted is True
    assert result.status == ActionStatus.queued


def test_action_arbiter_precondition_fail_rejects() -> None:
    queue = ActionQueue(max_per_bot=8)
    snapshot_cache = SnapshotCache(ttl_seconds=60)
    arbiter = ActionArbiter(queue=queue, fleet_client=None, snapshot_cache=snapshot_cache)
    proposal = _make_proposal("action.precond.fail", preconditions=["progression.skill_points_available"])

    snapshot_cache.set(
        BotStateSnapshot(
            meta=ContractMeta(bot_id="bot:precond.fail"),
            tick_id="t-2",
            observed_at=datetime.now(UTC),
            position={"map": "prt_fild02"},
            progression={"skill_points": 0},
        )
    )

    result = arbiter.admit_sync(proposal, bot_id="bot:precond.fail")

    assert result.admitted is False
    assert result.reason == "precondition_failed:progression.skill_points_available"
    assert queue.count("bot:precond.fail") == 0


def test_action_arbiter_conflict_detection_blocks_lower_priority() -> None:
    queue = ActionQueue(max_per_bot=8)
    arbiter = ActionArbiter(queue=queue, fleet_client=None)
    now = datetime.now(UTC)
    primary = _make_proposal("action.primary", conflict_key="conflict.test", created_at=now)
    secondary = _make_proposal(
        "action.secondary",
        conflict_key="conflict.test",
        created_at=now + timedelta(seconds=5),
    )

    primary_result = arbiter.admit_sync(primary, bot_id="bot:conflict")
    secondary_result = arbiter.admit_sync(secondary, bot_id="bot:conflict")

    assert primary_result.admitted is True
    assert secondary_result.admitted is False
    assert secondary_result.reason == "conflict_key_blocked"
    assert secondary_result.action_id == primary.action_id


def test_action_arbiter_fleet_lease_required() -> None:
    queue = ActionQueue(max_per_bot=8)
    arbiter = ActionArbiter(queue=queue, fleet_client=_FleetClient(enabled=False))
    proposal = _make_proposal("action.lease", lease_id="lease-1")

    result = arbiter.admit_sync(proposal, bot_id="bot:fleet")

    assert result.admitted is False
    assert result.reason == "fleet_lease_unavailable"
    assert queue.count("bot:fleet") == 0


def test_action_arbiter_latency_budget_enforced(monkeypatch) -> None:
    queue = ActionQueue(max_per_bot=8)
    arbiter = ActionArbiter(queue=queue, fleet_client=None)
    proposal = _make_proposal("action.latency", latency_budget_ms=1)
    times = iter([0.0, 0.01])
    monkeypatch.setattr(arbiter_module, "perf_counter", lambda: next(times))

    result = arbiter.admit_sync(proposal, bot_id="bot:latency")

    assert result.admitted is False
    assert result.reason == "latency_budget_exceeded"


def test_action_arbiter_source_tracking_from_metadata() -> None:
    queue = ActionQueue(max_per_bot=8)
    arbiter = ActionArbiter(queue=queue, fleet_client=None)
    proposal = _make_proposal("action.source", metadata={"source": "planner"})

    result = arbiter.admit_sync(proposal, bot_id="bot:source")

    assert result.admitted is True
    snapshot = queue.snapshot().get("bot:source") or []
    assert snapshot
    assert snapshot[0].proposal.source == "planner"
