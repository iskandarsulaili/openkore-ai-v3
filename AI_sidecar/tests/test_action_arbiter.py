from __future__ import annotations

import logging
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
    command: str = "sit",
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
        "command": command,
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


def test_action_arbiter_precondition_snapshot_missing_logs_info_then_threshold_warning(caplog) -> None:
    queue = ActionQueue(max_per_bot=8)
    snapshot_cache = SnapshotCache(ttl_seconds=60)
    arbiter = ActionArbiter(queue=queue, fleet_client=None, snapshot_cache=snapshot_cache)

    caplog.set_level(logging.INFO, logger="ai_sidecar.runtime.action_arbiter")
    for idx in range(arbiter_module._PRECONDITION_SNAPSHOT_MISSING_WARN_EVERY):
        proposal = _make_proposal(f"action.precond.missing.threshold.{idx}", preconditions=["navigation.ready"])
        result = arbiter.admit_sync(proposal, bot_id="bot:precond.missing.threshold")
        assert result.admitted is True
        assert result.status == ActionStatus.queued

    missing_records = [
        item
        for item in caplog.records
        if getattr(item, "event", "") == "action_admission_precondition_snapshot_missing"
    ]
    assert len(missing_records) == 2

    info_records = [item for item in missing_records if item.levelno == logging.INFO]
    warning_records = [item for item in missing_records if item.levelno == logging.WARNING]
    assert len(info_records) == 1
    assert len(warning_records) == 1
    assert getattr(info_records[0], "miss_count", None) == 1
    assert getattr(warning_records[0], "miss_count", None) == arbiter_module._PRECONDITION_SNAPSHOT_MISSING_WARN_EVERY


def test_action_arbiter_precondition_snapshot_missing_counter_resets_after_snapshot(caplog) -> None:
    queue = ActionQueue(max_per_bot=8)
    snapshot_cache = SnapshotCache(ttl_seconds=60)
    arbiter = ActionArbiter(queue=queue, fleet_client=None, snapshot_cache=snapshot_cache)

    caplog.set_level(logging.INFO, logger="ai_sidecar.runtime.action_arbiter")

    proposal_first = _make_proposal("action.precond.missing.reset.first", preconditions=["navigation.ready"])
    first = arbiter.admit_sync(proposal_first, bot_id="bot:precond.missing.reset")
    assert first.admitted is True
    assert first.status == ActionStatus.queued

    snapshot_cache.set(
        BotStateSnapshot(
            meta=ContractMeta(bot_id="bot:precond.missing.reset"),
            tick_id="t-reset-valid",
            observed_at=datetime.now(UTC),
            position={"map": "prt_fild01", "x": 5, "y": 9},
            vitals={"hp": 100, "hp_max": 100},
            raw={"status": "alive"},
        )
    )

    proposal_second = _make_proposal("action.precond.missing.reset.second", preconditions=["navigation.ready"])
    second = arbiter.admit_sync(proposal_second, bot_id="bot:precond.missing.reset")
    assert second.admitted is True
    assert second.status == ActionStatus.queued

    snapshot_cache.set(
        BotStateSnapshot(
            meta=ContractMeta(bot_id="bot:precond.missing.reset"),
            tick_id="t-reset-stale",
            observed_at=datetime.now(UTC) - timedelta(seconds=120),
            position={"map": "prt_fild01", "x": 5, "y": 9},
            vitals={"hp": 100, "hp_max": 100},
            raw={"status": "alive"},
        )
    )

    proposal_third = _make_proposal("action.precond.missing.reset.third", preconditions=["navigation.ready"])
    third = arbiter.admit_sync(proposal_third, bot_id="bot:precond.missing.reset")
    assert third.admitted is True
    assert third.status == ActionStatus.queued

    missing_records = [
        item
        for item in caplog.records
        if getattr(item, "event", "") == "action_admission_precondition_snapshot_missing"
    ]
    assert len(missing_records) == 2
    assert all(item.levelno == logging.INFO for item in missing_records)
    assert [getattr(item, "miss_count", None) for item in missing_records] == [1, 1]


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


def test_action_arbiter_random_walk_rejected_without_scan_precondition() -> None:
    queue = ActionQueue(max_per_bot=8)
    arbiter = ActionArbiter(queue=queue, fleet_client=None)
    proposal = _make_proposal(
        "action.randomwalk.missing.scan",
        command="move random_walk_seek",
        metadata={"seek_only_random_walk": True, "target_scan_required": True},
    )

    result = arbiter.admit_sync(proposal, bot_id="bot:rw:missing")

    assert result.admitted is False
    assert result.reason == "random_walk_requires_target_scan"
    assert queue.count("bot:rw:missing") == 0


def test_action_arbiter_random_walk_rejected_when_targets_present() -> None:
    queue = ActionQueue(max_per_bot=8)
    arbiter = ActionArbiter(queue=queue, fleet_client=None)
    proposal = _make_proposal(
        "action.randomwalk.targets.present",
        command="move random_walk_seek",
        preconditions=["scan.targets_absent"],
        metadata={
            "seek_only_random_walk": True,
            "target_scan_required": True,
            "target_scan": {"targets_found": True, "source": "pytest"},
        },
    )

    result = arbiter.admit_sync(proposal, bot_id="bot:rw:present")

    assert result.admitted is False
    assert result.reason == "random_walk_target_scan_failed"
    assert queue.count("bot:rw:present") == 0


def test_action_arbiter_random_walk_admits_with_scan_absent_evidence() -> None:
    queue = ActionQueue(max_per_bot=8)
    arbiter = ActionArbiter(queue=queue, fleet_client=None)
    proposal = _make_proposal(
        "action.randomwalk.scan.absent",
        command="move random_walk_seek",
        preconditions=["scan.targets_absent"],
        metadata={
            "seek_only_random_walk": True,
            "target_scan_required": True,
            "target_scan": {"targets_found": False, "nearby_hostiles": 0, "source": "pytest"},
        },
    )

    result = arbiter.admit_sync(proposal, bot_id="bot:rw:absent")

    assert result.admitted is True
    assert result.status == ActionStatus.queued
    assert queue.count("bot:rw:absent") == 1
