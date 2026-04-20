from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock

from ai_sidecar.config import settings
from ai_sidecar.contracts.actions import ActionAckRequest, ActionProposal
from ai_sidecar.contracts.state import BotStateSnapshot
from ai_sidecar.contracts.telemetry import TelemetryEvent, TelemetryIngestResponse
from ai_sidecar.runtime.action_queue import ActionQueue
from ai_sidecar.runtime.bot_registry import BotRegistry
from ai_sidecar.runtime.latency_router import LatencyRouter
from ai_sidecar.runtime.snapshot_cache import SnapshotCache

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TelemetryStore:
    max_per_bot: int
    _lock: RLock = field(default_factory=RLock)
    _events: dict[str, list[TelemetryEvent]] = field(default_factory=dict)

    def push(self, bot_id: str, events: list[TelemetryEvent]) -> TelemetryIngestResponse:
        with self._lock:
            bucket = self._events.setdefault(bot_id, [])
            before = len(bucket)
            bucket.extend(events)
            dropped = 0
            if len(bucket) > self.max_per_bot:
                dropped = len(bucket) - self.max_per_bot
                del bucket[:dropped]
            accepted = len(bucket) - before
            return TelemetryIngestResponse(ok=True, accepted=accepted, dropped=dropped)

    def count(self, bot_id: str | None = None) -> int:
        with self._lock:
            if bot_id is None:
                return sum(len(items) for items in self._events.values())
            return len(self._events.get(bot_id, []))


@dataclass(slots=True)
class RuntimeState:
    started_at: datetime
    bot_registry: BotRegistry
    snapshot_cache: SnapshotCache
    action_queue: ActionQueue
    latency_router: LatencyRouter
    telemetry_store: TelemetryStore
    _counter_lock: RLock = field(default_factory=RLock)
    counters: dict[str, int] = field(default_factory=dict)

    def incr(self, key: str, n: int = 1) -> None:
        with self._counter_lock:
            self.counters[key] = self.counters.get(key, 0) + n

    def counter_snapshot(self) -> dict[str, int]:
        with self._counter_lock:
            return dict(self.counters)

    def ingest_snapshot(self, snapshot: BotStateSnapshot) -> None:
        self.bot_registry.upsert(snapshot.meta.bot_id, snapshot.tick_id)
        self.snapshot_cache.set(snapshot)
        self.incr("snapshots_ingested")

    def queue_action(self, proposal: ActionProposal, bot_id: str):
        accepted, status, action_id = self.action_queue.enqueue(bot_id, proposal)
        self.incr("actions_queued")
        return accepted, status, action_id

    def next_action(self, bot_id: str) -> ActionProposal | None:
        self.bot_registry.upsert(bot_id)
        self.incr("actions_polled")
        return self.action_queue.fetch_next(bot_id)

    def acknowledge(self, ack: ActionAckRequest) -> tuple[bool, str]:
        self.incr("actions_acknowledged")
        return self.action_queue.acknowledge(ack.action_id, ack.success, ack.message)


def create_runtime() -> RuntimeState:
    runtime = RuntimeState(
        started_at=datetime.now(UTC),
        bot_registry=BotRegistry(),
        snapshot_cache=SnapshotCache(ttl_seconds=settings.snapshot_cache_ttl_seconds),
        action_queue=ActionQueue(max_per_bot=settings.action_max_queue_per_bot),
        latency_router=LatencyRouter(budget_ms=settings.latency_budget_ms),
        telemetry_store=TelemetryStore(max_per_bot=settings.telemetry_max_per_bot),
    )
    logger.info("runtime_initialized", extra={"event": "runtime_initialized"})
    return runtime
