from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import RLock
from uuid import uuid4

from ai_sidecar.config import settings
from ai_sidecar.contracts.actions import ActionAckRequest, ActionPriorityTier, ActionProposal, ActionStatus
from ai_sidecar.contracts.macros import MacroPublishRequest
from ai_sidecar.contracts.state import BotStateSnapshot
from ai_sidecar.contracts.telemetry import TelemetryEvent, TelemetryIngestResponse
from ai_sidecar.domain.macro_compiler import MacroCompiler, MacroPublisher
from ai_sidecar.runtime.action_queue import ActionQueue
from ai_sidecar.runtime.bot_registry import BotRegistry
from ai_sidecar.runtime.latency_router import LatencyRouter
from ai_sidecar.runtime.snapshot_cache import SnapshotCache

logger = logging.getLogger(__name__)


def _default_macro_plugin_name() -> str:
    return "macro"


def _default_event_macro_plugin_name() -> str:
    return "eventMacro"


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
    macro_compiler: MacroCompiler
    macro_publisher: MacroPublisher
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
        accepted, status, action_id, reason = self.action_queue.enqueue(bot_id, proposal)
        self.incr("actions_queued")
        return accepted, status, action_id, reason

    def next_action(self, bot_id: str) -> ActionProposal | None:
        self.bot_registry.upsert(bot_id)
        self.incr("actions_polled")
        return self.action_queue.fetch_next(bot_id)

    def rollback_action_dispatch(self, action_id: str) -> bool:
        return self.action_queue.rollback_dispatched(action_id)

    def acknowledge(self, ack: ActionAckRequest) -> tuple[bool, ActionStatus]:
        self.incr("actions_acknowledged")
        return self.action_queue.acknowledge(ack.action_id, ack.success, ack.message)

    def publish_macros(self, request: MacroPublishRequest) -> tuple[bool, dict[str, object] | None, str]:
        target_bot_id = request.target_bot_id or request.meta.bot_id
        try:
            compiled = self.macro_compiler.compile(
                macros=request.macros,
                event_macros=request.event_macros,
                automacros=request.automacros,
            )
            self.macro_publisher.publish(compiled)
        except Exception as exc:
            logger.exception(
                "macro_publish_failed",
                extra={"event": "macro_publish_failed", "bot_id": target_bot_id},
            )
            return False, None, f"macro publication failed: {exc}"

        publication_info: dict[str, object] = {
            "publication_id": compiled.publication_id,
            "version": compiled.version,
            "content_sha256": compiled.content_sha256,
            "published_at": compiled.published_at,
            "macro_file": self.macro_publisher.relpath(self.macro_publisher.macro_file),
            "event_macro_file": self.macro_publisher.relpath(self.macro_publisher.event_macro_file),
            "catalog_file": self.macro_publisher.relpath(self.macro_publisher.catalog_file),
            "manifest_file": self.macro_publisher.relpath(self.macro_publisher.manifest_file),
        }

        if request.enqueue_reload:
            now = datetime.now(UTC)
            action_id = f"macro-reload-{uuid4().hex[:18]}"
            reload_proposal = ActionProposal(
                action_id=action_id,
                kind="macro_reload",
                command="",
                priority_tier=ActionPriorityTier.macro_management,
                conflict_key=request.reload_conflict_key,
                created_at=now,
                expires_at=now + timedelta(seconds=settings.action_default_ttl_seconds),
                idempotency_key=f"macro-reload:{compiled.content_sha256}",
                metadata={
                    "publication_id": compiled.publication_id,
                    "version": compiled.version,
                    "macro_file": self.macro_publisher.macro_file.name,
                    "event_macro_file": self.macro_publisher.event_macro_file.name,
                    "catalog_file": self.macro_publisher.catalog_file.name,
                    "manifest_file": self.macro_publisher.manifest_file.name,
                    "macro_plugin": request.macro_plugin or _default_macro_plugin_name(),
                    "event_macro_plugin": request.event_macro_plugin or _default_event_macro_plugin_name(),
                },
            )

            accepted, status, action_id, reason = self.queue_action(reload_proposal, bot_id=target_bot_id)
            publication_info["reload_queued"] = accepted
            publication_info["reload_action_id"] = action_id
            publication_info["reload_status"] = status
            publication_info["reload_reason"] = reason
        else:
            publication_info["reload_queued"] = False
            publication_info["reload_action_id"] = None
            publication_info["reload_status"] = None
            publication_info["reload_reason"] = "reload_not_requested"

        return True, publication_info, "macro artifacts published"


def create_runtime() -> RuntimeState:
    workspace_root = Path(__file__).resolve().parents[2]
    runtime = RuntimeState(
        started_at=datetime.now(UTC),
        bot_registry=BotRegistry(),
        snapshot_cache=SnapshotCache(ttl_seconds=settings.snapshot_cache_ttl_seconds),
        action_queue=ActionQueue(max_per_bot=settings.action_max_queue_per_bot),
        latency_router=LatencyRouter(budget_ms=settings.latency_budget_ms),
        telemetry_store=TelemetryStore(max_per_bot=settings.telemetry_max_per_bot),
        macro_compiler=MacroCompiler(),
        macro_publisher=MacroPublisher(workspace_root=workspace_root),
    )
    logger.info("runtime_initialized", extra={"event": "runtime_initialized"})
    return runtime
