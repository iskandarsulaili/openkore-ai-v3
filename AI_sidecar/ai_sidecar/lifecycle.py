from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Callable, TypeVar
from uuid import uuid4

from ai_sidecar.config import settings
from ai_sidecar.contracts.actions import ActionAckRequest, ActionPriorityTier, ActionProposal, ActionStatus
from ai_sidecar.contracts.common import ContractMeta
from ai_sidecar.contracts.events import (
    ActorDeltaPushRequest,
    ChatStreamIngestRequest,
    ConfigDoctrineFingerprintRequest,
    EventFamily,
    EventBatchIngestRequest,
    EventSeverity,
    IngestAcceptedResponse,
    NormalizedEvent,
    QuestTransitionRequest,
)
from ai_sidecar.contracts.macros import MacroPublishRequest
from ai_sidecar.contracts.reflex import ReflexRule
from ai_sidecar.contracts.reflex import ReflexBreakerStatusView, ReflexTriggerRecord
from ai_sidecar.contracts.state import BotRegistrationRequest, BotStateSnapshot
from ai_sidecar.contracts.state_graph import EnrichedWorldState
from ai_sidecar.contracts.telemetry import TelemetryEvent, TelemetryIngestResponse
from ai_sidecar.domain.macro_compiler import MacroCompiler, MacroPublisher
from ai_sidecar.ingestion.event_journal import EventJournal
from ai_sidecar.ingestion.adapters.actor_state_adapter import actor_delta_to_events
from ai_sidecar.ingestion.adapters.chat_adapter import chat_stream_to_events
from ai_sidecar.ingestion.adapters.config_adapter import config_update_to_events
from ai_sidecar.ingestion.adapters.quest_adapter import quest_transition_to_events
from ai_sidecar.ingestion.normalizer_bus import NormalizerBus
from ai_sidecar.memory.embeddings import LocalSemanticEmbedder
from ai_sidecar.memory.episodic_store import EpisodicMemoryStore
from ai_sidecar.memory.retrieval import (
    InMemoryMemoryProvider,
    MemoryRetrievalService,
    OpenMemoryProvider,
    SQLiteMemoryProvider,
)
from ai_sidecar.memory.semantic_store import SemanticMemoryStore
from ai_sidecar.observability.audit import AuditTrail
from ai_sidecar.observability.metrics import DurableTelemetryIngestor
from ai_sidecar.persistence.db import SQLiteDB
from ai_sidecar.persistence.repositories import SidecarRepositories, create_repositories
from ai_sidecar.runtime.action_queue import ActionQueue
from ai_sidecar.runtime.bot_registry import BotRegistry
from ai_sidecar.runtime.latency_router import LatencyRouter
from ai_sidecar.runtime.snapshot_cache import SnapshotCache
from ai_sidecar.reflex.rule_engine import ReflexRuleEngine

logger = logging.getLogger(__name__)
T = TypeVar("T")


def _default_macro_plugin_name() -> str:
    return "macro"


def _default_event_macro_plugin_name() -> str:
    return "eventMacro"


def _resolve_path(workspace_root: Path, configured_path: str) -> Path:
    candidate = Path(configured_path)
    if candidate.is_absolute():
        return candidate
    return workspace_root / candidate


@dataclass(slots=True)
class TelemetryStore:
    max_per_bot: int
    _ingestor: DurableTelemetryIngestor | None = None
    _repository: object | None = None
    _lock: RLock = field(default_factory=RLock)
    _events: dict[str, list[TelemetryEvent]] = field(default_factory=dict)

    def push(self, bot_id: str, events: list[TelemetryEvent]) -> TelemetryIngestResponse:
        if not events:
            return TelemetryIngestResponse(ok=True, accepted=0, dropped=0, queued_for_retry=0)

        if self._ingestor is not None:
            return self._ingestor.ingest(bot_id=bot_id, events=events)

        with self._lock:
            bucket = self._events.setdefault(bot_id, [])
            before = len(bucket)
            bucket.extend(events)
            dropped = 0
            if len(bucket) > self.max_per_bot:
                dropped = len(bucket) - self.max_per_bot
                del bucket[:dropped]
            accepted = len(bucket) - before
            return TelemetryIngestResponse(ok=True, accepted=accepted, dropped=dropped, queued_for_retry=0)

    def count(self, bot_id: str | None = None) -> int:
        repository = self._repository
        if repository is not None:
            try:
                return int(repository.count(bot_id=bot_id))
            except Exception:
                logger.exception(
                    "telemetry_count_failed",
                    extra={"event": "telemetry_count_failed", "bot_id": bot_id},
                )

        with self._lock:
            if bot_id is None:
                return sum(len(items) for items in self._events.values())
            return len(self._events.get(bot_id, []))

    def backlog_size(self) -> int:
        if self._ingestor is None:
            return 0
        return self._ingestor.backlog_size()


@dataclass(slots=True)
class RuntimeState:
    started_at: datetime
    workspace_root: Path
    bot_registry: BotRegistry
    snapshot_cache: SnapshotCache
    action_queue: ActionQueue
    latency_router: LatencyRouter
    telemetry_store: TelemetryStore
    macro_compiler: MacroCompiler
    macro_publisher: MacroPublisher
    memory: MemoryRetrievalService
    audit_trail: AuditTrail | None
    repositories: SidecarRepositories | None
    normalizer_bus: NormalizerBus
    reflex_engine: ReflexRuleEngine
    sqlite_path: Path | None
    _counter_lock: RLock = field(default_factory=RLock)
    counters: dict[str, int] = field(default_factory=dict)
    persistence_degraded: bool = False

    def incr(self, key: str, n: int = 1, *, bot_id: str | None = None) -> None:
        with self._counter_lock:
            self.counters[key] = self.counters.get(key, 0) + n
        self._safe_persist(
            "increment_counter",
            lambda: self.repositories.telemetry.increment_counter(bot_id=bot_id or "fleet", name=key, delta=n),
            bot_id=bot_id,
        )

    def counter_snapshot(self) -> dict[str, int]:
        with self._counter_lock:
            return dict(self.counters)

    def register_bot(self, payload: BotRegistrationRequest) -> dict[str, object]:
        bot_id = payload.meta.bot_id
        in_memory = self.bot_registry.upsert(bot_id)
        self.reflex_engine.ensure_bot(bot_id=bot_id)

        persisted = self._safe_persist(
            "register_bot",
            lambda: self.repositories.bots.upsert_registration(
                bot_id=bot_id,
                bot_name=payload.bot_name,
                capabilities=payload.capabilities,
                attributes=payload.attributes,
                role=payload.role,
                assignment=payload.assignment,
                tick_id=None,
                liveness_state="online",
            ),
            bot_id=bot_id,
        )

        self.incr("bot_registrations", bot_id=bot_id)
        self._audit(
            level="info",
            event_type="bot_registration",
            summary="bot registered",
            bot_id=bot_id,
            payload={
                "bot_name": payload.bot_name,
                "role": payload.role,
                "assignment": payload.assignment,
                "capabilities": payload.capabilities,
            },
        )

        if persisted is not None:
            return {
                "seen_at": persisted.last_seen_at,
                "role": persisted.role,
                "assignment": persisted.assignment,
                "liveness_state": persisted.liveness_state,
            }

        return {
            "seen_at": in_memory.last_seen_at,
            "role": payload.role,
            "assignment": payload.assignment,
            "liveness_state": "online",
        }

    def ingest_snapshot(self, snapshot: BotStateSnapshot) -> None:
        bot_id = snapshot.meta.bot_id
        self.bot_registry.upsert(bot_id, snapshot.tick_id)
        self.snapshot_cache.set(snapshot)
        self.incr("snapshots_ingested", bot_id=bot_id)

        self._safe_persist(
            "persist_snapshot",
            lambda: self.repositories.bots.touch(bot_id=bot_id, tick_id=snapshot.tick_id, liveness_state="online")
            if self.repositories
            else None,
            bot_id=bot_id,
        )
        self._safe_persist(
            "persist_snapshot_history",
            lambda: self.repositories.snapshots.save_snapshot(snapshot) if self.repositories else None,
            bot_id=bot_id,
        )

        summary = (
            f"tick={snapshot.tick_id} map={snapshot.position.map or 'unknown'} "
            f"pos=({snapshot.position.x},{snapshot.position.y}) "
            f"hp={snapshot.vitals.hp}/{snapshot.vitals.hp_max} "
            f"ai={snapshot.combat.ai_sequence or 'idle'}"
        )
        self._safe_memory(
            "capture_snapshot_memory",
            lambda: self.memory.capture_snapshot(
                bot_id=bot_id,
                tick_id=snapshot.tick_id,
                summary=summary,
                payload={
                    "map": snapshot.position.map,
                    "x": snapshot.position.x,
                    "y": snapshot.position.y,
                    "in_combat": snapshot.combat.is_in_combat,
                },
            ),
            bot_id=bot_id,
        )
        snapshot_event = NormalizedEvent(
            meta=snapshot.meta,
            observed_at=snapshot.observed_at,
            event_family=EventFamily.snapshot,
            event_type="snapshot.compact",
            source_hook="v1.ingest.snapshot",
            text="snapshot ingested",
            numeric={
                "hp": float(snapshot.vitals.hp or 0),
                "hp_max": float(snapshot.vitals.hp_max or 0),
                "sp": float(snapshot.vitals.sp or 0),
                "sp_max": float(snapshot.vitals.sp_max or 0),
                "x": float(snapshot.position.x or 0),
                "y": float(snapshot.position.y or 0),
                "zeny": float(snapshot.inventory.zeny or 0),
                "item_count": float(snapshot.inventory.item_count or 0),
            },
            tags={
                "tick_id": snapshot.tick_id,
                "map": snapshot.position.map or "",
                "ai_sequence": snapshot.combat.ai_sequence or "",
            },
            payload=snapshot.model_dump(mode="json"),
        )
        result = self.normalizer_bus.ingest_batch(EventBatchIngestRequest(meta=snapshot.meta, events=[snapshot_event]))
        self._evaluate_reflex_events(
            bot_id=bot_id,
            events=self._accepted_events(events=[snapshot_event], event_ids=result.event_ids),
            source="snapshot",
        )

    def ingest_event_batch(self, payload: EventBatchIngestRequest) -> IngestAcceptedResponse:
        result = self.normalizer_bus.ingest_batch(payload)
        self._evaluate_reflex_events(
            bot_id=payload.meta.bot_id,
            events=self._accepted_events(events=payload.events, event_ids=result.event_ids),
            source="v2.event",
        )
        return result

    def ingest_actor_delta(self, payload: ActorDeltaPushRequest) -> IngestAcceptedResponse:
        events = actor_delta_to_events(payload)
        result = self.normalizer_bus.ingest_batch(EventBatchIngestRequest(meta=payload.meta, events=events))
        self._evaluate_reflex_events(
            bot_id=payload.meta.bot_id,
            events=self._accepted_events(events=events, event_ids=result.event_ids),
            source="v2.actors",
        )
        return result

    def ingest_chat_stream(self, payload: ChatStreamIngestRequest) -> IngestAcceptedResponse:
        events = chat_stream_to_events(payload)
        result = self.normalizer_bus.ingest_batch(EventBatchIngestRequest(meta=payload.meta, events=events))
        self._evaluate_reflex_events(
            bot_id=payload.meta.bot_id,
            events=self._accepted_events(events=events, event_ids=result.event_ids),
            source="v2.chat",
        )
        return result

    def ingest_config_update(self, payload: ConfigDoctrineFingerprintRequest) -> IngestAcceptedResponse:
        events = config_update_to_events(payload)
        result = self.normalizer_bus.ingest_batch(EventBatchIngestRequest(meta=payload.meta, events=events))
        self._evaluate_reflex_events(
            bot_id=payload.meta.bot_id,
            events=self._accepted_events(events=events, event_ids=result.event_ids),
            source="v2.config",
        )
        return result

    def ingest_quest_transition(self, payload: QuestTransitionRequest) -> IngestAcceptedResponse:
        events = quest_transition_to_events(payload)
        result = self.normalizer_bus.ingest_batch(EventBatchIngestRequest(meta=payload.meta, events=events))
        self._evaluate_reflex_events(
            bot_id=payload.meta.bot_id,
            events=self._accepted_events(events=events, event_ids=result.event_ids),
            source="v2.quest",
        )
        return result

    def enriched_state(self, *, bot_id: str) -> EnrichedWorldState:
        return self.normalizer_bus.enriched_state(bot_id=bot_id)

    def normalized_state_graph(self, *, bot_id: str) -> dict[str, object]:
        return self.normalizer_bus.debug_graph(bot_id=bot_id)

    def recent_ingest_events(self, *, bot_id: str, limit: int = 100) -> list[dict[str, object]]:
        return self.normalizer_bus.recent_events(bot_id=bot_id, limit=limit)

    def upsert_reflex_rule(self, *, bot_id: str, rule: ReflexRule) -> tuple[bool, str]:
        try:
            self.reflex_engine.upsert_rule(bot_id=bot_id, rule=rule)
            self._audit(
                level="info",
                event_type="reflex_rule_upsert",
                summary="reflex rule upserted",
                bot_id=bot_id,
                payload={"rule_id": rule.rule_id, "enabled": rule.enabled, "priority": rule.priority},
            )
            return True, "rule_saved"
        except Exception as exc:
            logger.exception(
                "reflex_rule_upsert_failed",
                extra={"event": "reflex_rule_upsert_failed", "bot_id": bot_id, "rule_id": rule.rule_id},
            )
            return False, str(exc)

    def list_reflex_rules(self, *, bot_id: str) -> list[ReflexRule]:
        return self.reflex_engine.list_rules(bot_id=bot_id)

    def enable_reflex_rule(self, *, bot_id: str, rule_id: str, enabled: bool) -> bool:
        changed = self.reflex_engine.set_rule_enabled(bot_id=bot_id, rule_id=rule_id, enabled=enabled)
        if changed:
            self._audit(
                level="info",
                event_type="reflex_rule_enable",
                summary="reflex rule enablement updated",
                bot_id=bot_id,
                payload={"rule_id": rule_id, "enabled": enabled},
            )
        return changed

    def recent_reflex_triggers(self, *, bot_id: str, limit: int = 100) -> list[ReflexTriggerRecord]:
        return self.reflex_engine.recent_triggers(bot_id=bot_id, limit=limit)

    def reflex_breakers(self, *, bot_id: str) -> list[ReflexBreakerStatusView]:
        return self.reflex_engine.list_breakers(bot_id=bot_id)

    def queue_action(self, proposal: ActionProposal, bot_id: str) -> tuple[bool, ActionStatus, str, str]:
        accepted, status, action_id, reason = self.action_queue.enqueue(bot_id, proposal)
        self.incr("actions_queued", bot_id=bot_id)

        if action_id == proposal.action_id:
            self._safe_persist(
                "persist_action_proposal",
                lambda: self.repositories.actions.upsert_action(
                    bot_id=bot_id,
                    proposal=proposal,
                    status=status,
                    status_reason=reason,
                )
                if self.repositories
                else None,
                bot_id=bot_id,
            )

        self._safe_memory(
            "capture_action_queue_memory",
            lambda: self.memory.capture_action(
                bot_id=bot_id,
                action_id=action_id,
                kind=proposal.kind,
                message=f"queued={accepted} status={status.value} reason={reason}",
                metadata={"phase": "queue", "idempotency_key": proposal.idempotency_key},
            ),
            bot_id=bot_id,
        )
        self._audit(
            level="info",
            event_type="action_queue",
            summary="action queue decision",
            bot_id=bot_id,
            payload={
                "action_id": action_id,
                "requested_action_id": proposal.action_id,
                "accepted": accepted,
                "status": status.value,
                "reason": reason,
            },
        )
        self._emit_runtime_event(
            bot_id=bot_id,
            event_family=EventFamily.action,
            event_type="action.queue_decision",
            severity=EventSeverity.info if accepted else EventSeverity.warning,
            text=f"queue decision accepted={accepted} status={status.value} reason={reason}",
            numeric={"accepted": 1.0 if accepted else 0.0},
            payload={
                "action_id": action_id,
                "requested_action_id": proposal.action_id,
                "kind": proposal.kind,
                "status": status.value,
                "reason": reason,
                "priority_tier": proposal.priority_tier.value,
                "conflict_key": proposal.conflict_key,
                "idempotency_key": proposal.idempotency_key,
            },
        )
        return accepted, status, action_id, reason

    def next_action(self, bot_id: str, poll_id: str | None = None) -> ActionProposal | None:
        self.bot_registry.upsert(bot_id)
        self.incr("actions_polled", bot_id=bot_id)
        self._safe_persist(
            "touch_bot_poll",
            lambda: self.repositories.bots.touch(bot_id=bot_id, tick_id=None, liveness_state="online")
            if self.repositories
            else None,
            bot_id=bot_id,
        )

        proposal = self.action_queue.fetch_next(bot_id)
        if proposal is not None:
            self._safe_persist(
                "mark_action_dispatched",
                lambda: self.repositories.actions.mark_dispatched(action_id=proposal.action_id, poll_id=poll_id)
                if self.repositories
                else None,
                bot_id=bot_id,
            )
            self._safe_memory(
                "capture_action_dispatch_memory",
                lambda: self.memory.capture_action(
                    bot_id=bot_id,
                    action_id=proposal.action_id,
                    kind=proposal.kind,
                    message="action dispatched",
                    metadata={"phase": "dispatch", "poll_id": poll_id},
                ),
                bot_id=bot_id,
            )
            self._emit_runtime_event(
                bot_id=bot_id,
                event_family=EventFamily.action,
                event_type="action.dispatched",
                severity=EventSeverity.info,
                text="action dispatched",
                payload={
                    "action_id": proposal.action_id,
                    "poll_id": poll_id,
                    "kind": proposal.kind,
                    "priority_tier": proposal.priority_tier.value,
                },
            )
        return proposal

    def rollback_action_dispatch(self, action_id: str) -> bool:
        rolled_back = self.action_queue.rollback_dispatched(action_id)
        if rolled_back:
            self._audit(
                level="warning",
                event_type="action_dispatch_rollback",
                summary="action dispatch rolled back due to budget/fallback",
                bot_id=None,
                payload={"action_id": action_id},
            )
        return rolled_back

    def acknowledge(self, ack: ActionAckRequest) -> tuple[bool, ActionStatus]:
        self.incr("actions_acknowledged", bot_id=ack.meta.bot_id)
        acknowledged, status = self.action_queue.acknowledge(ack.action_id, ack.success, ack.message)
        if acknowledged:
            self._safe_persist(
                "mark_action_acknowledged",
                lambda: self.repositories.actions.mark_acknowledged(
                    action_id=ack.action_id,
                    success=ack.success,
                    result_code=ack.result_code,
                    message=ack.message,
                    poll_id=ack.poll_id,
                )
                if self.repositories
                else None,
                bot_id=ack.meta.bot_id,
            )
            self._safe_memory(
                "capture_action_ack_memory",
                lambda: self.memory.capture_action(
                    bot_id=ack.meta.bot_id,
                    action_id=ack.action_id,
                    kind="ack",
                    message=f"acknowledged={ack.success} result={ack.result_code} msg={ack.message}",
                    metadata={"phase": "ack", "poll_id": ack.poll_id},
                ),
                bot_id=ack.meta.bot_id,
            )

        self._audit(
            level="info" if ack.success else "warning",
            event_type="action_ack",
            summary="action acknowledgement received",
            bot_id=ack.meta.bot_id,
            payload={
                "action_id": ack.action_id,
                "success": ack.success,
                "result_code": ack.result_code,
                "status": status.value,
            },
        )
        self._emit_runtime_event(
            bot_id=ack.meta.bot_id,
            event_family=EventFamily.action,
            event_type="action.acknowledged",
            severity=EventSeverity.info if ack.success else EventSeverity.warning,
            text=f"action ack success={ack.success} result={ack.result_code}",
            numeric={"success": 1.0 if ack.success else 0.0, "observed_latency_ms": float(ack.observed_latency_ms or 0.0)},
            payload={
                "action_id": ack.action_id,
                "poll_id": ack.poll_id,
                "success": ack.success,
                "result_code": ack.result_code,
                "message": ack.message,
                "status": status.value,
            },
        )
        self.reflex_engine.handle_ack(
            bot_id=ack.meta.bot_id,
            action_id=ack.action_id,
            success=ack.success,
            result_code=ack.result_code,
            message=ack.message,
        )
        return acknowledged, status

    def _accepted_events(self, *, events: list[NormalizedEvent], event_ids: list[str]) -> list[NormalizedEvent]:
        if not events:
            return []
        if not event_ids:
            return []
        accepted = set(event_ids)
        return [item for item in events if item.event_id in accepted]

    def _evaluate_reflex_events(self, *, bot_id: str, events: list[NormalizedEvent], source: str) -> None:
        if not events:
            return

        filtered_events = [
            item
            for item in events
            if not (
                item.meta.source == "sidecar-runtime"
                and item.event_family == EventFamily.action
                and item.event_type.startswith("action.")
            )
        ]
        if not filtered_events:
            return

        started = self.latency_router.begin()
        records = self.reflex_engine.evaluate_events(
            bot_id=bot_id,
            events=filtered_events,
            get_enriched_state=lambda *, bot_id=bot_id: self.enriched_state(bot_id=bot_id),
            queue_action=self.queue_action,
            publish_macros=self.publish_macros,
        )
        elapsed_ms = self.latency_router.end("reflex.evaluate", started)

        if elapsed_ms > float(settings.reflex_latency_budget_ms):
            self._audit(
                level="warning",
                event_type="reflex_latency_budget_exceeded",
                summary="reflex evaluation exceeded latency budget",
                bot_id=bot_id,
                payload={
                    "elapsed_ms": elapsed_ms,
                    "budget_ms": settings.reflex_latency_budget_ms,
                    "source": source,
                    "event_count": len(filtered_events),
                },
            )

        if not records:
            return

        self.incr("reflex_triggers_total", n=len(records), bot_id=bot_id)
        emitted_total = sum(1 for item in records if item.emitted)
        suppressed_total = sum(1 for item in records if item.suppressed)
        if emitted_total:
            self.incr("reflex_actions_emitted", n=emitted_total, bot_id=bot_id)
        if suppressed_total:
            self.incr("reflex_actions_suppressed", n=suppressed_total, bot_id=bot_id)

        for record in records:
            self._audit(
                level="warning" if record.suppressed else "info",
                event_type="reflex_trigger",
                summary=f"reflex {record.outcome} for rule {record.rule_id}",
                bot_id=bot_id,
                payload={
                    "trigger_id": record.trigger_id,
                    "rule_id": record.rule_id,
                    "event_id": record.event_id,
                    "event_type": record.event_type,
                    "suppressed": record.suppressed,
                    "suppression_reason": record.suppression_reason,
                    "emitted": record.emitted,
                    "execution_target": record.execution_target,
                    "action_id": record.action_id,
                    "latency_ms": record.latency_ms,
                    "outcome": record.outcome,
                    "detail": record.detail,
                },
            )

    def ingest_telemetry(self, bot_id: str, events: list[TelemetryEvent]) -> TelemetryIngestResponse:
        result = self.telemetry_store.push(bot_id, events)
        if result.accepted:
            self.incr("telemetry_ingested", n=result.accepted, bot_id=bot_id)
        if result.dropped:
            self.incr("telemetry_dropped", n=result.dropped, bot_id=bot_id)
        if result.queued_for_retry:
            self.incr("telemetry_queued_retry", n=result.queued_for_retry, bot_id=bot_id)

        self._audit(
            level="warning" if result.queued_for_retry else "info",
            event_type="telemetry_ingest",
            summary="telemetry ingest processed",
            bot_id=bot_id,
            payload={
                "accepted": result.accepted,
                "dropped": result.dropped,
                "queued_for_retry": result.queued_for_retry,
                "batch_size": len(events),
            },
        )

        if events:
            for item in events:
                severity = EventSeverity.info
                if item.level.value == "debug":
                    severity = EventSeverity.debug
                elif item.level.value == "warning":
                    severity = EventSeverity.warning
                elif item.level.value == "error":
                    severity = EventSeverity.error
                self._emit_runtime_event(
                    bot_id=bot_id,
                    event_family=EventFamily.telemetry,
                    event_type=f"telemetry.{item.category}.{item.event}",
                    severity=severity,
                    text=item.message,
                    numeric={key: float(value) for key, value in item.metrics.items()},
                    payload={
                        "category": item.category,
                        "event": item.event,
                        "tags": item.tags,
                    },
                )
        return result

    def telemetry_operational_summary(self, *, bot_id: str | None = None) -> dict[str, object]:
        if self.repositories is None:
            return {
                "window_minutes": settings.telemetry_operational_window_minutes,
                "window_since": datetime.now(UTC).isoformat(),
                "total_events": self.telemetry_store.count(bot_id=bot_id),
                "levels": {},
                "top_events": [],
                "recent_incidents": [],
            }
        return self._safe_persist(
            "telemetry_operational_summary",
            lambda: self.repositories.telemetry.operational_summary(
                bot_id=bot_id,
                incidents_limit=settings.telemetry_recent_incidents_limit,
            ),
            default={
                "window_minutes": settings.telemetry_operational_window_minutes,
                "window_since": datetime.now(UTC).isoformat(),
                "total_events": 0,
                "levels": {},
                "top_events": [],
                "recent_incidents": [],
            },
            bot_id=bot_id,
        )

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
            self._audit(
                level="error",
                event_type="macro_publish",
                summary="macro publication failed",
                bot_id=target_bot_id,
                payload={"error": str(exc)},
            )
            self._emit_runtime_event(
                bot_id=target_bot_id,
                event_family=EventFamily.macro,
                event_type="macro.publish_failed",
                severity=EventSeverity.error,
                text=f"macro publication failed: {exc}",
                payload={"error": str(exc)},
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

        self._safe_persist(
            "persist_macro_publication",
            lambda: self.repositories.macros.save_publication(
                bot_id=target_bot_id,
                publication_id=compiled.publication_id,
                version=compiled.version,
                content_sha256=compiled.content_sha256,
                published_at=compiled.published_at,
                manifest=compiled.manifest,
                paths={
                    "macro_file": publication_info["macro_file"],
                    "event_macro_file": publication_info["event_macro_file"],
                    "catalog_file": publication_info["catalog_file"],
                    "manifest_file": publication_info["manifest_file"],
                },
            )
            if self.repositories
            else None,
            bot_id=target_bot_id,
        )

        self._safe_memory(
            "capture_macro_publish_memory",
            lambda: self.memory.capture_action(
                bot_id=target_bot_id,
                action_id=f"macro-publish:{compiled.publication_id}",
                kind="macro_publish",
                message=f"published macro bundle {compiled.version}",
                metadata={"publication_id": compiled.publication_id, "sha256": compiled.content_sha256},
            ),
            bot_id=target_bot_id,
        )

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

        self._audit(
            level="info",
            event_type="macro_publish",
            summary="macro artifacts published",
            bot_id=target_bot_id,
            payload={
                "publication_id": compiled.publication_id,
                "version": compiled.version,
                "reload_queued": publication_info["reload_queued"],
            },
        )
        self._emit_runtime_event(
            bot_id=target_bot_id,
            event_family=EventFamily.macro,
            event_type="macro.published",
            severity=EventSeverity.info,
            text=f"published macro bundle {compiled.version}",
            payload={
                "publication_id": compiled.publication_id,
                "version": compiled.version,
                "reload_queued": publication_info["reload_queued"],
                "reload_action_id": publication_info["reload_action_id"],
                "reload_reason": publication_info["reload_reason"],
            },
        )
        return True, publication_info, "macro artifacts published"

    def list_bots(self) -> list[dict[str, object]]:
        items: dict[str, dict[str, object]] = {}

        if self.repositories is not None:
            records = self._safe_persist(
                "list_persisted_bots",
                lambda: self.repositories.bots.list_all(),
                default=[],
                bot_id=None,
            )
            for item in records:
                items[item.bot_id] = {
                    "bot_id": item.bot_id,
                    "bot_name": item.bot_name,
                    "role": item.role,
                    "assignment": item.assignment,
                    "capabilities": item.capabilities,
                    "attributes": item.attributes,
                    "first_seen_at": item.first_seen_at,
                    "last_seen_at": item.last_seen_at,
                    "last_tick_id": item.last_tick_id,
                    "liveness_state": item.liveness_state,
                }

        for item in self.bot_registry.list():
            existing = items.get(item.bot_id)
            if existing is None:
                items[item.bot_id] = {
                    "bot_id": item.bot_id,
                    "bot_name": None,
                    "role": None,
                    "assignment": None,
                    "capabilities": [],
                    "attributes": {},
                    "first_seen_at": item.first_seen_at,
                    "last_seen_at": item.last_seen_at,
                    "last_tick_id": item.last_tick_id,
                    "liveness_state": "online",
                }
            else:
                existing["last_seen_at"] = max(existing["last_seen_at"], item.last_seen_at)
                existing["last_tick_id"] = item.last_tick_id or existing["last_tick_id"]

        result: list[dict[str, object]] = []
        for bot_id, rec in items.items():
            latest_snapshot = self.snapshot_cache.get(bot_id)
            rec["pending_actions"] = self.action_queue.count(bot_id)
            rec["latest_snapshot_at"] = latest_snapshot.observed_at if latest_snapshot else None
            rec["telemetry_events"] = self.telemetry_store.count(bot_id=bot_id)
            result.append(rec)

        result.sort(key=lambda row: row.get("last_seen_at") or datetime.min.replace(tzinfo=UTC), reverse=True)
        return result

    def bot_status(self, bot_id: str) -> dict[str, object] | None:
        for item in self.list_bots():
            if item["bot_id"] == bot_id:
                return item
        return None

    def update_assignment(
        self,
        *,
        bot_id: str,
        role: str | None,
        assignment: str | None,
        attributes: dict[str, str],
    ) -> dict[str, object] | None:
        if self.repositories is None:
            return None
        updated = self._safe_persist(
            "update_bot_assignment",
            lambda: self.repositories.bots.update_assignment(
                bot_id=bot_id,
                role=role,
                assignment=assignment,
                attributes=attributes,
            ),
            bot_id=bot_id,
        )
        if updated is None:
            return None
        self._audit(
            level="info",
            event_type="bot_assignment",
            summary="bot assignment updated",
            bot_id=bot_id,
            payload={"role": role, "assignment": assignment, "attributes": attributes},
        )
        return self.bot_status(bot_id)

    def recent_actions(self, *, bot_id: str, limit: int) -> list[dict[str, object]]:
        if self.repositories is None:
            return []
        rows = self._safe_persist(
            "recent_actions",
            lambda: self.repositories.actions.list_recent(bot_id=bot_id, limit=limit),
            default=[],
            bot_id=bot_id,
        )
        return [
            {
                "action_id": item.action_id,
                "kind": item.kind,
                "status": item.status,
                "priority_tier": item.priority_tier,
                "created_at": item.created_at,
                "expires_at": item.expires_at,
                "queued_at": item.queued_at,
                "acknowledged_at": item.acknowledged_at,
                "ack_success": item.ack_success,
                "ack_result_code": item.ack_result_code,
                "ack_message": item.ack_message,
                "status_reason": item.status_reason,
            }
            for item in rows
        ]

    def recent_snapshots(self, *, bot_id: str, limit: int) -> list[dict[str, object]]:
        if self.repositories is None:
            return []
        rows = self._safe_persist(
            "recent_snapshots",
            lambda: self.repositories.snapshots.list_recent(bot_id=bot_id, limit=limit),
            default=[],
            bot_id=bot_id,
        )
        return [
            {
                "id": item.id,
                "tick_id": item.tick_id,
                "observed_at": item.observed_at,
                "ingested_at": item.ingested_at,
                "snapshot": item.snapshot,
            }
            for item in rows
        ]

    def latest_macro_publication(self, *, bot_id: str) -> dict[str, object] | None:
        if self.repositories is None:
            return None
        item = self._safe_persist(
            "latest_macro_publication",
            lambda: self.repositories.macros.latest_for_bot(bot_id),
            default=None,
            bot_id=bot_id,
        )
        if item is None:
            return None
        return {
            "publication_id": item.publication_id,
            "version": item.version,
            "content_sha256": item.content_sha256,
            "published_at": item.published_at,
            "paths": item.paths,
            "macro_count": item.macro_count,
            "event_macro_count": item.event_macro_count,
            "automacro_count": item.automacro_count,
        }

    def recent_audit(self, *, limit: int, bot_id: str | None = None, event_type: str | None = None) -> list[dict[str, object]]:
        if self.audit_trail is None:
            return []
        rows = self.audit_trail.recent(limit=limit, bot_id=bot_id, event_type=event_type)
        return [
            {
                "id": item["id"],
                "timestamp": item["timestamp"],
                "level": item["level"],
                "event_type": item["event_type"],
                "bot_id": item["bot_id"],
                "summary": item["summary"],
                "payload": item["payload"],
            }
            for item in rows
        ]

    def fleet_status(self) -> dict[str, object]:
        bots = self.list_bots()
        total_pending_actions = sum(int(item["pending_actions"]) for item in bots)
        online_bots = sum(1 for item in bots if item.get("liveness_state") == "online")

        status_totals: dict[str, int] = {}
        if self.repositories is not None:
            for status in ActionStatus:
                status_totals[status.value] = self._safe_persist(
                    "count_actions_by_status",
                    lambda status=status: self.repositories.actions.count(status=status),
                    default=0,
                    bot_id=None,
                )

        telemetry_window = self.telemetry_operational_summary(bot_id=None)
        counters = self.counter_snapshot()
        if self.repositories is not None:
            persisted_counters = self._safe_persist(
                "load_persisted_counters",
                lambda: self.repositories.telemetry.get_counters(bot_id=None),
                default={},
                bot_id=None,
            )
            for key, value in persisted_counters.items():
                counters.setdefault(key, value)

        return {
            "generated_at": datetime.now(UTC),
            "total_bots": len(bots),
            "online_bots": online_bots,
            "total_pending_actions": total_pending_actions,
            "action_status_totals": status_totals,
            "telemetry_window": telemetry_window,
            "counters": counters,
        }

    def memory_context(self, *, bot_id: str, query: str, limit: int) -> list[dict[str, object]]:
        return self._safe_memory(
            "memory_context_search",
            lambda: self.memory.search_context(bot_id=bot_id, query=query, limit=limit),
            default=[],
            bot_id=bot_id,
        )

    def memory_recent_episodes(self, *, bot_id: str, limit: int) -> list[dict[str, object]]:
        return self._safe_memory(
            "memory_recent_episodes",
            lambda: self.memory.recent_episodes(bot_id=bot_id, limit=limit),
            default=[],
            bot_id=bot_id,
        )

    def memory_stats(self, *, bot_id: str) -> dict[str, int]:
        return self._safe_memory(
            "memory_stats",
            lambda: self.memory.stats(bot_id=bot_id),
            default={"episodes": 0, "semantic_records": 0},
            bot_id=bot_id,
        )

    def _audit(
        self,
        *,
        level: str,
        event_type: str,
        summary: str,
        bot_id: str | None,
        payload: dict[str, object],
    ) -> None:
        if self.audit_trail is None:
            return
        self.audit_trail.record(
            level=level,
            event_type=event_type,
            summary=summary,
            bot_id=bot_id,
            payload=payload,
        )

    def _safe_persist(
        self,
        operation: str,
        fn: Callable[[], T],
        *,
        default: T | None = None,
        bot_id: str | None,
    ) -> T | None:
        if self.repositories is None:
            return default
        try:
            return fn()
        except Exception:
            self.persistence_degraded = True
            logger.exception(
                "persistence_operation_failed",
                extra={"event": "persistence_operation_failed", "operation": operation, "bot_id": bot_id},
            )
            return default

    def _safe_memory(
        self,
        operation: str,
        fn: Callable[[], T],
        *,
        default: T | None = None,
        bot_id: str | None,
    ) -> T | None:
        try:
            return fn()
        except Exception:
            logger.exception(
                "memory_operation_failed",
                extra={"event": "memory_operation_failed", "operation": operation, "bot_id": bot_id},
            )
            return default

    def _emit_runtime_event(
        self,
        *,
        bot_id: str,
        event_family: EventFamily,
        event_type: str,
        severity: EventSeverity,
        text: str,
        numeric: dict[str, float] | None = None,
        payload: dict[str, object] | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        try:
            meta = ContractMeta(
                contract_version=settings.contract_version,
                source="sidecar-runtime",
                bot_id=bot_id,
            )
            event = NormalizedEvent(
                meta=meta,
                event_family=event_family,
                event_type=event_type,
                severity=severity,
                text=text,
                numeric=dict(numeric or {}),
                payload=dict(payload or {}),
                tags=dict(tags or {}),
            )
            result = self.normalizer_bus.ingest_batch(EventBatchIngestRequest(meta=meta, events=[event]))
            self._evaluate_reflex_events(
                bot_id=bot_id,
                events=self._accepted_events(events=[event], event_ids=result.event_ids),
                source="runtime_event",
            )
        except Exception:
            logger.exception(
                "runtime_event_emit_failed",
                extra={"event": "runtime_event_emit_failed", "bot_id": bot_id, "event_type": event_type},
            )


def create_runtime() -> RuntimeState:
    workspace_root = Path(__file__).resolve().parents[2]
    sqlite_path = _resolve_path(workspace_root, settings.sqlite_path)
    memory_sqlite_path = _resolve_path(workspace_root, settings.memory_openmemory_path)

    repositories: SidecarRepositories | None = None
    telemetry_store: TelemetryStore
    audit_trail: AuditTrail | None = None
    persistence_degraded = False
    memory_provider_error: str | None = None

    try:
        db = SQLiteDB(path=sqlite_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms)
        db.initialize()
        repositories = create_repositories(
            db=db,
            snapshot_history_per_bot=settings.persistence_snapshot_history_per_bot,
            telemetry_max_per_bot=settings.telemetry_max_per_bot,
            telemetry_operational_window_minutes=settings.telemetry_operational_window_minutes,
            audit_history=settings.persistence_audit_history,
        )
        telemetry_store = TelemetryStore(
            max_per_bot=settings.telemetry_max_per_bot,
            _ingestor=DurableTelemetryIngestor(
                repositories.telemetry,
                backlog_max_events=settings.telemetry_backlog_max_events,
            ),
            _repository=repositories.telemetry,
        )
        audit_trail = AuditTrail(repositories.audit)
    except Exception:
        logger.exception("runtime_persistence_init_failed", extra={"event": "runtime_persistence_init_failed"})
        telemetry_store = TelemetryStore(max_per_bot=settings.telemetry_max_per_bot)
        persistence_degraded = True
        repositories = None
        audit_trail = None

    if repositories is not None:
        memory_repo = repositories.memory
    else:
        fallback_db = SQLiteDB(path=sqlite_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms)
        try:
            fallback_db.initialize()
            fallback_repositories = create_repositories(
                db=fallback_db,
                snapshot_history_per_bot=settings.persistence_snapshot_history_per_bot,
                telemetry_max_per_bot=settings.telemetry_max_per_bot,
                telemetry_operational_window_minutes=settings.telemetry_operational_window_minutes,
                audit_history=settings.persistence_audit_history,
            )
            memory_repo = fallback_repositories.memory
        except Exception as exc:
            memory_repo = None
            memory_provider_error = str(exc)
            logger.exception("memory_fallback_repo_init_failed", extra={"event": "memory_fallback_repo_init_failed"})

    if memory_repo is not None:
        fallback_provider = SQLiteMemoryProvider(
            episodic=EpisodicMemoryStore(repository=memory_repo),
            semantic=SemanticMemoryStore(
                repository=memory_repo,
                embedder=LocalSemanticEmbedder(settings.memory_embedding_dimensions),
                candidates=settings.memory_semantic_candidates,
            ),
        )
    else:
        fallback_provider = InMemoryMemoryProvider(dimensions=settings.memory_embedding_dimensions)
        persistence_degraded = True

    provider = fallback_provider
    backend = settings.memory_backend.lower().strip()
    if backend in {"openmemory", "auto"}:
        openmemory_provider = OpenMemoryProvider(
            sqlite_fallback=fallback_provider,
            mode=settings.memory_openmemory_mode,
            path=str(memory_sqlite_path),
        )
        provider = openmemory_provider
        if backend == "openmemory" and not openmemory_provider.enabled:
            persistence_degraded = True
            memory_provider_error = openmemory_provider.init_error

    runtime = RuntimeState(
        started_at=datetime.now(UTC),
        workspace_root=workspace_root,
        bot_registry=BotRegistry(),
        snapshot_cache=SnapshotCache(ttl_seconds=settings.snapshot_cache_ttl_seconds),
        action_queue=ActionQueue(max_per_bot=settings.action_max_queue_per_bot),
        latency_router=LatencyRouter(budget_ms=settings.latency_budget_ms),
        telemetry_store=telemetry_store,
        macro_compiler=MacroCompiler(),
        macro_publisher=MacroPublisher(workspace_root=workspace_root),
        memory=MemoryRetrievalService(provider=provider),
        audit_trail=audit_trail,
        repositories=repositories,
        normalizer_bus=NormalizerBus.create(event_journal=EventJournal(repository=repositories.events if repositories else None)),
        reflex_engine=ReflexRuleEngine(
            workspace_root=workspace_root,
            contract_version=settings.contract_version,
            action_ttl_seconds=settings.action_default_ttl_seconds,
            trigger_history_per_bot=settings.reflex_trigger_history_per_bot,
        ),
        sqlite_path=sqlite_path if repositories is not None else None,
        persistence_degraded=persistence_degraded,
    )
    runtime._audit(
        level="warning" if persistence_degraded else "info",
        event_type="runtime_initialized",
        summary="runtime initialized",
        bot_id=None,
        payload={
            "sqlite_path": str(sqlite_path),
            "memory_sqlite_path": str(memory_sqlite_path),
            "persistence_enabled": repositories is not None,
            "memory_backend": backend,
            "telemetry_backlog_enabled": True,
            "memory_provider_error": memory_provider_error,
        },
    )
    logger.info(
        "runtime_initialized",
        extra={"event": "runtime_initialized", "sqlite_path": str(sqlite_path), "degraded": persistence_degraded},
    )
    return runtime
