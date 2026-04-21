from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Callable, TypeVar
from uuid import uuid4

import httpx

from ai_sidecar.config import settings
from ai_sidecar.contracts.actions import ActionAckRequest, ActionPriorityTier, ActionProposal, ActionStatus
from ai_sidecar.contracts.common import ContractMeta
from ai_sidecar.contracts.crewai import (
    CrewAgentsResponse,
    CrewCoordinateRequest,
    CrewCoordinateResponse,
    CrewStatusResponse,
    CrewStrategizeRequest,
    CrewStrategizeResponse,
    CrewToolExecuteRequest,
    CrewToolExecuteResponse,
)
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
from ai_sidecar.contracts.ml_subconscious import (
    MLDistillMacroRequest,
    MLDistillMacroResponse,
    MLModelsResponse,
    MLObserveRequest,
    MLObserveResponse,
    MLPerformanceResponse,
    MLPredictRequest,
    MLPredictResponse,
    MLPromoteRequest,
    MLPromoteResponse,
    MLTrainRequest,
    MLTrainResponse,
    MLTrainingEpisode,
    ModelFamily,
)
from ai_sidecar.contracts.reflex import ReflexRule
from ai_sidecar.contracts.reflex import ReflexBreakerStatusView, ReflexTriggerRecord
from ai_sidecar.contracts.state import BotRegistrationRequest, BotStateSnapshot
from ai_sidecar.contracts.state_graph import EnrichedWorldState
from ai_sidecar.contracts.telemetry import TelemetryEvent, TelemetryIngestResponse
from ai_sidecar.crewai import CrewManager
from ai_sidecar.domain.macro_compiler import MacroCompiler, MacroPublisher
from ai_sidecar.ingestion.event_journal import EventJournal
from ai_sidecar.ingestion.adapters.actor_state_adapter import actor_delta_to_events
from ai_sidecar.ingestion.adapters.chat_adapter import chat_stream_to_events
from ai_sidecar.ingestion.adapters.config_adapter import config_update_to_events
from ai_sidecar.ingestion.adapters.quest_adapter import quest_transition_to_events
from ai_sidecar.ingestion.normalizer_bus import NormalizerBus
from ai_sidecar.memory.embeddings import LocalSemanticEmbedder, ProviderSemanticEmbedder
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
from ai_sidecar.ml_subconscious import (
    GuardedPromotionPipeline,
    LabelingPipeline,
    MacroDistillationEngine,
    ModelRegistry,
    ObservationCapture,
    ShadowModeEvaluator,
    TrainingHarness,
)
from ai_sidecar.planner.context_assembler import PlannerContextAssembler
from ai_sidecar.planner.intent_synthesizer import IntentSynthesizer
from ai_sidecar.planner.macro_synthesizer import MacroSynthesizer
from ai_sidecar.planner.plan_generator import PlanGenerator
from ai_sidecar.planner.reflection_writer import ReflectionWriter
from ai_sidecar.planner.schemas import (
    PlannerExplainRequest,
    PlannerMacroPromoteRequest,
    PlannerPlanRequest,
    PlannerResponse,
    PlannerStatusResponse,
    ProviderPolicyUpdateRequest,
    ProviderRouteRequest,
    ProviderRouteResponse,
)
from ai_sidecar.planner.self_critic import SelfCritic
from ai_sidecar.planner.service import PlannerService
from ai_sidecar.providers.deepseek_adapter import DeepseekAdapter
from ai_sidecar.providers.model_router import DEFAULT_POLICY_RULES, ModelRouter
from ai_sidecar.providers.ollama_adapter import OllamaAdapter
from ai_sidecar.providers.openai_adapter import OpenAIAdapter
from ai_sidecar.providers.prompt_guard import PromptGuard
from ai_sidecar.persistence.db import SQLiteDB
from ai_sidecar.persistence.repositories import SidecarRepositories, create_repositories
from ai_sidecar.runtime.action_queue import ActionQueue
from ai_sidecar.runtime.bot_registry import BotRegistry
from ai_sidecar.runtime.latency_router import LatencyRouter
from ai_sidecar.runtime.snapshot_cache import SnapshotCache
from ai_sidecar.reflex.circuit_breaker import ReflexCircuitBreaker
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


def _build_provider_policy_rules() -> dict[str, dict[str, object]]:
    return {
        "reflex_explain": {"providers": [], "models": {}},
        "tactical_short_reasoning": {
            "providers": ["ollama", "deepseek"],
            "models": {
                "ollama": settings.provider_ollama_tactical_model,
                "deepseek": settings.provider_deepseek_tactical_model,
            },
        },
        "strategic_planning": {
            "providers": ["ollama", "openai", "deepseek"],
            "models": {
                "ollama": settings.provider_ollama_strategic_model,
                "openai": settings.provider_openai_strategic_model,
                "deepseek": settings.provider_deepseek_strategic_model,
            },
        },
        "long_reflection": {
            "providers": ["deepseek", "openai", "ollama"],
            "models": {
                "deepseek": settings.provider_deepseek_reflection_model,
                "openai": settings.provider_openai_reflection_model,
                "ollama": settings.provider_ollama_reflection_model,
            },
        },
        "embeddings": {
            "providers": ["ollama", "openai", "deepseek"],
            "models": {
                "ollama": settings.provider_ollama_embedding_model,
                "openai": settings.provider_openai_embedding_model,
                "deepseek": settings.provider_deepseek_embedding_model,
            },
        },
    }


def _provider_embedding_model(provider_name: str) -> str:
    if provider_name == "ollama":
        return settings.provider_ollama_embedding_model
    if provider_name == "openai":
        return settings.provider_openai_embedding_model
    if provider_name == "deepseek":
        return settings.provider_deepseek_embedding_model
    return ""


def _provider_embedding_endpoint(provider_name: str) -> tuple[str, dict[str, str], str]:
    if provider_name == "ollama":
        return settings.provider_ollama_base_url.rstrip("/") + "/api/embed", {"Content-Type": "application/json"}, "ollama"
    if provider_name == "openai":
        return (
            settings.provider_openai_base_url.rstrip("/") + "/embeddings",
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.provider_openai_api_key}",
            },
            "openai",
        )
    return (
        settings.provider_deepseek_base_url.rstrip("/") + "/embeddings",
        {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.provider_deepseek_api_key}",
        },
        "openai_like",
    )


def _provider_embed_sync(provider_name: str, *, model: str, texts: list[str], timeout_seconds: float) -> list[list[float]]:
    if not texts:
        return []
    endpoint, headers, kind = _provider_embedding_endpoint(provider_name)
    payload: dict[str, object]
    if kind == "ollama":
        payload = {"model": model, "input": texts, "truncate": True}
    else:
        payload = {"model": model, "input": texts}

    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json() if response.content else {}

    if not isinstance(data, dict):
        return []
    if kind == "ollama":
        rows = data.get("embeddings") if isinstance(data.get("embeddings"), list) else []
        return [[float(value) for value in row] for row in rows if isinstance(row, list)]

    rows = data.get("data") if isinstance(data.get("data"), list) else []
    vectors: list[list[float]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        emb = row.get("embedding")
        if isinstance(emb, list):
            vectors.append([float(value) for value in emb])
    return vectors


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
    sqlite_path: Path | None = None
    model_router: ModelRouter | None = None
    planner_service: PlannerService | None = None
    crew_manager: CrewManager | None = None
    ml_observer: ObservationCapture | None = None
    ml_labeling: LabelingPipeline | None = None
    ml_registry: ModelRegistry | None = None
    ml_training: TrainingHarness | None = None
    ml_shadow: ShadowModeEvaluator | None = None
    ml_promotion: GuardedPromotionPipeline | None = None
    ml_macro_distiller: MacroDistillationEngine | None = None
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

        if self.ml_observer is not None:
            try:
                state_features = self._ml_state_features(bot_id=bot_id)
                for record in records:
                    success = bool(record.emitted and not record.suppressed)
                    rule_confidence = 0.9 if success else (0.7 if not record.suppressed else 0.4)
                    safety_flags: list[str] = []
                    if record.suppressed:
                        safety_flags.append("rule_suppressed")
                    if record.suppression_reason:
                        safety_flags.append(str(record.suppression_reason))
                    episode = MLTrainingEpisode(
                        episode_id=f"reflex-{record.trigger_id}",
                        bot_id=bot_id,
                        state_features=state_features,
                        decision_source="rule",
                        decision_payload={
                            "rule_id": record.rule_id,
                            "event_type": record.event_type,
                            "execution_target": record.execution_target,
                            "rule_confidence": rule_confidence,
                            "outcome": record.outcome,
                        },
                        executed_action={
                            "action_id": record.action_id,
                            "execution_target": record.execution_target,
                            "detail": record.detail,
                        },
                        outcome={
                            "success": success,
                            "reward": 0.0,
                            "latency_ms": float(record.latency_ms),
                            "side_effects": [str(record.suppression_reason)] if record.suppression_reason else [],
                        },
                        safety_flags=safety_flags,
                        macro_version="",
                    )
                    observed, _, _ = self.ml_observer.capture(episode)
                    if self.ml_labeling is not None:
                        self.ml_labeling.label_episode(observed)
            except Exception:
                logger.exception(
                    "ml_reflex_observation_failed",
                    extra={"event": "ml_reflex_observation_failed", "bot_id": bot_id, "source": source},
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

    def _ml_state_features(self, *, bot_id: str, fallback: dict[str, object] | None = None) -> dict[str, object]:
        if fallback:
            return dict(fallback)
        try:
            state = self.enriched_state(bot_id=bot_id)
            return {
                "feature_values": dict(state.features.values),
                "feature_labels": dict(state.features.labels),
                "feature_raw": dict(state.features.raw),
                "risk": {
                    "danger_score": float(state.risk.danger_score),
                    "death_risk_score": float(state.risk.death_risk_score),
                    "pvp_risk_score": float(state.risk.pvp_risk_score),
                    "anomaly_flags": list(state.risk.anomaly_flags),
                },
                "encounter": {
                    "in_encounter": bool(state.encounter.in_encounter),
                    "nearby_hostiles": int(state.encounter.nearby_hostiles),
                    "nearby_allies": int(state.encounter.nearby_allies),
                    "risk_score": float(state.encounter.risk_score),
                },
                "navigation": {
                    "route_status": str(state.navigation.route_status),
                    "stuck_score": float(state.navigation.stuck_score),
                    "map": state.navigation.map,
                },
                "inventory": {
                    "zeny": int(state.inventory.zeny or 0),
                    "item_count": int(state.inventory.item_count),
                    "overweight_ratio": float(state.inventory.overweight_ratio or 0.0),
                },
                "social": {
                    "recent_chat_count": int(state.social.recent_chat_count),
                    "private_messages_5m": int(state.social.private_messages_5m),
                },
            }
        except Exception:
            logger.exception("ml_state_features_failed", extra={"event": "ml_state_features_failed", "bot_id": bot_id})
            return {}

    def _ml_planner_choices(self, response: PlannerResponse) -> dict[ModelFamily, dict[str, object]]:
        plan = response.strategic_plan
        if plan is None:
            return {
                ModelFamily.encounter_classifier: {},
                ModelFamily.loot_ranker: {},
                ModelFamily.route_recovery_classifier: {},
                ModelFamily.npc_dialogue_predictor: {},
                ModelFamily.risk_anomaly_detector: {},
                ModelFamily.memory_retrieval_ranker: {},
            }

        steps = list(plan.steps or [])
        loot_step = next((item for item in steps if (item.kind or "").lower() in {"loot", "collect"}), None)
        route_step = next((item for item in steps if (item.kind or "").lower() in {"travel", "move", "route"}), None)
        npc_step = next((item for item in steps if (item.kind or "").lower() in {"npc", "dialog", "dialogue", "quest"}), None)
        combat_step = next((item for item in steps if (item.kind or "").lower() in {"combat", "fight", "attack"}), None)

        if plan.risk_score >= 0.7:
            encounter_profile = "safe"
        elif combat_step is not None:
            encounter_profile = "aggressive"
        else:
            encounter_profile = "balanced"

        return {
            ModelFamily.encounter_classifier: {"combat_profile": encounter_profile},
            ModelFamily.loot_ranker: {"loot_item": loot_step.target if loot_step is not None else ""},
            ModelFamily.route_recovery_classifier: {
                "stuck_strategy": route_step.fallbacks[0] if route_step is not None and route_step.fallbacks else "repath"
            },
            ModelFamily.npc_dialogue_predictor: {"npc_branch": npc_step.target if npc_step is not None else "default"},
            ModelFamily.risk_anomaly_detector: {"risk_label": "anomaly" if plan.risk_score >= 0.85 else "normal"},
            ModelFamily.memory_retrieval_ranker: {
                "memory_id": str((response.memory_writeback.metadata or {}).get("plan_id") if response.memory_writeback else "")
            },
        }

    def ml_observe(self, payload: MLObserveRequest) -> MLObserveResponse:
        if self.ml_observer is None:
            return MLObserveResponse(
                ok=False,
                message="ml_unavailable",
                trace_id=payload.meta.trace_id,
                episode_id=payload.episode.episode_id,
                bot_id=payload.episode.bot_id,
            )

        episode = payload.episode
        if not episode.state_features:
            episode = episode.model_copy(update={"state_features": self._ml_state_features(bot_id=episode.bot_id)})

        captured, reward, breakdown = self.ml_observer.capture(episode)
        labels_count = 0
        if self.ml_labeling is not None:
            labels_count = len(self.ml_labeling.label_episode(captured))

        self._safe_memory(
            "capture_ml_observation_memory",
            lambda: self.memory.capture_action(
                bot_id=captured.bot_id,
                action_id=f"ml-observe:{captured.episode_id}",
                kind="ml_observe",
                message=f"ml observation captured reward={reward:.3f} labels={labels_count}",
                metadata={"episode_id": captured.episode_id, "decision_source": captured.decision_source.value},
            ),
            bot_id=captured.bot_id,
        )

        return MLObserveResponse(
            ok=True,
            message="observed",
            trace_id=payload.meta.trace_id,
            episode_id=captured.episode_id,
            bot_id=captured.bot_id,
            reward=reward,
            reward_breakdown=breakdown,
            labels_generated=labels_count,
        )

    def ml_train(self, payload: MLTrainRequest) -> MLTrainResponse:
        if self.ml_training is None:
            return MLTrainResponse(
                ok=False,
                message="ml_unavailable",
                trace_id=payload.meta.trace_id,
                model_family=payload.model_family,
            )

        version, trained, metrics, ab = self.ml_training.train(
            family=payload.model_family,
            bot_id=payload.bot_id,
            incremental=payload.incremental,
            max_samples=payload.max_samples,
        )
        ok = bool(version)
        return MLTrainResponse(
            ok=ok,
            message="trained" if ok else "insufficient_training_data",
            trace_id=payload.meta.trace_id,
            model_family=payload.model_family,
            model_version=version,
            trained_samples=trained,
            metrics=metrics,
            ab_test=ab,
        )

    def ml_models(self) -> MLModelsResponse:
        if self.ml_registry is None:
            return MLModelsResponse(ok=False, models=[])
        return self.ml_registry.list_models()

    def ml_predict(self, payload: MLPredictRequest) -> MLPredictResponse:
        if self.ml_training is None:
            return MLPredictResponse(
                ok=False,
                message="ml_unavailable",
                trace_id=payload.meta.trace_id,
                model_family=payload.model_family,
            )

        state_features = dict(payload.state_features)
        if not state_features:
            state_features = self._ml_state_features(bot_id=payload.meta.bot_id)

        model_version, recommendation, confidence = self.ml_training.predict(
            family=payload.model_family,
            state_features=state_features,
            context=payload.context,
        )

        shadow: dict[str, object] = {"mode": "shadow_only", "matched": False, "planned": "", "predicted": ""}
        if self.ml_shadow is not None:
            shadow = self.ml_shadow.compare(
                bot_id=payload.meta.bot_id,
                trace_id=payload.meta.trace_id,
                family=payload.model_family,
                model_version=model_version,
                planner_choice=payload.planner_choice,
                recommendation=recommendation,
                confidence=confidence,
            )

        return MLPredictResponse(
            ok=True,
            message="predicted",
            trace_id=payload.meta.trace_id,
            model_family=payload.model_family,
            model_version=model_version,
            recommendation=recommendation,
            confidence=confidence,
            shadow=shadow,
        )

    def ml_promote(self, payload: MLPromoteRequest) -> MLPromoteResponse:
        if self.ml_promotion is None:
            return MLPromoteResponse(
                ok=False,
                message="ml_unavailable",
                trace_id=payload.meta.trace_id,
                model_family=payload.model_family,
                promotion={},
            )
        if self.ml_registry is not None:
            self.ml_registry.activate_version(family=payload.model_family, version=payload.model_version)
        state = self.ml_promotion.configure(
            family=payload.model_family,
            model_version=payload.model_version,
            canary_percentage=payload.canary_percentage,
            rollback_threshold=payload.rollback_threshold,
            scope=payload.scope,
        )
        return MLPromoteResponse(
            ok=True,
            message="promotion_updated",
            trace_id=payload.meta.trace_id,
            model_family=payload.model_family,
            promotion=state,
        )

    def ml_performance(self) -> MLPerformanceResponse:
        shadow_metrics = self.ml_shadow.metrics() if self.ml_shadow is not None else {}
        promotion_metrics = self.ml_promotion.metrics() if self.ml_promotion is not None else {}
        training_metrics = {
            "training": self.ml_training.metrics() if self.ml_training is not None else {},
            "labels": self.ml_labeling.counters() if self.ml_labeling is not None else {},
            "observations": self.ml_observer.counters() if self.ml_observer is not None else {},
            "distillation": self.ml_macro_distiller.stats() if self.ml_macro_distiller is not None else {},
        }
        return MLPerformanceResponse(
            ok=True,
            shadow_metrics=shadow_metrics,
            promotion_metrics=promotion_metrics,
            training_metrics=training_metrics,
        )

    def ml_distill_macro(self, payload: MLDistillMacroRequest) -> MLDistillMacroResponse:
        if self.ml_macro_distiller is None or self.ml_observer is None:
            return MLDistillMacroResponse(
                ok=False,
                message="ml_unavailable",
                trace_id=payload.meta.trace_id,
                bot_id=payload.bot_id or payload.meta.bot_id,
            )

        target_bot = payload.bot_id or payload.meta.bot_id
        episodes = self.ml_observer.recent(bot_id=target_bot, limit=5000)
        if payload.episode_ids:
            allowed = set(payload.episode_ids)
            episodes = [item for item in episodes if item.episode_id in allowed]

        result = self.ml_macro_distiller.distill(
            meta=payload.meta,
            episodes=episodes,
            min_support=payload.min_support,
            max_steps=payload.max_steps,
            enqueue_reload=payload.enqueue_reload,
            publish_macro=self.publish_macros,
        )
        return MLDistillMacroResponse(
            ok=bool(result.get("ok")),
            message=str(result.get("message") or "distilled"),
            trace_id=payload.meta.trace_id,
            bot_id=str(result.get("bot_id") or target_bot),
            proposal_id=str(result.get("proposal_id") or ""),
            support=int(result.get("support") or 0),
            success_rate=float(result.get("success_rate") or 0.0),
            macro=dict(result.get("macro") or {}),
            automacro=dict(result.get("automacro") or {}),
            publication=dict(result.get("publication") or {}) if result.get("publication") is not None else None,
        )

    async def planner_plan(self, payload: PlannerPlanRequest) -> PlannerResponse:
        if self.planner_service is None:
            return PlannerResponse(ok=False, message="planner_unavailable", trace_id=payload.meta.trace_id)
        result = await self.planner_service.plan(payload)

        if self.ml_observer is not None:
            try:
                safety_flags: list[str] = []
                try:
                    state_for_flags = self.enriched_state(bot_id=payload.meta.bot_id)
                    if state_for_flags.risk.anomaly_flags:
                        safety_flags.extend([str(item) for item in state_for_flags.risk.anomaly_flags])
                except Exception:
                    logger.exception(
                        "ml_safety_state_read_failed",
                        extra={"event": "ml_safety_state_read_failed", "bot_id": payload.meta.bot_id},
                    )

                try:
                    breakers = self.reflex_breakers(bot_id=payload.meta.bot_id)
                    if any((item.state or "").lower() != "closed" for item in breakers):
                        safety_flags.append("reflex_breaker_open")
                except Exception:
                    logger.exception(
                        "ml_safety_breaker_read_failed",
                        extra={"event": "ml_safety_breaker_read_failed", "bot_id": payload.meta.bot_id},
                    )

                episode = MLTrainingEpisode(
                    episode_id=f"planner-{uuid4().hex[:24]}",
                    bot_id=payload.meta.bot_id,
                    state_features=self._ml_state_features(bot_id=payload.meta.bot_id),
                    decision_source="llm",
                    decision_payload={
                        "objective": payload.objective,
                        "horizon": payload.horizon.value,
                        "provider": result.provider,
                        "model": result.model,
                        "route": dict(result.route),
                        "risk_score": float(result.strategic_plan.risk_score) if result.strategic_plan is not None else 0.0,
                    },
                    executed_action=(
                        result.strategic_plan.recommended_actions[0].model_dump(mode="json")
                        if result.strategic_plan is not None and result.strategic_plan.recommended_actions
                        else {}
                    ),
                    outcome={
                        "success": bool(result.ok),
                        "reward": float(result.learning_label.reward) if result.learning_label is not None else 0.0,
                        "latency_ms": float(result.latency_ms),
                        "side_effects": [],
                    },
                    safety_flags=(["self_critic_rejected"] if not result.ok else []) + safety_flags,
                    macro_version=(
                        str((result.route.get("macro_publish") or {}).get("publication", {}).get("version") or "")
                        if isinstance(result.route, dict)
                        else ""
                    ),
                )
                observed, _, _ = self.ml_observer.capture(episode)
                if self.ml_labeling is not None:
                    self.ml_labeling.label_episode(observed)

                if self.ml_training is not None and self.ml_shadow is not None:
                    planner_choices = self._ml_planner_choices(result)
                    shadow_rows: dict[str, object] = {}
                    for family, planner_choice in planner_choices.items():
                        model_version, recommendation, confidence = self.ml_training.predict(
                            family=family,
                            state_features=observed.state_features,
                            context={"objective": payload.objective, "horizon": payload.horizon.value},
                        )
                        shadow = self.ml_shadow.compare(
                            bot_id=payload.meta.bot_id,
                            trace_id=payload.meta.trace_id,
                            family=family,
                            model_version=model_version,
                            planner_choice=planner_choice,
                            recommendation=recommendation,
                            confidence=confidence,
                        )
                        shadow_rows[family.value] = {
                            "model_version": model_version,
                            "recommendation": recommendation,
                            "shadow": shadow,
                        }

                        if self.ml_promotion is not None:
                            promotion = self.ml_promotion.should_execute(
                                family=family,
                                bot_id=payload.meta.bot_id,
                                trace_id=payload.meta.trace_id,
                                confidence=confidence,
                                safety_flags=safety_flags,
                                context={
                                    "map": observed.state_features.get("navigation", {}).get("map")
                                    if isinstance(observed.state_features.get("navigation"), dict)
                                    else None,
                                },
                            )
                            self.ml_promotion.record_outcome(
                                family=family,
                                executed=bool(promotion.get("allowed")),
                                success=bool(result.ok),
                            )

                    route = dict(result.route)
                    route["ml_shadow"] = shadow_rows
                    result = result.model_copy(update={"route": route})
            except Exception:
                logger.exception(
                    "ml_planner_observation_failed",
                    extra={"event": "ml_planner_observation_failed", "bot_id": payload.meta.bot_id, "trace_id": payload.meta.trace_id},
                )

        return result

    async def planner_replan(self, payload: PlannerPlanRequest) -> PlannerResponse:
        replanned = payload.model_copy(update={"force_replan": True})
        return await self.planner_plan(replanned)

    async def planner_promote_macro(self, payload: PlannerMacroPromoteRequest) -> PlannerResponse:
        if self.planner_service is None:
            return PlannerResponse(ok=False, message="planner_unavailable", trace_id=payload.meta.trace_id)
        result = await self.planner_service.promote_macro(payload)
        proposal = result.macro_proposal
        if not result.ok or proposal is None:
            return result

        ok, publication, message = self.publish_macros(
            MacroPublishRequest(
                meta=payload.meta,
                target_bot_id=payload.meta.bot_id,
                macros=proposal.macros,
                event_macros=proposal.event_macros,
                automacros=proposal.automacros,
                enqueue_reload=True,
            )
        )
        route = dict(result.route)
        route["macro_publish"] = {
            "ok": ok,
            "message": message,
            "publication": publication,
        }
        return result.model_copy(
            update={
                "message": "macro_promoted" if ok else "macro_promotion_publish_failed",
                "route": route,
            }
        )

    def planner_explain(self, payload: PlannerExplainRequest) -> dict[str, object]:
        if self.planner_service is None:
            return {
                "ok": False,
                "bot_id": payload.meta.bot_id,
                "trace_id": payload.meta.trace_id,
                "message": "planner_unavailable",
                "rationale": "",
            }
        return self.planner_service.explain(payload)

    def planner_status(self, *, bot_id: str) -> PlannerStatusResponse:
        if self.planner_service is None:
            return PlannerStatusResponse(ok=True, bot_id=bot_id, planner_healthy=False, counters={})
        return self.planner_service.status(bot_id=bot_id)

    async def crewai_strategize(self, payload: CrewStrategizeRequest) -> CrewStrategizeResponse:
        if self.crew_manager is None:
            return CrewStrategizeResponse(
                ok=False,
                message="crewai_unavailable",
                trace_id=payload.meta.trace_id,
                bot_id=payload.meta.bot_id,
                objective=payload.objective,
                errors=["crewai_unavailable"],
            )
        return await self.crew_manager.strategize(payload)

    async def crewai_coordinate(self, payload: CrewCoordinateRequest) -> CrewCoordinateResponse:
        if self.crew_manager is None:
            return CrewCoordinateResponse(
                ok=False,
                message="crewai_unavailable",
                trace_id=payload.meta.trace_id,
                bot_id=payload.meta.bot_id,
                task=payload.task,
                errors=["crewai_unavailable"],
            )
        return await self.crew_manager.coordinate(payload)

    def crewai_agents(self) -> CrewAgentsResponse:
        if self.crew_manager is None:
            return CrewAgentsResponse(ok=False, total_agents=0, agents=[])
        return self.crew_manager.agents()

    def crewai_status(self) -> CrewStatusResponse:
        if self.crew_manager is None:
            return CrewStatusResponse(ok=False, crew_available=False, crewai_enabled=False, active_runs=0, counters={}, agents=[])
        return self.crew_manager.status()

    def crewai_execute_tool(self, payload: CrewToolExecuteRequest) -> CrewToolExecuteResponse:
        if self.crew_manager is None:
            return CrewToolExecuteResponse(
                ok=False,
                message="crewai_unavailable",
                trace_id=payload.meta.trace_id,
                bot_id=payload.meta.bot_id,
                tool_name=payload.tool_name,
                result={},
            )
        result = self.crew_manager.execute_tool(
            bot_id=payload.meta.bot_id,
            tool_name=payload.tool_name,
            arguments=payload.arguments,
        )
        return CrewToolExecuteResponse(
            ok=bool(result.get("ok", True)),
            message=str(result.get("message") or "ok"),
            trace_id=payload.meta.trace_id,
            bot_id=payload.meta.bot_id,
            tool_name=payload.tool_name,
            result=result,
        )

    async def providers_health(self, *, bot_id: str) -> list[dict[str, object]]:
        if self.model_router is None:
            return []
        rows = await self.model_router.health(bot_id=bot_id)
        return [
            {
                "provider": item.provider,
                "available": item.available,
                "latency_ms": item.latency_ms,
                "models": list(item.models),
                "breaker_state": item.breaker_state,
                "message": item.message,
            }
            for item in rows
        ]

    def provider_route(self, payload: ProviderRouteRequest) -> ProviderRouteResponse:
        if self.model_router is None:
            return ProviderRouteResponse(
                ok=False,
                workload=payload.workload,
                selected_provider="none",
                selected_model="",
                fallback_chain=[],
                policy_version="unavailable",
            )
        decision = self.model_router.decide(workload=payload.workload)
        return ProviderRouteResponse(
            ok=True,
            workload=payload.workload,
            selected_provider=decision.selected_provider,
            selected_model=decision.selected_model,
            fallback_chain=decision.fallback_chain,
            policy_version=decision.policy_version,
        )

    def provider_policy(self) -> dict[str, object]:
        if self.model_router is None:
            return {"ok": False, "version": "unavailable", "updated_at": datetime.now(UTC), "rules": {}}
        policy = self.model_router.current_policy()
        return {
            "ok": True,
            "version": policy.version,
            "updated_at": policy.updated_at,
            "rules": policy.rules,
        }

    def update_provider_policy(self, payload: ProviderPolicyUpdateRequest) -> dict[str, object]:
        if self.model_router is None:
            return {"ok": False, "version": "unavailable", "updated_at": datetime.now(UTC), "rules": {}}
        policy = self.model_router.update_policy(rules=payload.rules)
        return {
            "ok": True,
            "version": policy.version,
            "updated_at": policy.updated_at,
            "rules": policy.rules,
        }

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

    base_embedder = LocalSemanticEmbedder(settings.memory_embedding_dimensions)
    semantic_embedder = base_embedder
    embedding_mode = settings.memory_embedding_mode.lower().strip()
    embedding_provider = settings.memory_embedding_provider.lower().strip()
    requested_embedding_model = settings.memory_embedding_model.strip()

    if embedding_mode == "provider":
        provider_model = requested_embedding_model or _provider_embedding_model(embedding_provider)
        if embedding_provider in {"ollama", "openai", "deepseek"} and provider_model:
            semantic_embedder = ProviderSemanticEmbedder(
                dimensions=settings.memory_embedding_dimensions,
                fallback=base_embedder,
                embed_texts=lambda texts, provider_name=embedding_provider, model_name=provider_model: _provider_embed_sync(
                    provider_name,
                    model=model_name,
                    texts=texts,
                    timeout_seconds=min(settings.planner_timeout_seconds, settings.llm_timeout_seconds),
                ),
            )
            logger.info(
                "memory_embedding_provider_enabled",
                extra={
                    "event": "memory_embedding_provider_enabled",
                    "provider": embedding_provider,
                    "model": provider_model,
                },
            )
        else:
            logger.warning(
                "memory_embedding_provider_invalid_config_fallback_local",
                extra={
                    "event": "memory_embedding_provider_invalid_config_fallback_local",
                    "provider": embedding_provider,
                    "model": provider_model,
                },
            )

    if memory_repo is not None:
        fallback_provider = SQLiteMemoryProvider(
            episodic=EpisodicMemoryStore(repository=memory_repo),
            semantic=SemanticMemoryStore(
                repository=memory_repo,
                embedder=semantic_embedder,
                candidates=settings.memory_semantic_candidates,
            ),
        )
    else:
        fallback_provider = InMemoryMemoryProvider(
            dimensions=settings.memory_embedding_dimensions,
            embedder=semantic_embedder,
        )
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

    guard = PromptGuard(max_prompt_chars=settings.llm_prompt_max_chars)
    provider_breaker = ReflexCircuitBreaker()
    telemetry_push = telemetry_store.push

    provider_adapters: dict[str, object] = {}
    if settings.provider_ollama_enabled:
        provider_adapters["ollama"] = OllamaAdapter(
            base_url=settings.provider_ollama_base_url,
            default_model=settings.provider_ollama_default_model,
            embedding_model=settings.provider_ollama_embedding_model,
            guard=guard,
            breaker=provider_breaker,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            telemetry_push=telemetry_push,
        )
    if settings.provider_openai_enabled:
        provider_adapters["openai"] = OpenAIAdapter(
            base_url=settings.provider_openai_base_url,
            api_key=settings.provider_openai_api_key,
            default_model=settings.provider_openai_default_model,
            embedding_model=settings.provider_openai_embedding_model,
            guard=guard,
            breaker=provider_breaker,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            telemetry_push=telemetry_push,
        )
    if settings.provider_deepseek_enabled:
        provider_adapters["deepseek"] = DeepseekAdapter(
            base_url=settings.provider_deepseek_base_url,
            api_key=settings.provider_deepseek_api_key,
            default_model=settings.provider_deepseek_default_model,
            embedding_model=settings.provider_deepseek_embedding_model,
            guard=guard,
            breaker=provider_breaker,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            telemetry_push=telemetry_push,
        )

    policy_rules = _build_provider_policy_rules()
    if settings.provider_policy_json.strip():
        try:
            override_rules = json.loads(settings.provider_policy_json)
            if isinstance(override_rules, dict):
                for workload, config in override_rules.items():
                    if isinstance(config, dict):
                        policy_rules[str(workload)] = {
                            "providers": [str(item) for item in list(config.get("providers") or [])],
                            "models": dict(config.get("models") or {}),
                        }
        except Exception:
            logger.exception("provider_policy_json_parse_failed", extra={"event": "provider_policy_json_parse_failed"})

    model_router = ModelRouter(
        providers=provider_adapters,
        initial_rules=policy_rules,
    )

    ml_registry = ModelRegistry(workspace_root=workspace_root)
    ml_observer = ObservationCapture(workspace_root=workspace_root)
    ml_labeling = LabelingPipeline()
    ml_training = TrainingHarness(labels=ml_labeling, registry=ml_registry)
    ml_shadow = ShadowModeEvaluator()
    ml_promotion = GuardedPromotionPipeline()
    ml_macro_distiller = MacroDistillationEngine()

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
        model_router=model_router,
        ml_observer=ml_observer,
        ml_labeling=ml_labeling,
        ml_registry=ml_registry,
        ml_training=ml_training,
        ml_shadow=ml_shadow,
        ml_promotion=ml_promotion,
        ml_macro_distiller=ml_macro_distiller,
    )

    planner_service = PlannerService(
        runtime=runtime,
        context_assembler=PlannerContextAssembler(runtime=runtime),
        intent_synthesizer=IntentSynthesizer(),
        plan_generator=PlanGenerator(
            model_router=model_router,
            planner_timeout_seconds=settings.planner_timeout_seconds,
            planner_retries=settings.planner_retries,
        ),
        self_critic=SelfCritic(
            tactical_budget_ms=settings.planner_tactical_budget_ms,
            strategic_budget_ms=settings.planner_strategic_budget_ms,
        ),
        macro_synthesizer=MacroSynthesizer(),
        reflection_writer=ReflectionWriter(memory_service=runtime.memory),
    )
    runtime.planner_service = planner_service
    crew_manager = CrewManager(
        runtime=runtime,
        model_router=model_router,
        enabled=settings.crewai_enabled,
        verbose=settings.crewai_verbose,
    )
    runtime.crew_manager = crew_manager

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
            "memory_embedding_mode": embedding_mode,
            "memory_embedding_provider": embedding_provider,
            "memory_embedding_model": requested_embedding_model,
            "providers_enabled": sorted(provider_adapters.keys()),
            "planner_enabled": planner_service is not None,
            "crewai_enabled": settings.crewai_enabled,
            "crewai_available": crew_manager.status().crew_available,
            "ml_subconscious_enabled": True,
            "ml_models_registered": len(runtime.ml_models().models) if runtime.ml_registry is not None else 0,
        },
    )
    logger.info(
        "runtime_initialized",
        extra={"event": "runtime_initialized", "sqlite_path": str(sqlite_path), "degraded": persistence_degraded},
    )
    return runtime
