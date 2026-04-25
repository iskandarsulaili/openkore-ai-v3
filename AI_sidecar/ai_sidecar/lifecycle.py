from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event, RLock, Thread
from typing import Awaitable, Callable, TypeVar
from uuid import uuid4

import httpx
import structlog

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
from ai_sidecar.contracts.control_domain import (
    ControlApplyRequest,
    ControlApplyResponse,
    ControlArtifactsResponse,
    ControlPlanRequest,
    ControlPlanResponse,
    ControlRollbackRequest,
    ControlRollbackResponse,
    ControlValidateRequest,
    ControlValidateResponse,
)
from ai_sidecar.contracts.events import (
    ActorDeltaPushRequest,
    ActorObservation,
    ChatStreamIngestRequest,
    ConfigDoctrineFingerprintRequest,
    EventFamily,
    EventBatchIngestRequest,
    EventSeverity,
    IngestAcceptedResponse,
    NormalizedEvent,
    QuestTransitionRequest,
)
from ai_sidecar.contracts.fleet_v2 import (
    FleetBlackboardLocalResponse,
    FleetClaimRequestV2,
    FleetClaimResponseV2,
    FleetConstraintResponse,
    FleetOutcomeReportRequest,
    FleetOutcomeReportResponse,
    FleetRoleResponse,
    FleetSyncRequest,
    FleetSyncResponse,
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
from ai_sidecar.domain.control_executor import ControlExecutor
from ai_sidecar.domain.control_parser import ControlParser
from ai_sidecar.domain.control_planner import ControlPlanner
from ai_sidecar.domain.control_policy import default_control_policy
from ai_sidecar.domain.control_registry import ControlRegistry
from ai_sidecar.domain.control_service import ControlDomainService
from ai_sidecar.domain.control_state import ControlStateStore
from ai_sidecar.domain.control_storage import ControlStorage
from ai_sidecar.domain.control_validator import ControlValidator
from ai_sidecar.fleet import (
    ConstraintIngestionState,
    FleetConflictResolver,
    FleetSyncClient,
    OutcomeReporter,
    RoleManager,
)
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
from ai_sidecar.observability.audit_logger import ObservabilityAuditLogger
from ai_sidecar.observability.doctrine_manager import DoctrineManager
from ai_sidecar.observability.explainability import ExplainabilityStore
from ai_sidecar.observability.incident_taxonomy import IncidentRegistry
from ai_sidecar.observability.metrics import DurableTelemetryIngestor
from ai_sidecar.observability.metrics_collector import SLOMetricsCollector
from ai_sidecar.observability.security_auditor import SecurityAuditor
from ai_sidecar.observability.tracing import TraceStore
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
from ai_sidecar.planner.validator import PlanValidator
from ai_sidecar.providers.deepseek_adapter import DeepseekAdapter
from ai_sidecar.providers.model_router import DEFAULT_POLICY_RULES, ModelRouter
from ai_sidecar.providers.ollama_adapter import OllamaAdapter
from ai_sidecar.providers.openai_adapter import OpenAIAdapter
from ai_sidecar.providers.prompt_guard import PromptGuard
from ai_sidecar.persistence.db import SQLiteDB
from ai_sidecar.persistence.repositories import SidecarRepositories, bot_id_aliases, canonicalize_bot_id, create_repositories
from ai_sidecar.runtime.action_arbiter import ActionArbiter
from ai_sidecar.runtime.action_queue import ActionQueue
from ai_sidecar.runtime.bot_registry import BotRegistry
from ai_sidecar.runtime.latency_router import LatencyRouter
from ai_sidecar.runtime.snapshot_cache import SnapshotCache
from ai_sidecar.reflex.circuit_breaker import ReflexCircuitBreaker
from ai_sidecar.reflex.rule_engine import ReflexRuleEngine

logger = logging.getLogger(__name__)
fleet_logger = structlog.get_logger("ai_sidecar.fleet_sync")
T = TypeVar("T")

_PROVIDER_HARD_DENY_BY_WORKLOAD: dict[str, set[str]] = {
    "strategic_planning": {"openai"},
    "long_reflection": {"openai"},
    "embeddings": {"openai"},
}

_LOCAL_STARTUP_PREFERRED_GRIND_MAPS: tuple[str, ...] = ("prt_fild08",)


def _default_macro_plugin_name() -> str:
    return "macro"


def _default_event_macro_plugin_name() -> str:
    return "eventMacro"


def _resolve_path(workspace_root: Path, configured_path: str) -> Path:
    candidate = Path(configured_path)
    if candidate.is_absolute():
        return candidate
    return workspace_root / candidate


def _parse_csv_tokens(value: str) -> list[str]:
    seen: set[str] = set()
    rows: list[str] = []
    for raw in value.split(","):
        token = raw.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        rows.append(token)
    return rows


def _sanitize_provider_policy_rules(
    rules: dict[str, dict[str, object]],
    *,
    available_providers: set[str],
) -> dict[str, dict[str, object]]:
    defaults = _build_provider_policy_rules()
    sanitized: dict[str, dict[str, object]] = {}

    for workload, config in rules.items():
        workload_key = str(workload).strip()
        if not workload_key:
            continue
        cfg = config if isinstance(config, dict) else {}
        deny_for_workload = _PROVIDER_HARD_DENY_BY_WORKLOAD.get(workload_key, set())

        raw_providers = [str(item).strip().lower() for item in list(cfg.get("providers") or [])]
        providers: list[str] = []
        for provider_name in raw_providers:
            if not provider_name:
                continue
            if provider_name in deny_for_workload:
                continue
            if provider_name not in available_providers:
                continue
            if provider_name in providers:
                continue
            providers.append(provider_name)

        default_rule = defaults.get(workload_key, {})
        default_providers = [str(item).strip().lower() for item in list(default_rule.get("providers") or [])]
        if not providers:
            for provider_name in default_providers:
                if not provider_name:
                    continue
                if provider_name in deny_for_workload:
                    continue
                if provider_name not in available_providers:
                    continue
                if provider_name in providers:
                    continue
                providers.append(provider_name)

        models: dict[str, str] = {}
        raw_models = cfg.get("models")
        if isinstance(raw_models, dict):
            for provider_name, model_name in raw_models.items():
                key = str(provider_name).strip().lower()
                if key not in providers:
                    continue
                model_text = str(model_name or "").strip()
                if model_text:
                    models[key] = model_text

        default_models = default_rule.get("models") if isinstance(default_rule, dict) else {}
        if isinstance(default_models, dict):
            for provider_name in providers:
                if provider_name in models:
                    continue
                model_name = str(default_models.get(provider_name) or "").strip()
                if model_name:
                    models[provider_name] = model_name

        sanitized[workload_key] = {
            "providers": providers,
            "models": models,
        }

    return sanitized


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
            "providers": ["ollama", "deepseek"],
            "models": {
                "ollama": settings.provider_ollama_strategic_model,
                "deepseek": settings.provider_deepseek_strategic_model,
            },
        },
        "long_reflection": {
            "providers": ["deepseek", "ollama"],
            "models": {
                "deepseek": settings.provider_deepseek_reflection_model,
                "ollama": settings.provider_ollama_reflection_model,
            },
        },
        "embeddings": {
            "providers": ["ollama", "deepseek"],
            "models": {
                "ollama": settings.provider_ollama_embedding_model,
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
    action_arbiter: ActionArbiter | None = None
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
    fleet_sync_client: FleetSyncClient | None = None
    fleet_constraint_state: ConstraintIngestionState | None = None
    fleet_outcome_reporter: OutcomeReporter | None = None
    fleet_conflict_resolver: FleetConflictResolver | None = None
    observability_audit: ObservabilityAuditLogger | None = None
    slo_metrics: SLOMetricsCollector | None = None
    trace_store: TraceStore | None = None
    incident_registry: IncidentRegistry | None = None
    explainability: ExplainabilityStore | None = None
    security_auditor: SecurityAuditor | None = None
    doctrine_manager: DoctrineManager | None = None
    control_domain: ControlDomainService | None = None
    autonomy_policy: dict[str, object] = field(default_factory=dict)
    autonomy_scheduler_degraded: bool = False
    autonomy_scheduler_degraded_reason: str = ""
    planner_stale_threshold_s: float = 60.0
    pdca_loop: object | None = None
    _fleet_role_lock: RLock = field(default_factory=RLock)
    _fleet_roles: dict[str, RoleManager] = field(default_factory=dict)
    _counter_lock: RLock = field(default_factory=RLock)
    counters: dict[str, int] = field(default_factory=dict)
    _action_kind_index: dict[str, str] = field(default_factory=dict)
    _bot_plan_family: dict[str, str] = field(default_factory=dict)
    _actor_presence_by_bot: dict[str, set[str]] = field(default_factory=dict)
    _actor_last_revision_fingerprint: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = field(default_factory=dict)
    _background_tasks: list[asyncio.Task[None]] = field(default_factory=list)
    _background_loop: asyncio.AbstractEventLoop | None = None
    _background_thread: Thread | None = None
    _background_loop_ready: Event = field(default_factory=Event)
    _background_loop_lock: RLock = field(default_factory=RLock)
    persistence_degraded: bool = False

    def incr(self, key: str, n: int = 1, *, bot_id: str | None = None) -> None:
        with self._counter_lock:
            self.counters[key] = self.counters.get(key, 0) + n
        self._safe_persist(
            "increment_counter",
            lambda: self.repositories.telemetry.increment_counter(bot_id=bot_id or "fleet", name=key, delta=n),
            bot_id=bot_id,
        )

    def _trace_id_for_bot(self, bot_id: str, *, prefix: str = "runtime") -> str:
        return f"{prefix}-{bot_id}-{uuid4().hex[:12]}"

    def counter_snapshot(self) -> dict[str, int]:
        with self._counter_lock:
            return dict(self.counters)

    def _background_task_done(self, task: asyncio.Task[None]) -> None:
        try:
            self._background_tasks.remove(task)
        except ValueError:
            pass
        try:
            task.result()
        except asyncio.CancelledError:
            logger.debug(
                "background_task_cancelled",
                extra={"event": "background_task_cancelled", "label": getattr(task, "_ai_sidecar_label", None)},
            )
        except Exception:
            logger.exception(
                "background_task_failed",
                extra={"event": "background_task_failed", "label": getattr(task, "_ai_sidecar_label", None)},
            )

    def _ensure_background_loop(self, *, label: str, bot_id: str | None, tick_id: str | None) -> asyncio.AbstractEventLoop | None:
        with self._background_loop_lock:
            loop = self._background_loop
            if loop is not None and not loop.is_closed():
                return loop

            thread = self._background_thread
            if thread is not None and thread.is_alive():
                logger.info(
                    "background_task_loop_reuse_pending",
                    extra={"event": "background_task_loop_reuse_pending", "label": label, "bot_id": bot_id, "tick_id": tick_id},
                )
            else:
                self._background_loop_ready.clear()

                def _run_loop() -> None:
                    loop_local = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop_local)
                    self._background_loop = loop_local
                    self._background_loop_ready.set()
                    logger.info(
                        "background_task_loop_started",
                        extra={"event": "background_task_loop_started", "label": label, "bot_id": bot_id, "tick_id": tick_id},
                    )
                    loop_local.run_forever()
                    pending = asyncio.all_tasks(loop_local)
                    if pending:
                        for task in pending:
                            task.cancel()
                        loop_local.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    loop_local.close()

                thread = Thread(target=_run_loop, name="ai-sidecar-background", daemon=True)
                self._background_thread = thread
                thread.start()

        ready = self._background_loop_ready.wait(timeout=2.0)
        loop = self._background_loop
        if not ready or loop is None or loop.is_closed():
            logger.warning(
                "background_task_loop_unavailable",
                extra={"event": "background_task_loop_unavailable", "label": label, "bot_id": bot_id, "tick_id": tick_id},
            )
            return None
        return loop

    def _enqueue_background(
        self,
        coro: Awaitable[None],
        *,
        label: str,
        bot_id: str | None = None,
        tick_id: str | None = None,
    ) -> None:
        async def _runner() -> None:
            try:
                await coro
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "background_task_exception",
                    extra={"event": "background_task_exception", "label": label, "bot_id": bot_id, "tick_id": tick_id},
                )

        def _register(task: asyncio.Task[None]) -> None:
            setattr(task, "_ai_sidecar_label", label)
            self._background_tasks.append(task)
            task.add_done_callback(self._background_task_done)

        running_loop: asyncio.AbstractEventLoop | None
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is not None:
            self._background_loop = running_loop
            task = running_loop.create_task(_runner())
            _register(task)
            return

        loop = self._ensure_background_loop(label=label, bot_id=bot_id, tick_id=tick_id)
        if loop is None:
            asyncio.run(_runner())
            return

        def _schedule() -> None:
            task = loop.create_task(_runner())
            _register(task)

        loop.call_soon_threadsafe(_schedule)

    async def shutdown(self) -> None:
        if not self._background_tasks:
            loop = self._background_loop
            if loop is not None and not loop.is_closed():
                try:
                    loop.call_soon_threadsafe(loop.stop)
                    logger.info("background_task_loop_stopping", extra={"event": "background_task_loop_stopping"})
                except Exception:
                    logger.exception("background_task_loop_stop_failed", extra={"event": "background_task_loop_stop_failed"})
            thread = self._background_thread
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)
                if thread.is_alive():
                    logger.warning(
                        "background_task_loop_thread_join_timeout",
                        extra={"event": "background_task_loop_thread_join_timeout"},
                    )
            self._background_thread = None
            self._background_loop = None
            return
        pending = list(self._background_tasks)
        for task in pending:
            task.cancel()
        results = await asyncio.gather(*pending, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                logger.exception(
                    "background_task_shutdown_error",
                    extra={"event": "background_task_shutdown_error", "error": str(result)},
                )
        self._background_tasks.clear()
        loop = self._background_loop
        if loop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(loop.stop)
                logger.info("background_task_loop_stopping", extra={"event": "background_task_loop_stopping"})
            except Exception:
                logger.exception("background_task_loop_stop_failed", extra={"event": "background_task_loop_stop_failed"})
        thread = self._background_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
            if thread.is_alive():
                logger.warning(
                    "background_task_loop_thread_join_timeout",
                    extra={"event": "background_task_loop_thread_join_timeout"},
                )
        self._background_thread = None
        self._background_loop = None

    def register_bot(self, payload: BotRegistrationRequest) -> dict[str, object]:
        requested_bot_id = payload.meta.bot_id
        bot_id = canonicalize_bot_id(
            bot_id=requested_bot_id,
            bot_name=payload.bot_name,
            attributes=payload.attributes,
        )
        stale_aliases = [item for item in bot_id_aliases(canonical_bot_id=bot_id, bot_name=payload.bot_name, attributes=payload.attributes) if item != bot_id]
        if requested_bot_id != bot_id and requested_bot_id not in stale_aliases:
            stale_aliases.append(requested_bot_id)

        if bot_id != requested_bot_id:
            logger.warning(
                "bot_id_canonicalized",
                extra={
                    "event": "bot_id_canonicalized",
                    "requested_bot_id": requested_bot_id,
                    "canonical_bot_id": bot_id,
                    "bot_name": payload.bot_name,
                },
            )

        if stale_aliases:
            removed_in_memory = self.bot_registry.delete_many(stale_aliases)
            if removed_in_memory > 0:
                logger.warning(
                    "bot_registry_in_memory_alias_cleanup",
                    extra={
                        "event": "bot_registry_in_memory_alias_cleanup",
                        "bot_id": bot_id,
                        "removed": removed_in_memory,
                        "aliases": stale_aliases,
                    },
                )

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

        alias_candidates = self._safe_persist(
            "find_alias_bot_ids",
            lambda: self.repositories.bots.find_alias_bot_ids(
                canonical_bot_id=bot_id,
                bot_name=payload.bot_name,
                attributes=payload.attributes,
            )
            if self.repositories
            else [],
            default=[],
            bot_id=bot_id,
        ) or []
        alias_removed = 0
        if alias_candidates:
            alias_removed = int(
                self._safe_persist(
                    "delete_alias_bot_ids",
                    lambda: self.repositories.bots.delete_bot_ids(bot_ids=alias_candidates) if self.repositories else 0,
                    default=0,
                    bot_id=bot_id,
                )
                or 0
            )
            if alias_removed > 0:
                logger.warning(
                    "bot_registry_alias_cleanup",
                    extra={
                        "event": "bot_registry_alias_cleanup",
                        "bot_id": bot_id,
                        "removed": alias_removed,
                        "aliases": alias_candidates,
                    },
                )
                self._audit(
                    level="warning",
                    event_type="bot_registry_alias_cleanup",
                    summary="stale bot aliases removed",
                    bot_id=bot_id,
                    payload={
                        "requested_bot_id": requested_bot_id,
                        "canonical_bot_id": bot_id,
                        "removed": alias_removed,
                        "aliases": alias_candidates,
                    },
                )

        self.incr("bot_registrations", bot_id=bot_id)
        self._audit(
            level="info",
            event_type="bot_registration",
            summary="bot registered",
            bot_id=bot_id,
            payload={
                "requested_bot_id": requested_bot_id,
                "canonical_bot_id": bot_id,
                "bot_name": payload.bot_name,
                "role": payload.role,
                "assignment": payload.assignment,
                "capabilities": payload.capabilities,
                "alias_cleanup_count": alias_removed,
            },
        )

        if persisted is not None:
            return {
                "bot_id": bot_id,
                "seen_at": persisted.last_seen_at,
                "role": persisted.role,
                "assignment": persisted.assignment,
                "liveness_state": persisted.liveness_state,
                "alias_cleanup_count": alias_removed,
            }

        return {
            "bot_id": bot_id,
            "seen_at": in_memory.last_seen_at,
            "role": payload.role,
            "assignment": payload.assignment,
            "liveness_state": "online",
            "alias_cleanup_count": alias_removed,
        }

    def ingest_snapshot(self, snapshot: BotStateSnapshot) -> None:
        bot_id = snapshot.meta.bot_id
        self.bot_registry.upsert(bot_id, snapshot.tick_id)
        self.snapshot_cache.set(snapshot)
        self.incr("snapshots_ingested", bot_id=bot_id)

        if self.slo_metrics is not None:
            plan_family = self._bot_plan_family.get(bot_id, "unknown")
            progression = snapshot.progression
            exp_value = float(progression.base_exp or progression.job_exp or 0.0)
            self.slo_metrics.record_economy(
                bot_id=bot_id,
                plan_family=plan_family,
                zeny=float(snapshot.inventory.zeny or 0.0),
                exp_value=exp_value,
                observed_at=snapshot.observed_at,
            )

        self._enqueue_background(
            self._background_ingest_snapshot(snapshot),
            label="snapshot_ingest",
            bot_id=bot_id,
            tick_id=snapshot.tick_id,
        )

    async def _background_ingest_snapshot(self, snapshot: BotStateSnapshot) -> None:
        snapshot_copy = snapshot.model_copy(deep=True)
        bot_id = snapshot_copy.meta.bot_id
        tick_id = snapshot_copy.tick_id

        try:
            snapshot_payload = snapshot_copy.model_dump(mode="json")
            snapshot_event = NormalizedEvent(
                meta=snapshot_copy.meta,
                observed_at=snapshot_copy.observed_at,
                event_family=EventFamily.snapshot,
                event_type="snapshot.compact",
                source_hook="v1.ingest.snapshot",
                text="snapshot ingested",
                numeric={
                    "hp": float(snapshot_copy.vitals.hp or 0),
                    "hp_max": float(snapshot_copy.vitals.hp_max or 0),
                    "sp": float(snapshot_copy.vitals.sp or 0),
                    "sp_max": float(snapshot_copy.vitals.sp_max or 0),
                    "x": float(snapshot_copy.position.x or 0),
                    "y": float(snapshot_copy.position.y or 0),
                    "zeny": float(snapshot_copy.inventory.zeny or 0),
                    "item_count": float(snapshot_copy.inventory.item_count or 0),
                    "base_level": float(snapshot_copy.progression.base_level or 0),
                    "job_level": float(snapshot_copy.progression.job_level or 0),
                    "base_exp": float(snapshot_copy.progression.base_exp or 0),
                    "base_exp_max": float(snapshot_copy.progression.base_exp_max or 0),
                    "job_exp": float(snapshot_copy.progression.job_exp or 0),
                    "job_exp_max": float(snapshot_copy.progression.job_exp_max or 0),
                    "job_id": float(snapshot_copy.progression.job_id or 0),
                    "skill_points": float(snapshot_copy.progression.skill_points or 0),
                    "stat_points": float(snapshot_copy.progression.stat_points or 0),
                },
                tags={
                    "tick_id": snapshot_copy.tick_id,
                    "map": snapshot_copy.position.map or "",
                    "ai_sequence": snapshot_copy.combat.ai_sequence or "",
                    "job_name": snapshot_copy.progression.job_name or "",
                },
                payload=snapshot_payload,
            )
            result = self.normalizer_bus.ingest_batch(
                EventBatchIngestRequest(meta=snapshot_copy.meta, events=[snapshot_event])
            )
            self._evaluate_reflex_events(
                bot_id=bot_id,
                events=self._accepted_events(events=[snapshot_event], event_ids=result.event_ids),
                source="snapshot",
            )
        except Exception:
            logger.exception(
                "snapshot_normalization_failed",
                extra={"event": "snapshot_normalization_failed", "bot_id": bot_id, "tick_id": tick_id},
            )

        try:
            snapshot_actor_delta = self._snapshot_actor_delta_payload(snapshot=snapshot_copy)
            if snapshot_actor_delta is not None:
                self.ingest_actor_delta(snapshot_actor_delta)
        except Exception:
            logger.exception(
                "snapshot_actor_delta_failed",
                extra={"event": "snapshot_actor_delta_failed", "bot_id": bot_id, "tick_id": tick_id},
            )

        self._safe_persist(
            "persist_snapshot",
            lambda: self.repositories.bots.touch(
                bot_id=bot_id,
                tick_id=tick_id,
                liveness_state="online",
            )
            if self.repositories
            else None,
            bot_id=bot_id,
        )
        self._safe_persist(
            "persist_snapshot_history",
            lambda: self.repositories.snapshots.save_snapshot(snapshot_copy) if self.repositories else None,
            bot_id=bot_id,
        )

        summary = (
            f"tick={snapshot_copy.tick_id} map={snapshot_copy.position.map or 'unknown'} "
            f"pos=({snapshot_copy.position.x},{snapshot_copy.position.y}) "
            f"hp={snapshot_copy.vitals.hp}/{snapshot_copy.vitals.hp_max} "
            f"ai={snapshot_copy.combat.ai_sequence or 'idle'}"
        )
        self._safe_memory(
            "capture_snapshot_memory",
            lambda: self.memory.capture_snapshot(
                bot_id=bot_id,
                tick_id=snapshot_copy.tick_id,
                summary=summary,
                payload={
                    "map": snapshot_copy.position.map,
                    "x": snapshot_copy.position.x,
                    "y": snapshot_copy.position.y,
                    "in_combat": snapshot_copy.combat.is_in_combat,
                },
            ),
            bot_id=bot_id,
        )

    def _snapshot_actor_delta_payload(self, *, snapshot: BotStateSnapshot) -> ActorDeltaPushRequest | None:
        bot_id = snapshot.meta.bot_id
        known_actor_ids = set(self._actor_presence_by_bot.get(bot_id, set()))
        observed_ids: set[str] = set()
        observations: list[ActorObservation] = []
        default_map = snapshot.position.map

        for actor in snapshot.actors:
            actor_id = str(actor.actor_id or "").strip()
            if not actor_id:
                continue
            observed_ids.add(actor_id)
            observations.append(
                ActorObservation(
                    actor_id=actor_id,
                    actor_type=str(actor.actor_type or "unknown"),
                    name=actor.name,
                    map=default_map,
                    x=actor.x,
                    y=actor.y,
                    hp=actor.hp,
                    hp_max=actor.hp_max,
                    level=actor.level,
                    relation=actor.relation,
                    raw={},
                )
            )

        removed_actor_ids = sorted(known_actor_ids - observed_ids)
        if not observations and not removed_actor_ids:
            return None

        return ActorDeltaPushRequest(
            meta=snapshot.meta,
            observed_at=snapshot.observed_at,
            revision=snapshot.tick_id,
            actors=observations,
            removed_actor_ids=removed_actor_ids,
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
        bot_id = payload.meta.bot_id
        observed_actor_ids = {str(item.actor_id).strip() for item in payload.actors if str(item.actor_id).strip()}
        removed_actor_ids = {str(item).strip() for item in payload.removed_actor_ids if str(item).strip()}
        known_actor_ids = set(self._actor_presence_by_bot.get(bot_id, set()))
        hostile_count = sum(
            1
            for item in payload.actors
            if (str(item.relation or "").strip().lower() in {"hostile", "enemy", "monster"})
            or (str(item.actor_type or "").strip().lower() == "monster")
        )

        logger.info(
            "actor_delta_received",
            extra={
                "event": "actor_delta_received",
                "bot_id": bot_id,
                "revision": str(payload.revision or "").strip(),
                "observed_count": len(observed_actor_ids),
                "removed_count": len(removed_actor_ids),
                "known_count_before": len(known_actor_ids),
                "hostile_count": hostile_count,
            },
        )

        revision = str(payload.revision or "").strip()
        revision_fingerprint = (revision, tuple(sorted(observed_actor_ids)), tuple(sorted(removed_actor_ids)))
        if revision and self._actor_last_revision_fingerprint.get(bot_id) == revision_fingerprint:
            logger.info(
                "actor_delta_duplicate_revision_skipped",
                extra={
                    "event": "actor_delta_duplicate_revision_skipped",
                    "bot_id": bot_id,
                    "revision": revision,
                    "observed": len(observed_actor_ids),
                    "removed": len(removed_actor_ids),
                },
            )
            return IngestAcceptedResponse(
                ok=True,
                accepted=0,
                dropped=0,
                bot_id=bot_id,
                event_ids=[],
                message="duplicate_actor_revision_skipped",
            )

        appeared_actor_ids = observed_actor_ids - known_actor_ids
        disappeared_actor_ids = removed_actor_ids & known_actor_ids

        if appeared_actor_ids or disappeared_actor_ids:
            logger.info(
                "actor_lifecycle_delta",
                extra={
                    "event": "actor_lifecycle_delta",
                    "bot_id": bot_id,
                    "revision": revision,
                    "appeared_count": len(appeared_actor_ids),
                    "disappeared_count": len(disappeared_actor_ids),
                    "appeared_sample": sorted(list(appeared_actor_ids))[:10],
                    "disappeared_sample": sorted(list(disappeared_actor_ids))[:10],
                },
            )

        events = actor_delta_to_events(
            payload,
            appeared_actor_ids=appeared_actor_ids,
            disappeared_actor_ids=disappeared_actor_ids,
        )
        if not events:
            return IngestAcceptedResponse(
                ok=True,
                accepted=0,
                dropped=0,
                bot_id=bot_id,
                event_ids=[],
                message="no_actor_changes",
            )

        result = self.normalizer_bus.ingest_batch(EventBatchIngestRequest(meta=payload.meta, events=events))
        accepted_events = self._accepted_events(events=events, event_ids=result.event_ids)

        next_presence = set(known_actor_ids)
        for event in accepted_events:
            actor_id = str(event.payload.get("actor_id") or "").strip()
            if not actor_id:
                continue
            if event.event_type in {"actor.observed", "actor.appeared"}:
                next_presence.add(actor_id)
            elif event.event_type in {"actor.removed", "actor.disappeared"}:
                next_presence.discard(actor_id)
        self._actor_presence_by_bot[bot_id] = next_presence

        if revision and accepted_events:
            self._actor_last_revision_fingerprint[bot_id] = revision_fingerprint

        logger.info(
            "actor_delta_ingested",
            extra={
                "event": "actor_delta_ingested",
                "bot_id": bot_id,
                "revision": revision,
                "observed_count": len(observed_actor_ids),
                "removed_count": len(removed_actor_ids),
                "appeared_count": len(appeared_actor_ids),
                "disappeared_count": len(disappeared_actor_ids),
                "hostile_count": hostile_count,
                "accepted": result.accepted,
                "dropped": result.dropped,
                "known_count_after": len(next_presence),
            },
        )

        if result.accepted <= 0 and (observed_actor_ids or removed_actor_ids):
            logger.warning(
                "actor_delta_effectively_dropped",
                extra={
                    "event": "actor_delta_effectively_dropped",
                    "bot_id": bot_id,
                    "revision": revision,
                    "observed_count": len(observed_actor_ids),
                    "removed_count": len(removed_actor_ids),
                    "hostile_count": hostile_count,
                    "accepted": result.accepted,
                    "dropped": result.dropped,
                    "response_message": result.message,
                },
            )

        self._evaluate_reflex_events(
            bot_id=bot_id,
            events=accepted_events,
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
        if (
            proposal.priority_tier == ActionPriorityTier.strategic
            and self.fleet_constraint_state is not None
            and self.fleet_conflict_resolver is not None
        ):
            try:
                constraints = self.fleet_conflict_resolver.resolve_constraints(
                    constraints=self.fleet_constraint_state.constraints_for_bot(bot_id=bot_id)
                )
                action_metadata = dict(proposal.metadata)
                if proposal.conflict_key:
                    action_metadata.setdefault("conflict_key", proposal.conflict_key)
                proposal = proposal.model_copy(
                    update={
                        "metadata": self.fleet_conflict_resolver.rearbitrate_action_metadata(
                            action_metadata=action_metadata,
                            constraints=constraints,
                        )
                    }
                )
            except Exception:
                logger.exception(
                    "fleet_rearbitration_failed",
                    extra={"event": "fleet_rearbitration_failed", "bot_id": bot_id, "action_id": proposal.action_id},
                )

        if self.action_arbiter is not None:
            admission = self.action_arbiter.admit_sync(proposal, bot_id=bot_id)
            accepted = admission.admitted
            status = admission.status or (ActionStatus.queued if accepted else ActionStatus.dropped)
            action_id = admission.action_id or proposal.action_id
            reason = admission.reason
        else:
            accepted, status, action_id, reason = self.action_queue.enqueue(bot_id, proposal)
        self.incr("actions_queued", bot_id=bot_id)

        if self.slo_metrics is not None:
            self.slo_metrics.record_queue_decision(
                tier=proposal.priority_tier.value,
                status=status.value,
                reason=reason,
            )
            self.slo_metrics.set_queue_backlog(tier=proposal.priority_tier.value, depth=self.action_queue.count(bot_id))

        self._action_kind_index[action_id] = proposal.kind

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

        if self.slo_metrics is not None:
            action_kind = self._action_kind_index.get(ack.action_id, "unknown")
            self.slo_metrics.record_ack(source=ack.meta.source, action_kind=action_kind, success=ack.success)
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
            get_planner_context=lambda *, bot_id=bot_id: self.reflex_runtime_context(bot_id=bot_id),
        )
        elapsed_ms = self.latency_router.end("reflex.evaluate", started)

        if self.slo_metrics is not None:
            self.slo_metrics.observe_latency(domain="reflex", elapsed_ms=elapsed_ms)

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

            if self.explainability is not None:
                self.explainability.add(
                    kind="reflex",
                    bot_id=bot_id,
                    trace_id=self._trace_id_for_bot(bot_id, prefix="reflex"),
                    summary=f"rule={record.rule_id} outcome={record.outcome}",
                    details={
                        "event_type": record.event_type,
                        "suppressed": record.suppressed,
                        "suppression_reason": record.suppression_reason,
                        "execution_target": record.execution_target,
                        "action_id": record.action_id,
                        "latency_ms": record.latency_ms,
                        "detail": record.detail,
                    },
                )

            if self.slo_metrics is not None and (record.suppression_reason.startswith("open") or "breaker" in record.suppression_reason):
                self.slo_metrics.record_breaker(family="reflex", key=record.rule_id, state="open")

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
        if not events:
            return TelemetryIngestResponse(ok=True, accepted=0, dropped=0, queued_for_retry=0)

        if self.telemetry_store._ingestor is None:
            return self._process_telemetry_ingest(bot_id=bot_id, events=events)

        events_copy = list(events)
        self._enqueue_background(
            self._background_ingest_telemetry(bot_id=bot_id, events=events_copy),
            label="telemetry_ingest",
            bot_id=bot_id,
        )
        return TelemetryIngestResponse(
            ok=True,
            accepted=len(events_copy),
            dropped=0,
            queued_for_retry=self.telemetry_store.backlog_size(),
        )

    def _process_telemetry_ingest(self, *, bot_id: str, events: list[TelemetryEvent]) -> TelemetryIngestResponse:
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

        for item in events:
            if self.slo_metrics is not None:
                if "death" in item.event.lower() or "death" in item.category.lower():
                    self.slo_metrics.record_death(
                        map_name=str(item.tags.get("map") or item.tags.get("map_name") or "unknown"),
                        doctrine_version=str(item.tags.get("doctrine_version") or "unknown"),
                    )
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

    async def _background_ingest_telemetry(self, *, bot_id: str, events: list[TelemetryEvent]) -> None:
        try:
            self._process_telemetry_ingest(bot_id=bot_id, events=events)
        except Exception:
            logger.exception(
                "telemetry_ingest_failed",
                extra={"event": "telemetry_ingest_failed", "bot_id": bot_id, "batch_size": len(events)},
            )

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

        if self.security_auditor is not None:
            macro_lines = [line for routine in request.macros for line in routine.lines]
            macro_lines.extend(line for routine in request.event_macros for line in routine.lines)
            automacro_conditions = [line for item in request.automacros for line in item.conditions]
            allowed, reason = self.security_auditor.validate_macro_policy(
                macro_lines=macro_lines,
                automacro_conditions=automacro_conditions,
            )
            if not allowed:
                self.security_auditor.record(
                    kind="macro_policy_violation",
                    source="publish_macros",
                    bot_id=target_bot_id,
                    detail=reason,
                    severity="error",
                )
                self._audit(
                    level="error",
                    event_type="macro_publish_security_blocked",
                    summary="macro publication blocked by security policy",
                    bot_id=target_bot_id,
                    payload={"reason": reason},
                )
                return False, None, reason

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
        if self.slo_metrics is not None:
            self.slo_metrics.record_macro_publish(version=str(compiled.version), success=True)
        return True, publication_info, "macro artifacts published"

    def control_plan(self, payload: ControlPlanRequest) -> ControlPlanResponse:
        return self.control_domain.plan(payload)

    def control_apply(self, payload: ControlApplyRequest) -> ControlApplyResponse:
        return self.control_domain.apply(payload)

    def control_validate(self, payload: ControlValidateRequest) -> ControlValidateResponse:
        return self.control_domain.validate(payload)

    def control_rollback(self, payload: ControlRollbackRequest) -> ControlRollbackResponse:
        return self.control_domain.rollback(payload)

    def control_artifacts(self, *, bot_id: str) -> ControlArtifactsResponse:
        return self.control_domain.artifacts(bot_id=bot_id)

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

    def _fleet_role_manager(self, *, bot_id: str) -> RoleManager:
        with self._fleet_role_lock:
            manager = self._fleet_roles.get(bot_id)
            if manager is None:
                manager = RoleManager(bot_id=bot_id)
                self._fleet_roles[bot_id] = manager
            return manager

    def _fleet_status(self) -> dict[str, object]:
        if self.fleet_constraint_state is None:
            return {
                "mode": "local",
                "central_available": False,
                "stale": True,
                "last_sync_at": None,
                "doctrine_version": "local",
                "last_error": "fleet_constraint_state_unavailable",
            }
        return self.fleet_constraint_state.status()

    def _fleet_constraints_for_bot(self, *, bot_id: str) -> dict[str, object]:
        def _parse_maps(rows: object) -> list[str]:
            if not isinstance(rows, list):
                return []
            out: list[str] = []
            for item in rows:
                token = str(item or "").strip()
                if token and token not in out:
                    out.append(token)
            return out

        def _apply_local_startup_preference(payload: dict[str, object]) -> dict[str, object]:
            constraints = dict(payload)
            assignment = str(constraints.get("assignment") or "").strip()
            preferred = _parse_maps(constraints.get("preferred_grind_maps"))
            preferred_alias = _parse_maps(constraints.get("preferred_maps"))
            for map_name in preferred_alias:
                if map_name not in preferred:
                    preferred.append(map_name)
            if not assignment:
                for map_name in _LOCAL_STARTUP_PREFERRED_GRIND_MAPS:
                    if map_name not in preferred:
                        preferred.append(map_name)
            if preferred:
                constraints["preferred_grind_maps"] = preferred
            return constraints

        if self.fleet_constraint_state is None:
            return _apply_local_startup_preference({
                "avoid": [],
                "required": [],
                "sources": ["local_default"],
                "policy": {
                    "step_1_detect_conflict": True,
                    "step_2_compare_priority_and_lease": True,
                    "step_3_apply_doctrine": True,
                    "step_4_emit_constraints": True,
                    "step_5_rearbitrate_pending_strategic": True,
                },
            })
        constraints = _apply_local_startup_preference(self.fleet_constraint_state.constraints_for_bot(bot_id=bot_id))
        if self.fleet_conflict_resolver is None:
            return constraints
        return self.fleet_conflict_resolver.resolve_constraints(constraints=constraints)

    def _fleet_refresh_role_from_blackboard(self, *, bot_id: str, blackboard: dict[str, object]) -> None:
        rows = blackboard.get("role_leases") if isinstance(blackboard.get("role_leases"), list) else []
        latest: dict[str, object] | None = None
        latest_expires: datetime | None = None
        for item in rows:
            if not isinstance(item, dict):
                continue
            if str(item.get("bot_id") or "") != bot_id:
                continue
            row_expires_at = item.get("expires_at")
            row_expires: datetime | None = None
            if isinstance(row_expires_at, datetime):
                row_expires = row_expires_at if row_expires_at.tzinfo is not None else row_expires_at.replace(tzinfo=UTC)
            elif isinstance(row_expires_at, str) and row_expires_at:
                try:
                    row_expires = datetime.fromisoformat(row_expires_at.replace("Z", "+00:00")).astimezone(UTC)
                except Exception:
                    row_expires = None

            if latest is None:
                latest = item
                latest_expires = row_expires
                continue

            if row_expires is not None and (latest_expires is None or row_expires > latest_expires):
                latest = item
                latest_expires = row_expires
        if latest is None:
            return

        expires_at = latest.get("expires_at")
        expires: datetime | None = None
        if isinstance(expires_at, datetime):
            expires = expires_at if expires_at.tzinfo is not None else expires_at.replace(tzinfo=UTC)
        elif isinstance(expires_at, str) and expires_at:
            try:
                expires = datetime.fromisoformat(expires_at.replace("Z", "+00:00")).astimezone(UTC)
            except Exception:
                expires = None

        ttl_seconds = settings.action_default_ttl_seconds
        if expires is not None:
            ttl_seconds = max(5, int((expires - datetime.now(UTC)).total_seconds()))

        self._fleet_role_manager(bot_id=bot_id).update(
            role=(None if latest.get("role") is None else str(latest.get("role"))),
            confidence=float(latest.get("confidence") or 0.0),
            ttl_seconds=ttl_seconds,
            source=str(latest.get("lease_owner") or "central"),
        )

    def fleet_sync(self, payload: FleetSyncRequest) -> FleetSyncResponse:
        bot_id = payload.meta.bot_id
        client = self.fleet_sync_client
        state = self.fleet_constraint_state
        if client is None or state is None:
            constraints = self._fleet_constraints_for_bot(bot_id=bot_id)
            return FleetSyncResponse(
                ok=True,
                mode="local",
                central_available=False,
                doctrine_version="local",
                constraints=constraints,
                blackboard={},
                message="fleet_sync_unavailable",
            )

        ok, blackboard, reason = client.ping_blackboard()
        if ok:
            state.update_from_blackboard(blackboard=blackboard)
            self._fleet_refresh_role_from_blackboard(bot_id=bot_id, blackboard=blackboard)
            drained = self.fleet_outcome_reporter.flush_backlog() if self.fleet_outcome_reporter is not None else 0
            status = self._fleet_status()
            constraints = self._fleet_constraints_for_bot(bot_id=bot_id)
            logger.info(
                "fleet_sync_ok",
                extra={
                    "event": "fleet_sync_ok",
                    "bot_id": bot_id,
                    "mode": status.get("mode"),
                    "doctrine_version": status.get("doctrine_version"),
                    "backlog_flushed": drained,
                },
            )
            return FleetSyncResponse(
                ok=True,
                mode=str(status.get("mode") or "central"),
                central_available=bool(status.get("central_available")),
                doctrine_version=str(status.get("doctrine_version") or "unknown"),
                constraints=constraints,
                blackboard=dict(blackboard) if payload.include_blackboard else {},
                message="synced" if drained <= 0 else f"synced_backlog_flushed:{drained}",
            )

        state.mark_unavailable(reason=reason)
        status = self._fleet_status()
        constraints = self._fleet_constraints_for_bot(bot_id=bot_id)
        logger.warning(
            "fleet_sync_fallback_local",
            extra={"event": "fleet_sync_fallback_local", "bot_id": bot_id, "reason": reason},
        )
        return FleetSyncResponse(
            ok=True,
            mode=str(status.get("mode") or "local"),
            central_available=False,
            doctrine_version=str(status.get("doctrine_version") or "local"),
            constraints=constraints,
            blackboard=state.blackboard() if payload.include_blackboard else {},
            message=f"central_unavailable:{reason}",
        )

    def fleet_constraints(self, *, bot_id: str) -> FleetConstraintResponse:
        status = self._fleet_status()
        constraints = self._fleet_constraints_for_bot(bot_id=bot_id)
        return FleetConstraintResponse(
            ok=True,
            bot_id=bot_id,
            mode=str(status.get("mode") or "local"),
            doctrine_version=str(status.get("doctrine_version") or "local"),
            constraints=constraints,
        )

    def fleet_report_outcome(self, payload: FleetOutcomeReportRequest) -> FleetOutcomeReportResponse:
        reporter = self.fleet_outcome_reporter
        status = self._fleet_status()
        if reporter is None:
            return FleetOutcomeReportResponse(
                ok=True,
                accepted=True,
                central_available=False,
                queued_for_retry=True,
                mode=str(status.get("mode") or "local"),
                result={},
            )

        ok, queued_for_retry, result = reporter.report(
            bot_id=payload.meta.bot_id,
            event_type=payload.event_type,
            priority_class=payload.priority_class,
            lease_owner=payload.lease_owner,
            conflict_key=payload.conflict_key,
            payload=dict(payload.payload),
        )
        if not ok and self.fleet_constraint_state is not None:
            self.fleet_constraint_state.mark_unavailable(reason="outcome_submit_failed")

        current_status = self._fleet_status()
        mode = "central" if ok else str(current_status.get("mode") or "local")
        if ok:
            self.incr("fleet_outcome_reported", bot_id=payload.meta.bot_id)
        if queued_for_retry:
            self.incr("fleet_outcome_queued_retry", bot_id=payload.meta.bot_id)

        self._emit_runtime_event(
            bot_id=payload.meta.bot_id,
            event_family=EventFamily.system,
            event_type="fleet.outcome_report",
            severity=EventSeverity.info if ok else EventSeverity.warning,
            text=f"fleet outcome reported event_type={payload.event_type} ok={ok}",
            numeric={"queued_for_retry": 1.0 if queued_for_retry else 0.0},
            payload={
                "event_type": payload.event_type,
                "priority_class": payload.priority_class,
                "lease_owner": payload.lease_owner,
                "conflict_key": payload.conflict_key,
                "central_available": ok,
            },
        )

        return FleetOutcomeReportResponse(
            ok=True,
            accepted=True,
            central_available=ok,
            queued_for_retry=queued_for_retry,
            mode=mode,
            result=result,
        )

    def fleet_role(self, *, bot_id: str) -> FleetRoleResponse:
        if self.fleet_constraint_state is not None:
            self._fleet_refresh_role_from_blackboard(bot_id=bot_id, blackboard=self.fleet_constraint_state.blackboard())
        role = self._fleet_role_manager(bot_id=bot_id).current()
        status = self._fleet_status()
        return FleetRoleResponse(
            ok=True,
            bot_id=bot_id,
            role=(None if role.get("role") is None else str(role.get("role"))),
            confidence=float(role.get("confidence") or 0.0),
            expires_at=(role.get("expires_at") if isinstance(role.get("expires_at"), datetime) else None),
            source=str(role.get("source") or "local"),
            mode=str(status.get("mode") or "local"),
        )

    def fleet_claim(self, payload: FleetClaimRequestV2) -> FleetClaimResponseV2:
        bot_id = payload.meta.bot_id
        client = self.fleet_sync_client
        status = self._fleet_status()
        if client is None:
            return FleetClaimResponseV2(
                ok=True,
                accepted=True,
                central_available=False,
                mode=str(status.get("mode") or "local"),
                reason="accepted_local_fallback",
                claim={
                    "bot_id": bot_id,
                    "claim_type": payload.claim_type,
                    "map_name": payload.map_name,
                    "channel": payload.channel,
                    "resource_type": payload.resource_type,
                    "resource_id": payload.resource_id,
                    "quantity": payload.quantity,
                    "ttl_seconds": payload.ttl_seconds,
                    "priority": payload.priority,
                    "fallback": True,
                },
                conflicts=[],
            )

        ok, result, reason = client.claim(
            bot_id=bot_id,
            claim_type=payload.claim_type,
            map_name=payload.map_name,
            channel=payload.channel,
            objective_id=payload.objective_id,
            resource_type=payload.resource_type,
            resource_id=payload.resource_id,
            quantity=payload.quantity,
            ttl_seconds=payload.ttl_seconds,
            priority=payload.priority,
            metadata=dict(payload.metadata),
        )
        if ok:
            if self.fleet_constraint_state is not None:
                bb_ok, blackboard, bb_reason = client.ping_blackboard()
                if bb_ok:
                    self.fleet_constraint_state.update_from_blackboard(blackboard=blackboard)
                    self._fleet_refresh_role_from_blackboard(bot_id=bot_id, blackboard=blackboard)
                else:
                    self.fleet_constraint_state.mark_unavailable(reason=bb_reason)

            current = self._fleet_status()
            claim_domain = str(result.get("claim_domain") or ("zone" if payload.claim_type in {"territory", "map", "route"} else "resource"))
            claim_view = {
                "claim_id": int(result.get("claim_id") or 0),
                "claim_domain": claim_domain,
                "bot_id": bot_id,
                "claim_type": payload.claim_type,
                "map_name": payload.map_name,
                "channel": payload.channel,
                "resource_type": payload.resource_type,
                "resource_id": payload.resource_id,
                "quantity": payload.quantity,
                "ttl_seconds": payload.ttl_seconds,
                "priority": payload.priority,
            }
            conflicts = [item for item in list(result.get("conflicts") or []) if isinstance(item, dict)]
            return FleetClaimResponseV2(
                ok=True,
                accepted=bool(result.get("accepted", True)),
                central_available=True,
                mode=str(current.get("mode") or "central"),
                reason=str(result.get("reason") or "accepted"),
                claim=claim_view,
                conflicts=conflicts,
            )

        if self.fleet_constraint_state is not None:
            self.fleet_constraint_state.mark_unavailable(reason=reason)
        current = self._fleet_status()
        constraints = self._fleet_constraints_for_bot(bot_id=bot_id)
        local_conflicts: list[dict[str, object]] = []
        conflict_key = str(payload.metadata.get("conflict_key") or "")
        if conflict_key:
            for item in constraints.get("avoid") if isinstance(constraints.get("avoid"), list) else []:
                if not isinstance(item, dict):
                    continue
                if str(item.get("conflict_key") or "") == conflict_key:
                    local_conflicts.append(
                        {
                            "type": str(item.get("type") or "local_constraint"),
                            "key": conflict_key,
                            "owner_bot_id": bot_id,
                            "contender_bot_id": bot_id,
                        }
                    )

        accepted_local = len(local_conflicts) == 0
        return FleetClaimResponseV2(
            ok=True,
            accepted=accepted_local,
            central_available=False,
            mode=str(current.get("mode") or "local"),
            reason=("accepted_local_fallback" if accepted_local else "conflict_detected_local"),
            claim={
                "claim_id": 0,
                "claim_domain": "local",
                "bot_id": bot_id,
                "claim_type": payload.claim_type,
                "map_name": payload.map_name,
                "channel": payload.channel,
                "resource_type": payload.resource_type,
                "resource_id": payload.resource_id,
                "quantity": payload.quantity,
                "ttl_seconds": payload.ttl_seconds,
                "priority": payload.priority,
                "fallback": True,
            },
            conflicts=local_conflicts,
        )

    def fleet_blackboard(self, *, bot_id: str) -> FleetBlackboardLocalResponse:
        client = self.fleet_sync_client
        if client is not None and self.fleet_constraint_state is not None:
            ok, blackboard, reason = client.ping_blackboard()
            if ok:
                self.fleet_constraint_state.update_from_blackboard(blackboard=blackboard)
                self._fleet_refresh_role_from_blackboard(bot_id=bot_id, blackboard=blackboard)
                if self.fleet_outcome_reporter is not None:
                    self.fleet_outcome_reporter.flush_backlog()
            else:
                self.fleet_constraint_state.mark_unavailable(reason=reason)

        status = self._fleet_status()
        constraints = self._fleet_constraints_for_bot(bot_id=bot_id)
        blackboard_payload = self.fleet_constraint_state.blackboard() if self.fleet_constraint_state is not None else {}
        local_summary = {
            "queue_depth": self.action_queue.count(bot_id),
            "outcome_backlog": self.fleet_outcome_reporter.backlog_size() if self.fleet_outcome_reporter is not None else 0,
            "persistence_degraded": self.persistence_degraded,
            "last_sync_at": status.get("last_sync_at"),
            "last_error": status.get("last_error"),
        }
        return FleetBlackboardLocalResponse(
            ok=True,
            bot_id=bot_id,
            mode=str(status.get("mode") or "local"),
            constraints=constraints,
            blackboard=blackboard_payload,
            local_summary=local_summary,
        )

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
                "progression": {
                    "job_id": int(state.operational.job_id or 0),
                    "job_name": str(state.operational.job_name or ""),
                    "base_level": int(state.operational.base_level or 0),
                    "job_level": int(state.operational.job_level or 0),
                    "base_exp": int(state.operational.base_exp or 0),
                    "base_exp_max": int(state.operational.base_exp_max or 0),
                    "job_exp": int(state.operational.job_exp or 0),
                    "job_exp_max": int(state.operational.job_exp_max or 0),
                    "skill_points": int(state.operational.skill_points or 0),
                    "stat_points": int(state.operational.stat_points or 0),
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

        if self.slo_metrics is not None:
            self.slo_metrics.record_shadow(
                family=payload.model_family.value,
                matched=bool(shadow.get("matched", False)),
                confidence=float(confidence),
            )
        if self.explainability is not None:
            self.explainability.add(
                kind="ml_shadow",
                bot_id=payload.meta.bot_id,
                trace_id=payload.meta.trace_id,
                summary=f"family={payload.model_family.value} matched={bool(shadow.get('matched', False))}",
                details={
                    "model_version": model_version,
                    "confidence": float(confidence),
                    "planned": shadow.get("planned"),
                    "predicted": shadow.get("predicted"),
                },
            )
        if self.trace_store is not None:
            self.trace_store.add_event(
                trace_id=payload.meta.trace_id,
                name="ml.predict",
                attributes={
                    "bot_id": payload.meta.bot_id,
                    "family": payload.model_family.value,
                    "model_version": model_version,
                    "confidence": float(confidence),
                    "matched": bool(shadow.get("matched", False)),
                },
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

        self._bot_plan_family[payload.meta.bot_id] = payload.horizon.value
        if self.slo_metrics is not None:
            self.slo_metrics.observe_latency(domain="planner", elapsed_ms=float(result.latency_ms))
            ml_shadow_rows = result.route.get("ml_shadow") if isinstance(result.route, dict) else None
            if isinstance(ml_shadow_rows, dict):
                for family, item in ml_shadow_rows.items():
                    if not isinstance(item, dict):
                        continue
                    shadow = item.get("shadow")
                    if not isinstance(shadow, dict):
                        continue
                    self.slo_metrics.record_shadow(
                        family=str(family),
                        matched=bool(shadow.get("matched", False)),
                        confidence=float(shadow.get("confidence") or 0.0),
                    )
        if self.explainability is not None:
            self.explainability.add(
                kind="planner",
                bot_id=payload.meta.bot_id,
                trace_id=payload.meta.trace_id,
                summary=f"provider={result.provider} model={result.model} ok={result.ok}",
                details={
                    "objective": payload.objective,
                    "horizon": payload.horizon.value,
                    "provider": result.provider,
                    "model": result.model,
                    "ok": result.ok,
                    "latency_ms": float(result.latency_ms),
                },
            )
        if self.trace_store is not None:
            self.trace_store.add_event(
                trace_id=payload.meta.trace_id,
                name="planner.plan",
                attributes={
                    "bot_id": payload.meta.bot_id,
                    "horizon": payload.horizon.value,
                    "provider": result.provider,
                    "model": result.model,
                    "ok": result.ok,
                    "latency_ms": float(result.latency_ms),
                },
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

    def reflex_runtime_context(self, *, bot_id: str) -> dict[str, object]:
        planner = self.planner_status(bot_id=bot_id)
        return {
            "active": bool(planner.current_objective),
            "planner_healthy": bool(planner.planner_healthy),
            "current_objective": planner.current_objective,
            "current_horizon": self._bot_plan_family.get(bot_id, ""),
            "last_plan_id": planner.last_plan_id,
            "queue_depth": self.action_queue.count(bot_id),
        }

    def _readiness_bot_id(self) -> str:
        def _as_utc(value: object) -> datetime | None:
            if not isinstance(value, datetime):
                return None
            if value.tzinfo is None:
                return value.replace(tzinfo=UTC)
            return value.astimezone(UTC)

        def _as_int(value: object) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        try:
            bots = self.list_bots()
            candidates: list[
                tuple[
                    str,
                    bool,
                    bool,
                    datetime,
                    int,
                    int,
                    bool,
                    datetime,
                ]
            ] = []
            for row in bots:
                if not isinstance(row, dict):
                    continue
                bot_id = str(row.get("bot_id") or "").strip()
                if not bot_id:
                    continue

                last_seen_at = _as_utc(row.get("last_seen_at"))
                latest_snapshot_at = _as_utc(row.get("latest_snapshot_at"))
                pending_actions = max(_as_int(row.get("pending_actions")), 0)
                telemetry_events = max(_as_int(row.get("telemetry_events")), 0)

                planner = self.planner_status(bot_id=bot_id)
                planner_updated_at = _as_utc(getattr(planner, "updated_at", None))
                planner_has_state = bool(planner_updated_at or planner.current_objective or planner.last_plan_id)

                activity_points = [item for item in (planner_updated_at, latest_snapshot_at, last_seen_at) if item is not None]
                activity_at = max(activity_points) if activity_points else datetime.min.replace(tzinfo=UTC)

                has_activity = bool(
                    planner_has_state
                    or latest_snapshot_at is not None
                    or pending_actions > 0
                    or telemetry_events > 0
                    or last_seen_at is not None
                )
                is_online = str(row.get("liveness_state") or "").strip().lower() in {"online", "active"}

                candidates.append(
                    (
                        bot_id,
                        planner_has_state,
                        has_activity,
                        activity_at,
                        pending_actions,
                        telemetry_events,
                        is_online,
                        last_seen_at or datetime.min.replace(tzinfo=UTC),
                    )
                )

            if candidates:
                candidates.sort(
                    key=lambda item: (
                        int(item[1]),  # prefer bots with real planner state
                        int(item[2]),  # then bots showing runtime activity
                        item[3],  # freshest planner/snapshot/seen signal
                        int(item[4] > 0),
                        item[4],
                        item[5],
                        int(item[6]),
                        item[7],
                        item[0],
                    ),
                    reverse=True,
                )
                return candidates[0][0]

            if bots:
                first = bots[0]
                if isinstance(first, dict):
                    fallback_bot_id = str(first.get("bot_id") or "").strip()
                    if fallback_bot_id:
                        return fallback_bot_id
        except Exception:
            logger.exception("readiness_bot_resolution_failed", extra={"event": "readiness_bot_resolution_failed"})
        return "openkoreai"

    def readiness_indicators(self) -> dict[str, object]:
        now = datetime.now(UTC)
        bot_id = self._readiness_bot_id()
        planner = self.planner_status(bot_id=bot_id)

        planner_updated_at = getattr(planner, "updated_at", None)
        if isinstance(planner_updated_at, datetime) and planner_updated_at.tzinfo is None:
            planner_updated_at = planner_updated_at.replace(tzinfo=UTC)
        planner_stale_seconds: float | None = None
        if isinstance(planner_updated_at, datetime):
            planner_stale_seconds = max(0.0, (now - planner_updated_at.astimezone(UTC)).total_seconds())

        planner_stale_threshold_s = max(float(self.planner_stale_threshold_s), 1.0)
        planner_stale = (
            (not bool(planner.planner_healthy))
            or planner_stale_seconds is None
            or planner_stale_seconds > planner_stale_threshold_s
        )

        fleet_status = self._fleet_status()
        fleet_last_sync_at = fleet_status.get("last_sync_at")
        if isinstance(fleet_last_sync_at, datetime) and fleet_last_sync_at.tzinfo is None:
            fleet_last_sync_at = fleet_last_sync_at.replace(tzinfo=UTC)

        objective_scheduler_degraded = bool(self.autonomy_scheduler_degraded)
        objective_scheduler_reason = str(self.autonomy_scheduler_degraded_reason or "")
        pdca_running = False
        pdca_breaker_tripped: bool | None = None
        if self.pdca_loop is None:
            objective_scheduler_degraded = True
            if not objective_scheduler_reason:
                objective_scheduler_reason = "pdca_loop_unavailable"
        else:
            try:
                pdca_running = bool(getattr(self.pdca_loop, "running", False))
            except Exception:
                pdca_running = False
            if not pdca_running:
                objective_scheduler_degraded = True
                if not objective_scheduler_reason:
                    objective_scheduler_reason = "pdca_loop_stopped"
            try:
                breaker_probe = getattr(self.pdca_loop, "_circuit_breaker_tripped", None)
                if callable(breaker_probe):
                    pdca_breaker_tripped = bool(breaker_probe())
                    if pdca_breaker_tripped:
                        objective_scheduler_degraded = True
                        if not objective_scheduler_reason:
                            objective_scheduler_reason = "pdca_circuit_breaker_tripped"
            except Exception:
                pdca_breaker_tripped = None

        return {
            "planner_bot_id": bot_id,
            "planner_healthy": bool(planner.planner_healthy),
            "planner_stale": planner_stale,
            "planner_stale_seconds": planner_stale_seconds,
            "planner_stale_threshold_s": planner_stale_threshold_s,
            "planner_last_updated_at": planner_updated_at,
            "fleet_mode": str(fleet_status.get("mode") or "local"),
            "fleet_central_available": bool(fleet_status.get("central_available", False)),
            "fleet_central_stale": bool(fleet_status.get("stale", True)),
            "fleet_last_sync_at": fleet_last_sync_at,
            "objective_scheduler_degraded": objective_scheduler_degraded,
            "objective_scheduler_degraded_reason": objective_scheduler_reason,
            "pdca_running": pdca_running,
            "pdca_circuit_breaker_tripped": pdca_breaker_tripped,
        }

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
        if self.slo_metrics is not None:
            self.slo_metrics.record_provider_route(
                workload=payload.workload,
                provider=decision.selected_provider,
                model=decision.selected_model,
            )
        trace_id = self._trace_id_for_bot("fleet", prefix="provider")
        if self.explainability is not None:
            self.explainability.add(
                kind="provider_route",
                bot_id="fleet",
                trace_id=trace_id,
                summary=f"workload={payload.workload} provider={decision.selected_provider}",
                details={
                    "workload": payload.workload,
                    "selected_provider": decision.selected_provider,
                    "selected_model": decision.selected_model,
                    "fallback_chain": list(decision.fallback_chain),
                    "policy_version": decision.policy_version,
                },
            )
        if self.trace_store is not None:
            self.trace_store.add_event(
                trace_id=trace_id,
                name="providers.route",
                attributes={
                    "workload": payload.workload,
                    "provider": decision.selected_provider,
                    "model": decision.selected_model,
                    "policy_version": decision.policy_version,
                },
            )
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

    def observability_metrics_text(self) -> str:
        if self.slo_metrics is None:
            return ""
        return self.slo_metrics.render_prometheus()

    def observability_recent_traces(self, *, limit: int = 50) -> list[dict[str, object]]:
        if self.trace_store is None:
            return []
        return self.trace_store.recent(limit=limit)

    def observability_trace(self, *, trace_id: str) -> list[dict[str, object]]:
        if self.trace_store is None:
            return []
        return self.trace_store.get_trace(trace_id=trace_id)

    def observability_incidents(self, *, include_closed: bool = False, limit: int = 100) -> list[dict[str, object]]:
        if self.incident_registry is None:
            return []
        return self.incident_registry.list_incidents(include_closed=include_closed, limit=limit)

    def observability_ack_incident(self, *, incident_id: str, assignee: str = "") -> dict[str, object]:
        if self.incident_registry is None:
            return {"ok": False, "message": "incident_registry_unavailable", "incident_id": incident_id}
        result = self.incident_registry.ack(incident_id=incident_id, assignee=assignee)
        self._audit(
            level="info" if bool(result.get("ok")) else "warning",
            event_type="incident_ack",
            summary="incident acknowledged" if bool(result.get("ok")) else "incident acknowledge failed",
            bot_id=None,
            payload={"incident_id": incident_id, "assignee": assignee, "result": dict(result)},
        )
        return result

    def observability_escalate_incident(self, *, incident_id: str, assignee: str = "") -> dict[str, object]:
        if self.incident_registry is None:
            return {"ok": False, "message": "incident_registry_unavailable", "incident_id": incident_id}
        result = self.incident_registry.escalate(incident_id=incident_id, assignee=assignee)
        self._audit(
            level="warning" if bool(result.get("ok")) else "error",
            event_type="incident_escalate",
            summary="incident escalated" if bool(result.get("ok")) else "incident escalation failed",
            bot_id=None,
            payload={"incident_id": incident_id, "assignee": assignee, "result": dict(result)},
        )
        return result

    def observability_explainability(
        self,
        *,
        kind: str | None = None,
        bot_id: str | None = None,
        trace_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        if self.explainability is None:
            return []
        return self.explainability.list(kind=kind, bot_id=bot_id, trace_id=trace_id, limit=limit)

    def observability_security_violations(self, *, limit: int = 100) -> list[dict[str, object]]:
        if self.security_auditor is None:
            return []
        return self.security_auditor.recent(limit=limit)

    def observability_doctrine_active(self) -> dict[str, object]:
        if self.doctrine_manager is None:
            return {"ok": False, "message": "doctrine_manager_unavailable"}
        return {"ok": True, "doctrine": self.doctrine_manager.active()}

    def observability_doctrine_versions(self, *, limit: int = 50) -> dict[str, object]:
        if self.doctrine_manager is None:
            return {"ok": False, "message": "doctrine_manager_unavailable", "versions": []}
        return {
            "ok": True,
            "versions": self.doctrine_manager.list_versions(limit=limit),
            "active": self.doctrine_manager.active(),
        }

    def observability_doctrine_publish(
        self,
        *,
        version: str,
        policy: dict[str, object],
        canary_percentage: float,
        activate: bool,
        author: str = "",
        trace_id: str | None = None,
    ) -> dict[str, object]:
        if self.doctrine_manager is None:
            return {"ok": False, "message": "doctrine_manager_unavailable"}

        if self.security_auditor is not None:
            allowed, reason = self.security_auditor.validate_doctrine(doctrine=policy)
            if not allowed:
                self.security_auditor.record(
                    kind="doctrine_policy_violation",
                    source="observability_doctrine_publish",
                    bot_id="fleet",
                    detail=reason,
                    severity="error",
                )
                self._audit(
                    level="error",
                    event_type="doctrine_publish_blocked",
                    summary="doctrine publish blocked by security policy",
                    bot_id=None,
                    payload={"version": version, "reason": reason},
                )
                return {"ok": False, "message": reason, "version": version}

        result = self.doctrine_manager.publish(
            version=version,
            policy=policy,
            canary_percentage=canary_percentage,
            activate=activate,
            author=author,
        )

        level = "info" if bool(result.get("ok")) else "warning"
        self._audit(
            level=level,
            event_type="doctrine_publish",
            summary="doctrine published" if bool(result.get("ok")) else "doctrine publish failed",
            bot_id=None,
            payload={
                "version": version,
                "activate": activate,
                "canary_percentage": float(canary_percentage),
                "author": author,
                "result": dict(result),
            },
        )
        tid = trace_id or self._trace_id_for_bot("fleet", prefix="doctrine")
        if self.trace_store is not None:
            self.trace_store.add_event(
                trace_id=tid,
                name="doctrine.publish",
                attributes={
                    "ok": bool(result.get("ok")),
                    "version": version,
                    "activate": activate,
                    "canary_percentage": float(canary_percentage),
                },
            )
        if self.explainability is not None:
            self.explainability.add(
                kind="doctrine",
                bot_id="fleet",
                trace_id=tid,
                summary=f"doctrine publish version={version} ok={bool(result.get('ok'))}",
                details={
                    "activate": activate,
                    "canary_percentage": float(canary_percentage),
                    "author": author,
                    "result": dict(result),
                },
            )
        return result

    def observability_doctrine_rollback(
        self,
        *,
        target_version: str | None = None,
        author: str = "",
        trace_id: str | None = None,
    ) -> dict[str, object]:
        if self.doctrine_manager is None:
            return {"ok": False, "message": "doctrine_manager_unavailable"}

        result = self.doctrine_manager.rollback(target_version=target_version)
        level = "warning" if bool(result.get("ok")) else "error"
        self._audit(
            level=level,
            event_type="doctrine_rollback",
            summary="doctrine rolled back" if bool(result.get("ok")) else "doctrine rollback failed",
            bot_id=None,
            payload={
                "target_version": target_version,
                "author": author,
                "result": dict(result),
            },
        )
        tid = trace_id or self._trace_id_for_bot("fleet", prefix="doctrine")
        if self.trace_store is not None:
            self.trace_store.add_event(
                trace_id=tid,
                name="doctrine.rollback",
                attributes={
                    "ok": bool(result.get("ok")),
                    "target_version": target_version or "",
                },
            )
        if self.explainability is not None:
            self.explainability.add(
                kind="doctrine",
                bot_id="fleet",
                trace_id=tid,
                summary=f"doctrine rollback target={target_version or 'previous'} ok={bool(result.get('ok'))}",
                details={"author": author, "result": dict(result)},
            )
        return result

    def _audit(
        self,
        *,
        level: str,
        event_type: str,
        summary: str,
        bot_id: str | None,
        payload: dict[str, object],
    ) -> None:
        if self.observability_audit is not None:
            self.observability_audit.record(
                level=level,
                event_type=event_type,
                summary=summary,
                bot_id=bot_id,
                payload=payload,
            )
            return
        if self.audit_trail is not None:
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
        if openmemory_provider.enabled:
            provider = openmemory_provider
        else:
            provider = fallback_provider
            memory_provider_error = openmemory_provider.init_error
            logger.warning(
                "memory_backend_fallback_to_sqlite",
                extra={
                    "event": "memory_backend_fallback_to_sqlite",
                    "requested_backend": backend,
                    "fallback_backend": "sqlite" if isinstance(fallback_provider, SQLiteMemoryProvider) else "in_memory",
                    "reason": openmemory_provider.init_error,
                },
            )
            if backend == "openmemory" and not isinstance(fallback_provider, SQLiteMemoryProvider):
                persistence_degraded = True

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

    policy_rules = _sanitize_provider_policy_rules(
        policy_rules,
        available_providers={str(key).strip().lower() for key in provider_adapters.keys()},
    )

    autonomy_ranked_objectives = _parse_csv_tokens(settings.autonomy_ranked_objectives)
    autonomy_preferred_grind_maps = _parse_csv_tokens(settings.autonomy_preferred_grind_maps)
    autonomy_policy: dict[str, object] = {
        "objective_max_age_cycles": int(settings.autonomy_objective_max_age_cycles),
        "max_active_objectives": int(settings.autonomy_max_active_objectives),
        "priority_decay_per_cycle": float(settings.autonomy_priority_decay_per_cycle),
        "objective_rotation_cooldown_s": float(settings.autonomy_objective_rotation_cooldown_s),
        "ranked_objectives": autonomy_ranked_objectives,
        "stale_plan_threshold_s": float(settings.autonomy_stale_plan_threshold_s),
        "death_recovery_cooldown_s": float(settings.autonomy_death_recovery_cooldown_s),
        "reconnect_grace_s": float(settings.autonomy_reconnect_grace_s),
        "preferred_grind_maps": autonomy_preferred_grind_maps,
        "preferred_grind_map_policy": str(settings.autonomy_preferred_grind_map_policy),
    }
    autonomy_scheduler_degraded = not bool(autonomy_ranked_objectives)
    autonomy_scheduler_degraded_reason = "autonomy_ranked_objectives_empty" if autonomy_scheduler_degraded else ""

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

    denylist = [item.strip() for item in settings.security_doctrine_denylist.split(",") if item.strip()]
    trace_store = (
        TraceStore(
            max_traces=settings.observability_trace_max_traces,
            max_events_per_trace=settings.observability_trace_max_events_per_trace,
        )
        if settings.observability_enable_tracing
        else None
    )
    incident_registry = IncidentRegistry(max_open=settings.observability_incident_max_open)
    explainability_store = ExplainabilityStore(max_records=settings.observability_explainability_max_records)
    security_auditor = SecurityAuditor(doctrine_denylist=denylist)
    doctrine_manager = DoctrineManager()
    slo_metrics = SLOMetricsCollector() if settings.observability_enable_metrics else None
    if slo_metrics is not None:
        model_router.set_route_metric_observer(
            lambda workload, provider, model: slo_metrics.record_provider_route(
                workload=workload,
                provider=provider,
                model=model,
            )
        )
    observability_audit = ObservabilityAuditLogger(
        audit_trail=audit_trail,
        incident_registry=incident_registry,
        security_auditor=security_auditor,
    )

    fleet_sync_client = FleetSyncClient(
        base_url=settings.fleet_central_base_url,
        timeout_seconds=settings.fleet_request_timeout_seconds,
        enabled=settings.fleet_central_enabled,
    )
    fleet_constraint_state = ConstraintIngestionState()
    fleet_outcome_reporter = OutcomeReporter(client=fleet_sync_client)
    fleet_conflict_resolver = FleetConflictResolver()

    action_queue = ActionQueue(max_per_bot=settings.action_max_queue_per_bot)
    snapshot_cache = SnapshotCache(ttl_seconds=settings.snapshot_cache_ttl_seconds)
    action_arbiter = ActionArbiter(
        queue=action_queue,
        fleet_client=fleet_sync_client,
        constraint_state=fleet_constraint_state,
        snapshot_cache=snapshot_cache,
    )
    control_policy = default_control_policy()
    control_parser = ControlParser()
    control_storage = ControlStorage(workspace_root=workspace_root, parser=control_parser)
    control_registry = ControlRegistry()
    control_state = ControlStateStore()
    control_planner = ControlPlanner(storage=control_storage)
    control_executor = ControlExecutor(runtime=None, storage=control_storage)
    control_validator = ControlValidator(storage=control_storage)
    control_domain = ControlDomainService(
        storage=control_storage,
        policy=control_policy,
        registry=control_registry,
        planner=control_planner,
        executor=control_executor,
        validator=control_validator,
        state=control_state,
    )

    runtime = RuntimeState(
        started_at=datetime.now(UTC),
        workspace_root=workspace_root,
        bot_registry=BotRegistry(),
        snapshot_cache=snapshot_cache,
        action_queue=action_queue,
        action_arbiter=action_arbiter,
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
        fleet_sync_client=fleet_sync_client,
        fleet_constraint_state=fleet_constraint_state,
        fleet_outcome_reporter=fleet_outcome_reporter,
        fleet_conflict_resolver=fleet_conflict_resolver,
        observability_audit=observability_audit,
        slo_metrics=slo_metrics,
        trace_store=trace_store,
        incident_registry=incident_registry,
        explainability=explainability_store,
        security_auditor=security_auditor,
        doctrine_manager=doctrine_manager,
        autonomy_policy=autonomy_policy,
        autonomy_scheduler_degraded=autonomy_scheduler_degraded,
        autonomy_scheduler_degraded_reason=autonomy_scheduler_degraded_reason,
        planner_stale_threshold_s=float(settings.autonomy_stale_plan_threshold_s),
        control_domain=control_domain,
    )

    control_executor.runtime = runtime

    planner_service = PlannerService(
        runtime=runtime,
        context_assembler=PlannerContextAssembler(runtime=runtime),
        intent_synthesizer=IntentSynthesizer(),
        plan_generator=PlanGenerator(
            model_router=model_router,
            planner_timeout_seconds=settings.planner_timeout_seconds,
            planner_retries=settings.planner_retries,
            max_user_prompt_chars=settings.llm_prompt_max_chars,
        ),
        plan_validator=PlanValidator(
            tactical_budget_ms=settings.planner_tactical_budget_ms,
            strategic_budget_ms=settings.planner_strategic_budget_ms,
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
            "fleet_central_enabled": settings.fleet_central_enabled,
            "fleet_central_base_url": settings.fleet_central_base_url,
            "fleet_mode": runtime.fleet_constraint_state.status().get("mode") if runtime.fleet_constraint_state is not None else "local",
            "fleet_outcome_backlog": runtime.fleet_outcome_reporter.backlog_size() if runtime.fleet_outcome_reporter is not None else 0,
            "autonomy_policy": autonomy_policy,
            "autonomy_scheduler_degraded": autonomy_scheduler_degraded,
            "autonomy_scheduler_degraded_reason": autonomy_scheduler_degraded_reason,
            "control_domain_enabled": runtime.control_domain is not None,
        },
    )
    logger.info(
        "runtime_initialized",
        extra={"event": "runtime_initialized", "sqlite_path": str(sqlite_path), "degraded": persistence_degraded},
    )
    return runtime


def start_fleet_sync_loop(runtime: RuntimeState) -> asyncio.Task[None]:
    async def _fleet_sync_loop() -> None:
        while True:
            try:
                await asyncio.sleep(30)
                if runtime.fleet_sync_client is None or runtime.fleet_constraint_state is None:
                    continue
                bots = runtime.list_bots()
                if not bots:
                    continue
                for bot in bots:
                    bot_id = str(bot.get("bot_id") or "")
                    if not bot_id:
                        continue
                    payload = FleetSyncRequest(
                        meta=ContractMeta(
                            contract_version=settings.contract_version,
                            source="fleet_sync_loop",
                            bot_id=bot_id,
                        ),
                        include_blackboard=True,
                    )
                    response = runtime.fleet_sync(payload)
                    if response.ok and response.blackboard:
                        runtime.fleet_constraint_state.parse_blackboard(response.blackboard)
                        runtime._fleet_refresh_role_from_blackboard(
                            bot_id=bot_id,
                            blackboard=runtime.fleet_constraint_state.blackboard(),
                        )

                    if runtime.fleet_outcome_reporter is not None:
                        drained = runtime.fleet_outcome_reporter.flush_backlog()
                        if drained:
                            fleet_logger.info("fleet_outcome_backlog_flushed", bot_id=bot_id, drained=drained)

                    role_state = runtime._fleet_role_manager(bot_id=bot_id).current()
                    lease_id = None
                    role_name = None
                    blackboard = runtime.fleet_constraint_state.blackboard()
                    role_leases = blackboard.get("role_leases") if isinstance(blackboard.get("role_leases"), list) else []
                    for lease in role_leases:
                        if not isinstance(lease, dict):
                            continue
                        if str(lease.get("bot_id") or "") != bot_id:
                            continue
                        lease_id = str(lease.get("lease_id") or lease.get("id") or "")
                        role_name = str(lease.get("role") or "")
                        if lease_id:
                            break

                    if lease_id and role_name and runtime.fleet_sync_client.enabled:
                        expires_at = role_state.get("expires_at")
                        if isinstance(expires_at, datetime):
                            expires_at = expires_at if expires_at.tzinfo is not None else expires_at.replace(tzinfo=UTC)
                            remaining = (expires_at - datetime.now(UTC)).total_seconds()
                            if remaining <= 60:
                                ok, result, err = await runtime.fleet_sync_client.renew_role(
                                    role_name,
                                    lease_id,
                                    bot_id=bot_id,
                                )
                                if ok:
                                    fleet_logger.info(
                                        "fleet_role_renewed",
                                        bot_id=bot_id,
                                        role=role_name,
                                        lease_id=lease_id,
                                    )
                                else:
                                    fleet_logger.warning(
                                        "fleet_role_renew_failed",
                                        bot_id=bot_id,
                                        role=role_name,
                                        lease_id=lease_id,
                                        error=err,
                                    )

                    snapshot = runtime.snapshot_cache.get(bot_id)
                    map_name = None
                    if snapshot is not None:
                        map_name = str(snapshot.position.map or "") if snapshot.position.map else None
                    if map_name:
                        claim = runtime.fleet_constraint_state.get_zone_claim(map_name)
                        if claim and claim.claimed_by and claim.claimed_by != bot_id:
                            fleet_logger.warning(
                                "fleet_zone_claim_conflict",
                                bot_id=bot_id,
                                map=map_name,
                                claimed_by=claim.claimed_by,
                            )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                fleet_logger.error("fleet_sync_loop_error", error=str(exc))

    return asyncio.create_task(_fleet_sync_loop())
