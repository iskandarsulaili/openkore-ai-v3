from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from typing import Any, Callable

from ai_sidecar.providers.base import LLMProvider, PlannerModelRequest, PlannerModelResponse, ProviderHealth

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RoutingDecision:
    workload: str
    provider_order: list[str]
    selected_provider: str
    selected_model: str
    fallback_chain: list[str]
    policy_version: str
    planned_provider: str = ""
    planned_model: str = ""
    attempted_providers: list[str] = field(default_factory=list)
    attempted_models: dict[str, str] = field(default_factory=dict)
    fallback_used: bool = False


@dataclass(slots=True)
class RoutePolicy:
    version: str
    updated_at: datetime
    rules: dict[str, dict[str, Any]] = field(default_factory=dict)


DEFAULT_POLICY_RULES: dict[str, dict[str, Any]] = {
    "reflex_explain": {
        "providers": [],
        "models": {},
    },
    "tactical_short_reasoning": {
        "providers": ["ollama", "deepseek"],
        "models": {"ollama": "qwen3.6:35b-a3b-q4_K_M", "deepseek": "deepseek-chat"},
    },
    "strategic_planning": {
        "providers": ["ollama", "openai", "deepseek"],
        "models": {"ollama": "qwen3.6:35b-a3b-q4_K_M", "openai": "gpt-4o-mini", "deepseek": "deepseek-chat"},
    },
    "long_reflection": {
        "providers": ["deepseek", "openai", "ollama"],
        "models": {"deepseek": "deepseek-chat", "openai": "gpt-4o-mini", "ollama": "qwen3.6:35b-a3b-q4_K_M"},
    },
    "embeddings": {
        "providers": ["ollama", "openai", "deepseek"],
        "models": {"ollama": "nomic-embed-text", "openai": "text-embedding-3-small", "deepseek": "text-embedding-3-small"},
    },
}


class ModelRouter:
    def __init__(
        self,
        *,
        providers: dict[str, LLMProvider],
        initial_rules: dict[str, dict[str, Any]] | None = None,
        route_metric_observer: Callable[[str, str, str], None] | None = None,
    ) -> None:
        self._providers = providers
        self._lock = RLock()
        self._route_metric_observer = route_metric_observer
        seed_rules = initial_rules if initial_rules is not None else DEFAULT_POLICY_RULES
        self._policy = RoutePolicy(
            version=f"bootstrap-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
            updated_at=datetime.now(UTC),
            rules=json.loads(json.dumps(seed_rules)),
        )

    def set_route_metric_observer(self, observer: Callable[[str, str, str], None] | None) -> None:
        with self._lock:
            self._route_metric_observer = observer

    def provider_names(self) -> set[str]:
        with self._lock:
            return {str(name).strip().lower() for name in self._providers.keys()}

    def _emit_route_metric(self, *, workload: str, provider: str, model: str) -> None:
        with self._lock:
            observer = self._route_metric_observer
        if observer is None:
            return
        try:
            observer(workload, provider, model)
        except Exception:
            logger.exception(
                "provider_route_metric_emit_failed",
                extra={
                    "event": "provider_route_metric_emit_failed",
                    "workload": workload,
                    "provider": provider,
                    "model": model,
                },
            )

    def decide(self, *, workload: str) -> RoutingDecision:
        with self._lock:
            rule = self._policy.rules.get(workload) or self._policy.rules.get("strategic_planning") or {"providers": [], "models": {}}
            providers = [name for name in list(rule.get("providers") or []) if name in self._providers]
            models = rule.get("models") if isinstance(rule.get("models"), dict) else {}

            selected_provider = providers[0] if providers else "none"
            selected_model = str(models.get(selected_provider) or "") if selected_provider != "none" else ""
            decision = RoutingDecision(
                workload=workload,
                provider_order=providers,
                selected_provider=selected_provider,
                selected_model=selected_model,
                fallback_chain=providers[1:] if len(providers) > 1 else [],
                policy_version=self._policy.version,
                planned_provider=selected_provider,
                planned_model=selected_model,
            )
            logger.info(
                "provider_route_decided",
                extra={
                    "event": "provider_route_decided",
                    "workload": workload,
                    "selected_provider": decision.selected_provider,
                    "selected_model": decision.selected_model,
                    "fallback_chain": list(decision.fallback_chain),
                    "policy_version": decision.policy_version,
                },
            )
            return decision

    async def generate_with_fallback(self, *, request: PlannerModelRequest) -> tuple[PlannerModelResponse, RoutingDecision]:
        decision = self.decide(workload=request.task)
        if decision.selected_provider == "none":
            self._emit_route_metric(workload=request.task, provider="none", model="")
            return (
                PlannerModelResponse(
                    ok=False,
                    provider="none",
                    model="",
                    trace_id=request.trace_id,
                    latency_ms=0.0,
                    content=None,
                    raw_text="",
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    error="no_provider_for_workload",
                ),
                decision,
            )

        provider_order = [decision.selected_provider, *decision.fallback_chain]
        last_response: PlannerModelResponse | None = None
        attempted_providers: list[str] = []
        attempted_models: dict[str, str] = {}
        for idx, provider_name in enumerate(provider_order):
            provider = self._providers.get(provider_name)
            if provider is None:
                logger.warning(
                    "provider_route_missing_adapter",
                    extra={
                        "event": "provider_route_missing_adapter",
                        "workload": request.task,
                        "provider": provider_name,
                        "trace_id": request.trace_id,
                        "bot_id": request.bot_id,
                    },
                )
                continue
            model = decision.selected_model
            with self._lock:
                rule = self._policy.rules.get(request.task) or {}
                models = rule.get("models") if isinstance(rule.get("models"), dict) else {}
                model = str(models.get(provider_name) or model)

            attempted_providers.append(provider_name)
            attempted_models[provider_name] = model
            logger.info(
                "provider_route_attempt",
                extra={
                    "event": "provider_route_attempt",
                    "workload": request.task,
                    "provider": provider_name,
                    "model": model,
                    "attempt_index": idx,
                    "trace_id": request.trace_id,
                    "bot_id": request.bot_id,
                },
            )

            response = await provider.generate_structured(
                PlannerModelRequest(
                    bot_id=request.bot_id,
                    trace_id=request.trace_id,
                    task=request.task,
                    model=model,
                    system_prompt=request.system_prompt,
                    user_prompt=request.user_prompt,
                    schema=request.schema,
                    timeout_seconds=request.timeout_seconds,
                    max_retries=request.max_retries,
                    metadata=dict(request.metadata),
                )
            )
            if response.ok:
                actual_provider = str(response.provider or provider_name)
                actual_model = str(response.model or model)
                if idx > 0:
                    logger.warning(
                        "provider_route_fallback_used",
                        extra={
                            "event": "provider_route_fallback_used",
                            "workload": request.task,
                            "planned_provider": decision.selected_provider,
                            "actual_provider": actual_provider,
                            "actual_model": actual_model,
                            "attempted_providers": list(attempted_providers),
                            "trace_id": request.trace_id,
                            "bot_id": request.bot_id,
                        },
                    )
                else:
                    logger.info(
                        "provider_route_primary_succeeded",
                        extra={
                            "event": "provider_route_primary_succeeded",
                            "workload": request.task,
                            "provider": actual_provider,
                            "model": actual_model,
                            "trace_id": request.trace_id,
                            "bot_id": request.bot_id,
                        },
                    )
                self._emit_route_metric(workload=request.task, provider=actual_provider, model=actual_model)
                return response, RoutingDecision(
                    workload=decision.workload,
                    provider_order=list(decision.provider_order),
                    selected_provider=actual_provider,
                    selected_model=actual_model,
                    fallback_chain=provider_order[idx + 1 :],
                    policy_version=decision.policy_version,
                    planned_provider=decision.selected_provider,
                    planned_model=decision.selected_model,
                    attempted_providers=list(attempted_providers),
                    attempted_models=dict(attempted_models),
                    fallback_used=idx > 0,
                )

            logger.warning(
                "provider_route_attempt_failed",
                extra={
                    "event": "provider_route_attempt_failed",
                    "workload": request.task,
                    "provider": provider_name,
                    "model": model,
                    "attempt_index": idx,
                    "error": response.error,
                    "latency_ms": float(response.latency_ms),
                    "trace_id": request.trace_id,
                    "bot_id": request.bot_id,
                },
            )
            last_response = response
        if last_response is None:
            last_response = PlannerModelResponse(
                ok=False,
                provider="none",
                model="",
                trace_id=request.trace_id,
                latency_ms=0.0,
                content=None,
                raw_text="",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                error="no_available_provider_adapter",
            )

        failed_provider = str(last_response.provider or (attempted_providers[-1] if attempted_providers else decision.selected_provider))
        failed_model = str(last_response.model or attempted_models.get(failed_provider, ""))
        logger.error(
            "provider_route_exhausted",
            extra={
                "event": "provider_route_exhausted",
                "workload": request.task,
                "planned_provider": decision.selected_provider,
                "failed_provider": failed_provider,
                "failed_model": failed_model,
                "attempted_providers": list(attempted_providers),
                "trace_id": request.trace_id,
                "bot_id": request.bot_id,
            },
        )
        self._emit_route_metric(workload=request.task, provider="none", model="")
        return last_response, RoutingDecision(
            workload=decision.workload,
            provider_order=list(decision.provider_order),
            selected_provider=failed_provider,
            selected_model=failed_model,
            fallback_chain=[],
            policy_version=decision.policy_version,
            planned_provider=decision.selected_provider,
            planned_model=decision.selected_model,
            attempted_providers=list(attempted_providers),
            attempted_models=dict(attempted_models),
            fallback_used=len(attempted_providers) > 1,
        )

    async def health(self, *, bot_id: str) -> list[ProviderHealth]:
        rows: list[ProviderHealth] = []
        for name in sorted(self._providers):
            rows.append(await self._providers[name].health(bot_id=bot_id))
        return rows

    def update_policy(self, *, rules: dict[str, dict[str, Any]]) -> RoutePolicy:
        with self._lock:
            merged = json.loads(json.dumps(self._policy.rules))
            for key, value in rules.items():
                if not isinstance(value, dict):
                    continue
                merged[key] = {
                    "providers": [str(item) for item in list(value.get("providers") or [])],
                    "models": dict(value.get("models") or {}),
                }
            self._policy = RoutePolicy(
                version=f"policy-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
                updated_at=datetime.now(UTC),
                rules=merged,
            )
            return self._policy

    def current_policy(self) -> RoutePolicy:
        with self._lock:
            return RoutePolicy(
                version=self._policy.version,
                updated_at=self._policy.updated_at,
                rules=json.loads(json.dumps(self._policy.rules)),
            )
