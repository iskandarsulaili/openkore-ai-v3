from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from typing import Any

from ai_sidecar.providers.base import LLMProvider, PlannerModelRequest, PlannerModelResponse, ProviderHealth


@dataclass(slots=True)
class RoutingDecision:
    workload: str
    provider_order: list[str]
    selected_provider: str
    selected_model: str
    fallback_chain: list[str]
    policy_version: str


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
        "models": {"ollama": "qwen2.5:7b", "deepseek": "deepseek-chat"},
    },
    "strategic_planning": {
        "providers": ["ollama", "openai", "deepseek"],
        "models": {"ollama": "qwen2.5:14b", "openai": "gpt-4o-mini", "deepseek": "deepseek-chat"},
    },
    "long_reflection": {
        "providers": ["deepseek", "openai", "ollama"],
        "models": {"deepseek": "deepseek-chat", "openai": "gpt-4o-mini", "ollama": "qwen2.5:14b"},
    },
    "embeddings": {
        "providers": ["ollama", "openai", "deepseek"],
        "models": {"ollama": "nomic-embed-text", "openai": "text-embedding-3-small", "deepseek": "text-embedding-3-small"},
    },
}


class ModelRouter:
    def __init__(self, *, providers: dict[str, LLMProvider], initial_rules: dict[str, dict[str, Any]] | None = None) -> None:
        self._providers = providers
        self._lock = RLock()
        seed_rules = initial_rules if initial_rules is not None else DEFAULT_POLICY_RULES
        self._policy = RoutePolicy(
            version=f"bootstrap-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
            updated_at=datetime.now(UTC),
            rules=json.loads(json.dumps(seed_rules)),
        )

    def decide(self, *, workload: str) -> RoutingDecision:
        with self._lock:
            rule = self._policy.rules.get(workload) or self._policy.rules.get("strategic_planning") or {"providers": [], "models": {}}
            providers = [name for name in list(rule.get("providers") or []) if name in self._providers]
            models = rule.get("models") if isinstance(rule.get("models"), dict) else {}

            selected_provider = providers[0] if providers else "none"
            selected_model = str(models.get(selected_provider) or "") if selected_provider != "none" else ""
            return RoutingDecision(
                workload=workload,
                provider_order=providers,
                selected_provider=selected_provider,
                selected_model=selected_model,
                fallback_chain=providers[1:] if len(providers) > 1 else [],
                policy_version=self._policy.version,
            )

    async def generate_with_fallback(self, *, request: PlannerModelRequest) -> tuple[PlannerModelResponse, RoutingDecision]:
        decision = self.decide(workload=request.task)
        if decision.selected_provider == "none":
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
        for provider_name in provider_order:
            provider = self._providers.get(provider_name)
            if provider is None:
                continue
            model = decision.selected_model
            with self._lock:
                rule = self._policy.rules.get(request.task) or {}
                models = rule.get("models") if isinstance(rule.get("models"), dict) else {}
                model = str(models.get(provider_name) or model)

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
                return response, decision
            last_response = response

        assert last_response is not None
        return last_response, decision

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
