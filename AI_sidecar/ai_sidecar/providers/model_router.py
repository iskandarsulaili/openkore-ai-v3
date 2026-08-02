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
    "autonomy_mission_decision": {
        "providers": ["openai"],
        "models": {"openai": "opencode-go/deepseek-v4-flash"},
    },
    "tactical_short_reasoning": {
        "providers": ["openai"],
        "models": {"openai": "opencode-go/deepseek-v4-flash"},
    },
    "strategic_planning": {
        "providers": ["openai"],
        "models": {"openai": "opencode-go/deepseek-v4-flash"},
    },
    "long_reflection": {
        "providers": ["openai"],
        "models": {"openai": "opencode-go/deepseek-v4-flash"},
    },
    "embeddings": {
        "providers": ["openai"],
        "models": {"openai": "text-embedding-3-small"},
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
        self._validate_policy_targets()

    def _validate_policy_targets(self) -> list[str]:
        """Warn about policy rules that reference unregistered providers/models.

        A policy rule whose providers/models are not backed by a registered
        adapter would silently no-op (`decide()` returns selected_provider
        "none"), so we surface it loudly at construction time instead of
        letting requests fall into a confusing "no_provider_for_workload"
        with no explanation.
        """
        registered = self.provider_names()
        problems: list[str] = []
        for workload, rule in (self._policy.rules or {}).items():
            if not isinstance(rule, dict):
                continue
            for prov in list(rule.get("providers") or []):
                p = str(prov).strip().lower()
                if p and p not in registered:
                    problems.append(f"{workload}: provider '{prov}' not registered")
                    logger.warning(
                        "model_router_policy_unregistered_provider",
                        extra={
                            "event": "model_router_policy_unregistered_provider",
                            "workload": workload,
                            "provider": prov,
                            "registered": sorted(registered),
                        },
                    )
        if problems:
            logger.warning(
                "model_router_policy_has_unregistered_targets total=%d",
                len(problems),
                extra={
                    "event": "model_router_policy_unregistered_targets",
                    "problems": problems,
                },
            )
        return problems

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
        
        # ── Cost gate ──────────────────────────────────────
        _ct = getattr(self, '_cost_tracker', None)
        if _ct is not None:
            _allowed, _reason = _ct.check(
                daily_budget_tokens=getattr(self, '_daily_budget', 100000),
                max_calls_per_hour=getattr(self, '_max_calls_per_hour', 30),
                tier=getattr(self, '_cost_tier', 'standard'),
                bot_id=request.bot_id if hasattr(request, 'bot_id') else 'default',
            )
            if not _allowed:
                logger.warning("cost_gate: %s", _reason)
                decision.fallback_chain = []
                return (
                    PlannerModelResponse(
                        ok=False, provider="cost_gate", model="",
                        trace_id=request.trace_id, latency_ms=0.0,
                        content=None, raw_text="",
                        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        error=f"cost_gate:{_reason}",
                    ),
                    decision,
                )

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
        
        # ── Retry with exponential backoff + jitter ──
        import random as _random
        _max_retries = 3
        _base_delay = 1.0
        response: PlannerModelResponse = PlannerModelResponse(
            ok=False, provider="", model="",
            trace_id=request.trace_id, latency_ms=0.0,
            content=None, raw_text="",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            error="no_provider_attempted",
        )
        
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
            
            # Retry loop for this provider
            for retry in range(_max_retries):
                logger.info(
                    "provider_route_attempt",
                    extra={
                        "event": "provider_route_attempt",
                        "workload": request.task,
                        "provider": provider_name,
                        "model": model,
                        "attempt_index": idx,
                        "retry": retry,
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
                
                # ── Structured parse recovery ──
                if not response.ok and response.error == "structured_parse_failed" and response.raw_text:
                    # Try to extract JSON from raw text
                    recovered = self._recover_structured(response.raw_text)
                    if recovered is not None:
                        response = PlannerModelResponse(
                            ok=True,
                            provider=response.provider or provider_name,
                            model=response.model or model,
                            trace_id=response.trace_id,
                            latency_ms=response.latency_ms,
                            content=recovered,
                            raw_text=response.raw_text,
                            usage=response.usage,
                            error="",
                        )
                        logger.info(
                            "provider_route_parse_recovered",
                            extra={
                                "event": "provider_route_parse_recovered",
                                "workload": request.task,
                                "provider": provider_name,
                                "model": model,
                                "trace_id": request.trace_id,
                                "bot_id": request.bot_id,
                            },
                        )
                
                if response.ok:
                    actual_provider = str(response.provider or provider_name)
                    actual_model = str(response.model or model)
                    if idx > 0 or retry > 0:
                        logger.warning(
                            "provider_route_fallback_used",
                            extra={
                                "event": "provider_route_fallback_used",
                                "workload": request.task,
                                "planned_provider": decision.selected_provider,
                                "actual_provider": actual_provider,
                                "actual_model": actual_model,
                                "attempted_providers": list(attempted_providers),
                                "retry": retry,
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
                        fallback_used=idx > 0 or retry > 0,
                    )

                # Check if this error is retryable
                if not self._is_retryable_error(response.error):
                    break  # Don't retry non-retryable errors
                
                # Exponential backoff with jitter
                if retry < _max_retries - 1:
                    delay = _base_delay * (2 ** retry) + _random.random() * 0.5
                    logger.info(
                        "provider_route_retry",
                        extra={
                            "event": "provider_route_retry",
                            "workload": request.task,
                            "provider": provider_name,
                            "model": model,
                            "retry": retry,
                            "delay_ms": int(delay * 1000),
                            "error": response.error,
                            "trace_id": request.trace_id,
                            "bot_id": request.bot_id,
                        },
                    )
                    # Exponential backoff with jitter before next retry
                    if retry < _max_retries - 1:
                        import asyncio
                        await asyncio.sleep(delay)

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

    def _recover_structured(self, raw_text: str) -> dict | None:
        """Attempt to recover structured JSON from raw LLM output.
        
        Handles common failure modes:
        - JSON embedded in markdown code blocks (```json ... ```)
        - Trailing commas
        - Single quotes instead of double quotes
        - Leading/trailing non-JSON text
        """
        import re as _re
        
        text = raw_text.strip()
        if not text:
            return None
        
        # Try 1: Extract from markdown code block
        code_match = _re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, _re.DOTALL)
        if code_match:
            candidate = code_match.group(1).strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        
        # Try 2: Find first { and last } — extract JSON object
        start = text.find('{')
        end = text.rfind('}')
        if start >= 0 and end > start:
            candidate = text[start:end+1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        
        # Try 3: Repair common issues
        try:
            # Replace single quotes with double quotes (for simple cases)
            repaired = _re.sub(r"(?<!\\)'", '"', candidate)
            # Remove trailing commas before closing braces
            repaired = _re.sub(r',\s*}', '}', repaired)
            repaired = _re.sub(r',\s*]', ']', repaired)
            return json.loads(repaired)
        except (json.JSONDecodeError, UnboundLocalError, NameError):
            pass
        
        return None

    def _is_retryable_error(self, error: str) -> bool:
        """Determine if an error is worth retrying.
        
        Retryable: timeout, rate limit, service unavailable, parse failures
        Non-retryable: auth errors, invalid requests, context overflow
        """
        if not error:
            return True
        error_lower = error.lower()
        # Non-retryable
        if any(kw in error_lower for kw in ["auth", "unauthorized", "forbidden", "invalid_api", "context_length", "context_window"]):
            return False
        # Retryable
        if any(kw in error_lower for kw in ["timeout", "rate_limit", "unavailable", "overloaded", "parse", "structured", "server_error", "503", "502", "429"]):
            return True
        # Default: retry once
        return True

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

    CONTEXT_BUDGETS: dict[str, int] = {"off": 0, "economy": 512, "standard": 2048, "premium": 8192}

    def max_context_tokens(self, tier: str = "standard") -> int:
        """Return max context tokens allowed for the given cost tier."""
        return self.CONTEXT_BUDGETS.get(tier, 2048)

    def set_cost_controls(self, *, tracker, daily_budget: int, max_calls_per_hour: int, tier: str) -> None:
        self._cost_tracker = tracker
        self._daily_budget = daily_budget
        self._max_calls_per_hour = max_calls_per_hour
        self._cost_tier = tier

    def current_policy(self) -> RoutePolicy:
        with self._lock:
            return RoutePolicy(
                version=self._policy.version,
                updated_at=self._policy.updated_at,
                rules=json.loads(json.dumps(self._policy.rules)),
            )
