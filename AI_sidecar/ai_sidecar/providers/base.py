from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from typing import Any, Callable

import httpx

from ai_sidecar.contracts.telemetry import TelemetryEvent, TelemetryLevel
from ai_sidecar.providers.prompt_guard import PromptGuard
from ai_sidecar.reflex.circuit_breaker import ReflexCircuitBreaker

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PlannerModelRequest:
    bot_id: str
    trace_id: str
    task: str
    model: str
    system_prompt: str
    user_prompt: str
    schema: dict[str, Any]
    timeout_seconds: float
    max_retries: int
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class PlannerModelResponse:
    ok: bool
    provider: str
    model: str
    trace_id: str
    latency_ms: float
    content: dict[str, Any] | None
    raw_text: str
    usage: dict[str, int]
    error: str = ""
    refusal: str = ""


@dataclass(slots=True)
class EmbeddingResponse:
    ok: bool
    provider: str
    model: str
    vectors: list[list[float]]
    dimensions: int
    usage: dict[str, int] = field(default_factory=dict)
    error: str = ""


@dataclass(slots=True)
class ProviderHealth:
    provider: str
    available: bool
    latency_ms: float
    models: list[str]
    breaker_state: str
    message: str = ""


class LLMProvider:
    provider_name: str = "base"

    def __init__(
        self,
        *,
        guard: PromptGuard,
        breaker: ReflexCircuitBreaker,
        timeout_seconds: float,
        max_retries: int,
        telemetry_push: Callable[[str, list[TelemetryEvent]], object] | None = None,
    ) -> None:
        self._guard = guard
        self._breaker = breaker
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._telemetry_push = telemetry_push
        self._stats_lock = RLock()
        self._calls = 0
        self._failures = 0

    async def generate_structured(self, request: PlannerModelRequest) -> PlannerModelResponse:
        raise NotImplementedError

    async def embed(self, *, bot_id: str, trace_id: str, model: str, texts: list[str]) -> EmbeddingResponse:
        raise NotImplementedError

    async def health(self, *, bot_id: str) -> ProviderHealth:
        raise NotImplementedError

    @property
    def timeout_seconds(self) -> float:
        return self._timeout_seconds

    @property
    def max_retries(self) -> int:
        return self._max_retries

    async def _post_json(
        self,
        *,
        bot_id: str,
        trace_id: str,
        breaker_key: str,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
        max_retries: int,
    ) -> tuple[dict[str, Any] | None, float, str]:
        allowed, state = self._breaker.allow(bot_id=bot_id, key=breaker_key, family="provider")
        if not allowed:
            return None, 0.0, state

        error = ""
        last_latency = 0.0
        retries = max(0, max_retries)
        for attempt in range(retries + 1):
            started = datetime.now(UTC)
            try:
                async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                    response = await client.post(url, headers=headers, json=payload)
                    elapsed = (datetime.now(UTC) - started).total_seconds() * 1000.0
                    last_latency = elapsed
                    if response.status_code >= 400:
                        error = f"http_{response.status_code}"
                        if response.status_code in {408, 429, 500, 502, 503, 504} and attempt < retries:
                            await asyncio.sleep(min(0.2 * (attempt + 1), 1.0))
                            continue
                        self._breaker.record_failure(bot_id=bot_id, key=breaker_key, family="provider", reason=error)
                        self._record_failure()
                        return None, elapsed, error

                    data = response.json() if response.content else {}
                    if not isinstance(data, dict):
                        error = "invalid_json_object"
                        self._breaker.record_failure(bot_id=bot_id, key=breaker_key, family="provider", reason=error)
                        self._record_failure()
                        return None, elapsed, error

                    self._breaker.record_success(bot_id=bot_id, key=breaker_key, family="provider")
                    self._record_call()
                    return data, elapsed, ""
            except httpx.TimeoutException:
                elapsed = (datetime.now(UTC) - started).total_seconds() * 1000.0
                last_latency = elapsed
                error = "timeout"
                if attempt < retries:
                    await asyncio.sleep(min(0.2 * (attempt + 1), 1.0))
                    continue
                self._breaker.record_failure(bot_id=bot_id, key=breaker_key, family="provider", reason=error)
                self._record_failure()
                return None, elapsed, error
            except Exception as exc:
                elapsed = (datetime.now(UTC) - started).total_seconds() * 1000.0
                last_latency = elapsed
                error = f"exception:{type(exc).__name__}"
                if attempt < retries:
                    await asyncio.sleep(min(0.2 * (attempt + 1), 1.0))
                    continue
                self._breaker.record_failure(bot_id=bot_id, key=breaker_key, family="provider", reason=error)
                self._record_failure()
                logger.exception(
                    "provider_post_json_failed",
                    extra={
                        "event": "provider_post_json_failed",
                        "provider": self.provider_name,
                        "trace_id": trace_id,
                        "bot_id": bot_id,
                        "url": url,
                    },
                )
                return None, elapsed, error

        return None, last_latency, error or "unknown"

    def _extract_usage(self, payload: dict[str, Any]) -> dict[str, int]:
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return {
            "prompt_tokens": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
            "total_tokens": int(
                usage.get("total_tokens")
                or (
                    int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
                    + int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
                )
            ),
        }

    async def _get_json(
        self,
        *,
        bot_id: str,
        trace_id: str,
        breaker_key: str,
        url: str,
        headers: dict[str, str],
        timeout_seconds: float,
        max_retries: int,
    ) -> tuple[dict[str, Any] | None, float, str]:
        allowed, state = self._breaker.allow(bot_id=bot_id, key=breaker_key, family="provider")
        if not allowed:
            return None, 0.0, state

        error = ""
        last_latency = 0.0
        retries = max(0, max_retries)
        for attempt in range(retries + 1):
            started = datetime.now(UTC)
            try:
                async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                    response = await client.get(url, headers=headers)
                    elapsed = (datetime.now(UTC) - started).total_seconds() * 1000.0
                    last_latency = elapsed
                    if response.status_code >= 400:
                        error = f"http_{response.status_code}"
                        if response.status_code in {408, 429, 500, 502, 503, 504} and attempt < retries:
                            await asyncio.sleep(min(0.2 * (attempt + 1), 1.0))
                            continue
                        self._breaker.record_failure(bot_id=bot_id, key=breaker_key, family="provider", reason=error)
                        self._record_failure()
                        return None, elapsed, error
                    data = response.json() if response.content else {}
                    if not isinstance(data, dict):
                        error = "invalid_json_object"
                        self._breaker.record_failure(bot_id=bot_id, key=breaker_key, family="provider", reason=error)
                        self._record_failure()
                        return None, elapsed, error
                    self._breaker.record_success(bot_id=bot_id, key=breaker_key, family="provider")
                    self._record_call()
                    return data, elapsed, ""
            except httpx.TimeoutException:
                elapsed = (datetime.now(UTC) - started).total_seconds() * 1000.0
                last_latency = elapsed
                error = "timeout"
                if attempt < retries:
                    await asyncio.sleep(min(0.2 * (attempt + 1), 1.0))
                    continue
                self._breaker.record_failure(bot_id=bot_id, key=breaker_key, family="provider", reason=error)
                self._record_failure()
                return None, elapsed, error
            except Exception as exc:
                elapsed = (datetime.now(UTC) - started).total_seconds() * 1000.0
                last_latency = elapsed
                error = f"exception:{type(exc).__name__}"
                if attempt < retries:
                    await asyncio.sleep(min(0.2 * (attempt + 1), 1.0))
                    continue
                self._breaker.record_failure(bot_id=bot_id, key=breaker_key, family="provider", reason=error)
                self._record_failure()
                logger.exception(
                    "provider_get_json_failed",
                    extra={
                        "event": "provider_get_json_failed",
                        "provider": self.provider_name,
                        "trace_id": trace_id,
                        "bot_id": bot_id,
                        "url": url,
                    },
                )
                return None, elapsed, error
        return None, last_latency, error or "unknown"

    def _parse_json_content(self, content: str, schema: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
        parsed = self._guard.parse_json_object(content)
        if parsed is None:
            logger.warning(
                "structured_parse_failed",
                extra={
                    "event": "structured_parse_failed",
                    "provider": self.provider_name,
                    "content_preview": self._guard.preview(content or ""),
                },
            )
            return None, "structured_parse_failed"
        try:
            self._guard.validate_schema(parsed, schema)
        except Exception as exc:
            normalized = self._guard.normalize_for_schema(parsed, schema)
            try:
                self._guard.validate_schema(normalized, schema)
                logger.info(
                    "structured_schema_normalized",
                    extra={
                        "event": "structured_schema_normalized",
                        "provider": self.provider_name,
                    },
                )
                return normalized, ""
            except Exception:
                pass
            logger.warning(
                "structured_schema_invalid",
                extra={
                    "event": "structured_schema_invalid",
                    "provider": self.provider_name,
                    "error": str(exc),
                    "content_preview": self._guard.preview(content or ""),
                },
            )
            return None, str(exc)
        return parsed, ""

    def _record_call(self) -> None:
        with self._stats_lock:
            self._calls += 1

    def _record_failure(self) -> None:
        with self._stats_lock:
            self._failures += 1

    def stats(self) -> dict[str, int]:
        with self._stats_lock:
            return {"calls": self._calls, "failures": self._failures}

    def emit_telemetry(
        self,
        *,
        bot_id: str,
        level: TelemetryLevel,
        event: str,
        message: str,
        metrics: dict[str, float] | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        if self._telemetry_push is None:
            return
        payload = TelemetryEvent(
            timestamp=datetime.now(UTC),
            level=level,
            category="provider",
            event=event,
            message=message,
            metrics=dict(metrics or {}),
            tags={"provider": self.provider_name, **dict(tags or {})},
        )
        try:
            self._telemetry_push(bot_id, [payload])
        except Exception:
            logger.exception(
                "provider_telemetry_emit_failed",
                extra={"event": "provider_telemetry_emit_failed", "provider": self.provider_name, "bot_id": bot_id},
            )
