from __future__ import annotations

from typing import Any

from ai_sidecar.contracts.telemetry import TelemetryLevel
from ai_sidecar.providers.base import EmbeddingResponse, LLMProvider, PlannerModelRequest, PlannerModelResponse, ProviderHealth


class OllamaAdapter(LLMProvider):
    provider_name = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
        default_model: str,
        embedding_model: str,
        guard,
        breaker,
        timeout_seconds: float,
        max_retries: int,
        telemetry_push=None,
    ) -> None:
        super().__init__(
            guard=guard,
            breaker=breaker,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            telemetry_push=telemetry_push,
        )
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._embedding_model = embedding_model

    async def generate_structured(self, request: PlannerModelRequest) -> PlannerModelResponse:
        model = request.model or self._default_model
        system_prompt = self._guard.ensure_prompt_safe(request.system_prompt, field="system_prompt")
        user_prompt = self._guard.ensure_prompt_safe(request.user_prompt, field="user_prompt")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": request.schema,
            "options": {
                "temperature": 0.2,
            },
        }
        data, latency_ms, error = await self._post_json(
            bot_id=request.bot_id,
            trace_id=request.trace_id,
            breaker_key="provider.ollama",
            url=f"{self._base_url}/api/chat",
            headers={"Content-Type": "application/json"},
            payload=payload,
            timeout_seconds=request.timeout_seconds or self.timeout_seconds,
            max_retries=request.max_retries,
        )
        if data is None:
            self.emit_telemetry(
                bot_id=request.bot_id,
                level=TelemetryLevel.warning,
                event="llm_request_failed",
                message=error,
                metrics={"latency_ms": latency_ms},
                tags={"task": request.task},
            )
            return PlannerModelResponse(
                ok=False,
                provider=self.provider_name,
                model=model,
                trace_id=request.trace_id,
                latency_ms=latency_ms,
                content=None,
                raw_text="",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                error=error,
            )

        message = data.get("message") if isinstance(data.get("message"), dict) else {}
        raw_text = str(message.get("content") or "")
        content, schema_error = self._parse_json_content(raw_text, request.schema)
        if content is None:
            self.emit_telemetry(
                bot_id=request.bot_id,
                level=TelemetryLevel.warning,
                event="llm_schema_invalid",
                message=schema_error,
                metrics={"latency_ms": latency_ms},
                tags={"task": request.task},
            )
            return PlannerModelResponse(
                ok=False,
                provider=self.provider_name,
                model=model,
                trace_id=request.trace_id,
                latency_ms=latency_ms,
                content=None,
                raw_text=raw_text,
                usage={
                    "prompt_tokens": int(data.get("prompt_eval_count") or 0),
                    "completion_tokens": int(data.get("eval_count") or 0),
                    "total_tokens": int((data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0)),
                },
                error=schema_error,
            )

        usage = {
            "prompt_tokens": int(data.get("prompt_eval_count") or 0),
            "completion_tokens": int(data.get("eval_count") or 0),
            "total_tokens": int((data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0)),
        }
        return PlannerModelResponse(
            ok=True,
            provider=self.provider_name,
            model=model,
            trace_id=request.trace_id,
            latency_ms=latency_ms,
            content=content,
            raw_text=raw_text,
            usage=usage,
        )

    async def embed(self, *, bot_id: str, trace_id: str, model: str, texts: list[str]) -> EmbeddingResponse:
        target_model = model or self._embedding_model
        payload: dict[str, Any] = {
            "model": target_model,
            "input": texts,
            "truncate": True,
        }
        data, _, error = await self._post_json(
            bot_id=bot_id,
            trace_id=trace_id,
            breaker_key="provider.ollama.embed",
            url=f"{self._base_url}/api/embed",
            headers={"Content-Type": "application/json"},
            payload=payload,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
        )
        if data is None:
            return EmbeddingResponse(ok=False, provider=self.provider_name, model=target_model, vectors=[], dimensions=0, error=error)
        embeddings = data.get("embeddings") if isinstance(data.get("embeddings"), list) else []
        vectors = [[float(v) for v in row] for row in embeddings if isinstance(row, list)]
        dims = len(vectors[0]) if vectors else 0
        return EmbeddingResponse(
            ok=True,
            provider=self.provider_name,
            model=target_model,
            vectors=vectors,
            dimensions=dims,
            usage={"prompt_tokens": int(data.get("prompt_eval_count") or 0), "total_tokens": int(data.get("prompt_eval_count") or 0)},
        )

    async def health(self, *, bot_id: str) -> ProviderHealth:
        data, latency_ms, error = await self._get_json(
            bot_id=bot_id,
            trace_id="health",
            breaker_key="provider.ollama.health",
            url=f"{self._base_url}/api/tags",
            headers={"Content-Type": "application/json"},
            timeout_seconds=min(5.0, self.timeout_seconds),
            max_retries=0,
        )
        if data is None:
            return ProviderHealth(
                provider=self.provider_name,
                available=False,
                latency_ms=latency_ms,
                models=[],
                breaker_state="open",
                message=error,
            )
        models = []
        rows = data.get("models") if isinstance(data.get("models"), list) else []
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("name"), str):
                models.append(row["name"])
        return ProviderHealth(
            provider=self.provider_name,
            available=True,
            latency_ms=latency_ms,
            models=models,
            breaker_state="closed",
            message="ok",
        )
