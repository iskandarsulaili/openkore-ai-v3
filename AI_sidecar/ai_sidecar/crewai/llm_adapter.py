from __future__ import annotations

import asyncio
import json
from threading import Thread
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

try:
    from crewai.llms.base_llm import BaseLLM
except Exception:  # pragma: no cover - optional dependency fallback
    BaseLLM = BaseModel  # type: ignore[misc,assignment]

from ai_sidecar.providers.base import PlannerModelRequest


class ProviderBackedCrewLLM(BaseLLM):
    """CrewAI-compatible local LLM bridge backed by the sidecar provider router."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm_type: str = Field(default="base")
    model: str = Field(min_length=1)
    provider: str = Field(default="provider-router")
    model_router: Any
    workload: str = Field(default="strategic_planning")
    timeout_seconds: float = Field(default=45.0, ge=1.0, le=600.0)
    max_retries: int = Field(default=1, ge=0, le=8)
    bot_id: str = Field(default="fleet", min_length=1, max_length=128)
    trace_id: str = Field(default_factory=lambda: uuid4().hex)

    # CrewAI expects this method on custom LLM adapters.
    def call(
        self,
        messages: str | list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str:
        del callbacks, available_functions, from_task, from_agent, response_model

        prompt = self._messages_to_prompt(messages)
        schema: dict[str, object] = {
            "type": "object",
            "required": ["response"],
            "properties": {
                "response": {"type": "string"},
                "notes": {"type": "array", "items": {"type": "string"}},
            },
        }
        if tools:
            schema["properties"] = {
                **dict(schema["properties"]),
                "tool_plan": {"type": "array", "items": {"type": "string"}},
            }

        request = PlannerModelRequest(
            bot_id=self.bot_id,
            trace_id=self.trace_id,
            task=self.workload,
            model=self.model,
            system_prompt=(
                "You are a specialist strategic planning assistant in a local Ragnarok Online"
                " sidecar. Respond with compact JSON only."
            ),
            user_prompt=prompt,
            schema=schema,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            metadata={"source": "crewai", "provider": self.provider},
        )
        response, _decision = self._run_async_blocking(self.model_router.generate_with_fallback(request=request))
        if not response.ok:
            return f"Provider routing failure: {response.error or 'unknown_error'}"

        if isinstance(response.content, dict):
            candidate = response.content.get("response")
            if isinstance(candidate, str) and candidate.strip():
                return candidate
            return json.dumps(response.content, ensure_ascii=False)
        if response.raw_text:
            return response.raw_text
        return "No provider content returned."

    def supports_function_calling(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 8192

    def _messages_to_prompt(self, messages: str | list[dict[str, Any]]) -> str:
        if isinstance(messages, str):
            return messages
        parts: list[str] = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "user")
            content = str(item.get("content") or "")
            parts.append(f"[{role}] {content}")
        return "\n".join(parts)

    def _run_async_blocking(self, awaitable: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(awaitable)

        result: dict[str, Any] = {}

        def worker() -> None:
            try:
                result["value"] = asyncio.run(awaitable)
            except Exception as exc:  # pragma: no cover - passthrough error branch
                result["error"] = exc

        thread = Thread(target=worker, daemon=True)
        thread.start()
        thread.join()
        if "error" in result:
            raise result["error"]
        return result.get("value")
