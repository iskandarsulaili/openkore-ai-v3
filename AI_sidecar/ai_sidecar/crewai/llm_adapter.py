from __future__ import annotations

import asyncio
import json
import logging
from threading import Thread
from typing import Any, get_args, get_origin
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

try:
    from crewai.llms.base_llm import BaseLLM
except Exception:  # pragma: no cover - optional dependency fallback
    BaseLLM = BaseModel  # type: ignore[misc,assignment]

from ai_sidecar.providers.base import PlannerModelRequest


logger = logging.getLogger(__name__)


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
    max_prompt_chars: int = Field(default=24000, ge=1024, le=200000)
    max_message_count: int = Field(default=24, ge=4, le=200)

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
        del callbacks, available_functions, from_task, from_agent

        prompt = self._messages_to_prompt(messages)
        schema = self._schema_for_request(tools=tools, response_model=response_model)

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
            return self._structured_error_payload(
                code="provider_routing_failure",
                detail=str(response.error or "unknown_error"),
                response_model=response_model,
            )

        if isinstance(response.content, dict):
            if self._is_response_model_class(response_model):
                return json.dumps(response.content, ensure_ascii=False)
            candidate = response.content.get("response")
            if isinstance(candidate, str) and candidate.strip():
                return candidate
            return json.dumps(response.content, ensure_ascii=False)
        if response.raw_text:
            return response.raw_text
        return self._structured_error_payload(
            code="provider_empty_content",
            detail="no provider content returned",
            response_model=response_model,
        )

    async def acall(
        self,
        messages: str | list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str:
        del callbacks, available_functions, from_task, from_agent

        prompt = self._messages_to_prompt(messages)
        schema = self._schema_for_request(tools=tools, response_model=response_model)

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
        response, _decision = await self.model_router.generate_with_fallback(request=request)
        if not response.ok:
            return self._structured_error_payload(
                code="provider_routing_failure",
                detail=str(response.error or "unknown_error"),
                response_model=response_model,
            )

        if isinstance(response.content, dict):
            if self._is_response_model_class(response_model):
                return json.dumps(response.content, ensure_ascii=False)
            candidate = response.content.get("response")
            if isinstance(candidate, str) and candidate.strip():
                return candidate
            return json.dumps(response.content, ensure_ascii=False)
        if response.raw_text:
            return response.raw_text
        return self._structured_error_payload(
            code="provider_empty_content",
            detail="no provider content returned",
            response_model=response_model,
        )

    def _structured_error_payload(
        self,
        *,
        code: str,
        detail: str,
        response_model: type[BaseModel] | None = None,
    ) -> str:
        model_payload = self._model_compatible_error_payload(response_model=response_model)
        if model_payload is not None:
            return model_payload

        code_value = (code or "unknown_error").strip() or "unknown_error"
        detail_value = (detail or "unknown_error").strip() or "unknown_error"
        if len(detail_value) > 240:
            detail_value = f"{detail_value[:237]}..."
        payload = {
            "response": "",
            "notes": [f"{code_value}:{detail_value}"],
            # CrewAI planning can still request plain-text output, then parse it later
            # into PlannerTaskPydanticOutput. Keep this compatibility field present on
            # all structured error payloads so planning fallback paths never explode on
            # missing `list_of_plans_per_task` during model re-validation.
            "list_of_plans_per_task": [],
            "error": {
                "code": code_value,
                "detail": detail_value,
            },
        }
        return json.dumps(payload, ensure_ascii=False)

    def _schema_for_request(
        self,
        *,
        tools: list[dict[str, Any]] | None,
        response_model: type[BaseModel] | None,
    ) -> dict[str, object]:
        if self._is_response_model_class(response_model):
            try:
                schema = response_model.model_json_schema()
                if isinstance(schema, dict) and schema:
                    return schema
            except Exception:
                logger.exception(
                    "crewai_response_model_schema_failed",
                    extra={
                        "event": "crewai_response_model_schema_failed",
                        "response_model": getattr(response_model, "__name__", str(response_model)),
                    },
                )

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
        return schema

    def _model_compatible_error_payload(self, *, response_model: type[BaseModel] | None) -> str | None:
        if not self._is_response_model_class(response_model):
            return None
        try:
            payload = self._placeholder_for_model(response_model=response_model)
            validated = response_model.model_validate(payload)
            return validated.model_dump_json(exclude_none=False)
        except Exception:
            logger.exception(
                "crewai_error_payload_model_fallback_failed",
                extra={
                    "event": "crewai_error_payload_model_fallback_failed",
                    "response_model": getattr(response_model, "__name__", str(response_model)),
                },
            )
            return None

    def _placeholder_for_model(self, *, response_model: type[BaseModel], depth: int = 0) -> dict[str, Any]:
        if depth > 4:
            return {}
        payload: dict[str, Any] = {}
        for field_name, field in response_model.model_fields.items():
            if not field.is_required():
                continue
            payload[field_name] = self._placeholder_for_annotation(field.annotation, depth=depth + 1)
        return payload

    def _placeholder_for_annotation(self, annotation: Any, *, depth: int = 0) -> Any:
        if depth > 4:
            return None

        origin = get_origin(annotation)
        if origin is None:
            if annotation in (str, Any):
                return ""
            if annotation is bool:
                return False
            if annotation is int:
                return 0
            if annotation is float:
                return 0.0
            if annotation in (dict,):
                return {}
            if annotation in (list, tuple, set, frozenset):
                return []
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                return self._placeholder_for_model(response_model=annotation, depth=depth + 1)
            return None

        if origin in (list, tuple, set, frozenset):
            return []
        if origin in (dict,):
            return {}

        args = [item for item in get_args(annotation) if item is not type(None)]
        if args:
            first = args[0]
            if isinstance(first, (str, int, float, bool)):
                return first
            return self._placeholder_for_annotation(first, depth=depth + 1)

        return None

    def _is_response_model_class(self, response_model: type[BaseModel] | None) -> bool:
        return isinstance(response_model, type) and issubclass(response_model, BaseModel)

    def supports_function_calling(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 8192

    def _messages_to_prompt(self, messages: str | list[dict[str, Any]]) -> str:
        if isinstance(messages, str):
            return self._compact_prompt(messages)
        if len(messages) > self.max_message_count:
            messages = messages[-self.max_message_count :]
        parts: list[str] = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "user")
            content = str(item.get("content") or "")
            parts.append(f"[{role}] {content}")
        return self._compact_prompt("\n".join(parts))

    def _compact_prompt(self, prompt: str) -> str:
        text = (prompt or "").strip()
        if len(text) <= self.max_prompt_chars:
            return text

        marker = "\n...[truncated for provider budget]...\n"
        tail_budget = max(0, int(self.max_prompt_chars * 0.75) - len(marker))
        head_budget = max(0, self.max_prompt_chars - tail_budget - len(marker))
        if head_budget == 0:
            return text[-self.max_prompt_chars :]
        return f"{text[:head_budget]}{marker}{text[-tail_budget:]}"

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
