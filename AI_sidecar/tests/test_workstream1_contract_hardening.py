from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from ai_sidecar.contracts.common import ContractMeta
from ai_sidecar.planner.context_assembler import PlannerContextAssembler
from ai_sidecar.planner.schemas import PlanHorizon
from ai_sidecar.providers.base import PlannerModelRequest, PlannerModelResponse
from ai_sidecar.providers.model_router import ModelRouter
from ai_sidecar.providers.prompt_guard import PromptGuard


@dataclass(slots=True)
class _DummyProvider:
    provider_name: str
    ok: bool
    model_suffix: str = ""

    async def generate_structured(self, request: PlannerModelRequest) -> PlannerModelResponse:
        model_name = request.model or f"{self.provider_name}-default"
        return PlannerModelResponse(
            ok=self.ok,
            provider=self.provider_name,
            model=model_name + self.model_suffix,
            trace_id=request.trace_id,
            latency_ms=4.2,
            content={"response": "ok"} if self.ok else None,
            raw_text="{}",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            error="boom" if not self.ok else "",
        )

    async def health(self, *, bot_id: str):
        del bot_id
        raise NotImplementedError


def _request() -> PlannerModelRequest:
    return PlannerModelRequest(
        bot_id="bot:w1",
        trace_id="trace-w1",
        task="strategic_planning",
        model="",
        system_prompt="system",
        user_prompt="user",
        schema={"type": "object", "required": ["response"], "properties": {"response": {"type": "string"}}},
        timeout_seconds=5.0,
        max_retries=0,
    )


def test_model_router_reports_actual_fallback_provider_and_metrics() -> None:
    metrics: list[tuple[str, str, str]] = []
    router = ModelRouter(
        providers={
            "ollama": _DummyProvider(provider_name="ollama", ok=False),
            "openai": _DummyProvider(provider_name="openai", ok=True),
        },
        initial_rules={
            "strategic_planning": {
                "providers": ["ollama", "openai"],
                "models": {"ollama": "qwen2.5:14b", "openai": "gpt-4o-mini"},
            }
        },
        route_metric_observer=lambda workload, provider, model: metrics.append((workload, provider, model)),
    )

    response, decision = asyncio.run(router.generate_with_fallback(request=_request()))

    assert response.ok is True
    assert response.provider == "openai"
    assert decision.selected_provider == "openai"
    assert decision.planned_provider == "ollama"
    assert decision.fallback_used is True
    assert decision.attempted_providers == ["ollama", "openai"]
    assert metrics == [("strategic_planning", "openai", "gpt-4o-mini")]


def test_model_router_no_provider_emits_none_metric() -> None:
    metrics: list[tuple[str, str, str]] = []
    router = ModelRouter(
        providers={},
        initial_rules={"strategic_planning": {"providers": [], "models": {}}},
        route_metric_observer=lambda workload, provider, model: metrics.append((workload, provider, model)),
    )
    response, decision = asyncio.run(router.generate_with_fallback(request=_request()))
    assert response.ok is False
    assert decision.selected_provider == "none"
    assert metrics == [("strategic_planning", "none", "")]


def test_prompt_guard_schema_validation_hardening() -> None:
    guard = PromptGuard()
    schema = {
        "type": "object",
        "required": ["response", "risk_score", "steps"],
        "additionalProperties": False,
        "properties": {
            "response": {"type": "string", "minLength": 1, "maxLength": 20},
            "risk_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "steps": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["kind"],
                    "additionalProperties": False,
                    "properties": {
                        "kind": {"type": "string", "enum": ["move", "attack"]},
                    },
                },
            },
        },
    }

    guard.validate_schema(
        {
            "response": "safe",
            "risk_score": 0.5,
            "steps": [{"kind": "move"}],
        },
        schema,
    )

    with pytest.raises(ValueError, match="schema_additional_properties_forbidden"):
        guard.validate_schema(
            {
                "response": "safe",
                "risk_score": 0.5,
                "steps": [{"kind": "move"}],
                "extra": True,
            },
            schema,
        )

    with pytest.raises(ValueError, match="schema_unsafe_key"):
        guard.validate_schema(
            {
                "response": "safe",
                "risk_score": 0.5,
                "steps": [{"kind": "move", "__proto__": "pollute"}],
            },
            {
                "type": "object",
                "required": ["response", "risk_score", "steps"],
                "properties": {
                    "response": {"type": "string"},
                    "risk_score": {"type": "number"},
                    "steps": {
                        "type": "array",
                        "items": {"type": "object", "additionalProperties": True},
                    },
                },
            },
        )


class _RuntimeDegraded:
    def __init__(self) -> None:
        self.action_queue = type("Queue", (), {"count": lambda self, _bot_id: 1})()
        self.latency_router = type("Latency", (), {"average_ms": lambda self: 2.0})()

    def enriched_state(self, *, bot_id: str):
        return type(
            "State",
            (),
            {
                "model_dump": lambda self, mode="json": {
                    "fleet_intent": {
                        "role": "grinder",
                        "assignment": "prt_fild08",
                        "objective": "farm",
                        "constraints": {"config.doctrine_version": "legacy-v1"},
                    }
                }
            },
        )()

    def recent_ingest_events(self, *, bot_id: str, limit: int = 100) -> list[dict[str, object]]:
        return [{"bot_id": bot_id, "limit": limit}]

    def memory_context(self, *, bot_id: str, query: str, limit: int) -> list[dict[str, object]]:
        return [{"bot_id": bot_id, "query": query, "limit": limit}]

    def memory_recent_episodes(self, *, bot_id: str, limit: int) -> list[dict[str, object]]:
        return [{"bot_id": bot_id, "limit": limit}]

    def latest_macro_publication(self, *, bot_id: str) -> dict[str, object]:
        return {"bot_id": bot_id, "version": "macro-v1"}

    def fleet_constraints(self, *, bot_id: str):
        del bot_id
        raise RuntimeError("central unavailable")

    def fleet_blackboard(self, *, bot_id: str):
        del bot_id
        raise RuntimeError("blackboard unavailable")


def test_context_assembler_marks_degradation_when_fleet_calls_fail() -> None:
    assembler = PlannerContextAssembler(runtime=_RuntimeDegraded())
    context = assembler.assemble(
        meta=ContractMeta(contract_version="v1", source="pytest", bot_id="bot:w1", trace_id="trace-w1"),
        objective="farm safely",
        horizon=PlanHorizon.strategic,
        event_limit=8,
        memory_limit=4,
    )
    coordination = context.fleet_constraints["coordination"]
    assert coordination["degraded"] is True
    reasons = coordination["degradation_reasons"]
    assert any(str(item).startswith("fleet_constraints_unavailable") for item in reasons)
    assert any(str(item).startswith("fleet_blackboard_unavailable") for item in reasons)
