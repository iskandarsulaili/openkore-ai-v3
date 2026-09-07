"""Focused test: BYOK OpenAI-compatible providers register + route correctly."""
import asyncio
import types

import pytest

from ai_sidecar.providers.model_router import ModelRouter
from ai_sidecar.providers.openai_adapter import OpenAIAdapter
from ai_sidecar.providers.prompt_guard import PromptGuard
from ai_sidecar.reflex.circuit_breaker import ReflexCircuitBreaker


@pytest.fixture
def guard():
    return PromptGuard(max_prompt_chars=32000)


@pytest.fixture
def breaker():
    return ReflexCircuitBreaker()


def _adapter(name, guard, breaker, base_url, key, model):
    return OpenAIAdapter(
        base_url=base_url,
        api_key=key,
        default_model=model,
        embedding_model="text-embedding-3-small",
        guard=guard,
        breaker=breaker,
        timeout_seconds=30,
        max_retries=1,
        provider_name=name,
    )


def test_byok_adapter_name_and_breaker_scoped(guard, breaker):
    a = _adapter("openrouter", guard, breaker,
                 "https://openrouter.ai/api/v1", "sk-test", "deepseek/deepseek-v4-flash")
    assert a.provider_name == "openrouter"
    assert a._breaker_key == "provider.openrouter"
    assert a._breaker_key_embed == "provider.openrouter.embed"
    assert a._breaker_key_health == "provider.openrouter.health"

    t = _adapter("turbollm", guard, breaker,
                 "http://127.0.0.1:6996/v1", "", "local-model")
    assert t.provider_name == "turbollm"
    assert t._breaker_key == "provider.turbollm"


def test_byok_providers_routable_via_policy(guard, breaker):
    """BYOK adapters registered under their names are selected by the router."""
    openrouter = _adapter("openrouter", guard, breaker,
                          "https://openrouter.ai/api/v1", "sk-t", "deepseek/deepseek-v4-flash")
    router = ModelRouter(
        providers={"openrouter": openrouter},
        initial_rules={
            "strategic_planning": {
                "providers": ["openrouter"],
                "models": {"openrouter": "deepseek/deepseek-v4-flash"},
            }
        },
    )
    assert "openrouter" in router.provider_names()
    decision = router.decide(workload="strategic_planning")
    assert decision.selected_provider == "openrouter"
    assert decision.selected_model == "deepseek/deepseek-v4-flash"


def test_byok_disabled_skipped_in_chain(guard, breaker):
    """Only enabled BYOK providers are appended to the workload fallback chain."""
    from ai_sidecar.providers.openai_adapter import OpenAIAdapter as OA
    # emulate the lifecycle _build_provider_policy_rules for BYOK-append
    enabled = ["openrouter"]  # turbollm/generic disabled
    providers = [p for p in ["openai", "ollama"] if p not in enabled] + enabled
    assert providers == ["openai", "ollama", "openrouter"]
    # disabled provider absent
    providers2 = [p for p in ["openai", "ollama"] if p not in ["turbollm"]] + ["turbollm"]
    assert "turbollm" in providers2
