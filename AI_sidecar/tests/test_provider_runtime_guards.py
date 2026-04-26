from __future__ import annotations

from ai_sidecar.config import settings
from ai_sidecar.lifecycle import (
    _provider_registration_viability,
    _sanitize_provider_policy_rules,
    create_runtime,
)


def test_provider_registration_viability_requires_api_key_for_openai_and_deepseek(monkeypatch) -> None:
    monkeypatch.setattr(settings, "provider_openai_api_key", "")
    monkeypatch.setattr(settings, "provider_openai_base_url", "https://api.openai.com/v1")
    monkeypatch.setattr(settings, "provider_openai_default_model", "gpt-4o-mini")

    monkeypatch.setattr(settings, "provider_deepseek_api_key", "")
    monkeypatch.setattr(settings, "provider_deepseek_base_url", "https://api.deepseek.com/v1")
    monkeypatch.setattr(settings, "provider_deepseek_default_model", "deepseek-chat")

    openai_usable, openai_reason = _provider_registration_viability("openai")
    deepseek_usable, deepseek_reason = _provider_registration_viability("deepseek")

    assert openai_usable is False
    assert openai_reason == "api_key_missing"
    assert deepseek_usable is False
    assert deepseek_reason == "api_key_missing"


def test_provider_policy_sanitize_does_not_widen_when_default_fallback_disabled() -> None:
    rules = {
        "strategic_planning": {
            "providers": ["openai"],
            "models": {"openai": "gpt-4o-mini"},
        }
    }

    sanitized = _sanitize_provider_policy_rules(
        rules,
        available_providers={"ollama", "deepseek"},
        allow_default_fallback=False,
    )

    assert sanitized["strategic_planning"]["providers"] == []
    assert sanitized["strategic_planning"]["models"] == {}


def test_runtime_provider_registration_skips_enabled_but_unusable_openai(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings, "sqlite_path", str(tmp_path / "sidecar.sqlite"))
    monkeypatch.setattr(settings, "memory_openmemory_path", str(tmp_path / "openmemory.sqlite"))

    monkeypatch.setattr(settings, "provider_ollama_enabled", True)
    monkeypatch.setattr(settings, "provider_ollama_base_url", "http://127.0.0.1:11434")
    monkeypatch.setattr(settings, "provider_ollama_default_model", "qwen3.6:35b-a3b-q4_K_M")

    monkeypatch.setattr(settings, "provider_openai_enabled", True)
    monkeypatch.setattr(settings, "provider_openai_api_key", "")
    monkeypatch.setattr(settings, "provider_openai_base_url", "https://api.openai.com/v1")
    monkeypatch.setattr(settings, "provider_openai_default_model", "gpt-4o-mini")

    monkeypatch.setattr(settings, "provider_deepseek_enabled", False)
    monkeypatch.setattr(settings, "provider_policy_json", "")

    runtime = create_runtime()
    route = runtime.provider_route(type("Payload", (), {"workload": "strategic_planning"})())

    assert route.ok is True
    assert route.selected_provider == "ollama"
    assert "openai" not in route.fallback_chain

