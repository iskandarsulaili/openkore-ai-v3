"""Provider configuration from environment / settings.

Reads LLM_* environment variables by default. Falls back to the
existing SidecarSettings for seamless integration with the sidecar.
"""
from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ProviderEndpoint(BaseModel):
    """Connection details for one LLM provider."""

    api_key: str = ""
    base_url: str = ""
    default_model: str = ""
    timeout_seconds: float = 30.0
    max_retries: int = 2
    requests_per_minute: int = 30
    enabled: bool = True
    extra_headers: dict[str, str] = Field(default_factory=dict)


class ProviderConfig(BaseModel):
    """Configuration for the full LLM provider chain."""

    # Ordered list of provider names to try (fallback chain)
    provider_chain: list[str] = Field(
        default_factory=lambda: ["openai", "deepseek", "anthropic"]
    )

    # Per-provider endpoint configs
    openai: ProviderEndpoint = Field(
        default_factory=lambda: ProviderEndpoint(
            base_url=os.getenv("LLM_OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("LLM_OPENAI_API_KEY", ""),
            default_model=os.getenv("LLM_OPENAI_MODEL", "gpt-4o-mini"),
            timeout_seconds=float(os.getenv("LLM_OPENAI_TIMEOUT", "30.0")),
            max_retries=int(os.getenv("LLM_OPENAI_MAX_RETRIES", "2")),
            requests_per_minute=int(os.getenv("LLM_OPENAI_RPM", "60")),
            enabled=os.getenv("LLM_OPENAI_ENABLED", "true").lower() == "true",
        )
    )

    azure: ProviderEndpoint = Field(
        default_factory=lambda: ProviderEndpoint(
            base_url=os.getenv("LLM_AZURE_BASE_URL", ""),
            api_key=os.getenv("LLM_AZURE_API_KEY", ""),
            default_model=os.getenv("LLM_AZURE_MODEL", "gpt-4o-mini"),
            timeout_seconds=float(os.getenv("LLM_AZURE_TIMEOUT", "30.0")),
            max_retries=int(os.getenv("LLM_AZURE_MAX_RETRIES", "2")),
            requests_per_minute=int(os.getenv("LLM_AZURE_RPM", "30")),
            enabled=os.getenv("LLM_AZURE_ENABLED", "true").lower() == "true",
            extra_headers={"api-key": os.getenv("LLM_AZURE_API_KEY", "")},
        )
    )

    deepseek: ProviderEndpoint = Field(
        default_factory=lambda: ProviderEndpoint(
            base_url=os.getenv("LLM_DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            api_key=os.getenv("LLM_DEEPSEEK_API_KEY", ""),
            default_model=os.getenv("LLM_DEEPSEEK_MODEL", "deepseek-chat"),
            timeout_seconds=float(os.getenv("LLM_DEEPSEEK_TIMEOUT", "30.0")),
            max_retries=int(os.getenv("LLM_DEEPSEEK_MAX_RETRIES", "2")),
            requests_per_minute=int(os.getenv("LLM_DEEPSEEK_RPM", "60")),
            enabled=os.getenv("LLM_DEEPSEEK_ENABLED", "true").lower() == "true",
        )
    )

    anthropic: ProviderEndpoint = Field(
        default_factory=lambda: ProviderEndpoint(
            base_url=os.getenv("LLM_ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"),
            api_key=os.getenv("LLM_ANTHROPIC_API_KEY", ""),
            default_model=os.getenv("LLM_ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
            timeout_seconds=float(os.getenv("LLM_ANTHROPIC_TIMEOUT", "30.0")),
            max_retries=int(os.getenv("LLM_ANTHROPIC_MAX_RETRIES", "2")),
            requests_per_minute=int(os.getenv("LLM_ANTHROPIC_RPM", "30")),
            enabled=os.getenv("LLM_ANTHROPIC_ENABLED", "true").lower() == "true",
            extra_headers={
                "anthropic-version": os.getenv("LLM_ANTHROPIC_VERSION", "2023-06-01"),
            },
        )
    )

    # Global defaults
    default_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    default_max_tokens: int = Field(default=2048, ge=1, le=128_000)
    global_timeout_seconds: float = Field(default=45.0, ge=1.0, le=600.0)
    global_max_retries: int = Field(default=2, ge=0, le=8)
    global_rpm: int = Field(default=30, ge=1, le=1000)

    @field_validator("provider_chain", mode="before")
    @classmethod
    def _parse_provider_chain(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [p.strip().lower() for p in v.split(",") if p.strip()]
        if isinstance(v, list):
            return [p.strip().lower() for p in v if isinstance(p, str) and p.strip()]
        return ["openai", "deepseek", "anthropic"]

    @classmethod
    def from_env(cls) -> "ProviderConfig":
        """Build config from LLM_* environment variables."""
        chain_raw = os.getenv("LLM_PROVIDER_CHAIN", "")
        providers = cls()
        if chain_raw:
            providers.provider_chain = cls._parse_provider_chain(chain_raw)
        return providers

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProviderConfig":
        """Build config from a dictionary (e.g., from SidecarSettings)."""
        return cls.model_validate(data)


class LLMConfig(BaseModel):
    """Top-level LLM configuration consumed by LLMManager."""

    providers: ProviderConfig = Field(default_factory=ProviderConfig.from_env)
    temperature: float = Field(
        default=float(os.getenv("LLM_TEMPERATURE", "0.7")), ge=0.0, le=2.0
    )
    max_tokens: int = Field(
        default=int(os.getenv("LLM_MAX_TOKENS", "2048")), ge=1, le=128_000
    )
    system_prompt: str = Field(
        default=os.getenv("LLM_SYSTEM_PROMPT", "You are a helpful assistant.")
    )
    # Cost control
    cost_tier: str = Field(
        default=os.getenv("LLM_COST_TIER", "standard")
    )  # off | saver | standard | max
    daily_budget_tokens: int = Field(
        default=int(os.getenv("LLM_DAILY_BUDGET_TOKENS", "100000")), ge=0
    )
    max_calls_per_hour: int = Field(
        default=int(os.getenv("LLM_MAX_CALLS_PER_HOUR", "100")), ge=0
    )

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls()

    def get_provider(self, name: str) -> ProviderEndpoint | None:
        """Get a provider endpoint config by name."""
        return getattr(self.providers, name.lower(), None)


# Singleton config (lazy-loaded)
_config: LLMConfig | None = None


def get_llm_config() -> LLMConfig:
    """Get the global LLM config, building from env on first call."""
    global _config
    if _config is None:
        _config = LLMConfig.from_env()
    return _config


def reload_llm_config() -> LLMConfig:
    """Reload config from environment (for dynamic updates)."""
    global _config
    _config = LLMConfig.from_env()
    return _config
