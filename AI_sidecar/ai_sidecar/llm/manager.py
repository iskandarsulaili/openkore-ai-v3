"""LLMManager — provider selection, fallback chain, retry logic, and rate limiting."""
from __future__ import annotations

import logging
import time
from typing import Any

from ai_sidecar.llm.config import LLMConfig, ProviderConfig
from ai_sidecar.llm.providers import BaseLLMProvider, LLMProviderError

logger = logging.getLogger(__name__)

# Registry mapping provider names to implementation classes
_PROVIDER_REGISTRY: dict[str, type[BaseLLMProvider]] = {}


def register_provider(name: str, cls: type[BaseLLMProvider]) -> None:
    """Register a provider class for a given name."""
    _PROVIDER_REGISTRY[name.lower()] = cls


def get_provider_class(name: str) -> type[BaseLLMProvider] | None:
    """Get a registered provider class by name."""
    return _PROVIDER_REGISTRY.get(name.lower())


# Lazy-import and register built-in providers
def _register_builtins() -> None:
    if "openai" not in _PROVIDER_REGISTRY:
        from ai_sidecar.llm.providers.openai import OpenAIProvider

        register_provider("openai", OpenAIProvider)
    if "azure" not in _PROVIDER_REGISTRY:
        from ai_sidecar.llm.providers.azure import AzureProvider

        register_provider("azure", AzureProvider)
    if "deepseek" not in _PROVIDER_REGISTRY:
        from ai_sidecar.llm.providers.deepseek import DeepSeekProvider

        register_provider("deepseek", DeepSeekProvider)
    if "anthropic" not in _PROVIDER_REGISTRY:
        from ai_sidecar.llm.providers.anthropic import AnthropicProvider

        register_provider("anthropic", AnthropicProvider)


_register_builtins()


class LLMManager:
    """Orchestrates LLM providers with fallback chain, retry, and rate limiting.

    Usage:
        manager = LLMManager()
        result = await manager.complete("Explain quantum computing")
        result = await manager.complete_json("{\\"role\\": ...}", system_prompt="...")
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        provider_overrides: dict[str, BaseLLMProvider] | None = None,
    ) -> None:
        self._config = config or LLMConfig.from_env()
        self._providers: dict[str, BaseLLMProvider] = {}

        if provider_overrides:
            self._providers.update(provider_overrides)
        else:
            self._init_providers()

        # Track which providers are available (after health checks)
        self._available: set[str] = set(self._providers.keys())

        # Global rate limiter — calls per hour across all providers
        self._hourly_calls: list[float] = []
        self._daily_tokens: int = 0

    def _init_providers(self) -> None:
        """Instantiate configured providers from config."""
        provider_configs: ProviderConfig = self._config.providers
        chain = provider_configs.provider_chain

        for name in chain:
            if name in self._providers:
                continue  # already loaded (override)

            endpoint = self._config.get_provider_endpoint(name)
            if endpoint is None or not endpoint.enabled:
                logger.info("Provider '%s' not configured or disabled, skipping", name)
                continue

            cls = get_provider_class(name)
            if cls is None:
                logger.warning("Unknown provider '%s', skipping", name)
                continue

            # Build config dict from endpoint + global defaults
            provider_cfg: dict[str, Any] = {
                "api_key": endpoint.api_key,
                "base_url": endpoint.base_url,
                "default_model": endpoint.default_model,
                "timeout_seconds": endpoint.timeout_seconds or self._config.providers.global_timeout_seconds,
                "max_retries": endpoint.max_retries or self._config.providers.global_max_retries,
                "requests_per_minute": endpoint.requests_per_minute or self._config.providers.global_rpm,
                "enabled": endpoint.enabled,
                "extra_headers": dict(endpoint.extra_headers),
            }

            # Check for required config
            if not provider_cfg["api_key"]:
                logger.warning(
                    "Provider '%s' has no API key configured, marking disabled",
                    name,
                )
                provider_cfg["enabled"] = False

            try:
                instance = cls(config=provider_cfg)
                self._providers[name] = instance
                if instance.enabled:
                    logger.info(
                        "Registered provider '%s' (model=%s, url=%s)",
                        name,
                        instance.model,
                        endpoint.base_url,
                    )
            except Exception as e:
                logger.error("Failed to instantiate provider '%s': %s", name, e)

        if not self._providers:
            logger.warning("No LLM providers configured — all completions will fail")

    @property
    def available_providers(self) -> list[str]:
        """List of provider names currently considered available."""
        return [
            name
            for name, p in self._providers.items()
            if p.enabled and name in self._available
        ]

    @property
    def provider_chain(self) -> list[str]:
        """The ordered fallback chain of provider names."""
        return [
            name
            for name in self._config.providers.provider_chain
            if name in self._providers
        ]

    # ── Global rate limit checks ──

    def _check_hourly_budget(self) -> None:
        """Enforce max_calls_per_hour limit."""
        max_calls = self._config.max_calls_per_hour
        if max_calls <= 0:
            return
        now = time.monotonic()
        window_start = now - 3600.0
        self._hourly_calls = [t for t in self._hourly_calls if t > window_start]
        if len(self._hourly_calls) >= max_calls:
            sleep_for = (self._hourly_calls[0] + 3600.0) - now
            if sleep_for > 0:
                logger.warning(
                    "Hourly call limit reached (%d/%d), sleeping %.1fs",
                    len(self._hourly_calls),
                    max_calls,
                    sleep_for,
                )
                # Clamp sleep to 60s max — after that we let calls through.
                # Use a plain blocking time.sleep (this is a synchronous rate-limiter
                # budget check); asyncio.get_event_loop().run_until_complete(asyncio.sleep)
                # would raise "Cannot run the event loop while another loop is running"
                # when called from within the sidecar's running event loop.
                time.sleep(min(sleep_for, 60.0))
        self._hourly_calls.append(now)

    def _check_daily_budget(self, estimated_tokens: int = 0) -> bool:
        """Check daily token budget. Returns False if budget exceeded."""
        budget = self._config.daily_budget_tokens
        if budget <= 0:
            return True
        if self._daily_tokens + estimated_tokens > budget:
            logger.warning(
                "Daily token budget exceeded (%d/%d), skipping LLM call",
                self._daily_tokens,
                budget,
            )
            return False
        return True

    # ── Public API ──

    async def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        *,
        preferred_provider: str | None = None,
        fallback: bool = True,
    ) -> str:
        """Generate text with automatic fallback across providers.

        Args:
            prompt: The user message / prompt.
            system_prompt: System-level instruction.
            temperature: Sampling temperature (default from config).
            max_tokens: Max tokens (default from config).
            preferred_provider: If set, try this provider first.
            fallback: If True, fall back through provider chain on failure.

        Returns:
            Generated text.

        Raises:
            LLMProviderError: If all providers fail.
        """
        self._check_hourly_budget()
        if not self._check_daily_budget(estimated_tokens=len(prompt) // 4):
            raise LLMProviderError("Daily token budget exceeded", provider="manager")

        temperature = temperature if temperature is not None else self._config.temperature
        max_tokens = max_tokens if max_tokens is not None else self._config.max_tokens

        if not self._providers:
            raise LLMProviderError("No LLM providers configured", provider="manager")

        # Build ordered provider list
        chain = list(self.provider_chain)
        if preferred_provider and preferred_provider in chain:
            chain.remove(preferred_provider)
            chain.insert(0, preferred_provider)

        errors: list[tuple[str, str]] = []
        for provider_name in chain:
            provider = self._providers.get(provider_name)
            if provider is None or not provider.enabled:
                continue

            try:
                result = await provider.complete(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                logger.info(
                    "LLM completion succeeded via '%s' (model=%s)",
                    provider_name,
                    provider.model,
                )
                return result
            except LLMProviderError as e:
                logger.warning(
                    "Provider '%s' failed: %s", provider_name, e
                )
                errors.append((provider_name, str(e)))
                if not fallback:
                    raise
                continue
            except Exception as e:
                logger.error(
                    "Unexpected error from provider '%s': %s",
                    provider_name,
                    e,
                )
                errors.append((provider_name, f"unexpected:{type(e).__name__}:{e}"))
                if not fallback:
                    raise LLMProviderError(str(e), provider=provider_name) from e
                continue

        raise LLMProviderError(
            f"All providers failed: {errors}",
            provider="manager",
        )

    async def complete_json(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        *,
        preferred_provider: str | None = None,
        fallback: bool = True,
    ) -> dict[str, Any]:
        """Generate structured JSON with automatic fallback across providers.

        Args:
            prompt: The user message / prompt.
            system_prompt: System-level instruction with schema guidance.
            temperature: Sampling temperature (default 0.2 for deterministic output).
            max_tokens: Max tokens (default 4096 for structured output).
            preferred_provider: If set, try this provider first.
            fallback: If True, fall back through provider chain on failure.

        Returns:
            Parsed JSON dictionary.

        Raises:
            LLMProviderError: If all providers fail or JSON parsing fails.
        """
        self._check_hourly_budget()
        if not self._check_daily_budget(estimated_tokens=len(prompt) // 4):
            raise LLMProviderError("Daily token budget exceeded", provider="manager")

        temperature = temperature if temperature is not None else 0.2
        max_tokens = max_tokens if max_tokens is not None else 4096

        if not self._providers:
            raise LLMProviderError("No LLM providers configured", provider="manager")

        chain = list(self.provider_chain)
        if preferred_provider and preferred_provider in chain:
            chain.remove(preferred_provider)
            chain.insert(0, preferred_provider)

        errors: list[tuple[str, str]] = []
        for provider_name in chain:
            provider = self._providers.get(provider_name)
            if provider is None or not provider.enabled:
                continue

            try:
                result = await provider.complete_json(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                logger.info(
                    "LLM JSON completion succeeded via '%s' (model=%s)",
                    provider_name,
                    provider.model,
                )
                return result
            except LLMProviderError as e:
                logger.warning(
                    "Provider '%s' JSON completion failed: %s",
                    provider_name,
                    e,
                )
                errors.append((provider_name, str(e)))
                if not fallback:
                    raise
                continue
            except Exception as e:
                logger.error(
                    "Unexpected error from provider '%s' (JSON): %s",
                    provider_name,
                    e,
                )
                errors.append((provider_name, f"unexpected:{type(e).__name__}:{e}"))
                if not fallback:
                    raise LLMProviderError(str(e), provider=provider_name) from e
                continue

        raise LLMProviderError(
            f"All providers failed for JSON completion: {errors}",
            provider="manager",
        )

    # ── Health & Status ──

    def get_provider(self, name: str) -> BaseLLMProvider | None:
        """Get a provider instance by name."""
        return self._providers.get(name.lower())

    def is_available(self) -> bool:
        """Check if at least one provider is configured and enabled."""
        return any(
            p.enabled for p in self._providers.values()
        )

    def status(self) -> dict[str, Any]:
        """Return a detailed status dictionary for all providers."""
        return {
            "available": self.is_available(),
            "provider_count": len(self._providers),
            "provider_chain": self.provider_chain,
            "providers": {
                name: {
                    "enabled": p.enabled,
                    "model": p.model,
                    "name": p.name,
                    "available": name in self._available,
                }
                for name, p in self._providers.items()
            },
            "config": {
                "temperature": self._config.temperature,
                "max_tokens": self._config.max_tokens,
                "cost_tier": self._config.cost_tier,
                "max_calls_per_hour": self._config.max_calls_per_hour,
                "daily_budget_tokens": self._config.daily_budget_tokens,
            },
        }

    def mark_unavailable(self, provider_name: str) -> None:
        """Mark a provider as temporarily unavailable."""
        if provider_name in self._available:
            self._available.remove(provider_name)
            logger.warning("Provider '%s' marked unavailable", provider_name)

    def mark_available(self, provider_name: str) -> None:
        """Mark a provider as available again."""
        if provider_name in self._providers:
            self._available.add(provider_name)
            logger.info("Provider '%s' marked available", provider_name)

    def update_config(self, config: LLMConfig) -> None:
        """Update configuration at runtime. Re-initializes providers on change."""
        self._config = config
        self._providers.clear()
        self._available.clear()
        self._init_providers()
        self._available = set(self._providers.keys())
