"""Base provider class for the LLM integration module."""
from __future__ import annotations

import abc
import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class LLMProviderError(Exception):
    """Base error for LLM provider failures."""

    def __init__(self, message: str, provider: str = "", status_code: int = 0) -> None:
        self.provider = provider
        self.status_code = status_code
        super().__init__(message)


class RateLimitError(LLMProviderError):
    """Raised when rate limited by the provider."""


class TimeoutError(LLMProviderError):
    """Raised when a provider request times out."""


class AuthenticationError(LLMProviderError):
    """Raised when API authentication fails."""


class BaseLLMProvider(abc.ABC):
    """Abstract base class for LLM providers.

    Each provider subclass implements:
      - complete()        — free-form text completion
      - complete_json()   — structured JSON completion
      - name              — provider identifier
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._api_key: str = config.get("api_key", "")
        self._base_url: str = config.get("base_url", "").rstrip("/")
        self._model: str = config.get("default_model", "")
        self._timeout: float = float(config.get("timeout_seconds", 30.0))
        self._max_retries: int = int(config.get("max_retries", 2))
        self._rpm: int = int(config.get("requests_per_minute", 30))
        self._extra_headers: dict[str, str] = config.get("extra_headers", {})
        self._enabled: bool = bool(config.get("enabled", True))

        # Rate limiting state
        self._call_timestamps: list[float] = []
        self._last_rate_limit_warn: float = 0.0

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Provider identifier string (e.g., 'openai', 'anthropic')."""
        ...

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a free-form text completion.

        Args:
            prompt: User message / prompt text.
            system_prompt: Optional system-level instruction.
            temperature: Sampling temperature (0.0–2.0).
            max_tokens: Maximum tokens in the response.

        Returns:
            Generated text content.

        Raises:
            LLMProviderError: On provider failure.
        """
        return await self._completion_request(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def complete_json(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Generate a structured JSON completion.

        By default uses a lower temperature for deterministic output.
        Subclasses may add schema hints to the request.

        Args:
            prompt: User message / prompt text.
            system_prompt: Optional system-level instruction.
            temperature: Sampling temperature (lower = more deterministic).
            max_tokens: Maximum tokens in the response.

        Returns:
            Parsed JSON dictionary.

        Raises:
            LLMProviderError: On provider failure or invalid JSON.
        """
        raw = await self._completion_request(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        return self._parse_json(raw)

    # ── Rate limit helpers ──

    def _check_rate_limit(self) -> None:
        """Enforce requests-per-minute limit. Blocks (sleeps) if exceeded."""
        if self._rpm <= 0:
            return
        now = time.monotonic()
        window_start = now - 60.0
        # Prune old timestamps
        self._call_timestamps = [t for t in self._call_timestamps if t > window_start]
        if len(self._call_timestamps) >= self._rpm:
            # We're at the limit — warn occasionally and sleep
            if now - self._last_rate_limit_warn > 5.0:
                logger.warning(
                    "Rate limit reached for %s (%d calls in last 60s), sleeping",
                    self.name,
                    self._rpm,
                )
                self._last_rate_limit_warn = now
            # Sleep until a slot opens
            sleep_for = (self._call_timestamps[0] + 60.0) - now
            if sleep_for > 0:
                time.sleep(min(sleep_for, 10.0))
        self._call_timestamps.append(time.monotonic())

    # ── Subclass hooks ──

    @abc.abstractmethod
    async def _completion_request(
        self,
        *,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool = False,
    ) -> str:
        """Execute the actual API call. Must be implemented by subclasses."""
        ...

    # ── Shared utilities ──

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        """Parse JSON from a provider response, with recovery attempts."""
        text = raw.strip()
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code blocks
        for marker in ("```json", "```JSON", "```"):
            if marker in text:
                parts = text.split(marker)
                if len(parts) >= 2:
                    candidate = parts[1].split("```")[0].strip()
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        continue

        # Last resort: try to find {...} or [...] block
        import re

        brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        bracket_match = re.search(r"\[[^\[\]]*\]", text, re.DOTALL)
        if bracket_match:
            try:
                result = json.loads(bracket_match.group(0))
                if isinstance(result, list):
                    return {"data": result}
            except json.JSONDecodeError:
                pass

        raise LLMProviderError(
            f"Failed to parse JSON response: {text[:500]}",
            provider=self.name,
        )
