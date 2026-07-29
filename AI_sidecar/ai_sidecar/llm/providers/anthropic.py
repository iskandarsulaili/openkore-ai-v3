"""Anthropic/Claude provider — uses Anthropic's Messages API."""
from __future__ import annotations

import json as json_lib
import logging
from typing import Any

import httpx

from ai_sidecar.llm.providers import (
    BaseLLMProvider,
    LLMProviderError,
)

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider.

    Uses the Anthropic Messages API format (v1/messages).
    Authentication is via x-api-key header + anthropic-version.
    """

    @property
    def name(self) -> str:
        return "anthropic"

    async def _completion_request(
        self,
        *,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool = False,
    ) -> str:
        self._check_rate_limit()

        # Anthropic messages format:
        # { model, system, messages: [{role, content}], max_tokens, temperature }
        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }

        if system_prompt:
            payload["system"] = system_prompt

        if json_mode:
            # Anthropic doesn't have native JSON mode — we add instruction to system prompt
            if "system" in payload:
                payload["system"] += "\n\nYou must respond with valid JSON only, no other text."
            else:
                payload["system"] = "You must respond with valid JSON only, no other text."

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": self._extra_headers.get(
                "anthropic-version", "2023-06-01"
            ),
        }

        url = f"{self._base_url}/messages"

        last_error: str | None = None
        for attempt in range(self._max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(url, headers=headers, json=payload)

                if response.status_code == 429:
                    logger.warning("Anthropic rate limited (429), retrying...")
                    last_error = "rate_limited"
                    if attempt < self._max_retries:
                        import asyncio

                        await asyncio.sleep(2.0 * (attempt + 1))
                        continue
                    raise LLMProviderError(
                        "Anthropic rate limited after retries",
                        provider=self.name,
                        status_code=429,
                    )

                if response.status_code == 401:
                    raise LLMProviderError(
                        "Anthropic authentication failed — check API key",
                        provider=self.name,
                        status_code=401,
                    )

                response.raise_for_status()
                data = response.json()

                # Anthropic response format: { content: [{ type: "text", text: "..." }] }
                content_blocks = data.get("content", [])
                if not content_blocks:
                    raise LLMProviderError(
                        "Empty content in Anthropic response",
                        provider=self.name,
                    )

                # Concatenate all text blocks
                parts: list[str] = []
                for block in content_blocks:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            parts.append(text)

                if not parts:
                    raise LLMProviderError(
                        "No text content in Anthropic response",
                        provider=self.name,
                    )

                return "\n".join(parts).strip()

            except httpx.TimeoutException:
                last_error = "timeout"
                if attempt < self._max_retries:
                    import asyncio

                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                raise LLMProviderError(
                    f"Anthropic timeout after {self._max_retries + 1} attempts",
                    provider=self.name,
                ) from None

            except httpx.HTTPStatusError as e:
                last_error = f"http_{e.response.status_code}"
                if e.response.status_code in (500, 502, 503, 504) and attempt < self._max_retries:
                    import asyncio

                    await asyncio.sleep(3.0 * (attempt + 1))
                    continue
                raise LLMProviderError(
                    f"Anthropic HTTP {e.response.status_code}: {e.response.text[:200]}",
                    provider=self.name,
                    status_code=e.response.status_code,
                ) from e

            except Exception as e:
                last_error = f"exception:{type(e).__name__}"
                if attempt < self._max_retries:
                    logger.warning("Anthropic attempt %d failed: %s", attempt + 1, e)
                    import asyncio

                    await asyncio.sleep(1.0)
                    continue
                raise LLMProviderError(
                    f"Anthropic request failed: {e}",
                    provider=self.name,
                ) from e

        raise LLMProviderError(
            f"Anthropic exhausted retries: {last_error}",
            provider=self.name,
        )
