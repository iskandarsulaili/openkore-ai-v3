"""DeepSeek provider — uses the OpenAI-compatible DeepSeek API."""
from __future__ import annotations

import logging
from typing import Any

import httpx

from ai_sidecar.llm.providers import (
    BaseLLMProvider,
    LLMProviderError,
)

logger = logging.getLogger(__name__)


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek chat completions provider.

    DeepSeek uses an OpenAI-compatible API format. Supports
    deepseek-chat and deepseek-reasoner models.
    """

    @property
    def name(self) -> str:
        return "deepseek"

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

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            **self._extra_headers,
        }

        url = f"{self._base_url}/chat/completions"

        last_error: str | None = None
        for attempt in range(self._max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(url, headers=headers, json=payload)

                if response.status_code == 429:
                    logger.warning("DeepSeek rate limited (429), retrying...")
                    last_error = "rate_limited"
                    if attempt < self._max_retries:
                        import asyncio

                        await asyncio.sleep(2.0 * (attempt + 1))
                        continue
                    raise LLMProviderError(
                        "DeepSeek rate limited after retries",
                        provider=self.name,
                        status_code=429,
                    )

                if response.status_code == 401:
                    raise LLMProviderError(
                        "DeepSeek authentication failed — check API key",
                        provider=self.name,
                        status_code=401,
                    )

                response.raise_for_status()
                data = response.json()

                choices = data.get("choices", [])
                if not choices:
                    raise LLMProviderError(
                        "Empty choices in DeepSeek response",
                        provider=self.name,
                    )

                content = choices[0].get("message", {}).get("content", "")
                if content is None:
                    content = ""
                return content.strip()

            except httpx.TimeoutException:
                last_error = "timeout"
                if attempt < self._max_retries:
                    import asyncio

                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                raise LLMProviderError(
                    f"DeepSeek timeout after {self._max_retries + 1} attempts",
                    provider=self.name,
                ) from None

            except httpx.HTTPStatusError as e:
                last_error = f"http_{e.response.status_code}"
                if e.response.status_code in (500, 502, 503, 504) and attempt < self._max_retries:
                    import asyncio

                    await asyncio.sleep(3.0 * (attempt + 1))
                    continue
                raise LLMProviderError(
                    f"DeepSeek HTTP {e.response.status_code}: {e.response.text[:200]}",
                    provider=self.name,
                    status_code=e.response.status_code,
                ) from e

            except Exception as e:
                last_error = f"exception:{type(e).__name__}"
                if attempt < self._max_retries:
                    logger.warning("DeepSeek attempt %d failed: %s", attempt + 1, e)
                    import asyncio

                    await asyncio.sleep(1.0)
                    continue
                raise LLMProviderError(
                    f"DeepSeek request failed: {e}",
                    provider=self.name,
                ) from e

        raise LLMProviderError(
            f"DeepSeek exhausted retries: {last_error}",
            provider=self.name,
        )
