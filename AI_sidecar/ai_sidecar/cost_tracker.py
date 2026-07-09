from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CostSnapshot:
    daily_tokens_used: int = 0
    daily_calls_made: int = 0
    hourly_calls_made: int = 0
    monthly_cost_usd: float = 0.0
    tier: str = "standard"
    budget_exhausted: bool = False
    budget_reason: str = ""


class CostTracker:
    """Tracks LLM usage and enforces budget limits.

    All state is in-memory with optional SQLite persistence.
    The PDCA loop and model router check this before making LLM calls.
    """

    def __init__(self):
        self._lock = RLock()
        self._daily_tokens = 0
        self._daily_calls = 0
        self._hourly_calls: list[float] = []  # timestamps
        self._monthly_cost = 0.0
        self._day_start = int(time.time())

    def record_call(self, tokens: int, *, model: str, tier: str) -> None:
        """Record an LLM call and its token usage."""
        with self._lock:
            self._check_day_rollover()
            self._daily_tokens += tokens
            self._daily_calls += 1
            self._hourly_calls.append(time.time())
            # Purge calls older than 1 hour
            cutoff = time.time() - 3600
            self._hourly_calls = [t for t in self._hourly_calls if t > cutoff]

            # Estimate cost based on tier pricing
            pricing = {"off": 0.0, "economy": 0.15, "standard": 0.30, "premium": 0.60}
            price_per_m = pricing.get(tier, 0.30)
            self._monthly_cost += (tokens / 1_000_000) * price_per_m

    def check(self, *, daily_budget_tokens: int, max_calls_per_hour: int, tier: str) -> tuple[bool, str]:
        """Returns (allowed: bool, reason: str)."""
        with self._lock:
            self._check_day_rollover()

            if tier == "off":
                return False, "cost_tier_off"

            if daily_budget_tokens > 0 and self._daily_tokens >= daily_budget_tokens:
                return False, f"daily_token_budget_exceeded:{self._daily_tokens}/{daily_budget_tokens}"

            if max_calls_per_hour > 0:
                cutoff = time.time() - 3600
                recent = sum(1 for t in self._hourly_calls if t > cutoff)
                if recent >= max_calls_per_hour:
                    return False, f"hourly_call_limit_exceeded:{recent}/{max_calls_per_hour}"

            return True, "ok"

    def snapshot(self) -> CostSnapshot:
        with self._lock:
            self._check_day_rollover()
            cutoff = time.time() - 3600
            hourly = sum(1 for t in self._hourly_calls if t > cutoff)
            allowed, reason = self.check(
                daily_budget_tokens=100000,
                max_calls_per_hour=30,
                tier="standard",
            )
            return CostSnapshot(
                daily_tokens_used=self._daily_tokens,
                daily_calls_made=self._daily_calls,
                hourly_calls_made=hourly,
                monthly_cost_usd=round(self._monthly_cost, 4),
                tier="standard",
                budget_exhausted=not allowed,
                budget_reason=reason,
            )

    def _check_day_rollover(self) -> None:
        now = int(time.time())
        # Rollover after 86400 seconds since last reset
        if now - self._day_start >= 86400:
            self._daily_tokens = 0
            self._daily_calls = 0
            self._day_start = now
            logger.info("cost_tracker: daily budget reset")
