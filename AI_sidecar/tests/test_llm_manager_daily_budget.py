"""Regression tests for ZERO_INTERVENTION sweep: LLMManager daily token budget.

The daily gate read `_daily_tokens` but it was NEVER incremented (only the hourly
list was tracked), so the daily cap never tripped — a dead gate on the conscious-
tier advisory LLM path. Now _record_usage() increments it on each successful
completion and _rollover_daily() resets it per 24h (was never reset -> once tripped
it stayed tripped until process restart).
"""

from __future__ import annotations

import time

from ai_sidecar.llm.config import LLMConfig
from ai_sidecar.llm.manager import LLMManager


def _manager(daily: int) -> LLMManager:
    cfg = LLMConfig()
    cfg.daily_budget_tokens = daily
    cfg.max_calls_per_hour = 100
    m = LLMManager(config=cfg)
    m._daily_tokens = 0
    m._daily_start_ts = 0.0
    return m


def test_record_usage_increments_daily_tokens() -> None:
    m = _manager(daily=1_000_000)
    m._record_usage("a" * 4000, "b" * 4000)  # ~2000 tokens
    assert m._daily_tokens > 0
    assert m._daily_tokens <= 2000  # prompt+completion estimate


def test_daily_gate_trips_when_over() -> None:
    m = _manager(daily=1000)
    m._record_usage("a" * 4000, "b" * 4000)  # ~2000 > 1000
    allowed = m._check_daily_budget(estimated_tokens=0)
    assert allowed is False


def test_daily_gate_allows_under() -> None:
    m = _manager(daily=1_000_000)
    m._record_usage("a" * 400, "b" * 400)  # ~200 << budget
    assert m._check_daily_budget(estimated_tokens=0) is True


def test_daily_rollover_resets() -> None:
    m = _manager(daily=1000)
    m._record_usage("a" * 4000, "b" * 4000)  # trip the gate
    assert m._check_daily_budget(estimated_tokens=0) is False
    # Simulate 24h rollover.
    m._daily_start_ts = time.time() - 90000
    m._rollover_daily()
    assert m._daily_tokens == 0
    assert m._check_daily_budget(estimated_tokens=0) is True
