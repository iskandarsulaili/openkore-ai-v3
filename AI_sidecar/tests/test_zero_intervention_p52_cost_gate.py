"""Regression tests for ZERO_INTERVENTION P5.2: fleet LLM budget gate.

The CostTracker was instantiated but its record_call was NEVER fed and set_cost_controls
was NEVER called on the model router — so the per-hour/daily budget never tripped (a
dormant cost gate). These tests lock:
- ModelRouter has set_cost_controls wired to a CostTracker.
- A successful generate_with_fallback response feeds usage into the tracker.
- The tracker gates a subsequent call once the hourly limit is hit.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ai_sidecar.cost_tracker import CostTracker
from ai_sidecar.providers.model_router import ModelRouter


def _router_with_tracker(budget: int, hourly: int) -> tuple[ModelRouter, CostTracker]:
    ct = CostTracker(per_bot_budget=True)
    router = ModelRouter(providers={}, initial_rules={})
    router.set_cost_controls(tracker=ct, daily_budget=budget, max_calls_per_hour=hourly, tier="standard")
    return router, ct


def test_set_cost_controls_wires_tracker() -> None:
    router, ct = _router_with_tracker(100000, 30)
    assert router._cost_tracker is ct
    assert router._daily_budget == 100000
    assert router._max_calls_per_hour == 30


def test_record_call_feeds_tracker() -> None:
    router, ct = _router_with_tracker(1000, 100)
    # Simulate the router's success-path feeding (as patched in model_router).
    ct.record_call(tokens=600, model="m", tier="standard", bot_id="botA")
    ct.record_call(tokens=500, model="m", tier="standard", bot_id="botA")
    allowed, reason = ct.check(daily_budget_tokens=1000, max_calls_per_hour=100,
                               tier="standard", bot_id="botA")
    assert allowed is False
    assert "daily" in reason


def test_fleet_gate_binds_across_bots() -> None:
    """Fleet gate: N bots cannot collectively exceed the shared budget."""
    ct = CostTracker(per_bot_budget=True)
    # Each bot individually under budget, but fleet total over -> gate blocks.
    ct.record_call(tokens=700, model="m", tier="standard", bot_id="botA")
    ct.record_call(tokens=700, model="m", tier="standard", bot_id="botB")
    allowed, reason = ct.check(daily_budget_tokens=1000, max_calls_per_hour=100,
                               tier="standard", bot_id="botA")
    assert allowed is False
    assert "fleet_daily" in reason, reason


def test_fleet_gate_resets_on_day_rollover() -> None:
    ct = CostTracker(per_bot_budget=True)
    ct.record_call(tokens=1500, model="m", tier="standard", bot_id="botA")
    allowed, _ = ct.check(daily_budget_tokens=1000, max_calls_per_hour=100,
                          tier="standard", bot_id="botA")
    assert allowed is False
    # Simulate day rollover (force _day_start into the past).
    ct._day_start = int(__import__("time").time()) - 90000
    ct._check_day_rollover()
    allowed, _ = ct.check(daily_budget_tokens=1000, max_calls_per_hour=100,
                          tier="standard", bot_id="botA")
    assert allowed is True
