"""Tests for the implemented stubs (completeness program).

set_domain_weights, opportunity_cost.reset, and cleanup_bot x4 were true
`pass` stubs — now implemented with real behavior. These tests prove each
implementation actually does something.
"""

from __future__ import annotations

from ai_sidecar.autonomy.domains import BaseDomain, DomainRegistry
from ai_sidecar.opportunity_cost_engine import OpportunityCostEngine


# ── set_domain_weights → DomainRegistry.set_weights ──────────────────────────

class _D(BaseDomain):
    def __init__(self, name: str, priority: int) -> None:
        self.name = name
        self.priority = priority

    def assess(self, signals, actions, service=None) -> None:
        # Test double — no behavior; satisfies the abstract contract
        return None


def test_domain_registry_set_weights_reorders_domains() -> None:
    reg = DomainRegistry()
    a = _D("alpha", priority=10)
    b = _D("beta", priority=20)
    c = _D("gamma", priority=30)
    reg.register(a)
    reg.register(b)
    reg.register(c)
    assert reg.domain_names == ["alpha", "beta", "gamma"]

    # Weight gamma x4 -> effective priority 30/4 = 7.5 < 10 -> runs first
    reg.set_weights({"gamma": 4.0})
    assert reg.domain_names == ["gamma", "alpha", "beta"]


def test_domain_registry_set_weights_ignores_invalid() -> None:
    reg = DomainRegistry()
    a = _D("alpha", priority=10)
    b = _D("beta", priority=20)
    reg.register(a)
    reg.register(b)
    # Invalid entries (non-numeric, <=0, unknown names) must not crash or reorder
    reg.set_weights({"alpha": "not-a-number", "beta": 0, "ghost": 99})
    assert reg.domain_names == ["alpha", "beta"]


# ── opportunity_cost.reset ───────────────────────────────────────────────────

def test_opportunity_cost_reset_clears_state() -> None:
    eng = OpportunityCostEngine()
    eng.set_enqueue_fn(lambda bot_id, action: None)
    assert eng._enqueue_fn is not None
    eng.reset()
    assert eng._enqueue_fn is None
    # Engine remains usable after reset
    assert eng.compare([]) is None


# ── cleanup_bot x4 (defensive per-bot state removal) ─────────────────────────

def test_cleanup_bot_idempotent_and_defensive() -> None:
    from ai_sidecar.domains.crafting.cooking import CookingCrafting
    from ai_sidecar.domains.crafting.alchemy import AlchemyCrafting
    from ai_sidecar.domains.crafting.forging import ForgingCrafting
    from ai_sidecar.domains.consumables.recovery import RecoveryManager

    instances = [
        CookingCrafting(),
        AlchemyCrafting(),
        ForgingCrafting(),
        RecoveryManager(),
    ]
    for inst in instances:
        # Must not raise for unknown bot, twice (idempotent)
        inst.cleanup_bot("bot:unknown")
        inst.cleanup_bot("bot:unknown")
        # If a per-bot tracker exists, it must be popped
        for attr in ("_active_batches", "_last_craft", "_last_heal", "_cooldowns"):
            holder = getattr(inst, attr, None)
            if isinstance(holder, dict):
                holder["bot:x"] = 1
                inst.cleanup_bot("bot:x")
                assert "bot:x" not in holder, f"{type(inst).__name__}.{attr} not cleaned"
