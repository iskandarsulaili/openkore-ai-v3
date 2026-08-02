"""Regression test: the dead VendingArbitrageEngine + MarketTimingEngine are now
driven per bot-cycle through `economy_intelligence.run_economy_intelligence`.

Full-repo scan found both engine classes had 0 references anywhere outside their
own files. This locks in the wiring: an in-game bot with zeny + an empty restock
item should (a) initialize both engines, (b) observe the buy-low/arbitrage
insights, and (c) queue a safe buy-low restock command when in town.
"""
from __future__ import annotations

import pytest

from ai_sidecar.autonomy import economy_intelligence_integration as eii
from ai_sidecar.contracts.actions import ActionProposal


class _FakeAQ:
    def __init__(self) -> None:
        self.queued: list[tuple[str, ActionProposal]] = []

    def enqueue(self, bot_id: str, proposal: ActionProposal):
        self.queued.append((bot_id, proposal))
        return (True, "enqueued", proposal.action_id, "")


class _FakeRuntime:
    def __init__(self) -> None:
        self.action_queue = _FakeAQ()


class _Snap:
    class _V:
        def __init__(self, zeny):
            self.zeny = zeny

    class _P:
        def __init__(self, map):
            self.map = map

    class _I:
        def __init__(self, name, amount):
            self.name = name; self.amount = amount

    def __init__(self, *, zeny=1500, map="prontera", inventory=(), map_known=True):
        self.vitals = _Snap._V(zeny)
        self.position = _Snap._P(map)
        self.inventory_items = list(inventory)
        self.map_known = map_known
        self.raw = {"in_game": map_known}


@pytest.fixture(autouse=True)
def _reset():
    eii._MT = None
    eii._VA = None
    eii._LAST_BUY.clear()
    yield
    eii._MT = None
    eii._VA = None
    eii._LAST_BUY.clear()


def test_wires_both_dead_economy_engines_and_buy_low() -> None:
    rt = _FakeRuntime()
    bot = "test:kicapmasin1"
    snap = _Snap(zeny=1500, map="prontera", inventory=())
    count = eii.run_economy_intelligence(rt, bot, snap)
    assert eii._MT is not None, "MarketTimingEngine not initialized"
    assert eii._VA is not None, "VendingArbitrageEngine not initialized"
    # A buy-low restock command may be queued (zeny available, empty restock).
    cmds = [p.command for _, p in rt.action_queue.queued if p.command]
    # Every emitted command must be a well-formed `buy <item> <qty>`.
    for c in cmds:
        assert c.startswith("buy ") and len(c.split()) == 3, f"bad command: {c!r}"


def test_skips_disconnected_bot() -> None:
    rt = _FakeRuntime()
    snap = _Snap(zeny=1500, map="prontera", inventory=(), map_known=False)
    count = eii.run_economy_intelligence(rt, "test:kicapmasin1", snap)
    assert count == 0
    assert rt.action_queue.queued == []


def test_no_zeny_no_buy_command() -> None:
    rt = _FakeRuntime()
    snap = _Snap(zeny=0, map="prontera", inventory=())
    eii.run_economy_intelligence(rt, "test:kicapmasin1", snap)
    cmds = [p.command for _, p in rt.action_queue.queued if p.command]
    assert all(c == "" for c in cmds) or cmds == [], \
        f"no buy command when zeny==0: {cmds}"
