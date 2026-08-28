"""Test the edge-case self-heal chain end-to-end.

Regression for the ORPHANED dispatch: EdgeCaseHandler had 8 complete
handle_* methods + a check_all dispatcher, but (1) periodic_review (the bus
entry) was NEVER called from the PDCA loop, (2) the handler was created with a
WRONG import path (ai_sidecar.edge.* -> ImportError -> never created), and
(3) the bus was wired with edge_handler=None (lifecycle never set it). The
whole edge-case self-heal system silently returned 0 every cycle.

This test locks: bus.periodic_review -> _check_edge_cases -> handler.check_all
-> handle_* -> ActionProposal enqueued.
"""

from __future__ import annotations

from ai_sidecar.integration.bus import IntegrationBus
from ai_sidecar.resilience.edge_case_handler import EdgeCaseHandler


class _FakeQueue:
    def __init__(self) -> None:
        self.items: list = []

    def enqueue(self, bot_id: str, proposal) -> None:
        self.items.append((bot_id, proposal))


class _FakeEconomy:
    def budget_planning(self, **kwargs):
        return {"buy": []}


def _snap(**over):
    s = {
        "position": {"x": 10, "y": 10, "map": "prt_fild05"},
        "map": "prt_fild05",
        "base_level": 5,
        "job": "novice",
        "zeny": 100,
        "inventory_items": [],
        "skill_points": 0,
        "stat_points": 0,
    }
    s.update(over)
    return s


def test_bus_periodic_review_dispatches_unstuck() -> None:
    """A stuck bot (same position across cycles) must get a move action."""
    q = _FakeQueue()
    edge = EdgeCaseHandler(unstuck_timeout_s=1)
    bus = IntegrationBus(
        highfreq_reflex=None,
        learning_loop=None,
        combat_intel=None,
        economy_engine=_FakeEconomy(),
        map_intel=None,
        edge_handler=edge,
    )
    # First call seeds the position tracker; second call (same pos, after
    # timeout) must trigger unstuck.
    n1 = bus.periodic_review("bot:x", _snap(), q)
    import time
    time.sleep(1.1)
    n2 = bus.periodic_review("bot:x", _snap(), q)
    assert n1 == 0, f"first review must not trigger (seeds tracker): {n1}"
    assert n2 >= 1, f"stuck bot must trigger unstuck heal: {n2}"
    cmds = [p.command for _b, p in q.items]
    assert any(c.startswith("move") for c in cmds), f"must emit a move: {cmds}"


def test_bus_periodic_review_dispatches_death_recovery() -> None:
    """3+ consecutive deaths must trigger a safer-zone move."""
    q = _FakeQueue()
    edge = EdgeCaseHandler()
    bus = IntegrationBus(
        highfreq_reflex=None, learning_loop=None, combat_intel=None,
        economy_engine=_FakeEconomy(), map_intel=None, edge_handler=edge,
    )
    for i in range(4):
        n = bus.periodic_review("bot:y", _snap(dead=True, vitals={"hp": 0, "hp_max": 100}), q)
    cmds = [p.command for _b, p in q.items]
    assert any(c.startswith("move") for c in cmds), f"death spiral must move: {cmds}"


def test_bus_periodic_review_dispatches_skill_points() -> None:
    """Unspent skill points must queue an auto-assign."""
    q = _FakeQueue()
    edge = EdgeCaseHandler()
    bus = IntegrationBus(
        highfreq_reflex=None, learning_loop=None, combat_intel=None,
        economy_engine=_FakeEconomy(), map_intel=None, edge_handler=edge,
    )
    n = bus.periodic_review("bot:z", _snap(skill_points=9), q)
    assert n >= 1, f"unspent skill points must trigger: {n}"


def test_wiring_from_pdca_emit_heuristic_actions() -> None:
    """The pdca _emit_heuristic_actions must reach the bus periodic_review."""
    from types import SimpleNamespace
    from ai_sidecar.autonomy.pdca_loop import _emit_heuristic_actions

    q = _FakeQueue()
    edge = EdgeCaseHandler(unstuck_timeout_s=0)
    bus = IntegrationBus(
        highfreq_reflex=None, learning_loop=None, combat_intel=None,
        economy_engine=_FakeEconomy(), map_intel=None, edge_handler=edge,
    )
    # Runtime-shaped object: heuristic_service present (so hs check passes),
    # snapshot_cache returning a dict-like snapshot, integration_bus wired.
    class _Snap:
        def model_dump(self, mode="json"):
            return _snap()

    _assess_ret = SimpleNamespace(actions=[])
    rt = SimpleNamespace(
        heuristic_service=SimpleNamespace(assess=lambda *a, **k: _assess_ret),
        action_queue=q,
        snapshot_cache=SimpleNamespace(get=lambda b: _Snap()),
        integration_bus=bus,
        highfreq_reflex=None,
    )
    # First call seeds; second triggers unstuck (timeout 0).
    _emit_heuristic_actions(rt, "short", "bot:w")
    n = _emit_heuristic_actions(rt, "short", "bot:w")
    assert n >= 1, f"pdca emit must reach edge heals: {n}"
