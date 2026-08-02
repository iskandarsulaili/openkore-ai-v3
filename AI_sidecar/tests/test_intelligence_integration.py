"""Regression test: the three previously-unwired intelligence subsystems are now
wired into the runtime through `intelligence_integration.run_intelligence`.

ConsciousDecisionEngine, PreemptiveIntelligence and ProgressionDriver were each
fully implemented with public `evaluate`/`update_from_snapshot`/`process_decisions`
APIs but the `get_*` singletons were never imported or called anywhere in the tree
(confirmed by full-repo reference scan). This test locks in the wiring:
`run_intelligence(runtime, bot_id, snapshot)` must convert each subsystem's
decisions/actions into real queued `ActionProposal`s on the runtime action queue
(observability for non-executable intents), and gate to in-game bots.
"""
from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import pytest

from ai_sidecar.autonomy import intelligence_integration as ii
from ai_sidecar.contracts.actions import ActionProposal


class _FakeAQ:
    """Minimal in-memory ActionQueue: records every proposal, returns accepted."""

    def __init__(self) -> None:
        self.queued: list[tuple[str, ActionProposal]] = []

    def enqueue(self, bot_id: str, proposal: ActionProposal):
        self.queued.append((bot_id, proposal))
        return (True, "enqueued", proposal.action_id, "")


class _FakeRuntime:
    def __init__(self) -> None:
        self.action_queue = _FakeAQ()


class _Snapshot:
    """Minimal BotStateSnapshot stand-in exposing the fields the subsystems read."""

    def __init__(self, *, map_known=True, vitals=None, position=None,
                 inventory=None, skills=None, stats=None):
        self.map_known = map_known
        self.vitals = vitals
        self.position = position
        self.inventory_items = inventory or []
        self.skills = skills or []
        self.stats = stats
        self.raw = {"in_game": map_known}

    class _V:
        def __init__(self, hp_ratio, base_level, job_name, zeny, weight_ratio, job_level):
            self.hp_ratio = hp_ratio; self.base_level = base_level
            self.job_name = job_name; self.zeny = zeny
            self.weight_ratio = weight_ratio; self.job_level = job_level; self.hp = 100

    class _P:
        def __init__(self, map):
            self.map = map


def _kills(s):
    return s

def _snapshot(**kw) -> _Snapshot:
    return _Snapshot(**kw)


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset module-level singletons so tests start clean."""
    ii._CE = ii._PI = ii._PD = None
    yield
    ii._CE = ii._PI = ii._PD = None


def _novice_snapshot(map_known=True, hp=1.0):
    inv = type("I", (), {"name": "White Potion", "amount": 5})
    sk = type("K", (), {"name": "NV_BASIC"})
    st = type("S", (), {"str": 9, "agi": 9})
    return _snapshot(
        map_known=map_known,
        vitals=_Snapshot._V(hp, 1, "novice", 1500, 0.2, 1),
        position=_Snapshot._P("prt_fild08"),
        inventory=[inv],
        skills=[sk],
        stats=st,
    )


def test_wires_all_three_subsystems_and_queues_actions() -> None:
    rt = _FakeRuntime()
    bot = "test:kicapmasin1"
    snap = _novice_snapshot()
    ii.run_intelligence(rt, bot, snap)
    # Each subsystem was initialized and run.
    assert ii._CE is not None, "ConsciousDecisionEngine not initialized"
    assert ii._PI is not None, "PreemptiveIntelligence not initialized"
    assert ii._PD is not None, "ProgressionDriver not initialized"
    # Decisions reached the queue (or were observed). At minimum, the conscious
    # + progression layers should have produced real commands given the snapshot.
    assert isinstance(rt.action_queue, _FakeAQ)
    cmds = [p.command for _, p in rt.action_queue.queued if p.command]
    # Fresh novice with NV_BASIC + missing NV_FIRSTAID and low restock: expect
    # real, well-formed commands (skill/stats/buy), the same actions the wired
    # runtime would emit.
    assert cmds, f"intelligence should have queued commands: {cmds}"
    assert all(c and not c.endswith(("move", "buy", "skills_add", "stats_add", " ")) for c in cmds), \
        f"no empty-target commands: {cmds}"


def test_skips_disconnected_bot() -> None:
    rt = _FakeRuntime()
    bot = "test:kicapmasin1"
    snap = _novice_snapshot(map_known=False)  # at char-select / disconnected
    count = ii.run_intelligence(rt, bot, snap)
    assert count == 0, "must not emit for a bot that is not in-game"
    assert rt.action_queue.queued == []


def test_observe_only_actions_do_not_emit_bogus_commands() -> None:
    # The ensure_services + _in_game gate + command translation should never
    # produce 'move ' / 'buy ' / 'skills_add ' with an empty target.
    rt = _FakeRuntime()
    bot = "test:kicapmasin1"
    # A malicious/empty snapshot that would produce an empty-target decision
    # must be observed, not emitted as a malformed command.
    snap = _snapshot(
        map_known=True,
        vitals=_Snapshot._V(0.3, 1, "novice", 0, 0.9, 1),
        position=_Snapshot._P("prt_fild08"),
        inventory=[],
        skills=[],
        stats=None,
    )
    ii.run_intelligence(rt, bot, snap)
    for _, p in rt.action_queue.queued:
        assert not p.command.rstrip().endswith((" ", "move", "buy", "skills_add")), \
            f"must not emit empty-target command: {p.command!r}"
