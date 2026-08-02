"""Regression test: the dormant CombatTactics class is now driven per bot-cycle
through `combat_tactics_integration.run_combat_tactics`.

CombatTactics (combat_tactics.py) held the Pro-RO per-class skill-combo knowledge
base but was only CONSTRUCTED onto the runtime — none of its methods were called
(full-repo reference scan: get_combo/should_kite/... had 0 callers). This locks
in the wiring: an in-combat, in-game bot must have gated `ss <skill>` casts
queued from its best combo, and the `counters()` stub must report real state.
"""
from __future__ import annotations

import pytest

from ai_sidecar.autonomy import combat_tactics_integration as cti
from ai_sidecar.contracts.actions import ActionProposal
from ai_sidecar import combat_tactics


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
        def __init__(self, hp_ratio, sp, hp_max, job_name, hp):
            self.hp_ratio = hp_ratio; self.sp = sp; self.hp_max = hp_max
            self.job_name = job_name; self.hp = hp

    class _C:
        def __init__(self, aggro_count, target_element):
            self.aggro_count = aggro_count; self.target_element = target_element

    def __init__(self, *, hp=100, hp_ratio=1.0, sp=50, job="novice",
                 aggro=2, elem="water", map_known=True):
        self.vitals = _Snap._V(hp_ratio, sp, 100, job, hp)
        self.combat = _Snap._C(aggro, elem)
        self.map_known = map_known
        self.raw = {"in_game": map_known}
        self.inventory_items = []; self.skills = []


@pytest.fixture(autouse=True)
def _reset():
    cti._CT = None
    cti._LAST_CAST.clear()
    yield
    cti._CT = None
    cti._LAST_CAST.clear()


def test_in_combat_bot_queues_gated_skill_cast() -> None:
    rt = _FakeRuntime()
    bot = "test:kicapmasin1"
    # A mage in combat vs a water-element monster -> get_combo yields a combo
    # (e.g. freeze then fire); the first gated skill should be queued.
    snap = _Snap(hp=100, sp=200, job="mage", aggro=3, elem="water", map_known=True)
    count = cti.run_combat_tactics(rt, bot, snap)
    assert count >= 1, "in-combat mage should queue at least one gated skill cast"

    cmds = [p.command for _, p in rt.action_queue.queued if p.command]
    assert cmds, f"expected a skill command, got: {cmds}"
    # All emitted skill commands must be real `ss <skill>` casts (not empty/garbage).
    assert all(c.startswith("ss ") and len(c.split()) == 2 for c in cmds), \
        f"unexpected command form: {cmds}"


def test_no_aggro_no_skill_spam() -> None:
    rt = _FakeRuntime()
    bot = "test:kicapmasin1"
    # Out of combat (aggro 0) -> the combo engine is not consulted, no skill spam.
    snap = _Snap(hp=100, sp=200, job="mage", aggro=0, elem="water")
    count = cti.run_combat_tactics(rt, bot, snap)
    assert count == 0, "no combat -> no combat-tactics action"

    cmds = [p.command for _, p in rt.action_queue.queued if p.command]
    assert cmds == [], f"no skills outside combat: {cmds}"


def test_skips_disconnected_bot() -> None:
    rt = _FakeRuntime()
    snap = _Snap(hp=100, sp=200, job="mage", aggro=3, elem="water", map_known=False)
    count = cti.run_combat_tactics(rt, "test:kicapmasin1", snap)
    assert count == 0
    assert rt.action_queue.queued == []


def test_counts_now_reflect_real_combos() -> None:
    """The old `counters()` stub returned {'combos': 0}; now it reports live data."""
    info = combat_tactics.counters()
    assert info["combos"] > 0, f"real combo registry should be non-empty: {info}"
    assert info["classes"] > 0, f"real class registry should be non-empty: {info}"
    assert "kite_classes" in info and "size_weapons" in info
