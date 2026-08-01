"""Tests for the exploration scout (FLAW 2).

The scout is the route_failure/stuck -> exploration wiring: a bot stuck 60+
PDCA cycles on the same map gets a move-to-unexplored-map proposal (preferring
DPD portal-reachable targets), level-gated so academy bots (<=5) stay on the
izlude route, with a per-bot 5-minute cooldown.
"""

from __future__ import annotations

import time

from ai_sidecar.autonomy.pdca_loop import _emit_exploration_scout
from ai_sidecar.dynamic_portal_discovery import DynamicPortalDiscovery


class _ScoutRuntime:
    def __init__(self, dpd: DynamicPortalDiscovery) -> None:
        self.dynamic_portal_discovery = dpd
        self.action_queue = _FakeQueue()


class _FakeQueue:
    def __init__(self) -> None:
        self.enqueued: list[tuple[str, object]] = []

    def enqueue(self, bot_id: str, proposal: object) -> None:
        self.enqueued.append((bot_id, proposal))


def _dpd_with(portal_targets: list[str], explored: list[str] | None = None) -> DynamicPortalDiscovery:
    import tempfile
    import os

    tmp = tempfile.mkdtemp()
    dpd = DynamicPortalDiscovery(db_path=os.path.join(tmp, "shared_learning.db"))
    for m in explored or []:
        dpd.record_map_visit("bot:scout", m)
    # Seed a discovered portal chain from prt_fild08 -> each target
    for i, t in enumerate(portal_targets):
        dpd.record_portal_entry("bot:scout", "prt_fild08", 100 + i, 100)
        dpd.record_portal_exit("bot:scout", t, 50, 50)
    return dpd


def test_scout_queues_move_to_unexplored_map() -> None:
    dpd = _dpd_with(portal_targets=["prt_fild09"], explored=["prt_fild08"])
    rt = _ScoutRuntime(dpd)
    result = _emit_exploration_scout(rt, "bot:scout", "prt_fild08", base_level=12)
    assert result == 1
    assert len(rt.action_queue.enqueued) == 1
    bot_id, proposal = rt.action_queue.enqueued[0]
    assert bot_id == "bot:scout"
    assert proposal.command == "move prt_fild09"
    assert proposal.conflict_key == "nav.scout"


def test_scout_level_gated_academy_bot_stays() -> None:
    dpd = _dpd_with(portal_targets=["prt_fild09"], explored=["prt_fild08"])
    rt = _ScoutRuntime(dpd)
    # Level 5 academy bot must NOT be sent to an unexplored map
    result = _emit_exploration_scout(rt, "bot:scout", "prt_fild08", base_level=5)
    assert result == 0
    assert rt.action_queue.enqueued == []


def test_scout_falls_back_to_any_unexplored_map() -> None:
    # No portals FROM prt_fild08, but hardcoded portal knowledge knows other
    # maps (aldebaran, mjolnir01 via a discovered portal) — unexplored and not
    # reachable from the current map, so branch 2 (any unexplored map) picks
    # the alphabetically-first known-but-unexplored map.
    dpd = _dpd_with(portal_targets=[], explored=["prt_fild08"])
    dpd.record_portal_entry("bot:other", "geffen", 120, 120)
    dpd.record_portal_exit("bot:other", "mjolnir01", 30, 30)
    rt = _ScoutRuntime(dpd)
    result = _emit_exploration_scout(rt, "bot:scout", "prt_fild08", base_level=20)
    assert result == 1
    assert len(rt.action_queue.enqueued) == 1
    command = str(rt.action_queue.enqueued[0][1].command)
    assert command.startswith("move ")
    target = command.split(" ", 1)[1]
    assert target != "prt_fild08"
    assert target in dpd.get_unexplored_maps()


def test_scout_cooldown_prevents_spam() -> None:
    dpd = _dpd_with(portal_targets=["prt_fild09"], explored=["prt_fild08"])
    rt = _ScoutRuntime(dpd)
    first = _emit_exploration_scout(rt, "bot:scout", "prt_fild08", base_level=12)
    assert first == 1
    second = _emit_exploration_scout(rt, "bot:scout", "prt_fild08", base_level=12)
    assert second == 0
    assert len(rt.action_queue.enqueued) == 1
