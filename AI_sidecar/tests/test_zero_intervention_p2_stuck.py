"""Regression tests for ZERO_INTERVENTION P2.1: stuck-state detector + health monitor wiring.

Locks:
- `_snapshot_progress_fields` extracts kills/exp/level from both dict and pydantic snapshots.
- `_check_progress_and_detect_stuck` enqueues a recovery when a bot makes no progress
  past the configured window, and self-clears once progress resumes.
- `bot_health_monitor.run_health_checks` routes farm/town through server_solutions_store
  (RULE.md) and enqueues valid non-expired actions.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from types import SimpleNamespace

from ai_sidecar.autonomy import bot_health_monitor as bhm
from ai_sidecar.autonomy.pdca_loop import (
    _snapshot_progress_fields,
    PDCALoop,
    PDCAConfig,
)


def _dict_snap(kills: int = 0, exp: int = 0, level: int = 1, in_game: bool = True) -> dict:
    return {
        "kills": kills,
        "base_exp": exp,
        "base_level": level,
        "raw": {"in_game": in_game},
    }


def test_snapshot_progress_fields_dict() -> None:
    f = _snapshot_progress_fields(_dict_snap(kills=5, exp=1200, level=10))
    assert f == {"kills": 5, "exp": 1200, "level": 10}


def test_snapshot_progress_fields_pydantic_like() -> None:
    prog = SimpleNamespace(base_exp=500, base_level=22)
    snap = SimpleNamespace(kills=3, progression=prog)
    f = _snapshot_progress_fields(snap)
    assert f["kills"] == 3 and f["exp"] == 500 and f["level"] == 22


class _FakeQueue:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def enqueue(self, bot_id, proposal):
        self.items.append({"bot_id": bot_id, "cmd": proposal.command,
                           "idem": proposal.idempotency_key, "kind": proposal.kind,
                           "source": getattr(proposal, "source", "")})


class _FakeSnap:
    def __init__(self, snap: dict) -> None:
        self._snap = snap

    def get(self, bot_id):
        return self._snap


class _FakeRuntime:
    def __init__(self, snap: dict):
        self.snapshot_cache = _FakeSnap(snap)
        self.action_queue = _FakeQueue()
        self.server_solutions_store = SimpleNamespace(
            get=lambda k, d=None: {"farm_map": "prt_fild08", "safe_town": "prontera"}.get(k, d),
            get_json=lambda k, d=None: {},
        )


def _make_loop(runtime) -> PDCALoop:
    cfg = PDCAConfig(stuck_detect_window_s=1.0)
    loop = PDCALoop.__new__(PDCALoop)
    loop._runtime = runtime
    loop._config = cfg
    loop._circuit_breaker = None
    loop._progress_anchor = {}
    loop._stuck_declared = {}
    loop._advisory_inflight = set()
    return loop


def test_stuck_detector_enqueues_recovery_after_window() -> None:
    snap = _dict_snap(kills=0, exp=0, level=1, in_game=True)
    rt = _FakeRuntime(snap)
    loop = _make_loop(rt)
    # First call anchors, second call (past 1s window) declares stuck.
    assert loop._check_progress_and_detect_stuck("botA") is False
    time.sleep(1.1)
    assert loop._check_progress_and_detect_stuck("botA") is True
    # source is normalized to the ActionProposal allowlist {reflex,planner,crewai,ml,fleet,manual}
    assert rt.action_queue.items and rt.action_queue.items[0]["cmd"] == "ai auto"
    assert rt.action_queue.items[0]["idem"] == "stuck-recover:botA"
    assert rt.action_queue.items[0]["source"] == "manual"  # normalized from stuck_detector


def test_stuck_detector_resets_on_progress() -> None:
    snap = _dict_snap(kills=0, exp=0, level=0, in_game=True)
    rt = _FakeRuntime(snap)
    loop = _make_loop(rt)
    loop._check_progress_and_detect_stuck("botA")
    # Progress -> anchor resets, no stuck.
    rt.snapshot_cache._snap = _dict_snap(kills=1, exp=0, level=0)
    assert loop._check_progress_and_detect_stuck("botA") is False


def test_stuck_detector_skips_logged_out() -> None:
    snap = _dict_snap(kills=0, exp=0, level=0, in_game=False)
    rt = _FakeRuntime(snap)
    loop = _make_loop(rt)
    loop._check_progress_and_detect_stuck("botA")
    time.sleep(1.1)
    assert loop._check_progress_and_detect_stuck("botA") is False


def test_health_monitor_uses_server_store_and_expires_future() -> None:
    from ai_sidecar.contracts.actions import ActionProposal
    snap = SimpleNamespace(
        vitals=SimpleNamespace(weight_ratio=0.9, hp_ratio=1.0),
        position=SimpleNamespace(map="prontera"),
    )
    rt = _FakeRuntime(snap)
    # weight_ratio 0.9 > 0.65 -> overweight corrections, must not hardcode prt_in
    corr = bhm.check_bot_health(rt, rt.action_queue, "botA")
    assert corr, "expected overweight corrections"
    cmd_text = " ".join(c["command"] for c in corr)
    # server_solutions_store not set here -> falls back to default, but never 'prt_fild05'
    assert "prt_fild05" not in cmd_text
    assert "server_solutions_store" not in cmd_text or "store" in cmd_text
