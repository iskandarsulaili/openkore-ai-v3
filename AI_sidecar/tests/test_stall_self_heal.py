"""Tests for the self-heal NO-PROGRESS stall detector (2026-08-28).

The bot stalls on an empty map (wedge: map ownership flapping) — the 60-cycle
stuck-tracker resets on map change so it never fires. The stall detector keys
on EXP progress (immune to map flapping): in-game + EXP frozen for
STALL_NO_PROGRESS_MIN minutes -> emit a map-change heal.
"""
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

sys.path.insert(0, ".")
from ai_sidecar.autonomy.pdca_loop import _pick_stall_target


def _snap(exp: int, map_name: str, age_s: float = 30.0):
    """Build a snapshot-like SimpleNamespace with observed_at in the past."""
    obs = datetime.fromtimestamp(time.time() - age_s, tz=timezone.utc).isoformat()
    return SimpleNamespace(
        raw={"map": map_name, "death_count": 0},
        position=SimpleNamespace(map=map_name),
        vitals=SimpleNamespace(hp_ratio=1.0),
        progression=SimpleNamespace(base_exp=exp),
        inventory_items=[],
        has_weapon_in_inventory=True,
        observed_at=obs,
    )


class _FakeLTM:
    def __init__(self):
        self.stored = []
    def store(self, category, content, importance=1, **kw):
        self.stored.append((category, content, importance))


class _FakeAQ:
    def __init__(self):
        self.enqueued = []
    def enqueue(self, bot_id, proposal):
        self.enqueued.append((bot_id, proposal))


def _make_pdca():
    """Build a PDCALoop with fake runtime (snapshot_cache/action_queue/LTM)."""
    from ai_sidecar.autonomy.pdca_loop import PDCALoop
    rt = MagicMock()
    rt.action_queue = _FakeAQ()
    rt.long_term_memory = _FakeLTM()
    rt.server_solutions_store = None
    rt.dynamic_portal_discovery = None
    o = PDCALoop.__new__(PDCALoop)
    o._runtime = rt
    o._stall_no_progress_min = 5
    o._log = MagicMock()
    return o, rt


def test_pick_stall_target_farm_map_first():
    rt = MagicMock()
    rt.server_solutions_store = MagicMock()
    rt.server_solutions_store.get = lambda k, d=None: "prt_fild08" if k == "farm_map" else None
    rt.dynamic_portal_discovery = None
    assert _pick_stall_target(rt, "prt_fild05") == "prt_fild08"


def test_pick_stall_target_empty_when_no_data():
    """No store + no DPD -> falls back to a CITY from tables/cities.txt (safe)."""
    rt = MagicMock()
    rt.server_solutions_store = None
    with patch("ai_sidecar.dynamic_portal_discovery.get_dynamic_portal_discovery", side_effect=Exception("no dpd")):
        tgt = _pick_stall_target(rt, "prt_fild05")
        assert tgt != "", "empty store + no DPD must still find a city target"
        assert tgt != "prt_fild05"


def test_stall_detector_fires_after_frozen_exp():
    """EXP frozen 5+ min while in-game -> heal action enqueued."""
    o, rt = _make_pdca()
    prev = {"_seeded": True, "deaths": 0, "exp": 100, "weapon": True,
            "map": "prt_fild05", "_exp_change_ts": time.time() - 360}
    rt.snapshot_cache.get.return_value = _snap(100, "prt_fild05", 30.0)
    with patch.object(o, "_memory_snapshot_key", return_value=prev):
        o._remember_significant_deltas("testbotA")
    assert len(rt.action_queue.enqueued) == 1, "heal action must fire on frozen EXP"
    cmd = rt.action_queue.enqueued[0][1].command
    assert cmd.startswith("move "), f"heal must be a move, got {cmd}"
    assert "self_heal_stall" in rt.action_queue.enqueued[0][1].metadata["source"]


def test_stall_detector_silent_when_exp_changes():
    """EXP changed -> progress, no stall."""
    o, rt = _make_pdca()
    prev = {"_seeded": True, "deaths": 0, "exp": 100, "weapon": True,
            "map": "prt_fild05", "_exp_change_ts": time.time() - 360}
    rt.snapshot_cache.get.return_value = _snap(101, "prt_fild05", 30.0)
    with patch.object(o, "_memory_snapshot_key", return_value=prev):
        o._remember_significant_deltas("testbotA")
    assert len(rt.action_queue.enqueued) == 0


def test_stall_detector_silent_when_snapshot_stale():
    """Stale snapshot (bot disconnected) must NOT trigger a heal."""
    o, rt = _make_pdca()
    prev = {"_seeded": True, "deaths": 0, "exp": 100, "weapon": True,
            "map": "prt_fild05", "_exp_change_ts": time.time() - 360}
    rt.snapshot_cache.get.return_value = _snap(100, "prt_fild05", 600.0)
    with patch.object(o, "_memory_snapshot_key", return_value=prev):
        o._remember_significant_deltas("testbotA")
    assert len(rt.action_queue.enqueued) == 0


def test_stall_detector_arms_timestamp_on_exp_change():
    """EXP change re-arms the window (no heal while gaining)."""
    o, rt = _make_pdca()
    prev = {"_seeded": True, "deaths": 0, "exp": 100, "weapon": True,
            "map": "prt_fild05", "_exp_change_ts": time.time() - 720}
    rt.snapshot_cache.get.return_value = _snap(102, "prt_fild05", 30.0)
    with patch.object(o, "_memory_snapshot_key", return_value=prev):
        o._remember_significant_deltas("testbotA")
    # exp changed -> _exp_change_ts re-armed to now; no heal
    assert len(rt.action_queue.enqueued) == 0
    assert prev["_exp_change_ts"] > time.time() - 5


def test_stall_detector_fires_even_when_never_gained_exp():
    """A bot that NEVER gains EXP (frozen from start) must still stall-fire:
    the first-observation seed sets _exp_change_ts=now so the 5-min window
    starts at baseline (bug: _last_change stayed 0 -> never fired)."""
    from ai_sidecar.autonomy.pdca_loop import PDCALoop
    rt = MagicMock()
    rt.action_queue = _FakeAQ()
    rt.long_term_memory = _FakeLTM()
    rt.server_solutions_store = None
    rt.dynamic_portal_discovery = None
    o = PDCALoop.__new__(PDCALoop)
    o._runtime = rt
    o._stall_no_progress_min = 5
    o._log = MagicMock()
    # First observation: seeds baseline + _exp_change_ts (no stall yet)
    rt.snapshot_cache.get.return_value = _snap(626, "prt_fild05", 30.0)
    prev = {}
    with patch.object(o, "_memory_snapshot_key", return_value=prev):
        o._remember_significant_deltas("testbotA")
    assert len(rt.action_queue.enqueued) == 0, "first obs must not stall-fire"
    assert prev["_exp_change_ts"] > time.time() - 5, "baseline window must start at seed"
    # 6 min later, STILL 626 -> stall fires
    prev["_exp_change_ts"] = time.time() - 360
    rt.snapshot_cache.get.return_value = _snap(626, "prt_fild05", 30.0)
    with patch.object(o, "_memory_snapshot_key", return_value=prev):
        o._remember_significant_deltas("testbotA")
    assert len(rt.action_queue.enqueued) == 1, "frozen-from-start bot must stall-fire"


def test_stall_detector_route_failure_triggers_heal():
    """F13: high route_failure_count (8+) while in-game -> map-change heal,
    rate-limited by the same window (one per _stall_min)."""
    from ai_sidecar.autonomy.pdca_loop import PDCALoop
    rt = MagicMock()
    rt.action_queue = _FakeAQ()
    rt.long_term_memory = _FakeLTM()
    rt.server_solutions_store = None
    rt.dynamic_portal_discovery = None
    o = PDCALoop.__new__(PDCALoop)
    o._runtime = rt
    o._stall_no_progress_min = 5
    o._log = MagicMock()
    # Fresh snapshot with route_failure_count=10, EXP unchanged
    snap = _snap(626, "prt_fild05", 30.0)
    snap.raw["route_failure_count"] = 10
    prev = {"_seeded": True, "deaths": 0, "exp": 626, "weapon": True,
            "map": "prt_fild05", "_exp_change_ts": time.time() - 10}
    rt.snapshot_cache.get.return_value = snap
    with patch.object(o, "_memory_snapshot_key", return_value=prev):
        o._remember_significant_deltas("testbotA")
    assert len(rt.action_queue.enqueued) == 1, "route-failure stall must heal"
    meta = rt.action_queue.enqueued[0][1].metadata
    assert meta["reason"] == "route_failure"


def test_stall_detector_resolves_profile_key_mismatch_via_latest():
    """bot_id is the PROFILE key but snapshot_cache keys by meta.bot_id (FULL
    key). get(profile) misses -> latest() fallback must resolve the snapshot
    so the stall detector still fires (the live bug: snapshot_missing logs)."""
    from ai_sidecar.autonomy.pdca_loop import PDCALoop
    rt = MagicMock()
    rt.action_queue = _FakeAQ()
    rt.long_term_memory = _FakeLTM()
    rt.server_solutions_store = None
    rt.dynamic_portal_discovery = None
    # exact get(profile) MISSES; latest() returns the full-key snapshot
    rt.snapshot_cache.get.return_value = None
    rt.snapshot_cache.latest.return_value = _snap(100, "prt_fild05", 30.0)
    o = PDCALoop.__new__(PDCALoop)
    o._runtime = rt
    o._stall_no_progress_min = 5
    o._log = MagicMock()
    prev = {"_seeded": True, "deaths": 0, "exp": 100, "weapon": True,
            "map": "prt_fild05", "_exp_change_ts": time.time() - 360}
    with patch.object(o, "_memory_snapshot_key", return_value=prev):
        o._remember_significant_deltas("TestBotA:testbot99")
    assert len(rt.action_queue.enqueued) == 1, "profile-key miss must fall back to latest() and fire"
    assert rt.snapshot_cache.latest.called
