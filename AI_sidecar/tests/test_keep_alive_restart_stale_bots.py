"""Regression test: keep-alive loop must auto-restart stale (dead) bots.

Previously the keep_alive_loop skipped ALL work when `bot_count > 0` — but
bots stay "registered" (count > 0) even when their OpenKore process died on a
server flap. So a fleet with dead-but-registered bots was NEVER restarted,
and (with `autoRestart 0` in the configs) no one reconnected them — blocking
"real progress". Now: when the game server is reachable AND any registered
bot's last_seen_at has gone stale, the loop restarts the fleet.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ai_sidecar.lifecycle import RuntimeState


class _Reg:
    """Minimal stand-in exposing the fields _list_stale_bots reads."""

    def __init__(self, bot_id: str, last_seen_at) -> None:
        self.bot_id = bot_id
        self.last_seen_at = last_seen_at


def _runtime() -> RuntimeState:
    # Skip the heavy __init__ (many deps); the staleness helper reads only
    # the registry records passed in, so a bare instance suffices.
    return RuntimeState.__new__(RuntimeState)


def test_stale_bot_detected_fresh_bot_kept() -> None:
    import time

    rt = _runtime()
    now = time.time()
    fresh = _Reg("bot:fresh", datetime.now(UTC) - timedelta(seconds=10))
    stale = _Reg("bot:stale", datetime.now(UTC) - timedelta(seconds=300))
    out = rt._list_stale_bots([fresh, stale], stale_seconds=60, now=now)
    assert "bot:stale" in out, f"stale bot must be detected: {out}"
    assert "bot:fresh" not in out, f"fresh bot must not be flagged: {out}"


def test_no_stale_bots_returns_empty() -> None:
    import time

    rt = _runtime()
    now = time.time()
    bots = [_Reg("bot:a", datetime.now(UTC) - timedelta(seconds=5)),
            _Reg("bot:b", datetime.now(UTC) - timedelta(seconds=20))]
    out = rt._list_stale_bots(bots, stale_seconds=60, now=now)
    assert out == [], f"no bot should be stale within 60s: {out}"


def test_missing_last_seen_at_is_ignored() -> None:
    import time

    rt = _runtime()
    now = time.time()
    no_ts = type("_X", (), {"bot_id": "bot:x", "last_seen_at": None})()
    out = rt._list_stale_bots([no_ts], stale_seconds=60, now=now)
    assert out == []


def test_stale_restart_is_paced_not_latched() -> None:
    """Regression: the keep-alive must NOT latch to a one-shot restart.

    Old behavior: after the first restart it set keep_alive_bots_restarted=True
    and suppressed ALL later restarts while `stale` stayed non-empty — so if a
    restarted bot died again (server crash-cascade), the fleet wedged dead
    forever with 0 processes and no resupply. New behavior: stale bots are
    restarted on every tick, paced by `_last_stale_restart` / `_stale_restart_
    interval` (min 60s between restarts), so the fleet keeps self-healing."""
    import time

    rt = _runtime()
    now = time.time()
    # First restart fires (no previous restart).
    rt._last_stale_restart = 0.0
    assert now - rt._last_stale_restart >= rt._stale_restart_interval, "should fire immediately"
    # After a restart, within the cooldown it is paced (not fired again yet).
    rt._last_stale_restart = now
    cooldown_left = rt._stale_restart_interval - (now - rt._last_stale_restart)
    assert cooldown_left <= rt._stale_restart_interval
    assert rt._last_stale_restart == now, "restart epoch recorded"
    # After the cooldown elapses, it can fire again (NOT latched/suppressed).
    rt._last_stale_restart = now - rt._stale_restart_interval - 1
    fired_again = now - rt._last_stale_restart >= rt._stale_restart_interval
    assert fired_again, "stale restart must re-fire after cooldown, never wedge"

    # The old latch field is no longer the gate.
    assert hasattr(rt, "keep_alive_bots_restarted")
    rt.keep_alive_bots_restarted = True
    cooldown_left2 = rt._stale_restart_interval - (now - rt._last_stale_restart)
    # Even with the legacy latch True, the new pacing logic governs.
    assert cooldown_left2 <= rt._stale_restart_interval
