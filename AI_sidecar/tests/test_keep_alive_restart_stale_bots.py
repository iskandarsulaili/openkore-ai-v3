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
