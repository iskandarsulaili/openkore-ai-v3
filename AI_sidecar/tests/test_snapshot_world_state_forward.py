"""Regression: snapshot ingest MUST populate the enriched/world state.

The runtime.ingest_snapshot() path previously wrote ONLY to snapshot_cache;
it never forwarded to normalizer_bus.ingest_snapshot(), so world_state (the
store enriched_state()/PDCA/heuristics/state_v2 read) stayed at map=None /
x=None / y=None — the entire decision pipeline was blind to bot positions.
This test pins the forward-wiring so it cannot regress.
"""
from __future__ import annotations

from datetime import UTC, datetime

from ai_sidecar.contracts.state import BotStateSnapshot, ContractMeta
from ai_sidecar.lifecycle import create_runtime


def _snapshot(bot_id: str, *, map_name: str, x: int, y: int, tick: str) -> BotStateSnapshot:
    return BotStateSnapshot(
        meta=ContractMeta(bot_id=bot_id),
        tick_id=tick,
        observed_at=datetime.now(UTC),
        position={"map": map_name, "x": x, "y": y},
        vitals={"hp": 100, "hp_max": 100, "sp": 50, "sp_max": 60},
        raw={"in_game": True},
    )


def test_snapshot_ingest_populates_enriched_state_position():
    runtime = create_runtime()
    bot_id = "bot:snapshot-forward"
    snap = _snapshot(bot_id, map_name="iz_ac01", x=100, y=39, tick="t-fwd-1")

    runtime.ingest_snapshot(snap)

    st = runtime.enriched_state(bot_id=bot_id)
    assert st.operational.map == "iz_ac01"
    assert st.operational.x == 100
    assert st.operational.y == 39
    assert st.navigation.map == "iz_ac01"
    assert st.navigation.x == 100
    assert st.navigation.y == 39


def test_snapshot_forward_updates_both_caches():
    runtime = create_runtime()
    bot_id = "bot:snapshot-both"
    snap = _snapshot(bot_id, map_name="prt_fild08", x=367, y=212, tick="t-both-1")

    runtime.ingest_snapshot(snap)

    # snapshot_cache must still hold it (existing behavior preserved)
    cached = runtime.snapshot_cache.get(bot_id)
    assert cached is not None
    assert cached.position.map == "prt_fild08"

    # world_state (enriched_state) must ALSO hold it (the fix)
    st = runtime.enriched_state(bot_id=bot_id)
    assert st.operational.map == "prt_fild08"
    assert st.operational.x == 367
    assert st.operational.y == 212


def test_snapshot_forward_survives_bad_payload():
    """A failing normalizer_bus must not crash ingest; the snapshot cache
    still stores it (fail-open, never break the ingest path)."""
    runtime = create_runtime()
    bot_id = "bot:snapshot-bad"

    # Force the normalizer forward to fail -> the guard must swallow it.
    class _BoomNormalizer:
        def ingest_snapshot(self, _snap):
            raise RuntimeError("simulated normalizer failure")

    runtime.normalizer_bus = _BoomNormalizer()  # type: ignore[assignment]

    snap = _snapshot(bot_id, map_name="iz_ac01", x=1, y=2, tick="t-bad-1")
    runtime.ingest_snapshot(snap)  # must NOT raise

    cached = runtime.snapshot_cache.get(bot_id)
    assert cached is not None
    assert cached.position.map == "iz_ac01"
