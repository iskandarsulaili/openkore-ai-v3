"""Regression test: FleetLearningSystem must actually learn when fed.

FleetLearningSystem was initialized at every startup but record_outcome was
NEVER called anywhere in production — the zone_outcomes / zone_elo /
party_compositions tables stayed at 0 rows forever. The map-change feed
(pdca_loop) now closes each zone session and calls record_outcome. This test
locks the write path: outcome -> ELO update -> party composition -> readable
scores.
"""

from __future__ import annotations

from ai_sidecar.fleet.self_learning import FleetLearningSystem, ZoneOutcome


def _outcome(bot_id: str, map_name: str, *, exp: float = 100.0,
             zeny: float = 50.0, deaths: int = 0, level: int = 12,
             dur: float = 3.0, composition: dict[str, int] | None = None) -> ZoneOutcome:
    return ZoneOutcome(
        bot_id=bot_id,
        map_name=map_name,
        bot_level=level,
        duration_minutes=dur,
        base_exp_gained=exp,
        job_exp_gained=0.0,
        zeny_gained=zeny,
        death_count=deaths,
        party_size=1 if not composition else sum(composition.values()),
        party_composition=composition or {},
        success=deaths == 0,
    )


def test_record_outcome_updates_zone_stats() -> None:
    fls = FleetLearningSystem()
    fls.record_outcome(_outcome("bot:a", "prt_fild08", exp=250.0, zeny=120.0, deaths=0))
    fls.record_outcome(_outcome("bot:a", "prt_fild08", exp=180.0, zeny=90.0, deaths=1))
    fls.record_outcome(_outcome("bot:b", "pay_fild08", exp=300.0, zeny=200.0, deaths=0))

    s = fls.get_zone_score("prt_fild08")
    assert s["total_outcomes"] == 2
    assert s["avg_base_exp_rate"] > 0.0
    assert s["avg_zeny_rate"] > 0.0
    assert s["success_rate"] < 1.0  # one death dragged it below 1.0

    stats = fls.stats()
    assert stats["total_outcomes"] >= 3
    assert stats["known_zones"] >= 2
    assert stats["unique_bots"] >= 2
    assert stats["unique_maps"] >= 2


def test_best_zone_ranks_by_score() -> None:
    fls = FleetLearningSystem()
    for _ in range(3):
        fls.record_outcome(_outcome("bot:a", "prt_fild08", exp=100.0, deaths=0))
    for _ in range(3):
        fls.record_outcome(_outcome("bot:a", "pay_fild08", exp=500.0, deaths=0))
    best = fls.get_best_zone(bot_level=12, min_samples=3)
    assert best, "best zone list must not be empty"
    assert best[0]["map_name"] == "pay_fild08", f"pay_fild08 should rank first, got {best[0]}"


def test_party_composition_learned() -> None:
    fls = FleetLearningSystem()
    fls.record_outcome(_outcome("bot:a", "prt_fild08", composition={"Knight": 1, "Priest": 1}))
    comp = fls.get_party_composition("prt_fild08")
    assert comp is not None
    inner = comp.get("composition", comp)
    assert inner.get("Knight", 0) >= 1
    assert inner.get("Priest", 0) >= 1


def test_mined_out_detection() -> None:
    fls = FleetLearningSystem()
    # Mined-out = exp rate dropped 50%+ across the last MINED_OUT_WINDOW (10)
    # outcomes. Start rich, then starve: first 6 at high exp, next 10 at ~0.
    for _ in range(6):
        fls.record_outcome(_outcome("bot:a", "gef_fild01", exp=1000.0, deaths=0))
    for _ in range(10):
        fls.record_outcome(_outcome("bot:a", "gef_fild01", exp=5.0, deaths=0))
    mined = fls.get_mined_out_zones()
    maps = [m["map_name"] if isinstance(m, dict) else str(m) for m in mined]
    assert "gef_fild01" in maps, f"zone with 50%+ exp drop must be mined out, got {maps}"
