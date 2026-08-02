"""Regression test: adaptive save-point binding at a town Kafra.

The AI must bind its respawn point at the nearest town Kafra (one-shot per
town) so deaths / Butterfly-Wing returns land in a safe town instead of the
hostile start field. It must emit the Kafra `talknpc` + Save (`talk resp 0`)
sequence once per town, and never re-bind the same town.
"""
from __future__ import annotations

from ai_sidecar.autonomy.heuristic_service import HeuristicService


def _assess_in_town(h, map_name: str, bot_id: str = "bot:transit"):
    signals = {
        "bot_id": bot_id,
        "map": map_name,
        "map_known": True,  # in-game (passes the not-in-game guard)
        "lockMap": "prt_fild08",
        "base_level": 1,
        "job": "novice",
        "hp_ratio": 1.0,
        "zeny": 0,
        "weight_ratio": 0,
        "job_level": 1,
        "job_name": "novice",
    }
    result = h.assess(signals, bot_id_override=bot_id)
    return [a.command for a in result.actions]


def test_save_point_bound_once_in_prontera() -> None:
    h = HeuristicService()
    h._init_new_domains()
    # Bot visits Prontera (Kafra at 158,180) for the first time -> bind.
    cmds1 = _assess_in_town(h, "prontera", "kicapmasin4")
    assert any(c.startswith("talknpc 158") for c in cmds1), f"must talk to Prontera Kafra: {cmds1}"
    assert "talk resp 0" in cmds1, "must choose Kafra Save option"
    # Second visit to the SAME town -> must NOT re-bind (one-shot per town).
    cmds2 = _assess_in_town(h, "prontera", "kicapmasin4")
    assert not any(c.startswith("talknpc 158") for c in cmds2), \
        f"must NOT re-bind same town: {cmds2}"


def test_save_point_binds_different_town() -> None:
    h = HeuristicService()
    h._init_new_domains()
    # Bind in Prontera, then visit Izlude (Kafra 108,138) -> bind there too.
    _assess_in_town(h, "prontera", "kicapmasin5")
    cmds = _assess_in_town(h, "izlude", "kicapmasin5")
    assert any(c.startswith("talknpc 108") for c in cmds), f"must bind Izlude Kafra: {cmds}"


def test_save_point_not_bound_in_non_kafra_map() -> None:
    h = HeuristicService()
    h._init_new_domains()
    # A field (prt_fild08) has no Kafra -> no save binding.
    cmds = _assess_in_town(h, "prt_fild08", "kicapmasin6")
    assert not any(c.startswith("talknpc") and "158" in c for c in cmds), f"no kafra on field: {cmds}"
