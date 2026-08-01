"""Regression test: cold-start transit protection must fire for level<=5 bots.

The block guarded on `self._cold_start_planner` — an attribute that was NEVER
assigned, making transit protection (and the whole level-1-5 cold-start block)
dead code. Fixed to use the wired `_cold_start_manager`. This test locks the
behavior in: a level-1 bot on a fild map MUST get attackAuto 0 (run, don't
fight), and a level-6+ bot must NOT.
"""

from __future__ import annotations

from ai_sidecar.autonomy.heuristic_service import HeuristicService


def _assess(base_level: int, map_name: str, lock_map: str = "prontera"):
    h = HeuristicService()
    h._init_new_domains()
    assert h._cold_start_manager is not None, "cold_start_manager must be wired"
    signals = {
        "bot_id": "bot:transit",
        "map": map_name,
        "lockMap": lock_map,
        "base_level": base_level,
        "job": "novice",
    }
    result = h.assess(signals, "COLD_START")
    return [a.command for a in result.actions]


def test_level1_on_fild_triggers_transit_protection() -> None:
    cmds = _assess(base_level=1, map_name="prt_fild05")
    assert "set attackAuto 0" in cmds, f"transit protection missing: {cmds}"
    assert "set attackAuto_inLockOnly 1" in cmds
    assert "set lockMap prontera" in cmds


def test_level5_on_fild_triggers_transit_protection() -> None:
    cmds = _assess(base_level=5, map_name="prt_fild08")
    assert "set attackAuto 0" in cmds


def test_level6_on_fild_uses_field_config_not_transit() -> None:
    cmds = _assess(base_level=6, map_name="prt_fild05")
    # Level 6-15: field hunting with lockMap prt_fild05 — NOT transit-run mode
    assert "set attackAuto 0" not in cmds, f"level-6 must not transit: {cmds}"
    assert "set lockMap prt_fild05" in cmds


def test_level1_in_town_enables_attacking() -> None:
    cmds = _assess(base_level=1, map_name="prontera", lock_map="prontera")
    # Arrived at town lockMap -> attacking enabled (farming Porings)
    assert "set attackAuto 3" in cmds, f"town arrival must enable attack: {cmds}"
    assert "set attackAuto 0" not in cmds
