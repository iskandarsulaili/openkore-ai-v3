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
    # level 1-5 now targets the academy field prt_fild08 (rathena-ai-world
    # starter zone), not the town map
    assert "set lockMap prt_fild08" in cmds


def test_level5_on_academy_field_enables_attacking() -> None:
    # prt_fild08 is the safe Cryptura Academy starter field (level-1
    # Porings/Lunatics) — a level-5 bot should FIGHT there, not transit-run.
    cmds = _assess(base_level=5, map_name="prt_fild08", lock_map="prt_fild08")
    assert "set attackAuto 3" in cmds, f"academy field must enable attack: {cmds}"
    assert "set attackAuto 0" not in cmds


def test_level5_on_non_academy_fild_still_transits() -> None:
    # A level-5 bot on a tougher NON-academy field still runs (don't fight).
    cmds = _assess(base_level=5, map_name="prt_fild05", lock_map="prt_fild08")
    assert "set attackAuto 0" in cmds


def test_level6_on_fild_uses_field_config_not_transit() -> None:
    cmds = _assess(base_level=6, map_name="prt_fild05")
    # Level 6-15: field hunting with lockMap prt_fild05 — NOT transit-run mode
    assert "set attackAuto 0" not in cmds, f"level-6 must not transit: {cmds}"
    assert "set lockMap prt_fild05" in cmds


def test_level1_in_town_enables_attacking() -> None:
    cmds = _assess(base_level=1, map_name="prontera", lock_map="prt_fild08")
    # Level 1-5 targets the academy field prt_fild08. A bot still in town is
    # navigating toward it (not stuck idle); it does NOT set attackAuto 0 —
    # transit protection only fires while on a *_fild map en route.
    assert any("prt_fild08" in c for c in cmds), f"must navigate to academy: {cmds}"


def test_level1_in_academy_registers_at_receptionist() -> None:
    """A level-1 bot spawned in the Izlude Academy must register for starter
    gear (Novice_Knife + potions) before hunting prt_fild08."""
    cmds = _assess(base_level=1, map_name="iz_ac01", lock_map="prt_fild08")
    assert any("talknpc" in c and "100 39" in c for c in cmds), \
        f"must talk to Academy Receptionist (100,39): {cmds}"


def test_level1_on_secluded_island_sails_to_izlude() -> None:
    """A level-1 bot on int_land (Secluded Island intro; disconnected map)
    must RUN (attackAuto 0) to the sailor (49,57) and sail to Izlude instead
    of trying to route to prt_fild08 or fighting the island Poring."""
    cmds = _assess(base_level=1, map_name="int_land", lock_map="prt_fild08")
    assert "move 49 57" in cmds, f"must walk to sailor: {cmds}"
    assert any("talknpc 49 57" in c for c in cmds), f"must talk to sailor warp: {cmds}"
    assert "set attackAuto 0" in cmds, f"must run (attackAuto 0) on the island: {cmds}"
    # Must NOT force prt_fild08 lockMap OR move prontera while stranded (both
    # cannot route from int_land and would leave the bot stuck at spawn).
    assert not any("prt_fild08" in c for c in cmds), f"no lockMap on island: {cmds}"
    assert not any(c == "move prontera" for c in cmds), \
        f"must NOT move prontera from island (cannot route): {cmds}"
