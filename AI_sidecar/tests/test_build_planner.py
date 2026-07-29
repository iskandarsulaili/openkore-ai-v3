"""Tests for BuildPlanner and StatBreakpointPlanner.

Verifies all 7 meta RO builds, stat breakpoint thresholds,
stat recommendations, skill builds, and trap skills.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to path so imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # AI_sidecar/
sys.path.insert(0, str(PROJECT_ROOT))

from ai_sidecar.domains.planning.build_planner import BuildPlanner
from ai_sidecar.domains.planning.stat_planner import StatBreakpointPlanner


# ── Test data path ──────────────────────────────────────────────────

YAML_PATH = PROJECT_ROOT / "data" / "build_plans.yaml"


# ── Fixtures ────────────────────────────────────────────────────────

def make_planners():
    bp = BuildPlanner(yaml_path=YAML_PATH)
    sp = StatBreakpointPlanner(yaml_path=YAML_PATH)
    return bp, sp


# ═══════════════════════════════════════════════════════════════════
# 1. BuildPlanner — listing and lookup
# ═══════════════════════════════════════════════════════════════════

def test_list_all_builds():
    bp, _ = make_planners()
    builds = bp.list_builds()
    assert len(builds) == 7, f"Expected 7 builds, got {len(builds)}"
    names = [b["name"] for b in builds]
    assert "Bowling Bash Knight" in names
    assert "Falcon Hunter (Blitz Beat)" in names
    assert "ME Priest (Magnus Exorcismus)" in names
    assert "MS Wizard (Meteor Storm)" in names
    assert "SB Assassin (Sonic Blow)" in names
    assert "Full Support Priest" in names
    assert "Agile Crit Sin" in names
    print(f"  ✓ All 7 builds listed: {', '.join(names)}")


def test_list_builds_by_job():
    bp, _ = make_planners()
    priest_builds = bp.list_builds(job="Priest")
    assert len(priest_builds) == 2, f"Expected 2 Priest builds, got {len(priest_builds)}"
    names = [b["name"] for b in priest_builds]
    assert "ME Priest (Magnus Exorcismus)" in names
    assert "Full Support Priest" in names
    print(f"  ✓ Priest builds: {', '.join(names)}")

    assassin_builds = bp.list_builds(job="Assassin")
    assert len(assassin_builds) == 2, f"Expected 2 Assassin builds, got {len(assassin_builds)}"


def test_get_build_by_id():
    bp, _ = make_planners()
    build = bp.get_build_by_id("bowling_bash_knight")
    assert build is not None
    assert build["name"] == "Bowling Bash Knight"
    assert build["job"] == "Knight"
    assert build["target_stats"]["STR"] == 80
    assert build["target_stats"]["AGI"] == 40
    assert build["target_stats"]["VIT"] == 40
    assert build["target_stats"]["DEX"] == 40
    print(f"  ✓ Bowling Bash Knight: {build['target_stats']}")

    # None for missing
    assert bp.get_build_by_id("nonexistent") is None


def test_get_builds_for_job():
    bp, _ = make_planners()
    builds = bp.get_builds_for_job("Hunter")
    assert len(builds) == 1
    assert builds[0]["id"] == "falcon_hunter"


def test_get_jobs():
    bp, _ = make_planners()
    jobs = bp.get_jobs()
    assert "Knight" in jobs
    assert "Hunter" in jobs
    assert "Priest" in jobs
    assert "Wizard" in jobs
    assert "Assassin" in jobs
    print(f"  ✓ Jobs with builds: {jobs}")


# ═══════════════════════════════════════════════════════════════════
# 2. BuildPlanner — stat recommendations
# ═══════════════════════════════════════════════════════════════════

def test_get_target_stats_by_id():
    bp, _ = make_planners()
    stats = bp.get_target_stats(build_id="bowling_bash_knight")
    assert stats is not None
    assert stats["STR"] == 80
    assert stats["AGI"] == 40


def test_get_target_stats_by_job():
    bp, _ = make_planners()
    stats = bp.get_target_stats(job="Wizard")
    assert stats is not None
    assert stats["INT"] == 99
    assert stats["DEX"] == 50


def test_recommend_next_stat_basic():
    bp, _ = make_planners()
    target = {"STR": 80, "AGI": 40, "VIT": 40, "DEX": 40, "INT": 1, "LUK": 1}

    # Starting from 1 all: STR is furthest from target
    current = {"STR": 1, "AGI": 1, "VIT": 1, "DEX": 1, "INT": 1, "LUK": 1}
    rec = bp.recommend_next_stat(current, target)
    assert rec == "STR", f"Expected STR (furthest gap), got {rec}"
    print(f"  ✓ Starting stats → recommends {rec}")

    # STR near target, AGI way behind
    current2 = {"STR": 75, "AGI": 1, "VIT": 1, "DEX": 1, "INT": 1, "LUK": 1}
    rec2 = bp.recommend_next_stat(current2, target)
    assert rec2 == "AGI", f"Expected AGI (next furthest), got {rec2}"
    print(f"  ✓ STR near target → recommends {rec2}")

    # All at target
    current3 = {"STR": 80, "AGI": 40, "VIT": 40, "DEX": 40, "INT": 1, "LUK": 1}
    rec3 = bp.recommend_next_stat(current3, target)
    print(f"  ✓ All at target → recommends {rec3} (fallback)")


def test_get_stat_priority():
    bp, _ = make_planners()
    prio = bp.get_stat_priority("falcon_hunter")
    assert prio == ["DEX", "AGI", "INT"]
    print(f"  ✓ Falcon Hunter stat priority: {prio}")


# ═══════════════════════════════════════════════════════════════════
# 3. BuildPlanner — skill builds
# ═══════════════════════════════════════════════════════════════════

def test_get_skill_build():
    bp, _ = make_planners()
    skills = bp.get_skill_build(build_id="bowling_bash_knight")
    skill_ids = [s[0] for s in skills]
    assert ("KN_BOWLINGBASH", 10) in skills
    assert ("SM_SPEARMASTERY", 10) in skills
    assert ("SM_BASH", 0) not in skills  # level 0 should be excluded
    assert "SM_BASH" not in skill_ids  # trap skill excluded
    print(f"  ✓ Bowling Bash Knight skills ({len(skills)}): {skill_ids}")


def test_get_skill_build_falcon_hunter():
    bp, _ = make_planners()
    skills = bp.get_skill_build(build_id="falcon_hunter")
    skill_ids = [s[0] for s in skills]
    assert ("HT_BLITZBEAT", 5) in skills
    assert ("HT_BEASTBANE", 10) in skills
    assert ("HT_TRUESIGHT", 10) in skills
    assert "AC_CONCENTRATION" not in skill_ids  # trap
    print(f"  ✓ Falcon Hunter skills ({len(skills)}): {skill_ids}")


def test_get_skill_build_me_priest():
    bp, _ = make_planners()
    skills = bp.get_skill_build(build_id="me_priest")
    skill_ids = [s[0] for s in skills]
    assert ("PR_MAGNUS", 10) in skills
    assert ("PR_TURNUNDEAD", 5) in skills
    assert ("PR_SANCTUARY", 5) in skills
    print(f"  ✓ ME Priest skills ({len(skills)}): {skill_ids}")


def test_get_skill_build_ms_wizard():
    bp, _ = make_planners()
    skills = bp.get_skill_build(build_id="ms_wizard")
    skill_ids = [s[0] for s in skills]
    assert ("WZ_METEOR", 10) in skills
    assert ("WZ_STORMGUST", 10) in skills
    assert ("WZ_LORDVERMILION", 5) in skills
    print(f"  ✓ MS Wizard skills ({len(skills)}): {skill_ids}")


def test_get_skill_build_sb_assassin():
    bp, _ = make_planners()
    skills = bp.get_skill_build(build_id="sb_assassin")
    skill_ids = [s[0] for s in skills]
    assert ("AS_SONICBLOW", 10) in skills
    assert ("AS_KATARMASTERY", 10) in skills
    print(f"  ✓ SB Assassin skills ({len(skills)}): {skill_ids}")


def test_get_skill_build_full_support_priest():
    bp, _ = make_planners()
    skills = bp.get_skill_build(build_id="full_support_priest")
    skill_ids = [s[0] for s in skills]
    assert ("AL_BLESSING", 10) in skills
    assert ("AL_HEAL", 10) in skills
    assert ("PR_KYRIE", 10) in skills
    print(f"  ✓ Full Support Priest skills ({len(skills)}): {skill_ids}")


def test_get_skill_build_agile_crit_sin():
    bp, _ = make_planners()
    skills = bp.get_skill_build(build_id="agile_crit_sin")
    skill_ids = [s[0] for s in skills]
    assert ("AS_KATARMASTERY", 10) in skills
    assert ("AS_RIGHTHANDMASTERY", 10) in skills
    assert ("AS_ENCHANTPOISON", 5) in skills
    print(f"  ✓ Agile Crit Sin skills ({len(skills)}): {skill_ids}")


# ═══════════════════════════════════════════════════════════════════
# 4. BuildPlanner — trap skills
# ═══════════════════════════════════════════════════════════════════

def test_trap_skills():
    bp, _ = make_planners()
    # Knight trap
    traps = bp.get_trap_skills(build_id="bowling_bash_knight")
    assert "SM_BASH" in traps
    print(f"  ✓ Bowling Bash Knight traps: {traps}")

    # Hunter trap
    traps = bp.get_trap_skills(build_id="falcon_hunter")
    assert "AC_CONCENTRATION" in traps
    print(f"  ✓ Falcon Hunter traps: {traps}")

    # Wizard trap
    traps = bp.get_trap_skills(build_id="ms_wizard")
    assert "MG_FROSTDIVER" in traps
    print(f"  ✓ MS Wizard traps: {traps}")

    # All assassin traps (aggregated)
    traps = bp.get_trap_skills_for_job("Assassin")
    assert "AS_GRIMTOOTH" in traps
    print(f"  ✓ Assassin traps: {traps}")


# ═══════════════════════════════════════════════════════════════════
# 5. BuildPlanner — breakpoints
# ═══════════════════════════════════════════════════════════════════

def test_get_all_breakpoints():
    bp, _ = make_planners()
    bps = bp.get_all_breakpoints()
    assert "DEX" in bps
    assert "AGI" in bps
    assert "INT" in bps
    assert "STR" in bps
    assert "VIT" in bps
    assert "LUK" in bps
    print(f"  ✓ All 6 stats have breakpoints: {list(bps.keys())}")


def test_get_breakpoint_info():
    bp, _ = make_planners()
    info = bp.get_breakpoint_info("DEX", 95)
    assert info is not None
    assert "instant cast" in info["description"].lower()
    print(f"  ✓ DEX 95 breakpoint: {info['description']}")

    # Below breakpoint — should return None (no breakpoint below value)
    info_low = bp.get_breakpoint_info("DEX", 10)
    assert info_low is None, f"Expected None for DEX 10, got {info_low}"
    print(f"  ✓ DEX 10 below first breakpoint → None")


def test_is_stat_breakpoint():
    bp, _ = make_planners()
    assert bp.is_stat_breakpoint("DEX", 95) is True
    assert bp.is_stat_breakpoint("DEX", 150) is True
    assert bp.is_stat_breakpoint("INT", 75) is True
    assert bp.is_stat_breakpoint("INT", 99) is True
    assert bp.is_stat_breakpoint("VIT", 100) is True
    assert bp.is_stat_breakpoint("STR", 99) is True
    assert bp.is_stat_breakpoint("LUK", 100) is True
    # Not a breakpoint
    assert bp.is_stat_breakpoint("DEX", 80) is False
    print(f"  ✓ All defined breakpoints correctly detected")


# ═══════════════════════════════════════════════════════════════════
# 6. StatBreakpointPlanner — advanced features
# ═══════════════════════════════════════════════════════════════════

def test_sp_get_target_stats():
    _, sp = make_planners()
    stats = sp.get_target_stats("Knight")
    assert stats is not None
    assert stats["STR"] == 80

    # Named build
    me_stats = sp.get_target_stats("Priest", build_name="ME Priest")
    assert me_stats is not None
    assert me_stats["INT"] == 90


def test_sp_get_all_breakpoints():
    _, sp = make_planners()
    bps = sp.get_all_breakpoints()
    assert bps["DEX"][95]["effect"] == "cast_time_zero"
    assert bps["VIT"][100]["effect"] == "stun_immunity"


def test_sp_breakpoint_info_returns_closest():
    _, sp = make_planners()
    # DEX 96: should return DEX 95 breakpoint info
    info = sp.get_breakpoint_info("DEX", 96)
    assert info is not None
    assert info["breakpoint_value"] == 95
    assert "instant cast" in info["description"].lower()

    # DEX 150: should return DEX 150
    info = sp.get_breakpoint_info("DEX", 150)
    assert info is not None
    assert info["breakpoint_value"] == 150


def test_sp_next_breakpoint():
    _, sp = make_planners()
    next_bp = sp.get_next_breakpoint("DEX", 80)
    assert next_bp is not None
    assert next_bp["breakpoint_value"] == 95
    assert next_bp["remaining"] == 15
    print(f"  ✓ DEX 80 → next breakpoint at 95 ({next_bp['description']})")

    # Already past all breakpoints
    past = sp.get_next_breakpoint("STR", 120)
    assert past is None
    print(f"  ✓ STR 120 past all breakpoints → None")


def test_sp_breakpoints_for_stat():
    _, sp = make_planners()
    bps = sp.get_breakpoints_for_stat("DEX")
    assert len(bps) == 2
    assert bps[0]["breakpoint_value"] == 95
    assert bps[1]["breakpoint_value"] == 150
    print(f"  ✓ DEX breakpoints: {[b['breakpoint_value'] for b in bps]}")


def test_sp_recommend_next_breakpoint_aware():
    _, sp = make_planners()

    # MS Wizard target: INT 99, DEX 50, VIT 40
    # INT has a breakpoint at 99 — same as the target
    target = {"STR": 1, "AGI": 1, "VIT": 40, "INT": 99, "DEX": 50, "LUK": 1}

    # INT at 95 (within 5 of both INT 99 breakpoint and target)
    # Should recommend INT because it's approaching the max-MATK breakpoint
    current = {"STR": 1, "AGI": 1, "VIT": 1, "INT": 95, "DEX": 1, "LUK": 1}
    rec = sp.recommend_next_stat(current, target)
    assert rec == "INT", f"Expected INT (imminent breakpoint at 99), got {rec}"
    print(f"  ✓ INT 95 (within 5 of breakpoint 99) → recommends INT")

    # Falcon Hunter target: DEX 90, AGI 50, INT 30
    # DEX has breakpoint at 95, but target is 90. So DEX at 86 is within 5 of target.
    # But DEX breakpoint at 95 is above target, so it won't trigger imminent-bp logic.
    # Instead, stats furthest from target: INT target 30, current 1 → gap 29
    target2 = {"STR": 1, "AGI": 50, "VIT": 1, "INT": 30, "DEX": 90, "LUK": 1}
    current2 = {"STR": 1, "AGI": 40, "VIT": 1, "INT": 1, "DEX": 86, "LUK": 1}
    rec2 = sp.recommend_next_stat(current2, target2)
    # DEX 86 is within 5 of DEX target 90. But DEX breakpoint at 95 > target 90, so bp skipped.
    # AGI needs 10 more, INT needs 29 more → should recommend INT (furthest gap)
    assert rec2 == "INT", f"Expected INT (furthest gap, breakpoint above target), got {rec2}"
    print(f"  ✓ DEX 86 (breakpoint above target) → recommends INT (highest remaining gap)")

    # Breakpoint not imminent, STR furthest
    current3 = {"STR": 1, "AGI": 1, "VIT": 1, "DEX": 1, "INT": 1, "LUK": 1}
    rec3 = sp.recommend_next_stat(current3, target2)
    assert rec3 == "DEX", f"Expected DEX (largest gap), got {rec3}"
    print(f"  ✓ Starting stats → recommends DEX (largest gap)")


def test_sp_get_skill_build():
    _, sp = make_planners()
    skills = sp.get_skill_build("Wizard")
    skill_ids = [s[0] for s in skills]
    assert "WZ_METEOR" in skill_ids
    assert "MG_FROSTDIVER" not in skill_ids  # trap

    # Named
    skills2 = sp.get_skill_build("Priest", build_name="Full Support Priest")
    skill_ids2 = [s[0] for s in skills2]
    assert "AL_BLESSING" in skill_ids2
    assert "PR_ASSUMPTIO" in skill_ids2

    print(f"  ✓ Wizard skills via StatBreakpointPlanner: {skill_ids}")


def test_sp_get_trap_skills():
    _, sp = make_planners()
    traps = sp.get_trap_skills("Assassin")
    assert "AS_GRIMTOOTH" in traps
    assert len(traps) == 1  # both assassin builds list grimtooth as trap

    traps_wizard = sp.get_trap_skills("Wizard")
    assert "MG_FROSTDIVER" in traps_wizard

    print(f"  ✓ Assassin traps: {traps}")


def test_sp_get_stat_summary():
    _, sp = make_planners()
    current = {"STR": 60, "AGI": 50, "VIT": 30, "INT": 90, "DEX": 70, "LUK": 10}
    summary = sp.get_stat_summary(current)
    assert len(summary) == 6

    dex_info = [s for s in summary if s["stat"] == "DEX"][0]
    assert dex_info["current_value"] == 70
    assert dex_info["next_breakpoint"] == 95
    assert dex_info["remaining_to_next"] == 25

    int_info = [s for s in summary if s["stat"] == "INT"][0]
    assert int_info["current_value"] == 90
    # INT breakpoints are 75 and 99: at 90 the closest >= is 75
    assert int_info["breakpoint_active"] is not None

    print(f"  ✓ Stat summary for all 6 stats: OK")


# ═══════════════════════════════════════════════════════════════════
# 7. Edge cases
# ═══════════════════════════════════════════════════════════════════

def test_empty_current_stats():
    bp, sp = make_planners()
    target = {"STR": 80, "AGI": 40, "VIT": 40, "DEX": 40, "INT": 1, "LUK": 1}

    # Empty current stats (brand new character)
    current: dict[str, int] = {}
    rec = bp.recommend_next_stat(current, target)
    assert rec is not None
    print(f"  ✓ Empty current stats → recommends {rec}")


def test_nonexistent_build():
    bp, _ = make_planners()
    assert bp.get_build_by_id("fnord") is None
    assert bp.get_target_stats(build_id="fnord") is None
    assert bp.get_skill_build(build_id="fnord") == []
    assert bp.get_trap_skills(build_id="fnord") == []
    print(f"  ✓ Nonexistent build handled gracefully")


def test_nonexistent_job():
    _, sp = make_planners()
    assert sp.get_target_stats("Bard") is None
    assert sp.get_skill_build("Bard") == []
    assert sp.get_trap_skills("Bard") == []
    print(f"  ✓ Nonexistent job handled gracefully")


# ═══════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("RO Build Planner & Stat Breakpoint Planner — Verification")
    print("=" * 60)
    print()

    total = 0
    passed = 0
    failed = 0

    # Gather all test functions
    test_fns = [
        v for k, v in globals().items()
        if k.startswith("test_") and callable(v)
    ]
    # Sort by name for consistent order
    test_fns.sort(key=lambda f: f.__name__)

    for fn in test_fns:
        total += 1
        try:
            fn()
            passed += 1
        except Exception as e:
            failed += 1
            import traceback
            print(f"  ✗ {fn.__name__}: {e}")
            traceback.print_exc()

    print()
    print("=" * 60)
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f", {failed} FAILED", end="")
    print()
    if failed:
        print("  ❌ SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("  ✅ ALL TESTS PASSED")
    print("=" * 60)
