"""Tests for ServerAdapter, EXPObserver, DropObserver, StrategyAdjuster.

Validates that:
1. EXPObserver converges to the correct EXP multiplier
2. DropObserver converges to the correct drop multiplier
3. ServerAdapter correctly detects server type from damage observations
4. StrategyAdjuster produces appropriate adjustments for each rate/profile
5. Confidence grows with observation count
6. Edge cases (no observations, single observation, zero values)
"""
from __future__ import annotations

import math
import sys
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ai_sidecar.domains.world.server_adapter import (
    EXPObserver,
    DropObserver,
    ServerAdapter,
    ServerProfile,
    ServerType,
    ServerRateCategory,
    StrategyAdjuster,
    StrategyAdjustment,
    reference_base_exp,
    pre_renewal_damage_taken,
    renewal_damage_taken,
)


# ═══════════════════════════════════════════════════════════════
# 1. EXPObserver Tests
# ═══════════════════════════════════════════════════════════════

def test_exp_observer_initial_state():
    obs = EXPObserver()
    assert obs.get_mult_estimate() == 1.0
    assert obs.confidence() == 0.0
    assert obs.observation_count() == 0
    print("  ✓ EXPObserver initial state: mult=1.0, confidence=0.0")


def test_exp_observer_single_kill():
    obs = EXPObserver()
    # Kill a Lv20 monster on a 10x server
    reference = reference_base_exp(20)
    actual = reference * 10
    obs.observe_kill(monster_level=20, base_exp_gained=actual)
    est = obs.get_mult_estimate()
    assert est == 10.0, f"Expected 10.0, got {est}"
    # Confidence should be low with 1 observation
    assert obs.confidence() == 0.0, f"Expected 0.0 confidence, got {obs.confidence()}"
    assert obs.observation_count() == 1
    print(f"  ✓ Single kill 10x server: est={est}, confidence=0.0")


def test_exp_observer_convergence():
    """Core convergence test: EXPObserver should approach true multiplier."""
    true_mult = 25.0

    for smoothing in [0.05, 0.15, 0.30]:
        obs = EXPObserver(smoothing_alpha=smoothing)
        # Simulate 100 kills at various levels
        for _ in range(100):
            lv = random.randint(10, 80)
            expected = reference_base_exp(lv)
            # Add ±15% noise to simulate real variance
            noise = random.gauss(0, 0.15 * expected)
            actual = max(1, expected * true_mult + noise)
            obs.observe_kill(monster_level=lv, base_exp_gained=int(actual))

        est = obs.get_mult_estimate()
        error_pct = abs(est - true_mult) / true_mult * 100
        conf = obs.confidence()

        print(f"  ✓ EXPObserver convergence (α={smoothing}): true={true_mult}x, "
              f"est={est:.2f}x, error={error_pct:.1f}%, confidence={conf:.3f}, "
              f"obs={obs.observation_count()}")
        # After 100 kills, estimate should be within 20%
        assert error_pct < 20, f"Convergence error too high: {error_pct}%"
        # Confidence should be > 0 after 5+ observations
        assert conf > 0.0, "Confidence should be >0 after 100 observations"


def test_exp_observer_low_rate_convergence():
    """Verify detection works at low rates (1x) too."""
    obs = EXPObserver(smoothing_alpha=0.1)
    true_mult = 1.0

    for _ in range(80):
        lv = random.randint(5, 50)
        expected = reference_base_exp(lv)
        noise = random.gauss(0, 0.1 * expected)
        actual = max(1, expected * true_mult + noise)
        obs.observe_kill(monster_level=lv, base_exp_gained=int(actual))

    est = obs.get_mult_estimate()
    error_pct = abs(est - true_mult) / true_mult * 100
    print(f"  ✓ Low rate (1x) detection: est={est:.2f}x, error={error_pct:.1f}%")
    assert error_pct < 15, f"1x convergence error too high: {error_pct}%"


def test_exp_observer_extreme_rate_convergence():
    """Verify detection works at extreme rates (500x)."""
    obs = EXPObserver(smoothing_alpha=0.1)
    true_mult = 500.0

    for _ in range(60):
        lv = random.randint(5, 60)
        expected = reference_base_exp(lv)
        noise = random.gauss(0, 0.1 * expected)
        actual = max(1, expected * true_mult + noise)
        obs.observe_kill(monster_level=lv, base_exp_gained=int(actual))

    est = obs.get_mult_estimate()
    error_pct = abs(est - true_mult) / true_mult * 100
    print(f"  ✓ Extreme rate (500x) detection: est={est:.2f}x, error={error_pct:.1f}%")
    assert error_pct < 15, f"500x convergence error too high: {error_pct}%"


def test_exp_observer_confidence_growth():
    """Confidence should increase with more observations."""
    obs = EXPObserver()
    confidences = []
    for i in range(1, 51):
        lv = 30
        expected = reference_base_exp(lv)
        obs.observe_kill(monster_level=lv, base_exp_gained=int(expected * 10))
        confidences.append((i, obs.confidence()))

    # Confidence should be monotonically non-decreasing (or at least not dropping)
    # after the min_observations threshold
    print(f"  ✓ Confidence after 10 obs: {confidences[9][1]:.3f}")
    print(f"  ✓ Confidence after 50 obs: {confidences[49][1]:.3f}")
    assert confidences[49][1] >= confidences[9][1], "Confidence should increase with data"
    assert confidences[49][1] > 0.5, f"After 50 obs, confidence should be > 0.5, got {confidences[49][1]}"


def test_exp_observer_zero_or_negative():
    """Edge case: zero or negative EXP observations should be ignored."""
    obs = EXPObserver()
    obs.observe_kill(monster_level=20, base_exp_gained=0)
    assert obs.observation_count() == 0, "Zero EXP should not be counted"

    obs.observe_kill(monster_level=20, base_exp_gained=-100)
    assert obs.observation_count() == 0, "Negative EXP should not be counted"

    # Normal data should still work
    obs.observe_kill(monster_level=20, base_exp_gained=reference_base_exp(20) * 10)
    assert obs.observation_count() == 1
    print("  ✓ Zero/negative EXP correctly ignored")


def test_exp_observer_per_level_tracking():
    """Per-level multiplier breakdown should be available."""
    obs = EXPObserver()
    for lv in [10, 20, 30, 40]:
        for _ in range(5):
            expected = reference_base_exp(lv)
            obs.observe_kill(monster_level=lv, base_exp_gained=int(expected * 15))

    per_level = obs.per_level_multipliers()
    assert len(per_level) == 4, f"Expected 4 levels tracked, got {len(per_level)}"
    for lv, mult in per_level.items():
        assert abs(mult - 15.0) < 0.5, f"Level {lv}: expected ~15x, got {mult:.2f}"
    print(f"  ✓ Per-level tracking: {per_level}")


# ═══════════════════════════════════════════════════════════════
# 2. DropObserver Tests
# ═══════════════════════════════════════════════════════════════

def test_drop_observer_initial_state():
    obs = DropObserver()
    assert obs.get_mult_estimate() == 1.0
    assert obs.confidence() == 0.0
    print("  ✓ DropObserver initial state: mult=1.0, confidence=0.0")


def test_drop_observer_simple():
    """Basic drop recording with known expected rate."""
    obs = DropObserver()
    # Simulate 100 Poring kills with rare drops (5% base rate) on a 2x server
    # Expected: ~5% * 2 = ~10% observed rate → drives mult ~2.0
    for _ in range(200):
        obs.record_kill("Poring")
        if random.random() < 0.05 * 2:
            obs.record_drop("Poring", "Card", category="rare")

    est = obs.get_mult_estimate()
    print(f"  ✓ DropObserver simple: est={est:.2f}x")
    assert est > 1.0, f"Expected est > 1.0 for 2x drops, got {est}"
    assert est < 10.0, f"Expected sane estimate, got {est}"


def test_drop_observer_convergence():
    """DropObserver should converge towards true drop multiplier.

    Uses rare-category drops (5% base rate) to avoid the 100% ceiling effect
    that occurs when base_rate * mult > 1.0.
    """
    true_mult = 5.0
    obs = DropObserver()

    # Simulate 1000 kills with rare drops (5% base rate * 5x = 25% observed)
    for _ in range(1000):
        monster = random.choice(["Poring", "Fabre", "Lunatic", "PecoPeco"])
        obs.record_kill(monster)
        if random.random() < 0.05 * true_mult:
            obs.record_drop(monster, "rare_item", category="rare")

    est = obs.get_mult_estimate()
    error_pct = abs(est - true_mult) / true_mult * 100
    conf = obs.confidence()
    print(f"  ✓ DropObserver convergence: true={true_mult}x, "
          f"est={est:.2f}x, error={error_pct:.1f}%, confidence={conf:.3f}")
    assert error_pct < 30, f"Drop convergence too high: {error_pct}%"


def test_drop_observer_per_monster():
    """Per-monster multiplier estimates should work given enough data."""
    obs = DropObserver()
    # 500 kills of Poring with 2x drops on common (capped at 1.0 observed)
    # Since per_monster_drop_multiplier uses "common" as baseline, we keep
    # the multiplier low enough that observed rate < 1.0 for reliable estimate
    # At 1.5x on 0.55 common: 0.825 < 1.0 ✓
    _true_mult = 1.5
    for _ in range(500):
        obs.record_kill("Poring")
        if random.random() < 0.55 * _true_mult:
            obs.record_drop("Poring", "Apple", category="common")

    mult = obs.per_monster_drop_multiplier("Poring")
    if mult is None:
        obs.update_aggregate_estimate()
        mult = obs.per_monster_drop_multiplier("Poring")
    print(f"  ✓ Per-monster multiplier for Poring: {mult} (true={_true_mult})")
    assert mult is not None, "Should have per-monster estimate after 500 kills"
    assert mult > 1.0, f"Expected mult > 1.0, got {mult}"
    # Should converge within 50% given the variance
    error_pct = abs(mult - _true_mult) / _true_mult * 100
    assert error_pct < 50, f"Per-monster error too high: {error_pct}%"


# ═══════════════════════════════════════════════════════════════
# 3. ServerAdapter Tests
# ═══════════════════════════════════════════════════════════════

def test_server_adapter_initial():
    adapter = ServerAdapter()
    profile = adapter.get_profile()
    assert profile.server_type == ServerType.UNKNOWN
    assert profile.exp_multiplier == 1.0
    assert profile.drop_multiplier == 1.0
    assert profile.exp_confidence == 0.0
    print("  ✓ ServerAdapter initial state: all defaults")


def test_server_adapter_full_high_rate():
    """End-to-end: 50x EXP, 10x drops, pre-renewal."""
    adapter = ServerAdapter()
    true_exp_mult = 50.0
    true_drop_mult = 10.0

    # EXP observations (50 kills across various levels)
    for _ in range(80):
        lv = random.randint(15, 70)
        expected = reference_base_exp(lv)
        noise = random.gauss(0, 0.12 * expected)
        actual = max(1, expected * true_exp_mult + noise)
        adapter.observe_exp_kill(monster_level=lv, base_exp_gained=int(actual))

    # Drop observations (300 kills)
    for _ in range(300):
        monster = random.choice(["Poring", "Fabre", "Lunatic"])
        adapter.observe_kill_no_drop(monster)
        if random.random() < 0.55 * true_drop_mult:
            adapter.observe_drop(monster, "common_item", category="common")

    # Damage observations (pre-renewal server — formula matches)
    for _ in range(20):
        defence = random.randint(5, 80)
        base = 200
        actual_damage = int(base * pre_renewal_damage_taken(defence))
        adapter.observe_damage(monster_defence=defence, observed_damage=actual_damage,
                               estimated_base_damage=base)

    profile = adapter.get_profile()

    # EXP rate should be detected
    assert profile.exp_confidence > 0.5, f"EXP confidence too low: {profile.exp_confidence}"
    exp_error = abs(profile.exp_multiplier - true_exp_mult) / true_exp_mult * 100
    print(f"  ✓ High-rate EXP: true={true_exp_mult}x, est={profile.exp_multiplier:.1f}x "
          f"({exp_error:.1f}% error), conf={profile.exp_confidence:.3f}")
    assert exp_error < 30, f"EXP error too high: {exp_error}%"

    # Server type should be pre-renewal
    print(f"  ✓ Server type: {profile.server_type} (conf={profile.type_confidence:.3f})")
    assert profile.server_type == ServerType.PRE_RENEWAL, \
        f"Expected PRE_RENEWAL, got {profile.server_type}"

    # Rate category
    print(f"  ✓ Rate category: {profile.rate_category}")
    assert profile.rate_category == ServerRateCategory.HIGH_RATE

    summary = adapter.summary()
    print(f"  ✓ Summary: {summary['profile']}")


def test_server_adapter_renewal_detection():
    """ServerAdapter should detect renewal server from damage observations."""
    adapter = ServerAdapter()

    # Damage observations matching renewal formula
    for _ in range(30):
        defence = random.randint(10, 200)
        base = 300
        actual_damage = int(base * renewal_damage_taken(defence))
        adapter.observe_damage(monster_defence=defence, observed_damage=actual_damage,
                               estimated_base_damage=base)

    profile = adapter.get_profile()
    print(f"  ✓ Renewal detection: type={profile.server_type}, "
          f"conf={profile.type_confidence:.3f}")
    assert profile.server_type == ServerType.RENEWAL, \
        f"Expected RENEWAL, got {profile.server_type}"
    assert profile.type_confidence >= 0.5, f"Type confidence too low: {profile.type_confidence}"


def test_server_adapter_pre_renewal_detection():
    """ServerAdapter should detect pre-renewal server from damage observations."""
    adapter = ServerAdapter()

    for _ in range(30):
        defence = random.randint(5, 100)
        base = 250
        actual_damage = int(base * pre_renewal_damage_taken(defence))
        adapter.observe_damage(monster_defence=defence, observed_damage=actual_damage,
                               estimated_base_damage=base)

    profile = adapter.get_profile()
    print(f"  ✓ Pre-renewal detection: type={profile.server_type}, "
          f"conf={profile.type_confidence:.3f}")
    assert profile.server_type == ServerType.PRE_RENEWAL, \
        f"Expected PRE_RENEWAL, got {profile.server_type}"


def test_server_adapter_low_rate():
    """End-to-end: 1x EXP, low drops, pre-renewal."""
    adapter = ServerAdapter()

    for _ in range(100):
        lv = random.randint(5, 40)
        expected = reference_base_exp(lv)
        noise = random.gauss(0, 0.1 * expected)
        actual = max(1, expected * 1.0 + noise)
        adapter.observe_exp_kill(monster_level=lv, base_exp_gained=int(actual))

    for _ in range(200):
        monster = random.choice(["Poring", "Fabre"])
        adapter.observe_kill_no_drop(monster)
        if random.random() < 0.55:  # 1x drops
            adapter.observe_drop(monster, "item", category="common")

    profile = adapter.get_profile()
    exp_error = abs(profile.exp_multiplier - 1.0) / 1.0 * 100
    print(f"  ✓ Low rate (1x): est={profile.exp_multiplier:.2f}x, "
          f"error={exp_error:.1f}%, conf={profile.exp_confidence:.3f}")
    assert profile.rate_category == ServerRateCategory.LOW_RATE, \
        f"Expected LOW_RATE, got {profile.rate_category}"
    assert exp_error < 15, f"Low rate detection error too high: {exp_error}%"


def test_server_adapter_custom_mechanics():
    """Custom item/NPC/warp detection flags."""
    adapter = ServerAdapter()

    adapter.check_standard_item("Red_Potion")
    adapter.report_missing_item("White_Potion")
    adapter.report_missing_item("Blue_Potion")
    adapter.report_custom_npc()
    adapter.report_custom_warp()

    profile = adapter.get_profile()
    assert profile.has_custom_items, "Should detect custom items"
    assert profile.has_custom_npcs, "Should detect custom NPCs"
    assert profile.has_custom_warps, "Should detect custom warps"

    print(f"  ✓ Custom mechanics: items={profile.has_custom_items}, "
          f"npcs={profile.has_custom_npcs}, warps={profile.has_custom_warps}")


def test_server_adapter_insufficient_data():
    """With very few observations, confidence should be low."""
    adapter = ServerAdapter()
    adapter.observe_exp_kill(monster_level=30, base_exp_gained=300)
    adapter.observe_drop("Poring", "Apple", category="common")

    profile = adapter.get_profile()
    assert profile.exp_confidence == 0.0
    assert profile.server_type == ServerType.UNKNOWN
    assert profile.rate_category == ServerRateCategory.UNKNOWN
    print("  ✓ Insufficient data: low confidence, unknown type/rate")


# ═══════════════════════════════════════════════════════════════
# 4. StrategyAdjuster Tests
# ═══════════════════════════════════════════════════════════════

def test_adjuster_low_rate():
    """Low-rate profile should produce frugal strategy."""
    adjuster = StrategyAdjuster()
    profile = ServerProfile(
        server_type=ServerType.PRE_RENEWAL,
        exp_multiplier=1.0,
        drop_multiplier=1.0,
        exp_confidence=0.9,
        drop_confidence=0.8,
        type_confidence=0.85,
    )
    adj = adjuster.adjust(profile)
    assert adj.farm_frugally is True
    assert adj.allow_vending is True
    assert adj.skip_levels_under == 0
    assert adj.buy_equipment_at_level == 40
    assert adj.damage_formula == "pre_renewal"
    assert adj.item_valuation_bias == "npc"
    print(f"  ✓ Low-rate strategy: frugal={adj.farm_frugally}, "
          f"vending={adj.allow_vending}, buy_at={adj.buy_equipment_at_level}")


def test_adjuster_high_rate():
    """High-rate profile should produce aggressive, skip-low strategy."""
    adjuster = StrategyAdjuster()
    profile = ServerProfile(
        server_type=ServerType.PRE_RENEWAL,
        exp_multiplier=50.0,
        drop_multiplier=10.0,
        exp_confidence=0.9,
        drop_confidence=0.8,
        type_confidence=0.85,
    )
    adj = adjuster.adjust(profile)
    assert adj.farm_frugally is False
    assert adj.allow_vending is False
    assert adj.skip_levels_under == 15
    assert adj.buy_equipment_at_level == 1
    assert adj.grind_efficiency_mode is True
    assert adj.item_valuation_bias == "market"
    print(f"  ✓ High-rate strategy: skip_under={adj.skip_levels_under}, "
          f"buy_at={adj.buy_equipment_at_level}, "
          f"grind_efficiency={adj.grind_efficiency_mode}")


def test_adjuster_extreme_rate():
    """Extreme-rate profile should max level ASAP."""
    adjuster = StrategyAdjuster()
    profile = ServerProfile(
        server_type=ServerType.RENEWAL,
        exp_multiplier=500.0,
        drop_multiplier=50.0,
        exp_confidence=0.9,
        drop_confidence=0.8,
        type_confidence=0.9,
    )
    adj = adjuster.adjust(profile)
    assert adj.skip_levels_under == 40
    assert adj.buy_equipment_at_level == 1
    assert adj.target_level_before_gear == 99
    assert adj.keep_minimum_items == 1
    assert adj.npc_price_tolerance == 5.0
    assert adj.damage_formula == "renewal"
    assert adj.priority_stat == "agi"
    print(f"  ✓ Extreme-rate strategy: skip_under={adj.skip_levels_under}, "
          f"target_level={adj.target_level_before_gear}, stat={adj.priority_stat}")


def test_adjuster_medium_rate():
    """Medium-rate should balance all strategies."""
    adjuster = StrategyAdjuster()
    profile = ServerProfile(
        server_type=ServerType.PRE_RENEWAL,
        exp_multiplier=10.0,
        drop_multiplier=5.0,
        exp_confidence=0.9,
        drop_confidence=0.8,
        type_confidence=0.0,
    )
    adj = adjuster.adjust(profile)
    assert adj.skip_levels_under == 5
    assert adj.buy_equipment_at_level == 25
    assert adj.farm_frugally is True
    print(f"  ✓ Medium-rate strategy: skip_under={adj.skip_levels_under}, "
          f"buy_at={adj.buy_equipment_at_level}")


def test_adjuster_low_confidence():
    """Low-confidence profile should produce default (sane) adjustments."""
    adjuster = StrategyAdjuster()
    profile = ServerProfile(
        server_type=ServerType.UNKNOWN,
        exp_multiplier=1.0,
        drop_multiplier=1.0,
        exp_confidence=0.0,
        drop_confidence=0.0,
        type_confidence=0.0,
    )
    adj = adjuster.adjust(profile)
    # Should keep defaults — not crash
    assert adj.damage_formula == "pre_renewal"
    assert adj.farm_frugally is True
    print(f"  ✓ Low-confidence strategy: default values preserved")


def test_adjuster_renewal_strategy():
    """Renewal profile should set damage_formula to renewal."""
    adjuster = StrategyAdjuster()
    profile = ServerProfile(
        server_type=ServerType.RENEWAL,
        exp_multiplier=15.0,
        drop_multiplier=3.0,
        exp_confidence=0.8,
        drop_confidence=0.7,
        type_confidence=0.9,
    )
    adj = adjuster.adjust(profile)
    assert adj.damage_formula == "renewal"
    assert adj.favor_elemental_damage is True
    print(f"  ✓ Renewal strategy: formula={adj.damage_formula}, "
          f"elemental={adj.favor_elemental_damage}")


def test_adjuster_produce_actions():
    """Produce actions should generate HeuristicAction-like dicts."""
    adjuster = StrategyAdjuster()
    profile = ServerProfile(
        server_type=ServerType.PRE_RENEWAL,
        exp_multiplier=50.0,
        drop_multiplier=10.0,
        exp_confidence=0.9,
        drop_confidence=0.8,
        type_confidence=0.85,
    )
    actions = adjuster.produce_actions(profile)
    assert len(actions) >= 3, f"Expected at least 3 actions, got {len(actions)}"

    # Verify action structure
    for action in actions:
        assert hasattr(action, "kind")
        assert hasattr(action, "command")
        assert hasattr(action, "confidence")
        assert hasattr(action, "domain")
        assert action.domain == "server_adapter"

    # Should have cold-start, equipment, combat, and economy commands
    commands = [a.command for a in actions]
    print(f"  ✓ Produced {len(actions)} actions: {commands[:4]}")
    assert any("skip_under" in c for c in commands), "Missing cold_start skip_under"
    assert any("formula" in c for c in commands), "Missing combat formula command"
    assert any("bias" in c for c in commands), "Missing economy adjustment"


# ═══════════════════════════════════════════════════════════════
# 5. Formula Correctness Tests
# ═══════════════════════════════════════════════════════════════

def test_pre_renewal_damage_formula():
    """Pre-renewal formula should give expected values at known DEF points."""
    # At DEF=0: (0 + 2) / (0 + 2) = 1.0
    assert abs(pre_renewal_damage_taken(0) - 1.0) < 0.001
    # At DEF=10: (5+2)/(10+2) = 7/12 ≈ 0.583
    assert abs(pre_renewal_damage_taken(10) - 7/12) < 0.001
    # At DEF=100: (50+2)/(100+2) = 52/102 ≈ 0.510
    assert abs(pre_renewal_damage_taken(100) - 52/102) < 0.001
    # High DEF asymptotes at about 0.5
    assert 0.50 < pre_renewal_damage_taken(1000) < 0.51
    print("  ✓ Pre-renewal damage formula correct")


def test_renewal_damage_formula():
    """Renewal formula should give expected values at known DEF points."""
    # At DEF=0: 300/300 = 1.0
    assert abs(renewal_damage_taken(0) - 1.0) < 0.001
    # At DEF=300: 300/600 = 0.5
    assert abs(renewal_damage_taken(300) - 0.5) < 0.001
    # At DEF=600: 300/900 = 0.333...
    assert abs(renewal_damage_taken(600) - 300/900) < 0.001
    # At DEF=100: 300/400 = 0.75
    assert abs(renewal_damage_taken(100) - 0.75) < 0.001
    print("  ✓ Renewal damage formula correct")


def test_formulas_divergence():
    """Pre-renewal and renewal formulas should give different enough values for detection."""
    # At moderate DEF, the difference should be detectable
    for def_value in [20, 40, 60, 80, 100]:
        pre = pre_renewal_damage_taken(def_value)
        ren = renewal_damage_taken(def_value)
        diff = abs(pre - ren)
        assert diff > 0.05, f"At DEF={def_value}: pre={pre:.4f}, ren={ren:.4f}, diff={diff:.4f} — too close for reliable detection"
    print("  ✓ Formulas diverge sufficiently for detection")


def test_reference_base_exp():
    """Reference base EXP should be monotonic with monster level."""
    prev = 0
    for lv in range(1, 100):
        exp = reference_base_exp(lv)
        assert exp > prev, f"EXP not monotonic at Lv{lv}: {exp} <= {prev}"
        prev = exp
    # Sanity check: Lv99 monster should give reasonable EXP on 1x server
    exp99 = reference_base_exp(99)
    assert 2000 < exp99 < 20000, \
        f"Reference EXP for Lv99 out of range: {exp99}"
    print(f"  ✓ Reference EXP: Lv1={reference_base_exp(1)}, "
          f"Lv50={reference_base_exp(50)}, Lv99={reference_base_exp(99)} "
          f"— monotonic and sane")


# ═══════════════════════════════════════════════════════════════
# 6. ServerProfile Tests
# ═══════════════════════════════════════════════════════════════

def test_profile_rate_category():
    """Rate category should correctly classify multipliers."""
    assert ServerProfile(exp_multiplier=1.0, exp_confidence=0.9).rate_category == ServerRateCategory.LOW_RATE
    assert ServerProfile(exp_multiplier=10.0, exp_confidence=0.9).rate_category == ServerRateCategory.MEDIUM_RATE
    assert ServerProfile(exp_multiplier=50.0, exp_confidence=0.9).rate_category == ServerRateCategory.HIGH_RATE
    assert ServerProfile(exp_multiplier=500.0, exp_confidence=0.9).rate_category == ServerRateCategory.EXTREME_RATE
    assert ServerProfile(exp_multiplier=50.0, exp_confidence=0.0).rate_category == ServerRateCategory.UNKNOWN
    print("  ✓ Rate category classification correct")


def test_profile_summary():
    """Summary should contain all expected keys."""
    profile = ServerProfile(
        server_type=ServerType.RENEWAL,
        exp_multiplier=25.0,
        drop_multiplier=5.0,
        exp_confidence=0.9,
        drop_confidence=0.8,
        type_confidence=0.85,
    )
    s = profile.summary()
    assert "server_type" in s
    assert "exp_multiplier" in s
    assert "drop_multiplier" in s
    assert "rate_category" in s
    assert "confidence" in s
    assert "custom" in s
    assert "observations" in s
    # Verify serializable types
    import json
    json.dumps(s)  # should not raise
    print(f"  ✓ Profile summary JSON-serializable: {s['server_type']} @ {s['exp_multiplier']}x")


# ═══════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_fns = [
        # ── Formula correctness ──
        test_pre_renewal_damage_formula,
        test_renewal_damage_formula,
        test_formulas_divergence,
        test_reference_base_exp,

        # ── EXPObserver ──
        test_exp_observer_initial_state,
        test_exp_observer_single_kill,
        test_exp_observer_zero_or_negative,
        test_exp_observer_confidence_growth,
        test_exp_observer_per_level_tracking,
        test_exp_observer_low_rate_convergence,
        test_exp_observer_convergence,
        test_exp_observer_extreme_rate_convergence,

        # ── DropObserver ──
        test_drop_observer_initial_state,
        test_drop_observer_simple,
        test_drop_observer_convergence,
        test_drop_observer_per_monster,

        # ── ServerAdapter ──
        test_server_adapter_initial,
        test_server_adapter_insufficient_data,
        test_server_adapter_pre_renewal_detection,
        test_server_adapter_renewal_detection,
        test_server_adapter_low_rate,
        test_server_adapter_custom_mechanics,
        test_server_adapter_full_high_rate,

        # ── ServerProfile ──
        test_profile_rate_category,
        test_profile_summary,

        # ── StrategyAdjuster ──
        test_adjuster_low_confidence,
        test_adjuster_low_rate,
        test_adjuster_medium_rate,
        test_adjuster_high_rate,
        test_adjuster_extreme_rate,
        test_adjuster_renewal_strategy,
        test_adjuster_produce_actions,
    ]

    passed = 0
    failed = 0
    for fn in test_fns:
        try:
            fn()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  ✗ FAIL: {fn.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'═' * 50}")
    print(f"Results: {passed} passed, {failed} failed, {len(test_fns)} total")
    if failed > 0:
        sys.exit(1)
