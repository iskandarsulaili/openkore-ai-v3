"""
Synthetic reflex pipeline test — verifies highfreq_reflex fires under combat conditions.

This test feeds synthetic snapshots directly into check_and_act() without
needing a live bot. It validates the entire reflex pipeline:
  snapshot → check_and_act() → emit_direct() → action queue

Run: python -m pytest AI_sidecar/tests/test_reflex_pipeline.py -v
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root / "AI_sidecar"))

from ai_sidecar.reflex.highfreq_reflex import HighFreqReflex, DEFAULT_THRESHOLDS
from ai_sidecar.reflex.reflex_pipeline import ReflexPipeline
from ai_sidecar.reflex.healing_optimizer import HealingOptimizer


class MockActionQueue:
    """Minimal action queue mock that records what was enqueued."""
    
    def __init__(self):
        self.actions: list[dict] = []
        self.accepted = True
    
    def enqueue(self, bot_id: str, proposal: dict) -> tuple[bool, str, str, str]:
        self.actions.append({"proposal": proposal, "bot_id": bot_id})
        return (self.accepted, "queued", f"test-{len(self.actions)}", "ok")


def test_reflex_fires_at_low_hp():
    """HP ≤ 50% in combat → should return a heal command."""
    reflex = HighFreqReflex()
    cmd = reflex.check_and_act(
        bot_id="test_bot", hp=300, max_hp=1000,
        sp=100, max_sp=500, aggro_count=3,
        is_dead=False, is_town=False, has_potions=True,
        current_map="gef_fild01", zeny=50000, level=50,
    )
    assert cmd is not None, "check_and_act should return a command at 30% HP"
    assert "use" in cmd or "ai manual" in cmd, f"Expected heal/escape command, got: {cmd}"
    print(f"  PASS: low HP → '{cmd}'")


def test_reflex_does_not_fire_at_high_hp():
    """HP ≥ 90% with no aggro → should return None."""
    reflex = HighFreqReflex()
    cmd = reflex.check_and_act(
        bot_id="test_bot", hp=950, max_hp=1000,
        sp=400, max_sp=500, aggro_count=0,
        is_dead=False, is_town=False, has_potions=True,
        current_map="gef_fild01", zeny=50000, level=50,
    )
    assert cmd is None, f"check_and_act should return None at 95% HP, got: {cmd}"
    print("  PASS: high HP → None")


def test_reflex_escape_at_critical_hp():
    """HP ≤ 15% in combat → should return escape command."""
    reflex = HighFreqReflex()
    cmd = reflex.check_and_act(
        bot_id="test_bot", hp=100, max_hp=1000,
        sp=100, max_sp=500, aggro_count=5,
        is_dead=False, is_town=False, has_potions=True,
        current_map="gef_fild01", zeny=50000, level=50,
    )
    assert cmd is not None, "check_and_act should return a command at 10% HP"
    # Escape must NOT use 'ai manual' (freezes auto-attack forever, blocking EXP).
    # It stops combat via attackAuto 0 so the bot can flee while staying navigable.
    assert "ai manual" not in cmd, f"Escape must not freeze AI, got: {cmd}"
    assert "attackAuto 0" in cmd, f"Expected escape to stop combat, got: {cmd}"
    print(f"  PASS: critical HP → '{cmd}'")


def test_reflex_does_not_fire_when_dead():
    """Dead bot → should return None regardless of HP."""
    reflex = HighFreqReflex()
    cmd = reflex.check_and_act(
        bot_id="test_bot", hp=0, max_hp=1000,
        sp=0, max_sp=500, aggro_count=0,
        is_dead=True, is_town=False, has_potions=True,
        current_map="gef_fild01", zeny=50000, level=50,
    )
    assert cmd is None, f"check_and_act should return None when dead, got: {cmd}"
    print("  PASS: dead → None")


def test_reflex_emit_direct_through_pipeline():
    """emit_direct() should push action through the pipeline."""
    pipeline = ReflexPipeline()
    queue = MockActionQueue()
    pipeline._action_queue = queue
    
    pipeline.emit_direct("test_bot", "use Red Potion")
    
    assert len(queue.actions) == 1, f"Expected 1 action queued, got {len(queue.actions)}"
    action = queue.actions[0]
    assert action["bot_id"] == "test_bot"
    assert action["proposal"].command == "use Red Potion"
    print(f"  PASS: emit_direct → action queued: {action['proposal'].command}")


def test_healing_optimizer_selects_correct_potion():
    """HealingOptimizer should select level-appropriate potions, not food items."""
    opt = HealingOptimizer()
    loaded = opt.load()
    assert loaded, "HealingOptimizer should load healing items"
    
    # Low level → should select a real potion, not Monster Bread
    cmd = opt.select_healing_command(
        hp=50, max_hp=100, sp=50, max_sp=100,
        zeny=1000, level=10,
    )
    assert cmd is not None, "Should select a healing item for low level"
    assert "use " in cmd, f"Expected 'use <item>', got: {cmd}"
    assert "bread" not in cmd.lower(), f"Should not select food items, got: {cmd}"
    assert "food" not in cmd.lower(), f"Should not select food items, got: {cmd}"
    print(f"  PASS: low level heal → '{cmd}'")
    
    # High level → should select something stronger
    cmd2 = opt.select_healing_command(
        hp=500, max_hp=5000, sp=500, max_sp=5000,
        zeny=100000, level=99,
    )
    assert cmd2 is not None, "Should select a healing item for high level"
    assert "use " in cmd2, f"Expected 'use <item>', got: {cmd2}"
    assert "bread" not in cmd2.lower(), f"Should not select food items, got: {cmd2}"
    print(f"  PASS: high level heal → '{cmd2}'")
    
    # Emergency (10% HP) → should select strongest available, not cheapest
    cmd3 = opt.select_healing_command(
        hp=100, max_hp=1000, sp=50, max_sp=500,
        zeny=50000, level=60,
    )
    assert cmd3 is not None, "Should select a healing item for emergency"
    # In combat mode, should prefer high-heal items over cheap ones
    print(f"  PASS: emergency heal → '{cmd3}'")


def test_reflex_cooldown_respects_timer():
    """Reflex should not fire again within cooldown period."""
    reflex = HighFreqReflex()
    
    # First call should fire
    cmd1 = reflex.check_and_act(
        bot_id="test_bot", hp=300, max_hp=1000,
        sp=100, max_sp=500, aggro_count=3,
        is_dead=False, is_town=False, has_potions=True,
        current_map="gef_fild01", zeny=50000, level=50,
    )
    assert cmd1 is not None, "First call should fire"
    
    # Immediate second call should NOT fire (cooldown)
    cmd2 = reflex.check_and_act(
        bot_id="test_bot", hp=200, max_hp=1000,
        sp=100, max_sp=500, aggro_count=3,
        is_dead=False, is_town=False, has_potions=True,
        current_map="gef_fild01", zeny=50000, level=50,
    )
    assert cmd2 is None, "Second call within cooldown should return None"
    print("  PASS: cooldown respected")


def test_reflex_stats_tracking():
    """Reflex should track check/action/miss counts."""
    reflex = HighFreqReflex()
    
    # Fire a few times with different bot IDs
    reflex.check_and_act("bot_a", 300, 1000, 100, 500, 3, False, False, True, "map", zeny=50000, level=50)
    reflex.check_and_act("bot_b", 300, 1000, 100, 500, 3, False, False, True, "map", zeny=50000, level=50)
    
    stats = reflex.get_stats()
    assert stats.get("actions", 0) >= 2, f"Expected at least 2 actions, got {stats}"
    print(f"  PASS: stats tracking → {stats}")


if __name__ == "__main__":
    print("=== REFLEX PIPELINE TESTS ===\n")
    
    tests = [
        ("Low HP fires reflex", test_reflex_fires_at_low_hp),
        ("High HP no reflex", test_reflex_does_not_fire_at_high_hp),
        ("Critical HP escape", test_reflex_escape_at_critical_hp),
        ("Dead bot no reflex", test_reflex_does_not_fire_when_dead),
        ("emit_direct through pipeline", test_reflex_emit_direct_through_pipeline),
        ("Healing optimizer selection", test_healing_optimizer_selects_correct_potion),
        ("Cooldown respected", test_reflex_cooldown_respects_timer),
        ("Stats tracking", test_reflex_stats_tracking),
    ]
    
    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
            print(f"  ✅ {name}")
        except Exception as e:
            failed += 1
            print(f"  ❌ {name}: {e}")
    
    print(f"\n=== {passed}/{len(tests)} PASSED, {failed} FAILED ===")
    sys.exit(1 if failed > 0 else 0)
