"""Regression: a broke novice carrying Apple/Green Herb/Red Herb (itemheal
consumables, no literal "Potion" in name) must be treated as heal-capable and
the reflex must recommend the CARRIED heal — not a potion it can't afford/own.

Root cause (2026-09-14): has_potions only matched names containing "potion",
and HealingOptimizer only scored BUYABLE potions (zeny=0 -> every item skipped).
A gearless novice with 17 Apple but no Red Potion was reported heal-less, never
healed at low HP, and died on the field before the sell chain could convert junk.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ai_sidecar.app  # bootstraps the circular import chain  # noqa: F401

from ai_sidecar.reflex.healing_optimizer import HealingOptimizer
from ai_sidecar.reflex.highfreq_reflex import HighFreqReflex


INVENTORY_APPLE = [
    {"name": "Apple", "quantity": 17},
    {"name": "Jellopy", "quantity": 63},
    {"name": "Green Herb", "quantity": 6},
    {"name": "Red Herb", "quantity": 4},
]


def test_optimizer_prefers_carried_apple_when_broke():
    opt = HealingOptimizer()
    opt.load()
    cmd = opt.select_healing_command(
        hp=60, max_hp=275, sp=50, max_sp=80, zeny=0, level=1,
        prefer_hp=True, inventory=INVENTORY_APPLE,
    )
    # A broke bot must heal with a carried item, never a potion it can't afford.
    assert cmd and cmd.startswith("use "), f"expected a heal command, got {cmd!r}"
    item = cmd[4:]
    names = {i["name"].lower() for i in INVENTORY_APPLE}
    assert item.lower() in names, \
        f"reflex recommended {item!r} which is NOT carried; carried={names}"


def test_optimizer_broke_without_inventory_returns_late():
    # Without inventory, zeny=0 skips all buyable potions -> no recommendation.
    opt = HealingOptimizer()
    opt.load()
    cmd = opt.select_healing_command(
        hp=60, max_hp=275, sp=50, max_sp=80, zeny=0, level=1, prefer_hp=True,
    )
    assert cmd is None, f"expected None without inventory (broke can't buy), got {cmd!r}"


def test_has_potions_true_for_carried_apple():
    # The reflex's has_potions gate must treat a carried Apple as heal-capable.
    r = HighFreqReflex()
    names = r._heal_capable_names()
    # The optimizer's loaded table must include apple (heal item) by name/id.
    assert "apple" in names or "512" in names, \
        f"apple not recognized as heal-capable; names sample={sorted(names)[:10]}"
