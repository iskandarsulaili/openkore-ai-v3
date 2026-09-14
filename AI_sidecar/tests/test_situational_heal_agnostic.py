"""Regression: SituationalAwareness heal adapter must pick a CARRIED heal item
(Apple/Green Herb/Red Herb) via the data-driven HealingOptimizer, never fall
back to a hardcoded "use Red Potion" for an item the bot doesn't own.

Root cause (2026-09-14): situational.py:_get_best_heal_item used a hardcoded
HEALING_ITEMS list (White/Orange/Red/Novice Potion only). A broke novice that
carries 17 Apple but no potion got NO inventory match and (zeny=0) no buyable
-> fell back to (501,"Red Potion") -> "use Red Potion" -> "Error in use item"
-> NO heal -> death despite real carried heals. Now routes through the
inventory-aware HealingOptimizer.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ai_sidecar.app  # bootstraps the circular import chain  # noqa: F401

from ai_sidecar.runtime.situational import SituationalAwareness


def _adapt(aw, signals):
    from ai_sidecar.actions import HeuristicAction
    a = HeuristicAction(
        kind="command", command="use Red Potion",
        confidence=1.0, domain="survival",
    )
    out = aw._adapt_heal(a, signals, "TestBotA:testbot99")
    return out.command if out else None


def test_situational_uses_carried_apple_not_unowned_red_potion():
    aw = SituationalAwareness()
    signals = {
        "inventory": {"items": [{"name": "Apple", "quantity": 17},
                                {"name": "Green Herb", "quantity": 6},
                                {"name": "Red Herb", "quantity": 4}]},
        "zeny": 0,
        "hp": 60, "hp_max": 275,
        "sp": 50, "max_sp": 80,
        "base_level": 1,
    }
    cmd = _adapt(aw, signals)
    assert cmd and cmd.startswith("use "), f"expected a heal, got {cmd!r}"
    item = cmd[4:].strip().lower()
    carried = {"apple", "green herb", "red herb"}
    assert item in carried, (
        f"situational recommended {item!r} — NOT carried. "
        f"A broke bot must use a carried heal, never an unowned potion."
    )


def test_situational_without_carried_heal_does_not_emit_unowned_potion():
    # Bot carries only junk (no heal item) and is broke — must NOT return the
    # hardcoded Red Potion it doesn't own.
    aw = SituationalAwareness()
    signals = {
        "inventory": {"items": [{"name": "Jellopy", "quantity": 63},
                                {"name": "Yellow Gemstone", "quantity": 50}]},
        "zeny": 0,
        "hp": 60, "hp_max": 275,
        "sp": 50, "max_sp": 80,
        "base_level": 1,
    }
    # Must NOT emit "use Red Potion" for an unowned potion. It may return None
    # or a buy-potion fallback — but NEVER a use of an unowned heal.
    cmd = _adapt(aw, signals)
    assert cmd is None or cmd.startswith("buy "), (
        f"no carried heal -> should return None or a buy fallback, got {cmd!r}"
    )
