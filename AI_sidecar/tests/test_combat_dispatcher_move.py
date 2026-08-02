"""Regression test: combat dispatcher must not emit a bogus `move 0 0`.

The kiting/positioning modules signal retreat/back_up/approach/waiter with a
`tactic` label but leave move_x/move_y at 0 (TacticsContext has no bot/target
absolute coordinates). Previously _make_move_action emitted `move 0 0` —
pathing the bot to the map origin (no-op / teleport hazard).
"""
from __future__ import annotations

from ai_sidecar.domains.combat.dispatcher import TacticsDispatcher
from ai_sidecar.domains.combat.tactics.base import TacticsContext
from ai_sidecar.domains.combat.tactics.kiting import KitingTactics

# Minimal target carrying the actor_id used for metadata.
class _MinTarget:
    actor_id = 7


def test_positioning_with_zero_coords_emits_log_not_move0() -> None:
    d = TacticsDispatcher()
    ctx = TacticsContext()
    action = d._make_move_action(
        {"move_x": 0, "move_y": 0, "urgency": 0.9,
         "reason": "emergency_retreat_from_Poring_dist_3", "tactic": "retreat"},
        ctx, KitingTactics(), _MinTarget(),
    )
    # Must NOT be an executable `move 0 0`.
    assert action.kind == "log"
    assert "move 0 0" not in action.command


def test_positioning_with_real_coords_emits_move() -> None:
    d = TacticsDispatcher()
    ctx = TacticsContext()
    action = d._make_move_action(
        {"move_x": 150, "move_y": 200, "urgency": 0.8,
         "reason": "kite_position", "tactic": "approach"},
        ctx, KitingTactics(), _MinTarget(),
    )
    assert action.kind == "command"
    assert action.command == "move 150 200"
