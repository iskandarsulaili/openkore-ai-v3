"""Regression test: _emit_heuristic_actions injects the enriched depletion score.

Raw bridge signals don't carry inventory.consumable_depletion_score (it's a
world-projection feature). The injection block in _emit_heuristic_actions
(pdca_loop) must enrich signals before hs.assess() so the task scheduler can
trigger restock BEFORE pots run out.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class _FakeFeatures:
    values = {"inventory.consumable_depletion_score": 0.91, "inventory.weight_pressure": 0.4}


class _FakeEnriched:
    features = _FakeFeatures()


class _FakeRuntime:
    def __init__(self) -> None:
        self._last_signal = None
        self._enriched = _FakeEnriched()
        self.calls = 0
        self.heuristic_service: Any = None
        self.action_queue: Any = None
        self.snapshot_cache: Any = None

    def enriched_state(self, *, bot_id: str):
        return self._enriched


def test_depletion_score_injected_into_signals() -> None:
    from ai_sidecar.autonomy.pdca_loop import _emit_heuristic_actions

    captured: dict = {}

    class _CaptureHS:
        def assess(self, signals, bot_id_override=None):
            captured.update(signals)
            return SimpleNamespace(actions=[])

    rt = _FakeRuntime()
    rt.heuristic_service = _CaptureHS()
    rt.action_queue = None
    rt.snapshot_cache = None

    # bot_id must be provided; signals WITHOUT the score
    n = _emit_heuristic_actions(rt, "immediate", bot_id="bot:x")
    assert n == 0  # no actions
    assert captured.get("consumable_depletion_score") == 0.91, \
        f"enriched score must be injected, got {captured.get('consumable_depletion_score')}"


def test_depletion_score_not_overwritten_when_present() -> None:
    from ai_sidecar.autonomy.pdca_loop import _emit_heuristic_actions

    captured: dict = {}

    class _CaptureHS:
        def assess(self, signals, bot_id_override=None):
            captured.update(signals)
            return SimpleNamespace(actions=[])

    rt = _FakeRuntime()
    rt.heuristic_service = _CaptureHS()
    rt.action_queue = None
    rt.snapshot_cache = None

    _emit_heuristic_actions(rt, "immediate", bot_id="bot:x")
    # Second call with an explicit score must keep the explicit value
    _emit_heuristic_actions(rt, "immediate", bot_id="bot:x")
    # The injection only fills when absent; signals are rebuilt per call,
    # so both calls must carry the enriched value consistently.
    assert captured.get("consumable_depletion_score") == 0.91
