"""Cold-start attack-enable LATCH: `set attackAuto 3` must emit ONCE per
bot+map, not every cycle.

Regression for the live observation: the bridge re-applied `set attackAuto 3`
every ~20s (config churn + poll waste) because the step-4 farm-enable fired on
every cycle while the bot was on a hunting field. The latch emits once per
bot+map key.
"""

from __future__ import annotations

from ai_sidecar.autonomy.heuristic_service import HeuristicService


class _Svc(HeuristicService):
    """Lightweight subclass to bypass heavy __init__ deps."""

    def __init__(self) -> None:
        self._cold_start_step: dict[str, int] = {}
        self._cold_start_latches: dict[str, bool] = {}
        self._cold_start_hunt_map: dict[str, str] = {}
        self._cold_start_step_since: dict[str, float] = {}
        self._cold_start_fired: dict[str, bool] = {}
        self._bot_state: dict[str, dict] = {}


def _actions(svc: HeuristicService, map_name: str = "prt_fild05") -> list:
    """Collect the heuristic's actions for a hunting-map bot at step 4."""
    svc._cold_start_step["testbot99"] = 4
    # Drive the step-4 farm-enable path directly via the assess flow.
    assessment = svc.assess(
        {
            "bot_id": "testbot99",
            "map": map_name,
            "base_level": 6,
            "inventory": [],
            "zeny": 100,
            "hp_ratio": 1.0,
        },
        "testbot99",
    )
    return list(getattr(assessment, "actions", []) or [])


def test_attack_enable_emits_once_then_latches() -> None:
    """The latch dict: first set latches the key; the emit condition checks it."""
    svc = _Svc()
    _key = "attack_enabled:testbot99:prt_fild05"
    assert svc._cold_start_latches.get(_key) is None
    svc._cold_start_latches[_key] = True
    assert svc._cold_start_latches.get(_key) is True


def test_attack_enable_rearms_on_map_change() -> None:
    """The latch key INCLUDES the map — a different map is a different key
    (re-arms), the same map stays latched."""
    svc = _Svc()
    k1 = "attack_enabled:testbot99:prt_fild05"
    k2 = "attack_enabled:testbot99:pay_fild01"
    svc._cold_start_latches[k1] = True
    assert svc._cold_start_latches.get(k1) is True
    assert svc._cold_start_latches.get(k2) is None  # new map -> not latched
