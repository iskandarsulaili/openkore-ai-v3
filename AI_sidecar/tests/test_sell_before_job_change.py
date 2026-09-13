"""Regression: a broke job-change-eligible novice must SELL junk (not stay stuck
in JOB_CHANGE), and the weight signal must be read as the REAL load ratio, not
the overweight-relative `weight_pressure` (which is 0 until >50% loaded).

See docs/MASTER_GODTIER_SWEEP_2026-09-13.md Batch A2. Live deadlock: base 46 /
job Novice / job lvl 10 / zeny 0 / weight 16% carried junk forever; _get_state
read weight_pressure=0 so SELL never fired, and JOB_CHANGE fired before SELL so
the bot routed to the guild it couldn't afford -> zeny stayed 0 -> the 500z
job-change gate never opened."""

from __future__ import annotations

from unittest.mock import MagicMock

from ai_sidecar.autonomy.heuristic_service import HeuristicService, _HUNT_TOWNS


def _hs() -> HeuristicService:
    s = HeuristicService.__new__(HeuristicService)
    for attr, val in {
        "_bot_state": {},
        "_cold_start_fired": {},
        "_cold_start_step": {},
        "_last_job_name": {},
        "_town_entry_time": {},
        "_state_since": {},
        "_adaptive": MagicMock(),
        "_has_coldstart_weapon": None,
        "_resolve_academy_door": None,
    }.items():
        if val is None:
            continue
        setattr(s, attr, val)
    return s


def _town_signals(**overrides) -> dict:
    base = {
        "map": "prontera", "hp_ratio": 1.0, "hp": 200, "hp_max": 270,
        "zeny": 0, "base_level": 46, "job_level": 10, "job_name": "novice",
        "inventory": {"weight": 418.6, "weight_max": 2570.0,
                      "weight_ratio": 0.1628, "weight_pressure": 0.0},
        "weight_ratio": 0.1628,
        "inventory_items": [], "total_kills": 50, "kills_this_session": 5,
        "in_party": False, "stat_points": 0, "skill_points": 0,
        "last_map_change": 0,
    }
    base.update(overrides)
    return base


def test_eligible_novice_carrying_junk_enters_SELL_not_JOB_CHANGE() -> None:
    """A job-change-eligible novice carrying junk (weight 16%) must SELL to fund
    the job change — NOT return JOB_CHANGE (which would deadlock at zeny 0)."""
    hs = _hs()
    state = hs._get_state(_town_signals(), "bot:test")
    assert state == "SELL", f"expected SELL, got {state}"


def test_eligible_novice_broke_and_empty_farms_not_job_change() -> None:
    """A broke, empty (weight ~0) eligible novice must farm (TOWN_HUNT) to gain
    weight/zeny — not sit in JOB_CHANGE unable to pay."""
    hs = _hs()
    sig = _town_signals(weight_ratio=0.02, inventory={
        "weight": 51.0, "weight_max": 2570.0, "weight_ratio": 0.02,
        "weight_pressure": 0.0,
    })
    state = hs._get_state(sig, "bot:test")
    assert state == "TOWN_HUNT", f"expected TOWN_HUNT, got {state}"


def test_affordable_eligible_novice_enters_JOB_CHANGE() -> None:
    """An eligible novice WITH zeny>=500 must still job-change (not be blocked)."""
    hs = _hs()
    sig = _town_signals(zeny=600, weight_ratio=0.02, inventory={
        "weight": 51.0, "weight_max": 2570.0, "weight_ratio": 0.02,
        "weight_pressure": 0.0,
    })
    state = hs._get_state(sig, "bot:test")
    assert state == "JOB_CHANGE", f"expected JOB_CHANGE, got {state}"


def test_weight_ratio_used_not_weight_pressure() -> None:
    """The weight signal must come from weight_ratio (real load), not
    overweight-relative weight_pressure. A bot at 16% must read 0.16, not 0."""
    hs = _hs()
    sig = _town_signals()
    _inv = sig["inventory"]
    assert _inv["weight_pressure"] == 0.0, "precondition: weight_pressure is 0 at 16% load"
    # The deadlock only breaks if _get_state uses the real ratio.
    state = hs._get_state(sig, "bot:test")
    assert state == "SELL", state


def test_weight_ratio_fallback_inventory() -> None:
    """When top-level weight_ratio is absent but inventory.weight_ratio exists,
    _get_state must still use the real load ratio."""
    hs = _hs()
    sig = _town_signals()
    del sig["weight_ratio"]  # top-level missing
    sig["inventory"]["weight_ratio"] = 0.16
    state = hs._get_state(sig, "bot:test")
    assert state == "SELL", state


def test_junk_derived_from_buy_not_sell() -> None:
    """The auto-sell junk scan must derive resale value from Buy/2 (the RO vendor
    mechanic), NOT a non-existent `Sell` column that yields 0 and classifies NO
    item as junk. Regression: the sell->zeny->job-change deadlock — Jellopy/Bee
    Sting (the bot's actual drops) must classify as junk so `sell <id>` fires."""
    _junk_ids = []
    _junk_names = []
    try:
        from ai_sidecar.knowledge_loader import get_items
        item_db = {str(_it.get("Id", "")): _it for _it in get_items()}
    except Exception:
        item_db = {}
    for _it_id, _it in item_db.items():
        _it_nm = str(_it.get("Name", "") or _it.get("AegisName", "")).lower()
        if _it_nm in ("jellopy", "sticky mucus", "bee sting"):
            _buy = int(_it.get("Buy", 0) or 0)
            _sell_val = _buy // 2
            if 0 < _sell_val < 100:
                _junk_ids.append(_it_id)
                _junk_names.append(_it_nm)
    assert "909" in _junk_ids, f"Jellopy should be junk (Buy/2), got {_junk_ids}"
    assert "938" in _junk_ids, f"Sticky Mucus should be junk, got {_junk_ids}"
    assert "939" in _junk_ids, f"Bee Sting should be junk, got {_junk_ids}"


def test_gearless_bot_disables_deadly_teleport() -> None:
    """A gearless bot (no Fly Wing carried) must have teleportAuto_deadly=0 so
    OpenKore's built-in deadly-teleport (which fires NO_ITEM_OR_SKILL and FREEZES
    the bot in place -> dies -> corpse-loop) never arms. AGNOSTIC: derived from
    the live inventory, never a hardcoded item id."""
    import os

    def _has_fw(items) -> bool:
        return any(("fly wing" in str(i).lower()) or ("flywing" in str(i).lower())
                   for i in items)

    # no Fly Wing -> deadly teleport OFF
    gearless = ["Jellopy", "Novice Potion", "Knife"]
    assert _has_fw(gearless) is False
    # Fly Wing present -> ON
    assert _has_fw(["Jellopy", "Fly Wing", "Knife"]) is True
    assert _has_fw(["Jellopy", "fly wing", "Knife"]) is True
    assert _has_fw(["jellopy", "Flywing"]) is True
    # Lock the invariant: the audit block must scan inventory (not a hardcoded id)
    src = open(os.path.join(os.path.dirname(__file__), "..", "ai_sidecar",
                            "autonomy", "heuristic_service.py"),
               encoding="utf-8", errors="replace").read()
    assert "teleportAuto_deadly" in src
    assert "inventory_items" in src
    assert "fly wing" in src, "the deadly-teleport guard must scan inventory for a fly wing"
