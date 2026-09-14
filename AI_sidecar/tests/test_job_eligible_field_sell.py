"""Regression: a job-eligible broke novice on the FIELD must return to town to
sell (the sell->zeny->job-change chain). Prior: the field branch only returned
on HP<30% or weight>70%, so a novice farming light junk at ~15% weight never
sold, zeny stayed 0, and the 500z job-change gate stayed closed.

The trigger reads job_name/base_level/job_level/weight_ratio/zeny from signals.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ai_sidecar.app  # noqa: F401

from ai_sidecar.autonomy.heuristic_service import HeuristicService


def _assess_cmds(signals):
    hs = HeuristicService()
    a = hs.assess(signals, bot_id_override="TestBotA:testbot99")
    return [act.command for act in (a.actions or []) if getattr(act, "kind", "") == "command"]


def test_job_eligible_broke_field_returns_to_town():
    # base 47 / novice job 10 -> eligible; broke (zeny 0); carrying junk (15% wt)
    signals = {
        "map": "prt_fild05", "hp": 200, "hp_max": 275, "hp_ratio": 0.73,
        "sp": 50, "max_sp": 80,
        "base_level": 47, "job_level": 10, "job_name": "novice",
        "zeny": 0, "weight_ratio": 0.158,
        "inventory_items": ["Jellopy", "Bee Sting", "Clover", "Apple", "Red Herb"],
        "kills_this_session": 5, "in_game": True,
    }
    cmds = _assess_cmds(signals)
    assert any("move prontera" in c for c in cmds), (
        f"job-eligible broke novice with junk must return to town to sell; "
        f"commands={cmds}"
    )


def test_non_eligible_low_level_stays_on_field():
    # base 3 / job 1 -> NOT eligible. move prontera may fire legitimately from
    # OTHER emitters (cold-start economy), so assert the job-change-specific
    # reason is ABSENT (i.e. our new field-sell trigger did NOT fire).
    signals = {
        "map": "prt_fild05", "hp": 200, "hp_max": 200, "hp_ratio": 1.0,
        "base_level": 3, "job_level": 1, "job_name": "novice",
        "zeny": 0, "weight_ratio": 0.15,
        "inventory_items": ["Jellopy"], "kills_this_session": 5, "in_game": True,
    }
    hs = HeuristicService()
    a = hs.assess(signals, bot_id_override="TestBotA:testbot99")
    reasons = [act.reason for act in (a.actions or []) if getattr(act, "kind", "") == "command"]
    assert not any("fund job change" in (r or "") for r in reasons), (
        f"non-eligible bot must NOT get the job-change field-sell return; reasons={reasons}"
    )
