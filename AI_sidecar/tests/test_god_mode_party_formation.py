"""Regression test: god_mode party_organize forms a COMPLETE party.

The fleet coordinator (god_mode) is the party actor. party_organize must
emit party create for the leader AND party request for every other known
bot (stripped to char names), so the party actually fills instead of
leaving an empty leader-only party.
"""
from __future__ import annotations

from ai_sidecar.autonomy.god_mode import GodModeOrchestrator


class _FakeQueue:
    def __init__(self) -> None:
        self.enqueued: list[object] = []

    def enqueue(self, bot_id: str, proposal: object) -> tuple[bool, object, str, str]:
        self.enqueued.append((bot_id, proposal))
        return True, None, getattr(proposal, "action_id", ""), "ok"


class _FakeRuntime:
    def __init__(self) -> None:
        self.action_queue = _FakeQueue()


def _snap(bot_id: str, in_party: bool = False, level: int = 50) -> dict:
    return {
        bot_id: {
            "hp": 1000,
            "hp_max": 1000,
            "level": level,
            "in_party": in_party,
            "class": "novice",
            "map": "prontera",
            "attack_power": 10,
            "matk": 0,
            "target_element": "neutral",
            "target_size": "medium",
            "target_race": "brute",
            "target_hp": 100,
            "target_def": 0,
            "target_mdef": 0,
        }
    }


def test_party_organize_emits_create_and_requests() -> None:
    rt = _FakeRuntime()
    gm = GodModeOrchestrator()
    bots = {
        "Local rAthena AI World:kicapmasin4": {
            "hp": 1000, "hp_max": 1000, "level": 50, "in_party": False,
            "class": "novice", "map": "prontera", "attack_power": 10,
            "matk": 0, "target_element": "neutral", "target_size": "medium",
            "target_race": "brute", "target_hp": 100, "target_def": 0, "target_mdef": 0,
        },
        "Local rAthena AI World:kicapmasin5": {
            "hp": 1000, "hp_max": 1000, "level": 50, "in_party": False,
            "class": "novice", "map": "prontera", "attack_power": 10,
            "matk": 0, "target_element": "neutral", "target_size": "medium",
            "target_race": "brute", "target_hp": 100, "target_def": 0, "target_mdef": 0,
        },
    }
    actions = gm.assess(bots)
    organize = [a for a in actions if a.get("type") == "party_organize"]
    assert len(organize) == 1
    assert organize[0]["data"]["all_bots"] == sorted(bots.keys())

    enqueued = gm.enqueue_actions(rt, actions, "strategic")
    # leader create + 1 request for the other bot
    assert enqueued == 2
    cmds = [p.command for _b, p in rt.action_queue.enqueued]
    assert any(c.startswith("party create AI") for c in cmds)
    assert "party request kicapmasin5" in cmds
    assert not any("Local rAthena" in c for c in cmds), "char names must be stripped"


def test_party_organize_gated_below_level_40() -> None:
    rt = _FakeRuntime()
    gm = GodModeOrchestrator()
    bots = dict(_snap("Local rAthena AI World:kicapmasin4", level=10))
    bots.update(_snap("Local rAthena AI World:kicapmasin5", level=10))
    actions = gm.assess(bots)
    assert not [a for a in actions if a.get("type") == "party_organize"]
    assert gm.enqueue_actions(rt, actions, "strategic") == 0
