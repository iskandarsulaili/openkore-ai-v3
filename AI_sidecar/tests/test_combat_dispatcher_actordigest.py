"""Regression: combat dispatcher build_context must tolerate ActorDigest
pydantic actors (this fork's contract delivers ActorDigest objects, NOT dicts).

The dispatcher previously did `a.get("type", "")` on pydantic models →
AttributeError every combat tick → tactics_dispatcher.assess() failed and the
combat-tactics domain was dead (dormant). Fix normalizes actors to plain dicts
with the key aliases the dict-API consumers expect.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Bootstrap the package in the same order as the app entry (ai_sidecar.app)
# to resolve a pre-existing circular import (combat.skills -> autonomy ->
# heuristic_service -> combat.dispatcher). Importing app first makes the
# dispatcher symbol available for the direct import below.
import ai_sidecar.app  # noqa: F401
from ai_sidecar.domains.combat.dispatcher import TacticsDispatcher
from ai_sidecar.contracts.state import ActorDigest


def _dispatcher():
    return TacticsDispatcher()


def _signals(actors=None, monster=True):
    actors = actors if actors is not None else [
        ActorDigest(actor_id="mob1", actor_type="monster", hp=100, hp_max=100, distance=3.0, relation="hostile"),
        ActorDigest(actor_id="plr1", actor_type="player", hp=100, hp_max=100, distance=5.0, relation="party"),
        ActorDigest(actor_id="npc1", actor_type="npc", name="Kafra", distance=9.0),
    ]
    return {
        "vitals": {"hp": 30, "hp_max": 100, "sp": 50, "sp_max": 100, "job_name": "novice", "base_level": 1},
        "combat": {"target_id": 0, "aggro_count": 0, "in_combat": False},
        "actors": actors,
        "position": {"map": "prt_fild01", "x": 100, "y": 100},
        "status": {"sitting": False},
        "cooldowns": {},
        "buffs": [],
        "skills": [],
    }


def test_build_context_with_actordigest_monsters():
    d = _dispatcher()
    ctx = d.build_context(_signals())
    # monsters normalized from ActorDigest objects
    assert len(ctx.monsters) == 1, f"expected 1 monster, got {ctx.monsters}"
    assert ctx.monsters[0]["type"] == "monster"
    assert ctx.monsters[0]["actor_id"] == "mob1"
    assert ctx.enemies_nearby == 1
    # party member recognized via relation=="party" -> is_party alias
    assert ctx.party_members_nearby == 1
    assert ctx.has_party is True
    # npc excluded (not monster or player)
    assert len([a for a in ctx.monsters if a.get("type") == "npc"]) == 0


def test_build_context_with_mixed_dict_and_actordigest():
    d = _dispatcher()
    actors = [
        ActorDigest(actor_id="m1", actor_type="monster", hp=50, distance=2.0),
        {"actor_id": "m2", "actor_type": "monster", "hp": 60, "distance_to": 4.0, "type": "monster"},
    ]
    ctx = d.build_context(_signals(actors))
    assert len(ctx.monsters) == 2
    ids = {m["actor_id"] for m in ctx.monsters}
    assert ids == {"m1", "m2"}


def test_assert_assess_runs_without_exception():
    d = _dispatcher()
    actions = []
    # The pre-fix bug: assess() crashed on build_context .get() AttributeError.
    # This must not raise.
    d.assess(_signals(), actions, "testbot99")
    assert isinstance(actions, list)
