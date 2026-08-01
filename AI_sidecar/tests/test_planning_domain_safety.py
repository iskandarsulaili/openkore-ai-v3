"""Regression test: PlanningDomain.assess must emit through the translator.

PlanningDomain was a complete but zero-call-site domain that used the naive
scheduler.execute_task() — raw commands (bare `party`, unknown intents)
would execute in the bridge and reintroduce the frozen-party-spam bug. Its
assess() now routes every scheduled task through TaskCommandTranslator
(return_town gated on inventory_full, party/guild never emitted, moves
deduped). This test locks the safety contract so wiring PlanningDomain can
never reintroduce spam.
"""

from __future__ import annotations

from ai_sidecar.domains.planning import PlanningDomain


def _emitted_commands(pd: PlanningDomain, signals: dict) -> list[tuple[str, str]]:
    actions: list = []
    pd.assess(signals, actions, "bot:x")
    return [(a.kind, getattr(a, "command", "")) for a in actions]


def test_planning_domain_emits_via_translator() -> None:
    pd = PlanningDomain()
    pd.initialize()
    # Grind schedule on a fild map — must produce only log/observed intents
    # and safe real commands, never bare party or unknown commands.
    out = _emitted_commands(pd, {"map": "prt_fild05", "base_level": 8})
    for kind, cmd in out:
        assert kind in ("log", "command"), f"unexpected kind {kind}: {cmd}"
        if kind == "command":
            # REAL commands must never touch party/guild — log intents may
            # describe them (task:party_check:party) but are never executed
            # (bridge routes kind=log to unsupported_kind).
            assert not cmd.strip().startswith("party"), f"party must never emit as command: {cmd}"
            assert "party " not in cmd, f"party command: {cmd}"


def test_planning_domain_return_town_gated_on_inventory_full() -> None:
    pd = PlanningDomain()
    pd.initialize()
    # No inventory_full -> no real move-to-town command (only log intents)
    out = _emitted_commands(pd, {"map": "prt_fild05", "base_level": 8})
    real_moves = [c for k, c in out if k == "command" and c.startswith("move")]
    assert real_moves == [], f"move must not fire without inventory_full: {real_moves}"

    # inventory_full -> return_town translates to a real move prontera
    pd2 = PlanningDomain()
    pd2.initialize()
    out2 = _emitted_commands(pd2, {"map": "prt_fild05", "base_level": 8, "inventory_full": True})
    assert any(c.startswith("move prontera") for _, c in out2), \
        f"inventory_full must produce move prontera, got {out2}"


def test_planning_domain_dedupes_moves() -> None:
    pd = PlanningDomain()
    pd.initialize()
    out = _emitted_commands(pd, {"map": "prt_fild05", "base_level": 8, "inventory_full": True})
    moves = [c for _, c in out if c.startswith("move")]
    assert len(moves) == len(set(moves)), f"identical moves must be deduped: {moves}"
