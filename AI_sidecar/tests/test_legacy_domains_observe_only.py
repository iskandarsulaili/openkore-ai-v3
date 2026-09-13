"""Regression test: legacy domains activate in observe-only mode.

The 12 autonomy/domains/* modules were a documented-empty guard: populating
them would double-emit with the modern wired managers AND reintroduce party
spam. Both hazards are now fixed at the source (social.py emits party_* log
intents; ai_mode_* replaces `ai manual`/`ai auto` flips), and the registry
converts any remaining command emission to an observable log intent. This
test locks: (1) all 12 modules load, (2) assess_all never produces a
kind="command" action, (3) no party command survives anywhere.
"""

from __future__ import annotations

from ai_sidecar.autonomy.domains import DomainRegistry

_ALL_12 = {"combat", "consumables", "economy", "environment", "equipment",
           "learning", "mimicry", "npc", "progression", "quests",
           "routing", "social"}


def test_all_legacy_domains_load() -> None:
    reg = DomainRegistry()
    reg.load_all()
    names = set(reg.domain_names)
    assert _ALL_12 <= names, f"missing modules: {_ALL_12 - names}"


def test_assess_all_emits_no_commands() -> None:
    reg = DomainRegistry()
    reg.load_all()
    actions: list = []
    signals = {
        "map": "prt_fild05", "base_level": 12, "zeny": 5000,
        "in_party": False, "party_members": [], "all_bots": ["bot:x"],
        "inventory": [{"name": "Red Potion", "quantity": 5}],
        "hp": 500, "hp_max": 1000, "weight": 3000, "weight_max": 8000,
        "job_name": "novice",
    }
    reg.assess_all(signals, actions, service=_StubService())
    assert actions, "observe-only domains must still produce log intents"
    for a in actions:
        assert a.kind == "log", f"observe-only must never emit commands: {a}"
        assert not str(getattr(a, "command", "")).startswith("party"), \
            f"party must never survive: {a}"


def test_observe_only_domains_do_not_poison_config_dedupe_cache() -> None:
    """Legacy observe-only domains must not mutate the shared _last_config_set
    dedupe cache that the REAL config owner (heuristic_service) uses.

    Regression (2026-09-13): EconomyDomain._apply_sell_config called
    _set_config_once('sellAuto_maxWeight','70') on the shared cache, so the
    heuristic hunting block's '25' read '70' as "changed" every cycle and
    re-emitted sellAuto_maxWeight forever (live flood 19:10-19:16)."""

    from ai_sidecar.autonomy import heuristic_service as hs

    class _Svc:
        _last_config_set: dict[str, dict[str, str]] = {}

    svc = _Svc()
    # The single owner already set sellAuto_maxWeight=25 (authoritative).
    svc._last_config_set.setdefault("bot", {})["sellAuto_maxWeight"] = "25"

    reg = DomainRegistry()
    reg.load_all()
    actions: list = []
    signals = {
        "map": "prt_fild05", "base_level": 20, "zeny": 5000,
        "in_party": False, "party_members": [], "all_bots": ["bot:x"],
        "inventory": [{"name": "Jellopy", "quantity": 30}],
        "hp": 500, "hp_max": 1000, "weight": 3000, "weight_max": 8000,
        "job_name": "novice", "weight_ratio": 0.4,
        "state": "AGGRESSIVE", "bot_id": "bot:test",
    }
    reg.assess_all(signals, actions, service=svc)  # type: ignore[arg-type]

    # The owner's value must be untouched by any observe-only domain.
    assert svc._last_config_set["bot"]["sellAuto_maxWeight"] == "25", (
        "observe-only domain poisoned the config-owner's dedupe cache -> "
        "perpetual config re-emission flood"
    )
    # Any log intent is observe-only (never a live command).
    for a in actions:
        assert a.kind == "log"


def test_no_party_or_ai_mode_commands_in_sources() -> None:
    import os
    import re

    base = os.path.join(os.path.dirname(__file__), "..", "ai_sidecar", "autonomy", "domains")
    for f in os.listdir(base):
        if not f.endswith(".py") or f == "__init__.py":
            continue
        src = open(os.path.join(base, f), errors="replace").read()
        # kind="command" paired with party / ai manual / ai auto / set partyAuto
        for m in re.finditer(r'kind="command",\s*command="((?:party|ai (?:manual|auto)|set partyAuto)[^"]*)"', src):
            raise AssertionError(f"{f} still emits banned command: {m.group(1)}")


class _Anything:
    """Recursive stand-in: any attribute access, call, or index works."""

    def __getattr__(self, name: str):
        return _Anything()

    def __call__(self, *args, **kwargs):
        return _Anything()

    def __getitem__(self, key):
        return _Anything()

    def __setitem__(self, key, value):
        pass

    def __iter__(self):
        return iter(())

    def __bool__(self):
        return False

    def __float__(self) -> float:
        return 0.0

    def __int__(self) -> int:
        return 0

    def __sub__(self, other):
        return 0.0

    def __rsub__(self, other):
        return 0.0

    def __add__(self, other):
        return 0.0

    def __radd__(self, other):
        return 0.0

    def __lt__(self, other):
        return True

    def __gt__(self, other):
        return False

    def __le__(self, other):
        return True

    def __ge__(self, other):
        return False

    def __eq__(self, other):
        return False


class _StubService(_Anything):
    """Minimal HeuristicService stand-in with the state dicts domains touch.

    Inherits _Anything so any unknown service attribute (e.g.
    service._adaptive.record_visit) resolves to a usable stand-in, while
    the explicitly declared dicts below are real for state tracking.
    """

    def __init__(self) -> None:
        # __getattr__ would shadow these on first access; set them directly
        object.__setattr__(self, "_last_party_seen", {})
        object.__setattr__(self, "_last_party_members", {})
        object.__setattr__(self, "_all_bots_cache", {})
        object.__setattr__(self, "_last_party_leave", {})
        object.__setattr__(self, "_profile_to_char", {})
        object.__setattr__(self, "_cold_start_step", {})
        object.__setattr__(self, "_last_mon_control_map", {})
        object.__setattr__(self, "_last_lockmap", {})
        object.__setattr__(self, "_sit_start_time", {})

    def _resolve_bot_id(self, signals) -> str:
        return "bot:x"

    def _get_state(self, signals, bot_id) -> str:
        return "HUNTING"
