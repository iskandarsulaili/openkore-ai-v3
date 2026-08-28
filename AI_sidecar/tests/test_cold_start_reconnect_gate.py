"""Cold-start reconnect gate: char-creation must NEVER fire during a reconnect.

Regression for the 11:11 live misfire: the bot was mid-reconnect (raw.map still
showed the last playable map prt_fild08, reconnect_age_s growing), the
_disconnected_safe gate treats a real map as in-game (so the heuristic assess
ran), and cold-start saw characters=[] (char list only populates at char-select)
-> it decided "no character exists" and emitted a char CREATE against the live
account. With delete-recreate enabled this could destroy the leveled character.
"""

from __future__ import annotations

from ai_sidecar.domains.progression.cold_start import ColdStartManager


class _Cfg:
    max_creation_retries = 3
    enable_delete_recreate = False
    verify_after_creation = False
    job_class = "swordman"
    character_name_prefix = "BotTestBot"
    stat_allocation = None
    def __init__(self) -> None:
        self.cooldown_after_block = 120


class _Act:
    def __init__(self) -> None:
        self.kind = ""
        self.command = ""
        self.confidence = 0.0
        self.domain = ""
        self.reason = ""


def _signals(**over):
    s = {
        "bot_id": "b:x",
        "characters": [],
        "character_list": [],
        "base_level": 0,
        "map": "",
        "map_known": False,
        "in_game": False,
    }
    s.update(over)
    return s


def test_no_char_creation_during_reconnect() -> None:
    """A bot mid-reconnect (growing reconnect_age_s + stale raw.map) must NOT
    get a create-character action."""
    mgr = ColdStartManager(config=_Cfg())
    actions: list[_Act] = []
    mgr.assess(
        _signals(reconnect_age_s=12.0, raw_in_game=False,
                 raw={"map": "prt_fild08", "in_game": False}),
        actions, "b:x",
    )
    cmds = [a.command for a in actions]
    assert not any("create" in str(c) or "delete" in str(c) for c in cmds), (
        f"char create/delete during reconnect: {cmds}"
    )


def test_no_char_creation_when_raw_in_game_false() -> None:
    """Explicit raw_in_game=False must gate creation even without reconnect age."""
    mgr = ColdStartManager(config=_Cfg())
    actions: list[_Act] = []
    mgr.assess(_signals(raw_in_game=False, raw={"in_game": False}), actions, "b:x")
    cmds = [a.command for a in actions]
    assert not any("create" in str(c) or "delete" in str(c) for c in cmds), cmds


def test_char_creation_still_fires_when_truly_empty() -> None:
    """A genuinely-empty account (no reconnect, no map, no in_game) may still
    create — the cold-start purpose is intact."""
    mgr = ColdStartManager(config=_Cfg())
    actions: list[_Act] = []
    mgr.assess(_signals(), actions, "b:x")
    cmds = [a.command for a in actions]
    assert any("create" in str(c) for c in cmds), f"empty account must create: {cmds}"


def test_in_game_bot_never_creates() -> None:
    """A live bot (base_level + map) never creates — the pre-existing gate."""
    mgr = ColdStartManager(config=_Cfg())
    actions: list[_Act] = []
    mgr.assess(
        _signals(base_level=6, map="prt_fild08", in_game=True,
                 raw={"map": "prt_fild08", "in_game": True}),
        actions, "b:x",
    )
    cmds = [a.command for a in actions]
    assert not any("create" in str(c) or "delete" in str(c) for c in cmds), cmds
