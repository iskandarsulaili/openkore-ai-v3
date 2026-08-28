from __future__ import annotations

import asyncio
import types
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from ai_sidecar.autonomy.pdca_loop import PDCALoop


@dataclass
class FakeActionQueue:
    queued: list = field(default_factory=list)

    def enqueue(self, bot_id: str, proposal: Any) -> None:
        self.queued.append((bot_id, proposal))


@dataclass
class FakeLLM:
    """Minimal LLMManager stand-in for the cold-start advisory."""

    _plan: dict | None = None
    _raise: bool = False
    available: bool = True

    def is_available(self) -> bool:
        return self.available

    async def complete_json(self, prompt, system_prompt="", temperature=None,
                            max_tokens=None, *, preferred_provider=None, fallback=True):
        if self._raise:
            raise RuntimeError("gateway down")
        if self._plan is not None:
            return self._plan
        # Default: decide academy_kit when a warp fact is present.
        return {"analysis": "novice needs a weapon; academy kit is the fastest route",
                "root_cause": "no weapon -> cannot farm",
                "plan": "academy_kit", "command": "", "reason": "free starter kit"}


@dataclass
class FakeRuntime:
    llm_manager: FakeLLM = field(default_factory=FakeLLM)
    action_queue: FakeActionQueue = field(default_factory=FakeActionQueue)
    server_solutions_store: Any = None
    _last_snapshot: dict = field(default_factory=dict)
    snapshot_cache: Any = None

    def __post_init__(self):
        if self.snapshot_cache is None:
            # a minimal snapshot_cache backed by _last_snapshot-style dicts
            class _SC:
                def __init__(self, owner):
                    self._owner = owner
                def get(self, bot_id):
                    d = self._owner._last_snapshot.get(bot_id) or {}
                    return _Snap(d)
            class _Snap:
                def __init__(self, d):
                    self._d = d
                    self.position = SimpleNamespace(map=d.get("map", ""))
                    self.vitals = SimpleNamespace(hp_ratio=d.get("hp_ratio", 0.0))
                    self.progression = SimpleNamespace(
                        base_level=d.get("base_level", 1),
                        base_exp=d.get("base_exp"),
                        base_exp_max=d.get("base_exp_max"),
                        job_level=d.get("job_level", 1),
                    )
                    self.raw = d.get("raw") or d
                    self.inventory_items = d.get("inventory_items", [])
            self.snapshot_cache = _SC(self)


def _pdca(fake_rt: FakeRuntime, tmp_path) -> PDCALoop:
    loop = PDCALoop(fake_rt)
    # make _spawn_advisory run inline for determinism
    loop._spawn_advisory = types.MethodType(
        lambda self, name, coro: asyncio.create_task(coro), loop
    )
    return loop


def test_cold_start_advisory_enqueues_academy_move(tmp_path):
    """LLM (conscious) should enqueue the academy-door move when facts show a
    reachable academy warp and the bot is a weapon-less novice."""
    import pathlib
    import pytest
    tables = tmp_path / "tables"
    tables.mkdir()
    (tables / "portals.txt").write_text(
        "# from x y to tx ty\nizlude 125 257 iz_ac01 99 29\n", encoding="utf-8"
    )
    # point Path(__file__) resolution at tmp by monkeypatching the module __file__
    import ai_sidecar.autonomy.pdca_loop as mod
    orig = mod.__file__
    try:
        mod.__file__ = str(tmp_path / "ai_sidecar/autonomy/pdca_loop.py")
        fake_rt = FakeRuntime(llm_manager=FakeLLM())
        fake_rt._last_snapshot = {
            "bot:test": {
                "map": "izlude",
                "base_level": 1,
                "inventory_items": [],
            }
        }
        loop = _pdca(fake_rt, tmp_path)
        asyncio.run(loop._llm_cold_start_advisory("bot:test"))
        assert fake_rt.action_queue.queued, "expected the LLM cold-start command to be enqueued"
        _bid, _p = fake_rt.action_queue.queued[0]
        assert _bid == "bot:test"
        assert _p.command == "move 125 257"
    finally:
        mod.__file__ = orig


def test_cold_start_advisory_llm_down_no_crash(tmp_path):
    """LLM gateway blip must not crash; no command enqueued, no exception."""
    fake_rt = FakeRuntime(llm_manager=FakeLLM(_raise=True))
    fake_rt._last_snapshot = {
        "bot:x": {"map": "izlude", "base_level": 1, "inventory_items": []}
    }
    loop = _pdca(fake_rt, tmp_path)
    asyncio.run(loop._llm_cold_start_advisory("bot:x"))
    assert fake_rt.action_queue.queued == []


def test_llm_npc_dialog_selects_option_from_menu(tmp_path):
    """CONSCIOUS-tier dialog responder must read the ACTUAL menu and pick the
    option that advances the goal (AGNOSTIC — no hardcoded r<idx> for a specific
    NPC). The fake LLM returns option_index=2 (matching 'Register for the
    Academy' text at [2]) and the responder emits talknpc r2."""
    fake_rt = FakeRuntime(llm_manager=FakeLLM(_plan={
        "analysis": "register option advances the starter-kit goal",
        "choice": "option", "option_index": 2, "reason": "Register for the Academy"}))
    fake_rt._last_snapshot = {
        "bot:m": {
            "map": "iz_ac01",
            "base_level": 1,
            "inventory_items": [],
            "raw": {
                "menu_options": ["Explanation about academy", "Location for trainers",
                                 "Register for the Academy", "Conversation finished"],
                "npc_name": "Academy Receptionist#1",
                "npc_x": 100, "npc_y": 39,
                "goal": "register",
            },
        }
    }
    loop = _pdca(fake_rt, tmp_path)
    asyncio.run(loop._llm_npc_dialog_response("bot:m"))
    assert fake_rt.action_queue.queued, "expected the LLM dialog response to be enqueued"
    _bid, _p = fake_rt.action_queue.queued[0]
    assert _p.command == "talknpc 100 39 c r2 n"


def test_llm_npc_dialog_no_menu_skips(tmp_path):
    """No open menu -> the responder does nothing (no fabricuated command)."""
    fake_rt = FakeRuntime(llm_manager=FakeLLM())
    fake_rt._last_snapshot = {"bot:no": {"map": "prt_fild08", "raw": {}}}
    loop = _pdca(fake_rt, tmp_path)
    asyncio.run(loop._llm_npc_dialog_response("bot:no"))
    assert fake_rt.action_queue.queued == []