from __future__ import annotations

import asyncio
import types
from dataclasses import dataclass, field
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