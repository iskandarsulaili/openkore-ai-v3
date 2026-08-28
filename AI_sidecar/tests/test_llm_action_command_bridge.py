"""Tests for the LLM action->command bridge: an LLM sustain action must always produce
an executable command so the bot can act on the conscious tier's decision."""
import asyncio
from ai_sidecar.autonomy.pdca_loop import PDCALoop


class _FakeLLM:
    def __init__(self, result):
        self._result = result
        self.called = 0

    def is_available(self):
        return True

    async def complete_json(self, prompt, system_prompt="", temperature=0.2):
        self.called += 1
        return self._result


class _FakeQueue:
    def __init__(self):
        self.enqueued = []

    def enqueue(self, bot_id, proposal):
        self.enqueued.append((bot_id, proposal.command, proposal.source))


class _FakeRuntime:
    def __init__(self, llm):
        self.llm_manager = llm
        self.action_queue = _FakeQueue()
        from types import SimpleNamespace
        _snap = SimpleNamespace(
            raw={"map": "prt_fild08c", "death_count": 3},
            position=SimpleNamespace(map="prt_fild08c"),
            vitals=SimpleNamespace(hp_ratio=0.3),
            progression=SimpleNamespace(base_exp=100),
            inventory_items=[],
            has_weapon_in_inventory=True,
            observed_at="2026-08-28T00:00:00+00:00",
        )
        self.snapshot_cache = SimpleNamespace(get=lambda _b: _snap, latest=lambda: _snap)
        # store with a REAL learned potion solution (agnostic, not hardcoded)
        self.server_solutions_store = SimpleNamespace(
            get=lambda k, d=None: {
                "potion_solution": "buy 569 30",
                "safe_town": "prontera",
                "farm_map": "prt_fild08",
            }.get(k, d),
            get_json=lambda k, d=None: ({"buy_command": "buy 569 30", "potion_id": "569"} if k == "potion_solution" else (d or {})),
            get_origin=lambda k: "learned",
        )


def _make_loop(rt):
    loop = object.__new__(PDCALoop)
    loop._runtime = rt
    return loop


def test_action_with_empty_command_is_translated_to_executable():
    """An LLM 'retreat' action with no command must still enqueue a concrete move command."""
    llm = _FakeLLM({"action": "retreat", "command": "", "reason": "Low HP, no potions"})
    rt = _FakeRuntime(llm)
    loop = _make_loop(rt)
    asyncio.run(loop._llm_gear_advisory("botA"))
    assert llm.called == 1
    assert len(rt.action_queue.enqueued) == 1, "empty command must be bridged to a concrete one"
    _, cmd, src = rt.action_queue.enqueued[0]
    assert cmd != "", "must not enqueue an empty command"
    assert src == "crewai"


def test_action_with_command_uses_llm_command():
    """When the LLM gives a concrete command, it is used directly (not overridden)."""
    llm = _FakeLLM({"action": "restock", "command": "buy 501 50", "reason": "restock pots"})
    rt = _FakeRuntime(llm)
    loop = _make_loop(rt)
    asyncio.run(loop._llm_gear_advisory("botA"))
    assert len(rt.action_queue.enqueued) == 1
    assert rt.action_queue.enqueued[0][1] == "buy 501 50"


def test_acquire_potions_action_emits_buy_command():
    """'acquire_potions' with no command must become a potion buy."""
    llm = _FakeLLM({"action": "acquire_potions", "command": "", "reason": "no potions"})
    rt = _FakeRuntime(llm)
    loop = _make_loop(rt)
    asyncio.run(loop._llm_gear_advisory("botA"))
    assert len(rt.action_queue.enqueued) == 1
    assert "buy" in rt.action_queue.enqueued[0][1]


def test_unmappable_action_stays_empty():
    """A keep_farming action maps to 'ai auto'; equip with no item stays empty (no enqueue)."""
    llm = _FakeLLM({"action": "equip", "command": "", "reason": "nothing to equip"})
    rt = _FakeRuntime(llm)
    loop = _make_loop(rt)
    asyncio.run(loop._llm_gear_advisory("botA"))
    assert rt.action_queue.enqueued == [], "empty equip must not enqueue a command"
