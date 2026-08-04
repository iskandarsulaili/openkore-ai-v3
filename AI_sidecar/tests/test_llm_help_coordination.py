"""Tests for the conscious-tier team-play help coordination (LLM-driven)."""
import asyncio
from ai_sidecar.autonomy.pdca_loop import PDCALoop


class _FakeLLM:
    """Minimal fake LLM manager exposing complete_json + is_available."""
    def __init__(self, result):
        self._result = result
        self.called = 0
        self.last_prompt = ""

    def is_available(self) -> bool:
        return True

    async def complete_json(self, prompt, system_prompt="", temperature=0.2):
        self.called += 1
        self.last_prompt = prompt
        return self._result


class _FakeQueue:
    def __init__(self):
        self.enqueued = []

    def enqueue(self, bot_id, proposal):
        self.enqueued.append((bot_id, proposal.command, proposal.source))


class _FakeRuntime:
    """Minimal runtime with llm_manager, action_queue, and a snapshot."""
    def __init__(self, llm, members=None):
        self.llm_manager = llm
        self.action_queue = _FakeQueue()
        self._last_snapshot = {"botA": {"map": "prt_fild08c", "party_members": members or []}}


def _make_loop(runtime):
    loop = object.__new__(PDCALoop)
    loop._runtime = runtime
    return loop


def test_help_coordination_helps_teammate_in_danger():
    """A teammate at low HP should trigger an LLM-driven help command enqueued."""
    llm = _FakeLLM({"should_help": True, "target": "botB", "command": "use_skill Heal botB", "reason": "teammate low"})
    rt = _FakeRuntime(llm, members=[{"name": "botB", "hp_ratio": 0.2, "map": "prt_fild08c", "dead": False}])
    loop = _make_loop(rt)
    asyncio.run(loop._llm_help_coordination("botA"))
    assert llm.called == 1, "LLM should be consulted when party state exists"
    assert len(rt.action_queue.enqueued) == 1, "help command should be enqueued"
    _, cmd, src = rt.action_queue.enqueued[0]
    assert cmd == "use_skill Heal botB"
    assert src == "crewai"


def test_help_coordination_noop_without_party():
    """With no party state, the LLM should NOT be consulted (nothing to coordinate)."""
    llm = _FakeLLM({"should_help": False, "command": ""})
    rt = _FakeRuntime(llm, members=[])
    loop = _make_loop(rt)
    asyncio.run(loop._llm_help_coordination("botA"))
    assert llm.called == 0, "LLM should not be called without party members"
    assert rt.action_queue.enqueued == []


def test_help_coordination_stays_put_when_llm_says_no():
    """LLM deciding 'should_help=false' must NOT enqueue a command."""
    llm = _FakeLLM({"should_help": False, "target": "", "command": "", "reason": "teammates fine"})
    rt = _FakeRuntime(llm, members=[{"name": "botB", "hp_ratio": 0.9, "map": "prt_fild08c", "dead": False}])
    loop = _make_loop(rt)
    asyncio.run(loop._llm_help_coordination("botA"))
    assert llm.called == 1
    assert rt.action_queue.enqueued == []
