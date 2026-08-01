"""Regression test: CrewToolFacade must be reachable via CrewManager.execute_tool.

The facade was implemented and its tool name declared in the crew config,
but execute_tool never dispatched ml_shadow_predict — the tool was
unreachable by any agent. The dispatcher now routes it through the facade
to runtime.ml_predict.
"""

from __future__ import annotations

from types import SimpleNamespace

from ai_sidecar.crewai.crew_manager import CrewManager


class _FakeMLResult:
    ok = True
    model_family = SimpleNamespace(value="heuristic_decision")
    model_version = "v1-test"
    recommendation = {"target_map": "prt_fild08"}
    confidence = 0.87
    shadow = {"delta": 0.2}


class _FakeRuntime:
    def ml_predict(self, payload):
        return _FakeMLResult()


def _cm() -> CrewManager:
    cm = CrewManager(runtime=_FakeRuntime(), model_router=None)
    return cm


def test_ml_shadow_predict_dispatched_via_execute_tool() -> None:
    cm = _cm()
    out = cm.execute_tool(
        bot_id="bot:x",
        tool_name="ml_shadow_predict",
        arguments={"model_family": "heuristic_decision", "objective": "farm efficiently"},
    )
    assert out.get("ok") is True
    assert out.get("family") == "heuristic_decision"
    assert out.get("model_version") == "v1-test"
    assert out.get("recommendation", {}).get("target_map") == "prt_fild08"
    assert out.get("confidence") == 0.87


def test_ml_shadow_predict_unknown_family_returns_allowed() -> None:
    cm = _cm()
    out = cm.execute_tool(
        bot_id="bot:x",
        tool_name="ml_shadow_predict",
        arguments={"model_family": "not_a_family"},
    )
    assert out.get("ok") is False
    assert "allowed_families" in out


def test_other_tools_still_work() -> None:
    cm = _cm()
    out = cm.execute_tool(bot_id="bot:x", tool_name="get_bot_state", arguments={})
    assert out.get("ok") is True
    assert "queue_depth" in out
    unknown = cm.execute_tool(bot_id="bot:x", tool_name="nope", arguments={})
    assert unknown.get("ok") is False
