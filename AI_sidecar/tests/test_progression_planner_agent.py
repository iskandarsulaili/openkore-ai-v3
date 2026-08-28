"""Progression planner: empty job-change NPC must not crash or loop.

Regression for the job_change_available=True but job_change_npc="" case
(cold start / unknown server / table missing the job): the old code returned
None from get_action -> crew_manager line 285 `action.get(...)` threw
AttributeError -> crew_strategize_failed every cycle (a crash-loop), OR when
guarded, the 0.8 can_handle score kept re-selecting the same profile forever
(a no-op loop). The fix: get_action degrades to a graceful no-op action +
can_handle scores LOW (0.2) when the NPC is unresolved so the LLM conscious
brain / other profiles take over.
"""

from __future__ import annotations

from ai_sidecar.crewai.agents.progression_planner_agent import (
    ProgressionPlannerProfile,
)


def test_get_action_never_none_when_eligible_but_unresolved() -> None:
    """Eligible (job_change_available) + NPC unresolved -> a dict action (never
    None), command empty, so crew_manager's `action.get("command")` is safe."""
    agent = ProgressionPlannerProfile()
    action = agent.get_action({
        "job_change_available": True,
        "job_change_npc": "",
        "equipment": [],
        "level": 6,
    })
    assert isinstance(action, dict), f"expected dict, got {type(action)}"
    assert action["command"] == ""
    assert action["kind"] == "command"


def test_get_action_full_flow_when_npc_resolved() -> None:
    """Eligible + NPC resolved -> the full move+talknpc flow (unchanged)."""
    agent = ProgressionPlannerProfile()
    action = agent.get_action({
        "job_change_available": True,
        "job_change_npc": "move izlude_in 74 172",
        "equipment": [],
        "level": 6,
    })
    assert isinstance(action, dict)
    assert action["kind"] == "job_change"
    assert action["command"] == "move izlude_in 74 172"
    assert action["metadata"]["followup_command"] == "talknpc 74 172"


def test_can_handle_low_when_unresolved() -> None:
    """Eligible but NPC unresolved -> LOW score (0.2, not 0.8) so the profile
    doesn't monopolize selection and loop a no-op plan."""
    agent = ProgressionPlannerProfile()
    low = agent.can_handle({
        "job_change_available": True,
        "job_change_npc": "",
        "level": 6,
    })
    high = agent.can_handle({
        "job_change_available": True,
        "job_change_npc": "move izlude_in 74 172",
        "level": 6,
    })
    assert low < 0.5, f"unresolved NPC must score low, got {low}"
    assert high >= 0.8, f"resolved NPC must score high, got {high}"


def test_get_action_none_only_when_nothing_to_do() -> None:
    """Not eligible + no outdated equipment -> None is still legal (the crew
    manager now guards it); the bot is genuinely idle."""
    agent = ProgressionPlannerProfile()
    action = agent.get_action({
        "job_change_available": False,
        "job_change_npc": None,
        "equipment": [],
        "level": 6,
    })
    assert action is None
