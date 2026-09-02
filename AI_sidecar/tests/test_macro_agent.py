"""Tests for the macro-agent system (verifier + agent + job_change skill-set).

Covers:
- MacroVerifier: parse-check (valid/invalid), security (forbidden tokens),
  dry-run (command resolution), outcome (talknpc needs prior move).
- MacroAgent: LLM generation + verification + register + reward/punish.
- job_change skill-set: the MacroIntelligence process_triggers switch returns
  the job_change pattern for an eligible novice and NOT for a non-eligible one.
"""
from __future__ import annotations

from ai_sidecar.autonomy.macro_agent import MacroAgent
from ai_sidecar.autonomy.macro_intelligence import MacroIntelligence
from ai_sidecar.autonomy.macro_verifier import MacroVerifier, verify_macro_text


# ── MacroVerifier ─────────────────────────────────────────────────────────

def test_verifier_accepts_valid_macro():
    text = (
        "# Generated\n"
        "macro test_case {\n"
        "    log handling case\n"
        "    move alberta_in 53 43\n"
        "    talknpc 53 43 c\n"
        "    stop\n"
        "}\n"
    )
    result = verify_macro_text(text)
    assert result.ok, result.errors
    assert result.macro_count == 1


def test_verifier_rejects_unknown_command():
    text = (
        "macro bad_case {\n"
        "    frobnicate the widget\n"
        "    stop\n"
        "}\n"
    )
    result = verify_macro_text(text)
    assert not result.ok
    assert any("unknown command" in e for e in result.errors)


def test_verifier_rejects_forbidden_token():
    text = (
        "macro evil_case {\n"
        "    do eval system('rm -rf /')\n"
        "    stop\n"
        "}\n"
    )
    result = verify_macro_text(text)
    assert not result.ok
    assert any("forbidden" in e for e in result.errors)


def test_verifier_rejects_move_0_0():
    text = (
        "macro bad_move {\n"
        "    move 0 0\n"
        "    stop\n"
        "}\n"
    )
    result = verify_macro_text(text)
    assert not result.ok
    assert any("move 0 0" in e for e in result.errors)


def test_verifier_rejects_unbalanced_braces():
    text = (
        "macro bad_brace {\n"
        "    log no close\n"
    )
    result = verify_macro_text(text)
    assert not result.ok
    assert any("unbalanced" in e for e in result.errors)


def test_verifier_dry_run_resolves_commands():
    text = (
        "macro test_case {\n"
        "    log handling case\n"
        "    move alberta_in 53 43\n"
        "    stop\n"
        "}\n"
    )
    result = verify_macro_text(text)
    assert result.ok
    assert any("move alberta_in 53 43" in c for c in result.commands)


# ── MacroAgent ────────────────────────────────────────────────────────────

def test_macro_agent_generate_verifies_and_registers():
    def fake_llm(prompt, bot_id="default", workload="macro_generation"):
        return (
            "macro job_change_route {\n"
            "    log routing to job change\n"
            "    move alberta_in 53 43\n"
            "    talknpc 53 43 c\n"
            "    stop\n"
            "}\n"
        )

    agent = MacroAgent(llm_generate=fake_llm)
    macro = agent.generate(
        case="novice job change",
        context={"category": "job_change", "priority": 20},
        bot_id="testbot",
    )
    assert macro is not None
    assert macro.verified, macro.verification.errors if macro.verification else "no verification"
    assert macro.name == "job_change_route"

    # register into a real MacroIntelligence engine
    mi = MacroIntelligence()
    assert agent.register(macro, mi)
    assert "job_change_route" in mi.get_all_patterns()


def test_macro_agent_reward_punish_updates_score():
    def fake_llm(prompt, bot_id="default", workload="macro_generation"):
        return "macro x {\n    log x\n    stop\n}\n"

    agent = MacroAgent(llm_generate=fake_llm)
    macro = agent.generate(case="test", bot_id="b")
    assert macro is not None and macro.verified
    agent.register(macro)
    agent.reward(macro.name, bot_id="b")
    agent.punish(macro.name, bot_id="b")
    entry = next(m for m in agent.registry() if m["name"] == macro.name)
    assert entry["uses"] == 2


def test_macro_agent_rejects_unverified():
    def fake_llm(prompt, bot_id="default", workload="macro_generation"):
        return "macro bad {\n    frobnicate x\n    stop\n}\n"

    agent = MacroAgent(llm_generate=fake_llm)
    macro = agent.generate(case="test", bot_id="b")
    assert macro is not None
    assert not macro.verified
    assert not agent.register(macro)


# ── job_change skill-set (process_triggers switch) ────────────────────────

def test_job_change_pattern_wins_for_eligible_novice():
    mi = MacroIntelligence()
    state = {
        "progression": {
            "job_name": "novice",
            "base_level": 26,
            "job_level": 10,
            "job_changed": False,
        },
        "combat": {"is_in_combat": False},
    }
    winner = mi.process_triggers(bot_state=state, bot_id="testbot")
    assert winner is not None
    assert winner.category == "job_change"
    assert winner.pattern_id == "job_change_novice"


def test_job_change_pattern_not_for_non_eligible():
    mi = MacroIntelligence()
    state = {
        "progression": {
            "job_name": "novice",
            "base_level": 5,
            "job_level": 3,
            "job_changed": False,
        },
        "combat": {"is_in_combat": False},
    }
    winner = mi.process_triggers(bot_state=state, bot_id="testbot")
    # no job_change pattern should win for a level-5 novice
    assert winner is None or winner.category != "job_change"


def test_job_change_pattern_not_after_changed():
    mi = MacroIntelligence()
    state = {
        "progression": {
            "job_name": "merchant",
            "base_level": 26,
            "job_level": 10,
            "job_changed": True,
        },
        "combat": {"is_in_combat": False},
    }
    winner = mi.process_triggers(bot_state=state, bot_id="testbot")
    assert winner is None or winner.category != "job_change"
