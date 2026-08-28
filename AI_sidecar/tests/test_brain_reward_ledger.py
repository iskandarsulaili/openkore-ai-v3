"""Tests for the Brain Reward Ledger (punish/reward for all brains)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from ai_sidecar.learning.brain_reward_ledger import (
    BRAINS,
    BrainRewardLedger,
    PUNISH_DEATH,
    REWARD_KILL,
    get_brain_reward_ledger,
)


def _fresh_ledger() -> tuple[BrainRewardLedger, Path]:
    tmp = Path(tempfile.mkdtemp(prefix="brain_reward_test_"))
    return BrainRewardLedger(workspace_root=tmp), tmp


def test_ledger_records_and_scores():
    ledger, _ = _fresh_ledger()
    ledger.record("botA", "conscious_llm", REWARD_KILL, "map=prt_fild08")
    ledger.record("botA", "conscious_llm", REWARD_KILL, "map=prt_fild08")
    ledger.record("botA", "conscious_llm", PUNISH_DEATH, "map=prt_fild08")

    scores = ledger.scores("botA")
    assert len(scores) == 1
    s = scores[0]
    assert s.brain == "conscious_llm"
    assert s.events == 3
    assert s.rewards == 2
    assert s.punishments == 1
    # 0.8 + 0.8 - 1.0 = 0.6
    assert abs(s.score - 0.6) < 1e-6


def test_unknown_brain_falls_back_to_conscious():
    ledger, _ = _fresh_ledger()
    ledger.record("botA", "not_a_real_brain", REWARD_KILL)
    scores = ledger.scores("botA")
    assert len(scores) == 1
    assert scores[0].brain == "conscious_llm"


def test_unknown_event_ignored():
    ledger, _ = _fresh_ledger()
    ledger.record("botA", "heuristic", "not_an_event")
    assert ledger.scores("botA") == []


def test_win_rate_and_discount():
    ledger, _ = _fresh_ledger()
    # 10 kills, 0 deaths → win_rate 1.0 → no discount
    for _ in range(10):
        ledger.record("botB", "heuristic", REWARD_KILL)
    assert ledger.discounted_confidence("botB", "heuristic") == 1.0

    # 2 kills, 8 deaths → win_rate 0.2 → heavy discount
    for _ in range(2):
        ledger.record("botE", "heuristic", REWARD_KILL)
    for _ in range(8):
        ledger.record("botE", "heuristic", PUNISH_DEATH)
    assert ledger.discounted_confidence("botE", "heuristic") == 0.5


def test_llm_context_block():
    ledger, _ = _fresh_ledger()
    assert ledger.context_for_llm("botC") == ""
    ledger.record("botC", "conscious_llm", REWARD_KILL)
    ctx = ledger.context_for_llm("botC")
    assert "BRAIN PERFORMANCE" in ctx
    assert "conscious_llm" in ctx


def test_jsonl_persisted():
    ledger, tmp = _fresh_ledger()
    ledger.record("botD", "reflex", REWARD_KILL, "map=prt_fild08")
    files = list((tmp / "AI_sidecar" / "data" / "brain_rewards").glob("rewards-*.jsonl"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert "botD" in content and "reflex" in content


def test_singleton():
    a = get_brain_reward_ledger()
    b = get_brain_reward_ledger()
    assert a is b


def test_ledger_loads_persisted_history_across_restart():
    """A NEW ledger instance (simulating a sidecar restart) must replay the
    JSONL so the brain-rewards observability + LLM feedback survive restarts."""
    tmp = Path(tempfile.mkdtemp(prefix="brain_reward_restart_"))
    l1 = BrainRewardLedger(workspace_root=tmp)
    l1.record("botR", "conscious_llm", REWARD_KILL, "map=prt_fild08")
    l1.record("botR", "conscious_llm", REWARD_KILL, "map=prt_fild08")
    l1.record("botR", "heuristic", PUNISH_DEATH, "map=prt_fild08")

    # new instance = "restart": empty in-memory, must replay from JSONL
    l2 = BrainRewardLedger(workspace_root=tmp)
    scores = l2.scores("botR")
    by_brain = {s.brain: s for s in scores}
    assert by_brain["conscious_llm"].events == 2, "kill history must replay"
    assert by_brain["heuristic"].events == 1, "death history must replay"
    assert abs(by_brain["conscious_llm"].score - 1.6) < 1e-6

    # record on the restarted instance must APPLY ON TOP (not double-count)
    l2.record("botR", "conscious_llm", REWARD_KILL, "map=prt_fild08")
    scores2 = l2.scores("botR")
    assert next(s for s in scores2 if s.brain == "conscious_llm").events == 3
