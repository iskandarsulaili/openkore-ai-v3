"""
Brain Reward Ledger — unified reward/punish feedback for ALL brains + self-*.

User directive (2026-08-28): "We must have punish/reward system for all the
brains and self-*".

Every brain's decision (conscious/LLM, heuristic, reflex, subconscious/ML,
goal decomposer, memory) is scored on OUTCOME:
  REWARD:  kill landed, EXP gained, level-up, HP recovered from critical,
           survival after danger, objective progressed.
  PUNISH:  death, repeated failure, wasted resource, stuck loop, HP critical
           with no recovery, rule violation.

Scores are per-bot, per-brain, persisted to a JSONL ledger, and fed BACK into:
  1. The Conscious (LLM) prompt — each LLM advisory sees the bot's own recent
     brain-performance history (self-aware: it knows which of its plans worked
     and which got the bot killed → preemptive, not just reactive).
  2. Long-term memory (personal_history) — durable lessons.
  3. Heuristic/reflex weighting hints — brains that repeatedly fail get their
     confidence discounted; brains that consistently win are boosted.

Design notes:
- Outcome events are the SAME deltas the memory store watches (kills/deaths/
  EXP/HP), plus explicit action-outcome pairs from the PDCA cycle.
- The ledger is append-only JSONL (cheap, crash-safe); a bounded in-memory
  rolling window feeds prompts. No new DB required.
- Agnostic: no hardcoded item/map/server names — only brain names + outcomes.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

# ── Brain identifiers (the full self-* stack) ─────────────────────────
BRAINS = (
    "conscious_llm",     # LLM advisories (cold-start, gear/sustain)
    "heuristic",         # deterministic heuristic service decisions
    "reflex",            # reflex rule engine (safety floor)
    "subconscious_ml",   # ML/DQN subconscious trained policies
    "goal_decomposer",   # multi-horizon goal planning
    "memory",            # memory store/recall decisions
    "strategy",          # strategy optimizer A/B selection
)

# Outcome event types that credit/punish brains
REWARD_KILL = "kill"                # +0.8
REWARD_EXP = "exp_gain"             # +0.4 (scaled by delta)
REWARD_LEVEL_UP = "level_up"        # +1.0
REWARD_RECOVER = "hp_recovered"     # +0.5 (survived critical)
REWARD_OBJECTIVE = "objective"      # +0.6 (task/quest progress)
PUNISH_DEATH = "death"              # -1.0
PUNISH_FAIL = "repeated_failure"    # -0.4 per recurrence
PUNISH_STUCK = "stuck_loop"         # -0.5
PUNISH_CRITICAL = "hp_critical"     # -0.3 (no recovery)
PUNISH_WASTE = "resource_waste"     # -0.2

_EVENT_DELTA = {
    REWARD_KILL: 0.8,
    REWARD_EXP: 0.4,
    REWARD_LEVEL_UP: 1.0,
    REWARD_RECOVER: 0.5,
    REWARD_OBJECTIVE: 0.6,
    PUNISH_DEATH: -1.0,
    PUNISH_FAIL: -0.4,
    PUNISH_STUCK: -0.5,
    PUNISH_CRITICAL: -0.3,
    PUNISH_WASTE: -0.2,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class BrainScore:
    """Rolling score for one (bot, brain) pair."""
    bot_id: str
    brain: str
    score: float = 0.0          # cumulative signed score (bounded)
    events: int = 0             # total scored events
    rewards: int = 0            # positive events
    punishments: int = 0        # negative events
    last_event_ts: float = 0.0
    last_event: str = ""
    window: deque = field(default_factory=lambda: deque(maxlen=64))

    @property
    def win_rate(self) -> float:
        if self.events == 0:
            return 0.0
        return self.rewards / self.events

    def apply(self, event: str, delta: float, detail: str = "") -> None:
        self.score = max(-10.0, min(10.0, self.score + delta))
        self.events += 1
        if delta > 0:
            self.rewards += 1
        elif delta < 0:
            self.punishments += 1
        self.last_event_ts = time.time()
        self.last_event = event
        self.window.append({"event": event, "delta": delta, "detail": detail,
                            "ts": self.last_event_ts})

    def summary(self) -> str:
        return (
            f"{self.brain}: score={self.score:+.2f} events={self.events} "
            f"(r{self.rewards}/p{self.punishments}) win_rate={self.win_rate:.0%} "
            f"last={self.last_event}"
        )


class BrainRewardLedger:
    """Unified reward/punish ledger for all brains (self-* stack)."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self._lock = RLock()
        self._scores: dict[str, dict[str, BrainScore]] = {}   # bot_id -> brain -> score
        # DURABLE + CWD-INDEPENDENT path: resolve from this file's location so
        # the ledger persists in the SAME place regardless of how the sidecar
        # was started (repo root / AI_sidecar cwd / systemd). A cwd-relative
        # path (Path(".")) silently splits the ledger across restarts.
        if workspace_root is None:
            workspace_root = Path(__file__).resolve().parents[3]  # repo root
        self._root = workspace_root
        self._ledger_dir = self._root / "AI_sidecar" / "data" / "brain_rewards"
        self._ledger_dir.mkdir(parents=True, exist_ok=True)

    # ── API ────────────────────────────────────────────────────────
    def load(self) -> None:
        """Replay persisted JSONL into in-memory scores (idempotent).

        The ledger was WRITE-ONLY: after a sidecar restart the scores/observ-
        ability/LLM-feedback all reset to empty even though the JSONL holds
        the full history. Load replays today's file(s) so the brain-rewards
        endpoint + context_for_llm show the REAL track record across restarts.
        """
        with self._lock:
            if getattr(self, "_loaded", False):
                return
            self._loaded = True
            if not self._ledger_dir.exists():
                return
            for _f in sorted(self._ledger_dir.glob("*.jsonl")):
                try:
                    with open(_f) as _fh:
                        for _line in _fh:
                            _line = _line.strip()
                            if not _line:
                                continue
                            try:
                                _row = json.loads(_line)
                            except Exception:
                                continue
                            _bid = _row.get("bot_id")
                            _br = _row.get("brain")
                            _ev = _row.get("event")
                            if not _bid or not _br or _ev not in _EVENT_DELTA:
                                continue
                            scores = self._scores.setdefault(_bid, {})
                            _s = scores.get(_br)
                            if _s is None:
                                _s = BrainScore(bot_id=_bid, brain=_br)
                                scores[_br] = _s
                            _s.apply(_ev, _EVENT_DELTA[_ev], _row.get("detail", ""))
                except Exception:
                    logger.debug("brain_reward_load_failed", exc_info=True)

    def record(self, bot_id: str, brain: str, event: str, detail: str = "") -> None:
        """Credit/punish a brain for an outcome event (agnostic event names)."""
        if brain not in BRAINS:
            brain = "conscious_llm"  # unknown brain → conscious (safe default)
        if event not in _EVENT_DELTA:
            logger.debug("brain_reward_unknown_event brain=%s event=%s", brain, event)
            return
        self.load()  # replay persisted history BEFORE adding this row (idempotent)
        with self._lock:
            scores = self._scores.setdefault(bot_id, {})
            score = scores.get(brain)
            if score is None:
                score = BrainScore(bot_id=bot_id, brain=brain)
                scores[brain] = score
            score.apply(event, _EVENT_DELTA[event], detail)
        self._append_row(bot_id, brain, event, _EVENT_DELTA[event], detail)

    def _append_row(self, bot_id: str, brain: str, event: str, delta: float,
                    detail: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        row = {
            "ts": _now_iso(), "bot_id": bot_id, "brain": brain,
            "event": event, "delta": delta, "detail": detail,
        }
        try:
            with (self._ledger_dir / f"rewards-{stamp}.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            logger.debug("brain_reward_write_failed", exc_info=True)

    def scores(self, bot_id: str) -> list[BrainScore]:
        self.load()  # replay persisted JSONL first (idempotent)
        with self._lock:
            return list(self._scores.get(bot_id, {}).values())

    def context_for_llm(self, bot_id: str, limit: int = 6) -> str:
        """Formatted brain-performance block for the Conscious LLM prompt.

        The Conscious brain SEES its own + every other brain's recent track
        record — self-aware feedback: a plan that repeatedly failed is
        discounted; a winning approach is reinforced. This is the preemptive
        (not merely reactive) loop the directive asks for.
        """
        scores = self.scores(bot_id)
        if not scores:
            return ""
        scores.sort(key=lambda s: s.score, reverse=True)
        lines = [s.summary() for s in scores[:limit]]
        return "BRAIN PERFORMANCE (reward/punish history): " + " | ".join(lines)

    def discounted_confidence(self, bot_id: str, brain: str) -> float:
        """Confidence multiplier (0.0-1.0) for a brain based on its track record.

        Brains that repeatedly fail are discounted; winners are kept at 1.0.
        """
        with self._lock:
            score = self._scores.get(bot_id, {}).get(brain)
        if score is None:
            self.load()  # replay persisted history before judging a brain
            with self._lock:
                score = self._scores.get(bot_id, {}).get(brain)
        if score is None or score.events < 3:
            return 1.0
        win = score.win_rate
        if win >= 0.6:
            return 1.0
        if win >= 0.4:
            return 0.8
        return 0.5


_LEDGER_SINGLETON: BrainRewardLedger | None = None


def get_brain_reward_ledger(workspace_root: Path | None = None) -> BrainRewardLedger:
    """Module-level singleton accessor (mirrors failure_reasoning pattern)."""
    global _LEDGER_SINGLETON
    if _LEDGER_SINGLETON is None:
        _LEDGER_SINGLETON = BrainRewardLedger(workspace_root=workspace_root)
    return _LEDGER_SINGLETON
