"""Regression test: ActionRepository must cap per-bot history.

The actions table persisted every enqueued action forever (unbounded
growth — no retention unlike snapshots/telemetry/audit). The trim now
keeps the newest max_history_per_bot actions per bot.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from ai_sidecar.contracts.actions import ActionPriorityTier, ActionProposal, ActionStatus
from ai_sidecar.persistence.db import SQLiteDB
from ai_sidecar.persistence.repositories import ActionRepository


def _proposal(i: int) -> ActionProposal:
    return ActionProposal(
        action_id=f"a{i:05d}", kind="command", command="move prontera",
        conflict_key="", priority_tier=ActionPriorityTier.tactical, source="test",
        created_at=datetime.now(UTC), expires_at=datetime.now(UTC) + timedelta(seconds=30),
        idempotency_key=f"k{i:05d}",
    )


def _repo(tmp_path: Path, cap: int = 100) -> ActionRepository:
    db = SQLiteDB(path=tmp_path / "actions.db", busy_timeout_ms=5000)
    db.initialize()
    return ActionRepository(db, max_history_per_bot=cap)


def test_action_history_capped_per_bot(tmp_path: Path) -> None:
    repo = _repo(tmp_path, cap=100)
    for i in range(250):
        repo.upsert_action(bot_id="bot:x", proposal=_proposal(i),
                           status=ActionStatus.queued, status_reason="test")
    assert repo.count() == 100, f"cap 100 must hold, got {repo.count()}"


def test_newest_actions_survive_trim(tmp_path: Path) -> None:
    repo = _repo(tmp_path, cap=100)
    for i in range(250):
        repo.upsert_action(bot_id="bot:x", proposal=_proposal(i),
                           status=ActionStatus.queued, status_reason="test")
    recent = repo.list_recent(bot_id="bot:x", limit=3)
    ids = [r.action_id for r in recent]
    assert "a00249" in ids, f"newest action must survive, got {ids}"


def test_trim_is_per_bot(tmp_path: Path) -> None:
    repo = _repo(tmp_path, cap=50)
    for i in range(200):
        repo.upsert_action(bot_id="bot:a", proposal=_proposal(i),
                           status=ActionStatus.queued, status_reason="test")
    for i in range(10):
        repo.upsert_action(bot_id="bot:b", proposal=_proposal(i),
                           status=ActionStatus.queued, status_reason="test")
    assert repo.count(bot_id="bot:a") == 50
    assert repo.count(bot_id="bot:b") == 10, "small bot must be untouched by other bot's trim"
