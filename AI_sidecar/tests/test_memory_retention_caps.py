"""Regression test: memory retention caps must be enforced on the SQLite path.

max_episodes_per_bot / max_semantic_per_bot were declared but NEVER enforced
in MemoryRepository — only the in-memory provider pruned. The persistent path
grew unbounded (495K episodes / 1.9GB sidecar.sqlite observed). This test
locks the retention behavior: after exceeding the cap, only the newest N
records per bot survive, and orphaned embeddings are cleaned.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ai_sidecar.persistence.db import SQLiteDB
from ai_sidecar.persistence.models import MemoryEpisodeRecord, MemorySemanticRecord
from ai_sidecar.persistence.repositories import MemoryRepository


@pytest.fixture()
def repo(tmp_path: Path):
    """Return (repository, raw_connection) for retention-cap assertions."""
    db_path = tmp_path / "memory.db"
    raw = sqlite3.connect(str(db_path))
    raw.executescript(
        """
        CREATE TABLE memory_episodes(
            id TEXT PRIMARY KEY, bot_id TEXT, event_type TEXT, content TEXT,
            metadata_json TEXT, created_at TEXT
        );
        CREATE TABLE memory_semantic_records(
            id TEXT PRIMARY KEY, bot_id TEXT, source TEXT, content TEXT,
            lexical_signature TEXT, metadata_json TEXT, created_at TEXT
        );
        CREATE TABLE memory_embeddings(
            memory_id TEXT PRIMARY KEY, dimensions INT, vector_json TEXT, norm REAL
        );
        """
    )
    raw.commit()
    db = SQLiteDB(path=db_path, busy_timeout_ms=5000)
    repo = MemoryRepository(db, max_episodes_per_bot=3, max_semantic_per_bot=2)
    yield repo, raw
    raw.close()


def _episode(i: int) -> MemoryEpisodeRecord:
    return MemoryEpisodeRecord(
        id=f"e{i}", bot_id="bot:1", event_type="t", content=f"c{i}",
        metadata={}, created_at=datetime.now(UTC),
    )


def _semantic(i: int) -> MemorySemanticRecord:
    return MemorySemanticRecord(
        id=f"s{i}", bot_id="bot:1", source="src", content=f"c{i}",
        lexical_signature="", metadata={}, created_at=datetime.now(UTC),
        vector=[0.1] * 8, dimensions=8, norm=1.0,
    )


def test_episode_cap_enforced_keeps_newest(repo) -> None:
    repo_obj, conn = repo
    for i in range(10):
        repo_obj.add_episode(_episode(i))
    n = conn.execute("SELECT COUNT(*) FROM memory_episodes").fetchone()[0]
    assert n == 3, f"expected cap 3, got {n}"
    kept = [r[0] for r in conn.execute(
        "SELECT id FROM memory_episodes ORDER BY created_at DESC, id DESC LIMIT 3"
    ).fetchall()]
    assert kept == ["e9", "e8", "e7"], f"newest must survive: {kept}"


def test_semantic_cap_enforced_and_embeddings_cleaned(repo) -> None:
    repo_obj, conn = repo
    for i in range(10):
        repo_obj.add_semantic(_semantic(i))
    n = conn.execute("SELECT COUNT(*) FROM memory_semantic_records").fetchone()[0]
    emb = conn.execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()[0]
    assert n == 2, f"expected cap 2, got {n}"
    assert emb == 2, f"orphaned embeddings must be cleaned: {emb}"


def test_cap_zero_disables_trim(tmp_path: Path) -> None:
    db_path = tmp_path / "memory-zero.db"
    raw = sqlite3.connect(str(db_path))
    raw.executescript(
        """
        CREATE TABLE memory_episodes(
            id TEXT PRIMARY KEY, bot_id TEXT, event_type TEXT, content TEXT,
            metadata_json TEXT, created_at TEXT
        );
        CREATE TABLE memory_semantic_records(
            id TEXT PRIMARY KEY, bot_id TEXT, source TEXT, content TEXT,
            lexical_signature TEXT, metadata_json TEXT, created_at TEXT
        );
        CREATE TABLE memory_embeddings(
            memory_id TEXT PRIMARY KEY, dimensions INT, vector_json TEXT, norm REAL
        );
        """
    )
    raw.commit()
    db = SQLiteDB(path=db_path, busy_timeout_ms=5000)
    repo2 = MemoryRepository(db, max_episodes_per_bot=0, max_semantic_per_bot=0)
    for i in range(5):
        repo2.add_episode(_episode(100 + i))
    n = raw.execute("SELECT COUNT(*) FROM memory_episodes").fetchone()[0]
    raw.close()
    assert n == 5, f"cap 0 must disable trimming: {n}"
