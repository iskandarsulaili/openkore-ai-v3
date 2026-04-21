from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path
from threading import RLock

logger = logging.getLogger(__name__)


SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS bot_registry (
        bot_id TEXT PRIMARY KEY,
        bot_name TEXT,
        role TEXT,
        assignment TEXT,
        capabilities_json TEXT NOT NULL DEFAULT '[]',
        attributes_json TEXT NOT NULL DEFAULT '{}',
        first_seen_at TEXT NOT NULL,
        last_seen_at TEXT NOT NULL,
        last_tick_id TEXT,
        liveness_state TEXT NOT NULL DEFAULT 'unknown'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_bot_registry_last_seen ON bot_registry(last_seen_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        bot_id TEXT NOT NULL,
        tick_id TEXT NOT NULL,
        observed_at TEXT NOT NULL,
        ingested_at TEXT NOT NULL,
        snapshot_json TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_snapshots_bot_observed ON snapshots(bot_id, observed_at DESC, id DESC)",
    """
    CREATE TABLE IF NOT EXISTS actions (
        action_id TEXT PRIMARY KEY,
        bot_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        status TEXT NOT NULL,
        priority_tier TEXT NOT NULL,
        conflict_key TEXT,
        idempotency_key TEXT NOT NULL,
        proposal_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        queued_at TEXT NOT NULL,
        dispatched_at TEXT,
        acknowledged_at TEXT,
        ack_success INTEGER,
        ack_result_code TEXT,
        ack_message TEXT NOT NULL DEFAULT '',
        poll_id TEXT,
        status_reason TEXT NOT NULL DEFAULT ''
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_actions_bot_status ON actions(bot_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_actions_bot_queued ON actions(bot_id, queued_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS telemetry_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        bot_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        level TEXT NOT NULL,
        category TEXT NOT NULL,
        event TEXT NOT NULL,
        message TEXT NOT NULL,
        metrics_json TEXT NOT NULL,
        tags_json TEXT NOT NULL,
        ingested_at TEXT NOT NULL,
        is_incident INTEGER NOT NULL DEFAULT 0
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_telemetry_bot_ts ON telemetry_events(bot_id, timestamp DESC, id DESC)",
    "CREATE INDEX IF NOT EXISTS idx_telemetry_incident ON telemetry_events(bot_id, is_incident, id DESC)",
    """
    CREATE TABLE IF NOT EXISTS telemetry_counters (
        counter_key TEXT PRIMARY KEY,
        bot_id TEXT NOT NULL,
        name TEXT NOT NULL,
        value INTEGER NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_telemetry_counters_bot_name ON telemetry_counters(bot_id, name)",
    """
    CREATE TABLE IF NOT EXISTS macro_publications (
        publication_id TEXT PRIMARY KEY,
        bot_id TEXT NOT NULL,
        version TEXT NOT NULL,
        content_sha256 TEXT NOT NULL,
        published_at TEXT NOT NULL,
        manifest_json TEXT NOT NULL,
        paths_json TEXT NOT NULL,
        macro_count INTEGER NOT NULL DEFAULT 0,
        event_macro_count INTEGER NOT NULL DEFAULT 0,
        automacro_count INTEGER NOT NULL DEFAULT 0
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_macro_publications_bot_ts ON macro_publications(bot_id, published_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS audit_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        level TEXT NOT NULL,
        event_type TEXT NOT NULL,
        bot_id TEXT,
        summary TEXT NOT NULL,
        payload_json TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_audit_event_type_ts ON audit_events(event_type, timestamp DESC, id DESC)",
    "CREATE INDEX IF NOT EXISTS idx_audit_bot_ts ON audit_events(bot_id, timestamp DESC, id DESC)",
    """
    CREATE TABLE IF NOT EXISTS memory_episodes (
        id TEXT PRIMARY KEY,
        bot_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        content TEXT NOT NULL,
        metadata_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_memory_episodes_bot_ts ON memory_episodes(bot_id, created_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS memory_semantic_records (
        id TEXT PRIMARY KEY,
        bot_id TEXT NOT NULL,
        source TEXT NOT NULL,
        content TEXT NOT NULL,
        lexical_signature TEXT NOT NULL,
        metadata_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_memory_semantic_bot_ts ON memory_semantic_records(bot_id, created_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS memory_embeddings (
        memory_id TEXT PRIMARY KEY,
        dimensions INTEGER NOT NULL,
        vector_json TEXT NOT NULL,
        norm REAL NOT NULL,
        FOREIGN KEY(memory_id) REFERENCES memory_semantic_records(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ingest_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT NOT NULL UNIQUE,
        bot_id TEXT NOT NULL,
        observed_at TEXT NOT NULL,
        ingested_at TEXT NOT NULL,
        event_family TEXT NOT NULL,
        event_type TEXT NOT NULL,
        severity TEXT NOT NULL,
        source_hook TEXT,
        correlation_id TEXT,
        text TEXT NOT NULL,
        tags_json TEXT NOT NULL,
        numeric_json TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        event_json TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ingest_events_bot_ts ON ingest_events(bot_id, observed_at DESC, id DESC)",
    "CREATE INDEX IF NOT EXISTS idx_ingest_events_family_type ON ingest_events(event_family, event_type)",
)


class SQLiteDB:
    def __init__(self, *, path: Path, busy_timeout_ms: int) -> None:
        self._path = path
        self._busy_timeout_ms = int(busy_timeout_ms)
        self._lock = RLock()

    @property
    def path(self) -> Path:
        return self._path

    def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            for sql in SCHEMA_STATEMENTS:
                conn.execute(sql)
            conn.commit()
        logger.info(
            "sqlite_initialized",
            extra={"event": "sqlite_initialized", "sqlite_path": str(self._path)},
        )

    def execute(self, sql: str, params: Sequence[object] | None = None) -> int:
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(sql, tuple(params or ()))
                conn.commit()
                return cursor.rowcount

    def executemany(self, sql: str, rows: Iterable[Sequence[object]]) -> int:
        with self._lock:
            with self._connect() as conn:
                cursor = conn.executemany(sql, list(rows))
                conn.commit()
                return cursor.rowcount

    def fetchone(self, sql: str, params: Sequence[object] | None = None) -> sqlite3.Row | None:
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(sql, tuple(params or ()))
                return cursor.fetchone()

    def fetchall(self, sql: str, params: Sequence[object] | None = None) -> list[sqlite3.Row]:
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(sql, tuple(params or ()))
                return cursor.fetchall()

    def transaction(self, statements: Iterable[tuple[str, Sequence[object]]]) -> None:
        with self._lock:
            with self._connect() as conn:
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    for sql, params in statements:
                        conn.execute(sql, tuple(params))
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self._path,
            timeout=max(self._busy_timeout_ms / 1000.0, 0.05),
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
        return conn
