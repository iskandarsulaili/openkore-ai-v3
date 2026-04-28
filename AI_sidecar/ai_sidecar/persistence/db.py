from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from threading import RLock
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


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
    """
    CREATE TABLE IF NOT EXISTS autonomy_goal_states (
        bot_id TEXT PRIMARY KEY,
        tick_id TEXT,
        horizon TEXT NOT NULL,
        decision_version TEXT NOT NULL,
        selected_goal_key TEXT NOT NULL,
        selected_objective TEXT NOT NULL,
        assessment_json TEXT NOT NULL,
        goal_stack_json TEXT NOT NULL,
        goal_state_json TEXT NOT NULL,
        decided_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_autonomy_goal_states_updated ON autonomy_goal_states(updated_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS sidecar_operations (
        operation_id TEXT PRIMARY KEY,
        bot_id TEXT NOT NULL,
        operation_kind TEXT NOT NULL,
        artifact_kind TEXT NOT NULL,
        artifact_path TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        status TEXT NOT NULL,
        status_reason TEXT NOT NULL DEFAULT '',
        base_checksum TEXT,
        desired_checksum TEXT,
        observed_checksum TEXT,
        linked_action_id TEXT,
        payload_json TEXT NOT NULL,
        error_message TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        reconciled_at TEXT,
        attempt_count INTEGER NOT NULL DEFAULT 0,
        last_attempt_at TEXT,
        last_error_at TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_sidecar_operations_bot_updated ON sidecar_operations(bot_id, updated_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_sidecar_operations_status_updated ON sidecar_operations(status, updated_at DESC)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_sidecar_operations_idempotency ON sidecar_operations(bot_id, idempotency_key)",
)


class SQLiteDB:
    def __init__(self, *, path: Path, busy_timeout_ms: int) -> None:
        self._path = path
        self._busy_timeout_ms = int(busy_timeout_ms)
        self._lock = RLock()
        self._lock_retry_attempts = 4

    @property
    def path(self) -> Path:
        return self._path

    def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            for sql in SCHEMA_STATEMENTS:
                conn.execute(sql)
            conn.commit()
        logger.info(
            "sqlite_initialized",
            extra={"event": "sqlite_initialized", "sqlite_path": str(self._path)},
        )

    def execute(self, sql: str, params: Sequence[object] | None = None) -> int:
        with self._lock:
            return self._with_lock_retry(
                operation="execute",
                sql=sql,
                fn=lambda: self._execute_once(sql=sql, params=params),
            )

    def executemany(self, sql: str, rows: Iterable[Sequence[object]]) -> int:
        with self._lock:
            materialized_rows = list(rows)
            return self._with_lock_retry(
                operation="executemany",
                sql=sql,
                fn=lambda: self._executemany_once(sql=sql, rows=materialized_rows),
            )

    def fetchone(self, sql: str, params: Sequence[object] | None = None) -> sqlite3.Row | None:
        with self._lock:
            return self._with_lock_retry(
                operation="fetchone",
                sql=sql,
                fn=lambda: self._fetchone_once(sql=sql, params=params),
            )

    def fetchall(self, sql: str, params: Sequence[object] | None = None) -> list[sqlite3.Row]:
        with self._lock:
            return self._with_lock_retry(
                operation="fetchall",
                sql=sql,
                fn=lambda: self._fetchall_once(sql=sql, params=params),
            )

    def transaction(self, statements: Iterable[tuple[str, Sequence[object]]]) -> None:
        with self._lock:
            materialized = list(statements)
            self._with_lock_retry(
                operation="transaction",
                sql="BEGIN IMMEDIATE",
                fn=lambda: self._transaction_once(statements=materialized),
            )

    def _execute_once(self, *, sql: str, params: Sequence[object] | None) -> int:
        with self._connect() as conn:
            cursor = conn.execute(sql, tuple(params or ()))
            conn.commit()
            return cursor.rowcount

    def _executemany_once(self, *, sql: str, rows: Sequence[Sequence[object]]) -> int:
        with self._connect() as conn:
            cursor = conn.executemany(sql, list(rows))
            conn.commit()
            return cursor.rowcount

    def _fetchone_once(self, *, sql: str, params: Sequence[object] | None) -> sqlite3.Row | None:
        with self._connect() as conn:
            cursor = conn.execute(sql, tuple(params or ()))
            return cursor.fetchone()

    def _fetchall_once(self, *, sql: str, params: Sequence[object] | None) -> list[sqlite3.Row]:
        with self._connect() as conn:
            cursor = conn.execute(sql, tuple(params or ()))
            return cursor.fetchall()

    def _transaction_once(self, *, statements: Sequence[tuple[str, Sequence[object]]]) -> None:
        with self._connect() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                for sql, params in statements:
                    conn.execute(sql, tuple(params))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _with_lock_retry(self, *, operation: str, sql: str, fn: Callable[[], T]) -> T:
        for attempt in range(1, self._lock_retry_attempts + 1):
            try:
                return fn()
            except sqlite3.OperationalError as exc:
                if not self._is_retryable_lock_error(exc) or attempt >= self._lock_retry_attempts:
                    raise
                backoff_s = min(0.02 * (2 ** (attempt - 1)), 0.25)
                logger.info(
                    "sqlite_retry_on_lock",
                    extra={
                        "event": "sqlite_retry_on_lock",
                        "operation": operation,
                        "attempt": attempt,
                        "max_attempts": self._lock_retry_attempts,
                        "sql": sql[:120],
                        "backoff_s": backoff_s,
                    },
                )
                time.sleep(backoff_s)
        raise RuntimeError("sqlite retry loop exhausted unexpectedly")

    def _is_retryable_lock_error(self, exc: sqlite3.OperationalError) -> bool:
        message = str(exc).strip().lower()
        return "database is locked" in message or "database table is locked" in message or "database schema is locked" in message

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self._path,
            timeout=max(self._busy_timeout_ms / 1000.0, 0.05),
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
        return conn
