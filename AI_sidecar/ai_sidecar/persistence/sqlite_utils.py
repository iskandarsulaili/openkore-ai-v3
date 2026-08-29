"""Shared hardening helpers for ad-hoc sqlite3.connect() call sites.

The central SQLiteDB (persistence/db.py) already applies WAL, busy_timeout
and lock-retry. Several modules open their own per-call connections and were
missing those pragmas, which makes them vulnerable to transient
"database is locked" failures when the PDCA thread, HTTP request threads and
the HighFreqReflex thread write the same files concurrently.

Use `connect()` from this module instead of a bare sqlite3.connect() so every
ad-hoc connection gets the same durability/concurrency posture:

- journal_mode=WAL  (readers never block the writer)
- busy_timeout      (wait instead of failing instantly on lock contention)
- synchronous=NORMAL (safe with WAL, much faster than FULL)

Retry behavior mirrors SQLiteDB._with_lock_retry: retryable lock errors are
retried with exponential backoff.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_DEFAULT_TIMEOUT_S = 5.0
_RETRY_ATTEMPTS = 4


def _is_retryable_lock_error(exc: sqlite3.OperationalError) -> bool:
    message = str(exc).strip().lower()
    return (
        "database is locked" in message
        or "database table is locked" in message
        or "database schema is locked" in message
    )


def connect(
    path: str | Path, timeout: float = _DEFAULT_TIMEOUT_S
) -> sqlite3.Connection:
    """Open a hardened sqlite connection (WAL + busy_timeout + NORMAL sync)."""
    conn = sqlite3.connect(str(path), timeout=max(timeout, _DEFAULT_TIMEOUT_S))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(
            f"PRAGMA busy_timeout={int(max(timeout, _DEFAULT_TIMEOUT_S) * 1000)}"
        )
    except sqlite3.Error:
        # A read-only filesystem or an in-memory DB must not break callers.
        pass
    return conn


def run_with_retry(
    operation: str, fn: Callable[[], T], *, attempts: int = _RETRY_ATTEMPTS
) -> T:
    """Run fn, retrying transient sqlite lock errors with backoff."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except sqlite3.OperationalError as exc:
            if not _is_retryable_lock_error(exc) or attempt >= attempts:
                raise
            backoff_s = min(0.02 * (2 ** (attempt - 1)), 0.25)
            logger.info(
                "sqlite_adhoc_retry_on_lock",
                extra={
                    "event": "sqlite_adhoc_retry_on_lock",
                    "operation": operation,
                    "attempt": attempt,
                    "max_attempts": attempts,
                    "backoff_s": backoff_s,
                },
            )
            time.sleep(backoff_s)
    raise RuntimeError("sqlite adhoc retry loop exhausted unexpectedly")


def execute(
    path: str | Path,
    sql: str,
    params: tuple | list = (),
    *,
    timeout: float = _DEFAULT_TIMEOUT_S,
    many: bool = False,
) -> int:
    """Hardened write helper: connect, execute (+commit), return rowcount."""

    def _once() -> int:
        conn = connect(path, timeout=timeout)
        try:
            cursor = (
                conn.executemany(sql, [tuple(p) for p in params])
                if many
                else conn.execute(sql, tuple(params))
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    return run_with_retry(f"execute:{sql[:60]}", _once)


def query_all(
    path: str | Path,
    sql: str,
    params: tuple | list = (),
    *,
    timeout: float = _DEFAULT_TIMEOUT_S,
) -> list[dict[str, Any]]:
    """Hardened read helper returning rows as dicts."""

    def _once() -> list[dict[str, Any]]:
        conn = connect(path, timeout=timeout)
        try:
            cursor = conn.execute(sql, tuple(params))
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        finally:
            conn.close()

    return run_with_retry(f"query:{sql[:60]}", _once)
