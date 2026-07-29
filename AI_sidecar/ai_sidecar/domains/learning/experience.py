"""Experience tracker — SQLite-backed persistence for bot outcomes.

Records kills, deaths, loot, and experience gained per map, and
provides aggregate statistics for adaptation decisions.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# ── Default database path ───────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
_DEFAULT_DB_PATH = os.path.normpath(os.path.join(DATA_DIR, "learning.db"))

# ── Global singleton (thread-safe) ─────────────────────────────────

_instance: ExperienceTracker | None = None
_lock = threading.Lock()


def get_experience_tracker(db_path: str = "") -> ExperienceTracker:
    """Get or create the singleton ExperienceTracker."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                path = db_path or _DEFAULT_DB_PATH
                _instance = ExperienceTracker(path)
    return _instance


# ── SQL helpers ─────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS map_stats (
    map_name       TEXT NOT NULL,
    bot_id         TEXT NOT NULL DEFAULT 'default',
    total_kills    INTEGER NOT NULL DEFAULT 0,
    total_deaths   INTEGER NOT NULL DEFAULT 0,
    total_loot     INTEGER NOT NULL DEFAULT 0,
    total_exp      INTEGER NOT NULL DEFAULT 0,
    session_seconds REAL NOT NULL DEFAULT 0.0,
    last_updated   REAL NOT NULL,
    PRIMARY KEY (map_name, bot_id)
);

CREATE TABLE IF NOT EXISTS event_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    bot_id         TEXT NOT NULL DEFAULT 'default',
    map_name       TEXT NOT NULL,
    event_type     TEXT NOT NULL,  -- 'kill', 'death', 'loot', 'exp'
    value          REAL NOT NULL DEFAULT 0,
    timestamp      REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_event_log_bot_map
    ON event_log (bot_id, map_name);

CREATE INDEX IF NOT EXISTS idx_event_log_time
    ON event_log (timestamp);

CREATE TABLE IF NOT EXISTS map_metadata (
    map_name       TEXT NOT NULL,
    bot_id         TEXT NOT NULL DEFAULT 'default',
    suggested_level_min INTEGER NOT NULL DEFAULT 1,
    suggested_level_max INTEGER NOT NULL DEFAULT 99,
    optimal_party_size  INTEGER NOT NULL DEFAULT 1,
    is_safe            INTEGER NOT NULL DEFAULT 1,
    last_visited       REAL NOT NULL,
    PRIMARY KEY (map_name, bot_id)
);

CREATE TABLE IF NOT EXISTS session_stats (
    bot_id         TEXT NOT NULL,
    session_start  REAL NOT NULL,
    session_end    REAL,
    total_exp      INTEGER NOT NULL DEFAULT 0,
    total_zeny     INTEGER NOT NULL DEFAULT 0,
    total_kills    INTEGER NOT NULL DEFAULT 0,
    total_deaths   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (bot_id, session_start)
);
"""


# ── ExperienceTracker ───────────────────────────────────────────────

class ExperienceTracker:
    """Thread-safe SQLite-backed experience tracker.

    Records outcome events per map and provides aggregate statistics
    used by the strategy adapter.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._local = threading.local()
        self._lock = threading.RLock()
        self._session_start = time.time()
        # Per-bot session accumulators (flushed to DB periodically)
        self._session_cache: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "kills": 0, "deaths": 0, "loot": 0, "exp": 0,
                "start": time.time(),
            }
        )
        self._ensure_schema()
        logger.info("ExperienceTracker: db=%s", db_path)

    # ── Connection management ───────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        """Get a thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
            conn = sqlite3.connect(self._db_path, timeout=10)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return self._local.conn

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        conn = self._get_conn()
        conn.executescript(_SCHEMA_SQL)
        conn.commit()

    def close(self) -> None:
        """Close the thread-local connection if open."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None

    # ── Recording events ────────────────────────────────────────────

    def record_kill(self, map_name: str, bot_id: str = "default") -> None:
        """Record a single kill on a map."""
        now = time.time()
        with self._lock:
            self._cache_incr(bot_id, "kills")
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO event_log (bot_id, map_name, event_type, value, timestamp)
                   VALUES (?, ?, 'kill', 1, ?)""",
                (bot_id, map_name, now),
            )
            conn.execute(
                """INSERT INTO map_stats (map_name, bot_id, total_kills, total_deaths,
                                          total_loot, total_exp, session_seconds, last_updated)
                   VALUES (?, ?, 1, 0, 0, 0, 0.0, ?)
                   ON CONFLICT(map_name, bot_id) DO UPDATE SET
                       total_kills = total_kills + 1,
                       last_updated = ?""",
                (map_name, bot_id, now, now),
            )
            conn.commit()

    def record_death(self, map_name: str, bot_id: str = "default") -> None:
        """Record a single death on a map."""
        now = time.time()
        with self._lock:
            self._cache_incr(bot_id, "deaths")
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO event_log (bot_id, map_name, event_type, value, timestamp)
                   VALUES (?, ?, 'death', 1, ?)""",
                (bot_id, map_name, now),
            )
            conn.execute(
                """INSERT INTO map_stats (map_name, bot_id, total_kills, total_deaths,
                                          total_loot, total_exp, session_seconds, last_updated)
                   VALUES (?, ?, 0, 1, 0, 0, 0.0, ?)
                   ON CONFLICT(map_name, bot_id) DO UPDATE SET
                       total_deaths = total_deaths + 1,
                       last_updated = ?""",
                (map_name, bot_id, now, now),
            )
            conn.commit()

    def record_loot(
        self, map_name: str, zeny: float, bot_id: str = "default",
    ) -> None:
        """Record loot/zeny gained on a map."""
        now = time.time()
        with self._lock:
            self._cache_incr(bot_id, "loot", zeny)
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO event_log (bot_id, map_name, event_type, value, timestamp)
                   VALUES (?, ?, 'loot', ?, ?)""",
                (bot_id, map_name, zeny, now),
            )
            conn.execute(
                """INSERT INTO map_stats (map_name, bot_id, total_kills, total_deaths,
                                          total_loot, total_exp, session_seconds, last_updated)
                   VALUES (?, ?, 0, 0, ?, 0, 0.0, ?)
                   ON CONFLICT(map_name, bot_id) DO UPDATE SET
                       total_loot = total_loot + ?,
                       last_updated = ?""",
                (map_name, bot_id, zeny, now, zeny, now),
            )
            conn.commit()

    def record_exp(
        self, map_name: str, exp: int, bot_id: str = "default",
    ) -> None:
        """Record experience gained on a map."""
        now = time.time()
        with self._lock:
            self._cache_incr(bot_id, "exp", exp)
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO event_log (bot_id, map_name, event_type, value, timestamp)
                   VALUES (?, ?, 'exp', ?, ?)""",
                (bot_id, map_name, exp, now),
            )
            conn.execute(
                """INSERT INTO map_stats (map_name, bot_id, total_kills, total_deaths,
                                          total_loot, total_exp, session_seconds, last_updated)
                   VALUES (?, ?, 0, 0, 0, ?, 0.0, ?)
                   ON CONFLICT(map_name, bot_id) DO UPDATE SET
                       total_exp = total_exp + ?,
                       last_updated = ?""",
                (map_name, bot_id, exp, now, exp, now),
            )
            conn.commit()

    def record_session_time(
        self, map_name: str, seconds: float, bot_id: str = "default",
    ) -> None:
        """Record time spent on a map."""
        now = time.time()
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO map_stats (map_name, bot_id, total_kills, total_deaths,
                                          total_loot, total_exp, session_seconds, last_updated)
                   VALUES (?, ?, 0, 0, 0, 0, ?, ?)
                   ON CONFLICT(map_name, bot_id) DO UPDATE SET
                       session_seconds = session_seconds + ?,
                       last_updated = ?""",
                (map_name, bot_id, seconds, now, seconds, now),
            )
            conn.commit()

    def _cache_incr(
        self, bot_id: str, key: str, amount: float = 1,
    ) -> None:
        """Increment a session cache counter."""
        self._session_cache[bot_id][key] += amount

    # ── Querying ────────────────────────────────────────────────────

    def get_map_stats(
        self, map_name: str, bot_id: str = "default",
    ) -> dict[str, Any]:
        """Get aggregate stats for a specific map and bot.

        Returns:
            dict with keys: map_name, bot_id, total_kills, total_deaths,
            total_loot, total_exp, session_seconds, kill_rate, death_rate,
            loot_rate, exp_rate, last_updated
        """
        conn = self._get_conn()
        row = conn.execute(
            """SELECT * FROM map_stats WHERE map_name = ? AND bot_id = ?""",
            (map_name, bot_id),
        ).fetchone()

        if row is None:
            return {
                "map_name": map_name,
                "bot_id": bot_id,
                "total_kills": 0,
                "total_deaths": 0,
                "total_loot": 0.0,
                "total_exp": 0,
                "session_seconds": 0.0,
                "kill_rate": 0.0,
                "death_rate": 0.0,
                "loot_rate": 0.0,
                "exp_rate": 0.0,
                "last_updated": 0.0,
            }

        result = dict(row)
        hours = max(result["session_seconds"] / 3600.0, 0.001)
        result["kill_rate"] = result["total_kills"] / hours
        result["death_rate"] = result["total_deaths"] / hours
        result["loot_rate"] = result["total_loot"] / hours
        result["exp_rate"] = result["total_exp"] / hours
        return result

    def get_all_map_stats(self, bot_id: str = "default") -> list[dict[str, Any]]:
        """Get stats for all maps a bot has visited."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT * FROM map_stats WHERE bot_id = ? ORDER BY last_updated DESC""",
            (bot_id,),
        ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            d = dict(row)
            hours = max(d["session_seconds"] / 3600.0, 0.001)
            d["kill_rate"] = d["total_kills"] / hours
            d["death_rate"] = d["total_deaths"] / hours
            d["loot_rate"] = d["total_loot"] / hours
            d["exp_rate"] = d["total_exp"] / hours
            results.append(d)
        return results

    def get_recent_events(
        self,
        map_name: str = "",
        bot_id: str = "default",
        minutes: int = 60,
    ) -> list[dict[str, Any]]:
        """Get recent events, optionally filtered by map."""
        cutoff = time.time() - (minutes * 60)
        conn = self._get_conn()
        if map_name:
            rows = conn.execute(
                """SELECT * FROM event_log
                   WHERE bot_id = ? AND map_name = ? AND timestamp >= ?
                   ORDER BY timestamp DESC""",
                (bot_id, map_name, cutoff),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM event_log
                   WHERE bot_id = ? AND timestamp >= ?
                   ORDER BY timestamp DESC""",
                (bot_id, cutoff),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_death_rate_per_hour(
        self, map_name: str, bot_id: str = "default",
    ) -> float:
        """Get death rate per hour for a map."""
        stats = self.get_map_stats(map_name, bot_id)
        return stats["death_rate"]

    def get_exp_rate_per_hour(
        self, map_name: str, bot_id: str = "default",
    ) -> float:
        """Get experience rate per hour for a map."""
        stats = self.get_map_stats(map_name, bot_id)
        return stats["exp_rate"]

    def get_loot_rate_per_hour(
        self, map_name: str, bot_id: str = "default",
    ) -> float:
        """Get loot/zeny rate per hour for a map."""
        stats = self.get_map_stats(map_name, bot_id)
        return stats["loot_rate"]

    def get_kill_rate_per_hour(
        self, map_name: str, bot_id: str = "default",
    ) -> float:
        """Get kill rate per hour for a map."""
        stats = self.get_map_stats(map_name, bot_id)
        return stats["kill_rate"]

    def get_session_stats(self, bot_id: str = "default") -> dict[str, Any]:
        """Get current session aggregate from cache + DB."""
        with self._lock:
            cache = self._session_cache.get(bot_id, {})
            conn = self._get_conn()
            row = conn.execute(
                """SELECT COALESCE(SUM(total_kills), 0) as kills,
                          COALESCE(SUM(total_deaths), 0) as deaths,
                          COALESCE(SUM(total_loot), 0) as loot,
                          COALESCE(SUM(total_exp), 0) as exp
                   FROM map_stats WHERE bot_id = ?""",
                (bot_id,),
            ).fetchone()
            elapsed = time.time() - self._session_start
            hours = max(elapsed / 3600.0, 0.001)
            db_kills = row["kills"] if row else 0
            db_deaths = row["deaths"] if row else 0
            db_loot = row["loot"] if row else 0
            db_exp = row["exp"] if row else 0
            return {
                "total_kills": int(db_kills),
                "total_deaths": int(db_deaths),
                "total_loot": float(db_loot),
                "total_exp": int(db_exp),
                "session_seconds": elapsed,
                "kill_rate": (db_kills or 0) / hours,
                "death_rate": (db_deaths or 0) / hours,
                "loot_rate": (db_loot or 0) / hours,
                "exp_rate": (db_exp or 0) / hours,
            }

    def get_map_count(self) -> int:
        """Get the number of distinct maps tracked."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(DISTINCT map_name) as cnt FROM map_stats"
        ).fetchone()
        return row["cnt"] if row else 0

    def update_map_metadata(
        self,
        map_name: str,
        bot_id: str = "default",
        level_min: int | None = None,
        level_max: int | None = None,
        is_safe: bool | None = None,
    ) -> None:
        """Update metadata for a map (level range, safety)."""
        now = time.time()
        conn = self._get_conn()
        existing = conn.execute(
            "SELECT * FROM map_metadata WHERE map_name = ? AND bot_id = ?",
            (map_name, bot_id),
        ).fetchone()

        if existing:
            updates = []
            params = []
            if level_min is not None:
                updates.append("suggested_level_min = ?")
                params.append(level_min)
            if level_max is not None:
                updates.append("suggested_level_max = ?")
                params.append(level_max)
            if is_safe is not None:
                updates.append("is_safe = ?")
                params.append(1 if is_safe else 0)
            if updates:
                updates.append("last_visited = ?")
                params.append(now)
                params.extend([map_name, bot_id])
                conn.execute(
                    f"UPDATE map_metadata SET {', '.join(updates)} "
                    "WHERE map_name = ? AND bot_id = ?",
                    params,
                )
        else:
            conn.execute(
                """INSERT INTO map_metadata
                   (map_name, bot_id, suggested_level_min, suggested_level_max,
                    optimal_party_size, is_safe, last_visited)
                   VALUES (?, ?, ?, ?, 1, ?, ?)""",
                (
                    map_name,
                    bot_id,
                    level_min or 1,
                    level_max or 99,
                    1 if (is_safe is None or is_safe) else 0,
                    now,
                ),
            )
        conn.commit()

    def get_map_metadata(
        self, map_name: str, bot_id: str = "default",
    ) -> dict[str, Any]:
        """Get metadata for a map."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM map_metadata WHERE map_name = ? AND bot_id = ?",
            (map_name, bot_id),
        ).fetchone()
        if row is None:
            return {
                "map_name": map_name,
                "bot_id": bot_id,
                "suggested_level_min": 1,
                "suggested_level_max": 99,
                "optimal_party_size": 1,
                "is_safe": True,
                "last_visited": 0.0,
            }
        return dict(row)

    def __repr__(self) -> str:
        count = self.get_map_count()
        return f"<ExperienceTracker: {count} maps, db={self._db_path}>"
