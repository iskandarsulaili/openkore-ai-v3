"""
Shared Learning Database — Redis-backed shared knowledge across all bot instances.

One bot learns → all bots know. Death records, successful strategies, MVP kill data,
and market prices are shared across all instances.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

from ai_sidecar.persistence.sqlite_utils import connect as _hardened_connect


@dataclass
class SharedDeathRecord:
    """A death record shared across all bot instances."""
    monster_id: int
    monster_name: str
    map_name: str
    cause: str
    timestamp: float
    bot_id: str = ""
    count: int = 1


@dataclass
class SharedMVPKill:
    """An MVP kill shared across all bot instances."""
    monster_id: int
    monster_name: str
    map_name: str
    kill_time: float
    bot_id: str = ""
    strategy_used: str = ""
    successful: bool = True


@dataclass
class SharedPrice:
    """A price observation shared across all bot instances."""
    item_name: str
    item_id: int = 0
    buy_price: int = 0
    sell_price: int = 0
    map_name: str = ""
    timestamp: float = 0.0
    bot_id: str = ""


class SharedLearningDB:
    """SQLite-backed shared learning database for all bot instances."""

    def __init__(self, db_path: str = "") -> None:
        self._lock = RLock()
        if not db_path:
            db_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "shared_learning.db")
        self._db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._lock:
            conn = _hardened_connect(self._db_path)
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS deaths (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        monster_id INTEGER NOT NULL,
                        monster_name TEXT NOT NULL,
                        map_name TEXT NOT NULL,
                        cause TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        bot_id TEXT NOT NULL DEFAULT '',
                        UNIQUE(monster_id, map_name, cause, bot_id)
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS mvp_kills (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        monster_id INTEGER NOT NULL,
                        monster_name TEXT NOT NULL,
                        map_name TEXT NOT NULL,
                        kill_time REAL NOT NULL,
                        bot_id TEXT NOT NULL DEFAULT '',
                        strategy_used TEXT NOT NULL DEFAULT '',
                        successful INTEGER NOT NULL DEFAULT 1
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS prices (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        item_name TEXT NOT NULL,
                        item_id INTEGER NOT NULL DEFAULT 0,
                        buy_price INTEGER NOT NULL DEFAULT 0,
                        sell_price INTEGER NOT NULL DEFAULT 0,
                        map_name TEXT NOT NULL DEFAULT '',
                        timestamp REAL NOT NULL,
                        bot_id TEXT NOT NULL DEFAULT ''
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS strategies (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        monster_id INTEGER NOT NULL,
                        strategy TEXT NOT NULL,
                        success_count INTEGER NOT NULL DEFAULT 1,
                        fail_count INTEGER NOT NULL DEFAULT 0,
                        last_used REAL NOT NULL,
                        bot_id TEXT NOT NULL DEFAULT '',
                        UNIQUE(monster_id, strategy)
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS failures (
                        id TEXT PRIMARY KEY,
                        server_id TEXT NOT NULL DEFAULT 'default',
                        bot_id TEXT NOT NULL DEFAULT '',
                        category TEXT NOT NULL,
                        subcategory TEXT,
                        timestamp REAL NOT NULL,
                        context TEXT NOT NULL DEFAULT '{}',
                        reasoning TEXT NOT NULL DEFAULT '',
                        lesson_learned TEXT NOT NULL DEFAULT '',
                        action_taken TEXT NOT NULL DEFAULT '',
                        action_effective INTEGER,
                        resolved INTEGER NOT NULL DEFAULT 0,
                        resolved_at REAL,
                        recurrence_count INTEGER NOT NULL DEFAULT 1,
                        recurrence_key TEXT NOT NULL DEFAULT '',
                        applied_to_config TEXT NOT NULL DEFAULT '[]',
                        peer_shared INTEGER NOT NULL DEFAULT 0
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_deaths_monster ON deaths(monster_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_mvp_kills_monster ON mvp_kills(monster_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_prices_item ON prices(item_name)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_failures_recurrence
                    ON failures(server_id, category, recurrence_key)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_failures_server
                    ON failures(server_id)
                """)
                conn.commit()
            finally:
                conn.close()

    # ── Death Records ──

    def record_death(self, record: SharedDeathRecord) -> None:
        """Record a death shared across all instances."""
        with self._lock:
            conn = _hardened_connect(self._db_path)
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO deaths (monster_id, monster_name, map_name, cause, timestamp, bot_id) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (record.monster_id, record.monster_name, record.map_name, record.cause, record.timestamp, record.bot_id)
                )
                conn.commit()
            finally:
                conn.close()

    def is_monster_dangerous(self, monster_id: int) -> bool:
        """Check if any instance has died to this monster."""
        with self._lock:
            conn = _hardened_connect(self._db_path)
            try:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM deaths WHERE monster_id = ?", (monster_id,)
                )
                count = cursor.fetchone()[0]
                return count > 0
            finally:
                conn.close()

    def get_death_count(self, monster_id: int) -> int:
        """Get total death count for a monster across all instances."""
        with self._lock:
            conn = _hardened_connect(self._db_path)
            try:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM deaths WHERE monster_id = ?", (monster_id,)
                )
                return cursor.fetchone()[0]
            finally:
                conn.close()

    def get_most_dangerous_monsters(self, limit: int = 10) -> list[dict]:
        """Get the monsters that have killed us the most across all instances."""
        with self._lock:
            conn = _hardened_connect(self._db_path)
            try:
                cursor = conn.execute(
                    "SELECT monster_id, monster_name, COUNT(*) as deaths FROM deaths "
                    "GROUP BY monster_id ORDER BY deaths DESC LIMIT ?", (limit,)
                )
                return [{"monster_id": r[0], "monster_name": r[1], "deaths": r[2]} for r in cursor.fetchall()]
            finally:
                conn.close()

    # ── MVP Kills ──

    def record_mvp_kill(self, record: SharedMVPKill) -> None:
        """Record an MVP kill shared across all instances."""
        with self._lock:
            conn = _hardened_connect(self._db_path)
            try:
                conn.execute(
                    "INSERT INTO mvp_kills (monster_id, monster_name, map_name, kill_time, bot_id, strategy_used, successful) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (record.monster_id, record.monster_name, record.map_name, record.kill_time,
                     record.bot_id, record.strategy_used, 1 if record.successful else 0)
                )
                conn.commit()
            finally:
                conn.close()

    def get_mvp_kill_count(self, monster_id: int) -> int:
        """Get total MVP kill count for a monster across all instances."""
        with self._lock:
            conn = _hardened_connect(self._db_path)
            try:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM mvp_kills WHERE monster_id = ? AND successful = 1", (monster_id,)
                )
                return cursor.fetchone()[0]
            finally:
                conn.close()

    def get_mvp_spawn_time(self, monster_id: int) -> float | None:
        """Get the last known spawn time for an MVP."""
        with self._lock:
            conn = _hardened_connect(self._db_path)
            try:
                cursor = conn.execute(
                    "SELECT kill_time FROM mvp_kills WHERE monster_id = ? ORDER BY kill_time DESC LIMIT 1",
                    (monster_id,)
                )
                row = cursor.fetchone()
                return row[0] if row else None
            finally:
                conn.close()

    # ── Prices ──

    def record_price(self, record: SharedPrice) -> None:
        """Record a price observation shared across all instances."""
        with self._lock:
            conn = _hardened_connect(self._db_path)
            try:
                conn.execute(
                    "INSERT INTO prices (item_name, item_id, buy_price, sell_price, map_name, timestamp, bot_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (record.item_name, record.item_id, record.buy_price, record.sell_price,
                     record.map_name, record.timestamp, record.bot_id)
                )
                conn.commit()
            finally:
                conn.close()

    def get_average_price(self, item_name: str, hours: int = 24) -> dict:
        """Get average buy/sell price for an item over the last N hours."""
        with self._lock:
            conn = _hardened_connect(self._db_path)
            try:
                cutoff = time.time() - (hours * 3600)
                cursor = conn.execute(
                    "SELECT AVG(buy_price), AVG(sell_price), COUNT(*) FROM prices "
                    "WHERE item_name = ? AND timestamp > ?",
                    (item_name, cutoff)
                )
                row = cursor.fetchone()
                return {
                    "avg_buy": int(row[0] or 0),
                    "avg_sell": int(row[1] or 0),
                    "samples": row[2] or 0,
                }
            finally:
                conn.close()

    def _query_arbitrage_candidates(self, limit: int = 5) -> list[tuple[str, int, int, int]]:
        """Return items whose observed buy_price < sell_price (potential arbitrage).

        COMPLETED per completeness mandate: the innovation engine's arbitrage detector
        needed real price data to detect buy-low-sell-high windows. This queries the
        `prices` table for items where the average sell price EXCEEDS the average buy
        price (a genuine mismatch observed from real shop interactions), returning
        (item_name, avg_buy, avg_sell, samples). Empty when no such mismatch is observed
        yet — never fabricated.
        """
        try:
            with self._lock:
                conn = _hardened_connect(self._db_path)
                try:
                    cursor = conn.execute(
                        "SELECT item_name, AVG(buy_price), AVG(sell_price), COUNT(*) "
                        "FROM prices GROUP BY item_name "
                        "HAVING AVG(sell_price) > AVG(buy_price) "
                        "ORDER BY (AVG(sell_price) - AVG(buy_price)) DESC LIMIT ?",
                        (limit,)
                    )
                    return [
                        (str(r[0]), int(r[1] or 0), int(r[2] or 0), int(r[3] or 0))
                        for r in cursor.fetchall()
                    ]
                finally:
                    conn.close()
        except Exception:
            return []

    # ── Strategies ──

    def record_strategy(self, monster_id: int, strategy: str, success: bool, bot_id: str = "") -> None:
        """Record a strategy outcome shared across all instances."""
        with self._lock:
            conn = _hardened_connect(self._db_path)
            try:
                if success:
                    conn.execute(
                        "INSERT OR REPLACE INTO strategies (monster_id, strategy, success_count, fail_count, last_used, bot_id) "
                        "VALUES (?, ?, COALESCE((SELECT success_count FROM strategies WHERE monster_id=? AND strategy=?), 0) + 1, "
                        "COALESCE((SELECT fail_count FROM strategies WHERE monster_id=? AND strategy=?), 0), ?, ?)",
                        (monster_id, strategy, monster_id, strategy, monster_id, strategy, time.time(), bot_id)
                    )
                else:
                    conn.execute(
                        "INSERT OR REPLACE INTO strategies (monster_id, strategy, success_count, fail_count, last_used, bot_id) "
                        "VALUES (?, ?, COALESCE((SELECT success_count FROM strategies WHERE monster_id=? AND strategy=?), 0), "
                        "COALESCE((SELECT fail_count FROM strategies WHERE monster_id=? AND strategy=?), 0) + 1, ?, ?)",
                        (monster_id, strategy, monster_id, strategy, monster_id, strategy, time.time(), bot_id)
                    )
                conn.commit()
            finally:
                conn.close()

    def get_best_strategy(self, monster_id: int) -> str | None:
        """Get the best strategy for a monster based on success rate."""
        with self._lock:
            conn = _hardened_connect(self._db_path)
            try:
                cursor = conn.execute(
                    "SELECT strategy, success_count, fail_count FROM strategies WHERE monster_id = ? "
                    "ORDER BY (CAST(success_count AS REAL) / MAX(success_count + fail_count, 1)) DESC LIMIT 1",
                    (monster_id,)
                )
                row = cursor.fetchone()
                return row[0] if row else None
            finally:
                conn.close()

    def get_shared_summary(self) -> str:
        with self._lock:
            conn = _hardened_connect(self._db_path)
            try:
                death_count = conn.execute("SELECT COUNT(*) FROM deaths").fetchone()[0]
                mvp_count = conn.execute("SELECT COUNT(*) FROM mvp_kills").fetchone()[0]
                price_count = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
                strat_count = conn.execute("SELECT COUNT(*) FROM strategies").fetchone()[0]
                fail_count = conn.execute("SELECT COUNT(*) FROM failures").fetchone()[0]

                lines = [f"── Shared Learning Database ──"]
                lines.append(f"Path: {self._db_path}")
                lines.append(f"Total deaths: {death_count}")
                lines.append(f"Total MVP kills: {mvp_count}")
                lines.append(f"Total price records: {price_count}")
                lines.append(f"Total strategies: {strat_count}")
                lines.append(f"Total failures: {fail_count}")

                dangerous = self.get_most_dangerous_monsters(3)
                if dangerous:
                    lines.append("Most dangerous: " + ", ".join(f'{d["monster_name"]}({d["deaths"]})' for d in dangerous))
                return "\n".join(lines)
            finally:
                conn.close()

    # ── Failures ──

    def _ensure_failures_table(self) -> None:
        """Create the failures table if it doesn't exist (idempotent)."""
        with self._lock:
            conn = _hardened_connect(self._db_path)
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS failures (
                        id TEXT PRIMARY KEY,
                        server_id TEXT NOT NULL DEFAULT 'default',
                        bot_id TEXT NOT NULL DEFAULT '',
                        category TEXT NOT NULL,
                        subcategory TEXT,
                        timestamp REAL NOT NULL,
                        context TEXT NOT NULL DEFAULT '{}',
                        reasoning TEXT NOT NULL DEFAULT '',
                        lesson_learned TEXT NOT NULL DEFAULT '',
                        action_taken TEXT NOT NULL DEFAULT '',
                        action_effective INTEGER,
                        resolved INTEGER NOT NULL DEFAULT 0,
                        resolved_at REAL,
                        recurrence_count INTEGER NOT NULL DEFAULT 1,
                        recurrence_key TEXT NOT NULL DEFAULT '',
                        applied_to_config TEXT NOT NULL DEFAULT '[]',
                        peer_shared INTEGER NOT NULL DEFAULT 0
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_failures_recurrence
                    ON failures(server_id, category, recurrence_key)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_failures_server
                    ON failures(server_id)
                """)
                conn.commit()
            finally:
                conn.close()

    def record_failure(self, record_dict: dict) -> None:
        """Record a failure in the shared database."""
        with self._lock:
            self._ensure_failures_table()
            conn = _hardened_connect(self._db_path)
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO failures
                       (id, server_id, bot_id, category, subcategory, timestamp,
                        context, reasoning, lesson_learned, action_taken,
                        action_effective, resolved, resolved_at,
                        recurrence_count, recurrence_key, applied_to_config, peer_shared)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        record_dict.get("id", ""),
                        record_dict.get("server_id", "default"),
                        record_dict.get("bot_id", ""),
                        record_dict.get("category", "unknown"),
                        record_dict.get("subcategory"),
                        record_dict.get("timestamp", time.time()),
                        record_dict.get("context", "{}"),
                        record_dict.get("reasoning", ""),
                        record_dict.get("lesson_learned", ""),
                        record_dict.get("action_taken", ""),
                        record_dict.get("action_effective"),
                        record_dict.get("resolved", 0),
                        record_dict.get("resolved_at"),
                        record_dict.get("recurrence_count", 1),
                        record_dict.get("recurrence_key", ""),
                        record_dict.get("applied_to_config", "[]"),
                        record_dict.get("peer_shared", 0),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def get_failures(
        self,
        server_id: str | None = None,
        category: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Get failures with optional filters."""
        with self._lock:
            self._ensure_failures_table()
            conn = _hardened_connect(self._db_path)
            try:
                query = "SELECT * FROM failures WHERE 1=1"
                params: list = []
                if server_id:
                    query += " AND server_id = ?"
                    params.append(server_id)
                if category:
                    query += " AND category = ?"
                    params.append(category)
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                columns = [d[0] for d in cursor.description]
                return [dict(zip(columns, row)) for row in rows]
            finally:
                conn.close()

    def get_recurring_failures(
        self,
        server_id: str | None = None,
        min_count: int = 3,
        limit: int = 20,
    ) -> list[dict]:
        """Find failures with recurrence_count >= min_count."""
        with self._lock:
            self._ensure_failures_table()
            conn = _hardened_connect(self._db_path)
            try:
                query = "SELECT * FROM failures WHERE recurrence_count >= ?"
                params: list = [min_count]
                if server_id:
                    query += " AND server_id = ?"
                    params.append(server_id)
                query += " ORDER BY recurrence_count DESC, timestamp DESC LIMIT ?"
                params.append(limit)
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                columns = [d[0] for d in cursor.description]
                return [dict(zip(columns, row)) for row in rows]
            finally:
                conn.close()

    def get_failure_summary(self, server_id: str | None = None) -> str:
        """Return a formatted failure summary for LLM context injection."""
        with self._lock:
            self._ensure_failures_table()
            conn = _hardened_connect(self._db_path)
            try:
                if server_id:
                    total = conn.execute(
                        "SELECT COUNT(*) FROM failures WHERE server_id = ?",
                        (server_id,),
                    ).fetchone()[0]
                else:
                    total = conn.execute(
                        "SELECT COUNT(*) FROM failures"
                    ).fetchone()[0]

                if total == 0:
                    return "No failures recorded."

                # By category
                if server_id:
                    cat_cursor = conn.execute(
                        "SELECT category, COUNT(*) as cnt FROM failures "
                        "WHERE server_id = ? GROUP BY category ORDER BY cnt DESC",
                        (server_id,),
                    )
                else:
                    cat_cursor = conn.execute(
                        "SELECT category, COUNT(*) as cnt FROM failures "
                        "GROUP BY category ORDER BY cnt DESC"
                    )
                cat_counts = {row[0]: row[1] for row in cat_cursor.fetchall()}

                # By server
                server_cursor = conn.execute(
                    "SELECT server_id, COUNT(*) as cnt FROM failures "
                    "GROUP BY server_id ORDER BY cnt DESC"
                )
                server_counts = {row[0]: row[1] for row in server_cursor.fetchall()}

                # Top recurring
                if server_id:
                    rec_cursor = conn.execute(
                        "SELECT category, subcategory, recurrence_count, lesson_learned "
                        "FROM failures WHERE server_id = ? AND recurrence_count >= 3 "
                        "ORDER BY recurrence_count DESC LIMIT 5",
                        (server_id,),
                    )
                else:
                    rec_cursor = conn.execute(
                        "SELECT category, subcategory, recurrence_count, lesson_learned "
                        "FROM failures WHERE recurrence_count >= 3 "
                        "ORDER BY recurrence_count DESC LIMIT 5"
                    )
                recurring = rec_cursor.fetchall()

                # Recent lessons
                if server_id:
                    lesson_cursor = conn.execute(
                        "SELECT lesson_learned FROM failures WHERE server_id = ? "
                        "AND lesson_learned != '' ORDER BY timestamp DESC LIMIT 3",
                        (server_id,),
                    )
                else:
                    lesson_cursor = conn.execute(
                        "SELECT lesson_learned FROM failures "
                        "WHERE lesson_learned != '' ORDER BY timestamp DESC LIMIT 3"
                    )
                recent_lessons = [r[0] for r in lesson_cursor.fetchall()]

                lines = [f"── Failure Summary ──"]
                lines.append(f"Total failures: {total}")
                lines.append(f"By category: {', '.join(f'{k}={v}' for k, v in cat_counts.items())}")
                lines.append(f"By server: {', '.join(f'{k}={v}' for k, v in server_counts.items())}")
                if recurring:
                    lines.append("Top recurring issues:")
                    for r in recurring:
                        lines.append(f"  {r[0]}/{r[1] or '-'} x{r[2]}: {r[3][:80]}")
                if recent_lessons:
                    lines.append("Recent lessons:")
                    for l in recent_lessons:
                        lines.append(f"  - {l[:100]}")
                return "\n".join(lines)
            finally:
                conn.close()

    def mark_failure_resolved(self, failure_id: str, effective: bool | None = None) -> None:
        """Mark a failure as resolved."""
        with self._lock:
            self._ensure_failures_table()
            conn = _hardened_connect(self._db_path)
            try:
                if effective is not None:
                    conn.execute(
                        "UPDATE failures SET resolved = 1, resolved_at = ?, action_effective = ? "
                        "WHERE id = ?",
                        (time.time(), 1 if effective else 0, failure_id),
                    )
                else:
                    conn.execute(
                        "UPDATE failures SET resolved = 1, resolved_at = ? WHERE id = ?",
                        (time.time(), failure_id),
                    )
                conn.commit()
            finally:
                conn.close()

    def increment_failure_recurrence(self, recurrence_key: str) -> int:
        """Increment recurrence count for a matching failure within 1 hour.

        Returns the new recurrence count, or 0 if no matching failure found.
        """
        with self._lock:
            self._ensure_failures_table()
            conn = _hardened_connect(self._db_path)
            try:
                cutoff = time.time() - 3600
                cursor = conn.execute(
                    "SELECT id, recurrence_count FROM failures "
                    "WHERE recurrence_key = ? AND timestamp > ? "
                    "ORDER BY timestamp DESC LIMIT 1",
                    (recurrence_key, cutoff),
                )
                row = cursor.fetchone()
                if row is None:
                    return 0
                failure_id, current_count = row
                new_count = current_count + 1
                conn.execute(
                    "UPDATE failures SET recurrence_count = ? WHERE id = ?",
                    (new_count, failure_id),
                )
                conn.commit()
                return new_count
            finally:
                conn.close()


# ── Global Singleton ──

_shared_db: SharedLearningDB | None = None
_shared_db_lock = RLock()


def get_shared_learning_db(db_path: str = "") -> SharedLearningDB:
    global _shared_db
    with _shared_db_lock:
        if _shared_db is None:
            _shared_db = SharedLearningDB(db_path=db_path)
        return _shared_db
