"""
Shared Learning Database — Redis-backed shared knowledge across all bot instances.

One bot learns → all bots know. Death records, successful strategies, MVP kill data,
and market prices are shared across all instances.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


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
            conn = sqlite3.connect(self._db_path)
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
                    CREATE INDEX IF NOT EXISTS idx_deaths_monster ON deaths(monster_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_mvp_kills_monster ON mvp_kills(monster_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_prices_item ON prices(item_name)
                """)
                conn.commit()
            finally:
                conn.close()

    # ── Death Records ──

    def record_death(self, record: SharedDeathRecord) -> None:
        """Record a death shared across all instances."""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
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
            conn = sqlite3.connect(self._db_path)
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
            conn = sqlite3.connect(self._db_path)
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
            conn = sqlite3.connect(self._db_path)
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
            conn = sqlite3.connect(self._db_path)
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
            conn = sqlite3.connect(self._db_path)
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
            conn = sqlite3.connect(self._db_path)
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
            conn = sqlite3.connect(self._db_path)
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
            conn = sqlite3.connect(self._db_path)
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

    # ── Strategies ──

    def record_strategy(self, monster_id: int, strategy: str, success: bool, bot_id: str = "") -> None:
        """Record a strategy outcome shared across all instances."""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
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
            conn = sqlite3.connect(self._db_path)
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
            conn = sqlite3.connect(self._db_path)
            try:
                death_count = conn.execute("SELECT COUNT(*) FROM deaths").fetchone()[0]
                mvp_count = conn.execute("SELECT COUNT(*) FROM mvp_kills").fetchone()[0]
                price_count = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
                strat_count = conn.execute("SELECT COUNT(*) FROM strategies").fetchone()[0]

                lines = [f"── Shared Learning Database ──"]
                lines.append(f"Path: {self._db_path}")
                lines.append(f"Total deaths: {death_count}")
                lines.append(f"Total MVP kills: {mvp_count}")
                lines.append(f"Total price records: {price_count}")
                lines.append(f"Total strategies: {strat_count}")

                dangerous = self.get_most_dangerous_monsters(3)
                if dangerous:
                    lines.append("Most dangerous: " + ", ".join(f'{d["monster_name"]}({d["deaths"]})' for d in dangerous))
                return "\n".join(lines)
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
