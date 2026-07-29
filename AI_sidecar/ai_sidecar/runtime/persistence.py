"""Persistent State — SQLite-backed learning data that survives crashes.

Each learning module gets its own table. Data persists across
OpenKore restarts so accumulated knowledge isn't lost.

Tables:
- death_records (map, monster, cause, count, last_seen)
- market_prices (item, buy_price, sell_price, volume, last_trade)
- server_profile (exp_mult, drop_mult, server_type, confidence)
- bot_state (bot_id, zeny, level, job, map, last_seen)
- domain_state (domain, key, value_json, updated_at)
"""
from __future__ import annotations
import json
import sqlite3
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "bot_memory.db"


class PersistentState:
    """SQLite-backed persistent memory for learning modules.
    
    Thread-safe (connection per thread). Creates tables on init.
    Data survives process restarts — only clear if file is deleted.
    """
    
    _local = threading.local()
    _db_path: Path = _DEFAULT_DB_PATH
    _init_lock = threading.Lock()
    _initialized = False
    
    @classmethod
    def configure(cls, db_path: str | Path) -> None:
        cls._db_path = Path(db_path)
        cls._initialized = False
    
    @classmethod
    def _get_conn(cls) -> sqlite3.Connection:
        if not hasattr(cls._local, "conn") or cls._local.conn is None:
            cls._local.conn = sqlite3.connect(str(cls._db_path))
            cls._local.conn.row_factory = sqlite3.Row
        return cls._local.conn
    
    @classmethod
    def _init_db(cls) -> None:
        if cls._initialized:
            return
        with cls._init_lock:
            if cls._initialized:
                return
            conn = cls._get_conn()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS death_records (
                    map TEXT, monster TEXT, cause TEXT,
                    count INTEGER DEFAULT 1,
                    last_seen TEXT,
                    countermeasure TEXT,
                    PRIMARY KEY (map, monster)
                );
                CREATE TABLE IF NOT EXISTS market_prices (
                    item TEXT PRIMARY KEY,
                    buy_price INTEGER DEFAULT 0,
                    sell_price INTEGER DEFAULT 0,
                    volume INTEGER DEFAULT 0,
                    last_trade TEXT
                );
                CREATE TABLE IF NOT EXISTS server_profile (
                    key TEXT PRIMARY KEY,
                    value_json TEXT,
                    updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS bot_state (
                    bot_id TEXT,
                    key TEXT,
                    value_json TEXT,
                    updated_at TEXT,
                    PRIMARY KEY (bot_id, key)
                );
                CREATE TABLE IF NOT EXISTS domain_state (
                    domain TEXT,
                    key TEXT,
                    value_json TEXT,
                    updated_at TEXT,
                    PRIMARY KEY (domain, key)
                );
            """)
            conn.commit()
            cls._initialized = True
    
    # ── Death Records ──
    
    @classmethod
    def record_death(cls, map_name: str, monster: str, cause: str, countermeasure: str = "") -> None:
        cls._init_db()
        conn = cls._get_conn()
        now = datetime.now().isoformat()
        conn.execute("""
            INSERT INTO death_records (map, monster, cause, count, last_seen, countermeasure)
            VALUES (?, ?, ?, 1, ?, ?)
            ON CONFLICT(map, monster) DO UPDATE SET
                count = count + 1,
                last_seen = excluded.last_seen,
                cause = excluded.cause,
                countermeasure = CASE WHEN excluded.countermeasure != '' THEN excluded.countermeasure ELSE countermeasure END
        """, (map_name, monster, cause, now, countermeasure))
        conn.commit()
    
    @classmethod
    def get_death_count(cls, map_name: str, monster: str = "") -> int:
        cls._init_db()
        conn = cls._get_conn()
        if monster:
            row = conn.execute("SELECT count FROM death_records WHERE map=? AND monster=?", (map_name, monster)).fetchone()
        else:
            row = conn.execute("SELECT SUM(count) as count FROM death_records WHERE map=?", (map_name,)).fetchone()
        return row["count"] if row and row["count"] else 0
    
    @classmethod
    def get_deadliest_maps(cls, limit: int = 5) -> list[dict]:
        cls._init_db()
        conn = cls._get_conn()
        rows = conn.execute("""
            SELECT map, SUM(count) as deaths, GROUP_CONCAT(DISTINCT monster) as monsters
            FROM death_records GROUP BY map ORDER BY deaths DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]
    
    # ── Market Prices ──
    
    @classmethod
    def record_trade(cls, item: str, buy_price: int, sell_price: int) -> None:
        cls._init_db()
        conn = cls._get_conn()
        now = datetime.now().isoformat()
        conn.execute("""
            INSERT INTO market_prices (item, buy_price, sell_price, volume, last_trade)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(item) DO UPDATE SET
                buy_price = (buy_price + excluded.buy_price) / 2,
                sell_price = (sell_price + excluded.sell_price) / 2,
                volume = volume + 1,
                last_trade = excluded.last_trade
        """, (item, buy_price, sell_price, now))
        conn.commit()
    
    @classmethod
    def get_market_data(cls, item: str) -> dict | None:
        cls._init_db()
        conn = cls._get_conn()
        row = conn.execute("SELECT * FROM market_prices WHERE item=?", (item,)).fetchone()
        return dict(row) if row else None
    
    # ── Server Profile ──
    
    @classmethod
    def save_server_profile(cls, key: str, value: Any) -> None:
        cls._init_db()
        conn = cls._get_conn()
        now = datetime.now().isoformat()
        conn.execute("""
            INSERT INTO server_profile (key, value_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at
        """, (key, json.dumps(value), now))
        conn.commit()
    
    @classmethod
    def load_server_profile(cls, key: str) -> Any:
        cls._init_db()
        conn = cls._get_conn()
        row = conn.execute("SELECT value_json FROM server_profile WHERE key=?", (key,)).fetchone()
        if row:
            try:
                return json.loads(row["value_json"])
            except (json.JSONDecodeError, TypeError):
                pass
        return None
    
    # ── Generic Domain State ──
    
    @classmethod
    def save_domain_state(cls, domain: str, key: str, value: Any) -> None:
        cls._init_db()
        conn = cls._get_conn()
        now = datetime.now().isoformat()
        conn.execute("""
            INSERT INTO domain_state (domain, key, value_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(domain, key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at
        """, (domain, key, json.dumps(value), now))
        conn.commit()
    
    @classmethod
    def load_domain_state(cls, domain: str, key: str) -> Any:
        cls._init_db()
        conn = cls._get_conn()
        row = conn.execute("SELECT value_json FROM domain_state WHERE domain=? AND key=?", (domain, key)).fetchone()
        if row:
            try:
                return json.loads(row["value_json"])
            except (json.JSONDecodeError, TypeError):
                pass
        return None
    
    @classmethod
    def load_all_domain_state(cls, domain: str) -> dict[str, Any]:
        cls._init_db()
        conn = cls._get_conn()
        rows = conn.execute("SELECT key, value_json FROM domain_state WHERE domain=?", (domain,)).fetchall()
        result = {}
        for row in rows:
            try:
                result[row["key"]] = json.loads(row["value_json"])
            except (json.JSONDecodeError, TypeError):
                result[row["key"]] = row["value_json"]
        return result
    
    # ── Bot State ──
    
    @classmethod
    def save_bot_state(cls, bot_id: str, key: str, value: Any) -> None:
        cls._init_db()
        conn = cls._get_conn()
        now = datetime.now().isoformat()
        conn.execute("""
            INSERT INTO bot_state (bot_id, key, value_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(bot_id, key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at
        """, (bot_id, key, json.dumps(value), now))
        conn.commit()
    
    @classmethod
    def load_bot_state(cls, bot_id: str, key: str) -> Any:
        cls._init_db()
        conn = cls._get_conn()
        row = conn.execute("SELECT value_json FROM bot_state WHERE bot_id=? AND key=?", (bot_id, key)).fetchone()
        if row:
            try:
                return json.loads(row["value_json"])
            except (json.JSONDecodeError, TypeError):
                pass
        return None
    
    @classmethod
    def get_all_bot_ids(cls) -> list[str]:
        cls._init_db()
        conn = cls._get_conn()
        rows = conn.execute("SELECT DISTINCT bot_id FROM bot_state").fetchall()
        return [r["bot_id"] for r in rows]
    
    # ── Maintenance ──
    
    @classmethod
    def get_stats(cls) -> dict:
        cls._init_db()
        conn = cls._get_conn()
        stats = {}
        for table in ["death_records", "market_prices", "server_profile", "bot_state", "domain_state"]:
            row = conn.execute(f"SELECT COUNT(*) as c FROM {table}").fetchone()
            stats[table] = row["c"] if row else 0
        return stats
    
    @classmethod
    def reset(cls) -> None:
        """Clear all data (for testing)."""
        cls._init_db()
        conn = cls._get_conn()
        for table in ["death_records", "market_prices", "server_profile", "bot_state", "domain_state"]:
            conn.execute(f"DELETE FROM {table}")
        conn.commit()
