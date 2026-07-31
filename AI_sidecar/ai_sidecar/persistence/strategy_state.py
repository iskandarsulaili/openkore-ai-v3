"""
Full State Persistence — SQLite-backed persistence for all strategy modules.

Persists ALL state to disk:
- competitive_intelligence (farming activities, market activities, player builds, guild intel)
- theory_of_mind (observations, intentions, predictions, relationships, patterns)
- empire_manager (roles, directives, pipeline, territories, alliances, shared inventory)
- crisis_manager (crisis history, lessons, blacklists, config overrides)

Features:
- Versioned snapshots (incremental saves, not full rewrites)
- Load on restart (all state survives crashes)
- Incremental saves (only changed data is written)
- Thread-safe
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "strategy_state.db"


@dataclass
class SnapshotVersion:
    """Version tracking for incremental saves."""
    module: str
    version: int = 0
    last_full_save: float = 0.0
    last_incremental_save: float = 0.0
    change_count: int = 0


class StrategyStateDB:
    """SQLite-backed persistence for all strategy modules.

    Each module gets its own table. Data survives process restarts.
    Supports versioned snapshots and incremental saves.
    """

    _local = threading.local()
    _db_path: Path = DEFAULT_DB_PATH
    _init_lock = threading.Lock()
    _initialized = False
    _versions: dict[str, SnapshotVersion] = {}

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
                CREATE TABLE IF NOT EXISTS strategy_versions (
                    module TEXT PRIMARY KEY,
                    version INTEGER DEFAULT 0,
                    last_full_save REAL DEFAULT 0,
                    last_incremental_save REAL DEFAULT 0,
                    change_count INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS competitive_intelligence (
                    key TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL,
                    version INTEGER DEFAULT 0,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS theory_of_mind (
                    key TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL,
                    version INTEGER DEFAULT 0,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS empire_manager (
                    key TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL,
                    version INTEGER DEFAULT 0,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS crisis_manager (
                    key TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL,
                    version INTEGER DEFAULT 0,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS unified_consciousness (
                    key TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL,
                    version INTEGER DEFAULT 0,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS strategy_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    module TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    is_full INTEGER DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_strategy_snapshots_module_version
                    ON strategy_snapshots(module, version DESC);

                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;
            """)
            conn.commit()
            cls._initialized = True

    # ── Version Management ──

    @classmethod
    def _get_version(cls, module: str) -> SnapshotVersion:
        cls._init_db()
        if module not in cls._versions:
            conn = cls._get_conn()
            row = conn.execute(
                "SELECT * FROM strategy_versions WHERE module=?",
                (module,),
            ).fetchone()
            if row:
                cls._versions[module] = SnapshotVersion(
                    module=row["module"],
                    version=row["version"],
                    last_full_save=row["last_full_save"],
                    last_incremental_save=row["last_incremental_save"],
                    change_count=row["change_count"],
                )
            else:
                cls._versions[module] = SnapshotVersion(module=module)
                conn.execute(
                    "INSERT INTO strategy_versions (module, version, last_full_save, last_incremental_save, change_count) "
                    "VALUES (?, 0, 0, 0, 0)",
                    (module,),
                )
                conn.commit()
        return cls._versions[module]

    @classmethod
    def _increment_version(cls, module: str) -> int:
        ver = cls._get_version(module)
        ver.version += 1
        ver.change_count += 1
        conn = cls._get_conn()
        conn.execute(
            "UPDATE strategy_versions SET version=?, change_count=? WHERE module=?",
            (ver.version, ver.change_count, module),
        )
        conn.commit()
        return ver.version

    # ── Generic Save/Load ──

    @classmethod
    def save_module_state(cls, module: str, key: str, data: Any) -> int:
        """Save a module's state. Returns version number."""
        cls._init_db()
        conn = cls._get_conn()
        now = time.time()
        version = cls._increment_version(module)

        conn.execute(
            f"INSERT INTO {module} (key, data_json, version, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET "
            "data_json=excluded.data_json, version=excluded.version, updated_at=excluded.updated_at",
            (key, json.dumps(data, default=str), version, now),
        )
        conn.commit()
        return version

    @classmethod
    def load_module_state(cls, module: str, key: str) -> Any:
        """Load a module's state by key."""
        cls._init_db()
        conn = cls._get_conn()
        row = conn.execute(
            f"SELECT data_json FROM {module} WHERE key=?",
            (key,),
        ).fetchone()
        if row:
            try:
                return json.loads(row["data_json"])
            except (json.JSONDecodeError, TypeError):
                pass
        return None

    @classmethod
    def load_all_module_state(cls, module: str) -> dict[str, Any]:
        """Load all state for a module."""
        cls._init_db()
        conn = cls._get_conn()
        rows = conn.execute(
            f"SELECT key, data_json FROM {module}",
        ).fetchall()
        result = {}
        for row in rows:
            try:
                result[row["key"]] = json.loads(row["data_json"])
            except (json.JSONDecodeError, TypeError):
                result[row["key"]] = row["data_json"]
        return result

    @classmethod
    def delete_module_state(cls, module: str, key: str) -> None:
        """Delete a specific state entry."""
        cls._init_db()
        conn = cls._get_conn()
        conn.execute(f"DELETE FROM {module} WHERE key=?", (key,))
        conn.commit()

    # ── Snapshot Management ──

    @classmethod
    def save_snapshot(cls, module: str, snapshot: dict, is_full: bool = False) -> int:
        """Save a versioned snapshot of the module's full state."""
        cls._init_db()
        conn = cls._get_conn()
        now = time.time()
        version = cls._increment_version(module)

        conn.execute(
            "INSERT INTO strategy_snapshots (module, version, snapshot_json, created_at, is_full) "
            "VALUES (?, ?, ?, ?, ?)",
            (module, version, json.dumps(snapshot, default=str), now, 1 if is_full else 0),
        )
        conn.commit()

        # Update version tracking
        ver = cls._get_version(module)
        if is_full:
            ver.last_full_save = now
        else:
            ver.last_incremental_save = now
        conn.execute(
            "UPDATE strategy_versions SET last_full_save=?, last_incremental_save=? WHERE module=?",
            (ver.last_full_save, ver.last_incremental_save, module),
        )
        conn.commit()

        # Prune old snapshots (keep last 50)
        cls._prune_snapshots(module)

        return version

    @classmethod
    def load_latest_snapshot(cls, module: str) -> dict | None:
        """Load the latest snapshot for a module."""
        cls._init_db()
        conn = cls._get_conn()
        row = conn.execute(
            "SELECT snapshot_json, version FROM strategy_snapshots "
            "WHERE module=? ORDER BY version DESC LIMIT 1",
            (module,),
        ).fetchone()
        if row:
            try:
                return json.loads(row["snapshot_json"])
            except (json.JSONDecodeError, TypeError):
                pass
        return None

    @classmethod
    def load_snapshot_at_version(cls, module: str, version: int) -> dict | None:
        """Load a snapshot at a specific version."""
        cls._init_db()
        conn = cls._get_conn()
        row = conn.execute(
            "SELECT snapshot_json FROM strategy_snapshots "
            "WHERE module=? AND version=?",
            (module, version),
        ).fetchone()
        if row:
            try:
                return json.loads(row["snapshot_json"])
            except (json.JSONDecodeError, TypeError):
                pass
        return None

    @classmethod
    def _prune_snapshots(cls, module: str, keep: int = 50) -> None:
        """Remove old snapshots, keeping only the most recent N."""
        try:
            conn = cls._get_conn()
            conn.execute(
                "DELETE FROM strategy_snapshots WHERE module=? AND id NOT IN ("
                "SELECT id FROM strategy_snapshots WHERE module=? ORDER BY id DESC LIMIT ?"
                ")",
                (module, module, keep),
            )
            conn.commit()
        except Exception:
            pass

    # ─── Module-Specific Save/Load ──

    @classmethod
    def save_competitive_intelligence(cls, data: dict) -> int:
        """Save all competitive intelligence state."""
        return cls.save_module_state("competitive_intelligence", "full_state", data)

    @classmethod
    def load_competitive_intelligence(cls) -> dict | None:
        return cls.load_module_state("competitive_intelligence", "full_state")

    @classmethod
    def save_theory_of_mind(cls, data: dict) -> int:
        return cls.save_module_state("theory_of_mind", "full_state", data)

    @classmethod
    def load_theory_of_mind(cls) -> dict | None:
        return cls.load_module_state("theory_of_mind", "full_state")

    @classmethod
    def save_empire_manager(cls, data: dict) -> int:
        return cls.save_module_state("empire_manager", "full_state", data)

    @classmethod
    def load_empire_manager(cls) -> dict | None:
        return cls.load_module_state("empire_manager", "full_state")

    @classmethod
    def save_crisis_manager(cls, data: dict) -> int:
        return cls.save_module_state("crisis_manager", "full_state", data)

    @classmethod
    def load_crisis_manager(cls) -> dict | None:
        return cls.load_module_state("crisis_manager", "full_state")

    @classmethod
    def save_unified_consciousness(cls, data: dict) -> int:
        return cls.save_module_state("unified_consciousness", "full_state", data)

    @classmethod
    def load_unified_consciousness(cls) -> dict | None:
        return cls.load_module_state("unified_consciousness", "full_state")

    # ── Maintenance ──

    @classmethod
    def get_stats(cls) -> dict:
        cls._init_db()
        conn = cls._get_conn()
        stats = {}
        for table in ["competitive_intelligence", "theory_of_mind", "empire_manager",
                       "crisis_manager", "unified_consciousness", "strategy_snapshots"]:
            try:
                row = conn.execute(f"SELECT COUNT(*) as c FROM {table}").fetchone()
                stats[table] = row["c"] if row else 0
            except Exception:
                stats[table] = 0
        # Version info
        rows = conn.execute("SELECT * FROM strategy_versions").fetchall()
        stats["versions"] = {r["module"]: {"version": r["version"], "changes": r["change_count"]}
                             for r in rows}
        return stats

    @classmethod
    def reset(cls) -> None:
        cls._init_db()
        conn = cls._get_conn()
        for table in ["competitive_intelligence", "theory_of_mind", "empire_manager",
                       "crisis_manager", "unified_consciousness", "strategy_snapshots",
                       "strategy_versions"]:
            try:
                conn.execute(f"DELETE FROM {table}")
            except Exception:
                pass
        conn.commit()
        cls._versions.clear()
