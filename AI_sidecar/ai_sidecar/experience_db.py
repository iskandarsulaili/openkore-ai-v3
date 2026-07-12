"""Experience database — per-bot EXP tracking with SQLite persistence.

Tracks base_level, job_level, base_exp, job_exp, zeny per bot per map.
Computes EXP rates per hour and detects leveling plateaus.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from threading import RLock
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExpSnapshot:
    """A single EXP datapoint for a bot at a moment in time."""
    bot_id: str
    base_level: int
    job_level: int
    base_exp: int
    job_exp: int
    zeny: int
    map_name: str
    timestamp: float = field(default_factory=time.time)


class ExperienceDB:
    """SQLite-backed per-bot experience tracking.

    Stores EXP snapshots and computes rates per hour.
    Detects leveling plateaus (same level for >60 minutes).
    Thread-safe.
    """

    def __init__(self, db_path: str | Path, max_snapshots_per_bot: int = 10000):
        self._lock = threading.RLock()
        self._db_path = str(db_path)
        self._max_snapshots_per_bot = max_snapshots_per_bot
        self._init_db()

    # ── Schema ──────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        try:
            db = sqlite3.connect(self._db_path, timeout=5.0)
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
            db.execute("""CREATE TABLE IF NOT EXISTS exp_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_id TEXT NOT NULL,
                base_level INTEGER NOT NULL,
                job_level INTEGER NOT NULL,
                base_exp INTEGER NOT NULL,
                job_exp INTEGER NOT NULL,
                zeny INTEGER NOT NULL DEFAULT 0,
                map_name TEXT NOT NULL DEFAULT '',
                timestamp REAL NOT NULL
            )""")
            db.execute("""CREATE INDEX IF NOT EXISTS idx_exp_bot_time
                ON exp_snapshots(bot_id, timestamp DESC)""")
            db.execute("""CREATE INDEX IF NOT EXISTS idx_exp_bot_level
                ON exp_snapshots(bot_id, base_level, job_level)""")
            db.commit()
            db.close()
        except Exception as e:
            logger.warning("ExperienceDB: DB init failed: %s", e)

    # ── Record ──────────────────────────────────────────────────────────

    def record_exp_snapshot(self, snapshot: ExpSnapshot) -> None:
        """Record an EXP snapshot for a bot."""
        with self._lock:
            try:
                db = sqlite3.connect(self._db_path, timeout=5.0)
                db.execute(
                    """INSERT INTO exp_snapshots
                       (bot_id, base_level, job_level, base_exp, job_exp, zeny, map_name, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (snapshot.bot_id, snapshot.base_level, snapshot.job_level,
                     snapshot.base_exp, snapshot.job_exp, snapshot.zeny,
                     snapshot.map_name, snapshot.timestamp),
                )
                db.commit()
                db.close()
            except Exception as e:
                logger.warning("ExperienceDB: record failed: %s", e)

            # Prune oldest if over limit
            self._prune(snapshot.bot_id)

    def record(self, entry: object) -> None:
        """Legacy compatibility — accepts ExperienceEntry from _seed_db."""
        # Seed data is non-critical; PDCA loop handles zone discovery dynamically
        pass

    def _prune(self, bot_id: str) -> None:
        """Remove oldest snapshots for a bot when over the limit."""
        try:
            db = sqlite3.connect(self._db_path, timeout=5.0)
            count = db.execute(
                "SELECT COUNT(*) FROM exp_snapshots WHERE bot_id = ?", (bot_id,)
            ).fetchone()[0]
            if count > self._max_snapshots_per_bot:
                excess = count - self._max_snapshots_per_bot
                # Delete oldest records (by id order which correlates with insert order)
                db.execute(
                    """DELETE FROM exp_snapshots WHERE id IN (
                        SELECT id FROM exp_snapshots WHERE bot_id = ?
                        ORDER BY id ASC LIMIT ?
                    )""",
                    (bot_id, excess),
                )
                db.commit()
            db.close()
        except Exception as e:
            logger.debug("ExperienceDB: prune failed: %s", e)

    # ── Query ───────────────────────────────────────────────────────────

    def get_exp_rate(
        self, bot_id: str, window_minutes: float = 5.0
    ) -> dict[str, float]:
        """Compute EXP rates per hour over the given window.

        Returns dict with base_exp_rate, job_exp_rate, zeny_rate, map_name.
        Returns zero rates if insufficient data.
        """
        with self._lock:
            try:
                db = sqlite3.connect(self._db_path, timeout=5.0)
                cutoff = time.time() - window_minutes * 60
                rows = db.execute(
                    """SELECT base_level, job_level, base_exp, job_exp, zeny, map_name, timestamp
                       FROM exp_snapshots
                       WHERE bot_id = ? AND timestamp >= ?
                       ORDER BY timestamp ASC""",
                    (bot_id, cutoff),
                ).fetchall()
                db.close()
            except Exception as e:
                logger.debug("ExperienceDB: get_exp_rate failed: %s", e)
                return {"base_exp_rate": 0.0, "job_exp_rate": 0.0, "zeny_rate": 0.0, "map_name": ""}

            if len(rows) < 2:
                return {"base_exp_rate": 0.0, "job_exp_rate": 0.0, "zeny_rate": 0.0, "map_name": ""}

            first = rows[0]
            last = rows[-1]
            elapsed_hours = max(0.001, (last[6] - first[6]) / 3600.0)

            base_exp_gain = last[2] - first[2]
            job_exp_gain = last[3] - first[3]
            zeny_gain = last[4] - first[4]

            return {
                "base_exp_rate": max(0.0, base_exp_gain / elapsed_hours),
                "job_exp_rate": max(0.0, job_exp_gain / elapsed_hours),
                "zeny_rate": max(0.0, zeny_gain / elapsed_hours),
                "map_name": last[5],
            }

    def get_leveling_speed(self, bot_id: str) -> dict[str, Any]:
        """Return leveling speed statistics for a bot.

        Returns:
            current_base_level, current_job_level,
            time_at_current_base (seconds), base_levels_per_hour,
            job_levels_per_hour, total_tracking_time, snapshot_count
        """
        with self._lock:
            try:
                db = sqlite3.connect(self._db_path, timeout=5.0)
                rows = db.execute(
                    """SELECT base_level, job_level, base_exp, job_exp, timestamp
                       FROM exp_snapshots
                       WHERE bot_id = ?
                       ORDER BY timestamp ASC""",
                    (bot_id,),
                ).fetchall()
                db.close()
            except Exception as e:
                logger.debug("ExperienceDB: get_leveling_speed failed: %s", e)
                return {"error": str(e)}

            if not rows:
                return {
                    "current_base_level": 0,
                    "current_job_level": 0,
                    "time_at_current_base": 0.0,
                    "base_levels_per_hour": 0.0,
                    "job_levels_per_hour": 0.0,
                    "total_tracking_time": 0.0,
                    "snapshot_count": 0,
                }

            first = rows[0]
            last = rows[-1]
            total_hours = max(0.001, (last[4] - first[4]) / 3600.0)

            # Count how many unique base levels gained
            base_levels = len(set(r[0] for r in rows))
            base_gained = last[0] - first[0]
            job_gained = last[1] - first[1]

            # Time at current base level
            now = time.time()
            latest_base_level = last[0]
            time_at_current = 0.0
            for r in reversed(rows):
                if r[0] == latest_base_level:
                    time_at_current = now - r[4]
                else:
                    break

            return {
                "current_base_level": last[0],
                "current_job_level": last[1],
                "time_at_current_base": time_at_current,
                "base_levels_per_hour": base_gained / total_hours if base_gained > 0 else 0.0,
                "job_levels_per_hour": job_gained / total_hours if job_gained > 0 else 0.0,
                "total_tracking_time": last[4] - first[4],
                "snapshot_count": len(rows),
            }

    def get_plateau_warnings(self, bot_id: str) -> list[dict[str, Any]]:
        """Detect leveling plateaus — same level for >60 minutes.

        Returns list of warnings, each with level_type ('base' or 'job'),
        level, duration_minutes, and current_map.
        """
        warnings: list[dict[str, Any]] = []
        with self._lock:
            try:
                db = sqlite3.connect(self._db_path, timeout=5.0)
                rows = db.execute(
                    """SELECT base_level, job_level, map_name, timestamp
                       FROM exp_snapshots
                       WHERE bot_id = ?
                       ORDER BY timestamp ASC""",
                    (bot_id,),
                ).fetchall()
                db.close()
            except Exception as e:
                logger.debug("ExperienceDB: get_plateau_warnings failed: %s", e)
                return warnings

            if len(rows) < 2:
                return warnings

            now = time.time()

            # Check base level plateau
            latest_base = rows[-1][0]
            base_first_ts = rows[0][3]
            for r in reversed(rows):
                if r[0] == latest_base:
                    base_first_ts = r[3]
                else:
                    break
            base_duration = (now - base_first_ts) / 60.0
            if base_duration > 60:
                warnings.append({
                    "level_type": "base",
                    "level": latest_base,
                    "duration_minutes": round(base_duration, 1),
                    "current_map": rows[-1][2],
                })

            # Check job level plateau
            latest_job = rows[-1][1]
            job_first_ts = rows[0][3]
            for r in reversed(rows):
                if r[1] == latest_job:
                    job_first_ts = r[3]
                else:
                    break
            job_duration = (now - job_first_ts) / 60.0
            if job_duration > 60:
                warnings.append({
                    "level_type": "job",
                    "level": latest_job,
                    "duration_minutes": round(job_duration, 1),
                    "current_map": rows[-1][2],
                })

            return warnings

    # ── Per-map EXP rates ───────────────────────────────────────────────

    def get_map_exp_rates(self, bot_id: str) -> dict[str, dict[str, float]]:
        """Return per-map average EXP rates.

        Returns dict[map_name] -> {base_exp_rate, job_exp_rate, zeny_rate, hours_tracked}
        """
        with self._lock:
            try:
                db = sqlite3.connect(self._db_path, timeout=5.0)
                rows = db.execute(
                    """SELECT map_name, base_exp, job_exp, zeny, timestamp
                       FROM exp_snapshots
                       WHERE bot_id = ?
                       ORDER BY map_name, timestamp ASC""",
                    (bot_id,),
                ).fetchall()
                db.close()
            except Exception as e:
                logger.debug("ExperienceDB: get_map_exp_rates failed: %s", e)
                return {}

            if not rows:
                return {}

            # Group by map
            map_data: dict[str, list[tuple[int, int, int, float]]] = {}
            for r in rows:
                map_name = r[0] or "unknown"
                map_data.setdefault(map_name, []).append((r[1], r[2], r[3], r[4]))

            result: dict[str, dict[str, float]] = {}
            for map_name, snapshots in map_data.items():
                if len(snapshots) < 2:
                    result[map_name] = {
                        "base_exp_rate": 0.0,
                        "job_exp_rate": 0.0,
                        "zeny_rate": 0.0,
                        "hours_tracked": 0.0,
                    }
                    continue
                first = snapshots[0]
                last = snapshots[-1]
                hours = max(0.001, (last[3] - first[3]) / 3600.0)
                result[map_name] = {
                    "base_exp_rate": max(0.0, (last[0] - first[0]) / hours),
                    "job_exp_rate": max(0.0, (last[1] - first[1]) / hours),
                    "zeny_rate": max(0.0, (last[2] - first[2]) / hours),
                    "hours_tracked": hours,
                }

            return result

    # ── Utility ─────────────────────────────────────────────────────────

    def get_latest_snapshot(self, bot_id: str) -> ExpSnapshot | None:
        """Get the most recent EXP snapshot for a bot, or None."""
        with self._lock:
            try:
                db = sqlite3.connect(self._db_path, timeout=5.0)
                row = db.execute(
                    """SELECT bot_id, base_level, job_level, base_exp, job_exp,
                              zeny, map_name, timestamp
                       FROM exp_snapshots
                       WHERE bot_id = ?
                       ORDER BY timestamp DESC LIMIT 1""",
                    (bot_id,),
                ).fetchone()
                db.close()
            except Exception:
                return None

            if row is None:
                return None
            return ExpSnapshot(
                bot_id=row[0],
                base_level=row[1],
                job_level=row[2],
                base_exp=row[3],
                job_exp=row[4],
                zeny=row[5],
                map_name=row[6],
                timestamp=row[7],
            )

    def stats(self) -> dict[str, Any]:
        """Return aggregate statistics."""
        with self._lock:
            try:
                db = sqlite3.connect(self._db_path, timeout=5.0)
                total = db.execute("SELECT COUNT(*) FROM exp_snapshots").fetchone()[0]
                unique_bots = db.execute(
                    "SELECT COUNT(DISTINCT bot_id) FROM exp_snapshots"
                ).fetchone()[0]
                unique_maps = db.execute(
                    "SELECT COUNT(DISTINCT map_name) FROM exp_snapshots WHERE map_name != ''"
                ).fetchone()[0]
                oldest = db.execute("SELECT MIN(timestamp) FROM exp_snapshots").fetchone()[0]
                newest = db.execute("SELECT MAX(timestamp) FROM exp_snapshots").fetchone()[0]
                db.close()
            except Exception:
                return {"total_snapshots": 0, "unique_bots": 0}

            return {
                "total_snapshots": total,
                "unique_bots": unique_bots,
                "unique_maps": unique_maps,
                "oldest_timestamp": oldest,
                "newest_timestamp": newest,
                "tracking_span_hours": (newest - oldest) / 3600.0 if oldest and newest else 0.0,
            }

# ── Backward-compatible aliases ─────────────────────────────────────────
# The rest of the codebase imports ExperienceDatabase and ExperienceEntry.
# ── Backward-compatible aliases ─────────────────────────────────────────
# The rest of the codebase imports ExperienceDatabase and ExperienceEntry
# with their original schemas. Keep them working alongside the new classes.


@dataclass
class ExperienceEntry:
    """A single experience datapoint for a bot in a specific context.
    (Legacy — maintained for backward compatibility)
    """
    bot_id: str
    timestamp: float
    context_type: str  # "combat" | "economy" | "survival" | "quest" | "craft" | "trade" | "refine" | "pvp" | "gvg" | "mvp"
    map_name: str = ""
    monster_name: str = ""
    role: str = ""
    action_taken: str = ""
    success: bool = False
    reward: float = 0.0  # XP gained, zeny earned, etc.
    details: dict[str, Any] = field(default_factory=dict)


class ExperienceDatabase:
    """Cross-bot experience database — shared learning across all bots.
    (Legacy — maintained for backward compatibility)

    Stores outcomes of actions in various contexts (combat, economy, quest, etc.)
    so that when bot A learns something, bot B can benefit from that knowledge.
    All state is in-memory with optional SQLite persistence.
    """

    def __init__(self, max_entries: int = 50000):
        self._lock = RLock()
        self._max_entries = max_entries
        self._entries: list[ExperienceEntry] = []
        # Indexes for fast lookup
        self._by_context: dict[str, list[int]] = {}
        self._by_map: dict[str, list[int]] = {}
        self._by_monster: dict[str, list[int]] = {}
        self._by_role: dict[str, list[int]] = {}

    def record(self, entry: ExperienceEntry) -> None:
        """Record an experience entry."""
        with self._lock:
            idx = len(self._entries)
            self._entries.append(entry)
            self._by_context.setdefault(entry.context_type, []).append(idx)
            if entry.map_name:
                self._by_map.setdefault(entry.map_name, []).append(idx)
            if entry.monster_name:
                self._by_monster.setdefault(entry.monster_name, []).append(idx)
            if entry.role:
                self._by_role.setdefault(entry.role, []).append(idx)
            # Trim oldest entries if over limit
            if len(self._entries) > self._max_entries:
                excess = len(self._entries) - self._max_entries
                self._entries = self._entries[excess:]
                self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        self._by_context.clear()
        self._by_map.clear()
        self._by_monster.clear()
        self._by_role.clear()
        for idx, entry in enumerate(self._entries):
            self._by_context.setdefault(entry.context_type, []).append(idx)
            if entry.map_name:
                self._by_map.setdefault(entry.map_name, []).append(idx)
            if entry.monster_name:
                self._by_monster.setdefault(entry.monster_name, []).append(idx)
            if entry.role:
                self._by_role.setdefault(entry.role, []).append(idx)

    def query(self, *, context_type: str | None = None, map_name: str | None = None,
              monster_name: str | None = None, role: str | None = None,
              limit: int = 100) -> list[ExperienceEntry]:
        with self._lock:
            candidates = set(range(len(self._entries)))
            if context_type:
                candidates &= set(self._by_context.get(context_type, []))
            if map_name:
                candidates &= set(self._by_map.get(map_name, []))
            if monster_name:
                candidates &= set(self._by_monster.get(monster_name, []))
            if role:
                candidates &= set(self._by_role.get(role, []))
            results = [self._entries[i] for i in sorted(candidates, reverse=True)[:limit]]
            return results

    def success_rate(self, *, context_type: str | None = None, map_name: str | None = None,
                     monster_name: str | None = None, action: str | None = None) -> float:
        with self._lock:
            entries = self.query(context_type=context_type, map_name=map_name,
                                 monster_name=monster_name, limit=1000)
            if not entries:
                return 0.5
            if action:
                entries = [e for e in entries if e.action_taken == action]
                if not entries:
                    return 0.5
            successes = sum(1 for e in entries if e.success)
            return successes / len(entries)

    def best_action(self, *, context_type: str, map_name: str = "",
                    monster_name: str = "", min_samples: int = 1) -> tuple[str, float]:
        with self._lock:
            entries = self.query(context_type=context_type, map_name=map_name,
                                 monster_name=monster_name, limit=1000)
            if not entries:
                return "", 0.0
            action_stats: dict[str, list[bool]] = {}
            for e in entries:
                if e.action_taken:
                    action_stats.setdefault(e.action_taken, []).append(e.success)
            best_action = ""
            best_rate = 0.0
            for action, outcomes in action_stats.items():
                if len(outcomes) < min_samples:
                    continue
                rate = sum(1 for s in outcomes if s) / len(outcomes)
                if rate > best_rate:
                    best_rate = rate
                    best_action = action
            return best_action, best_rate

    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            context_counts = {k: len(v) for k, v in sorted(self._by_context.items())}
            role_counts = {k: len(v) for k, v in sorted(self._by_role.items())}
            return {
                "total_entries": len(self._entries),
                "by_context": context_counts,
                "by_role": role_counts,
                "unique_maps": len(self._by_map),
                "unique_monsters": len(self._by_monster),
            }

    def persist(self, sqlite_path: str | None = None) -> None:
        if not sqlite_path:
            return
        try:
            import sqlite3
            db = sqlite3.connect(sqlite_path, timeout=5.0)
            db.execute("""CREATE TABLE IF NOT EXISTS experience (
                bot_id TEXT, timestamp REAL, context_type TEXT, map_name TEXT,
                monster_name TEXT, role TEXT, action_taken TEXT, success INTEGER,
                reward REAL, details TEXT
            )""")
            for e in self._entries:
                db.execute(
                    "INSERT INTO experience VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (e.bot_id, e.timestamp, e.context_type, e.map_name,
                     e.monster_name, e.role, e.action_taken, int(e.success),
                     e.reward, json.dumps(e.details)),
                )
            db.commit()
            db.close()
            logger.info("experience_persisted: %d entries to %s", len(self._entries), sqlite_path)
        except Exception:
            logger.warning("experience_persist_failed")

    def load(self, sqlite_path: str | None = None) -> int:
        if not sqlite_path:
            return 0
        try:
            import sqlite3
            db = sqlite3.connect(sqlite_path, timeout=5.0)
            cursor = db.execute("SELECT * FROM experience")
            loaded = 0
            for row in cursor:
                entry = ExperienceEntry(
                    bot_id=row[0], timestamp=row[1], context_type=row[2],
                    map_name=row[3] or "", monster_name=row[4] or "",
                    role=row[5] or "", action_taken=row[6] or "",
                    success=bool(row[7]), reward=float(row[8] or 0.0),
                    details=json.loads(row[9]) if row[9] else {},
                )
                self._entries.append(entry)
                loaded += 1
            db.close()
            self._rebuild_indexes()
            logger.info("experience_loaded: %d entries from %s", loaded, sqlite_path)
            return loaded
        except Exception:
            logger.warning("experience_load_failed")
            return 0
