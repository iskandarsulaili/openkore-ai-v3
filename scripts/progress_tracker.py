#!/usr/bin/env python3
"""
Progress Tracker — Real farming metrics that matter to a pro player.
===============================================================
Measures: XP/hour, zeny/hour, items/hour, levels gained, deaths.
Stored in SQLite, viewable via API or CLI.
"""

import json
import logging
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FarmingMetrics:
    """Real farming metrics that matter to a pro player."""
    bot_id: str
    timestamp: float = field(default_factory=time.time)
    base_level: int = 0
    job_level: int = 0
    base_exp: int = 0
    job_exp: int = 0
    zeny: int = 0
    items_looted: int = 0
    monsters_killed: int = 0
    deaths: int = 0
    map_name: str = ""
    runtime_hours: float = 0.0
    session_active: bool = True

    @property
    def base_exp_hour(self) -> float:
        return self.base_exp / max(self.runtime_hours, 0.01)

    @property
    def zeny_hour(self) -> float:
        return self.zeny / max(self.runtime_hours, 0.01)

    @property
    def items_hour(self) -> float:
        return self.items_looted / max(self.runtime_hours, 0.01)

    @property
    def deaths_hour(self) -> float:
        return self.deaths / max(self.runtime_hours, 0.01)


class ProgressTracker:
    """Tracks real farming progress across all bots.

    A pro player checks this dashboard to decide if the bot is worth keeping.
    """

    def __init__(self, db_path: str = "data/progress.db"):
        self._lock = Lock()
        self._db_path = db_path
        self._snapshots: dict[str, FarmingMetrics] = {}  # bot_id -> last snapshot
        self._session_start: dict[str, float] = {}  # bot_id -> start time
        self._init_db()

    def _init_db(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            db = sqlite3.connect(self._db_path, timeout=5.0)
            db.execute("""
                CREATE TABLE IF NOT EXISTS farming_progress (
                    bot_id TEXT, timestamp REAL,
                    base_level INTEGER, job_level INTEGER,
                    base_exp INTEGER, job_exp INTEGER,
                    zeny INTEGER, items_looted INTEGER,
                    monsters_killed INTEGER, deaths INTEGER,
                    map_name TEXT, runtime_hours REAL,
                    session_active INTEGER
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS farming_sessions (
                    bot_id TEXT, session_start REAL, session_end REAL,
                    base_exp_gained INTEGER, job_exp_gained INTEGER,
                    zeny_gained INTEGER, items_gained INTEGER,
                    deaths INTEGER, avg_base_exp_hour REAL,
                    avg_zeny_hour REAL, status TEXT
                )
            """)
            db.commit()
            db.close()
        except Exception as e:
            logger.warning("Failed to init progress DB: %s", e)

    def start_session(self, bot_id: str) -> None:
        with self._lock:
            self._session_start[bot_id] = time.time()
            self._snapshots[bot_id] = FarmingMetrics(bot_id=bot_id)

    def record_snapshot(self, metrics: FarmingMetrics) -> None:
        with self._lock:
            bot_id = metrics.bot_id
            prev = self._snapshots.get(bot_id)

            # Calculate runtime
            start = self._session_start.get(bot_id, time.time())
            metrics.runtime_hours = (time.time() - start) / 3600

            # Persist
            try:
                db = sqlite3.connect(self._db_path, timeout=5.0)
                db.execute(
                    "INSERT INTO farming_progress VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (metrics.bot_id, metrics.timestamp,
                     metrics.base_level, metrics.job_level,
                     metrics.base_exp, metrics.job_exp,
                     metrics.zeny, metrics.items_looted,
                     metrics.monsters_killed, metrics.deaths,
                     metrics.map_name, metrics.runtime_hours,
                     1 if metrics.session_active else 0)
                )
                db.commit()
                db.close()
            except Exception as e:
                logger.warning("Failed to record progress: %s", e)

            self._snapshots[bot_id] = metrics

    def end_session(self, bot_id: str) -> dict[str, Any] | None:
        with self._lock:
            start = self._session_start.pop(bot_id, None)
            current = self._snapshots.pop(bot_id, None)
            if not start or not current:
                return None

            end = time.time()
            runtime_h = (end - start) / 3600

            session = {
                "bot_id": bot_id,
                "session_start": start,
                "session_end": end,
                "runtime_hours": round(runtime_h, 2),
                "base_exp_gained": current.base_exp,
                "job_exp_gained": current.job_exp,
                "zeny_gained": current.zeny,
                "items_gained": current.items_looted,
                "deaths": current.deaths,
                "avg_base_exp_hour": round(current.base_exp / max(runtime_h, 0.01), 0),
                "avg_zeny_hour": round(current.zeny / max(runtime_h, 0.01), 0),
                "levels_gained": f"{current.base_level} / {current.job_level}",
                "status": "completed",
            }

            try:
                db = sqlite3.connect(self._db_path, timeout=5.0)
                db.execute(
                    "INSERT INTO farming_sessions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (session["bot_id"], session["session_start"], session["session_end"],
                     session["base_exp_gained"], session["job_exp_gained"],
                     session["zeny_gained"], session["items_gained"],
                     session["deaths"], session["avg_base_exp_hour"],
                     session["avg_zeny_hour"], session["status"])
                )
                # Mark progress records as inactive
                db.execute(
                    "UPDATE farming_progress SET session_active = 0 WHERE bot_id = ? AND session_active = 1",
                    (bot_id,))
                db.commit()
                db.close()
            except Exception as e:
                logger.warning("Failed to end session: %s", e)

            return session

    def dashboard(self, bot_id: str | None = None) -> dict[str, Any]:
        """Return the farming dashboard — what a pro player checks first."""
        try:
            db = sqlite3.connect(self._db_path, timeout=5.0)
            db.row_factory = sqlite3.Row

            result = {"bots": {}, "totals": {}, "sessions": []}

            if bot_id:
                rows = db.execute(
                    "SELECT * FROM farming_sessions WHERE bot_id = ? ORDER BY session_end DESC LIMIT 10",
                    (bot_id,)).fetchall()
            else:
                rows = db.execute(
                    "SELECT * FROM farming_sessions ORDER BY session_end DESC LIMIT 50").fetchall()

            total_exp = 0
            total_zeny = 0
            total_items = 0
            total_deaths = 0
            total_runtime = 0.0

            for row in rows:
                d = dict(row)
                result["sessions"].append(d)
                bid = d["bot_id"]
                if bid not in result["bots"]:
                    result["bots"][bid] = {
                        "total_exp": 0, "total_zeny": 0,
                        "total_items": 0, "total_deaths": 0,
                        "total_runtime": 0.0, "sessions": 0,
                    }
                b = result["bots"][bid]
                b["total_exp"] += d["base_exp_gained"]
                b["total_zeny"] += d["zeny_gained"]
                b["total_items"] += d["items_gained"]
                b["total_deaths"] += d["deaths"]
                b["total_runtime"] += d["runtime_hours"]
                b["sessions"] += 1
                total_exp += d["base_exp_gained"]
                total_zeny += d["zeny_gained"]
                total_items += d["items_gained"]
                total_deaths += d["deaths"]
                total_runtime += d["runtime_hours"]

            result["totals"] = {
                "total_exp": total_exp,
                "total_zeny": total_zeny,
                "total_items": total_items,
                "total_deaths": total_deaths,
                "total_runtime_hours": round(total_runtime, 2),
                "avg_exp_hour": round(total_exp / max(total_runtime, 0.01), 0),
                "avg_zeny_hour": round(total_zeny / max(total_runtime, 0.01), 0),
            }

            db.close()
            return result
        except Exception as e:
            logger.warning("Dashboard query failed: %s", e)
            return {"error": str(e)}

    def live_metrics(self, bot_id: str) -> dict[str, Any]:
        """Return live metrics for the current session."""
        with self._lock:
            snapshot = self._snapshots.get(bot_id)
            if not snapshot:
                return {"status": "no_active_session"}
            start = self._session_start.get(bot_id, time.time())
            runtime_h = (time.time() - start) / 3600

            return {
                "bot_id": bot_id,
                "runtime_hours": round(runtime_h, 2),
                "base_level": snapshot.base_level,
                "job_level": snapshot.job_level,
                "base_exp_gained": snapshot.base_exp,
                "zeny_gained": snapshot.zeny,
                "items_looted": snapshot.items_looted,
                "monsters_killed": snapshot.monsters_killed,
                "deaths": snapshot.deaths,
                "map": snapshot.map_name,
                "exp_hour": round(snapshot.base_exp / max(runtime_h, 0.01), 0),
                "zeny_hour": round(snapshot.zeny / max(runtime_h, 0.01), 0),
                "items_hour": round(snapshot.items_looted / max(runtime_h, 0.01), 1),
                "deaths_hour": round(snapshot.deaths / max(runtime_h, 0.01), 2),
            }


def format_dashboard(data: dict[str, Any]) -> str:
    """Format dashboard data for terminal display."""
    lines = []
    lines.append("=" * 60)
    lines.append("FARMING DASHBOARD — What a Pro Player Cares About")
    lines.append("=" * 60)

    # Totals
    totals = data.get("totals", {})
    lines.append(f"\nTotal Runtime: {totals.get('total_runtime_hours', 0):.1f}h")
    lines.append(f"Total EXP:     {totals.get('total_exp', 0):,}  ({totals.get('avg_exp_hour', 0):,.0f}/h)")
    lines.append(f"Total Zeny:    {totals.get('total_zeny', 0):,}z  ({totals.get('avg_zeny_hour', 0):,.0f}z/h)")
    lines.append(f"Total Items:   {totals.get('total_items', 0):,}")
    lines.append(f"Deaths:        {totals.get('total_deaths', 0)}")
    lines.append(f"Profit:        {totals.get('total_zeny', 0) - totals.get('total_deaths', 0) * 1000:,}z (est)")

    # Per-bot breakdown
    for bot_id, b in data.get("bots", {}).items():
        lines.append(f"\n── {bot_id} ──")
        lines.append(f"  Runtime: {b['total_runtime']:.1f}h over {b['sessions']} sessions")
        lines.append(f"  EXP:     {b['total_exp']:,}  ({b['total_exp']/max(b['total_runtime'],0.01):,.0f}/h)")
        lines.append(f"  Zeny:    {b['total_zeny']:,}z  ({b['total_zeny']/max(b['total_runtime'],0.01):,.0f}z/h)")
        lines.append(f"  Items:   {b['total_items']:,}")
        lines.append(f"  Deaths:  {b['total_deaths']}  ({b['total_deaths']/max(b['total_runtime'],0.01):.2f}/h)")

    return "\n".join(lines)