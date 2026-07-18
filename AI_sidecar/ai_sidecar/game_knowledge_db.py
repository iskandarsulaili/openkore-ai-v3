"""
GameKnowledgeDB — server-agnostic, database-backed RO game knowledge.

All zone ladders, stat builds, skill builds, NPC interactions are
queried from SQLite instead of hardcoded. Adapts to any server
by storing learned knowledge in player_memory table.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "bot_knowledge.db"


class GameKnowledgeDB:
    """Thread-safe, cached, database-backed RO game knowledge service."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else _DB_PATH
        self._local = None  # thread-local connection

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if self._local is None:
            self._local = sqlite3.connect(str(self._db_path))
            self._local.row_factory = sqlite3.Row
            self._local.execute("PRAGMA journal_mode=WAL")
            self._local.execute("PRAGMA busy_timeout=5000")
        return self._local

    # ── Zone Ladder ────────────────────────────────────────────────

    def get_hunting_zone(self, base_level: int, class_hint: str = "all") -> dict | None:
        """Find the best hunting zone for a given level and class."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM zone_ladder WHERE ? BETWEEN min_level AND max_level "
            "ORDER BY difficulty ASC, max_level ASC LIMIT 1",
            (base_level,),
        )
        row = cur.fetchone()
        if row:
            return dict(row)
        # Fallback: closest zone
        cur.execute(
            "SELECT * FROM zone_ladder ORDER BY ABS((min_level+max_level)/2 - ?) ASC LIMIT 1",
            (base_level,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_zone_ladder(self) -> list[dict]:
        """Return the full zone ladder sorted by level range."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT * FROM zone_ladder ORDER BY min_level ASC")
        return [dict(r) for r in cur.fetchall()]

    # ── Stat Builds ────────────────────────────────────────────────

    def get_stat_build(self, job_name: str, priority: int = 1) -> dict | None:
        """Get the recommended stat build for a job class."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM stat_builds WHERE LOWER(job_name)=LOWER(?) AND priority=? ORDER BY priority ASC LIMIT 1",
            (job_name, priority),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def allocate_stats(self, bot_level: int, current_stats: dict, job_name: str = "novice",
                       available_points: int = 36) -> dict:
        """Distribute available stat points according to build ratios.
        
        Uses the DB-backed build plan (ratios + breakpoints) to allocate
        available_points across stats. Handles any starting values and
        ensures breakpoints aren't exceeded.
        """
        build = self.get_stat_build(job_name)
        if not build or available_points <= 0:
            return {}

        stat_order = json.loads(build["stat_order"])
        stat_ratios = json.loads(build["stat_ratios"])
        breakpoints = json.loads(build.get("stat_breakpoints", "{}"))

        total_ratio = sum(stat_ratios.values())
        if total_ratio == 0:
            total_ratio = 1

        # Calculate how many of the available points go to each stat
        raw_allocation = {}
        remaining = available_points
        for stat in stat_order:
            ratio = stat_ratios.get(stat, 0) / total_ratio
            points = int(available_points * ratio)
            current = current_stats.get(stat, 1) or 1
            
            # Respect breakpoints
            stat_bp = breakpoints.get(stat, [])
            if stat_bp:
                next_bp = min((bp for bp in stat_bp if bp > current), default=999)
                max_to_next_bp = max(0, next_bp - current)
                points = min(points, max_to_next_bp)
            
            points = min(points, remaining)
            if points > 0:
                raw_allocation[stat] = points
                remaining -= points

        # Distribute any leftover points (due to rounding) round-robin
        if remaining > 0:
            for stat in stat_order:
                if remaining <= 0:
                    break
                current = current_stats.get(stat, 1) or 1
                stat_bp = breakpoints.get(stat, [])
                if stat_bp:
                    next_bp = min((bp for bp in stat_bp if bp > current), default=999)
                    if current + raw_allocation.get(stat, 0) < next_bp:
                        raw_allocation[stat] = raw_allocation.get(stat, 0) + 1
                        remaining -= 1
                else:
                    raw_allocation[stat] = raw_allocation.get(stat, 0) + 1
                    remaining -= 1

        return raw_allocation

    # ── Skill Builds ───────────────────────────────────────────────

    def get_skill_build(self, job_name: str, priority: int = 1) -> dict | None:
        """Get recommended skill build for a job."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM skill_builds WHERE LOWER(job_name)=LOWER(?) AND priority=? ORDER BY priority ASC LIMIT 1",
            (job_name, priority),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_next_skill(self, job_name: str, current_skills: dict) -> tuple | None:
        """Find the next skill to train based on build plan."""
        build = self.get_skill_build(job_name)
        if not build:
            return None
        skill_order = json.loads(build["skill_order"])
        for skill_id, max_lv in skill_order:
            current_lv = current_skills.get(skill_id, 0) or 0
            if current_lv < max_lv:
                return (skill_id, current_lv + 1)
        return None

    # ── NPC Interactions ───────────────────────────────────────────

    def get_npc_interaction(self, npc_name: str, map_name: str | None = None) -> dict | None:
        """Get NPC interaction steps by name."""
        conn = self._get_conn()
        cur = conn.cursor()
        if map_name:
            cur.execute(
                "SELECT * FROM npc_interactions WHERE LOWER(npc_name) LIKE LOWER(?) AND LOWER(map_name)=LOWER(?) LIMIT 1",
                (f"%{npc_name}%", map_name),
            )
        else:
            cur.execute(
                "SELECT * FROM npc_interactions WHERE LOWER(npc_name) LIKE LOWER(?) LIMIT 1",
                (f"%{npc_name}%",),
            )
        row = cur.fetchone()
        return dict(row) if row else None

    @lru_cache(maxsize=64)
    def find_npc_for_task(self, task_type: str, map_name: str = "prontera") -> dict | None:
        """Find an NPC that can fulfill a task (buy, sell, heal, storage, job_change, quest)."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM npc_interactions WHERE interaction_type=? AND LOWER(map_name)=LOWER(?) LIMIT 1",
            (task_type, map_name),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    # ── Player Memory (Learned Knowledge) ──────────────────────────

    def remember(self, bot_id: str, key: str, value: str, confidence: float = 0.5) -> None:
        """Store a piece of learned knowledge."""
        conn = self._get_conn()
        now = time.time()
        conn.execute(
            "INSERT OR REPLACE INTO player_memory (bot_id, knowledge_key, knowledge_value, confidence, discovered_at, last_verified_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (bot_id, key, value, confidence, now, now),
        )
        conn.commit()

    def recall(self, bot_id: str, key: str) -> str | None:
        """Recall a stored knowledge by key."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT knowledge_value FROM player_memory WHERE bot_id=? AND knowledge_key=?",
            (bot_id, key),
        )
        row = cur.fetchone()
        return row[0] if row else None

    # ── Exp Efficiency ─────────────────────────────────────────────

    def track_exp(self, bot_id: str, map_name: str, base_exp: int, job_exp: int,
                  zeny: int, kills: int, deaths: int, duration_s: float) -> None:
        """Record exp efficiency for a session."""
        conn = self._get_conn()
        now = time.time()
        conn.execute(
            "INSERT INTO exp_efficiency (bot_id, map_name, base_level, base_exp_gained, job_exp_gained, "
            "zeny_gained, time_spent_seconds, kills, deaths, session_start, session_end) "
            "VALUES (?, ?, (SELECT COALESCE(base_level, 1) FROM exp_efficiency WHERE bot_id=? ORDER BY session_end DESC LIMIT 1), "
            "?, ?, ?, ?, ?, ?, ?, ?)",
            (bot_id, map_name, bot_id, base_exp, job_exp, zeny, duration_s, kills, deaths, now - duration_s, now),
        )
        conn.commit()

    def get_best_zones(self, bot_id: str, base_level: int, top_n: int = 3) -> list[dict]:
        """Get the most exp-efficient zones for this bot based on history."""
        conn = self._get_conn()
        cur = conn.cursor()
        # Prefer database-backed efficiency, fallback to zone ladder
        cur.execute("""
            SELECT e.map_name,
                   SUM(e.base_exp_gained) / MAX(SUM(e.time_spent_seconds), 1) * 3600 AS exp_per_hour,
                   SUM(e.kills) AS total_kills,
                   SUM(e.deaths) AS total_deaths
            FROM exp_efficiency e
            WHERE e.bot_id=? AND e.base_level BETWEEN ?-5 AND ?+5
            GROUP BY e.map_name
            ORDER BY exp_per_hour DESC
            LIMIT ?
        """, (bot_id, base_level, base_level, top_n))
        rows = cur.fetchall()
        if rows:
            return [{"map_name": r[0], "exp_per_hour": r[1], "kills": r[2], "deaths": r[3]} for r in rows]
        # Fallback: zone ladder
        zone = self.get_hunting_zone(base_level)
        if zone:
            return [{"map_name": zone["map_name"], "exp_per_hour": 0, "kills": 0, "deaths": 0}]
        return []

    def optimize_hunting_map(self, bot_id: str, base_level: int, known_maps: set[str]) -> str | None:
        """Choose the best map to hunt on between DB data and zone ladder."""
        best = self.get_best_zones(bot_id, base_level, 1)
        if best:
            return best[0]["map_name"]
        zone = self.get_hunting_zone(base_level)
        if zone:
            return zone["map_name"]
        return None
