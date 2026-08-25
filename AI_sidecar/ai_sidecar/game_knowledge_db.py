"""
GameKnowledgeDB — server-agnostic, database-backed RO game knowledge.

All zone ladders, stat builds, skill builds, NPC interactions are
queried from SQLite instead of hardcoded. Adapts to any server
by storing learned knowledge in player_memory table.
"""
from __future__ import annotations

import json
import threading
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
        self._local = threading.local()
        self._ensure_seeded()

    # ── Schema + seed ────────────────────────────────────────────────────

    def _ensure_seeded(self) -> None:
        """Create all tables and seed from knowledge.json if empty.

        The DB-backed knowledge layer was previously a read-only facade over
        tables that NOTHING ever created or populated — every query returned
        empty and consumers silently fell back to hardcoded paths. This
        guarantees schema existence and a baseline seed (zone ladder from
        monster spawn levels, skill builds from skill trees, job paths from
        job stats) so the DB path is actually functional.
        """
        try:
            conn = self._get_conn()
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS zone_ladder (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    map_name TEXT NOT NULL,
                    min_level INTEGER NOT NULL DEFAULT 1,
                    max_level INTEGER NOT NULL DEFAULT 99,
                    difficulty INTEGER NOT NULL DEFAULT 1,
                    monster_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS stat_builds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_name TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 1,
                    stats_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS skill_builds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_name TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 1,
                    skill_order TEXT NOT NULL DEFAULT '[]'
                );
                CREATE TABLE IF NOT EXISTS npc_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    npc_name TEXT NOT NULL,
                    map_name TEXT NOT NULL DEFAULT '',
                    task_type TEXT NOT NULL DEFAULT '',
                    x INTEGER NOT NULL DEFAULT 0,
                    y INTEGER NOT NULL DEFAULT 0,
                    steps_json TEXT NOT NULL DEFAULT '[]'
                );
                CREATE TABLE IF NOT EXISTS player_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bot_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS exp_efficiency (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    map_name TEXT NOT NULL,
                    bot_level INTEGER NOT NULL DEFAULT 1,
                    exp_per_hour REAL NOT NULL DEFAULT 0,
                    deaths_per_hour REAL NOT NULL DEFAULT 0,
                    sample_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS job_paths (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    from_job TEXT NOT NULL,
                    to_job TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 1,
                    requirements TEXT NOT NULL DEFAULT '',
                    order_index INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_zone_ladder_level ON zone_ladder(min_level, max_level);
                CREATE INDEX IF NOT EXISTS idx_npc_task ON npc_interactions(task_type, map_name);
                CREATE INDEX IF NOT EXISTS idx_skill_build_job ON skill_builds(job_name);
                """
            )
            conn.commit()
            # Schema migration: drop tables whose shape changed between
            # versions (job_paths gained from_job/to_job/requirements).
            try:
                _cols = [r[1] for r in conn.execute("PRAGMA table_info(job_paths)").fetchall()]
                if _cols and "from_job" not in _cols:
                    conn.execute("DROP TABLE IF EXISTS job_paths")
                    conn.execute(
                        """
                        CREATE TABLE job_paths (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            from_job TEXT NOT NULL,
                            to_job TEXT NOT NULL,
                            priority INTEGER NOT NULL DEFAULT 1,
                            requirements TEXT NOT NULL DEFAULT '',
                            order_index INTEGER NOT NULL DEFAULT 0
                        )
                        """
                    )
                    conn.commit()
            except Exception:
                pass
            # Migration: npc_interactions gained x/y coordinates (data-driven
            # NPC/portal resolution). Add if missing (idempotent).
            try:
                _ncols = [r[1] for r in conn.execute("PRAGMA table_info(npc_interactions)").fetchall()]
                if _ncols and "x" not in _ncols:
                    conn.execute("ALTER TABLE npc_interactions ADD COLUMN x INTEGER NOT NULL DEFAULT 0")
                    conn.execute("ALTER TABLE npc_interactions ADD COLUMN y INTEGER NOT NULL DEFAULT 0")
                    conn.commit()
            except Exception:
                pass
            # Seed only when the DB is genuinely empty (first run)
            empty = conn.execute("SELECT COUNT(*) AS c FROM zone_ladder").fetchone()[0] == 0
            if empty:
                self._seed_from_knowledge(conn)
                conn.commit()
            # Always ensure the baseline NPC facts exist (idempotent — the bot
            # needs these coordinates for data-driven portal/shop resolution on
            # ANY server; they are FACTS, never decision logic).
            self._seed_npc_interaction_facts(conn)
            conn.commit()
        except Exception as e:
            logger.warning("game_knowledge_db_seed_failed: %s", e)

    def _seed_from_knowledge(self, conn: sqlite3.Connection) -> None:
        """Populate baseline tables from knowledge/knowledge.json."""
        kpath = Path(__file__).resolve().parent.parent / "knowledge" / "knowledge.json"
        if not kpath.exists():
            logger.info("game_knowledge_db_seed_skip: no knowledge.json")
            return
        try:
            with open(kpath) as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("game_knowledge_db_seed_read_failed: %s", e)
            return

        # ── zone_ladder: authoritative LEVEL_LADDER tiers + map_drops ──
        # The mobs in knowledge.json carry no spawn-map keys (ingestion
        # limitation), so the level ladder comes from the curated
        # GameKnowledgeService.LEVEL_LADDER, extended with monster-level
        # aggregates from map_drops for the higher-tier zones.
        try:
            from ai_sidecar.game_knowledge import LEVEL_LADDER
            for min_lv, max_lv, map_name, _desc in LEVEL_LADDER:
                conn.execute(
                    "INSERT INTO zone_ladder(map_name, min_level, max_level, difficulty, monster_count) "
                    "VALUES (?, ?, ?, ?, 0)",
                    (str(map_name).lower(), min_lv, max_lv,
                     max(1, min(10, (max_lv - min_lv) // 10 + 1))),
                )
        except Exception:
            pass

        # map_drops: maps -> monster names -> levels (high-tier zones)
        mobs = data.get("mobs", []) or []
        mob_levels = {
            str(m.get("AegisName", "") or "").upper(): int(m.get("Level", 0) or 0)
            for m in mobs
        }
        for entry in data.get("map_drops", []) or []:
            mn = str(entry.get("Map", "") or "").lower().replace(".gat", "")
            if not mn:
                continue
            lvls = [
                mob_levels.get(str(s.get("Monster", "") or "").upper(), 0)
                for s in (entry.get("SpecificDrops", []) or [])
            ]
            lvls = [l for l in lvls if l > 0]
            if not lvls:
                continue
            conn.execute(
                "INSERT INTO zone_ladder(map_name, min_level, max_level, difficulty, monster_count) "
                "VALUES (?, ?, ?, ?, ?)",
                (mn, min(lvls), max(lvls),
                 max(1, min(10, (max(lvls) - min(lvls)) // 10 + 1)),
                 len(lvls)),
            )

        # ── skill_builds: first skill-tree per job ──
        skill_trees = data.get("skill_trees", []) or []
        for tree in skill_trees:
            job = str(tree.get("Job", "") or "").lower()
            tree_list = tree.get("Tree") or []
            if not job or not tree_list:
                continue
            order = [
                [str(s.get("Name", "") or ""), int(s.get("MaxLevel", 1) or 1)]
                for s in tree_list if isinstance(s, dict) and s.get("Name")
            ]
            if order:
                conn.execute(
                    "INSERT INTO skill_builds(job_name, priority, skill_order) VALUES (?, 1, ?)",
                    (job, json.dumps(order)),
                )

        # ── job_paths: parent/child from job_stats naming ──
        job_stats = data.get("job_stats", {}) or {}
        order_idx = 0
        for job in job_stats:
            base = str(job).lower()
            parent = ""
            for suffix in ("_high", "_baby"):
                if base.endswith(suffix):
                    parent = base[: -len(suffix)]
                    break
            if parent:
                conn.execute(
                    "INSERT INTO job_paths(from_job, to_job, priority, order_index) "
                    "VALUES (?, ?, 1, ?)",
                    (parent, base, order_idx),
                )
            order_idx += 1

        n_zones = conn.execute("SELECT COUNT(*) FROM zone_ladder").fetchone()[0]
        n_skills = conn.execute("SELECT COUNT(*) FROM skill_builds").fetchone()[0]
        logger.info(
            "game_knowledge_db_seeded: %d zones, %d skill builds, %d job paths",
            n_zones, n_skills, order_idx,
        )

    def _seed_npc_interaction_facts(self, conn: sqlite3.Connection) -> None:
        """Seed baseline NPC-interaction FACTS (data-driven, idempotent).

        These are DATA (server-agnostic baseline facts for common tasks) — NOT
        decision logic. On this server the observation layer can refine them;
        a different server would learn its own. The conscious LLM + the
        knowledge-DB-driven blocks consume them. INSERT OR IGNORE keeps it
        idempotent (bot-learned refinements are never overwritten).
        """
        try:
            _facts = [
                # (task_type, map_name, npc_name, x, y)
                ("weapon_shop",  "prontera", "Weapon Shop",  160, 133),
                ("academy_receptionist", "iz_ac01", "Academy Receptionist", 100, 39),
                ("portal_to_town", "prt_fild05", "Prontera gate", 22, 203),
                ("portal_to_hunt", "prontera", "Prontera field gate", 156, 164),
                ("portal_to_town", "izlude", "Izlude dock", 128, 260),
                ("kafra",         "prontera", "Kafra Employee", 145, 122),
                ("healer",        "prontera", "Townsfolk", 154, 177),
            ]
            for _task, _map, _name, _x, _y in _facts:
                conn.execute(
                    "INSERT OR IGNORE INTO npc_interactions "
                    "(npc_name, map_name, task_type, x, y) VALUES (?,?,?,?,?)",
                    (_name, _map, _task, _x, _y),
                )
        except Exception:
            pass

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self._db_path))
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=5000")
        return self._local.conn

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
            "SELECT * FROM npc_interactions WHERE task_type=? AND LOWER(map_name)=LOWER(?) LIMIT 1",
            (task_type, map_name),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def list_npcs_on_map(self, map_name: str) -> list[dict]:
        """List all known NPC interactions on a given map from the DB."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM npc_interactions WHERE LOWER(map_name)=LOWER(?) ORDER BY npc_name",
            (map_name,),
        )
        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]

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

    def get_job_path(self, from_job: str, priority: int = 1) -> dict | None:
        """Get the next job change recommendation from the job path DB."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM job_paths WHERE LOWER(from_job)=LOWER(?) AND priority=? ORDER BY priority ASC LIMIT 1",
            (from_job, priority),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_job_tree(self, current_job: str, base_level: int, current_stats: dict) -> list[dict]:
        """Build a progression tree from current job through all available upgrades."""
        conn = self._get_conn()
        cur = conn.cursor()
        paths = []
        cur.execute(
            "SELECT * FROM job_paths WHERE LOWER(from_job)=LOWER(?) ORDER BY priority ASC",
            (current_job,),
        )
        for row in cur.fetchall():
            jp = dict(row)
            # Check if stats meet requirements
            meets_reqs = True
            if jp.get("requirements"):
                reqs = jp["requirements"]
                import re
                for stat_req in re.findall(r'([A-Z]+)\s+(\d+)\+', reqs):
                    stat_name, needed = stat_req[0].lower(), int(stat_req[1])
                    if (current_stats.get(stat_name, 1) or 1) < needed:
                        meets_reqs = False
            jp["meets_requirements"] = meets_reqs
            paths.append(jp)
        return paths

    def plan_build(self, current_job: str, base_level: int, current_stats: dict) -> dict:
        """Plan the optimal build path: which job to aim for, what stats to prioritize."""
        # Get primary job path
        job_path = self.get_job_path(current_job)
        target_job = job_path["to_job"] if job_path else "adventurer"
        
        # Get stat build for target job
        stat_build = self.get_stat_build(target_job)
        if not stat_build:
            stat_build = self.get_stat_build(current_job)
        
        return {
            "current_job": current_job,
            "target_job": target_job,
            "stat_build": stat_build,
            "job_tree": self.get_job_tree(current_job, base_level, current_stats),
            "next_job_change": self.get_job_path(current_job),
        }

    def optimize_hunting_map(self, bot_id: str, base_level: int, known_maps: set[str]) -> str | None:
        """Choose the best map to hunt on between DB data and zone ladder."""
        best = self.get_best_zones(bot_id, base_level, 1)
        if best:
            return best[0]["map_name"]
        zone = self.get_hunting_zone(base_level)
        if zone:
            return zone["map_name"]
        return None
