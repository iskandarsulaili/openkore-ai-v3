"""Quest tracking — active quests, progress, objectives stored in memory/sqlite."""
from __future__ import annotations

import json
import logging
import sqlite3
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class QuestObjective:
    """A single quest objective."""
    description: str = ""
    target: str = ""  # monster/item/npc name
    target_type: str = ""  # kill, collect, talk, deliver
    current: int = 0
    required: int = 1
    completed: bool = False


@dataclass
class QuestState:
    """Full state of a quest."""
    quest_id: str = ""
    quest_name: str = ""
    quest_level: int = 1
    status: str = "inactive"  # inactive, active, completed, failed
    objectives: list[QuestObjective] = field(default_factory=list)
    npc_start: str = ""
    npc_complete: str = ""
    rewards: dict[str, Any] = field(default_factory=dict)
    started_at: float = 0.0
    completed_at: float = 0.0
    expires_at: float = 0.0
    map_name: str = ""


_QUEST_DB_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "quest_data.db"


class QuestTracker:
    """Track active quests, progress, objectives.

    Persists quest state to SQLite for cross-session tracking.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else _QUEST_DB_PATH
        self._local = threading.local()
        self._active_quests: dict[str, dict[str, QuestState]] = {}  # bot_id -> {quest_id: QuestState}
        self._ensure_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._local.conn = sqlite3.connect(str(self._db_path))
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn

    def _ensure_db(self) -> None:
        """Ensure quest tracking tables exist."""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS quest_tracking (
                bot_id TEXT NOT NULL,
                quest_id TEXT NOT NULL,
                quest_name TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'inactive',
                objectives TEXT NOT NULL DEFAULT '[]',
                npc_start TEXT NOT NULL DEFAULT '',
                npc_complete TEXT NOT NULL DEFAULT '',
                rewards TEXT NOT NULL DEFAULT '{}',
                started_at REAL NOT NULL DEFAULT 0,
                completed_at REAL NOT NULL DEFAULT 0,
                expires_at REAL NOT NULL DEFAULT 0,
                map_name TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (bot_id, quest_id)
            );
            CREATE TABLE IF NOT EXISTS quest_progress (
                bot_id TEXT NOT NULL,
                quest_id TEXT NOT NULL,
                objective_idx INTEGER NOT NULL DEFAULT 0,
                current INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL DEFAULT (strftime('%s','now')),
                PRIMARY KEY (bot_id, quest_id, objective_idx)
            );
        """)
        conn.commit()

    def track_quest(self, bot_id: str, quest_state: QuestState) -> None:
        """Track a quest in memory and persist to DB."""
        bot_quests = self._active_quests.setdefault(bot_id, {})
        bot_quests[quest_state.quest_id or quest_state.quest_name] = quest_state
        self._persist_quest(bot_id, quest_state)

    def _persist_quest(self, bot_id: str, qs: QuestState) -> None:
        """Persist a quest state to SQLite."""
        conn = self._get_conn()
        quest_id = qs.quest_id or qs.quest_name
        conn.execute(
            """INSERT OR REPLACE INTO quest_tracking
               (bot_id, quest_id, quest_name, status, objectives,
                npc_start, npc_complete, rewards,
                started_at, completed_at, expires_at, map_name)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                bot_id, quest_id, qs.quest_name, qs.status,
                json.dumps([{
                    "description": o.description,
                    "target": o.target,
                    "target_type": o.target_type,
                    "current": o.current,
                    "required": o.required,
                    "completed": o.completed,
                } for o in qs.objectives]),
                qs.npc_start, qs.npc_complete, json.dumps(qs.rewards),
                qs.started_at, qs.completed_at, qs.expires_at, qs.map_name,
            ),
        )
        conn.commit()

    def update_objective_progress(
        self,
        bot_id: str,
        quest_id: str,
        objective_idx: int,
        current: int,
    ) -> None:
        """Update progress on a specific objective."""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO quest_progress
               (bot_id, quest_id, objective_idx, current, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (bot_id, quest_id, objective_idx, current, time.time()),
        )
        conn.commit()

        # Update in-memory state
        bot_quests = self._active_quests.get(bot_id, {})
        qs = bot_quests.get(quest_id)
        if qs and objective_idx < len(qs.objectives):
            qs.objectives[objective_idx].current = current
            if current >= qs.objectives[objective_idx].required:
                qs.objectives[objective_idx].completed = True

    def get_active_quests(self, bot_id: str) -> list[QuestState]:
        """Get all active quests for a bot."""
        bot_quests = self._active_quests.get(bot_id, {})
        return [qs for qs in bot_quests.values() if qs.status == "active"]

    def get_completed_quests(self, bot_id: str) -> list[QuestState]:
        """Get all completed quests for a bot."""
        bot_quests = self._active_quests.get(bot_id, {})
        return [qs for qs in bot_quests.values() if qs.status == "completed"]

    def get_quest(self, bot_id: str, quest_id: str) -> QuestState | None:
        """Get quest state for a specific quest."""
        bot_quests = self._active_quests.get(bot_id, {})
        return bot_quests.get(quest_id)

    def has_active_quests(self, bot_id: str) -> bool:
        """Check if bot has any active quests."""
        return len(self.get_active_quests(bot_id)) > 0

    def complete_quest(self, bot_id: str, quest_id: str) -> None:
        """Mark a quest as completed."""
        now = time.time()
        bot_quests = self._active_quests.get(bot_id, {})
        qs = bot_quests.get(quest_id)
        if qs:
            qs.status = "completed"
            qs.completed_at = now
            self._persist_quest(bot_id, qs)
            logger.info("[quests] %s: marked quest %s as completed", bot_id, quest_id)

    def fail_quest(self, bot_id: str, quest_id: str) -> None:
        """Mark a quest as failed."""
        bot_quests = self._active_quests.get(bot_id, {})
        qs = bot_quests.get(quest_id)
        if qs:
            qs.status = "failed"
            self._persist_quest(bot_id, qs)

    def get_available_for_level(
        self,
        base_level: int,
        map_name: str = "prontera",
        limit: int = 5,
    ) -> list[dict]:
        """Get quests available for the bot's level.

        Returns the trackable quests this tracker knows about (persisted in
        quest_tracking) that are either already active or not yet complete,
        optionally filtered to the given map. Falls back to active in-memory
        quests when none are persisted yet.
        """
        out: list[dict] = []
        try:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT bot_id, quest_id, quest_name, status, map_name, "
                "npc_start, npc_complete, objectives, rewards "
                "FROM quest_tracking "
                "WHERE status IN ('active','inactive','in_progress') "
                "ORDER BY (status='active') DESC, quest_name "
                "LIMIT ?",
                (limit,),
            ).fetchall()
            for r in rows:
                out.append(dict(r))
        except Exception:
            pass
        # Supplement with any in-memory active quests not already included.
        seen_ids = {q.get("quest_id") or q.get("quest_name") for q in out}
        for _bot_quests in self._active_quests.values():
            for _q in _bot_quests.values():
                _key = _q.quest_id or _q.quest_name
                if _key and _key not in seen_ids:
                    out.append(
                        {
                            "quest_id": _q.quest_id or "",
                            "quest_name": _q.quest_name or "",
                            "status": getattr(_q, "status", "active"),
                            "map_name": getattr(_q, "map_name", "") or "",
                        }
                    )
                    seen_ids.add(_key)
                    if len(out) >= limit:
                        return out
        return out[:limit]

    def parse_quest_info(self, signals: dict[str, Any], bot_id: str) -> None:
        """Parse quest info from signals and update tracking.

        Checks for quest-related signal keys like quest_window, active_quests, etc.
        """
        active_list = signals.get("active_quests", []) or []
        quest_window = signals.get("quest_window", {}) or {}

        if active_list:
            for entry in active_list:
                qid = entry.get("id", "") or entry.get("name", "")
                if qid and qid not in self._active_quests.get(bot_id, {}):
                    qs = QuestState(
                        quest_id=qid,
                        quest_name=qid,
                        status="active",
                        started_at=time.time(),
                        map_name=str(signals.get("map", "") or "").replace(".gat", ""),
                    )
                    self.track_quest(bot_id, qs)

    def get_quests_near_completion(self, bot_id: str) -> list[QuestState]:
        """Get quests where most objectives are done."""
        nearly_done: list[QuestState] = []
        for qs in self.get_active_quests(bot_id):
            if not qs.objectives:
                continue
            done = sum(1 for o in qs.objectives if o.completed)
            if done >= len(qs.objectives) - 1 and done < len(qs.objectives):
                nearly_done.append(qs)
        return nearly_done

    def load_from_db(self, bot_id: str) -> list[QuestState]:
        """Load quest tracking from DB for a bot."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM quest_tracking WHERE bot_id=? ORDER BY started_at DESC",
            (bot_id,),
        )
        loaded: list[QuestState] = []
        for row in cur.fetchall():
            row = dict(row)
            objectives_data = json.loads(row.get("objectives", "[]") or "[]")
            objectives = [
                QuestObjective(
                    description=o.get("description", ""),
                    target=o.get("target", ""),
                    target_type=o.get("target_type", ""),
                    current=o.get("current", 0),
                    required=o.get("required", 1),
                    completed=o.get("completed", False),
                )
                for o in objectives_data
            ]
            qs = QuestState(
                quest_id=row.get("quest_id", ""),
                quest_name=row.get("quest_name", ""),
                status=row.get("status", "inactive"),
                objectives=objectives,
                npc_start=row.get("npc_start", ""),
                npc_complete=row.get("npc_complete", ""),
                rewards=json.loads(row.get("rewards", "{}") or "{}"),
                started_at=row.get("started_at", 0),
                completed_at=row.get("completed_at", 0),
                expires_at=row.get("expires_at", 0),
                map_name=row.get("map_name", ""),
            )
            loaded.append(qs)
            self._active_quests.setdefault(bot_id, {})[qs.quest_id or qs.quest_name] = qs
        return loaded

    def cleanup_bot(self, bot_id: str) -> None:
        """Remove quest tracking for a bot."""
        self._active_quests.pop(bot_id, None)
