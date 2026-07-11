from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExperienceEntry:
    """A single experience datapoint for a bot in a specific context."""
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
    
    Stores outcomes of actions in various contexts (combat, economy, quest, etc.)
    so that when bot A learns something, bot B can benefit from that knowledge.
    All state is in-memory with optional SQLite persistence.
    """

    def __init__(self, max_entries: int = 50000):
        self._lock = RLock()
        self._max_entries = max_entries
        self._entries: list[ExperienceEntry] = []
        # Indexes for fast lookup
        self._by_context: dict[str, list[int]] = {}  # context_type -> indices
        self._by_map: dict[str, list[int]] = {}  # map_name -> indices
        self._by_monster: dict[str, list[int]] = {}  # monster_name -> indices
        self._by_role: dict[str, list[int]] = {}  # role -> indices

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
        """Rebuild all indexes after trimming."""
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
        """Query experience entries by context type, map, monster, or role."""
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
        """Return success rate [0.0, 1.0] for the given context."""
        with self._lock:
            entries = self.query(context_type=context_type, map_name=map_name,
                                 monster_name=monster_name, limit=1000)
            if not entries:
                return 0.5  # Unknown -> neutral
            if action:
                entries = [e for e in entries if e.action_taken == action]
                if not entries:
                    return 0.5
            successes = sum(1 for e in entries if e.success)
            return successes / len(entries)

    def best_action(self, *, context_type: str, map_name: str = "",
                    monster_name: str = "", min_samples: int = 3) -> tuple[str, float]:
        """Return the action with the highest success rate for the given context."""
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
        """Persist experience to SQLite."""
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
            for e in self._entries:  # persist ALL entries, not just last 1000
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
        """Load experience from SQLite on startup.
        
        This fixes the critical bug where ExpDB lost all knowledge on restart.
        """
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
