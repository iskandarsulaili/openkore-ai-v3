"""
Dynamic Portal Discovery — learns portal locations from exploration.

Instead of hardcoding all 132 portals, this module:
1. Records portal locations when the bot walks through them
2. Shares portal data across bots via shared_learning_db
3. Builds the portal map dynamically from exploration
4. Falls back to hardcoded portals when dynamic data is insufficient

This is how a pro player learns: by exploring and remembering.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from ai_sidecar.portal_knowledge import Portal, get_portal_knowledge
from ai_sidecar.learning.shared_learning_db import get_shared_learning_db

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DiscoveredPortal:
    """A portal discovered through exploration."""
    source_map: str
    source_x: int
    source_y: int
    target_map: str
    target_x: int = 0
    target_y: int = 0
    portal_type: str = "two_way"
    discovered_by: str = ""
    discovered_at: float = 0.0
    use_count: int = 0
    confirmed: bool = False  # confirmed by walking through both directions


class DynamicPortalDiscovery:
    """Learns portal locations from exploration.

    Thread-safe. Records portals as the bot walks through them,
    shares data across bots, and builds the portal graph dynamically.
    """

    def __init__(self, db_path: str = "") -> None:
        self._lock = RLock()
        self._discovered_portals: list[DiscoveredPortal] = []
        self._pending_portals: list[DiscoveredPortal] = []  # not yet added to portal_knowledge
        self._portal_knowledge = get_portal_knowledge()
        self._shared_db = get_shared_learning_db(db_path)

        # Track which maps we've explored
        self._explored_maps: set[str] = set()

        # Track portal entry/exit pairs for confirmation
        self._recent_entries: dict[str, tuple[str, int, int, float]] = {}
        # bot_id -> (map_name, x, y, timestamp)

        # Load previously discovered portals
        self._load_discovered()

    def _get_db_path(self) -> str:
        """Get the database path for storing discovered portals."""
        return self._shared_db._db_path  # type: ignore[arg-type]

    def _load_discovered(self) -> None:
        """Load previously discovered portals from the shared database."""
        try:
            conn = sqlite3.connect(self._get_db_path())
            try:
                cursor = conn.execute(
                    "SELECT source_map, source_x, source_y, target_map, target_x, target_y, "
                    "portal_type, discovered_by, discovered_at, use_count, confirmed "
                    "FROM discovered_portals ORDER BY use_count DESC"
                )
                for row in cursor.fetchall():
                    portal = DiscoveredPortal(
                        source_map=row[0], source_x=row[1], source_y=row[2],
                        target_map=row[3], target_x=row[4], target_y=row[5],
                        portal_type=row[6], discovered_by=row[7],
                        discovered_at=row[8], use_count=row[9],
                        confirmed=bool(row[10]),
                    )
                    self._discovered_portals.append(portal)
                    if portal.confirmed:
                        self._add_to_knowledge(portal)
                logger.info("dynamic_portal_loaded: %d discovered portals", len(self._discovered_portals))
                # Load explored maps so the unexplored-map scout survives restarts
                _explored_rows = conn.execute(
                    "SELECT map_name FROM explored_maps ORDER BY first_explored_at"
                ).fetchall()
                for (_em,) in _explored_rows:
                    self._explored_maps.add(str(_em))
                if _explored_rows:
                    logger.info("dynamic_explored_loaded: %d explored maps", len(_explored_rows))
            finally:
                conn.close()
        except (sqlite3.OperationalError, Exception) as e:
            logger.debug("dynamic_portal_load_skipped: %s", e)

    def _ensure_schema(self) -> None:
        """Ensure the discovered_portals table exists."""
        try:
            conn = sqlite3.connect(self._get_db_path())
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS discovered_portals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_map TEXT NOT NULL,
                        source_x INTEGER NOT NULL,
                        source_y INTEGER NOT NULL,
                        target_map TEXT NOT NULL,
                        target_x INTEGER NOT NULL DEFAULT 0,
                        target_y INTEGER NOT NULL DEFAULT 0,
                        portal_type TEXT NOT NULL DEFAULT 'two_way',
                        discovered_by TEXT NOT NULL DEFAULT '',
                        discovered_at REAL NOT NULL,
                        use_count INTEGER NOT NULL DEFAULT 1,
                        confirmed INTEGER NOT NULL DEFAULT 0,
                        UNIQUE(source_map, source_x, source_y, target_map)
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_dp_source ON discovered_portals(source_map)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_dp_target ON discovered_portals(target_map)
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS explored_maps (
                        map_name TEXT PRIMARY KEY,
                        first_explored_at REAL NOT NULL,
                        last_seen_at REAL NOT NULL,
                        explored_by TEXT NOT NULL DEFAULT ''
                    )
                """)
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.warning("dynamic_portal_schema_error: %s", e)

    def _add_to_knowledge(self, portal: DiscoveredPortal) -> None:
        """Add a discovered portal to the main portal knowledge."""
        p = Portal(
            source_map=portal.source_map,
            source_x=portal.source_x,
            source_y=portal.source_y,
            target_map=portal.target_map,
            target_x=portal.target_x,
            target_y=portal.target_y,
            portal_type=portal.portal_type,
            name=f"Discovered: {portal.source_map}→{portal.target_map}",
        )
        self._portal_knowledge.add_portal(p)

    # ── Portal Recording ────────────────────────────────────────────

    def record_portal_entry(self, bot_id: str, map_name: str, x: int, y: int) -> None:
        """Record that a bot entered a portal at a specific location.

        This is called when the bot walks onto a portal cell.
        We save the entry point and wait for the exit to confirm.
        """
        if not map_name:
            return
        with self._lock:
            self._recent_entries[bot_id] = (map_name, x, y, time.time())

    def record_portal_exit(self, bot_id: str, map_name: str, x: int, y: int) -> None:
        """Record that a bot exited a portal onto a new map.

        This confirms a portal connection. We pair it with the last entry
        to determine the portal's target.
        """
        if not map_name:
            # Discard the pending entry: bot left to nowhere (char-select,
            # disconnect) — this is NOT a portal transition.
            self._recent_entries.pop(bot_id, None)
            return
        with self._lock:
            entry = self._recent_entries.pop(bot_id, None)
            if entry is None:
                # No entry recorded — this might be a spawn or warp
                return

            source_map, source_x, source_y, entry_time = entry
            target_map = map_name
            target_x = x
            target_y = y

            # Don't record if it's the same map (not a real portal)
            if source_map == target_map:
                return

            # Check if we already know this portal
            for dp in self._discovered_portals:
                if (dp.source_map == source_map and dp.source_x == source_x
                        and dp.source_y == source_y and dp.target_map == target_map):
                    dp.use_count += 1
                    dp.confirmed = True
                    self._save_portal(dp)
                    return

            # New portal discovery!
            portal = DiscoveredPortal(
                source_map=source_map,
                source_x=source_x,
                source_y=source_y,
                target_map=target_map,
                target_x=target_x,
                target_y=target_y,
                portal_type="two_way",
                discovered_by=bot_id,
                discovered_at=time.time(),
                use_count=1,
                confirmed=True,
            )

            self._discovered_portals.append(portal)
            self._add_to_knowledge(portal)
            self._save_portal(portal)

            logger.info(
                "dynamic_portal_discovered: %s→%s at (%d,%d) by %s",
                source_map, target_map, source_x, source_y, bot_id,
            )

    def record_map_visit(self, bot_id: str, map_name: str) -> None:
        """Record that a bot visited a map (persisted for restart survival)."""
        if not map_name:
            return
        with self._lock:
            if map_name not in self._explored_maps:
                self._explored_maps.add(map_name)
                logger.info("dynamic_map_explored: %s by %s", map_name, bot_id)
                try:
                    self._ensure_schema()
                    conn = sqlite3.connect(self._get_db_path())
                    try:
                        conn.execute(
                            "INSERT OR REPLACE INTO explored_maps (map_name, first_explored_at, last_seen_at, explored_by) "
                            "VALUES (?, ?, ?, ?)",
                            (map_name, time.time(), time.time(), bot_id),
                        )
                        conn.commit()
                    finally:
                        conn.close()
                except Exception as e:
                    logger.warning("dynamic_map_explored_save_error: %s", e)
            else:
                # Refresh last_seen for existing maps (cheap, keeps recency signal)
                try:
                    conn = sqlite3.connect(self._get_db_path())
                    try:
                        conn.execute(
                            "UPDATE explored_maps SET last_seen_at = ? WHERE map_name = ?",
                            (time.time(), map_name),
                        )
                        conn.commit()
                    finally:
                        conn.close()
                except Exception:
                    pass

    def _save_portal(self, portal: DiscoveredPortal) -> None:
        """Save a discovered portal to the shared database."""
        try:
            self._ensure_schema()
            conn = sqlite3.connect(self._get_db_path())
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO discovered_portals "
                    "(source_map, source_x, source_y, target_map, target_x, target_y, "
                    "portal_type, discovered_by, discovered_at, use_count, confirmed) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (portal.source_map, portal.source_x, portal.source_y,
                     portal.target_map, portal.target_x, portal.target_y,
                     portal.portal_type, portal.discovered_by,
                     portal.discovered_at, portal.use_count,
                     1 if portal.confirmed else 0),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.warning("dynamic_portal_save_error: %s", e)

    # ── Query API ────────────────────────────────────────────────────

    def get_discovered_portals(self) -> list[DiscoveredPortal]:
        """Get all discovered portals."""
        with self._lock:
            return list(self._discovered_portals)

    def get_discovered_portals_from(self, map_name: str) -> list[DiscoveredPortal]:
        """Get discovered portals leaving a map."""
        with self._lock:
            return [p for p in self._discovered_portals if p.source_map == map_name]

    def get_discovered_portals_to(self, map_name: str) -> list[DiscoveredPortal]:
        """Get discovered portals entering a map."""
        with self._lock:
            return [p for p in self._discovered_portals if p.target_map == map_name]

    def get_explored_maps(self) -> set[str]:
        """Get all maps that have been explored."""
        with self._lock:
            return set(self._explored_maps)

    def is_map_explored(self, map_name: str) -> bool:
        """Check if a map has been explored."""
        with self._lock:
            return map_name in self._explored_maps

    def get_unexplored_maps(self) -> list[str]:
        """Get maps that are in the hardcoded portal knowledge but not yet explored."""
        all_known = self._portal_knowledge.get_all_maps()
        with self._lock:
            return sorted(all_known - self._explored_maps)

    def get_discovery_count(self) -> int:
        """Get the number of portals discovered through exploration."""
        with self._lock:
            return len(self._discovered_portals)

    def get_confirmed_count(self) -> int:
        """Get the number of confirmed (bidirectional) portal discoveries."""
        with self._lock:
            return sum(1 for p in self._discovered_portals if p.confirmed)

    def get_status_summary(self) -> str:
        """Get a human-readable status summary."""
        with self._lock:
            lines = [
                f"── Dynamic Portal Discovery ──",
                f"Discovered portals: {len(self._discovered_portals)}",
                f"Confirmed portals: {self.get_confirmed_count()}",
                f"Explored maps: {len(self._explored_maps)}",
                f"Unexplored maps (known): {len(self.get_unexplored_maps())}",
            ]
            return "\n".join(lines)


# ── Global Singleton ──

_dynamic_portal_discovery: DynamicPortalDiscovery | None = None
_dynamic_portal_discovery_lock = RLock()


def get_dynamic_portal_discovery(db_path: str = "") -> DynamicPortalDiscovery:
    global _dynamic_portal_discovery
    with _dynamic_portal_discovery_lock:
        if _dynamic_portal_discovery is None:
            _dynamic_portal_discovery = DynamicPortalDiscovery(db_path=db_path)
        return _dynamic_portal_discovery
