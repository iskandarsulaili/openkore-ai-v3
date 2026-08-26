"""Self-Awareness Layer for the conscious tier.

Mirrors Hermes's memory architecture (reverse-engineered from hermes-agent
source) so the openkore-ai-v3 LLM brain self-learns / self-heals /
self-improves coherently across every reasoning call:

- SOUL.md : curated identity + decision doctrine (Hermes has no identity doc;
  this is OUR addition). Injected VERBATIM into every conscious-tier LLM call.
- MEMORY.md : CURATED durable lessons the conscious LLM writes when it decides
  something is worth remembering (Hermes pattern: the agent curates, it is not
  a DB dump). Char-bounded, injected verbatim each call.
- P2P CROWDSOURCE: every new lesson is pushed to a central HTTP sink (config
  memory.sink_endpoint, RAW-style). Bots pull shared lessons on boot and merge.
  A WebRTC mesh can later gossip lesson hashes instead.

The DB stores (long_term_memory / episodic / semantic) remain the structured
retrieval backend and coexist; MEMORY.md is the curated in-context layer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

# Character budget for MEMORY.md (Hermes MEMORY.md = 440_000; keep bot tighter).
MEMORY_CHAR_LIMIT = 100_000
ENTRY_DELIMITER = "\n\u00a7\n"  # Hermes uses \n\u00a7\n

# Header shown above the memory block in the injected system context.
MEMORY_BLOCK_HEADERS = {
    "soul": "SOUL — conscious identity & doctrine",
    "memory": "MEMORY — durable lessons (self-curated)",
}

_SEPARATOR = "\u2550" * 46


class MemorySink:
    """Central P2P-crowdsource sink for shared lessons (RAW-style HTTP).

    Later a WebRTC mesh can gossip lesson hashes directly between bots; for now
    a central HTTP endpoint is the source of truth (config memory.sink_endpoint).
    """

    def __init__(
        self,
        endpoint: str = "",
        *,
        token: str = "",
        enabled: bool = True,
        timeout: float = 8.0,
    ) -> None:
        self.endpoint = (endpoint or "").rstrip("/")
        self.token = token
        self.enabled = enabled and bool(self.endpoint) and requests is not None
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return self.enabled

    def push_lessons(self, lessons: list[dict[str, Any]]) -> bool:
        """Push new lessons to the central sink. Idempotent by lesson id."""
        if not self.enabled or not lessons:
            return True
        try:
            _resp = requests.post(
                self.endpoint + "/lessons",
                json={"lessons": lessons},
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
            if _resp.status_code in (200, 201):
                logger.info("memory_sink_push_ok count=%d", len(lessons))
                return True
            logger.warning(
                "memory_sink_push_failed status=%s body=%s",
                _resp.status_code,
                _resp.text[:200],
            )
            return False
        except Exception as e:  # pragma: no cover
            logger.warning("memory_sink_push_error %s", e)
            return False

    def pull_lessons(self) -> list[dict[str, Any]]:
        """Pull shared lessons from the central sink for cross-bot learning."""
        if not self.enabled:
            return []
        try:
            _resp = requests.get(
                self.endpoint + "/lessons",
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Accept": "application/json",
                },
                timeout=self.timeout,
            )
            if _resp.status_code == 200:
                data = _resp.json()
                return data.get("lessons", []) if isinstance(data, dict) else []
            logger.warning("memory_sink_pull_failed status=%s", _resp.status_code)
            return []
        except Exception as e:  # pragma: no cover
            logger.warning("memory_sink_pull_error %s", e)
            return []


class LessonsHub:
    """Local central sink for the fleet's shared durable lessons (SQLite).

    Serves as the "central sink now" — every bot on the same box pushes lessons
    here and pulls them back, so the fleet cross-improves even before the
    external RAW-hosted mesh/HTTP sink is live. The remote MemorySink remains
    available (config memory_sink_endpoint) for cross-box RAW-style telemetry.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS lessons (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    importance INTEGER NOT NULL DEFAULT 5,
                    source TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    def push(self, lessons: list[dict[str, Any]]) -> int:
        """Upsert lessons (idempotent by id). Returns number stored/updated."""
        if not lessons:
            return 0
        added = 0
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                for l in lessons:
                    content = (l.get("content") or "").strip()
                    if not content:
                        continue
                    lid = l.get("id") or _lesson_id(content)
                    now = int(time.time())
                    try:
                        conn.execute(
                            """
                            INSERT INTO lessons(id, content, importance, source, created_at, updated_at)
                            VALUES(?, ?, ?, ?, ?, ?)
                            ON CONFLICT(id) DO UPDATE SET
                              content=excluded.content,
                              importance=excluded.importance,
                              updated_at=excluded.updated_at
                            """,
                            (lid, content, int(l.get("importance", 5) or 5),
                             l.get("source", ""), now, now),
                        )
                    except sqlite3.Error as e:  # pragma: no cover
                        logger.warning("lessons_hub_push_err %s", e)
                conn.commit()
                added = conn.total_changes
        return added

    def pull(self, limit: int = 200) -> list[dict[str, Any]]:
        """Return shared lessons newest-first, capped."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, content, importance, source, created_at, updated_at "
                "FROM lessons ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0])


class SelfAwareness:
    """Wires SOUL.md + MEMORY.md into every conscious-tier LLM call."""

    def __init__(
        self,
        data_dir: Path,
        *,
        sink_endpoint: str = "",
        sink_token: str = "",
        sink_enabled: bool = False,
        memory_char_limit: int = MEMORY_CHAR_LIMIT,
        hub: Any | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.soul_path = self.data_dir / "SOUL.md"
        self.memory_path = self.data_dir / "MEMORY.md"
        self.memory_char_limit = memory_char_limit
        self._lock = RLock()

        self._soul: str = ""
        self._memory_entries: list[str] = []

        self.sink = MemorySink(
            endpoint=sink_endpoint,
            token=sink_token,
            enabled=sink_enabled,
        )
        # Local shared fleet hub (SQLite) — the working "central sink now".
        self.hub = hub

        self.load_from_disk()
        # Boot-time cross-bot learning: pull shared lessons from the hub.
        if self.hub is not None:
            try:
                n = self.pull_shared_from_hub()
                if n:
                    logger.info("self_awareness_hub_pulled merged=%d", n)
            except Exception as e:  # pragma: no cover
                logger.warning("self_awareness_hub_pull_failed %s", e)

    # ── Hub (fleet cross-learning) ────────────────────────────────────
    def push_to_hub(self, content: str) -> bool:
        """Push a single lesson to the shared fleet hub (idempotent)."""
        if self.hub is None:
            return False
        try:
            self.hub.push(
                [{"id": _lesson_id(content), "content": content, "importance": 5,
                  "source": "self_awareness"}]
            )
            return True
        except Exception as e:  # pragma: no cover
            logger.warning("self_awareness_hub_push_failed %s", e)
            return False

    def pull_shared_from_hub(self) -> int:
        """Merge hub lessons into local MEMORY.md (dedup by lesson id)."""
        if self.hub is None:
            return 0
        try:
            shared = self.hub.pull(limit=500)
        except Exception as e:  # pragma: no cover
            logger.warning("self_awareness_hub_pull_failed %s", e)
            return 0
        if not shared:
            return 0
        with self._lock:
            existing = set(_lesson_id(e) for e in self._memory_entries)
            added = 0
            for l in shared:
                content = (l.get("content") or "").strip()
                if not content or _lesson_id(content) in existing:
                    continue
                new_total = self.memory_char_count + len(content) + len(ENTRY_DELIMITER)
                if new_total > self.memory_char_limit:
                    continue
                self._memory_entries.append(content)
                existing.add(_lesson_id(content))
                added += 1
            if added:
                self.save_to_disk()
        return added

    # ── Disk ──────────────────────────────────────────────────────────
    def load_from_disk(self) -> None:
        with self._lock:
            # SOUL.md — curated identity.
            try:
                if self.soul_path.exists():
                    self._soul = self.soul_path.read_text(encoding="utf-8")
            except Exception as e:  # pragma: no cover
                logger.warning("soul_md_load_failed %s", e)
                self._soul = ""

            # MEMORY.md — entries split by \n§\n (Hermes format).
            self._memory_entries = []
            try:
                if self.memory_path.exists():
                    raw = self.memory_path.read_text(encoding="utf-8")
                    for ent in raw.split(ENTRY_DELIMITER):
                        ent = ent.strip()
                        if ent:
                            self._memory_entries.append(ent)
            except Exception as e:  # pragma: no cover
                logger.warning("memory_md_load_failed %s", e)
                self._memory_entries = []

    def save_to_disk(self) -> None:
        with self._lock:
            self.memory_path.write_text(
                ENTRY_DELIMITER.join(self._memory_entries), encoding="utf-8"
            )

    # ── Accessors ─────────────────────────────────────────────────────
    @property
    def soul(self) -> str:
        return self._soul

    @property
    def memory_entries(self) -> list[str]:
        return list(self._memory_entries)

    @property
    def memory_char_count(self) -> int:
        return len(ENTRY_DELIMITER.join(self._memory_entries))

    def inject(self, system_prompt: str = "") -> str:
        """Return the system prompt with SOUL + MEMORY blocks prepended.

        Matches Hermes: the blocks are rendered with separators + headers and
        prepended to whatever system prompt the caller already had. Called on
        every conscious-tier LLM call so the brain always sees its identity
        and accumulated lessons in-context.
        """
        parts: list[str] = []
        if self._soul:
            parts.append(
                f"{_SEPARATOR}\n{MEMORY_BLOCK_HEADERS['soul']}\n{_SEPARATOR}\n{self._soul}"
            )
        if self._memory_entries:
            pct = min(100, int((self.memory_char_count / self.memory_char_limit) * 100))
            parts.append(
                f"{_SEPARATOR}\n{MEMORY_BLOCK_HEADERS['memory']} [{pct}% — "
                f"{self.memory_char_count:,}/{self.memory_char_limit:,} chars]\n"
                f"{_SEPARATOR}\n{ENTRY_DELIMITER.join(self._memory_entries)}"
            )
        if system_prompt:
            parts.append(system_prompt)
        return "\n\n---\n\n".join(parts) if parts else ""

    # ── Memory ops (Hermes memory tool contract) ───────────────────────
    def add_lesson(self, content: str) -> dict[str, Any]:
        """Append a durable lesson to MEMORY.md. Returns success/error dict."""
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}
        with self._lock:
            # Re-read under lock to pick up other sessions' writes.
            self.load_from_disk()
            entries = self._memory_entries
            new_total = len(ENTRY_DELIMITER.join(entries + [content]))
            if new_total > self.memory_char_limit:
                return {
                    "success": False,
                    "error": (
                        f"Memory full: would be {new_total:,} / "
                        f"{self.memory_char_limit:,} chars. Consolidate/remove first."
                    ),
                }
            entries.append(content)
            self._memory_entries = entries
            self.save_to_disk()

        # Crowdsource: push the new lesson to the fleet hub + central sink.
        lesson = {
            "id": _lesson_id(content),
            "content": content,
            "importance": 5,
            "updated_at": time.time(),
        }
        if self.hub is not None:
            try:
                self.hub.push([lesson])
            except Exception:  # pragma: no cover
                pass
        if self.sink.available:
            self.sink.push_lessons([lesson])

        pct = min(100, int((self.memory_char_count / self.memory_char_limit) * 100))
        return {
            "success": True,
            "done": True,
            "usage": f"{pct}% — {self.memory_char_count:,}/{self.memory_char_limit:,} chars",
            "entry_count": len(self._memory_entries),
            "note": "Lesson saved. This update is complete — do not repeat it.",
        }

    def remove_lesson(self, index: int) -> dict[str, Any]:
        """Remove a lesson by its position in MEMORY.md (Hermes remove)."""
        with self._lock:
            self.load_from_disk()
            if not 0 <= index < len(self._memory_entries):
                return {"success": False, "error": f"No lesson at index {index}."}
            self._memory_entries.pop(index)
            self.save_to_disk()
        return {
            "success": True,
            "done": True,
            "entry_count": len(self._memory_entries),
        }

    def pull_shared_lessons(self) -> int:
        """Merge lessons from the central sink into MEMORY.md (dedup by id)."""
        if not self.sink.available:
            return 0
        shared = self.sink.pull_lessons()
        if not shared:
            return 0
        with self._lock:
            existing = set(_lesson_id(e) for e in self._memory_entries)
            added = 0
            for l in shared:
                content = (l.get("content") or "").strip()
                if not content or _lesson_id(content) in existing:
                    continue
                # Respect budget — don't push over the limit.
                new_total = self.memory_char_count + len(content) + len(ENTRY_DELIMITER)
                if new_total > self.memory_char_limit:
                    continue
                self._memory_entries.append(content)
                existing.add(_lesson_id(content))
                added += 1
            if added:
                self.save_to_disk()
                logger.info("memory_sink_pull_merged added=%d", added)
        return added


def _lesson_id(content: str) -> str:
    return hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]
