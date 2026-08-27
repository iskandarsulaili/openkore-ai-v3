"""CharStatusReader — reads the durable real-time charstatus.json contract.

The bridge (aiSidecarBridge.pl) writes a complete, enriched char+world state to
``data/charstatus/charstatus_<bot>.json`` on every snapshot tick (atomic
temp+rename, monotonic ``seq``). This reader gives the Conscious (LLM),
Subconscious (ML) and Reflex brains a single authoritative view of a bot's
full state — identity, vitals, position, inventory, stats/skills, combat,
environment, party, economy, AI internals and telemetry.

The file is the durable record; the in-memory ``SnapshotCache`` holds the
last POSTed snapshot. This reader prefers the durable file (it carries the
full enriched contract the bridge computed) and falls back to the cache.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import RLock

logger = logging.getLogger("ai_sidecar.runtime.charstatus")


class CharStatusReader:
    """Reads the durable charstatus.json files written by the bridge."""

    def __init__(self, data_dir: Path) -> None:
        self._dir = Path(data_dir) / "charstatus"
        self._lock = RLock()
        self._cache: dict[str, dict] = {}
        self._mtime: dict[str, float] = {}

    def _path_for(self, bot_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in bot_id)
        safe = safe or "unknown"
        return self._dir / f"charstatus_{safe}.json"

    def get(self, bot_id: str, *, max_age_s: float = 30.0) -> dict | None:
        """Return the latest charstatus contract for a bot, or None.

        Reads the durable file, re-reading only when its mtime changed
        (cheap poll). Rejects files older than ``max_age_s`` (stale guard —
        a bot that went offline leaves a frozen file; callers must not act
        on it).
        """
        path = self._path_for(bot_id)
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            return None
        now = __import__("time").time()
        if now - mtime > max_age_s:
            return None
        with self._lock:
            if self._mtime.get(bot_id) == mtime and bot_id in self._cache:
                return self._cache[bot_id]
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("charstatus_read_failed bot=%s err=%s", bot_id, exc)
                return None
            self._cache[bot_id] = data
            self._mtime[bot_id] = mtime
            return data

    def list_bots(self) -> list[str]:
        """Return bot ids that have a fresh charstatus file."""
        out: list[str] = []
        try:
            for p in self._dir.glob("charstatus_*.json"):
                bot = p.stem[len("charstatus_"):]
                if self.get(bot) is not None:
                    out.append(bot)
        except OSError:
            return out
        return out
