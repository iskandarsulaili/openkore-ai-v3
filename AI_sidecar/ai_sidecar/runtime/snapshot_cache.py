from __future__ import annotations

from datetime import UTC, datetime, timedelta
from threading import RLock

from ai_sidecar.contracts.state import BotStateSnapshot


class SnapshotCache:
    def __init__(self, ttl_seconds: int) -> None:
        self._ttl = timedelta(seconds=ttl_seconds)
        self._lock = RLock()
        self._snapshots: dict[str, BotStateSnapshot] = {}

    def set(self, snapshot: BotStateSnapshot) -> None:
        with self._lock:
            self._snapshots[snapshot.meta.bot_id] = snapshot

    def get(self, bot_id: str) -> BotStateSnapshot | None:
        now = datetime.now(UTC)
        with self._lock:
            snapshot = self._snapshots.get(bot_id)
            if snapshot is None:
                return None
            if now - snapshot.observed_at > self._ttl:
                del self._snapshots[bot_id]
                return None
            return snapshot

    def count(self) -> int:
        self._expire_all()
        with self._lock:
            return len(self._snapshots)

    def latest(self) -> BotStateSnapshot | None:
        """Return the most recently updated snapshot across all bots."""
        self._expire_all()
        with self._lock:
            best: BotStateSnapshot | None = None
            for s in self._snapshots.values():
                if best is None or s.observed_at > best.observed_at:
                    best = s
            return best

    def bot_ids(self) -> list[str]:
        """Return all bot IDs with valid (non-expired) snapshots."""
        self._expire_all()
        with self._lock:
            return list(self._snapshots.keys())

    def _expire_all(self) -> None:
        now = datetime.now(UTC)
        with self._lock:
            stale = [
                bot_id
                for bot_id, snapshot in self._snapshots.items()
                if now - snapshot.observed_at > self._ttl
            ]
            for bot_id in stale:
                del self._snapshots[bot_id]

