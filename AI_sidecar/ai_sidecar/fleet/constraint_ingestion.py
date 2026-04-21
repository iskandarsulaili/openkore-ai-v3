from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from threading import RLock

from ai_sidecar.config import settings


@dataclass(slots=True)
class ConstraintIngestionState:
    _lock: RLock = field(default_factory=RLock)
    _last_sync_at: datetime | None = None
    _central_available: bool = False
    _doctrine_version: str = "local"
    _constraints_by_bot: dict[str, dict[str, object]] = field(default_factory=dict)
    _blackboard: dict[str, object] = field(default_factory=dict)
    _last_error: str = ""

    def update_from_blackboard(self, *, blackboard: dict[str, object]) -> None:
        now = datetime.now(UTC)
        constraints = blackboard.get("constraints") if isinstance(blackboard.get("constraints"), dict) else {}
        doctrine = blackboard.get("doctrine") if isinstance(blackboard.get("doctrine"), dict) else {}
        doctrine_version = str(doctrine.get("version") or "unknown")
        with self._lock:
            self._last_sync_at = now
            self._central_available = True
            self._doctrine_version = doctrine_version
            self._constraints_by_bot = {str(k): dict(v) for k, v in constraints.items() if isinstance(v, dict)}
            self._blackboard = dict(blackboard)
            self._last_error = ""

    def mark_unavailable(self, *, reason: str) -> None:
        with self._lock:
            self._central_available = False
            self._last_error = reason

    def status(self) -> dict[str, object]:
        with self._lock:
            now = datetime.now(UTC)
            stale = True
            if self._last_sync_at is not None:
                stale = now - self._last_sync_at > timedelta(seconds=settings.fleet_local_partition_ttl_seconds)
            mode = "central" if self._central_available and not stale else "local"
            return {
                "mode": mode,
                "central_available": self._central_available,
                "stale": stale,
                "last_sync_at": self._last_sync_at,
                "doctrine_version": self._doctrine_version,
                "last_error": self._last_error,
            }

    def constraints_for_bot(self, *, bot_id: str) -> dict[str, object]:
        with self._lock:
            default = {"avoid": [], "required": [], "sources": ["local_default"]}
            row = self._constraints_by_bot.get(bot_id)
            if row is None:
                return default
            merged = dict(default)
            merged.update(row)
            return merged

    def blackboard(self) -> dict[str, object]:
        with self._lock:
            return dict(self._blackboard)

