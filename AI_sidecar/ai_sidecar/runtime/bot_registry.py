from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from threading import RLock


@dataclass(slots=True)
class BotRecord:
    bot_id: str
    first_seen_at: datetime
    last_seen_at: datetime
    last_tick_id: str | None = None


class BotRegistry:
    def __init__(self) -> None:
        self._lock = RLock()
        self._bots: dict[str, BotRecord] = {}

    def upsert(self, bot_id: str, tick_id: str | None = None) -> BotRecord:
        now = datetime.now(UTC)
        with self._lock:
            existing = self._bots.get(bot_id)
            if existing is None:
                existing = BotRecord(
                    bot_id=bot_id,
                    first_seen_at=now,
                    last_seen_at=now,
                    last_tick_id=tick_id,
                )
                self._bots[bot_id] = existing
            else:
                existing.last_seen_at = now
                if tick_id is not None:
                    existing.last_tick_id = tick_id
            return existing

    def get(self, bot_id: str) -> BotRecord | None:
        with self._lock:
            return self._bots.get(bot_id)

    def count(self) -> int:
        with self._lock:
            return len(self._bots)

