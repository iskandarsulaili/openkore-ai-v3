from __future__ import annotations

from dataclasses import dataclass

from ai_sidecar.contracts.events import NormalizedEvent
from ai_sidecar.persistence.repositories import EventJournalRepository


@dataclass(slots=True)
class EventJournal:
    repository: EventJournalRepository | None = None

    def append(self, event: NormalizedEvent) -> None:
        if self.repository is None:
            return
        self.repository.append(event)

    def list_recent(self, *, bot_id: str, limit: int) -> list[dict[str, object]]:
        if self.repository is None:
            return []
        rows = self.repository.list_recent(bot_id=bot_id, limit=limit)
        return [
            {
                "id": item.id,
                "event_id": item.event_id,
                "bot_id": item.bot_id,
                "observed_at": item.observed_at,
                "ingested_at": item.ingested_at,
                "event_family": item.event_family,
                "event_type": item.event_type,
                "severity": item.severity,
                "source_hook": item.source_hook,
                "correlation_id": item.correlation_id,
                "text": item.text,
                "tags": item.tags,
                "numeric": item.numeric,
                "payload": item.payload,
                "event": item.event,
            }
            for item in rows
        ]

    def count(self, *, bot_id: str | None = None) -> int:
        if self.repository is None:
            return 0
        return self.repository.count(bot_id=bot_id)

