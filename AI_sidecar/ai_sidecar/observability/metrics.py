from __future__ import annotations

import logging
from threading import RLock

from ai_sidecar.contracts.telemetry import TelemetryEvent, TelemetryIngestResponse
from ai_sidecar.persistence.repositories import TelemetryRepository

logger = logging.getLogger(__name__)


class DurableTelemetryIngestor:
    def __init__(self, repository: TelemetryRepository, backlog_max_events: int) -> None:
        self._repository = repository
        self._backlog_max_events = backlog_max_events
        self._lock = RLock()
        self._backlog: list[tuple[str, TelemetryEvent]] = []

    def ingest(self, *, bot_id: str, events: list[TelemetryEvent]) -> TelemetryIngestResponse:
        if not events:
            return TelemetryIngestResponse(ok=True, accepted=0, dropped=0, queued_for_retry=0)

        with self._lock:
            accepted = 0
            dropped = 0
            queued_for_retry = 0

            pending: list[tuple[str, TelemetryEvent]] = list(self._backlog)
            pending.extend((bot_id, item) for item in events)
            self._backlog.clear()

            per_bot: dict[str, list[TelemetryEvent]] = {}
            for item_bot_id, item in pending:
                per_bot.setdefault(item_bot_id, []).append(item)

            try:
                for group_bot_id, batch in per_bot.items():
                    persisted, trimmed = self._repository.ingest(bot_id=group_bot_id, events=batch)
                    accepted += persisted
                    dropped += trimmed
            except Exception:
                logger.exception(
                    "telemetry_persist_failed",
                    extra={"event": "telemetry_persist_failed", "bot_id": bot_id},
                )
                self._backlog = pending[-self._backlog_max_events :]
                queued_for_retry = len(self._backlog)
                return TelemetryIngestResponse(
                    ok=True,
                    accepted=0,
                    dropped=0,
                    queued_for_retry=queued_for_retry,
                )

            overflow = max(0, len(pending) - accepted)
            if overflow > 0:
                self._backlog.extend(pending[-overflow:])
            if len(self._backlog) > self._backlog_max_events:
                self._backlog = self._backlog[-self._backlog_max_events :]
            queued_for_retry = len(self._backlog)

            return TelemetryIngestResponse(
                ok=True,
                accepted=accepted,
                dropped=dropped,
                queued_for_retry=queued_for_retry,
            )

    def backlog_size(self) -> int:
        with self._lock:
            return len(self._backlog)

