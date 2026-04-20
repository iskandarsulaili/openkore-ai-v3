from __future__ import annotations

import logging
from threading import RLock

from ai_sidecar.persistence.repositories import AuditRepository

logger = logging.getLogger(__name__)


class AuditTrail:
    def __init__(self, repository: AuditRepository) -> None:
        self._repository = repository
        self._lock = RLock()

    def record(
        self,
        *,
        level: str,
        event_type: str,
        summary: str,
        bot_id: str | None,
        payload: dict[str, object] | None = None,
    ) -> None:
        body = dict(payload or {})
        try:
            with self._lock:
                self._repository.record(
                    level=level,
                    event_type=event_type,
                    summary=summary,
                    bot_id=bot_id,
                    payload=body,
                )
        except Exception:
            logger.exception(
                "audit_record_failed",
                extra={"event": "audit_record_failed", "event_type": event_type, "bot_id": bot_id},
            )

    def recent(
        self,
        *,
        limit: int,
        bot_id: str | None = None,
        event_type: str | None = None,
    ) -> list[dict[str, object]]:
        try:
            with self._lock:
                records = self._repository.recent(limit=limit, bot_id=bot_id, event_type=event_type)
        except Exception:
            logger.exception(
                "audit_recent_failed",
                extra={"event": "audit_recent_failed", "event_type": event_type, "bot_id": bot_id},
            )
            return []

        return [
            {
                "id": item.id,
                "timestamp": item.timestamp,
                "level": item.level,
                "event_type": item.event_type,
                "bot_id": item.bot_id,
                "summary": item.summary,
                "payload": item.payload,
            }
            for item in records
        ]

