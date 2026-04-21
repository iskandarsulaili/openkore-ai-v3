from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from threading import RLock


@dataclass(slots=True)
class ExplainabilityRecord:
    timestamp: datetime
    kind: str
    bot_id: str
    trace_id: str
    summary: str
    details: dict[str, object] = field(default_factory=dict)


class ExplainabilityStore:
    def __init__(self, *, max_records: int = 10000) -> None:
        self._lock = RLock()
        self._max_records = max(500, int(max_records))
        self._records: list[ExplainabilityRecord] = []

    def add(
        self,
        *,
        kind: str,
        bot_id: str,
        trace_id: str,
        summary: str,
        details: dict[str, object] | None = None,
    ) -> None:
        row = ExplainabilityRecord(
            timestamp=datetime.now(UTC),
            kind=str(kind or "unknown"),
            bot_id=str(bot_id or ""),
            trace_id=str(trace_id or ""),
            summary=str(summary or ""),
            details=dict(details or {}),
        )
        with self._lock:
            self._records.append(row)
            if len(self._records) > self._max_records:
                self._records = self._records[-self._max_records :]

    def list(
        self,
        *,
        kind: str | None = None,
        bot_id: str | None = None,
        trace_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        with self._lock:
            rows = list(self._records)
        if kind:
            rows = [item for item in rows if item.kind == kind]
        if bot_id:
            rows = [item for item in rows if item.bot_id == bot_id]
        if trace_id:
            rows = [item for item in rows if item.trace_id == trace_id]
        rows = rows[-max(1, min(int(limit), 2000)) :]
        return [asdict(item) for item in reversed(rows)]

