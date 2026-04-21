from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from threading import RLock
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request

TRACE_ID_HEADER = "X-Trace-Id"


def ensure_trace_id(value: str | None = None) -> str:
    text = (value or "").strip()
    if 8 <= len(text) <= 128:
        return text
    return uuid4().hex


@dataclass(slots=True)
class TraceEvent:
    trace_id: str
    timestamp: datetime
    name: str
    attributes: dict[str, object] = field(default_factory=dict)


class TraceStore:
    def __init__(self, *, max_traces: int = 5000, max_events_per_trace: int = 200) -> None:
        self._lock = RLock()
        self._max_traces = max(100, int(max_traces))
        self._max_events_per_trace = max(10, int(max_events_per_trace))
        self._order: deque[str] = deque(maxlen=self._max_traces)
        self._events: dict[str, deque[TraceEvent]] = {}

    def add_event(self, *, trace_id: str, name: str, attributes: dict[str, object] | None = None) -> None:
        record = TraceEvent(
            trace_id=ensure_trace_id(trace_id),
            timestamp=datetime.now(UTC),
            name=name,
            attributes=dict(attributes or {}),
        )
        with self._lock:
            bucket = self._events.get(record.trace_id)
            if bucket is None:
                if len(self._order) >= self._max_traces:
                    stale = self._order.popleft()
                    if stale in self._events:
                        del self._events[stale]
                self._order.append(record.trace_id)
                bucket = deque(maxlen=self._max_events_per_trace)
                self._events[record.trace_id] = bucket
            bucket.append(record)

    def get_trace(self, *, trace_id: str) -> list[dict[str, object]]:
        with self._lock:
            bucket = list(self._events.get(trace_id, ()))
        return [asdict(item) for item in bucket]

    def recent(self, *, limit: int = 50) -> list[dict[str, object]]:
        size = max(1, min(int(limit), 500))
        with self._lock:
            ids = list(self._order)[-size:]
        return [{"trace_id": item, "events": self.get_trace(trace_id=item)} for item in reversed(ids)]


def install_fastapi_tracing(app: FastAPI, *, trace_store: TraceStore | None = None) -> None:
    @app.middleware("http")
    async def _trace_middleware(request: Request, call_next):
        trace_id = ensure_trace_id(request.headers.get(TRACE_ID_HEADER) or request.headers.get("X-Request-Id"))
        request.state.trace_id = trace_id
        started = perf_counter()
        active_store = trace_store
        if active_store is None:
            runtime = getattr(request.app.state, "runtime", None)
            active_store = getattr(runtime, "trace_store", None)
        if active_store is not None:
            active_store.add_event(
                trace_id=trace_id,
                name="http.request.start",
                attributes={"method": request.method, "path": request.url.path},
            )

        response = await call_next(request)
        latency_ms = (perf_counter() - started) * 1000.0
        response.headers[TRACE_ID_HEADER] = trace_id
        if active_store is not None:
            active_store.add_event(
                trace_id=trace_id,
                name="http.request.finish",
                attributes={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "latency_ms": latency_ms,
                },
            )
        return response
