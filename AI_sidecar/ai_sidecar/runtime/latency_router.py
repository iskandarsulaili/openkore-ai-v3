from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import RLock
from time import perf_counter
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(slots=True)
class LatencySample:
    route: str
    elapsed_ms: float


class LatencyRouter:
    def __init__(self, budget_ms: int) -> None:
        self._budget_ms = float(budget_ms)
        self._lock = RLock()
        self._samples: list[LatencySample] = []

    def begin(self) -> float:
        return perf_counter()

    def end(self, route: str, started_at: float) -> float:
        elapsed_ms = (perf_counter() - started_at) * 1000.0
        with self._lock:
            self._samples.append(LatencySample(route=route, elapsed_ms=elapsed_ms))
            if len(self._samples) > 5000:
                self._samples = self._samples[-2000:]
        return elapsed_ms

    def within_budget(self, elapsed_ms: float) -> bool:
        return elapsed_ms <= self._budget_ms

    def average_ms(self) -> float:
        with self._lock:
            if not self._samples:
                return 0.0
            return sum(sample.elapsed_ms for sample in self._samples) / len(self._samples)

    def run_with_budget(
        self,
        route: str,
        operation: Callable[[], T],
        fallback: Callable[[], T],
    ) -> tuple[T, float, bool]:
        started = self.begin()
        used_fallback = False
        try:
            result = operation()
        except Exception:
            logger.exception("latency_route_operation_failed", extra={"event": "latency_route_operation_failed"})
            result = fallback()
            used_fallback = True

        elapsed_ms = self.end(route=route, started_at=started)
        if not self.within_budget(elapsed_ms):
            logger.warning(
                "latency_budget_exceeded",
                extra={"event": "latency_budget_exceeded"},
            )
            result = fallback()
            used_fallback = True
        return result, elapsed_ms, used_fallback
