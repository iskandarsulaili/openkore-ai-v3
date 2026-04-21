from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from threading import RLock

from ai_sidecar.contracts.reflex import ReflexBreakerStatusView


@dataclass(slots=True)
class _BreakerConfig:
    failure_threshold: int
    open_seconds: int


@dataclass(slots=True)
class _BreakerState:
    key: str
    family: str
    state: str = "closed"
    failure_count: int = 0
    total_failures: int = 0
    total_successes: int = 0
    opened_until: datetime | None = None
    last_failure_reason: str = ""
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class ReflexCircuitBreaker:
    _FAMILIES: dict[str, _BreakerConfig] = {
        "provider": _BreakerConfig(failure_threshold=3, open_seconds=30),
        "macro": _BreakerConfig(failure_threshold=3, open_seconds=20),
        "combat": _BreakerConfig(failure_threshold=2, open_seconds=15),
        "social": _BreakerConfig(failure_threshold=1, open_seconds=60),
        "fleet": _BreakerConfig(failure_threshold=3, open_seconds=20),
        "queue": _BreakerConfig(failure_threshold=5, open_seconds=10),
    }

    def __init__(self) -> None:
        self._lock = RLock()
        self._states: dict[tuple[str, str], _BreakerState] = {}

    def allow(self, *, bot_id: str, key: str, family: str) -> tuple[bool, str]:
        now = datetime.now(UTC)
        with self._lock:
            state = self._states.get((bot_id, key))
            if state is None:
                return True, "closed"

            if state.state == "open":
                if state.opened_until is not None and now < state.opened_until:
                    remaining = int((state.opened_until - now).total_seconds() * 1000.0)
                    return False, f"breaker_open:{max(0, remaining)}"
                state.state = "half_open"
                state.updated_at = now
                self._states[(bot_id, key)] = state
                return True, "half_open_trial"

            return True, state.state

    def record_success(self, *, bot_id: str, key: str, family: str) -> None:
        now = datetime.now(UTC)
        with self._lock:
            state = self._states.get((bot_id, key))
            if state is None:
                state = _BreakerState(key=key, family=self._normalize_family(family), updated_at=now)
            state.total_successes += 1
            state.failure_count = 0
            state.state = "closed"
            state.opened_until = None
            state.updated_at = now
            self._states[(bot_id, key)] = state

    def record_failure(self, *, bot_id: str, key: str, family: str, reason: str) -> None:
        now = datetime.now(UTC)
        normalized_family = self._normalize_family(family)
        cfg = self._FAMILIES[normalized_family]

        with self._lock:
            state = self._states.get((bot_id, key))
            if state is None:
                state = _BreakerState(key=key, family=normalized_family, updated_at=now)
            state.family = normalized_family
            state.total_failures += 1
            state.failure_count += 1
            state.last_failure_reason = reason
            state.updated_at = now

            if state.failure_count >= cfg.failure_threshold:
                state.state = "open"
                state.opened_until = now + timedelta(seconds=cfg.open_seconds)
                state.failure_count = 0

            self._states[(bot_id, key)] = state

    def statuses(self, *, bot_id: str) -> list[ReflexBreakerStatusView]:
        with self._lock:
            now = datetime.now(UTC)
            self._ensure_family_defaults(bot_id=bot_id, now=now)
            rows = [state for (owner_bot, _), state in self._states.items() if owner_bot == bot_id]
            rows.sort(key=lambda item: (item.family, item.key))
            return [
                ReflexBreakerStatusView(
                    key=item.key,
                    family=item.family,
                    state=item.state,
                    failure_count=item.failure_count,
                    total_failures=item.total_failures,
                    total_successes=item.total_successes,
                    opened_until=item.opened_until,
                    last_failure_reason=item.last_failure_reason,
                    updated_at=item.updated_at,
                )
                for item in rows
            ]

    def ensure_bot(self, *, bot_id: str) -> None:
        with self._lock:
            self._ensure_family_defaults(bot_id=bot_id, now=datetime.now(UTC))

    def _ensure_family_defaults(self, *, bot_id: str, now: datetime) -> None:
        for family in self._FAMILIES:
            key = f"{family}.default"
            if (bot_id, key) in self._states:
                continue
            self._states[(bot_id, key)] = _BreakerState(key=key, family=family, updated_at=now)

    def _normalize_family(self, family: str) -> str:
        candidate = (family or "").strip().lower()
        if candidate in self._FAMILIES:
            return candidate
        return "queue"
