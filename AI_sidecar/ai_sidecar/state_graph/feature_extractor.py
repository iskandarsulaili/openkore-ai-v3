from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from threading import RLock

from ai_sidecar.contracts.events import EventFamily, NormalizedEvent
from ai_sidecar.contracts.state_graph import LearningFeatureVector


@dataclass(slots=True)
class _FeatureWindow:
    chat_events: deque[datetime] = field(default_factory=deque)
    warnings: deque[datetime] = field(default_factory=deque)
    errors: deque[datetime] = field(default_factory=deque)
    zeny_points: deque[tuple[datetime, int]] = field(default_factory=deque)
    exp_points: deque[tuple[datetime, int]] = field(default_factory=deque)
    latest_values: dict[str, float] = field(default_factory=dict)


class FeatureExtractor:
    def __init__(self) -> None:
        self._lock = RLock()
        self._by_bot: dict[str, _FeatureWindow] = {}

    def observe_event(self, event: NormalizedEvent) -> None:
        now = self._normalize_dt(event.observed_at)
        with self._lock:
            window = self._by_bot.setdefault(event.meta.bot_id, _FeatureWindow())

            if event.event_family == EventFamily.chat:
                window.chat_events.append(now)

            if event.severity.value == "warning":
                window.warnings.append(now)
            elif event.severity.value in {"error", "critical"}:
                window.errors.append(now)

            zeny_value = event.payload.get("zeny")
            if isinstance(zeny_value, int):
                window.zeny_points.append((now, zeny_value))
            elif isinstance(event.numeric.get("zeny"), float):
                window.zeny_points.append((now, int(event.numeric.get("zeny") or 0.0)))

            exp_value = event.payload.get("base_exp")
            if not isinstance(exp_value, int):
                job_exp = event.payload.get("job_exp")
                if isinstance(job_exp, int):
                    exp_value = job_exp
                else:
                    numeric_base = event.numeric.get("base_exp")
                    numeric_job = event.numeric.get("job_exp")
                    if isinstance(numeric_base, float):
                        exp_value = int(numeric_base)
                    elif isinstance(numeric_job, float):
                        exp_value = int(numeric_job)
            if isinstance(exp_value, int):
                window.exp_points.append((now, exp_value))

            for key, value in event.numeric.items():
                window.latest_values[f"event.{event.event_type}.{key}"] = float(value)

            self._trim(window, now)

    def extract(
        self,
        *,
        bot_id: str,
        basis: dict[str, float],
        labels: dict[str, str] | None = None,
        raw: dict[str, object] | None = None,
    ) -> LearningFeatureVector:
        now = datetime.now(UTC)
        with self._lock:
            window = self._by_bot.setdefault(bot_id, _FeatureWindow())
            self._trim(window, now)

            features = dict(basis)
            features["chat.events_5m"] = float(len(window.chat_events))
            features["risk.warnings_5m"] = float(len(window.warnings))
            features["risk.errors_5m"] = float(len(window.errors))

            zeny_delta_10m = 0.0
            if len(window.zeny_points) >= 2:
                zeny_delta_10m = float(window.zeny_points[-1][1] - window.zeny_points[0][1])
            features["economy.zeny_delta_10m"] = zeny_delta_10m

            exp_delta_10m = 0.0
            if len(window.exp_points) >= 2:
                exp_delta_10m = float(window.exp_points[-1][1] - window.exp_points[0][1])
            features["economy.exp_delta_10m"] = exp_delta_10m

            for key, value in window.latest_values.items():
                features[key] = value

            return LearningFeatureVector(
                feature_version="v1",
                observed_at=now,
                values=features,
                labels=dict(labels or {}),
                raw=dict(raw or {}),
            )

    def _trim(self, window: _FeatureWindow, now: datetime) -> None:
        self._trim_deque(window.chat_events, now, timedelta(minutes=5))
        self._trim_deque(window.warnings, now, timedelta(minutes=5))
        self._trim_deque(window.errors, now, timedelta(minutes=5))

        min_zeny_ts = now - timedelta(minutes=10)
        while window.zeny_points and window.zeny_points[0][0] < min_zeny_ts:
            window.zeny_points.popleft()

        min_exp_ts = now - timedelta(minutes=10)
        while window.exp_points and window.exp_points[0][0] < min_exp_ts:
            window.exp_points.popleft()

    def _trim_deque(self, bucket: deque[datetime], now: datetime, keep: timedelta) -> None:
        min_ts = now - keep
        while bucket and bucket[0] < min_ts:
            bucket.popleft()

    def _normalize_dt(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
