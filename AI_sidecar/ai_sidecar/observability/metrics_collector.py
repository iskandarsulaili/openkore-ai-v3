from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock


def _esc(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _labels(parts: dict[str, str]) -> str:
    if not parts:
        return ""
    body = ",".join(f'{key}="{_esc(str(val))}"' for key, val in sorted(parts.items()))
    return "{" + body + "}"


@dataclass(slots=True)
class _Histogram:
    bounds: list[float]
    counts: list[int]
    total: int = 0
    total_sum: float = 0.0

    @classmethod
    def build(cls, bounds: list[float]) -> _Histogram:
        ordered = sorted(float(item) for item in bounds)
        return cls(bounds=ordered, counts=[0 for _ in ordered])

    def observe(self, value: float) -> None:
        value = max(0.0, float(value))
        self.total += 1
        self.total_sum += value
        for idx, upper in enumerate(self.bounds):
            if value <= upper:
                self.counts[idx] += 1


@dataclass(slots=True)
class _EconomyPoint:
    observed_at: datetime
    zeny: float
    exp: float


class SLOMetricsCollector:
    def __init__(self) -> None:
        self._lock = RLock()
        self._started_at = datetime.now(UTC)

        self._latency: dict[str, _Histogram] = {}
        self._latency_buckets = [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]

        self._queue_backlog_by_tier: dict[str, int] = {}
        self._queue_decisions: dict[tuple[str, str, str], int] = {}
        self._ack_success: dict[tuple[str, str, str], int] = {}
        self._ack_total: dict[tuple[str, str], int] = {}

        self._death_rate: dict[tuple[str, str], int] = {}
        self._macro_success: dict[str, dict[str, int]] = {}
        self._shadow_disagreement: dict[str, dict[str, float]] = {}
        self._provider_route: dict[tuple[str, str, str], int] = {}
        self._breaker_events: dict[tuple[str, str, str], int] = {}

        self._economy_last: dict[tuple[str, str], _EconomyPoint] = {}
        self._economy_rates: dict[str, dict[str, float]] = {}

    def observe_latency(self, *, domain: str, elapsed_ms: float) -> None:
        key = (domain or "unknown").strip().lower()
        with self._lock:
            hist = self._latency.get(key)
            if hist is None:
                hist = _Histogram.build(self._latency_buckets)
                self._latency[key] = hist
            hist.observe(elapsed_ms)

    def set_queue_backlog(self, *, tier: str, depth: int) -> None:
        with self._lock:
            self._queue_backlog_by_tier[(tier or "unknown").strip().lower()] = max(0, int(depth))

    def record_queue_decision(self, *, tier: str, status: str, reason: str) -> None:
        key = ((tier or "unknown").strip().lower(), (status or "unknown").strip().lower(), (reason or "").strip().lower())
        with self._lock:
            self._queue_decisions[key] = self._queue_decisions.get(key, 0) + 1

    def record_ack(self, *, source: str, action_kind: str, success: bool) -> None:
        src = (source or "unknown").strip().lower()
        kind = (action_kind or "unknown").strip().lower()
        with self._lock:
            total_key = (src, kind)
            self._ack_total[total_key] = self._ack_total.get(total_key, 0) + 1
            ok_key = (src, kind, "success" if success else "failure")
            self._ack_success[ok_key] = self._ack_success.get(ok_key, 0) + 1

    def record_death(self, *, map_name: str, doctrine_version: str) -> None:
        key = ((map_name or "unknown").strip().lower(), (doctrine_version or "unknown").strip().lower())
        with self._lock:
            self._death_rate[key] = self._death_rate.get(key, 0) + 1

    def record_macro_publish(self, *, version: str, success: bool) -> None:
        ver = (version or "unknown").strip().lower()
        with self._lock:
            row = self._macro_success.setdefault(ver, {"ok": 0, "total": 0})
            row["total"] += 1
            if success:
                row["ok"] += 1

    def record_shadow(self, *, family: str, matched: bool, confidence: float) -> None:
        key = (family or "unknown").strip().lower()
        with self._lock:
            row = self._shadow_disagreement.setdefault(
                key,
                {"total": 0.0, "disagree": 0.0, "high_conf_disagree": 0.0},
            )
            row["total"] += 1.0
            if not matched:
                row["disagree"] += 1.0
                if float(confidence) >= 0.75:
                    row["high_conf_disagree"] += 1.0

    def record_economy(
        self,
        *,
        bot_id: str,
        plan_family: str,
        zeny: float,
        exp_value: float,
        observed_at: datetime | None = None,
    ) -> None:
        when = observed_at if isinstance(observed_at, datetime) else datetime.now(UTC)
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        else:
            when = when.astimezone(UTC)

        family = (plan_family or "unknown").strip().lower()
        key = ((bot_id or "fleet").strip().lower(), family)
        point = _EconomyPoint(observed_at=when, zeny=float(zeny), exp=float(exp_value))
        with self._lock:
            previous = self._economy_last.get(key)
            self._economy_last[key] = point
            if previous is None:
                return

            elapsed_hours = max(1e-6, (point.observed_at - previous.observed_at).total_seconds() / 3600.0)
            zeny_per_hour = max(0.0, (point.zeny - previous.zeny) / elapsed_hours)
            exp_per_hour = max(0.0, (point.exp - previous.exp) / elapsed_hours)

            fam = self._economy_rates.setdefault(family, {"zeny_per_hour": 0.0, "exp_per_hour": 0.0, "samples": 0.0})
            samples = fam["samples"] + 1.0
            fam["zeny_per_hour"] = ((fam["zeny_per_hour"] * fam["samples"]) + zeny_per_hour) / samples
            fam["exp_per_hour"] = ((fam["exp_per_hour"] * fam["samples"]) + exp_per_hour) / samples
            fam["samples"] = samples

    def record_provider_route(self, *, workload: str, provider: str, model: str) -> None:
        key = (
            (workload or "unknown").strip().lower(),
            (provider or "unknown").strip().lower(),
            (model or "unknown").strip().lower(),
        )
        with self._lock:
            self._provider_route[key] = self._provider_route.get(key, 0) + 1

    def record_breaker(self, *, family: str, key: str, state: str) -> None:
        idx = ((family or "unknown").strip().lower(), (key or "unknown").strip().lower(), (state or "unknown").strip().lower())
        with self._lock:
            self._breaker_events[idx] = self._breaker_events.get(idx, 0) + 1

    def render_prometheus(self) -> str:
        lines: list[str] = []

        with self._lock:
            latency = dict(self._latency)
            queue_backlog = dict(self._queue_backlog_by_tier)
            queue_decisions = dict(self._queue_decisions)
            ack_success = dict(self._ack_success)
            ack_total = dict(self._ack_total)
            deaths = dict(self._death_rate)
            macro = {k: dict(v) for k, v in self._macro_success.items()}
            shadow = {k: dict(v) for k, v in self._shadow_disagreement.items()}
            economy = {k: dict(v) for k, v in self._economy_rates.items()}
            routes = dict(self._provider_route)
            breakers = dict(self._breaker_events)

        lines.append("# HELP sidecar_up Sidecar runtime health")
        lines.append("# TYPE sidecar_up gauge")
        lines.append("sidecar_up 1")

        lines.append("# HELP sidecar_uptime_seconds Sidecar uptime in seconds")
        lines.append("# TYPE sidecar_uptime_seconds gauge")
        lines.append(f"sidecar_uptime_seconds {(datetime.now(UTC) - self._started_at).total_seconds():.3f}")

        lines.append("# HELP sidecar_latency_ms Runtime latency by domain")
        lines.append("# TYPE sidecar_latency_ms histogram")
        for domain, hist in sorted(latency.items()):
            running = 0
            for bound, count in zip(hist.bounds, hist.counts):
                running += count
                lines.append(f"sidecar_latency_ms_bucket{_labels({'domain': domain, 'le': str(bound)})} {running}")
            lines.append(f"sidecar_latency_ms_bucket{_labels({'domain': domain, 'le': '+Inf'})} {hist.total}")
            lines.append(f"sidecar_latency_ms_sum{_labels({'domain': domain})} {hist.total_sum:.6f}")
            lines.append(f"sidecar_latency_ms_count{_labels({'domain': domain})} {hist.total}")

        lines.append("# HELP sidecar_queue_backlog Queue backlog by priority tier")
        lines.append("# TYPE sidecar_queue_backlog gauge")
        for tier, depth in sorted(queue_backlog.items()):
            lines.append(f"sidecar_queue_backlog{_labels({'tier': tier})} {int(depth)}")

        lines.append("# HELP sidecar_queue_decisions_total Queue arbitration decisions")
        lines.append("# TYPE sidecar_queue_decisions_total counter")
        for (tier, status, reason), count in sorted(queue_decisions.items()):
            lines.append(f"sidecar_queue_decisions_total{_labels({'tier': tier, 'status': status, 'reason': reason})} {count}")

        lines.append("# HELP sidecar_ack_total Ack total grouped by source and action kind")
        lines.append("# TYPE sidecar_ack_total counter")
        for (source, kind), count in sorted(ack_total.items()):
            lines.append(f"sidecar_ack_total{_labels({'source': source, 'action_kind': kind})} {count}")

        lines.append("# HELP sidecar_ack_outcome_total Ack outcomes grouped by source/action")
        lines.append("# TYPE sidecar_ack_outcome_total counter")
        for (source, kind, outcome), count in sorted(ack_success.items()):
            lines.append(f"sidecar_ack_outcome_total{_labels({'source': source, 'action_kind': kind, 'outcome': outcome})} {count}")

        lines.append("# HELP sidecar_deaths_total Death events by map and doctrine")
        lines.append("# TYPE sidecar_deaths_total counter")
        for (map_name, doctrine_version), count in sorted(deaths.items()):
            lines.append(f"sidecar_deaths_total{_labels({'map': map_name, 'doctrine': doctrine_version})} {count}")

        lines.append("# HELP sidecar_macro_publish_total Macro publish attempts by version")
        lines.append("# TYPE sidecar_macro_publish_total counter")
        lines.append("# HELP sidecar_macro_publish_success_total Macro publish success by version")
        lines.append("# TYPE sidecar_macro_publish_success_total counter")
        for version, row in sorted(macro.items()):
            lines.append(f"sidecar_macro_publish_total{_labels({'version': version})} {int(row.get('total') or 0)}")
            lines.append(f"sidecar_macro_publish_success_total{_labels({'version': version})} {int(row.get('ok') or 0)}")

        lines.append("# HELP sidecar_shadow_disagreement_total Shadow model disagreement by family")
        lines.append("# TYPE sidecar_shadow_disagreement_total counter")
        lines.append("# HELP sidecar_shadow_total Shadow model comparisons by family")
        lines.append("# TYPE sidecar_shadow_total counter")
        for family, row in sorted(shadow.items()):
            lines.append(f"sidecar_shadow_total{_labels({'family': family})} {int(row.get('total') or 0)}")
            lines.append(
                f"sidecar_shadow_disagreement_total{_labels({'family': family})} {int(row.get('disagree') or 0)}"
            )
            lines.append(
                f"sidecar_shadow_high_conf_disagreement_total{_labels({'family': family})} {int(row.get('high_conf_disagree') or 0)}"
            )

        lines.append("# HELP sidecar_economy_zeny_per_hour Mean zeny per hour by plan family")
        lines.append("# TYPE sidecar_economy_zeny_per_hour gauge")
        lines.append("# HELP sidecar_economy_exp_per_hour Mean exp per hour by plan family")
        lines.append("# TYPE sidecar_economy_exp_per_hour gauge")
        for family, row in sorted(economy.items()):
            lines.append(f"sidecar_economy_zeny_per_hour{_labels({'plan_family': family})} {float(row.get('zeny_per_hour') or 0.0):.6f}")
            lines.append(f"sidecar_economy_exp_per_hour{_labels({'plan_family': family})} {float(row.get('exp_per_hour') or 0.0):.6f}")

        lines.append("# HELP sidecar_provider_routes_total Provider route decisions")
        lines.append("# TYPE sidecar_provider_routes_total counter")
        for (workload, provider, model), count in sorted(routes.items()):
            lines.append(
                f"sidecar_provider_routes_total{_labels({'workload': workload, 'provider': provider, 'model': model})} {count}"
            )

        lines.append("# HELP sidecar_breaker_events_total Breaker transitions")
        lines.append("# TYPE sidecar_breaker_events_total counter")
        for (family, key, state), count in sorted(breakers.items()):
            lines.append(f"sidecar_breaker_events_total{_labels({'family': family, 'key': key, 'state': state})} {count}")

        return "\n".join(lines) + "\n"

