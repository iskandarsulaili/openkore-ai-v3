from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from threading import RLock


class IncidentTaxonomy:
    DUPLICATE_FARMING = "duplicate_farming"
    QUEST_COLLISION = "quest_collision"
    STORAGE_CONTENTION = "storage_contention"
    ROLE_DESYNC = "role_desync"
    MACRO_BROKEN = "macro_broken"
    PROVIDER_OUTAGE = "provider_outage"
    BOT_STALL = "bot_stall"
    ECONOMY_ANOMALY = "economy_anomaly"
    SOCIAL_RISK = "social_risk"
    PVP_RISK = "pvp_risk"
    RULE_CONFLICT = "rule_conflict"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class IncidentRecord:
    incident_id: str
    taxonomy: str
    severity: str
    title: str
    bot_id: str | None
    first_seen_at: datetime
    last_seen_at: datetime
    status: str = "open"
    count: int = 1
    details: dict[str, object] = field(default_factory=dict)
    acked_at: datetime | None = None
    escalated_at: datetime | None = None
    assignee: str = ""


class IncidentRegistry:
    def __init__(self, *, max_open: int = 2000) -> None:
        self._lock = RLock()
        self._max_open = max(100, int(max_open))
        self._by_key: dict[str, IncidentRecord] = {}

    def classify(self, *, event_type: str, severity: str, payload: dict[str, object] | None = None) -> str:
        text = event_type.lower().strip()
        if "duplicate_farming" in text:
            return IncidentTaxonomy.DUPLICATE_FARMING
        if "quest_collision" in text or "conflict_key_collision" in text:
            return IncidentTaxonomy.QUEST_COLLISION
        if "storage" in text and "contention" in text:
            return IncidentTaxonomy.STORAGE_CONTENTION
        if "role" in text and "lease" in text:
            return IncidentTaxonomy.ROLE_DESYNC
        if "macro" in text and ("failed" in text or "error" in text):
            return IncidentTaxonomy.MACRO_BROKEN
        if "provider" in text and ("error" in text or "timeout" in text or "failed" in text):
            return IncidentTaxonomy.PROVIDER_OUTAGE
        if "stuck" in text or "stall" in text:
            return IncidentTaxonomy.BOT_STALL
        if "economy" in text or "zeny" in text:
            return IncidentTaxonomy.ECONOMY_ANOMALY
        if "social" in text:
            return IncidentTaxonomy.SOCIAL_RISK
        if "pvp" in text:
            return IncidentTaxonomy.PVP_RISK
        if "rule" in text and "conflict" in text:
            return IncidentTaxonomy.RULE_CONFLICT

        body = payload or {}
        if str(body.get("type") or "") in {
            IncidentTaxonomy.DUPLICATE_FARMING,
            IncidentTaxonomy.QUEST_COLLISION,
            IncidentTaxonomy.STORAGE_CONTENTION,
        }:
            return str(body.get("type"))

        if severity.lower() in {"warning", "error", "critical"}:
            return IncidentTaxonomy.UNKNOWN
        return ""

    def record_event(
        self,
        *,
        event_type: str,
        severity: str,
        title: str,
        bot_id: str | None,
        payload: dict[str, object] | None = None,
    ) -> IncidentRecord | None:
        taxonomy = self.classify(event_type=event_type, severity=severity, payload=payload)
        if not taxonomy:
            return None

        now = datetime.now(UTC)
        key = f"{taxonomy}:{bot_id or 'fleet'}:{str((payload or {}).get('key') or (payload or {}).get('conflict_key') or event_type)}"

        with self._lock:
            row = self._by_key.get(key)
            if row is None:
                if len(self._by_key) >= self._max_open:
                    oldest_key = min(self._by_key, key=lambda item: self._by_key[item].last_seen_at)
                    del self._by_key[oldest_key]
                row = IncidentRecord(
                    incident_id=f"inc-{now.strftime('%Y%m%d%H%M%S')}-{abs(hash(key)) % 1000000:06d}",
                    taxonomy=taxonomy,
                    severity=severity.lower(),
                    title=title,
                    bot_id=bot_id,
                    first_seen_at=now,
                    last_seen_at=now,
                    details=dict(payload or {}),
                )
                self._by_key[key] = row
            else:
                row.last_seen_at = now
                row.count += 1
                row.severity = severity.lower()
                row.details.update(dict(payload or {}))
            return row

    def list_incidents(self, *, include_closed: bool = False, limit: int = 100) -> list[dict[str, object]]:
        with self._lock:
            rows = list(self._by_key.values())
        rows.sort(key=lambda item: item.last_seen_at, reverse=True)
        if not include_closed:
            rows = [item for item in rows if item.status != "closed"]
        rows = rows[: max(1, min(int(limit), 2000))]
        return [asdict(item) for item in rows]

    def ack(self, *, incident_id: str, assignee: str = "") -> dict[str, object]:
        with self._lock:
            for item in self._by_key.values():
                if item.incident_id != incident_id:
                    continue
                item.status = "acked"
                item.assignee = assignee
                item.acked_at = datetime.now(UTC)
                return {"ok": True, "incident": asdict(item)}
        return {"ok": False, "message": "incident_not_found", "incident_id": incident_id}

    def escalate(self, *, incident_id: str, assignee: str = "") -> dict[str, object]:
        with self._lock:
            for item in self._by_key.values():
                if item.incident_id != incident_id:
                    continue
                item.status = "escalated"
                item.assignee = assignee
                item.escalated_at = datetime.now(UTC)
                if item.severity in {"info", "debug"}:
                    item.severity = "warning"
                return {"ok": True, "incident": asdict(item)}
        return {"ok": False, "message": "incident_not_found", "incident_id": incident_id}

