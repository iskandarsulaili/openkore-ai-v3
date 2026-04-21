from __future__ import annotations

from dataclasses import asdict

from ai_sidecar.observability.audit import AuditTrail
from ai_sidecar.observability.incident_taxonomy import IncidentRecord, IncidentRegistry
from ai_sidecar.observability.security_auditor import SecurityAuditor


class ObservabilityAuditLogger:
    def __init__(
        self,
        *,
        audit_trail: AuditTrail | None,
        incident_registry: IncidentRegistry | None = None,
        security_auditor: SecurityAuditor | None = None,
    ) -> None:
        self._audit_trail = audit_trail
        self._incident_registry = incident_registry
        self._security = security_auditor

    def record(
        self,
        *,
        level: str,
        event_type: str,
        summary: str,
        bot_id: str | None,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        clean_payload = dict(payload or {})
        if self._security is not None:
            clean_payload = self._security.sanitize_payload(clean_payload)

        if self._audit_trail is not None:
            self._audit_trail.record(
                level=level,
                event_type=event_type,
                summary=summary,
                bot_id=bot_id,
                payload=clean_payload,
            )

        incident: IncidentRecord | None = None
        if self._incident_registry is not None:
            incident = self._incident_registry.record_event(
                event_type=event_type,
                severity=level,
                title=summary,
                bot_id=bot_id,
                payload=clean_payload,
            )

        result = {
            "ok": True,
            "level": level,
            "event_type": event_type,
            "summary": summary,
            "bot_id": bot_id,
            "payload": clean_payload,
        }
        if incident is not None:
            result["incident"] = asdict(incident)
        return result

