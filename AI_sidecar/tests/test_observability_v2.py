from __future__ import annotations

from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_sidecar.api.deps import get_runtime
from ai_sidecar.api.routers import observability_v2
from ai_sidecar.observability import install_fastapi_tracing
from ai_sidecar.observability.doctrine_manager import DoctrineManager
from ai_sidecar.observability.explainability import ExplainabilityStore
from ai_sidecar.observability.incident_taxonomy import IncidentRegistry
from ai_sidecar.observability.metrics_collector import SLOMetricsCollector
from ai_sidecar.observability.security_auditor import SecurityAuditor
from ai_sidecar.observability.tracing import TRACE_ID_HEADER, TraceStore


class _ObservabilityRuntime:
    def __init__(self) -> None:
        self.slo_metrics = SLOMetricsCollector()
        self.trace_store = TraceStore(max_traces=500, max_events_per_trace=50)
        self.incident_registry = IncidentRegistry(max_open=500)
        self.explainability = ExplainabilityStore(max_records=500)
        self.security_auditor = SecurityAuditor(doctrine_denylist=["dupe", "exploit"])
        self.doctrine_manager = DoctrineManager()

        self.slo_metrics.observe_latency(domain="planner", elapsed_ms=21.5)
        self.slo_metrics.record_provider_route(workload="strategic_planning", provider="openai", model="gpt-4o-mini")
        self.trace_store.add_event(
            trace_id="trace-fixed",
            name="planner.plan",
            attributes={"bot_id": "bot:obs", "ok": True},
        )
        self.incident_registry.record_event(
            event_type="provider_error_timeout",
            severity="warning",
            title="provider timeout",
            bot_id="bot:obs",
            payload={"key": "provider.openai"},
        )
        self.explainability.add(
            kind="planner",
            bot_id="bot:obs",
            trace_id="trace-fixed",
            summary="planner selected openai",
            details={"provider": "openai"},
        )
        self.security_auditor.record(
            kind="policy_warning",
            source="bootstrap",
            bot_id="bot:obs",
            detail="sample warning",
            severity="warning",
        )

    def observability_metrics_text(self) -> str:
        return self.slo_metrics.render_prometheus()

    def observability_recent_traces(self, *, limit: int = 50) -> list[dict[str, object]]:
        return self.trace_store.recent(limit=limit)

    def observability_trace(self, *, trace_id: str) -> list[dict[str, object]]:
        return self.trace_store.get_trace(trace_id=trace_id)

    def observability_incidents(self, *, include_closed: bool = False, limit: int = 100) -> list[dict[str, object]]:
        return self.incident_registry.list_incidents(include_closed=include_closed, limit=limit)

    def observability_ack_incident(self, *, incident_id: str, assignee: str = "") -> dict[str, object]:
        return self.incident_registry.ack(incident_id=incident_id, assignee=assignee)

    def observability_escalate_incident(self, *, incident_id: str, assignee: str = "") -> dict[str, object]:
        return self.incident_registry.escalate(incident_id=incident_id, assignee=assignee)

    def observability_explainability(
        self,
        *,
        kind: str | None = None,
        bot_id: str | None = None,
        trace_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        return self.explainability.list(kind=kind, bot_id=bot_id, trace_id=trace_id, limit=limit)

    def observability_security_violations(self, *, limit: int = 100) -> list[dict[str, object]]:
        return self.security_auditor.recent(limit=limit)

    def observability_doctrine_active(self) -> dict[str, object]:
        return {"ok": True, "doctrine": self.doctrine_manager.active()}

    def observability_doctrine_versions(self, *, limit: int = 50) -> dict[str, object]:
        return {
            "ok": True,
            "versions": self.doctrine_manager.list_versions(limit=limit),
            "active": self.doctrine_manager.active(),
        }

    def observability_doctrine_publish(
        self,
        *,
        version: str,
        policy: dict[str, object],
        canary_percentage: float,
        activate: bool,
        author: str = "",
        trace_id: str | None = None,
    ) -> dict[str, object]:
        if self.security_auditor is not None:
            allowed, reason = self.security_auditor.validate_doctrine(doctrine=policy)
            if not allowed:
                self.security_auditor.record(
                    kind="doctrine_policy_violation",
                    source="observability_doctrine_publish",
                    bot_id="fleet",
                    detail=reason,
                    severity="error",
                )
                return {"ok": False, "message": reason, "version": version}
        result = self.doctrine_manager.publish(
            version=version,
            policy=policy,
            canary_percentage=canary_percentage,
            activate=activate,
            author=author,
        )
        event_trace_id = trace_id or f"doctrine-{datetime.now(UTC).strftime('%H%M%S')}"
        self.trace_store.add_event(
            trace_id=event_trace_id,
            name="doctrine.publish",
            attributes={"ok": bool(result.get("ok")), "version": version},
        )
        return result

    def observability_doctrine_rollback(
        self,
        *,
        target_version: str | None = None,
        author: str = "",
        trace_id: str | None = None,
    ) -> dict[str, object]:
        result = self.doctrine_manager.rollback(target_version=target_version)
        event_trace_id = trace_id or f"doctrine-{datetime.now(UTC).strftime('%H%M%S')}"
        self.trace_store.add_event(
            trace_id=event_trace_id,
            name="doctrine.rollback",
            attributes={"ok": bool(result.get("ok")), "target_version": target_version or ""},
        )
        return result


def test_observability_v2_router_endpoints_and_tracing_middleware() -> None:
    runtime = _ObservabilityRuntime()
    app = FastAPI()
    install_fastapi_tracing(app)

    @app.get("/trace-probe")
    def trace_probe() -> dict[str, object]:
        return {"ok": True}

    app.include_router(observability_v2.router)
    app.dependency_overrides[get_runtime] = lambda: runtime

    try:
        with TestClient(app) as client:
            trace_resp = client.get("/trace-probe", headers={TRACE_ID_HEADER: "trace-http-observability"})
            assert trace_resp.status_code == 200
            assert trace_resp.headers.get(TRACE_ID_HEADER) == "trace-http-observability"

            metrics_resp = client.get("/v2/observability/metrics")
            assert metrics_resp.status_code == 200
            assert "text/plain" in str(metrics_resp.headers.get("content-type", ""))
            assert "sidecar_up" in metrics_resp.text

            traces_resp = client.get("/v2/observability/traces", params={"limit": 20})
            assert traces_resp.status_code == 200
            assert traces_resp.json()["ok"] is True
            assert traces_resp.json()["count"] >= 1

            trace_id_resp = client.get("/v2/observability/traces/trace-fixed")
            assert trace_id_resp.status_code == 200
            assert trace_id_resp.json()["trace_id"] == "trace-fixed"
            assert trace_id_resp.json()["count"] >= 1

            incidents_resp = client.get("/v2/observability/incidents")
            assert incidents_resp.status_code == 200
            assert incidents_resp.json()["count"] >= 1
            incident_id = incidents_resp.json()["incidents"][0]["incident_id"]

            ack_resp = client.post(
                f"/v2/observability/incidents/{incident_id}/ack",
                json={"assignee": "ops-a"},
            )
            assert ack_resp.status_code == 200
            assert ack_resp.json()["ok"] is True

            escalate_resp = client.post(
                f"/v2/observability/incidents/{incident_id}/escalate",
                json={"assignee": "ops-b"},
            )
            assert escalate_resp.status_code == 200
            assert escalate_resp.json()["ok"] is True

            explain_resp = client.get("/v2/observability/explainability", params={"kind": "planner", "limit": 10})
            assert explain_resp.status_code == 200
            assert explain_resp.json()["count"] >= 1

            sec_resp = client.get("/v2/observability/security/violations", params={"limit": 10})
            assert sec_resp.status_code == 200
            assert sec_resp.json()["count"] >= 1

            active_resp = client.get("/v2/observability/doctrine/active")
            assert active_resp.status_code == 200
            assert active_resp.json()["ok"] is True
            assert "doctrine" in active_resp.json()

            publish_resp = client.post(
                "/v2/observability/doctrine/publish",
                json={
                    "version": "doctrine-v3",
                    "policy": {"conflict_resolution": {"on_equal": "lease_owner"}},
                    "canary_percentage": 20.0,
                    "activate": True,
                    "author": "ops",
                    "trace_id": "trace-doctrine-publish",
                },
            )
            assert publish_resp.status_code == 200
            assert publish_resp.json()["ok"] is True

            blocked_publish = client.post(
                "/v2/observability/doctrine/publish",
                json={
                    "version": "doctrine-bad",
                    "policy": {"script": "attempt dupe route"},
                    "canary_percentage": 10.0,
                    "activate": False,
                    "author": "ops",
                },
            )
            assert blocked_publish.status_code == 200
            assert blocked_publish.json()["ok"] is False
            assert "doctrine_denylist_violation" in blocked_publish.json()["message"]

            versions_resp = client.get("/v2/observability/doctrine/versions", params={"limit": 10})
            assert versions_resp.status_code == 200
            assert versions_resp.json()["ok"] is True
            assert any(item["version"] == "doctrine-v3" for item in versions_resp.json()["versions"])

            rollback_resp = client.post(
                "/v2/observability/doctrine/rollback",
                json={"target_version": "local-default", "author": "ops"},
            )
            assert rollback_resp.status_code == 200
            assert rollback_resp.json()["ok"] is True

            active_after_rollback = client.get("/v2/observability/doctrine/active")
            assert active_after_rollback.status_code == 200
            assert active_after_rollback.json()["doctrine"]["version"] == "local-default"
    finally:
        app.dependency_overrides = {}

