from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ai_sidecar.api.routers.health import ready as ready_route
from ai_sidecar.contracts.common import ContractMeta
from ai_sidecar.contracts.fleet_v2 import FleetSyncRequest
from ai_sidecar.fleet import ConstraintIngestionState, FleetSyncClient, OutcomeReporter
from ai_sidecar.lifecycle import RuntimeState
from ai_sidecar.runtime.action_queue import ActionQueue
from ai_sidecar.runtime.bot_registry import BotRegistry
from ai_sidecar.runtime.latency_router import LatencyRouter
from ai_sidecar.runtime.snapshot_cache import SnapshotCache


def _runtime_with_fleet_client(*, client_enabled: bool) -> RuntimeState:
    fleet_client = FleetSyncClient(base_url="http://fleet.invalid", timeout_seconds=0.2, enabled=client_enabled)
    return RuntimeState(
        started_at=datetime.now(UTC),
        workspace_root=Path("."),
        bot_registry=BotRegistry(),
        snapshot_cache=SnapshotCache(ttl_seconds=120),
        action_queue=ActionQueue(max_per_bot=8),
        latency_router=LatencyRouter(budget_ms=500),
        telemetry_store=None,  # type: ignore[arg-type]
        macro_compiler=None,  # type: ignore[arg-type]
        macro_publisher=None,  # type: ignore[arg-type]
        memory=None,  # type: ignore[arg-type]
        audit_trail=None,
        repositories=None,
        normalizer_bus=None,  # type: ignore[arg-type]
        reflex_engine=None,  # type: ignore[arg-type]
        fleet_sync_client=fleet_client,
        fleet_constraint_state=ConstraintIngestionState(central_enabled=client_enabled),
        fleet_outcome_reporter=OutcomeReporter(client=fleet_client),
    )


def test_constraint_ingestion_status_disabled_central_is_local_not_stale() -> None:
    state = ConstraintIngestionState(central_enabled=False)

    status = state.status()

    assert status["mode"] == "local"
    assert status["central_enabled"] is False
    assert status["central_available"] is False
    assert status["stale"] is False


def test_constraint_ingestion_status_enabled_central_without_sync_is_stale() -> None:
    state = ConstraintIngestionState(central_enabled=True)

    status = state.status()

    assert status["mode"] == "local"
    assert status["central_enabled"] is True
    assert status["central_available"] is False
    assert status["stale"] is True


def test_runtime_fleet_sync_central_disabled_is_quiet_info(caplog: pytest.LogCaptureFixture) -> None:
    runtime = _runtime_with_fleet_client(client_enabled=False)
    payload = FleetSyncRequest(
        meta=ContractMeta(contract_version="v1", source="pytest", bot_id="bot:fleet-disabled", trace_id="trace-fleet-disabled"),
        include_blackboard=False,
    )

    caplog.set_level("INFO", logger="ai_sidecar.lifecycle")
    response = runtime.fleet_sync(payload)

    assert response.ok is True
    assert response.mode == "local"
    assert response.central_available is False
    assert response.message == "central_unavailable:fleet_central_disabled"

    warning_events = [item for item in caplog.records if getattr(item, "event", "") == "fleet_sync_fallback_local"]
    assert warning_events == []
    info_events = [item for item in caplog.records if getattr(item, "event", "") == "fleet_sync_central_disabled_local"]
    assert info_events


def test_runtime_fleet_sync_central_unavailable_keeps_warning(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime_with_fleet_client(client_enabled=True)

    def _fail_ping(self: FleetSyncClient) -> tuple[bool, dict[str, object], str]:
        del self
        return False, {}, "timeout"

    monkeypatch.setattr(FleetSyncClient, "ping_blackboard", _fail_ping)
    payload = FleetSyncRequest(
        meta=ContractMeta(contract_version="v1", source="pytest", bot_id="bot:fleet-timeout", trace_id="trace-fleet-timeout"),
        include_blackboard=False,
    )

    caplog.set_level("WARNING", logger="ai_sidecar.lifecycle")
    response = runtime.fleet_sync(payload)

    assert response.ok is True
    assert response.mode == "local"
    assert response.central_available is False
    assert response.message == "central_unavailable:timeout"

    warning_events = [item for item in caplog.records if getattr(item, "event", "") == "fleet_sync_fallback_local"]
    assert warning_events


def test_lifespan_skips_fleet_sync_loop_when_central_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_sidecar import app as app_module

    class _StubPDCA:
        def __init__(self, runtime_state, config) -> None:
            del config
            self.runtime_state = runtime_state
            self.running = False

        def start(self) -> None:
            self.running = True

        async def stop(self) -> None:
            self.running = False

    class _StubRuntime:
        def __init__(self) -> None:
            self.fleet_sync_client = type("FleetClient", (), {"enabled": False})()
            self.pdca_loop = None
            self.shutdown_calls = 0

        async def shutdown(self) -> None:
            self.shutdown_calls += 1

    stub_runtime = _StubRuntime()
    calls: list[str] = []

    def _fake_create_runtime():
        return stub_runtime

    def _fake_start_fleet_sync_loop(runtime):
        del runtime
        calls.append("start_fleet_sync_loop")
        return asyncio.create_task(asyncio.sleep(0.01))

    monkeypatch.setattr(app_module, "create_runtime", _fake_create_runtime)
    monkeypatch.setattr(app_module, "start_fleet_sync_loop", _fake_start_fleet_sync_loop)
    monkeypatch.setattr("ai_sidecar.autonomy.pdca_loop.PDCALoop", _StubPDCA)
    monkeypatch.setattr("ai_sidecar.api.routers.autonomy.set_pdca_loop", lambda loop: None)

    async def _exercise() -> None:
        app = app_module.create_app()
        async with app.router.lifespan_context(app):
            assert app.state.runtime is stub_runtime

    asyncio.run(_exercise())

    assert calls == []
    assert stub_runtime.shutdown_calls == 1


def test_lifespan_starts_fleet_sync_loop_when_central_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_sidecar import app as app_module

    class _StubPDCA:
        def __init__(self, runtime_state, config) -> None:
            del config
            self.runtime_state = runtime_state
            self.running = False

        def start(self) -> None:
            self.running = True

        async def stop(self) -> None:
            self.running = False

    class _StubRuntime:
        def __init__(self) -> None:
            self.fleet_sync_client = type("FleetClient", (), {"enabled": True})()
            self.pdca_loop = None
            self.shutdown_calls = 0

        async def shutdown(self) -> None:
            self.shutdown_calls += 1

    stub_runtime = _StubRuntime()
    calls: list[str] = []

    def _fake_create_runtime():
        return stub_runtime

    async def _never_loop() -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise

    def _fake_start_fleet_sync_loop(runtime):
        del runtime
        calls.append("start_fleet_sync_loop")
        return asyncio.create_task(_never_loop())

    monkeypatch.setattr(app_module, "create_runtime", _fake_create_runtime)
    monkeypatch.setattr(app_module, "start_fleet_sync_loop", _fake_start_fleet_sync_loop)
    monkeypatch.setattr("ai_sidecar.autonomy.pdca_loop.PDCALoop", _StubPDCA)
    monkeypatch.setattr("ai_sidecar.api.routers.autonomy.set_pdca_loop", lambda loop: None)

    async def _exercise() -> None:
        app = app_module.create_app()
        async with app.router.lifespan_context(app):
            assert app.state.runtime is stub_runtime

    asyncio.run(_exercise())

    assert calls == ["start_fleet_sync_loop"]
    assert stub_runtime.shutdown_calls == 1


def test_ready_ignores_fleet_stale_when_central_is_disabled() -> None:
    class _Counter:
        def __init__(self, value: int) -> None:
            self._value = value

        def count(self) -> int:
            return self._value

    class _TelemetryStore:
        def backlog_size(self) -> int:
            return 0

        def count(self) -> int:
            return 0

    class _LatencyRouter:
        def average_ms(self) -> float:
            return 0.0

    class _Runtime:
        repositories = None
        persistence_degraded = False
        sqlite_path = None
        autonomy_policy: dict[str, object] = {}

        def __init__(self) -> None:
            self.bot_registry = _Counter(1)
            self.snapshot_cache = _Counter(1)
            self.telemetry_store = _TelemetryStore()
            self.latency_router = _LatencyRouter()

        def readiness_indicators(self) -> dict[str, object]:
            return {
                "planner_stale": False,
                "fleet_central_enabled": False,
                "fleet_central_stale": True,
                "fleet_central_available": False,
                "fleet_mode": "local",
                "fleet_last_sync_at": None,
                "objective_scheduler_degraded": False,
                "objective_scheduler_degraded_reason": "",
                "pdca_running": True,
                "pdca_circuit_breaker_tripped": False,
            }

        def counter_snapshot(self) -> dict[str, int]:
            return {}

    payload = ready_route(_Runtime())

    assert payload["fleet_central_enabled"] is False
    assert payload["fleet_central_stale"] is True
    assert payload["readiness_degraded"] is False


def test_ready_flags_degraded_when_central_enabled_and_stale() -> None:
    class _Counter:
        def __init__(self, value: int) -> None:
            self._value = value

        def count(self) -> int:
            return self._value

    class _TelemetryStore:
        def backlog_size(self) -> int:
            return 0

        def count(self) -> int:
            return 0

    class _LatencyRouter:
        def average_ms(self) -> float:
            return 0.0

    class _Runtime:
        repositories = None
        persistence_degraded = False
        sqlite_path = None
        autonomy_policy: dict[str, object] = {}

        def __init__(self) -> None:
            self.bot_registry = _Counter(1)
            self.snapshot_cache = _Counter(1)
            self.telemetry_store = _TelemetryStore()
            self.latency_router = _LatencyRouter()

        def readiness_indicators(self) -> dict[str, object]:
            return {
                "planner_stale": False,
                "fleet_central_enabled": True,
                "fleet_central_stale": True,
                "fleet_central_available": False,
                "fleet_mode": "local",
                "fleet_last_sync_at": None,
                "objective_scheduler_degraded": False,
                "objective_scheduler_degraded_reason": "",
                "pdca_running": True,
                "pdca_circuit_breaker_tripped": False,
            }

        def counter_snapshot(self) -> dict[str, int]:
            return {}

    payload = ready_route(_Runtime())

    assert payload["fleet_central_enabled"] is True
    assert payload["fleet_central_stale"] is True
    assert payload["readiness_degraded"] is True
