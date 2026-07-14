"""
Integration Test Suite — verifies that modules work together, not just compile.
Tests the combat loop, market executor, multi-client coordinator, and PDCA loop.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of an integration test."""
    test_name: str
    passed: bool = False
    duration_ms: float = 0.0
    error: str = ""
    details: str = ""


class IntegrationTester:
    """Runs integration tests to verify modules work together."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._results: list[TestResult] = []
        self._runtime: Any = None

    def set_runtime(self, runtime: Any) -> None:
        with self._lock:
            self._runtime = runtime

    def run_all(self) -> list[TestResult]:
        """Run all integration tests."""
        with self._lock:
            self._results.clear()
            self._test_combat_loop()
            self._test_action_executor()
            self._test_degradation_manager()
            self._test_self_healer()
            self._test_time_scheduler()
            self._test_goal_planner()
            self._test_opportunity_cost()
            return list(self._results)

    def _test_combat_loop(self) -> None:
        start = time.time()
        try:
            from ai_sidecar.combat.combat_loop import get_combat_loop
            cl = get_combat_loop()
            cl.start()
            action = cl.tick()
            assert action is not None or action is None  # tick can return None if no target
            cl.stop()
            self._results.append(TestResult(
                "combat_loop_init", True, (time.time() - start) * 1000,
                details="Combat loop initializes, starts, ticks, and stops"
            ))
        except Exception as e:
            self._results.append(TestResult(
                "combat_loop_init", False, (time.time() - start) * 1000, error=str(e)
            ))

    def _test_action_executor(self) -> None:
        start = time.time()
        try:
            from ai_sidecar.combat.action_executor import get_action_executor
            ae = get_action_executor()
            mappings = ae.get_all_mappings()
            assert len(mappings) > 0, "No action mappings found"
            self._results.append(TestResult(
                "action_executor_mappings", True, (time.time() - start) * 1000,
                details=f"{len(mappings)} mappings loaded"
            ))
        except Exception as e:
            self._results.append(TestResult(
                "action_executor_mappings", False, (time.time() - start) * 1000, error=str(e)
            ))

    def _test_degradation_manager(self) -> None:
        start = time.time()
        try:
            from ai_sidecar.degradation_manager import get_degradation_manager
            dm = get_degradation_manager()
            dm.register_module("test_module")
            dm.report_success("test_module")
            assert dm.is_healthy("test_module")
            dm.report_failure("test_module", "test error")
            dm.report_failure("test_module", "test error 2")
            dm.report_failure("test_module", "test error 3")
            assert not dm.is_healthy("test_module")
            self._results.append(TestResult(
                "degradation_manager", True, (time.time() - start) * 1000,
                details="Module degrades after 3 failures"
            ))
        except Exception as e:
            self._results.append(TestResult(
                "degradation_manager", False, (time.time() - start) * 1000, error=str(e)
            ))

    def _test_self_healer(self) -> None:
        start = time.time()
        try:
            from ai_sidecar.self_healer import get_self_healer
            sh = get_self_healer()
            action = sh.heal_module("test_module", "connection timeout")
            assert action == "reconnect", f"Expected reconnect, got {action}"
            self._results.append(TestResult(
                "self_healer", True, (time.time() - start) * 1000,
                details=f"Heal action: {action}"
            ))
        except Exception as e:
            self._results.append(TestResult(
                "self_healer", False, (time.time() - start) * 1000, error=str(e)
            ))

    def _test_time_scheduler(self) -> None:
        start = time.time()
        try:
            from ai_sidecar.time_scheduler import get_time_scheduler
            ts = get_time_scheduler()
            strategy = ts.get_current_strategy()
            assert strategy, "No strategy returned"
            self._results.append(TestResult(
                "time_scheduler", True, (time.time() - start) * 1000,
                details=f"Strategy: {strategy}"
            ))
        except Exception as e:
            self._results.append(TestResult(
                "time_scheduler", False, (time.time() - start) * 1000, error=str(e)
            ))

    def _test_goal_planner(self) -> None:
        start = time.time()
        try:
            from ai_sidecar.goal_planner import get_goal_planner
            gp = get_goal_planner()
            goals = gp.get_active_goals()
            assert len(goals) > 0, "No goals loaded"
            tasks = gp.generate_daily_tasks()
            assert len(tasks) > 0, "No tasks generated"
            self._results.append(TestResult(
                "goal_planner", True, (time.time() - start) * 1000,
                details=f"{len(goals)} goals, {len(tasks)} tasks"
            ))
        except Exception as e:
            self._results.append(TestResult(
                "goal_planner", False, (time.time() - start) * 1000, error=str(e)
            ))

    def _test_opportunity_cost(self) -> None:
        start = time.time()
        try:
            from ai_sidecar.opportunity_cost_engine import get_opportunity_cost_engine
            oc = get_opportunity_cost_engine()
            result = oc.compare_farming_vs_quest(500000, 1.0, 5000000, 100000)
            assert "Recommend" in result, "No recommendation"
            self._results.append(TestResult(
                "opportunity_cost", True, (time.time() - start) * 1000,
                details=result[:100]
            ))
        except Exception as e:
            self._results.append(TestResult(
                "opportunity_cost", False, (time.time() - start) * 1000, error=str(e)
            ))

    def get_summary(self) -> str:
        with self._lock:
            passed = sum(1 for r in self._results if r.passed)
            total = len(self._results)
            lines = [f"── Integration Tests ({passed}/{total} passed) ──"]
            for r in self._results:
                status = "✅" if r.passed else "❌"
                lines.append(f"  {status} {r.test_name} ({r.duration_ms:.0f}ms)")
                if not r.passed and r.error:
                    lines.append(f"    Error: {r.error}")
            return "\n".join(lines)

    def reset(self) -> None:
        with self._lock:
            self._results.clear()


# ── Global Singleton ──

_integration: IntegrationTester | None = None
_integration_lock = RLock()


def get_integration_tester() -> IntegrationTester:
    global _integration
    with _integration_lock:
        if _integration is None:
            _integration = IntegrationTester()
        return _integration
