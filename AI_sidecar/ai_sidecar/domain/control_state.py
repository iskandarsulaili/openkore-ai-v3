from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock

from ai_sidecar.contracts.control_domain import ControlChangePlan, ControlChangeState


@dataclass(slots=True)
class ControlPlanState:
    plan: ControlChangePlan
    state: ControlChangeState
    expected: dict[str, str]
    lineage: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ControlStateStore:
    _lock: RLock = field(default_factory=RLock)
    _plans: dict[str, ControlPlanState] = field(default_factory=dict)

    def upsert(self, plan: ControlPlanState) -> None:
        with self._lock:
            self._plans[plan.plan.plan_id] = plan

    def get(self, plan_id: str) -> ControlPlanState | None:
        with self._lock:
            return self._plans.get(plan_id)

