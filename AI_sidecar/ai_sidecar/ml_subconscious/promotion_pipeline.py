from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from threading import RLock

from ai_sidecar.contracts.ml_subconscious import ModelFamily

logger = logging.getLogger(__name__)


def _stable_percent(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    val = int.from_bytes(digest[:8], "little")
    return float((val % 10000) / 100.0)


@dataclass(slots=True)
class GuardedPromotionPipeline:
    min_confidence: float = 0.75
    min_canary_evals: int = 20
    _lock: RLock = field(default_factory=RLock)
    _state: dict[str, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for family in ModelFamily:
            self._state[family.value] = {
                "enabled": False,
                "model_version": "",
                "canary_percentage": 0.0,
                "rollback_threshold": 0.25,
                "scope": {},
                "guardrails": {
                    "reflex_safety_rules": True,
                    "action_conflict_arbitration": True,
                    "queue_admission_guards": True,
                    "macro_reload_safety": True,
                    "central_fleet_constraints": True,
                },
                "stats": {
                    "decisions": 0,
                    "executed": 0,
                    "success": 0,
                    "failures": 0,
                    "rolled_back": 0,
                    "rejected_safety": 0,
                },
            }

    def configure(
        self,
        *,
        family: ModelFamily,
        model_version: str,
        canary_percentage: float,
        rollback_threshold: float,
        scope: dict[str, object],
    ) -> dict[str, object]:
        with self._lock:
            row = self._state[family.value]
            row["enabled"] = True
            row["model_version"] = model_version
            row["canary_percentage"] = max(0.0, min(100.0, float(canary_percentage)))
            row["rollback_threshold"] = max(0.01, min(1.0, float(rollback_threshold)))
            row["scope"] = dict(scope)
        logger.warning(
            "ml_promotion_configured",
            extra={
                "event": "ml_promotion_configured",
                "family": family.value,
                "model_version": model_version,
                "canary_percentage": canary_percentage,
                "rollback_threshold": rollback_threshold,
            },
        )
        return self.get(family=family)

    def _scope_match(self, scope: dict[str, object], context: dict[str, object]) -> bool:
        if not scope:
            return True
        for key, expected in scope.items():
            if expected in (None, "", []):
                continue
            actual = context.get(key)
            if isinstance(expected, list):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True

    def should_execute(
        self,
        *,
        family: ModelFamily,
        bot_id: str,
        trace_id: str,
        confidence: float,
        safety_flags: list[str],
        context: dict[str, object],
    ) -> dict[str, object]:
        with self._lock:
            row = dict(self._state.get(family.value, {}))

        if not bool(row.get("enabled")):
            return {"allowed": False, "reason": "promotion_disabled", "mode": "shadow"}

        if safety_flags:
            with self._lock:
                self._state[family.value]["stats"]["rejected_safety"] += 1
                self._state[family.value]["stats"]["decisions"] += 1
            return {"allowed": False, "reason": "safety_flags_present", "mode": "shadow"}

        if float(confidence) < self.min_confidence:
            with self._lock:
                self._state[family.value]["stats"]["decisions"] += 1
            return {"allowed": False, "reason": "confidence_below_threshold", "mode": "shadow"}

        scope = dict(row.get("scope") or {})
        if not self._scope_match(scope, context):
            with self._lock:
                self._state[family.value]["stats"]["decisions"] += 1
            return {"allowed": False, "reason": "outside_scope", "mode": "shadow"}

        canary = float(row.get("canary_percentage") or 0.0)
        bucket = _stable_percent(f"{family.value}:{bot_id}:{trace_id}")
        allowed = bucket < canary

        with self._lock:
            self._state[family.value]["stats"]["decisions"] += 1
            if allowed:
                self._state[family.value]["stats"]["executed"] += 1

        return {
            "allowed": allowed,
            "reason": "canary_allow" if allowed else "canary_shadow_hold",
            "mode": "guarded_execute" if allowed else "shadow",
            "bucket": bucket,
            "canary_percentage": canary,
            "guardrails_enforced": list((row.get("guardrails") or {}).keys()),
        }

    def record_outcome(self, *, family: ModelFamily, executed: bool, success: bool) -> dict[str, object]:
        with self._lock:
            row = self._state[family.value]
            if executed:
                if success:
                    row["stats"]["success"] += 1
                else:
                    row["stats"]["failures"] += 1

            executed_n = int(row["stats"].get("executed") or 0)
            failures = int(row["stats"].get("failures") or 0)
            threshold = float(row.get("rollback_threshold") or 0.25)
            degrade = (failures / executed_n) if executed_n > 0 else 0.0
            if executed_n >= self.min_canary_evals and degrade > threshold and bool(row.get("enabled")):
                row["enabled"] = False
                row["stats"]["rolled_back"] += 1
                logger.error(
                    "ml_promotion_auto_rollback",
                    extra={
                        "event": "ml_promotion_auto_rollback",
                        "family": family.value,
                        "failure_rate": degrade,
                        "rollback_threshold": threshold,
                        "executed": executed_n,
                    },
                )
        return self.get(family=family)

    def get(self, *, family: ModelFamily) -> dict[str, object]:
        with self._lock:
            row = dict(self._state.get(family.value, {}))
            row["stats"] = dict(row.get("stats") or {})
            row["family"] = family.value
            return row

    def metrics(self) -> dict[str, object]:
        with self._lock:
            snapshot = {key: dict(value) for key, value in self._state.items()}
            for key in snapshot:
                snapshot[key]["stats"] = dict(snapshot[key].get("stats") or {})
        return {
            "families": snapshot,
        }

