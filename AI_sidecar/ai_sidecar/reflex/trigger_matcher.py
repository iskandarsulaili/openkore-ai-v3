from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from time import perf_counter

from ai_sidecar.contracts.reflex import ReflexPredicate, ReflexRule


_MISSING = object()


@dataclass(slots=True)
class TriggerDecision:
    matched: bool
    suppressed: bool
    reason: str
    elapsed_ms: float


class TriggerMatcher:
    def __init__(self) -> None:
        self._lock = RLock()
        self._last_fired_ms: dict[tuple[str, str], float] = {}

    def evaluate(self, *, bot_id: str, rule: ReflexRule, facts: dict[str, object]) -> TriggerDecision:
        started = perf_counter()

        if not rule.enabled:
            return TriggerDecision(matched=False, suppressed=True, reason="rule_disabled", elapsed_ms=self._elapsed(started))

        if not self._match_clause(rule.trigger.all, match_all=True, facts=facts):
            return TriggerDecision(matched=False, suppressed=False, reason="trigger_all_unmet", elapsed_ms=self._elapsed(started))

        if rule.trigger.any and not self._match_clause(rule.trigger.any, match_all=False, facts=facts):
            return TriggerDecision(matched=False, suppressed=False, reason="trigger_any_unmet", elapsed_ms=self._elapsed(started))

        if not self._match_clause(rule.guards, match_all=True, facts=facts):
            return TriggerDecision(matched=False, suppressed=False, reason="guard_unmet", elapsed_ms=self._elapsed(started))

        if rule.cooldown_ms > 0:
            cooldown_hit, remaining = self._is_in_cooldown(bot_id=bot_id, rule_id=rule.rule_id, cooldown_ms=rule.cooldown_ms)
            if cooldown_hit:
                return TriggerDecision(
                    matched=False,
                    suppressed=True,
                    reason=f"cooldown_active:{remaining}",
                    elapsed_ms=self._elapsed(started),
                )

        return TriggerDecision(matched=True, suppressed=False, reason="matched", elapsed_ms=self._elapsed(started))

    def mark_fired(self, *, bot_id: str, rule_id: str) -> None:
        now_ms = perf_counter() * 1000.0
        with self._lock:
            self._last_fired_ms[(bot_id, rule_id)] = now_ms

    def _match_clause(self, predicates: list[ReflexPredicate], *, match_all: bool, facts: dict[str, object]) -> bool:
        if not predicates:
            return True
        if match_all:
            return all(self._match_predicate(item, facts=facts) for item in predicates)
        return any(self._match_predicate(item, facts=facts) for item in predicates)

    def _match_predicate(self, predicate: ReflexPredicate, *, facts: dict[str, object]) -> bool:
        left = facts.get(predicate.fact, _MISSING)
        op = predicate.op.value
        right = predicate.value

        if op == "exists":
            exists = left is not _MISSING and left is not None
            want = True if right is None else bool(right)
            return exists if want else not exists

        if left is _MISSING:
            return False

        if op == "eq":
            return left == right
        if op == "neq":
            return left != right
        if op == "gt":
            return self._safe_cmp(left, right, "gt")
        if op == "gte":
            return self._safe_cmp(left, right, "gte")
        if op == "lt":
            return self._safe_cmp(left, right, "lt")
        if op == "lte":
            return self._safe_cmp(left, right, "lte")
        if op == "in":
            return left in right if isinstance(right, (list, tuple, set)) else False
        if op == "not_in":
            return left not in right if isinstance(right, (list, tuple, set)) else False
        if op == "contains":
            if isinstance(left, str):
                return str(right) in left
            if isinstance(left, (list, tuple, set)):
                return right in left
            if isinstance(left, dict):
                return str(right) in left
            return False
        if op == "startswith":
            return isinstance(left, str) and str(left).startswith(str(right))
        if op == "endswith":
            return isinstance(left, str) and str(left).endswith(str(right))
        return False

    def _safe_cmp(self, left: object, right: object, mode: str) -> bool:
        try:
            lf = float(left)
            rf = float(right)
        except (TypeError, ValueError):
            return False
        if mode == "gt":
            return lf > rf
        if mode == "gte":
            return lf >= rf
        if mode == "lt":
            return lf < rf
        if mode == "lte":
            return lf <= rf
        return False

    def _is_in_cooldown(self, *, bot_id: str, rule_id: str, cooldown_ms: int) -> tuple[bool, int]:
        now_ms = perf_counter() * 1000.0
        with self._lock:
            last_ms = self._last_fired_ms.get((bot_id, rule_id))
            if last_ms is None:
                return False, 0
            elapsed = now_ms - last_ms
        if elapsed >= float(cooldown_ms):
            return False, 0
        remaining = int(max(0.0, float(cooldown_ms) - elapsed))
        return True, remaining

    def _elapsed(self, started: float) -> float:
        return (perf_counter() - started) * 1000.0

