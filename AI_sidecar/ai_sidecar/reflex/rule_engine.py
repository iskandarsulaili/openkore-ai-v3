from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
import logging
from pathlib import Path
from threading import RLock
from time import perf_counter
from uuid import uuid4

from ai_sidecar.contracts.events import NormalizedEvent
from ai_sidecar.contracts.reflex import (
    ReflexCategory,
    ReflexActionTemplate,
    ReflexPlannerInterop,
    ReflexPredicate,
    ReflexRule,
    ReflexTriggerClause,
    ReflexTriggerRecord,
)
from ai_sidecar.contracts.state_graph import EnrichedWorldState
import json
import logging
from pathlib import Path
from ai_sidecar.reflex.action_emitter import ActionEmitter
from ai_sidecar.reflex.circuit_breaker import ReflexCircuitBreaker
from ai_sidecar.reflex.trigger_matcher import TriggerMatcher



logger = logging.getLogger(__name__)
@dataclass(slots=True)
class _RuleDescriptor:
    rule: ReflexRule
    event_type_hints: set[str]


@dataclass(slots=True)
class _PendingOutcome:
    bot_id: str
    trigger_id: str
    rule_id: str
    breaker_key: str
    breaker_family: str


class ReflexRuleEngine:
    _CATEGORY_ORDER: dict[ReflexCategory, int] = {
        ReflexCategory.survival: 0,
        ReflexCategory.combat: 1,
        ReflexCategory.interaction: 2,
    }

    def __init__(
        self,
        *,
        workspace_root: Path,
        contract_version: str,
        action_ttl_seconds: int,
        trigger_history_per_bot: int = 1000,
    ) -> None:
        self._lock = RLock()
        self._history_per_bot = max(100, int(trigger_history_per_bot))

        self._rules_by_bot: dict[str, dict[str, ReflexRule]] = {}
        self._descriptors_by_bot: dict[str, list[_RuleDescriptor]] = {}
        self._recent_triggers: dict[str, deque[ReflexTriggerRecord]] = {}
        self._pending_outcomes: dict[str, _PendingOutcome] = {}
        self._triggers_by_id: dict[tuple[str, str], ReflexTriggerRecord] = {}
        self._conflict_reservations_ms: dict[tuple[str, str], float] = {}

        self._matcher = TriggerMatcher()
        self._breakers = ReflexCircuitBreaker()
        self._emitter = ActionEmitter(
            workspace_root=workspace_root,
            contract_version=contract_version,
            action_ttl_seconds=action_ttl_seconds,
        )

    def ensure_bot(self, *, bot_id: str) -> None:
        with self._lock:
            existing = self._rules_by_bot.get(bot_id)
            if existing:
                self._breakers.ensure_bot(bot_id=bot_id)
                return

            defaults = self._default_rules()
            self._rules_by_bot[bot_id] = {item.rule_id: item for item in defaults}
            self._descriptors_by_bot[bot_id] = self._build_descriptors(defaults)
            self._recent_triggers.setdefault(bot_id, deque(maxlen=self._history_per_bot))
            self._breakers.ensure_bot(bot_id=bot_id)

    def upsert_rule(self, *, bot_id: str, rule: ReflexRule) -> None:
        self.ensure_bot(bot_id=bot_id)
        with self._lock:
            rules = self._rules_by_bot.setdefault(bot_id, {})
            rules[rule.rule_id] = rule
            self._descriptors_by_bot[bot_id] = self._build_descriptors(list(rules.values()))

    def set_rule_enabled(self, *, bot_id: str, rule_id: str, enabled: bool) -> bool:
        self.ensure_bot(bot_id=bot_id)
        with self._lock:
            rules = self._rules_by_bot.get(bot_id, {})
            current = rules.get(rule_id)
            if current is None:
                return False
            updated = current.model_copy(update={"enabled": enabled})
            rules[rule_id] = updated
            self._descriptors_by_bot[bot_id] = self._build_descriptors(list(rules.values()))
            return True

    def list_rules(self, *, bot_id: str) -> list[ReflexRule]:
        self.ensure_bot(bot_id=bot_id)
        with self._lock:
            rules = list(self._rules_by_bot.get(bot_id, {}).values())
        rules.sort(key=lambda item: (item.priority, item.rule_id))
        return rules

    def list_breakers(self, *, bot_id: str):
        self.ensure_bot(bot_id=bot_id)
        return self._breakers.statuses(bot_id=bot_id)

    def recent_triggers(self, *, bot_id: str, limit: int = 100) -> list[ReflexTriggerRecord]:
        self.ensure_bot(bot_id=bot_id)
        with self._lock:
            rows = list(self._recent_triggers.get(bot_id, deque()))
        rows = rows[-max(1, limit) :]
        rows.reverse()
        return rows

    def evaluate_events(
        self,
        *,
        bot_id: str,
        events: list[NormalizedEvent],
        get_enriched_state,
        queue_action,
        publish_macros,
        get_planner_context=None,
    ) -> list[ReflexTriggerRecord]:
        self.ensure_bot(bot_id=bot_id)

        output: list[ReflexTriggerRecord] = []
        if not events:
            return output

        for event in events:
            state = get_enriched_state(bot_id=bot_id)
            planner_context = get_planner_context(bot_id=bot_id) if callable(get_planner_context) else {}
            facts = self._build_fact_map(event=event, state=state, planner_context=planner_context)
            descriptors = self._candidate_descriptors(bot_id=bot_id, event=event)
            override_emitted = False
            for descriptor in descriptors:
                rule = descriptor.rule

                if override_emitted:
                    output.append(
                        self._record_trigger(
                            bot_id=bot_id,
                            rule=rule,
                            event=event,
                            elapsed_ms=0.0,
                            suppressed=True,
                            suppression_reason="override_in_effect",
                            emitted=False,
                            execution_target=None,
                            action_id=None,
                            outcome="suppressed",
                            detail="higher-priority override reflex already emitted for this event",
                        )
                    )
                    continue

                decision = self._matcher.evaluate(bot_id=bot_id, rule=rule, facts=facts)
                if not decision.matched:
                    if decision.suppressed:
                        output.append(
                            self._record_trigger(
                                bot_id=bot_id,
                                rule=rule,
                                event=event,
                                elapsed_ms=decision.elapsed_ms,
                                suppressed=True,
                                suppression_reason=decision.reason,
                                emitted=False,
                                execution_target=None,
                                action_id=None,
                                outcome="suppressed",
                                detail=decision.reason,
                            )
                        )
                    continue

                reserved, reservation_reason = self._reserve_conflict_key(bot_id=bot_id, rule=rule)
                if not reserved:
                    output.append(
                        self._record_trigger(
                            bot_id=bot_id,
                            rule=rule,
                            event=event,
                            elapsed_ms=decision.elapsed_ms,
                            suppressed=True,
                            suppression_reason=reservation_reason,
                            emitted=False,
                            execution_target=None,
                            action_id=None,
                            outcome="suppressed",
                            detail=reservation_reason,
                        )
                    )
                    continue

                breaker_key = rule.circuit_breaker_key or "queue.default"
                breaker_family = self._breaker_family(breaker_key)
                allowed, allow_reason = self._breakers.allow(bot_id=bot_id, key=breaker_key, family=breaker_family)
                if not allowed:
                    output.append(
                        self._record_trigger(
                            bot_id=bot_id,
                            rule=rule,
                            event=event,
                            elapsed_ms=decision.elapsed_ms,
                            suppressed=True,
                            suppression_reason=allow_reason,
                            emitted=False,
                            execution_target=None,
                            action_id=None,
                            outcome="suppressed",
                            detail=allow_reason,
                        )
                    )
                    continue

                trigger_id = f"rfx-{uuid4().hex[:20]}"
                started_emit = perf_counter()
                outcome = self._emitter.emit_chain(
                    bot_id=bot_id,
                    rule=rule,
                    trigger_id=trigger_id,
                    queue_action=queue_action,
                    publish_macros=publish_macros,
                )
                elapsed_ms = decision.elapsed_ms + ((perf_counter() - started_emit) * 1000.0)

                if outcome.emitted:
                    self._matcher.mark_fired(bot_id=bot_id, rule_id=rule.rule_id)
                    self._remember_pending_outcome(
                        action_id=outcome.action_id,
                        bot_id=bot_id,
                        trigger_id=trigger_id,
                        rule_id=rule.rule_id,
                        breaker_key=breaker_key,
                        breaker_family=breaker_family,
                    )
                    record = self._record_trigger(
                        bot_id=bot_id,
                        rule=rule,
                        event=event,
                        elapsed_ms=elapsed_ms,
                        suppressed=False,
                        suppression_reason="",
                        emitted=True,
                        execution_target=outcome.execution_target,
                        action_id=outcome.action_id,
                        outcome="emitted",
                        detail=outcome.reason,
                        trigger_id=trigger_id,
                    )
                    output.append(record)
                    if rule.planner_interop == ReflexPlannerInterop.override:
                        override_emitted = True
                    continue

                self._breakers.record_failure(
                    bot_id=bot_id,
                    key=breaker_key,
                    family=breaker_family,
                    reason=outcome.reason,
                )
                output.append(
                    self._record_trigger(
                        bot_id=bot_id,
                        rule=rule,
                        event=event,
                        elapsed_ms=elapsed_ms,
                        suppressed=True,
                        suppression_reason="emit_failed",
                        emitted=False,
                        execution_target=outcome.execution_target,
                        action_id=outcome.action_id,
                        outcome="emit_failed",
                        detail=outcome.reason,
                    )
                )

        return output

    def handle_ack(
        self,
        *,
        bot_id: str,
        action_id: str,
        success: bool,
        result_code: str,
        message: str,
    ) -> None:
        with self._lock:
            pending = self._pending_outcomes.pop(action_id, None)
        if pending is None:
            return

        if success:
            self._breakers.record_success(bot_id=bot_id, key=pending.breaker_key, family=pending.breaker_family)
        else:
            self._breakers.record_failure(
                bot_id=bot_id,
                key=pending.breaker_key,
                family=pending.breaker_family,
                reason=f"ack_failed:{result_code}:{message}",
            )

        # Feed acknowledgment outcome into outcome tracking
        try:
            self.record_outcome(bot_id=bot_id, rule_id=pending.rule_id, success=success)
        except Exception:
            logger.exception("handle_ack_outcome_tracking_failed")

        with self._lock:
            record = self._triggers_by_id.get((bot_id, pending.trigger_id))
            if record is None:
                return
            updated = record.model_copy(
                update={
                    "outcome": "ack_success" if success else "ack_failed",
                    "detail": f"result_code={result_code} message={message}",
                }
            )
            self._replace_trigger_record(bot_id=bot_id, trigger_id=pending.trigger_id, updated=updated)

    def _candidate_descriptors(self, *, bot_id: str, event: NormalizedEvent) -> list[_RuleDescriptor]:
        with self._lock:
            all_descriptors = list(self._descriptors_by_bot.get(bot_id, []))

        if not all_descriptors:
            return []

        event_type = event.event_type
        selected: list[_RuleDescriptor] = []
        for item in all_descriptors:
            if not item.event_type_hints or event_type in item.event_type_hints:
                selected.append(item)
        return selected

    def _build_descriptors(self, rules: list[ReflexRule]) -> list[_RuleDescriptor]:
        sorted_rules = sorted(
            rules,
            key=lambda item: (
                item.priority,
                self._CATEGORY_ORDER.get(item.category, 99),
                item.rule_id,
            ),
        )
        out: list[_RuleDescriptor] = []
        for rule in sorted_rules:
            hints = self._extract_event_hints(rule)
            out.append(_RuleDescriptor(rule=rule, event_type_hints=hints))
        return out

    def _extract_event_hints(self, rule: ReflexRule) -> set[str]:
        hint_facts = {"event.event_type", "event_type"}
        predicates = list(rule.trigger.all) + list(rule.trigger.any)
        out: set[str] = set()
        for predicate in predicates:
            if predicate.fact not in hint_facts:
                continue
            if predicate.op.value == "eq" and isinstance(predicate.value, str):
                out.add(predicate.value)
                continue
            if predicate.op.value == "in" and isinstance(predicate.value, list):
                for item in predicate.value:
                    if isinstance(item, str):
                        out.add(item)
        return out

    def _build_fact_map(
        self,
        *,
        event: NormalizedEvent,
        state: EnrichedWorldState,
        planner_context: dict[str, object],
    ) -> dict[str, object]:
        facts: dict[str, object] = {
            "event.event_id": event.event_id,
            "event.event_type": event.event_type,
            "event.event_family": event.event_family.value,
            "event.severity": event.severity.value,
            "event.source_hook": event.source_hook,
            "event.text": event.text,
            "event_type": event.event_type,
            "event_family": event.event_family.value,
            "bot_id": state.bot_id,
        }

        event_payload = event.model_dump(mode="json")
        state_payload = state.model_dump(mode="json")

        self._flatten(prefix="event", value=event_payload, out=facts)
        self._flatten(prefix="state", value=state_payload, out=facts)
        self._flatten(prefix="", value=state_payload, out=facts)

        planner_payload = planner_context if isinstance(planner_context, dict) else {}
        self._flatten(prefix="planner", value=planner_payload, out=facts)

        facts["planner.active"] = bool(planner_payload.get("active", False))
        facts["planner.current_horizon"] = planner_payload.get("current_horizon", "")
        facts["planner.current_objective"] = planner_payload.get("current_objective", "")
        facts["planner.last_plan_id"] = planner_payload.get("last_plan_id", "")
        facts["planner.queue_depth"] = planner_payload.get("queue_depth", 0)

        hp = self._safe_float(facts.get("operational.hp"))
        hp_max = self._safe_float(facts.get("operational.hp_max"))
        hp_ratio = (hp / hp_max) if hp_max > 0 else 1.0
        facts["vitals.hp_ratio"] = hp_ratio
        facts["combat.is_in_combat"] = bool(facts.get("operational.in_combat"))
        facts["inventory.overweight_ratio"] = facts.get("inventory.overweight_ratio")
        facts["social.private_messages_5m"] = facts.get("social.private_messages_5m", 0)
        facts["risk.death_risk_score"] = facts.get("risk.death_risk_score", 0.0)
        facts["risk.danger_score"] = facts.get("risk.danger_score", 0.0)

        # --- Progression shorthands (alias deep paths for ergonomic rule authoring) ---
        facts["state.base_level"] = facts.get("operational.base_level")
        facts["state.job_level"] = facts.get("operational.job_level")
        facts["state.job_id"] = facts.get("operational.job_id")
        facts["state.skill_points"] = facts.get("operational.skill_points")
        facts["state.stat_points"] = facts.get("operational.stat_points")
        facts["state.skill_points_delta"] = facts.get("event.numeric.skill_points_delta", 0.0)
        facts["state.stat_points_delta"] = facts.get("event.numeric.stat_points_delta", 0.0)
        facts["state.skill_points_newly_pending"] = facts.get("event.numeric.skill_points_newly_pending", 0.0)
        facts["state.stat_points_newly_pending"] = facts.get("event.numeric.stat_points_newly_pending", 0.0)
        facts["state.job_name"] = facts.get("operational.job_name")

        # --- Encounter shorthands ---
        facts["encounter.nearby_hostiles"] = facts.get("encounter.nearby_hostiles", 0)
        facts["encounter.nearby_allies"] = facts.get("encounter.nearby_allies", 0)
        facts["encounter.risk_score"] = facts.get("encounter.risk_score", 0.0)
        facts["encounter.in_encounter"] = facts.get("encounter.in_encounter", False)

        return facts


    def _flatten(self, *, prefix: str, value: object, out: dict[str, object]) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                key_str = str(key)
                target = f"{prefix}.{key_str}" if prefix else key_str
                self._flatten(prefix=target, value=item, out=out)
            return
        if isinstance(value, list):
            out[prefix] = value
            return
        out[prefix] = value

    def _reserve_conflict_key(self, *, bot_id: str, rule: ReflexRule) -> tuple[bool, str]:
        conflict_key = rule.action_template.conflict_key
        if not conflict_key:
            return True, "reserved"

        now_ms = perf_counter() * 1000.0
        ttl_ms = float(max(100, min(rule.cooldown_ms if rule.cooldown_ms > 0 else 250, 1000)))
        ident = (bot_id, conflict_key)
        with self._lock:
            stale = [item for item, ts in self._conflict_reservations_ms.items() if ts <= now_ms]
            for item in stale:
                del self._conflict_reservations_ms[item]

            existing = self._conflict_reservations_ms.get(ident)
            if existing is not None and existing > now_ms:
                remaining = int(existing - now_ms)
                return False, f"conflict_reserved:{max(0, remaining)}"

            self._conflict_reservations_ms[ident] = now_ms + ttl_ms
            return True, "reserved"

    def _record_trigger(
        self,
        *,
        bot_id: str,
        rule: ReflexRule,
        event: NormalizedEvent,
        elapsed_ms: float,
        suppressed: bool,
        suppression_reason: str,
        emitted: bool,
        execution_target: str | None,
        action_id: str | None,
        outcome: str,
        detail: str,
        trigger_id: str | None = None,
    ) -> ReflexTriggerRecord:
        record = ReflexTriggerRecord(
            trigger_id=trigger_id or f"rfx-{uuid4().hex[:20]}",
            bot_id=bot_id,
            rule_id=rule.rule_id,
            priority=rule.priority,
            category=rule.category,
            planner_interop=rule.planner_interop,
            event_id=event.event_id,
            event_family=event.event_family.value,
            event_type=event.event_type,
            matched_at=datetime.now(UTC),
            latency_ms=float(elapsed_ms),
            suppressed=suppressed,
            suppression_reason=suppression_reason,
            emitted=emitted,
            execution_target=execution_target,
            action_id=action_id,
            outcome=outcome,
            detail=detail,
        )
        with self._lock:
            bucket = self._recent_triggers.setdefault(bot_id, deque(maxlen=self._history_per_bot))
            bucket.append(record)
            self._triggers_by_id[(bot_id, record.trigger_id)] = record
        return record

    def _replace_trigger_record(self, *, bot_id: str, trigger_id: str, updated: ReflexTriggerRecord) -> None:
        bucket = self._recent_triggers.get(bot_id)
        if bucket is None:
            self._triggers_by_id[(bot_id, trigger_id)] = updated
            return

        rebuilt = deque(maxlen=bucket.maxlen)
        for item in bucket:
            if item.trigger_id == trigger_id:
                rebuilt.append(updated)
            else:
                rebuilt.append(item)
        self._recent_triggers[bot_id] = rebuilt
        self._triggers_by_id[(bot_id, trigger_id)] = updated

    def _remember_pending_outcome(
        self,
        *,
        action_id: str | None,
        bot_id: str,
        trigger_id: str,
        rule_id: str,
        breaker_key: str,
        breaker_family: str,
    ) -> None:
        if not action_id:
            return
        with self._lock:
            self._pending_outcomes[action_id] = _PendingOutcome(
                bot_id=bot_id,
                trigger_id=trigger_id,
                rule_id=rule_id,
                breaker_key=breaker_key,
                breaker_family=breaker_family,
            )

    def _breaker_family(self, breaker_key: str) -> str:
        if not breaker_key:
            return "queue"
        return breaker_key.split(".", 1)[0].strip().lower() or "queue"

    def _safe_float(self, value: object) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def load_rules_from_yaml(self, path: str | Path) -> int:
        """Load reflex rules from a YAML file. Returns number of rules loaded."""
        path = Path(path)
        if not path.exists():
            logger.warning("reflex_rules_yaml_not_found: %s", path)
            return 0
        try:
            import yaml
        except ImportError:
            logger.warning("reflex_rules_yaml_unavailable: PyYAML not installed, skipping YAML rules")
            return 0
        
        with open(path) as f:
            data = yaml.safe_load(f)
        
        if not isinstance(data, dict) or "rules" not in data:
            logger.warning("reflex_rules_yaml_invalid: missing 'rules' key in %s", path)
            return 0
        
        raw_rules = data["rules"]
        if not isinstance(raw_rules, list):
            return 0
        
        count = 0
        for raw in raw_rules:
            if not isinstance(raw, dict) or not raw.get("rule_id"):
                continue
            try:
                rule = self._yaml_to_rule(raw)
                self.upsert_rule(bot_id="default", rule=rule)
                count += 1
            except Exception as exc:
                logger.warning("reflex_rules_yaml_parse_failed: rule=%s error=%s", raw.get("rule_id"), exc)
        
        logger.info("reflex_rules_yaml_loaded: %d rules from %s", count, path)
        return count

    def _yaml_to_rule(self, raw: dict) -> ReflexRule:
        """Convert YAML rule dict to ReflexRule contract."""
        from ai_sidecar.contracts.reflex import (
            ReflexRule, ReflexTriggerClause, ReflexPredicate,
            ReflexActionTemplate, ReflexCategory, ReflexPlannerInterop,
        )
        
        trigger_raw = raw.get("trigger", {})
        all_preds = []
        for pred_raw in trigger_raw.get("all", []):
            all_preds.append(ReflexPredicate(
                fact=pred_raw["fact"],
                op=pred_raw["op"],
                value=pred_raw["value"],
            ))
        
        action_raw = raw.get("action", {})
        action = ReflexActionTemplate(
            kind=action_raw.get("kind", "command"),
            command=action_raw.get("command", ""),
            priority_tier=action_raw.get("priority_tier", "reflex"),
            conflict_key=action_raw.get("conflict_key") or "",
        )
        
        return ReflexRule(
            rule_id=raw["rule_id"],
            enabled=raw.get("enabled", True),
            priority=int(raw.get("priority", 10)),
            trigger=ReflexTriggerClause(all=all_preds),
            guards=[],
            action_template=action,
            fallback_macro=str(raw.get("fallback_macro") or "") if raw.get("fallback_macro") else None,
            cooldown_ms=int(raw.get("cooldown_ms", 1000)),
            circuit_breaker_key=raw.get("circuit_breaker_key", "default"),
            category=ReflexCategory(raw.get("category", "interaction")),
        )

    def record_outcome(self, *, bot_id: str, rule_id: str, success: bool) -> None:
        """Record the outcome of a fired reflex rule.
        
        Only counts explicit successes/failures from bot acknowledgements.
        Missing acks are NOT counted as failures — the action was queued
        successfully, the bot-side execution is a separate concern.
        """
        with self._lock:
            rules = self._rules_by_bot.get(bot_id)
            if rules and rule_id in rules:
                rule = rules[rule_id]
                rule.outcome_tracking["fires"] = rule.outcome_tracking.get("fires", 0) + 1
                if success:
                    rule.outcome_tracking["successes"] = rule.outcome_tracking.get("successes", 0) + 1
                else:
                    rule.outcome_tracking["failures"] = rule.outcome_tracking.get("failures", 0) + 1
                # Log high failure rates but NEVER auto-disable — let the LLM decide
                fires = rule.outcome_tracking.get("fires", 0)
                failures = rule.outcome_tracking.get("failures", 0)
                if fires >= 10 and failures / fires > 0.8:
                    logger.warning("reflex_rule_high_failure_rate: %s (%d/%d failures) — LLM will review on next kaizen cycle", rule_id, failures, fires)

    def outcome_stats(self, *, bot_id: str) -> dict[str, dict[str, int]]:
        """Return outcome stats for all rules for a bot."""
        with self._lock:
            rules = self._rules_by_bot.get(bot_id, {})
            return {rid: dict(r.outcome_tracking) for rid, r in rules.items()}

    def _default_rules(self) -> list[ReflexRule]:
        return [
            # ═══════════════════════════════════════════════════════════
            # SURVIVAL: Heal when HP low in combat
            # ═══════════════════════════════════════════════════════════
            ReflexRule(
                rule_id="emergency_heal_potion",
                enabled=True,
                priority=5,
                trigger=ReflexTriggerClause(
                    all=[
                        ReflexPredicate(fact="vitals.hp_ratio", op="lte", value=0.50),
                        ReflexPredicate(fact="combat.is_in_combat", op="eq", value=True),
                    ]
                ),
                guards=[],
                action_template=ReflexActionTemplate(
                    kind="command",
                    command="use red_potion",
                    priority_tier="reflex",
                    conflict_key="survival.heal",
                    metadata={"category": "emergency_heal"},
                ),
                fallback_macro="reflex_survival_heal",
                cooldown_ms=2000,
                circuit_breaker_key="combat.default",
                category=ReflexCategory.survival,
                planner_interop=ReflexPlannerInterop.override,
            ),
            # ═══════════════════════════════════════════════════════════
            # SURVIVAL: Heal when HP < 30% (deadly)
            # ═══════════════════════════════════════════════════════════
            ReflexRule(
                rule_id="emergency_orange_potion",
                enabled=True,
                priority=3,
                trigger=ReflexTriggerClause(
                    all=[
                        ReflexPredicate(fact="vitals.hp_ratio", op="lte", value=0.30),
                    ]
                ),
                guards=[],
                action_template=ReflexActionTemplate(
                    kind="command",
                    command="use orange_potion",
                    priority_tier="reflex",
                    conflict_key="survival.heal",
                    metadata={"category": "emergency_heal"},
                ),
                fallback_macro="reflex_survival_orange",
                cooldown_ms=2000,
                circuit_breaker_key="combat.default",
                category=ReflexCategory.survival,
                planner_interop=ReflexPlannerInterop.override,
            ),
            # ═══════════════════════════════════════════════════════════
            # SURVIVAL: Escape teleport at 15% HP (lethal)
            # ═══════════════════════════════════════════════════════════
            ReflexRule(
                rule_id="lethal_escape_teleport",
                enabled=True,
                priority=1,
                trigger=ReflexTriggerClause(
                    all=[
                        ReflexPredicate(fact="vitals.hp_ratio", op="lte", value=0.15),
                    ]
                ),
                guards=[],
                action_template=ReflexActionTemplate(
                    kind="command",
                    command="ai manual",
                    priority_tier="reflex",
                    conflict_key="survival.escape",
                    metadata={"category": "lethal_escape"},
                ),
                fallback_macro="reflex_survival_escape",
                cooldown_ms=5000,
                circuit_breaker_key="combat.default",
                category=ReflexCategory.survival,
                planner_interop=ReflexPlannerInterop.override,
            ),
            # ═══════════════════════════════════════════════════════════
            # SURVIVAL: Death recovery — respawn and return to hunting
            # ═══════════════════════════════════════════════════════════
            ReflexRule(
                rule_id="death_recovery",
                enabled=True,
                priority=1,
                trigger=ReflexTriggerClause(
                    any=[
                        ReflexPredicate(fact="event.event_type", op="eq", value="player_died"),
                        ReflexPredicate(fact="vitals.hp", op="eq", value=0),
                        ReflexPredicate(fact="state.is_dead", op="eq", value=True),
                    ]
                ),
                guards=[],
                action_template=ReflexActionTemplate(
                    kind="command",
                    command="ai auto",
                    priority_tier="reflex",
                    conflict_key="survival.respawn",
                    metadata={"category": "death_recovery", "action": "respawn_and_return"},
                ),
                fallback_macro="reflex_death_recovery",
                cooldown_ms=10000,
                circuit_breaker_key="queue.default",
                category=ReflexCategory.survival,
                planner_interop=ReflexPlannerInterop.override,
            ),
            # ═══════════════════════════════════════════════════════════
            # NAVIGATION: Route stuck recovery
            # ═══════════════════════════════════════════════════════════
            ReflexRule(
                rule_id="route_stuck_recovery",
                enabled=True,
                priority=20,
                trigger=ReflexTriggerClause(
                    any=[
                        ReflexPredicate(fact="navigation.stuck_score", op="gte", value=0.85),
                        ReflexPredicate(fact="event.event_type", op="eq", value="navigation.stuck"),
                    ]
                ),
                guards=[],
                action_template=ReflexActionTemplate(
                    kind="command",
                    command="ai auto",
                    priority_tier="reflex",
                    conflict_key="navigation.unstuck",
                    metadata={"category": "route_stuck"},
                ),
                fallback_macro="reflex_route_recovery",
                cooldown_ms=5000,
                circuit_breaker_key="queue.default",
                category=ReflexCategory.combat,
                planner_interop=ReflexPlannerInterop.complement,
            ),
            # ═══════════════════════════════════════════════════════════
            # INVENTORY: Weight overflow → return to town to sell
            # ═══════════════════════════════════════════════════════════
            ReflexRule(
                rule_id="weight_overflow_handling",
                enabled=True,
                priority=30,
                trigger=ReflexTriggerClause(
                    any=[
                        ReflexPredicate(fact="inventory.overweight_ratio", op="gte", value=0.9),
                    ]
                ),
                guards=[],
                action_template=ReflexActionTemplate(
                    kind="command",
                    command="move prontera",
                    priority_tier="reflex",
                    conflict_key="inventory.weight_guard",
                    metadata={"category": "weight_overflow", "target_town": "prontera"},
                ),
                fallback_macro="reflex_weight_overflow",
                cooldown_ms=4000,
                circuit_breaker_key="queue.default",
                category=ReflexCategory.survival,
                planner_interop=ReflexPlannerInterop.override,
            ),
            ReflexRule(
                rule_id="hostile_player_detection_response",
                enabled=True,
                priority=3,
                trigger=ReflexTriggerClause(
                    all=[
                        ReflexPredicate(fact="event.event_type", op="eq", value="actor.observed"),
                        ReflexPredicate(fact="event.payload.actor_type", op="eq", value="player"),
                    ],
                    any=[
                        ReflexPredicate(fact="event.payload.relation", op="in", value=["enemy", "hostile", "unknown"]),
                    ],
                ),
                guards=[],
                action_template=ReflexActionTemplate(
                    kind="command",
                    command="",
                    priority_tier="reflex",
                    conflict_key="social.escape",
                    metadata={
                        "category": "hostile_player",
                        "bridge_compat": {
                            "status": "suppressed",
                            "original_command": "teleport",
                            "reason": "bridge_root_not_allowed",
                            "fallback_strategy": "fallback_macro",
                        },
                    },
                ),
                fallback_macro="reflex_social_escape",
                cooldown_ms=10000,
                circuit_breaker_key="social.default",
                category=ReflexCategory.interaction,
                planner_interop=ReflexPlannerInterop.override,
            ),
            ReflexRule(
                rule_id="disconnect_relog_orchestration",
                enabled=True,
                priority=4,
                trigger=ReflexTriggerClause(
                    any=[
                        ReflexPredicate(fact="event.event_type", op="eq", value="lifecycle.disconnected"),
                    ]
                ),
                guards=[],
                action_template=ReflexActionTemplate(
                    kind="command",
                    command="",
                    priority_tier="reflex",
                    conflict_key="lifecycle.relog",
                    metadata={
                        "category": "disconnect_relog",
                        "bridge_compat": {
                            "status": "suppressed",
                            "original_command": "relog",
                            "reason": "bridge_root_not_allowed",
                            "fallback_strategy": "fallback_macro",
                        },
                    },
                ),
                fallback_macro="reflex_relog_sequence",
                cooldown_ms=15000,
                circuit_breaker_key="fleet.default",
                category=ReflexCategory.interaction,
                planner_interop=ReflexPlannerInterop.override,
            ),
            ReflexRule(
                rule_id="anti_loop_deadlock_escape",
                enabled=True,
                priority=25,
                trigger=ReflexTriggerClause(
                    any=[
                        ReflexPredicate(fact="event.event_type", op="eq", value="action.loop_detected"),
                        ReflexPredicate(fact="risk.anomaly_flags", op="contains", value="action.loop"),
                    ]
                ),
                guards=[],
                action_template=ReflexActionTemplate(
                    kind="command",
                    command="ai clear",
                    priority_tier="reflex",
                    conflict_key="runtime.deadlock_escape",
                    metadata={"category": "anti_loop"},
                ),
                fallback_macro="reflex_deadlock_escape",
                cooldown_ms=6000,
                circuit_breaker_key="queue.default",
                category=ReflexCategory.survival,
                planner_interop=ReflexPlannerInterop.override,
            ),
            ReflexRule(
                rule_id="macro_crash_fallback",
                enabled=True,
                priority=8,
                trigger=ReflexTriggerClause(
                    any=[
                        ReflexPredicate(fact="event.event_type", op="eq", value="macro.publish_failed"),
                        ReflexPredicate(fact="macro_execution.last_result", op="contains", value="failed"),
                    ]
                ),
                guards=[],
                action_template=ReflexActionTemplate(
                    kind="command",
                    command="macro reflex_macro_recovery",
                    priority_tier="reflex",
                    conflict_key="macro.recovery",
                    metadata={"category": "macro_crash_fallback"},
                ),
                fallback_macro="reflex_macro_recovery",
                cooldown_ms=8000,
                circuit_breaker_key="macro.default",
                event_macro_conditions=["OnCharLogIn"],
                category=ReflexCategory.interaction,
                planner_interop=ReflexPlannerInterop.complement,
            ),
            # --- Progression & encounter auto-responses ---
            ReflexRule(
                rule_id="mob_swarm_combat_stance",
                enabled=True,
                priority=15,
                trigger=ReflexTriggerClause(
                    all=[
                        ReflexPredicate(fact="encounter.nearby_hostiles", op="gte", value=3),
                        ReflexPredicate(fact="combat.is_in_combat", op="eq", value=False),
                    ]
                ),
                guards=[],
                action_template=ReflexActionTemplate(
                    kind="command",
                    command="ai auto",
                    priority_tier="reflex",
                    conflict_key="encounter.combat_stance",
                    metadata={
                        "category": "mob_swarm",
                        "bridge_compat": {
                            "status": "rewritten",
                            "original_command": "do ai auto",
                            "rewritten_command": "ai auto",
                            "reason": "bridge_root_not_allowed",
                        },
                    },
                ),
                fallback_macro="reflex_mob_swarm_combat",
                cooldown_ms=5000,
                circuit_breaker_key="combat.default",
                category=ReflexCategory.combat,
                planner_interop=ReflexPlannerInterop.complement,
            ),
            ReflexRule(
                rule_id="extreme_overweight_sell_run",
                enabled=True,
                priority=22,
                trigger=ReflexTriggerClause(
                    all=[
                        ReflexPredicate(fact="inventory.overweight_ratio", op="gte", value=0.95),
                    ]
                ),
                guards=[],
                action_template=ReflexActionTemplate(
                    kind="command",
                    command="ai manual",
                    priority_tier="reflex",
                    conflict_key="inventory.extreme_weight_guard",
                    metadata={
                        "category": "extreme_overweight",
                        "bridge_compat": {
                            "status": "rewritten",
                            "original_command": "sit",
                            "rewritten_command": "ai manual",
                            "reason": "bridge_root_not_allowed",
                        },
                    },
                ),
                fallback_macro="reflex_extreme_weight_sell",
                cooldown_ms=10000,
                circuit_breaker_key="queue.default",
                category=ReflexCategory.survival,
                planner_interop=ReflexPlannerInterop.override,
            ),
            ReflexRule(
                rule_id="skill_points_available_alert",
                enabled=True,
                priority=50,
                trigger=ReflexTriggerClause(
                    all=[
                        ReflexPredicate(fact="event.event_type", op="eq", value="snapshot.compact"),
                        ReflexPredicate(fact="state.skill_points", op="gte", value=1),
                        ReflexPredicate(fact="state.skill_points_newly_pending", op="gte", value=1),
                        ReflexPredicate(fact="state.base_level", op="gte", value=2),
                    ]
                ),
                guards=[],
                action_template=ReflexActionTemplate(
                    kind="command",
                    command="",
                    priority_tier="tactical",
                    conflict_key="progression.skill_points_pending",
                    metadata={"category": "skill_points_available"},
                ),
                fallback_macro="reflex_skill_point_allocation",
                cooldown_ms=60000,
                circuit_breaker_key="queue.default",
                category=ReflexCategory.interaction,
                planner_interop=ReflexPlannerInterop.complement,
            ),
            # --- Equipment, party, market, NPC dialogue, stat/skill, PvP escape ---
            ReflexRule(
                rule_id="equipment_auto_swap",
                enabled=True,
                priority=18,
                trigger=ReflexTriggerClause(
                    any=[
                        ReflexPredicate(fact="inventory.recommended_equip", op="neq", value=""),
                        ReflexPredicate(fact="event.event_type", op="eq", value="equipment.update"),
                    ]
                ),
                guards=[],
                action_template=ReflexActionTemplate(
                    kind="command",
                    command="",
                    priority_tier="reflex",
                    conflict_key="inventory.equip",
                    metadata={
                        "category": "equipment",
                        "bridge_compat": {
                            "status": "suppressed",
                            "original_command": "equip",
                            "reason": "bridge_root_not_allowed",
                            "fallback_strategy": "fallback_macro",
                        },
                    },
                ),
                fallback_macro="reflex_equipment_swap",
                cooldown_ms=8000,
                circuit_breaker_key="queue.default",
                category=ReflexCategory.interaction,
                planner_interop=ReflexPlannerInterop.complement,
            ),
            ReflexRule(
                rule_id="party_coordination_reflex",
                enabled=True,
                priority=24,
                trigger=ReflexTriggerClause(
                    any=[
                        ReflexPredicate(fact="event.event_type", op="eq", value="party.invite"),
                        ReflexPredicate(fact="event.event_type", op="eq", value="party.request"),
                    ]
                ),
                guards=[],
                action_template=ReflexActionTemplate(
                    kind="command",
                    command="",
                    priority_tier="reflex",
                    conflict_key="social.party",
                    metadata={
                        "category": "party_coordination",
                        "bridge_compat": {
                            "status": "suppressed",
                            "original_command": "party",
                            "reason": "bridge_root_not_allowed",
                            "fallback_strategy": "fallback_macro",
                        },
                    },
                ),
                fallback_macro="reflex_party_coordination",
                cooldown_ms=12000,
                circuit_breaker_key="social.default",
                category=ReflexCategory.interaction,
                planner_interop=ReflexPlannerInterop.complement,
            ),
            ReflexRule(
                rule_id="market_overflow_guard",
                enabled=True,
                priority=21,
                trigger=ReflexTriggerClause(
                    all=[
                        ReflexPredicate(fact="inventory.overweight_ratio", op="gte", value=0.88),
                        ReflexPredicate(fact="economy.inventory_value", op="gte", value=1),
                    ]
                ),
                guards=[],
                action_template=ReflexActionTemplate(
                    kind="command",
                    command="",
                    priority_tier="reflex",
                    conflict_key="economy.market_guard",
                    metadata={
                        "category": "market_ops",
                        "bridge_compat": {
                            "status": "suppressed",
                            "original_command": "storage",
                            "reason": "bridge_root_not_allowed",
                            "fallback_strategy": "fallback_macro",
                        },
                    },
                ),
                fallback_macro="reflex_market_safety",
                cooldown_ms=10000,
                circuit_breaker_key="queue.default",
                category=ReflexCategory.survival,
                planner_interop=ReflexPlannerInterop.override,
            ),
            ReflexRule(
                rule_id="npc_dialogue_retry",
                enabled=True,
                priority=12,
                trigger=ReflexTriggerClause(
                    any=[
                        ReflexPredicate(fact="event.event_type", op="eq", value="npc.dialogue_failed"),
                        ReflexPredicate(fact="event.event_type", op="eq", value="npc.missing"),
                    ]
                ),
                guards=[],
                action_template=ReflexActionTemplate(
                    kind="command",
                    command="macro reflex_npc_dialogue_recover",
                    priority_tier="reflex",
                    conflict_key="npc.dialogue_retry",
                    metadata={"category": "npc_dialogue"},
                ),
                fallback_macro="reflex_npc_dialogue_recover",
                cooldown_ms=7000,
                circuit_breaker_key="npc.default",
                category=ReflexCategory.interaction,
                planner_interop=ReflexPlannerInterop.complement,
            ),
            ReflexRule(
                rule_id="stat_skill_auto_allocation",
                enabled=True,
                priority=26,
                trigger=ReflexTriggerClause(
                    all=[
                        ReflexPredicate(fact="event.event_type", op="eq", value="snapshot.compact"),
                        ReflexPredicate(fact="combat.is_in_combat", op="eq", value=False),
                    ],
                    any=[
                        ReflexPredicate(fact="state.skill_points_newly_pending", op="gte", value=1),
                        ReflexPredicate(fact="state.stat_points_newly_pending", op="gte", value=1),
                    ]
                ),
                guards=[],
                action_template=ReflexActionTemplate(
                    kind="command",
                    command="",
                    priority_tier="reflex",
                    conflict_key="progression.auto_allocate",
                    metadata={
                        "category": "stat_skill",
                        "bridge_compat": {
                            "status": "suppressed",
                            "original_command": "skills",
                            "reason": "bridge_root_not_allowed",
                            "fallback_strategy": "fallback_macro",
                        },
                    },
                ),
                fallback_macro="reflex_stat_skill_allocation",
                cooldown_ms=45000,
                circuit_breaker_key="queue.default",
                category=ReflexCategory.interaction,
                planner_interop=ReflexPlannerInterop.complement,
            ),
            ReflexRule(
                rule_id="pvp_escape_extended",
                enabled=False,
                priority=2,
                trigger=ReflexTriggerClause(
                    all=[
                        ReflexPredicate(fact="event.event_type", op="eq", value="actor.observed"),
                        ReflexPredicate(fact="event.payload.actor_type", op="eq", value="player"),
                    ],
                    any=[
                        ReflexPredicate(fact="encounter.risk_score", op="gte", value=0.75),
                        ReflexPredicate(fact="risk.anomaly_flags", op="contains", value="pvp.threat"),
                    ],
                ),
                guards=[],
                action_template=ReflexActionTemplate(
                    kind="command",
                    command="",
                    priority_tier="reflex",
                    conflict_key="social.escape.extended",
                    metadata={
                        "category": "pvp_escape",
                        "bridge_compat": {
                            "status": "suppressed",
                            "original_command": "teleport",
                            "reason": "bridge_root_not_allowed",
                            "fallback_strategy": "fallback_macro",
                        },
                    },
                ),
                fallback_macro="reflex_pvp_escape",
                cooldown_ms=12000,
                circuit_breaker_key="social.default",
                category=ReflexCategory.survival,
                planner_interop=ReflexPlannerInterop.override,
            ),
        ]
