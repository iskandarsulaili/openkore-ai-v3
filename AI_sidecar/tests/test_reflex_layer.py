from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta

from ai_sidecar.contracts.actions import ActionPriorityTier, ActionProposal
from ai_sidecar.contracts.common import ContractMeta
from ai_sidecar.contracts.events import EventFamily, EventSeverity, NormalizedEvent
from ai_sidecar.contracts.reflex import (
    ReflexActionTemplate,
    ReflexCategory,
    ReflexPlannerInterop,
    ReflexPredicate,
    ReflexRule,
    ReflexTriggerClause,
)
from ai_sidecar.contracts.state_graph import (
    BotOperationalState,
    EconomyState,
    EncounterState,
    EnrichedWorldState,
    FleetIntentState,
    InventoryState,
    LearningFeatureVector,
    MacroExecutionState,
    NavigationState,
    QuestState,
    RiskState,
    SocialState,
)
from ai_sidecar.domain.macro_compiler import MacroCompiler, MacroPublisher
from ai_sidecar.reflex.rule_engine import ReflexRuleEngine
from ai_sidecar.runtime.action_queue import ActionQueue


def _make_state(bot_id: str, *, hp: int = 100, hp_max: int = 100, in_combat: bool = False) -> EnrichedWorldState:
    return EnrichedWorldState(
        bot_id=bot_id,
        operational=BotOperationalState(bot_id=bot_id, hp=hp, hp_max=hp_max, in_combat=in_combat),
        encounter=EncounterState(),
        navigation=NavigationState(),
        quest=QuestState(),
        inventory=InventoryState(overweight_ratio=0.1),
        economy=EconomyState(),
        social=SocialState(),
        risk=RiskState(),
        macro_execution=MacroExecutionState(),
        fleet_intent=FleetIntentState(),
        features=LearningFeatureVector(),
    )


def _make_event(bot_id: str, *, event_type: str, family: EventFamily = EventFamily.system) -> NormalizedEvent:
    return NormalizedEvent(
        meta=ContractMeta(contract_version="v1", source="pytest", bot_id=bot_id),
        event_family=family,
        event_type=event_type,
        observed_at=datetime.now(UTC),
        severity=EventSeverity.info,
        payload={},
        tags={},
        numeric={},
        text=event_type,
    )


def _callbacks(workspace_root, queue: ActionQueue):
    compiler = MacroCompiler()
    publisher = MacroPublisher(workspace_root=workspace_root)

    def queue_action(proposal, bot_id):
        return queue.enqueue(bot_id, proposal)

    def publish_macros(request):
        compiled = compiler.compile(
            macros=request.macros,
            event_macros=request.event_macros,
            automacros=request.automacros,
        )
        publisher.publish(compiled)
        return True, {"version": compiled.version}, "published"

    return queue_action, publish_macros


def test_reflex_latency_under_load(tmp_path):
    bot_id = "bot:latency"
    queue = ActionQueue(max_per_bot=256)
    queue_action, publish_macros = _callbacks(tmp_path, queue)
    engine = ReflexRuleEngine(workspace_root=tmp_path, contract_version="v1", action_ttl_seconds=20)

    rule = ReflexRule(
        rule_id="latency.rule",
        enabled=True,
        priority=1,
        trigger=ReflexTriggerClause(all=[ReflexPredicate(fact="event.event_type", op="eq", value="unit.latency")]),
        guards=[],
        action_template=ReflexActionTemplate(kind="command", command="sit", conflict_key="latency.key"),
        cooldown_ms=2000,
        circuit_breaker_key="queue.default",
    )
    engine.upsert_rule(bot_id=bot_id, rule=rule)

    events = [_make_event(bot_id, event_type="unit.latency") for _ in range(120)]
    started = time.perf_counter()
    records = engine.evaluate_events(
        bot_id=bot_id,
        events=events,
        get_enriched_state=lambda *, bot_id=bot_id: _make_state(bot_id),
        queue_action=queue_action,
        publish_macros=publish_macros,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    assert records
    assert max(item.latency_ms for item in records) < 100.0
    assert elapsed_ms < 1000.0


def test_reflex_conflict_resolution_between_rules(tmp_path):
    bot_id = "bot:conflict"
    queue = ActionQueue(max_per_bot=64)
    queue_action, publish_macros = _callbacks(tmp_path, queue)
    engine = ReflexRuleEngine(workspace_root=tmp_path, contract_version="v1", action_ttl_seconds=20)

    event = _make_event(bot_id, event_type="unit.conflict")
    rule_a = ReflexRule(
        rule_id="conflict.a",
        enabled=True,
        priority=1,
        trigger=ReflexTriggerClause(all=[ReflexPredicate(fact="event.event_type", op="eq", value="unit.conflict")]),
        guards=[],
        action_template=ReflexActionTemplate(kind="command", command="sit", conflict_key="same.conflict"),
        cooldown_ms=0,
        circuit_breaker_key="queue.default",
    )
    rule_b = rule_a.model_copy(update={"rule_id": "conflict.b", "priority": 2})
    engine.upsert_rule(bot_id=bot_id, rule=rule_a)
    engine.upsert_rule(bot_id=bot_id, rule=rule_b)

    records = engine.evaluate_events(
        bot_id=bot_id,
        events=[event],
        get_enriched_state=lambda *, bot_id=bot_id: _make_state(bot_id),
        queue_action=queue_action,
        publish_macros=publish_macros,
    )

    relevant = [item for item in records if item.rule_id in {"conflict.a", "conflict.b"}]
    emitted = [item for item in relevant if item.emitted]
    suppressed = [item for item in relevant if item.suppressed]
    assert len(emitted) == 1
    assert suppressed
    assert any(
        (item.suppression_reason or "").startswith("conflict_reserved")
        or (item.suppression_reason or "") == "override_in_effect"
        for item in suppressed
    )


def test_reflex_fallback_chain_micro_macro_then_eventmacro(tmp_path):
    bot_id = "bot:fallback"
    queue = ActionQueue(max_per_bot=64)
    queue_action, publish_macros = _callbacks(tmp_path, queue)
    engine = ReflexRuleEngine(workspace_root=tmp_path, contract_version="v1", action_ttl_seconds=20)

    rule_micro = ReflexRule(
        rule_id="chain.micro",
        enabled=True,
        priority=1,
        trigger=ReflexTriggerClause(all=[ReflexPredicate(fact="event.event_type", op="eq", value="unit.chain.micro")]),
        guards=[],
        action_template=ReflexActionTemplate(kind="command", command="", conflict_key="chain.micro"),
        fallback_macro="reflex_chain_micro",
        cooldown_ms=0,
        circuit_breaker_key="macro.default",
    )
    engine.upsert_rule(bot_id=bot_id, rule=rule_micro)

    records_micro = engine.evaluate_events(
        bot_id=bot_id,
        events=[_make_event(bot_id, event_type="unit.chain.micro")],
        get_enriched_state=lambda *, bot_id=bot_id: _make_state(bot_id),
        queue_action=queue_action,
        publish_macros=publish_macros,
    )
    emitted_micro = [item for item in records_micro if item.rule_id == "chain.micro" and item.emitted]
    assert emitted_micro
    assert emitted_micro[0].execution_target == "published_micro_macro"
    queued_micro = queue.fetch_next(bot_id)
    assert queued_micro is not None
    assert queued_micro.command == "macro reflex_chain_micro"

    rule_event = ReflexRule(
        rule_id="chain.event",
        enabled=True,
        priority=1,
        trigger=ReflexTriggerClause(all=[ReflexPredicate(fact="event.event_type", op="eq", value="unit.chain.event")]),
        guards=[],
        action_template=ReflexActionTemplate(kind="command", command="", conflict_key="chain.event"),
        fallback_macro=None,
        event_macro_conditions=["OnCharLogIn"],
        cooldown_ms=0,
        circuit_breaker_key="macro.default",
    )
    engine.upsert_rule(bot_id=bot_id, rule=rule_event)

    records_event = engine.evaluate_events(
        bot_id=bot_id,
        events=[_make_event(bot_id, event_type="unit.chain.event")],
        get_enriched_state=lambda *, bot_id=bot_id: _make_state(bot_id),
        queue_action=queue_action,
        publish_macros=publish_macros,
    )
    emitted_event = [item for item in records_event if item.rule_id == "chain.event" and item.emitted]
    assert emitted_event
    assert emitted_event[0].execution_target == "eventmacro_trigger"
    queued_event = queue.fetch_next(bot_id)
    assert queued_event is not None
    assert queued_event.command == "eventMacro reflex_auto_chain.event"


def test_reflex_race_condition_prevention(tmp_path):
    bot_id = "bot:race"
    queue = ActionQueue(max_per_bot=512)
    queue_action, publish_macros = _callbacks(tmp_path, queue)
    engine = ReflexRuleEngine(workspace_root=tmp_path, contract_version="v1", action_ttl_seconds=20)

    race_rule = ReflexRule(
        rule_id="race.rule",
        enabled=True,
        priority=1,
        trigger=ReflexTriggerClause(all=[ReflexPredicate(fact="event.event_type", op="eq", value="unit.race")]),
        guards=[],
        action_template=ReflexActionTemplate(kind="command", command="sit", conflict_key="race.conflict"),
        cooldown_ms=1000,
        circuit_breaker_key="queue.default",
    )
    engine.upsert_rule(bot_id=bot_id, rule=race_rule)

    results = []
    lock = threading.Lock()

    def worker():
        recs = engine.evaluate_events(
            bot_id=bot_id,
            events=[_make_event(bot_id, event_type="unit.race")],
            get_enriched_state=lambda *, bot_id=bot_id: _make_state(bot_id),
            queue_action=queue_action,
            publish_macros=publish_macros,
        )
        with lock:
            results.extend(recs)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for item in threads:
        item.start()
    for item in threads:
        item.join()

    relevant = [item for item in results if item.rule_id == "race.rule"]
    emitted = [item for item in relevant if item.emitted]
    assert len(emitted) <= 1
    assert relevant


def test_reflex_circuit_breaker_opens_after_failures(tmp_path):
    bot_id = "bot:breaker"
    engine = ReflexRuleEngine(workspace_root=tmp_path, contract_version="v1", action_ttl_seconds=20)

    failing_rule = ReflexRule(
        rule_id="breaker.rule",
        enabled=True,
        priority=1,
        trigger=ReflexTriggerClause(all=[ReflexPredicate(fact="event.event_type", op="eq", value="unit.breaker")]),
        guards=[],
        action_template=ReflexActionTemplate(kind="command", command="", conflict_key=None),
        fallback_macro="reflex_breaker_macro",
        event_macro_conditions=[],
        cooldown_ms=0,
        circuit_breaker_key="macro.default",
    )
    engine.upsert_rule(bot_id=bot_id, rule=failing_rule)

    def queue_action(_proposal, _bot_id):
        return False, type("S", (), {"value": "dropped"})(), "", "forced_failure"

    def publish_macros(_request):
        return False, None, "forced_publish_failure"

    for _ in range(3):
        engine.evaluate_events(
            bot_id=bot_id,
            events=[_make_event(bot_id, event_type="unit.breaker")],
            get_enriched_state=lambda *, bot_id=bot_id: _make_state(bot_id),
            queue_action=queue_action,
            publish_macros=publish_macros,
        )

    records = engine.evaluate_events(
        bot_id=bot_id,
        events=[_make_event(bot_id, event_type="unit.breaker")],
        get_enriched_state=lambda *, bot_id=bot_id: _make_state(bot_id),
        queue_action=queue_action,
        publish_macros=publish_macros,
    )
    breaker_suppressed = [
        item
        for item in records
        if item.rule_id == "breaker.rule" and item.suppressed and (item.suppression_reason or "").startswith("breaker_open")
    ]
    assert breaker_suppressed

    breakers = engine.list_breakers(bot_id=bot_id)
    macro_default = [item for item in breakers if item.key == "macro.default"]
    assert macro_default
    assert macro_default[0].state == "open"


def test_reflex_default_rules_cover_all_categories(tmp_path):
    bot_id = "bot:categories"
    engine = ReflexRuleEngine(workspace_root=tmp_path, contract_version="v1", action_ttl_seconds=20)
    rules = engine.list_rules(bot_id=bot_id)
    categories = {item.category for item in rules}
    assert ReflexCategory.combat in categories
    assert ReflexCategory.survival in categories
    assert ReflexCategory.interaction in categories


def test_reflex_default_rules_bridge_compat_no_unsupported_direct_roots(tmp_path):
    bot_id = "bot:bridge-compat"
    engine = ReflexRuleEngine(workspace_root=tmp_path, contract_version="v1", action_ttl_seconds=20)
    rules = engine.list_rules(bot_id=bot_id)

    allowed_roots = {"ai", "move", "macro", "eventmacro", "talknpc", "take"}
    checked = 0
    for rule in rules:
        command = (rule.action_template.command or "").strip()
        if not command:
            compat = dict(rule.action_template.metadata).get("bridge_compat")
            if compat:
                assert compat.get("status") == "suppressed"
            continue
        checked += 1
        root = command.split(maxsplit=1)[0].strip().lower()
        assert root in allowed_roots
        if dict(rule.action_template.metadata).get("bridge_compat"):
            assert dict(rule.action_template.metadata)["bridge_compat"].get("status") in {"rewritten", "suppressed"}

    assert checked > 0


def test_reflex_default_rules_suppressed_direct_commands_still_have_macro_fallback(tmp_path):
    bot_id = "bot:suppressed-defaults"
    engine = ReflexRuleEngine(workspace_root=tmp_path, contract_version="v1", action_ttl_seconds=20)
    rules = engine.list_rules(bot_id=bot_id)

    suppressed = [
        item
        for item in rules
        if not (item.action_template.command or "").strip()
        and isinstance(dict(item.action_template.metadata).get("bridge_compat"), dict)
        and dict(item.action_template.metadata)["bridge_compat"].get("status") == "suppressed"
    ]

    assert suppressed
    assert all(bool(item.fallback_macro) for item in suppressed)


def test_reflex_override_interop_suppresses_later_rules_for_same_event(tmp_path):
    bot_id = "bot:override"
    queue = ActionQueue(max_per_bot=64)
    queue_action, publish_macros = _callbacks(tmp_path, queue)
    engine = ReflexRuleEngine(workspace_root=tmp_path, contract_version="v1", action_ttl_seconds=20)

    override_rule = ReflexRule(
        rule_id="override.first",
        enabled=True,
        priority=1,
        trigger=ReflexTriggerClause(all=[ReflexPredicate(fact="event.event_type", op="eq", value="unit.override")]),
        guards=[],
        action_template=ReflexActionTemplate(kind="command", command="sit", conflict_key="override.a"),
        cooldown_ms=0,
        circuit_breaker_key="queue.default",
        category=ReflexCategory.survival,
        planner_interop=ReflexPlannerInterop.override,
    )
    complement_rule = ReflexRule(
        rule_id="override.second",
        enabled=True,
        priority=2,
        trigger=ReflexTriggerClause(all=[ReflexPredicate(fact="event.event_type", op="eq", value="unit.override")]),
        guards=[],
        action_template=ReflexActionTemplate(kind="command", command="attack", conflict_key="override.b"),
        cooldown_ms=0,
        circuit_breaker_key="queue.default",
        category=ReflexCategory.combat,
        planner_interop=ReflexPlannerInterop.complement,
    )
    engine.upsert_rule(bot_id=bot_id, rule=override_rule)
    engine.upsert_rule(bot_id=bot_id, rule=complement_rule)

    records = engine.evaluate_events(
        bot_id=bot_id,
        events=[_make_event(bot_id, event_type="unit.override")],
        get_enriched_state=lambda *, bot_id=bot_id: _make_state(bot_id),
        queue_action=queue_action,
        publish_macros=publish_macros,
    )

    first = [item for item in records if item.rule_id == "override.first"]
    second = [item for item in records if item.rule_id == "override.second"]
    assert first and first[0].emitted
    assert second and second[0].suppressed
    assert second[0].suppression_reason == "override_in_effect"


def test_reflex_complement_interop_allows_multiple_rules(tmp_path):
    bot_id = "bot:complement"
    queue = ActionQueue(max_per_bot=64)
    queue_action, publish_macros = _callbacks(tmp_path, queue)
    engine = ReflexRuleEngine(workspace_root=tmp_path, contract_version="v1", action_ttl_seconds=20)

    first = ReflexRule(
        rule_id="complement.a",
        enabled=True,
        priority=1,
        trigger=ReflexTriggerClause(all=[ReflexPredicate(fact="event.event_type", op="eq", value="unit.complement")]),
        guards=[],
        action_template=ReflexActionTemplate(kind="command", command="sit", conflict_key=None),
        cooldown_ms=0,
        circuit_breaker_key="queue.default",
        planner_interop=ReflexPlannerInterop.complement,
    )
    second = ReflexRule(
        rule_id="complement.b",
        enabled=True,
        priority=2,
        trigger=ReflexTriggerClause(all=[ReflexPredicate(fact="event.event_type", op="eq", value="unit.complement")]),
        guards=[],
        action_template=ReflexActionTemplate(kind="command", command="attack", conflict_key=None),
        cooldown_ms=0,
        circuit_breaker_key="queue.default",
        planner_interop=ReflexPlannerInterop.complement,
    )
    engine.upsert_rule(bot_id=bot_id, rule=first)
    engine.upsert_rule(bot_id=bot_id, rule=second)

    records = engine.evaluate_events(
        bot_id=bot_id,
        events=[_make_event(bot_id, event_type="unit.complement")],
        get_enriched_state=lambda *, bot_id=bot_id: _make_state(bot_id),
        queue_action=queue_action,
        publish_macros=publish_macros,
    )

    emitted = [item for item in records if item.rule_id in {"complement.a", "complement.b"} and item.emitted]
    assert len(emitted) == 2


def test_reflex_uses_planner_context_facts(tmp_path):
    bot_id = "bot:planner-facts"
    queue = ActionQueue(max_per_bot=64)
    queue_action, publish_macros = _callbacks(tmp_path, queue)
    engine = ReflexRuleEngine(workspace_root=tmp_path, contract_version="v1", action_ttl_seconds=20)

    rule = ReflexRule(
        rule_id="planner.interrupt",
        enabled=True,
        priority=1,
        trigger=ReflexTriggerClause(
            all=[
                ReflexPredicate(fact="event.event_type", op="eq", value="unit.planner"),
                ReflexPredicate(fact="planner.active", op="eq", value=True),
                ReflexPredicate(fact="planner.current_horizon", op="eq", value="strategic"),
            ]
        ),
        guards=[],
        action_template=ReflexActionTemplate(kind="command", command="sit", conflict_key="planner.interrupt"),
        cooldown_ms=0,
        circuit_breaker_key="queue.default",
    )
    engine.upsert_rule(bot_id=bot_id, rule=rule)

    records = engine.evaluate_events(
        bot_id=bot_id,
        events=[_make_event(bot_id, event_type="unit.planner")],
        get_enriched_state=lambda *, bot_id=bot_id: _make_state(bot_id),
        queue_action=queue_action,
        publish_macros=publish_macros,
        get_planner_context=lambda *, bot_id=bot_id: {
            "active": True,
            "current_horizon": "strategic",
            "current_objective": "farm",
            "last_plan_id": "plan-1",
            "queue_depth": 2,
        },
    )
    assert any(item.rule_id == "planner.interrupt" and item.emitted for item in records)


def test_reflex_micro_and_eventmacro_plugin_override_from_metadata(tmp_path):
    bot_id = "bot:plugin-override"
    queue = ActionQueue(max_per_bot=64)
    queue_action, publish_macros = _callbacks(tmp_path, queue)
    engine = ReflexRuleEngine(workspace_root=tmp_path, contract_version="v1", action_ttl_seconds=20)

    micro_rule = ReflexRule(
        rule_id="plugin.micro",
        enabled=True,
        priority=1,
        trigger=ReflexTriggerClause(all=[ReflexPredicate(fact="event.event_type", op="eq", value="unit.plugin.micro")]),
        guards=[],
        action_template=ReflexActionTemplate(
            kind="command",
            command="",
            conflict_key="plugin.micro",
            metadata={"macro_plugin": "macroCustom"},
        ),
        fallback_macro="reflex_plugin_micro",
        cooldown_ms=0,
        circuit_breaker_key="macro.default",
    )
    engine.upsert_rule(bot_id=bot_id, rule=micro_rule)

    records_micro = engine.evaluate_events(
        bot_id=bot_id,
        events=[_make_event(bot_id, event_type="unit.plugin.micro")],
        get_enriched_state=lambda *, bot_id=bot_id: _make_state(bot_id),
        queue_action=queue_action,
        publish_macros=publish_macros,
    )
    assert any(item.rule_id == "plugin.micro" and item.emitted for item in records_micro)
    queued_micro = queue.fetch_next(bot_id)
    assert queued_micro is not None
    assert queued_micro.command == "macroCustom reflex_plugin_micro"

    event_rule = ReflexRule(
        rule_id="plugin.event",
        enabled=True,
        priority=1,
        trigger=ReflexTriggerClause(all=[ReflexPredicate(fact="event.event_type", op="eq", value="unit.plugin.event")]),
        guards=[],
        action_template=ReflexActionTemplate(
            kind="command",
            command="",
            conflict_key="plugin.event",
            metadata={"event_macro_plugin": "eventMacroCustom"},
        ),
        event_macro_conditions=["OnCharLogIn"],
        cooldown_ms=0,
        circuit_breaker_key="macro.default",
    )
    engine.upsert_rule(bot_id=bot_id, rule=event_rule)

    records_event = engine.evaluate_events(
        bot_id=bot_id,
        events=[_make_event(bot_id, event_type="unit.plugin.event")],
        get_enriched_state=lambda *, bot_id=bot_id: _make_state(bot_id),
        queue_action=queue_action,
        publish_macros=publish_macros,
    )
    assert any(item.rule_id == "plugin.event" and item.emitted for item in records_event)
    queued_event = queue.fetch_next(bot_id)
    assert queued_event is not None
    assert queued_event.command == "eventMacroCustom reflex_auto_plugin.event"


def test_reflex_eventmacro_route_uses_distinct_conflict_key_when_base_route_is_blocked(tmp_path):
    bot_id = "bot:eventmacro-conflict"
    queue = ActionQueue(max_per_bot=64)
    queue_action, publish_macros = _callbacks(tmp_path, queue)
    engine = ReflexRuleEngine(workspace_root=tmp_path, contract_version="v1", action_ttl_seconds=20)

    now = datetime.now(UTC)
    accepted, _, _, _ = queue.enqueue(
        bot_id,
        ActionProposal(
            action_id="existing-macro-recovery",
            kind="command",
            command="macro reflex_macro_recovery",
            priority_tier=ActionPriorityTier.reflex,
            conflict_key="macro.recovery",
            created_at=now,
            expires_at=now + timedelta(seconds=30),
            idempotency_key="existing:macro.recovery",
            metadata={"source": "reflex"},
        ),
    )
    assert accepted

    rule = ReflexRule(
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
            conflict_key="macro.recovery",
            metadata={"category": "macro_crash_fallback"},
        ),
        fallback_macro="reflex_macro_recovery",
        cooldown_ms=8000,
        circuit_breaker_key="macro.default",
        event_macro_conditions=["OnCharLogIn"],
        category=ReflexCategory.interaction,
        planner_interop=ReflexPlannerInterop.complement,
    )
    engine.upsert_rule(bot_id=bot_id, rule=rule)

    records = engine.evaluate_events(
        bot_id=bot_id,
        events=[_make_event(bot_id, event_type="macro.publish_failed")],
        get_enriched_state=lambda *, bot_id=bot_id: _make_state(bot_id),
        queue_action=queue_action,
        publish_macros=publish_macros,
    )

    emitted = [item for item in records if item.rule_id == "macro_crash_fallback" and item.emitted]
    assert emitted
    assert emitted[0].execution_target == "eventmacro_trigger"

    queued_actions = queue.snapshot()[bot_id]
    eventmacro_actions = [
        item for item in queued_actions if item.proposal.command == "eventMacro reflex_auto_macro_crash_fallback"
    ]
    assert len(eventmacro_actions) == 1
    assert eventmacro_actions[0].proposal.conflict_key == "macro.recovery.eventmacro"

    direct_or_micro_actions = [
        item for item in queued_actions if item.proposal.command == "macro reflex_macro_recovery"
    ]
    assert len(direct_or_micro_actions) == 1
