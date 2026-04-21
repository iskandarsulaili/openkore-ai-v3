from __future__ import annotations

import threading
import time
from datetime import UTC, datetime

from ai_sidecar.contracts.common import ContractMeta
from ai_sidecar.contracts.events import EventFamily, EventSeverity, NormalizedEvent
from ai_sidecar.contracts.reflex import ReflexActionTemplate, ReflexPredicate, ReflexRule, ReflexTriggerClause
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
    assert any((item.suppression_reason or "").startswith("conflict_reserved") for item in suppressed)


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
        fallback_macro=None,
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
