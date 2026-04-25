from __future__ import annotations

import re
import time
from datetime import UTC, datetime, timedelta

import pytest

from ai_sidecar.contracts.common import ContractMeta
from ai_sidecar.contracts.events import (
    ActorDeltaPushRequest,
    ActorObservation,
    EventFamily,
    IngestAcceptedResponse,
    NormalizedEvent,
)
from ai_sidecar.contracts.state import (
    ActorDigest,
    BotStateSnapshot,
    CombatState,
    InventoryDigest,
    InventoryItemDigest,
    MarketDigest,
    MarketQuoteDigest,
    NpcRelationshipDigest,
    Position,
    ProgressionDigest,
    QuestDigest,
    QuestObjectiveDigest,
    SkillDigest,
    Vitals,
)
from ai_sidecar.contracts.state_graph import LearningFeatureVector
from ai_sidecar.ingestion.adapters.actor_state_adapter import actor_delta_to_events
from ai_sidecar.lifecycle import RuntimeState, create_runtime
from ai_sidecar.state_graph.feature_extractor import FeatureExtractor
from ai_sidecar.state_graph.world_state import WorldStateProjector


def _meta(bot_id: str) -> ContractMeta:
    return ContractMeta(
        contract_version="v1",
        source="pytest",
        bot_id=bot_id,
        trace_id=f"trace-{bot_id.replace(':', '-')}",
    )


def _snapshot(
    *,
    bot_id: str,
    tick_id: str,
    observed_at: datetime,
    base_exp: int = 0,
    job_exp: int = 0,
    actors: list[ActorDigest] | None = None,
    raw: dict[str, object] | None = None,
) -> BotStateSnapshot:
    return BotStateSnapshot(
        meta=_meta(bot_id),
        tick_id=tick_id,
        observed_at=observed_at,
        position=Position(map="prt_fild08", x=100, y=120),
        vitals=Vitals(hp=800, hp_max=1000, sp=120, sp_max=200, weight=1000, weight_max=8000),
        combat=CombatState(ai_sequence="route", target_id=None, is_in_combat=False),
        inventory=InventoryDigest(zeny=5000, item_count=42),
        progression=ProgressionDigest(
            job_id=7,
            job_name="knight",
            base_level=45,
            job_level=20,
            base_exp=base_exp,
            base_exp_max=1000,
            job_exp=job_exp,
            job_exp_max=500,
            skill_points=3,
            stat_points=5,
        ),
        actors=list(actors or []),
        raw=dict(raw or {}),
    )


def test_actor_delta_to_events_emits_lifecycle_and_backward_compat() -> None:
    payload = ActorDeltaPushRequest(
        meta=_meta("bot:ws2-adapter"),
        observed_at=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        revision="tick-1",
        actors=[
            ActorObservation(
                actor_id="mob-1",
                actor_type="monster",
                name="Poring",
                map="prt_fild08",
                x=10,
                y=20,
                hp=50,
                hp_max=100,
                level=3,
                distance=5.0,
                relation="hostile",
                raw={},
            )
        ],
        removed_actor_ids=["mob-2"],
    )

    events = actor_delta_to_events(
        payload,
        appeared_actor_ids={"mob-1"},
        disappeared_actor_ids={"mob-2"},
    )

    event_types = [item.event_type for item in events]
    assert event_types == ["actor.batch", "actor.appeared", "actor.observed", "actor.disappeared", "actor.removed"]
    assert events[0].numeric["observed_count"] == pytest.approx(1.0)
    assert events[0].numeric["removed_count"] == pytest.approx(1.0)
    assert events[0].numeric["appeared_count"] == pytest.approx(1.0)
    assert events[0].numeric["disappeared_count"] == pytest.approx(1.0)
    assert events[0].numeric["hostile_count"] == pytest.approx(1.0)
    for item in events:
        assert item.payload.get("revision") == "tick-1"


def test_world_state_projector_recomputes_actor_truth_on_disappearance() -> None:
    bot_id = "bot:ws2-world"
    projector = WorldStateProjector()
    observed_at = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)

    snapshot_event = NormalizedEvent(
        meta=_meta(bot_id),
        observed_at=observed_at,
        event_family=EventFamily.snapshot,
        event_type="snapshot.compact",
        source_hook="pytest",
        payload={
            "position": {"map": "prt_fild08", "x": 100, "y": 100},
            "vitals": {"hp": 900, "hp_max": 1000},
            "combat": {"is_in_combat": False, "target_id": None, "ai_sequence": "idle"},
            "inventory": {"zeny": 1000, "item_count": 2},
            "progression": {"base_level": 10, "job_level": 5},
            "actors": [
                {"actor_id": "mob-1", "actor_type": "monster", "relation": "hostile"},
                {"actor_id": "player-1", "actor_type": "player", "relation": "party"},
            ],
        },
    )
    projector.observe_event(snapshot_event)

    first = projector.export(bot_id=bot_id, features=LearningFeatureVector())
    assert first["encounter"].nearby_hostiles == 1
    assert first["encounter"].nearby_allies == 1
    assert first["encounter"].in_encounter is True

    disappear = NormalizedEvent(
        meta=_meta(bot_id),
        observed_at=observed_at + timedelta(seconds=5),
        event_family=EventFamily.actor_state,
        event_type="actor.disappeared",
        source_hook="pytest",
        payload={"actor_id": "mob-1"},
    )
    projector.observe_event(disappear)

    second = projector.export(bot_id=bot_id, features=LearningFeatureVector())
    assert second["encounter"].nearby_hostiles == 0
    assert second["encounter"].nearby_allies == 1
    assert second["encounter"].in_encounter is False


def test_feature_extractor_tracks_exp_delta_and_latest_numeric_values() -> None:
    bot_id = "bot:ws2-features"
    extractor = FeatureExtractor()
    t0 = datetime.now(UTC) - timedelta(minutes=3)

    e1 = NormalizedEvent(
        meta=_meta(bot_id),
        observed_at=t0,
        event_family=EventFamily.snapshot,
        event_type="snapshot.compact",
        source_hook="pytest",
        numeric={"zeny": 1000.0, "base_exp": 100.0, "hp": 10.0},
        payload={"zeny": 1000, "base_exp": 100},
    )
    e2 = NormalizedEvent(
        meta=_meta(bot_id),
        observed_at=t0 + timedelta(minutes=3),
        event_family=EventFamily.snapshot,
        event_type="snapshot.compact",
        source_hook="pytest",
        numeric={"zeny": 1300.0, "base_exp": 250.0, "hp": 30.0},
        payload={"zeny": 1300, "base_exp": 250},
    )

    extractor.observe_event(e1)
    extractor.observe_event(e2)
    features = extractor.extract(bot_id=bot_id, basis={"basis.signal": 1.0})

    assert features.values["economy.zeny_delta_10m"] == pytest.approx(300.0)
    assert features.values["economy.exp_delta_10m"] == pytest.approx(150.0)
    assert features.values["event.snapshot.compact.hp"] == pytest.approx(30.0)


def test_world_state_projector_enriches_npc_economy_quest_and_skill_state() -> None:
    bot_id = "bot:wsA-world"
    projector = WorldStateProjector()
    t0 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)

    first = NormalizedEvent(
        meta=_meta(bot_id),
        observed_at=t0,
        event_family=EventFamily.snapshot,
        event_type="snapshot.compact",
        source_hook="pytest",
        payload={
            "position": {"map": "prt_fild08", "x": 100, "y": 100},
            "vitals": {"hp": 900, "hp_max": 1000, "sp": 120, "sp_max": 200, "weight": 1000, "weight_max": 8000},
            "combat": {"is_in_combat": False, "target_id": None, "ai_sequence": "idle"},
            "inventory": {"zeny": 1000, "item_count": 5},
            "inventory_items": [
                {"item_id": "red_potion", "name": "Red Potion", "category": "consumable", "quantity": 12, "sell_price": 25}
            ],
            "progression": {"base_level": 10, "job_level": 5, "skill_points": 3, "stat_points": 4},
            "skills": [
                {"skill_id": "AL_HEAL", "skill_name": "Heal", "level": 5},
                {"skill_id": "SM_BASH", "skill_name": "Bash", "level": 7},
            ],
            "quests": [
                {
                    "quest_id": "quest.alpha",
                    "state": "active",
                    "npc": "Tool Dealer",
                    "title": "Collect Apples",
                    "objectives": [
                        {"objective_id": "obj.1", "description": "Collect 10 apples", "status": "in_progress", "current": 2, "target": 10}
                    ],
                }
            ],
            "npc_relationships": [
                {
                    "npc_id": "npc:tool_dealer",
                    "npc_name": "Tool Dealer",
                    "relation": "quest",
                    "affinity_score": 0.3,
                    "trust_score": 0.2,
                    "interaction_count": 4,
                }
            ],
            "market": {
                "listings": [
                    {"item_id": "red_potion", "item_name": "Red Potion", "buy_price": 45, "sell_price": 25, "quantity": 200, "source": "npc_shop"}
                ],
                "vendor_exposure": 1,
                "transaction_count_10m": 2,
            },
            "actors": [
                {"actor_id": "npc:tool_dealer", "actor_type": "npc", "relation": "neutral", "name": "Tool Dealer"}
            ],
        },
    )
    projector.observe_event(first)

    second = NormalizedEvent(
        meta=_meta(bot_id),
        observed_at=t0 + timedelta(minutes=5),
        event_family=EventFamily.snapshot,
        event_type="snapshot.compact",
        source_hook="pytest",
        payload={
            "position": {"map": "prt_fild08", "x": 102, "y": 101},
            "vitals": {"hp": 900, "hp_max": 1000, "sp": 120, "sp_max": 200, "weight": 1010, "weight_max": 8000},
            "combat": {"is_in_combat": False, "target_id": None, "ai_sequence": "route"},
            "inventory": {"zeny": 1600, "item_count": 6},
            "progression": {"base_level": 10, "job_level": 5, "skill_points": 2, "stat_points": 4},
        },
    )
    projector.observe_event(second)

    state = projector.export(bot_id=bot_id, features=LearningFeatureVector())

    assert state["operational"].skill_count == 2
    assert state["operational"].skills["Heal"] == 5
    assert state["inventory"].consumables["Red Potion"] == 12

    assert "quest.alpha" in state["quest"].active_quests
    assert state["quest"].quest_status["quest.alpha"] == "active"
    assert state["quest"].quest_titles["quest.alpha"] == "Collect Apples"
    assert state["quest"].active_objective_count >= 1

    assert state["npc"].total_known_npcs >= 1
    assert state["npc"].last_interacted_npc in {"Tool Dealer", "npc:tool_dealer"}

    assert state["economy"].vendor_exposure >= 1
    assert state["economy"].zeny_delta_10m == 600
    assert state["economy"].inventory_value_estimate > 0


def test_feature_extractor_tracks_npc_quest_and_market_activity() -> None:
    bot_id = "bot:wsA-features"
    extractor = FeatureExtractor()
    t0 = datetime.now(UTC) - timedelta(minutes=2)

    quest_evt = NormalizedEvent(
        meta=_meta(bot_id),
        observed_at=t0,
        event_family=EventFamily.quest,
        event_type="quest.transition",
        source_hook="pytest",
        payload={"quest_id": "quest.alpha", "state_to": "active", "npc": "Tool Dealer"},
    )
    market_evt = NormalizedEvent(
        meta=_meta(bot_id),
        observed_at=t0 + timedelta(seconds=30),
        event_family=EventFamily.action,
        event_type="market.buy",
        source_hook="pytest",
        payload={"zeny": 800},
    )
    chat_npc_evt = NormalizedEvent(
        meta=_meta(bot_id),
        observed_at=t0 + timedelta(seconds=60),
        event_family=EventFamily.chat,
        event_type="chat.intent",
        source_hook="pytest",
        payload={"interaction_intent": {"npc": "Tool Dealer", "kind": "quest_dialogue"}},
    )

    extractor.observe_event(quest_evt)
    extractor.observe_event(market_evt)
    extractor.observe_event(chat_npc_evt)

    features = extractor.extract(bot_id=bot_id, basis={})
    assert features.values["quest.transitions_10m"] >= 1.0
    assert features.values["social.npc_interactions_10m"] >= 1.0
    assert features.values["economy.market_transactions_10m"] >= 1.0


def test_world_state_projector_tracks_lifecycle_recovery_signals() -> None:
    bot_id = "bot:ws2-lifecycle"
    projector = WorldStateProjector()
    t0 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)

    projector.observe_event(
        NormalizedEvent(
            meta=_meta(bot_id),
            observed_at=t0,
            event_family=EventFamily.snapshot,
            event_type="snapshot.compact",
            source_hook="pytest",
            payload={
                "position": {"map": "prt_fild08", "x": 100, "y": 100},
                "vitals": {"hp": 900, "hp_max": 1000, "weight": 1000, "weight_max": 8000},
                "combat": {"is_in_combat": False, "target_id": None, "ai_sequence": "route"},
                "inventory": {"zeny": 1000, "item_count": 5},
                "raw": {"in_game": True, "death_count": 0, "respawn_state": "alive", "reconnect_age_s": 0.0},
            },
        )
    )

    projector.observe_event(
        NormalizedEvent(
            meta=_meta(bot_id),
            observed_at=t0 + timedelta(seconds=1),
            event_family=EventFamily.lifecycle,
            event_type="lifecycle.disconnected",
            source_hook="pytest",
            payload={"net_state": "disconnected"},
        )
    )
    projector.observe_event(
        NormalizedEvent(
            meta=_meta(bot_id),
            observed_at=t0 + timedelta(seconds=13),
            event_family=EventFamily.lifecycle,
            event_type="lifecycle.reconnected",
            source_hook="pytest",
            payload={"reconnect_age_s": 12.0},
        )
    )
    projector.observe_event(
        NormalizedEvent(
            meta=_meta(bot_id),
            observed_at=t0 + timedelta(seconds=14),
            event_family=EventFamily.lifecycle,
            event_type="lifecycle.death",
            source_hook="pytest",
            payload={"death_count": 1},
        )
    )
    projector.observe_event(
        NormalizedEvent(
            meta=_meta(bot_id),
            observed_at=t0 + timedelta(seconds=15),
            event_family=EventFamily.lifecycle,
            event_type="lifecycle.respawn",
            source_hook="pytest",
            payload={"respawn_state": "respawned"},
        )
    )
    projector.observe_event(
        NormalizedEvent(
            meta=_meta(bot_id),
            observed_at=t0 + timedelta(seconds=16),
            event_family=EventFamily.lifecycle,
            event_type="lifecycle.map_transfer",
            source_hook="pytest",
            payload={"from_map": "prt_fild08", "to_map": "prt_fild09"},
        )
    )
    projector.observe_event(
        NormalizedEvent(
            meta=_meta(bot_id),
            observed_at=t0 + timedelta(seconds=17),
            event_family=EventFamily.lifecycle,
            event_type="lifecycle.route_failure",
            source_hook="pytest",
            payload={"route_failure_count": 1},
        )
    )

    state = projector.export(bot_id=bot_id, features=LearningFeatureVector())
    assert state["operational"].liveness_state == "online"
    assert state["operational"].reconnect_age_s == pytest.approx(12.0)
    assert state["operational"].death_count >= 1
    assert state["operational"].respawn_state == "respawned"
    assert state["navigation"].route_churn_count >= 1
    assert state["navigation"].route_failure_count >= 1
    assert state["navigation"].route_status == "failed"


def test_runtime_enriched_state_exports_phase1_recovery_features() -> None:
    runtime = create_runtime()
    bot_id = "bot:ws2-recovery-features"
    t0 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)

    s1 = BotStateSnapshot(
        meta=_meta(bot_id),
        tick_id="tick-recovery-1",
        observed_at=t0,
        position=Position(map="prt_fild08", x=100, y=100),
        vitals=Vitals(hp=900, hp_max=1000, sp=100, sp_max=200, weight=7600, weight_max=8000),
        combat=CombatState(ai_sequence="route", is_in_combat=False),
        inventory=InventoryDigest(zeny=1500, item_count=8),
        inventory_items=[
            InventoryItemDigest(item_id="red_potion", name="Red Potion", quantity=1, category="consumable", sell_price=25)
        ],
        progression=ProgressionDigest(base_level=12, job_level=6),
        raw={"in_game": True, "death_count": 2, "respawn_state": "alive", "reconnect_age_s": 11.5, "route_failure_count": 3},
    )
    s2 = s1.model_copy(update={"tick_id": "tick-recovery-2", "observed_at": t0 + timedelta(seconds=2)})

    runtime.ingest_snapshot(s1)
    runtime.ingest_snapshot(s2)

    deadline = time.monotonic() + 2.0
    enriched = runtime.enriched_state(bot_id=bot_id)
    while (
        (
            enriched.features.values.get("navigation.route_churn_count", 0.0) < 1.0
            or enriched.features.values.get("inventory.weight_pressure", 0.0) <= 0.8
        )
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
        enriched = runtime.enriched_state(bot_id=bot_id)

    values = enriched.features.values
    assert values["operational.death_count"] == pytest.approx(2.0)
    assert values["operational.reconnect_age_s"] == pytest.approx(11.5)
    assert values["inventory.weight_pressure"] > 0.8
    assert values["inventory.consumable_depletion_score"] >= 0.9
    assert values["navigation.route_failure_count"] >= 3.0
    assert values["navigation.route_churn_count"] >= 1.0


def test_runtime_fleet_constraints_include_local_startup_preferred_map() -> None:
    runtime = create_runtime()

    response = runtime.fleet_constraints(bot_id="bot:ws2-fleet-local-default")

    preferred = response.constraints.get("preferred_grind_maps")
    assert isinstance(preferred, list)
    assert "prt_fild08" in preferred


def test_runtime_fleet_constraints_respect_assignment_without_local_startup_injection() -> None:
    runtime = create_runtime()
    bot_id = "bot:ws2-fleet-assignment"
    assert runtime.fleet_constraint_state is not None

    runtime.fleet_constraint_state.update_from_blackboard(
        blackboard={
            "doctrine": {"version": "doctrine-test"},
            "constraints": {
                bot_id: {
                    "assignment": "pay_fild08",
                    "preferred_grind_maps": ["pay_fild08"],
                    "sources": ["fleet-central"],
                }
            },
        }
    )

    response = runtime.fleet_constraints(bot_id=bot_id)

    assert response.constraints.get("assignment") == "pay_fild08"
    preferred = response.constraints.get("preferred_grind_maps")
    assert preferred == ["pay_fild08"]
    assert "prt_fild08" not in preferred


def test_runtime_snapshot_builds_typed_actor_delta_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = create_runtime()
    bot_id = "bot:ws2-runtime-actors"
    runtime._actor_presence_by_bot[bot_id] = {"stale-actor"}

    captured: dict[str, ActorDeltaPushRequest] = {}

    def _fake_ingest_actor_delta(self: RuntimeState, payload: ActorDeltaPushRequest) -> IngestAcceptedResponse:
        captured["payload"] = payload
        return IngestAcceptedResponse(ok=True, accepted=0, dropped=0, bot_id=payload.meta.bot_id, event_ids=[])

    monkeypatch.setattr(RuntimeState, "ingest_actor_delta", _fake_ingest_actor_delta)

    snapshot = _snapshot(
        bot_id=bot_id,
        tick_id="tick-actors-1",
        observed_at=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        actors=[
            ActorDigest(actor_id="mob-1", actor_type="monster", name="Poring", relation="hostile", x=10, y=20),
            ActorDigest(actor_id="npc-1", actor_type="npc", name="Tool Dealer", relation="neutral", x=11, y=21),
        ],
    )

    runtime.ingest_snapshot(snapshot)

    deadline = time.monotonic() + 2.0
    while "payload" not in captured and time.monotonic() < deadline:
        time.sleep(0.01)

    assert "payload" in captured
    payload = captured["payload"]
    assert payload.revision == "tick-actors-1"
    assert [item.actor_id for item in payload.actors] == ["mob-1", "npc-1"]
    assert payload.removed_actor_ids == ["stale-actor"]


def test_runtime_ingest_actor_delta_emits_batch_and_presence_observability() -> None:
    runtime = create_runtime()
    bot_id = "bot:ws2-runtime-ingest-actors"

    payload = ActorDeltaPushRequest(
        meta=_meta(bot_id),
        observed_at=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        revision="tick-actor-obs-1",
        actors=[
            ActorObservation(
                actor_id="mob-1",
                actor_type="monster",
                name="Poring",
                map="prt_fild08",
                x=100,
                y=120,
                hp=50,
                hp_max=100,
                level=3,
                relation="hostile",
                raw={},
            )
        ],
        removed_actor_ids=[],
    )

    result = runtime.ingest_actor_delta(payload)

    assert result.accepted >= 2
    assert runtime._actor_presence_by_bot.get(bot_id) == {"mob-1"}

    recent = runtime.recent_ingest_events(bot_id=bot_id, limit=16)
    types = [str(item.get("event_type") or "") for item in recent]
    assert "actor.batch" in types
    assert "actor.observed" in types

    batch_rows = [item for item in recent if item.get("event_type") == "actor.batch"]
    assert batch_rows
    latest_batch = batch_rows[0]
    numeric = latest_batch.get("numeric") or {}
    assert float(numeric.get("observed_count") or 0.0) == pytest.approx(1.0)
    assert float(numeric.get("removed_count") or 0.0) == pytest.approx(0.0)
    assert float(numeric.get("hostile_count") or 0.0) == pytest.approx(1.0)


def test_runtime_ingest_actor_delta_duplicate_revision_is_explicitly_skipped() -> None:
    runtime = create_runtime()
    bot_id = "bot:ws2-runtime-ingest-actors-dup"

    payload = ActorDeltaPushRequest(
        meta=_meta(bot_id),
        observed_at=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        revision="tick-actor-dup-1",
        actors=[
            ActorObservation(
                actor_id="mob-1",
                actor_type="monster",
                name="Poring",
                map="prt_fild08",
                x=100,
                y=120,
                hp=50,
                hp_max=100,
                level=3,
                relation="hostile",
                raw={},
            )
        ],
        removed_actor_ids=[],
    )

    first = runtime.ingest_actor_delta(payload)
    second = runtime.ingest_actor_delta(payload)

    assert first.accepted >= 2
    assert second.accepted == 0
    assert second.message == "duplicate_actor_revision_skipped"


def test_runtime_snapshot_accepts_rich_ingestion_fields() -> None:
    runtime = create_runtime()
    bot_id = "bot:wsA-runtime-rich"
    snapshot = BotStateSnapshot(
        meta=_meta(bot_id),
        tick_id="tick-rich-1",
        observed_at=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        position=Position(map="prt_fild08", x=100, y=100),
        vitals=Vitals(hp=900, hp_max=1000, sp=100, sp_max=200, weight=500, weight_max=8000),
        combat=CombatState(ai_sequence="route", is_in_combat=False),
        inventory=InventoryDigest(zeny=1500, item_count=8),
        inventory_items=[
            InventoryItemDigest(item_id="red_potion", name="Red Potion", quantity=10, category="consumable", sell_price=25)
        ],
        progression=ProgressionDigest(base_level=12, job_level=6, skill_points=3, stat_points=4),
        skills=[SkillDigest(skill_id="AL_HEAL", skill_name="Heal", level=5)],
        quests=[
            QuestDigest(
                quest_id="quest.alpha",
                state="active",
                npc="Tool Dealer",
                title="Collect Apples",
                objectives=[
                    QuestObjectiveDigest(
                        objective_id="obj.1",
                        description="Collect 10 apples",
                        status="in_progress",
                        current=3,
                        target=10,
                    )
                ],
            )
        ],
        npc_relationships=[
            NpcRelationshipDigest(
                npc_id="npc:tool_dealer",
                npc_name="Tool Dealer",
                relation="quest",
                affinity_score=0.2,
                trust_score=0.1,
                interaction_count=2,
            )
        ],
        market=MarketDigest(
            listings=[
                MarketQuoteDigest(
                    item_id="red_potion",
                    item_name="Red Potion",
                    buy_price=45,
                    sell_price=25,
                    quantity=200,
                    source="npc_shop",
                )
            ]
        ),
        actors=[
            ActorDigest(actor_id="npc-1", actor_type="npc", name="Tool Dealer", relation="neutral", x=11, y=21)
        ],
    )

    runtime.ingest_snapshot(snapshot)
    deadline = time.monotonic() + 2.0
    enriched = runtime.enriched_state(bot_id=bot_id)
    while enriched.operational.skill_count < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
        enriched = runtime.enriched_state(bot_id=bot_id)

    assert enriched.operational.skill_count >= 1
    assert enriched.quest.active_quests
    assert enriched.npc.total_known_npcs >= 1
    assert enriched.economy.vendor_exposure >= 1


def test_runtime_slo_economy_uses_progression_digest_for_exp() -> None:
    runtime = create_runtime()
    if runtime.slo_metrics is None:
        pytest.skip("metrics disabled")

    bot_id = "bot:ws2-runtime-progression"
    t0 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)

    s1 = _snapshot(
        bot_id=bot_id,
        tick_id="tick-exp-1",
        observed_at=t0,
        base_exp=100,
        job_exp=0,
        raw={"base_exp": 0, "job_exp": 0},
    )
    s2 = _snapshot(
        bot_id=bot_id,
        tick_id="tick-exp-2",
        observed_at=t0 + timedelta(hours=1),
        base_exp=250,
        job_exp=0,
        raw={"base_exp": 0, "job_exp": 0},
    )

    runtime.ingest_snapshot(s1)
    runtime.ingest_snapshot(s2)

    rendered = runtime.slo_metrics.render_prometheus()
    match = re.search(r'sidecar_economy_exp_per_hour\{plan_family="unknown"\}\s+([0-9]+\.[0-9]+)', rendered)
    assert match is not None
    assert float(match.group(1)) == pytest.approx(150.0, rel=1e-6)
