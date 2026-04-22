from __future__ import annotations

import re
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
    Position,
    ProgressionDigest,
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
    assert event_types == ["actor.appeared", "actor.observed", "actor.disappeared", "actor.removed"]
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

    payload = captured["payload"]
    assert payload.revision == "tick-actors-1"
    assert [item.actor_id for item in payload.actors] == ["mob-1", "npc-1"]
    assert payload.removed_actor_ids == ["stale-actor"]


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
