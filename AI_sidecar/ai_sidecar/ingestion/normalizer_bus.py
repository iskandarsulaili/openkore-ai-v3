from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import RLock

from ai_sidecar.contracts.events import (
    ActorDeltaPushRequest,
    ChatStreamIngestRequest,
    ConfigDoctrineFingerprintRequest,
    EventBatchIngestRequest,
    EventFamily,
    IngestAcceptedResponse,
    NormalizedEvent,
    QuestTransitionRequest,
)
from ai_sidecar.contracts.state import BotStateSnapshot
from ai_sidecar.contracts.state_graph import EnrichedWorldState
from ai_sidecar.ingestion.adapters.actor_state_adapter import actor_delta_to_events
from ai_sidecar.ingestion.adapters.chat_adapter import chat_stream_to_events
from ai_sidecar.ingestion.adapters.config_adapter import config_update_to_events
from ai_sidecar.ingestion.adapters.quest_adapter import quest_transition_to_events
from ai_sidecar.ingestion.event_journal import EventJournal
from ai_sidecar.state_graph.entity_graph import EntityGraphStore
from ai_sidecar.state_graph.feature_extractor import FeatureExtractor
from ai_sidecar.state_graph.world_state import WorldStateProjector

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class NormalizerBus:
    event_journal: EventJournal
    world_state: WorldStateProjector
    entity_graph: EntityGraphStore
    feature_extractor: FeatureExtractor
    _lock: RLock

    @classmethod
    def create(cls, *, event_journal: EventJournal) -> "NormalizerBus":
        return cls(
            event_journal=event_journal,
            world_state=WorldStateProjector(),
            entity_graph=EntityGraphStore(),
            feature_extractor=FeatureExtractor(),
            _lock=RLock(),
        )

    def ingest_batch(self, payload: EventBatchIngestRequest) -> IngestAcceptedResponse:
        return self._ingest(payload.events, bot_id=payload.meta.bot_id)

    def ingest_actors(self, payload: ActorDeltaPushRequest) -> IngestAcceptedResponse:
        return self._ingest(actor_delta_to_events(payload), bot_id=payload.meta.bot_id)

    def ingest_chat(self, payload: ChatStreamIngestRequest) -> IngestAcceptedResponse:
        return self._ingest(chat_stream_to_events(payload), bot_id=payload.meta.bot_id)

    def ingest_config(self, payload: ConfigDoctrineFingerprintRequest) -> IngestAcceptedResponse:
        return self._ingest(config_update_to_events(payload), bot_id=payload.meta.bot_id)

    def ingest_quest(self, payload: QuestTransitionRequest) -> IngestAcceptedResponse:
        return self._ingest(quest_transition_to_events(payload), bot_id=payload.meta.bot_id)

    def ingest_snapshot(self, snapshot: BotStateSnapshot) -> IngestAcceptedResponse:
        event = NormalizedEvent(
            meta=snapshot.meta,
            observed_at=snapshot.observed_at,
            event_family=EventFamily.snapshot,
            event_type="snapshot.compact",
            source_hook="v1.ingest.snapshot",
            text="snapshot ingested",
            numeric={
                "hp": float(snapshot.vitals.hp or 0),
                "hp_max": float(snapshot.vitals.hp_max or 0),
                "sp": float(snapshot.vitals.sp or 0),
                "sp_max": float(snapshot.vitals.sp_max or 0),
                "x": float(snapshot.position.x or 0),
                "y": float(snapshot.position.y or 0),
                "zeny": float(snapshot.inventory.zeny or 0),
                "item_count": float(snapshot.inventory.item_count or 0),
            },
            tags={
                "tick_id": snapshot.tick_id,
                "map": snapshot.position.map or "",
                "ai_sequence": snapshot.combat.ai_sequence or "",
            },
            payload=snapshot.model_dump(mode="json"),
        )
        return self._ingest([event], bot_id=snapshot.meta.bot_id)

    def enriched_state(self, *, bot_id: str) -> EnrichedWorldState:
        basis = {
            "operational.in_combat": 1.0,
        }
        features = self.feature_extractor.extract(bot_id=bot_id, basis=basis)
        world_projection = self.world_state.export(bot_id=bot_id, features=features)
        graph = self.entity_graph.export(bot_id=bot_id)
        return EnrichedWorldState(
            bot_id=bot_id,
            operational=world_projection["operational"],
            encounter=world_projection["encounter"],
            navigation=world_projection["navigation"],
            quest=world_projection["quest"],
            inventory=world_projection["inventory"],
            economy=world_projection["economy"],
            social=world_projection["social"],
            risk=world_projection["risk"],
            macro_execution=world_projection["macro_execution"],
            fleet_intent=world_projection["fleet_intent"],
            entities=graph.nodes,
            features=world_projection["features"],
            recent_event_ids=world_projection["recent_event_ids"],
        )

    def debug_graph(self, *, bot_id: str) -> dict[str, object]:
        graph = self.entity_graph.export(bot_id=bot_id)
        return graph.model_dump(mode="json")

    def recent_events(self, *, bot_id: str, limit: int = 100) -> list[dict[str, object]]:
        return self.event_journal.list_recent(bot_id=bot_id, limit=limit)

    def _ingest(self, events: list[NormalizedEvent], *, bot_id: str) -> IngestAcceptedResponse:
        accepted_ids: list[str] = []
        with self._lock:
            for event in events:
                try:
                    self.event_journal.append(event)
                    self.world_state.observe_event(event)
                    self.entity_graph.observe_event(event)
                    self.feature_extractor.observe_event(event)
                    accepted_ids.append(event.event_id)
                except Exception:
                    logger.exception(
                        "normalizer_bus_event_failed",
                        extra={
                            "event": "normalizer_bus_event_failed",
                            "bot_id": bot_id,
                            "event_id": event.event_id,
                            "event_type": event.event_type,
                        },
                    )
        return IngestAcceptedResponse(
            ok=True,
            accepted=len(accepted_ids),
            dropped=max(0, len(events) - len(accepted_ids)),
            bot_id=bot_id,
            event_ids=accepted_ids,
            message="events processed",
        )

