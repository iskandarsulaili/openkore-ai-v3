from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from threading import RLock

from ai_sidecar.contracts.events import EventFamily, NormalizedEvent
from ai_sidecar.contracts.state_graph import EntityEdge, NormalizedStateGraph, WorldEntity


@dataclass(slots=True)
class _BotEntityGraph:
    nodes: dict[str, WorldEntity] = field(default_factory=dict)
    edges: dict[tuple[str, str, str], EntityEdge] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class EntityGraphStore:
    def __init__(self) -> None:
        self._lock = RLock()
        self._by_bot: dict[str, _BotEntityGraph] = {}

    def observe_event(self, event: NormalizedEvent) -> None:
        bot_id = event.meta.bot_id
        now = self._normalize_dt(event.observed_at)
        with self._lock:
            graph = self._by_bot.setdefault(bot_id, _BotEntityGraph())
            graph.updated_at = now

            if event.event_family == EventFamily.actor_state:
                self._apply_actor_event(graph, bot_id, event, now)
            elif event.event_family == EventFamily.chat:
                self._apply_chat_event(graph, bot_id, event, now)

            self._trim(graph, now)

    def export(self, *, bot_id: str) -> NormalizedStateGraph:
        with self._lock:
            graph = self._by_bot.get(bot_id)
            if graph is None:
                return NormalizedStateGraph(bot_id=bot_id, nodes=[], edges=[], projections={})
            return NormalizedStateGraph(
                bot_id=bot_id,
                generated_at=datetime.now(UTC),
                nodes=sorted(graph.nodes.values(), key=lambda item: (item.entity_type, item.entity_id)),
                edges=sorted(graph.edges.values(), key=lambda item: (item.source_id, item.target_id, item.relation)),
                projections={
                    "node_count": len(graph.nodes),
                    "edge_count": len(graph.edges),
                    "updated_at": graph.updated_at,
                },
            )

    def _apply_actor_event(self, graph: _BotEntityGraph, bot_id: str, event: NormalizedEvent, now: datetime) -> None:
        payload = event.payload
        if event.event_type == "actor.removed":
            actor_id = str(payload.get("actor_id") or "")
            if actor_id:
                graph.nodes.pop(actor_id, None)
                for key in list(graph.edges.keys()):
                    if key[0] == actor_id or key[1] == actor_id:
                        del graph.edges[key]
            return

        actor_id = str(payload.get("actor_id") or "")
        actor_type = str(payload.get("actor_type") or "unknown")
        if not actor_id:
            return

        node = WorldEntity(
            entity_id=actor_id,
            entity_type=actor_type,
            name=payload.get("name"),
            map=payload.get("map"),
            x=payload.get("x"),
            y=payload.get("y"),
            hp=payload.get("hp"),
            hp_max=payload.get("hp_max"),
            relation=payload.get("relation"),
            threat_score=float(payload.get("distance") or 0.0),
            last_seen_at=now,
            attributes={"raw": payload.get("raw", {})},
        )
        graph.nodes[actor_id] = node

        if actor_id != bot_id:
            relation = str(payload.get("relation") or "observes")
            edge_key = (bot_id, actor_id, relation)
            graph.edges[edge_key] = EntityEdge(
                source_id=bot_id,
                target_id=actor_id,
                relation=relation,
                weight=1.0,
                observed_at=now,
                metadata={"event_id": event.event_id},
            )

    def _apply_chat_event(self, graph: _BotEntityGraph, bot_id: str, event: NormalizedEvent, now: datetime) -> None:
        sender = str(event.tags.get("sender") or "")
        if not sender:
            return
        sender_id = f"chat:{sender}"
        graph.nodes.setdefault(
            sender_id,
            WorldEntity(
                entity_id=sender_id,
                entity_type="chat_participant",
                name=sender,
                map=event.tags.get("map") or None,
                relation="social",
                threat_score=0.0,
                last_seen_at=now,
            ),
        )
        edge_key = (sender_id, bot_id, "chat_interaction")
        graph.edges[edge_key] = EntityEdge(
            source_id=sender_id,
            target_id=bot_id,
            relation="chat_interaction",
            weight=1.0,
            observed_at=now,
            metadata={"channel": event.tags.get("channel") or "unknown", "event_id": event.event_id},
        )

    def _trim(self, graph: _BotEntityGraph, now: datetime) -> None:
        expiry = now - timedelta(minutes=30)
        stale_nodes = [node_id for node_id, node in graph.nodes.items() if self._normalize_dt(node.last_seen_at) < expiry]
        for node_id in stale_nodes:
            del graph.nodes[node_id]
            for key in list(graph.edges.keys()):
                if key[0] == node_id or key[1] == node_id:
                    del graph.edges[key]

    def _normalize_dt(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

