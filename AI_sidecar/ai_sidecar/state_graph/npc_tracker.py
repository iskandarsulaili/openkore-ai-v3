from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from ai_sidecar.contracts.events import EventFamily, NormalizedEvent
from ai_sidecar.contracts.state_graph import NpcState


@dataclass(slots=True)
class _NpcRelation:
    npc_id: str
    npc_name: str | None = None
    relation: str = "neutral"
    affinity_score: float = 0.0
    trust_score: float = 0.0
    interaction_count: int = 0
    last_interaction_at: datetime | None = None


@dataclass(slots=True)
class _NpcWindow:
    relations: dict[str, _NpcRelation] = field(default_factory=dict)
    interactions_10m: deque[datetime] = field(default_factory=deque)


class NpcInteractionTracker:
    """Tracks NPC relationship and interaction dynamics from normalized events."""

    def __init__(self) -> None:
        self._by_bot: dict[str, _NpcWindow] = {}

    def observe_snapshot(self, *, bot_id: str, payload: dict[str, object], observed_at: datetime) -> None:
        window = self._by_bot.setdefault(bot_id, _NpcWindow())

        rels = payload.get("npc_relationships")
        if isinstance(rels, list):
            for item in rels:
                if not isinstance(item, dict):
                    continue
                npc_id = str(item.get("npc_id") or "").strip()
                if not npc_id:
                    continue
                relation = window.relations.setdefault(npc_id, _NpcRelation(npc_id=npc_id))
                relation.npc_name = str(item.get("npc_name") or "").strip() or relation.npc_name
                relation.relation = str(item.get("relation") or relation.relation or "neutral").strip().lower() or "neutral"
                relation.affinity_score = _clamp(_float_or_default(item.get("affinity_score"), relation.affinity_score), -1.0, 1.0)
                relation.trust_score = _clamp(_float_or_default(item.get("trust_score"), relation.trust_score), 0.0, 1.0)
                relation.interaction_count = max(relation.interaction_count, _int_or_default(item.get("interaction_count"), 0))
                relation.last_interaction_at = observed_at
                self._record_interaction(window, observed_at)

        actors = payload.get("actors")
        if isinstance(actors, list):
            for item in actors:
                if not isinstance(item, dict):
                    continue
                if str(item.get("actor_type") or "").strip().lower() != "npc":
                    continue
                npc_id = str(item.get("actor_id") or "").strip()
                if not npc_id:
                    continue
                relation = window.relations.setdefault(npc_id, _NpcRelation(npc_id=npc_id))
                relation.npc_name = str(item.get("name") or "").strip() or relation.npc_name
                rel_value = str(item.get("relation") or "").strip().lower()
                if rel_value:
                    relation.relation = rel_value

        self._trim(window, observed_at)

    def observe_event(self, event: NormalizedEvent) -> None:
        bot_id = event.meta.bot_id
        now = _normalize_dt(event.observed_at)
        window = self._by_bot.setdefault(bot_id, _NpcWindow())

        if event.event_family == EventFamily.actor_state and event.event_type in {"actor.removed", "actor.disappeared"}:
            actor_id = str(event.payload.get("actor_id") or "").strip()
            if actor_id:
                window.relations.pop(actor_id, None)
            self._trim(window, now)
            return

        npc_id = ""
        npc_name = ""

        if event.event_family == EventFamily.quest:
            npc_id = str(event.payload.get("npc") or event.tags.get("npc") or "").strip()
            npc_name = npc_id
        elif event.event_family == EventFamily.actor_state:
            actor_type = str(event.payload.get("actor_type") or event.tags.get("actor_type") or "").strip().lower()
            if actor_type == "npc":
                npc_id = str(event.payload.get("actor_id") or "").strip()
                npc_name = str(event.payload.get("name") or "").strip()
        elif event.event_family == EventFamily.chat and event.event_type == "chat.intent":
            intent = event.payload.get("interaction_intent") if isinstance(event.payload.get("interaction_intent"), dict) else {}
            npc_id = str(intent.get("npc") or intent.get("npc_id") or "").strip()
            npc_name = str(intent.get("npc_name") or "").strip()

        if npc_id:
            relation = window.relations.setdefault(npc_id, _NpcRelation(npc_id=npc_id))
            if npc_name:
                relation.npc_name = npc_name
            if event.event_family == EventFamily.quest:
                relation.relation = "quest"
                relation.affinity_score = _clamp(relation.affinity_score + 0.03, -1.0, 1.0)
                relation.trust_score = _clamp(relation.trust_score + 0.02, 0.0, 1.0)
            relation.interaction_count += 1
            relation.last_interaction_at = now
            self._record_interaction(window, now)

        self._trim(window, now)

    def export(self, *, bot_id: str, observed_at: datetime) -> NpcState:
        window = self._by_bot.get(bot_id)
        if window is None:
            return NpcState(updated_at=observed_at)

        self._trim(window, observed_at)
        relations = sorted(
            window.relations.values(),
            key=lambda item: (item.interaction_count, item.trust_score, item.affinity_score),
            reverse=True,
        )

        payload = [
            {
                "npc_id": item.npc_id,
                "npc_name": item.npc_name,
                "relation": item.relation,
                "affinity_score": round(item.affinity_score, 4),
                "trust_score": round(item.trust_score, 4),
                "interaction_count": item.interaction_count,
                "last_interaction_at": item.last_interaction_at,
            }
            for item in relations[:64]
        ]

        last_npc = payload[0]["npc_name"] or payload[0]["npc_id"] if payload else None
        return NpcState(
            last_interacted_npc=last_npc,
            relationships=payload,
            total_known_npcs=len(window.relations),
            interaction_count_10m=len(window.interactions_10m),
            updated_at=observed_at,
            raw={
                "npc_ids": [item["npc_id"] for item in payload[:16]],
            },
        )

    def _record_interaction(self, window: _NpcWindow, observed_at: datetime) -> None:
        window.interactions_10m.append(observed_at)

    def _trim(self, window: _NpcWindow, observed_at: datetime) -> None:
        threshold = observed_at - timedelta(minutes=10)
        while window.interactions_10m and window.interactions_10m[0] < threshold:
            window.interactions_10m.popleft()


def _float_or_default(value: object, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _int_or_default(value: object, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _normalize_dt(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
