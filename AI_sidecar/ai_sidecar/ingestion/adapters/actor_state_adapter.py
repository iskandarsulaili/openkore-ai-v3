from __future__ import annotations

from ai_sidecar.contracts.events import ActorDeltaPushRequest, EventFamily, NormalizedEvent


def actor_delta_to_events(
    payload: ActorDeltaPushRequest,
    *,
    appeared_actor_ids: set[str] | None = None,
    disappeared_actor_ids: set[str] | None = None,
) -> list[NormalizedEvent]:
    events: list[NormalizedEvent] = []
    appeared = appeared_actor_ids or set()
    disappeared = disappeared_actor_ids or set()

    hostile_count = 0
    for actor in payload.actors:
        relation = str(actor.relation or "").strip().lower()
        actor_type = str(actor.actor_type or "").strip().lower()
        if relation in {"hostile", "enemy", "monster"} or actor_type == "monster":
            hostile_count += 1

    events.append(
        NormalizedEvent(
            meta=payload.meta,
            observed_at=payload.observed_at,
            event_family=EventFamily.actor_state,
            event_type="actor.batch",
            source_hook="v2.ingest.actors",
            text="actor delta batch received",
            numeric={
                "observed_count": float(len(payload.actors)),
                "removed_count": float(len(payload.removed_actor_ids)),
                "appeared_count": float(len(appeared)),
                "disappeared_count": float(len(disappeared)),
                "hostile_count": float(hostile_count),
            },
            payload={
                "revision": payload.revision,
                "observed_actor_ids": [item.actor_id for item in payload.actors[:32]],
                "removed_actor_ids": list(payload.removed_actor_ids[:32]),
            },
        )
    )

    for actor in payload.actors:
        actor_payload = actor.model_dump(mode="json")
        actor_id = actor.actor_id

        if actor_id in appeared:
            events.append(
                NormalizedEvent(
                    meta=payload.meta,
                    observed_at=payload.observed_at,
                    event_family=EventFamily.actor_state,
                    event_type="actor.appeared",
                    source_hook="v2.ingest.actors",
                    text=f"actor appeared {actor_id}",
                    tags={"actor_type": actor.actor_type, "relation": actor.relation or "unknown"},
                    numeric={
                        "x": float(actor.x or 0),
                        "y": float(actor.y or 0),
                        "distance": float(actor.distance or 0.0),
                        "hp": float(actor.hp or 0),
                        "hp_max": float(actor.hp_max or 0),
                        "level": float(actor.level or 0),
                    },
                    payload={**actor_payload, "revision": payload.revision},
                )
            )

        events.append(
            NormalizedEvent(
                meta=payload.meta,
                observed_at=payload.observed_at,
                event_family=EventFamily.actor_state,
                event_type="actor.observed",
                source_hook="v2.ingest.actors",
                text=f"observed actor {actor_id}",
                tags={"actor_type": actor.actor_type, "relation": actor.relation or "unknown"},
                numeric={
                    "x": float(actor.x or 0),
                    "y": float(actor.y or 0),
                    "distance": float(actor.distance or 0.0),
                    "hp": float(actor.hp or 0),
                    "hp_max": float(actor.hp_max or 0),
                    "level": float(actor.level or 0),
                },
                payload={**actor_payload, "revision": payload.revision},
            )
        )

    for actor_id in payload.removed_actor_ids:
        if actor_id in disappeared:
            events.append(
                NormalizedEvent(
                    meta=payload.meta,
                    observed_at=payload.observed_at,
                    event_family=EventFamily.actor_state,
                    event_type="actor.disappeared",
                    source_hook="v2.ingest.actors",
                    text=f"actor disappeared {actor_id}",
                    payload={"actor_id": actor_id, "revision": payload.revision},
                )
            )

        events.append(
            NormalizedEvent(
                meta=payload.meta,
                observed_at=payload.observed_at,
                event_family=EventFamily.actor_state,
                event_type="actor.removed",
                source_hook="v2.ingest.actors",
                text=f"removed actor {actor_id}",
                payload={"actor_id": actor_id, "revision": payload.revision},
            )
        )

    return events
