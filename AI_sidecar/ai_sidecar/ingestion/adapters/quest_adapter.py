from __future__ import annotations

from ai_sidecar.contracts.events import EventFamily, NormalizedEvent, QuestTransitionRequest


def quest_transition_to_events(payload: QuestTransitionRequest) -> list[NormalizedEvent]:
    events: list[NormalizedEvent] = []

    for item in payload.transitions:
        events.append(
            NormalizedEvent(
                meta=payload.meta,
                observed_at=payload.observed_at,
                event_family=EventFamily.quest,
                event_type="quest.transition",
                source_hook="v2.ingest.quest",
                text=f"quest {item.quest_id} moved to {item.state_to}",
                tags={
                    "quest_id": item.quest_id,
                    "state_from": item.state_from or "",
                    "state_to": item.state_to,
                    "npc": item.npc or "",
                },
                payload=item.model_dump(mode="json"),
            )
        )

    if payload.active_quests:
        events.append(
            NormalizedEvent(
                meta=payload.meta,
                observed_at=payload.observed_at,
                event_family=EventFamily.quest,
                event_type="quest.active_set",
                source_hook="v2.ingest.quest",
                text="active quest set updated",
                numeric={"active_quest_count": float(len(payload.active_quests))},
                payload={"active_quests": payload.active_quests},
            )
        )

    return events

