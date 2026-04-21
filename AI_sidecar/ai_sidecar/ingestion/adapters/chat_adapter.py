from __future__ import annotations

from ai_sidecar.contracts.events import ChatStreamIngestRequest, EventFamily, NormalizedEvent


def chat_stream_to_events(payload: ChatStreamIngestRequest) -> list[NormalizedEvent]:
    events: list[NormalizedEvent] = []

    for item in payload.events:
        message = item.message.strip()
        events.append(
            NormalizedEvent(
                meta=payload.meta,
                observed_at=payload.observed_at,
                event_family=EventFamily.chat,
                event_type=f"chat.{item.channel}",
                source_hook="v2.ingest.chat",
                text=message,
                tags={
                    "channel": item.channel,
                    "sender": item.sender or "",
                    "target": item.target or "",
                    "kind": item.kind or "",
                    "map": item.map or "",
                },
                numeric={"message_length": float(len(message))},
                payload=item.model_dump(mode="json"),
            )
        )

    if payload.interaction_intent:
        events.append(
            NormalizedEvent(
                meta=payload.meta,
                observed_at=payload.observed_at,
                event_family=EventFamily.chat,
                event_type="chat.intent",
                source_hook="v2.ingest.chat",
                text="chat interaction intent update",
                payload={"interaction_intent": payload.interaction_intent},
            )
        )

    return events
