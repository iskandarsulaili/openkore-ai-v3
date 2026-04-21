from __future__ import annotations

from ai_sidecar.contracts.events import ConfigDoctrineFingerprintRequest, EventFamily, NormalizedEvent


def config_update_to_events(payload: ConfigDoctrineFingerprintRequest) -> list[NormalizedEvent]:
    changed_keys = payload.changed_keys or list(payload.values.keys())
    events: list[NormalizedEvent] = [
        NormalizedEvent(
            meta=payload.meta,
            observed_at=payload.observed_at,
            event_family=EventFamily.config,
            event_type="config.fingerprint",
            source_hook="v2.ingest.config",
            text="config doctrine fingerprint updated",
            tags={
                "fingerprint": payload.fingerprint,
                "doctrine_version": payload.doctrine_version or "",
            },
            numeric={"changed_key_count": float(len(changed_keys))},
            payload=payload.model_dump(mode="json"),
        )
    ]

    for key in changed_keys:
        events.append(
            NormalizedEvent(
                meta=payload.meta,
                observed_at=payload.observed_at,
                event_family=EventFamily.config,
                event_type="config.key_changed",
                source_hook="v2.ingest.config",
                text=f"config key changed: {key}",
                tags={"key": key},
                payload={
                    "key": key,
                    "value": payload.values.get(key),
                    "fingerprint": payload.fingerprint,
                    "source_files": payload.source_files,
                },
            )
        )

    return events

