from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class BotIdentityRecord:
    bot_id: str
    bot_name: str | None
    role: str | None
    assignment: str | None
    capabilities: list[str] = field(default_factory=list)
    attributes: dict[str, str] = field(default_factory=dict)
    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None
    last_tick_id: str | None = None
    liveness_state: str = "unknown"


@dataclass(slots=True)
class SnapshotRecord:
    id: int
    bot_id: str
    tick_id: str
    observed_at: datetime
    ingested_at: datetime
    snapshot: dict[str, object]


@dataclass(slots=True)
class ActionRecord:
    action_id: str
    bot_id: str
    kind: str
    status: str
    priority_tier: str
    conflict_key: str | None
    idempotency_key: str
    proposal: dict[str, object]
    created_at: datetime
    expires_at: datetime
    queued_at: datetime
    dispatched_at: datetime | None
    acknowledged_at: datetime | None
    ack_success: bool | None
    ack_result_code: str | None
    ack_message: str
    poll_id: str | None
    status_reason: str


@dataclass(slots=True)
class TelemetryEventRecord:
    id: int
    bot_id: str
    timestamp: datetime
    level: str
    category: str
    event: str
    message: str
    metrics: dict[str, float]
    tags: dict[str, str]
    ingested_at: datetime
    is_incident: bool


@dataclass(slots=True)
class MacroPublicationRecord:
    publication_id: str
    bot_id: str
    version: str
    content_sha256: str
    published_at: datetime
    manifest: dict[str, object]
    paths: dict[str, str]
    macro_count: int
    event_macro_count: int
    automacro_count: int


@dataclass(slots=True)
class AuditEventRecord:
    id: int
    timestamp: datetime
    level: str
    event_type: str
    bot_id: str | None
    summary: str
    payload: dict[str, object]


@dataclass(slots=True)
class MemoryEpisodeRecord:
    id: str
    bot_id: str
    event_type: str
    content: str
    metadata: dict[str, object]
    created_at: datetime


@dataclass(slots=True)
class MemorySemanticRecord:
    id: str
    bot_id: str
    source: str
    content: str
    lexical_signature: str
    metadata: dict[str, object]
    created_at: datetime
    dimensions: int
    vector: list[float]
    norm: float

