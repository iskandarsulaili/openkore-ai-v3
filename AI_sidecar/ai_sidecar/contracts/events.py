from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from ai_sidecar.contracts.common import ContractMeta, utc_now


def new_event_id() -> str:
    return f"evt-{uuid4().hex}"


class EventFamily(StrEnum):
    snapshot = "snapshot"
    hook = "hook"
    packet = "packet"
    config = "config"
    actor_state = "actor_state"
    chat = "chat"
    quest = "quest"
    telemetry = "telemetry"
    macro = "macro"
    action = "action"
    lifecycle = "lifecycle"
    system = "system"


class EventSeverity(StrEnum):
    debug = "debug"
    info = "info"
    warning = "warning"
    error = "error"
    critical = "critical"


class NormalizedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    event_id: str = Field(default_factory=new_event_id, min_length=8, max_length=128)
    event_family: EventFamily
    event_type: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:/-]+$")
    observed_at: datetime = Field(default_factory=utc_now)
    sequence: int | None = Field(default=None, ge=0)
    source_hook: str | None = Field(default=None, max_length=256)
    correlation_id: str | None = Field(default=None, max_length=128)
    severity: EventSeverity = EventSeverity.info
    text: str = Field(default="", max_length=1024)
    tags: dict[str, str] = Field(default_factory=dict)
    numeric: dict[str, float] = Field(default_factory=dict)
    payload: dict[str, object] = Field(default_factory=dict)


class ActorObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor_id: str = Field(min_length=1, max_length=128)
    actor_type: str = Field(min_length=1, max_length=64)
    name: str | None = Field(default=None, max_length=128)
    map: str | None = Field(default=None, max_length=64)
    x: int | None = None
    y: int | None = None
    hp: int | None = None
    hp_max: int | None = None
    level: int | None = None
    distance: float | None = None
    relation: str | None = Field(default=None, max_length=64)
    raw: dict[str, object] = Field(default_factory=dict)


class ActorDeltaPushRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    observed_at: datetime = Field(default_factory=utc_now)
    revision: str | None = Field(default=None, max_length=128)
    actors: list[ActorObservation] = Field(default_factory=list)
    removed_actor_ids: list[str] = Field(default_factory=list)


class ChatMessageEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel: str = Field(min_length=1, max_length=64)
    sender: str | None = Field(default=None, max_length=128)
    target: str | None = Field(default=None, max_length=128)
    message: str = Field(default="", max_length=1024)
    map: str | None = Field(default=None, max_length=64)
    kind: str | None = Field(default=None, max_length=64)
    raw: dict[str, object] = Field(default_factory=dict)


class ChatStreamIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    observed_at: datetime = Field(default_factory=utc_now)
    events: list[ChatMessageEvent] = Field(default_factory=list)
    interaction_intent: dict[str, object] = Field(default_factory=dict)


class ConfigDoctrineFingerprintRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    observed_at: datetime = Field(default_factory=utc_now)
    fingerprint: str = Field(min_length=1, max_length=256)
    doctrine_version: str | None = Field(default=None, max_length=128)
    changed_keys: list[str] = Field(default_factory=list)
    values: dict[str, str] = Field(default_factory=dict)
    source_files: list[str] = Field(default_factory=list)


class QuestTransitionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quest_id: str = Field(min_length=1, max_length=128)
    npc: str | None = Field(default=None, max_length=128)
    state_from: str | None = Field(default=None, max_length=64)
    state_to: str = Field(min_length=1, max_length=64)
    detail: str = Field(default="", max_length=512)
    metadata: dict[str, object] = Field(default_factory=dict)


class QuestTransitionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    observed_at: datetime = Field(default_factory=utc_now)
    transitions: list[QuestTransitionEvent] = Field(default_factory=list)
    active_quests: list[str] = Field(default_factory=list)


class IngestAcceptedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    accepted: int = 0
    dropped: int = 0
    bot_id: str
    event_ids: list[str] = Field(default_factory=list)
    message: str = "accepted"


class EventBatchIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    events: list[NormalizedEvent] = Field(default_factory=list)
