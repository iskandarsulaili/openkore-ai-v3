from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_trace_id() -> str:
    return uuid4().hex


class ContractMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: str = Field(default="v1")
    emitted_at: datetime = Field(default_factory=utc_now)
    trace_id: str = Field(default_factory=new_trace_id)
    source: str = Field(default="openkore")
    bot_id: str = Field(min_length=1, max_length=128)


class ApiEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    message: str = "ok"
    meta: ContractMeta


class NoopActionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str = "noop"
    kind: str = "noop"
    command: str = ""
    priority_tier: str | None = None
    conflict_key: str | None = None
    expires_at: datetime | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
