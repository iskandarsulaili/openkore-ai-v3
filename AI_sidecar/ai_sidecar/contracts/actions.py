from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ai_sidecar.contracts.common import NoopActionPayload
from ai_sidecar.contracts.common import ContractMeta


class ActionStatus(StrEnum):
    queued = "queued"
    dispatched = "dispatched"
    acknowledged = "acknowledged"
    expired = "expired"
    dropped = "dropped"


class ActionProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(min_length=1, max_length=128)
    kind: str = Field(default="command", min_length=1, max_length=64)
    command: str = Field(min_length=0, max_length=256)
    conflict_key: str | None = Field(default=None, max_length=128)
    created_at: datetime
    expires_at: datetime
    idempotency_key: str = Field(min_length=1, max_length=128)
    metadata: dict[str, object] = Field(default_factory=dict)


class QueueActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    proposal: ActionProposal


class QueueActionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    accepted: bool = True
    message: str = "action queued"
    bot_id: str
    action_id: str
    status: ActionStatus


class NextActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    poll_id: str = Field(min_length=1, max_length=128)


class NextActionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    bot_id: str
    poll_id: str
    has_action: bool
    action: ActionProposal | NoopActionPayload | None = None
    reason: str = ""


class ActionAckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    action_id: str = Field(min_length=1, max_length=128)
    poll_id: str = Field(min_length=1, max_length=128)
    success: bool
    result_code: str = Field(default="ok", max_length=64)
    message: str = Field(default="", max_length=512)
    observed_latency_ms: int | None = None


class ActionAckResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    acknowledged: bool
    action_id: str
    status: ActionStatus
