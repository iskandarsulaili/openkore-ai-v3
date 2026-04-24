from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai_sidecar.contracts.common import NoopActionPayload
from ai_sidecar.contracts.common import ContractMeta


class ActionStatus(StrEnum):
    queued = "queued"
    dispatched = "dispatched"
    acknowledged = "acknowledged"
    expired = "expired"
    dropped = "dropped"
    superseded = "superseded"


class ActionPriorityTier(StrEnum):
    reflex = "reflex"
    tactical = "tactical"
    strategic = "strategic"
    macro_management = "macro_management"


class ActionProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(min_length=1, max_length=128)
    kind: str = Field(default="command", min_length=1, max_length=64)
    command: str = Field(min_length=0, max_length=256)
    priority_tier: ActionPriorityTier = ActionPriorityTier.strategic
    conflict_key: str | None = Field(default=None, max_length=128)
    source: str = Field(default="manual", max_length=32)
    lease_id: str | None = Field(default=None, max_length=128)
    preconditions: list[str] = Field(default_factory=list)
    rollback_action: str | None = Field(default=None, max_length=256)
    latency_budget_ms: int | None = Field(default=None, ge=0, le=600000)
    ttl_seconds: int | None = Field(default=None, ge=1, le=86400)
    created_at: datetime
    expires_at: datetime
    idempotency_key: str = Field(min_length=1, max_length=128)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("source")
    @classmethod
    def _normalize_source(cls, value: str) -> str:
        allowed = {"reflex", "planner", "crewai", "ml", "fleet", "manual"}
        normalized = str(value or "").strip().lower()
        return normalized if normalized in allowed else "manual"

    @field_validator("preconditions")
    @classmethod
    def _normalize_preconditions(cls, value: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in value or []:
            normalized = str(item or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            out.append(normalized)
        return out

    @model_validator(mode="after")
    def _normalize_ttl(self) -> "ActionProposal":
        if self.ttl_seconds is None:
            return self
        ttl_seconds = max(1, int(self.ttl_seconds))
        target_expires = self.created_at + timedelta(seconds=ttl_seconds)
        if self.expires_at > target_expires:
            try:
                object.__setattr__(self, "expires_at", target_expires)
            except Exception:
                return self.model_copy(update={"expires_at": target_expires})
        return self


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
