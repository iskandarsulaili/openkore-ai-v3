from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from ai_sidecar.contracts.common import ContractMeta, utc_now


class FleetSyncRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    include_blackboard: bool = True


class FleetSyncResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    mode: str = "local"
    central_available: bool = False
    synced_at: datetime = Field(default_factory=utc_now)
    doctrine_version: str = "local"
    constraints: dict[str, object] = Field(default_factory=dict)
    blackboard: dict[str, object] = Field(default_factory=dict)
    message: str = "ok"


class FleetConstraintResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    bot_id: str
    mode: str = "local"
    doctrine_version: str = "local"
    constraints: dict[str, object] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=utc_now)


class FleetOutcomeReportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    event_type: str = Field(min_length=1, max_length=128)
    priority_class: int = Field(default=100, ge=0, le=1000)
    lease_owner: str = Field(default="", max_length=128)
    conflict_key: str = Field(default="", max_length=128)
    payload: dict[str, object] = Field(default_factory=dict)


class FleetOutcomeReportResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    accepted: bool = True
    central_available: bool = False
    queued_for_retry: bool = False
    mode: str = "local"
    result: dict[str, object] = Field(default_factory=dict)


class FleetRoleResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    bot_id: str
    role: str | None = None
    confidence: float = 0.0
    expires_at: datetime | None = None
    source: str = "local"
    mode: str = "local"


class FleetClaimRequestV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    claim_type: str = Field(default="territory", max_length=64)
    map_name: str | None = Field(default=None, max_length=128)
    channel: str = Field(default="0", max_length=64)
    objective_id: int | None = None
    resource_type: str | None = Field(default=None, max_length=64)
    resource_id: str | None = Field(default=None, max_length=128)
    quantity: int = Field(default=1, ge=1, le=100000)
    ttl_seconds: int = Field(default=180, ge=5, le=86400)
    priority: int = Field(default=100, ge=0, le=1000)
    metadata: dict[str, object] = Field(default_factory=dict)


class FleetClaimResponseV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    accepted: bool = True
    central_available: bool = False
    mode: str = "local"
    reason: str = "accepted"
    claim: dict[str, object] = Field(default_factory=dict)
    conflicts: list[dict[str, object]] = Field(default_factory=list)


class FleetBlackboardLocalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    bot_id: str
    mode: str = "local"
    generated_at: datetime = Field(default_factory=utc_now)
    constraints: dict[str, object] = Field(default_factory=dict)
    blackboard: dict[str, object] = Field(default_factory=dict)
    local_summary: dict[str, object] = Field(default_factory=dict)

