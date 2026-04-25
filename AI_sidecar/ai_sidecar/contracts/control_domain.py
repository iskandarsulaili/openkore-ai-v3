from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ai_sidecar.contracts.common import ContractMeta


class ControlArtifactType(StrEnum):
    config = "config"
    table = "table"
    macro = "macro"
    policy = "policy"


class ControlChangeState(StrEnum):
    drafted = "drafted"
    planned = "planned"
    applied = "applied"
    validated = "validated"
    rolled_back = "rolled_back"
    failed = "failed"


class ControlOwnerScope(StrEnum):
    sidecar = "sidecar"
    operator = "operator"
    system = "system"


class ControlArtifactIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bot_id: str = Field(min_length=1, max_length=128)
    profile: str | None = Field(default=None, max_length=128)
    artifact_type: ControlArtifactType
    name: str = Field(min_length=1, max_length=128)
    path: str = Field(min_length=1, max_length=256)


class ControlArtifactRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    identity: ControlArtifactIdentity
    owner: ControlOwnerScope = ControlOwnerScope.sidecar
    checksum: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=128)
    updated_at: datetime
    metadata: dict[str, object] = Field(default_factory=dict)


class ControlPolicyRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key_pattern: str = Field(min_length=1, max_length=256)
    owner: ControlOwnerScope = ControlOwnerScope.sidecar
    allow_write: bool = True
    reason: str = Field(default="", max_length=256)


class ControlPolicySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = Field(min_length=1, max_length=128)
    protected_prefixes: list[str] = Field(default_factory=list)
    protected_exact: list[str] = Field(default_factory=list)
    rules: list[ControlPolicyRule] = Field(default_factory=list)


class ControlChangeItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1, max_length=256)
    previous: str = Field(default="", max_length=4096)
    updated: str = Field(default="", max_length=4096)
    owner: ControlOwnerScope = ControlOwnerScope.sidecar
    reason: str = Field(default="", max_length=256)


class ControlChangePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(min_length=1, max_length=128)
    bot_id: str = Field(min_length=1, max_length=128)
    profile: str | None = Field(default=None, max_length=128)
    artifact: ControlArtifactIdentity
    created_at: datetime
    policy_version: str = Field(min_length=1, max_length=128)
    checksum_before: str = Field(min_length=1, max_length=128)
    checksum_after: str = Field(min_length=1, max_length=128)
    changes: list[ControlChangeItem] = Field(default_factory=list)
    annotations: dict[str, object] = Field(default_factory=dict)


class ControlPlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    bot_id: str = Field(min_length=1, max_length=128)
    profile: str | None = Field(default=None, max_length=128)
    artifact_type: ControlArtifactType
    name: str = Field(min_length=1, max_length=128)
    target_path: str = Field(min_length=1, max_length=256)
    desired: dict[str, str] = Field(default_factory=dict)
    source: str = Field(default="manual", max_length=32)


class ControlPlanResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    message: str = "planned"
    plan: ControlChangePlan


class ControlApplyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    plan_id: str = Field(min_length=1, max_length=128)
    dry_run: bool = False


class ControlApplyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    message: str = "applied"
    state: ControlChangeState
    plan: ControlChangePlan
    reload_action_id: str | None = None
    reload_status: str | None = None
    reload_reason: str = ""


class ControlValidateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    plan_id: str = Field(min_length=1, max_length=128)


class ControlValidateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    message: str = "validated"
    state: ControlChangeState
    plan: ControlChangePlan
    drift: list[ControlChangeItem] = Field(default_factory=list)


class ControlRollbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    plan_id: str = Field(min_length=1, max_length=128)
    reason: str = Field(default="", max_length=256)


class ControlRollbackResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    message: str = "rolled_back"
    state: ControlChangeState
    plan: ControlChangePlan
    rollback_lineage: list[str] = Field(default_factory=list)


class ControlArtifactsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    artifacts: list[ControlArtifactRecord] = Field(default_factory=list)

