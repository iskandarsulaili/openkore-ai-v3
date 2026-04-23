from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai_sidecar.contracts.actions import ActionPriorityTier
from ai_sidecar.contracts.common import ContractMeta, utc_now


class ReflexFactOp(StrEnum):
    eq = "eq"
    neq = "neq"
    gt = "gt"
    gte = "gte"
    lt = "lt"
    lte = "lte"
    in_ = "in"
    not_in = "not_in"
    contains = "contains"
    startswith = "startswith"
    endswith = "endswith"
    exists = "exists"


class ReflexPredicate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fact: str = Field(min_length=1, max_length=256)
    op: ReflexFactOp
    value: object | None = None


class ReflexTriggerClause(BaseModel):
    model_config = ConfigDict(extra="forbid")

    all: list[ReflexPredicate] = Field(default_factory=list)
    any: list[ReflexPredicate] = Field(default_factory=list)


class ReflexActionTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = Field(default="command", min_length=1, max_length=64)
    command: str = Field(default="", max_length=256)
    priority_tier: ActionPriorityTier = ActionPriorityTier.reflex
    conflict_key: str | None = Field(default=None, max_length=128)
    metadata: dict[str, object] = Field(default_factory=dict)


class ReflexCategory(StrEnum):
    combat = "combat"
    survival = "survival"
    interaction = "interaction"


class ReflexPlannerInterop(StrEnum):
    override = "override"
    complement = "complement"


class ReflexRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    enabled: bool = True
    priority: int = Field(default=100, ge=0, le=1000)
    trigger: ReflexTriggerClause
    guards: list[ReflexPredicate] = Field(default_factory=list)
    action_template: ReflexActionTemplate
    fallback_macro: str | None = Field(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    cooldown_ms: int = Field(default=1000, ge=0, le=600000)
    circuit_breaker_key: str | None = Field(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    event_macro_conditions: list[str] = Field(default_factory=list)
    category: ReflexCategory = ReflexCategory.survival
    planner_interop: ReflexPlannerInterop = ReflexPlannerInterop.override

    @field_validator("event_macro_conditions")
    @classmethod
    def _normalize_event_macro_conditions(cls, value: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in value:
            normalized = str(item).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            out.append(normalized)
        return out

    @model_validator(mode="after")
    def _validate_rule_contract(self) -> Self:
        has_trigger = bool(self.trigger.all or self.trigger.any)
        if not has_trigger:
            raise ValueError("reflex_rule_trigger_missing")

        has_direct_action = bool((self.action_template.command or "").strip())
        has_macro_action = bool(self.fallback_macro or self.event_macro_conditions)
        if not has_direct_action and not has_macro_action:
            raise ValueError("reflex_rule_action_missing")

        return self


class ReflexRuleUpsertRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    rule: ReflexRule


class ReflexRuleUpsertResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    saved: bool
    rule_id: str
    message: str = ""


class ReflexRuleEnableRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    enabled: bool


class ReflexRuleEnableResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    updated: bool
    rule_id: str
    enabled: bool
    message: str = ""


class ReflexRuleListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    bot_id: str
    rules: list[ReflexRule] = Field(default_factory=list)


class ReflexTriggerRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trigger_id: str = Field(min_length=1, max_length=128)
    bot_id: str = Field(min_length=1, max_length=128)
    rule_id: str = Field(min_length=1, max_length=128)
    priority: int = 100
    category: ReflexCategory = ReflexCategory.survival
    planner_interop: ReflexPlannerInterop = ReflexPlannerInterop.override
    event_id: str | None = Field(default=None, max_length=128)
    event_family: str = Field(min_length=1, max_length=64)
    event_type: str = Field(min_length=1, max_length=128)
    matched_at: datetime = Field(default_factory=utc_now)
    latency_ms: float = 0.0
    suppressed: bool = False
    suppression_reason: str = ""
    emitted: bool = False
    execution_target: str | None = Field(default=None, max_length=64)
    action_id: str | None = Field(default=None, max_length=128)
    outcome: str = ""
    detail: str = ""


class ReflexTriggerListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    bot_id: str
    triggers: list[ReflexTriggerRecord] = Field(default_factory=list)


class ReflexBreakerStatusView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1, max_length=128)
    family: str = Field(min_length=1, max_length=64)
    state: str = Field(min_length=1, max_length=32)
    failure_count: int = 0
    total_failures: int = 0
    total_successes: int = 0
    opened_until: datetime | None = None
    last_failure_reason: str = ""
    updated_at: datetime = Field(default_factory=utc_now)


class ReflexBreakerListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    bot_id: str
    breakers: list[ReflexBreakerStatusView] = Field(default_factory=list)
