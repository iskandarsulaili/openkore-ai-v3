from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ai_sidecar.contracts.actions import ActionProposal
from ai_sidecar.contracts.common import ContractMeta, utc_now
from ai_sidecar.contracts.macros import EventAutomacro, MacroRoutine


class PlanHorizon(StrEnum):
    tactical = "tactical"
    strategic = "strategic"
    reflection = "reflection"


class PlannerStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(min_length=1, max_length=128)
    kind: str = Field(min_length=1, max_length=64)
    target: str | None = Field(default=None, max_length=256)
    description: str = Field(default="", max_length=1024)
    success_predicates: list[str] = Field(default_factory=list)
    fallbacks: list[str] = Field(default_factory=list)


class StrategicPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(min_length=1, max_length=128)
    bot_id: str = Field(min_length=1, max_length=128)
    objective: str = Field(min_length=1, max_length=512)
    horizon: PlanHorizon = PlanHorizon.strategic
    assumptions: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    policies: list[str] = Field(default_factory=list)
    steps: list[PlannerStep] = Field(default_factory=list)
    recommended_actions: list[ActionProposal] = Field(default_factory=list)
    recommended_macros: list[MacroRoutine] = Field(default_factory=list)
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    requires_fleet_coordination: bool = False
    rationale: str = Field(default="", max_length=2000)
    expires_at: datetime


class TacticalIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_id: str = Field(min_length=1, max_length=128)
    objective: str = Field(min_length=1, max_length=512)
    priority: int = Field(default=100, ge=0, le=1000)
    constraints: list[str] = Field(default_factory=list)
    expected_latency_ms: int = Field(default=1000, ge=1, le=120000)


class TacticalIntentBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bundle_id: str = Field(min_length=1, max_length=128)
    bot_id: str = Field(min_length=1, max_length=128)
    generated_at: datetime = Field(default_factory=utc_now)
    intents: list[TacticalIntent] = Field(default_factory=list)
    actions: list[ActionProposal] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class MacroSynthesisProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str = Field(min_length=1, max_length=128)
    bot_id: str = Field(min_length=1, max_length=128)
    rationale: str = Field(default="", max_length=1024)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    macros: list[MacroRoutine] = Field(default_factory=list)
    event_macros: list[MacroRoutine] = Field(default_factory=list)
    automacros: list[EventAutomacro] = Field(default_factory=list)


class MemoryWriteback(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bot_id: str = Field(min_length=1, max_length=128)
    summary: str = Field(min_length=1, max_length=2048)
    semantic_tags: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)


class LearningLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bot_id: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=256)
    reward: float = Field(default=0.0, ge=-1.0, le=1.0)
    details: dict[str, object] = Field(default_factory=dict)


class EscalationNotice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bot_id: str = Field(min_length=1, max_length=128)
    severity: str = Field(min_length=1, max_length=64)
    reason: str = Field(min_length=1, max_length=512)
    recommended_action: str = Field(default="", max_length=256)


class PlannerContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bot_id: str = Field(min_length=1, max_length=128)
    objective: str = Field(min_length=1, max_length=512)
    horizon: PlanHorizon
    state: dict[str, object] = Field(default_factory=dict)
    recent_events: list[dict[str, object]] = Field(default_factory=list)
    memory_matches: list[dict[str, object]] = Field(default_factory=list)
    episodes: list[dict[str, object]] = Field(default_factory=list)
    doctrine: dict[str, object] = Field(default_factory=dict)
    fleet_constraints: dict[str, object] = Field(default_factory=dict)
    queue: dict[str, object] = Field(default_factory=dict)
    macros: dict[str, object] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=utc_now)


class PlannerPlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    objective: str = Field(min_length=1, max_length=512)
    horizon: PlanHorizon = PlanHorizon.tactical
    force_replan: bool = False
    max_steps: int = Field(default=8, ge=1, le=64)


class PlannerExplainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    plan_id: str | None = Field(default=None, max_length=128)
    action_id: str | None = Field(default=None, max_length=128)
    query: str = Field(default="", max_length=512)


class PlannerMacroPromoteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    objective: str = Field(min_length=1, max_length=512)
    min_repeat: int = Field(default=3, ge=2, le=20)


class PlannerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    message: str = "ok"
    trace_id: str
    strategic_plan: StrategicPlan | None = None
    tactical_bundle: TacticalIntentBundle | None = None
    macro_proposal: MacroSynthesisProposal | None = None
    memory_writeback: MemoryWriteback | None = None
    learning_label: LearningLabel | None = None
    escalation: EscalationNotice | None = None
    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0
    route: dict[str, object] = Field(default_factory=dict)


class PlannerStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    bot_id: str
    planner_healthy: bool
    current_objective: str | None = None
    last_plan_id: str | None = None
    last_provider: str | None = None
    last_model: str | None = None
    updated_at: datetime = Field(default_factory=utc_now)
    counters: dict[str, int] = Field(default_factory=dict)


class ProviderRouteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workload: str = Field(min_length=1, max_length=128)


class ProviderRouteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    workload: str
    selected_provider: str
    selected_model: str
    fallback_chain: list[str] = Field(default_factory=list)
    policy_version: str


class ProviderPolicyUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rules: dict[str, dict[str, object]] = Field(default_factory=dict)

