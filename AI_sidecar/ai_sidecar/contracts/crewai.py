from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from ai_sidecar.contracts.common import ContractMeta, utc_now
from ai_sidecar.planner.schemas import PlanHorizon, PlannerResponse


class CrewAgentDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(min_length=1, max_length=64)
    role: str = Field(min_length=1, max_length=128)
    goal: str = Field(min_length=1, max_length=512)
    tools: list[str] = Field(default_factory=list)
    operating_model: str = Field(default="", max_length=1024)
    responsibilities: list[str] = Field(default_factory=list)
    handoff_inputs: list[str] = Field(default_factory=list)
    handoff_outputs: list[str] = Field(default_factory=list)
    enabled: bool = True


class CrewStrategizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    objective: str = Field(min_length=1, max_length=512)
    horizon: PlanHorizon = PlanHorizon.strategic
    force_replan: bool = False
    max_steps: int = Field(default=12, ge=1, le=64)
    context_overrides: dict[str, object] = Field(default_factory=dict)


class CrewCoordinateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    task: str = Field(min_length=1, max_length=512)
    objective: str | None = Field(default=None, max_length=512)
    target_bots: list[str] = Field(default_factory=list)
    required_agents: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)


class CrewStrategizeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    message: str = "ok"
    trace_id: str
    bot_id: str
    objective: str
    generated_at: datetime = Field(default_factory=utc_now)
    agent_outputs: list[dict[str, object]] = Field(default_factory=list)
    consolidated_output: str = ""
    planner_response: PlannerResponse | None = None
    orchestrator: dict[str, object] = Field(default_factory=dict)
    duration_ms: float = 0.0
    errors: list[str] = Field(default_factory=list)


class CrewCoordinateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    message: str = "ok"
    trace_id: str
    bot_id: str
    task: str
    generated_at: datetime = Field(default_factory=utc_now)
    agent_outputs: list[dict[str, object]] = Field(default_factory=list)
    consolidated_output: str = ""
    planner_response: PlannerResponse | None = None
    orchestrator: dict[str, object] = Field(default_factory=dict)
    duration_ms: float = 0.0
    errors: list[str] = Field(default_factory=list)


class CrewToolExecuteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    tool_name: str = Field(min_length=1, max_length=128)
    arguments: dict[str, object] = Field(default_factory=dict)


class CrewToolExecuteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    message: str = "ok"
    trace_id: str
    bot_id: str
    tool_name: str
    result: dict[str, object] = Field(default_factory=dict)


class CrewAgentsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    total_agents: int
    agents: list[CrewAgentDescriptor] = Field(default_factory=list)


class CrewStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    generated_at: datetime = Field(default_factory=utc_now)
    crew_available: bool = True
    crewai_enabled: bool = True
    active_runs: int = 0
    counters: dict[str, int] = Field(default_factory=dict)
    agents: list[CrewAgentDescriptor] = Field(default_factory=list)
