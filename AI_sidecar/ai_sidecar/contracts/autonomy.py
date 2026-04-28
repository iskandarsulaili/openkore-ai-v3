from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ai_sidecar.contracts.common import ContractMeta, utc_now


class GoalCategory(StrEnum):
    survival = "survival"
    job_advancement = "job_advancement"
    opportunistic_upgrades = "opportunistic_upgrades"
    leveling = "leveling"


class SituationalAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bot_id: str = Field(min_length=1, max_length=128)
    tick_id: str | None = Field(default=None, max_length=128)
    observed_at: datetime = Field(default_factory=utc_now)
    map_name: str | None = Field(default=None, max_length=64)
    in_combat: bool = False
    hp_ratio: float = Field(default=1.0, ge=0.0, le=2.0)
    danger_score: float = Field(default=0.0, ge=0.0, le=1.0)
    death_risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reconnect_age_s: float | None = Field(default=None, ge=0.0)
    is_dead: bool = False
    is_disconnected: bool = False
    skill_points: int = Field(default=0, ge=0)
    stat_points: int = Field(default=0, ge=0)
    base_level: int = Field(default=0, ge=0)
    job_level: int = Field(default=0, ge=0)
    job_name: str | None = Field(default=None, max_length=64)
    base_exp_ratio: float = Field(default=0.0, ge=0.0, le=2.0)
    job_exp_ratio: float = Field(default=0.0, ge=0.0, le=2.0)
    active_quest_count: int = Field(default=0, ge=0)
    objective_completion_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    overweight_ratio: float = Field(default=0.0, ge=0.0, le=2.0)
    item_count: int = Field(default=0, ge=0)
    zeny: int = Field(default=0, ge=0)
    vendor_exposure: int = Field(default=0, ge=0)
    replan_reasons: list[str] = Field(default_factory=list)
    trigger_flags: list[str] = Field(default_factory=list)
    placeholders: list[str] = Field(default_factory=list)
    progression_recommendation: dict[str, object] = Field(default_factory=dict)
    job_advancement: dict[str, object] = Field(default_factory=dict)
    opportunistic_upgrades: dict[str, object] = Field(default_factory=dict)
    raw_signals: dict[str, object] = Field(default_factory=dict)


class GoalDirective(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal_key: GoalCategory
    priority_rank: int = Field(default=999, ge=1, le=9999)
    active: bool = False
    objective: str = Field(min_length=1, max_length=512)
    rationale: str = Field(default="", max_length=1024)
    blockers: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)


class GoalStackState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bot_id: str = Field(min_length=1, max_length=128)
    tick_id: str | None = Field(default=None, max_length=128)
    horizon: str = Field(default="tactical", max_length=32)
    decision_version: str = Field(default="stage1-deterministic-v1", max_length=64)
    decided_at: datetime = Field(default_factory=utc_now)
    assessment: SituationalAssessment
    goal_stack: list[GoalDirective] = Field(default_factory=list)
    selected_goal: GoalDirective


class MissionDecisionClass(StrEnum):
    maintain = "maintain"
    adjust_objective = "adjust_objective"
    replan = "replan"


class AutonomyMissionContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bot_id: str = Field(min_length=1, max_length=128)
    horizon: str = Field(min_length=1, max_length=32)
    objective_input: str = Field(min_length=1, max_length=512)
    assessment: SituationalAssessment
    selected_goal: GoalDirective
    deterministic_goal_stack: list[GoalDirective] = Field(default_factory=list)
    deterministic_goal_summary: dict[str, object] = Field(default_factory=dict)
    replan_reasons: list[str] = Field(default_factory=list)
    trigger_reasons: list[str] = Field(default_factory=list)
    queue: dict[str, object] = Field(default_factory=dict)
    startup_gate: dict[str, object] = Field(default_factory=dict)
    planner_status: dict[str, object] = Field(default_factory=dict)
    snapshot: dict[str, object] = Field(default_factory=dict)
    enriched_state: dict[str, object] = Field(default_factory=dict)
    fleet_constraints: dict[str, object] = Field(default_factory=dict)
    fleet_blackboard: dict[str, object] = Field(default_factory=dict)
    recent_events: list[dict[str, object]] = Field(default_factory=list)
    memory_matches: list[dict[str, object]] = Field(default_factory=list)
    recent_episodes: list[dict[str, object]] = Field(default_factory=list)


class AutonomyMissionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_class: MissionDecisionClass = MissionDecisionClass.maintain
    selected_goal_key: str = Field(min_length=1, max_length=64)
    mission_objective: str = Field(min_length=1, max_length=512)
    mission_rationale: str = Field(default="", max_length=2048)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    execution_hints: list[dict[str, object]] = Field(default_factory=list)
    planner_handoff: dict[str, object] = Field(default_factory=dict)
    annotations: dict[str, object] = Field(default_factory=dict)


class AutonomyMissionDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    workload: str = Field(default="autonomy_mission_decision", min_length=1, max_length=128)
    trigger_reasons: list[str] = Field(default_factory=list)
    context: AutonomyMissionContext
    metadata: dict[str, object] = Field(default_factory=dict)


class AutonomyMissionDecisionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    message: str = "ok"
    trace_id: str
    bot_id: str
    generated_at: datetime = Field(default_factory=utc_now)
    decision: AutonomyMissionDecision | None = None
    errors: list[str] = Field(default_factory=list)
    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0
    route: dict[str, object] = Field(default_factory=dict)
