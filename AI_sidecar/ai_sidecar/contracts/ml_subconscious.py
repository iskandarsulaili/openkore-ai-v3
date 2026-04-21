from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ai_sidecar.contracts.common import ContractMeta, utc_now


class DecisionSource(StrEnum):
    llm = "llm"
    rule = "rule"
    ml = "ml"


class ModelFamily(StrEnum):
    encounter_classifier = "encounter_classifier"
    loot_ranker = "loot_ranker"
    route_recovery_classifier = "route_recovery_classifier"
    npc_dialogue_predictor = "npc_dialogue_predictor"
    risk_anomaly_detector = "risk_anomaly_detector"
    memory_retrieval_ranker = "memory_retrieval_ranker"


class MLOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool = True
    reward: float = Field(default=0.0, ge=-1.0, le=1.0)
    latency_ms: float = Field(default=0.0, ge=0.0, le=600000.0)
    side_effects: list[str] = Field(default_factory=list)


class MLTrainingEpisode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(min_length=1, max_length=128)
    bot_id: str = Field(min_length=1, max_length=128)
    state_features: dict[str, object] = Field(default_factory=dict)
    decision_source: DecisionSource = DecisionSource.llm
    decision_payload: dict[str, object] = Field(default_factory=dict)
    executed_action: dict[str, object] = Field(default_factory=dict)
    outcome: MLOutcome = Field(default_factory=MLOutcome)
    safety_flags: list[str] = Field(default_factory=list)
    macro_version: str = Field(default="", max_length=128)


class MLObserveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    episode: MLTrainingEpisode


class MLObserveResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    message: str = "observed"
    trace_id: str
    episode_id: str
    bot_id: str
    reward: float = 0.0
    reward_breakdown: dict[str, float] = Field(default_factory=dict)
    labels_generated: int = 0


class MLTrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    model_family: ModelFamily
    bot_id: str | None = Field(default=None, min_length=1, max_length=128)
    incremental: bool = True
    max_samples: int = Field(default=2000, ge=50, le=50000)


class MLTrainResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    message: str = "trained"
    trace_id: str
    model_family: ModelFamily
    model_version: str = ""
    trained_samples: int = 0
    metrics: dict[str, float] = Field(default_factory=dict)
    ab_test: dict[str, object] = Field(default_factory=dict)


class MLModelVersionView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    created_at: datetime
    active: bool = False
    metrics: dict[str, float] = Field(default_factory=dict)
    path: str = ""


class MLModelFamilyView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: ModelFamily
    active_version: str | None = None
    versions: list[MLModelVersionView] = Field(default_factory=list)


class MLModelsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    generated_at: datetime = Field(default_factory=utc_now)
    models: list[MLModelFamilyView] = Field(default_factory=list)


class MLPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    model_family: ModelFamily
    state_features: dict[str, object] = Field(default_factory=dict)
    context: dict[str, object] = Field(default_factory=dict)
    planner_choice: dict[str, object] = Field(default_factory=dict)


class MLPredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    message: str = "predicted"
    trace_id: str
    model_family: ModelFamily
    model_version: str = ""
    recommendation: dict[str, object] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    shadow: dict[str, object] = Field(default_factory=dict)


class MLPromoteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    model_family: ModelFamily
    model_version: str = Field(min_length=1, max_length=128)
    canary_percentage: float = Field(default=10.0, ge=0.0, le=100.0)
    rollback_threshold: float = Field(default=0.25, ge=0.01, le=1.0)
    scope: dict[str, object] = Field(default_factory=dict)


class MLPromoteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    message: str = "promotion_updated"
    trace_id: str
    model_family: ModelFamily
    promotion: dict[str, object] = Field(default_factory=dict)


class MLPerformanceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    generated_at: datetime = Field(default_factory=utc_now)
    shadow_metrics: dict[str, object] = Field(default_factory=dict)
    promotion_metrics: dict[str, object] = Field(default_factory=dict)
    training_metrics: dict[str, object] = Field(default_factory=dict)


class MLDistillMacroRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    bot_id: str | None = Field(default=None, min_length=1, max_length=128)
    episode_ids: list[str] = Field(default_factory=list)
    min_support: int = Field(default=3, ge=2, le=100)
    max_steps: int = Field(default=8, ge=2, le=20)
    enqueue_reload: bool = False


class MLDistillMacroResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    message: str = "distilled"
    trace_id: str
    bot_id: str
    proposal_id: str = ""
    support: int = 0
    success_rate: float = 0.0
    macro: dict[str, object] = Field(default_factory=dict)
    automacro: dict[str, object] = Field(default_factory=dict)
    publication: dict[str, object] | None = None

