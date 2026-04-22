from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from ai_sidecar.contracts.common import utc_now


class BotOperationalState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bot_id: str
    map: str | None = None
    x: int | None = None
    y: int | None = None
    hp: int | None = None
    hp_max: int | None = None
    sp: int | None = None
    sp_max: int | None = None
    ai_sequence: str | None = None
    in_combat: bool = False
    target_id: str | None = None
    liveness_state: str = "unknown"
    updated_at: datetime = Field(default_factory=utc_now)
    raw: dict[str, object] = Field(default_factory=dict)
    # --- progression (populated from enriched snapshot) ---
    job_id: int | None = None
    job_name: str | None = None
    base_level: int | None = None
    job_level: int | None = None
    base_exp: int | None = None
    base_exp_max: int | None = None
    job_exp: int | None = None
    job_exp_max: int | None = None
    skill_points: int | None = None
    stat_points: int | None = None


class WorldEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(min_length=1, max_length=128)
    entity_type: str = Field(min_length=1, max_length=64)
    name: str | None = None
    map: str | None = None
    x: int | None = None
    y: int | None = None
    hp: int | None = None
    hp_max: int | None = None
    relation: str | None = None
    threat_score: float = 0.0
    last_seen_at: datetime = Field(default_factory=utc_now)
    attributes: dict[str, object] = Field(default_factory=dict)


class EncounterState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    in_encounter: bool = False
    target_id: str | None = None
    nearby_hostiles: int = 0
    nearby_allies: int = 0
    estimated_ttk_seconds: float | None = None
    risk_score: float = 0.0
    updated_at: datetime = Field(default_factory=utc_now)
    raw: dict[str, object] = Field(default_factory=dict)


class NavigationState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    map: str | None = None
    x: int | None = None
    y: int | None = None
    destination_map: str | None = None
    destination_x: int | None = None
    destination_y: int | None = None
    route_status: str = "idle"
    stuck_score: float = 0.0
    leash_state: str | None = None
    updated_at: datetime = Field(default_factory=utc_now)
    raw: dict[str, object] = Field(default_factory=dict)


class QuestState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_quests: list[str] = Field(default_factory=list)
    quest_status: dict[str, str] = Field(default_factory=dict)
    last_npc: str | None = None
    updated_at: datetime = Field(default_factory=utc_now)
    raw: dict[str, object] = Field(default_factory=dict)


class InventoryState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    zeny: int | None = None
    item_count: int = 0
    weight: int | None = None
    weight_max: int | None = None
    overweight_ratio: float | None = None
    consumables: dict[str, int] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=utc_now)
    raw: dict[str, object] = Field(default_factory=dict)


class EconomyState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    zeny: int | None = None
    zeny_delta_1m: int = 0
    zeny_delta_10m: int = 0
    vendor_exposure: int = 0
    updated_at: datetime = Field(default_factory=utc_now)
    raw: dict[str, object] = Field(default_factory=dict)


class SocialState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recent_chat_count: int = 0
    private_messages_5m: int = 0
    party_messages_5m: int = 0
    guild_messages_5m: int = 0
    last_interaction_at: datetime | None = None
    updated_at: datetime = Field(default_factory=utc_now)
    raw: dict[str, object] = Field(default_factory=dict)


class RiskState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    danger_score: float = 0.0
    death_risk_score: float = 0.0
    pvp_risk_score: float = 0.0
    anomaly_flags: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=utc_now)
    raw: dict[str, object] = Field(default_factory=dict)


class MacroExecutionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    macro_running: bool = False
    macro_name: str | None = None
    event_macro_running: bool = False
    event_macro_name: str | None = None
    last_result: str | None = None
    updated_at: datetime = Field(default_factory=utc_now)
    raw: dict[str, object] = Field(default_factory=dict)


class FleetIntentState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str | None = None
    assignment: str | None = None
    objective: str | None = None
    constraints: dict[str, object] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=utc_now)
    raw: dict[str, object] = Field(default_factory=dict)


class LearningFeatureVector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature_version: str = "v1"
    observed_at: datetime = Field(default_factory=utc_now)
    values: dict[str, float] = Field(default_factory=dict)
    labels: dict[str, str] = Field(default_factory=dict)
    raw: dict[str, object] = Field(default_factory=dict)


class EnrichedWorldState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bot_id: str
    generated_at: datetime = Field(default_factory=utc_now)
    operational: BotOperationalState
    encounter: EncounterState
    navigation: NavigationState
    quest: QuestState
    inventory: InventoryState
    economy: EconomyState
    social: SocialState
    risk: RiskState
    macro_execution: MacroExecutionState
    fleet_intent: FleetIntentState
    entities: list[WorldEntity] = Field(default_factory=list)
    features: LearningFeatureVector
    recent_event_ids: list[str] = Field(default_factory=list)


class EntityEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1, max_length=128)
    target_id: str = Field(min_length=1, max_length=128)
    relation: str = Field(min_length=1, max_length=64)
    weight: float = 1.0
    observed_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)


class NormalizedStateGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bot_id: str
    generated_at: datetime = Field(default_factory=utc_now)
    nodes: list[WorldEntity] = Field(default_factory=list)
    edges: list[EntityEdge] = Field(default_factory=list)
    projections: dict[str, object] = Field(default_factory=dict)

