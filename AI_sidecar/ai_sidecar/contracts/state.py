from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from ai_sidecar.contracts.common import ContractMeta


class Position(BaseModel):
    model_config = ConfigDict(extra="forbid")

    map: str | None = None
    x: int | None = None
    y: int | None = None


class Vitals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hp: int | None = None
    hp_max: int | None = None
    sp: int | None = None
    sp_max: int | None = None
    weight: int | None = None
    weight_max: int | None = None


class CombatState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ai_sequence: str | None = None
    target_id: str | None = None
    is_in_combat: bool = False


class InventoryDigest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    zeny: int | None = None
    item_count: int | None = None


class InventoryItemDigest(BaseModel):
    """Compact inventory item details for state enrichment."""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1, max_length=128)
    name: str | None = Field(default=None, max_length=128)
    quantity: int = Field(default=0, ge=0)
    category: str | None = Field(default=None, max_length=64)
    equipped: bool = False
    marketable: bool = False
    buy_price: int | None = Field(default=None, ge=0)
    sell_price: int | None = Field(default=None, ge=0)
    metadata: dict[str, object] = Field(default_factory=dict)


class ProgressionDigest(BaseModel):
    """Character job/level progression snapshot — optional, provided by bridge v2."""

    model_config = ConfigDict(extra="forbid")

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


class SkillDigest(BaseModel):
    """Skill progression details from bridge v2+."""

    model_config = ConfigDict(extra="forbid")

    skill_id: str = Field(min_length=1, max_length=128)
    skill_name: str | None = Field(default=None, max_length=128)
    level: int = Field(default=0, ge=0)
    max_level: int | None = Field(default=None, ge=0)
    category: str | None = Field(default=None, max_length=64)
    cooldown_ms: int | None = Field(default=None, ge=0)
    metadata: dict[str, object] = Field(default_factory=dict)


class QuestObjectiveDigest(BaseModel):
    """Objective-level quest progression details."""

    model_config = ConfigDict(extra="forbid")

    objective_id: str = Field(min_length=1, max_length=128)
    description: str = Field(default="", max_length=256)
    status: str = Field(default="unknown", max_length=64)
    current: int | None = Field(default=None, ge=0)
    target: int | None = Field(default=None, ge=0)
    metadata: dict[str, object] = Field(default_factory=dict)


class QuestDigest(BaseModel):
    """Quest state payload for compact snapshot ingestion."""

    model_config = ConfigDict(extra="forbid")

    quest_id: str = Field(min_length=1, max_length=128)
    state: str = Field(default="unknown", max_length=64)
    npc: str | None = Field(default=None, max_length=128)
    title: str | None = Field(default=None, max_length=256)
    objectives: list[QuestObjectiveDigest] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)


class NpcRelationshipDigest(BaseModel):
    """NPC social affinity/relationship signal snapshot."""

    model_config = ConfigDict(extra="forbid")

    npc_id: str = Field(min_length=1, max_length=128)
    npc_name: str | None = Field(default=None, max_length=128)
    relation: str = Field(default="neutral", max_length=64)
    affinity_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    trust_score: float = Field(default=0.0, ge=0.0, le=1.0)
    interaction_count: int = Field(default=0, ge=0)
    metadata: dict[str, object] = Field(default_factory=dict)


class MarketQuoteDigest(BaseModel):
    """Market quote sample for economy/market enrichment."""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1, max_length=128)
    item_name: str | None = Field(default=None, max_length=128)
    buy_price: int | None = Field(default=None, ge=0)
    sell_price: int | None = Field(default=None, ge=0)
    quantity: int | None = Field(default=None, ge=0)
    source: str | None = Field(default=None, max_length=64)
    metadata: dict[str, object] = Field(default_factory=dict)


class MarketDigest(BaseModel):
    """Compact market snapshot for planner-side economic reasoning."""

    model_config = ConfigDict(extra="forbid")

    listings: list[MarketQuoteDigest] = Field(default_factory=list)
    vendor_exposure: int = Field(default=0, ge=0)
    transaction_count_10m: int = Field(default=0, ge=0)
    metadata: dict[str, object] = Field(default_factory=dict)


class ActorDigest(BaseModel):
    """Compact summary of a nearby actor (mob/player/NPC)."""

    model_config = ConfigDict(extra="ignore")  # forward-compat for extra actor fields

    actor_id: str = Field(min_length=1, max_length=64)
    actor_type: str = Field(default="unknown", max_length=32)  # monster | player | npc
    name: str | None = None
    relation: str | None = None  # hostile | ally | neutral | party | unknown
    x: int | None = None
    y: int | None = None
    hp: int | None = None
    hp_max: int | None = None
    level: int | None = None
    distance: float | None = None


class BotStateSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    tick_id: str = Field(min_length=1, max_length=128)
    observed_at: datetime
    position: Position = Field(default_factory=Position)
    vitals: Vitals = Field(default_factory=Vitals)
    combat: CombatState = Field(default_factory=CombatState)
    inventory: InventoryDigest = Field(default_factory=InventoryDigest)
    inventory_items: list[InventoryItemDigest] = Field(default_factory=list)
    progression: ProgressionDigest = Field(default_factory=ProgressionDigest)
    skills: list[SkillDigest] = Field(default_factory=list)
    quests: list[QuestDigest] = Field(default_factory=list)
    npc_relationships: list[NpcRelationshipDigest] = Field(default_factory=list)
    market: MarketDigest = Field(default_factory=MarketDigest)
    actors: list[ActorDigest] = Field(default_factory=list, max_length=64)
    raw: dict[str, object] = Field(default_factory=dict)


class SnapshotIngestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    accepted: bool = True
    message: str = "snapshot accepted"
    bot_id: str
    tick_id: str


class BotRegistrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    bot_name: str | None = None
    role: str | None = Field(default=None, max_length=64)
    assignment: str | None = Field(default=None, max_length=128)
    capabilities: list[str] = Field(default_factory=list)
    attributes: dict[str, str] = Field(default_factory=dict)


class BotRegistrationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    registered: bool = True
    bot_id: str
    seen_at: datetime
    role: str | None = None
    assignment: str | None = None
    liveness_state: str = "online"
