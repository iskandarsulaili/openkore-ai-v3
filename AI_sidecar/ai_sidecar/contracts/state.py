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


class BotStateSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    tick_id: str = Field(min_length=1, max_length=128)
    observed_at: datetime
    position: Position = Field(default_factory=Position)
    vitals: Vitals = Field(default_factory=Vitals)
    combat: CombatState = Field(default_factory=CombatState)
    inventory: InventoryDigest = Field(default_factory=InventoryDigest)
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
    capabilities: list[str] = Field(default_factory=list)
    attributes: dict[str, str] = Field(default_factory=dict)


class BotRegistrationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    registered: bool = True
    bot_id: str
    seen_at: datetime
