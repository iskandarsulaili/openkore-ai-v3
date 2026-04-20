from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ai_sidecar.contracts.common import ContractMeta


class TelemetryLevel(StrEnum):
    debug = "debug"
    info = "info"
    warning = "warning"
    error = "error"


class TelemetryEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    level: TelemetryLevel = TelemetryLevel.info
    category: str = Field(min_length=1, max_length=64)
    event: str = Field(min_length=1, max_length=128)
    message: str = Field(default="", max_length=512)
    metrics: dict[str, float] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)


class TelemetryIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    events: list[TelemetryEvent] = Field(default_factory=list)


class TelemetryIngestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    accepted: int
    dropped: int

