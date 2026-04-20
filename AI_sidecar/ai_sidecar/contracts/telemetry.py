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
    queued_for_retry: int = 0


class TelemetryIncident(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    bot_id: str
    timestamp: datetime
    level: str
    category: str
    event: str
    message: str
    tags: dict[str, str] = Field(default_factory=dict)


class TelemetrySummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    window_minutes: int
    window_since: datetime
    total_events: int
    levels: dict[str, int] = Field(default_factory=dict)
    top_events: list[dict[str, object]] = Field(default_factory=list)
    recent_incidents: list[TelemetryIncident] = Field(default_factory=list)
