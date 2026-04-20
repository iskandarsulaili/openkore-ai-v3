from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from ai_sidecar.contracts.actions import ActionStatus


class BotAssignmentUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str | None = Field(default=None, max_length=64)
    assignment: str | None = Field(default=None, max_length=128)
    attributes: dict[str, str] = Field(default_factory=dict)


class BotStatusView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bot_id: str
    bot_name: str | None = None
    role: str | None = None
    assignment: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    attributes: dict[str, str] = Field(default_factory=dict)
    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None
    last_tick_id: str | None = None
    liveness_state: str = "unknown"
    pending_actions: int = 0
    latest_snapshot_at: datetime | None = None
    telemetry_events: int = 0


class BotListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    total_bots: int
    bots: list[BotStatusView] = Field(default_factory=list)


class BotStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    bot: BotStatusView
    recent_actions: list[dict[str, object]] = Field(default_factory=list)
    recent_snapshots: list[dict[str, object]] = Field(default_factory=list)
    latest_macro_publication: dict[str, object] | None = None
    recent_audit: list[dict[str, object]] = Field(default_factory=list)


class FleetStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    generated_at: datetime
    total_bots: int
    online_bots: int
    total_pending_actions: int
    action_status_totals: dict[str, int] = Field(default_factory=dict)
    telemetry_window: dict[str, object] = Field(default_factory=dict)
    counters: dict[str, int] = Field(default_factory=dict)


class AssignmentUpdateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    updated: bool
    bot: BotStatusView | None = None
    message: str = ""


def action_status_key(status: ActionStatus) -> str:
    return status.value

