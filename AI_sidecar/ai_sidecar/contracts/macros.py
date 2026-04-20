from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from ai_sidecar.contracts.actions import ActionStatus
from ai_sidecar.contracts.common import ContractMeta


class MacroRoutine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    lines: list[str] = Field(default_factory=list)


class EventAutomacro(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    conditions: list[str] = Field(default_factory=list)
    call: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    parameters: dict[str, str] = Field(default_factory=dict)


class MacroPublishRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContractMeta
    target_bot_id: str | None = Field(default=None, min_length=1, max_length=128)
    macros: list[MacroRoutine] = Field(default_factory=list)
    event_macros: list[MacroRoutine] = Field(default_factory=list)
    automacros: list[EventAutomacro] = Field(default_factory=list)
    enqueue_reload: bool = True
    reload_conflict_key: str = Field(default="macro_reload", min_length=1, max_length=128)
    macro_plugin: str | None = Field(default="macro", min_length=1, max_length=64)
    event_macro_plugin: str | None = Field(default="eventMacro", min_length=1, max_length=64)


class MacroArtifactPaths(BaseModel):
    model_config = ConfigDict(extra="forbid")

    macro_file: str
    event_macro_file: str
    catalog_file: str
    manifest_file: str


class MacroPublication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    publication_id: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=128)
    content_sha256: str = Field(min_length=64, max_length=64)
    published_at: datetime
    paths: MacroArtifactPaths


class MacroPublishResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    published: bool
    message: str = ""
    publication: MacroPublication | None = None
    target_bot_id: str
    reload_queued: bool = False
    reload_action_id: str | None = None
    reload_status: ActionStatus | None = None
    reload_reason: str = ""
