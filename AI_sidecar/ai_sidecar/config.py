from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SidecarSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="OPENKORE_AI_",
        extra="ignore",
    )

    app_name: str = "openkore-ai-sidecar"
    env: str = "development"
    host: str = "127.0.0.1"
    port: int = 18081
    log_level: str = "INFO"
    log_json: bool = True
    enable_docs: bool = True

    contract_version: str = "v1"

    action_default_ttl_seconds: int = Field(default=30, ge=1, le=600)
    action_max_queue_per_bot: int = Field(default=128, ge=1, le=4096)
    snapshot_cache_ttl_seconds: int = Field(default=120, ge=1, le=3600)
    telemetry_max_per_bot: int = Field(default=500, ge=10, le=10000)

    latency_budget_ms: int = Field(default=12, ge=1, le=500)


settings = SidecarSettings()

