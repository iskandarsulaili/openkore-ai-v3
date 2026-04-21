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
    telemetry_operational_window_minutes: int = Field(default=60, ge=1, le=1440)
    telemetry_recent_incidents_limit: int = Field(default=100, ge=1, le=1000)
    telemetry_backlog_max_events: int = Field(default=10000, ge=100, le=200000)

    latency_budget_ms: int = Field(default=12, ge=1, le=500)
    reflex_latency_budget_ms: int = Field(default=100, ge=10, le=1000)
    reflex_trigger_history_per_bot: int = Field(default=1000, ge=100, le=20000)

    sqlite_path: str = "AI_sidecar/data/sidecar.sqlite"
    sqlite_busy_timeout_ms: int = Field(default=300, ge=50, le=10000)
    persistence_snapshot_history_per_bot: int = Field(default=5000, ge=100, le=200000)
    persistence_audit_history: int = Field(default=50000, ge=1000, le=500000)

    memory_backend: str = "openmemory"  # sqlite | openmemory | auto
    memory_openmemory_mode: str = "local"
    memory_openmemory_path: str = "AI_sidecar/data/openmemory.sqlite"
    memory_embedding_dimensions: int = Field(default=384, ge=64, le=4096)
    memory_semantic_candidates: int = Field(default=500, ge=20, le=5000)
    memory_default_search_limit: int = Field(default=5, ge=1, le=50)


settings = SidecarSettings()
