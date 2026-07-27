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
    observability_enable_metrics: bool = True
    observability_enable_tracing: bool = True
    observability_trace_max_traces: int = Field(default=5000, ge=100, le=200000)
    observability_trace_max_events_per_trace: int = Field(default=200, ge=10, le=2000)
    observability_incident_max_open: int = Field(default=2000, ge=100, le=100000)
    observability_explainability_max_records: int = Field(default=10000, ge=500, le=200000)
    security_doctrine_denylist: str = "exploit,dupe,rmt,botting service"

    contract_version: str = "v1"

    action_default_ttl_seconds: int = Field(default=120, ge=1, le=600)
    action_max_queue_per_bot: int = Field(default=128, ge=1, le=4096)
    snapshot_cache_ttl_seconds: int = Field(default=120, ge=1, le=3600)
    telemetry_max_per_bot: int = Field(default=500, ge=10, le=10000)

    # ── Skills system ──
    skills_enabled: bool = True
    skills_max_context_tokens: int = Field(default=10000, ge=500, le=100000)
    skills_stale_after_days: int = Field(default=7, ge=1, le=365)
    skills_archive_after_days: int = Field(default=14, ge=1, le=730)
    skills_max_matched: int = Field(default=3, ge=1, le=10)
    skills_curator_interval_hours: int = Field(default=1, ge=1, le=168)
    skills_confidence_increment: float = Field(default=0.02, ge=0.0, le=1.0)
    skills_confidence_decrement: float = Field(default=0.05, ge=0.0, le=1.0)
    skills_confidence_decay: float = Field(default=0.9, ge=0.0, le=1.0)

    telemetry_operational_window_minutes: int = Field(default=60, ge=1, le=1440)
    telemetry_recent_incidents_limit: int = Field(default=100, ge=1, le=1000)
    telemetry_backlog_max_events: int = Field(default=10000, ge=100, le=200000)

    latency_budget_ms: int = Field(default=5000, ge=1, le=30000)
    reflex_latency_budget_ms: int = Field(default=500, ge=10, le=5000)
    # Reflex chain fallback: when all emit targets fail, use this as last resort
    # Set to True to enable direct command injection as fallback
    reflex_chain_fallback_enabled: bool = Field(default=True)
    reflex_trigger_history_per_bot: int = Field(default=1000, ge=100, le=20000)

    sqlite_path: str = "AI_sidecar/data/sidecar.sqlite"
    sqlite_busy_timeout_ms: int = Field(default=300, ge=50, le=10000)
    persistence_snapshot_history_per_bot: int = Field(default=5000, ge=100, le=200000)
    persistence_audit_history: int = Field(default=50000, ge=1000, le=500000)

    memory_backend: str = "openmemory"  # sqlite | openmemory | auto
    memory_openmemory_mode: str = "local"
    memory_openmemory_path: str = "AI_sidecar/data/openmemory.sqlite"
    memory_embedding_dimensions: int = Field(default=384, ge=64, le=4096)
    memory_embedding_mode: str = "local_hash"  # local_hash | provider
    memory_embedding_provider: str = "ollama"  # ollama | openai | deepseek
    memory_embedding_model: str = ""
    memory_semantic_candidates: int = Field(default=500, ge=20, le=5000)
    memory_default_search_limit: int = Field(default=5, ge=1, le=50)

    llm_timeout_seconds: float = Field(default=45.0, ge=1.0, le=600.0)
    llm_max_retries: int = Field(default=2, ge=0, le=8)
    llm_prompt_max_chars: int = Field(default=32000, ge=1024, le=200000)
    # ── Cost Control ──────────────────────────────────────────────
    llm_cost_tier: str = "standard"  # off | saver | standard | max
    llm_daily_budget_tokens: int = Field(default=100000, ge=0, le=10000000)
    llm_max_calls_per_hour: int = Field(default=30, ge=0, le=1000)
    llm_skip_if_heuristic: bool = True
    llm_heuristic_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    llm_cost_tier_pricing: str = '{"off":0,"saver":0.05,"standard":0.30,"max":0.60}'  # $/M tokens

    # Cost mode: saver = minimal LLM, standard = balanced, max = full LLM
    # saver: heuristic + game engine for 95% of decisions, LLM only for novel situations
    # standard: heuristic + game engine + occasional LLM for strategic planning
    # max: full LLM for all horizons, game engine as fallback
    cost_mode: str = "standard"  # saver | standard | max

    # Game engine settings
    game_engine_enabled: bool = True
    game_engine_knowledge_path: str = "knowledge/knowledge.json"
    game_engine_auto_learn: bool = True
    game_engine_learning_rate: float = Field(default=0.1, ge=0.0, le=1.0)

    # Hunting zone auto-discovery
    hunting_zone_auto_discover: bool = True
    hunting_zone_min_monsters: int = Field(default=3, ge=1, le=100)
    hunting_zone_max_distance: int = Field(default=5, ge=1, le=20)

    # Anti-detection
    anti_detection_enabled: bool = True
    anti_detection_random_delay_ms: tuple = (100, 500)
    anti_detection_human_like_movement: bool = True
    anti_detection_session_rotation_hours: float = Field(default=4.0, ge=0.5, le=24.0)

    # Multi-bot (unlimited)
    multi_bot_max_bots: int = Field(default=100, ge=1, le=1000)
    multi_bot_stagger_delay_s: float = Field(default=10.0, ge=1.0, le=60.0)
    multi_bot_server_timeout_s: float = Field(default=120.0, ge=30.0, le=600.0)
    multi_bot_auto_restart: bool = True
    multi_bot_auto_restart_interval_s: float = Field(default=60.0, ge=10.0, le=600.0)

    # Swarm AI
    swarm_ai_enabled: bool = True
    swarm_ai_formation: str = "auto"  # auto | vanguard | wedge | line | spread | surround | protect | diamond | flank | retreat
    swarm_ai_skill_combos: bool = True
    swarm_ai_party_heal: bool = True
    swarm_ai_aggro_share: bool = True
    swarm_ai_loot_share: bool = True

    # ── API Authentication ────────────────────────────────────────
    api_auth_enabled: bool = False  # Enable manually + set api_auth_token for the bridge
    api_auth_token: str = ""  # Set via env or auto-generated at startup

    provider_ollama_enabled: bool = True
    provider_ollama_base_url: str = "http://127.0.0.1:11434"  # Override via PROVIDER_OLLAMA_BASE_URL env
    provider_ollama_default_model: str = "qwen3.6:35b-a3b-q4_K_M"
    provider_ollama_tactical_model: str = "qwen3.6:35b-a3b-q4_K_M"
    provider_ollama_strategic_model: str = "qwen3.6:35b-a3b-q4_K_M"
    provider_ollama_reflection_model: str = "qwen3.6:35b-a3b-q4_K_M"
    provider_ollama_embedding_model: str = "nomic-embed-text"

    provider_deepseek_enabled: bool = True
    provider_deepseek_base_url: str = "https://api.deepseek.com/v1"
    provider_deepseek_api_key: str = ""
    provider_deepseek_default_model: str = "deepseek-chat"
    provider_deepseek_tactical_model: str = "deepseek-chat"
    provider_deepseek_strategic_model: str = "deepseek-chat"
    provider_deepseek_reflection_model: str = "deepseek-chat"
    provider_deepseek_embedding_model: str = "text-embedding-3-small"

    provider_openai_enabled: bool = True
    provider_openai_base_url: str = "https://api.openai.com/v1"
    provider_openai_api_key: str = ""
    provider_openai_default_model: str = "gpt-4o-mini"
    provider_openai_tactical_model: str = "gpt-4o-mini"
    provider_openai_strategic_model: str = "gpt-4o-mini"
    provider_openai_reflection_model: str = "gpt-4o-mini"
    provider_openai_embedding_model: str = "text-embedding-3-small"

    provider_policy_json: str = ""

    planner_tactical_budget_ms: int = Field(default=15000, ge=100, le=120000)
    planner_strategic_budget_ms: int = Field(default=30000, ge=500, le=300000)
    planner_timeout_seconds: float = Field(default=45.0, ge=1.0, le=600.0)
    planner_retries: int = Field(default=2, ge=0, le=8)

    autonomy_objective_max_age_cycles: int = Field(default=6, ge=1, le=10000)
    autonomy_max_active_objectives: int = Field(default=3, ge=1, le=64)
    autonomy_priority_decay_per_cycle: float = Field(default=0.10, ge=0.0, le=1.0)
    autonomy_objective_rotation_cooldown_s: float = Field(default=20.0, ge=0.0, le=3600.0)
    autonomy_ranked_objectives: str = "grind,recovery,economy,quest"
    autonomy_stale_plan_threshold_s: float = Field(default=60.0, ge=1.0, le=36000.0)
    autonomy_death_recovery_cooldown_s: float = Field(default=15.0, ge=0.0, le=3600.0)
    autonomy_reconnect_grace_s: float = Field(default=20.0, ge=0.0, le=3600.0)
    autonomy_preferred_grind_maps: str = ""
    autonomy_preferred_grind_map_policy: str = "prefer"

    crewai_enabled: bool = True
    crewai_verbose: bool = False
    crewai_memory_enabled: bool = False

    fleet_central_enabled: bool = True
    fleet_central_base_url: str = "http://127.0.0.1:18090"
    fleet_request_timeout_seconds: float = Field(default=8.0, ge=0.1, le=30.0)
    fleet_outcome_backlog_limit: int = Field(default=2000, ge=100, le=200000)
    fleet_local_partition_ttl_seconds: int = Field(default=600, ge=30, le=86400)

    # ── Keep Alive Mode ──────────────────────────────────────────
    keep_alive_enabled: bool = False
    keep_alive_timeout_minutes: int = Field(default=30, ge=1, le=1440)
    keep_alive_poll_interval_s: float = Field(default=30.0, ge=5.0, le=300.0)
    game_server_host: str = "asgardsglory.ddns.net"
    game_server_port: int = 6900


settings = SidecarSettings()
