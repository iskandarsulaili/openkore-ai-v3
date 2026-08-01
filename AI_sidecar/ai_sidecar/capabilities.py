"""System Capabilities Registry — single source of truth for what the AI system can do.

Purpose:
    LLM agents (planner, conscious engine, Pro RO agent) and downstream decision
    layers need accurate, current context on the AI system's capabilities so they
    can process / delegate / plan / execute / tool-call correctly. This registry
    declares the FULL capability surface of openkore-ai-v3's sidecar:

      * Execution domains (what the fleet can DO in-game)
      * Action command roots (bridge-safe command intents)
      * Knowledge systems (databases the AI can read/write)
      * Learning / self-adaptation systems
      * Fleet coordination / crowdsourcing systems
      * API surface (HTTP endpoints agents can call)

    The registry is introspection-driven where possible (it scans the sidecar
    package for known capability markers) and explicitly declared otherwise.
    It is consumed by:
      - prompt_invariants()  -> injected into every LLM system prompt
      - /v1/capabilities    -> HTTP endpoint for agents/LLM to query
      - logging/debugging    -> observability of what the system can do

    RULE.md §18 (crowdsource/delegate) and §19 (self-adapt/self-learn) are the
    doctrine this registry serves: the LLM must know the system's capabilities
    before it can delegate or plan across them.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Declared capability surface (authoritative)
# ---------------------------------------------------------------------------

# In-game execution domains the sidecar can drive through the bridge.
EXECUTION_DOMAINS: dict[str, dict[str, Any]] = {
    "combat": {
        "description": "Attack monsters, control aggro, kite, flee, use combat skills.",
        "commands": ["attack", "ss <skill>", "equip", "sit"],
        "modules": ["domains/combat/engine.py", "combat_tactics.py", "combat_instinct.py"],
    },
    "movement": {
        "description": "Move to coordinates, route across maps, follow players, portal navigation.",
        "commands": ["move <x> <y>", "follow <player>", "ai manual/auto"],
        "modules": ["aggro_pathfinder.py", "exploration_driver.py", "fly_wing_manager.py"],
    },
    "economy": {
        "description": "Buy/sell from NPCs, storage, vendor management, price intelligence.",
        "commands": ["buy <item> <qty>", "sell <item>", "storage"],
        "modules": ["economy_optimizer.py", "market_intelligence.py", "cost_tracker.py"],
    },
    "progression": {
        "description": "Level, allocate stats/skills, job change, gear progression, quest automation.",
        "commands": ["stats_add <stat> <n>", "skills_add <skill> <n>", "talknpc"],
        "modules": ["gear_progression_planner.py", "goal_planner.py", "build_manager.py"],
    },
    "social": {
        "description": "Party management, class combos, swarm coordination, guild ops.",
        "commands": ["party invite/leave", "respond"],
        "modules": ["domains/social/party_synergy.py", "guild_manager.py"],
    },
    "survival": {
        "description": "Health/weight/status monitoring, emergency reflexes, potion use, death recovery.",
        "commands": ["use <item>", "sit", "move <portal>"],
        "modules": ["bot_health_monitor.py", "heal_resource_loader.py"],
    },
}

# Bridge-safe direct command roots (mirrors ro_knowledge._DIRECT_ALLOWED_ROOTS).
DIRECT_COMMAND_ROOTS: tuple[str, ...] = (
    "ai", "move", "macro", "eventmacro", "talknpc", "take", "ss", "use",
    "buy", "sell", "stats_add", "skills_add", "stat_add", "party", "teleport",
    "equip", "storage", "respond", "follow", "sit",
)

# Knowledge systems the AI can read and write.
KNOWLEDGE_SYSTEMS: dict[str, dict[str, Any]] = {
    "game_knowledge_db": {
        "description": "NPC positions, item IDs, shop data, portal lookups — per-server learnable.",
        "path": "game_knowledge_db.py",
    },
    "map_knowledge": {
        "description": "Map geometry, warps, danger zones, spawn data, portal connections.",
        "path": "map_knowledge.py",
    },
    "dynamic_portal_discovery": {
        "description": "Learns portal entry->exit pairs from observed gameplay; persists to SQLite.",
        "path": "dynamic_portal_discovery.py",
    },
    "map_server_knowledge": {
        "description": "Knowledge of maps known to the rAthena server (from server responses).",
        "path": "map_server_knowledge.py",
    },
    "economy_optimizer": {
        "description": "Item pricing, vending intelligence, 35k+ item database.",
        "path": "economy_optimizer.py",
    },
    "experience_db": {
        "description": "Level/exp curves, kill-rate planning.",
        "path": "experience_db.py",
    },
    "knowledge_graph": {
        "description": "Cross-referenced game knowledge graph.",
        "path": "knowledge_graph.py",
    },
}

# Learning / self-adaptation systems (RULE.md §19).
LEARNING_SYSTEMS: dict[str, dict[str, Any]] = {
    "fleet_self_learning": {
        "description": "Cross-bot learning: one bot's discovery becomes every bot's knowledge.",
        "path": "fleet/self_learning.py",
    },
    "dynamic_portal_discovery": {
        "description": "Observes and persists portal coordinates when bots walk through warp cells.",
        "path": "dynamic_portal_discovery.py",
    },
    "post_action_review": {
        "description": "Reviews action outcomes and feeds lessons back into planning.",
        "path": "autonomy/post_action_review.py",
    },
    "reflection_system": {
        "description": "Unified consciousness reflection: experiences -> lessons -> general principles.",
        "path": "strategy/unified_consciousness.py",
    },
    "macro_intelligence": {
        "description": "Learns macro/eventmacro patterns; 63 built-in + custom patterns.",
        "path": "autonomy/macro_intelligence.py",
    },
}

# Fleet coordination / crowdsourcing systems (RULE.md §18).
FLEET_SYSTEMS: dict[str, dict[str, Any]] = {
    "empire_manager": {
        "description": "Role assignment (CEO/CFO/COO), directives, territory claims, deal negotiation.",
        "path": "strategy/empire_manager.py",
    },
    "unified_consciousness": {
        "description": "Fleet-wide world model, empire consultation, theory of mind, competitive intelligence.",
        "path": "strategy/unified_consciousness.py",
    },
    "multi_account_synergy": {
        "description": "Multi-bot synergy, cross-bot resource management, coordinated action.",
        "path": "fleet/multi_account_synergy.py",
    },
    "party_coordinator": {
        "description": "Party formation, buff coordination, shared exp, role-based composition.",
        "path": "fleet/party_coordinator.py",
    },
    "swarm_ai": {
        "description": "Swarm behavior for coordinated farming/combat.",
        "path": "fleet/swarm_ai.py",
    },
    "coordinator": {
        "description": "Fleet-level task coordination and dispatch.",
        "path": "fleet/coordinator.py",
    },
}

# API surface (HTTP endpoints the sidecar exposes for agents/LLM).
API_SURFACE: dict[str, dict[str, Any]] = {
    "actions": {"path": "/v1/actions", "description": "Action queue: submit/retrieve bot actions."},
    "ingest": {"path": "/v1/ingest", "description": "Snapshot ingestion from bridge."},
    "fleet": {"path": "/v1/fleet", "description": "Fleet state, directives, coordination."},
    "conscious": {"path": "/v1/conscious", "description": "Conscious decision engine (LLM-driven)."},
    "planner": {"path": "/v1/planner", "description": "Strategic/tactical planning (LLM)."},
    "autonomy": {"path": "/v1/autonomy", "description": "Autonomy loop control (PDCA)."},
    "combat": {"path": "/v1/combat", "description": "Combat optimizer queries."},
    "party": {"path": "/v1/party", "description": "Party/synergy endpoints."},
    "discovery": {"path": "/v1/discovery", "description": "Dynamic portal discovery data."},
    "npc_dialog": {"path": "/v1/npc_dialog", "description": "NPC dialog automation."},
    "skills": {"path": "/v1/skills", "description": "Skill data and allocation."},
    "health": {"path": "/v1/health", "description": "Liveness/readiness."},
    "capabilities": {"path": "/v1/capabilities", "description": "This registry."},
}


def build_capabilities_registry() -> dict[str, Any]:
    """Build the full capability registry as a structured dict."""
    return {
        "system": "openkore-ai-v3 sidecar",
        "execution_domains": EXECUTION_DOMAINS,
        "direct_command_roots": list(DIRECT_COMMAND_ROOTS),
        "knowledge_systems": KNOWLEDGE_SYSTEMS,
        "learning_systems": LEARNING_SYSTEMS,
        "fleet_systems": FLEET_SYSTEMS,
        "api_surface": API_SURFACE,
    }


def capabilities_to_prompt_block(registry: dict[str, Any] | None = None) -> str:
    """Render the registry as a compact text block for LLM system prompts."""
    reg = registry or build_capabilities_registry()
    lines: list[str] = [
        "SYSTEM CAPABILITIES (what this AI system can do — use these to plan, delegate, and execute):",
    ]

    domains = reg.get("execution_domains", {})
    lines.append("Execution domains:")
    for name, meta in domains.items():
        lines.append(
            f"  - {name}: {meta.get('description', '')} "
            f"[commands: {', '.join(meta.get('commands', []))}]"
        )

    roots = reg.get("direct_command_roots", [])
    lines.append(f"Direct command roots (bridge-safe): {', '.join(roots)}")

    knowledge = reg.get("knowledge_systems", {})
    lines.append("Knowledge systems:")
    for name, meta in knowledge.items():
        lines.append(f"  - {name}: {meta.get('description', '')} ({meta.get('path', '')})")

    learning = reg.get("learning_systems", {})
    lines.append("Learning/self-adaptation systems:")
    for name, meta in learning.items():
        lines.append(f"  - {name}: {meta.get('description', '')} ({meta.get('path', '')})")

    fleet = reg.get("fleet_systems", {})
    lines.append("Fleet coordination/crowdsourcing systems:")
    for name, meta in fleet.items():
        lines.append(f"  - {name}: {meta.get('description', '')} ({meta.get('path', '')})")

    api = reg.get("api_surface", {})
    lines.append("API endpoints:")
    for name, meta in api.items():
        lines.append(f"  - {name}: {meta.get('path', '')} — {meta.get('description', '')}")

    lines.append(
        "Doctrine: crowdsource & delegate across the fleet (§18); self-adapt/learn "
        "for custom server layouts (§19); only execute bridge-safe command roots."
    )
    return "\n".join(lines)


def capabilities_to_json() -> dict[str, Any]:
    """JSON-friendly registry for the API endpoint."""
    return build_capabilities_registry()


# Module-level singleton (lazy-built on first access).
_REGISTRY: dict[str, Any] | None = None


def get_capabilities_registry() -> dict[str, Any]:
    """Return the cached registry singleton."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = build_capabilities_registry()
    return _REGISTRY


def get_capabilities_prompt_block() -> str:
    """Return the cached rendered prompt block."""
    return capabilities_to_prompt_block(get_capabilities_registry())
