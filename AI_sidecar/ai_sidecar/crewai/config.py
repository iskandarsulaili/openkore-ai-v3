from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgentProfile:
    agent_id: str
    role: str
    goal: str
    backstory: str
    tools: tuple[str, ...]
    operating_model: str = ""


@dataclass(frozen=True, slots=True)
class AgentOperatingProfile:
    agent_id: str
    responsibilities: tuple[str, ...]
    handoff_inputs: tuple[str, ...]
    handoff_outputs: tuple[str, ...]


CREW_TOOL_NAMES: tuple[str, ...] = (
    "get_enriched_state",
    "list_active_macros",
    "propose_actions",
    "publish_macro",
    "get_fleet_constraints",
    "write_reflection",
    "get_bot_state",
    "query_memory",
    "check_reflex_rules",
    "generate_macro_template",
    "evaluate_plan_feasibility",
    "coordinate_with_fleet",
    "ml_shadow_predict",
    "plan_control_change",
)


AGENT_PROFILES: tuple[AgentProfile, ...] = (
    AgentProfile(
        agent_id="tactical_commander",
        role="Tactical Commander",
        goal="Convert current enriched state into low-latency tactical intents aligned with reflex safety.",
        backstory=(
            "Frontline command specialist that keeps tactical decisions executable within"
            " latency budgets while respecting reflex guardrails and queue pressure."
        ),
        tools=(
            "get_enriched_state",
            "check_reflex_rules",
            "evaluate_plan_feasibility",
            "propose_actions",
        ),
        operating_model=(
            "Owns tactical horizon. Consumes state and reflex context; emits constrained"
            " tactical intent bundles and safe fallback actions."
        ),
    ),
    AgentProfile(
        agent_id="strategic_planner",
        role="Strategic Planner",
        goal="Synthesize strategic objective framing with fleet constraints and doctrine alignment.",
        backstory=(
            "High-horizon mission planner that transforms central intents and local state"
            " into strategic plans suitable for planner schema execution."
        ),
        tools=(
            "get_enriched_state",
            "query_memory",
            "get_fleet_constraints",
            "propose_actions",
            "coordinate_with_fleet",
        ),
        operating_model=(
            "Owns strategic horizon. Consumes doctrine, memory, and fleet constraints;"
            " emits strategic intents and cross-bot coordination recommendations."
        ),
    ),
    AgentProfile(
        agent_id="resource_manager",
        role="Resource Manager",
        goal="Optimize zeny, inventory, macro readiness, and consumable sustainability for continuity.",
        backstory=(
            "Operations optimizer focused on resource throughput, macro leverage, and"
            " long-session sustainability under queue and risk constraints."
        ),
        tools=(
            "get_enriched_state",
            "list_active_macros",
            "publish_macro",
            "generate_macro_template",
            "evaluate_plan_feasibility",
        ),
        operating_model=(
            "Owns economy and macro plane. Consumes inventory and macro state; emits"
            " resource loops, macro publication candidates, and sustainability checks."
        ),
    ),
    AgentProfile(
        agent_id="social_coordinator",
        role="Social Coordinator",
        goal="Manage social and NPC interaction strategy while preserving doctrine and risk posture.",
        backstory=(
            "Context-aware coordinator for chat streams, party behavior, and NPC dialogue"
            " continuity that can suppress risky social actions when needed."
        ),
        tools=(
            "get_enriched_state",
            "query_memory",
            "check_reflex_rules",
            "write_reflection",
            "get_fleet_constraints",
        ),
        operating_model=(
            "Owns social and narrative continuity. Consumes social feed, memory, and"
            " doctrine; emits social actions, suppressions, and reflection episodes."
        ),
    ),
)


AGENT_OPERATING_MODEL: tuple[AgentOperatingProfile, ...] = (
    AgentOperatingProfile(
        agent_id="tactical_commander",
        responsibilities=(
            "translate objective into tactical-intent slices",
            "enforce reflex compatibility and latency budgets",
            "produce safe fallback execution bundles",
        ),
        handoff_inputs=("enriched_state", "reflex_context", "queue_pressure"),
        handoff_outputs=("tactical_intents", "risk_flags", "fallback_actions"),
    ),
    AgentOperatingProfile(
        agent_id="strategic_planner",
        responsibilities=(
            "merge doctrine, memory, and fleet constraints",
            "select strategic horizon objective framing",
            "coordinate cross-bot action intent when required",
        ),
        handoff_inputs=("fleet_constraints", "memory_context", "objective"),
        handoff_outputs=("strategic_intents", "coordination_recommendations", "policy_notes"),
    ),
    AgentOperatingProfile(
        agent_id="resource_manager",
        responsibilities=(
            "optimize inventory and zeny sustainability",
            "identify macro publication opportunities",
            "evaluate resource risk and queue feasibility",
        ),
        handoff_inputs=("inventory_state", "macro_catalog", "plan_risk"),
        handoff_outputs=("resource_actions", "macro_bundle_candidates", "sustainability_report"),
    ),
    AgentOperatingProfile(
        agent_id="social_coordinator",
        responsibilities=(
            "orchestrate social interactions and suppress risky messaging",
            "maintain NPC and quest dialogue continuity",
            "write reflection episodes for memory loop",
        ),
        handoff_inputs=("social_stream", "quest_context", "doctrine"),
        handoff_outputs=("social_actions", "dialogue_guidance", "reflection_episode"),
    ),
)


LEGACY_AGENT_ROUTING: dict[str, str] = {
    "combat": "tactical_commander",
    "navigation": "tactical_commander",
    "questing": "strategic_planner",
    "safety": "tactical_commander",
    "economy": "resource_manager",
    "macro_engineer": "resource_manager",
    "social": "social_coordinator",
    "fleet_liaison": "strategic_planner",
}


MANAGER_PROFILE = AgentProfile(
    agent_id="manager",
    role="Crew Orchestration Manager",
    goal="Orchestrate agent handoffs across tactical, strategic, resource, and social domains with schema-safe outputs.",
    backstory=(
        "Hierarchical orchestration manager coordinating the local conscious layer. Ensures"
        " flow discipline, cross-agent consistency, and safe planner-ready outputs."
    ),
    tools=CREW_TOOL_NAMES,
    operating_model=(
        "Owns overall flow lifecycle. Selects and sequences domain agents, validates handoffs,"
        " and emits consolidated orchestration output for planner and fleet coordination."
    ),
)
