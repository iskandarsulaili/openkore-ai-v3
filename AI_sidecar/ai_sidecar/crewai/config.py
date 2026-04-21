from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgentProfile:
    agent_id: str
    role: str
    goal: str
    backstory: str
    tools: tuple[str, ...]


CREW_TOOL_NAMES: tuple[str, ...] = (
    "get_bot_state",
    "query_memory",
    "check_reflex_rules",
    "generate_macro_template",
    "evaluate_plan_feasibility",
    "coordinate_with_fleet",
)


AGENT_PROFILES: tuple[AgentProfile, ...] = (
    AgentProfile(
        agent_id="combat",
        role="Combat Agent",
        goal="Design safe and efficient combat tactics, target priority, and skill rotations.",
        backstory=(
            "Veteran RO combat tactician who optimizes DPS and survivability while honoring"
            " local reflex safety boundaries and queue conflict policies."
        ),
        tools=("get_bot_state", "check_reflex_rules", "evaluate_plan_feasibility"),
    ),
    AgentProfile(
        agent_id="navigation",
        role="Navigation Agent",
        goal="Generate robust movement paths, transitions, and stuck-recovery routes.",
        backstory=(
            "Field navigation specialist with deep map transition knowledge, route risk"
            " awareness, and practical fallback path heuristics."
        ),
        tools=("get_bot_state", "evaluate_plan_feasibility", "coordinate_with_fleet"),
    ),
    AgentProfile(
        agent_id="questing",
        role="Questing Agent",
        goal="Plan quest chains, NPC interactions, and objective progression order.",
        backstory=(
            "Quest operations planner focused on minimizing deadtime and ensuring prerequisite"
            " correctness for reliable progression."
        ),
        tools=("get_bot_state", "query_memory", "evaluate_plan_feasibility"),
    ),
    AgentProfile(
        agent_id="economy",
        role="Economy Agent",
        goal="Optimize item valuation, market timing, and zeny growth loops.",
        backstory=(
            "Market analyst trained on item liquidity, storage flow, and farming-to-trade"
            " conversion strategy."
        ),
        tools=("get_bot_state", "query_memory", "evaluate_plan_feasibility"),
    ),
    AgentProfile(
        agent_id="social",
        role="Social Agent",
        goal="Coordinate party or guild interactions while reducing social risk.",
        backstory=(
            "Social protocol specialist for party etiquette, assist callouts, and chat-safe"
            " responses under doctrine constraints."
        ),
        tools=("get_bot_state", "query_memory", "check_reflex_rules"),
    ),
    AgentProfile(
        agent_id="safety",
        role="Safety Agent",
        goal="Assess risk, detect unsafe actions, and enforce emergency protocol overlays.",
        backstory=(
            "Hardline safety reviewer that prioritizes account survivability and policy"
            " compliance before throughput."
        ),
        tools=("get_bot_state", "check_reflex_rules", "evaluate_plan_feasibility"),
    ),
    AgentProfile(
        agent_id="fleet_liaison",
        role="Fleet Liaison Agent",
        goal="Coordinate cross-bot assignments, map partitioning, and role synchronization.",
        backstory=(
            "Fleet coordinator focused on preventing overlap, resource contention, and"
            " role drift across bots."
        ),
        tools=("get_bot_state", "coordinate_with_fleet", "evaluate_plan_feasibility"),
    ),
    AgentProfile(
        agent_id="macro_engineer",
        role="Macro Engineer Agent",
        goal="Generate and optimize macro templates for repeatable execution patterns.",
        backstory=(
            "Macro systems engineer translating repeated tactical sequences into maintainable"
            " macro bundles with safe hot-reload intent."
        ),
        tools=("generate_macro_template", "query_memory", "evaluate_plan_feasibility"),
    ),
)


MANAGER_PROFILE = AgentProfile(
    agent_id="manager",
    role="Crew Manager",
    goal="Resolve inter-agent conflicts and produce actionable strategic output.",
    backstory=(
        "Orchestration manager specialized in balancing combat, navigation, economy, social,"
        " safety, and macro dimensions into one coherent plan."
    ),
    tools=CREW_TOOL_NAMES,
)

