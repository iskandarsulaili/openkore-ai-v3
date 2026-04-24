from __future__ import annotations

from typing import Any, Callable


def _merge_context(base: dict[str, Any], extra: dict[str, Any] | None) -> dict[str, Any]:
    merged = {key: value for key, value in base.items() if value not in (None, "", [], {})}
    if extra:
        for key, value in extra.items():
            if value in (None, "", [], {}):
                continue
            merged[key] = value
    return merged


def _format_context(context: dict[str, Any]) -> str:
    if not context:
        return ""
    lines = [f"- {key}: {value}" for key, value in context.items()]
    return "\nContext:\n" + "\n".join(lines)


def npc_dialogue_prompt(
    *,
    npc_name: str | None = None,
    objective: str | None = None,
    quest_name: str | None = None,
    quest_stage: str | None = None,
    job: str | None = None,
    level: int | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    merged = _merge_context(
        {
            "npc_name": npc_name,
            "objective": objective,
            "quest_name": quest_name,
            "quest_stage": quest_stage,
            "job": job,
            "level": level,
        },
        context,
    )
    return (
        "Domain Focus: NPC Dialogue\n"
        "- Prefer kind=npc or kind=quest steps with explicit NPC names and locations when known.\n"
        "- Include prerequisites (items, quest flags, job/level gates) before initiating dialogue.\n"
        "- Add a retry or fallback step if the NPC is missing or dialogue fails.\n"
        "- Keep descriptions concrete and confirm next expected response or action."
        f"{_format_context(merged)}"
    )


def job_advancement_prompt(
    *,
    job: str | None = None,
    job_level: int | None = None,
    base_level: int | None = None,
    required_items: list[str] | None = None,
    advancement_npc: str | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    merged = _merge_context(
        {
            "job": job,
            "job_level": job_level,
            "base_level": base_level,
            "required_items": required_items,
            "advancement_npc": advancement_npc,
        },
        context,
    )
    return (
        "Domain Focus: Job Advancement\n"
        "- Ensure job/base level requirements are met before traveling to the advancement NPC.\n"
        "- Plan to gather or purchase required items first if missing.\n"
        "- Use npc/quest steps for the advancement dialogue chain; include success predicates.\n"
        "- If requirements are unmet, include a safe training or acquisition fallback."
        f"{_format_context(merged)}"
    )


def market_operations_prompt(
    *,
    objective: str | None = None,
    sell_items: list[str] | None = None,
    buy_items: list[str] | None = None,
    budget: int | None = None,
    market_location: str | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    merged = _merge_context(
        {
            "objective": objective,
            "sell_items": sell_items,
            "buy_items": buy_items,
            "budget": budget,
            "market_location": market_location,
        },
        context,
    )
    return (
        "Domain Focus: Market Operations\n"
        "- Use econ steps with explicit buy/sell targets and price sensitivity notes.\n"
        "- Prefer storage or vending actions after verifying inventory weight limits.\n"
        "- Include a travel step to market hubs if needed and a fallback to pause trading.\n"
        "- Keep steps compact and avoid risky detours during trade runs."
        f"{_format_context(merged)}"
    )


def equipment_prompt(
    *,
    objective: str | None = None,
    target_slot: str | None = None,
    upgrade_goal: str | None = None,
    available_gear: list[str] | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    merged = _merge_context(
        {
            "objective": objective,
            "target_slot": target_slot,
            "upgrade_goal": upgrade_goal,
            "available_gear": available_gear,
        },
        context,
    )
    return (
        "Domain Focus: Equipment Management\n"
        "- Use equip steps to swap or upgrade gear with explicit slot and item names.\n"
        "- Consider weight, durability, and class restrictions before equipping.\n"
        "- Add a fallback to keep the current loadout if items are missing or unsafe.\n"
        "- If upgrades require services, plan npc/quest steps for the artisan interaction."
        f"{_format_context(merged)}"
    )


def pvp_tactics_prompt(
    *,
    objective: str | None = None,
    threat_level: str | None = None,
    escape_route: str | None = None,
    party_role: str | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    merged = _merge_context(
        {
            "objective": objective,
            "threat_level": threat_level,
            "escape_route": escape_route,
            "party_role": party_role,
        },
        context,
    )
    return (
        "Domain Focus: PvP Tactics\n"
        "- Prefer safety-first plans unless the objective explicitly requires engagement.\n"
        "- Include escape or regroup steps when risk is high; avoid dead-end routes.\n"
        "- Coordinate with party roles and avoid initiating fights without support.\n"
        "- Use combat or travel steps with clear triggers and fallbacks."
        f"{_format_context(merged)}"
    )


def quest_completion_prompt(
    *,
    quest_name: str | None = None,
    quest_stage: str | None = None,
    objectives: list[str] | None = None,
    handoff_npc: str | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    merged = _merge_context(
        {
            "quest_name": quest_name,
            "quest_stage": quest_stage,
            "objectives": objectives,
            "handoff_npc": handoff_npc,
        },
        context,
    )
    return (
        "Domain Focus: Quest Completion\n"
        "- Use quest steps with objective checkpoints and item/kill counts.\n"
        "- Include navigation to quest zones and NPC handoff steps with explicit names.\n"
        "- Add a recovery step if objectives stall (e.g., insufficient drops).\n"
        "- Keep the plan ordered to minimize backtracking."
        f"{_format_context(merged)}"
    )


def chat_response_prompt(
    *,
    intent: str | None = None,
    recipient: str | None = None,
    language: str | None = None,
    tone: str | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    merged = _merge_context(
        {
            "intent": intent,
            "recipient": recipient,
            "language": language,
            "tone": tone,
        },
        context,
    )
    return (
        "Domain Focus: Chat Responses\n"
        "- Use social steps with concise, polite responses aligned with doctrine.\n"
        "- Avoid revealing sensitive automation details or risky behaviors.\n"
        "- If the chat is about coordination, add a party or travel step as needed.\n"
        "- Confirm the recipient and intent in the description."
        f"{_format_context(merged)}"
    )


def map_navigation_prompt(
    *,
    destination: str | None = None,
    route_hint: str | None = None,
    risk_level: str | None = None,
    mobility_tools: list[str] | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    merged = _merge_context(
        {
            "destination": destination,
            "route_hint": route_hint,
            "risk_level": risk_level,
            "mobility_tools": mobility_tools,
        },
        context,
    )
    return (
        "Domain Focus: Map Navigation\n"
        "- Use travel steps with explicit destinations and waypoints.\n"
        "- Prefer safer routes when risk is elevated; note alternative paths.\n"
        "- Include a fallback if navigation is blocked or the map changes.\n"
        "- Keep movement steps short and ordered."
        f"{_format_context(merged)}"
    )


def party_coordination_prompt(
    *,
    party_goal: str | None = None,
    leader: str | None = None,
    role: str | None = None,
    rendezvous: str | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    merged = _merge_context(
        {
            "party_goal": party_goal,
            "leader": leader,
            "role": role,
            "rendezvous": rendezvous,
        },
        context,
    )
    return (
        "Domain Focus: Party Coordination\n"
        "- Use party and social steps for invites, follow orders, and comms.\n"
        "- Align movement and combat steps with leader instructions and rendezvous points.\n"
        "- Add fallback steps if party members are missing or unresponsive.\n"
        "- Ensure descriptions clearly state who to follow or assist."
        f"{_format_context(merged)}"
    )


def stat_skill_allocation_prompt(
    *,
    build_goal: str | None = None,
    available_points: int | None = None,
    priority_stats: list[str] | None = None,
    priority_skills: list[str] | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    merged = _merge_context(
        {
            "build_goal": build_goal,
            "available_points": available_points,
            "priority_stats": priority_stats,
            "priority_skills": priority_skills,
        },
        context,
    )
    return (
        "Domain Focus: Stat & Skill Allocation\n"
        "- Use skill_up or task steps with explicit stat/skill targets and point counts.\n"
        "- Validate points are available before allocating; avoid wasteful spending.\n"
        "- Prefer safe moments (out of combat) for allocation actions.\n"
        "- Include a fallback to pause if build requirements are unclear."
        f"{_format_context(merged)}"
    )


def domain_prompt_builders() -> dict[str, Callable[..., str]]:
    return {
        "npc_dialogue": npc_dialogue_prompt,
        "job_advancement": job_advancement_prompt,
        "market_operations": market_operations_prompt,
        "equipment": equipment_prompt,
        "pvp_tactics": pvp_tactics_prompt,
        "quest_completion": quest_completion_prompt,
        "chat_response": chat_response_prompt,
        "map_navigation": map_navigation_prompt,
        "party_coordination": party_coordination_prompt,
        "stat_skill_allocation": stat_skill_allocation_prompt,
    }
