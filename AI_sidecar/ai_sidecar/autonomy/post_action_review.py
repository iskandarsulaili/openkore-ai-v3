"""Post-Action Review — trigger skill creation when agents make verified discoveries.

After every key action (discovery, heal strategy, navigation route, economy finding),
this module evaluates whether a skill should be created or updated.

Inspired by Hermes Agent's background_review.py pattern.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ai_sidecar import skills_manager, skills_usage
from ai_sidecar.crewai.agents.base_agent import maybe_create_skill

logger = logging.getLogger(__name__)


def record_lesson(
    self_awareness,
    content: str,
    *,
    importance: int = 5,
    dedupe: bool = True,
) -> Dict[str, Any]:
    """Persist a durable lesson into MEMORY.md via the SelfAwareness layer.

    This is the self-learning write-path of the loop: the conscious tier, after
    a verified action/outcome, records what it learned so future reasoning calls
    (which have SOUL + MEMORY injected by ``SelfAwareness.inject``) honor it.

    Fully server-agnostic: ``content`` must be a general lesson (a decision
    heuristic, a routing fact derived from live discovery, an anti-pattern), NOT
    a hardcoded map/item/coord literal. Server-specific facts belong in the
    DB-backed ``server_solutions`` store, not MEMORY.md.

    Args:
        self_awareness: The runtime's SelfAwareness instance (SOUL/MEMORY owner).
        content: The lesson text (general, agnostic).
        importance: 1-10; >=8 shortlists for the curated top block.
        dedupe: If True, skip when an identical lesson already exists.

    Returns:
        Dict with success/error + usage stats, mirroring add_lesson's contract.
    """
    if self_awareness is None or not hasattr(self_awareness, "add_lesson"):
        return {"success": False, "error": "self_awareness_unavailable"}

    content = (content or "").strip()
    if not content:
        return {"success": False, "error": "Content cannot be empty."}

    # Dedupe: never let the same lesson accumulate across cycles.
    if dedupe:
        existing = {e.strip().lower() for e in self_awareness.memory_entries}
        if content.lower() in existing:
            return {"success": False, "error": "duplicate_lesson", "duplicate": True}

    result = self_awareness.add_lesson(content)
    if result.get("success"):
        logger.info("Lesson recorded to MEMORY.md (importance=%d)", importance)
    return result


def review_action(
    agent_name: str,
    action_type: str,
    context: Dict[str, Any],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Review an action and auto-create/update skills based on discoveries.

    Args:
        agent_name: Name of the agent that performed the action
        action_type: Type of action (discovery, heal, navigation, economy, party, combat)
        context: Situation context (map, hp, zeny, level, etc.)
        result: Action result (discovered facts, success/failure, data)

    Returns:
        Dict with created/updated skill names, if any
    """
    outcome: Dict[str, Any] = {
        "reviewed": True,
        "action_type": action_type,
        "agent": agent_name,
        "created": [],
        "updated": [],
        "skipped_reason": None,
    }

    # Don't create skills for failed actions or trivial operations
    if result.get("error") or result.get("status") in ("error", "failed"):
        outcome["skipped_reason"] = "action_failed"
        return outcome

    # Determine domain from action type
    domain_map = {
        "discovery": "navigation",
        "heal": "healing",
        "navigation": "navigation",
        "economy": "economy",
        "party": "social",
        "combat": "combat",
        "grinding": "grinding",
    }
    domain = domain_map.get(action_type, "general")

    # Build skill from result data
    skill_name = f"{domain}-{agent_name}-{action_type}"
    skill_content = _build_skill_content(
        name=skill_name,
        action_type=action_type,
        context=context,
        result=result,
        domain=domain,
        agent_name=agent_name,
    )

    # Try to create the skill
    if skill_content:
        try:
            existing = skills_manager._find_skill(skill_name)
            if existing:
                # Update if new info is different
                outcome["skipped_reason"] = "already_exists"
                outcome["existing"] = skill_name
            else:
                create_result = skills_manager.create_skill(
                    name=skill_name,
                    content=skill_content,
                    category=domain,
                    provenance="background_review",
                )
                if create_result.get("success"):
                    outcome["created"].append(skill_name)
                    logger.info(
                        "Post-action review created skill: %s (agent=%s, action=%s)",
                        skill_name, agent_name, action_type,
                    )
                else:
                    outcome["skipped_reason"] = create_result.get("error")
        except Exception as exc:
            outcome["skipped_reason"] = str(exc)
            logger.debug("Post-action review failed: %s", exc)

    return outcome


def _build_skill_content(
    name: str,
    action_type: str,
    context: Dict[str, Any],
    result: Dict[str, Any],
    domain: str,
    agent_name: str,
) -> Optional[str]:
    """Build SKILL.md content from action context and result."""
    # Extract relevant facts from result
    discovered = result.get("discovered", result.get("data", {}))
    if not discovered:
        return None

    facts = []
    if isinstance(discovered, dict):
        for k, v in discovered.items():
            facts.append(f"- **{k}**: {v}")
    elif isinstance(discovered, list):
        for item in discovered:
            if isinstance(item, dict):
                for k, v in item.items():
                    facts.append(f"- **{k}**: {v}")
            else:
                facts.append(f"- {item}")
    else:
        facts.append(f"- {discovered}")

    if not facts:
        return None

    facts_text = "\n".join(facts)
    trigger = f"{action_type}_strategy" if action_type in ("heal",) else f"{domain}_{action_type}"

    content = f"""---
name: {name}
description: "{agent_name} discovered strategy for {domain}/{action_type}"
version: 1.0.0
triggers:
  - {trigger}
  - {action_type}
when_to_use:
  - action == {action_type}
  - domain == {domain}
metadata:
  domain: {domain}
  source: post_action_review
  agent: {agent_name}
  confidence: 0.6
---

# {domain.title()} Strategy: {action_type.title()}

## Discovered by {agent_name}

{facts_text}

## Context at discovery
- **Map**: {context.get("map", "unknown")}
- **HP**: {context.get("hp", "?")}/{context.get("hp_max", "?")}
- **Zeny**: {context.get("zeny", "?")}
- **Level**: {context.get("level", "?")}
"""
    return content


def review_heal_strategy(
    strategy: str,
    target_map: str,
    target_npc: str,
    confidence: float,
    bot_id: str,
) -> Dict[str, Any]:
    """Review a heal strategy discovery and create skill if new."""
    return review_action(
        agent_name="pro_ro_llm",
        action_type="heal",
        context={"map": target_map},
        result={
            "discovered": {
                "strategy": strategy,
                "target_map": target_map,
                "target_npc": target_npc,
                "confidence": confidence,
                "discovered_by": bot_id,
            },
            "status": "success",
        },
    )
