"""Skills Loader — match active skills to the current situation and load into context.

Skills are loaded from AI_sidecar/ai_sidecar/skills/*/SKILL.md based on
trigger matching. Progressive disclosure: metadata first, full content on demand.

Inspired by Hermes Agent's skills_tool.py progressive disclosure pattern.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai_sidecar import skills_usage
from ai_sidecar.skills_manager import view_skill, _parse_frontmatter, _read_skill_md, _find_skill

logger = logging.getLogger(__name__)

_SKILLS_DIR = Path(__file__).resolve().parent / "skills"


def load_for_context(situation: Dict[str, Any], max_skills: int = 5) -> List[str]:
    """Load full SKILL.md content for skills whose triggers match the situation.

    Args:
        situation: Dict with keys like hp_ratio, map, zeny, level, action_type, bot_id
        max_skills: Max skills to load (token budget control)

    Returns:
        List of SKILL.md content strings ready to inject into LLM context.
    """
    active = skills_usage.get_active_skills()
    scored: List[tuple[float, str, str]] = []

    for name in active:
        existing = _find_skill(name)
        if not existing:
            continue
        content = _read_skill_md(existing["path"])
        if not content:
            continue
        meta = _parse_frontmatter(content)

        # Score: higher = better match
        score = _score_match(meta, situation)

        usage = skills_usage.get_skill(name)
        confidence = usage.get("confidence", 0.5) if usage else 0.5

        # Combine match score + confidence
        combined = score * confidence
        if combined > 0:
            scored.append((combined, name, content))

    # Sort by combined score descending
    scored.sort(key=lambda x: -x[0])

    # Load top N
    selected = scored[:max_skills]
    for _, name, _ in selected:
        skills_usage.bump(name, event="use")

    return [content for _, name, content in selected]


def _score_match(meta: Dict[str, Any], situation: Dict[str, Any]) -> float:
    """Score how well a skill's triggers match the current situation.
    Returns 0.0 (no match) to 1.0 (perfect match)."""
    score = 0.0
    triggers = meta.get("triggers", [])
    if isinstance(triggers, str):
        triggers = [triggers]

    if not triggers:
        return 0.0

    hp = situation.get("hp_ratio", None)
    zeny = situation.get("zeny", None)
    action = situation.get("action_type", "")

    for trigger in triggers:
        t = trigger.lower()
        # HP-based triggers
        if "low_hp" in t and hp is not None and hp < 0.3:
            score += 0.4
        elif "heal" in t and hp is not None and hp < 0.5:
            score += 0.3
        elif "safe_hp" in t and hp is not None and hp > 0.5:
            score += 0.2

        # Zeny-based triggers
        if "low_zeny" in t and zeny is not None and zeny < 500:
            score += 0.3
        elif "economy" in t and zeny is not None:
            score += 0.2

        # Action-based triggers
        if action and t in action.lower():
            score += 0.3

        # Discovery triggers (always match when situation has discovery flags)
        if "discovery" in t and situation.get("_discovery_flag", False):
            score += 0.5

        # Generic triggers always give a small base score
        if "strategy" in t or "default" in t:
            score += 0.1

    return min(score, 1.0)


def get_matching_skills(situation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return metadata for matching skills without loading full content.
    Used for progressive disclosure (tier 1: metadata only)."""
    from ai_sidecar.skills_manager import list_skills

    all_skills = list_skills()
    scored: List[tuple[float, Dict[str, Any]]] = []

    for skill in all_skills:
        if skill["state"] != "active":
            continue
        existing = _find_skill(skill["name"])
        if not existing:
            continue
        content = _read_skill_md(existing["path"])
        if not content:
            continue
        meta = _parse_frontmatter(content)
        score = _score_match(meta, situation)
        if score > 0:
            scored.append((score, skill))

    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored]
