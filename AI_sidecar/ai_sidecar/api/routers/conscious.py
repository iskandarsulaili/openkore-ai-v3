"""Conscious decision engine API — LLM-driven team composition, build planning, and progression advice."""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ai_sidecar.api.deps import get_runtime
from ai_sidecar.lifecycle import RuntimeState
from ai_sidecar.providers.base import PlannerModelRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/conscious", tags=["conscious"])


class BotProfile(BaseModel):
    """Profile of a single bot for team synergy evaluation."""
    bot_id: str = Field(..., min_length=1, max_length=128)
    profile_name: str = Field(..., min_length=1, max_length=64)
    base_level: int = Field(ge=1, le=255)
    job_level: int = Field(ge=1, le=70)
    current_job: str = Field(default="Novice", max_length=64)
    desired_role: str = Field(default="", max_length=64)
    str_stat: int = Field(default=1, ge=1)
    agi_stat: int = Field(default=1, ge=1)
    vit_stat: int = Field(default=1, ge=1)
    int_stat: int = Field(default=1, ge=1)
    dex_stat: int = Field(default=1, ge=1)
    luk_stat: int = Field(default=1, ge=1)


class TeamSynergyRequest(BaseModel):
    """Request to evaluate optimal team composition for a group of bots."""
    bots: list[BotProfile] = Field(..., min_length=1, max_length=12)
    skip_llm: bool = Field(default=False, description="Skip LLM call, use knowledge rules only")


class JobAssignment(BaseModel):
    """Recommended job assignment for one bot."""
    profile_name: str = Field(..., max_length=64)
    bot_id: str = Field(..., max_length=128)
    recommended_job: str = Field(..., max_length=64)
    role: str = Field(..., max_length=64)  # tank, healer, dps, support, ranged, aoe
    reason: str = Field(..., max_length=256)
    build_focus: str = Field(default="", max_length=256)


class TeamSynergyResponse(BaseModel):
    """Team composition recommendation from LLM."""
    ok: bool = True
    message: str = "ok"
    assignments: list[JobAssignment] = Field(default_factory=list)
    team_synergy_note: str = Field(default="", max_length=1024)
    source: str = Field(default="llm")  # llm or knowledge



# ── Knowledge-driven fallback team compositions ──

# Synergy templates: (bot_count, [jobs]) -> roles
_SYNERGY_TEMPLATES: dict[str, list[str]] = {
    # 3-bot teams
    "3_magic": ["Mage", "Acolyte", "Swordsman"],
    "3_physical": ["Swordsman", "Hunter", "Acolyte"],
    "3_balanced": ["Acolyte", "Mage", "Hunter"],
    "3_ranged": ["Hunter", "Hunter", "Acolyte"],
    "3_melee": ["Swordsman", "Thief", "Acolyte"],
    # 2-bot teams
    "2_duo": ["Acolyte", "Mage"],
    "2_grind": ["Swordsman", "Acolyte"],
}

# Profile templates for each build (stat allocation)
_BUILD_PROFILES: dict[str, dict[str, str]] = {
    "Acolyte": {"build_focus": "INT > DEX. Max Heal, increase SP recovery. Support build with Blessing + Increase AGI."},
    "Mage": {"build_focus": "INT > DEX. Max Fire Bolt/Cold Bolt, then Safety Wall. INT 40 -> DEX 30 -> rest INT."},
    "Swordsman": {"build_focus": "STR > VIT > DEX. STR 40 -> VIT 30 -> DEX 20. Use Sword + Shield."},
    "Hunter": {"build_focus": "AGI > DEX. AGI 40 -> DEX 40. Use Bow + Arrows. Train Falcon for auto-attack."},
    "Thief": {"build_focus": "AGI > DEX. AGI 50 -> DEX 30. Dual daggers for Double Attack proc."},
    "Merchant": {"build_focus": "STR > VIT > DEX. STR 40 -> VIT 30. Pushcart for weight, Overcharge for profits."},
}


def _knowledge_team_synergy(bots: list[BotProfile]) -> TeamSynergyResponse:
    """Determine team composition using RO game knowledge (non-LLM fallback)."""
    n = len(bots)
    best_key = None
    best_overlap = -1

    for key, jobs in _SYNERGY_TEMPLATES.items():
        key_count = int(key.split("_")[0])
        if key_count != n:
            continue
        # Count overlap with current jobs (Novice = no preference)
        overlap = sum(
            1 for i, bot in enumerate(bots)
            if i < len(jobs) and (bot.current_job == "Novice" or bot.current_job == jobs[i])
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_key = key

    if best_key is None:
        # Fallback: assign based on position
        recommended_roles = ["Acolyte", "Mage", "Hunter", "Swordsman", "Thief", "Merchant"][:n]
    else:
        recommended_roles = _SYNERGY_TEMPLATES[best_key]

    role_labels = {
        "Acolyte": "healer",
        "Mage": "aoe_dps",
        "Swordsman": "tank",
        "Hunter": "ranged_dps",
        "Thief": "melee_dps",
        "Merchant": "economy",
    }

    assignments = []
    for i, bot in enumerate(bots):
        job = recommended_roles[i] if i < len(recommended_roles) else "Acolyte"
        profile = _BUILD_PROFILES.get(job, {})
        role = role_labels.get(job, "unknown")
        assignments.append(JobAssignment(
            profile_name=bot.profile_name,
            bot_id=bot.bot_id,
            recommended_job=job,
            role=role,
            reason=f"{job} fills {role} role for balanced team synergy",
            build_focus=profile.get("build_focus", ""),
        ))

    return TeamSynergyResponse(
        assignments=assignments,
        team_synergy_note=f"Knowledge-based {n}-bot team composition",
        source="knowledge",
    )


# ── LLM team synergy prompt ──

_TEAM_SYNERGY_SYSTEM_PROMPT = """You are an expert Ragnarok Online player (20+ years, 50+ max-level characters). You design optimal team compositions for bot automation.

RULES:
1. Each bot must get a DIFFERENT first class (no duplicates for first job change).
2. Consider stat distribution when recommending (e.g., high INT = Mage/Acolyte, high AGI = Thief/Hunter, high STR = Swordsman/Merchant).
3. Team synergy: healer + tank + dps works best for party farming.
4. Gear non-conflict: jobs should use different weapon types (Mace vs Staff vs Bow vs Sword vs Dagger vs Knuckle).
5. Available first job classes: Acolyte (heal/buff), Mage (AoE), Swordsman (tank/damage), Hunter (ranged), Thief (melee/crit), Merchant (economy/tank).
6. Always justify each recommendation with a short reason.

Return JSON matching this schema:
{
  "assignments": [
    {
      "profile_name": "<profile name>",
      "bot_id": "<bot id>",
      "recommended_job": "<job class>",
      "role": "<role>",
      "reason": "<short justification>",
      "build_focus": "<stat priority + skills to max>"
    }
  ],
  "team_synergy_note": "<one-sentence summary of team composition strength>"
}"""


def _build_team_synergy_user_prompt(bots: list[BotProfile]) -> str:
    """Build a user prompt describing the current team."""
    parts = [f"Evaluate team composition for {len(bots)} bots:"]
    for i, bot in enumerate(bots):
        parts.append(
            f"Bot {i+1}: {bot.profile_name} (id={bot.bot_id})\n"
            f"  Level {bot.base_level}/{bot.job_level} {bot.current_job}\n"
            f"  STR={bot.str_stat} AGI={bot.agi_stat} VIT={bot.vit_stat} "
            f"INT={bot.int_stat} DEX={bot.dex_stat} LUK={bot.luk_stat}\n"
            f"  Desired role: {bot.desired_role or 'automatic'}"
        )
    parts.append("\nRecommend optimal first job class and build focus for each bot.")
    return "\n".join(parts)


_TEAM_SYNERGY_SCHEMA = {
    "type": "object",
    "properties": {
        "assignments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "profile_name": {"type": "string"},
                    "bot_id": {"type": "string"},
                    "recommended_job": {
                        "type": "string",
                        "enum": ["Acolyte", "Mage", "Swordsman", "Hunter", "Thief", "Merchant"]
                    },
                    "role": {
                        "type": "string",
                        "enum": ["healer", "aoe_dps", "tank", "ranged_dps", "melee_dps", "economy"]
                    },
                    "reason": {"type": "string", "maxLength": 256},
                    "build_focus": {"type": "string", "maxLength": 256},
                },
                "required": ["profile_name", "bot_id", "recommended_job", "role", "reason"],
            },
        },
        "team_synergy_note": {"type": "string", "maxLength": 1024},
    },
    "required": ["assignments", "team_synergy_note"],
}


@router.post("/team-synergy", response_model=TeamSynergyResponse)
async def team_synergy(
    payload: TeamSynergyRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> TeamSynergyResponse:
    """Evaluate optimal team composition for a group of bots.

    Uses LLM (via model_router) by default, falls back to knowledge rules.
    The leader bot calls this to decide job change assignments for all bots.
    """
    if not payload.bots:
        raise HTTPException(status_code=400, detail="At least one bot required")

    if payload.skip_llm or runtime.model_router is None:
        logger.info("team_synergy: using knowledge fallback (%d bots)", len(payload.bots))
        return _knowledge_team_synergy(payload.bots)

    # Build LLM request
    system_prompt = _TEAM_SYNERGY_SYSTEM_PROMPT
    user_prompt = _build_team_synergy_user_prompt(payload.bots)

    req = PlannerModelRequest(
        bot_id=payload.bots[0].bot_id,
        trace_id=f"team_synergy_{payload.bots[0].profile_name}",
        task="team_synergy_evaluation",
        model="gpt-4o-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=_TEAM_SYNERGY_SCHEMA,
        timeout_seconds=30.0,
        max_retries=1,
    )

    try:
        resp, _decision = await runtime.model_router.generate_with_fallback(request=req)
    except Exception as e:
        logger.warning("team_synergy: LLM call failed (%s), falling back to knowledge", e)
        return _knowledge_team_synergy(payload.bots)

    if not resp.ok or resp.content is None:
        logger.warning("team_synergy: LLM returned error (%s), falling back", resp.error)
        return _knowledge_team_synergy(payload.bots)

    try:
        data = resp.content
        assignments_data = data.get("assignments", [])
        assignments = []
        for a in assignments_data:
            assignments.append(JobAssignment(
                profile_name=a.get("profile_name", ""),
                bot_id=a.get("bot_id", ""),
                recommended_job=a.get("recommended_job", "Acolyte"),
                role=a.get("role", "unknown"),
                reason=a.get("reason", "")[:256],
                build_focus=a.get("build_focus", "")[:256],
            ))
        return TeamSynergyResponse(
            assignments=assignments,
            team_synergy_note=data.get("team_synergy_note", "")[:1024],
            source="llm",
        )
    except Exception as e:
        logger.warning("team_synergy: failed to parse LLM response (%s), falling back", e)
        return _knowledge_team_synergy(payload.bots)


@router.get("/health", response_model=dict)
async def conscious_health(
    runtime: RuntimeState = Depends(get_runtime),
) -> dict:
    """Health check for the conscious engine API."""
    llm_available = runtime.model_router is not None
    return {
        "ok": True,
        "llm_available": llm_available,
        "template_count": len(_SYNERGY_TEMPLATES),
    }
