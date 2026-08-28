"""Conscious decision engine API — LLM-driven team composition, build planning, and progression advice.

Uses the new LLMManager multi-provider system as the primary LLM layer,
falls back to the legacy model_router, and ultimately to knowledge rules.
"""
from __future__ import annotations

import json
import logging
from typing import Any

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
    use_llm_manager: bool = Field(default=False, description="Use new LLMManager instead of legacy model_router")


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
    source: str = Field(default="llm")  # llm, llm_manager, or knowledge


# ── LLM Manager prompts ──

_TEAM_SYNERGY_LLM_SYSTEM_PROMPT = """You are an expert Ragnarok Online player (20+ years, 50+ max-level characters). You design optimal team compositions for bot automation.

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


# ── Knowledge-driven fallback team compositions ──

_SYNERGY_TEMPLATES: dict[str, list[str]] = {
    "3_magic": ["Mage", "Acolyte", "Swordsman"],
    "3_physical": ["Swordsman", "Hunter", "Acolyte"],
    "3_balanced": ["Acolyte", "Mage", "Hunter"],
    "3_ranged": ["Hunter", "Hunter", "Acolyte"],
    "3_melee": ["Swordsman", "Thief", "Acolyte"],
    "2_duo": ["Acolyte", "Mage"],
    "2_grind": ["Swordsman", "Acolyte"],
}

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
        overlap = sum(
            1 for i, bot in enumerate(bots)
            if i < len(jobs) and (bot.current_job == "Novice" or bot.current_job == jobs[i])
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_key = key

    if best_key is None:
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


# ── LLM-via-LLMManager handler ──

async def _llm_manager_team_synergy(
    bots: list[BotProfile],
    llm_manager: Any,
) -> TeamSynergyResponse | None:
    """Attempt team synergy via the new LLMManager. Returns None on failure."""
    try:
        user_prompt = _build_team_synergy_user_prompt(bots)
        data = await llm_manager.complete_json(
            prompt=user_prompt,
            system_prompt=_TEAM_SYNERGY_LLM_SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=4096,
        )
        if not data or not isinstance(data, dict):
            return None

        assignments_data = data.get("assignments", [])
        if not assignments_data:
            return None

        assignments = []
        for a in assignments_data:
            assignments.append(JobAssignment(
                profile_name=str(a.get("profile_name", "")),
                bot_id=str(a.get("bot_id", "")),
                recommended_job=str(a.get("recommended_job", "Acolyte")),
                role=str(a.get("role", "unknown")),
                reason=str(a.get("reason", ""))[:256],
                build_focus=str(a.get("build_focus", ""))[:256],
            ))

        return TeamSynergyResponse(
            assignments=assignments,
            team_synergy_note=str(data.get("team_synergy_note", ""))[:1024],
            source="llm_manager",
        )
    except Exception as e:
        logger.warning("LLMManager team_synergy failed: %s", e)
        return None


# ── Legacy model_router prompt ──

_TEAM_SYNERGY_SYSTEM_PROMPT = _TEAM_SYNERGY_LLM_SYSTEM_PROMPT  # same prompt, reused

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


# ── Endpoints ──


@router.post("/team-synergy", response_model=TeamSynergyResponse)
async def team_synergy(
    payload: TeamSynergyRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> TeamSynergyResponse:
    """Evaluate optimal team composition for a group of bots.

    LLM capability priority:
      1. LLMManager (new — multi-provider, fallback chain)  [if use_llm_manager=True]
      2. Legacy model_router                                [default]
      3. Knowledge rules                                    [fallback]
    """
    if not payload.bots:
        raise HTTPException(status_code=400, detail="At least one bot required")

    if payload.skip_llm:
        logger.info("team_synergy: LLM skipped, using knowledge fallback (%d bots)", len(payload.bots))
        return _knowledge_team_synergy(payload.bots)

    # ── Path 1: LLMManager (new multi-provider system) ──
    if payload.use_llm_manager and runtime.llm_manager is not None:
        if runtime.llm_manager.is_available():
            result = await _llm_manager_team_synergy(payload.bots, runtime.llm_manager)
            if result is not None:
                logger.info("team_synergy: LLMManager succeeded (%d bots)", len(payload.bots))
                return result
            logger.warning("team_synergy: LLMManager failed, falling back")
        else:
            logger.warning("team_synergy: LLMManager not available, falling back")

    # ── Path 2: Legacy model_router ──
    if runtime.model_router is not None:
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
            logger.warning("team_synergy: model_router failed (%s), falling back to knowledge", e)
            return _knowledge_team_synergy(payload.bots)

        if not resp.ok or resp.content is None:
            logger.warning("team_synergy: model_router returned error (%s), falling back", resp.error)
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
            logger.warning("team_synergy: failed to parse model_router response (%s), falling back", e)
            return _knowledge_team_synergy(payload.bots)

    # ── Path 3: Knowledge fallback ──
    logger.info("team_synergy: no LLM available, using knowledge rules (%d bots)", len(payload.bots))
    return _knowledge_team_synergy(payload.bots)


# ── LLM-powered NPC dialogue decision ──

_NPC_DIALOGUE_SYSTEM_PROMPT = """You are an AI playing Ragnarok Online. Given an NPC dialogue situation,
choose the optimal response option. Consider quest progression, rewards, and class-specific benefits.

Return JSON:
{
  "choice_index": <0-based index of chosen option>,
  "reason": "<why this choice>"
}"""


@router.post("/npc-decide", response_model=dict)
async def npc_dialogue_decision(
    payload: dict,
    runtime: RuntimeState = Depends(get_runtime),
) -> dict:
    """Use LLM to decide NPC dialogue choices."""
    npc_name = payload.get("npc_name", "unknown")
    dialogue_text = payload.get("dialogue_text", "")
    options = payload.get("options", [])

    if not options:
        return {"ok": True, "choice_index": 0, "source": "fallback"}

    # Try LLMManager first
    if runtime.llm_manager is not None and runtime.llm_manager.is_available():
        prompt = (
            f"NPC: {npc_name}\n"
            f"Dialogue: {dialogue_text}\n\n"
            f"Options:\n" + "\n".join(f"[{i}] {opt}" for i, opt in enumerate(options)) + "\n\n"
            "Choose the best option index."
        )
        try:
            data = await runtime.llm_manager.complete_json(
                prompt=prompt,
                system_prompt=_NPC_DIALOGUE_SYSTEM_PROMPT,
                temperature=0.2,
                max_tokens=512,
            )
            choice = int(data.get("choice_index", 0))
            if 0 <= choice < len(options):
                return {
                    "ok": True,
                    "choice_index": choice,
                    "reason": data.get("reason", ""),
                    "source": "llm_manager",
                }
        except Exception as e:
            logger.warning("LLMManager NPC decide failed: %s", e)

    # Fallback: first non-repeat option
    return {"ok": True, "choice_index": 0, "source": "fallback"}


# ── Strategic planning ──

_STRATEGIC_PLAN_SYSTEM_PROMPT = """You are a strategic planner for Ragnarok Online bot automation.
Given the current game state, plan the next actions for the bot.

Return JSON:
{
  "plan": [
    {
      "action": "<action name>",
      "target": "<target or location>",
      "priority": <1-10>,
      "reason": "<why this action>"
    }
  ],
  "overall_strategy": "<one-sentence strategy summary>"
}"""


@router.post("/strategize", response_model=dict)
async def strategic_planning(
    payload: dict,
    runtime: RuntimeState = Depends(get_runtime),
) -> dict:
    """Use LLM for high-level strategic planning decisions."""
    context = payload.get("context", {})

    if runtime.llm_manager is not None and runtime.llm_manager.is_available():
        prompt = (
            f"Current state:\n"
            f"  Level: {context.get('base_level', '?')}/{context.get('job_level', '?')}\n"
            f"  Job: {context.get('current_job', 'Novice')}\n"
            f"  Map: {context.get('map', 'unknown')}\n"
            f"  HP: {context.get('hp', 0)}/{context.get('max_hp', 0)}\n"
            f"  SP: {context.get('sp', 0)}/{context.get('max_sp', 0)}\n"
            f"  Zenny: {context.get('zenny', 0)}\n"
            f"  Weight: {context.get('weight', 0)}/{context.get('max_weight', 0)}\n"
            f"  Party size: {context.get('party_size', 1)}\n"
            f"  Current objective: {context.get('current_objective', 'grind')}\n"
            f"\n"
            f"Recent events:\n" + "\n".join(
                f"  - {ev}" for ev in (context.get("recent_events", []) or [])
            ) + "\n\n"
            "What should the bot do next?"
        )
        try:
            data = await runtime.llm_manager.complete_json(
                prompt=prompt,
                system_prompt=_STRATEGIC_PLAN_SYSTEM_PROMPT,
                temperature=0.4,
                max_tokens=2048,
            )
            return {
                "ok": True,
                "plan": data.get("plan", []),
                "overall_strategy": data.get("overall_strategy", ""),
                "source": "llm_manager",
            }
        except Exception as e:
            logger.warning("LLMManager strategize failed: %s", e)

    return {"ok": True, "plan": [], "overall_strategy": "continue current objective", "source": "fallback"}


# ── Quest choice ──

_QUEST_DECISION_SYSTEM_PROMPT = """You are a quest advisor for Ragnarok Online.
Given the available quests and the bot's current state, recommend which quests to prioritize.

Return JSON:
{
  "recommended_quests": [
    {
      "quest_name": "<quest name>",
      "priority": <1-10>,
      "reason": "<why this quest>"
    }
  ],
  "explanation": "<brief reasoning>"
}"""


@router.post("/quest-decide", response_model=dict)
async def quest_decision(
    payload: dict,
    runtime: RuntimeState = Depends(get_runtime),
) -> dict:
    """Use LLM to decide which quests to pursue."""
    available_quests = payload.get("available_quests", [])
    bot_state = payload.get("bot_state", {})

    if runtime.llm_manager is not None and runtime.llm_manager.is_available() and available_quests:
        quest_list = "\n".join(
            f"  - {q.get('name', '?')} (level {q.get('level', '?')}, reward: {q.get('reward', '?')})"
            for q in available_quests[:10]
        )
        prompt = (
            f"Bot: Level {bot_state.get('base_level', '?')} {bot_state.get('current_job', 'Novice')}\n"
            f"Available quests:\n{quest_list}\n\n"
            "Which quests should this bot prioritize?"
        )
        try:
            data = await runtime.llm_manager.complete_json(
                prompt=prompt,
                system_prompt=_QUEST_DECISION_SYSTEM_PROMPT,
                temperature=0.3,
                max_tokens=2048,
            )
            return {
                "ok": True,
                "recommended_quests": data.get("recommended_quests", []),
                "explanation": data.get("explanation", ""),
                "source": "llm_manager",
            }
        except Exception as e:
            logger.warning("LLMManager quest decide failed: %s", e)

    return {"ok": True, "recommended_quests": [], "explanation": "No quest recommendations", "source": "fallback"}


@router.get("/health", response_model=dict)
async def conscious_health(
    runtime: RuntimeState = Depends(get_runtime),
) -> dict:
    """Health check for the conscious engine API."""
    llm_available = runtime.model_router is not None
    llm_manager_available = (
        runtime.llm_manager is not None and runtime.llm_manager.is_available()
    )
    return {
        "ok": True,
        "llm_available": llm_available,
        "llm_manager_available": llm_manager_available,
        "llm_manager_providers": (
            runtime.llm_manager.available_providers if runtime.llm_manager else []
        ),
        "template_count": len(_SYNERGY_TEMPLATES),
    }


@router.get("/brain-rewards", response_model=dict)
async def brain_rewards(
    bot_id: str = "",
    runtime: RuntimeState = Depends(get_runtime),
) -> dict:
    """Reward/punish ledger for ALL brains (conscious, heuristic, reflex,
    subconscious, goal, memory, strategy) — self-* feedback observability.

    User directive (2026-08-28): punish/reward system for all the brains.
    """
    try:
        from ai_sidecar.learning.brain_reward_ledger import get_brain_reward_ledger
        _ledger = get_brain_reward_ledger()
        _ledger.load()  # replay persisted JSONL so scores survive restarts
        _sd = _ledger._score_dict
        if bot_id:
            scores = _ledger.scores(bot_id)
            return {"ok": True, "bot_id": bot_id,
                    "scores": [_sd(s) for s in scores]}
        # No bot_id → aggregate across all bots seen.
        bots = sorted({s.bot_id for s in _ledger._scores.values() for s in s.values()})
        out = {}
        for bid in bots:
            out[bid] = [_sd(s) for s in _ledger.scores(bid)]
        return {"ok": True, "bots": out}
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "error": str(exc)}


@router.get("/self-heal-status", response_model=dict)
async def self_heal_status(
    runtime: RuntimeState = Depends(get_runtime),
) -> dict:
    """Self-heal observability: every correctable-failure surface's state.

    Aggregates the edge-case handler outcomes, crisis summary, comeback
    fix-registry, degradation module health, and healer log — the surfaces
    the user directive (2026-08-28) requires self-healing to own BEFORE
    self-learning/self-improving.
    """
    out: dict = {
        "edge_case": {},
        "crisis": {},
        "comeback": {},
        "degradation": {},
        "self_healer": {},
        "time_scheduler": {},
    }
    try:
        _edge = getattr(runtime, "edge_case_handler", None)
        if _edge is not None and hasattr(_edge, "_outcomes"):
            _oc = _edge._outcomes
            if hasattr(_oc, "summary"):
                out["edge_case"] = _oc.summary()
            elif hasattr(_oc, "records"):
                out["edge_case"] = {"count": len(_oc.records)}
            else:
                out["edge_case"] = {"note": "outcome history active"}
    except Exception:
        pass
    try:
        _cm = getattr(runtime, "crisis_manager", None)
        if _cm is not None and hasattr(_cm, "get_crisis_summary"):
            out["crisis"] = _cm.get_crisis_summary()
    except Exception:
        pass
    try:
        _cb = getattr(runtime, "comeback_engine", None)
        if _cb is not None:
            out["comeback"]["fix_summary"] = _cb.get_fix_summary() if hasattr(_cb, "get_fix_summary") else {}
            out["comeback"]["pending"] = _cb.get_pending_count() if hasattr(_cb, "get_pending_count") else 0
            out["comeback"]["recovery_rate"] = _cb.get_recovery_rate("death") if hasattr(_cb, "get_recovery_rate") else 0.0
    except Exception:
        pass
    try:
        _dm = getattr(runtime, "degradation_manager", None)
        if _dm is not None and hasattr(_dm, "get_health_summary"):
            out["degradation"] = _dm.get_health_summary()
    except Exception:
        pass
    try:
        _sh = getattr(runtime, "self_healer", None)
        if _sh is not None and hasattr(_sh, "get_heal_summary"):
            out["self_healer"] = _sh.get_heal_summary()
    except Exception:
        pass
    try:
        _ts = getattr(runtime, "time_scheduler", None)
        if _ts is not None and hasattr(_ts, "get_scheduler_summary"):
            out["time_scheduler"] = _ts.get_scheduler_summary()
    except Exception:
        pass
    return out
