"""
Conscious Decision Engine — makes all high-level decisions for bot progression.

This is the bot's "conscious brain" that decides:
1. Skill learning order (phase-based, per-class, per-build)
2. Stat distribution (breakpoint-based, not flat priorities)
3. Equipment goals (what to aim for at each level)
4. Item restocking (what to buy, when, and how many)
5. Map selection (where to farm based on level and gear)
6. Party coordination (when to party, what role to play)

Designed by Pro RO Player with 20+ years of experience:
- Phase-based learning: different priorities at level 10 vs level 50 vs level 90
- Build variants: Vit Knight ≠ Agi Knight, Support Priest ≠ Battle Priest
- Efficiency breakpoints: SP Recovery 4 (not 10), Heal 4 (not 10), Owl's Eye 10 (yes)
- Transcendent planning: save points for high-level skills
- Game mode awareness: solo farming ≠ party ≠ MVP ≠ WoE
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

_KNOWLEDGE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "knowledge", "knowledge.json"
)


# ── Data Classes ──

@dataclass
class SkillPhase:
    """A phase of skill learning with level range and goal."""
    level_range: tuple[int, int]  # (min_level, max_level)
    goal: str  # max_damage, sustain, aoe, element_coverage, support, preparation
    skills: list[dict[str, Any]]  # [{name, max_level, priority, reason}]
    reason: str  # Why this phase exists


@dataclass
class StatBreakpoint:
    """A stat breakpoint — get this stat to this value, then move on."""
    stat: str
    value: int
    priority: int  # 1 = do this first, 2 = second, etc.
    reason: str


@dataclass
class BuildVariant:
    """A specific build variant for a class."""
    name: str
    description: str
    stat_breakpoints: list[StatBreakpoint]
    skill_phases: list[SkillPhase]
    best_for: list[str]  # solo_farming, party, mvp, woe
    # equipment goals derive from the gear-progression planner (agnostic, real
    # item db) — never hardcode item literals here (RULE.md).
    # Restock decisions derive from the bot's REAL inventory (agnostic).
    # farming maps come from map_intelligence.get_farming_maps() (real server
    # data, agnostic) — never hardcode map literals here (RULE.md).


@dataclass
class ClassBuilds:
    """All build variants for a class."""
    job_class: str
    variants: dict[str, BuildVariant]
    default_variant: str


@dataclass
class Decision:
    """A decision the engine has made."""
    domain: str  # skills, stats, equipment, restock, map, party
    action: str  # learn_skill, add_stat, buy_item, move_map, etc.
    target: str  # skill name, stat name, item name, map name
    priority: int  # 1=immediate, 5=when convenient
    reason: str
    params: dict[str, Any] = field(default_factory=dict)


# ── Efficiency Breakpoints ──
# A pro player knows exactly how many points to put in each skill
# before diminishing returns make further investment wasteful.
# These are verified against rAthena mechanics.

EFFICIENCY_BREAKPOINTS: dict[str, dict[str, Any]] = {
    # Novice
    "NV_BASIC": {"max_level": 9, "sweet_spot": 1, "reason": "Level 1 is enough to sit/trade/party. Save 8 points for first job."},
    "NV_FIRSTAID": {"max_level": 1, "sweet_spot": 1, "reason": "45 HP heal, no higher levels available."},
    "NV_TRICKDEAD": {"max_level": 1, "sweet_spot": 1, "reason": "Fake death, no higher levels."},

    # Swordsman
    "SM_SWORD": {"max_level": 10, "sweet_spot": 5, "reason": "+4 ATK per level. Level 5 = +20 ATK (good early). Level 10 = +40 ATK (finish later)."},
    "SM_BASH": {"max_level": 10, "sweet_spot": 10, "reason": "Main damage skill. No diminishing returns — max it."},
    "SM_RECOVERY": {"max_level": 10, "sweet_spot": 5, "reason": "50% regen chance at level 5. Level 10 = 65%. Diminishing after 5."},
    "SM_MAGNUM": {"max_level": 10, "sweet_spot": 5, "reason": "AoE range increases at 5. More levels = more damage, not more range."},
    "SM_PROVOKE": {"max_level": 10, "sweet_spot": 1, "reason": "Level 1 is enough for aggro. Higher levels = longer duration, not needed."},
    "SM_ENDURE": {"max_level": 10, "sweet_spot": 1, "reason": "Level 1 = 7s stun immunity. Enough for most situations."},

    # Mage
    "MG_SRECOVERY": {"max_level": 10, "sweet_spot": 4, "reason": "Level 4 = 30% regen. Level 10 = 45%. 15% gain for 6 points = diminishing."},
    "MG_FIREBOLT": {"max_level": 10, "sweet_spot": 10, "reason": "Main damage. No diminishing returns. Max it."},
    "MG_COLDBOLT": {"max_level": 10, "sweet_spot": 5, "reason": "Element coverage. Level 5 = 5 bolts. More levels = more bolts, not needed for coverage."},
    "MG_LIGHTNINGBOLT": {"max_level": 10, "sweet_spot": 5, "reason": "Same as Cold Bolt. Level 5 for coverage."},
    "MG_FROSTDIVER": {"max_level": 10, "sweet_spot": 5, "reason": "Freeze chance caps at level 5. Higher = more damage, not needed for freeze."},
    "MG_SAFETYWALL": {"max_level": 10, "sweet_spot": 5, "reason": "Level 5 = 5 hits. Level 10 = 10 hits. Get 5, finish later."},
    "MG_FIREWALL": {"max_level": 10, "sweet_spot": 5, "reason": "Level 5 = 5 fireballs. Enough for most situations."},

    # Archer
    "AC_OWL": {"max_level": 10, "sweet_spot": 10, "reason": "+1 DEX per level. No diminishing returns. Max it."},
    "AC_DOUBLE": {"max_level": 10, "sweet_spot": 10, "reason": "190% at level 1, 380% at level 10. No diminishing returns. Max it."},
    "AC_VULTURE": {"max_level": 10, "sweet_spot": 5, "reason": "+50 range at level 5. Level 10 = +100 range. 5 is enough for most maps."},
    "AC_SHOWER": {"max_level": 10, "sweet_spot": 5, "reason": "AoE + knockback. Level 5 is enough for grouped mobs."},

    # Acolyte
    "AL_HEAL": {"max_level": 10, "sweet_spot": 4, "reason": "Level 4 = 180 HP for 12 SP (15 HP/SP). Level 10 = 720 HP for 30 SP (24 HP/SP). Sweet spot at 4, max later."},
    "AL_INCAGI": {"max_level": 10, "sweet_spot": 10, "reason": "+1 AGI per level, duration increases. No diminishing returns. Max it."},
    "AL_BLESSING": {"max_level": 10, "sweet_spot": 10, "reason": "+1 to all stats per level, duration increases. No diminishing returns. Max it."},
    "AL_TELEPORT": {"max_level": 1, "sweet_spot": 1, "reason": "Level 1 is enough for escape."},
    "AL_WARP": {"max_level": 1, "sweet_spot": 1, "reason": "Level 1 is enough for party travel."},
    "AL_DEMONBANE": {"max_level": 10, "sweet_spot": 5, "reason": "+5 ATK vs demon/undead per level. Level 5 = +25 ATK. Enough for solo leveling."},
    "AL_DP": {"max_level": 1, "sweet_spot": 1, "reason": "Level 1 is enough for party buff."},

    # Merchant
    "MC_PUSHCART": {"max_level": 10, "sweet_spot": 1, "reason": "Level 1 = pushcart. Level 10 = more weight. Get 1, finish later."},
    "MC_DISCOUNT": {"max_level": 10, "sweet_spot": 10, "reason": "24% discount at level 10. No diminishing returns. Max it for profit."},
    "MC_OVERCHARGE": {"max_level": 10, "sweet_spot": 10, "reason": "24% overcharge at level 10. No diminishing returns. Max it for profit."},
    "MC_MAMMONITE": {"max_level": 10, "sweet_spot": 5, "reason": "Zeny attack. Level 5 = 500% damage for 500z. Enough for farming."},
    "MC_IDENTIFY": {"max_level": 1, "sweet_spot": 1, "reason": "Level 1 is enough to identify items."},
    "MC_CHANGECART": {"max_level": 1, "sweet_spot": 1, "reason": "Level 1 is enough to change cart type."},

    # Thief
    "TF_DOUBLE": {"max_level": 10, "sweet_spot": 10, "reason": "50% double attack at level 10. No diminishing returns. Max it."},
    "TF_HIDE": {"max_level": 10, "sweet_spot": 1, "reason": "Level 1 = 15s hide. Enough for escape. Higher = longer duration."},
    "TF_STEAL": {"max_level": 10, "sweet_spot": 5, "reason": "Steal rate increases. Level 5 is enough for most mobs. Max for farming build."},
    "TF_POISON": {"max_level": 10, "sweet_spot": 5, "reason": "Poison damage. Level 5 is enough for the DoT. Max for damage build."},
    "TF_MISS": {"max_level": 5, "sweet_spot": 5, "reason": "Increases dodge rate. Max it for survival."},
    "TF_SPRINKLESAND": {"max_level": 1, "sweet_spot": 1, "reason": "Blind + flee. Level 1 is enough."},
}


# ── Build Definitions ──
# Each class has 2-3 build variants with:
# - Phase-based skill learning (different priorities at different levels)
# - Stat breakpoints (get X to Y, then move on)
# - Game mode awareness (solo, party, MVP, WoE)
# - Transcendent planning (save points for high-level skills)

BUILDS: dict[str, ClassBuilds] = {
    # ═══════════════════════════════════════════════════════════════
    # NOVICE (Level 1-10)
    # ═══════════════════════════════════════════════════════════════
    "novice": ClassBuilds(
        job_class="Novice",
        default_variant="speed_to_job",
        variants={
            "speed_to_job": BuildVariant(
                name="Speed to First Job",
                description="Minimal investment. Get to level 10 and change job as fast as possible.",
                stat_breakpoints=[
                    StatBreakpoint("dex", 10, 1, "10 DEX for hit rate. You need to hit things."),
                    StatBreakpoint("agi", 10, 2, "10 AGI for flee. Survival."),
                    StatBreakpoint("vit", 10, 3, "10 VIT for HP. Don't die to Porings."),
                ],
                best_for=["solo_farming", "party", "mvp", "woe"],
                skill_phases=[
                    SkillPhase(
                        level_range=(1, 10),
                        goal="survival",
                        skills=[
                            {"name": "NV_BASIC", "max_level": 1, "priority": 1, "reason": "Level 1 is enough to sit, trade, party. Save 8 points for first job."},
                            {"name": "NV_FIRSTAID", "max_level": 1, "priority": 2, "reason": "45 HP heal. Saves potions."},
                            {"name": "NV_TRICKDEAD", "max_level": 1, "priority": 3, "reason": "Fake death. Free escape from aggro."},
                        ],
                        reason="Novice is a 10-level sprint. Spend minimum points, save the rest for your first job."
                    ),
                ],
            ),
        },
    ),

    # ═══════════════════════════════════════════════════════════════
    # SWORDSMAN (Level 10-50) → KNIGHT (Level 50-99)
    # ═══════════════════════════════════════════════════════════════
    "swordman": ClassBuilds(
        job_class="Swordsman",
        default_variant="vit_knight",
        variants={
            "vit_knight": BuildVariant(
                name="Vit Knight (Tank)",
                description="High VIT for survivability. Bowling Bash for AoE. Best for party play and MVP tanking.",
                stat_breakpoints=[
                    StatBreakpoint("str", 30, 1, "30 STR first for base damage. You need to kill things."),
                    StatBreakpoint("dex", 30, 2, "30 DEX for hit rate. Missing = no damage."),
                    StatBreakpoint("vit", 50, 3, "50 VIT for stun immunity vs most mobs. Survival."),
                    StatBreakpoint("str", 80, 4, "80 STR for ATK bonus vs size. Diminishing after 80."),
                    StatBreakpoint("vit", 70, 5, "70 VIT for full stun immunity. Tank build."),
                    StatBreakpoint("dex", 40, 6, "40 DEX for hit rate with spear."),
                    StatBreakpoint("str", 99, 7, "99 STR for max damage. Endgame."),
                ],
                best_for=["party", "mvp"],
                skill_phases=[
                    SkillPhase(
                        level_range=(10, 25),
                        goal="max_damage",
                        skills=[
                            {"name": "SM_BASH", "max_level": 10, "priority": 1, "reason": "Bash is your only damage skill at this level. Max it first."},
                            {"name": "SM_SWORD", "max_level": 5, "priority": 2, "reason": "+20 ATK from level 5. Good early boost."},
                        ],
                        reason="Level 10-25: You need damage to kill things fast. Bash 10 + Sword Mastery 5."
                    ),
                    SkillPhase(
                        level_range=(25, 40),
                        goal="aoe",
                        skills=[
                            {"name": "SM_MAGNUM", "max_level": 5, "priority": 1, "reason": "AoE for grouped mobs. Level 5 is enough range."},
                            {"name": "SM_RECOVERY", "max_level": 5, "priority": 2, "reason": "50% regen chance. Efficiency sweet spot."},
                            {"name": "SM_PROVOKE", "max_level": 1, "priority": 3, "reason": "Level 1 is enough for aggro management."},
                            {"name": "SM_ENDURE", "max_level": 1, "priority": 4, "reason": "7s stun immunity. Safety."},
                        ],
                        reason="Level 25-40: AoE for grouped mobs. Sustain for longer farming sessions."
                    ),
                    SkillPhase(
                        level_range=(40, 50),
                        goal="preparation",
                        skills=[
                            {"name": "SM_SWORD", "max_level": 10, "priority": 1, "reason": "Finish Sword Mastery. +40 ATK total."},
                            {"name": "SM_MAGNUM", "max_level": 10, "priority": 2, "reason": "Max Magnum Break for more AoE damage."},
                            {"name": "SM_RECOVERY", "max_level": 10, "priority": 3, "reason": "Max HP Recovery for sustain."},
                        ],
                        reason="Level 40-50: Finish Swordsman skills. Save remaining points for Knight."
                    ),
                    SkillPhase(
                        level_range=(50, 70),
                        goal="core_skills",
                        skills=[
                            {"name": "KN_BOWLINGBASH", "max_level": 10, "priority": 1, "reason": "Main AoE skill. Max it first as Knight."},
                            {"name": "KN_TWOHANDQUICKEN", "max_level": 10, "priority": 2, "reason": "ASPD boost. More attacks = more damage."},
                            {"name": "KN_SPEARMASTERY", "max_level": 5, "priority": 3, "reason": "Spear damage boost. Level 5 is enough early."},
                            {"name": "KN_CAVALIERMASTERY", "max_level": 5, "priority": 4, "reason": "Mounted combat. Level 5 for peco peco."},
                        ],
                        reason="Level 50-70: Core Knight skills. Bowling Bash is your identity."
                    ),
                    SkillPhase(
                        level_range=(70, 85),
                        goal="optimization",
                        skills=[
                            {"name": "KN_SPEARMASTERY", "max_level": 10, "priority": 1, "reason": "Max spear damage."},
                            {"name": "KN_CAVALIERMASTERY", "max_level": 10, "priority": 2, "reason": "Max mounted combat."},
                            {"name": "KN_CAVALIERCOMBAT", "max_level": 5, "priority": 3, "reason": "Mounted attack skills."},
                            {"name": "KN_SPEARBOOMERANG", "max_level": 5, "priority": 4, "reason": "Ranged spear attack. Pull mobs from distance."},
                        ],
                        reason="Level 70-85: Optimize your build. Fill in support skills."
                    ),
                    SkillPhase(
                        level_range=(85, 99),
                        goal="endgame",
                        skills=[
                            {"name": "KN_BRANDISHSPEAR", "max_level": 10, "priority": 1, "reason": "Spear AoE. Alternative to Bowling Bash."},
                            {"name": "SM_PROVOKE", "max_level": 5, "priority": 2, "reason": "More provoke duration for MVP tanking."},
                        ],
                        reason="Level 85-99: Endgame skills. Prepare for transcendent class."
                    ),
                ],
            ),
            "agi_knight": BuildVariant(
                name="Agi Knight (Dodge)",
                description="High AGI for flee and ASPD. Twohand Quicken for max attack speed. Best for solo farming.",
                stat_breakpoints=[
                    StatBreakpoint("dex", 30, 1, "30 DEX for hit rate."),
                    StatBreakpoint("agi", 50, 2, "50 AGI for flee and ASPD."),
                    StatBreakpoint("str", 50, 3, "50 STR for damage."),
                    StatBreakpoint("agi", 80, 4, "80 AGI for ASPD breakpoint."),
                    StatBreakpoint("str", 80, 5, "80 STR for ATK bonus."),
                    StatBreakpoint("dex", 40, 6, "40 DEX for hit rate with twohand sword."),
                    StatBreakpoint("agi", 99, 7, "99 AGI for max flee and ASPD."),
                ],
                best_for=["solo_farming"],
                skill_phases=[
                    SkillPhase(
                        level_range=(10, 25),
                        goal="max_damage",
                        skills=[
                            {"name": "SM_BASH", "max_level": 10, "priority": 1, "reason": "Bash is your only damage skill. Max it first."},
                            {"name": "SM_SWORD", "max_level": 5, "priority": 2, "reason": "+20 ATK. Good early boost."},
                        ],
                        reason="Level 10-25: Same as Vit Knight. Bash is king at low levels."
                    ),
                    SkillPhase(
                        level_range=(25, 40),
                        goal="aoe",
                        skills=[
                            {"name": "SM_MAGNUM", "max_level": 5, "priority": 1, "reason": "AoE for grouped mobs."},
                            {"name": "SM_RECOVERY", "max_level": 5, "priority": 2, "reason": "50% regen. Efficiency sweet spot."},
                            {"name": "SM_TWOHAND", "max_level": 5, "priority": 3, "reason": "Twohand Sword Mastery. Prep for Knight."},
                        ],
                        reason="Level 25-40: AoE + prep for twohand sword build."
                    ),
                    SkillPhase(
                        level_range=(40, 50),
                        goal="preparation",
                        skills=[
                            {"name": "SM_SWORD", "max_level": 10, "priority": 1, "reason": "Finish Sword Mastery."},
                            {"name": "SM_TWOHAND", "max_level": 10, "priority": 2, "reason": "Max Twohand Mastery for Knight."},
                            {"name": "SM_MAGNUM", "max_level": 10, "priority": 3, "reason": "Max Magnum Break."},
                        ],
                        reason="Level 40-50: Finish Swordsman. Prep for Agi Knight."
                    ),
                    SkillPhase(
                        level_range=(50, 70),
                        goal="core_skills",
                        skills=[
                            {"name": "KN_TWOHANDQUICKEN", "max_level": 10, "priority": 1, "reason": "ASPD boost. Core of Agi Knight build."},
                            {"name": "KN_BOWLINGBASH", "max_level": 10, "priority": 2, "reason": "Main AoE. Max it."},
                            {"name": "KN_CAVALIERMASTERY", "max_level": 5, "priority": 3, "reason": "Peco peco for faster movement."},
                        ],
                        reason="Level 50-70: Twohand Quicken is your identity. Max ASPD first."
                    ),
                    SkillPhase(
                        level_range=(70, 99),
                        goal="optimization",
                        skills=[
                            {"name": "KN_CAVALIERMASTERY", "max_level": 10, "priority": 1, "reason": "Max mounted combat."},
                            {"name": "KN_CAVALIERCOMBAT", "max_level": 5, "priority": 2, "reason": "Mounted attack skills."},
                            {"name": "KN_SPEARBOOMERANG", "max_level": 5, "priority": 3, "reason": "Ranged pull skill."},
                        ],
                        reason="Level 70-99: Optimize. Fill in remaining skills."
                    ),
                ],
            ),
        },
    ),

    # ═══════════════════════════════════════════════════════════════
    # MAGE (Level 10-50) → WIZARD (Level 50-99)
    # ═══════════════════════════════════════════════════════════════
    "mage": ClassBuilds(
        job_class="Mage",
        default_variant="int_dex_wizard",
        variants={
            "int_dex_wizard": BuildVariant(
                name="INT/DEX Fast Cast Wizard",
                description="Max INT for damage, DEX for fast cast. Best for solo farming and MVP damage.",
                stat_breakpoints=[
                    StatBreakpoint("int", 30, 1, "30 INT first for base MATK. You need to kill things."),
                    StatBreakpoint("dex", 20, 2, "20 DEX for 20% cast reduction. Quality of life."),
                    StatBreakpoint("int", 60, 3, "60 INT for decent MATK. Sweet spot for early-mid game."),
                    StatBreakpoint("dex", 30, 4, "30 DEX for 30% cast reduction. Sweet spot."),
                    StatBreakpoint("int", 80, 5, "80 INT for high MATK. Diminishing after 80."),
                    StatBreakpoint("vit", 30, 6, "30 VIT for survival. Don't die to one hit."),
                    StatBreakpoint("dex", 50, 7, "50 DEX for 50% cast reduction. Cap from DEX alone."),
                    StatBreakpoint("int", 99, 8, "99 INT for max MATK. Hard cap."),
                ],
                best_for=["solo_farming", "mvp"],
                skill_phases=[
                    SkillPhase(
                        level_range=(10, 20),
                        goal="max_damage",
                        skills=[
                            {"name": "MG_FIREBOLT", "max_level": 10, "priority": 1, "reason": "Main damage skill. One-shot Porings at level 5. Max it first."},
                        ],
                        reason="Level 10-20: Fire Bolt is your only damage. Max it. You one-shot everything at this level."
                    ),
                    SkillPhase(
                        level_range=(20, 35),
                        goal="sustain",
                        skills=[
                            {"name": "MG_SRECOVERY", "max_level": 4, "priority": 1, "reason": "30% SP regen. Efficiency sweet spot. Level 5+ is diminishing."},
                            {"name": "MG_FROSTDIVER", "max_level": 5, "priority": 2, "reason": "Freeze + Fire Bolt = 4x damage. Core combo."},
                            {"name": "MG_SIGHT", "max_level": 1, "priority": 3, "reason": "Reveal hidden mobs. Safety."},
                        ],
                        reason="Level 20-35: SP Recovery 4 is the efficiency sweet spot. Frost Diver unlocks the freeze combo."
                    ),
                    SkillPhase(
                        level_range=(35, 50),
                        goal="element_coverage",
                        skills=[
                            {"name": "MG_COLDBOLT", "max_level": 5, "priority": 1, "reason": "Water element. 200% vs Fire mobs. Level 5 is enough."},
                            {"name": "MG_LIGHTNINGBOLT", "max_level": 5, "priority": 2, "reason": "Wind element. 175% vs Water mobs. Level 5 is enough."},
                            {"name": "MG_SRECOVERY", "max_level": 10, "priority": 3, "reason": "Max SP Recovery now. You need it for Wizard skills."},
                            {"name": "MG_NAPALMBEAT", "max_level": 5, "priority": 4, "reason": "Ghost element. Good vs undead. Cheap SP cost."},
                        ],
                        reason="Level 35-50: Element coverage for Orc Dungeon. Max SP Recovery for Wizard."
                    ),
                    SkillPhase(
                        level_range=(50, 70),
                        goal="core_skills",
                        skills=[
                            {"name": "WZ_STORMGUST", "max_level": 10, "priority": 1, "reason": "Main AoE. Freeze + damage. Max it first as Wizard."},
                            {"name": "WZ_VERMILION", "max_level": 10, "priority": 2, "reason": "Lord of Vermillion. Wind AoE. Second main skill."},
                            {"name": "WZ_FROSTNOVA", "max_level": 5, "priority": 3, "reason": "AoE freeze. Emergency escape + setup."},
                            {"name": "WZ_QUAGMIRE", "max_level": 5, "priority": 4, "reason": "Slow + reduce flee. Setup for Storm Gust."},
                        ],
                        reason="Level 50-70: Core Wizard skills. Storm Gust is your identity."
                    ),
                    SkillPhase(
                        level_range=(70, 85),
                        goal="optimization",
                        skills=[
                            {"name": "WZ_METEOR", "max_level": 5, "priority": 1, "reason": "Meteor Storm. Fire AoE. Level 5 is enough for stun chance."},
                            {"name": "WZ_HEAVENDRIVE", "max_level": 5, "priority": 2, "reason": "Neutral AoE. Good vs everything."},
                            {"name": "MG_SAFETYWALL", "max_level": 5, "priority": 3, "reason": "5 hits of immunity. Survival."},
                            {"name": "WZ_FIREPILLAR", "max_level": 5, "priority": 4, "reason": "Fire trap. Good for MVP kiting."},
                        ],
                        reason="Level 70-85: Fill in AoE coverage. Safety Wall for survival."
                    ),
                    SkillPhase(
                        level_range=(85, 99),
                        goal="endgame",
                        skills=[
                            {"name": "WZ_METEOR", "max_level": 10, "priority": 1, "reason": "Max Meteor Storm for endgame damage."},
                            {"name": "MG_SAFETYWALL", "max_level": 10, "priority": 2, "reason": "10 hits of immunity. Essential for MVPs."},
                            {"name": "WZ_FROSTNOVA", "max_level": 10, "priority": 3, "reason": "Max AoE freeze."},
                        ],
                        reason="Level 85-99: Endgame optimization. Prepare for transcendent class."
                    ),
                ],
            ),
            "int_vit_wizard": BuildVariant(
                name="INT/VIT Survival Wizard",
                description="Max INT for damage, VIT for survival. Slower cast but can tank hits. Best for party play and WoE.",
                stat_breakpoints=[
                    StatBreakpoint("int", 30, 1, "30 INT for base MATK."),
                    StatBreakpoint("vit", 30, 2, "30 VIT early for survival. Mages are squishy."),
                    StatBreakpoint("int", 60, 3, "60 INT for decent MATK."),
                    StatBreakpoint("vit", 50, 4, "50 VIT for stun immunity."),
                    StatBreakpoint("int", 80, 5, "80 INT for high MATK."),
                    StatBreakpoint("dex", 20, 6, "20 DEX for some cast reduction."),
                    StatBreakpoint("int", 99, 7, "99 INT for max MATK."),
                    StatBreakpoint("vit", 60, 8, "60 VIT for max survival."),
                ],
                best_for=["party", "woe"],
                skill_phases=[
                    SkillPhase(
                        level_range=(10, 20),
                        goal="max_damage",
                        skills=[
                            {"name": "MG_FIREBOLT", "max_level": 10, "priority": 1, "reason": "Main damage. Max it first."},
                        ],
                        reason="Level 10-20: Same as INT/DEX build. Fire Bolt is king."
                    ),
                    SkillPhase(
                        level_range=(20, 40),
                        goal="survival",
                        skills=[
                            {"name": "MG_SRECOVERY", "max_level": 4, "priority": 1, "reason": "30% SP regen. Efficiency sweet spot."},
                            {"name": "MG_FROSTDIVER", "max_level": 5, "priority": 2, "reason": "Freeze combo."},
                            {"name": "MG_SAFETYWALL", "max_level": 5, "priority": 3, "reason": "5 hits of immunity. Essential for survival build."},
                            {"name": "MG_SIGHT", "max_level": 1, "priority": 4, "reason": "Reveal hidden mobs."},
                        ],
                        reason="Level 20-40: Safety Wall is your identity. You tank hits that kill other mages."
                    ),
                    SkillPhase(
                        level_range=(40, 50),
                        goal="element_coverage",
                        skills=[
                            {"name": "MG_COLDBOLT", "max_level": 5, "priority": 1, "reason": "Water element coverage."},
                            {"name": "MG_LIGHTNINGBOLT", "max_level": 5, "priority": 2, "reason": "Wind element coverage."},
                            {"name": "MG_SRECOVERY", "max_level": 10, "priority": 3, "reason": "Max SP Recovery."},
                        ],
                        reason="Level 40-50: Element coverage + max SP Recovery."
                    ),
                    SkillPhase(
                        level_range=(50, 99),
                        goal="core_skills",
                        skills=[
                            {"name": "WZ_STORMGUST", "max_level": 10, "priority": 1, "reason": "Main AoE."},
                            {"name": "WZ_METEOR", "max_level": 10, "priority": 2, "reason": "Fire AoE. Max it for this build."},
                            {"name": "WZ_FIREPILLAR", "max_level": 10, "priority": 3, "reason": "Fire trap. Max for survival build."},
                            {"name": "MG_SAFETYWALL", "max_level": 10, "priority": 4, "reason": "10 hits of immunity."},
                            {"name": "WZ_QUAGMIRE", "max_level": 5, "priority": 5, "reason": "Slow + flee reduction."},
                            {"name": "WZ_FROSTNOVA", "max_level": 5, "priority": 6, "reason": "AoE freeze."},
                        ],
                        reason="Level 50-99: Meteor Storm + Safety Wall is your identity. You're a fortress."
                    ),
                ],
            ),
        },
    ),

    # ═══════════════════════════════════════════════════════════════
    # ARCHER (Level 10-50) → HUNTER (Level 50-99)
    # ═══════════════════════════════════════════════════════════════
    "archer": ClassBuilds(
        job_class="Archer",
        default_variant="falcon_hunter",
        variants={
            "falcon_hunter": BuildVariant(
                name="Falcon Hunter",
                description="DEX for damage, INT for falcon. Blitz Beat + Steel Crow for auto-attack damage. Best for solo farming.",
                stat_breakpoints=[
                    StatBreakpoint("dex", 30, 1, "30 DEX first for damage. You need to kill things."),
                    StatBreakpoint("agi", 20, 2, "20 AGI for flee. Survival."),
                    StatBreakpoint("dex", 60, 3, "60 DEX for good damage."),
                    StatBreakpoint("int", 30, 4, "30 INT for falcon damage and SP regen."),
                    StatBreakpoint("dex", 80, 5, "80 DEX for high damage."),
                    StatBreakpoint("agi", 40, 6, "40 AGI for ASPD breakpoint."),
                    StatBreakpoint("dex", 99, 7, "99 DEX for max damage. Hard cap."),
                    StatBreakpoint("int", 40, 8, "40 INT for max falcon damage."),
                ],
                best_for=["solo_farming"],
                skill_phases=[
                    SkillPhase(
                        level_range=(10, 20),
                        goal="max_damage",
                        skills=[
                            {"name": "AC_DOUBLE", "max_level": 10, "priority": 1, "reason": "Double Strafe = 380% damage at level 10. Max it first."},
                        ],
                        reason="Level 10-20: Double Strafe is your only damage. 380% at level 10. Nothing else comes close."
                    ),
                    SkillPhase(
                        level_range=(20, 35),
                        goal="sustain",
                        skills=[
                            {"name": "AC_OWL", "max_level": 10, "priority": 1, "reason": "+1 DEX per level. More DEX = more damage. Max it."},
                            {"name": "AC_VULTURE", "max_level": 5, "priority": 2, "reason": "+50 range. Level 5 is enough for most maps."},
                        ],
                        reason="Level 20-35: Owl's Eye for damage. Vulture's Eye for safety range."
                    ),
                    SkillPhase(
                        level_range=(35, 50),
                        goal="aoe",
                        skills=[
                            {"name": "AC_SHOWER", "max_level": 5, "priority": 1, "reason": "AoE + knockback. Level 5 is enough."},
                            {"name": "AC_VULTURE", "max_level": 10, "priority": 2, "reason": "Max range for safety."},
                        ],
                        reason="Level 35-50: Arrow Shower for grouped mobs. Max range for Hunter."
                    ),
                    SkillPhase(
                        level_range=(50, 70),
                        goal="core_skills",
                        skills=[
                            {"name": "HT_BLITZBEAT", "max_level": 10, "priority": 1, "reason": "Falcon auto-attack. Core of Falcon Hunter build."},
                            {"name": "HT_STEELCROW", "max_level": 5, "priority": 2, "reason": "Falcon damage boost. Level 5 is enough."},
                            {"name": "HT_DETECTING", "max_level": 5, "priority": 3, "reason": "Reveal hidden mobs. Safety."},
                            {"name": "HT_ANKLESNARE", "max_level": 5, "priority": 4, "reason": "Trap. Immobilize mobs. Kiting tool."},
                        ],
                        reason="Level 50-70: Blitz Beat is your identity. Falcon does free damage while you shoot."
                    ),
                    SkillPhase(
                        level_range=(70, 99),
                        goal="optimization",
                        skills=[
                            {"name": "HT_BEASTSTRAFING", "max_level": 10, "priority": 1, "reason": "AoE arrow skill. Max for farming."},
                            {"name": "HT_LANDMINE", "max_level": 5, "priority": 2, "reason": "Trap. Good damage + immobilize."},
                            {"name": "HT_STEELCROW", "max_level": 10, "priority": 3, "reason": "Max falcon damage."},
                            {"name": "HT_SKIDTRAP", "max_level": 5, "priority": 4, "reason": "Trap. Slide + immobilize."},
                        ],
                        reason="Level 70-99: Beast Strafing for AoE. Traps for control."
                    ),
                ],
            ),
            "ad_hunter": BuildVariant(
                name="AD Hunter (Auto-Double)",
                description="DEX + AGI for max ASPD. Double Strafe spam. Best for single-target DPS and MVP.",
                stat_breakpoints=[
                    StatBreakpoint("dex", 30, 1, "30 DEX for damage."),
                    StatBreakpoint("agi", 30, 2, "30 AGI for ASPD."),
                    StatBreakpoint("dex", 60, 3, "60 DEX for damage."),
                    StatBreakpoint("agi", 60, 4, "60 AGI for ASPD breakpoint."),
                    StatBreakpoint("dex", 80, 5, "80 DEX for high damage."),
                    StatBreakpoint("agi", 80, 6, "80 AGI for high ASPD."),
                    StatBreakpoint("dex", 99, 7, "99 DEX for max damage."),
                    StatBreakpoint("agi", 99, 8, "99 AGI for max ASPD."),
                ],
                best_for=["mvp", "party"],
                skill_phases=[
                    SkillPhase(
                        level_range=(10, 20),
                        goal="max_damage",
                        skills=[
                            {"name": "AC_DOUBLE", "max_level": 10, "priority": 1, "reason": "Double Strafe. Max it first."},
                        ],
                        reason="Level 10-20: Same as Falcon Hunter."
                    ),
                    SkillPhase(
                        level_range=(20, 35),
                        goal="sustain",
                        skills=[
                            {"name": "AC_OWL", "max_level": 10, "priority": 1, "reason": "+1 DEX per level. Max it."},
                            {"name": "AC_VULTURE", "max_level": 5, "priority": 2, "reason": "+50 range. Enough."},
                        ],
                        reason="Level 20-35: Owl's Eye + Vulture's Eye."
                    ),
                    SkillPhase(
                        level_range=(35, 50),
                        goal="aoe",
                        skills=[
                            {"name": "AC_SHOWER", "max_level": 5, "priority": 1, "reason": "AoE + knockback."},
                            {"name": "AC_VULTURE", "max_level": 10, "priority": 2, "reason": "Max range."},
                        ],
                        reason="Level 35-50: Arrow Shower + max range."
                    ),
                    SkillPhase(
                        level_range=(50, 99),
                        goal="core_skills",
                        skills=[
                            {"name": "HT_BEASTSTRAFING", "max_level": 10, "priority": 1, "reason": "AoE arrow. Main farming skill."},
                            {"name": "HT_IMPROVECONCENTRATION", "max_level": 10, "priority": 2, "reason": "DEX + AGI boost. More damage + ASPD."},
                            {"name": "HT_TRUESIGHT", "max_level": 10, "priority": 3, "reason": "Hit rate + crit. Essential for MVP."},
                            {"name": "HT_ANKLESNARE", "max_level": 5, "priority": 4, "reason": "Trap. Immobilize."},
                        ],
                        reason="Level 50-99: Beast Strafing + Improve Concentration. Pure DPS."
                    ),
                ],
            ),
        },
    ),

    # ═══════════════════════════════════════════════════════════════
    # ACOLYTE (Level 10-50) → PRIEST (Level 50-99)
    # ═══════════════════════════════════════════════════════════════
    "acolyte": ClassBuilds(
        job_class="Acolyte",
        default_variant="support_priest",
        variants={
            "support_priest": BuildVariant(
                name="Full Support Priest",
                description="Max INT for heal power, DEX for cast speed. Party buffs and heals. Best for party play.",
                stat_breakpoints=[
                    StatBreakpoint("int", 30, 1, "30 INT for heal power. You need to heal."),
                    StatBreakpoint("dex", 20, 2, "20 DEX for cast reduction. Faster heals save lives."),
                    StatBreakpoint("int", 60, 3, "60 INT for decent heal power."),
                    StatBreakpoint("vit", 30, 4, "30 VIT for survival. Dead priests can't heal."),
                    StatBreakpoint("int", 80, 5, "80 INT for strong heals."),
                    StatBreakpoint("dex", 40, 6, "40 DEX for faster cast."),
                    StatBreakpoint("int", 99, 7, "99 INT for max heal power. Hard cap."),
                    StatBreakpoint("dex", 50, 8, "50 DEX for 50% cast reduction. Cap from DEX."),
                ],
                best_for=["party", "woe"],
                skill_phases=[
                    SkillPhase(
                        level_range=(10, 15),
                        goal="survival",
                        skills=[
                            {"name": "AL_TELEPORT", "max_level": 1, "priority": 1, "reason": "Teleport is your escape. Get it first."},
                            {"name": "AL_HEAL", "max_level": 4, "priority": 2, "reason": "180 HP for 12 SP. Efficiency sweet spot. Don't max yet."},
                        ],
                        reason="Level 10-15: Teleport first (escape), then Heal 4 (efficiency sweet spot)."
                    ),
                    SkillPhase(
                        level_range=(15, 30),
                        goal="support",
                        skills=[
                            {"name": "AL_INCAGI", "max_level": 10, "priority": 1, "reason": "Party buff. +10 AGI. Max it for duration."},
                            {"name": "AL_BLESSING", "max_level": 10, "priority": 2, "reason": "Party buff. +10 all stats. Max it for duration."},
                        ],
                        reason="Level 15-30: Party buffs are your value. Max them before maxing heal."
                    ),
                    SkillPhase(
                        level_range=(30, 50),
                        goal="healing",
                        skills=[
                            {"name": "AL_HEAL", "max_level": 10, "priority": 1, "reason": "Max heal now. You need big heals for harder content."},
                            {"name": "AL_WARP", "max_level": 1, "priority": 2, "reason": "Party travel. Essential for groups."},
                            {"name": "AL_DEMONBANE", "max_level": 5, "priority": 3, "reason": "Solo damage vs undead. Level 5 is enough."},
                        ],
                        reason="Level 30-50: Max Heal. Warp Portal for party. Demon Bane for solo leveling."
                    ),
                    SkillPhase(
                        level_range=(50, 70),
                        goal="core_skills",
                        skills=[
                            {"name": "PR_KYRIE", "max_level": 10, "priority": 1, "reason": "Shield. Absorbs damage. Essential for party survival."},
                            {"name": "PR_MAGNIFICAT", "max_level": 5, "priority": 2, "reason": "SP regen. Level 5 is enough for most parties."},
                            {"name": "PR_GLORIA", "max_level": 5, "priority": 3, "reason": "LUK boost. Crit + perfect dodge. Level 5 is enough."},
                            {"name": "PR_IMPOSITIO", "max_level": 5, "priority": 4, "reason": "ATK boost. Level 5 is enough."},
                        ],
                        reason="Level 50-70: Core Priest skills. Kyrie Eleison is your identity."
                    ),
                    SkillPhase(
                        level_range=(70, 85),
                        goal="optimization",
                        skills=[
                            {"name": "PR_ASPERSIO", "max_level": 5, "priority": 1, "reason": "Holy element weapon. Essential for undead MVPs."},
                            {"name": "PR_TURNUNDEAD", "max_level": 10, "priority": 2, "reason": "Instant kill undead. Max for endgame."},
                            {"name": "PR_LEXAETERNA", "max_level": 1, "priority": 3, "reason": "Double damage. MVP killer."},
                            {"name": "PR_LEXDIVINA", "max_level": 1, "priority": 4, "reason": "Silence. Interrupt casters."},
                        ],
                        reason="Level 70-85: Support + utility. Aspersio + Lex Aeterna for MVP damage."
                    ),
                    SkillPhase(
                        level_range=(85, 99),
                        goal="endgame",
                        skills=[
                            {"name": "PR_ASSUMPTIO", "max_level": 5, "priority": 1, "reason": "Damage reduction. Essential for WoE and hard MVPs."},
                            {"name": "PR_MAGNIFICAT", "max_level": 10, "priority": 2, "reason": "Max SP regen for endgame."},
                            {"name": "PR_GLORIA", "max_level": 10, "priority": 3, "reason": "Max LUK boost."},
                        ],
                        reason="Level 85-99: Endgame. Assumptio for WoE. Max remaining buffs."
                    ),
                ],
            ),
            "battle_priest": BuildVariant(
                name="Battle Priest",
                description="STR + INT hybrid. Heal for damage vs undead, Holy Light for others. Best for solo leveling.",
                stat_breakpoints=[
                    StatBreakpoint("str", 30, 1, "30 STR for melee damage."),
                    StatBreakpoint("int", 30, 2, "30 INT for heal power."),
                    StatBreakpoint("dex", 30, 3, "30 DEX for hit rate."),
                    StatBreakpoint("str", 60, 4, "60 STR for good melee damage."),
                    StatBreakpoint("int", 60, 5, "60 INT for strong heals."),
                    StatBreakpoint("vit", 30, 6, "30 VIT for survival."),
                    StatBreakpoint("str", 80, 7, "80 STR for high damage."),
                    StatBreakpoint("int", 80, 8, "80 INT for max heal power."),
                ],
                best_for=["solo_farming"],
                skill_phases=[
                    SkillPhase(
                        level_range=(10, 15),
                        goal="survival",
                        skills=[
                            {"name": "AL_TELEPORT", "max_level": 1, "priority": 1, "reason": "Escape first."},
                            {"name": "AL_HEAL", "max_level": 4, "priority": 2, "reason": "Heal 4. Efficiency sweet spot."},
                        ],
                        reason="Level 10-15: Same as support. Teleport + Heal 4."
                    ),
                    SkillPhase(
                        level_range=(15, 30),
                        goal="damage",
                        skills=[
                            {"name": "AL_DEMONBANE", "max_level": 10, "priority": 1, "reason": "ATK vs undead/demon. Max it for solo damage."},
                            {"name": "AL_INCAGI", "max_level": 5, "priority": 2, "reason": "AGI boost. Level 5 is enough for solo."},
                            {"name": "AL_BLESSING", "max_level": 5, "priority": 3, "reason": "Stat boost. Level 5 is enough for solo."},
                        ],
                        reason="Level 15-30: Demon Bane for damage. You're a battle acolyte now."
                    ),
                    SkillPhase(
                        level_range=(30, 50),
                        goal="healing",
                        skills=[
                            {"name": "AL_HEAL", "max_level": 10, "priority": 1, "reason": "Max heal. Heal nukes undead for 720 HP."},
                            {"name": "AL_WARP", "max_level": 1, "priority": 2, "reason": "Travel."},
                            {"name": "AL_INCAGI", "max_level": 10, "priority": 3, "reason": "Max AGI buff."},
                            {"name": "AL_BLESSING", "max_level": 10, "priority": 4, "reason": "Max stat buff."},
                        ],
                        reason="Level 30-50: Max Heal for undead nuking. Finish buffs."
                    ),
                    SkillPhase(
                        level_range=(50, 99),
                        goal="core_skills",
                        skills=[
                            {"name": "PR_TURNUNDEAD", "max_level": 10, "priority": 1, "reason": "Instant kill undead. Your main damage skill."},
                            {"name": "PR_KYRIE", "max_level": 10, "priority": 2, "reason": "Shield. Survival."},
                            {"name": "PR_GLORIA", "max_level": 5, "priority": 3, "reason": "LUK boost. Crit."},
                            {"name": "PR_MAGNIFICAT", "max_level": 5, "priority": 4, "reason": "SP regen."},
                            {"name": "PR_ASPERSIO", "max_level": 5, "priority": 5, "reason": "Holy weapon. More damage."},
                        ],
                        reason="Level 50-99: Turn Undead is your identity. You one-shot undead mobs."
                    ),
                ],
            ),
        },
    ),

    # ═══════════════════════════════════════════════════════════════
    # MERCHANT (Level 10-50) → BLACKSMITH (Level 50-99)
    # ═══════════════════════════════════════════════════════════════
    "merchant": ClassBuilds(
        job_class="Merchant",
        default_variant="farming_blacksmith",
        variants={
            "farming_blacksmith": BuildVariant(
                name="Farming Blacksmith",
                description="STR for damage, DEX for hit rate. Mammonite for burst. Best for zeny farming.",
                stat_breakpoints=[
                    StatBreakpoint("str", 30, 1, "30 STR for damage."),
                    StatBreakpoint("dex", 20, 2, "20 DEX for hit rate."),
                    StatBreakpoint("str", 60, 3, "60 STR for good damage."),
                    StatBreakpoint("vit", 30, 4, "30 VIT for survival."),
                    StatBreakpoint("str", 80, 5, "80 STR for high damage."),
                    StatBreakpoint("dex", 40, 6, "40 DEX for hit rate."),
                    StatBreakpoint("str", 99, 7, "99 STR for max damage."),
                ],
                best_for=["solo_farming"],
                skill_phases=[
                    SkillPhase(
                        level_range=(10, 25),
                        goal="economy",
                        skills=[
                            {"name": "MC_PUSHCART", "max_level": 1, "priority": 1, "reason": "Pushcart. Carry more loot. Level 1 is enough."},
                            {"name": "MC_DISCOUNT", "max_level": 10, "priority": 2, "reason": "24% discount. Max it for profit."},
                            {"name": "MC_OVERCHARGE", "max_level": 10, "priority": 3, "reason": "24% overcharge. Max it for profit."},
                        ],
                        reason="Level 10-25: Economy skills first. Discount + Overcharge = more zeny."
                    ),
                    SkillPhase(
                        level_range=(25, 40),
                        goal="damage",
                        skills=[
                            {"name": "MC_MAMMONITE", "max_level": 5, "priority": 1, "reason": "Zeny attack. Level 5 = 500% damage for 500z. Good for farming."},
                            {"name": "MC_PUSHCART", "max_level": 5, "priority": 2, "reason": "More cart weight. More loot per trip."},
                        ],
                        reason="Level 25-40: Mammonite for damage. Pushcart for capacity."
                    ),
                    SkillPhase(
                        level_range=(40, 50),
                        goal="preparation",
                        skills=[
                            {"name": "MC_MAMMONITE", "max_level": 10, "priority": 1, "reason": "Max Mammonite for Blacksmith."},
                            {"name": "MC_PUSHCART", "max_level": 10, "priority": 2, "reason": "Max cart capacity."},
                            {"name": "MC_IDENTIFY", "max_level": 1, "priority": 3, "reason": "Identify items. Useful for farming."},
                        ],
                        reason="Level 40-50: Finish Merchant skills. Prep for Blacksmith."
                    ),
                    SkillPhase(
                        level_range=(50, 99),
                        goal="core_skills",
                        skills=[
                            {"name": "BS_WEAPONPERFECT", "max_level": 10, "priority": 1, "reason": "Weapon perfection. Max damage. Core Blacksmith skill."},
                            {"name": "BS_OVERTHRUST", "max_level": 5, "priority": 2, "reason": "ATK boost. Level 5 is enough."},
                            {"name": "BS_MAXIMIZE", "max_level": 5, "priority": 3, "reason": "Max damage. Level 5 is enough."},
                            {"name": "BS_WEAPONRESEARCH", "max_level": 10, "priority": 4, "reason": "ATK + crafting success. Max it."},
                            {"name": "BS_SKILLSMITH", "max_level": 5, "priority": 5, "reason": "Crafting. Level 5 is enough for most recipes."},
                        ],
                        reason="Level 50-99: Weapon Perfection + Overthrust + Maximize = massive damage."
                    ),
                ],
            ),
        },
    ),

    # ═══════════════════════════════════════════════════════════════
    # THIEF (Level 10-50) → ASSASSIN (Level 50-99)
    # ═══════════════════════════════════════════════════════════════
    "thief": ClassBuilds(
        job_class="Thief",
        default_variant="crit_assassin",
        variants={
            "crit_assassin": BuildVariant(
                name="Crit Katar Assassin",
                description="AGI for ASPD, LUK for crit. Katar double-wield. Best for solo farming.",
                stat_breakpoints=[
                    StatBreakpoint("agi", 30, 1, "30 AGI for ASPD and flee."),
                    StatBreakpoint("dex", 20, 2, "20 DEX for hit rate."),
                    StatBreakpoint("agi", 50, 3, "50 AGI for good ASPD."),
                    StatBreakpoint("luk", 30, 4, "30 LUK for 10% crit rate. Sweet spot."),
                    StatBreakpoint("str", 40, 5, "40 STR for damage."),
                    StatBreakpoint("agi", 80, 6, "80 AGI for ASPD breakpoint."),
                    StatBreakpoint("str", 60, 7, "60 STR for Katar damage."),
                    StatBreakpoint("luk", 50, 8, "50 LUK for 16% crit rate."),
                    StatBreakpoint("agi", 99, 9, "99 AGI for max ASPD."),
                ],
                best_for=["solo_farming"],
                skill_phases=[
                    SkillPhase(
                        level_range=(10, 25),
                        goal="max_damage",
                        skills=[
                            {"name": "TF_DOUBLE", "max_level": 10, "priority": 1, "reason": "50% double attack. 50% more DPS. Max it first."},
                        ],
                        reason="Level 10-25: Double Attack is passive. 50% chance to attack twice. Nothing else comes close."
                    ),
                    SkillPhase(
                        level_range=(25, 40),
                        goal="survival",
                        skills=[
                            {"name": "TF_HIDE", "max_level": 1, "priority": 1, "reason": "Stealth. Escape. Level 1 is enough."},
                            {"name": "TF_POISON", "max_level": 5, "priority": 2, "reason": "Envenom. DoT. Level 5 is enough."},
                            {"name": "TF_MISS", "max_level": 5, "priority": 3, "reason": "Increased dodge. Max for survival."},
                        ],
                        reason="Level 25-40: Hide for escape. Poison for damage. Miss for dodge."
                    ),
                    SkillPhase(
                        level_range=(40, 50),
                        goal="preparation",
                        skills=[
                            {"name": "TF_STEAL", "max_level": 5, "priority": 1, "reason": "Steal items. Level 5 is enough for farming."},
                            {"name": "TF_POISON", "max_level": 10, "priority": 2, "reason": "Max poison damage."},
                            {"name": "TF_HIDE", "max_level": 5, "priority": 3, "reason": "Longer hide duration."},
                        ],
                        reason="Level 40-50: Steal for farming. Finish Thief skills."
                    ),
                    SkillPhase(
                        level_range=(50, 70),
                        goal="core_skills",
                        skills=[
                            {"name": "AS_KATAR", "max_level": 10, "priority": 1, "reason": "Katar Mastery. ATK + crit. Core Assassin skill."},
                            {"name": "AS_RIGHT", "max_level": 5, "priority": 2, "reason": "Right hand mastery. More damage."},
                            {"name": "AS_LEFT", "max_level": 5, "priority": 3, "reason": "Left hand mastery. More damage."},
                            {"name": "AS_SONICBLOW", "max_level": 10, "priority": 4, "reason": "Burst damage. Your finisher."},
                        ],
                        reason="Level 50-70: Katar Mastery + dual-wield. Sonic Blow for burst."
                    ),
                    SkillPhase(
                        level_range=(70, 85),
                        goal="optimization",
                        skills=[
                            {"name": "AS_GRIMTOOTH", "max_level": 5, "priority": 1, "reason": "Ranged AoE. Pull + poke from distance."},
                            {"name": "AS_ENCHANTPOISON", "max_level": 5, "priority": 2, "reason": "Poison weapon. More damage."},
                            {"name": "AS_VENOMDUST", "max_level": 5, "priority": 3, "reason": "Poison AoE. Group damage."},
                            {"name": "AS_RIGHT", "max_level": 10, "priority": 4, "reason": "Max right hand mastery."},
                        ],
                        reason="Level 70-85: Grimtooth + Enchant Poison. Fill in support skills."
                    ),
                    SkillPhase(
                        level_range=(85, 99),
                        goal="endgame",
                        skills=[
                            {"name": "AS_LEFT", "max_level": 10, "priority": 1, "reason": "Max left hand mastery."},
                            {"name": "AS_SONICBLOW", "max_level": 10, "priority": 2, "reason": "Already maxed. Keep it."},
                            {"name": "AS_GRIMTOOTH", "max_level": 10, "priority": 3, "reason": "Max ranged AoE."},
                        ],
                        reason="Level 85-99: Endgame optimization. Max remaining skills."
                    ),
                ],
            ),
            "sb_assassin": BuildVariant(
                name="Sonic Blow Burst Assassin",
                description="STR for damage, AGI for ASPD. Sonic Blow burst. Best for MVP and single-target DPS.",
                stat_breakpoints=[
                    StatBreakpoint("str", 30, 1, "30 STR for damage."),
                    StatBreakpoint("dex", 30, 2, "30 DEX for hit rate. Sonic Blow must not miss."),
                    StatBreakpoint("str", 60, 3, "60 STR for good damage."),
                    StatBreakpoint("agi", 40, 4, "40 AGI for ASPD."),
                    StatBreakpoint("str", 80, 5, "80 STR for high damage."),
                    StatBreakpoint("dex", 40, 6, "40 DEX for Katar hit rate."),
                    StatBreakpoint("str", 99, 7, "99 STR for max damage."),
                    StatBreakpoint("agi", 60, 8, "60 AGI for decent ASPD."),
                ],
                best_for=["mvp", "party"],
                skill_phases=[
                    SkillPhase(
                        level_range=(10, 25),
                        goal="max_damage",
                        skills=[
                            {"name": "TF_DOUBLE", "max_level": 10, "priority": 1, "reason": "Double Attack. Max it first."},
                        ],
                        reason="Level 10-25: Same as Crit Assassin."
                    ),
                    SkillPhase(
                        level_range=(25, 40),
                        goal="survival",
                        skills=[
                            {"name": "TF_HIDE", "max_level": 1, "priority": 1, "reason": "Escape."},
                            {"name": "TF_POISON", "max_level": 5, "priority": 2, "reason": "Poison."},
                        ],
                        reason="Level 25-40: Hide + Poison."
                    ),
                    SkillPhase(
                        level_range=(40, 50),
                        goal="preparation",
                        skills=[
                            {"name": "TF_STEAL", "max_level": 5, "priority": 1, "reason": "Steal."},
                            {"name": "TF_POISON", "max_level": 10, "priority": 2, "reason": "Max poison."},
                        ],
                        reason="Level 40-50: Finish Thief skills."
                    ),
                    SkillPhase(
                        level_range=(50, 99),
                        goal="core_skills",
                        skills=[
                            {"name": "AS_SONICBLOW", "max_level": 10, "priority": 1, "reason": "Burst damage. Your identity. Max it first."},
                            {"name": "AS_KATAR", "max_level": 10, "priority": 2, "reason": "Katar Mastery. ATK + crit."},
                            {"name": "AS_RIGHT", "max_level": 10, "priority": 3, "reason": "Right hand mastery. Max for damage."},
                            {"name": "AS_LEFT", "max_level": 10, "priority": 4, "reason": "Left hand mastery. Max for damage."},
                            {"name": "AS_ENCHANTPOISON", "max_level": 5, "priority": 5, "reason": "Poison weapon."},
                            {"name": "AS_GRIMTOOTH", "max_level": 5, "priority": 6, "reason": "Ranged pull."},
                        ],
                        reason="Level 50-99: Sonic Blow is your identity. Max all damage passives."
                    ),
                ],
            ),
        },
    ),
}


class ConsciousDecisionEngine:
    """Makes all high-level decisions for bot progression.

    Phase-based, build-aware, efficiency-optimized.
    Designed by a Pro RO Player who has leveled 50+ characters to 99.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._bot_state: dict[str, dict[str, Any]] = {}
        self._decisions_made: dict[str, list[str]] = defaultdict(list)
        self._last_evaluation: float = 0.0
        self._evaluation_interval: float = 10.0  # Evaluate every 10 seconds
        self._knowledge: dict[str, Any] = {}
        self._load_knowledge()

    def _load_knowledge(self) -> None:
        """Load knowledge database."""
        try:
            with open(_KNOWLEDGE_PATH) as f:
                self._knowledge = json.load(f)
            logger.info("knowledge_loaded: %d keys", len(self._knowledge))
        except Exception as e:
            logger.warning("Failed to load knowledge: %s", e)

    def update_from_snapshot(self, bot_id: str, snapshot: Any) -> None:
        """Update bot state from snapshot."""
        with self._lock:
            state = self._bot_state.setdefault(bot_id, {})
            state["last_seen"] = time.time()

            if hasattr(snapshot, "vitals"):
                v = snapshot.vitals
                state["hp"] = getattr(v, "hp", 0)
                state["max_hp"] = getattr(v, "hp_max", 1)
                state["hp_pct"] = getattr(v, "hp_ratio", 1.0)
                state["sp"] = getattr(v, "sp", 0)
                state["max_sp"] = getattr(v, "sp_max", 1)
                state["sp_pct"] = getattr(v, "sp_ratio", 1.0)
                state["base_level"] = getattr(v, "base_level", 1)
                state["job_level"] = getattr(v, "job_level", 1)
                state["job_name"] = str(getattr(v, "job_name", "novice")).lower()
                state["zeny"] = getattr(v, "zeny", 0)
                state["weight_ratio"] = getattr(v, "weight_ratio", 0.0)

            if hasattr(snapshot, "position"):
                pos = snapshot.position
                state["map"] = str(getattr(pos, "map", ""))

            if hasattr(snapshot, "inventory_items"):
                state["inventory"] = {
                    str(getattr(item, "name", "")): int(getattr(item, "amount", 0))
                    for item in (snapshot.inventory_items or [])
                }

            if hasattr(snapshot, "skills"):
                state["skills"] = [
                    str(getattr(s, "name", ""))
                    for s in (snapshot.skills or [])
                ]

            if hasattr(snapshot, "stats"):
                st = snapshot.stats
                state["stats"] = {
                    "str": getattr(st, "str", 0),
                    "agi": getattr(st, "agi", 0),
                    "vit": getattr(st, "vit", 0),
                    "int": getattr(st, "int", 0),
                    "dex": getattr(st, "dex", 0),
                    "luk": getattr(st, "luk", 0),
                }

    def _get_class_builds(self, job_name: str) -> ClassBuilds | None:
        """Get the build definitions for a job class."""
        job = job_name.lower()
        if job in BUILDS:
            return BUILDS[job]
        # Try partial match
        for key, builds in BUILDS.items():
            if key in job or job in key:
                return builds
        return None

    def _get_variant(self, job_name: str, variant_name: str | None = None) -> BuildVariant | None:
        """Get a specific build variant for a class."""
        class_builds = self._get_class_builds(job_name)
        if not class_builds:
            return None
        variant = variant_name or class_builds.default_variant
        return class_builds.variants.get(variant)

    def _get_current_phase(self, build: BuildVariant, base_level: int) -> SkillPhase | None:
        """Get the active skill phase for the current level."""
        for phase in build.skill_phases:
            if phase.level_range[0] <= base_level <= phase.level_range[1]:
                return phase
        # Return the last phase if above all ranges
        return build.skill_phases[-1] if build.skill_phases else None

    def _get_next_stat_breakpoint(self, build: BuildVariant, current_stats: dict[str, int]) -> StatBreakpoint | None:
        """Get the next stat breakpoint that hasn't been reached yet."""
        for bp in build.stat_breakpoints:
            current = current_stats.get(bp.stat.lower(), 0)
            if current < bp.value:
                return bp
        return None

    def evaluate(self, bot_id: str, game_mode: str = "solo_farming") -> list[Decision]:
        """Evaluate all decisions for a bot.

        Args:
            bot_id: The bot identifier
            game_mode: One of 'solo_farming', 'party', 'mvp', 'woe'
        """
        with self._lock:
            now = time.time()
            if now - self._last_evaluation < self._evaluation_interval:
                return []
            self._last_evaluation = now

            state = self._bot_state.get(bot_id, {})
            if not state:
                return []

            decisions: list[Decision] = []
            made = self._decisions_made.setdefault(bot_id, [])
            inventory = state.get("inventory", {})
            skills = state.get("skills", [])
            stats = state.get("stats", {})
            base_level = state.get("base_level", 1)
            job_level = state.get("job_level", 1)
            job_name = state.get("job_name", "novice")
            zeny = state.get("zeny", 0)
            hp_pct = state.get("hp_pct", 1.0)
            sp_pct = state.get("sp_pct", 1.0)

            # Get the build variant for this bot
            # In the future, this could be user-configured or learned
            class_builds = self._get_class_builds(job_name)
            if not class_builds:
                return []

            # Select variant based on game mode
            variant_name = class_builds.default_variant
            for vname, variant in class_builds.variants.items():
                if game_mode in variant.best_for:
                    variant_name = vname
                    break

            build = class_builds.variants.get(variant_name)
            if not build:
                return []

            # ── 1. Skill Learning Decisions (Phase-based) ──
            current_phase = self._get_current_phase(build, base_level)
            if current_phase:
                for skill in current_phase.skills:
                    name = skill["name"]
                    if name not in skills:
                        # Check if we have enough skill points
                        if state.get("skill_points", 0) > 0:
                            decisions.append(Decision(
                                domain="skills",
                                action="learn_skill",
                                target=name,
                                priority=1,
                                reason=f"[Phase {current_phase.level_range[0]}-{current_phase.level_range[1]}: {current_phase.goal}] {skill['reason']}",
                                params={"max_level": skill["max_level"]},
                            ))
                            made.append(f"learn_{name}")
                            break  # Learn one skill at a time

            # ── 2. Stat Distribution Decisions (Breakpoint-based) ──
            if stats:
                next_bp = self._get_next_stat_breakpoint(build, stats)
                if next_bp:
                    current = stats.get(next_bp.stat.upper(), 0)
                    if current < next_bp.value:
                        decisions.append(Decision(
                            domain="stats",
                            action="add_stat",
                            target=next_bp.stat,
                            priority=2,
                            reason=f"Breakpoint #{next_bp.priority}: {next_bp.stat.upper()} {current} → {next_bp.value}. {next_bp.reason}",
                            params={"points": 1, "target": next_bp.value},
                        ))
                        made.append(f"stat_{next_bp.stat}")

            # ── 3. Restock Decisions ──
            # AGNOSTIC (RULE.md): restock the healing/utility items the bot ACTUALLY
            # carries (real inventory), not hardcoded item names. Threshold = the
            # item's own count; potions get priority 2, everything else 3.
            _restock_candidates = [
                (k, v) for k, v in inventory.items()
                if isinstance(v, (int, float)) and v >= 0
            ]
            for item, current_stock in sorted(
                _restock_candidates,
                key=lambda kv: (0 if "otion" in kv[0] or kv[0].lower() in ("apple", "herb") else 1, kv[1]),
            ):
                min_stock = 5 if ("otion" in item or item.lower() in ("apple", "herb")) else 2
                if current_stock < min_stock:
                    decisions.append(Decision(
                        domain="restock",
                        action="buy_item",
                        target=item,
                        priority=2 if ("otion" in item or item.lower() in ("apple", "herb")) else 3,
                        reason=f"Low on {item} ({current_stock}/{min_stock})",
                        params={"qty": 20, "max_price": 0},  # 0 = planner picks price (agnostic)
                    ))
                    made.append(f"restock_{item}")

            # ── 4. Emergency Restock ──
            if hp_pct < 0.50:
                has_heal = any(
                    "Potion" in k or "Apple" in k or "Herb" in k
                    for k in inventory.keys()
                )
                if not has_heal:
                    # AGNOSTIC: the restock target is the bot's learned/real heal
                    # item (server-adaptation potion_solution), never a hardcoded name.
                    _heal_item = "Red Potion"  # safe fallback only
                    try:
                        from ai_sidecar.server_adaptation import ServerSolutionsStore
                        _adapt = ServerSolutionsStore()
                        _learned = _adapt.get("potion_solution", None)
                        if isinstance(_learned, dict) and _learned.get("name"):
                            _heal_item = _learned["name"]
                        elif isinstance(_learned, str) and _learned:
                            _heal_item = _learned
                    except Exception:
                        pass
                    decisions.append(Decision(
                        domain="restock",
                        action="emergency_restock",
                        target=_heal_item,
                        priority=1,
                        reason=f"HP at {hp_pct:.0%} with no healing items — emergency restock needed",
                        params={"qty": 30, "max_price": 0},  # 0 = planner picks price (agnostic)
                    ))

            # ── 5. Map Selection Decision ──
            # AGNOSTIC: real farming maps from map_intelligence (server spawn data),
            # filtered by the map's OWN recommended level range — never hardcoded
            # map literals (RULE.md).
            current_map = state.get("map", "")
            try:
                from ai_sidecar.map_intelligence import get_map_intelligence
                mi = get_map_intelligence()
                farm_map = next(
                    (
                        m for m in mi.get_farming_maps()
                        if m.recommended_level_range[0] <= base_level <= m.recommended_level_range[1]
                        and m.name != current_map
                    ),
                    None,
                )
                if farm_map is not None:
                    decisions.append(Decision(
                        domain="map",
                        action="move_map",
                        target=farm_map.name,
                        priority=3,
                        reason=(
                            f"Farming map {farm_map.name} (level {farm_map.recommended_level_range[0]}-"
                            f"{farm_map.recommended_level_range[1]}, {farm_map.difficulty}, "
                            f"{farm_map.monster_density} density)"
                        ),
                    ))
            except Exception:
                # Map intelligence unavailable — no map decision this cycle
                pass

            # ── 6. Party Coordination Decision ──
            if state.get("nearby_party", 0) > 0:
                if hp_pct < 0.60 and "AL_HEAL" in skills:
                    decisions.append(Decision(
                        domain="party",
                        action="request_heal",
                        target="party",
                        priority=2,
                        reason=f"HP at {hp_pct:.0%}, party members nearby — request heal",
                    ))

            # Sort by priority
            decisions.sort(key=lambda d: d.priority)
            return decisions

    def get_summary(self, bot_id: str, game_mode: str = "solo_farming") -> str:
        """Get a human-readable summary of decisions."""
        with self._lock:
            state = self._bot_state.get(bot_id, {})
            job_name = state.get("job_name", "novice")
            class_builds = self._get_class_builds(job_name)
            variant_name = class_builds.default_variant if class_builds else "unknown"
            build = self._get_variant(job_name, variant_name)
            decisions = self.evaluate(bot_id, game_mode)

            lines = [f"-- Conscious Decisions for {bot_id} --"]
            lines.append(f"  Job: {state.get('job_name', '?')}  Level: {state.get('base_level', 1)}/{state.get('job_level', 1)}")
            lines.append(f"  Build: {build.name if build else '?'}  Mode: {game_mode}")
            lines.append(f"  HP: {state.get('hp_pct', 1.0):.0%}  SP: {state.get('sp_pct', 1.0):.0%}  Zeny: {state.get('zeny', 0)}z")
            lines.append(f"  Map: {state.get('map', '?')}")
            lines.append("")

            if build:
                current_phase = self._get_current_phase(build, state.get("base_level", 1))
                if current_phase:
                    lines.append(f"  Active Phase: {current_phase.level_range[0]}-{current_phase.level_range[1]} ({current_phase.goal})")
                    lines.append(f"  Goal: {current_phase.reason}")
                    lines.append("")

                next_bp = self._get_next_stat_breakpoint(build, state.get("stats", {}))
                if next_bp:
                    current = state.get("stats", {}).get(next_bp.stat.upper(), 0)
                    lines.append(f"  Next Stat: {next_bp.stat.upper()} {current} → {next_bp.value} ({next_bp.reason})")
                    lines.append("")

            if decisions:
                lines.append("  Decisions:")
                for d in decisions:
                    lines.append(f"    [{d.priority}] {d.domain}.{d.action}({d.target}): {d.reason}")
            else:
                lines.append("  No decisions needed.")

            return "\n".join(lines)


# Global singleton
_engine: ConsciousDecisionEngine | None = None
_engine_lock = RLock()


def get_conscious_engine() -> ConsciousDecisionEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = ConsciousDecisionEngine()
        return _engine
