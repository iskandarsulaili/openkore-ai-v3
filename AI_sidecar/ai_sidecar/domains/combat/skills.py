"""Skill registry and rotation engine — skill definitions and rotation management.

Provides:
  - SkillRegistry: maps internal skill IDs to full skill metadata.
  - SkillRotationEngine: manages rotations (list of skills in sequence).
  - Support functions for SP management and cooldown tracking.

Integrates with the existing ro_mechanics.py SKILL_DAMAGE, SKILL_SP_COSTS,
and CLASS_SKILL_TRAINING constants for data consistency.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from ai_sidecar.autonomy.ro_mechanics import SKILL_DAMAGE, SKILL_SP_COSTS

logger = logging.getLogger(__name__)


# ── Skill Metadata ──

@dataclass
class SkillDef:
    """Complete skill definition.

    Mirrors the existing ai_sidecar.combat.skill_rotation.Skill dataclass
    but is isolated in our domain package to avoid circular imports.
    """
    skill_id: str          # Internal RO ID (e.g. "MG_FIREBOLT")
    name: str              # Display name (e.g. "Fire Bolt")
    sp_cost: int           # SP cost at base level
    range: int             # Max range in cells
    cooldown_ms: int       # Cooldown in milliseconds
    is_aoe: bool           # Area of effect flag
    aoe_radius: int = 0    # AoE radius in cells
    cast_time_ms: int = 0  # Cast time in milliseconds
    delay_ms: int = 0      # Post-cast delay in milliseconds
    element: str = "neutral"
    damage_type: str = "melee"  # melee, magic, ranged
    heal_pct: float = 0.0  # Heal as % of MATK (for Heal skill)
    buff_duration_ms: int = 0  # Duration for buffs
    tags: list[str] = field(default_factory=list)


# ── Skill Registry ──

class SkillRegistry:
    """Maps internal skill IDs to SkillDef objects.

    All 40+ commonly used RO skills for pre-renewal.
    Data sourced from rAthena SKILL_DAMAGE and SKILL_SP_COSTS where available.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._skills: dict[str, SkillDef] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load all skill definitions."""
        s = {}  # Build dict, then assign atomically

        # ── Novice Skills ──
        s["NV_BASIC"] = SkillDef("NV_BASIC", "Basic Skill", 0, 1, 0, False, tags=["passive"])
        s["NV_FIRSTAID"] = SkillDef("NV_FIRSTAID", "First Aid", 5, 1, 0, False, heal_pct=0.03, tags=["heal"])

        # ── Swordman Skills ──
        s["SM_BASH"] = SkillDef("SM_BASH", "Bash", 8, 1, 500, False, element="neutral", tags=["physical", "stun"])
        s["SM_RECOVERY"] = SkillDef("SM_RECOVERY", "HP Recovery", 0, 0, 0, False, buff_duration_ms=60000, tags=["buff", "passive"])
        s["SM_MAGNUM"] = SkillDef("SM_MAGNUM", "Magnum Break", 12, 1, 3000, True, aoe_radius=3, element="fire", tags=["physical", "aoe"])
        s["SM_ENDURE"] = SkillDef("SM_ENDURE", "Endure", 10, 0, 30000, False, buff_duration_ms=20000, tags=["buff", "defensive"])
        s["SM_PROVOKE"] = SkillDef("SM_PROVOKE", "Provoke", 3, 9, 5000, False, tags=["taunt"])

        # ── Knight Skills ──
        s["KN_SPEARMASTERY"] = SkillDef("KN_SPEARMASTERY", "Spear Mastery", 0, 0, 0, False, tags=["passive"])
        s["KN_BRANDISHSPEAR"] = SkillDef("KN_BRANDISHSPEAR", "Brandish Spear", 15, 1, 2000, True, aoe_radius=3, element="neutral", tags=["physical", "aoe"])
        s["KN_PIERCE"] = SkillDef("KN_PIERCE", "Pierce", 10, 1, 1000, False, tags=["physical"])
        s["KN_BOWLINGBASH"] = SkillDef("KN_BOWLINGBASH", "Bowling Bash", 15, 1, 2000, True, aoe_radius=3, tags=["physical", "aoe"])
        s["KN_TWOHANDQUICKEN"] = SkillDef("KN_TWOHANDQUICKEN", "Two-Hand Quicken", 15, 0, 30000, False, buff_duration_ms=300000, tags=["buff", "aspd"])
        s["KN_SPEARBOOMERANG"] = SkillDef("KN_SPEARBOOMERANG", "Spear Boomerang", 12, 7, 2000, False, tags=["physical", "ranged"])
        s["KN_AURA"] = SkillDef("KN_AURA", "Aura Blade", 20, 0, 30000, False, buff_duration_ms=120000, element="neutral", tags=["buff"])

        # ── Crusader / Paladin Skills ──
        s["CR_SHIELDBOOMERANG"] = SkillDef("CR_SHIELDBOOMERANG", "Shield Boomerang", 15, 7, 3000, False, tags=["physical", "ranged"])
        s["CR_SHIELDCHARGE"] = SkillDef("CR_SHIELDCHARGE", "Shield Charge", 8, 1, 1000, False, tags=["physical", "stun"])
        s["CR_AUTOGUARD"] = SkillDef("CR_AUTOGUARD", "Auto Guard", 15, 0, 0, False, buff_duration_ms=300000, tags=["buff", "defensive"])
        s["CR_REFLECTSHIELD"] = SkillDef("CR_REFLECTSHIELD", "Reflect Shield", 20, 0, 0, False, buff_duration_ms=300000, tags=["buff", "defensive"])
        s["CR_PROVOCATE"] = SkillDef("CR_PROVOCATE", "Provocatum", 5, 9, 5000, False, tags=["taunt"])
        s["CR_DEVOTION"] = SkillDef("CR_DEVOTION", "Devotion", 20, 0, 0, False, tags=["support", "defensive"])

        # ── Mage Skills ──
        s["MG_SRECOVERY"] = SkillDef("MG_SRECOVERY", "SP Recovery", 0, 0, 0, False, tags=["passive"])
        s["MG_FIREBOLT"] = SkillDef("MG_FIREBOLT", "Fire Bolt", 12, 9, 500, False, element="fire", cast_time_ms=1500, delay_ms=500, tags=["magic", "fire"])
        s["MG_COLD"] = SkillDef("MG_COLD", "Cold Bolt", 12, 9, 500, False, element="water", cast_time_ms=1500, delay_ms=500, tags=["magic", "water"])
        s["MG_LIGHTNING"] = SkillDef("MG_LIGHTNING", "Lightning Bolt", 15, 9, 500, False, element="wind", cast_time_ms=1500, delay_ms=500, tags=["magic", "wind"])
        s["MG_FIREBALL"] = SkillDef("MG_FIREBALL", "Fire Ball", 25, 9, 2000, True, aoe_radius=3, element="fire", cast_time_ms=2000, delay_ms=1000, tags=["magic", "aoe"])
        s["MG_FROSTDIVER"] = SkillDef("MG_FROSTDIVER", "Frost Diver", 12, 9, 500, False, element="water", cast_time_ms=1000, delay_ms=500, tags=["magic", "freeze"])
        s["MG_FROSTNOVA"] = SkillDef("MG_FROSTNOVA", "Frost Nova", 20, 0, 3000, True, aoe_radius=4, element="water", cast_time_ms=2000, delay_ms=1000, tags=["magic", "aoe", "freeze"])
        s["MG_THUNDERSTORM"] = SkillDef("MG_THUNDERSTORM", "Thunderstorm", 35, 9, 3000, True, aoe_radius=5, element="wind", cast_time_ms=3000, delay_ms=1500, tags=["magic", "aoe"])
        s["MG_NAPALMBEAT"] = SkillDef("MG_NAPALMBEAT", "Napalm Beat", 10, 9, 500, False, element="neutral", cast_time_ms=1000, delay_ms=500, tags=["magic", "ghost", "stun"])
        s["MG_SOULSTRIKE"] = SkillDef("MG_SOULSTRIKE", "Soul Strike", 18, 9, 500, False, element="neutral", cast_time_ms=1500, delay_ms=500, tags=["magic", "ghost"])
        s["MG_ENERGYCOAT"] = SkillDef("MG_ENERGYCOAT", "Energy Coat", 15, 0, 0, False, buff_duration_ms=600000, tags=["buff", "defensive"])

        # ── Wizard Skills ──
        s["WZ_FIREPILLAR"] = SkillDef("WZ_FIREPILLAR", "Fire Pillar", 30, 9, 3000, True, aoe_radius=2, element="fire", cast_time_ms=2000, delay_ms=1000, tags=["magic", "aoe", "trap"])
        s["WZ_STORMGUST"] = SkillDef("WZ_STORMGUST", "Storm Gust", 80, 9, 5000, True, aoe_radius=7, element="water", cast_time_ms=5000, delay_ms=3000, tags=["magic", "aoe", "nuke"])
        s["WZ_METEOR"] = SkillDef("WZ_METEOR", "Meteor Storm", 90, 9, 5000, True, aoe_radius=7, element="fire", cast_time_ms=6000, delay_ms=3000, tags=["magic", "aoe", "nuke", "stun"])
        s["WZ_VERMILION"] = SkillDef("WZ_VERMILION", "Lord of Vermilion", 85, 9, 5000, True, aoe_radius=7, element="wind", cast_time_ms=5000, delay_ms=3000, tags=["magic", "aoe", "nuke"])
        s["WZ_HEAVENDRIVE"] = SkillDef("WZ_HEAVENDRIVE", "Heaven's Drive", 45, 9, 5000, True, aoe_radius=5, element="neutral", cast_time_ms=4000, delay_ms=2000, tags=["magic", "aoe"])
        s["WZ_FROSTNOVA"] = SkillDef("WZ_FROSTNOVA", "Frost Nova", 20, 0, 3000, True, aoe_radius=4, element="water", cast_time_ms=2000, delay_ms=1000, tags=["magic", "aoe", "freeze"])
        s["WZ_AMPLIFY"] = SkillDef("WZ_AMPLIFY", "Amplify Magic Power", 40, 0, 30000, False, buff_duration_ms=60000, tags=["buff", "magic_amp"])

        # ── Archer Skills ──
        s["AC_OWL"] = SkillDef("AC_OWL", "Owl's Eye", 0, 0, 0, False, buff_duration_ms=300000, tags=["buff", "passive"])
        s["AC_DOUBLE"] = SkillDef("AC_DOUBLE", "Double Strafe", 12, 9, 200, False, element="neutral", tags=["physical", "ranged"])
        s["AC_SHOWER"] = SkillDef("AC_SHOWER", "Arrow Shower", 15, 9, 2000, True, aoe_radius=3, element="neutral", tags=["physical", "aoe", "knockback"])
        s["AC_CONCENTRATION"] = SkillDef("AC_CONCENTRATION", "Improve Concentration", 20, 0, 30000, False, buff_duration_ms=180000, tags=["buff", "aspd"])

        # ── Hunter Skills ──
        s["HT_BEASTBANE"] = SkillDef("HT_BEASTBANE", "Beast Bane", 0, 0, 0, False, tags=["passive"])
        s["HT_FALCON"] = SkillDef("HT_FALCON", "Falcon", 0, 0, 0, False, tags=["passive"])
        s["HT_STEELCROW"] = SkillDef("HT_STEELCROW", "Steel Crow", 0, 0, 0, False, tags=["passive"])
        s["HT_TRUESIGHT"] = SkillDef("HT_TRUESIGHT", "True Sight", 25, 0, 30000, False, buff_duration_ms=120000, tags=["buff"])
        s["HT_BLITZBEAT"] = SkillDef("HT_BLITZBEAT", "Blitz Beat", 0, 0, 0, True, aoe_radius=3, tags=["passive", "aoe"])  # Falcon auto-proc

        # ── Acolyte Skills ──
        s["AL_HEAL"] = SkillDef("AL_HEAL", "Heal", 15, 9, 500, False, element="holy", cast_time_ms=1000, delay_ms=1000, heal_pct=0.3, tags=["heal", "holy"])
        s["AL_DEMONBANE"] = SkillDef("AL_DEMONBANE", "Demon Bane", 0, 0, 0, False, tags=["passive"])
        s["AL_BLESSING"] = SkillDef("AL_BLESSING", "Blessing", 15, 9, 0, False, buff_duration_ms=300000, tags=["buff", "stat_boost"])
        s["AL_INCAGI"] = SkillDef("AL_INCAGI", "Increase AGI", 15, 9, 0, False, buff_duration_ms=300000, tags=["buff", "aspd"])
        s["AL_ANGELUS"] = SkillDef("AL_ANGELUS", "Angelus", 15, 0, 0, False, buff_duration_ms=300000, tags=["buff", "defensive"])
        s["AL_TELEPORT"] = SkillDef("AL_TELEPORT", "Teleport", 10, 0, 0, False, tags=["utility", "escape"])
        s["AL_HOLYLIGHT"] = SkillDef("AL_HOLYLIGHT", "Holy Light", 15, 9, 500, False, element="holy", cast_time_ms=1500, delay_ms=500, tags=["magic", "holy"])

        # ── Priest Skills ──
        s["PR_TURNUNDEAD"] = SkillDef("PR_TURNUNDEAD", "Turn Undead", 20, 9, 3000, False, element="holy", cast_time_ms=2000, delay_ms=1000, tags=["magic", "holy", "instant_kill"])
        s["PR_SANCTUARY"] = SkillDef("PR_SANCTUARY", "Sanctuary", 30, 9, 5000, True, aoe_radius=3, element="holy", cast_time_ms=4000, delay_ms=2000, heal_pct=0.5, tags=["heal", "aoe", "holy"])
        s["PR_BENEDICTIO"] = SkillDef("PR_BENEDICTIO", "Benedictio", 10, 9, 0, False, buff_duration_ms=300000, element="holy", tags=["buff", "holy"])
        s["PR_SLOWPOISON"] = SkillDef("PR_SLOWPOISON", "Slow Poison", 5, 9, 0, False, tags=["heal", "cure"])
        s["PR_GLORIA"] = SkillDef("PR_GLORIA", "Gloria", 20, 0, 0, False, buff_duration_ms=120000, tags=["buff", "luk"])
        s["PR_MAGNIFICAT"] = SkillDef("PR_MAGNIFICAT", "Magnificat", 20, 0, 0, False, buff_duration_ms=120000, tags=["buff", "sp_regen"])
        s["PR_IMPOSITIO"] = SkillDef("PR_IMPOSITIO", "Impositio Manus", 15, 9, 0, False, buff_duration_ms=300000, tags=["buff", "atk_boost"])
        s["PR_SUFFRAGIUM"] = SkillDef("PR_SUFFRAGIUM", "Suffragium", 15, 9, 0, False, buff_duration_ms=15000, tags=["buff", "fast_cast"])
        s["PR_KYRIE"] = SkillDef("PR_KYRIE", "Kyrie Eleison", 20, 9, 0, False, buff_duration_ms=120000, tags=["buff", "defensive", "absorb"])
        s["PR_ASSUMPTIO"] = SkillDef("PR_ASSUMPTIO", "Assumptio", 30, 0, 0, False, buff_duration_ms=60000, tags=["buff", "defensive"])

        # ── Thief Skills ──
        s["TF_DOUBLE"] = SkillDef("TF_DOUBLE", "Double Attack", 0, 1, 0, False, tags=["passive", "physical"])
        s["TF_HIDING"] = SkillDef("TF_HIDING", "Hiding", 10, 0, 0, False, tags=["utility", "escape"])
        s["TF_MISS"] = SkillDef("TF_MISS", "Improve Dodge", 0, 0, 0, False, tags=["passive", "defensive"])
        s["TF_POISON"] = SkillDef("TF_POISON", "Envenom", 12, 1, 2000, False, element="poison", tags=["physical", "poison", "dot"])

        # ── Assassin Skills ──
        s["AS_RIGHT"] = SkillDef("AS_RIGHT", "Right Hand Mastery", 0, 0, 0, False, tags=["passive"])
        s["AS_LEFT"] = SkillDef("AS_LEFT", "Left Hand Mastery", 0, 0, 0, False, tags=["passive"])
        s["AS_KATAR"] = SkillDef("AS_KATAR", "Katar Mastery", 0, 0, 0, False, tags=["passive"])
        s["AS_SONICBLOW"] = SkillDef("AS_SONICBLOW", "Sonic Blow", 20, 1, 2000, False, element="neutral", tags=["physical", "burst"])
        s["AS_GRIMTOOTH"] = SkillDef("AS_GRIMTOOTH", "Grimtooth", 12, 7, 1000, False, element="neutral", tags=["physical", "ranged"])
        s["AS_VENOMDUST"] = SkillDef("AS_VENOMDUST", "Venom Dust", 15, 1, 3000, True, aoe_radius=3, element="poison", tags=["physical", "aoe", "poison"])
        s["AS_CLOAKING"] = SkillDef("AS_CLOAKING", "Cloaking", 15, 0, 0, False, tags=["utility", "escape"])
        s["AS_ENCHANTPOISON"] = SkillDef("AS_ENCHANTPOISON", "Enchant Poison", 15, 0, 0, False, buff_duration_ms=300000, element="poison", tags=["buff", "element"])

        # ── Merchant Skills ──
        s["MC_DISCOUNT"] = SkillDef("MC_DISCOUNT", "Discount", 0, 0, 0, False, tags=["passive", "economy"])
        s["MC_OVERCHARGE"] = SkillDef("MC_OVERCHARGE", "Overcharge", 0, 0, 0, False, tags=["passive", "economy"])
        s["MC_VENDING"] = SkillDef("MC_VENDING", "Vending", 0, 0, 0, False, tags=["passive", "economy"])
        s["MC_PUSHCART"] = SkillDef("MC_PUSHCART", "Pushcart", 0, 0, 0, False, tags=["passive", "utility"])
        s["MC_MAMMONITE"] = SkillDef("MC_MAMMONITE", "Mammonite", 20, 1, 2000, False, element="neutral", tags=["physical", "burst"])

        # ── Blacksmith Skills ──
        s["BS_HAMMERFALL"] = SkillDef("BS_HAMMERFALL", "Hammerfall", 20, 1, 3000, True, aoe_radius=3, element="earth", tags=["physical", "aoe"])
        s["BS_IRON"] = SkillDef("BS_IRON", "Iron Tempering", 0, 0, 0, False, tags=["passive"])
        s["BS_STEEL"] = SkillDef("BS_STEEL", "Steel Tempering", 0, 0, 0, False, tags=["passive"])
        s["BS_WEAPONPERFECT"] = SkillDef("BS_WEAPONPERFECT", "Weapon Perfection", 15, 0, 0, False, buff_duration_ms=300000, tags=["buff", "size"])
        s["BS_MAXIMIZE"] = SkillDef("BS_MAXIMIZE", "Maximize Power", 20, 0, 30000, False, buff_duration_ms=300000, tags=["buff", "atk_boost"])

        # ── Sage Skills ──
        s["SA_DRAGONOLOGY"] = SkillDef("SA_DRAGONOLOGY", "Dragonology", 0, 0, 0, False, tags=["passive"])
        s["SA_MAGICROD"] = SkillDef("SA_MAGICROD", "Magic Rod", 10, 9, 3000, False, tags=["magic", "absorb"])
        s["SA_CASTCANCEL"] = SkillDef("SA_CASTCANCEL", "Cast Cancel", 5, 0, 0, False, tags=["utility"])
        s["SA_LANDPROTECTOR"] = SkillDef("SA_LANDPROTECTOR", "Land Protector", 40, 9, 60000, True, aoe_radius=5, tags=["magic", "aoe", "defensive"])

        # ── Monk Skills ──
        s["MO_IRONHAND"] = SkillDef("MO_IRONHAND", "Iron Hand", 0, 0, 0, False, tags=["passive"])
        s["MO_SPIRITSRECOVERY"] = SkillDef("MO_SPIRITSRECOVERY", "Spirit's Recovery", 0, 0, 0, False, tags=["passive"])
        s["MO_FINGEROFFENSIVE"] = SkillDef("MO_FINGEROFFENSIVE", "Finger Offensive", 15, 7, 500, False, element="neutral", tags=["physical", "ranged"])
        s["MO_TRIPLEATTACK"] = SkillDef("MO_TRIPLEATTACK", "Triple Attack", 0, 1, 0, False, tags=["passive", "physical"])
        s["MO_STEELBODY"] = SkillDef("MO_STEELBODY", "Steel Body", 30, 0, 30000, False, buff_duration_ms=300000, tags=["buff", "defensive"])
        s["MO_EXTREMITYFIST"] = SkillDef("MO_EXTREMITYFIST", "Extremity Fist", 30, 1, 5000, False, element="neutral", tags=["physical", "burst", "nuke"])

        # ── Bard / Dancer Skills ──
        s["BA_MUSICAL"] = SkillDef("BA_MUSICAL", "Musical Lesson", 0, 0, 0, False, tags=["passive"])
        s["BA_APPLAUSE"] = SkillDef("BA_APPLAUSE", "Lesson of Applause", 0, 0, 0, False, tags=["passive"])
        s["DC_DANCING"] = SkillDef("DC_DANCING", "Dancing Lesson", 0, 0, 0, False, tags=["passive"])

        # ── Rogue Skills ──
        s["RG_SNATCHER"] = SkillDef("RG_SNATCHER", "Snatcher", 0, 0, 0, False, tags=["passive"])
        s["RG_BACKSTAB"] = SkillDef("RG_BACKSTAB", "Back Stab", 10, 1, 2000, False, element="neutral", tags=["physical", "burst"])
        s["RG_STEAL"] = SkillDef("RG_STEAL", "Steal", 10, 1, 2000, False, tags=["utility", "steal"])
        s["RG_STRIPWEAPON"] = SkillDef("RG_STRIPWEAPON", "Strip Weapon", 15, 2, 3000, False, tags=["debuff"])
        s["RG_STRIPSHIELD"] = SkillDef("RG_STRIPSHIELD", "Strip Shield", 15, 2, 3000, False, tags=["debuff"])
        s["RG_STRIPARMOR"] = SkillDef("RG_STRIPARMOR", "Strip Armor", 15, 2, 3000, False, tags=["debuff"])
        s["RG_STRIPHELM"] = SkillDef("RG_STRIPHELM", "Strip Helm", 15, 2, 3000, False, tags=["debuff"])

        # ── Alchemist Skills ──
        s["AM_LEARNING"] = SkillDef("AM_LEARNING", "Learning Potion", 0, 0, 0, False, tags=["passive"])
        s["AM_POTIONRESEARCH"] = SkillDef("AM_POTIONRESEARCH", "Potion Research", 0, 0, 0, False, tags=["passive"])
        s["AM_CALLHOMUN"] = SkillDef("AM_CALLHOMUN", "Call Homunculus", 50, 0, 0, False, tags=["utility"])
        s["AM_DEMONSTRATION"] = SkillDef("AM_DEMONSTRATION", "Demonstration", 25, 9, 5000, True, aoe_radius=3, element="poison", tags=["magic", "aoe", "poison"])
        s["AM_ACIDTERROR"] = SkillDef("AM_ACIDTERROR", "Acid Terror", 30, 9, 3000, False, element="neutral", tags=["magic", "ranged"])

        # ── Soul Linker Skills ──
        s["SL_SOULCOLLECT"] = SkillDef("SL_SOULCOLLECT", "Soul Collect", 10, 0, 0, False, tags=["utility"])
        s["SL_KAINA"] = SkillDef("SL_KAINA", "Kaina", 30, 9, 0, False, buff_duration_ms=300000, element="fire", tags=["buff", "element"])
        s["SL_KAUPE"] = SkillDef("SL_KAUPE", "Kaupe", 30, 9, 0, False, buff_duration_ms=60000, tags=["buff", "defensive"])

        # ── Ninja Skills ──
        s["NJ_KUNAI"] = SkillDef("NJ_KUNAI", "Throw Kunai", 8, 7, 500, False, element="neutral", tags=["physical", "ranged"])
        s["NJ_HUUMA"] = SkillDef("NJ_HUUMA", "Huuma Shuriken", 15, 7, 2000, True, aoe_radius=3, element="neutral", tags=["physical", "aoe"])
        s["NJ_ZENYNAGE"] = SkillDef("NJ_ZENYNAGE", "Throw Zeny", 0, 7, 500, False, element="neutral", tags=["physical", "ranged"])
        s["NJ_KASUMIKIRI"] = SkillDef("NJ_KASUMIKIRI", "Kasumikiri", 25, 1, 3000, False, element="neutral", tags=["physical", "burst"])

        # ── Gunslinger Skills ──
        s["GS_SINGLE"] = SkillDef("GS_SINGLE", "Single Action", 0, 0, 0, False, tags=["passive"])
        s["GS_CHAIN"] = SkillDef("GS_CHAIN", "Chain Action", 0, 0, 0, False, tags=["passive"])
        s["GS_TRACK"] = SkillDef("GS_TRACK", "Tracking", 10, 14, 2000, False, element="neutral", tags=["physical", "ranged"])
        s["GS_DESPERADO"] = SkillDef("GS_DESPERADO", "Desperado", 30, 0, 3000, True, aoe_radius=5, element="neutral", tags=["physical", "aoe"])
        s["GS_GATLINGFEVER"] = SkillDef("GS_GATLINGFEVER", "Gatling Fever", 15, 0, 30000, False, buff_duration_ms=120000, tags=["buff", "aspd"])

        # ── Taekwon / Star Gladiator Skills ──
        s["TK_PUNCH"] = SkillDef("TK_PUNCH", "Punch", 0, 1, 0, False, tags=["physical"])
        s["TK_KICK"] = SkillDef("TK_KICK", "Kick", 5, 1, 500, False, tags=["physical"])
        s["TK_COUNTER"] = SkillDef("TK_COUNTER", "Counter Kick", 8, 1, 2000, False, tags=["physical", "counter"])
        s["SG_FEEL"] = SkillDef("SG_FEEL", "Feel", 30, 0, 0, False, buff_duration_ms=300000, tags=["buff"])
        s["SG_SUNWARM"] = SkillDef("SG_SUNWARM", "Sun Warm", 20, 0, 0, False, buff_duration_ms=300000, tags=["buff"])

        # ── Super Novice ──
        s["SN_BASIC"] = SkillDef("SN_BASIC", "Super Basic", 0, 1, 0, False, tags=["passive"])

        self._skills = s
        logger.info("skill_registry_loaded: %d skills", len(s))

    def get(self, skill_id: str) -> SkillDef | None:
        """Get a skill definition by ID (case-insensitive)."""
        with self._lock:
            return self._skills.get(skill_id.upper())

    def get_by_name(self, name: str) -> SkillDef | None:
        """Get a skill definition by display name (case-insensitive)."""
        name_lower = name.lower()
        with self._lock:
            for skill in self._skills.values():
                if skill.name.lower() == name_lower:
                    return skill
            return None

    def all_skills(self) -> list[SkillDef]:
        """Return all registered skills."""
        with self._lock:
            return list(self._skills.values())

    def skill_ids_for_job(self, job_name: str) -> list[str]:
        """Get likely skill IDs for a given job from CLASS_SKILL_TRAINING.

        Uses the CLASS_SKILL_TRAINING constant in ro_mechanics to map
        job names to their training skill IDs.
        """
        from ai_sidecar.autonomy.ro_mechanics import CLASS_SKILL_TRAINING
        training = CLASS_SKILL_TRAINING.get(job_name.lower(), [])
        return [skill_id for skill_id, _, _ in training]

    def get_sp_cost(self, skill_id: str, level: int = 1) -> int:
        """Get SP cost from ro_mechanics SKILL_SP_COSTS or fallback."""
        upper = skill_id.upper()
        if upper in SKILL_SP_COSTS:
            return SKILL_SP_COSTS[upper]
        skill = self.get(skill_id)
        if skill:
            return skill.sp_cost
        return 10  # Default generic cost

    def is_aoe(self, skill_id: str) -> bool:
        """Check if a skill is AoE."""
        skill = self.get(skill_id)
        return skill.is_aoe if skill else False

    def get_element(self, skill_id: str) -> str:
        """Get the element of a skill."""
        skill = self.get(skill_id)
        return skill.element if skill else "neutral"

    def can_execute(self, skill_id: str, current_sp: int, current_hp: int,
                    cooldowns: dict[str, float] | None = None) -> tuple[bool, str]:
        """Check if a skill can be executed.

        Returns (can_execute, reason).
        """
        skill = self.get(skill_id)
        if not skill:
            return False, f"unknown skill: {skill_id}"

        if current_sp < skill.sp_cost:
            return False, f"not enough SP ({current_sp}/{skill.sp_cost})"

        if cooldowns:
            remaining = cooldowns.get(skill_id.lower(), 0)
            if remaining > 0:
                return False, f"on cooldown ({remaining:.1f}s remaining)"

        return True, "ok"


# ── Rotation Engine ──

@dataclass
class RotationStep:
    """A single step in a skill rotation."""
    skill_id: str
    level: int = 1
    condition: str = ""  # Python expression evaluated in context
    priority: int = 50
    repeat: int = 1  # How many times to repeat this step before moving on


@dataclass
class Rotation:
    """A named skill rotation with situational triggers."""
    name: str
    steps: list[RotationStep] = field(default_factory=list)
    trigger_condition: str = ""
    priority: int = 50


class SkillRotationEngine:
    """Manages skill rotations — ordered lists of skills for different situations.

    Designed for the combat domain: rotations are stateless descriptions that
    tactics modules can use to build their skill execution plan.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._rotations: dict[str, Rotation] = {}
        self._registry = SkillRegistry()
        self._load_default_rotations()

    def _load_default_rotations(self) -> None:
        """Load default rotations for common situations."""
        rotations: dict[str, Rotation] = {}

        # Mage DPS rotation
        rotations["mage_dps"] = Rotation(
            name="mage_dps",
            priority=80,
            steps=[
                RotationStep("MG_FROSTDIVER", 5, priority=90, condition="target.hp_pct > 0.5"),
                RotationStep("MG_FIREBOLT", 10, priority=80),
                RotationStep("MG_COLD", 10, priority=70, condition="target.element in ('fire', 'undead')"),
                RotationStep("MG_LIGHTNING", 10, priority=70, condition="target.element == 'water'"),
            ],
        )

        # Wizard AoE rotation
        rotations["wizard_aoe"] = Rotation(
            name="wizard_aoe",
            priority=90,
            steps=[
                RotationStep("WZ_METEOR", 10, priority=90, condition="aggro >= 4 and sp > 90"),
                RotationStep("WZ_STORMGUST", 10, priority=80, condition="aggro >= 3 and sp > 80"),
                RotationStep("WZ_VERMILION", 10, priority=70, condition="aggro >= 3 and sp > 85"),
                RotationStep("WZ_HEAVENDRIVE", 10, priority=60, condition="aggro >= 3 and sp > 45"),
            ],
        )

        # Archer DPS rotation
        rotations["archer_dps"] = Rotation(
            name="archer_dps",
            priority=80,
            steps=[
                RotationStep("AC_CONCENTRATION", 10, priority=95, condition="not has_buff('concentration')", repeat=0),
                RotationStep("AC_DOUBLE", 10, priority=80, repeat=3),
                RotationStep("AC_SHOWER", 5, priority=70, condition="aggro >= 3"),
            ],
        )

        # Hunter rotation
        rotations["hunter_dps"] = Rotation(
            name="hunter_dps",
            priority=80,
            steps=[
                RotationStep("AC_CONCENTRATION", 10, priority=95, condition="not has_buff('concentration')", repeat=0),
                RotationStep("HT_TRUESIGHT", 5, priority=90, condition="not has_buff('truesight')", repeat=0),
                RotationStep("AC_DOUBLE", 10, priority=80, repeat=3),
                RotationStep("AC_SHOWER", 5, priority=70, condition="aggro >= 3"),
            ],
        )

        # Swordman / Knight DPS rotation
        rotations["knight_dps"] = Rotation(
            name="knight_dps",
            priority=80,
            steps=[
                RotationStep("SM_PROVOKE", 5, priority=95, condition="not has_debuff('provoke')"),
                RotationStep("KN_TWOHANDQUICKEN", 10, priority=90, condition="not has_buff('twohandquicken')"),
                RotationStep("KN_BOWLINGBASH", 10, priority=80, condition="aggro >= 2"),
                RotationStep("SM_BASH", 10, priority=60),
            ],
        )

        # Assassin burst rotation
        rotations["assassin_burst"] = Rotation(
            name="assassin_burst",
            priority=80,
            steps=[
                RotationStep("AS_ENCHANTPOISON", 5, priority=90, condition="not has_buff('enchantpoison')"),
                RotationStep("TF_POISON", 5, priority=85, condition="target.hp_pct > 0.8"),
                RotationStep("AS_SONICBLOW", 10, priority=80, condition="sp > 20"),
                RotationStep("AS_VENOMDUST", 5, priority=70, condition="aggro >= 2"),
            ],
        )

        # Priest support rotation
        rotations["priest_support"] = Rotation(
            name="priest_support",
            priority=90,
            steps=[
                RotationStep("AL_BLESSING", 10, priority=95, condition="not has_buff('blessing')"),
                RotationStep("AL_INCAGI", 10, priority=90, condition="not has_buff('incagi')"),
                RotationStep("PR_GLORIA", 5, priority=85, condition="not has_buff('gloria')"),
                RotationStep("PR_KYRIE", 10, priority=80, condition="not has_buff('kyrie')"),
                RotationStep("AL_HEAL", 10, priority=70, condition="hp_pct < 0.6"),
                RotationStep("AL_HEAL", 5, priority=50, condition="target and target.element == 'undead'"),
                RotationStep("AL_HOLYLIGHT", 5, priority=40),
            ],
        )

        # Tank rotation
        rotations["tank"] = Rotation(
            name="tank",
            priority=85,
            steps=[
                RotationStep("SM_ENDURE", 5, priority=90, condition="not has_buff('endure')"),
                RotationStep("SM_PROVOKE", 5, priority=85, condition="target and not has_debuff('provoke')"),
                RotationStep("SM_MAGNUM", 5, priority=75, condition="aggro >= 3"),
                RotationStep("SM_BASH", 10, priority=60),
            ],
        )

        # Melee basic rotation
        rotations["melee_basic"] = Rotation(
            name="melee_basic",
            priority=50,
            steps=[
                RotationStep("TF_POISON", 5, priority=70, condition="target and target.hp_pct > 0.8"),
                RotationStep("SM_BASH", 5, priority=60),
            ],
        )

        self._rotations = rotations
        logger.info("rotation_engine_loaded: %d rotations", len(rotations))

    def get_rotation(self, name: str) -> Rotation | None:
        with self._lock:
            return self._rotations.get(name)

    def all_rotations(self) -> list[Rotation]:
        with self._lock:
            return list(self._rotations.values())

    def register_rotation(self, rotation: Rotation) -> None:
        with self._lock:
            self._rotations[rotation.name] = rotation

    def find_best_rotation(self, job_class: str, aggro: int = 0, sp_pct: float = 1.0,
                           target_element: str = "") -> Rotation | None:
        """Find the best rotation for the current situation."""
        with self._lock:
            candidates: list[Rotation] = []
            job_lower = job_class.lower()

            # Match by job name in rotation name
            for rot in self._rotations.values():
                if job_lower in rot.name:
                    candidates.append(rot)

            if not candidates:
                return None

            # AoE rotations for high aggro
            if aggro >= 3:
                aoe = [r for r in candidates if "aoe" in r.name]
                if aoe:
                    return aoe[0]

            # Sort by priority descending
            candidates.sort(key=lambda r: -r.priority)
            return candidates[0]

    def get_skill_registry(self) -> SkillRegistry:
        return self._registry


# ── Global Singletons ──

_registry: SkillRegistry | None = None
_registry_lock = RLock()

_rotation_engine: SkillRotationEngine | None = None
_rotation_engine_lock = RLock()


def get_skill_registry() -> SkillRegistry:
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = SkillRegistry()
        return _registry


def get_rotation_engine() -> SkillRotationEngine:
    global _rotation_engine
    with _rotation_engine_lock:
        if _rotation_engine is None:
            _rotation_engine = SkillRotationEngine()
        return _rotation_engine
