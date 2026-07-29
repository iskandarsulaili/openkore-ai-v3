"""Job Registry — defines all 45+ RO classes with tactics mappings, skills, and stat builds.

Each job is a dict with:
  - name: Display name
  - tactics: Which tactics module to use ("tank", "melee_dps", "ranged_dps", "magic_dps", "support", "hybrid")
  - attack_range: Base attack range in cells
  - skills: List of (skill_id, name, sp_cost, level, is_aoe) tuples
  - stat_build: Recommended stat allocation [(stat, target_value), ...]
  - weapon_type: Preferred weapon type
  - description: Role description

All 45+ pre-renewal RO classes included, from Novice through transcendent classes.
"""

from __future__ import annotations

import logging
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


# ── Skill ID constants (from rAthena) ──

NV_BASIC = "NV_BASIC"
NV_FIRSTAID = "NV_FIRSTAID"

SM_BASH = "SM_BASH"
SM_RECOVERY = "SM_RECOVERY"
SM_MAGNUM = "SM_MAGNUM"
SM_ENDURE = "SM_ENDURE"
SM_PROVOKE = "SM_PROVOKE"

KN_SPEARMASTERY = "KN_SPEARMASTERY"
KN_BRANDISHSPEAR = "KN_BRANDISHSPEAR"
KN_PIERCE = "KN_PIERCE"
KN_BOWLINGBASH = "KN_BOWLINGBASH"
KN_TWOHANDQUICKEN = "KN_TWOHANDQUICKEN"
KN_SPEARBOOMERANG = "KN_SPEARBOOMERANG"
KN_AURA = "KN_AURA"

CR_SHIELDBOOMERANG = "CR_SHIELDBOOMERANG"
CR_SHIELDCHARGE = "CR_SHIELDCHARGE"
CR_AUTOGUARD = "CR_AUTOGUARD"
CR_REFLECTSHIELD = "CR_REFLECTSHIELD"
CR_PROVOCATE = "CR_PROVOCATE"
CR_DEVOTION = "CR_DEVOTION"

MG_SRECOVERY = "MG_SRECOVERY"
MG_FIREBOLT = "MG_FIREBOLT"
MG_COLD = "MG_COLD"
MG_LIGHTNING = "MG_LIGHTNING"
MG_FIREBALL = "MG_FIREBALL"
MG_FROSTDIVER = "MG_FROSTDIVER"
MG_FROSTNOVA = "MG_FROSTNOVA"
MG_THUNDERSTORM = "MG_THUNDERSTORM"
MG_NAPALMBEAT = "MG_NAPALMBEAT"
MG_SOULSTRIKE = "MG_SOULSTRIKE"
MG_ENERGYCOAT = "MG_ENERGYCOAT"

WZ_FIREPILLAR = "WZ_FIREPILLAR"
WZ_STORMGUST = "WZ_STORMGUST"
WZ_METEOR = "WZ_METEOR"
WZ_VERMILION = "WZ_VERMILION"
WZ_HEAVENDRIVE = "WZ_HEAVENDRIVE"
WZ_FROSTNOVA = "WZ_FROSTNOVA"
WZ_AMPLIFY = "WZ_AMPLIFY"

AC_OWL = "AC_OWL"
AC_DOUBLE = "AC_DOUBLE"
AC_SHOWER = "AC_SHOWER"
AC_CONCENTRATION = "AC_CONCENTRATION"

HT_BEASTBANE = "HT_BEASTBANE"
HT_FALCON = "HT_FALCON"
HT_STEELCROW = "HT_STEELCROW"
HT_TRUESIGHT = "HT_TRUESIGHT"
HT_BLITZBEAT = "HT_BLITZBEAT"

SN_FALCON = "SN_FALCON"
SN_WARGMASTERY = "SN_WARGMASTERY"
SN_WARGSTRIKE = "SN_WARGSTRIKE"
SN_WINDCURTAIN = "SN_WINDCURTAIN"

AL_HEAL = "AL_HEAL"
AL_DEMONBANE = "AL_DEMONBANE"
AL_BLESSING = "AL_BLESSING"
AL_INCAGI = "AL_INCAGI"
AL_ANGELUS = "AL_ANGELUS"
AL_TELEPORT = "AL_TELEPORT"
AL_HOLYLIGHT = "AL_HOLYLIGHT"

PR_TURNUNDEAD = "PR_TURNUNDEAD"
PR_SANCTUARY = "PR_SANCTUARY"
PR_BENEDICTIO = "PR_BENEDICTIO"
PR_SLOWPOISON = "PR_SLOWPOISON"
PR_GLORIA = "PR_GLORIA"
PR_MAGNIFICAT = "PR_MAGNIFICAT"
PR_IMPOSITIO = "PR_IMPOSITIO"
PR_SUFFRAGIUM = "PR_SUFFRAGIUM"
PR_KYRIE = "PR_KYRIE"
PR_ASSUMPTIO = "PR_ASSUMPTIO"

TF_DOUBLE = "TF_DOUBLE"
TF_HIDING = "TF_HIDING"
TF_MISS = "TF_MISS"
TF_POISON = "TF_POISON"

AS_RIGHT = "AS_RIGHT"
AS_LEFT = "AS_LEFT"
AS_KATAR = "AS_KATAR"
AS_SONICBLOW = "AS_SONICBLOW"
AS_GRIMTOOTH = "AS_GRIMTOOTH"
AS_VENOMDUST = "AS_VENOMDUST"
AS_CLOAKING = "AS_CLOAKING"
AS_ENCHANTPOISON = "AS_ENCHANTPOISON"

MC_DISCOUNT = "MC_DISCOUNT"
MC_OVERCHARGE = "MC_OVERCHARGE"
MC_VENDING = "MC_VENDING"
MC_PUSHCART = "MC_PUSHCART"
MC_MAMMONITE = "MC_MAMMONITE"

BS_HAMMERFALL = "BS_HAMMERFALL"
BS_IRON = "BS_IRON"
BS_STEEL = "BS_STEEL"
BS_WEAPONPERFECT = "BS_WEAPONPERFECT"
BS_MAXIMIZE = "BS_MAXIMIZE"
BS_ENRICH = "BS_ENRICH"

SA_DRAGONOLOGY = "SA_DRAGONOLOGY"
SA_MAGICROD = "SA_MAGICROD"
SA_CASTCANCEL = "SA_CASTCANCEL"
SA_LANDPROTECTOR = "SA_LANDPROTECTOR"

MO_IRONHAND = "MO_IRONHAND"
MO_SPIRITSRECOVERY = "MO_SPIRITSRECOVERY"
MO_FINGEROFFENSIVE = "MO_FINGEROFFENSIVE"
MO_TRIPLEATTACK = "MO_TRIPLEATTACK"
MO_STEELBODY = "MO_STEELBODY"
MO_EXTREMITYFIST = "MO_EXTREMITYFIST"
MO_COMBOFINISH = "MO_COMBOFINISH"

BA_MUSICAL = "BA_MUSICAL"
BA_APPLAUSE = "BA_APPLAUSE"

DC_DANCING = "DC_DANCING"

RG_SNATCHER = "RG_SNATCHER"
RG_BACKSTAB = "RG_BACKSTAB"
RG_STEAL = "RG_STEAL"
RG_STRIPWEAPON = "RG_STRIPWEAPON"
RG_STRIPSHIELD = "RG_STRIPSHIELD"
RG_STRIPARMOR = "RG_STRIPARMOR"
RG_STRIPHELM = "RG_STRIPHELM"

AM_LEARNING = "AM_LEARNING"
AM_POTIONRESEARCH = "AM_POTIONRESEARCH"
AM_CALLHOMUN = "AM_CALLHOMUN"
AM_DEMONSTRATION = "AM_DEMONSTRATION"
AM_ACIDTERROR = "AM_ACIDTERROR"

SL_SOULCOLLECT = "SL_SOULCOLLECT"
SL_KAINA = "SL_KAINA"
SL_KAUPE = "SL_KAUPE"

NJ_KUNAI = "NJ_KUNAI"
NJ_HUUMA = "NJ_HUUMA"
NJ_ZENYNAGE = "NJ_ZENYNAGE"
NJ_KASUMIKIRI = "NJ_KASUMIKIRI"

GS_SINGLE = "GS_SINGLE"
GS_CHAIN = "GS_CHAIN"
GS_TRACK = "GS_TRACK"
GS_DESPERADO = "GS_DESPERADO"
GS_GATLINGFEVER = "GS_GATLINGFEVER"

TK_PUNCH = "TK_PUNCH"
TK_KICK = "TK_KICK"
TK_COUNTER = "TK_COUNTER"

SG_FEEL = "SG_FEEL"
SG_SUNWARM = "SG_SUNWARM"

SN_BASIC = "SN_BASIC"

# ── All Job Definitions ──

ALL_JOBS: dict[str, dict[str, Any]] = {
    # ═══════════════════════════════════════════════════════════════════
    # NOVICE CLASSES
    # ═══════════════════════════════════════════════════════════════════

    "novice": {
        "name": "Novice",
        "tactics": "hybrid",
        "attack_range": 1,
        "weapon_type": "dagger",
        "description": "Starting class — basic attack and First Aid",
        "skills": [
            (NV_BASIC, "Basic Skill", 0, 1, False),
            (NV_FIRSTAID, "First Aid", 5, 1, False),
        ],
        "stat_build": [("str", 10), ("agi", 10), ("dex", 10)],
    },

    "super_novice": {
        "name": "Super Novice",
        "tactics": "hybrid",
        "attack_range": 1,
        "weapon_type": "dagger",
        "description": "Expanded novice — can use most 1st class skills",
        "skills": [
            (NV_BASIC, "Basic Skill", 0, 1, False),
            (NV_FIRSTAID, "First Aid", 5, 1, False),
            (SN_BASIC, "Super Basic", 0, 1, False),
        ],
        "stat_build": [("str", 30), ("agi", 30), ("int", 30), ("dex", 30), ("vit", 20)],
    },

    "high_novice": {
        "name": "High Novice",
        "tactics": "hybrid",
        "attack_range": 1,
        "weapon_type": "dagger",
        "description": "Transcendent novice — higher stat caps",
        "skills": [
            (NV_BASIC, "Basic Skill", 0, 1, False),
            (NV_FIRSTAID, "First Aid", 5, 1, False),
        ],
        "stat_build": [("str", 10), ("agi", 10), ("dex", 10)],
    },

    # ═══════════════════════════════════════════════════════════════════
    # SWORDMAN BRANCH
    # ═══════════════════════════════════════════════════════════════════

    "swordman": {
        "name": "Swordsman",
        "tactics": "tank",
        "attack_range": 1,
        "weapon_type": "sword",
        "description": "Melee tank — Bash, Provoke, Magnum Break, Endure",
        "skills": [
            (SM_BASH, "Bash", 8, 10, False),
            (SM_RECOVERY, "HP Recovery", 0, 1, False),
            (SM_MAGNUM, "Magnum Break", 12, 5, True),
            (SM_ENDURE, "Endure", 10, 5, False),
            (SM_PROVOKE, "Provoke", 3, 10, False),
        ],
        "stat_build": [("str", 40), ("vit", 30), ("dex", 20)],
    },

    "knight": {
        "name": "Knight",
        "tactics": "tank",
        "attack_range": 1,
        "weapon_type": "spear",
        "description": "Advanced tank — Bowling Bash, Brandish Spear, Pierce, Two-Hand Quicken",
        "skills": [
            (SM_BASH, "Bash", 8, 10, False),
            (SM_MAGNUM, "Magnum Break", 12, 5, True),
            (SM_ENDURE, "Endure", 10, 5, False),
            (SM_PROVOKE, "Provoke", 3, 10, False),
            (KN_SPEARMASTERY, "Spear Mastery", 0, 10, False),
            (KN_BRANDISHSPEAR, "Brandish Spear", 15, 10, True),
            (KN_PIERCE, "Pierce", 10, 10, False),
            (KN_BOWLINGBASH, "Bowling Bash", 15, 10, True),
            (KN_TWOHANDQUICKEN, "Two-Hand Quicken", 15, 10, False),
            (KN_SPEARBOOMERANG, "Spear Boomerang", 12, 5, False),
            (KN_AURA, "Aura Blade", 20, 5, False),
        ],
        "stat_build": [("str", 60), ("vit", 40), ("dex", 30)],
    },

    "crusader": {
        "name": "Crusader",
        "tactics": "tank",
        "attack_range": 1,
        "weapon_type": "spear",
        "description": "Holy tank — Shield skills, Auto Guard, Reflect Shield, Devotion",
        "skills": [
            (SM_BASH, "Bash", 8, 10, False),
            (SM_ENDURE, "Endure", 10, 5, False),
            (SM_PROVOKE, "Provoke", 3, 10, False),
            (CR_SHIELDBOOMERANG, "Shield Boomerang", 15, 5, False),
            (CR_SHIELDCHARGE, "Shield Charge", 8, 5, False),
            (CR_AUTOGUARD, "Auto Guard", 15, 10, False),
            (CR_REFLECTSHIELD, "Reflect Shield", 20, 10, False),
            (CR_PROVOCATE, "Provocatum", 5, 5, False),
            (CR_DEVOTION, "Devotion", 20, 5, False),
        ],
        "stat_build": [("vit", 60), ("str", 40), ("dex", 20)],
    },

    "lord_knight": {
        "name": "Lord Knight",
        "tactics": "tank",
        "attack_range": 1,
        "weapon_type": "two_hand_sword",
        "description": "Transcendent knight — Fury, Concentration, advanced AoE",
        "skills": [
            (SM_BASH, "Bash", 8, 10, False),
            (SM_MAGNUM, "Magnum Break", 12, 5, True),
            (SM_ENDURE, "Endure", 10, 5, False),
            (SM_PROVOKE, "Provoke", 3, 10, False),
            (KN_BOWLINGBASH, "Bowling Bash", 15, 10, True),
            (KN_TWOHANDQUICKEN, "Two-Hand Quicken", 15, 10, False),
            (KN_AURA, "Aura Blade", 20, 10, False),
        ],
        "stat_build": [("str", 80), ("vit", 50), ("dex", 40)],
    },

    "paladin": {
        "name": "Paladin",
        "tactics": "tank",
        "attack_range": 1,
        "weapon_type": "spear",
        "description": "Transcendent crusader — holy tank with advanced defensive skills",
        "skills": [
            (SM_ENDURE, "Endure", 10, 5, False),
            (CR_AUTOGUARD, "Auto Guard", 15, 10, False),
            (CR_REFLECTSHIELD, "Reflect Shield", 20, 10, False),
            (CR_DEVOTION, "Devotion", 20, 5, False),
        ],
        "stat_build": [("vit", 80), ("str", 50), ("dex", 30)],
    },

    # ═══════════════════════════════════════════════════════════════════
    # MAGE BRANCH
    # ═══════════════════════════════════════════════════════════════════

    "mage": {
        "name": "Mage",
        "tactics": "magic_dps",
        "attack_range": 9,
        "weapon_type": "staff",
        "description": "Elemental caster — Fire/Cold/Lightning Bolt, Frost Diver, Napalm Beat",
        "skills": [
            (MG_SRECOVERY, "SP Recovery", 0, 10, False),
            (MG_FIREBOLT, "Fire Bolt", 12, 10, False),
            (MG_COLD, "Cold Bolt", 12, 10, False),
            (MG_LIGHTNING, "Lightning Bolt", 15, 10, False),
            (MG_FIREBALL, "Fire Ball", 25, 5, True),
            (MG_FROSTDIVER, "Frost Diver", 12, 5, False),
            (MG_NAPALMBEAT, "Napalm Beat", 10, 5, False),
            (MG_SOULSTRIKE, "Soul Strike", 18, 5, False),
            (MG_ENERGYCOAT, "Energy Coat", 15, 5, False),
        ],
        "stat_build": [("int", 70), ("dex", 30)],
    },

    "wizard": {
        "name": "Wizard",
        "tactics": "magic_dps",
        "attack_range": 9,
        "weapon_type": "staff",
        "description": "Master caster — Storm Gust, Meteor Storm, Lord of Vermilion, Heaven's Drive",
        "skills": [
            (MG_SRECOVERY, "SP Recovery", 0, 10, False),
            (MG_FIREBOLT, "Fire Bolt", 12, 10, False),
            (MG_COLD, "Cold Bolt", 12, 10, False),
            (MG_LIGHTNING, "Lightning Bolt", 15, 10, False),
            (MG_FIREBALL, "Fire Ball", 25, 5, True),
            (MG_FROSTDIVER, "Frost Diver", 12, 5, False),
            (MG_NAPALMBEAT, "Napalm Beat", 10, 5, False),
            (MG_SOULSTRIKE, "Soul Strike", 18, 5, False),
            (MG_ENERGYCOAT, "Energy Coat", 15, 5, False),
            (WZ_FIREPILLAR, "Fire Pillar", 30, 5, True),
            (WZ_STORMGUST, "Storm Gust", 80, 10, True),
            (WZ_METEOR, "Meteor Storm", 90, 10, True),
            (WZ_VERMILION, "Lord of Vermilion", 85, 10, True),
            (WZ_HEAVENDRIVE, "Heaven's Drive", 45, 10, True),
            (WZ_AMPLIFY, "Amplify Magic Power", 40, 5, False),
        ],
        "stat_build": [("int", 90), ("dex", 40)],
    },

    "sage": {
        "name": "Sage",
        "tactics": "magic_dps",
        "attack_range": 9,
        "weapon_type": "staff",
        "description": "Elemental scholar — element endow, dispel, Land Protector",
        "skills": [
            (MG_FIREBOLT, "Fire Bolt", 12, 10, False),
            (MG_COLD, "Cold Bolt", 12, 10, False),
            (MG_LIGHTNING, "Lightning Bolt", 15, 10, False),
            (SA_DRAGONOLOGY, "Dragonology", 0, 1, False),
            (SA_MAGICROD, "Magic Rod", 10, 5, False),
            (SA_CASTCANCEL, "Cast Cancel", 5, 1, False),
            (SA_LANDPROTECTOR, "Land Protector", 40, 5, True),
        ],
        "stat_build": [("int", 70), ("dex", 40)],
    },

    "high_wizard": {
        "name": "High Wizard",
        "tactics": "magic_dps",
        "attack_range": 9,
        "weapon_type": "staff",
        "description": "Transcendent wizard — enhanced AoE and element spells",
        "skills": [
            (MG_FIREBOLT, "Fire Bolt", 12, 10, False),
            (MG_COLD, "Cold Bolt", 12, 10, False),
            (MG_LIGHTNING, "Lightning Bolt", 15, 10, False),
            (WZ_STORMGUST, "Storm Gust", 80, 10, True),
            (WZ_METEOR, "Meteor Storm", 90, 10, True),
            (WZ_VERMILION, "Lord of Vermilion", 85, 10, True),
            (WZ_HEAVENDRIVE, "Heaven's Drive", 45, 10, True),
            (WZ_AMPLIFY, "Amplify Magic Power", 40, 5, False),
        ],
        "stat_build": [("int", 99), ("dex", 50)],
    },

    "professor": {
        "name": "Professor",
        "tactics": "magic_dps",
        "attack_range": 9,
        "weapon_type": "staff",
        "description": "Transcendent sage — advanced elemental manipulation",
        "skills": [
            (MG_FIREBOLT, "Fire Bolt", 12, 10, False),
            (SA_LANDPROTECTOR, "Land Protector", 40, 5, True),
        ],
        "stat_build": [("int", 90), ("dex", 40)],
    },

    # ═══════════════════════════════════════════════════════════════════
    # ARCHER BRANCH
    # ═══════════════════════════════════════════════════════════════════

    "archer": {
        "name": "Archer",
        "tactics": "ranged_dps",
        "attack_range": 9,
        "weapon_type": "bow",
        "description": "Ranged DPS — Double Strafe, Arrow Shower, Owl's Eye",
        "skills": [
            (AC_OWL, "Owl's Eye", 0, 10, False),
            (AC_DOUBLE, "Double Strafe", 12, 10, False),
            (AC_SHOWER, "Arrow Shower", 15, 5, True),
            (AC_CONCENTRATION, "Improve Concentration", 20, 10, False),
        ],
        "stat_build": [("dex", 50), ("agi", 30), ("luk", 20)],
    },

    "hunter": {
        "name": "Hunter",
        "tactics": "ranged_dps",
        "attack_range": 9,
        "weapon_type": "bow",
        "description": "Ranged DPS with falcon — Blitz Beat, Steel Crow, Beast Bane, trap support",
        "skills": [
            (AC_OWL, "Owl's Eye", 0, 10, False),
            (AC_DOUBLE, "Double Strafe", 12, 10, False),
            (AC_SHOWER, "Arrow Shower", 15, 5, True),
            (AC_CONCENTRATION, "Improve Concentration", 20, 10, False),
            (HT_BEASTBANE, "Beast Bane", 0, 5, False),
            (HT_FALCON, "Falcon", 0, 1, False),
            (HT_STEELCROW, "Steel Crow", 0, 5, False),
            (HT_TRUESIGHT, "True Sight", 25, 5, False),
            (HT_BLITZBEAT, "Blitz Beat", 0, 5, True),
        ],
        "stat_build": [("dex", 70), ("agi", 50), ("luk", 30)],
    },

    "sniper": {
        "name": "Sniper",
        "tactics": "ranged_dps",
        "attack_range": 14,
        "weapon_type": "bow",
        "description": "Transcendent hunter — extended range, Falcon Assault, Wind Curtain",
        "skills": [
            (AC_DOUBLE, "Double Strafe", 12, 10, False),
            (AC_SHOWER, "Arrow Shower", 15, 5, True),
            (AC_CONCENTRATION, "Improve Concentration", 20, 10, False),
            (HT_BLITZBEAT, "Blitz Beat", 0, 5, True),
            (SN_WARGMASTERY, "Warg Mastery", 0, 5, False),
            (SN_WARGSTRIKE, "Warg Strike", 0, 5, False),
            (SN_WINDCURTAIN, "Wind Curtain", 0, 5, False),
        ],
        "stat_build": [("dex", 99), ("agi", 60), ("luk", 40)],
    },

    "bard": {
        "name": "Bard",
        "tactics": "hybrid",
        "attack_range": 9,
        "weapon_type": "instrument",
        "description": "Support/DPS hybrid — music buffs, area songs, Arrow Shower",
        "skills": [
            (AC_DOUBLE, "Double Strafe", 12, 10, False),
            (AC_SHOWER, "Arrow Shower", 15, 5, True),
            (BA_MUSICAL, "Musical Lesson", 0, 10, False),
            (BA_APPLAUSE, "Lesson of Applause", 0, 1, False),
        ],
        "stat_build": [("dex", 50), ("int", 40), ("agi", 20)],
    },

    "dancer": {
        "name": "Dancer",
        "tactics": "hybrid",
        "attack_range": 9,
        "weapon_type": "whip",
        "description": "Support/DPS hybrid — dance buffs, Arrow Shower, area effects",
        "skills": [
            (AC_DOUBLE, "Double Strafe", 12, 10, False),
            (AC_SHOWER, "Arrow Shower", 15, 5, True),
            (DC_DANCING, "Dancing Lesson", 0, 10, False),
        ],
        "stat_build": [("dex", 50), ("int", 40), ("agi", 20)],
    },

    "clown": {
        "name": "Clown",
        "tactics": "hybrid",
        "attack_range": 9,
        "weapon_type": "instrument",
        "description": "Transcendent bard — advanced song and jester skills",
        "skills": [
            (AC_DOUBLE, "Double Strafe", 12, 10, False),
            (BA_MUSICAL, "Musical Lesson", 0, 10, False),
        ],
        "stat_build": [("dex", 60), ("int", 50)],
    },

    "gypsy": {
        "name": "Gypsy",
        "tactics": "hybrid",
        "attack_range": 9,
        "weapon_type": "whip",
        "description": "Transcendent dancer — advanced dance skills",
        "skills": [
            (AC_DOUBLE, "Double Strafe", 12, 10, False),
            (DC_DANCING, "Dancing Lesson", 0, 10, False),
        ],
        "stat_build": [("dex", 60), ("int", 50)],
    },

    # ═══════════════════════════════════════════════════════════════════
    # ACOLYTE BRANCH
    # ═══════════════════════════════════════════════════════════════════

    "acolyte": {
        "name": "Acolyte",
        "tactics": "support",
        "attack_range": 1,
        "weapon_type": "mace",
        "description": "Healer/support — Heal, Blessing, Increase AGI, Holy Light vs undead",
        "skills": [
            (AL_HEAL, "Heal", 15, 10, False),
            (AL_DEMONBANE, "Demon Bane", 0, 10, False),
            (AL_BLESSING, "Blessing", 15, 10, False),
            (AL_INCAGI, "Increase AGI", 15, 10, False),
            (AL_ANGELUS, "Angelus", 15, 5, False),
            (AL_TELEPORT, "Teleport", 10, 2, False),
            (AL_HOLYLIGHT, "Holy Light", 15, 5, False),
        ],
        "stat_build": [("int", 50), ("dex", 20), ("vit", 10)],
    },

    "priest": {
        "name": "Priest",
        "tactics": "support",
        "attack_range": 1,
        "weapon_type": "mace",
        "description": "Master healer — Turn Undead, Sanctuary, Kyrie Eleison, Assumptio, full buffs",
        "skills": [
            (AL_HEAL, "Heal", 15, 10, False),
            (AL_DEMONBANE, "Demon Bane", 0, 10, False),
            (AL_BLESSING, "Blessing", 15, 10, False),
            (AL_INCAGI, "Increase AGI", 15, 10, False),
            (AL_ANGELUS, "Angelus", 15, 5, False),
            (AL_TELEPORT, "Teleport", 10, 2, False),
            (AL_HOLYLIGHT, "Holy Light", 15, 5, False),
            (PR_TURNUNDEAD, "Turn Undead", 20, 10, False),
            (PR_SANCTUARY, "Sanctuary", 30, 5, True),
            (PR_BENEDICTIO, "Benedictio", 10, 5, False),
            (PR_SLOWPOISON, "Slow Poison", 5, 1, False),
            (PR_GLORIA, "Gloria", 20, 5, False),
            (PR_MAGNIFICAT, "Magnificat", 20, 5, False),
            (PR_IMPOSITIO, "Impositio Manus", 15, 5, False),
            (PR_SUFFRAGIUM, "Suffragium", 15, 5, False),
            (PR_KYRIE, "Kyrie Eleison", 20, 10, False),
            (PR_ASSUMPTIO, "Assumptio", 30, 5, False),
        ],
        "stat_build": [("int", 80), ("dex", 40), ("vit", 30)],
    },

    "monk": {
        "name": "Monk",
        "tactics": "melee_dps",
        "attack_range": 1,
        "weapon_type": "knuckle",
        "description": "Melee burst DPS — Triple Attack, Finger Offensive, Extremity Fist, Steel Body",
        "skills": [
            (AL_HEAL, "Heal", 15, 10, False),
            (AL_BLESSING, "Blessing", 15, 10, False),
            (AL_INCAGI, "Increase AGI", 15, 10, False),
            (AL_TELEPORT, "Teleport", 10, 2, False),
            (MO_IRONHAND, "Iron Hand", 0, 10, False),
            (MO_SPIRITSRECOVERY, "Spirit's Recovery", 0, 10, False),
            (MO_FINGEROFFENSIVE, "Finger Offensive", 15, 5, False),
            (MO_TRIPLEATTACK, "Triple Attack", 0, 10, False),
            (MO_STEELBODY, "Steel Body", 30, 5, False),
            (MO_EXTREMITYFIST, "Extremity Fist", 30, 5, False),
        ],
        "stat_build": [("str", 60), ("int", 30), ("dex", 30)],
    },

    "high_priest": {
        "name": "High Priest",
        "tactics": "support",
        "attack_range": 1,
        "weapon_type": "mace",
        "description": "Transcendent priest — enhanced healing, Basilica, Meditatio",
        "skills": [
            (AL_HEAL, "Heal", 15, 10, False),
            (AL_BLESSING, "Blessing", 15, 10, False),
            (AL_INCAGI, "Increase AGI", 15, 10, False),
            (PR_KYRIE, "Kyrie Eleison", 20, 10, False),
            (PR_ASSUMPTIO, "Assumptio", 30, 5, False),
            (PR_SANCTUARY, "Sanctuary", 30, 5, True),
        ],
        "stat_build": [("int", 99), ("dex", 40), ("vit", 30)],
    },

    "champion": {
        "name": "Champion",
        "tactics": "melee_dps",
        "attack_range": 1,
        "weapon_type": "knuckle",
        "description": "Transcendent monk — Asura Strike, Occult Impaction, combo finishers",
        "skills": [
            (MO_FINGEROFFENSIVE, "Finger Offensive", 15, 5, False),
            (MO_TRIPLEATTACK, "Triple Attack", 0, 10, False),
            (MO_STEELBODY, "Steel Body", 30, 5, False),
            (MO_EXTREMITYFIST, "Extremity Fist", 30, 5, False),
        ],
        "stat_build": [("str", 90), ("int", 40), ("dex", 30)],
    },

    # ═══════════════════════════════════════════════════════════════════
    # MERCHANT BRANCH
    # ═══════════════════════════════════════════════════════════════════

    "merchant": {
        "name": "Merchant",
        "tactics": "melee_dps",
        "attack_range": 1,
        "weapon_type": "sword",
        "description": "Economy/melee — Discount, Overcharge, Vending, Pushcart, Mammonite",
        "skills": [
            (MC_DISCOUNT, "Discount", 0, 10, False),
            (MC_OVERCHARGE, "Overcharge", 0, 10, False),
            (MC_VENDING, "Vending", 0, 1, False),
            (MC_PUSHCART, "Pushcart", 0, 5, False),
            (MC_MAMMONITE, "Mammonite", 20, 10, False),
        ],
        "stat_build": [("str", 50), ("vit", 30), ("dex", 10)],
    },

    "blacksmith": {
        "name": "Blacksmith",
        "tactics": "melee_dps",
        "attack_range": 1,
        "weapon_type": "mace",
        "description": "Melee DPS/crafter — Hammerfall, Weapon Perfection, Maximize Power",
        "skills": [
            (MC_MAMMONITE, "Mammonite", 20, 10, False),
            (BS_HAMMERFALL, "Hammerfall", 20, 10, True),
            (BS_IRON, "Iron Tempering", 0, 5, False),
            (BS_STEEL, "Steel Tempering", 0, 5, False),
            (BS_WEAPONPERFECT, "Weapon Perfection", 15, 5, False),
            (BS_MAXIMIZE, "Maximize Power", 20, 5, False),
        ],
        "stat_build": [("str", 70), ("vit", 40), ("dex", 20)],
    },

    "alchemist": {
        "name": "Alchemist",
        "tactics": "hybrid",
        "attack_range": 9,
        "weapon_type": "staff",
        "description": "Potion/creation hybrid — Acid Terror, Demonstration, Homunculus",
        "skills": [
            (AM_LEARNING, "Learning Potion", 0, 10, False),
            (AM_POTIONRESEARCH, "Potion Research", 0, 10, False),
            (AM_CALLHOMUN, "Call Homunculus", 50, 1, False),
            (AM_DEMONSTRATION, "Demonstration", 25, 5, True),
            (AM_ACIDTERROR, "Acid Terror", 30, 5, False),
        ],
        "stat_build": [("int", 60), ("str", 30), ("dex", 20)],
    },

    "whitesmith": {
        "name": "Whitesmith",
        "tactics": "melee_dps",
        "attack_range": 1,
        "weapon_type": "mace",
        "description": "Transcendent blacksmith — enhanced crafting and melee DPS",
        "skills": [
            (BS_HAMMERFALL, "Hammerfall", 20, 10, True),
            (BS_MAXIMIZE, "Maximize Power", 20, 5, False),
        ],
        "stat_build": [("str", 90), ("vit", 50), ("dex", 30)],
    },

    "creator": {
        "name": "Creator",
        "tactics": "hybrid",
        "attack_range": 9,
        "weapon_type": "staff",
        "description": "Transcendent alchemist — Acid Bomb, Hell Plant, advanced potions",
        "skills": [
            (AM_ACIDTERROR, "Acid Terror", 30, 5, False),
            (AM_DEMONSTRATION, "Demonstration", 25, 5, True),
        ],
        "stat_build": [("int", 80), ("dex", 30)],
    },

    # ═══════════════════════════════════════════════════════════════════
    # THIEF BRANCH
    # ═══════════════════════════════════════════════════════════════════

    "thief": {
        "name": "Thief",
        "tactics": "melee_dps",
        "attack_range": 1,
        "weapon_type": "dagger",
        "description": "Melee DPS — Double Attack, Hiding, Improve Dodge, Envenom",
        "skills": [
            (TF_DOUBLE, "Double Attack", 0, 10, False),
            (TF_HIDING, "Hiding", 10, 5, False),
            (TF_MISS, "Improve Dodge", 0, 5, False),
            (TF_POISON, "Envenom", 12, 5, False),
        ],
        "stat_build": [("agi", 50), ("dex", 20), ("str", 20)],
    },

    "assassin": {
        "name": "Assassin",
        "tactics": "melee_dps",
        "attack_range": 1,
        "weapon_type": "katar",
        "description": "Melee burst DPS — Sonic Blow, Grimtooth, Venom Dust, Cloaking, dual-wield",
        "skills": [
            (TF_DOUBLE, "Double Attack", 0, 10, False),
            (TF_HIDING, "Hiding", 10, 5, False),
            (TF_MISS, "Improve Dodge", 0, 5, False),
            (TF_POISON, "Envenom", 12, 5, False),
            (AS_RIGHT, "Right Hand Mastery", 0, 10, False),
            (AS_LEFT, "Left Hand Mastery", 0, 10, False),
            (AS_KATAR, "Katar Mastery", 0, 10, False),
            (AS_SONICBLOW, "Sonic Blow", 20, 10, False),
            (AS_GRIMTOOTH, "Grimtooth", 12, 5, False),
            (AS_VENOMDUST, "Venom Dust", 15, 5, True),
            (AS_CLOAKING, "Cloaking", 15, 5, False),
            (AS_ENCHANTPOISON, "Enchant Poison", 15, 5, False),
        ],
        "stat_build": [("agi", 70), ("str", 40), ("dex", 30)],
    },

    "rogue": {
        "name": "Rogue",
        "tactics": "melee_dps",
        "attack_range": 1,
        "weapon_type": "dagger",
        "description": "Melee DPS/utility — Back Stab, Steal, Strip skills, Snatcher",
        "skills": [
            (TF_DOUBLE, "Double Attack", 0, 10, False),
            (TF_HIDING, "Hiding", 10, 5, False),
            (TF_MISS, "Improve Dodge", 0, 5, False),
            (RG_SNATCHER, "Snatcher", 0, 10, False),
            (RG_BACKSTAB, "Back Stab", 10, 10, False),
            (RG_STEAL, "Steal", 10, 10, False),
            (RG_STRIPWEAPON, "Strip Weapon", 15, 5, False),
            (RG_STRIPSHIELD, "Strip Shield", 15, 5, False),
            (RG_STRIPARMOR, "Strip Armor", 15, 5, False),
            (RG_STRIPHELM, "Strip Helm", 15, 5, False),
        ],
        "stat_build": [("agi", 60), ("dex", 30), ("str", 30)],
    },

    "assassin_cross": {
        "name": "Assassin Cross",
        "tactics": "melee_dps",
        "attack_range": 1,
        "weapon_type": "katar",
        "description": "Transcendent assassin — Meteor Assault, Soul Destroyer, Enchant Deadly Poison",
        "skills": [
            (AS_SONICBLOW, "Sonic Blow", 20, 10, False),
            (AS_VENOMDUST, "Venom Dust", 15, 5, True),
            (AS_ENCHANTPOISON, "Enchant Poison", 15, 5, False),
        ],
        "stat_build": [("agi", 90), ("str", 50), ("dex", 40)],
    },

    "stalker": {
        "name": "Stalker",
        "tactics": "melee_dps",
        "attack_range": 1,
        "weapon_type": "dagger",
        "description": "Transcendent rogue — Full Strip, Preserve, Divest skills",
        "skills": [
            (RG_BACKSTAB, "Back Stab", 10, 10, False),
            (RG_STRIPWEAPON, "Strip Weapon", 15, 5, False),
            (RG_STRIPSHIELD, "Strip Shield", 15, 5, False),
            (RG_STRIPARMOR, "Strip Armor", 15, 5, False),
        ],
        "stat_build": [("agi", 80), ("dex", 40), ("str", 40)],
    },

    # ═══════════════════════════════════════════════════════════════════
    # EXTENDED / EXPANSION CLASSES
    # ═══════════════════════════════════════════════════════════════════

    "taekwon": {
        "name": "Taekwon",
        "tactics": "melee_dps",
        "attack_range": 1,
        "weapon_type": "knuckle",
        "description": "Martial artist — kick-focused melee, no weapons needed",
        "skills": [
            (TK_PUNCH, "Punch", 0, 10, False),
            (TK_KICK, "Kick", 5, 10, False),
            (TK_COUNTER, "Counter Kick", 8, 5, False),
        ],
        "stat_build": [("str", 40), ("agi", 30), ("dex", 20)],
    },

    "star_gladiator": {
        "name": "Star Gladiator",
        "tactics": "melee_dps",
        "attack_range": 1,
        "weapon_type": "knuckle",
        "description": "Star-powered melee — feel skill, star warm, map-specific buffs",
        "skills": [
            (TK_KICK, "Kick", 5, 10, False),
            (SG_FEEL, "Feel", 30, 5, False),
            (SG_SUNWARM, "Sun Warm", 20, 5, False),
        ],
        "stat_build": [("str", 60), ("agi", 40), ("dex", 20)],
    },

    "soul_linker": {
        "name": "Soul Linker",
        "tactics": "hybrid",
        "attack_range": 9,
        "weapon_type": "staff",
        "description": "Spirit support — links souls, elemental buffs for allies",
        "skills": [
            (SL_SOULCOLLECT, "Soul Collect", 10, 10, False),
            (SL_KAINA, "Kaina", 30, 5, False),
            (SL_KAUPE, "Kaupe", 30, 5, False),
        ],
        "stat_build": [("int", 50), ("dex", 20)],
    },

    "gunslinger": {
        "name": "Gunslinger",
        "tactics": "ranged_dps",
        "attack_range": 14,
        "weapon_type": "grenade",
        "description": "Ranged firearm DPS — Tracking, Desperado, Gatling Fever, Single/Chain Action",
        "skills": [
            (GS_SINGLE, "Single Action", 0, 10, False),
            (GS_CHAIN, "Chain Action", 0, 10, False),
            (GS_TRACK, "Tracking", 10, 10, False),
            (GS_DESPERADO, "Desperado", 30, 10, True),
            (GS_GATLINGFEVER, "Gatling Fever", 15, 5, False),
        ],
        "stat_build": [("dex", 70), ("agi", 40)],
    },

    "ninja": {
        "name": "Ninja",
        "tactics": "hybrid",
        "attack_range": 7,
        "weapon_type": "shuriken",
        "description": "Versatile ninja — Kunai, Huuma Shuriken, Throw Zeny, stealth attacks",
        "skills": [
            (NJ_KUNAI, "Throw Kunai", 8, 10, False),
            (NJ_HUUMA, "Huuma Shuriken", 15, 5, True),
            (NJ_ZENYNAGE, "Throw Zeny", 0, 10, False),
            (NJ_KASUMIKIRI, "Kasumikiri", 25, 5, False),
        ],
        "stat_build": [("dex", 60), ("str", 30), ("agi", 20)],
    },

    "rebel": {
        "name": "Rebel",
        "tactics": "ranged_dps",
        "attack_range": 14,
        "weapon_type": "grenade",
        "description": "Transcendent gunslinger — Crimson Mark, Shoot, enhanced firearm skills",
        "skills": [
            (GS_TRACK, "Tracking", 10, 10, False),
            (GS_DESPERADO, "Desperado", 30, 10, True),
        ],
        "stat_build": [("dex", 90), ("agi", 50)],
    },

    # ═══════════════════════════════════════════════════════════════════
    # TRANSCENDENT SECOND CLASSES (2-2 variants)
    # ═══════════════════════════════════════════════════════════════════

    "high_swordman": {
        "name": "High Swordsman",
        "tactics": "tank",
        "attack_range": 1,
        "weapon_type": "sword",
        "description": "Transcendent swordsman — higher stat caps, pre-2-1",
        "skills": [
            (SM_BASH, "Bash", 8, 10, False),
            (SM_MAGNUM, "Magnum Break", 12, 5, True),
            (SM_PROVOKE, "Provoke", 3, 10, False),
            (SM_ENDURE, "Endure", 10, 5, False),
        ],
        "stat_build": [("str", 50), ("vit", 40), ("dex", 30)],
    },

    "high_mage": {
        "name": "High Mage",
        "tactics": "magic_dps",
        "attack_range": 9,
        "weapon_type": "staff",
        "description": "Transcendent mage — pre-wizard transcendent",
        "skills": [
            (MG_FIREBOLT, "Fire Bolt", 12, 10, False),
            (MG_COLD, "Cold Bolt", 12, 10, False),
            (MG_LIGHTNING, "Lightning Bolt", 15, 10, False),
            (MG_FROSTDIVER, "Frost Diver", 12, 5, False),
        ],
        "stat_build": [("int", 80), ("dex", 30)],
    },

    "high_archer": {
        "name": "High Archer",
        "tactics": "ranged_dps",
        "attack_range": 9,
        "weapon_type": "bow",
        "description": "Transcendent archer — pre-hunter transcendent",
        "skills": [
            (AC_DOUBLE, "Double Strafe", 12, 10, False),
            (AC_SHOWER, "Arrow Shower", 15, 5, True),
            (AC_CONCENTRATION, "Improve Concentration", 20, 10, False),
        ],
        "stat_build": [("dex", 60), ("agi", 40)],
    },

    "high_acolyte": {
        "name": "High Acolyte",
        "tactics": "support",
        "attack_range": 1,
        "weapon_type": "mace",
        "description": "Transcendent acolyte — pre-priest transcendent",
        "skills": [
            (AL_HEAL, "Heal", 15, 10, False),
            (AL_BLESSING, "Blessing", 15, 10, False),
            (AL_INCAGI, "Increase AGI", 15, 10, False),
            (AL_HOLYLIGHT, "Holy Light", 15, 5, False),
        ],
        "stat_build": [("int", 60), ("dex", 30), ("vit", 20)],
    },

    "high_merchant": {
        "name": "High Merchant",
        "tactics": "melee_dps",
        "attack_range": 1,
        "weapon_type": "sword",
        "description": "Transcendent merchant — pre-blacksmith transcendent",
        "skills": [
            (MC_MAMMONITE, "Mammonite", 20, 10, False),
            (MC_DISCOUNT, "Discount", 0, 10, False),
            (MC_OVERCHARGE, "Overcharge", 0, 10, False),
        ],
        "stat_build": [("str", 60), ("vit", 40), ("dex", 20)],
    },

    "high_thief": {
        "name": "High Thief",
        "tactics": "melee_dps",
        "attack_range": 1,
        "weapon_type": "dagger",
        "description": "Transcendent thief — pre-assassin transcendent",
        "skills": [
            (TF_DOUBLE, "Double Attack", 0, 10, False),
            (TF_HIDING, "Hiding", 10, 5, False),
            (TF_POISON, "Envenom", 12, 5, False),
        ],
        "stat_build": [("agi", 60), ("dex", 30), ("str", 20)],
    },

    "baby_novice": {
        "name": "Baby Novice",
        "tactics": "hybrid",
        "attack_range": 1,
        "weapon_type": "dagger",
        "description": "Doram / Baby class — lower stats, cute appearance",
        "skills": [
            (NV_BASIC, "Basic Skill", 0, 1, False),
        ],
        "stat_build": [("str", 10), ("agi", 10), ("dex", 10)],
    },
}


class JobRegistry:
    """Registry of all job definitions.

    Provides lookup by name, tactics resolution, and skill access.
    Lazy-loaded to avoid import-time overhead.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._load_all()
        self._tactics_cache: dict[str, Any] = {}

    def _load_all(self) -> None:
        """Load all job definitions."""
        self._jobs = dict(ALL_JOBS)
        logger.info("job_registry_loaded: %d jobs", len(self._jobs))

    def get_job(self, job_name: str) -> dict[str, Any] | None:
        """Get a job definition by name (case-insensitive)."""
        key = job_name.lower().strip()
        with self._lock:
            return self._jobs.get(key)

    def get_tactics_name(self, job_name: str) -> str:
        """Get the tactics module name for a job.

        Returns "hybrid" for unknown jobs (safe default).
        """
        job = self.get_job(job_name)
        if job:
            return job.get("tactics", "hybrid")
        return "hybrid"

    def get_tactics_for_job(self, job_name: str) -> object:
        """Get the tactics instance for a job.

        Returns a cached BaseTactics instance appropriate for the job.
        """
        job_name_lower = job_name.lower().strip()

        with self._lock:
            if job_name_lower in self._tactics_cache:
                return self._tactics_cache[job_name_lower]

        tactics_name = self.get_tactics_name(job_name)

        # Lazy-import to avoid circular dependencies
        if tactics_name == "tank":
            from ai_sidecar.domains.combat.tactics.tank import TankTactics
            instance: object = TankTactics()
        elif tactics_name == "melee_dps":
            from ai_sidecar.domains.combat.tactics.melee_dps import MeleeDPSTactics
            instance = MeleeDPSTactics()
        elif tactics_name == "ranged_dps":
            from ai_sidecar.domains.combat.tactics.ranged_dps import RangedDPSTactics
            instance = RangedDPSTactics()
        elif tactics_name == "magic_dps":
            from ai_sidecar.domains.combat.tactics.magic_dps import MagicDPSTactics
            instance = MagicDPSTactics()
        elif tactics_name == "support":
            from ai_sidecar.domains.combat.tactics.support import SupportTactics
            instance = SupportTactics()
        else:
            from ai_sidecar.domains.combat.tactics.hybrid import HybridTactics
            instance = HybridTactics()

        with self._lock:
            self._tactics_cache[job_name_lower] = instance

        return instance

    def get_skills_for_job(self, job_name: str) -> list[tuple[str, str, int, int, bool]]:
        """Get the skill list for a job.

        Returns list of (skill_id, name, sp_cost, level, is_aoe).
        """
        job = self.get_job(job_name)
        if job:
            return job.get("skills", [])
        return []

    def get_stat_build(self, job_name: str) -> list[tuple[str, int]]:
        """Get the recommended stat build for a job."""
        job = self.get_job(job_name)
        if job:
            return job.get("stat_build", [])
        return []

    def get_weapon_type(self, job_name: str) -> str | None:
        """Get the preferred weapon type for a job."""
        job = self.get_job(job_name)
        if job:
            return job.get("weapon_type")
        return None

    def get_attack_range(self, job_name: str) -> int:
        """Get the base attack range for a job."""
        job = self.get_job(job_name)
        if job:
            return job.get("attack_range", 1)
        return 1

    def all_jobs(self) -> dict[str, dict[str, Any]]:
        """Get all registered job definitions."""
        with self._lock:
            return dict(self._jobs)

    def job_count(self) -> int:
        with self._lock:
            return len(self._jobs)

    def list_job_names(self) -> list[str]:
        """Get sorted list of all job names."""
        with self._lock:
            return sorted(self._jobs.keys())


# ── Global Singleton ──

_registry: JobRegistry | None = None
_registry_lock = RLock()


def get_job_registry() -> JobRegistry:
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = JobRegistry()
        return _registry


def get_tactics_for_job(job_name: str) -> object:
    """Convenience function to get tactics for a job by name.

    Uses the global JobRegistry singleton.
    """
    return get_job_registry().get_tactics_for_job(job_name)
