"""Skill registry, combo chains, status effects, and rotation engine.

Provides:
  - SkillRegistry: maps internal skill IDs to full skill metadata with timing, elements, weapon requirements
  - ComboChain: defines skill A → skill B chains with time windows and bonus damage
  - StatusEffect tracking: stun, freeze, poison, blind, silence, confusion, curse, sleep
  - Conditional damage bonuses: +50% to stunned, +25% to frozen, etc.
  - Dispel tracking (Lex Aeterna = +100% next damage)
  - Skill timing (cast time, delay, cooldown, animation time)
  - Weapon-type requirements for each skill
  - SkillRotationEngine: manages rotations with combo-aware selection
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any

from ai_sidecar.autonomy.ro_mechanics import SKILL_DAMAGE, SKILL_SP_COSTS

logger = logging.getLogger(__name__)


# ── Status Effects ─────────────────────────────────────────────────────────

class StatusEffect(str, Enum):
    """All status effects that can be applied to characters or targets."""
    STUN = "stun"
    FREEZE = "freeze"
    POISON = "poison"
    BLIND = "blind"
    SILENCE = "silence"
    CONFUSION = "confusion"
    CURSE = "curse"
    SLEEP = "sleep"
    PETRIFY = "petrify"
    BURN = "burn"
    BLEED = "bleed"
    DARKNESS = "darkness"
    SLOW = "slow"
    STONE = "stone"
    FEAR = "fear"
    DEEP_SLEEP = "deep_sleep"
    CRYSTAL = "crystal"


# ── Status effect damage bonuses ──────────────────────────────────────────

STATUS_DAMAGE_BONUSES: dict[StatusEffect, float] = {
    StatusEffect.STUN: 0.50,       # +50% damage to stunned targets
    StatusEffect.FREEZE: 0.25,     # +25% damage to frozen targets
    StatusEffect.SLEEP: 0.40,      # +40% damage to sleeping targets
    StatusEffect.PETRIFY: 0.30,    # +30% damage to petrified targets
    StatusEffect.STONE: 0.20,      # +20% damage to stone targets
    StatusEffect.BURN: 0.15,       # +15% damage to burning targets
    StatusEffect.POISON: 0.10,     # +10% damage to poisoned targets
    StatusEffect.BLIND: 0.05,      # +5% damage to blinded targets
    StatusEffect.CONFUSION: 0.05,  # +5% damage to confused targets
    StatusEffect.CURSE: 0.10,      # +10% damage to cursed targets
    StatusEffect.SLOW: 0.10,       # +10% damage to slowed targets
    StatusEffect.BLEED: 0.15,      # +15% damage to bleeding targets
    StatusEffect.FEAR: 0.20,       # +20% damage to feared targets
}

# Status effects that prevent the target from acting
HARD_CC_STATUSES = {
    StatusEffect.STUN, StatusEffect.FREEZE, StatusEffect.SLEEP,
    StatusEffect.PETRIFY, StatusEffect.STONE, StatusEffect.DEEP_SLEEP,
    StatusEffect.CRYSTAL, StatusEffect.FEAR,
}

# Status effects that reduce the target's effectiveness
SOFT_CC_STATUSES = {
    StatusEffect.BLIND, StatusEffect.SILENCE, StatusEffect.CONFUSION,
    StatusEffect.CURSE, StatusEffect.SLOW, StatusEffect.DARKNESS,
}

# Damage-over-time status effects
DOT_STATUSES = {
    StatusEffect.POISON, StatusEffect.BURN, StatusEffect.BLEED,
}


# ── Weapon Types ──────────────────────────────────────────────────────────

class WeaponType(str, Enum):
    """RO weapon types that skills may require."""
    DAGGER = "dagger"
    SWORD = "sword"
    ONE_HANDED_SWORD = "1h_sword"
    TWO_HANDED_SWORD = "2h_sword"
    SPEAR = "spear"
    ONE_HANDED_SPEAR = "1h_spear"
    TWO_HANDED_SPEAR = "2h_spear"
    AXE = "axe"
    ONE_HANDED_AXE = "1h_axe"
    TWO_HANDED_AXE = "2h_axe"
    MACE = "mace"
    STAFF = "staff"
    BOW = "bow"
    KATAR = "katar"
    KNUCKLE = "knuckle"
    MUSICAL_INSTRUMENT = "musical_instrument"
    WHIP = "whip"
    BOOK = "book"
    GUN = "gun"
    HUUMA_SHURIKEN = "huuma_shuriken"
    GRENADE = "grenade"
    ANY = "any"
    NONE = "none"  # No weapon required (e.g., heal, buffs)


# ── Skill Metadata ─────────────────────────────────────────────────────────

@dataclass
class SkillDef:
    """Complete skill definition with timing, elements, and weapon requirements.

    Fields:
      skill_id:      Internal RO ID (e.g. "MG_FIREBOLT")
      name:          Display name (e.g. "Fire Bolt")
      sp_cost:       SP cost at base level
      range:         Max range in cells
      cooldown_ms:   Cooldown in milliseconds
      is_aoe:        Area of effect flag
      aoe_radius:    AoE radius in cells
      cast_time_ms:  Cast time in milliseconds
      delay_ms:      Post-cast delay in milliseconds
      animation_ms:  Animation time in milliseconds
      element:       Element type (neutral, fire, water, wind, earth, holy, shadow, ghost, undead, poison)
      damage_type:   melee, magic, ranged
      heal_pct:      Heal as % of MATK (for Heal skill)
      buff_duration_ms: Duration for buffs
      weapon_required: Weapon type required (WeaponType enum)
      applies_status: Status effect this skill applies to the target
      status_chance:  Chance (0.0-1.0) of applying the status effect
      tags:          Categorization tags
    """
    skill_id: str
    name: str
    sp_cost: int
    range: int
    cooldown_ms: int
    is_aoe: bool
    aoe_radius: int = 0
    cast_time_ms: int = 0
    delay_ms: int = 0
    animation_ms: int = 0
    element: str = "neutral"
    damage_type: str = "melee"
    heal_pct: float = 0.0
    buff_duration_ms: int = 0
    weapon_required: WeaponType = WeaponType.ANY
    applies_status: StatusEffect | None = None
    status_chance: float = 0.0
    tags: list[str] = field(default_factory=list)

    @property
    def total_time_ms(self) -> int:
        """Total time from cast start to ready for next action."""
        return self.cast_time_ms + self.delay_ms + self.animation_ms

    @property
    def is_buff(self) -> bool:
        return "buff" in self.tags

    @property
    def is_passive(self) -> bool:
        return "passive" in self.tags

    @property
    def is_heal(self) -> bool:
        return "heal" in self.tags

    @property
    def is_utility(self) -> bool:
        return "utility" in self.tags or "escape" in self.tags


# ── Combo Chain ────────────────────────────────────────────────────────────

@dataclass
class ComboStep:
    """A single step in a combo chain.

    skill_id:     The skill to use
    time_window_ms:  How long after the previous skill this must be used
    bonus_damage_pct: Bonus damage multiplier (e.g., 1.5 = +50%)
    bonus_damage_type: How the bonus is applied: "multiply" or "add"
    required_status:  Status effect the target must have for this step
    consumes_status:  Whether this step consumes the status effect
    sp_cost_multiplier: SP cost multiplier for this step
    repeat:        How many times to repeat this step
    """
    skill_id: str
    time_window_ms: int = 3000
    bonus_damage_pct: float = 0.0
    bonus_damage_type: str = "multiply"  # "multiply" or "add"
    required_status: StatusEffect | None = None
    consumes_status: bool = False
    sp_cost_multiplier: float = 1.0
    repeat: int = 1


@dataclass
class ComboChain:
    """A chain of skills that combo together for bonus damage.

    name:        Combo name (e.g., "Raging Trifecta")
    steps:       Ordered list of combo steps
    description: Human-readable description
    job_required: Job class that can use this combo
    """
    name: str
    steps: list[ComboStep]
    description: str = ""
    job_required: str = ""


# ── Dispel Tracking ──────────────────────────────────────────────────────

@dataclass
class DispelState:
    """Tracks dispel effects on a target (e.g., Lex Aeterna).

    active:        Whether the dispel effect is active
    multiplier:    Damage multiplier (2.0 = +100%)
    expires_at:    Monotonic timestamp when the dispel expires
    source_skill:  The skill that applied the dispel
    """
    active: bool = False
    multiplier: float = 2.0
    expires_at: float = 0.0
    source_skill: str = ""


# ── Skill Registry ────────────────────────────────────────────────────────

class SkillRegistry:
    """Maps internal skill IDs to SkillDef objects.

    All 80+ commonly used RO skills for pre-renewal with full timing,
    element, weapon requirement, and status effect data.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._skills: dict[str, SkillDef] = {}
        self._combo_chains: list[ComboChain] = []
        self._load_all()
        self._load_combo_chains()

    def _load_all(self) -> None:
        """Load all skill definitions."""
        s: dict[str, SkillDef] = {}

        # ── Novice Skills ──
        s["NV_BASIC"] = SkillDef("NV_BASIC", "Basic Skill", 0, 1, 0, False, tags=["passive"])
        s["NV_FIRSTAID"] = SkillDef("NV_FIRSTAID", "First Aid", 5, 1, 0, False, heal_pct=0.03, tags=["heal"])

        # ── Swordman Skills ──
        s["SM_BASH"] = SkillDef("SM_BASH", "Bash", 8, 1, 500, False,
                                cast_time_ms=0, delay_ms=500, animation_ms=300,
                                element="neutral", damage_type="melee",
                                weapon_required=WeaponType.ANY,
                                applies_status=StatusEffect.STUN, status_chance=0.10,
                                tags=["physical", "stun"])
        s["SM_RECOVERY"] = SkillDef("SM_RECOVERY", "HP Recovery", 0, 0, 0, False,
                                    buff_duration_ms=60000, tags=["buff", "passive"])
        s["SM_MAGNUM"] = SkillDef("SM_MAGNUM", "Magnum Break", 12, 1, 3000, True, aoe_radius=3,
                                  cast_time_ms=0, delay_ms=1000, animation_ms=500,
                                  element="fire", damage_type="melee",
                                  weapon_required=WeaponType.ANY,
                                  tags=["physical", "aoe", "fire"])
        s["SM_ENDURE"] = SkillDef("SM_ENDURE", "Endure", 10, 0, 30000, False,
                                  cast_time_ms=0, delay_ms=500,
                                  buff_duration_ms=20000, tags=["buff", "defensive"])
        s["SM_PROVOKE"] = SkillDef("SM_PROVOKE", "Provoke", 3, 9, 5000, False,
                                   cast_time_ms=0, delay_ms=500,
                                   tags=["taunt", "debuff"])

        # ── Knight Skills ──
        s["KN_SPEARMASTERY"] = SkillDef("KN_SPEARMASTERY", "Spear Mastery", 0, 0, 0, False,
                                        weapon_required=WeaponType.SPEAR, tags=["passive"])
        s["KN_BRANDISHSPEAR"] = SkillDef("KN_BRANDISHSPEAR", "Brandish Spear", 15, 1, 2000, True, aoe_radius=3,
                                         cast_time_ms=0, delay_ms=1000, animation_ms=500,
                                         element="neutral", damage_type="melee",
                                         weapon_required=WeaponType.SPEAR,
                                         tags=["physical", "aoe"])
        s["KN_PIERCE"] = SkillDef("KN_PIERCE", "Pierce", 10, 1, 1000, False,
                                  cast_time_ms=0, delay_ms=500, animation_ms=300,
                                  element="neutral", damage_type="melee",
                                  weapon_required=WeaponType.SPEAR,
                                  tags=["physical"])
        s["KN_BOWLINGBASH"] = SkillDef("KN_BOWLINGBASH", "Bowling Bash", 15, 1, 2000, True, aoe_radius=3,
                                       cast_time_ms=0, delay_ms=1000, animation_ms=500,
                                       element="neutral", damage_type="melee",
                                       weapon_required=WeaponType.TWO_HANDED_SWORD,
                                       tags=["physical", "aoe"])
        s["KN_TWOHANDQUICKEN"] = SkillDef("KN_TWOHANDQUICKEN", "Two-Hand Quicken", 15, 0, 30000, False,
                                          cast_time_ms=0, delay_ms=500,
                                          buff_duration_ms=300000,
                                          weapon_required=WeaponType.TWO_HANDED_SWORD,
                                          tags=["buff", "aspd"])
        s["KN_SPEARBOOMERANG"] = SkillDef("KN_SPEARBOOMERANG", "Spear Boomerang", 12, 7, 2000, False,
                                          cast_time_ms=0, delay_ms=500, animation_ms=300,
                                          element="neutral", damage_type="ranged",
                                          weapon_required=WeaponType.SPEAR,
                                          tags=["physical", "ranged"])
        s["KN_AURA"] = SkillDef("KN_AURA", "Aura Blade", 20, 0, 30000, False,
                                cast_time_ms=0, delay_ms=500,
                                buff_duration_ms=120000, element="neutral",
                                tags=["buff"])

        # ── Crusader / Paladin Skills ──
        s["CR_SHIELDBOOMERANG"] = SkillDef("CR_SHIELDBOOMERANG", "Shield Boomerang", 15, 7, 3000, False,
                                            cast_time_ms=0, delay_ms=500, animation_ms=300,
                                            element="neutral", damage_type="ranged",
                                            weapon_required=WeaponType.ONE_HANDED_SWORD,
                                            tags=["physical", "ranged"])
        s["CR_SHIELDCHARGE"] = SkillDef("CR_SHIELDCHARGE", "Shield Charge", 8, 1, 1000, False,
                                        cast_time_ms=0, delay_ms=500, animation_ms=300,
                                        element="neutral", damage_type="melee",
                                        weapon_required=WeaponType.ONE_HANDED_SWORD,
                                        applies_status=StatusEffect.STUN, status_chance=0.20,
                                        tags=["physical", "stun"])
        s["CR_AUTOGUARD"] = SkillDef("CR_AUTOGUARD", "Auto Guard", 15, 0, 0, False,
                                     buff_duration_ms=300000, tags=["buff", "defensive"])
        s["CR_REFLECTSHIELD"] = SkillDef("CR_REFLECTSHIELD", "Reflect Shield", 20, 0, 0, False,
                                         buff_duration_ms=300000, tags=["buff", "defensive"])
        s["CR_PROVOCATE"] = SkillDef("CR_PROVOCATE", "Provocatum", 5, 9, 5000, False,
                                     cast_time_ms=0, delay_ms=500, tags=["taunt"])
        s["CR_DEVOTION"] = SkillDef("CR_DEVOTION", "Devotion", 20, 0, 0, False,
                                    tags=["support", "defensive"])

        # ── Mage Skills ──
        s["MG_SRECOVERY"] = SkillDef("MG_SRECOVERY", "SP Recovery", 0, 0, 0, False, tags=["passive"])
        s["MG_FIREBOLT"] = SkillDef("MG_FIREBOLT", "Fire Bolt", 12, 9, 500, False,
                                    cast_time_ms=1500, delay_ms=500, animation_ms=300,
                                    element="fire", damage_type="magic",
                                    weapon_required=WeaponType.STAFF,
                                    tags=["magic", "fire"])
        s["MG_COLD"] = SkillDef("MG_COLD", "Cold Bolt", 12, 9, 500, False,
                                cast_time_ms=1500, delay_ms=500, animation_ms=300,
                                element="water", damage_type="magic",
                                weapon_required=WeaponType.STAFF,
                                tags=["magic", "water"])
        s["MG_LIGHTNING"] = SkillDef("MG_LIGHTNING", "Lightning Bolt", 15, 9, 500, False,
                                     cast_time_ms=1500, delay_ms=500, animation_ms=300,
                                     element="wind", damage_type="magic",
                                     weapon_required=WeaponType.STAFF,
                                     tags=["magic", "wind"])
        s["MG_FIREBALL"] = SkillDef("MG_FIREBALL", "Fire Ball", 25, 9, 2000, True, aoe_radius=3,
                                    cast_time_ms=2000, delay_ms=1000, animation_ms=500,
                                    element="fire", damage_type="magic",
                                    weapon_required=WeaponType.STAFF,
                                    tags=["magic", "aoe"])
        s["MG_FROSTDIVER"] = SkillDef("MG_FROSTDIVER", "Frost Diver", 12, 9, 500, False,
                                      cast_time_ms=1000, delay_ms=500, animation_ms=300,
                                      element="water", damage_type="magic",
                                      weapon_required=WeaponType.STAFF,
                                      applies_status=StatusEffect.FREEZE, status_chance=0.30,
                                      tags=["magic", "freeze"])
        s["MG_FROSTNOVA"] = SkillDef("MG_FROSTNOVA", "Frost Nova", 20, 0, 3000, True, aoe_radius=4,
                                     cast_time_ms=2000, delay_ms=1000, animation_ms=500,
                                     element="water", damage_type="magic",
                                     weapon_required=WeaponType.STAFF,
                                     applies_status=StatusEffect.FREEZE, status_chance=0.20,
                                     tags=["magic", "aoe", "freeze"])
        s["MG_THUNDERSTORM"] = SkillDef("MG_THUNDERSTORM", "Thunderstorm", 35, 9, 3000, True, aoe_radius=5,
                                        cast_time_ms=3000, delay_ms=1500, animation_ms=500,
                                        element="wind", damage_type="magic",
                                        weapon_required=WeaponType.STAFF,
                                        tags=["magic", "aoe"])
        s["MG_NAPALMBEAT"] = SkillDef("MG_NAPALMBEAT", "Napalm Beat", 10, 9, 500, False,
                                      cast_time_ms=1000, delay_ms=500, animation_ms=300,
                                      element="neutral", damage_type="magic",
                                      weapon_required=WeaponType.STAFF,
                                      applies_status=StatusEffect.STUN, status_chance=0.05,
                                      tags=["magic", "ghost", "stun"])
        s["MG_SOULSTRIKE"] = SkillDef("MG_SOULSTRIKE", "Soul Strike", 18, 9, 500, False,
                                      cast_time_ms=1500, delay_ms=500, animation_ms=300,
                                      element="neutral", damage_type="magic",
                                      weapon_required=WeaponType.STAFF,
                                      tags=["magic", "ghost"])
        s["MG_ENERGYCOAT"] = SkillDef("MG_ENERGYCOAT", "Energy Coat", 15, 0, 0, False,
                                      buff_duration_ms=600000, tags=["buff", "defensive"])

        # ── Wizard Skills ──
        s["WZ_FIREPILLAR"] = SkillDef("WZ_FIREPILLAR", "Fire Pillar", 30, 9, 3000, True, aoe_radius=2,
                                      cast_time_ms=2000, delay_ms=1000, animation_ms=500,
                                      element="fire", damage_type="magic",
                                      weapon_required=WeaponType.STAFF,
                                      tags=["magic", "aoe", "trap"])
        s["WZ_STORMGUST"] = SkillDef("WZ_STORMGUST", "Storm Gust", 80, 9, 5000, True, aoe_radius=7,
                                     cast_time_ms=5000, delay_ms=3000, animation_ms=1000,
                                     element="water", damage_type="magic",
                                     weapon_required=WeaponType.STAFF,
                                     applies_status=StatusEffect.FREEZE, status_chance=0.15,
                                     tags=["magic", "aoe", "nuke"])
        s["WZ_METEOR"] = SkillDef("WZ_METEOR", "Meteor Storm", 90, 9, 5000, True, aoe_radius=7,
                                  cast_time_ms=6000, delay_ms=3000, animation_ms=1000,
                                  element="fire", damage_type="magic",
                                  weapon_required=WeaponType.STAFF,
                                  applies_status=StatusEffect.STUN, status_chance=0.20,
                                  tags=["magic", "aoe", "nuke", "stun"])
        s["WZ_VERMILION"] = SkillDef("WZ_VERMILION", "Lord of Vermilion", 85, 9, 5000, True, aoe_radius=7,
                                     cast_time_ms=5000, delay_ms=3000, animation_ms=1000,
                                     element="wind", damage_type="magic",
                                     weapon_required=WeaponType.STAFF,
                                     tags=["magic", "aoe", "nuke"])
        s["WZ_HEAVENDRIVE"] = SkillDef("WZ_HEAVENDRIVE", "Heaven's Drive", 45, 9, 5000, True, aoe_radius=5,
                                       cast_time_ms=4000, delay_ms=2000, animation_ms=500,
                                       element="neutral", damage_type="magic",
                                       weapon_required=WeaponType.STAFF,
                                       tags=["magic", "aoe"])
        s["WZ_FROSTNOVA"] = SkillDef("WZ_FROSTNOVA", "Frost Nova", 20, 0, 3000, True, aoe_radius=4,
                                      cast_time_ms=2000, delay_ms=1000, animation_ms=500,
                                      element="water", damage_type="magic",
                                      weapon_required=WeaponType.STAFF,
                                      applies_status=StatusEffect.FREEZE, status_chance=0.20,
                                      tags=["magic", "aoe", "freeze"])
        s["WZ_AMPLIFY"] = SkillDef("WZ_AMPLIFY", "Amplify Magic Power", 40, 0, 30000, False,
                                   cast_time_ms=1000, delay_ms=500,
                                   buff_duration_ms=60000, tags=["buff", "magic_amp"])

        # ── Archer Skills ──
        s["AC_OWL"] = SkillDef("AC_OWL", "Owl's Eye", 0, 0, 0, False,
                               buff_duration_ms=300000,
                               weapon_required=WeaponType.BOW, tags=["buff", "passive"])
        s["AC_DOUBLE"] = SkillDef("AC_DOUBLE", "Double Strafe", 12, 9, 200, False,
                                 cast_time_ms=0, delay_ms=200, animation_ms=200,
                                 element="neutral", damage_type="ranged",
                                 weapon_required=WeaponType.BOW,
                                 tags=["physical", "ranged"])
        s["AC_SHOWER"] = SkillDef("AC_SHOWER", "Arrow Shower", 15, 9, 2000, True, aoe_radius=3,
                                 cast_time_ms=0, delay_ms=1000, animation_ms=500,
                                 element="neutral", damage_type="ranged",
                                 weapon_required=WeaponType.BOW,
                                 tags=["physical", "aoe", "knockback"])
        s["AC_CONCENTRATION"] = SkillDef("AC_CONCENTRATION", "Improve Concentration", 20, 0, 30000, False,
                                        cast_time_ms=0, delay_ms=500,
                                        buff_duration_ms=180000,
                                        weapon_required=WeaponType.BOW,
                                        tags=["buff", "aspd"])

        # ── Hunter Skills ──
        s["HT_BEASTBANE"] = SkillDef("HT_BEASTBANE", "Beast Bane", 0, 0, 0, False,
                                     weapon_required=WeaponType.BOW, tags=["passive"])
        s["HT_FALCON"] = SkillDef("HT_FALCON", "Falcon", 0, 0, 0, False,
                                 weapon_required=WeaponType.BOW, tags=["passive"])
        s["HT_STEELCROW"] = SkillDef("HT_STEELCROW", "Steel Crow", 0, 0, 0, False,
                                     weapon_required=WeaponType.BOW, tags=["passive"])
        s["HT_TRUESIGHT"] = SkillDef("HT_TRUESIGHT", "True Sight", 25, 0, 30000, False,
                                     cast_time_ms=0, delay_ms=500,
                                     buff_duration_ms=120000,
                                     weapon_required=WeaponType.BOW, tags=["buff"])
        s["HT_BLITZBEAT"] = SkillDef("HT_BLITZBEAT", "Blitz Beat", 0, 0, 0, True, aoe_radius=3,
                                     weapon_required=WeaponType.BOW,
                                     tags=["passive", "aoe"])

        # ── Acolyte Skills ──
        s["AL_HEAL"] = SkillDef("AL_HEAL", "Heal", 15, 9, 500, False,
                               cast_time_ms=1000, delay_ms=1000, animation_ms=500,
                               element="holy", damage_type="magic", heal_pct=0.3,
                               weapon_required=WeaponType.NONE,
                               tags=["heal", "holy"])
        s["AL_DEMONBANE"] = SkillDef("AL_DEMONBANE", "Demon Bane", 0, 0, 0, False, tags=["passive"])
        s["AL_BLESSING"] = SkillDef("AL_BLESSING", "Blessing", 15, 9, 0, False,
                                   cast_time_ms=2000, delay_ms=500,
                                   buff_duration_ms=300000,
                                   weapon_required=WeaponType.NONE,
                                   tags=["buff", "stat_boost"])
        s["AL_INCAGI"] = SkillDef("AL_INCAGI", "Increase AGI", 15, 9, 0, False,
                                 cast_time_ms=2000, delay_ms=500,
                                 buff_duration_ms=300000,
                                 weapon_required=WeaponType.NONE,
                                 tags=["buff", "aspd"])
        s["AL_ANGELUS"] = SkillDef("AL_ANGELUS", "Angelus", 15, 0, 0, False,
                                  cast_time_ms=2000, delay_ms=500,
                                  buff_duration_ms=300000,
                                  weapon_required=WeaponType.NONE,
                                  tags=["buff", "defensive"])
        s["AL_TELEPORT"] = SkillDef("AL_TELEPORT", "Teleport", 10, 0, 0, False,
                                   cast_time_ms=2000, delay_ms=500,
                                   tags=["utility", "escape"])
        s["AL_HOLYLIGHT"] = SkillDef("AL_HOLYLIGHT", "Holy Light", 15, 9, 500, False,
                                    cast_time_ms=1500, delay_ms=500, animation_ms=300,
                                    element="holy", damage_type="magic",
                                    weapon_required=WeaponType.NONE,
                                    tags=["magic", "holy"])

        # ── Priest Skills ──
        s["PR_TURNUNDEAD"] = SkillDef("PR_TURNUNDEAD", "Turn Undead", 20, 9, 3000, False,
                                     cast_time_ms=2000, delay_ms=1000, animation_ms=500,
                                     element="holy", damage_type="magic",
                                     weapon_required=WeaponType.NONE,
                                     tags=["magic", "holy", "instant_kill"])
        s["PR_SANCTUARY"] = SkillDef("PR_SANCTUARY", "Sanctuary", 30, 9, 5000, True, aoe_radius=3,
                                    cast_time_ms=4000, delay_ms=2000, animation_ms=500,
                                    element="holy", damage_type="magic", heal_pct=0.5,
                                    weapon_required=WeaponType.NONE,
                                    tags=["heal", "aoe", "holy"])
        s["PR_BENEDICTIO"] = SkillDef("PR_BENEDICTIO", "Benedictio", 10, 9, 0, False,
                                     buff_duration_ms=300000, element="holy",
                                     weapon_required=WeaponType.NONE,
                                     tags=["buff", "holy"])
        s["PR_SLOWPOISON"] = SkillDef("PR_SLOWPOISON", "Slow Poison", 5, 9, 0, False,
                                     weapon_required=WeaponType.NONE,
                                     tags=["heal", "cure"])
        s["PR_GLORIA"] = SkillDef("PR_GLORIA", "Gloria", 20, 0, 0, False,
                                 cast_time_ms=2000, delay_ms=500,
                                 buff_duration_ms=120000,
                                 weapon_required=WeaponType.NONE,
                                 tags=["buff", "luk"])
        s["PR_MAGNIFICAT"] = SkillDef("PR_MAGNIFICAT", "Magnificat", 20, 0, 0, False,
                                     cast_time_ms=2000, delay_ms=500,
                                     buff_duration_ms=120000,
                                     weapon_required=WeaponType.NONE,
                                     tags=["buff", "sp_regen"])
        s["PR_IMPOSITIO"] = SkillDef("PR_IMPOSITIO", "Impositio Manus", 15, 9, 0, False,
                                    cast_time_ms=2000, delay_ms=500,
                                    buff_duration_ms=300000,
                                    weapon_required=WeaponType.NONE,
                                    tags=["buff", "atk_boost"])
        s["PR_SUFFRAGIUM"] = SkillDef("PR_SUFFRAGIUM", "Suffragium", 15, 9, 0, False,
                                      cast_time_ms=2000, delay_ms=500,
                                      buff_duration_ms=15000,
                                      weapon_required=WeaponType.NONE,
                                      tags=["buff", "fast_cast"])
        s["PR_KYRIE"] = SkillDef("PR_KYRIE", "Kyrie Eleison", 20, 9, 0, False,
                                cast_time_ms=2000, delay_ms=500,
                                buff_duration_ms=120000,
                                weapon_required=WeaponType.NONE,
                                tags=["buff", "defensive", "absorb"])
        s["PR_ASSUMPTIO"] = SkillDef("PR_ASSUMPTIO", "Assumptio", 30, 0, 0, False,
                                    cast_time_ms=2000, delay_ms=500,
                                    buff_duration_ms=60000,
                                    weapon_required=WeaponType.NONE,
                                    tags=["buff", "defensive"])
        s["PR_LEXAETERNA"] = SkillDef("PR_LEXAETERNA", "Lex Aeterna", 10, 9, 5000, False,
                                     cast_time_ms=1000, delay_ms=500,
                                     element="neutral", damage_type="magic",
                                     weapon_required=WeaponType.NONE,
                                     tags=["debuff", "dispel"])

        # ── Monk Skills ──
        s["MO_IRONHAND"] = SkillDef("MO_IRONHAND", "Iron Hand", 0, 0, 0, False,
                                   weapon_required=WeaponType.KNUCKLE, tags=["passive"])
        s["MO_SPIRITSRECOVERY"] = SkillDef("MO_SPIRITSRECOVERY", "Spirit's Recovery", 0, 0, 0, False,
                                          weapon_required=WeaponType.KNUCKLE, tags=["passive"])
        s["MO_FINGEROFFENSIVE"] = SkillDef("MO_FINGEROFFENSIVE", "Finger Offensive", 15, 7, 500, False,
                                           cast_time_ms=0, delay_ms=500, animation_ms=300,
                                           element="neutral", damage_type="ranged",
                                           weapon_required=WeaponType.KNUCKLE,
                                           tags=["physical", "ranged"])
        s["MO_TRIPLEATTACK"] = SkillDef("MO_TRIPLEATTACK", "Triple Attack", 0, 1, 0, False,
                                       weapon_required=WeaponType.KNUCKLE,
                                       tags=["passive", "physical"])
        s["MO_STEELBODY"] = SkillDef("MO_STEELBODY", "Steel Body", 30, 0, 30000, False,
                                    cast_time_ms=0, delay_ms=500,
                                    buff_duration_ms=300000,
                                    weapon_required=WeaponType.KNUCKLE,
                                    tags=["buff", "defensive"])
        s["MO_EXTREMITYFIST"] = SkillDef("MO_EXTREMITYFIST", "Extremity Fist", 30, 1, 5000, False,
                                        cast_time_ms=0, delay_ms=1000, animation_ms=500,
                                        element="neutral", damage_type="melee",
                                        weapon_required=WeaponType.KNUCKLE,
                                        tags=["physical", "burst", "nuke"])
        s["MO_CALLSPIRITS"] = SkillDef("MO_CALLSPIRITS", "Call Spirits", 20, 0, 0, False,
                                      cast_time_ms=0, delay_ms=500,
                                      buff_duration_ms=300000,
                                      weapon_required=WeaponType.KNUCKLE,
                                      tags=["buff", "spirit"])
        s["MO_ABSORBSPIRITS"] = SkillDef("MO_ABSORBSPIRITS", "Absorb Spirits", 0, 9, 0, False,
                                        cast_time_ms=0, delay_ms=500,
                                        weapon_required=WeaponType.KNUCKLE,
                                        tags=["utility", "sp_regen"])

        # ── Champion / Sura Skills ──
        s["MO_RAGINGQUAD"] = SkillDef("MO_RAGINGQUAD", "Raging Quadruple Blow", 25, 1, 2000, False,
                                     cast_time_ms=0, delay_ms=500, animation_ms=500,
                                     element="neutral", damage_type="melee",
                                     weapon_required=WeaponType.KNUCKLE,
                                     tags=["physical", "combo", "burst"])
        s["MO_TIGERCANNON"] = SkillDef("MO_TIGERCANNON", "Tiger Cannon", 30, 9, 3000, False,
                                      cast_time_ms=0, delay_ms=1000, animation_ms=500,
                                      element="neutral", damage_type="ranged",
                                      weapon_required=WeaponType.KNUCKLE,
                                      tags=["physical", "combo", "burst"])
        s["MO_ASURA"] = SkillDef("MO_ASURA", "Asura Strike", 50, 1, 5000, False,
                               cast_time_ms=0, delay_ms=1500, animation_ms=1000,
                               element="neutral", damage_type="melee",
                               weapon_required=WeaponType.KNUCKLE,
                               tags=["physical", "combo", "nuke", "burst"])

        # ── Thief Skills ──
        s["TF_DOUBLE"] = SkillDef("TF_DOUBLE", "Double Attack", 0, 1, 0, False,
                                 weapon_required=WeaponType.DAGGER, tags=["passive", "physical"])
        s["TF_HIDING"] = SkillDef("TF_HIDING", "Hiding", 10, 0, 0, False, tags=["utility", "escape"])
        s["TF_MISS"] = SkillDef("TF_MISS", "Improve Dodge", 0, 0, 0, False, tags=["passive", "defensive"])
        s["TF_POISON"] = SkillDef("TF_POISON", "Envenom", 12, 1, 2000, False,
                                 cast_time_ms=0, delay_ms=500, animation_ms=300,
                                 element="poison", damage_type="melee",
                                 weapon_required=WeaponType.DAGGER,
                                 applies_status=StatusEffect.POISON, status_chance=0.50,
                                 tags=["physical", "poison", "dot"])

        # ── Assassin Skills ──
        s["AS_RIGHT"] = SkillDef("AS_RIGHT", "Right Hand Mastery", 0, 0, 0, False,
                                weapon_required=WeaponType.DAGGER, tags=["passive"])
        s["AS_LEFT"] = SkillDef("AS_LEFT", "Left Hand Mastery", 0, 0, 0, False,
                               weapon_required=WeaponType.DAGGER, tags=["passive"])
        s["AS_KATAR"] = SkillDef("AS_KATAR", "Katar Mastery", 0, 0, 0, False,
                                weapon_required=WeaponType.KATAR, tags=["passive"])
        s["AS_SONICBLOW"] = SkillDef("AS_SONICBLOW", "Sonic Blow", 20, 1, 2000, False,
                                    cast_time_ms=0, delay_ms=500, animation_ms=500,
                                    element="neutral", damage_type="melee",
                                    weapon_required=WeaponType.KATAR,
                                    tags=["physical", "burst"])
        s["AS_GRIMTOOTH"] = SkillDef("AS_GRIMTOOTH", "Grimtooth", 12, 7, 1000, False,
                                    cast_time_ms=0, delay_ms=500, animation_ms=300,
                                    element="neutral", damage_type="ranged",
                                    weapon_required=WeaponType.KATAR,
                                    tags=["physical", "ranged"])
        s["AS_VENOMDUST"] = SkillDef("AS_VENOMDUST", "Venom Dust", 15, 1, 3000, True, aoe_radius=3,
                                    cast_time_ms=0, delay_ms=1000, animation_ms=500,
                                    element="poison", damage_type="melee",
                                    weapon_required=WeaponType.KATAR,
                                    applies_status=StatusEffect.POISON, status_chance=0.40,
                                    tags=["physical", "aoe", "poison"])
        s["AS_CLOAKING"] = SkillDef("AS_CLOAKING", "Cloaking", 15, 0, 0, False, tags=["utility", "escape"])
        s["AS_ENCHANTPOISON"] = SkillDef("AS_ENCHANTPOISON", "Enchant Poison", 15, 0, 0, False,
                                        cast_time_ms=0, delay_ms=500,
                                        buff_duration_ms=300000, element="poison",
                                        weapon_required=WeaponType.KATAR,
                                        tags=["buff", "element"])

        # ── Merchant Skills ──
        s["MC_DISCOUNT"] = SkillDef("MC_DISCOUNT", "Discount", 0, 0, 0, False, tags=["passive", "economy"])
        s["MC_OVERCHARGE"] = SkillDef("MC_OVERCHARGE", "Overcharge", 0, 0, 0, False, tags=["passive", "economy"])
        s["MC_VENDING"] = SkillDef("MC_VENDING", "Vending", 0, 0, 0, False, tags=["passive", "economy"])
        s["MC_PUSHCART"] = SkillDef("MC_PUSHCART", "Pushcart", 0, 0, 0, False, tags=["passive", "utility"])
        s["MC_MAMMONITE"] = SkillDef("MC_MAMMONITE", "Mammonite", 20, 1, 2000, False,
                                    cast_time_ms=0, delay_ms=500, animation_ms=300,
                                    element="neutral", damage_type="melee",
                                    weapon_required=WeaponType.ANY,
                                    tags=["physical", "burst"])

        # ── Blacksmith Skills ──
        s["BS_HAMMERFALL"] = SkillDef("BS_HAMMERFALL", "Hammerfall", 20, 1, 3000, True, aoe_radius=3,
                                      cast_time_ms=0, delay_ms=1000, animation_ms=500,
                                      element="earth", damage_type="melee",
                                      weapon_required=WeaponType.AXE,
                                      applies_status=StatusEffect.STUN, status_chance=0.15,
                                      tags=["physical", "aoe"])
        s["BS_IRON"] = SkillDef("BS_IRON", "Iron Tempering", 0, 0, 0, False, tags=["passive"])
        s["BS_STEEL"] = SkillDef("BS_STEEL", "Steel Tempering", 0, 0, 0, False, tags=["passive"])
        s["BS_WEAPONPERFECT"] = SkillDef("BS_WEAPONPERFECT", "Weapon Perfection", 15, 0, 0, False,
                                        buff_duration_ms=300000, tags=["buff", "size"])
        s["BS_MAXIMIZE"] = SkillDef("BS_MAXIMIZE", "Maximize Power", 20, 0, 30000, False,
                                   cast_time_ms=0, delay_ms=500,
                                   buff_duration_ms=300000, tags=["buff", "atk_boost"])

        # ── Sage Skills ──
        s["SA_DRAGONOLOGY"] = SkillDef("SA_DRAGONOLOGY", "Dragonology", 0, 0, 0, False, tags=["passive"])
        s["SA_MAGICROD"] = SkillDef("SA_MAGICROD", "Magic Rod", 10, 9, 3000, False,
                                   cast_time_ms=1000, delay_ms=500,
                                   weapon_required=WeaponType.STAFF,
                                   tags=["magic", "absorb"])
        s["SA_CASTCANCEL"] = SkillDef("SA_CASTCANCEL", "Cast Cancel", 5, 0, 0, False, tags=["utility"])
        s["SA_LANDPROTECTOR"] = SkillDef("SA_LANDPROTECTOR", "Land Protector", 40, 9, 60000, True, aoe_radius=5,
                                        cast_time_ms=3000, delay_ms=1000,
                                        weapon_required=WeaponType.STAFF,
                                        tags=["magic", "aoe", "defensive"])

        # ── Bard / Dancer Skills ──
        s["BA_MUSICAL"] = SkillDef("BA_MUSICAL", "Musical Lesson", 0, 0, 0, False,
                                  weapon_required=WeaponType.MUSICAL_INSTRUMENT, tags=["passive"])
        s["BA_APPLAUSE"] = SkillDef("BA_APPLAUSE", "Lesson of Applause", 0, 0, 0, False,
                                   weapon_required=WeaponType.MUSICAL_INSTRUMENT, tags=["passive"])
        s["DC_DANCING"] = SkillDef("DC_DANCING", "Dancing Lesson", 0, 0, 0, False,
                                  weapon_required=WeaponType.WHIP, tags=["passive"])

        # ── Rogue Skills ──
        s["RG_SNATCHER"] = SkillDef("RG_SNATCHER", "Snatcher", 0, 0, 0, False,
                                   weapon_required=WeaponType.DAGGER, tags=["passive"])
        s["RG_BACKSTAB"] = SkillDef("RG_BACKSTAB", "Back Stab", 10, 1, 2000, False,
                                   cast_time_ms=0, delay_ms=500, animation_ms=300,
                                   element="neutral", damage_type="melee",
                                   weapon_required=WeaponType.DAGGER,
                                   tags=["physical", "burst"])
        s["RG_STEAL"] = SkillDef("RG_STEAL", "Steal", 10, 1, 2000, False,
                                cast_time_ms=0, delay_ms=500,
                                weapon_required=WeaponType.DAGGER,
                                tags=["utility", "steal"])
        s["RG_STRIPWEAPON"] = SkillDef("RG_STRIPWEAPON", "Strip Weapon", 15, 2, 3000, False,
                                      cast_time_ms=0, delay_ms=500,
                                      weapon_required=WeaponType.DAGGER,
                                      tags=["debuff"])
        s["RG_STRIPSHIELD"] = SkillDef("RG_STRIPSHIELD", "Strip Shield", 15, 2, 3000, False,
                                      cast_time_ms=0, delay_ms=500,
                                      weapon_required=WeaponType.DAGGER,
                                      tags=["debuff"])
        s["RG_STRIPARMOR"] = SkillDef("RG_STRIPARMOR", "Strip Armor", 15, 2, 3000, False,
                                     cast_time_ms=0, delay_ms=500,
                                     weapon_required=WeaponType.DAGGER,
                                     tags=["debuff"])
        s["RG_STRIPHELM"] = SkillDef("RG_STRIPHELM", "Strip Helm", 15, 2, 3000, False,
                                    cast_time_ms=0, delay_ms=500,
                                    weapon_required=WeaponType.DAGGER,
                                    tags=["debuff"])

        # ── Alchemist Skills ──
        s["AM_LEARNING"] = SkillDef("AM_LEARNING", "Learning Potion", 0, 0, 0, False, tags=["passive"])
        s["AM_POTIONRESEARCH"] = SkillDef("AM_POTIONRESEARCH", "Potion Research", 0, 0, 0, False, tags=["passive"])
        s["AM_CALLHOMUN"] = SkillDef("AM_CALLHOMUN", "Call Homunculus", 50, 0, 0, False, tags=["utility"])
        s["AM_DEMONSTRATION"] = SkillDef("AM_DEMONSTRATION", "Demonstration", 25, 9, 5000, True, aoe_radius=3,
                                        cast_time_ms=2000, delay_ms=1000, animation_ms=500,
                                        element="poison", damage_type="magic",
                                        applies_status=StatusEffect.POISON, status_chance=0.60,
                                        tags=["magic", "aoe", "poison"])
        s["AM_ACIDTERROR"] = SkillDef("AM_ACIDTERROR", "Acid Terror", 30, 9, 3000, False,
                                     cast_time_ms=1500, delay_ms=1000, animation_ms=500,
                                     element="neutral", damage_type="magic",
                                     tags=["magic", "ranged"])

        # ── Soul Linker Skills ──
        s["SL_SOULCOLLECT"] = SkillDef("SL_SOULCOLLECT", "Soul Collect", 10, 0, 0, False, tags=["utility"])
        s["SL_KAINA"] = SkillDef("SL_KAINA", "Kaina", 30, 9, 0, False,
                                buff_duration_ms=300000, element="fire", tags=["buff", "element"])
        s["SL_KAUPE"] = SkillDef("SL_KAUPE", "Kaupe", 30, 9, 0, False,
                                buff_duration_ms=60000, tags=["buff", "defensive"])

        # ── Ninja Skills ──
        s["NJ_KUNAI"] = SkillDef("NJ_KUNAI", "Throw Kunai", 8, 7, 500, False,
                                cast_time_ms=0, delay_ms=200, animation_ms=200,
                                element="neutral", damage_type="ranged",
                                weapon_required=WeaponType.NONE,
                                tags=["physical", "ranged"])
        s["NJ_HUUMA"] = SkillDef("NJ_HUUMA", "Huuma Shuriken", 15, 7, 2000, True, aoe_radius=3,
                                cast_time_ms=0, delay_ms=500, animation_ms=300,
                                element="neutral", damage_type="ranged",
                                weapon_required=WeaponType.HUUMA_SHURIKEN,
                                tags=["physical", "aoe"])
        s["NJ_ZENYNAGE"] = SkillDef("NJ_ZENYNAGE", "Throw Zeny", 0, 7, 500, False,
                                   cast_time_ms=0, delay_ms=200, animation_ms=200,
                                   element="neutral", damage_type="ranged",
                                   tags=["physical", "ranged"])
        s["NJ_KASUMIKIRI"] = SkillDef("NJ_KASUMIKIRI", "Kasumikiri", 25, 1, 3000, False,
                                     cast_time_ms=0, delay_ms=500, animation_ms=500,
                                     element="neutral", damage_type="melee",
                                     applies_status=StatusEffect.BLEED, status_chance=0.30,
                                     tags=["physical", "burst"])

        # ── Gunslinger Skills ──
        s["GS_SINGLE"] = SkillDef("GS_SINGLE", "Single Action", 0, 0, 0, False,
                                 weapon_required=WeaponType.GUN, tags=["passive"])
        s["GS_CHAIN"] = SkillDef("GS_CHAIN", "Chain Action", 0, 0, 0, False,
                               weapon_required=WeaponType.GUN, tags=["passive"])
        s["GS_TRACK"] = SkillDef("GS_TRACK", "Tracking", 10, 14, 2000, False,
                                cast_time_ms=500, delay_ms=500, animation_ms=300,
                                element="neutral", damage_type="ranged",
                                weapon_required=WeaponType.GUN,
                                tags=["physical", "ranged"])
        s["GS_DESPERADO"] = SkillDef("GS_DESPERADO", "Desperado", 30, 0, 3000, True, aoe_radius=5,
                                    cast_time_ms=0, delay_ms=1000, animation_ms=500,
                                    element="neutral", damage_type="ranged",
                                    weapon_required=WeaponType.GUN,
                                    tags=["physical", "aoe"])
        s["GS_GATLINGFEVER"] = SkillDef("GS_GATLINGFEVER", "Gatling Fever", 15, 0, 30000, False,
                                       cast_time_ms=0, delay_ms=500,
                                       buff_duration_ms=120000,
                                       weapon_required=WeaponType.GUN,
                                       tags=["buff", "aspd"])

        # ── Taekwon / Star Gladiator Skills ──
        s["TK_PUNCH"] = SkillDef("TK_PUNCH", "Punch", 0, 1, 0, False,
                                weapon_required=WeaponType.NONE, tags=["physical"])
        s["TK_KICK"] = SkillDef("TK_KICK", "Kick", 5, 1, 500, False,
                               cast_time_ms=0, delay_ms=300, animation_ms=200,
                               weapon_required=WeaponType.NONE, tags=["physical"])
        s["TK_COUNTER"] = SkillDef("TK_COUNTER", "Counter Kick", 8, 1, 2000, False,
                                  cast_time_ms=0, delay_ms=500, animation_ms=300,
                                  weapon_required=WeaponType.NONE,
                                  tags=["physical", "counter"])
        s["SG_FEEL"] = SkillDef("SG_FEEL", "Feel", 30, 0, 0, False,
                               buff_duration_ms=300000, tags=["buff"])
        s["SG_SUNWARM"] = SkillDef("SG_SUNWARM", "Sun Warm", 20, 0, 0, False,
                                  buff_duration_ms=300000, tags=["buff"])

        # ── Super Novice ──
        s["SN_BASIC"] = SkillDef("SN_BASIC", "Super Basic", 0, 1, 0, False, tags=["passive"])

        self._skills = s
        logger.info("skill_registry_loaded: %d skills", len(s))

    def _load_combo_chains(self) -> None:
        """Load all combo chain definitions."""
        self._combo_chains = [
            # ── Monk: Raging Trifecta ──
            ComboChain(
                name="Raging Trifecta",
                job_required="monk",
                description="Raging Quadruple Blow → Tiger Cannon → Asura Strike. "
                            "Each step must follow within 3s for bonus damage.",
                steps=[
                    ComboStep("MO_RAGINGQUAD", time_window_ms=3000, bonus_damage_pct=0.0),
                    ComboStep("MO_TIGERCANNON", time_window_ms=3000, bonus_damage_pct=0.50,
                              sp_cost_multiplier=0.8),
                    ComboStep("MO_ASURA", time_window_ms=3000, bonus_damage_pct=1.00,
                              sp_cost_multiplier=1.5),
                ],
            ),
            # ── Wizard: Freeze → Nuke ──
            ComboChain(
                name="Freeze Shatter",
                job_required="wizard",
                description="Freeze target with Frost Diver/Nova, then shatter with "
                            "Storm Gust or Meteor Storm for +50% damage.",
                steps=[
                    ComboStep("MG_FROSTDIVER", time_window_ms=5000, bonus_damage_pct=0.0,
                              required_status=StatusEffect.FREEZE),
                    ComboStep("WZ_STORMGUST", time_window_ms=5000, bonus_damage_pct=0.50,
                              required_status=StatusEffect.FREEZE, consumes_status=True),
                ],
            ),
            ComboChain(
                name="Freeze Meteor",
                job_required="wizard",
                description="Freeze then Meteor Storm for massive AoE damage.",
                steps=[
                    ComboStep("WZ_FROSTNOVA", time_window_ms=5000, bonus_damage_pct=0.0,
                              required_status=StatusEffect.FREEZE),
                    ComboStep("WZ_METEOR", time_window_ms=5000, bonus_damage_pct=0.50,
                              required_status=StatusEffect.FREEZE, consumes_status=True),
                ],
            ),
            # ── Assassin: Poison → Burst ──
            ComboChain(
                name="Venom Strike",
                job_required="assassin",
                description="Poison the target first, then Sonic Blow for +30% damage.",
                steps=[
                    ComboStep("TF_POISON", time_window_ms=5000, bonus_damage_pct=0.0,
                              required_status=StatusEffect.POISON),
                    ComboStep("AS_SONICBLOW", time_window_ms=5000, bonus_damage_pct=0.30,
                              required_status=StatusEffect.POISON, consumes_status=False),
                ],
            ),
            # ── Knight: Provoke → Bowling Bash ──
            ComboChain(
                name="Provoke Smash",
                job_required="knight",
                description="Provoke to lower defense, then Bowling Bash for +25% damage.",
                steps=[
                    ComboStep("SM_PROVOKE", time_window_ms=5000, bonus_damage_pct=0.0),
                    ComboStep("KN_BOWLINGBASH", time_window_ms=5000, bonus_damage_pct=0.25),
                ],
            ),
            # ── Priest: Lex Aeterna → Holy Light ──
            ComboChain(
                name="Lex Judex",
                job_required="priest",
                description="Cast Lex Aeterna on target, then Holy Light for +100% damage.",
                steps=[
                    ComboStep("PR_LEXAETERNA", time_window_ms=5000, bonus_damage_pct=0.0),
                    ComboStep("AL_HOLYLIGHT", time_window_ms=5000, bonus_damage_pct=1.00),
                ],
            ),
            # ── Blacksmith: Maximize → Hammerfall ──
            ComboChain(
                name="Maximize Impact",
                job_required="blacksmith",
                description="Activate Maximize Power, then Hammerfall for +40% damage.",
                steps=[
                    ComboStep("BS_MAXIMIZE", time_window_ms=5000, bonus_damage_pct=0.0),
                    ComboStep("BS_HAMMERFALL", time_window_ms=5000, bonus_damage_pct=0.40),
                ],
            ),
            # ── Hunter: Concentration → Double Strafe ──
            ComboChain(
                name="Focused Fire",
                job_required="hunter",
                description="Activate Concentration, then spam Double Strafe for +20% damage.",
                steps=[
                    ComboStep("AC_CONCENTRATION", time_window_ms=5000, bonus_damage_pct=0.0),
                    ComboStep("AC_DOUBLE", time_window_ms=5000, bonus_damage_pct=0.20, repeat=3),
                ],
            ),
        ]
        logger.info("combo_chains_loaded: %d chains", len(self._combo_chains))

    # ── Skill queries ──────────────────────────────────────────────────

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
        """Get likely skill IDs for a given job from CLASS_SKILL_TRAINING."""
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
        return 10

    def is_aoe(self, skill_id: str) -> bool:
        """Check if a skill is AoE."""
        skill = self.get(skill_id)
        return skill.is_aoe if skill else False

    def get_element(self, skill_id: str) -> str:
        """Get the element of a skill."""
        skill = self.get(skill_id)
        return skill.element if skill else "neutral"

    def get_weapon_required(self, skill_id: str) -> WeaponType:
        """Get the weapon type required by a skill."""
        skill = self.get(skill_id)
        return skill.weapon_required if skill else WeaponType.ANY

    def get_status_applied(self, skill_id: str) -> tuple[StatusEffect | None, float]:
        """Get the status effect a skill applies and its chance."""
        skill = self.get(skill_id)
        if skill:
            return skill.applies_status, skill.status_chance
        return None, 0.0

    def can_execute(
        self,
        skill_id: str,
        current_sp: int,
        current_hp: int,
        cooldowns: dict[str, float] | None = None,
        equipped_weapon: WeaponType | None = None,
    ) -> tuple[bool, str]:
        """Check if a skill can be executed.

        Checks SP, cooldowns, and weapon requirements.
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

        if equipped_weapon and skill.weapon_required != WeaponType.ANY and skill.weapon_required != WeaponType.NONE:
            if equipped_weapon != skill.weapon_required:
                return False, (
                    f"wrong weapon: need {skill.weapon_required.value}, "
                    f"have {equipped_weapon.value}"
                )

        return True, "ok"

    # ── Combo chain queries ────────────────────────────────────────────

    def get_combo_chains(self) -> list[ComboChain]:
        """Get all registered combo chains."""
        with self._lock:
            return list(self._combo_chains)

    def get_combo_chains_for_job(self, job_name: str) -> list[ComboChain]:
        """Get combo chains available for a specific job class."""
        job_lower = job_name.lower()
        with self._lock:
            return [
                chain for chain in self._combo_chains
                if chain.job_required == job_lower
            ]

    def get_combo_chains_starting_with(self, skill_id: str) -> list[ComboChain]:
        """Get combo chains that start with a given skill."""
        upper = skill_id.upper()
        with self._lock:
            return [
                chain for chain in self._combo_chains
                if chain.steps and chain.steps[0].skill_id.upper() == upper
            ]

    def get_combo_follow_up(
        self,
        last_skill_id: str,
        job_name: str = "",
    ) -> list[ComboStep]:
        """Get possible follow-up combo steps after a given skill.

        Returns all combo steps that could follow the last used skill,
        filtered by time window and job if specified.
        """
        upper = last_skill_id.upper()
        follow_ups: list[ComboStep] = []
        now = time.monotonic() * 1000  # Convert to ms

        with self._lock:
            for chain in self._combo_chains:
                if job_name and chain.job_required != job_name.lower():
                    continue
                for i, step in enumerate(chain.steps):
                    if i > 0 and chain.steps[i - 1].skill_id.upper() == upper:
                        follow_ups.append(step)

        return follow_ups

    def get_combo_bonus(
        self,
        skill_id: str,
        previous_skill_id: str,
        time_since_previous_ms: float,
        target_statuses: set[StatusEffect] | None = None,
    ) -> float:
        """Calculate the total damage bonus for a skill in a combo context.

        Returns a multiplier (1.0 = no bonus, 1.5 = +50%, 2.0 = +100%).
        Combines:
        1. Combo chain bonus from previous skill
        2. Status effect bonus on target
        3. Dispel bonus (Lex Aeterna)
        """
        bonus = 1.0
        upper = skill_id.upper()
        prev_upper = previous_skill_id.upper()

        # 1. Combo chain bonus
        with self._lock:
            for chain in self._combo_chains:
                for i, step in enumerate(chain.steps):
                    if step.skill_id.upper() == upper and i > 0:
                        prev_step = chain.steps[i - 1]
                        if prev_step.skill_id.upper() == prev_upper:
                            if time_since_previous_ms <= step.time_window_ms:
                                if step.bonus_damage_type == "multiply":
                                    bonus *= (1.0 + step.bonus_damage_pct)
                                else:
                                    bonus += step.bonus_damage_pct

        # 2. Status effect bonus
        if target_statuses:
            for status in target_statuses:
                status_bonus = STATUS_DAMAGE_BONUSES.get(status, 0.0)
                if status_bonus > 0:
                    bonus *= (1.0 + status_bonus)

        return bonus


# ── Rotation Engine ────────────────────────────────────────────────────────

@dataclass
class RotationStep:
    """A single step in a skill rotation."""
    skill_id: str
    level: int = 1
    condition: str = ""  # Python expression evaluated in context
    priority: int = 50
    repeat: int = 1  # How many times to repeat this step before moving on
    combo_only: bool = False  # Only use this skill as part of a combo chain


@dataclass
class Rotation:
    """A named skill rotation with situational triggers."""
    name: str
    steps: list[RotationStep] = field(default_factory=list)
    trigger_condition: str = ""
    priority: int = 50


class SkillRotationEngine:
    """Manages skill rotations with combo-aware selection.

    Integrates with SkillRegistry for combo chain awareness:
    - When a combo chain is active, prioritizes follow-up skills
    - Applies status effect damage bonuses
    - Tracks dispel effects (Lex Aeterna)
    - Respects weapon requirements
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

        # Wizard freeze → nuke combo rotation
        rotations["wizard_freeze_nuke"] = Rotation(
            name="wizard_freeze_nuke",
            priority=95,
            steps=[
                RotationStep("MG_FROSTDIVER", 10, priority=95, condition="target and not target.is_frozen"),
                RotationStep("WZ_STORMGUST", 10, priority=90, condition="target.is_frozen and sp > 80"),
                RotationStep("WZ_METEOR", 10, priority=85, condition="target.is_frozen and sp > 90 and aggro >= 3"),
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

        # Knight provoke → bowling bash combo
        rotations["knight_provoke_smash"] = Rotation(
            name="knight_provoke_smash",
            priority=90,
            steps=[
                RotationStep("SM_PROVOKE", 10, priority=95, condition="target and not target.is_provoked"),
                RotationStep("KN_BOWLINGBASH", 10, priority=90, condition="target.is_provoked"),
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

        # Assassin poison → sonic blow combo
        rotations["assassin_venom_strike"] = Rotation(
            name="assassin_venom_strike",
            priority=90,
            steps=[
                RotationStep("TF_POISON", 10, priority=95, condition="target and not target.is_poisoned"),
                RotationStep("AS_SONICBLOW", 10, priority=90, condition="target.is_poisoned"),
            ],
        )

        # Monk Raging Trifecta combo rotation
        rotations["monk_trifecta"] = Rotation(
            name="monk_trifecta",
            priority=95,
            steps=[
                RotationStep("MO_CALLSPIRITS", 5, priority=99, condition="not has_buff('callspirits')", repeat=0),
                RotationStep("MO_RAGINGQUAD", 10, priority=95, condition="spirit_spheres >= 5"),
                RotationStep("MO_TIGERCANNON", 10, priority=90, condition="combo_active and sp > 30"),
                RotationStep("MO_ASURA", 10, priority=85, condition="combo_active and sp > 50 and target.hp_pct > 0.3"),
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

        # Priest Lex Aeterna → Holy Light combo
        rotations["priest_lex_judex"] = Rotation(
            name="priest_lex_judex",
            priority=95,
            steps=[
                RotationStep("PR_LEXAETERNA", 10, priority=95, condition="target and not target.has_lex"),
                RotationStep("AL_HOLYLIGHT", 10, priority=90, condition="target.has_lex"),
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

        # Blacksmith maximize → hammerfall combo
        rotations["blacksmith_max_impact"] = Rotation(
            name="blacksmith_max_impact",
            priority=90,
            steps=[
                RotationStep("BS_MAXIMIZE", 10, priority=95, condition="not has_buff('maximize')"),
                RotationStep("BS_HAMMERFALL", 10, priority=90, condition="has_buff('maximize') and aggro >= 2"),
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

    def find_best_rotation(
        self,
        job_class: str,
        aggro: int = 0,
        sp_pct: float = 1.0,
        target_element: str = "",
        has_combo_active: bool = False,
    ) -> Rotation | None:
        """Find the best rotation for the current situation.

        Prefers combo rotations when a combo is active.
        Prefers AoE rotations for high aggro.
        """
        with self._lock:
            candidates: list[Rotation] = []
            job_lower = job_class.lower()

            for rot in self._rotations.values():
                if job_lower in rot.name:
                    candidates.append(rot)

            if not candidates:
                return None

            # If a combo is active, prefer combo rotations
            if has_combo_active:
                combo_rots = [r for r in candidates if "combo" in r.name or "trifecta" in r.name
                             or "freeze" in r.name or "venom" in r.name or "provoke" in r.name
                             or "lex" in r.name or "max" in r.name]
                if combo_rots:
                    combo_rots.sort(key=lambda r: -r.priority)
                    return combo_rots[0]

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
