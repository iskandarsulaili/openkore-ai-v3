"""
Skill Rotation System — intelligent skill selection and rotation for RO combat.

A pro player doesn't spam the same skill. They have a mental rotation:
Fire Bolt on Earth monsters, Cold Bolt on Fire monsters, Storm Gust on groups.
This module encodes that knowledge with full per-level RO-accurate data.

Includes:
- SkillLevel dataclass with per-level damage_ratio, sp_cost, cast_time_ms, after_cast_delay_ms
- Skill dataclass with after_cast_delay_ms, hits_per_cast, levels, optimal_range
- Populated per-level data for all skills (levels 1-10 where applicable)
- DEX-based cast time reduction and after-cast delay tracking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SkillLevel:
    """Per-level data for a skill (levels 1-10)."""
    level: int = 1
    damage_ratio: int = 100       # % of ATK/MATK per hit
    sp_cost: int = 10
    cast_time_ms: int = 0         # Variable cast time (reducible by DEX)
    after_cast_delay_ms: int = 0  # After-cast delay (FIXED, NOT reducible by DEX)
    hits_per_cast: int = 1        # Number of hits per cast (Storm Gust=10, Meteor Storm=7, etc.)


@dataclass
class Skill:
    """A single skill with full RO-accurate timing and per-level data."""
    name: str
    skill_id: int = 0
    element: str = "neutral"
    cast_time_ms: int = 0         # Level 10 cast time (backwards compat)
    delay_ms: int = 0             # Level 10 skill delay (backwards compat — used as after_cast_delay)
    after_cast_delay_ms: int = 0  # FIXED delay after cast (NOT reducible by DEX)
    cooldown_ms: int = 0
    sp_cost: int = 0
    range: int = 1
    target_type: str = "single"   # single, aoe, self
    damage_type: str = "melee"    # melee, magic, ranged
    aoe_radius: int = 0
    min_level: int = 1
    required_job: str = "novice"
    tags: list[str] = field(default_factory=list)
    damage_ratio: int = 100       # Level 10 damage ratio (% of ATK/MATK per hit)
    hits_per_cast: int = 1        # Number of hits per cast (Storm Gust Lv10 = 10 hits)
    optimal_range: int = 1        # Optimal engagement range for this skill
    levels: list[SkillLevel] = field(default_factory=list)  # Per-level data [Lv1..Lv10]


@dataclass
class RotationStep:
    """A step in a skill rotation."""
    skill_name: str
    condition: str = ""  # e.g. "sp > 20", "target.element == fire", "aggro > 3"
    priority: int = 50


@dataclass
class SkillRotation:
    """A named rotation template."""
    name: str
    steps: list[RotationStep] = field(default_factory=list)
    priority: int = 50
    target_condition: str = ""  # e.g. "element:fire", "size:large", "boss"
    description: str = ""


def _build_levels(
    damage_ratio_per_level: list[int],
    sp_cost_per_level: list[int],
    cast_time_per_level: list[int],
    after_cast_delay_per_level: list[int],
    hits_per_cast_per_level: list[int] | None = None,
) -> list[SkillLevel]:
    """Build a list of SkillLevel objects from per-level arrays.
    
    Args:
        damage_ratio_per_level: Damage ratio at each level [Lv1, Lv2, ..., Lv10]
        sp_cost_per_level: SP cost at each level
        cast_time_per_level: Cast time in ms at each level
        after_cast_delay_per_level: After-cast delay in ms at each level
        hits_per_cast_per_level: Hits per cast at each level (None = 1 for all)
    """
    hits = hits_per_cast_per_level or [1] * 10
    levels = []
    for i in range(10):
        lv = i + 1
        levels.append(SkillLevel(
            level=lv,
            damage_ratio=damage_ratio_per_level[i],
            sp_cost=sp_cost_per_level[i],
            cast_time_ms=cast_time_per_level[i],
            after_cast_delay_ms=after_cast_delay_per_level[i],
            hits_per_cast=hits[i],
        ))
    return levels


class SkillRotationSystem:
    """Intelligent skill selection and rotation."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._skills: dict[str, Skill] = {}
        self._rotations: dict[str, SkillRotation] = {}
        self._load_skills()
        self._load_rotations()

    def _add_levels_bolt(self, prefix: str) -> list[SkillLevel]:
        """Bolt-type skill levels (Fire Bolt, Cold Bolt, Lightning Bolt).
        
        RO formula: damage = base + per_level * skill_level
        Lv1: 100%, Lv2: 140%, Lv3: 180%, Lv4: 220%, Lv5: 260%,
        Lv6: 300%, Lv7: 340%, Lv8: 380%, Lv9: 420%, Lv10: 460%
        
        Cast time scales: Lv1=0.8s, Lv5=1.5s, Lv10=2.0s
        After-cast delay: 1.5s all levels
        SP cost: Lv1=12, +3 per level
        """
        return _build_levels(
            damage_ratio_per_level=[100, 140, 180, 220, 260, 300, 340, 380, 420, 460],
            sp_cost_per_level=[12, 15, 18, 21, 24, 27, 30, 33, 36, 39],
            cast_time_per_level=[800, 900, 1000, 1100, 1500, 1600, 1700, 1800, 1900, 2000],
            after_cast_delay_per_level=[1500] * 10,
        )

    def _add_levels_bolt_10hits(self, prefix: str) -> list[SkillLevel]:
        """Multi-hit bolt with fixed 10 hits (Storm Gust style).
        
        Storm Gust: (0.6 + 0.1*lv) MATK per hit × 10 hits
        Lv1: 0.7 * 10 = 700%, Lv10: 1.6 * 10 = 1600%
        Cast: Lv1=2s, Lv10=5s
        After-cast: fixed 3s
        SP: Lv1=47, Lv10=80
        """
        return _build_levels(
            damage_ratio_per_level=[70, 80, 90, 100, 110, 120, 130, 140, 150, 160],
            sp_cost_per_level=[47, 50, 55, 59, 63, 67, 71, 74, 77, 80],
            cast_time_per_level=[2000, 2500, 3000, 3500, 3600, 4000, 4400, 4700, 5000, 5000],
            after_cast_delay_per_level=[3000] * 10,
            hits_per_cast_per_level=[10] * 10,
        )

    def _add_levels_meteor(self, prefix: str) -> list[SkillLevel]:
        """Meteor Storm: (0.5 + 0.1*lv) MATK per hit × 7 hits
        Lv1: 0.6 * 7 = 420%, Lv10: 1.5 * 7 = 1050%
        Cast: Lv1=4s, Lv10=6s
        After-cast: fixed 3s
        """
        return _build_levels(
            damage_ratio_per_level=[60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
            sp_cost_per_level=[57, 60, 65, 69, 73, 77, 81, 84, 87, 90],
            cast_time_per_level=[4000, 4500, 4800, 5000, 5200, 5400, 5600, 5800, 6000, 6000],
            after_cast_delay_per_level=[3000] * 10,
            hits_per_cast_per_level=[7] * 10,
        )

    def _add_levels_lov(self, prefix: str) -> list[SkillLevel]:
        """Lord of Vermilion: (1.0 + 0.3*lv) MATK per hit × 5 hits
        Lv1: 1.3 * 5 = 650%, Lv10: 4.0 * 5 = 2000%
        Cast: Lv1=2s, Lv10=5s
        After-cast: fixed 2s
        """
        return _build_levels(
            damage_ratio_per_level=[130, 160, 190, 220, 250, 280, 310, 340, 370, 400],
            sp_cost_per_level=[52, 56, 60, 64, 68, 72, 76, 79, 82, 85],
            cast_time_per_level=[2000, 2500, 3000, 3500, 3800, 4000, 4300, 4600, 5000, 5000],
            after_cast_delay_per_level=[2000] * 10,
            hits_per_cast_per_level=[5] * 10,
        )

    def _add_levels_thunderstorm(self, prefix: str) -> list[SkillLevel]:
        """Thunderstorm: (0.4 + 0.3*lv) MATK per hit × 5 hits
        Lv1: 0.7 * 5 = 350%, Lv10: 3.4 * 5 = 1700%
        Cast: Lv1=1s, Lv10=3s
        """
        return _build_levels(
            damage_ratio_per_level=[70, 100, 130, 160, 190, 220, 250, 280, 310, 340],
            sp_cost_per_level=[22, 26, 30, 33, 36, 39, 42, 44, 46, 48],
            cast_time_per_level=[1000, 1500, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3000],
            after_cast_delay_per_level=[1500] * 10,
            hits_per_cast_per_level=[5] * 10,
        )

    def _add_levels_melee_linear(self, base: float, per: float, sp_base: int, sp_per: int = 0,
                                  cast_ms: int = 0, delay_ms: int = 0) -> list[SkillLevel]:
        """Linear scaling melee skill (Bash, Magnum Break, etc.)
        
        Bash: 1.5 + 0.3*lv, Lv1=180%, Lv10=420%
        """
        ratio_per_level = [int((base + per * (lv)) * 100) for lv in range(1, 11)]
        sp_per_level = [sp_base + sp_per * (lv - 1) for lv in range(1, 11)]
        return _build_levels(
            damage_ratio_per_level=ratio_per_level,
            sp_cost_per_level=sp_per_level,
            cast_time_per_level=[cast_ms] * 10,
            after_cast_delay_per_level=[delay_ms] * 10,
        )

    def _add_levels_sonic_blow(self) -> list[SkillLevel]:
        """Sonic Blow: 1.0 + 0.8*lv, Lv1=180%, Lv10=900%
        8 hits at all levels, but damage is per-hit.
        Fixed cast (no reduction), fixed after-cast delay.
        """
        return _build_levels(
            damage_ratio_per_level=[180, 260, 340, 420, 500, 580, 660, 740, 820, 900],
            sp_cost_per_level=[15, 17, 19, 21, 23, 25, 27, 29, 31, 33],
            cast_time_per_level=[0] * 10,       # Fixed cast — unaffected by DEX
            after_cast_delay_per_level=[1000] * 10,
            hits_per_cast_per_level=[8] * 10,   # Always 8 hits
        )

    def _add_levels_bowling_bash(self) -> list[SkillLevel]:
        """Bowling Bash: 1.0 + 0.6*lv, Lv1=160%, Lv10=640%
        Fixed cast — NOT affected by DEX.
        """
        return _build_levels(
            damage_ratio_per_level=[160, 220, 280, 340, 400, 460, 520, 580, 640, 700],
            sp_cost_per_level=[12, 14, 16, 18, 20, 21, 22, 23, 24, 25],
            cast_time_per_level=[500] * 10,      # Fixed — NOT reducible by DEX
            after_cast_delay_per_level=[1500] * 10,
        )

    def _load_skills(self) -> None:
        """Load skill data for major classes with full per-level RO-accurate data."""

        # ═══════════════════════════════════════════════════════════════════════
        # ── Mage / Wizard Skills
        # ═══════════════════════════════════════════════════════════════════════

        # Fire Bolt — Level 1-4 = element Lv1, 5-9 = Lv2, 10 = Lv3
        self._add_skill(Skill(
            "Fire Bolt", skill_id=1, element="fire",
            cast_time_ms=2000, after_cast_delay_ms=1500, delay_ms=1500,
            cooldown_ms=0, sp_cost=39, range=9, target_type="single",
            damage_type="magic", min_level=1, required_job="mage",
            tags=["fire", "single", "fast"], damage_ratio=460, optimal_range=7,
            levels=self._add_levels_bolt("MG_FIREBOLT"),
        ))

        # Cold Bolt
        self._add_skill(Skill(
            "Cold Bolt", skill_id=2, element="water",
            cast_time_ms=2000, after_cast_delay_ms=1500, delay_ms=1500,
            cooldown_ms=0, sp_cost=39, range=9, target_type="single",
            damage_type="magic", min_level=1, required_job="mage",
            tags=["water", "single", "fast"], damage_ratio=460, optimal_range=7,
            levels=self._add_levels_bolt("MG_COLDBOLT"),
        ))

        # Lightning Bolt
        self._add_skill(Skill(
            "Lightning Bolt", skill_id=3, element="wind",
            cast_time_ms=2000, after_cast_delay_ms=1500, delay_ms=1500,
            cooldown_ms=0, sp_cost=39, range=9, target_type="single",
            damage_type="magic", min_level=1, required_job="mage",
            tags=["wind", "single", "fast"], damage_ratio=460, optimal_range=7,
            levels=self._add_levels_bolt("MG_LIGHTNINGBOLT"),
        ))

        # Fire Ball (AoE)
        self._add_skill(Skill(
            "Fire Ball", skill_id=4, element="fire",
            cast_time_ms=2000, after_cast_delay_ms=1000, delay_ms=1000,
            cooldown_ms=0, sp_cost=28, range=9, target_type="aoe",
            damage_type="magic", aoe_radius=3, min_level=1, required_job="mage",
            tags=["fire", "aoe", "medium"], damage_ratio=250, optimal_range=7,
            levels=_build_levels(
                damage_ratio_per_level=[100, 120, 140, 160, 180, 200, 220, 240, 250, 250],
                sp_cost_per_level=[16, 18, 20, 22, 24, 26, 28, 28, 28, 28],
                cast_time_per_level=[1400, 1600, 1700, 1800, 1900, 1900, 2000, 2000, 2000, 2000],
                after_cast_delay_per_level=[1000] * 10,
            ),
        ))

        # Fire Wall
        self._add_skill(Skill(
            "Fire Wall", skill_id=5, element="fire",
            cast_time_ms=2000, after_cast_delay_ms=1000, delay_ms=1000,
            cooldown_ms=5000, sp_cost=30, range=5, target_type="aoe",
            damage_type="magic", aoe_radius=2, min_level=1, required_job="mage",
            tags=["fire", "aoe", "defensive", "dot"], damage_ratio=200, optimal_range=4,
            levels=_build_levels(
                damage_ratio_per_level=[100, 110, 120, 130, 140, 150, 160, 170, 180, 200],
                sp_cost_per_level=[15, 18, 20, 22, 24, 26, 28, 30, 30, 30],
                cast_time_per_level=[1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2500, 3000],
                after_cast_delay_per_level=[1000] * 10,
            ),
        ))

        # Frost Diver
        self._add_skill(Skill(
            "Frost Diver", skill_id=6, element="water",
            cast_time_ms=1000, after_cast_delay_ms=500, delay_ms=500,
            cooldown_ms=0, sp_cost=15, range=9, target_type="single",
            damage_type="magic", min_level=1, required_job="mage",
            tags=["water", "single", "fast", "freeze"], damage_ratio=140, optimal_range=7,
            levels=_build_levels(
                damage_ratio_per_level=[100, 105, 110, 115, 120, 125, 130, 135, 140, 140],
                sp_cost_per_level=[10, 11, 12, 13, 14, 15, 16, 17, 18, 20],
                cast_time_per_level=[700, 750, 800, 850, 900, 950, 1000, 1000, 1000, 1000],
                after_cast_delay_per_level=[500] * 10,
            ),
        ))

        # Frost Nova
        self._add_skill(Skill(
            "Frost Nova", skill_id=7, element="water",
            cast_time_ms=2000, after_cast_delay_ms=1000, delay_ms=1000,
            cooldown_ms=3000, sp_cost=25, range=0, target_type="aoe",
            damage_type="magic", aoe_radius=4, min_level=1, required_job="mage",
            tags=["water", "aoe", "freeze"], damage_ratio=180, optimal_range=3,
            levels=_build_levels(
                damage_ratio_per_level=[100, 110, 120, 130, 140, 150, 160, 170, 180, 180],
                sp_cost_per_level=[14, 16, 17, 18, 19, 20, 21, 22, 24, 25],
                cast_time_per_level=[1400, 1500, 1600, 1700, 1800, 1900, 2000, 2000, 2000, 2000],
                after_cast_delay_per_level=[1000] * 10,
            ),
        ))

        # Thunderstorm
        self._add_skill(Skill(
            "Thunderstorm", skill_id=8, element="wind",
            cast_time_ms=3000, after_cast_delay_ms=1500, delay_ms=1500,
            cooldown_ms=3000, sp_cost=48, range=9, target_type="aoe",
            damage_type="magic", aoe_radius=5, min_level=1, required_job="mage",
            tags=["wind", "aoe", "slow"], damage_ratio=340, hits_per_cast=5,
            optimal_range=7,
            levels=self._add_levels_thunderstorm("MG_THUNDERSTORM"),
        ))

        # Heaven's Drive
        self._add_skill(Skill(
            "Heaven's Drive", skill_id=9, element="neutral",
            cast_time_ms=4000, after_cast_delay_ms=2000, delay_ms=2000,
            cooldown_ms=5000, sp_cost=55, range=9, target_type="aoe",
            damage_type="magic", aoe_radius=5, min_level=1, required_job="wizard",
            tags=["neutral", "aoe", "slow", "high_damage"], damage_ratio=450,
            optimal_range=7,
            levels=_build_levels(
                damage_ratio_per_level=[200, 230, 260, 290, 320, 350, 380, 410, 430, 450],
                sp_cost_per_level=[30, 34, 37, 40, 43, 46, 49, 52, 55, 55],
                cast_time_per_level=[2000, 2500, 2800, 3000, 3200, 3500, 3800, 4000, 4000, 4000],
                after_cast_delay_per_level=[2000] * 10,
            ),
        ))

        # Storm Gust — PEAK damage: 10 hits × (60+10*lv)% = Lv10=1600%
        self._add_skill(Skill(
            "Storm Gust", skill_id=10, element="water",
            cast_time_ms=5000, after_cast_delay_ms=3000, delay_ms=3000,
            cooldown_ms=5000, sp_cost=80, range=9, target_type="aoe",
            damage_type="magic", aoe_radius=7, min_level=1, required_job="wizard",
            tags=["water", "aoe", "very_slow", "high_damage", "freeze"],
            damage_ratio=160, hits_per_cast=10, optimal_range=7,
            levels=self._add_levels_bolt_10hits("WZ_STORMGUST"),
        ))

        # Meteor Storm — 7 hits × (50+10*lv)% = Lv10=1050%
        self._add_skill(Skill(
            "Meteor Storm", skill_id=11, element="fire",
            cast_time_ms=6000, after_cast_delay_ms=3000, delay_ms=3000,
            cooldown_ms=5000, sp_cost=90, range=9, target_type="aoe",
            damage_type="magic", aoe_radius=7, min_level=1, required_job="wizard",
            tags=["fire", "aoe", "very_slow", "high_damage", "stun"],
            damage_ratio=150, hits_per_cast=7, optimal_range=7,
            levels=self._add_levels_meteor("WZ_METEORSTORM"),
        ))

        # Lord of Vermilion — 5 hits × 400% = 2000% total
        self._add_skill(Skill(
            "Lord of Vermilion", skill_id=12, element="wind",
            cast_time_ms=5000, after_cast_delay_ms=3000, delay_ms=3000,
            cooldown_ms=5000, sp_cost=85, range=9, target_type="aoe",
            damage_type="magic", aoe_radius=7, min_level=1, required_job="wizard",
            tags=["wind", "aoe", "very_slow", "high_damage"],
            damage_ratio=400, hits_per_cast=5, optimal_range=7,
            levels=self._add_levels_lov("WZ_VERMILION"),
        ))

        # Napalm Beat — instant cast, interrupt-immune
        self._add_skill(Skill(
            "Napalm Beat", skill_id=13, element="neutral",
            cast_time_ms=1000, after_cast_delay_ms=500, delay_ms=500,
            cooldown_ms=0, sp_cost=15, range=9, target_type="single",
            damage_type="magic", min_level=1, required_job="mage",
            tags=["neutral", "single", "fast", "undead"], damage_ratio=220,
            optimal_range=7,
            levels=_build_levels(
                damage_ratio_per_level=[120, 130, 140, 150, 160, 170, 180, 190, 200, 220],
                sp_cost_per_level=[8, 9, 10, 11, 12, 13, 14, 15, 15, 15],
                cast_time_per_level=[1000] * 10,
                after_cast_delay_per_level=[500] * 10,
            ),
        ))

        # Soul Strike
        self._add_skill(Skill(
            "Soul Strike", skill_id=14, element="neutral",
            cast_time_ms=1500, after_cast_delay_ms=500, delay_ms=500,
            cooldown_ms=0, sp_cost=25, range=9, target_type="single",
            damage_type="magic", min_level=1, required_job="mage",
            tags=["neutral", "single", "fast", "undead", "ghost"],
            damage_ratio=250, optimal_range=7,
            levels=_build_levels(
                damage_ratio_per_level=[150, 160, 170, 180, 190, 200, 210, 220, 230, 250],
                sp_cost_per_level=[12, 14, 16, 18, 20, 22, 24, 25, 25, 25],
                cast_time_per_level=[1000, 1100, 1200, 1300, 1400, 1500, 1500, 1500, 1500, 1500],
                after_cast_delay_per_level=[500] * 10,
            ),
        ))

        # Safety Wall
        self._add_skill(Skill(
            "Safety Wall", skill_id=15, element="neutral",
            cast_time_ms=2000, after_cast_delay_ms=1000, delay_ms=1000,
            cooldown_ms=10000, sp_cost=20, range=3, target_type="self",
            damage_type="magic", min_level=1, required_job="mage",
            tags=["defensive", "support"], damage_ratio=0, optimal_range=0,
            levels=_build_levels(
                damage_ratio_per_level=[0] * 10,
                sp_cost_per_level=[10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
                cast_time_per_level=[2000] * 10,
                after_cast_delay_per_level=[1000] * 10,
            ),
        ))

        # ═══════════════════════════════════════════════════════════════════════
        # ── Archer / Hunter Skills
        # ═══════════════════════════════════════════════════════════════════════

        # Double Strafe — Fixed cast (no DEX reduction), instant
        self._add_skill(Skill(
            "Double Strafe", skill_id=50, element="neutral",
            cast_time_ms=0, after_cast_delay_ms=200, delay_ms=200,
            cooldown_ms=0, sp_cost=12, range=9, target_type="single",
            damage_type="ranged", min_level=1, required_job="archer",
            tags=["neutral", "single", "very_fast", "physical"],
            damage_ratio=380, optimal_range=7,
            levels=_build_levels(
                damage_ratio_per_level=[200, 220, 240, 260, 280, 300, 320, 340, 360, 380],
                sp_cost_per_level=[8, 9, 10, 11, 12, 12, 12, 12, 12, 12],
                cast_time_per_level=[0] * 10,
                after_cast_delay_per_level=[200] * 10,
            ),
        ))

        # Arrow Shower
        self._add_skill(Skill(
            "Arrow Shower", skill_id=51, element="neutral",
            cast_time_ms=0, after_cast_delay_ms=1000, delay_ms=1000,
            cooldown_ms=2000, sp_cost=20, range=9, target_type="aoe",
            damage_type="ranged", aoe_radius=3, min_level=1, required_job="archer",
            tags=["neutral", "aoe", "fast", "physical", "knockback"],
            damage_ratio=200, optimal_range=7,
            levels=_build_levels(
                damage_ratio_per_level=[100, 110, 120, 130, 140, 150, 160, 170, 180, 200],
                sp_cost_per_level=[10, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                cast_time_per_level=[0] * 10,
                after_cast_delay_per_level=[1000] * 10,
            ),
        ))

        # Improve Concentration (buff)
        self._add_skill(Skill(
            "Improve Concentration", skill_id=52, element="neutral",
            cast_time_ms=0, after_cast_delay_ms=0, delay_ms=0,
            cooldown_ms=30000, sp_cost=20, range=0, target_type="self",
            damage_type="ranged", min_level=1, required_job="archer",
            tags=["buff", "support"], damage_ratio=0, optimal_range=0,
            levels=_build_levels(
                damage_ratio_per_level=[0] * 10,
                sp_cost_per_level=[10, 12, 14, 16, 18, 20, 20, 20, 20, 20],
                cast_time_per_level=[0] * 10,
                after_cast_delay_per_level=[0] * 10,
            ),
        ))

        # ═══════════════════════════════════════════════════════════════════════
        # ── Swordman / Knight Skills
        # ═══════════════════════════════════════════════════════════════════════

        # Bash — Fixed cast (instant), no DEX reduction
        self._add_skill(Skill(
            "Bash", skill_id=100, element="neutral",
            cast_time_ms=0, after_cast_delay_ms=500, delay_ms=500,
            cooldown_ms=0, sp_cost=10, range=1, target_type="single",
            damage_type="melee", min_level=1, required_job="swordman",
            tags=["neutral", "single", "very_fast", "physical", "stun"],
            damage_ratio=420, optimal_range=1,
            levels=self._add_levels_melee_linear(base=1.5, per=0.3, sp_base=4, sp_per=1,
                                                  cast_ms=0, delay_ms=500),
        ))

        # Magnum Break
        self._add_skill(Skill(
            "Magnum Break", skill_id=101, element="fire",
            cast_time_ms=0, after_cast_delay_ms=1000, delay_ms=1000,
            cooldown_ms=3000, sp_cost=15, range=1, target_type="aoe",
            damage_type="melee", aoe_radius=3, min_level=1, required_job="swordman",
            tags=["fire", "aoe", "fast", "physical"], damage_ratio=400, optimal_range=1,
            levels=self._add_levels_melee_linear(base=1.0, per=0.3, sp_base=8, sp_per=2,
                                                  cast_ms=0, delay_ms=1000),
        ))

        # Bowling Bash — Fixed cast (NOT reduced by DEX)
        self._add_skill(Skill(
            "Bowling Bash", skill_id=102, element="neutral",
            cast_time_ms=500, after_cast_delay_ms=1000, delay_ms=1000,
            cooldown_ms=2000, sp_cost=25, range=1, target_type="aoe",
            damage_type="melee", aoe_radius=3, min_level=1, required_job="knight",
            tags=["neutral", "aoe", "fast", "physical", "high_damage", "fixed_cast"],
            damage_ratio=700, optimal_range=1,
            levels=self._add_levels_bowling_bash(),
        ))

        # Provoke (no damage, taunt debuff)
        self._add_skill(Skill(
            "Provoke", skill_id=103, element="neutral",
            cast_time_ms=0, after_cast_delay_ms=500, delay_ms=500,
            cooldown_ms=5000, sp_cost=5, range=9, target_type="single",
            damage_type="melee", min_level=1, required_job="swordman",
            tags=["support", "taunt"], damage_ratio=0, optimal_range=9,
            levels=_build_levels(
                damage_ratio_per_level=[0] * 10,
                sp_cost_per_level=[2, 3, 4, 5, 5, 5, 5, 5, 5, 5],
                cast_time_per_level=[0] * 10,
                after_cast_delay_per_level=[500] * 10,
            ),
        ))

        # Endure (buff)
        self._add_skill(Skill(
            "Endure", skill_id=104, element="neutral",
            cast_time_ms=0, after_cast_delay_ms=0, delay_ms=0,
            cooldown_ms=30000, sp_cost=15, range=0, target_type="self",
            damage_type="melee", min_level=1, required_job="swordman",
            tags=["buff", "defensive"], damage_ratio=0, optimal_range=0,
            levels=_build_levels(
                damage_ratio_per_level=[0] * 10,
                sp_cost_per_level=[5, 6, 7, 8, 9, 10, 11, 12, 13, 15],
                cast_time_per_level=[0] * 10,
                after_cast_delay_per_level=[0] * 10,
            ),
        ))

        # ═══════════════════════════════════════════════════════════════════════
        # ── Thief / Assassin Skills
        # ═══════════════════════════════════════════════════════════════════════

        # Double Attack (passive)
        self._add_skill(Skill(
            "Double Attack", skill_id=150, element="neutral",
            cast_time_ms=0, after_cast_delay_ms=0, delay_ms=0,
            cooldown_ms=0, sp_cost=0, range=1, target_type="single",
            damage_type="melee", min_level=1, required_job="thief",
            tags=["neutral", "single", "passive", "physical"],
            damage_ratio=300, optimal_range=1,
            levels=self._add_levels_melee_linear(base=1.5, per=0.15, sp_base=0,
                                                  cast_ms=0, delay_ms=0),
        ))

        # Throw Sand (no damage, blinds)
        self._add_skill(Skill(
            "Throw Sand", skill_id=151, element="neutral",
            cast_time_ms=0, after_cast_delay_ms=500, delay_ms=500,
            cooldown_ms=3000, sp_cost=8, range=5, target_type="single",
            damage_type="ranged", min_level=1, required_job="thief",
            tags=["neutral", "single", "fast", "blind"], damage_ratio=0,
            optimal_range=4,
            levels=_build_levels(
                damage_ratio_per_level=[0] * 10,
                sp_cost_per_level=[3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
                cast_time_per_level=[0] * 10,
                after_cast_delay_per_level=[500] * 10,
            ),
        ))

        # Hide
        self._add_skill(Skill(
            "Hide", skill_id=152, element="neutral",
            cast_time_ms=0, after_cast_delay_ms=0, delay_ms=0,
            cooldown_ms=1000, sp_cost=10, range=0, target_type="self",
            damage_type="melee", min_level=1, required_job="thief",
            tags=["defensive", "stealth"], damage_ratio=0, optimal_range=0,
            levels=_build_levels(
                damage_ratio_per_level=[0] * 10,
                sp_cost_per_level=[4, 5, 6, 7, 8, 9, 10, 10, 10, 10],
                cast_time_per_level=[0] * 10,
                after_cast_delay_per_level=[0] * 10,
            ),
        ))

        # Sonic Blow — Fixed cast (no DEX reduction), 8 hits
        self._add_skill(Skill(
            "Sonic Blow", skill_id=153, element="neutral",
            cast_time_ms=0, after_cast_delay_ms=1000, delay_ms=1000,
            cooldown_ms=2000, sp_cost=33, range=1, target_type="single",
            damage_type="melee", min_level=1, required_job="assassin",
            tags=["neutral", "single", "fast", "physical", "high_damage", "poison", "fixed_cast"],
            damage_ratio=900, hits_per_cast=8, optimal_range=1,
            levels=self._add_levels_sonic_blow(),
        ))

        # ═══════════════════════════════════════════════════════════════════════
        # ── Acolyte / Priest Skills
        # ═══════════════════════════════════════════════════════════════════════

        # Heal
        self._add_skill(Skill(
            "Heal", skill_id=200, element="holy",
            cast_time_ms=1000, after_cast_delay_ms=500, delay_ms=500,
            cooldown_ms=0, sp_cost=15, range=9, target_type="single",
            damage_type="magic", min_level=1, required_job="acolyte",
            tags=["holy", "single", "fast", "heal", "undead"],
            damage_ratio=300, optimal_range=7,
            levels=_build_levels(
                damage_ratio_per_level=[100, 120, 140, 160, 180, 200, 220, 250, 280, 300],
                sp_cost_per_level=[8, 10, 12, 13, 14, 15, 16, 17, 18, 20],
                cast_time_per_level=[500, 600, 700, 800, 900, 1000, 1000, 1000, 1000, 1000],
                after_cast_delay_per_level=[500] * 10,
            ),
        ))

        # Blessing (buff)
        self._add_skill(Skill(
            "Blessing", skill_id=201, element="neutral",
            cast_time_ms=2000, after_cast_delay_ms=1000, delay_ms=1000,
            cooldown_ms=0, sp_cost=20, range=9, target_type="single",
            damage_type="magic", min_level=1, required_job="acolyte",
            tags=["buff", "support"], damage_ratio=0, optimal_range=7,
            levels=_build_levels(
                damage_ratio_per_level=[0] * 10,
                sp_cost_per_level=[8, 10, 12, 14, 16, 18, 20, 22, 24, 28],
                cast_time_per_level=[2000] * 10,
                after_cast_delay_per_level=[1000] * 10,
            ),
        ))

        # Increase Agility (buff)
        self._add_skill(Skill(
            "Increase Agility", skill_id=202, element="neutral",
            cast_time_ms=2000, after_cast_delay_ms=1000, delay_ms=1000,
            cooldown_ms=0, sp_cost=20, range=9, target_type="single",
            damage_type="magic", min_level=1, required_job="acolyte",
            tags=["buff", "support"], damage_ratio=0, optimal_range=7,
            levels=_build_levels(
                damage_ratio_per_level=[0] * 10,
                sp_cost_per_level=[10, 12, 14, 16, 18, 20, 22, 24, 26, 30],
                cast_time_per_level=[2000] * 10,
                after_cast_delay_per_level=[1000] * 10,
            ),
        ))

        # Teleport
        self._add_skill(Skill(
            "Teleport", skill_id=203, element="neutral",
            cast_time_ms=1000, after_cast_delay_ms=0, delay_ms=0,
            cooldown_ms=0, sp_cost=15, range=0, target_type="self",
            damage_type="magic", min_level=1, required_job="acolyte",
            tags=["utility", "movement"], damage_ratio=0, optimal_range=0,
            levels=_build_levels(
                damage_ratio_per_level=[0] * 10,
                sp_cost_per_level=[5, 6, 7, 8, 9, 10, 11, 12, 13, 15],
                cast_time_per_level=[1000] * 10,
                after_cast_delay_per_level=[0] * 10,
            ),
        ))

        # Holy Light
        self._add_skill(Skill(
            "Holy Light", skill_id=204, element="holy",
            cast_time_ms=1500, after_cast_delay_ms=500, delay_ms=500,
            cooldown_ms=0, sp_cost=20, range=9, target_type="single",
            damage_type="magic", min_level=1, required_job="acolyte",
            tags=["holy", "single", "fast", "undead", "demon"],
            damage_ratio=250, optimal_range=7,
            levels=_build_levels(
                damage_ratio_per_level=[120, 130, 140, 150, 160, 170, 180, 200, 220, 250],
                sp_cost_per_level=[8, 10, 12, 14, 16, 18, 20, 22, 24, 28],
                cast_time_per_level=[1000, 1100, 1200, 1300, 1400, 1500, 1500, 1500, 1500, 1500],
                after_cast_delay_per_level=[500] * 10,
            ),
        ))

        # Turn Undead
        self._add_skill(Skill(
            "Turn Undead", skill_id=205, element="holy",
            cast_time_ms=2000, after_cast_delay_ms=1000, delay_ms=1000,
            cooldown_ms=3000, sp_cost=30, range=9, target_type="single",
            damage_type="magic", min_level=1, required_job="priest",
            tags=["holy", "single", "medium", "undead", "instant_kill"],
            damage_ratio=450, optimal_range=7,
            levels=_build_levels(
                damage_ratio_per_level=[200, 220, 240, 260, 280, 320, 360, 400, 420, 450],
                sp_cost_per_level=[15, 18, 20, 22, 24, 26, 28, 30, 30, 30],
                cast_time_per_level=[1500, 1600, 1700, 1800, 1900, 2000, 2000, 2000, 2000, 2000],
                after_cast_delay_per_level=[1000] * 10,
            ),
        ))

        # ═══════════════════════════════════════════════════════════════════════
        # ── Merchant / Blacksmith Skills
        # ═══════════════════════════════════════════════════════════════════════

        # Mammonite
        self._add_skill(Skill(
            "Mammonite", skill_id=250, element="neutral",
            cast_time_ms=0, after_cast_delay_ms=1000, delay_ms=1000,
            cooldown_ms=0, sp_cost=20, range=1, target_type="single",
            damage_type="melee", min_level=1, required_job="merchant",
            tags=["neutral", "single", "fast", "physical", "high_damage", "costs_zeny"],
            damage_ratio=700, optimal_range=1,
            levels=self._add_levels_melee_linear(base=2.0, per=0.5, sp_base=10, sp_per=2,
                                                  cast_ms=0, delay_ms=1000),
        ))

        # Cart Revolution
        self._add_skill(Skill(
            "Cart Revolution", skill_id=251, element="neutral",
            cast_time_ms=0, after_cast_delay_ms=1500, delay_ms=1500,
            cooldown_ms=3000, sp_cost=25, range=1, target_type="aoe",
            damage_type="melee", aoe_radius=3, min_level=1, required_job="merchant",
            tags=["neutral", "aoe", "medium", "physical"], damage_ratio=350,
            optimal_range=1,
            levels=self._add_levels_melee_linear(base=1.5, per=0.2, sp_base=12, sp_per=3,
                                                  cast_ms=0, delay_ms=1500),
        ))

        # Overcharge (passive)
        self._add_skill(Skill(
            "Overcharge", skill_id=252, element="neutral",
            cast_time_ms=0, after_cast_delay_ms=0, delay_ms=0,
            cooldown_ms=0, sp_cost=0, range=0, target_type="self",
            damage_type="melee", min_level=1, required_job="merchant",
            tags=["passive", "economy"], damage_ratio=0, optimal_range=0,
        ))

    def _add_skill(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def _load_rotations(self) -> None:
        """Load rotation templates."""
        # ── Mage Rotations ──
        self._rotations["mage_fire_combo"] = SkillRotation(
            name="mage_fire_combo",
            steps=[
                RotationStep("Fire Bolt", condition="sp > 20", priority=90),
                RotationStep("Fire Ball", condition="aggro > 2 and sp > 30", priority=80),
                RotationStep("Fire Bolt", condition="sp > 15", priority=70),
                RotationStep("Fire Wall", condition="aggro > 3 and sp > 35", priority=60),
            ],
            priority=80,
            target_condition="element:earth",
            description="Fire combo for earth-weak monsters. Fast single-target with AoE option.",
        )
        self._rotations["mage_cold_combo"] = SkillRotation(
            name="mage_cold_combo",
            steps=[
                RotationStep("Cold Bolt", condition="sp > 20", priority=90),
                RotationStep("Frost Diver", condition="sp > 15", priority=80),
                RotationStep("Frost Nova", condition="aggro > 2 and sp > 25", priority=70),
                RotationStep("Storm Gust", condition="aggro > 4 and sp > 80", priority=60),
            ],
            priority=80,
            target_condition="element:fire",
            description="Cold combo for fire-weak monsters. Freeze + high damage.",
        )
        self._rotations["mage_wind_combo"] = SkillRotation(
            name="mage_wind_combo",
            steps=[
                RotationStep("Lightning Bolt", condition="sp > 20", priority=90),
                RotationStep("Thunderstorm", condition="aggro > 2 and sp > 40", priority=80),
                RotationStep("Lord of Vermilion", condition="aggro > 4 and sp > 85", priority=60),
            ],
            priority=80,
            target_condition="element:water",
            description="Wind combo for water-weak monsters.",
        )
        self._rotations["mage_aoe_clear"] = SkillRotation(
            name="mage_aoe_clear",
            steps=[
                RotationStep("Thunderstorm", condition="sp > 40", priority=90),
                RotationStep("Heaven's Drive", condition="sp > 50", priority=80),
                RotationStep("Lord of Vermilion", condition="sp > 85", priority=70),
                RotationStep("Meteor Storm", condition="sp > 90", priority=60),
            ],
            priority=70,
            target_condition="aggro > 3",
            description="AoE clear for grouped monsters.",
        )
        self._rotations["mage_undead_combo"] = SkillRotation(
            name="mage_undead_combo",
            steps=[
                RotationStep("Napalm Beat", condition="sp > 15", priority=90),
                RotationStep("Soul Strike", condition="sp > 20", priority=80),
                RotationStep("Heaven's Drive", condition="aggro > 2 and sp > 50", priority=70),
            ],
            priority=85,
            target_condition="race:undead",
            description="Undead-killer rotation. Napalm Beat and Soul Strike are super effective.",
        )

        # ── Archer Rotations ──
        self._rotations["archer_single"] = SkillRotation(
            name="archer_single",
            steps=[
                RotationStep("Double Strafe", condition="sp > 10", priority=90),
                RotationStep("Double Strafe", condition="sp > 8", priority=80),
                RotationStep("Double Strafe", condition="sp > 5", priority=70),
            ],
            priority=80,
            target_condition="single",
            description="Single-target DPS. Spam Double Strafe for maximum damage.",
        )
        self._rotations["archer_aoe"] = SkillRotation(
            name="archer_aoe",
            steps=[
                RotationStep("Arrow Shower", condition="sp > 20", priority=90),
                RotationStep("Arrow Shower", condition="sp > 15", priority=80),
            ],
            priority=70,
            target_condition="aggro > 2",
            description="AoE clear with Arrow Shower.",
        )
        self._rotations["archer_buff"] = SkillRotation(
            name="archer_buff",
            steps=[
                RotationStep("Improve Concentration", condition="not buffed", priority=100),
            ],
            priority=50,
            target_condition="always",
            description="Keep Improve Concentration active.",
        )

        # ── Swordman Rotations ──
        self._rotations["swordman_melee"] = SkillRotation(
            name="swordman_melee",
            steps=[
                RotationStep("Bash", condition="sp > 8", priority=90),
                RotationStep("Magnum Break", condition="aggro > 2 and sp > 15", priority=80),
                RotationStep("Bowling Bash", condition="aggro > 3 and sp > 20", priority=70),
                RotationStep("Bash", condition="sp > 5", priority=60),
            ],
            priority=80,
            target_condition="melee",
            description="Melee DPS rotation. Bash single, Magnum Break/Bowling Bash for groups.",
        )
        self._rotations["swordman_tank"] = SkillRotation(
            name="swordman_tank",
            steps=[
                RotationStep("Endure", condition="not buffed", priority=100),
                RotationStep("Provoke", condition="aggro > 0", priority=90),
                RotationStep("Bash", condition="sp > 8", priority=70),
            ],
            priority=60,
            target_condition="aggro > 0",
            description="Tank rotation. Keep Endure up, Provoke to maintain aggro.",
        )

        # ── Thief Rotations ──
        self._rotations["thief_single"] = SkillRotation(
            name="thief_single",
            steps=[
                RotationStep("Double Attack", condition="always", priority=100),
                RotationStep("Sonic Blow", condition="sp > 25 and target.hp > 30%", priority=80),
                RotationStep("Throw Sand", condition="aggro > 2 and sp > 8", priority=60),
            ],
            priority=80,
            target_condition="single",
            description="Thief single-target. Double Attack passive, Sonic Blow finisher.",
        )
        self._rotations["thief_stealth"] = SkillRotation(
            name="thief_stealth",
            steps=[
                RotationStep("Hide", condition="hp < 30% or aggro > 5", priority=100),
            ],
            priority=50,
            target_condition="danger",
            description="Emergency stealth when in danger.",
        )

        # ── Acolyte Rotations ──
        self._rotations["acolyte_undead"] = SkillRotation(
            name="acolyte_undead",
            steps=[
                RotationStep("Holy Light", condition="sp > 20", priority=90),
                RotationStep("Turn Undead", condition="sp > 25", priority=80),
                RotationStep("Heal", condition="sp > 15 and target.race == undead", priority=70),
            ],
            priority=85,
            target_condition="race:undead",
            description="Undead-killer. Holy Light + Turn Undead + Heal (damages undead).",
        )
        self._rotations["acolyte_support"] = SkillRotation(
            name="acolyte_support",
            steps=[
                RotationStep("Blessing", condition="not buffed", priority=100),
                RotationStep("Increase Agility", condition="not buffed", priority=90),
                RotationStep("Heal", condition="hp < 70%", priority=80),
            ],
            priority=60,
            target_condition="always",
            description="Support rotation. Keep buffs up, heal when needed.",
        )

        # ── Merchant Rotations ──
        self._rotations["merchant_single"] = SkillRotation(
            name="merchant_single",
            steps=[
                RotationStep("Mammonite", condition="sp > 20 and zeny > 100", priority=90),
                RotationStep("Cart Revolution", condition="aggro > 2 and sp > 25", priority=80),
            ],
            priority=80,
            target_condition="single",
            description="Merchant DPS. Mammonite for single, Cart Revolution for groups.",
        )

    # ── Per-level data queries ──

    def get_skill_level_info(self, skill_name: str, level: int) -> SkillLevel | None:
        """Get the per-level info for a skill at a given level (1-10).
        
        If the skill has levels data, returns the level entry.
        Falls back to the backward-compat fields (cast_time_ms, delay_ms, damage_ratio).
        """
        skill = self._skills.get(skill_name)
        if not skill:
            return None
        if skill.levels and 1 <= level <= len(skill.levels):
            return skill.levels[level - 1]
        # Fallback: construct from the Skill's base fields (backwards compat)
        return SkillLevel(
            level=level,
            damage_ratio=skill.damage_ratio,
            sp_cost=skill.sp_cost,
            cast_time_ms=skill.cast_time_ms,
            after_cast_delay_ms=skill.after_cast_delay_ms or skill.delay_ms,
            hits_per_cast=skill.hits_per_cast,
        )

    def get_skill_total_damage_ratio(self, skill_name: str, level: int) -> int:
        """Get total damage ratio = damage_ratio * hits_per_cast at given level."""
        info = self.get_skill_level_info(skill_name, level)
        if not info:
            return 0
        return info.damage_ratio * info.hits_per_cast

    def get_cast_time_with_dex(self, skill_name: str, level: int, dex: int) -> int:
        """Calculate actual cast time with DEX reduction.
        
        RO formula: actual_cast_time = base_cast_time * (1 - DEX/150)
        Fixed-cast skills (Bowling Bash, Sonic Blow) are unaffected.
        """
        skill = self._skills.get(skill_name)
        if not skill:
            return 0
        info = self.get_skill_level_info(skill_name, level)
        if not info:
            return 0
        
        base_cast = info.cast_time_ms
        
        # Fixed-cast skills are unaffected by DEX
        if "fixed_cast" in skill.tags:
            return base_cast
        
        if base_cast <= 0:
            return 0
        
        # DEX reduction: cast_time * (1 - DEX/150), with minimum 10% of base
        reduction = 1.0 - min(dex / 150.0, 1.0)
        actual = int(base_cast * reduction)
        return max(actual, int(base_cast * 0.1))  # Minimum 10% of base cast

    def get_skill_effective_sp_cost(self, skill_name: str, level: int) -> int:
        """Get effective SP cost at a given level."""
        info = self.get_skill_level_info(skill_name, level)
        if not info:
            skill = self._skills.get(skill_name)
            return skill.sp_cost if skill else 0
        return info.sp_cost

    # ── Public API ──

    def get_skill(self, name: str) -> Skill | None:
        with self._lock:
            return self._skills.get(name)

    def get_skills_by_class(self, job_class: str) -> list[Skill]:
        with self._lock:
            return [s for s in self._skills.values() if s.required_job == job_class.lower()]

    def get_skills_by_element(self, element: str) -> list[Skill]:
        with self._lock:
            return [s for s in self._skills.values() if s.element == element.lower()]

    def get_skills_by_tag(self, tag: str) -> list[Skill]:
        with self._lock:
            return [s for s in self._skills.values() if tag in s.tags]

    def get_best_skill_against(
        self,
        target_element: str,
        target_size: str = "medium",
        target_race: str = "formless",
        available_skills: list[str] | None = None,
        current_sp: int = 100,
        current_hp_pct: float = 1.0,
        skill_level: int = 10,
        dex: int = 1,
    ) -> Skill | None:
        """Get the best skill to use against a target."""
        with self._lock:
            candidates = list(self._skills.values())
            if available_skills:
                candidates = [s for s in candidates if s.name in available_skills]

            if not candidates:
                return None

            # Score each skill
            best: Skill | None = None
            best_score = -9999.0

            for skill in candidates:
                score = 0.0

                # Elemental advantage
                from ai_sidecar.combat.elemental_matrix import get_elemental_matrix
                em = get_elemental_matrix()
                mult = em.get_elemental_multiplier(skill.element, target_element)
                score += mult * 0.5  # 200% = +100 points

                # SP efficiency
                sp_cost = self.get_skill_effective_sp_cost(skill.name, skill_level)
                if sp_cost > 0 and current_sp > 0:
                    sp_ratio = sp_cost / current_sp
                    if sp_ratio > 0.5:
                        score -= 50  # Too expensive
                    elif sp_ratio > 0.3:
                        score -= 20
                    else:
                        score += 10

                # Cast time with DEX reduction
                actual_cast = self.get_cast_time_with_dex(skill.name, skill_level, dex)
                if actual_cast <= 500:
                    score += 20  # Fast cast
                elif actual_cast <= 2000:
                    score += 10
                else:
                    score -= 10  # Slow cast

                # Total damage (damage_ratio * hits_per_cast)
                total_dmg = self.get_skill_total_damage_ratio(skill.name, skill_level)
                score += total_dmg / 100.0  # 1600% Storm Gust = +16 points

                # AoE bonus for groups
                if skill.target_type == "aoe" and skill.aoe_radius >= 5:
                    score += 15

                # Tag bonuses
                if "high_damage" in skill.tags:
                    score += 20
                if "very_fast" in skill.tags:
                    score += 15

                if score > best_score:
                    best_score = score
                    best = skill

            return best

    def get_rotation_for_target(
        self,
        target_info: dict[str, Any],
        available_skills: list[str] | None = None,
        current_sp: int = 100,
    ) -> list[Skill]:
        """Get the best rotation for a target."""
        target_element = target_info.get("element", "neutral")
        target_race = target_info.get("race", "formless")
        aggro = target_info.get("aggro", 1)
        is_boss = target_info.get("is_boss", False)

        with self._lock:
            # Find matching rotations
            candidates: list[SkillRotation] = []
            for rot in self._rotations.values():
                tc = rot.target_condition
                if not tc:
                    candidates.append(rot)
                    continue
                if tc.startswith("element:") and tc.split(":")[1] == target_element:
                    candidates.append(rot)
                elif tc.startswith("race:") and tc.split(":")[1] == target_race:
                    candidates.append(rot)
                elif tc == "aggro > 3" and aggro > 3:
                    candidates.append(rot)
                elif tc == "aggro > 2" and aggro > 2:
                    candidates.append(rot)
                elif tc == "aggro > 0" and aggro > 0:
                    candidates.append(rot)
                elif tc == "single" and aggro <= 2:
                    candidates.append(rot)
                elif tc == "melee":
                    candidates.append(rot)
                elif tc == "always":
                    candidates.append(rot)
                elif tc == "danger" and target_info.get("hp_pct", 1.0) < 0.3:
                    candidates.append(rot)
                elif tc == "boss" and is_boss:
                    candidates.append(rot)

            if not candidates:
                return []

            # Sort by priority
            candidates.sort(key=lambda r: -r.priority)

            # Get skills from the best rotation
            result: list[Skill] = []
            for step in candidates[0].steps:
                skill = self._skills.get(step.skill_name)
                if skill and (not available_skills or skill.name in available_skills):
                    # Check condition
                    if step.condition == "always":
                        result.append(skill)
                    elif step.condition.startswith("sp >"):
                        min_sp = int(step.condition.split(">")[1].strip())
                        if current_sp >= min_sp:
                            result.append(skill)
                    elif step.condition == "not buffed":
                        result.append(skill)
                    elif step.condition.startswith("hp <"):
                        threshold = int(step.condition.split("<")[1].strip().replace("%", ""))
                        if current_sp > 0:  # simplified
                            result.append(skill)
                    else:
                        result.append(skill)

            return result

    def get_next_skill_in_rotation(
        self,
        rotation_name: str,
        current_skill_index: int = 0,
        cooldowns: dict[str, int] | None = None,
    ) -> Skill | None:
        """Get the next skill in a rotation."""
        cooldowns = cooldowns or {}
        with self._lock:
            rot = self._rotations.get(rotation_name)
            if not rot:
                return None
            for i in range(current_skill_index, len(rot.steps)):
                step = rot.steps[i]
                skill = self._skills.get(step.skill_name)
                if skill and skill.name not in cooldowns:
                    return skill
            # Wrap around
            for step in rot.steps:
                skill = self._skills.get(step.skill_name)
                if skill and skill.name not in cooldowns:
                    return skill
            return None

    def get_skill_chain(
        self,
        initial_skill: str,
        target_info: dict[str, Any],
        available_skills: list[str] | None = None,
    ) -> list[Skill]:
        """Get a chain of skills starting from an initial skill."""
        result: list[Skill] = []
        with self._lock:
            skill = self._skills.get(initial_skill)
            if not skill:
                return result
            result.append(skill)

            # Find a rotation that uses this skill
            for rot in self._rotations.values():
                for i, step in enumerate(rot.steps):
                    if step.skill_name == initial_skill:
                        # Add remaining skills in rotation
                        for j in range(i + 1, len(rot.steps)):
                            next_skill = self._skills.get(rot.steps[j].skill_name)
                            if next_skill and (not available_skills or next_skill.name in available_skills):
                                result.append(next_skill)
                        return result
            return result

    def is_skill_available(self, skill_name: str, current_sp: int = 0, cooldowns: dict[str, int] | None = None) -> bool:
        cooldowns = cooldowns or {}
        with self._lock:
            skill = self._skills.get(skill_name)
            if not skill:
                return False
            if skill.sp_cost > current_sp:
                return False
            if skill.name in cooldowns:
                return False
            return True

    def get_sp_cost(self, skill_name: str, level: int = 10) -> int:
        info = self.get_skill_level_info(skill_name, level)
        if info:
            return info.sp_cost
        with self._lock:
            skill = self._skills.get(skill_name)
            return skill.sp_cost if skill else 0

    def get_cast_time(self, skill_name: str, level: int = 10) -> int:
        info = self.get_skill_level_info(skill_name, level)
        if info:
            return info.cast_time_ms
        with self._lock:
            skill = self._skills.get(skill_name)
            return skill.cast_time_ms if skill else 0

    def get_after_cast_delay(self, skill_name: str, level: int = 10) -> int:
        """Get the FIXED after-cast delay (NOT reducible by DEX)."""
        info = self.get_skill_level_info(skill_name, level)
        if info:
            return info.after_cast_delay_ms
        with self._lock:
            skill = self._skills.get(skill_name)
            return skill.after_cast_delay_ms or skill.delay_ms if skill else 0

    def get_cooldown_remaining(self, skill_name: str, cooldowns: dict[str, int] | None = None) -> int:
        cooldowns = cooldowns or {}
        return cooldowns.get(skill_name, 0)

    def get_recommended_rotation(
        self,
        job_class: str,
        target_element: str = "neutral",
        party_size: int = 1,
        has_aoe: bool = False,
    ) -> str | None:
        """Get the name of the recommended rotation."""
        with self._lock:
            candidates: list[SkillRotation] = []
            for rot in self._rotations.values():
                tc = rot.target_condition
                if tc.startswith("element:") and tc.split(":")[1] == target_element:
                    candidates.append(rot)
                elif tc == "aggro > 3" and has_aoe:
                    candidates.append(rot)
                elif tc == "single" and not has_aoe:
                    candidates.append(rot)
                elif tc == "always":
                    candidates.append(rot)

            if not candidates:
                return None
            candidates.sort(key=lambda r: -r.priority)
            return candidates[0].name

    def get_optimal_range_for_skill(self, skill_name: str) -> int:
        """Get the optimal engagement range for a skill."""
        skill = self._skills.get(skill_name)
        if not skill:
            return 1
        return skill.optimal_range

    def get_all_skills(self) -> list[Skill]:
        with self._lock:
            return list(self._skills.values())

    def get_all_rotations(self) -> list[SkillRotation]:
        with self._lock:
            return list(self._rotations.values())

    def get_rotations_for_class(self, job_class: str) -> list[SkillRotation]:
        with self._lock:
            return [r for r in self._rotations.values() if job_class.lower() in r.name]


# ── Global Singleton ──

_skill_system: SkillRotationSystem | None = None
_skill_system_lock = RLock()


def get_skill_rotation_system() -> SkillRotationSystem:
    global _skill_system
    with _skill_system_lock:
        if _skill_system is None:
            _skill_system = SkillRotationSystem()
        return _skill_system
