"""
Skill Rotation System — intelligent skill selection and rotation for RO combat.

A pro player doesn't spam the same skill. They have a mental rotation:
Fire Bolt on Earth monsters, Cold Bolt on Fire monsters, Storm Gust on groups.
This module encodes that knowledge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """A single skill."""
    name: str
    skill_id: int = 0
    element: str = "neutral"
    cast_time_ms: int = 0
    delay_ms: int = 0
    cooldown_ms: int = 0
    sp_cost: int = 0
    range: int = 1
    target_type: str = "single"  # single, aoe, self
    damage_type: str = "melee"  # melee, magic, ranged
    aoe_radius: int = 0
    min_level: int = 1
    required_job: str = "novice"
    tags: list[str] = field(default_factory=list)
    damage_ratio: int = 100  # % of ATK/MATK


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


class SkillRotationSystem:
    """Intelligent skill selection and rotation."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._skills: dict[str, Skill] = {}
        self._rotations: dict[str, SkillRotation] = {}
        self._load_skills()
        self._load_rotations()

    def _load_skills(self) -> None:
        """Load skill data for major classes."""
        # ── Mage Skills ──
        self._add_skill(Skill("Fire Bolt", skill_id=1, element="fire", cast_time_ms=1500, delay_ms=500, cooldown_ms=0, sp_cost=15, range=9, target_type="single", damage_type="magic", min_level=1, required_job="mage", tags=["fire", "single", "fast"], damage_ratio=100))
        self._add_skill(Skill("Cold Bolt", skill_id=2, element="water", cast_time_ms=1500, delay_ms=500, cooldown_ms=0, sp_cost=15, range=9, target_type="single", damage_type="magic", min_level=1, required_job="mage", tags=["water", "single", "fast"], damage_ratio=100))
        self._add_skill(Skill("Lightning Bolt", skill_id=3, element="wind", cast_time_ms=1500, delay_ms=500, cooldown_ms=0, sp_cost=20, range=9, target_type="single", damage_type="magic", min_level=1, required_job="mage", tags=["wind", "single", "fast"], damage_ratio=100))
        self._add_skill(Skill("Fire Ball", skill_id=4, element="fire", cast_time_ms=2000, delay_ms=1000, cooldown_ms=0, sp_cost=25, range=9, target_type="aoe", damage_type="magic", aoe_radius=3, min_level=1, required_job="mage", tags=["fire", "aoe", "medium"], damage_ratio=150))
        self._add_skill(Skill("Fire Wall", skill_id=5, element="fire", cast_time_ms=3000, delay_ms=1000, cooldown_ms=5000, sp_cost=30, range=5, target_type="aoe", damage_type="magic", aoe_radius=2, min_level=1, required_job="mage", tags=["fire", "aoe", "defensive", "dot"], damage_ratio=200))
        self._add_skill(Skill("Frost Diver", skill_id=6, element="water", cast_time_ms=1000, delay_ms=500, cooldown_ms=0, sp_cost=12, range=9, target_type="single", damage_type="magic", min_level=1, required_job="mage", tags=["water", "single", "fast", "freeze"], damage_ratio=80))
        self._add_skill(Skill("Frost Nova", skill_id=7, element="water", cast_time_ms=2000, delay_ms=1000, cooldown_ms=3000, sp_cost=20, range=0, target_type="aoe", damage_type="magic", aoe_radius=4, min_level=1, required_job="mage", tags=["water", "aoe", "freeze"], damage_ratio=120))
        self._add_skill(Skill("Thunderstorm", skill_id=8, element="wind", cast_time_ms=3000, delay_ms=1500, cooldown_ms=3000, sp_cost=35, range=9, target_type="aoe", damage_type="magic", aoe_radius=5, min_level=1, required_job="mage", tags=["wind", "aoe", "slow"], damage_ratio=180))
        self._add_skill(Skill("Heaven's Drive", skill_id=9, element="neutral", cast_time_ms=4000, delay_ms=2000, cooldown_ms=5000, sp_cost=45, range=9, target_type="aoe", damage_type="magic", aoe_radius=5, min_level=1, required_job="wizard", tags=["neutral", "aoe", "slow", "high_damage"], damage_ratio=250))
        self._add_skill(Skill("Storm Gust", skill_id=10, element="water", cast_time_ms=5000, delay_ms=3000, cooldown_ms=5000, sp_cost=80, range=9, target_type="aoe", damage_type="magic", aoe_radius=7, min_level=1, required_job="wizard", tags=["water", "aoe", "very_slow", "high_damage", "freeze"], damage_ratio=400))
        self._add_skill(Skill("Meteor Storm", skill_id=11, element="fire", cast_time_ms=6000, delay_ms=3000, cooldown_ms=5000, sp_cost=90, range=9, target_type="aoe", damage_type="magic", aoe_radius=7, min_level=1, required_job="wizard", tags=["fire", "aoe", "very_slow", "high_damage", "stun"], damage_ratio=500))
        self._add_skill(Skill("Lord of Vermilion", skill_id=12, element="wind", cast_time_ms=5000, delay_ms=3000, cooldown_ms=5000, sp_cost=85, range=9, target_type="aoe", damage_type="magic", aoe_radius=7, min_level=1, required_job="wizard", tags=["wind", "aoe", "very_slow", "high_damage"], damage_ratio=450))
        self._add_skill(Skill("Napalm Beat", skill_id=13, element="neutral", cast_time_ms=1000, delay_ms=500, cooldown_ms=0, sp_cost=10, range=9, target_type="single", damage_type="magic", min_level=1, required_job="mage", tags=["neutral", "single", "fast", "undead"], damage_ratio=120))
        self._add_skill(Skill("Soul Strike", skill_id=14, element="neutral", cast_time_ms=1500, delay_ms=500, cooldown_ms=0, sp_cost=18, range=9, target_type="single", damage_type="magic", min_level=1, required_job="mage", tags=["neutral", "single", "fast", "undead", "ghost"], damage_ratio=150))
        self._add_skill(Skill("Safety Wall", skill_id=15, element="neutral", cast_time_ms=2000, delay_ms=1000, cooldown_ms=10000, sp_cost=20, range=3, target_type="self", damage_type="magic", min_level=1, required_job="mage", tags=["defensive", "support"], damage_ratio=0))

        # ── Archer Skills ──
        self._add_skill(Skill("Double Strafe", skill_id=50, element="neutral", cast_time_ms=0, delay_ms=200, cooldown_ms=0, sp_cost=8, range=9, target_type="single", damage_type="ranged", min_level=1, required_job="archer", tags=["neutral", "single", "very_fast", "physical"], damage_ratio=190))
        self._add_skill(Skill("Arrow Shower", skill_id=51, element="neutral", cast_time_ms=0, delay_ms=1000, cooldown_ms=2000, sp_cost=15, range=9, target_type="aoe", damage_type="ranged", aoe_radius=3, min_level=1, required_job="archer", tags=["neutral", "aoe", "fast", "physical", "knockback"], damage_ratio=100))
        self._add_skill(Skill("Improve Concentration", skill_id=52, element="neutral", cast_time_ms=0, delay_ms=0, cooldown_ms=30000, sp_cost=20, range=0, target_type="self", damage_type="ranged", min_level=1, required_job="archer", tags=["buff", "support"], damage_ratio=0))

        # ── Swordman Skills ──
        self._add_skill(Skill("Bash", skill_id=100, element="neutral", cast_time_ms=0, delay_ms=500, cooldown_ms=0, sp_cost=5, range=1, target_type="single", damage_type="melee", min_level=1, required_job="swordman", tags=["neutral", "single", "very_fast", "physical", "stun"], damage_ratio=150))
        self._add_skill(Skill("Magnum Break", skill_id=101, element="fire", cast_time_ms=0, delay_ms=1000, cooldown_ms=3000, sp_cost=12, range=1, target_type="aoe", damage_type="melee", aoe_radius=3, min_level=1, required_job="swordman", tags=["fire", "aoe", "fast", "physical"], damage_ratio=200))
        self._add_skill(Skill("Bowling Bash", skill_id=102, element="neutral", cast_time_ms=0, delay_ms=1000, cooldown_ms=2000, sp_cost=15, range=1, target_type="aoe", damage_type="melee", aoe_radius=3, min_level=1, required_job="knight", tags=["neutral", "aoe", "fast", "physical", "high_damage"], damage_ratio=300))
        self._add_skill(Skill("Provoke", skill_id=103, element="neutral", cast_time_ms=0, delay_ms=500, cooldown_ms=5000, sp_cost=3, range=9, target_type="single", damage_type="melee", min_level=1, required_job="swordman", tags=["support", "taunt"], damage_ratio=0))
        self._add_skill(Skill("Endure", skill_id=104, element="neutral", cast_time_ms=0, delay_ms=0, cooldown_ms=30000, sp_cost=10, range=0, target_type="self", damage_type="melee", min_level=1, required_job="swordman", tags=["buff", "defensive"], damage_ratio=0))

        # ── Thief Skills ──
        self._add_skill(Skill("Double Attack", skill_id=150, element="neutral", cast_time_ms=0, delay_ms=0, cooldown_ms=0, sp_cost=0, range=1, target_type="single", damage_type="melee", min_level=1, required_job="thief", tags=["neutral", "single", "passive", "physical"], damage_ratio=200))
        self._add_skill(Skill("Throw Sand", skill_id=151, element="neutral", cast_time_ms=0, delay_ms=500, cooldown_ms=3000, sp_cost=5, range=5, target_type="single", damage_type="ranged", min_level=1, required_job="thief", tags=["neutral", "single", "fast", "blind"], damage_ratio=0))
        self._add_skill(Skill("Hide", skill_id=152, element="neutral", cast_time_ms=0, delay_ms=0, cooldown_ms=1000, sp_cost=5, range=0, target_type="self", damage_type="melee", min_level=1, required_job="thief", tags=["defensive", "stealth"], damage_ratio=0))
        self._add_skill(Skill("Sonic Blow", skill_id=153, element="neutral", cast_time_ms=0, delay_ms=1000, cooldown_ms=2000, sp_cost=20, range=1, target_type="single", damage_type="melee", min_level=1, required_job="assassin", tags=["neutral", "single", "fast", "physical", "high_damage", "poison"], damage_ratio=600))

        # ── Acolyte Skills ──
        self._add_skill(Skill("Heal", skill_id=200, element="holy", cast_time_ms=1000, delay_ms=500, cooldown_ms=0, sp_cost=10, range=9, target_type="single", damage_type="magic", min_level=1, required_job="acolyte", tags=["holy", "single", "fast", "heal", "undead"], damage_ratio=100))
        self._add_skill(Skill("Blessing", skill_id=201, element="neutral", cast_time_ms=2000, delay_ms=1000, cooldown_ms=0, sp_cost=15, range=9, target_type="single", damage_type="magic", min_level=1, required_job="acolyte", tags=["buff", "support"], damage_ratio=0))
        self._add_skill(Skill("Increase Agility", skill_id=202, element="neutral", cast_time_ms=2000, delay_ms=1000, cooldown_ms=0, sp_cost=15, range=9, target_type="single", damage_type="magic", min_level=1, required_job="acolyte", tags=["buff", "support"], damage_ratio=0))
        self._add_skill(Skill("Teleport", skill_id=203, element="neutral", cast_time_ms=1000, delay_ms=0, cooldown_ms=0, sp_cost=10, range=0, target_type="self", damage_type="magic", min_level=1, required_job="acolyte", tags=["utility", "movement"], damage_ratio=0))
        self._add_skill(Skill("Holy Light", skill_id=204, element="holy", cast_time_ms=1500, delay_ms=500, cooldown_ms=0, sp_cost=15, range=9, target_type="single", damage_type="magic", min_level=1, required_job="acolyte", tags=["holy", "single", "fast", "undead", "demon"], damage_ratio=120))
        self._add_skill(Skill("Turn Undead", skill_id=205, element="holy", cast_time_ms=2000, delay_ms=1000, cooldown_ms=3000, sp_cost=20, range=9, target_type="single", damage_type="magic", min_level=1, required_job="priest", tags=["holy", "single", "medium", "undead", "instant_kill"], damage_ratio=200))

        # ── Merchant Skills ──
        self._add_skill(Skill("Mammonite", skill_id=250, element="neutral", cast_time_ms=0, delay_ms=1000, cooldown_ms=0, sp_cost=15, range=1, target_type="single", damage_type="melee", min_level=1, required_job="merchant", tags=["neutral", "single", "fast", "physical", "high_damage", "costs_zeny"], damage_ratio=400))
        self._add_skill(Skill("Cart Revolution", skill_id=251, element="neutral", cast_time_ms=0, delay_ms=1500, cooldown_ms=3000, sp_cost=20, range=1, target_type="aoe", damage_type="melee", aoe_radius=3, min_level=1, required_job="merchant", tags=["neutral", "aoe", "medium", "physical"], damage_ratio=150))
        self._add_skill(Skill("Overcharge", skill_id=252, element="neutral", cast_time_ms=0, delay_ms=0, cooldown_ms=0, sp_cost=0, range=0, target_type="self", damage_type="melee", min_level=1, required_job="merchant", tags=["passive", "economy"], damage_ratio=0))

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
                if skill.sp_cost > 0 and current_sp > 0:
                    sp_ratio = skill.sp_cost / current_sp
                    if sp_ratio > 0.5:
                        score -= 50  # Too expensive
                    elif sp_ratio > 0.3:
                        score -= 20
                    else:
                        score += 10

                # Speed preference
                if skill.cast_time_ms <= 500:
                    score += 20  # Fast cast
                elif skill.cast_time_ms <= 2000:
                    score += 10
                else:
                    score -= 10  # Slow cast

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

    def get_sp_cost(self, skill_name: str) -> int:
        with self._lock:
            skill = self._skills.get(skill_name)
            return skill.sp_cost if skill else 0

    def get_cast_time(self, skill_name: str) -> int:
        with self._lock:
            skill = self._skills.get(skill_name)
            return skill.cast_time_ms if skill else 0

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
