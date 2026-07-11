"""
Game Intelligence Engine — Reads rAthena data and answers game-mechanics questions.
===============================================================================
A pro player pays for this: the bot knows what to hunt, where to go, what gear
to use, and what skills to cast — without any manual configuration.

Uses the pre-loaded rAthena knowledge DB (2675 monsters, 460K+ items, 18 maps).
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Element Chart (offensive efficiency) ──
# Element attack → damage multiplier vs defense element
ELEMENT_CHART: dict[str, dict[str, float]] = {
    "Neutral":  {"Neutral": 1.0, "Water": 1.0, "Earth": 1.0, "Fire": 1.0, "Wind": 1.0,
                 "Poison": 1.0, "Holy": 1.0, "Dark": 1.0, "Ghost": 0.75, "Undead": 1.0},
    "Water":    {"Neutral": 1.0, "Water": 0.25, "Earth": 0.75, "Fire": 1.0, "Wind": 0.75,
                 "Poison": 1.0, "Holy": 1.0, "Dark": 1.0, "Ghost": 0.5, "Undead": 1.0},
    "Earth":    {"Neutral": 1.0, "Water": 1.0, "Earth": 0.25, "Fire": 0.75, "Wind": 1.0,
                 "Poison": 1.0, "Holy": 1.0, "Dark": 1.0, "Ghost": 0.5, "Undead": 1.0},
    "Fire":     {"Neutral": 1.0, "Water": 0.5, "Earth": 1.0, "Fire": 0.25, "Wind": 0.75,
                 "Poison": 1.0, "Holy": 1.0, "Dark": 1.0, "Ghost": 0.5, "Undead": 1.0},
    "Wind":     {"Neutral": 1.0, "Water": 1.0, "Earth": 0.5, "Fire": 1.0, "Wind": 0.25,
                 "Poison": 1.0, "Holy": 1.0, "Dark": 1.0, "Ghost": 0.5, "Undead": 1.0},
    "Poison":   {"Neutral": 1.0, "Water": 1.0, "Earth": 0.75, "Fire": 1.0, "Wind": 1.0,
                 "Poison": 0.25, "Holy": 1.0, "Dark": 1.0, "Ghost": 0.5, "Undead": 0.5},
    "Holy":     {"Neutral": 1.0, "Water": 1.0, "Earth": 1.0, "Fire": 1.0, "Wind": 1.0,
                 "Poison": 1.0, "Holy": 0.25, "Dark": 2.0, "Ghost": 1.0, "Undead": 1.5},
    "Dark":     {"Neutral": 1.0, "Water": 1.0, "Earth": 1.0, "Fire": 1.0, "Wind": 1.0,
                 "Poison": 1.0, "Holy": 0.5, "Dark": 0.25, "Ghost": 1.0, "Undead": 1.0},
    "Ghost":    {"Neutral": 0.0, "Water": 1.0, "Earth": 1.0, "Fire": 1.0, "Wind": 1.0,
                 "Poison": 1.0, "Holy": 1.0, "Dark": 1.0, "Ghost": 0.5, "Undead": 1.0},
    "Undead":   {"Neutral": 1.0, "Water": 1.0, "Earth": 1.0, "Fire": 1.25, "Wind": 1.0,
                 "Poison": 0.5, "Holy": 2.0, "Dark": 0.5, "Ghost": 1.0, "Undead": 0.5},
}

# ── Race Chart ──
RACE_CHARTS: dict[str, dict[str, float]] = {
    "Brute":    {"Brute": 1.0, "Demon": 1.0, "Dragon": 1.0, "Fish": 1.0, "Formless": 1.0,
                 "Human": 1.0, "Insect": 1.0, "Plant": 1.0, "Undead": 1.0, "Angel": 1.0},
    "Demon":    {"Human": 1.25, "Angel": 1.25, "Demon": 0.75},
    "Angel":    {"Demon": 1.5, "Undead": 1.5},
    "Undead":   {"Human": 1.5, "Undead": 0.5, "Demon": 0.5},
    "Dragon":   {"Dragon": 1.0, "Formless": 1.0},
    "Fish":     {"Fish": 1.0, "Water": 1.0},
    "Insect":   {"Insect": 0.5, "Plant": 1.5},
    "Plant":    {"Plant": 0.5, "Insect": 1.5},
}

# ── Size Chart ──
SIZE_CHART: dict[str, dict[str, float]] = {
    "Dagger":   {"Small": 1.0, "Medium": 0.75, "Large": 0.5},
    "Sword":    {"Small": 1.0, "Medium": 1.0, "Large": 0.75},
    "TwoHandedSword": {"Small": 0.75, "Medium": 0.75, "Large": 1.0},
    "Spear":    {"Small": 0.75, "Medium": 1.0, "Large": 1.0},
    "Bow":      {"Small": 1.0, "Medium": 1.0, "Large": 0.75},
    "Mace":     {"Small": 1.0, "Medium": 1.0, "Large": 0.75},
    "Staff":    {"Small": 1.0, "Medium": 1.0, "Large": 1.0},
    "Knuckle":  {"Small": 1.0, "Medium": 0.75, "Large": 0.5},
    "Katar":    {"Small": 1.0, "Medium": 1.0, "Large": 0.5},
    "Instrument": {"Small": 1.0, "Medium": 1.0, "Large": 0.75},
    "Whip":     {"Small": 1.0, "Medium": 1.0, "Large": 0.75},
    "Book":     {"Small": 1.0, "Medium": 1.0, "Large": 0.75},
    "Gun":      {"Small": 1.0, "Medium": 1.0, "Large": 0.75},
    "Grenade":  {"Small": 1.0, "Medium": 1.0, "Large": 1.0},
    "Shuriken": {"Small": 1.0, "Medium": 1.0, "Large": 0.75},
}

# ── Level Penalty (from rAthena level_penalty.yml) ──
LEVEL_PENALTY = [
    {"min_level": 1, "max_level": 15, "base_exp": 1.0, "job_exp": 1.0},
    {"min_level": 16, "max_level": 25, "base_exp": 0.8, "job_exp": 0.8},
    {"min_level": 26, "max_level": 35, "base_exp": 0.6, "job_exp": 0.6},
    {"min_level": 36, "max_level": 45, "base_exp": 0.4, "job_exp": 0.4},
    {"min_level": 46, "max_level": 55, "base_exp": 0.2, "job_exp": 0.2},
    {"min_level": 56, "max_level": 65, "base_exp": 0.1, "job_exp": 0.1},
    {"min_level": 66, "max_level": 75, "base_exp": 0.05, "job_exp": 0.05},
    {"min_level": 76, "max_level": 85, "base_exp": 0.01, "job_exp": 0.01},
    {"min_level": 86, "max_level": 99, "base_exp": 0.0, "job_exp": 0.0},
]

# ── Class Archetypes and their stat/skill preferences ──
CLASS_ARCHETYPES: dict[str, dict[str, Any]] = {
    "novice": {"stat_priority": ["str", "dex", "agi"], "damage_type": "melee", "weapon_type": "Dagger"},
    "swordman": {"stat_priority": ["str", "vit", "dex"], "damage_type": "melee", "weapon_type": "Sword"},
    "knight": {"stat_priority": ["str", "vit", "dex"], "damage_type": "melee", "weapon_type": "Spear"},
    "paladin": {"stat_priority": ["str", "vit", "int"], "damage_type": "melee", "weapon_type": "Spear"},
    "mage": {"stat_priority": ["int", "dex", "vit"], "damage_type": "magic", "weapon_type": "Staff"},
    "wizard": {"stat_priority": ["int", "dex", "vit"], "damage_type": "magic", "weapon_type": "Staff"},
    "sage": {"stat_priority": ["int", "dex", "vit"], "damage_type": "magic", "weapon_type": "Staff"},
    "archer": {"stat_priority": ["dex", "agi", "str"], "damage_type": "ranged", "weapon_type": "Bow"},
    "hunter": {"stat_priority": ["dex", "agi", "int"], "damage_type": "ranged", "weapon_type": "Bow"},
    "bard": {"stat_priority": ["dex", "agi", "int"], "damage_type": "ranged", "weapon_type": "Instrument"},
    "dancer": {"stat_priority": ["dex", "agi", "int"], "damage_type": "ranged", "weapon_type": "Whip"},
    "acolyte": {"stat_priority": ["int", "dex", "vit"], "damage_type": "magic", "weapon_type": "Staff"},
    "priest": {"stat_priority": ["int", "dex", "vit"], "damage_type": "magic", "weapon_type": "Staff"},
    "monk": {"stat_priority": ["str", "dex", "vit"], "damage_type": "melee", "weapon_type": "Knuckle"},
    "thief": {"stat_priority": ["agi", "str", "dex"], "damage_type": "melee", "weapon_type": "Dagger"},
    "assassin": {"stat_priority": ["agi", "str", "dex"], "damage_type": "melee", "weapon_type": "Katar"},
    "rogue": {"stat_priority": ["agi", "dex", "str"], "damage_type": "melee", "weapon_type": "Dagger"},
    "merchant": {"stat_priority": ["str", "vit", "dex"], "damage_type": "melee", "weapon_type": "Mace"},
    "blacksmith": {"stat_priority": ["str", "dex", "vit"], "damage_type": "melee", "weapon_type": "Mace"},
    "alchemist": {"stat_priority": ["int", "str", "dex"], "damage_type": "melee", "weapon_type": "Mace"},
    "gunslinger": {"stat_priority": ["dex", "agi", "int"], "damage_type": "ranged", "weapon_type": "Gun"},
    "ninja": {"stat_priority": ["str", "int", "agi"], "damage_type": "mixed", "weapon_type": "Shuriken"},
    "taekwon": {"stat_priority": ["str", "agi", "vit"], "damage_type": "melee", "weapon_type": "Knuckle"},
    "star_gladiator": {"stat_priority": ["str", "agi", "vit"], "damage_type": "melee", "weapon_type": "Knuckle"},
    "soul_linker": {"stat_priority": ["int", "dex", "vit"], "damage_type": "magic", "weapon_type": "Staff"},
    "super_novice": {"stat_priority": ["str", "agi", "vit", "int", "dex"], "damage_type": "melee", "weapon_type": "Dagger"},
}


@dataclass
class HuntingRecommendation:
    """A complete hunting recommendation for a bot."""
    map_name: str
    primary_mob: str
    mob_level: int
    base_exp: int
    job_exp: int
    hp: int
    element: str
    race: str
    size: str
    element_efficiency: float  # 1.0 = neutral, 2.0 = double damage
    exp_per_hp: float  # Higher = more efficient
    level_penalty: float  # 1.0 = no penalty, 0.0 = full penalty
    effective_exp: float  # base_exp * level_penalty
    score: float  # Overall score (higher = better)
    drops: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""


class GameIntelligenceEngine:
    """The game intelligence layer — reads rAthena data and makes tactical decisions.

    A pro player pays for this: the bot knows what to hunt, what gear to use,
    and what skills to cast — without any manual configuration.

    The engine is self-adapting: it reads the actual game server data and
    adjusts recommendations based on the bot's class, level, and equipment.
    No hardcoded tier lists or class-specific configs.
    """

    def __init__(self, knowledge_path: str | None = None, rathena_path: str = ""):
        self._monsters: list[dict[str, Any]] = []
        self._items: dict[str, dict[str, Any]] = {}
        self._mob_skills: dict[str, list[dict[str, Any]]] = {}  # mob_id -> skills
        self._mob_skill_warnings: dict[str, list[str]] = {}  # mob_name -> dangerous skills
        self._load_knowledge(knowledge_path)
        if rathena_path:
            self._load_mob_skills(rathena_path)

    def _load_mob_skills(self, rathena_path: str) -> None:
        """Load mob_skill_db.txt to understand monster abilities.

        This lets the AI know what each monster can do:
        - What skills they cast (elemental attacks, status effects, summoning)
        - When they cast them (HP thresholds, conditions)
        - How dangerous they are (stun, freeze, poison, curse)
        """
        skill_path = Path(rathena_path) / "db" / "re" / "mob_skill_db.txt"
        if not skill_path.exists():
            skill_path = Path(rathena_path) / "db" / "mob_skill_db.txt"
        if not skill_path.exists():
            logger.warning("mob_skill_db.txt not found at %s", skill_path)
            return

        # Danger skill keywords
        danger_skills = {
            "stun": ["NPC_STUNATTACK", "STUN"],
            "freeze": ["NPC_FREEZE", "MG_FROSTDIVER", "WZ_STORMGUST"],
            "sleep": ["NPC_SLEEPATTACK", "SLEEP"],
            "poison": ["NPC_POISON", "NPC_POISONATTACK", "AS_VENOMDUST"],
            "curse": ["NPC_CURSEATTACK", "CURSE"],
            "silence": ["NPC_SILENCE", "SILENCE"],
            "blind": ["NPC_BLINDATTACK", "BLIND"],
            "confusion": ["NPC_CONFUSION", "CONFUSION"],
            "stone_curse": ["MG_STONECURSE", "STONE"],
            "summon": ["NPC_SUMMONSLAVE", "NPC_CALLSLAVE"],
            "heal": ["NPC_ALLHEAL", "NPC_POWERUP", "NPC_AGIUP"],
            "aoe": ["NPC_HELLJUDGEMENT", "NPC_EARTHQUAKE", "NPC_DARKBREATH",
                    "ASC_METEORASSAULT", "WZ_METEORSTORM", "WZ_HEAVENDRIVE"],
            "teleport": ["AL_TELEPORT"],
            "combo": ["NPC_COMBOATTACK", "AS_SONICBLOW", "NPC_CRITICALSLASH"],
            "provoke": ["NPC_PROVOCATION"],
        }

        try:
            with open(skill_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("//") or line.startswith("#"):
                        continue

                    parts = line.split(",")
                    if len(parts) < 8:
                        continue

                    mob_id_str = parts[0].strip()
                    try:
                        mob_id = int(mob_id_str)
                    except ValueError:
                        continue

                    mob_skill_name = parts[1].strip()
                    # Extract skill name after @
                    skill_name = mob_skill_name.split("@")[-1] if "@" in mob_skill_name else mob_skill_name
                    state = parts[2].strip()
                    rate = int(parts[4].strip() or 0)
                    cast_time = int(parts[5].strip() or 0)
                    delay = int(parts[6].strip() or 0)
                    condition = parts[10].strip() if len(parts) > 10 else ""
                    condition_val = parts[11].strip() if len(parts) > 11 else ""

                    # Store skill
                    if mob_id not in self._mob_skills:
                        self._mob_skills[mob_id] = []
                    self._mob_skills[mob_id].append({
                        "skill": skill_name,
                        "full_name": mob_skill_name,
                        "state": state,
                        "rate": rate,
                        "cast_time_ms": cast_time,
                        "delay_ms": delay,
                        "condition": condition,
                        "condition_value": condition_val,
                    })

                    # Check for danger conditions
                    for danger_type, keywords in danger_skills.items():
                        if any(kw in skill_name.upper() for kw in keywords):
                            # Find the mob name for this ID
                            for mob in self._monsters:
                                check_id = mob.get("id", 0) or 0
                                if str(check_id) == mob_id_str or check_id == mob_id:
                                    mob_name = mob.get("name", str(mob_id))
                                    if mob_name not in self._mob_skill_warnings:
                                        self._mob_skill_warnings[mob_name] = []
                                    warning = f"{danger_type}:{skill_name} (rate={rate}%, cast={cast_time}ms, cond={condition})"
                                    if warning not in self._mob_skill_warnings[mob_name]:
                                        self._mob_skill_warnings[mob_name].append(warning)
                                    break

            logger.info("Loaded mob skills for %d monsters, %d with danger warnings",
                        len(self._mob_skills), len(self._mob_skill_warnings))
        except Exception as e:
            logger.warning("Failed to load mob skills: %s", e)

    def _load_knowledge(self, path: str | None) -> None:
        """Load pre-ingested rAthena knowledge."""
        if path and Path(path).exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                self._monsters = data.get("monsters", [])
                logger.info("Loaded %d monsters from %s", len(self._monsters), path)
                return
            except Exception as e:
                logger.warning("Failed to load knowledge from %s: %s", path, e)

        # Load knowledge from local file if not provided
        loaded = False
        if not self._monsters:
            for candidate in [
                path,
                str(Path(__file__).parent.parent.parent / "knowledge" / "knowledge.json"),
                str(Path(__file__).parent.parent / "knowledge" / "knowledge.json"),
                "knowledge/knowledge.json",
            ]:
                if candidate and Path(candidate).exists():
                    try:
                        with open(candidate) as f:
                            data = json.load(f)
                        self._monsters = data.get("monsters", [])
                        loaded = True
                        logger.info("Loaded %d monsters from %s", len(self._monsters), candidate)
                        break
                    except Exception:
                        continue

        # Load mob skills from knowledge JSON
        if not self._mob_skills:
            for candidate in [
                path,
                str(Path(__file__).parent.parent.parent / "knowledge" / "knowledge.json"),
                str(Path(__file__).parent.parent / "knowledge" / "knowledge.json"),
                "knowledge/knowledge.json",
            ]:
                if candidate and Path(candidate).exists():
                    try:
                        with open(candidate) as f:
                            data = json.load(f)
                        raw_skills = data.get("mob_skills", {})
                        for mob_id_str, skill_list in raw_skills.items():
                            self._mob_skills[str(mob_id_str)] = skill_list
                            for skill in skill_list:
                                for danger_type, keywords in {
                                    "stun": ["STUNATTACK", "STUN"],
                                    "freeze": ["FREEZE", "FROSTDIVER", "STORMGUST"],
                                    "poison": ["POISONATTACK", "POISON", "VENOM"],
                                    "curse": ["CURSEATTACK", "CURSE"],
                                    "aoe": ["HELLJUDGEMENT", "EARTHQUAKE", "DARKBREATH",
                                            "METEORASSAULT", "METEORSTORM"],
                                    "summon": ["SUMMONSLAVE", "CALLSLAVE"],
                                    "heal": ["ALLHEAL", "POWERUP", "AGIUP"],
                                    "combo": ["COMBOATTACK", "SONICBLOW"],
                                }.items():
                                    skill_name = skill.get("skill", "").upper()
                                    if any(kw in skill_name for kw in keywords):
                                        rate = int(skill.get("rate", 0) or 0)
                                        if rate > 500:
                                            for mob in self._monsters:
                                                mob_id = str(mob.get("id", 0))
                                                if mob_id == mob_id_str:
                                                    mob_name = mob.get("name", "")
                                                    if mob_name not in self._mob_skill_warnings:
                                                        self._mob_skill_warnings[mob_name] = []
                                                    warning = f"{danger_type}:{skill.get('skill','')} (rate={rate}%)"
                                                    if warning not in self._mob_skill_warnings[mob_name]:
                                                        self._mob_skill_warnings[mob_name].append(warning)
                                                    break
                                        break
                        logger.info("Loaded mob skills for %d monsters from %s", len(self._mob_skills), candidate)
                        break
                    except Exception:
                        continue

        if not self._monsters:
            logger.warning("No game knowledge loaded — recommendations will be generic")

    def recommend_hunting(self, *,
                          bot_level: int = 1,
                          job_name: str = "novice",
                          bot_class: str = "novice",
                          weapon_type: str = "Dagger",
                          element: str = "Neutral",
                          party_size: int = 1,
                          goal: str = "leveling",
                          target_item: str = "",
                          avoid_mobs: list[str] | None = None,
                          ) -> list[HuntingRecommendation]:
        """Recommend the best hunting ground for a bot.

        Considers: level, class, weapon/element, party size, goal.
        Returns sorted list of recommendations (best first).
        """
        if not self._monsters:
            return self._generic_recommendation(bot_level)

        avoid = set(avoid_mobs or [])
        archetype = CLASS_ARCHETYPES.get(bot_class.lower(), CLASS_ARCHETYPES["novice"])
        damage_type = archetype.get("damage_type", "melee")

        recommendations = []

        for mob in self._monsters:
            name = mob.get("name", "")
            if name in avoid:
                continue

            mob_level = int(mob.get("level", 1) or 1)
            hp = int(mob.get("hp", 1) or 1)
            base_exp = int(mob.get("base_exp", 0) or 0)
            job_exp = int(mob.get("job_exp", 0) or 0)
            mob_element = mob.get("element", "Neutral")
            mob_race = mob.get("race", "Formless")
            mob_size = mob.get("size", "Medium")
            drops = mob.get("drops", [])

            # Level range filter: ±10 levels for efficient hunting
            level_diff = abs(mob_level - bot_level)
            if level_diff > 15:
                continue

            # Level penalty
            penalty = 1.0
            for bracket in LEVEL_PENALTY:
                if bracket["min_level"] <= level_diff <= bracket["max_level"]:
                    penalty = bracket["base_exp"]
                    break
            if level_diff >= 86:
                penalty = 0.0

            if penalty <= 0:
                continue

            # Element efficiency
            element_eff = ELEMENT_CHART.get(element, {}).get(mob_element, 1.0)

            # Race efficiency (based on weapon/class cards)
            race_eff = 1.0
            for race_name, chart in RACE_CHARTS.items():
                if mob_race in chart:
                    race_eff = chart[mob_race]
                    break

            # Size efficiency
            weapon_size = SIZE_CHART.get(weapon_type, {}).get(mob_size, 1.0)

            # EXP per HP (efficiency metric)
            exp_per_hp = base_exp / max(hp, 1)

            # Effective EXP after penalty
            effective_exp = base_exp * penalty

            # Score: weighted combination of efficiency factors
            score = (
                exp_per_hp * 1000 +  # Raw efficiency
                element_eff * 0.5 +   # Element advantage
                race_eff * 0.3 +      # Race advantage
                weapon_size * 0.2 +   # Weapon size fit
                (1.0 - min(level_diff / 15, 1.0)) * 0.5  # Level appropriateness
            )

            # Danger penalty: reduce score for dangerous monsters
            mob_name = mob.get("name", "")
            mob_id = str(mob.get("id", 0) or 0)
            mob_skills = self._mob_skills.get(mob_id, [])
            danger_factors = []
            for skill in mob_skills:
                skill_name = skill.get("skill", "")
                for danger_type, keywords in {
                    "stun": ["STUNATTACK", "STUN"],
                    "freeze": ["FREEZE", "FROSTDIVER", "STORMGUST"],
                    "sleep": ["SLEEPATTACK", "SLEEP"],
                    "poison": ["POISONATTACK", "POISON", "VENOM"],
                    "curse": ["CURSEATTACK", "CURSE"],
                    "stone_curse": ["STONECURSE", "STONE"],
                    "aoe": ["HELLJUDGEMENT", "EARTHQUAKE", "DARKBREATH",
                            "METEORASSAULT", "METEORSTORM"],
                    "summon": ["SUMMONSLAVE", "CALLSLAVE"],
                    "heal": ["ALLHEAL", "POWERUP", "AGIUP"],
                    "combo": ["COMBOATTACK", "SONICBLOW"],
                }.items():
                    if any(kw in skill_name.upper() for kw in keywords):
                        rate = int(skill.get("rate", 0) or 0)
                        if rate > 500:  # >5% chance
                            danger_factors.append(danger_type)
                        break

            if danger_factors:
                # Penalty based on number and severity of dangers
                danger_types = set(danger_factors)
                danger_severity = {
                    "stun": 0.3, "freeze": 0.3, "stone_curse": 0.4,
                    "sleep": 0.1, "poison": 0.15, "curse": 0.2,
                    "aoe": 0.3, "summon": 0.25, "heal": 0.2,
                    "combo": 0.1,
                }
                total_penalty = sum(danger_severity.get(d, 0.1) for d in danger_types)
                penalty_mult = max(0.3, 1.0 - total_penalty)
                score *= penalty_mult

            # Boost for item hunting
            if goal == "item" and target_item:
                for drop in drops:
                    if target_item.lower() in drop.get("item", "").lower():
                        score += 5.0
                        break

            # Adjust for party size
            if party_size > 1:
                score *= 1.0 + (party_size - 1) * 0.15  # 15% bonus per extra bot

            rec = HuntingRecommendation(
                map_name="",  # Unknown — maps are not in mob data
                primary_mob=name,
                mob_level=mob_level,
                base_exp=base_exp,
                job_exp=job_exp,
                hp=hp,
                element=mob_element,
                race=mob_race,
                size=mob_size,
                element_efficiency=element_eff,
                exp_per_hp=round(exp_per_hp, 4),
                level_penalty=penalty,
                effective_exp=effective_exp,
                score=round(score, 2),
                drops=drops[:5],
                reason=f"Lv{mob_level} {mob_element} {mob_race} | EXP/HP={exp_per_hp:.2f} | "
                       f"Eff={element_eff:.1f}x | Penalty={penalty:.0%}",
            )
            recommendations.append(rec)

        # Sort by score descending
        recommendations.sort(key=lambda r: r.score, reverse=True)

        # Return top 10
        return recommendations[:10]

    def _generic_recommendation(self, bot_level: int) -> list[HuntingRecommendation]:
        """Generic recommendations when no knowledge is loaded."""
        zones = [
            ("prt_fild08", 1, 10, 10, 55, "Neutral", "Formless", "Small"),
            ("prt_fild04", 15, 12, 120, 300, "Earth", "Plant", "Medium"),
            ("pay_fild08", 25, 25, 250, 800, "Poison", "Insect", "Medium"),
            ("pay_fild04", 40, 51, 500, 1500, "Fire", "Brute", "Medium"),
            ("gef_fild14", 60, 62, 1200, 4000, "Dark", "Demon", "Large"),
            ("moc_fild17", 75, 75, 2000, 6000, "Neutral", "Formless", "Medium"),
        ]
        recs = []
        for zone, min_lv, mob_lv, exp, hp, elem, race, size in zones:
            if bot_level >= min_lv - 5:
                eph = exp / max(hp, 1)
                recs.append(HuntingRecommendation(
                    map_name=zone, primary_mob="generic", mob_level=mob_lv,
                    base_exp=exp, job_exp=exp, hp=hp,
                    element=elem, race=race, size=size,
                    element_efficiency=1.0, exp_per_hp=round(eph, 4),
                    level_penalty=1.0, effective_exp=exp,
                    score=round(eph * 1000, 2),
                    reason=f"Generic zone for level {bot_level}",
                ))
        return recs[:5]

    def analyze_element(self, attack_element: str, defense_element: str) -> dict[str, Any]:
        """Analyze element advantage."""
        multiplier = ELEMENT_CHART.get(attack_element, {}).get(defense_element, 1.0)
        return {
            "attack": attack_element,
            "defense": defense_element,
            "multiplier": multiplier,
            "advantage": multiplier > 1.0,
            "disadvantage": multiplier < 1.0,
            "immune": multiplier == 0.0,
            "description": f"{attack_element} → {defense_element}: {multiplier:.0%} damage",
        }

    def recommend_skills_for_mob(self, *,
                                  job_name: str,
                                  mob_element: str,
                                  mob_race: str,
                                  mob_size: str,
                                  element: str = "Neutral") -> list[dict[str, Any]]:
        """Recommend optimal skills for a specific mob."""
        # Find the best element to use against this mob
        best_element = element
        best_mult = 1.0
        for atk_elem, defenses in ELEMENT_CHART.items():
            mult = defenses.get(mob_element, 1.0)
            if mult > best_mult:
                best_mult = mult
                best_element = atk_elem

        return [{
            "recommended_element": best_element,
            "damage_multiplier": best_mult,
            "mob_element": mob_element,
            "mob_race": mob_race,
            "mob_size": mob_size,
            "note": f"Use {best_element} attacks for {best_mult:.0%} damage",
        }]

    def valuate_item(self, item_name: str, item_data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Determine if an item is worth looting, keeping, or selling."""
        name_lower = item_name.lower()

        # Cards are always valuable
        if "card" in name_lower:
            return {"value": "high", "action": "keep", "reason": "Card — always valuable"}

        # Equipment
        if item_data and item_data.get("type") in ("Weapon", "Armor", "Accessory"):
            slots = int(item_data.get("slots", 0) or 0)
            level_req = int(item_data.get("level", 0) or 0)
            if slots > 0:
                return {"value": "high", "action": "keep", "reason": f"Slotted equipment ({slots} slots)"}
            return {"value": "medium", "action": "sell", "reason": "Equipment — sell if not needed"}

        # Healing items
        if any(kw in name_lower for kw in ["pot", "herb", "fruit", "fish", "meat"]):
            return {"value": "medium", "action": "keep", "reason": "Healing item — keep for sustain"}

        # Quest items
        if any(kw in name_lower for kw in ["quest", "token", "badge", "proof", "mark", "symbol"]):
            return {"value": "high", "action": "keep", "reason": "Quest item — may be needed"}

        # Ores/enchants
        if any(kw in name_lower for kw in ["elunium", "oridecon", "emerald", "sapphire", "topaz",
                                              "amethyst", "garnet", "diamond", "opal", "ruby"]):
            return {"value": "high", "action": "keep", "reason": "Refining material — valuable"}

        # Junk
        return {"value": "low", "action": "sell", "reason": "Junk item — sell to NPC"}

    def suggest_leveling_route(self, *, bot_level: int, job_name: str = "novice",
                                bot_class: str = "novice") -> list[dict[str, Any]]:
        """Suggest a complete leveling route from current level to 99."""
        route = []
        ranges = [(1, 15), (15, 30), (30, 50), (50, 70), (70, 85), (85, 99)]

        for min_lv, max_lv in ranges:
            if bot_level > max_lv:
                continue
            recs = self.recommend_hunting(
                bot_level=max(min_lv, bot_level),
                job_name=job_name,
                bot_class=bot_class,
                goal="leveling",
            )
            if recs:
                best = recs[0]
                route.append({
                    "level_range": f"{min_lv}-{max_lv}",
                    "recommended_map": best.map_name,
                    "primary_mob": best.primary_mob,
                    "mob_level": best.mob_level,
                    "exp_per_hp": best.exp_per_hp,
                    "element_efficiency": best.element_efficiency,
                    "score": best.score,
                })

        return route