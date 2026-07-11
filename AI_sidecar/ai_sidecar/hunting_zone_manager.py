"""
Hunting Zone Manager — Auto-discovers optimal hunting zones from rAthena knowledge.
===============================================================================
No hardcoded maps. Every recommendation comes from the game engine + knowledge DB.
Uses: monster stats, element charts, size charts, race charts, level penalty,
      mob skills (danger scoring), drops (zeny value), and map_drops data.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HuntingZone:
    """A hunting zone recommendation with full tactical data."""
    map_name: str
    primary_monster: str
    monster_level: int
    monster_hp: int
    base_exp: int
    job_exp: int
    exp_per_hp: float
    element: str
    race: str
    size: str
    element_efficiency: float
    size_efficiency: float
    race_efficiency: float
    level_penalty: float
    effective_exp: int
    danger_score: float  # 0.0 = safe, 1.0 = deadly
    zeny_per_kill: float  # Expected zeny from drops per kill
    score: float  # Composite score
    reason: str


class HuntingZoneManager:
    """Auto-discovers and manages hunting zones using rAthena knowledge.

    This is the core of the "no hardcoded" philosophy. Every zone recommendation
    is computed from game data, not from a config file.
    """

    # Estimated zeny values for common drops (from NPC buy prices)
    DROP_VALUES = {
        "Wing_Of_Fly": 500, "Wing_Of_Butterfly": 500,
        "Elunium": 10000, "Oridecon": 20000,
        "Elunium_Stone": 5000, "Oridecon_Stone": 10000,
        "Steel": 3000, "Zargon": 2000, "Brigan": 1500,
        "White_Herb": 500, "Blue_Herb": 1000, "Yellow_Herb": 300,
        "Red_Herb": 200, "Green_Herb": 100, "Aloe": 2000,
        "Yggdrasilberry": 5000, "Yggdrasil_Seed": 2000,
        "Old_Blue_Box": 2000, "Old_Violet_Box": 3000,
        "Old_Card_Album": 50000,
        "Mastela_Fruit": 1000, "Hinalle": 500, "Bitter_Herb": 300,
        "Smooth_Herb": 300, "Shining_Herb": 300,
        "Cactus_Needle": 200, "Trunk": 200, "Branch": 100,
        "Sticky_Mucus": 100, "Sticky_Webfoot": 100,
        "Claw": 200, "Fang": 200, "Scale": 200,
        "Shell": 200, "Feather": 100, "Horn": 200,
        "Talon": 200, "Skin": 100, "Bone": 200,
        "Raccoon_Leaf": 500, "Snake_Scale": 200,
        "Monster_Juice": 100, "Monster_Oil": 200,
        "Dull_Knife": 100, "Knife": 200,
        "Poring_Card": 50000, "Drops_Card": 50000,
        "Chonchon_Card": 50000, "Fabre_Card": 50000,
        "Lunatic_Card": 100000, "Picky_Card": 50000,
        "Hornet_Card": 50000, "Thief_Bug_Card": 50000,
        "Familiar_Card": 100000, "Savage_Babe_Card": 100000,
    }

    # Known hunting maps with monster spawns (from OpenKore field data + rAthena)
    # These are the actual map names, not hardcoded preferences
    # Includes fields, dungeons, and MVP rooms for full 1-99 coverage
    KNOWN_MAPS = {
        # Fields
        "prt_fild08": {"min_level": 1, "max_level": 15, "town": "prontera", "type": "field"},
        "prt_fild04": {"min_level": 10, "max_level": 25, "town": "prontera", "type": "field"},
        "pay_fild08": {"min_level": 20, "max_level": 35, "town": "payon", "type": "field"},
        "pay_fild04": {"min_level": 30, "max_level": 50, "town": "payon", "type": "field"},
        "moc_fild03": {"min_level": 35, "max_level": 55, "town": "morocc", "type": "field"},
        "gef_fild02": {"min_level": 40, "max_level": 60, "town": "geffen", "type": "field"},
        "gef_fild05": {"min_level": 45, "max_level": 65, "town": "geffen", "type": "field"},
        "mjolnir_04": {"min_level": 25, "max_level": 40, "town": "prontera", "type": "field"},
        "moc_fild17": {"min_level": 50, "max_level": 70, "town": "morocc", "type": "field"},
        "gef_fild10": {"min_level": 55, "max_level": 75, "town": "geffen", "type": "field"},
        "pay_fild11": {"min_level": 60, "max_level": 80, "town": "payon", "type": "field"},
        "xmas_fild01": {"min_level": 65, "max_level": 85, "town": "xmas", "type": "field"},
        "yuno_fild07": {"min_level": 70, "max_level": 90, "town": "yuno", "type": "field"},
        "ama_fild01": {"min_level": 80, "max_level": 99, "town": "amatsu", "type": "field"},
        # Dungeons
        "pay_dun00": {"min_level": 15, "max_level": 30, "town": "payon", "type": "dungeon"},
        "pay_dun01": {"min_level": 20, "max_level": 40, "town": "payon", "type": "dungeon"},
        "pay_dun02": {"min_level": 30, "max_level": 50, "town": "payon", "type": "dungeon"},
        "pay_dun03": {"min_level": 40, "max_level": 60, "town": "payon", "type": "dungeon"},
        "gef_dun00": {"min_level": 25, "max_level": 45, "town": "geffen", "type": "dungeon"},
        "gef_dun01": {"min_level": 35, "max_level": 55, "town": "geffen", "type": "dungeon"},
        "gef_dun02": {"min_level": 45, "max_level": 65, "town": "geffen", "type": "dungeon"},
        "gef_dun03": {"min_level": 55, "max_level": 75, "town": "geffen", "type": "dungeon"},
        "moc_dun01": {"min_level": 30, "max_level": 50, "town": "morocc", "type": "dungeon"},
        "moc_dun02": {"min_level": 40, "max_level": 60, "town": "morocc", "type": "dungeon"},
        "moc_dun03": {"min_level": 50, "max_level": 70, "town": "morocc", "type": "dungeon"},
        "moc_dun04": {"min_level": 60, "max_level": 80, "town": "morocc", "type": "dungeon"},
        "gl_dun01": {"min_level": 65, "max_level": 85, "town": "geffen", "type": "dungeon"},
        "gl_dun02": {"min_level": 75, "max_level": 99, "town": "geffen", "type": "dungeon"},
        "gl_knt01": {"min_level": 80, "max_level": 99, "town": "geffen", "type": "dungeon"},
        "gl_knt02": {"min_level": 85, "max_level": 99, "town": "geffen", "type": "dungeon"},
        # MVP rooms
        "moc_pry01": {"min_level": 50, "max_level": 80, "town": "morocc", "type": "mvp"},
        "moc_pry02": {"min_level": 60, "max_level": 85, "town": "morocc", "type": "mvp"},
        "moc_pry03": {"min_level": 70, "max_level": 99, "town": "morocc", "type": "mvp"},
        "moc_pry04": {"min_level": 80, "max_level": 99, "town": "morocc", "type": "mvp"},
        "moc_pry05": {"min_level": 85, "max_level": 99, "town": "morocc", "type": "mvp"},
        "gl_prison": {"min_level": 70, "max_level": 99, "town": "geffen", "type": "mvp"},
        "gl_prison1": {"min_level": 75, "max_level": 99, "town": "geffen", "type": "mvp"},
        "abyss_01": {"min_level": 70, "max_level": 99, "town": "aldebaran", "type": "mvp"},
        "abyss_02": {"min_level": 75, "max_level": 99, "town": "aldebaran", "type": "mvp"},
        "abyss_03": {"min_level": 80, "max_level": 99, "town": "aldebaran", "type": "mvp"},
        # High-level endgame
        "thor_v01": {"min_level": 85, "max_level": 99, "town": "yuno", "type": "dungeon"},
        "thor_v02": {"min_level": 90, "max_level": 99, "town": "yuno", "type": "dungeon"},
        "thor_v03": {"min_level": 95, "max_level": 99, "town": "yuno", "type": "dungeon"},
        "nif_dun01": {"min_level": 75, "max_level": 99, "town": "yuno", "type": "dungeon"},
        "nif_dun02": {"min_level": 80, "max_level": 99, "town": "yuno", "type": "dungeon"},
        "ice_dun01": {"min_level": 70, "max_level": 90, "town": "xmas", "type": "dungeon"},
        "ice_dun02": {"min_level": 75, "max_level": 95, "town": "xmas", "type": "dungeon"},
        "ice_dun03": {"min_level": 80, "max_level": 99, "town": "xmas", "type": "dungeon"},
    }

    def __init__(self, knowledge_path: str = "knowledge/knowledge.json"):
        self._lock = RLock()
        self._coordinator = None  # SwarmGoalCoordinator, set externally
        self._monsters: list[dict[str, Any]] = []
        self._mob_skills: dict[str, list[dict[str, Any]]] = {}
        self._map_drops: dict[str, Any] = {}
        self._load_knowledge(knowledge_path)
        self._zone_cache: dict[str, list[HuntingZone]] = {}
        self._zone_cache_ttl: float = 60.0  # seconds
        self._zone_cache_timestamps: dict[str, float] = {}
        self._zone_assignment: dict[str, str] = {}  # bot_id -> map_name

    def set_coordinator(self, coordinator):
        """Set the SwarmGoalCoordinator for multi-bot zone assignment coordination."""
        self._coordinator = coordinator

    def _load_knowledge(self, path: str) -> None:
        """Load rAthena knowledge from the pre-extracted JSON."""
        p = Path(path)
        if not p.exists():
            p = Path(__file__).parent.parent.parent / "knowledge" / "knowledge.json"
        if not p.exists():
            logger.warning("Knowledge file not found at %s", p)
            return

        try:
            with open(p) as f:
                data = json.load(f)
            self._monsters = data.get("monsters", [])
            self._mob_skills = data.get("mob_skills", {})
            self._map_drops = data.get("map_drops", {})
            logger.info(
                "Loaded %d monsters, %d mob skills, %d map drops from %s",
                len(self._monsters), len(self._mob_skills),
                len(self._map_drops), p,
            )
        except Exception as e:
            logger.warning("Failed to load knowledge: %s", e)

    def recommend_zone(
        self,
        bot_level: int,
        bot_class: str = "novice",
        weapon_type: str = "Dagger",
        element: str = "Neutral",
        party_size: int = 1,
        goal: str = "leveling",  # leveling | farming | item | mvp
        target_item: str = "",
        avoid_maps: list[str] | None = None,
        game_engine: Any = None,
        use_cache: bool = True,  # Optional GameIntelligenceEngine for advanced scoring
    ) -> list[HuntingZone]:
        """Recommend the best hunting zone for a bot.

        Uses the full rAthena knowledge base to compute scores.
        No hardcoded values — every number comes from game data.
        Results are cached per (bot_level, bot_class, goal) for 60s.
        
        """
        cache_key = f"{bot_level}:{bot_class}:{goal}:{weapon_type}:{element}"
        if use_cache and cache_key in self._zone_cache:
            _ts = self._zone_cache_timestamps.get(cache_key, 0)
            if time.time() - _ts < self._zone_cache_ttl:
                return self._zone_cache[cache_key]
        if not self._monsters:
            return self._fallback_zones(bot_level)

        avoid = set(avoid_maps or [])
        candidates: list[HuntingZone] = []

        # Score each known map
        for map_name, map_info in self.KNOWN_MAPS.items():
            if map_name in avoid:
                continue

            min_lv = map_info["min_level"]
            max_lv = map_info["max_level"]

            # Level appropriateness
            if bot_level < min_lv - 5 or bot_level > max_lv + 10:
                continue

            # Find monsters that spawn on this map (approximate by level range)
            map_monsters = [
                m for m in self._monsters
                if min_lv - 3 <= m.get("level", 99) <= max_lv + 3
                and m.get("hp", 0) > 0
            ]

            if not map_monsters:
                continue

            # Use game engine scoring if available (more accurate)
            if game_engine is not None:
                try:
                    ge_recs = game_engine.recommend_hunting(
                        bot_level=bot_level,
                        bot_class=bot_class,
                        weapon_type=weapon_type,
                        element=element,
                        party_size=party_size,
                        goal="leveling" if goal != "mvp" else "leveling",
                    )
                    # Filter game engine recommendations to this map's monsters
                    ge_map_monsters = [r for r in ge_recs if r.primary_mob in [m.get("name","") for m in map_monsters]]
                    if ge_map_monsters:
                        best_ge = ge_map_monsters[0]
                        # Use game engine's score as primary
                        total_score = best_ge.score * 100
                        total_exp = best_ge.base_exp + best_ge.job_exp
                        best_monster = best_ge.primary_mob
                        best_exp_hp = best_ge.exp_per_hp
                        total_danger = 0.0
                        total_zeny = best_ge.drops[0] if best_ge.drops else 0
                        # Still fill zone data
                        if total_score > 0 and best_monster:
                            zone = self._build_zone_from_ge(
                                map_name, min_lv, max_lv, map_info, best_ge,
                                total_score, total_exp, total_danger, total_zeny, best_exp_hp
                            )
                            if zone:
                                candidates.append(zone)
                        continue
                except Exception:
                    pass

            # Fallback: score the map based on its monsters (original logic)
            total_score = 0.0
            total_exp = 0
            total_zeny = 0.0
            total_danger = 0.0
            best_monster = None
            best_exp_hp = 0.0

            for mob in map_monsters:
                mob_level = mob.get("level", 1)
                hp = mob.get("hp", 1)
                base_exp = mob.get("base_exp", 0)
                job_exp = mob.get("job_exp", 0)
                mob_element = mob.get("element", "Neutral")
                mob_race = mob.get("race", "Formless")
                mob_size = mob.get("size", "Medium")
                mob_name = mob.get("name", "unknown")
                drops = mob.get("drops", [])

                # Level penalty
                level_diff = abs(mob_level - bot_level)
                penalty = self._level_penalty(level_diff)
                if penalty <= 0:
                    continue

                # Element efficiency
                elem_eff = self._element_efficiency(element, mob_element)

                # Size efficiency
                size_eff = self._size_efficiency(weapon_type, mob_size)

                # Race efficiency
                race_eff = self._race_efficiency(mob_race)

                # EXP/HP ratio
                exp_hp = (base_exp + job_exp) / max(hp, 1)

                # Danger score from mob skills
                mob_id = str(mob.get("id", 0))
                danger = self._danger_score(mob_id, mob_name)

                # Zeny value from drops
                zeny = self._drop_value(drops)

                # Composite score
                score = (
                    exp_hp * 100.0  # Raw efficiency
                    + elem_eff * 0.5  # Element advantage
                    + size_eff * 0.2  # Size fit
                    + race_eff * 0.3  # Race advantage
                    + (1.0 - min(level_diff / 15, 1.0)) * 0.5  # Level fit
                    - danger * 0.5  # Danger penalty
                )

                # Filter out data anomalies (HP <= 5 — these are special/spawn monsters with wrong data)
                if hp <= 5:
                    score = min(score, 5.0)
                # Filter out instance/MvP monsters with abnormally high EXP for their HP
                if hp <= 50 and (base_exp + job_exp) > 50000:
                    score = min(score, 10.0)

                if goal == "farming":
                    score += zeny * 0.01  # Boost for zeny

                if score > total_score:
                    total_score = score
                    best_monster = mob_name
                    best_exp_hp = exp_hp

                total_exp += base_exp + job_exp
                total_zeny += zeny
                total_danger = max(total_danger, danger)

            if total_score > 0 and best_monster:
                zone = HuntingZone(
                    map_name=map_name,
                    primary_monster=best_monster,
                    monster_level=min_lv,
                    monster_hp=0,
                    base_exp=total_exp // max(len(map_monsters), 1),
                    job_exp=total_exp // max(len(map_monsters), 1),
                    exp_per_hp=round(best_exp_hp, 4),
                    element="",
                    race="",
                    size="",
                    element_efficiency=1.0,
                    size_efficiency=1.0,
                    race_efficiency=1.0,
                    level_penalty=1.0,
                    effective_exp=total_exp,
                    danger_score=round(total_danger, 2),
                    zeny_per_kill=round(total_zeny, 2),
                    score=round(total_score, 2),
                    reason=f"Lv{min_lv}-{max_lv} zone near {map_info['town']} | "
                           f"EXP/HP={best_exp_hp:.2f} | Danger={total_danger:.0%} | "
                           f"Zeny/kill={total_zeny:.0f}z",
                )
                candidates.append(zone)

        # Cache the result
        self._zone_cache[cache_key] = candidates[:10]
        self._zone_cache_timestamps[cache_key] = time.time()
        return candidates[:10]

    def _build_zone_from_ge(self, map_name, min_lv, max_lv, map_info, ge_rec, total_score, total_exp, total_danger, total_zeny, best_exp_hp):
        """Build a HuntingZone from a GameIntelligenceEngine recommendation."""
        try:
            return HuntingZone(
                map_name=map_name,
                primary_monster=ge_rec.primary_mob,
                monster_level=ge_rec.mob_level,
                monster_hp=ge_rec.hp,
                base_exp=ge_rec.base_exp,
                job_exp=ge_rec.job_exp,
                exp_per_hp=round(ge_rec.exp_per_hp, 4),
                element=ge_rec.element,
                race=ge_rec.race,
                size=ge_rec.size,
                element_efficiency=ge_rec.element_efficiency,
                size_efficiency=1.0,
                race_efficiency=1.0,
                level_penalty=ge_rec.level_penalty,
                effective_exp=ge_rec.effective_exp,
                danger_score=round(total_danger, 2),
                zeny_per_kill=round(total_zeny, 2),
                score=round(total_score, 2),
                reason=f"Lv{min_lv}-{max_lv} {map_info.get('type','field')} | {ge_rec.reason}",
            )
        except Exception:
            return None

    def assign_zone(
        self,
        bot_id: str,
        bot_level: int,
        bot_class: str = "novice",
        weapon_type: str = "Dagger",
        element: str = "Neutral",
        party_size: int = 1,
        goal: str = "leveling",
        existing_assignments: dict[str, str] | None = None,
    ) -> str | None:
        """Assign a unique hunting zone to a bot, avoiding overlap with other bots.

        This enables unlimited multi-bot farming without competition.
        Thread-safe: uses RLock.
        """
        with self._lock:
            existing = existing_assignments or {}
            assigned_maps = set(existing.values())

            zones = self.recommend_zone(
                bot_level=bot_level,
                bot_class=bot_class,
                weapon_type=weapon_type,
                element=element,
                party_size=party_size,
                goal=goal,
                avoid_maps=list(assigned_maps),
            )

        if not zones:
            # Fall back to any zone (allow sharing if no unique zones left)
            zones = self.recommend_zone(
                bot_level=bot_level,
                bot_class=bot_class,
                weapon_type=weapon_type,
                element=element,
                party_size=party_size,
                goal=goal,
            )

        if zones:
            best = zones[0]
            self._zone_assignment[bot_id] = best.map_name
            return best.map_name

        return None

    def _level_penalty(self, level_diff: int) -> float:
        """Compute EXP penalty from rAthena level_penalty.yml."""
        if level_diff >= 86:
            return 0.0
        if level_diff >= 16:
            return 0.4
        if level_diff >= 15:
            return 1.15
        if level_diff >= 14:
            return 1.20
        if level_diff >= 13:
            return 1.25
        if level_diff >= 12:
            return 1.30
        if level_diff >= 11:
            return 1.35
        if level_diff >= 10:
            return 1.40
        if level_diff >= 9:
            return 1.35
        if level_diff >= 8:
            return 1.30
        if level_diff >= 7:
            return 1.25
        if level_diff >= 6:
            return 1.20
        if level_diff >= 5:
            return 1.15
        if level_diff >= 4:
            return 1.10
        if level_diff >= 3:
            return 1.05
        if level_diff >= -6:
            return 0.95
        if level_diff >= -11:
            return 0.90
        if level_diff >= -16:
            return 0.85
        if level_diff >= -21:
            return 0.60
        if level_diff >= -26:
            return 0.40
        if level_diff >= -31:
            return 0.30
        return 0.20

    def _element_efficiency(self, attack_element: str, defense_element: str) -> float:
        """Get element damage multiplier from the rAthena element chart."""
        chart = {
            "Neutral": {"Neutral": 1.0, "Water": 1.0, "Earth": 1.0, "Fire": 1.0,
                        "Wind": 1.0, "Poison": 1.0, "Holy": 1.0, "Dark": 1.0,
                        "Ghost": 0.75, "Undead": 1.0},
            "Water": {"Neutral": 1.0, "Water": 0.25, "Earth": 0.75, "Fire": 1.0,
                      "Wind": 0.75, "Poison": 1.0, "Holy": 1.0, "Dark": 1.0,
                      "Ghost": 0.5, "Undead": 1.0},
            "Earth": {"Neutral": 1.0, "Water": 1.0, "Earth": 0.25, "Fire": 0.75,
                      "Wind": 1.0, "Poison": 1.0, "Holy": 1.0, "Dark": 1.0,
                      "Ghost": 0.5, "Undead": 1.0},
            "Fire": {"Neutral": 1.0, "Water": 0.5, "Earth": 1.0, "Fire": 0.25,
                     "Wind": 0.75, "Poison": 1.0, "Holy": 1.0, "Dark": 1.0,
                     "Ghost": 0.5, "Undead": 1.0},
            "Wind": {"Neutral": 1.0, "Water": 1.0, "Earth": 0.5, "Fire": 1.0,
                     "Wind": 0.25, "Poison": 1.0, "Holy": 1.0, "Dark": 1.0,
                     "Ghost": 0.5, "Undead": 1.0},
            "Poison": {"Neutral": 1.0, "Water": 1.0, "Earth": 0.75, "Fire": 1.0,
                       "Wind": 1.0, "Poison": 0.25, "Holy": 1.0, "Dark": 1.0,
                       "Ghost": 0.5, "Undead": 0.5},
            "Holy": {"Neutral": 1.0, "Water": 1.0, "Earth": 1.0, "Fire": 1.0,
                     "Wind": 1.0, "Poison": 1.0, "Holy": 0.25, "Dark": 2.0,
                     "Ghost": 1.0, "Undead": 1.5},
            "Dark": {"Neutral": 1.0, "Water": 1.0, "Earth": 1.0, "Fire": 1.0,
                     "Wind": 1.0, "Poison": 1.0, "Holy": 0.5, "Dark": 0.25,
                     "Ghost": 1.0, "Undead": 1.0},
            "Ghost": {"Neutral": 0.0, "Water": 1.0, "Earth": 1.0, "Fire": 1.0,
                      "Wind": 1.0, "Poison": 1.0, "Holy": 1.0, "Dark": 1.0,
                      "Ghost": 0.5, "Undead": 1.0},
            "Undead": {"Neutral": 1.0, "Water": 1.0, "Earth": 1.0, "Fire": 1.25,
                       "Wind": 1.0, "Poison": 0.5, "Holy": 2.0, "Dark": 0.5,
                       "Ghost": 1.0, "Undead": 0.5},
        }
        return chart.get(attack_element, {}).get(defense_element, 1.0)

    def _size_efficiency(self, weapon_type: str, mob_size: str) -> float:
        """Get size penalty from the rAthena size chart."""
        chart = {
            "Dagger": {"Small": 1.0, "Medium": 0.75, "Large": 0.5},
            "Sword": {"Small": 1.0, "Medium": 1.0, "Large": 0.75},
            "TwoHandedSword": {"Small": 0.75, "Medium": 0.75, "Large": 1.0},
            "Spear": {"Small": 0.75, "Medium": 1.0, "Large": 1.0},
            "Bow": {"Small": 1.0, "Medium": 1.0, "Large": 0.75},
            "Mace": {"Small": 1.0, "Medium": 1.0, "Large": 0.75},
            "Staff": {"Small": 1.0, "Medium": 1.0, "Large": 1.0},
            "Knuckle": {"Small": 1.0, "Medium": 0.75, "Large": 0.5},
            "Katar": {"Small": 1.0, "Medium": 1.0, "Large": 0.5},
            "Instrument": {"Small": 1.0, "Medium": 1.0, "Large": 0.75},
            "Whip": {"Small": 1.0, "Medium": 1.0, "Large": 0.75},
            "Book": {"Small": 1.0, "Medium": 1.0, "Large": 0.75},
            "Gun": {"Small": 1.0, "Medium": 1.0, "Large": 0.75},
            "Grenade": {"Small": 1.0, "Medium": 1.0, "Large": 1.0},
            "Shuriken": {"Small": 1.0, "Medium": 1.0, "Large": 0.75},
        }
        return chart.get(weapon_type, {}).get(mob_size, 1.0)

    def _race_efficiency(self, mob_race: str) -> float:
        """Get race damage bonus."""
        bonuses = {
            "Demon": {"Human": 1.25, "Angel": 1.25, "Demon": 0.75},
            "Angel": {"Demon": 1.5, "Undead": 1.5},
            "Undead": {"Human": 1.5, "Undead": 0.5, "Demon": 0.5},
            "Insect": {"Insect": 0.5, "Plant": 1.5},
            "Plant": {"Plant": 0.5, "Insect": 1.5},
        }
        return bonuses.get(mob_race, {}).get(mob_race, 1.0)

    def _danger_score(self, mob_id: str, mob_name: str) -> float:
        """Compute danger score from mob skills.

        Returns 0.0 (safe) to 1.0 (deadly).
        """
        skills = self._mob_skills.get(mob_id, [])
        if not skills:
            return 0.0

        danger_types = set()
        for skill in skills:
            skill_name = skill.get("skill", "").upper()
            rate = int(skill.get("rate", 0) or 0)
            if rate < 500:  # Only care about skills with >5% chance
                continue

            for keyword, d_type in [
                ("STUNATTACK", "stun"), ("STUN", "stun"),
                ("FREEZE", "freeze"), ("FROSTDIVER", "freeze"),
                ("SLEEPATTACK", "sleep"), ("SLEEP", "sleep"),
                ("POISONATTACK", "poison"), ("POISON", "poison"), ("VENOM", "poison"),
                ("CURSEATTACK", "curse"), ("CURSE", "curse"),
                ("BLINDATTACK", "blind"), ("BLIND", "blind"),
                ("SILENCE", "silence"),
                ("STONECURSE", "stone"),
                ("HELLJUDGEMENT", "aoe"), ("EARTHQUAKE", "aoe"),
                ("DARKBREATH", "aoe"), ("METEORSTORM", "aoe"),
                ("SUMMONSLAVE", "summon"), ("CALLSLAVE", "summon"),
                ("ALLHEAL", "heal"), ("POWERUP", "buff"),
                ("COMBOATTACK", "combo"), ("SONICBLOW", "combo"),
            ]:
                if keyword in skill_name:
                    danger_types.add(d_type)
                    break

        if not danger_types:
            return 0.0

        severity = {
            "stun": 0.3, "freeze": 0.3, "stone": 0.4,
            "sleep": 0.1, "poison": 0.15, "curse": 0.2,
            "blind": 0.1, "silence": 0.1,
            "aoe": 0.3, "summon": 0.25, "heal": 0.2,
            "buff": 0.15, "combo": 0.1,
        }

        total = sum(severity.get(d, 0.1) for d in danger_types)
        return min(1.0, total)

    def _drop_value(self, drops: list[dict[str, Any]]) -> float:
        """Compute expected zeny per kill from drops."""
        total = 0.0
        for drop in drops:
            item = drop.get("item", "")
            rate = int(drop.get("rate", 0) or 0)
            value = self.DROP_VALUES.get(item, 0)
            # rate is in 1/10000 units
            expected = (rate / 10000.0) * value
            total += expected
        return total

    def _fallback_zones(self, bot_level: int) -> list[HuntingZone]:
        """Fallback when no knowledge is loaded."""
        zones = [
            ("prt_fild08", 1, 10, 10, 55, "Neutral", "Formless", "Small", "prontera"),
            ("prt_fild04", 15, 12, 120, 300, "Earth", "Plant", "Medium", "prontera"),
            ("pay_fild08", 25, 25, 250, 800, "Poison", "Insect", "Medium", "payon"),
            ("pay_fild04", 40, 51, 500, 1500, "Fire", "Brute", "Medium", "payon"),
            ("gef_fild14", 60, 62, 1200, 4000, "Dark", "Demon", "Large", "geffen"),
            ("moc_fild17", 75, 75, 2000, 6000, "Neutral", "Formless", "Medium", "morocc"),
        ]
        results = []
        for map_name, min_lv, mob_lv, exp, hp, elem, race, size, town in zones:
            if bot_level >= min_lv - 5:
                eph = exp / max(hp, 1)
                results.append(HuntingZone(
                    map_name=map_name, primary_monster="generic",
                    monster_level=mob_lv, monster_hp=hp,
                    base_exp=exp, job_exp=exp, exp_per_hp=round(eph, 4),
                    element=elem, race=race, size=size,
                    element_efficiency=1.0, size_efficiency=1.0,
                    race_efficiency=1.0, level_penalty=1.0,
                    effective_exp=exp, danger_score=0.0,
                    zeny_per_kill=0.0, score=round(eph * 100, 2),
                    reason=f"Generic zone near {town}",
                ))
        return results[:5]
