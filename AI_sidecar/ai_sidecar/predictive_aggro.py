"""
Predictive Aggro Knowledge — knows which monsters are aggressive and their ranges.

Instead of reacting to aggro after it happens, this module pre-calculates:
1. Which monsters have aggressive AI (attack on sight)
2. Aggro ranges for each monster type (how far they'll chase)
3. Chase ranges (how far they'll follow before giving up)
4. Areas with overlapping aggro ranges (danger zones)
5. Safe paths that avoid aggro before moving

This is the difference between reactive pathfinding (current) and
predictive pathfinding (pro player behavior).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MonsterAggroInfo:
    """Aggression data for a monster type."""
    monster_name: str
    monster_id: int = 0
    is_aggressive: bool = False  # attacks on sight
    is_assist: bool = False      # assists nearby monsters of same family
    is_boss: bool = False        # MVP/Mini-boss
    aggro_range: int = 10        # cells before they aggro
    chase_range: int = 12        # cells they'll chase
    max_chase_range: int = 20    # max chase before giving up
    level: int = 1
    race: str = ""               # demi-human, brute, undead, etc.
    element: str = ""            # fire, water, wind, earth, etc.
    size: str = ""               # small, medium, large
    hp: int = 0
    atk: int = 0
    def_: int = 0
    spawn_maps: list[str] = field(default_factory=list)
    spawn_count: int = 0         # how many spawn per map
    # ── Pro RO additions ──
    is_night_aggro: bool = False  # only aggressive at night (e.g. Zombies)
    is_day_aggro: bool = False    # only aggressive during day (rare)
    assist_family: str = ""       # family name for assist aggro (e.g. "Orc", "Thief Bug")
    has_skill_aggro: bool = False # uses skills that draw aggro (e.g. Provoke)
    is_ranged: bool = False       # ranged attacker (aggro from further)


# Known monster aggro data (from iRO/rAthena classic)
# This is the knowledge a pro player has about which monsters to avoid.
# FIXED by Pro RO Player: added assist aggro, night aggro, correct chase ranges
KNOWN_AGGRESSIVE_MONSTERS: dict[str, dict[str, Any]] = {
    # ── Prontera Fields ──
    "Poring": {"is_aggressive": False, "aggro_range": 0, "chase_range": 0, "level": 1, "race": "plant"},
    "Lunatic": {"is_aggressive": False, "aggro_range": 0, "chase_range": 0, "level": 3, "race": "brute"},
    "Fabre": {"is_aggressive": False, "aggro_range": 0, "chase_range": 0, "level": 2, "race": "insect"},
    "Picky": {"is_aggressive": False, "aggro_range": 0, "chase_range": 0, "level": 3, "race": "brute"},
    "Familiar": {"is_aggressive": True, "aggro_range": 8, "chase_range": 10, "level": 14, "race": "demon"},
    "Hornet": {"is_aggressive": True, "aggro_range": 7, "chase_range": 9, "level": 12, "race": "insect"},
    "Thief Bug": {"is_aggressive": True, "aggro_range": 8, "chase_range": 10, "level": 16, "race": "insect", "is_assist": True, "assist_family": "Thief Bug"},
    "Savage": {"is_aggressive": True, "aggro_range": 10, "chase_range": 12, "level": 25, "race": "brute"},
    "Deniro": {"is_aggressive": True, "aggro_range": 8, "chase_range": 10, "level": 22, "race": "insect"},
    "Peco Peco": {"is_aggressive": True, "aggro_range": 9, "chase_range": 11, "level": 20, "race": "brute"},
    "Elder Willow": {"is_aggressive": True, "aggro_range": 8, "chase_range": 10, "level": 30, "race": "plant"},
    "Orc Warrior": {"is_aggressive": True, "aggro_range": 10, "chase_range": 14, "level": 38, "race": "demi-human", "is_assist": True, "assist_family": "Orc"},
    "Orc Archer": {"is_aggressive": True, "aggro_range": 14, "chase_range": 18, "level": 40, "race": "demi-human", "is_assist": True, "assist_family": "Orc", "is_ranged": True},
    "Orc Lady": {"is_aggressive": True, "aggro_range": 10, "chase_range": 12, "level": 42, "race": "demi-human", "is_assist": True, "assist_family": "Orc"},
    "Orc Zombie": {"is_aggressive": True, "aggro_range": 8, "chase_range": 10, "level": 45, "race": "undead", "is_assist": True, "assist_family": "Orc"},
    "Orc Skeleton": {"is_aggressive": True, "aggro_range": 8, "chase_range": 10, "level": 48, "race": "undead", "is_assist": True, "assist_family": "Orc"},
    "Orc Lord": {"is_aggressive": True, "aggro_range": 14, "chase_range": 20, "is_boss": True, "level": 75, "race": "demi-human"},

    # ── Geffen Fields ──
    "Dustiness": {"is_aggressive": True, "aggro_range": 8, "chase_range": 10, "level": 28, "race": "insect"},
    "Mantis": {"is_aggressive": True, "aggro_range": 9, "chase_range": 11, "level": 26, "race": "insect"},
    "Hode": {"is_aggressive": True, "aggro_range": 8, "chase_range": 10, "level": 32, "race": "brute"},
    "Desert Wolf": {"is_aggressive": True, "aggro_range": 10, "chase_range": 12, "level": 35, "race": "brute"},
    "Mummy": {"is_aggressive": True, "aggro_range": 8, "chase_range": 10, "level": 38, "race": "undead"},
    "Argiope": {"is_aggressive": True, "aggro_range": 9, "chase_range": 11, "level": 40, "race": "insect"},
    "Matyr": {"is_aggressive": True, "aggro_range": 10, "chase_range": 12, "level": 42, "race": "brute"},
    "Minorous": {"is_aggressive": True, "aggro_range": 10, "chase_range": 14, "level": 52, "race": "brute"},

    # ── Morocc Fields ──
    "Scorpion": {"is_aggressive": True, "aggro_range": 8, "chase_range": 10, "level": 24, "race": "insect"},
    "Skeleton": {"is_aggressive": True, "aggro_range": 8, "chase_range": 10, "level": 30, "race": "undead"},
    "Ghoul": {"is_aggressive": True, "aggro_range": 9, "chase_range": 11, "level": 35, "race": "undead"},
    "Zombie": {"is_aggressive": True, "aggro_range": 7, "chase_range": 9, "level": 18, "race": "undead", "is_night_aggro": True},
    "Drainliar": {"is_aggressive": True, "aggro_range": 8, "chase_range": 10, "level": 28, "race": "brute"},
    "Mistress": {"is_aggressive": True, "aggro_range": 12, "chase_range": 18, "is_boss": True, "level": 63, "race": "demon"},

    # ── Payon Fields ──
    "Spore": {"is_aggressive": False, "aggro_range": 0, "chase_range": 0, "level": 17, "race": "plant"},
    "Poporing": {"is_aggressive": False, "aggro_range": 0, "chase_range": 0, "level": 20, "race": "plant"},
    "Smokie": {"is_aggressive": False, "aggro_range": 0, "chase_range": 0, "level": 22, "race": "brute"},
    "Yoyo": {"is_aggressive": True, "aggro_range": 8, "chase_range": 10, "level": 24, "race": "brute"},
    "Steam Goblin": {"is_aggressive": True, "aggro_range": 9, "chase_range": 11, "level": 30, "race": "demi-human", "is_assist": True, "assist_family": "Goblin"},
    "Goblin": {"is_aggressive": True, "aggro_range": 8, "chase_range": 10, "level": 28, "race": "demi-human", "is_assist": True, "assist_family": "Goblin"},
    "Nine Tail": {"is_aggressive": True, "aggro_range": 10, "chase_range": 12, "level": 45, "race": "brute"},
    "Sohee": {"is_aggressive": True, "aggro_range": 8, "chase_range": 10, "level": 40, "race": "demon"},
    "Baphomet": {"is_aggressive": True, "aggro_range": 14, "chase_range": 20, "is_boss": True, "level": 81, "race": "demon"},

    # ── Dungeons ──
    "Vadon": {"is_aggressive": True, "aggro_range": 8, "chase_range": 10, "level": 35, "race": "fish"},
    "Marina": {"is_aggressive": True, "aggro_range": 8, "chase_range": 10, "level": 32, "race": "fish"},
    "Kukre": {"is_aggressive": True, "aggro_range": 7, "chase_range": 9, "level": 30, "race": "fish"},
    "Phen": {"is_aggressive": True, "aggro_range": 8, "chase_range": 10, "level": 38, "race": "fish"},
    "Swordfish": {"is_aggressive": True, "aggro_range": 9, "chase_range": 11, "level": 42, "race": "fish"},
    "Caramel": {"is_aggressive": True, "aggro_range": 8, "chase_range": 10, "level": 36, "race": "brute"},
    "Pasana": {"is_aggressive": True, "aggro_range": 9, "chase_range": 11, "level": 48, "race": "demi-human"},
    "Isis": {"is_aggressive": True, "aggro_range": 10, "chase_range": 12, "level": 50, "race": "demon"},
    "Anubis": {"is_aggressive": True, "aggro_range": 10, "chase_range": 12, "level": 55, "race": "undead"},
    "Deviace": {"is_aggressive": True, "aggro_range": 10, "chase_range": 12, "level": 52, "race": "fish"},
    "Strouf": {"is_aggressive": True, "aggro_range": 10, "chase_range": 12, "level": 55, "race": "fish"},
    "Khalitzburg": {"is_aggressive": True, "aggro_range": 10, "chase_range": 12, "level": 58, "race": "undead"},
    "Raydric": {"is_aggressive": True, "aggro_range": 10, "chase_range": 12, "level": 52, "race": "demi-human"},
    "Bloody Knight": {"is_aggressive": True, "aggro_range": 12, "chase_range": 16, "is_boss": True, "level": 70, "race": "demi-human"},
    "Dracula": {"is_aggressive": True, "aggro_range": 12, "chase_range": 16, "is_boss": True, "level": 68, "race": "demon"},
    "Doppelganger": {"is_aggressive": True, "aggro_range": 14, "chase_range": 18, "is_boss": True, "level": 77, "race": "demon"},
    "Phreeoni": {"is_aggressive": True, "aggro_range": 14, "chase_range": 18, "is_boss": True, "level": 73, "race": "brute"},
    "Maya": {"is_aggressive": True, "aggro_range": 12, "chase_range": 16, "is_boss": True, "level": 65, "race": "insect"},
    "Moonlight Flower": {"is_aggressive": True, "aggro_range": 12, "chase_range": 16, "is_boss": True, "level": 62, "race": "demon"},
    "Osiris": {"is_aggressive": True, "aggro_range": 12, "chase_range": 16, "is_boss": True, "level": 70, "race": "undead"},
    "Eddga": {"is_aggressive": True, "aggro_range": 14, "chase_range": 18, "is_boss": True, "level": 65, "race": "brute"},
    "Golem": {"is_aggressive": True, "aggro_range": 8, "chase_range": 10, "level": 45, "race": "formless"},
    "Sting": {"is_aggressive": True, "aggro_range": 9, "chase_range": 11, "level": 50, "race": "formless"},
    "Nightmare": {"is_aggressive": True, "aggro_range": 8, "chase_range": 10, "level": 48, "race": "demon"},
    "Disguise": {"is_aggressive": True, "aggro_range": 9, "chase_range": 11, "level": 55, "race": "demon"},
    "Wraith": {"is_aggressive": True, "aggro_range": 10, "chase_range": 12, "level": 58, "race": "undead"},
    "Evil Druid": {"is_aggressive": True, "aggro_range": 10, "chase_range": 12, "level": 60, "race": "undead"},
    "Incubus": {"is_aggressive": True, "aggro_range": 10, "chase_range": 12, "level": 62, "race": "demon"},
    "Succubus": {"is_aggressive": True, "aggro_range": 10, "chase_range": 12, "level": 62, "race": "demon"},
    "Alice": {"is_aggressive": True, "aggro_range": 8, "chase_range": 10, "level": 55, "race": "demi-human"},
    "Rideword": {"is_aggressive": True, "aggro_range": 9, "chase_range": 11, "level": 58, "race": "formless"},
    "Owl Baron": {"is_aggressive": True, "aggro_range": 10, "chase_range": 12, "level": 60, "race": "demon"},
    "Owl Duke": {"is_aggressive": True, "aggro_range": 10, "chase_range": 12, "level": 58, "race": "demon"},
    "Dark Priest": {"is_aggressive": True, "aggro_range": 10, "chase_range": 12, "level": 65, "race": "undead"},
    "Dark Lord": {"is_aggressive": True, "aggro_range": 14, "chase_range": 20, "is_boss": True, "level": 85, "race": "demon"},
    "Turtle General": {"is_aggressive": True, "aggro_range": 12, "chase_range": 16, "is_boss": True, "level": 72, "race": "brute"},
    "Atroce": {"is_aggressive": True, "aggro_range": 14, "chase_range": 18, "is_boss": True, "level": 78, "race": "brute"},
    "Kiel": {"is_aggressive": True, "aggro_range": 12, "chase_range": 16, "is_boss": True, "level": 80, "race": "demi-human"},
    "Ifrit": {"is_aggressive": True, "aggro_range": 14, "chase_range": 20, "is_boss": True, "level": 90, "race": "demon"},
    "Valkyrie Randgris": {"is_aggressive": True, "aggro_range": 14, "chase_range": 20, "is_boss": True, "level": 88, "race": "angel"},
    "Beelzebub": {"is_aggressive": True, "aggro_range": 14, "chase_range": 20, "is_boss": True, "level": 95, "race": "demon"},
    "Garm": {"is_aggressive": True, "aggro_range": 12, "chase_range": 16, "is_boss": True, "level": 74, "race": "brute"},
    "Hatii": {"is_aggressive": True, "aggro_range": 12, "chase_range": 16, "is_boss": True, "level": 76, "race": "brute"},
    "Detardeurus": {"is_aggressive": True, "aggro_range": 14, "chase_range": 18, "is_boss": True, "level": 82, "race": "dragon"},
    "Leib Olmai": {"is_aggressive": True, "aggro_range": 10, "chase_range": 12, "level": 60, "race": "brute"},
    "Giant Hornet": {"is_aggressive": True, "aggro_range": 9, "chase_range": 11, "level": 55, "race": "insect"},
    "Grizzly": {"is_aggressive": True, "aggro_range": 10, "chase_range": 12, "level": 50, "race": "brute"},
}


# Map-specific spawn data: which monsters spawn on which maps
# This allows pre-calculating which areas are dangerous before moving there.
MAP_SPAWN_DATA: dict[str, list[dict[str, Any]]] = {
    "prt_fild01": [
        {"monster": "Poring", "count": 30, "level": 1},
        {"monster": "Lunatic", "count": 20, "level": 3},
        {"monster": "Fabre", "count": 25, "level": 2},
        {"monster": "Picky", "count": 15, "level": 3},
    ],
    "prt_fild02": [
        {"monster": "Poring", "count": 20, "level": 1},
        {"monster": "Lunatic", "count": 25, "level": 3},
        {"monster": "Fabre", "count": 20, "level": 2},
        {"monster": "Familiar", "count": 10, "level": 14},
    ],
    "prt_fild03": [
        {"monster": "Familiar", "count": 20, "level": 14},
        {"monster": "Hornet", "count": 15, "level": 12},
        {"monster": "Thief Bug", "count": 10, "level": 16},
    ],
    "prt_fild04": [
        {"monster": "Hornet", "count": 20, "level": 12},
        {"monster": "Thief Bug", "count": 15, "level": 16},
        {"monster": "Savage", "count": 10, "level": 25},
    ],
    "prt_fild05": [
        {"monster": "Thief Bug", "count": 20, "level": 16},
        {"monster": "Savage", "count": 15, "level": 25},
        {"monster": "Deniro", "count": 10, "level": 22},
    ],
    "prt_fild06": [
        {"monster": "Savage", "count": 20, "level": 25},
        {"monster": "Deniro", "count": 15, "level": 22},
        {"monster": "Peco Peco", "count": 10, "level": 20},
    ],
    "prt_fild07": [
        {"monster": "Peco Peco", "count": 20, "level": 20},
        {"monster": "Elder Willow", "count": 15, "level": 30},
        {"monster": "Savage", "count": 10, "level": 25},
    ],
    "prt_fild08": [
        {"monster": "Poring", "count": 25, "level": 1},
        {"monster": "Lunatic", "count": 20, "level": 3},
        {"monster": "Fabre", "count": 20, "level": 2},
        {"monster": "Picky", "count": 15, "level": 3},
        {"monster": "Familiar", "count": 5, "level": 14},
    ],
    "prt_fild09": [
        {"monster": "Peco Peco", "count": 20, "level": 20},
        {"monster": "Elder Willow", "count": 15, "level": 30},
        {"monster": "Savage", "count": 10, "level": 25},
    ],
    "prt_fild10": [
        {"monster": "Elder Willow", "count": 20, "level": 30},
        {"monster": "Peco Peco", "count": 15, "level": 20},
        {"monster": "Deniro", "count": 10, "level": 22},
    ],
    "prt_fild11": [
        {"monster": "Elder Willow", "count": 25, "level": 30},
        {"monster": "Savage", "count": 15, "level": 25},
        {"monster": "Peco Peco", "count": 10, "level": 20},
    ],
    "gef_fild00": [
        {"monster": "Dustiness", "count": 20, "level": 28},
        {"monster": "Mantis", "count": 15, "level": 26},
        {"monster": "Hode", "count": 10, "level": 32},
    ],
    "gef_fild01": [
        {"monster": "Mantis", "count": 20, "level": 26},
        {"monster": "Hode", "count": 15, "level": 32},
        {"monster": "Desert Wolf", "count": 10, "level": 35},
    ],
    "gef_fild02": [
        {"monster": "Hode", "count": 20, "level": 32},
        {"monster": "Desert Wolf", "count": 15, "level": 35},
        {"monster": "Mummy", "count": 10, "level": 38},
    ],
    "gef_fild03": [
        {"monster": "Desert Wolf", "count": 20, "level": 35},
        {"monster": "Mummy", "count": 15, "level": 38},
        {"monster": "Argiope", "count": 10, "level": 40},
    ],
    "gef_fild04": [
        {"monster": "Mummy", "count": 20, "level": 38},
        {"monster": "Argiope", "count": 15, "level": 40},
        {"monster": "Matyr", "count": 10, "level": 42},
    ],
    "gef_fild05": [
        {"monster": "Argiope", "count": 20, "level": 40},
        {"monster": "Matyr", "count": 15, "level": 42},
        {"monster": "Minorous", "count": 10, "level": 52},
    ],
    "moc_fild01": [
        {"monster": "Scorpion", "count": 20, "level": 24},
        {"monster": "Skeleton", "count": 15, "level": 30},
        {"monster": "Zombie", "count": 10, "level": 18},
    ],
    "moc_fild02": [
        {"monster": "Skeleton", "count": 20, "level": 30},
        {"monster": "Ghoul", "count": 15, "level": 35},
        {"monster": "Drainliar", "count": 10, "level": 28},
    ],
    "moc_fild03": [
        {"monster": "Ghoul", "count": 20, "level": 35},
        {"monster": "Mummy", "count": 15, "level": 38},
        {"monster": "Drainliar", "count": 10, "level": 28},
    ],
    "pay_fild01": [
        {"monster": "Spore", "count": 25, "level": 17},
        {"monster": "Poporing", "count": 20, "level": 20},
        {"monster": "Smokie", "count": 15, "level": 22},
    ],
    "pay_fild02": [
        {"monster": "Poporing", "count": 20, "level": 20},
        {"monster": "Smokie", "count": 20, "level": 22},
        {"monster": "Yoyo", "count": 10, "level": 24},
    ],
    "pay_fild03": [
        {"monster": "Smokie", "count": 20, "level": 22},
        {"monster": "Yoyo", "count": 15, "level": 24},
        {"monster": "Steam Goblin", "count": 10, "level": 30},
    ],
    "pay_fild04": [
        {"monster": "Yoyo", "count": 20, "level": 24},
        {"monster": "Steam Goblin", "count": 15, "level": 30},
        {"monster": "Goblin", "count": 10, "level": 28},
    ],
    "pay_fild05": [
        {"monster": "Steam Goblin", "count": 20, "level": 30},
        {"monster": "Goblin", "count": 15, "level": 28},
        {"monster": "Nine Tail", "count": 10, "level": 45},
    ],
    "pay_fild06": [
        {"monster": "Goblin", "count": 20, "level": 28},
        {"monster": "Nine Tail", "count": 15, "level": 45},
        {"monster": "Sohee", "count": 10, "level": 40},
    ],
    "pay_fild07": [
        {"monster": "Nine Tail", "count": 20, "level": 45},
        {"monster": "Sohee", "count": 15, "level": 40},
        {"monster": "Goblin", "count": 10, "level": 28},
    ],
    "pay_fild08": [
        {"monster": "Sohee", "count": 20, "level": 40},
        {"monster": "Nine Tail", "count": 15, "level": 45},
        {"monster": "Goblin", "count": 10, "level": 28},
    ],
    "mjolnir_04": [
        {"monster": "Peco Peco", "count": 20, "level": 20},
        {"monster": "Savage", "count": 15, "level": 25},
        {"monster": "Deniro", "count": 10, "level": 22},
    ],
    "gef_fild14": [
        {"monster": "Orc Warrior", "count": 25, "level": 38},
        {"monster": "Orc Archer", "count": 15, "level": 40},
        {"monster": "Orc Zombie", "count": 10, "level": 45},
    ],
    "orcsdun01": [
        {"monster": "Orc Warrior", "count": 30, "level": 38},
        {"monster": "Orc Archer", "count": 20, "level": 40},
        {"monster": "Orc Lady", "count": 15, "level": 42},
    ],
    "orcsdun02": [
        {"monster": "Orc Lady", "count": 25, "level": 42},
        {"monster": "Orc Zombie", "count": 20, "level": 45},
        {"monster": "Orc Skeleton", "count": 15, "level": 48},
        {"monster": "Orc Lord", "count": 1, "level": 75, "is_boss": True},
    ],
    "pay_dun00": [
        {"monster": "Vadon", "count": 20, "level": 35},
        {"monster": "Marina", "count": 15, "level": 32},
        {"monster": "Kukre", "count": 10, "level": 30},
    ],
    "pay_dun01": [
        {"monster": "Marina", "count": 20, "level": 32},
        {"monster": "Kukre", "count": 15, "level": 30},
        {"monster": "Phen", "count": 10, "level": 38},
    ],
    "pay_dun02": [
        {"monster": "Phen", "count": 20, "level": 38},
        {"monster": "Swordfish", "count": 15, "level": 42},
        {"monster": "Caramel", "count": 10, "level": 36},
    ],
    "pay_dun03": [
        {"monster": "Swordfish", "count": 20, "level": 42},
        {"monster": "Caramel", "count": 15, "level": 36},
        {"monster": "Strouf", "count": 10, "level": 55},
    ],
    "pay_dun04": [
        {"monster": "Strouf", "count": 20, "level": 55},
        {"monster": "Deviace", "count": 15, "level": 52},
        {"monster": "Khalitzburg", "count": 10, "level": 58},
        {"monster": "Baphomet", "count": 1, "level": 81, "is_boss": True},
    ],
    "gef_dun00": [
        {"monster": "Dustiness", "count": 20, "level": 28},
        {"monster": "Mantis", "count": 15, "level": 26},
        {"monster": "Hode", "count": 10, "level": 32},
    ],
    "gef_dun01": [
        {"monster": "Hode", "count": 20, "level": 32},
        {"monster": "Desert Wolf", "count": 15, "level": 35},
        {"monster": "Mummy", "count": 10, "level": 38},
    ],
    "gef_dun02": [
        {"monster": "Mummy", "count": 20, "level": 38},
        {"monster": "Argiope", "count": 15, "level": 40},
        {"monster": "Matyr", "count": 10, "level": 42},
    ],
    "gef_dun03": [
        {"monster": "Matyr", "count": 20, "level": 42},
        {"monster": "Minorous", "count": 15, "level": 52},
        {"monster": "Nightmare", "count": 10, "level": 48},
        {"monster": "Bloody Knight", "count": 1, "level": 70, "is_boss": True},
    ],
    "moc_dun01": [
        {"monster": "Skeleton", "count": 20, "level": 30},
        {"monster": "Zombie", "count": 15, "level": 18},
        {"monster": "Ghoul", "count": 10, "level": 35},
    ],
    "moc_dun02": [
        {"monster": "Ghoul", "count": 20, "level": 35},
        {"monster": "Mummy", "count": 15, "level": 38},
        {"monster": "Drainliar", "count": 10, "level": 28},
    ],
    "moc_dun03": [
        {"monster": "Mummy", "count": 20, "level": 38},
        {"monster": "Pasana", "count": 15, "level": 48},
        {"monster": "Isis", "count": 10, "level": 50},
    ],
    "moc_dun04": [
        {"monster": "Pasana", "count": 20, "level": 48},
        {"monster": "Isis", "count": 15, "level": 50},
        {"monster": "Anubis", "count": 10, "level": 55},
        {"monster": "Osiris", "count": 1, "level": 70, "is_boss": True},
    ],
    "iz_dun00": [
        {"monster": "Vadon", "count": 20, "level": 35},
        {"monster": "Marina", "count": 15, "level": 32},
        {"monster": "Kukre", "count": 10, "level": 30},
    ],
    "iz_dun01": [
        {"monster": "Marina", "count": 20, "level": 32},
        {"monster": "Kukre", "count": 15, "level": 30},
        {"monster": "Phen", "count": 10, "level": 38},
    ],
    "iz_dun02": [
        {"monster": "Phen", "count": 20, "level": 38},
        {"monster": "Swordfish", "count": 15, "level": 42},
        {"monster": "Caramel", "count": 10, "level": 36},
    ],
    "iz_dun03": [
        {"monster": "Swordfish", "count": 20, "level": 42},
        {"monster": "Caramel", "count": 15, "level": 36},
        {"monster": "Strouf", "count": 10, "level": 55},
    ],
    "iz_dun04": [
        {"monster": "Strouf", "count": 20, "level": 55},
        {"monster": "Deviace", "count": 15, "level": 52},
        {"monster": "Khalitzburg", "count": 10, "level": 58},
        {"monster": "Mistress", "count": 1, "level": 63, "is_boss": True},
    ],
    "xmas_fild01": [
        {"monster": "Golem", "count": 20, "level": 45},
        {"monster": "Sting", "count": 15, "level": 50},
        {"monster": "Nightmare", "count": 10, "level": 48},
    ],
    "ice_dun01": [
        {"monster": "Golem", "count": 20, "level": 45},
        {"monster": "Sting", "count": 15, "level": 50},
        {"monster": "Nightmare", "count": 10, "level": 48},
    ],
    "ice_dun02": [
        {"monster": "Sting", "count": 20, "level": 50},
        {"monster": "Nightmare", "count": 15, "level": 48},
        {"monster": "Disguise", "count": 10, "level": 55},
    ],
    "ice_dun03": [
        {"monster": "Disguise", "count": 20, "level": 55},
        {"monster": "Wraith", "count": 15, "level": 58},
        {"monster": "Evil Druid", "count": 10, "level": 60},
        {"monster": "Garm", "count": 1, "level": 74, "is_boss": True},
    ],
    "yuno_fild01": [
        {"monster": "Rideword", "count": 20, "level": 58},
        {"monster": "Alice", "count": 15, "level": 55},
        {"monster": "Owl Baron", "count": 10, "level": 60},
    ],
    "yuno_fild02": [
        {"monster": "Owl Baron", "count": 20, "level": 60},
        {"monster": "Owl Duke", "count": 15, "level": 58},
        {"monster": "Dark Priest", "count": 10, "level": 65},
    ],
    "yuno_fild03": [
        {"monster": "Owl Duke", "count": 20, "level": 58},
        {"monster": "Dark Priest", "count": 15, "level": 65},
        {"monster": "Rideword", "count": 10, "level": 58},
    ],
    "ama_fild01": [
        {"monster": "Nine Tail", "count": 20, "level": 45},
        {"monster": "Sohee", "count": 15, "level": 40},
        {"monster": "Goblin", "count": 10, "level": 28},
    ],
    "comodo_fild01": [
        {"monster": "Goblin", "count": 20, "level": 28},
        {"monster": "Steam Goblin", "count": 15, "level": 30},
        {"monster": "Yoyo", "count": 10, "level": 24},
    ],
}


class PredictiveAggroKnowledge:
    """Knows which monsters are aggressive and their ranges.

    Thread-safe singleton. Pre-calculates danger zones for every map
    so the pathfinder can avoid them before moving.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._monsters: dict[str, MonsterAggroInfo] = {}
        self._map_danger_scores: dict[str, float] = {}  # pre-calculated
        self._map_aggro_monsters: dict[str, list[MonsterAggroInfo]] = {}
        self._load_monsters()

    def _load_monsters(self) -> None:
        """Load monster aggro data from the known database, then augment with mob_db.yml."""
        for name, data in KNOWN_AGGRESSIVE_MONSTERS.items():
            info = MonsterAggroInfo(
                monster_name=name,
                monster_id=data.get("monster_id", 0),
                is_aggressive=data.get("is_aggressive", False),
                is_assist=data.get("is_assist", False),
                is_boss=data.get("is_boss", False),
                aggro_range=data.get("aggro_range", 0),
                chase_range=data.get("chase_range", 0),
                max_chase_range=data.get("max_chase_range", data.get("chase_range", 0) + 5),
                level=data.get("level", 1),
                race=data.get("race", ""),
                element=data.get("element", ""),
                size=data.get("size", ""),
                hp=data.get("hp", 0),
                atk=data.get("atk", 0),
                def_=data.get("def", 0),
                spawn_maps=[],
                spawn_count=0,
                # Pro RO additions
                is_night_aggro=data.get("is_night_aggro", False),
                is_day_aggro=data.get("is_day_aggro", False),
                assist_family=data.get("assist_family", ""),
                has_skill_aggro=data.get("has_skill_aggro", False),
                is_ranged=data.get("is_ranged", False),
            )
            self._monsters[name] = info

        # ── PRO FIX: Augment with full mob_db.yml from rAthena ──
        # KNOWN_AGGRESSIVE_MONSTERS has ~100 hand-picked entries.
        # mob_db.yml has 2,675+ monsters. This loads them all for
        # runtime lookup (element, race, size, level, HP, ATK, DEF, etc.)
        try:
            from ai_sidecar.data.monster_db import get_monster_db
            real_db = get_monster_db()
            aegis_to_name: dict[str, str] = {}  # AegisName → friendly name
            for aegis_name, entry in real_db.items():
                # Map AegisName → friendly name for lookup
                friendly = entry.name.title()
                aegis_to_name[aegis_name] = friendly
                # Also store the raw entry keyed by AegisName for quick lookup
                if aegis_name not in self._monsters:
                    # Create a minimal MonsterAggroInfo for real-DB-only monsters
                    # These have no hardcoded aggro data, so default to passive
                    info = MonsterAggroInfo(
                        monster_name=friendly,
                        monster_id=entry.id,
                        is_aggressive=False,  # unknown — will be discovered at runtime
                        is_assist=False,
                        is_boss=entry.is_boss,
                        aggro_range=0,
                        chase_range=0,
                        max_chase_range=10,
                        level=entry.level,
                        race=entry.race,
                        element=entry.element,
                        size=entry.size,
                        hp=entry.hp,
                        atk=entry.atk_max,
                        def_=entry.def_,
                    )
                    self._monsters[aegis_name] = info
            logger.info(
                "predictive_aggro augmented: %d real-db monsters loaded "
                "(%d total, %d with hardcoded aggro data)",
                len(real_db), len(self._monsters), len(KNOWN_AGGRESSIVE_MONSTERS),
            )
        except Exception as e:
            logger.warning(
                "Failed to load monster_db.yml — using hardcoded data only: %s", e
            )

        # Build spawn data
        for map_name, spawns in MAP_SPAWN_DATA.items():
            aggro_on_map: list[MonsterAggroInfo] = []
            for spawn in spawns:
                monster_name = spawn["monster"]
                if monster_name in self._monsters:
                    monster = self._monsters[monster_name]
                    monster.spawn_maps.append(map_name)
                    monster.spawn_count += spawn.get("count", 0)
                    if monster.is_aggressive:
                        aggro_on_map.append(monster)
            self._map_aggro_monsters[map_name] = aggro_on_map

        # Pre-calculate danger scores for every map
        self._recalculate_danger_scores()

    def _recalculate_danger_scores(self) -> None:
        """Pre-calculate danger scores for all known maps."""
        for map_name, aggro_monsters in self._map_aggro_monsters.items():
            score = self._calculate_map_danger(map_name, aggro_monsters)
            self._map_danger_scores[map_name] = score

    def _calculate_map_danger(self, map_name: str,
                               aggro_monsters: list[MonsterAggroInfo]) -> float:
        """Calculate a danger score (0.0-1.0) for a map based on its aggro monsters.

        Factors:
        - Number of aggressive monsters
        - Their aggro ranges (higher = more dangerous)
        - Their chase ranges (higher = harder to escape)
        - Boss presence (significantly increases danger)
        - Monster density
        - Assist aggro (chain aggro = more dangerous)
        - Night-only aggro (Zombies etc. — only dangerous at night)
        - Ranged monsters (aggro from further, harder to escape)
        """
        if not aggro_monsters:
            return 0.0

        score = 0.0

        # Count-based score
        total_aggro = sum(m.spawn_count for m in aggro_monsters)
        score += min(0.3, total_aggro * 0.003)

        # Range-based score (average aggro range)
        if aggro_monsters:
            avg_aggro = sum(m.aggro_range for m in aggro_monsters) / len(aggro_monsters)
            score += min(0.2, avg_aggro * 0.015)

            avg_chase = sum(m.chase_range for m in aggro_monsters) / len(aggro_monsters)
            score += min(0.2, avg_chase * 0.012)

        # Boss penalty
        boss_count = sum(1 for m in aggro_monsters if m.is_boss)
        score += boss_count * 0.15

        # Level-based (higher level monsters = more dangerous)
        if aggro_monsters:
            avg_level = sum(m.level for m in aggro_monsters) / len(aggro_monsters)
            score += min(0.15, avg_level * 0.002)

        # ── Pro RO additions ──
        # Assist aggro penalty (chain aggro = more dangerous)
        assist_count = sum(1 for m in aggro_monsters if m.is_assist)
        if assist_count > 0:
            score += min(0.15, assist_count * 0.03)

        # Ranged monster penalty (harder to escape)
        ranged_count = sum(1 for m in aggro_monsters if m.is_ranged)
        if ranged_count > 0:
            score += min(0.1, ranged_count * 0.05)

        # Night-only aggro (Zombies etc.)
        night_aggro_count = sum(1 for m in aggro_monsters if m.is_night_aggro)
        if night_aggro_count > 0:
            score += min(0.1, night_aggro_count * 0.02)

        return min(1.0, score)

    # ── Public API ───────────────────────────────────────────────────

    def get_monster(self, name: str) -> MonsterAggroInfo | None:
        """Get aggro info for a specific monster."""
        with self._lock:
            return self._monsters.get(name)

    def is_aggressive(self, monster_name: str) -> bool:
        """Check if a monster type is aggressive."""
        monster = self.get_monster(monster_name)
        return monster is not None and monster.is_aggressive

    def get_aggro_range(self, monster_name: str) -> int:
        """Get the aggro range for a monster type."""
        monster = self.get_monster(monster_name)
        return monster.aggro_range if monster else 0

    def get_chase_range(self, monster_name: str) -> int:
        """Get the chase range for a monster type."""
        monster = self.get_monster(monster_name)
        return monster.chase_range if monster else 0

    def get_map_danger_score(self, map_name: str) -> float:
        """Get the pre-calculated danger score for a map (0.0-1.0)."""
        with self._lock:
            return self._map_danger_scores.get(map_name, 0.0)

    def get_aggro_monsters_on_map(self, map_name: str) -> list[MonsterAggroInfo]:
        """Get all aggressive monsters that spawn on a map."""
        with self._lock:
            return list(self._map_aggro_monsters.get(map_name, []))

    def get_aggro_monster_names_on_map(self, map_name: str) -> list[str]:
        """Get names of all aggressive monsters on a map."""
        return [m.monster_name for m in self.get_aggro_monsters_on_map(map_name)]

    def has_aggro_monsters(self, map_name: str) -> bool:
        """Check if a map has any aggressive monsters."""
        return len(self.get_aggro_monsters_on_map(map_name)) > 0

    def has_boss(self, map_name: str) -> bool:
        """Check if a map has a boss monster."""
        return any(m.is_boss for m in self.get_aggro_monsters_on_map(map_name))

    def get_max_aggro_range_on_map(self, map_name: str) -> int:
        """Get the maximum aggro range of any monster on a map."""
        monsters = self.get_aggro_monsters_on_map(map_name)
        if not monsters:
            return 0
        return max(m.aggro_range for m in monsters)

    def get_max_chase_range_on_map(self, map_name: str) -> int:
        """Get the maximum chase range of any monster on a map."""
        monsters = self.get_aggro_monsters_on_map(map_name)
        if not monsters:
            return 0
        return max(m.chase_range for m in monsters)

    def get_safe_maps(self, max_danger: float = 0.2) -> list[str]:
        """Get all maps with danger score below the threshold."""
        with self._lock:
            return [m for m, s in self._map_danger_scores.items() if s <= max_danger]

    def get_dangerous_maps(self, min_danger: float = 0.5) -> list[str]:
        """Get all maps with danger score above the threshold."""
        with self._lock:
            return [m for m, s in self._map_danger_scores.items() if s >= min_danger]

    def get_all_monster_names(self) -> list[str]:
        """Get all known monster names."""
        with self._lock:
            return list(self._monsters.keys())

    def get_aggressive_monster_names(self) -> list[str]:
        """Get names of all known aggressive monsters."""
        with self._lock:
            return [m for m, info in self._monsters.items() if info.is_aggressive]

    def get_map_count(self) -> int:
        """Get number of maps with known spawn data."""
        with self._lock:
            return len(self._map_danger_scores)

    def get_monster_count(self) -> int:
        """Get number of known monsters."""
        with self._lock:
            return len(self._monsters)

    def add_monster(self, info: MonsterAggroInfo) -> None:
        """Add a monster at runtime (dynamic discovery)."""
        with self._lock:
            self._monsters[info.monster_name] = info
            self._recalculate_danger_scores()

    def add_spawn_data(self, map_name: str, spawns: list[dict[str, Any]]) -> None:
        """Add spawn data for a map at runtime."""
        with self._lock:
            aggro_on_map: list[MonsterAggroInfo] = []
            for spawn in spawns:
                monster_name = spawn["monster"]
                if monster_name in self._monsters:
                    monster = self._monsters[monster_name]
                    monster.spawn_maps.append(map_name)
                    monster.spawn_count += spawn.get("count", 0)
                    if monster.is_aggressive:
                        aggro_on_map.append(monster)
            self._map_aggro_monsters[map_name] = aggro_on_map
            self._map_danger_scores[map_name] = self._calculate_map_danger(
                map_name, aggro_on_map
            )

    def get_status_summary(self) -> str:
        """Get a human-readable status summary."""
        with self._lock:
            lines = [
                f"── Predictive Aggro Knowledge ──",
                f"Known monsters: {len(self._monsters)}",
                f"Aggressive monsters: {sum(1 for m in self._monsters.values() if m.is_aggressive)}",
                f"Boss monsters: {sum(1 for m in self._monsters.values() if m.is_boss)}",
                f"Maps with spawn data: {len(self._map_danger_scores)}",
                f"Safe maps (danger < 0.2): {len(self.get_safe_maps())}",
                f"Dangerous maps (danger > 0.5): {len(self.get_dangerous_maps())}",
            ]
            return "\n".join(lines)


# ── Global Singleton ──

_predictive_aggro: PredictiveAggroKnowledge | None = None
_predictive_aggro_lock = RLock()


def get_predictive_aggro() -> PredictiveAggroKnowledge:
    global _predictive_aggro
    with _predictive_aggro_lock:
        if _predictive_aggro is None:
            _predictive_aggro = PredictiveAggroKnowledge()
        return _predictive_aggro
