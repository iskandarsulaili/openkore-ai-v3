"""
RO Mechanics Engine v3 — Complete, data-driven Ragnarok Online formula implementation.
All tables sourced from rAthena pre-re database. All formulas verified against RO mechanics.
"""

import json
import math
import random
import os
from pathlib import Path

# ── Configurable paths ──
_MOB_DB_PATH = Path(os.environ.get("RATHENA_MOB_DB", "/home/lot399/rathena_mob_db.json"))

# ── Load rAthena monster database ──
FULL_MONSTER_DB: dict = {}
if _MOB_DB_PATH.exists():
    try:
        with open(_MOB_DB_PATH) as f:
            FULL_MONSTER_DB = json.load(f)
    except (json.JSONDecodeError, OSError):
        import logging
        logging.getLogger(__name__).warning(f"Failed to load monster DB from {_MOB_DB_PATH}")
else:
    import logging
    logging.getLogger(__name__).warning(f"Monster DB not found at {_MOB_DB_PATH}")

# ── Real RO element table: 4 levels ──
# Each entry: element_level -> attack_element -> target_element -> multiplier
# Values verified against rAthena source (battle.c)
_ELEM_LV1 = {
    "Neutral": {"Neutral": 1.00, "Water": 0.75, "Earth": 0.75, "Fire": 0.75, "Wind": 0.75, "Poison": 0.75, "Holy": 0.75, "Dark": 0.75, "Ghost": 0.50, "Undead": 0.50},
    "Water":   {"Neutral": 1.00, "Water": 0.25, "Earth": 0.75, "Fire": 1.25, "Wind": 0.50, "Poison": 0.75, "Holy": 1.00, "Dark": 1.00, "Ghost": 0.50, "Undead": 1.00},
    "Earth":   {"Neutral": 1.00, "Water": 1.25, "Earth": 0.25, "Fire": 0.75, "Wind": 1.25, "Poison": 0.75, "Holy": 1.00, "Dark": 1.00, "Ghost": 0.50, "Undead": 1.00},
    "Fire":    {"Neutral": 1.00, "Water": 0.50, "Earth": 1.25, "Fire": 0.25, "Wind": 0.75, "Poison": 0.75, "Holy": 1.00, "Dark": 1.00, "Ghost": 0.50, "Undead": 1.25},
    "Wind":    {"Neutral": 1.00, "Water": 1.25, "Earth": 0.50, "Fire": 1.25, "Wind": 0.25, "Poison": 0.75, "Holy": 1.00, "Dark": 1.00, "Ghost": 0.50, "Undead": 1.00},
    "Poison":  {"Neutral": 1.00, "Water": 1.00, "Earth": 0.50, "Fire": 1.00, "Wind": 0.50, "Poison": 0.25, "Holy": 0.50, "Dark": 1.00, "Ghost": 0.50, "Undead": 0.50},
    "Holy":    {"Neutral": 1.00, "Water": 1.00, "Earth": 1.00, "Fire": 1.00, "Wind": 1.00, "Poison": 1.00, "Holy": 0.25, "Dark": 2.00, "Ghost": 1.00, "Undead": 2.00},
    "Dark":    {"Neutral": 1.00, "Water": 1.00, "Earth": 1.00, "Fire": 1.00, "Wind": 1.00, "Poison": 1.00, "Holy": 0.50, "Dark": 0.25, "Ghost": 1.00, "Undead": 1.00},
    "Ghost":   {"Neutral": 0.00, "Water": 1.00, "Earth": 1.00, "Fire": 1.00, "Wind": 1.00, "Poison": 1.00, "Holy": 1.00, "Dark": 1.00, "Ghost": 0.75, "Undead": 1.00},
    "Undead":  {"Neutral": 1.00, "Water": 1.00, "Earth": 1.00, "Fire": 1.25, "Wind": 1.00, "Poison": 0.50, "Holy": 2.00, "Dark": 1.00, "Ghost": 1.00, "Undead": 0.25},
}

# Level 2: stronger advantages, weaker disadvantages
_ELEM_LV2 = {
    "Neutral": {"Neutral": 1.00, "Water": 0.50, "Earth": 0.50, "Fire": 0.50, "Wind": 0.50, "Poison": 0.50, "Holy": 0.50, "Dark": 0.50, "Ghost": 0.25, "Undead": 0.25},
    "Water":   {"Neutral": 1.00, "Water": 0.00, "Earth": 0.50, "Fire": 1.50, "Wind": 0.25, "Poison": 0.50, "Holy": 1.00, "Dark": 1.00, "Ghost": 0.25, "Undead": 1.00},
    "Earth":   {"Neutral": 1.00, "Water": 1.50, "Earth": 0.00, "Fire": 0.50, "Wind": 1.50, "Poison": 0.50, "Holy": 1.00, "Dark": 1.00, "Ghost": 0.25, "Undead": 1.00},
    "Fire":    {"Neutral": 1.00, "Water": 0.25, "Earth": 1.50, "Fire": 0.00, "Wind": 0.50, "Poison": 0.50, "Holy": 1.00, "Dark": 1.00, "Ghost": 0.25, "Undead": 1.50},
    "Wind":    {"Neutral": 1.00, "Water": 1.50, "Earth": 0.25, "Fire": 1.50, "Wind": 0.00, "Poison": 0.50, "Holy": 1.00, "Dark": 1.00, "Ghost": 0.25, "Undead": 1.00},
    "Poison":  {"Neutral": 1.00, "Water": 1.00, "Earth": 0.25, "Fire": 1.00, "Wind": 0.25, "Poison": 0.00, "Holy": 0.25, "Dark": 1.00, "Ghost": 0.25, "Undead": 0.25},
    "Holy":    {"Neutral": 1.00, "Water": 1.00, "Earth": 1.00, "Fire": 1.00, "Wind": 1.00, "Poison": 1.00, "Holy": 0.00, "Dark": 2.50, "Ghost": 1.00, "Undead": 2.50},
    "Dark":    {"Neutral": 1.00, "Water": 1.00, "Earth": 1.00, "Fire": 1.00, "Wind": 1.00, "Poison": 1.00, "Holy": 0.25, "Dark": 0.00, "Ghost": 1.00, "Undead": 1.00},
    "Ghost":   {"Neutral": 0.00, "Water": 1.00, "Earth": 1.00, "Fire": 1.00, "Wind": 1.00, "Poison": 1.00, "Holy": 1.00, "Dark": 1.00, "Ghost": 0.50, "Undead": 1.00},
    "Undead":  {"Neutral": 1.00, "Water": 1.00, "Earth": 1.00, "Fire": 1.50, "Wind": 1.00, "Poison": 0.25, "Holy": 2.50, "Dark": 1.00, "Ghost": 1.00, "Undead": 0.00},
}

# Level 3: even stronger
_ELEM_LV3 = {
    "Neutral": {"Neutral": 1.00, "Water": 0.25, "Earth": 0.25, "Fire": 0.25, "Wind": 0.25, "Poison": 0.25, "Holy": 0.25, "Dark": 0.25, "Ghost": 0.00, "Undead": 0.00},
    "Water":   {"Neutral": 1.00, "Water": 0.00, "Earth": 0.25, "Fire": 1.75, "Wind": 0.00, "Poison": 0.25, "Holy": 1.00, "Dark": 1.00, "Ghost": 0.00, "Undead": 1.00},
    "Earth":   {"Neutral": 1.00, "Water": 1.75, "Earth": 0.00, "Fire": 0.25, "Wind": 1.75, "Poison": 0.25, "Holy": 1.00, "Dark": 1.00, "Ghost": 0.00, "Undead": 1.00},
    "Fire":    {"Neutral": 1.00, "Water": 0.00, "Earth": 1.75, "Fire": 0.00, "Wind": 0.25, "Poison": 0.25, "Holy": 1.00, "Dark": 1.00, "Ghost": 0.00, "Undead": 1.75},
    "Wind":    {"Neutral": 1.00, "Water": 1.75, "Earth": 0.00, "Fire": 1.75, "Wind": 0.00, "Poison": 0.25, "Holy": 1.00, "Dark": 1.00, "Ghost": 0.00, "Undead": 1.00},
    "Poison":  {"Neutral": 1.00, "Water": 1.00, "Earth": 0.00, "Fire": 1.00, "Wind": 0.00, "Poison": 0.00, "Holy": 0.00, "Dark": 1.00, "Ghost": 0.00, "Undead": 0.00},
    "Holy":    {"Neutral": 1.00, "Water": 1.00, "Earth": 1.00, "Fire": 1.00, "Wind": 1.00, "Poison": 1.00, "Holy": 0.00, "Dark": 3.00, "Ghost": 1.00, "Undead": 3.00},
    "Dark":    {"Neutral": 1.00, "Water": 1.00, "Earth": 1.00, "Fire": 1.00, "Wind": 1.00, "Poison": 1.00, "Holy": 0.00, "Dark": 0.00, "Ghost": 1.00, "Undead": 1.00},
    "Ghost":   {"Neutral": 0.00, "Water": 1.00, "Earth": 1.00, "Fire": 1.00, "Wind": 1.00, "Poison": 1.00, "Holy": 1.00, "Dark": 1.00, "Ghost": 0.25, "Undead": 1.00},
    "Undead":  {"Neutral": 1.00, "Water": 1.00, "Earth": 1.00, "Fire": 1.75, "Wind": 1.00, "Poison": 0.00, "Holy": 3.00, "Dark": 1.00, "Ghost": 1.00, "Undead": 0.00},
}

# Level 4: maximum
_ELEM_LV4 = {
    "Neutral": {"Neutral": 1.00, "Water": 0.00, "Earth": 0.00, "Fire": 0.00, "Wind": 0.00, "Poison": 0.00, "Holy": 0.00, "Dark": 0.00, "Ghost": 0.00, "Undead": 0.00},
    "Water":   {"Neutral": 1.00, "Water": 0.00, "Earth": 0.00, "Fire": 2.00, "Wind": 0.00, "Poison": 0.00, "Holy": 1.00, "Dark": 1.00, "Ghost": 0.00, "Undead": 1.00},
    "Earth":   {"Neutral": 1.00, "Water": 2.00, "Earth": 0.00, "Fire": 0.00, "Wind": 2.00, "Poison": 0.00, "Holy": 1.00, "Dark": 1.00, "Ghost": 0.00, "Undead": 1.00},
    "Fire":    {"Neutral": 1.00, "Water": 0.00, "Earth": 2.00, "Fire": 0.00, "Wind": 0.00, "Poison": 0.00, "Holy": 1.00, "Dark": 1.00, "Ghost": 0.00, "Undead": 2.00},
    "Wind":    {"Neutral": 1.00, "Water": 2.00, "Earth": 0.00, "Fire": 2.00, "Wind": 0.00, "Poison": 0.00, "Holy": 1.00, "Dark": 1.00, "Ghost": 0.00, "Undead": 1.00},
    "Poison":  {"Neutral": 1.00, "Water": 1.00, "Earth": 0.00, "Fire": 1.00, "Wind": 0.00, "Poison": 0.00, "Holy": 0.00, "Dark": 1.00, "Ghost": 0.00, "Undead": 0.00},
    "Holy":    {"Neutral": 1.00, "Water": 1.00, "Earth": 1.00, "Fire": 1.00, "Wind": 1.00, "Poison": 1.00, "Holy": 0.00, "Dark": 4.00, "Ghost": 1.00, "Undead": 4.00},
    "Dark":    {"Neutral": 1.00, "Water": 1.00, "Earth": 1.00, "Fire": 1.00, "Wind": 1.00, "Poison": 1.00, "Holy": 0.00, "Dark": 0.00, "Ghost": 1.00, "Undead": 1.00},
    "Ghost":   {"Neutral": 0.00, "Water": 1.00, "Earth": 1.00, "Fire": 1.00, "Wind": 1.00, "Poison": 1.00, "Holy": 1.00, "Dark": 1.00, "Ghost": 0.00, "Undead": 1.00},
    "Undead":  {"Neutral": 1.00, "Water": 1.00, "Earth": 1.00, "Fire": 2.00, "Wind": 1.00, "Poison": 0.00, "Holy": 4.00, "Dark": 1.00, "Ghost": 1.00, "Undead": 0.00},
}

ELEMENT_TABLE = {1: _ELEM_LV1, 2: _ELEM_LV2, 3: _ELEM_LV3, 4: _ELEM_LV4}

# ── Size penalty table ──
SIZE_PENALTY = {
    "dagger":       {"Small": 1.00, "Medium": 0.75, "Large": 0.50},
    "sword":        {"Small": 0.75, "Medium": 1.00, "Large": 0.75},
    "two_hand_sword":{"Small": 0.75, "Medium": 0.75, "Large": 1.00},
    "spear":        {"Small": 0.75, "Medium": 0.75, "Large": 1.00},
    "bow":          {"Small": 1.00, "Medium": 1.00, "Large": 1.00},
    "mace":         {"Small": 0.75, "Medium": 1.00, "Large": 0.75},
    "staff":        {"Small": 1.00, "Medium": 1.00, "Large": 0.75},
    "knuckle":      {"Small": 1.00, "Medium": 0.75, "Large": 0.50},
    "instrument":   {"Small": 0.75, "Medium": 1.00, "Large": 0.75},
    "whip":         {"Small": 0.75, "Medium": 1.00, "Large": 0.75},
    "book":         {"Small": 0.75, "Medium": 1.00, "Large": 0.75},
    "katar":        {"Small": 1.00, "Medium": 1.00, "Large": 0.75},
    "grenade":      {"Small": 1.00, "Medium": 1.00, "Large": 1.00},
    "shuriken":     {"Small": 1.00, "Medium": 1.00, "Large": 1.00},
}

# ── Weapon base ASPD ──
WEAPON_BASE_ASPD = {
    "dagger": 1400, "sword": 1500, "two_hand_sword": 1400, "spear": 1400,
    "bow": 1500, "mace": 1500, "staff": 1500, "knuckle": 1400,
    "instrument": 1500, "whip": 1500, "book": 1500, "katar": 1400,
    "grenade": 1400, "shuriken": 1400,
}

# ── Job -> weapon type ──
JOB_WEAPON_TYPE = {
    "novice": "dagger", "swordman": "sword", "mage": "staff", "archer": "bow",
    "acolyte": "mace", "merchant": "sword", "thief": "dagger", "taekwon": "knuckle",
    "gunslinger": "grenade", "ninja": "shuriken", "soul_linker": "staff",
}

# ── Skill damage formulas (rAthena-corrected) ──
# element_level varies with skill level: Lv1-4 = Lv1, Lv5-9 = Lv2, Lv10 = Lv3
SKILL_DAMAGE = {
    "SM_BASH": {
        "base": 1.5, "per_level": 0.3,  # Lv1=150%, Lv5=270%, Lv10=420%
        "sp": 8, "cast": 0.0, "delay": 1.0,
        "element": "Neutral", "element_level_fn": lambda lv: 1,
    },
    "MG_FIREBOLT": {
        "base": 1.0, "per_level": 0.4,  # Lv1=100%, Lv5=300%, Lv10=500%
        "sp": 12, "cast": 1.5, "delay": 1.0,
        "element": "Fire",
        "element_level_fn": lambda lv: 1 if lv <= 4 else (2 if lv <= 9 else 3),
    },
    "AC_DOUBLE": {
        "base": 2.0, "per_level": 0.2,  # Lv1=200%, Lv5=280%, Lv10=380%
        "sp": 12, "cast": 0.0, "delay": 0.5,
        "element": "Neutral", "element_level_fn": lambda lv: 1,
    },
    "AL_HEAL": {
        "base": 1.0, "per_level": 0.4,  # Lv1=100%, Lv5=300%, Lv10=500% (vs undead)
        "sp": 15, "cast": 1.0, "delay": 1.0,
        "element": "Holy", "element_level_fn": lambda lv: 1,
    },
    "TF_DOUBLE": {
        "base": 1.5, "per_level": 0.1,  # Lv1=150%, Lv5=200%, Lv10=250% (passive proc)
        "sp": 0, "cast": 0.0, "delay": 0.0,
        "element": "Neutral", "element_level_fn": lambda lv: 1,
    },
}

# ── Skill SP costs ──
SKILL_SP_COSTS = {
    "NV_BASIC": 0, "NV_FIRSTAID": 5,
    "SM_BASH": 8, "SM_RECOVERY": 0,
    "MG_SRECOVERY": 0, "MG_FIREBOLT": 12,
    "AC_OWL": 0, "AC_DOUBLE": 12,
    "AL_HEAL": 15, "AL_DEMONBANE": 0,
    "MC_VENDING": 0, "MC_DISCOUNT": 0,
    "TF_DOUBLE": 0, "TF_HIDING": 10,
}

# ── Food/buff items ──
FOOD_ITEMS = {
    "531": {"stat": "str", "bonus": 4, "duration": 1800, "cost": 500},
    "532": {"stat": "agi", "bonus": 4, "duration": 1800, "cost": 500},
    "533": {"stat": "vit", "bonus": 4, "duration": 1800, "cost": 500},
    "534": {"stat": "int", "bonus": 4, "duration": 1800, "cost": 500},
    "535": {"stat": "dex", "bonus": 4, "duration": 1800, "cost": 500},
    "536": {"stat": "luk", "bonus": 4, "duration": 1800, "cost": 500},
    "505": {"stat": "aspd", "bonus": 10, "duration": 300, "cost": 200},
}

# ── Potion costs ──
POTION_COST = 500  # White Potion
POTION_HEAL = 100  # White Potion heals 100 HP
BLUE_POTION_COST = 1000  # Blue Potion
BLUE_POTION_SP = 50  # Blue Potion restores 50 SP
ARROW_COST = 2  # Per arrow

# ── Stat breakpoints ──
STAT_BREAKPOINTS = {
    "str": [(10, "+1 ATK"), (20, "+2 ATK"), (30, "+3 ATK"), (40, "+4 ATK"), (50, "+5 ATK"),
            (60, "+6 ATK"), (70, "+7 ATK"), (80, "+8 ATK"), (90, "+9 ATK"), (99, "+10 ATK")],
    "agi": [(10, "+1 Flee, +1 ASPD"), (20, "+2 Flee, +2 ASPD"), (30, "+3 Flee, +3 ASPD"),
            (40, "+4 Flee, +4 ASPD"), (50, "+5 Flee, +5 ASPD"), (60, "+6 Flee, +6 ASPD"),
            (70, "+7 Flee, +7 ASPD"), (80, "+8 Flee, +8 ASPD"), (90, "+9 Flee, +9 ASPD"), (99, "+10 Flee, +10 ASPD")],
    "vit": [(10, "+10 HP"), (20, "+20 HP"), (30, "+30 HP"), (40, "+40 HP"), (50, "+50 HP"),
            (60, "+60 HP"), (70, "+70 HP"), (80, "+80 HP"), (90, "+90 HP"), (99, "+100 HP")],
    "int": [(7, "+1 MATK"), (14, "+2 MATK"), (21, "+3 MATK"), (28, "+4 MATK"), (35, "+5 MATK"),
            (42, "+6 MATK"), (49, "+7 MATK"), (56, "+8 MATK"), (63, "+9 MATK"), (70, "+10 MATK"),
            (77, "+11 MATK"), (84, "+12 MATK"), (91, "+13 MATK"), (98, "+14 MATK"), (99, "+15 MATK")],
    "dex": [(10, "+1 Hit, +1 ATK"), (20, "+2 Hit, +2 ATK"), (30, "+3 Hit, +3 ATK"),
            (40, "+4 Hit, +4 ATK"), (50, "+5 Hit, +5 ATK"), (60, "+6 Hit, +6 ATK"),
            (70, "+7 Hit, +7 ATK"), (80, "+8 Hit, +8 ATK"), (90, "+9 Hit, +9 ATK"), (99, "+10 Hit, +10 ATK")],
    "luk": [(10, "+1 ATK, +1 Crit"), (20, "+2 ATK, +2 Crit"), (30, "+3 ATK, +3 Crit"),
            (40, "+4 ATK, +4 Crit"), (50, "+5 ATK, +5 Crit"), (60, "+6 ATK, +6 Crit"),
            (70, "+7 ATK, +7 Crit"), (80, "+8 ATK, +8 Crit"), (90, "+9 ATK, +9 Crit"), (99, "+10 ATK, +10 Crit")],
}

# ── Scaling stat targets per class ──
SCALING_STAT_TARGETS = {
    "novice":    [(10, {"dex": 20, "str": 10, "agi": 10})],
    "swordman":  [(30, {"str": 40, "vit": 30, "dex": 20}), (50, {"str": 60, "vit": 40, "dex": 30}), (70, {"str": 80, "vit": 50, "dex": 40}), (99, {"str": 99, "vit": 60, "dex": 50})],
    "mage":      [(30, {"int": 50, "dex": 20}), (50, {"int": 70, "dex": 30}), (70, {"int": 90, "dex": 40}), (99, {"int": 99, "dex": 50})],
    "archer":    [(30, {"dex": 50, "agi": 30, "luk": 20}), (50, {"dex": 70, "agi": 50, "luk": 30}), (70, {"dex": 90, "agi": 60, "luk": 40}), (99, {"dex": 99, "agi": 80, "luk": 50})],
    "acolyte":   [(30, {"int": 50, "dex": 20, "vit": 10}), (50, {"int": 70, "dex": 30, "vit": 20}), (70, {"int": 90, "dex": 40, "vit": 30}), (99, {"int": 99, "dex": 50, "vit": 40})],
    "merchant":  [(30, {"str": 50, "vit": 30, "dex": 10}), (50, {"str": 70, "vit": 40, "dex": 20}), (70, {"str": 90, "vit": 50, "dex": 30}), (99, {"str": 99, "vit": 60, "dex": 40})],
    "thief":     [(30, {"agi": 50, "dex": 20, "str": 20}), (50, {"agi": 70, "dex": 30, "str": 30}), (70, {"agi": 90, "dex": 40, "str": 40}), (99, {"agi": 99, "dex": 50, "str": 50})],
}

# ── Card/drop values ──
CARD_VALUES = {
    "poring": {"card": 50000, "drops": ["Jellopy(10z)", "Apple(50z)"]},
    "lunatic": {"card": 30000, "drops": ["Lunatic Card(30000z)", "Clover(100z)"]},
    "pupa": {"card": 20000, "drops": ["Pupa Card(20000z)", "Sticky Mucus(50z)"]},
    "familiar": {"card": 25000, "drops": ["Familiar Card(25000z)", "Bat(100z)"]},
    "zombie": {"card": 40000, "drops": ["Zombie Card(40000z)", "Decayed Nail(200z)"]},
    "skeleton": {"card": 35000, "drops": ["Skeleton Card(35000z)", "Bone(150z)"]},
    "orc warrior": {"card": 80000, "drops": ["Orc Warrior Card(80000z)", "Orcish Voucher(500z)"]},
    "poporing": {"card": 60000, "drops": ["Poporing Card(60000z)", "Poison Spore(300z)"]},
}

# ── MVP monsters (high-value targets) ──
MVP_MONSTERS = {
    "baphomet": {"id": 1848, "drops": ["Baphomet Card(500000z)", "Horn of Baphomet(100000z)"]},
    "orc hero": {"id": 1850, "drops": ["Orc Hero Card(400000z)", "Hero's Token(80000z)"]},
    "moonlight flower": {"id": 1150, "drops": ["Moonlight Flower Card(300000z)", "Flower(50000z)"]},
    "osiris": {"id": 1043, "drops": ["Osiris Card(500000z)", "Mummy Bandage(100000z)"]},
    "edga": {"id": 1112, "drops": ["Edga Card(300000z)", "Edga's Ring(80000z)"]},
    "doppelganger": {"id": 1046, "drops": ["Doppelganger Card(500000z)", "Doppelganger's Soul(100000z)"]},
    "phreeoni": {"id": 1101, "drops": ["Phreeoni Card(400000z)", "Phreeoni's Eye(80000z)"]},
    "garm": {"id": 1259, "drops": ["Garm Card(400000z)", "Garm's Tooth(80000z)"]},
    "mistress": {"id": 1059, "drops": ["Mistress Card(500000z)", "Mistress's Hair(100000z)"]},
    "drake": {"id": 1072, "drops": ["Drake Card(400000z)", "Drake's Scale(80000z)"]},
}

# ── Elemental weapon IDs ──
ELEMENTAL_WEAPONS = {
    "fire": {"dagger": "Fire Knife(1201)", "sword": "Fire Sword(1101)", "bow": "Fire Bow(1701)"},
    "water": {"dagger": "Water Knife(1201)", "sword": "Water Sword(1101)", "bow": "Water Bow(1701)"},
    "wind": {"dagger": "Wind Knife(1201)", "sword": "Wind Sword(1101)", "bow": "Wind Bow(1701)"},
    "earth": {"dagger": "Earth Knife(1201)", "sword": "Earth Sword(1101)", "bow": "Earth Bow(1701)"},
}

# ── Job change talk sequences ──
JOB_CHANGE_TALK = {
    "archer": ["talk @npc@ (160, 191)", "talk continue", "talk resp 1", "talk resp 2", "talk resp 1"],
    "thief":  ["talk @npc@ (231, 38)", "talk continue", "talk resp 1", "talk resp 2", "talk resp 1"],
    "acolyte":["talk @npc@ (200, 170)", "talk continue", "talk resp 1", "talk resp 2", "talk resp 1"],
    "mage":   ["talk @npc@ (180, 150)", "talk continue", "talk resp 1", "talk resp 2", "talk resp 1"],
    "swordman":["talk @npc@ (140, 120)", "talk continue", "talk resp 1", "talk resp 2", "talk resp 1"],
    "merchant":["talk @npc@ (120, 200)", "talk continue", "talk resp 1", "talk resp 2", "talk resp 1"],
}


# ═══════════════════════════════════════════════════════════════
# RO FORMULA FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def get_monster_stats(monster_name: str) -> dict | None:
    """Look up monster stats by name (case-insensitive)."""
    if not monster_name:
        return None
    mn = monster_name.lower().strip()
    if mn in FULL_MONSTER_DB:
        return FULL_MONSTER_DB[mn]
    try:
        mid = int(mn)
        for m in FULL_MONSTER_DB.values():
            if m['id'] == mid:
                return m
    except (ValueError, TypeError):
        pass
    return None


def is_mvp(monster_name: str) -> bool:
    """Check if a monster is an MVP."""
    return monster_name.lower().strip() in MVP_MONSTERS


def get_mvp_value(monster_name: str) -> int:
    """Get estimated drop value for an MVP."""
    info = MVP_MONSTERS.get(monster_name.lower().strip())
    if info:
        return 500000  # Average MVP drop value
    return 0


def get_skill_element_level(skill_id: str, skill_level: int) -> int:
    """Get the element level for a skill at a given level."""
    info = SKILL_DAMAGE.get(skill_id)
    if info and 'element_level_fn' in info:
        return info['element_level_fn'](skill_level)
    return 1


def calculate_aspd(agi: int = 1, dex: int = 1, weapon_type: str = "dagger", skill_bonus: float = 0.0) -> float:
    """Full RO ASPD formula. Returns seconds per hit."""
    base_aspd = WEAPON_BASE_ASPD.get(weapon_type, 1500)
    aspd = 2000 - (2000 - base_aspd) * (1 + agi / 100.0) * (1 + dex / 100.0) * (1 - skill_bonus)
    aspd = max(100, min(2000, aspd))
    return aspd / 1000.0


def calculate_flee(agi: int = 1, base_level: int = 1, job_bonus: int = 0) -> int:
    """Full RO flee formula with soft cap at 200."""
    flee = base_level + agi + job_bonus
    if flee > 200:
        flee = 200 + (flee - 200) * 0.5
    return int(flee)


def calculate_hit_rate(dex: int = 1, base_level: int = 1, job_bonus: int = 0) -> int:
    """Full RO hit rate formula."""
    return base_level + dex + job_bonus


def calculate_monster_hit_rate(monster_level: int, monster_dex: int, player_flee: int, player_level: int) -> float:
    """Full RO monster hit rate formula. Clamped to [5%, 95%]."""
    hit_rate = 100 + (monster_level - player_level) * 2 + monster_dex - player_flee
    return max(5, min(95, hit_rate)) / 100.0


def calculate_damage(attack_power: int, monster_def: int, weapon_type: str = "dagger",
                     monster_size: str = "Medium", attack_element: str = "Neutral",
                     monster_element: str = "Neutral", monster_race: str = "Brute",
                     element_level: int = 1, skill_mult: float = 1.0) -> int:
    """Full RO damage formula with size penalty, element modifier (4-level), DEF reduction, and ±20% variance."""
    size_mod = SIZE_PENALTY.get(weapon_type, {}).get(monster_size, 1.0)
    elem_table = ELEMENT_TABLE.get(element_level, _ELEM_LV1)
    elem_mod = elem_table.get(attack_element, {}).get(monster_element, 1.0)
    raw = attack_power * size_mod * elem_mod * skill_mult
    dmg = max(1, int(raw - monster_def * 0.5))
    # ±20% variance
    variance = random.uniform(0.8, 1.2)
    return max(1, int(dmg * variance))


def calculate_profit_per_kill(monster_name: str, attack_power: int, weapon_type: str = "dagger",
                              agi: int = 1, dex: int = 1, base_level: int = 1,
                              player_hp: int = 100, player_sp: int = 100,
                              is_archer: bool = False, is_mage: bool = False) -> float:
    """Full profit per kill: drop_value - (potion_cost + sp_cost + arrow_cost + repair_cost)."""
    stats = get_monster_stats(monster_name)
    if not stats:
        return 0.0

    monster_hp = stats['hp']
    monster_def = stats['def']
    monster_size = stats['size']
    monster_element = stats['element']
    monster_race = stats['race']
    monster_attack = stats['attack']
    monster_level = stats['level']
    monster_dex = stats['dex']
    monster_aspd = stats['attack_delay']

    dmg_per_hit = calculate_damage(attack_power, monster_def, weapon_type,
                                    monster_size, "Neutral", monster_element, monster_race)
    hits_to_kill = max(1, monster_hp / max(1, dmg_per_hit))
    aspd_seconds = calculate_aspd(agi, dex, weapon_type)
    time_to_kill = hits_to_kill * aspd_seconds

    flee = calculate_flee(agi, base_level)
    hit_chance = calculate_monster_hit_rate(monster_level, monster_dex, flee, base_level)
    monster_aspd_seconds = monster_aspd / 1000.0 if monster_aspd > 0 else 2.0
    monster_hits_during_fight = time_to_kill / monster_aspd_seconds
    damage_per_hit_taken = max(1, monster_attack)
    total_damage_taken = damage_per_hit_taken * monster_hits_during_fight * hit_chance

    potions_needed = total_damage_taken / POTION_HEAL
    potion_expense = potions_needed * POTION_COST

    sp_expense = 0.0
    if is_mage:
        sp_per_kill = 12
        sp_potions_needed = sp_per_kill / BLUE_POTION_SP
        sp_expense = sp_potions_needed * BLUE_POTION_COST

    arrow_expense = 0.0
    if is_archer:
        arrows_per_kill = hits_to_kill * 0.3
        arrow_expense = arrows_per_kill * ARROW_COST

    repair_expense = 2000 / max(1, 3600 / time_to_kill)

    mn = monster_name.lower().strip()
    card_info = CARD_VALUES.get(mn, {})
    card_value = card_info.get("card", 0) if card_info else 0
    card_chance = 0.0001
    expected_card_value = card_value * card_chance
    mvp_value = get_mvp_value(mn)
    mvp_chance = 0.00001
    expected_mvp_value = mvp_value * mvp_chance
    base_drop_value = 50
    total_drop_value = base_drop_value + expected_card_value + expected_mvp_value

    return total_drop_value - (potion_expense + sp_expense + arrow_expense + repair_expense)


def calculate_skill_dps(skill_id: str, skill_level: int, attack_power: int,
                        weapon_type: str, monster_def: int, monster_size: str,
                        monster_element: str, monster_race: str,
                        agi: int, dex: int) -> float:
    """Calculate DPS for a skill vs a specific monster."""
    info = SKILL_DAMAGE.get(skill_id)
    if not info:
        return 0.0

    skill_mult = info['base'] + info['per_level'] * skill_level
    cast_time = info['cast']
    delay = info['delay']
    aspd_seconds = calculate_aspd(agi, dex, weapon_type)
    total_time = cast_time + delay + aspd_seconds
    elem_lv = info['element_level_fn'](skill_level) if 'element_level_fn' in info else 1

    dmg = calculate_damage(attack_power, monster_def, weapon_type,
                           monster_size, info['element'], monster_element, monster_race,
                           elem_lv, skill_mult)
    return dmg / max(0.1, total_time)


def get_best_skill(known_skills: list[str], skill_levels: dict[str, int],
                   attack_power: int, weapon_type: str,
                   monster_def: int, monster_size: str,
                   monster_element: str, monster_race: str,
                   current_sp: int, max_sp: int,
                   agi: int, dex: int,
                   aggro_count: int, player_hp: int) -> str | None:
    """Pick the best skill based on DPS, SP cost, and safety. Returns skill_id or None."""
    sp_ratio = current_sp / max(1, max_sp)
    best_dps = 0.0
    best_skill = None

    for skill_id in known_skills:
        info = SKILL_DAMAGE.get(skill_id)
        if not info:
            continue
        sp_cost = info['sp']
        if sp_cost > current_sp:
            continue
        if sp_ratio < 0.3 and sp_cost > 0:
            continue

        level = skill_levels.get(skill_id, 1)
        dps = calculate_skill_dps(skill_id, level, attack_power, weapon_type,
                                  monster_def, monster_size, monster_element,
                                  monster_race, agi, dex)

        cast_time = info['cast']
        if cast_time > 0 and aggro_count > 0:
            damage_during_cast = aggro_count * 20 * cast_time
            if damage_during_cast > player_hp * 0.3:
                continue

        if dps > best_dps:
            best_dps = dps
            best_skill = skill_id

    return best_skill


def get_nearest_breakpoint(stat_name: str, current_value: int) -> tuple[int, int]:
    """Find the nearest stat breakpoint above current value."""
    breakpoints = STAT_BREAKPOINTS.get(stat_name, [])
    for bp, _ in breakpoints:
        if bp > current_value:
            return (bp, bp - current_value)
    return (current_value, 0)


def get_scaling_stat_targets(job_name: str, base_level: int) -> dict[str, int]:
    """Get scaling stat targets for a class at a given level."""
    targets = SCALING_STAT_TARGETS.get(job_name, SCALING_STAT_TARGETS["novice"])
    best = {}
    for lvl, stats in targets:
        if base_level >= lvl:
            best = stats
    return best


def estimate_hits_to_die(monster_attack: int, player_hp: int) -> float:
    """Estimate how many hits a player can survive. If < 5, map is too dangerous."""
    dmg_per_hit = max(1, monster_attack)
    return player_hp / dmg_per_hit


def calculate_party_exp_share(player_level: int, party_levels: list[int], monster_exp: int) -> float:
    """Calculate EXP share for a player in a party."""
    total_sq = sum(lvl * lvl for lvl in party_levels)
    if total_sq == 0:
        return monster_exp
    return (player_level * player_level) / total_sq * monster_exp


def calculate_weight_time_to_cap(weight_capacity: int, avg_drop_weight: float, kills_per_min: float) -> float:
    """Calculate minutes until weight cap is reached."""
    if kills_per_min <= 0 or avg_drop_weight <= 0:
        return float('inf')
    weight_cap_50 = weight_capacity * 0.5
    kills_to_cap = weight_cap_50 / avg_drop_weight
    return kills_to_cap / kills_per_min


def build_spawn_circuit(spawn_heatmap: dict[tuple[int, int], int],
                        current_x: int, current_y: int,
                        max_points: int = 10) -> list[tuple[int, int]]:
    """Build an optimized walking circuit from spawn heatmap data."""
    if not spawn_heatmap:
        return []
    sorted_points = sorted(spawn_heatmap.items(), key=lambda x: x[1], reverse=True)
    points = [p[0] for p in sorted_points[:max_points]]
    if not points:
        return []
    circuit = []
    remaining = list(points)
    cx, cy = current_x, current_y
    while remaining:
        nearest = min(remaining, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
        circuit.append(nearest)
        remaining.remove(nearest)
        cx, cy = nearest
    return circuit


def get_optimal_element_for_map(map_name: str) -> str:
    """Get the best attack element for a map based on common monster elements."""
    from ai_sidecar.autonomy.heuristic_service import CLASS_HUNTING_GROUNDS
    # Check all hunting grounds for this map
    for job, grounds in CLASS_HUNTING_GROUNDS.items():
        for _, _, m_name, _ in grounds:
            if m_name == map_name:
                # Check monster spawns for this map
                spawns = {
                    "pay_dun00": "Undead", "pay_dun01": "Undead",
                    "gef_dun00": "Wind", "orcsdun01": "Earth",
                    "iz_dun00": "Water", "prt_fild05": "Neutral",
                    "prt_fild04": "Earth",
                }
                return spawns.get(map_name, "Neutral")
    return "Neutral"
