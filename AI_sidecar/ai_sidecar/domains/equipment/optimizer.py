"""Equipment optimizer — upgrade levels, cards, elements, and slot-aware calculations.

Real RO data: weapon upgrade effects, card database, element multipliers,
elemental converters, and equipment swap recommendations.

Features:
  - Stat-to-equipment mapping: which equipment gives which stats
  - Set bonuses: track item set effects (e.g., Orleans set, Valkyrie set)
  - Card slot analysis: which cards for which hunting target
  - Refine-aware scoring: higher refine = more ATK but more weight
  - Loadout presets: farming, PvP, MVP, tank, flee
  - Auto-swap conditions: swap to fire weapon on earth mobs
  - Element weapon priority: given target element, recommend best weapon + card combo
  - Durability monitoring: track equipment durability and auto-repair
  - Weight optimization: maximize ATK/DEF per weight unit
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Weapon Upgrade Levels (+0 to +10)
# ──────────────────────────────────────────────

_WEAPON_UPGRADE_BONUS: dict[int, int] = {
    0: 0, 1: 3, 2: 6, 3: 9, 4: 12,
    5: 17, 6: 22, 7: 27, 8: 32, 9: 37, 10: 42,
}

_ARMOR_UPGRADE_BONUS: dict[int, int] = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4,
    5: 6, 6: 8, 7: 10, 8: 12, 9: 14, 10: 16,
}

_UPGRADE_SUCCESS_RATES: dict[int, float] = {
    0: 1.00, 1: 1.00, 2: 1.00, 3: 1.00,
    4: 0.60, 5: 0.40, 6: 0.40, 7: 0.20, 8: 0.20, 9: 0.08,
}

_UPGRADE_COST_PER_TRY: dict[int, int] = {
    0: 55000, 1: 55000, 2: 55000, 3: 55000,
    4: 55000, 5: 55000, 6: 55000, 7: 55000, 8: 55000, 9: 55000,
}

# ──────────────────────────────────────────────
# Equipment weight database
# ──────────────────────────────────────────────

_EQUIPMENT_WEIGHTS: dict[str, int] = {
    # Weapons
    "dagger": 40, "sword": 50, "two_handed_sword": 100,
    "spear": 80, "two_handed_spear": 120,
    "staff": 30, "two_handed_staff": 60,
    "bow": 50, "crossbow": 70,
    "mace": 60, "two_handed_mace": 100,
    "knuckle": 30, "instrument": 40, "whip": 30,
    "katar": 60, "claw": 40,
    "gun": 40, "grenade": 50,
    "huuma_shuriken": 50,
    # Armor
    "armor": 50, "robe": 30, "coat": 40,
    "shield": 40, "buckler": 20,
    "manteau": 20, "cloak": 15,
    "boots": 20, "shoes": 15,
    "helm": 20, "hat": 10, "crown": 15,
    "accessory": 5, "ring": 3, "earring": 3, "necklace": 5,
}

# ──────────────────────────────────────────────
# Set bonuses
# ──────────────────────────────────────────────

@dataclass
class SetBonus:
    """A set bonus effect from wearing multiple items together."""
    name: str
    items: list[str]  # Item names in the set
    bonus_description: str
    atk_bonus: int = 0
    matk_bonus: int = 0
    def_bonus: int = 0
    hp_bonus_pct: float = 0.0
    sp_bonus_pct: float = 0.0
    stat_bonus: dict[str, int] = field(default_factory=dict)
    damage_bonus_pct: float = 0.0
    reduction_pct: float = 0.0

_SET_BONUSES: dict[str, SetBonus] = {
    "orleans_set": SetBonus(
        name="Orleans Set",
        items=["orleans_gown", "orleans_glove"],
        bonus_description="+15% MATK, -15% cast time, +5% SP",
        matk_bonus=15,
        sp_bonus_pct=5.0,
    ),
    "valkyrie_set": SetBonus(
        name="Valkyrie Set",
        items=["valkyrie_armor", "valkyrie_shield", "valkyrie_manteau", "valkyrie_boots"],
        bonus_description="+10% HP, +5% DEF, +5 MDEF",
        hp_bonus_pct=10.0,
        def_bonus=5,
    ),
    "devil_set": SetBonus(
        name="Devil Set",
        items=["devil_helm", "devil_manteau"],
        bonus_description="+5% ATK, +5% ASPD",
        atk_bonus=5,
    ),
    "frozen_set": SetBonus(
        name="Frozen Set",
        items=["frozen_armor", "frozen_shield"],
        bonus_description="+20% resistance to Water, +5 MDEF",
        reduction_pct=20.0,
    ),
    "skeleton_set": SetBonus(
        name="Skeleton Set",
        items=["skeleton_armor", "skeleton_shield"],
        bonus_description="+10% resistance to Undead, +3 DEF",
        reduction_pct=10.0,
        def_bonus=3,
    ),
    "angelic_set": SetBonus(
        name="Angelic Set",
        items=["angelic_helm", "angelic_manteau", "angelic_armor"],
        bonus_description="+10% HP, +5% SP, +3 INT",
        hp_bonus_pct=10.0,
        sp_bonus_pct=5.0,
        stat_bonus={"int": 3},
    ),
    "bloody_set": SetBonus(
        name="Bloody Set",
        items=["bloody_armor", "bloody_shield"],
        bonus_description="+10% ATK, +5% HP",
        atk_bonus=10,
        hp_bonus_pct=5.0,
    ),
}

# ──────────────────────────────────────────────
# Stat-to-equipment mapping
# ──────────────────────────────────────────────

_STAT_EQUIPMENT_MAP: dict[str, list[str]] = {
    "str": ["mantis_card", "kukre_card", "skel_worker_card", "bloody_armor"],
    "agi": ["boned_card", "whisper_card", "sohee_card"],
    "vit": ["pecopeco_card", "verit_card", "valkyrie_armor"],
    "int": ["sohee_card", "zerom_card", "orleans_glove", "angelic_helm"],
    "dex": ["dex_card", "kobold_archer_card", "sniper_goggles"],
    "luk": ["zerom_card", "lunatic_card", "lucky_hat"],
    "crit": ["kobold_archer_card", "critical_ring"],
    "flee": ["whisper_card", "flee_boots", "flee_manteau"],
    "aspd": ["boned_card", "devil_manteau", "berserk_potion"],
}

# ──────────────────────────────────────────────
# Card Database — real RO cards
# ──────────────────────────────────────────────

@dataclass
class CardInfo:
    name: str
    item_id: int
    slot_type: str
    effect: str
    damage_bonus_pct: float = 0.0
    damage_bonus_race: str = ""
    damage_bonus_element: str = ""
    damage_bonus_size: str = ""
    atk_bonus: int = 0
    matk_bonus: int = 0
    def_bonus: int = 0
    hp_bonus_pct: float = 0.0
    sp_bonus_pct: float = 0.0
    reduction_pct: float = 0.0
    reduction_race: str = ""
    reduction_element: str = ""
    stat_bonus: dict[str, int] = field(default_factory=dict)
    market_price: int = 50000

    def calc_damage_multiplier(self, monster_race: str = "",
                               monster_element: str = "",
                               monster_size: str = "") -> float:
        mult = 1.0
        if self.damage_bonus_pct > 0:
            applicable = True
            if self.damage_bonus_race and self.damage_bonus_race != monster_race:
                applicable = False
            if self.damage_bonus_element and self.damage_bonus_element != monster_element:
                applicable = False
            if self.damage_bonus_size and self.damage_bonus_size != monster_size:
                applicable = False
            if applicable:
                mult *= (1.0 + self.damage_bonus_pct / 100.0)
        return mult


_CARD_DB: dict[str, CardInfo] = {
    # ── Weapon Cards ──
    "vadon_card": CardInfo(
        name="Vadon Card", item_id=4021, slot_type="weapon",
        effect="+20% damage to Water element monsters",
        damage_bonus_pct=20.0, damage_bonus_element="water",
        market_price=50000,
    ),
    "drainliar_card": CardInfo(
        name="Drainliar Card", item_id=4030, slot_type="weapon",
        effect="+20% damage to Earth element monsters",
        damage_bonus_pct=20.0, damage_bonus_element="earth",
        market_price=45000,
    ),
    "hydra_card": CardInfo(
        name="Hydra Card", item_id=4024, slot_type="weapon",
        effect="+20% damage to DemiHuman race",
        damage_bonus_pct=20.0, damage_bonus_race="demi_human",
        market_price=80000,
    ),
    "skeleton_worker_card": CardInfo(
        name="Skeleton Worker Card", item_id=4034, slot_type="weapon",
        effect="+5 ATK, +5% damage to Medium size",
        atk_bonus=5, damage_bonus_pct=5.0, damage_bonus_size="medium",
        market_price=35000,
    ),
    "mandragora_card": CardInfo(
        name="Mandragora Card", item_id=4055, slot_type="weapon",
        effect="+20% damage to Plant race",
        damage_bonus_pct=20.0, damage_bonus_race="plant",
        market_price=40000,
    ),
    "pecopeco_card_weapon": CardInfo(
        name="Peco Peco Card", item_id=4113, slot_type="weapon",
        effect="+15% damage to Formless race",
        damage_bonus_pct=15.0, damage_bonus_race="formless",
        market_price=30000,
    ),
    "dragon_tail_card": CardInfo(
        name="Dragon Tail Card", item_id=4214, slot_type="weapon",
        effect="+15% damage to Dragon race",
        damage_bonus_pct=15.0, damage_bonus_race="dragon",
        market_price=60000,
    ),
    "mino_card": CardInfo(
        name="Minorous Card", item_id=4047, slot_type="weapon",
        effect="+15% damage to Large size, +15% damage to Boss monsters",
        damage_bonus_pct=15.0, damage_bonus_size="large",
        market_price=500000,
    ),
    "abysmal_knight_card": CardInfo(
        name="Abysmal Knight Card", item_id=4138, slot_type="weapon",
        effect="+25% damage to Boss monsters",
        damage_bonus_pct=25.0, damage_bonus_race="boss",
        market_price=800000,
    ),
    "skel_bone_card": CardInfo(
        name="Skeleton Bone Card", item_id=4033, slot_type="weapon",
        effect="+10% damage to Undead element",
        damage_bonus_pct=10.0, damage_bonus_element="undead",
        market_price=25000,
    ),
    "kobold_archer_card": CardInfo(
        name="Kobold Archer Card", item_id=4120, slot_type="weapon",
        effect="CRI +9",
        stat_bonus={"crit": 9},
        market_price=60000,
    ),
    # ── Armor Cards ──
    "pecopeco_card": CardInfo(
        name="Peco Peco Card", item_id=4113, slot_type="armor",
        effect="Max HP +10%",
        hp_bonus_pct=10.0,
        market_price=30000,
    ),
    "raydric_card": CardInfo(
        name="Raydric Card", item_id=4167, slot_type="armor",
        effect="Reduce damage from Neutral element by 20%",
        reduction_pct=20.0, reduction_element="neutral",
        market_price=120000,
    ),
    "pasana_card": CardInfo(
        name="Pasana Card", item_id=4163, slot_type="armor",
        effect="Reduce damage from Fire element by 30%",
        reduction_pct=30.0, reduction_element="fire",
        market_price=20000,
    ),
    "phen_card": CardInfo(
        name="Phen Card", item_id=4129, slot_type="armor",
        effect="Prevent cast interruption (long cast only)",
        market_price=80000,
    ),
    "whisper_card": CardInfo(
        name="Whisper Card", item_id=4049, slot_type="armor",
        effect="FLEE +20",
        market_price=15000,
    ),
    # ── Garment Cards ──
    "raydric_garment_card": CardInfo(
        name="Raydric Card", item_id=4167, slot_type="garment",
        effect="Reduce damage from Neutral by 20%",
        reduction_pct=20.0, reduction_element="neutral",
        market_price=120000,
    ),
    "deviling_card": CardInfo(
        name="Deviling Card", item_id=4107, slot_type="garment",
        effect="Reduce all element damage by 30%, but increase Neutral damage taken by 50%",
        reduction_pct=30.0,
        market_price=200000,
    ),
    # ── Shield Cards ──
    "thara_frog_card": CardInfo(
        name="Thara Frog Card", item_id=4175, slot_type="shield",
        effect="Reduce damage from DemiHuman by 30%",
        reduction_pct=30.0, reduction_race="demi_human",
        market_price=50000,
    ),
    "hodremlin_card": CardInfo(
        name="Hodremlin Card", item_id=4207, slot_type="shield",
        effect="Reduce damage from Water element by 30%",
        reduction_pct=30.0, reduction_element="water",
        market_price=15000,
    ),
    # ── Footgear Cards ──
    "sohee_card": CardInfo(
        name="Sohee Card", item_id=4199, slot_type="shoes",
        effect="SP +15%",
        sp_bonus_pct=15.0,
        market_price=60000,
    ),
    "boned_card": CardInfo(
        name="Boned Card", item_id=4298, slot_type="shoes",
        effect="AGI +2",
        stat_bonus={"agi": 2},
        market_price=20000,
    ),
    "verit_card": CardInfo(
        name="Verit Card", item_id=4044, slot_type="shoes",
        effect="Max HP +10%, Max SP +10%",
        hp_bonus_pct=10.0, sp_bonus_pct=10.0,
        market_price=20000,
    ),
    # ── Accessory Cards ──
    "zerom_card": CardInfo(
        name="Zerom Card", item_id=4029, slot_type="accessory",
        effect="LUK +2",
        stat_bonus={"luk": 2},
        market_price=20000,
    ),
    "kukre_card": CardInfo(
        name="Kukre Card", item_id=4061, slot_type="accessory",
        effect="STR +1",
        stat_bonus={"str": 1},
        market_price=10000,
    ),
    "mantis_card": CardInfo(
        name="Mantis Card", item_id=4060, slot_type="accessory",
        effect="STR +2",
        stat_bonus={"str": 2},
        market_price=30000,
    ),
    "lunatic_card": CardInfo(
        name="Lunatic Card", item_id=4007, slot_type="accessory",
        effect="LUK +1",
        stat_bonus={"luk": 1},
        market_price=8000,
    ),
    "dex_card": CardInfo(
        name="Deleter Card", item_id=4135, slot_type="accessory",
        effect="DEX +2",
        stat_bonus={"dex": 2},
        market_price=40000,
    ),
}

# ──────────────────────────────────────────────
# Elemental Converters & Endow Skills
# ──────────────────────────────────────────────

@dataclass
class ElementalConverter:
    name: str
    element: str
    item_id: int = 0
    is_skill: bool = False
    skill_id: str = ""
    duration_s: int = 1800
    market_price: int = 0

    def multiplier_against(self, target_element: str) -> float:
        return _ELEMENT_TABLE_L1.get(self.element, {}).get(target_element, 1.0)


_ELEMENTAL_CONVERTERS: dict[str, ElementalConverter] = {
    "elemental_converter_fire": ElementalConverter(
        name="Fire Converter", element="fire", item_id=12713, market_price=5000,
    ),
    "endow_skill_fire": ElementalConverter(
        name="Endow Flame (Skill)", element="fire",
        is_skill=True, skill_id="PR_ENDOW_FLAME",
    ),
    "elemental_converter_water": ElementalConverter(
        name="Water Converter", element="water", item_id=12714, market_price=5000,
    ),
    "endow_skill_water": ElementalConverter(
        name="Endow Tsunami (Skill)", element="water",
        is_skill=True, skill_id="PR_ENDOW_TSUNAMI",
    ),
    "elemental_converter_wind": ElementalConverter(
        name="Wind Converter", element="wind", item_id=12715, market_price=5000,
    ),
    "endow_skill_wind": ElementalConverter(
        name="Endow Tornado (Skill)", element="wind",
        is_skill=True, skill_id="PR_ENDOW_TORNADO",
    ),
    "elemental_converter_earth": ElementalConverter(
        name="Earth Converter", element="earth", item_id=12716, market_price=5000,
    ),
    "endow_skill_earth": ElementalConverter(
        name="Endow Quake (Skill)", element="earth",
        is_skill=True, skill_id="PR_ENDOW_QUAKE",
    ),
    "holy_water": ElementalConverter(
        name="Holy Water", element="holy", item_id=12622, market_price=2000, duration_s=600,
    ),
    "aspersio_skill": ElementalConverter(
        name="Aspersio (Skill)", element="holy",
        is_skill=True, skill_id="PR_ASPERSIO",
    ),
    "shadow_converter": ElementalConverter(
        name="Shadow Converter", element="shadow", item_id=12764, market_price=8000,
    ),
    "endow_shadow_skill": ElementalConverter(
        name="Enchant Shadow (Skill)", element="shadow",
        is_skill=True, skill_id="NC_ENCHANT_SHADOW",
    ),
    "ghost_converter": ElementalConverter(
        name="Ghost Converter", element="ghost", item_id=12765, market_price=10000,
    ),
    "endow_ghost_skill": ElementalConverter(
        name="Enchant Ghost (Skill)", element="ghost",
        is_skill=True, skill_id="NC_ENCHANT_GHOST",
    ),
}

# ──────────────────────────────────────────────
# Element Table (Level 1 — pre-renewal)
# ──────────────────────────────────────────────

_ELEMENT_TABLE_L1: dict[str, dict[str, float]] = {
    "neutral":  {"neutral": 1.00, "water": 0.75, "earth": 0.75, "fire": 0.75,
                 "wind": 0.75, "poison": 0.75, "holy": 0.75, "shadow": 0.75,
                 "ghost": 0.50, "undead": 0.50},
    "water":    {"neutral": 1.00, "water": 0.25, "earth": 0.75, "fire": 1.25,
                 "wind": 0.50, "poison": 0.75, "holy": 1.00, "shadow": 1.00,
                 "ghost": 0.50, "undead": 1.00},
    "earth":    {"neutral": 1.00, "water": 1.25, "earth": 0.25, "fire": 0.75,
                 "wind": 1.25, "poison": 0.75, "holy": 1.00, "shadow": 1.00,
                 "ghost": 0.50, "undead": 1.00},
    "fire":     {"neutral": 1.00, "water": 0.50, "earth": 1.25, "fire": 0.25,
                 "wind": 0.75, "poison": 0.75, "holy": 1.00, "shadow": 1.00,
                 "ghost": 0.50, "undead": 1.25},
    "wind":     {"neutral": 1.00, "water": 1.25, "earth": 0.50, "fire": 1.25,
                 "wind": 0.25, "poison": 0.75, "holy": 1.00, "shadow": 1.00,
                 "ghost": 0.50, "undead": 1.00},
    "poison":   {"neutral": 1.00, "water": 1.00, "earth": 0.50, "fire": 1.00,
                 "wind": 0.50, "poison": 0.25, "holy": 0.50, "shadow": 1.00,
                 "ghost": 0.50, "undead": 0.50},
    "holy":     {"neutral": 1.00, "water": 1.00, "earth": 1.00, "fire": 1.00,
                 "wind": 1.00, "poison": 1.00, "holy": 0.25, "shadow": 2.00,
                 "ghost": 1.00, "undead": 2.00},
    "shadow":   {"neutral": 1.00, "water": 1.00, "earth": 1.00, "fire": 1.00,
                 "wind": 1.00, "poison": 1.00, "holy": 0.50, "shadow": 0.25,
                 "ghost": 1.00, "undead": 1.00},
    "ghost":    {"neutral": 0.00, "water": 1.00, "earth": 1.00, "fire": 1.00,
                 "wind": 1.00, "poison": 1.00, "holy": 1.00, "shadow": 1.00,
                 "ghost": 0.75, "undead": 1.00},
    "undead":   {"neutral": 1.00, "water": 1.00, "earth": 1.00, "fire": 1.25,
                 "wind": 1.00, "poison": 0.50, "holy": 2.00, "shadow": 1.00,
                 "ghost": 1.00, "undead": 0.25},
}

# ──────────────────────────────────────────────
# Monster Database Snapshot
# ──────────────────────────────────────────────

@dataclass
class MonsterInfo:
    name: str
    element: str
    element_level: int = 1
    race: str = "formless"
    size: str = "medium"
    hp: int = 1000
    level: int = 1


_GRINDING_MONSTERS: dict[str, MonsterInfo] = {
    "poring": MonsterInfo(name="Poring", element="water", race="plant", size="medium", hp=55, level=1),
    "poporing": MonsterInfo(name="Poporing", element="poison", race="plant", size="medium", hp=519, level=16),
    "lunatic": MonsterInfo(name="Lunatic", element="neutral", race="brute", size="small", hp=64, level=3),
    "fabre": MonsterInfo(name="Fabre", element="neutral", race="insect", size="small", hp=75, level=4),
    "pupa": MonsterInfo(name="Pupa", element="neutral", race="insect", size="small", hp=415, level=9),
    "condor": MonsterInfo(name="Condor", element="neutral", race="bird", size="small", hp=125, level=6),
    "wilow": MonsterInfo(name="Wilow", element="water", race="plant", size="medium", hp=166, level=8),
    "chonchon": MonsterInfo(name="Chonchon", element="wind", race="insect", size="small", hp=168, level=10),
    "roda_frog": MonsterInfo(name="Roda Frog", element="water", race="fish", size="medium", hp=193, level=12),
    "spore": MonsterInfo(name="Spore", element="water", race="plant", size="medium", hp=221, level=13),
    "hunter_fly": MonsterInfo(name="Hunter Fly", element="wind", race="insect", size="small", hp=379, level=17),
    "savage": MonsterInfo(name="Savage", element="earth", race="brute", size="large", hp=755, level=22),
    "hode": MonsterInfo(name="Hode", element="water", race="fish", size="medium", hp=552, level=12),
    "argiope": MonsterInfo(name="Argiope", element="poison", race="insect", size="large", hp=423, level=17),
    "orcs_warrior": MonsterInfo(name="Orc Warrior", element="earth", race="demi_human", size="large", hp=867, level=24),
    "orcs_archer": MonsterInfo(name="Orc Archer", element="earth", race="demi_human", size="medium", hp=801, level=25),
    "orcs_lady": MonsterInfo(name="Orc Lady", element="earth", race="demi_human", size="medium", hp=2030, level=33),
    "zenorc": MonsterInfo(name="Zenorc", element="shadow", race="demi_human", size="medium", hp=2343, level=35),
    "high_orc": MonsterInfo(name="High Orc", element="fire", race="demi_human", size="large", hp=5283, level=44),
    "metaller": MonsterInfo(name="Metaller", element="neutral", race="insect", size="small", hp=1430, level=28),
    "drainliar": MonsterInfo(name="Drainliar", element="earth", race="dragon", size="medium", hp=3644, level=40),
    "vadon": MonsterInfo(name="Vadon", element="water", race="dragon", size="medium", hp=3244, level=38),
    "golem": MonsterInfo(name="Golem", element="neutral", race="formless", size="large", hp=3100, level=32),
    "andre": MonsterInfo(name="Andre", element="earth", race="insect", size="medium", hp=237, level=15),
    "soldier_skel": MonsterInfo(name="Soldier Skeleton", element="undead", race="undead", size="medium", hp=758, level=26),
    "skel_archer": MonsterInfo(name="Skeleton Archer", element="undead", race="undead", size="medium", hp=880, level=29),
    "munak": MonsterInfo(name="Munak", element="undead", race="undead", size="medium", hp=903, level=27),
    "bongun": MonsterInfo(name="Bongun", element="undead", race="undead", size="medium", hp=904, level=30),
    "ghost": MonsterInfo(name="Ghostring", element="ghost", race="demon", size="medium", hp=1540, level=32),
    "deviace": MonsterInfo(name="Deviace", element="water", race="fish", size="large", hp=1174, level=28),
    "ferus": MonsterInfo(name="Ferus", element="fire", race="dragon", size="large", hp=6478, level=47),
    "magnolia": MonsterInfo(name="Magnolia", element="neutral", race="demi_human", size="medium", hp=2430, level=35),
    "desert_wolf": MonsterInfo(name="Desert Wolf", element="fire", race="brute", size="medium", hp=1041, level=24),
    "myst_case": MonsterInfo(name="Myst Case", element="neutral", race="formless", size="small", hp=1193, level=28),
    "sand_man": MonsterInfo(name="Sand Man", element="earth", race="formless", size="medium", hp=1305, level=28),
}

# ──────────────────────────────────────────────
# Loadout presets
# ──────────────────────────────────────────────

@dataclass
class LoadoutPreset:
    """A named equipment loadout preset."""
    name: str
    description: str
    priority_stats: list[str]
    recommended_armor_cards: list[str]
    recommended_garment_cards: list[str]
    recommended_shield_cards: list[str]
    recommended_shoe_cards: list[str]
    recommended_accessory_cards: list[str]
    recommended_weapon_cards: list[str]
    min_refine: int = 0
    max_weight: int = 2000

LOADOUT_PRESETS: dict[str, LoadoutPreset] = {
    "farming": LoadoutPreset(
        name="Farming",
        description="Optimized for sustained farming: weight efficiency, ASPD, and damage per weight",
        priority_stats=["str", "dex", "aspd"],
        recommended_armor_cards=["pecopeco_card"],
        recommended_garment_cards=["whisper_card"],
        recommended_shield_cards=["thara_frog_card"],
        recommended_shoe_cards=["boned_card"],
        recommended_accessory_cards=["mantis_card", "mantis_card"],
        recommended_weapon_cards=["vadon_card", "drainliar_card"],
        min_refine=4,
        max_weight=1500,
    ),
    "pvp": LoadoutPreset(
        name="PvP",
        description="Optimized for player vs player: demi-human damage, survivability",
        priority_stats=["vit", "str", "agi"],
        recommended_armor_cards=["raydric_card"],
        recommended_garment_cards=["deviling_card"],
        recommended_shield_cards=["thara_frog_card"],
        recommended_shoe_cards=["verit_card"],
        recommended_accessory_cards=["mantis_card", "dex_card"],
        recommended_weapon_cards=["hydra_card", "hydra_card"],
        min_refine=6,
        max_weight=2500,
    ),
    "mvp": LoadoutPreset(
        name="MVP",
        description="Optimized for MVP/boss hunting: boss damage, high HP",
        priority_stats=["str", "vit", "dex"],
        recommended_armor_cards=["pecopeco_card"],
        recommended_garment_cards=["raydric_garment_card"],
        recommended_shield_cards=["thara_frog_card"],
        recommended_shoe_cards=["verit_card"],
        recommended_accessory_cards=["mantis_card", "dex_card"],
        recommended_weapon_cards=["mino_card", "abysmal_knight_card"],
        min_refine=7,
        max_weight=2500,
    ),
    "tank": LoadoutPreset(
        name="Tank",
        description="Maximum survivability: HP, DEF, MDEF, elemental reduction",
        priority_stats=["vit", "agi", "int"],
        recommended_armor_cards=["pecopeco_card", "raydric_card"],
        recommended_garment_cards=["deviling_card"],
        recommended_shield_cards=["thara_frog_card"],
        recommended_shoe_cards=["verit_card"],
        recommended_accessory_cards=["zerom_card", "zerom_card"],
        recommended_weapon_cards=["skel_bone_card"],
        min_refine=4,
        max_weight=3000,
    ),
    "flee": LoadoutPreset(
        name="Flee",
        description="Maximum flee rate: avoid physical attacks entirely",
        priority_stats=["agi", "luk", "dex"],
        recommended_armor_cards=["whisper_card"],
        recommended_garment_cards=["whisper_card"],
        recommended_shield_cards=[],
        recommended_shoe_cards=["boned_card"],
        recommended_accessory_cards=["zerom_card", "lunatic_card"],
        recommended_weapon_cards=["kobold_archer_card"],
        min_refine=0,
        max_weight=1000,
    ),
}

# ──────────────────────────────────────────────
# Auto-swap conditions
# ──────────────────────────────────────────────

# Element -> recommended weapon element for auto-swap
_AUTO_SWAP_RULES: dict[str, str] = {
    "earth": "fire",       # Earth mobs -> fire weapon
    "fire": "water",       # Fire mobs -> water weapon
    "water": "wind",       # Water mobs -> wind weapon
    "wind": "earth",       # Wind mobs -> earth weapon
    "undead": "holy",      # Undead -> holy weapon
    "shadow": "holy",      # Shadow -> holy weapon
    "ghost": "shadow",     # Ghost -> shadow weapon
    "poison": "holy",      # Poison -> holy weapon
    "neutral": "neutral",  # Neutral -> no swap needed
}

# ──────────────────────────────────────────────
# Data classes for results
# ──────────────────────────────────────────────

@dataclass
class CardSlotResult:
    item_name: str
    card_name: str
    slot_index: int
    total_slots: int
    damage_multiplier: float
    command: str


@dataclass
class UpgradeResult:
    item_name: str
    current_level: int
    target_level: int
    success_chance: float
    atk_bonus: int
    def_bonus: int
    cost_per_try: int
    expected_cost: int
    command: str


@dataclass
class ElementSwapRecommendation:
    target_monster: str
    monster_element: str
    recommended_element: str
    elemental_multiplier: float
    card_multiplier: float
    total_multiplier: float
    converter_name: str
    command: str


@dataclass
class EquipRecommendation:
    monster_name: str
    recommended_weapon: str
    recommended_cards: list[str]
    recommended_converter: str
    expected_damage_multiplier: float
    priority: int = 5


@dataclass
class DurabilityInfo:
    """Equipment durability state."""
    item_name: str
    current_durability: int
    max_durability: int
    broken: bool = False
    last_repair_time: float = 0.0


@dataclass
class WeightOptimizationResult:
    """Weight optimization recommendation."""
    item_name: str
    weight: int
    atk_per_weight: float
    def_per_weight: float
    score: float
    recommendation: str


# ──────────────────────────────────────────────
# Equipment Optimizer
# ──────────────────────────────────────────────

class EquipmentOptimizer:
    """Advanced equipment optimizer with upgrade, card, element, set bonus,
    durability, and weight awareness.

    Features:
      - Stat-to-equipment mapping
      - Set bonuses tracking
      - Card slot analysis
      - Refine-aware scoring
      - Loadout presets (farming, PvP, MVP, tank, flee)
      - Auto-swap conditions
      - Element weapon priority
      - Durability monitoring
      - Weight optimization
    """

    def __init__(self, db: Any = None) -> None:
        self._gk_db = db
        if self._gk_db is None:
            from ai_sidecar.game_knowledge_db import GameKnowledgeDB
            self._gk_db = GameKnowledgeDB()
        self._durability: dict[str, DurabilityInfo] = {}
        self._last_repair_check: float = 0.0

    # ── Stat-to-equipment mapping ─────────────────────────────────

    def get_equipment_for_stat(self, stat_name: str) -> list[str]:
        """Get equipment/cards that provide a specific stat."""
        return _STAT_EQUIPMENT_MAP.get(stat_name.lower(), [])

    def get_stats_from_equipment(self, equipped_items: list[str]) -> dict[str, int]:
        """Calculate total stat bonuses from a list of equipped items."""
        stats: dict[str, int] = {}
        for item in equipped_items:
            item_lower = item.lower().replace(" ", "_")
            for stat_name, items in _STAT_EQUIPMENT_MAP.items():
                if any(eq_item in item_lower for eq_item in items):
                    stats[stat_name] = stats.get(stat_name, 0) + 1
        return stats

    # ── Set bonuses ────────────────────────────────────────────────

    def get_set_bonuses(self, equipped_items: list[str]) -> list[SetBonus]:
        """Get active set bonuses from equipped items."""
        active_bonuses: list[SetBonus] = []
        equipped_lower = [item.lower().replace(" ", "_") for item in equipped_items]

        for set_name, set_bonus in _SET_BONUSES.items():
            set_items_lower = [item.lower().replace(" ", "_") for item in set_bonus.items]
            # Check if all items in the set are equipped
            if all(any(set_item in eq for eq in equipped_lower) for set_item in set_items_lower):
                active_bonuses.append(set_bonus)
                logger.debug("Set bonus active: %s", set_bonus.name)

        return active_bonuses

    def calculate_set_bonus_stats(self, equipped_items: list[str]) -> dict[str, float]:
        """Calculate total stat bonuses from active set effects."""
        total: dict[str, float] = {}
        for bonus in self.get_set_bonuses(equipped_items):
            total["atk"] = total.get("atk", 0) + bonus.atk_bonus
            total["matk"] = total.get("matk", 0) + bonus.matk_bonus
            total["def"] = total.get("def", 0) + bonus.def_bonus
            total["hp_pct"] = total.get("hp_pct", 0) + bonus.hp_bonus_pct
            total["sp_pct"] = total.get("sp_pct", 0) + bonus.sp_bonus_pct
            total["damage_pct"] = total.get("damage_pct", 0) + bonus.damage_bonus_pct
            total["reduction_pct"] = total.get("reduction_pct", 0) + bonus.reduction_pct
            for stat, val in bonus.stat_bonus.items():
                total[stat] = total.get(stat, 0) + val
        return total

    # ── Card Methods ─────────────────────────────────────────────

    def get_card(self, card_name: str) -> CardInfo | None:
        name_lower = card_name.lower().replace(" ", "_")
        if name_lower in _CARD_DB:
            return _CARD_DB[name_lower]
        for key, card in _CARD_DB.items():
            if name_lower in key or key in name_lower:
                return card
            if card.name.lower() in name_lower or name_lower in card.name.lower():
                return card
        return None

    def get_cards_for_slot(self, slot_type: str) -> list[CardInfo]:
        return [c for c in _CARD_DB.values() if c.slot_type == slot_type]

    def calc_card_damage_multiplier(
        self,
        cards: list[str],
        monster_race: str = "",
        monster_element: str = "",
        monster_size: str = "",
    ) -> float:
        mult = 1.0
        for card_name in cards:
            card = self.get_card(card_name)
            if card:
                mult *= card.calc_damage_multiplier(monster_race, monster_element, monster_size)
        return mult

    def recommend_cards_for_monster(
        self,
        monster_name: str,
        monster_race: str = "",
        monster_element: str = "",
        monster_size: str = "",
        available_slots: int = 4,
    ) -> list[CardInfo]:
        monster_key = monster_name.lower().replace(" ", "_").replace("'", "")
        monster = _GRINDING_MONSTERS.get(monster_key)

        race = monster_race or (monster.race if monster else "")
        element = monster_element or (monster.element if monster else "")
        size = monster_size or (monster.size if monster else "")

        scored: list[tuple[CardInfo, float]] = []
        for card in _CARD_DB.values():
            if card.slot_type != "weapon":
                continue
            mult = card.calc_damage_multiplier(race, element, size)
            if mult > 1.0:
                scored.append((card, mult))

        scored.sort(key=lambda x: -x[1])

        if not scored:
            return self.get_cards_for_slot("weapon")[:available_slots]

        return [c[0] for c in scored[:available_slots]]

    def slot_command(self, item_name: str, card_name: str, slot_index: int = 0) -> str:
        return f"slot {item_name} {card_name}"

    def calc_full_weapon_multiplier(
        self,
        weapon_slots: int,
        cards: list[str],
        monster_race: str,
        monster_element: str,
        monster_size: str,
        weapon_element: str = "neutral",
    ) -> dict[str, float]:
        element_mult = _ELEMENT_TABLE_L1.get(weapon_element, {}).get(monster_element, 1.0)
        card_mult = self.calc_card_damage_multiplier(cards, monster_race, monster_element, monster_size)
        total_mult = element_mult * card_mult

        return {
            "element_mult": element_mult,
            "card_mult": card_mult,
            "total_mult": total_mult,
        }

    # ── Upgrade Methods ──────────────────────────────────────

    def get_upgrade_bonus(self, item_type: str, refine_level: int) -> int:
        if item_type == "weapon":
            return _WEAPON_UPGRADE_BONUS.get(refine_level, 0)
        else:
            return _ARMOR_UPGRADE_BONUS.get(refine_level, 0)

    def calc_upgrade_success_chance(self, current_level: int) -> float:
        return _UPGRADE_SUCCESS_RATES.get(current_level, 0.0)

    def calc_upgrade_cost(self, current_level: int, target_level: int) -> int:
        total_cost = 0
        level = current_level
        while level < target_level and level < 10:
            success_rate = _UPGRADE_SUCCESS_RATES.get(level, 0.0)
            cost_per_try = _UPGRADE_COST_PER_TRY.get(level, 55000)
            if success_rate <= 0:
                break
            expected_attempts = 1.0 / success_rate
            total_cost += int(expected_attempts * cost_per_try)
            level += 1
        return total_cost

    def upgrade_recommendation(
        self,
        item_name: str,
        item_type: str,
        current_level: int,
        zeny: int,
        target_level: int = 10,
    ) -> UpgradeResult | None:
        if current_level >= target_level:
            return None

        success_chance = self.calc_upgrade_success_chance(current_level)
        atk_bonus = self.get_upgrade_bonus("weapon", current_level + 1) if item_type == "weapon" else 0
        def_bonus = self.get_upgrade_bonus("armor", current_level + 1) if item_type != "weapon" else 0
        cost_per_try = _UPGRADE_COST_PER_TRY.get(current_level, 55000)
        expected_cost = self.calc_upgrade_cost(current_level, target_level)

        return UpgradeResult(
            item_name=item_name,
            current_level=current_level,
            target_level=target_level,
            success_chance=success_chance,
            atk_bonus=atk_bonus,
            def_bonus=def_bonus,
            cost_per_try=cost_per_try,
            expected_cost=expected_cost,
            command=f"upgrade {item_name}",
        )

    def is_upgrade_worthwhile(self, current_level: int, zeny: int) -> bool:
        if current_level >= 10:
            return False
        if current_level >= 7:
            return zeny > 500000
        if current_level >= 4:
            return zeny > 100000
        return True

    # ── Element Methods ──────────────────────────────────────

    def get_element_multiplier(self, attack_element: str, defense_element: str) -> float:
        return _ELEMENT_TABLE_L1.get(attack_element, {}).get(defense_element, 1.0)

    def find_best_converter(
        self,
        target_element: str,
        available_converters: list[str] | None = None,
        available_skills: list[str] | None = None,
    ) -> ElementalConverter | None:
        best: ElementalConverter | None = None
        best_mult = 1.0

        for conv in _ELEMENTAL_CONVERTERS.values():
            if conv.is_skill:
                if available_skills and conv.skill_id.upper() not in [s.upper() for s in available_skills]:
                    continue
            else:
                if available_converters and conv.name.lower() not in [c.lower() for c in available_converters]:
                    continue

            mult = conv.multiplier_against(target_element)
            if mult > best_mult:
                best_mult = mult
                best = conv

        return best

    def get_converter_command(self, converter: ElementalConverter, target: str = "") -> str:
        if converter.is_skill:
            return f"use_skill {converter.skill_id} {target}"
        else:
            return f"use_item {converter.name}"

    # ── Auto-swap conditions ─────────────────────────────────

    def get_auto_swap_element(self, monster_element: str) -> str | None:
        """Get the recommended weapon element for auto-swap against a monster element."""
        return _AUTO_SWAP_RULES.get(monster_element.lower())

    def should_auto_swap(self, current_element: str, monster_element: str) -> bool:
        """Check if we should auto-swap weapon element."""
        recommended = self.get_auto_swap_element(monster_element)
        if not recommended:
            return False
        return current_element.lower() != recommended

    # ── Loadout presets ──────────────────────────────────────

    def get_loadout_preset(self, preset_name: str) -> LoadoutPreset | None:
        """Get a named loadout preset."""
        return LOADOUT_PRESETS.get(preset_name.lower())

    def get_all_presets(self) -> dict[str, LoadoutPreset]:
        """Get all available loadout presets."""
        return dict(LOADOUT_PRESETS)

    def recommend_loadout(
        self,
        preset_name: str,
        current_equipment: list[str] = None,
    ) -> dict[str, Any]:
        """Get a full loadout recommendation for a preset."""
        preset = self.get_loadout_preset(preset_name)
        if not preset:
            return {"error": f"Unknown preset: {preset_name}"}

        return {
            "preset": preset.name,
            "description": preset.description,
            "priority_stats": preset.priority_stats,
            "recommended_armor_cards": preset.recommended_armor_cards,
            "recommended_garment_cards": preset.recommended_garment_cards,
            "recommended_shield_cards": preset.recommended_shield_cards,
            "recommended_shoe_cards": preset.recommended_shoe_cards,
            "recommended_accessory_cards": preset.recommended_accessory_cards,
            "recommended_weapon_cards": preset.recommended_weapon_cards,
            "min_refine": preset.min_refine,
            "max_weight": preset.max_weight,
        }

    # ── Durability monitoring ───────────────────────────────

    def update_durability(self, item_name: str, current: int, max_durability: int) -> None:
        """Update durability tracking for an item."""
        if item_name not in self._durability:
            self._durability[item_name] = DurabilityInfo(
                item_name=item_name,
                current_durability=current,
                max_durability=max_durability,
            )
        else:
            info = self._durability[item_name]
            info.current_durability = current
            info.max_durability = max_durability
            info.broken = current <= 0

    def check_durability(self, item_name: str) -> DurabilityInfo | None:
        """Get durability info for an item."""
        return self._durability.get(item_name)

    def get_items_needing_repair(self, threshold: float = 0.3) -> list[DurabilityInfo]:
        """Get items with durability below threshold."""
        now = time.time()
        needing_repair: list[DurabilityInfo] = []
        for info in self._durability.values():
            if info.broken:
                needing_repair.append(info)
            elif info.max_durability > 0:
                pct = info.current_durability / info.max_durability
                if pct < threshold and (now - info.last_repair_time) > 60:
                    needing_repair.append(info)
        return needing_repair

    def auto_repair_recommendation(self, threshold: float = 0.3) -> list[str]:
        """Get auto-repair commands for items with low durability."""
        commands: list[str] = []
        for info in self.get_items_needing_repair(threshold):
            commands.append(f"repair {info.item_name}")
            info.last_repair_time = time.time()
        return commands

    # ── Weight optimization ─────────────────────────────────

    def get_item_weight(self, item_type: str) -> int:
        """Get the weight of an item type."""
        return _EQUIPMENT_WEIGHTS.get(item_type.lower().replace(" ", "_"), 50)

    def calculate_weight_score(
        self,
        atk: int,
        def_val: int,
        weight: int,
    ) -> float:
        """Calculate ATK/DEF per weight unit score."""
        if weight <= 0:
            return float(atk + def_val)
        return (atk + def_val) / weight

    def optimize_for_weight(
        self,
        equipment_options: list[dict[str, Any]],
        max_weight: int,
    ) -> list[WeightOptimizationResult]:
        """Optimize equipment selection for weight efficiency.

        Args:
            equipment_options: List of dicts with 'name', 'atk', 'def', 'weight', 'type'
            max_weight: Maximum total weight allowed

        Returns:
            Sorted list of weight optimization results.
        """
        results: list[WeightOptimizationResult] = []
        for item in equipment_options:
            name = item.get("name", "unknown")
            atk = item.get("atk", 0)
            def_val = item.get("def", 0)
            weight = item.get("weight", self.get_item_weight(item.get("type", "armor")))
            score = self.calculate_weight_score(atk, def_val, weight)

            atk_per_weight = atk / max(1, weight)
            def_per_weight = def_val / max(1, weight)

            if weight <= max_weight:
                recommendation = "Recommended"
            else:
                recommendation = f"Over weight limit by {weight - max_weight}"

            results.append(WeightOptimizationResult(
                item_name=name,
                weight=weight,
                atk_per_weight=round(atk_per_weight, 2),
                def_per_weight=round(def_per_weight, 2),
                score=round(score, 2),
                recommendation=recommendation,
            ))

        results.sort(key=lambda r: -r.score)
        return results

    # ── Combined Recommendations ─────────────────────────────

    def recommend_equipment_for_monster(
        self,
        monster_name: str,
        weapon_slots: int = 4,
        weapon_level: int = 0,
        available_converters: list[str] | None = None,
        available_skills: list[str] | None = None,
    ) -> EquipRecommendation | None:
        monster_key = monster_name.lower().replace(" ", "_").replace("'", "")
        monster = _GRINDING_MONSTERS.get(monster_key)

        if not monster:
            return None

        # Best element
        best_element = "neutral"
        best_mult = 1.0
        for elem in _ELEMENT_TABLE_L1:
            mult = _ELEMENT_TABLE_L1[elem].get(monster.element, 1.0)
            if mult > best_mult:
                best_mult = mult
                best_element = elem

        # Best converter for that element
        converter = self.find_best_converter(
            monster.element,
            available_converters,
            available_skills,
        )
        converter_name = converter.name if converter else "None (neutral)"

        # Best cards
        recommended_cards = self.recommend_cards_for_monster(
            monster_name,
            monster.race,
            monster.element,
            monster.size,
            weapon_slots,
        )
        card_names = [c.name for c in recommended_cards]

        # Full multiplier
        card_mult = self.calc_card_damage_multiplier(
            card_names, monster.race, monster.element, monster.size
        )
        total_mult = best_mult * card_mult

        # Upgrade bonus
        if weapon_level > 0:
            atk_bonus_val = _WEAPON_UPGRADE_BONUS.get(weapon_level, 0)
            upgrade_factor = 1.0 + (atk_bonus_val / 100.0)
            total_mult *= upgrade_factor

        priority = 8 if total_mult > 2.0 else (6 if total_mult > 1.5 else 4)

        return EquipRecommendation(
            monster_name=monster.name,
            recommended_weapon=f"weapon({best_element})",
            recommended_cards=card_names,
            recommended_converter=converter_name,
            expected_damage_multiplier=round(total_mult, 2),
            priority=priority,
        )

    def recommend_equipment_for_map(
        self,
        map_name: str,
        weapon_slots: int = 4,
        weapon_level: int = 0,
        available_converters: list[str] | None = None,
        available_skills: list[str] | None = None,
    ) -> list[EquipRecommendation]:
        from ai_sidecar.autonomy.ro_mechanics import get_optimal_element_for_map

        target_element = get_optimal_element_for_map(map_name)
        if target_element:
            monster = MonsterInfo(
                name=f"{map_name}_monsters",
                element=target_element,
            )
            return [self.recommend_equipment_for_monster(
                monster.name, weapon_slots, weapon_level,
                available_converters, available_skills,
            )]  # type: ignore[list-item]

        return []

    # ── Utility Commands ─────────────────────────────────────

    def equip_command(self, item_id_or_name: str) -> str:
        return f"equip {item_id_or_name}"

    def upgrade_command(self, item_name: str) -> str:
        return f"upgrade {item_name}"

    def slot_card_command(self, item_name: str, card_name: str) -> str:
        return f"slot {item_name} {card_name}"

    def use_converter_command(self, converter_name: str) -> str:
        return f"use_item {converter_name}"

    def use_endow_command(self, skill_id: str, target: str = "") -> str:
        return f"use_skill {skill_id} {target}"

    # ── Batch Assess ─────────────────────────────────────────

    def assess(self, signals: dict[str, Any], bot_id: str = "") -> list[dict]:
        """Assess equipment state and return optimization recommendations."""
        actions: list[dict] = []

        inventory = signals.get("inventory", []) or []
        equipped = signals.get("equipment", []) or signals.get("inventory_equipped", []) or []
        current_monster = signals.get("current_monster", {}) or {}
        job_name = str(signals.get("job_name", "novice") or "novice").lower()
        base_level = int(signals.get("base_level", 1) or 1)
        zeny = int(signals.get("zeny", 0) or 0)
        current_map = str(signals.get("map", "") or "")

        # Current monster recommendation
        if current_monster:
            monster_name = ""
            if isinstance(current_monster, dict):
                monster_name = current_monster.get("name", "")
            elif isinstance(current_monster, str):
                monster_name = current_monster

            if monster_name:
                rec = self.recommend_equipment_for_monster(
                    monster_name,
                    available_converters=[i.get("name", "") for i in inventory],
                    available_skills=signals.get("available_skills", []),
                )
                if rec and rec.expected_damage_multiplier > 1.5:
                    actions.append({
                        "type": "optimize_equipment",
                        "priority": rec.priority,
                        "reason": (
                            f"vs {rec.monster_name}: {rec.recommended_converter} "
                            f"({rec.expected_damage_multiplier}x), "
                            f"cards: {', '.join(rec.recommended_cards[:2])}"
                        ),
                        "monster": rec.monster_name,
                        "converter": rec.recommended_converter,
                        "cards": rec.recommended_cards,
                        "expected_mult": rec.expected_damage_multiplier,
                    })

        # Auto-swap check
        if current_monster and isinstance(current_monster, dict):
            monster_element = current_monster.get("element", "")
            if monster_element:
                current_weapon_element = signals.get("weapon_element", "neutral")
                if self.should_auto_swap(current_weapon_element, monster_element):
                    recommended = self.get_auto_swap_element(monster_element)
                    if recommended:
                        actions.append({
                            "type": "auto_swap_weapon",
                            "priority": 7,
                            "reason": f"Auto-swap: {monster_element} mob -> {recommended} weapon",
                            "current_element": current_weapon_element,
                            "recommended_element": recommended,
                        })

        # Upgrade recommendations
        for item in equipped:
            eq_name = str(item.get("name", "") or "")
            refine = int(item.get("refine", item.get("refine_level", 0)) or 0)
            if eq_name and self.is_upgrade_worthwhile(refine, zeny):
                actions.append({
                    "type": "upgrade_equipment",
                    "priority": 6,
                    "reason": f"Upgrade {eq_name} from +{refine} to +{refine + 1}",
                    "item": eq_name,
                    "current_refine": refine,
                    "cost_estimate": _UPGRADE_COST_PER_TRY.get(refine, 55000),
                    "command": f"upgrade {eq_name}",
                })

        # Card recommendations for empty weapon slots
        equipped_weapon = None
        for item in equipped:
            slot = str(item.get("slot", "") or "").lower().replace(" ", "_")
            if slot == "weapon":
                equipped_weapon = item
                break

        if equipped_weapon:
            equip_name = str(equipped_weapon.get("name", "") or "")
            current_cards = equipped_weapon.get("cards", []) or []
            slot_count = 0
            if "[" in equip_name and "]" in equip_name:
                try:
                    slot_str = equip_name.split("[")[1].split("]")[0]
                    slot_count = int(slot_str)
                except (ValueError, IndexError):
                    pass

            empty_slots = max(0, slot_count - len(current_cards))
            if empty_slots > 0 and current_monster:
                monster_name = ""
                if isinstance(current_monster, dict):
                    monster_name = current_monster.get("name", "")
                cards = self.recommend_cards_for_monster(
                    monster_name, available_slots=empty_slots,
                ) or []
                for card in cards[:empty_slots]:
                    actions.append({
                        "type": "slot_card",
                        "priority": 5,
                        "reason": f"Insert {card.name} into {equip_name} ({card.effect})",
                        "item": equip_name,
                        "card": card.name,
                        "command": f"slot {equip_name} {card.name}",
                    })

        # Durability check
        now = time.time()
        if now - self._last_repair_check > 60:
            self._last_repair_check = now
            for item in equipped:
                eq_name = str(item.get("name", "") or "")
                durability = int(item.get("durability", item.get("dur", 100)) or 100)
                max_dur = int(item.get("max_durability", item.get("max_dur", 100)) or 100)
                self.update_durability(eq_name, durability, max_dur)

            repair_commands = self.auto_repair_recommendation()
            for cmd in repair_commands:
                actions.append({
                    "type": "repair_equipment",
                    "priority": 8,
                    "reason": f"Auto-repair: {cmd}",
                    "command": cmd,
                })

        # Set bonus check
        equipped_names = [str(item.get("name", "")) for item in equipped if item.get("name")]
        active_sets = self.get_set_bonuses(equipped_names)
        if active_sets:
            for bonus in active_sets:
                actions.append({
                    "type": "set_bonus_active",
                    "priority": 3,
                    "reason": f"Set bonus active: {bonus.name} — {bonus.bonus_description}",
                    "set_name": bonus.name,
                })

        # Loadout recommendation based on current activity
        current_activity = signals.get("activity", signals.get("mode", ""))
        if current_activity:
            preset_map = {
                "farming": "farming",
                "pvp": "pvp",
                "mvp": "mvp",
                "tank": "tank",
                "flee": "flee",
            }
            preset_name = preset_map.get(current_activity.lower())
            if preset_name:
                actions.append({
                    "type": "loadout_recommendation",
                    "priority": 4,
                    "reason": f"Loadout suggestion: {preset_name} mode",
                    "preset": preset_name,
                })

        return actions
