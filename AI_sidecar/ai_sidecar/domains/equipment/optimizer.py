"""Equipment optimizer — upgrade levels, cards, elements, and slot-aware calculations.

Real RO data: weapon upgrade effects, card database, element multipliers,
elemental converters, and equipment swap recommendations.
Commands: equip, slot, upgrade.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Weapon Upgrade Levels (+0 to +10)
# ──────────────────────────────────────────────
# Each upgrade level adds ATK and a small chance of breaking on failure.
# Level 1-9 upgrades use Enriched or normal Elunium/Oridecon.
# Level 10 is the maximum safe upgrade (safe to +4, risky +5+).
_WEAPON_UPGRADE_BONUS: dict[int, int] = {
    0: 0,    # Base
    1: 3,    # +1: +3 ATK
    2: 6,    # +2: +6 ATK
    3: 9,    # +3: +9 ATK
    4: 12,   # +4: +12 ATK
    5: 17,   # +5: +17 ATK
    6: 22,   # +6: +22 ATK
    7: 27,   # +7: +27 ATK
    8: 32,   # +8: +32 ATK
    9: 37,   # +9: +37 ATK
    10: 42,  # +10: +42 ATK
}

_ARMOR_UPGRADE_BONUS: dict[int, int] = {
    0: 0,   # Base
    1: 1,   # +1: +1 DEF
    2: 2,   # +2: +2 DEF
    3: 3,   # +3: +3 DEF
    4: 4,   # +4: +4 DEF
    5: 6,   # +5: +6 DEF
    6: 8,   # +6: +8 DEF
    7: 10,  # +7: +10 DEF
    8: 12,  # +8: +12 DEF
    9: 14,  # +9: +14 DEF
    10: 16, # +10: +16 DEF
}

# Upgrade success probabilities (pre-renewal formula)
# Safe: +0->+1 to +4->+5 with 100% on +0->+4, then decreasing
_UPGRADE_SUCCESS_RATES: dict[int, float] = {
    0: 1.00,   # +0 -> +1: 100%
    1: 1.00,   # +1 -> +2: 100%
    2: 1.00,   # +2 -> +3: 100%
    3: 1.00,   # +3 -> +4: 100%
    4: 0.60,   # +4 -> +5: 60%
    5: 0.40,   # +5 -> +6: 40%
    6: 0.40,   # +6 -> +7: 40%
    7: 0.20,   # +7 -> +8: 20%
    8: 0.20,   # +8 -> +9: 20%
    9: 0.08,   # +9 -> +10: 8%
}

# Upgrade costs per attempt (Oridecon cost for weapons)
_UPGRADE_COST_PER_TRY: dict[int, int] = {
    0: 55000,   # Oridecon cost for +0->+1
    1: 55000,   # +1->+2
    2: 55000,   # +2->+3
    3: 55000,   # +3->+4
    4: 55000,   # +4->+5
    5: 55000,   # +5->+6
    6: 55000,   # +6->+7
    7: 55000,   # +7->+8
    8: 55000,   # +8->+9
    9: 55000,   # +9->+10
}

# ──────────────────────────────────────────────
# Card Database — real RO cards
# ──────────────────────────────────────────────
@dataclass
class CardInfo:
    """Information about a card and its effects."""
    name: str
    item_id: int
    slot_type: str  # 'weapon', 'armor', 'garment', 'shield', 'shoes', 'accessory', 'head'
    effect: str
    damage_bonus_pct: float = 0.0       # +% damage to specific race/element/size
    damage_bonus_race: str = ""          # race this bonus applies to
    damage_bonus_element: str = ""       # element this bonus applies to
    damage_bonus_size: str = ""          # size this bonus applies to
    atk_bonus: int = 0                   # flat ATK bonus
    matk_bonus: int = 0                  # flat MATK bonus
    def_bonus: int = 0                   # flat DEF bonus
    hp_bonus_pct: float = 0.0            # +% HP
    sp_bonus_pct: float = 0.0            # +% SP
    reduction_pct: float = 0.0           # -% damage from specific element/race
    reduction_race: str = ""             # race reduced
    reduction_element: str = ""          # element reduced
    stat_bonus: dict[str, int] = field(default_factory=dict)  # e.g. {"str": 2}
    market_price: int = 50000

    def calc_damage_multiplier(self, monster_race: str = "",
                               monster_element: str = "",
                               monster_size: str = "") -> float:
        """Calculate the damage multiplier this card provides against a specific monster."""
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
        # Reduction doesn't multiply damage, it reduces incoming damage
        return mult


# Real RO Card Database
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
    """An item or skill that changes weapon element."""
    name: str
    element: str
    item_id: int = 0
    is_skill: bool = False
    skill_id: str = ""
    duration_s: int = 1800  # 30 min default
    market_price: int = 0

    def multiplier_against(self, target_element: str) -> float:
        """Get damage multiplier when using this element against a target."""
        return _ELEMENT_TABLE_L1.get(self.element, {}).get(target_element, 1.0)


_ELEMENTAL_CONVERTERS: dict[str, ElementalConverter] = {
    # ── Fire ──
    "elemental_converter_fire": ElementalConverter(
        name="Fire Converter", element="fire", item_id=12713,
        market_price=5000,
    ),
    "endow_skill_fire": ElementalConverter(
        name="Endow Flame (Skill)", element="fire",
        is_skill=True, skill_id="PR_ENDOW_FLAME",
    ),

    # ── Water ──
    "elemental_converter_water": ElementalConverter(
        name="Water Converter", element="water", item_id=12714,
        market_price=5000,
    ),
    "endow_skill_water": ElementalConverter(
        name="Endow Tsunami (Skill)", element="water",
        is_skill=True, skill_id="PR_ENDOW_TSUNAMI",
    ),

    # ── Wind ──
    "elemental_converter_wind": ElementalConverter(
        name="Wind Converter", element="wind", item_id=12715,
        market_price=5000,
    ),
    "endow_skill_wind": ElementalConverter(
        name="Endow Tornado (Skill)", element="wind",
        is_skill=True, skill_id="PR_ENDOW_TORNADO",
    ),

    # ── Earth ──
    "elemental_converter_earth": ElementalConverter(
        name="Earth Converter", element="earth", item_id=12716,
        market_price=5000,
    ),
    "endow_skill_earth": ElementalConverter(
        name="Endow Quake (Skill)", element="earth",
        is_skill=True, skill_id="PR_ENDOW_QUAKE",
    ),

    # ── Holy ──
    "holy_water": ElementalConverter(
        name="Holy Water", element="holy", item_id=12622,
        market_price=2000,
        duration_s=600,  # 10 min
    ),
    "aspersio_skill": ElementalConverter(
        name="Aspersio (Skill)", element="holy",
        is_skill=True, skill_id="PR_ASPERSIO",
    ),

    # ── Shadow ──
    "shadow_converter": ElementalConverter(
        name="Shadow Converter", element="shadow", item_id=12764,
        market_price=8000,
    ),
    "endow_shadow_skill": ElementalConverter(
        name="Enchant Shadow (Skill)", element="shadow",
        is_skill=True, skill_id="NC_ENCHANT_SHADOW",
    ),

    # ── Ghost ──
    "ghost_converter": ElementalConverter(
        name="Ghost Converter", element="ghost", item_id=12765,
        market_price=10000,
    ),
    "endow_ghost_skill": ElementalConverter(
        name="Enchant Ghost (Skill)", element="ghost",
        is_skill=True, skill_id="NC_ENCHANT_GHOST",
    ),
}

# ──────────────────────────────────────────────
# Element Table (Level 1 — pre-renewal)
# ──────────────────────────────────────────────
# Attack element rows × Target element columns
# Order: neutral, water, earth, fire, wind, poison, holy, shadow, ghost, undead
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
# Monster Database Snapshot (for recommendations)
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

# Key grinding monsters with real RO data
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
# Equipment Optimizer
# ──────────────────────────────────────────────

@dataclass
class CardSlotResult:
    """Result of slotting a card into an item."""
    item_name: str
    card_name: str
    slot_index: int
    total_slots: int
    damage_multiplier: float
    command: str

@dataclass
class UpgradeResult:
    """Result of a weapon/armor upgrade attempt."""
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
    """Recommendation to swap weapon element."""
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
    """Full equipment recommendation for a target."""
    monster_name: str
    recommended_weapon: str
    recommended_cards: list[str]
    recommended_converter: str
    expected_damage_multiplier: float
    priority: int = 5


class EquipmentOptimizer:
    """Advanced equipment optimizer with upgrade, card, and element awareness.

    Features:
        - Weapon/armor upgrade tracking (+0 to +10) with damage calculation
        - Card system: know which cards go in which slot
        - Slot-aware: 2-slot weapon + 2 cards = multiplicative bonus
        - Elemental advantage: converters, endow skills
        - Swap recommendations based on target monster
        - Commands: equip, slot, upgrade
    """

    def __init__(self, db: Any = None) -> None:
        self._gk_db = db
        if self._gk_db is None:
            from ai_sidecar.game_knowledge_db import GameKnowledgeDB
            self._gk_db = GameKnowledgeDB()

    # ── Card Methods ─────────────────────────────────────────

    def get_card(self, card_name: str) -> CardInfo | None:
        """Look up a card by name (partial match supported)."""
        name_lower = card_name.lower().replace(" ", "_")
        # Exact match first
        if name_lower in _CARD_DB:
            return _CARD_DB[name_lower]
        # Partial match
        for key, card in _CARD_DB.items():
            if name_lower in key or key in name_lower:
                return card
            if card.name.lower() in name_lower or name_lower in card.name.lower():
                return card
        return None

    def get_cards_for_slot(self, slot_type: str) -> list[CardInfo]:
        """Get all cards that can go in a particular slot."""
        return [c for c in _CARD_DB.values() if c.slot_type == slot_type]

    def calc_card_damage_multiplier(
        self,
        cards: list[str],
        monster_race: str = "",
        monster_element: str = "",
        monster_size: str = "",
    ) -> float:
        """Calculate total damage multiplier from multiple cards (multiplicative).

        Args:
            cards: List of card names in the weapon.
            monster_race: Target monster race.
            monster_element: Target monster element.
            monster_size: Target monster size.

        Returns:
            Total multiplier (e.g. 1.44 for 2x Vadon vs Water).
        """
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
        """Recommend the best cards to put in a weapon against a specific monster.

        Uses real card data and monster properties.
        """
        # Try to look up monster info
        monster_key = monster_name.lower().replace(" ", "_").replace("'", "")
        monster = _GRINDING_MONSTERS.get(monster_key)

        race = monster_race or (monster.race if monster else "")
        element = monster_element or (monster.element if monster else "")
        size = monster_size or (monster.size if monster else "")

        # Score each weapon card
        scored: list[tuple[CardInfo, float]] = []
        for card in _CARD_DB.values():
            if card.slot_type != "weapon":
                continue
            mult = card.calc_damage_multiplier(race, element, size)
            if mult > 1.0:
                scored.append((card, mult))

        # Sort by multiplier (highest first)
        scored.sort(key=lambda x: -x[1])

        if not scored:
            # Fallback: return generic recommendations
            return self.get_cards_for_slot("weapon")[:available_slots]

        # Return top N cards
        return [c[0] for c in scored[:available_slots]]

    def slot_command(self, item_name: str, card_name: str, slot_index: int = 0) -> str:
        """Generate a command to insert a card into an item."""
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
        """Calculate the total damage multiplier from element + cards + upgrades.

        Args:
            weapon_slots: Number of slots in the weapon (1, 2, 3, 4).
            cards: Card names inserted.
            monster_race: Target race.
            monster_element: Target element.
            monster_size: Target size.
            weapon_element: Current weapon element (from converter/endow).

        Returns:
            Dict with breakdown: element_mult, card_mult, total_mult.
        """
        # Element multiplier
        element_mult = _ELEMENT_TABLE_L1.get(weapon_element, {}).get(monster_element, 1.0)

        # Card multiplier (multiplicative with element)
        card_mult = self.calc_card_damage_multiplier(cards, monster_race, monster_element, monster_size)

        total_mult = element_mult * card_mult

        return {
            "element_mult": element_mult,
            "card_mult": card_mult,
            "total_mult": total_mult,
        }

    # ── Upgrade Methods ──────────────────────────────────────

    def get_upgrade_bonus(self, item_type: str, refine_level: int) -> int:
        """Get the stat bonus from a given upgrade level."""
        if item_type == "weapon":
            return _WEAPON_UPGRADE_BONUS.get(refine_level, 0)
        else:
            return _ARMOR_UPGRADE_BONUS.get(refine_level, 0)

    def calc_upgrade_success_chance(self, current_level: int) -> float:
        """Get the success probability for upgrading from current_level."""
        return _UPGRADE_SUCCESS_RATES.get(current_level, 0.0)

    def calc_upgrade_cost(self, current_level: int, target_level: int) -> int:
        """Calculate expected cost to go from current_level to target_level.

        Accounts for failure probability (item breaks on failure at >= +5 in pre-renewal).
        Simple model: each attempt costs Oridecon + fee, success rate per level.
        """
        total_cost = 0
        level = current_level
        while level < target_level and level < 10:
            success_rate = _UPGRADE_SUCCESS_RATES.get(level, 0.0)
            cost_per_try = _UPGRADE_COST_PER_TRY.get(level, 55000)
            if success_rate <= 0:
                break
            # Expected attempts = 1/success_rate (geometric distribution)
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
        """Recommend whether and how to upgrade an item."""
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
        """Decide if upgrading further makes economic sense."""
        if current_level >= 10:
            return False
        if current_level >= 7:
            return zeny > 500000  # Only rich players should push past +7
        if current_level >= 4:
            return zeny > 100000
        return True  # +0 to +4 is always worth it (100% success)

    # ── Element Methods ──────────────────────────────────────

    def get_element_multiplier(self, attack_element: str, defense_element: str) -> float:
        """Get the damage multiplier for attack element vs defense element."""
        return _ELEMENT_TABLE_L1.get(attack_element, {}).get(defense_element, 1.0)

    def find_best_converter(
        self,
        target_element: str,
        available_converters: list[str] | None = None,
        available_skills: list[str] | None = None,
    ) -> ElementalConverter | None:
        """Find the best converter/endow to use against a target element."""
        best: ElementalConverter | None = None
        best_mult = 1.0

        for conv in _ELEMENTAL_CONVERTERS.values():
            # Check if we have this converter
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
        """Generate the command to use a converter or cast an endow skill."""
        if converter.is_skill:
            return f"use_skill {converter.skill_id} {target}"
        else:
            return f"use_item {converter.name}"

    # ── Combined Recommendations ─────────────────────────────

    def recommend_equipment_for_monster(
        self,
        monster_name: str,
        weapon_slots: int = 4,
        weapon_level: int = 0,
        available_converters: list[str] | None = None,
        available_skills: list[str] | None = None,
    ) -> EquipRecommendation | None:
        """Full equipment recommendation against a specific monster.

        Combines element advantage, card recommendations, and upgrade advice.
        """
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
            upgrade_factor = 1.0 + (atk_bonus_val / 100.0)  # rough estimate
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
        """Get recommendations for the most dangerous/common monsters on a map.

        Falls back to element-based recommendations using the map's dominant
        monster element known in the knowledge DB.
        """
        from ai_sidecar.autonomy.ro_mechanics import get_optimal_element_for_map

        target_element = get_optimal_element_for_map(map_name)
        if target_element:
            # Mock a monster with this element
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
        """Generate a command to equip an item."""
        return f"equip {item_id_or_name}"

    def upgrade_command(self, item_name: str) -> str:
        """Generate a command to attempt upgrading an item."""
        return f"upgrade {item_name}"

    def slot_card_command(self, item_name: str, card_name: str) -> str:
        """Generate a command to slot a card into an item."""
        return f"slot {item_name} {card_name}"

    def use_converter_command(self, converter_name: str) -> str:
        """Generate a command to use an elemental converter."""
        return f"use_item {converter_name}"

    def use_endow_command(self, skill_id: str, target: str = "") -> str:
        """Generate a command to cast an endow weapon skill."""
        return f"use_skill {skill_id} {target}"

    # ── Batch Assess ─────────────────────────────────────────

    def assess(self, signals: dict[str, Any], bot_id: str = "") -> list[dict]:
        """Assess equipment state and return optimization recommendations.

        Integrates with the existing signal-based assessment pipeline.
        """
        actions: list[dict] = []

        inventory = signals.get("inventory", []) or []
        equipped = signals.get("equipment", []) or signals.get("inventory_equipped", []) or []
        current_monster = signals.get("current_monster", {}) or {}
        job_name = str(signals.get("job_name", "novice") or "novice").lower()
        base_level = int(signals.get("base_level", 1) or 1)
        zeny = int(signals.get("zeny", 0) or 0)

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
            # Extract slot count from name like "Blade[3]"
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
                )
                for card in cards[:empty_slots]:
                    actions.append({
                        "type": "slot_card",
                        "priority": 5,
                        "reason": f"Insert {card.name} into {equip_name} ({card.effect})",
                        "item": equip_name,
                        "card": card.name,
                        "command": f"slot {equip_name} {card.name}",
                    })

        return actions
