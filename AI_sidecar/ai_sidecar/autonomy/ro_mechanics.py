"""
RO Mechanics Engine v5 — Adaptive server-mode (classic / pre-renewal / renewal).
Auto-detected from monster stats every assess cycle; all formulas correct.
"""

import json
import math
import os
from enum import Enum
from pathlib import Path
from threading import Lock

# ── Server Mode ─────────────────────────────────────────────────────────────

class ServerMode(str, Enum):
    CLASSIC = "classic"           # Pre-2004: simple ATK-DEF, no element table
    PRE_RENEWAL = "pre_renewal"   # 2004-2009: DEF * 0.8%, size penalties
    RENEWAL = "renewal"           # 2010+: 1 - DEF/(DEF+400), mod element table
    UNKNOWN = "unknown"

_server_mode = ServerMode.RENEWAL
_server_mode_lock = Lock()

def get_server_mode() -> ServerMode:
    global _server_mode
    with _server_mode_lock:
        return _server_mode

def set_server_mode(mode: str | ServerMode) -> None:
    global _server_mode
    if isinstance(mode, str):
        mode = ServerMode(mode)
    with _server_mode_lock:
        _server_mode = mode


# ── Monster DB (optional JSON from rAthena export) ──────────────────────────

_MOB_DB_PATH = Path(os.environ.get(
    "RATHENA_MOB_DB",
    str(Path(__file__).resolve().parent.parent.parent.parent / "data" / "rAthena" / "mob_db.json")
))

FULL_MONSTER_DB: dict = {}
if _MOB_DB_PATH.exists():
    try:
        with open(_MOB_DB_PATH) as f:
            FULL_MONSTER_DB = json.load(f)
    except (json.JSONDecodeError, OSError):
        pass


# ── Formula Helpers ─────────────────────────────────────────────────────────

def _auto_detect_server_mode(
    max_monster_hp: int = 0,
    has_high_level_mobs: bool = False,
) -> ServerMode:
    """Auto-detect server mode from monster stats."""
    if max_monster_hp > 500000:
        return ServerMode.RENEWAL
    if max_monster_hp < 20000 and has_high_level_mobs:
        return ServerMode.CLASSIC
    return get_server_mode()  # Keep current


def calculate_damage(
    attack_power: int,
    defense: int = 0,
    defense2: int = 0,
    weapon_type: str = "sword",
    target_size: str = "Medium",
    attack_element: str = "Neutral",
    target_element: str = "Neutral",
    element_level: int = 1,
    refine_bonus: int = 0,
    card_mod: float = 1.0,
    size_penalty: float = 1.0,
    server_mode: ServerMode | None = None,
) -> int:
    """Damage formula correct for ANY server mode.

    CLASSIC:      ATK * element - DEF               (pre-2003)
    PRE_RENEWAL:  ATK * (100 - DEF*0.8)/100 * elem  (2004-2009)
    RENEWAL:      ATK * elem * (1 - DEF/(DEF+400))  (2010+)
    """
    mode = server_mode or _server_mode
    total_atk = max(1, attack_power + refine_bonus)

    # Size penalty applies in pre-renewal and renewal
    if mode != ServerMode.CLASSIC:
        size_pen = SIZE_PENALTY.get(weapon_type, {}).get(target_size, 1.0)
        total_atk = int(total_atk * size_pen)

    # Element modifier (only in pre-renewal and renewal)
    elem_mod = 1.0
    if mode != ServerMode.CLASSIC:
        elem_table = ELEMENT_TABLE.get(element_level, ELEMENT_TABLE[1])
        # Normalize case: 'fire' -> 'Fire', 'water' -> 'Water', etc.
        atk_elem = attack_element.capitalize() if attack_element else 'Neutral'
        tgt_elem = target_element.capitalize() if target_element else 'Neutral'
        elem_mod = elem_table.get(atk_elem, {}).get(tgt_elem, 1.0)

    if mode == ServerMode.CLASSIC:
        # Classic: simple subtraction
        total_atk = max(1, total_atk - defense - defense2)
    elif mode == ServerMode.PRE_RENEWAL:
        # Pre-renewal: DEF * 0.8% reduction, hard cap at 99 DEF
        effective_def = min(defense, 99)
        def_reduction = max(0, 100 - effective_def * 0.8) / 100
        total_atk = int(total_atk * def_reduction)
        total_atk = max(1, total_atk - defense2)
    else:
        # Renewal: 1 - DEF/(DEF+400), then subtract soft DEF
        if defense > 0:
            def_reduction = 1.0 - defense / (defense + 400)
            total_atk = int(total_atk * def_reduction)
        total_atk = max(1, total_atk - defense2)

    total_atk = int(total_atk * elem_mod)
    total_atk = int(total_atk * card_mod)

    return max(1, total_atk)


def calculate_aspd(
    agi: int = 1,
    dex: int = 1,
    base_aspd: int | None = None,
    skill_bonus: float = 0.0,
    item_aspd: int = 0,
    weapon_type: str = "sword",
    server_mode: ServerMode | None = None,
) -> float:
    """ASPD formula for any server mode.

    CLASSIC:       200 - ASPDBASE - AGI*0.5
    PRE_RENEWAL:   200 - (ASPDBASE*10 - sqrt(AGI+DEX*0.5)) / (200 - skill%) * 200
    RENEWAL:       200 - (200 - weapon - sqrt(agi²/2+dex²/4) - item) * (200 - skill)/200
    """
    mode = server_mode or _server_mode
    if base_aspd is None:
        base_aspd = WEAPON_BASE_ASPD.get(weapon_type, 1500)
    weapon_asdp = base_aspd / 10

    if mode == ServerMode.CLASSIC:
        raw = 200 - weapon_asdp - agi * 0.5 - item_aspd
        return min(193.0, max(100.0, round(raw, 1)))

    stat_reduction = math.sqrt(agi * agi / 2 + dex * dex / 4)
    total_bonus = item_aspd
    skill_mod = 200 - int(skill_bonus * 200)
    skill_mod = max(0, min(200, skill_mod))

    if mode == ServerMode.PRE_RENEWAL:
        # Pre-renewal: ASPD = 200 - (200 - weapon_aspd - sqrt(agi*0.5 + dex*0.5)/4 - item) * (200 - skill_mod)/200
        stat_reduction_pre = math.sqrt(agi * 0.5 + dex * 0.5) / 4
        base_result = 200 - (200 - weapon_asdp - stat_reduction_pre - total_bonus)
    else:
        # Renewal: ASPD = 200 - (200 - weapon_aspd - sqrt(agi²/2 + dex²/4) - item) * (200 - skill_mod)/200
        base_result = 200 - (200 - weapon_asdp - stat_reduction - total_bonus)

    final_aspd = 200 - (200 - base_result) * skill_mod / 200
    return min(193.0, max(100.0, round(final_aspd, 1)))


def calculate_flee(
    base_level: int,
    agi: int,
    item_bonus: int = 0,
    server_mode: ServerMode | None = None,
) -> int:
    """FLEE formula for any server mode."""
    mode = server_mode or _server_mode
    if mode == ServerMode.CLASSIC:
        return 100 + base_level + agi + item_bonus
    return 100 + base_level + agi + item_bonus  # Same for pre-renewal and renewal


def calculate_hit(
    base_level: int,
    dex: int,
    item_bonus: int = 0,
    server_mode: ServerMode | None = None,
) -> int:
    """HIT formula for any server mode."""
    mode = server_mode or _server_mode
    if mode == ServerMode.CLASSIC:
        return 100 + base_level + dex + item_bonus
    return 175 + base_level + dex + item_bonus


def calculate_monster_hit_rate(monster_base_level: int, monster_dex: int) -> int:
    return 95 + monster_base_level + monster_dex


def calculate_monster_flee(monster_level: int, monster_agi: int) -> int:
    return monster_level + monster_agi


def estimate_hits_to_die(
    monster_attack: int, player_hp: int, player_def: int = 0,
    server_mode: ServerMode | None = None,
) -> float:
    if monster_attack <= 0:
        return 999.0
    dmg_per_hit = calculate_damage(monster_attack, player_def, server_mode=server_mode)
    if dmg_per_hit <= 0:
        return 999.0
    return player_hp / dmg_per_hit


def calculate_hit_rate(
    player_hit: int,
    monster_flee: int,
) -> float:
    """Calculate hit probability based on your HIT vs monster FLEE.
    
    RO formula: 
      hit_chance = 100 - (monster_flee - player_hit) * 1
      Minimum 5%, Maximum 95%
    
    If your HIT >= monster FLEE, you have 95% hit rate (95% is cap).
    If monster FLEE is much higher, you have 5% minimum.
    """
    if player_hit >= monster_flee:
        return 0.95  # 95% cap
    raw = 0.95 - (monster_flee - player_hit) * 0.01
    return max(0.05, min(0.95, raw))


def calculate_renewal_defense(
    hard_def: int,
    soft_def: int = 0,
    use_simplified: bool = True,
) -> float:
    """Calculate Renewal defense reduction.
    
    Simplified (most servers): def / (def + 400)
    Actual Renewal:
      Hard DEF: (400 + defense) / (400 + defense * 10)
      Soft DEF: (400 + soft_def) / (400 + soft_def * 10)
    
    Args:
        hard_def: Equipment DEF value
        soft_def: VIT-based soft DEF
        use_simplified: If True, use simplified formula (most common)
    
    Returns:
        Defense multiplier (0.0 = immune, 1.0 = no reduction)
    """
    if hard_def <= 0:
        return 1.0
    
    if use_simplified:
        # Simplified: def / (def + 400)
        return 1.0 - hard_def / (hard_def + 400)
    else:
        # Actual Renewal hard DEF
        hard_mult = (400 + hard_def) / (400 + hard_def * 10)
        # Soft DEF
        soft_mult = (400 + soft_def) / (400 + soft_def * 10) if soft_def > 0 else 1.0
        return hard_mult * soft_mult


# ── MVP Element Shift ──
MVP_ELEMENT_SHIFT: dict[str, dict] = {
    "edga": {"element": "Fire", "element_level": 2, "hp_threshold": 0.25},
    "eddga": {"element": "Fire", "element_level": 2, "hp_threshold": 0.25},
    "drake": {"element": "Undead", "element_level": 2, "hp_threshold": 0.25},
    "garm": {"element": "Water", "element_level": 3, "hp_threshold": 0.25},
    "ifrit": {"element": "Fire", "element_level": 4, "hp_threshold": 0.25},
    "thanatos": {"element": "Ghost", "element_level": 3, "hp_threshold": 0.25},
    "beelzebub": {"element": "Dark", "element_level": 3, "hp_threshold": 0.25},
    "detardeuras": {"element": "Holy", "element_level": 3, "hp_threshold": 0.25},
    "gloom under night": {"element": "Dark", "element_level": 3, "hp_threshold": 0.25},
    "kiel": {"element": "Ghost", "element_level": 2, "hp_threshold": 0.25},
}


def get_mvp_low_hp_element(monster_name: str, current_hp_pct: float) -> tuple[str, int] | None:
    """Get the element shift for an MVP at low HP.
    
    Returns (element_name, element_level) if the MVP shifts element,
    or None if no shift occurs.
    
    Example: Eddga shifts to Fire Lv2 when HP < 25%.
    """
    info = MVP_ELEMENT_SHIFT.get(monster_name.lower())
    if info and current_hp_pct <= info["hp_threshold"]:
        return (info["element"], info["element_level"])
    return None


def get_stat_atk(str_stat: int) -> int:
    return str_stat + (str_stat // 10) ** 2


def get_stat_matk(int_stat: int) -> int:
    return int_stat + (int_stat // 10) ** 2 + (int_stat // 5)


def get_stat_max_hp(base_level: int, vit: int) -> int:
    return int((35 + vit * 100) * (1 + base_level / 100))


# ── Element table (correct for all modes) ──
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

# ── Size penalty table (same for pre-renewal & renewal) ──
SIZE_PENALTY = {
    "dagger":        {"Small": 1.00, "Medium": 0.75, "Large": 0.50},
    "sword":         {"Small": 0.75, "Medium": 1.00, "Large": 0.75},
    "two_hand_sword": {"Small": 0.75, "Medium": 0.75, "Large": 1.00},
    "spear":         {"Small": 0.75, "Medium": 0.75, "Large": 1.00},
    "bow":           {"Small": 1.00, "Medium": 1.00, "Large": 1.00},
    "mace":          {"Small": 0.75, "Medium": 1.00, "Large": 0.75},
    "staff":         {"Small": 1.00, "Medium": 1.00, "Large": 0.75},
    "knuckle":       {"Small": 1.00, "Medium": 0.75, "Large": 0.50},
    "instrument":    {"Small": 0.75, "Medium": 1.00, "Large": 0.75},
    "whip":          {"Small": 0.75, "Medium": 1.00, "Large": 0.75},
    "book":          {"Small": 0.75, "Medium": 1.00, "Large": 0.75},
    "katar":         {"Small": 1.00, "Medium": 1.00, "Large": 0.75},
    "grenade":       {"Small": 1.00, "Medium": 1.00, "Large": 1.00},
    "shuriken":      {"Small": 1.00, "Medium": 1.00, "Large": 1.00},
}

WEAPON_BASE_ASPD = {
    "dagger": 1400, "sword": 1500, "two_hand_sword": 1400, "spear": 1400,
    "bow": 1500, "mace": 1500, "staff": 1500, "knuckle": 1400,
    "instrument": 1500, "whip": 1500, "book": 1500, "katar": 1400,
    "grenade": 1400, "shuriken": 1400,
}

JOB_WEAPON_TYPE = {
    "novice": "dagger", "swordman": "sword", "mage": "staff", "archer": "bow",
    "acolyte": "mace", "merchant": "sword", "thief": "dagger", "taekwon": "knuckle",
    "gunslinger": "grenade", "ninja": "shuriken", "soul_linker": "staff",
    "knight": "spear", "wizard": "staff", "hunter": "bow", "priest": "mace",
    "blacksmith": "sword", "assassin": "katar", "crusader": "spear",
    "monk": "knuckle", "sage": "staff", "rogue": "dagger", "alchemist": "sword",
    "bard": "instrument", "dancer": "whip",
}

# ── Potion values (range-based) ──
POTION_COST = 500
POTION_HEAL_MIN = 235
POTION_HEAL_MAX = 355
BLUE_POTION_COST = 1000
BLUE_POTION_SP_MIN = 72
BLUE_POTION_SP_MAX = 144
ARROW_COST = 2

RED_POTION_COST = 35
RED_POTION_HEAL_MIN = 45
RED_POTION_HEAL_MAX = 65
ORANGE_POTION_COST = 200
ORANGE_POTION_HEAL_MIN = 105
ORANGE_POTION_HEAL_MAX = 145

# ── Skill damage formulas ──
SKILL_DAMAGE = {
    "NV_BASIC": {"base": 1.0, "per_level": 0.0, "sp": 0, "cast": 0.0, "delay": 0.0, "element": "Neutral", "element_level_fn": lambda lv: 1},
    "NV_FIRSTAID": {"base": 1.0, "per_level": 0.5, "sp": 5, "cast": 0.5, "delay": 0.5, "element": "Neutral", "element_level_fn": lambda lv: 1},
    "SM_BASH": {"base": 1.5, "per_level": 0.3, "sp": 8, "cast": 0.0, "delay": 1.0, "element": "Neutral", "element_level_fn": lambda lv: 1, "stun_chance": lambda lv: lv * 3},
    "SM_MAGNUM": {"base": 0.7, "per_level": 0.6, "sp": 12, "cast": 0.0, "delay": 1.5, "element": "Neutral", "element_level_fn": lambda lv: 1, "aoe": True},
    "SM_PROVOKE": {"base": 0.0, "per_level": 0.0, "sp": 5, "cast": 0.0, "delay": 0.5, "element": "Neutral", "element_level_fn": lambda lv: 1, "debuff": True},
    "SM_RECOVERY": {"base": 0.0, "per_level": 0.0, "sp": 0, "cast": 0.0, "delay": 0.0, "element": "Neutral", "element_level_fn": lambda lv: 1, "passive": True},
    "MG_FIREBOLT": {"base": 1.0, "per_level": 0.4, "sp": 12, "cast": 1.5, "delay": 1.0, "element": "Fire", "element_level_fn": lambda lv: 1 if lv <= 4 else (2 if lv <= 9 else 3)},
    "MG_COLDBOLT": {"base": 1.0, "per_level": 0.4, "sp": 12, "cast": 1.5, "delay": 1.0, "element": "Water", "element_level_fn": lambda lv: 1 if lv <= 4 else (2 if lv <= 9 else 3)},
    "MG_LIGHTNINGBOLT": {"base": 1.0, "per_level": 0.4, "sp": 12, "cast": 1.5, "delay": 1.0, "element": "Wind", "element_level_fn": lambda lv: 1 if lv <= 4 else (2 if lv <= 9 else 3)},
    "MG_FIREWALL": {"base": 0.5, "per_level": 0.6, "sp": 15, "cast": 0.5, "delay": 2.0, "element": "Fire", "element_level_fn": lambda lv: 1 if lv <= 4 else (2 if lv <= 9 else 3), "aoe": True},
    "MG_FROSTDIVER": {"base": 1.0, "per_level": 0.4, "sp": 15, "cast": 1.0, "delay": 1.0, "element": "Water", "element_level_fn": lambda lv: 1 if lv <= 4 else (2 if lv <= 9 else 3), "freeze_chance": lambda lv: lv * 5},
    "MG_THUNDERSTORM": {"base": 0.4, "per_level": 0.4, "sp": 20, "cast": 2.0, "delay": 2.0, "element": "Wind", "element_level_fn": lambda lv: 1 if lv <= 4 else (2 if lv <= 9 else 3), "aoe": True},
    "MG_SRECOVERY": {"base": 0.0, "per_level": 0.0, "sp": 0, "cast": 0.0, "delay": 0.0, "element": "Neutral", "element_level_fn": lambda lv: 1, "passive": True},
    "MG_ENERGYCOAT": {"base": 0.0, "per_level": 0.0, "sp": 10, "cast": 0.0, "delay": 0.0, "element": "Neutral", "element_level_fn": lambda lv: 1, "buff": True},
    "AC_DOUBLE": {"base": 2.0, "per_level": 0.2, "sp": 12, "cast": 0.0, "delay": 0.5, "element": "Neutral", "element_level_fn": lambda lv: 1},
    "AC_OWL": {"base": 0.0, "per_level": 0.0, "sp": 0, "cast": 0.0, "delay": 0.0, "element": "Neutral", "element_level_fn": lambda lv: 1, "passive": True},
    "AC_SHOWER": {"base": 1.0, "per_level": 0.2, "sp": 15, "cast": 1.0, "delay": 1.0, "element": "Neutral", "element_level_fn": lambda lv: 1, "aoe": True},
    "AL_HEAL": {"base": 1.0, "per_level": 0.4, "sp": 15, "cast": 1.0, "delay": 1.0, "element": "Holy", "element_level_fn": lambda lv: 1, "heal": True},
    "AL_DEMONBANE": {"base": 0.0, "per_level": 0.0, "sp": 0, "cast": 0.0, "delay": 0.0, "element": "Neutral", "element_level_fn": lambda lv: 1, "passive": True},
    "AL_DP": {"base": 0.0, "per_level": 0.0, "sp": 10, "cast": 0.0, "delay": 0.0, "element": "Neutral", "element_level_fn": lambda lv: 1, "buff": True},
    "AL_BLESS": {"base": 0.0, "per_level": 0.0, "sp": 8, "cast": 0.5, "delay": 0.5, "element": "Neutral", "element_level_fn": lambda lv: 1, "buff": True},
    "AL_INCAGI": {"base": 0.0, "per_level": 0.0, "sp": 10, "cast": 0.5, "delay": 0.5, "element": "Neutral", "element_level_fn": lambda lv: 1, "buff": True},
    "MC_MAMMON": {"base": 0.0, "per_level": 0.0, "sp": 0, "cast": 0.0, "delay": 0.0, "element": "Neutral", "element_level_fn": lambda lv: 1, "passive": True},
    "MC_DISCOUNT": {"base": 0.0, "per_level": 0.0, "sp": 0, "cast": 0.0, "delay": 0.0, "element": "Neutral", "element_level_fn": lambda lv: 1, "passive": True},
    "MC_OVERCHARGE": {"base": 0.0, "per_level": 0.0, "sp": 0, "cast": 0.0, "delay": 0.0, "element": "Neutral", "element_level_fn": lambda lv: 1, "passive": True},
    "MC_PUSHCART": {"base": 0.0, "per_level": 0.0, "sp": 0, "cast": 0.0, "delay": 0.0, "element": "Neutral", "element_level_fn": lambda lv: 1, "passive": True},
    "MC_VENDING": {"base": 0.0, "per_level": 0.0, "sp": 0, "cast": 0.0, "delay": 0.0, "element": "Neutral", "element_level_fn": lambda lv: 1, "passive": True},
    "MC_IDENTIFY": {"base": 0.0, "per_level": 0.0, "sp": 10, "cast": 0.0, "delay": 0.0, "element": "Neutral", "element_level_fn": lambda lv: 1},
    "TF_DOUBLE": {"base": 1.5, "per_level": 0.1, "sp": 0, "cast": 0.0, "delay": 0.0, "element": "Neutral", "element_level_fn": lambda lv: 1, "passive": True},
    "TF_HIDING": {"base": 0.0, "per_level": 0.0, "sp": 10, "cast": 0.0, "delay": 0.0, "element": "Neutral", "element_level_fn": lambda lv: 1, "buff": True, "stealth": True},
    "TF_POISON": {"base": 0.0, "per_level": 0.0, "sp": 10, "cast": 0.0, "delay": 0.5, "element": "Neutral", "element_level_fn": lambda lv: 1, "debuff": True},
    "TF_STEAL": {"base": 0.0, "per_level": 0.0, "sp": 5, "cast": 0.0, "delay": 0.5, "element": "Neutral", "element_level_fn": lambda lv: 1},
    "TF_DETOXIFY": {"base": 0.0, "per_level": 0.0, "sp": 5, "cast": 0.0, "delay": 0.0, "element": "Neutral", "element_level_fn": lambda lv: 1},
    "KN_BOWLINGBASH": {"base": 1.0, "per_level": 0.6, "sp": 18, "cast": 0.5, "delay": 1.5, "element": "Neutral", "element_level_fn": lambda lv: 1, "aoe": True},
    "CR_SHIELDBOOMERANG": {"base": 1.0, "per_level": 0.5, "sp": 20, "cast": 0.0, "delay": 2.0, "element": "Neutral", "element_level_fn": lambda lv: 1},
    "MO_TRIPLEATTACK": {"base": 1.0, "per_level": 0.4, "sp": 0, "cast": 0.0, "delay": 0.0, "element": "Neutral", "element_level_fn": lambda lv: 1, "passive": True},
    "MO_CHAINCOMBO": {"base": 1.5, "per_level": 0.4, "sp": 10, "cast": 0.0, "delay": 1.0, "element": "Neutral", "element_level_fn": lambda lv: 1},
    "AS_SONICBLOW": {"base": 1.0, "per_level": 0.8, "sp": 25, "cast": 1.0, "delay": 1.0, "element": "Neutral", "element_level_fn": lambda lv: 1},
}

SKILL_SP_COSTS = {
    "NV_BASIC": 0, "NV_FIRSTAID": 5,
    "SM_BASH": 8, "SM_MAGNUM": 12, "SM_PROVOKE": 5, "SM_RECOVERY": 0,
    "MG_SRECOVERY": 0, "MG_FIREBOLT": 12, "MG_COLDBOLT": 12, "MG_LIGHTNINGBOLT": 12,
    "MG_FIREWALL": 15, "MG_FROSTDIVER": 15, "MG_THUNDERSTORM": 20, "MG_ENERGYCOAT": 10,
    "AC_OWL": 0, "AC_DOUBLE": 12, "AC_SHOWER": 15,
    "AL_HEAL": 15, "AL_DEMONBANE": 0, "AL_DP": 10, "AL_BLESS": 8, "AL_INCAGI": 10,
    "MC_VENDING": 0, "MC_DISCOUNT": 0, "MC_OVERCHARGE": 0, "MC_PUSHCART": 0,
    "MC_MAMMON": 0, "MC_IDENTIFY": 10,
    "TF_DOUBLE": 0, "TF_HIDING": 10, "TF_POISON": 10, "TF_STEAL": 5, "TF_DETOXIFY": 5,
    "KN_BOWLINGBASH": 18, "CR_SHIELDBOOMERANG": 20, "MO_TRIPLEATTACK": 0,
    "MO_CHAINCOMBO": 10, "AS_SONICBLOW": 25,
}

FOOD_ITEMS = {
    "531": {"stat": "str", "bonus": 4, "duration": 1800, "cost": 500},
    "532": {"stat": "agi", "bonus": 4, "duration": 1800, "cost": 500},
    "533": {"stat": "vit", "bonus": 4, "duration": 1800, "cost": 500},
    "534": {"stat": "int", "bonus": 4, "duration": 1800, "cost": 500},
    "535": {"stat": "dex", "bonus": 4, "duration": 1800, "cost": 500},
    "536": {"stat": "luk", "bonus": 4, "duration": 1800, "cost": 500},
    "505": {"stat": "aspd", "bonus": 10, "duration": 300, "cost": 200},
}

STAT_BREAKPOINTS = {
    "str": [(10, "+10 ATK"), (20, "+20 ATK"), (30, "+30 ATK"), (40, "+41 ATK"), (50, "+52 ATK"),
            (60, "+63 ATK"), (70, "+74 ATK"), (80, "+86 ATK"), (90, "+99 ATK"), (99, "+108 ATK")],
    "agi": [(10, "+10 Flee"), (20, "+20 Flee"), (30, "+31 Flee"), (40, "+42 Flee"),
            (50, "+53 Flee"), (60, "+64 Flee"), (70, "+75 Flee"), (80, "+86 Flee"), (90, "+97 Flee"), (99, "+108 Flee")],
    "vit": [(10, "+100 HP"), (20, "+200 HP"), (30, "+300 HP"), (40, "+400 HP"), (50, "+500 HP"),
            (60, "+600 HP"), (70, "+700 HP"), (80, "+800 HP"), (90, "+900 HP"), (99, "+1000 HP")],
    "int": [(7, "+7 MATK"), (14, "+15 MATK"), (21, "+24 MATK"), (28, "+34 MATK"), (35, "+45 MATK"),
            (42, "+57 MATK"), (49, "+70 MATK"), (56, "+84 MATK"), (63, "+99 MATK"), (70, "+115 MATK"),
            (77, "+132 MATK"), (84, "+150 MATK"), (91, "+169 MATK"), (98, "+189 MATK"), (99, "+199 MATK")],
    "dex": [(10, "+10 ATK, +10 Hit"), (20, "+20 ATK, +20 Hit"), (30, "+31 ATK, +30 Hit"),
            (40, "+42 ATK, +40 Hit"), (50, "+53 ATK, +50 Hit"), (60, "+64 ATK, +60 Hit"),
            (70, "+75 ATK, +70 Hit"), (80, "+86 ATK, +80 Hit"), (90, "+97 ATK, +90 Hit"), (99, "+108 ATK, +99 Hit")],
    "luk": [(10, "+1 ATK, +1 Crit"), (20, "+2 ATK, +2 Crit"), (30, "+3 ATK, +3 Crit"),
            (40, "+4 ATK, +4 Crit"), (50, "+5 ATK, +5 Crit"), (60, "+6 ATK, +6 Crit"),
            (70, "+7 ATK, +7 Crit"), (80, "+8 ATK, +8 Crit"), (90, "+9 ATK, +9 Crit"), (99, "+10 ATK, +10 Crit")],
}

SCALING_STAT_TARGETS = {
    "novice":    [(10, {"dex": 20, "str": 10, "agi": 10})],
    "swordman":  [(30, {"str": 40, "vit": 30, "dex": 20}), (50, {"str": 60, "vit": 40, "dex": 30}), (70, {"str": 80, "vit": 50, "dex": 40}), (99, {"str": 99, "vit": 60, "dex": 50})],
    "mage":      [(30, {"int": 50, "dex": 20}), (50, {"int": 70, "dex": 30}), (70, {"int": 90, "dex": 40}), (99, {"int": 99, "dex": 50})],
    "archer":    [(30, {"dex": 50, "agi": 30, "luk": 20}), (50, {"dex": 70, "agi": 50, "luk": 30}), (70, {"dex": 90, "agi": 60, "luk": 40}), (99, {"dex": 99, "agi": 80, "luk": 50})],
    "acolyte":   [(30, {"int": 50, "dex": 20, "vit": 10}), (50, {"int": 70, "dex": 30, "vit": 20}), (70, {"int": 90, "dex": 40, "vit": 30}), (99, {"int": 99, "dex": 50, "vit": 40})],
    "merchant":  [(30, {"str": 50, "vit": 30, "dex": 10}), (50, {"str": 70, "vit": 40, "dex": 20}), (70, {"str": 90, "vit": 50, "dex": 30}), (99, {"str": 99, "vit": 60, "dex": 40})],
    "thief":     [(30, {"agi": 50, "dex": 20, "str": 20}), (50, {"agi": 70, "dex": 30, "str": 30}), (70, {"agi": 90, "dex": 40, "str": 40}), (99, {"agi": 99, "dex": 50, "str": 50})],
}

CARD_VALUES = {
    "poring": {"card": 50000, "drops": ["Jellopy(10z)", "Apple(50z)"]},
    "lunatic": {"card": 30000, "drops": ["Lunatic Card(30000z)", "Clover(100z)"]},
    "pupa": {"card": 20000, "drops": ["Pupa Card(20000z)", "Sticky Mucus(50z)"]},
    "familiar": {"card": 25000, "drops": ["Familiar Card(25000z)", "Bat(100z)"]},
    "zombie": {"card": 40000, "drops": ["Zombie Card(40000z)", "Decayed Nail(200z)"]},
    "skeleton": {"card": 35000, "drops": ["Skeleton Card(35000z)", "Bone(150z)"]},
    "orc warrior": {"card": 80000, "drops": ["Orc Warrior Card(80000z)", "Orcish Voucher(500z)"]},
    "poporing": {"card": 60000, "drops": ["Poporing Card(60000z)", "Poison Spore(300z)"]},
    "muk": {"card": 25000, "drops": ["Muk Card(25000z)", "Sticky Web(80z)"]},
    "hunter fly": {"card": 80000, "drops": ["Hunter Fly Card(80000z)", "Insect Wing(100z)"]},
    "drainliar": {"card": 150000, "drops": ["Drainliar Card(150000z)", "Scale Shell(500z)"]},
    "elder willow": {"card": 35000, "drops": ["Elder Willow Card(35000z)", "Bark(100z)"]},
    "cream": {"card": 120000, "drops": ["Creamy Card(120000z)", "Carrion(200z)"]},
    "pecopeco": {"card": 150000, "drops": ["Peco Peco Card(150000z)", "Feather(100z)"]},
    "savage": {"card": 200000, "drops": ["Savage Card(200000z)", "Savage Bellow(1000z)"]},
    "marc": {"card": 250000, "drops": ["Marc Card(250000z)", "Bitter Herb(200z)"]},
    "hydra": {"card": 100000, "drops": ["Hydra Card(100000z)", "Tentacle(200z)"]},
    "petite": {"card": 150000, "drops": ["Petite Card(150000z)", "Dragon Scale(800z)"]},
    "munak": {"card": 60000, "drops": ["Munak Card(60000z)", "Old Piece of Kimono(200z)"]},
    "sohee": {"card": 300000, "drops": ["Sohee Card(300000z)", "Crystal Fragment(500z)"]},
    "archer skeleton": {"card": 100000, "drops": ["Archer Skeleton Card(100000z)", "Bone Fragment(150z)"]},
    "eclipse": {"card": 350000, "drops": ["Eclipse Card(350000z)", "Moon Stone(2000z)"]},
    "deviruchi": {"card": 500000, "drops": ["Deviruchi Card(500000z)", "Wing(200z)"]},
    "wild rose": {"card": 120000, "drops": ["Wild Rose Card(120000z)", "Rose(300z)"]},
}

MVP_MONSTERS = {
    "baphomet": {"id": 1848, "map": "iz_dun04", "respawn_s": 7200, "drops": ["Baphomet Card(500000z)", "Horn of Baphomet(100000z)"]},
    "orc hero": {"id": 1850, "map": "orcsdun02", "respawn_s": 3000, "drops": ["Orc Hero Card(400000z)", "Hero's Token(80000z)"]},
    "moonlight flower": {"id": 1150, "map": "um_dun01", "respawn_s": 3600, "drops": ["Moonlight Flower Card(300000z)", "Flower(50000z)"]},
    "osiris": {"id": 1043, "map": "moc_pryd04", "respawn_s": 3600, "drops": ["Osiris Card(500000z)", "Mummy Bandage(100000z)"]},
    "edga": {"id": 1112, "map": "moc_pryd05", "respawn_s": 3600, "drops": ["Edga Card(300000z)", "Edga's Ring(80000z)"]},
    "doppelganger": {"id": 1046, "map": "gef_dun02", "respawn_s": 3600, "drops": ["Doppelganger Card(500000z)", "Doppelganger's Soul(100000z)"]},
    "phreeoni": {"id": 1101, "map": "mi_dun01", "respawn_s": 3600, "drops": ["Phreeoni Card(400000z)", "Phreeoni's Eye(80000z)"]},
    "garm": {"id": 1259, "map": "xmas_dun02", "respawn_s": 3600, "drops": ["Garm Card(400000z)", "Garm's Tooth(80000z)"]},
    "mistress": {"id": 1059, "map": "moc_fild12", "respawn_s": 3600, "drops": ["Mistress Card(500000z)", "Mistress's Hair(100000z)"]},
    "drake": {"id": 1072, "map": "treasure02", "respawn_s": 3600, "drops": ["Drake Card(400000z)", "Drake's Scale(80000z)"]},
    "atros": {"id": 1107, "map": "um_boss", "respawn_s": 7200, "drops": ["Atroce Card(500000z)", "Atroce's Tooth(100000z)"]},
    "kiel": {"id": 1291, "map": "kiel_dun01", "respawn_s": 7200, "drops": ["Kiel Card(500000z)", "Kiel's Orb(100000z)"]},
    "kades": {"id": 1292, "map": "abyss_03", "respawn_s": 7200, "drops": ["Kades Card(500000z)", "Kades' Necklace(100000z)"]},
    "turtle general": {"id": 1312, "map": "tur_dun04", "respawn_s": 7200, "drops": ["Turtle General Card(500000z)", "Turtle's Shell(100000z)"]},
    "gloom under night": {"id": 1871, "map": "ra_fild01", "respawn_s": 7200, "drops": ["Gloom Card(500000z)", "Gloom's Shard(100000z)"]},
    "detardeuras": {"id": 3301, "map": "boss_rash", "respawn_s": 7200, "drops": ["Detardeuras Card(500000z)", "Deta's Scale(100000z)"]},
    "ifrit": {"id": 3452, "map": "mocboss", "respawn_s": 7200, "drops": ["Ifrit Card(500000z)", "Ifrit's Horn(100000z)"]},
    "thanatos": {"id": 3309, "map": "than_d", "respawn_s": 7200, "drops": ["Thanatos Card(500000z)", "Thanatos' Fragment(100000z)"]},
    "beelzebub": {"id": 3305, "map": "beach_dun", "respawn_s": 7200, "drops": ["Beelzebub Card(500000z)", "Bee's Stinger(100000z)"]},
}

ELEMENTAL_WEAPONS = {
    "fire": {"dagger": "Fire Knife", "sword": "Fire Sword", "bow": "Fire Bow", "spear": "Flame Spear", "mace": "Fire Mace"},
    "water": {"dagger": "Water Knife", "sword": "Water Sword", "bow": "Water Bow", "spear": "Aqua Spear", "mace": "Water Mace"},
    "wind": {"dagger": "Wind Knife", "sword": "Wind Sword", "bow": "Wind Bow", "spear": "Storm Spear", "mace": "Wind Mace"},
    "earth": {"dagger": "Earth Knife", "sword": "Earth Sword", "bow": "Earth Bow", "spear": "Gaia Spear", "mace": "Earth Mace"},
}

JOB_CHANGE_TALK = {
    "archer": ["talk @npc@ (160, 191)", "talk continue", "talk resp 1", "talk resp 2", "talk resp 1"],
    "thief":  ["talk @npc@ (231, 38)", "talk continue", "talk resp 1", "talk resp 2", "talk resp 1"],
    "acolyte":["talk @npc@ (200, 170)", "talk continue", "talk resp 1", "talk resp 2", "talk resp 1"],
    "mage":   ["talk @npc@ (180, 150)", "talk continue", "talk resp 1", "talk resp 2", "talk resp 1"],
    "swordman":["talk @npc@ (140, 120)", "talk continue", "talk resp 1", "talk resp 2", "talk resp 1"],
    "merchant":["talk @npc@ (120, 200)", "talk continue", "talk resp 1", "talk resp 2", "talk resp 1"],
}
JOB_CHANGE_2_1 = {
    "archer":    ("prontera", 160, 191, ["talk @npc@ (160, 191)", "talk continue", "talk resp 1", "talk resp 2", "talk resp 1"]),
    "thief":     ("prontera", 231, 38,  ["talk @npc@ (231, 38)", "talk continue", "talk resp 1", "talk resp 2", "talk resp 1"]),
    "acolyte":   ("prontera", 200, 170, ["talk @npc@ (200, 170)", "talk continue", "talk resp 1", "talk resp 2", "talk resp 1"]),
    "mage":      ("prontera", 180, 150, ["talk @npc@ (180, 150)", "talk continue", "talk resp 1", "talk resp 2", "talk resp 1"]),
    "swordman":  ("prontera", 140, 120, ["talk @npc@ (140, 120)", "talk continue", "talk resp 1", "talk resp 2", "talk resp 1"]),
    "merchant":  ("prontera", 120, 200, ["talk @npc@ (120, 200)", "talk continue", "talk resp 1", "talk resp 2", "talk resp 1"]),
}
JOB_2_1_CLASSES = {
    "swordman":  "knight", "mage": "wizard", "archer": "hunter", "acolyte": "priest",
    "merchant":  "blacksmith", "thief": "assassin", "taekwon": "star_gladiator",
    "gunslinger": "gunslinger_2", "ninja": "ninja_2", "soul_linker": "soul_linker_2",
}


CLASS_SKILL_TRAINING = {
    "swordman": [("SM_BASH", 10, "Bash — core damage skill"), ("SM_RECOVERY", 5, "Increase HP Recovery")],
    "mage":     [("MG_FIREBOLT", 10, "Fire Bolt — main damage"), ("MG_SRECOVERY", 5, "SP Recovery passive")],
    "archer":   [("AC_DOUBLE", 10, "Double Strafe — core skill"), ("AC_OWL", 1, "Owl's Eye passive")],
    "acolyte":  [("AL_HEAL", 10, "Heal — primary skill"), ("AL_DP", 5, "Demon Bane for undead")],
    "merchant": [("MC_MAMMON", 10, "Mammonite — damage"), ("MC_DISCOUNT", 10, "Discount passive")],
    "thief":    [("TF_DOUBLE", 10, "Double Attack passive"), ("TF_HIDING", 5, "Hiding utility")],
}

EQUIPMENT_PROGRESSION = {
    "archer": [(1, "1701", "Bow (ATK 15, 3 slots)"), (15, "1704", "Composite Bow (ATK 25, 3 slots)"), (30, "1710", "Crossbow (ATK 65, 2 slots)")],
    "swordman": [(1, "1101", "Sword (ATK 25, 3 slots)"), (15, "1107", "Blade (ATK 45, 3 slots)"), (30, "1113", "Scimitar (ATK 70, 2 slots)")],
    "mage": [(1, "1601", "Staff (ATK 15, 3 slots)"), (15, "1605", "Staff (ATK 30, 2 slots)"), (30, "1609", "Wand (ATK 50, 2 slots)")],
    "acolyte": [(1, "1501", "Mace (ATK 30, 3 slots)"), (15, "1503", "Mace (ATK 45, 2 slots)"), (30, "1510", "Sword Mace (ATK 75, 2 slots)")],
    "merchant": [(1, "1101", "Sword (ATK 25, 3 slots)"), (15, "1107", "Blade (ATK 45, 3 slots)"), (30, "1113", "Scimitar (ATK 70, 2 slots)")],
    "thief": [(1, "1301", "Knife (ATK 18, 3 slots)"), (15, "1307", "Knife (ATK 30, 2 slots)"), (30, "1313", "Knife (ATK 50, 2 slots)")],
}


# ── Helper functions ───────────────────────────────────────────────────────

def get_monster_stats(name: str) -> dict | None:
    if name in FULL_MONSTER_DB:
        return FULL_MONSTER_DB[name]
    for mob in FULL_MONSTER_DB.values():
        if isinstance(mob, dict) and mob.get("name", "").lower() == name.lower():
            return mob
    return None


def calculate_skill_dps(skill_name: str, skill_level: int, attack_power: int = 25) -> float:
    info = SKILL_DAMAGE.get(skill_name)
    if not info:
        return 0.0
    multiplier = info.get("base", 1.0) + info.get("per_level", 0.0) * skill_level
    sp_cost = info.get("sp", 0)
    delay = info.get("delay", 1.0)
    cast = info.get("cast", 0.0)
    damage = int(attack_power * multiplier)
    total_time = cast + delay
    if total_time <= 0:
        return float(damage)
    return damage / total_time


def get_skill_element(skill_name: str, skill_level: int) -> tuple[str, int]:
    info = SKILL_DAMAGE.get(skill_name)
    if not info:
        return ("Neutral", 1)
    element = info.get("element", "Neutral")
    fn = info.get("element_level_fn", lambda lv: 1)
    return (element, fn(skill_level))


def get_skill_sp_cost(skill_name: str, skill_level: int = 1) -> int:
    base = SKILL_SP_COSTS.get(skill_name, 0)
    if skill_level > 1 and base > 0:
        return base + (skill_level - 1) * 3
    return base


def get_best_skill(attack_power: int, available_skills: dict[str, int], sp_available: int) -> tuple[str, int, float]:
    best_skill = ""
    best_level = 0
    best_dps = 0.0
    for skill_name, skill_level in available_skills.items():
        if skill_level <= 0:
            continue
        sp_cost = get_skill_sp_cost(skill_name, skill_level)
        if sp_cost > sp_available:
            continue
        dps = calculate_skill_dps(skill_name, skill_level, attack_power)
        if dps > best_dps:
            best_dps = dps
            best_skill = skill_name
            best_level = skill_level
    return best_skill, best_level, best_dps


def get_nearest_breakpoint(stat_name: str, current_value: int) -> tuple[int, str]:
    breaks = STAT_BREAKPOINTS.get(stat_name, [])
    next_bp = None
    for bp_val, bp_desc in breaks:
        if bp_val > current_value:
            return (bp_val, bp_desc)
        next_bp = (bp_val, bp_desc)
    return next_bp or (99, "Max")


def get_scaling_stat_targets(job_name: str, base_level: int) -> dict[str, int]:
    milestones = SCALING_STAT_TARGETS.get(job_name, [])
    best = {}
    for mlvl, stats in milestones:
        if mlvl <= base_level:
            best = stats
    return best


def calculate_party_exp_share(party_size: int, is_full_party: bool = False) -> float:
    base_bonus = 1.0 + (party_size - 1) * 0.1
    if is_full_party:
        base_bonus *= 1.5
    return base_bonus


def calculate_weight_time_to_cap(current_weight: int, max_weight: int, avg_loot_per_kill: int = 50) -> int:
    remaining = max_weight - current_weight
    if remaining <= 0:
        return 0
    kills = remaining / max(avg_loot_per_kill, 1)
    return int(kills)


def calculate_profit_per_kill(monster_name: str, attack_power: int = 25, weapon_type: str = "dagger") -> float:
    import re
    card_info = CARD_VALUES.get(monster_name.lower(), {})
    drops_str = card_info.get("drops", [])
    total_value = 0.0
    for drop_str in drops_str:
        m = re.search(r'\((\d+)z\)', drop_str)
        if m:
            total_value += int(m.group(1))
    card_value = card_info.get("card", 0)
    drop_rate = 0.01
    card_ev = card_value * drop_rate
    return total_value + card_ev


def is_mvp(monster_name: str) -> bool:
    return monster_name.lower() in MVP_MONSTERS


def get_mvp_value(monster_name: str) -> int:
    info = MVP_MONSTERS.get(monster_name.lower(), {})
    drops = info.get("drops", [])
    total = 0
    for drop in drops:
        import re
        m = re.search(r'\((\d+)z\)', drop)
        if m:
            total += int(m.group(1))
    return total


def get_optimal_element_for_map(map_name: str) -> str:
    # AGNOSTIC (RULE.md): derive the map's element mix from the LIVE server's
    # real spawn data (spawn_loader) + the renewal mob db — never hardcoded
    # per-map monster lists.
    try:
        from ai_sidecar.autonomy.spawn_loader import load_map_spawns
        _spawns = load_map_spawns()
    except Exception:
        _spawns = {}
    map_mobs = [(mob_name, count) for mob_name, count, _r in _spawns.get(map_name, [])]
    element_counts: dict[str, int] = {}
    for mob_name, _count in map_mobs:
        stats = get_monster_stats(mob_name)
        if stats:
            elem = stats.get("element", "Neutral")
            element_counts[elem] = element_counts.get(elem, 0) + 1
    if not element_counts:
        return "Neutral"
    return max(element_counts.items(), key=lambda x: x[1])[0]


def build_spawn_circuit(map_name: str) -> list[tuple[int, int]]:
    known_spawns = {
        "prt_fild05": [(200, 200), (220, 180), (180, 220), (150, 200)],
        "pay_dun00": [(50, 50), (80, 80), (100, 30), (30, 100)],
        "orcsdun01": [(100, 100), (150, 80), (80, 150), (120, 120)],
    }
    return known_spawns.get(map_name, [(150, 150), (200, 200)])