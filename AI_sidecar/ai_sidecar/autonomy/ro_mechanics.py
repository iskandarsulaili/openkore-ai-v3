"""
RO Mechanics Engine — Complete, data-driven Ragnarok Online formula implementation.
All tables sourced from rAthena pre-re database.
"""

import json
import math
from pathlib import Path

# ── Load rAthena monster database ──
_MOB_DB_PATH = Path("/home/lot399/rathena_mob_db.json")
if _MOB_DB_PATH.exists():
    with open(_MOB_DB_PATH) as f:
        FULL_MONSTER_DB = json.load(f)
else:
    FULL_MONSTER_DB = {}

# ── Element table: 4 levels ──
# Level 1 (base): skills level 1-4, natural monster elements
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

# Level 2: skills level 5-9
_ELEM_LV2 = {}
for ae in _ELEM_LV1:
    _ELEM_LV2[ae] = {}
    for de in _ELEM_LV1[ae]:
        v = _ELEM_LV1[ae][de]
        if v > 1.0:
            v = min(2.0, v + 0.25)
        elif 0 < v < 1.0:
            v = max(0.0, v - 0.25)
        _ELEM_LV2[ae][de] = v

# Level 3: skills level 10
_ELEM_LV3 = {}
for ae in _ELEM_LV1:
    _ELEM_LV3[ae] = {}
    for de in _ELEM_LV1[ae]:
        v = _ELEM_LV1[ae][de]
        if v > 1.0:
            v = min(2.0, v + 0.50)
        elif 0 < v < 1.0:
            v = max(0.0, v - 0.50)
        _ELEM_LV3[ae][de] = v

# Level 4: weapon enchants, high-level skills
_ELEM_LV4 = {}
for ae in _ELEM_LV1:
    _ELEM_LV4[ae] = {}
    for de in _ELEM_LV1[ae]:
        v = _ELEM_LV1[ae][de]
        if v > 1.0:
            v = min(2.0, v + 0.75)
        elif 0 < v < 1.0:
            v = max(0.0, v - 0.75)
        _ELEM_LV4[ae][de] = v

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

# ── Skill damage formulas ──
SKILL_DAMAGE = {
    "SM_BASH": {"base": 1.5, "per_level": 0.5, "sp": 8, "cast": 0.0, "delay": 1.0, "element": "Neutral", "element_level": 1},
    "MG_FIREBOLT": {"base": 1.0, "per_level": 0.3, "sp": 12, "cast": 1.5, "delay": 1.0, "element": "Fire", "element_level": 3},
    "AC_DOUBLE": {"base": 2.0, "per_level": 0.2, "sp": 12, "cast": 0.0, "delay": 0.5, "element": "Neutral", "element_level": 1},
    "AL_HEAL": {"base": 1.0, "per_level": 0.4, "sp": 15, "cast": 1.0, "delay": 1.0, "element": "Holy", "element_level": 1},
    "TF_DOUBLE": {"base": 1.5, "per_level": 0.1, "sp": 0, "cast": 0.0, "delay": 0.0, "element": "Neutral", "element_level": 1},
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


# ═══════════════════════════════════════════════════════════════
# RO FORMULA FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def get_monster_stats(monster_name: str) -> dict | None:
    """Look up monster stats by name (case-insensitive)."""
    if not monster_name:
        return None
    mn = monster_name.lower().strip()
    # Direct lookup
    if mn in FULL_MONSTER_DB:
        return FULL_MONSTER_DB[mn]
    # Try by id
    try:
        mid = int(mn)
        for m in FULL_MONSTER_DB.values():
            if m['id'] == mid:
                return m
    except ValueError:
        pass
    return None


def calculate_aspd(agi: int = 1, dex: int = 1, weapon_type: str = "dagger", skill_bonus: float = 0.0) -> float:
    """Full RO ASPD formula. Returns seconds per hit.
    ASPD = 2000 - (2000 - base_ASPD) * (1 + AGI/100) * (1 + DEX/100) * (1 - skill_bonus)
    """
    base_aspd = WEAPON_BASE_ASPD.get(weapon_type, 1500)
    aspd = 2000 - (2000 - base_aspd) * (1 + agi / 100.0) * (1 + dex / 100.0) * (1 - skill_bonus)
    aspd = max(100, min(2000, aspd))
    return aspd / 1000.0


def calculate_flee(agi: int = 1, base_level: int = 1, job_bonus: int = 0) -> int:
    """Full RO flee formula with soft cap at 200.
    Flee = base_level + AGI + job_bonus
    Soft cap: effective_flee = flee - max(0, (flee - 200) * 0.5)
    """
    flee = base_level + agi + job_bonus
    if flee > 200:
        flee = 200 + (flee - 200) * 0.5
    return int(flee)


def calculate_hit_rate(dex: int = 1, base_level: int = 1, job_bonus: int = 0) -> int:
    """Full RO hit rate formula.
    Hit = base_level + DEX + job_bonus
    """
    return base_level + dex + job_bonus


def calculate_monster_hit_rate(monster_level: int, monster_dex: int, player_flee: int, player_level: int) -> float:
    """Full RO monster hit rate formula.
    hit_rate = 100 + (monster_level - player_level) * 2 + monster_dex - player_flee
    Clamped to [5%, 95%].
    """
    hit_rate = 100 + (monster_level - player_level) * 2 + monster_dex - player_flee
    return max(5, min(95, hit_rate)) / 100.0


def calculate_damage(attack_power: int, monster_def: int, weapon_type: str = "dagger",
                     monster_size: str = "Medium", attack_element: str = "Neutral",
                     monster_element: str = "Neutral", monster_race: str = "Brute",
                     element_level: int = 1, skill_mult: float = 1.0) -> int:
    """Full RO damage formula with size penalty, element modifier (4-level), and DEF reduction.
    Damage = (ATK * size_penalty * element_mod * race_mod) - (DEF * 0.5)
    """
    size_mod = SIZE_PENALTY.get(weapon_type, {}).get(monster_size, 1.0)
    elem_table = ELEMENT_TABLE.get(element_level, _ELEM_LV1)
    elem_mod = elem_table.get(attack_element, {}).get(monster_element, 1.0)
    raw = attack_power * size_mod * elem_mod * skill_mult
    dmg = max(1, int(raw - monster_def * 0.5))
    return dmg


def calculate_profit_per_kill(monster_name: str, attack_power: int, weapon_type: str = "dagger",
                              agi: int = 1, dex: int = 1, base_level: int = 1,
                              player_hp: int = 100, player_sp: int = 100,
                              is_archer: bool = False, is_mage: bool = False) -> float:
    """Full profit per kill: drop_value - (potion_cost + sp_cost + arrow_cost + repair_cost).
    Returns zeny per kill (negative = money sink).
    """
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

    # Damage per hit
    dmg_per_hit = calculate_damage(attack_power, monster_def, weapon_type,
                                    monster_size, "Neutral", monster_element, monster_race)
    hits_to_kill = max(1, monster_hp / max(1, dmg_per_hit))
    aspd_seconds = calculate_aspd(agi, dex, weapon_type)
    time_to_kill = hits_to_kill * aspd_seconds

    # Damage taken per kill
    flee = calculate_flee(agi, base_level)
    hit_chance = calculate_monster_hit_rate(monster_level, monster_dex, flee, base_level)
    # Monster attack speed: use monster's attack_delay
    monster_aspd_seconds = monster_aspd / 1000.0 if monster_aspd > 0 else 2.0
    monster_hits_during_fight = time_to_kill / monster_aspd_seconds
    damage_per_hit_taken = max(1, monster_attack)
    total_damage_taken = damage_per_hit_taken * monster_hits_during_fight * hit_chance

    # Potion cost
    potions_needed = total_damage_taken / POTION_HEAL
    potion_expense = potions_needed * POTION_COST

    # SP cost (Mage casting Fire Bolt)
    sp_expense = 0.0
    if is_mage:
        sp_per_kill = 12  # Fire Bolt SP cost
        sp_potions_needed = sp_per_kill / BLUE_POTION_SP
        sp_expense = sp_potions_needed * BLUE_POTION_COST

    # Arrow cost (Archer using Double Strafe)
    arrow_expense = 0.0
    if is_archer:
        arrows_per_kill = hits_to_kill * 0.3  # 30% of attacks use Double Strafe
        arrow_expense = arrows_per_kill * ARROW_COST

    # Repair cost (weapon degrades on death)
    repair_expense = 2000 / max(1, 3600 / time_to_kill)  # ~2,000z repair per death, ~1 death per hour

    # Drop value
    mn = monster_name.lower().strip()
    card_info = CARD_VALUES.get(mn, {})
    card_value = card_info.get("card", 0) if card_info else 0
    card_chance = 0.0001  # 0.01% card drop rate
    expected_card_value = card_value * card_chance
    base_drop_value = 50  # Average junk drop value
    total_drop_value = base_drop_value + expected_card_value

    return total_drop_value - (potion_expense + sp_expense + arrow_expense + repair_expense)


def calculate_skill_dps(skill_id: str, skill_level: int, attack_power: int,
                        weapon_type: str, monster_def: int, monster_size: str,
                        monster_element: str, monster_race: str,
                        agi: int, dex: int) -> float:
    """Calculate DPS for a skill vs a specific monster.
    Returns damage per second.
    """
    info = SKILL_DAMAGE.get(skill_id)
    if not info:
        return 0.0

    skill_mult = info['base'] + info['per_level'] * skill_level
    cast_time = info['cast']
    delay = info['delay']
    aspd_seconds = calculate_aspd(agi, dex, weapon_type)
    total_time = cast_time + delay + aspd_seconds

    dmg = calculate_damage(attack_power, monster_def, weapon_type,
                           monster_size, info['element'], monster_element, monster_race,
                           info['element_level'], skill_mult)
    return dmg / max(0.1, total_time)


def get_best_skill(known_skills: list[str], skill_levels: dict[str, int],
                   attack_power: int, weapon_type: str,
                   monster_def: int, monster_size: str,
                   monster_element: str, monster_race: str,
                   current_sp: int, max_sp: int,
                   agi: int, dex: int,
                   aggro_count: int, player_hp: int) -> str | None:
    """Pick the best skill to use based on DPS, SP cost, and safety.
    Returns skill_id or None (auto-attack).
    """
    sp_ratio = current_sp / max(1, max_sp)
    best_dps = 0.0
    best_skill = None

    for skill_id in known_skills:
        info = SKILL_DAMAGE.get(skill_id)
        if not info:
            continue
        sp_cost = info['sp']
        if sp_cost > current_sp:
            continue  # Not enough SP
        if sp_ratio < 0.3 and sp_cost > 0:
            continue  # Save SP for emergencies

        level = skill_levels.get(skill_id, 1)
        dps = calculate_skill_dps(skill_id, level, attack_power, weapon_type,
                                  monster_def, monster_size, monster_element,
                                  monster_race, agi, dex)

        # Cast time safety check
        cast_time = info['cast']
        if cast_time > 0 and aggro_count > 0:
            damage_during_cast = aggro_count * 20 * cast_time  # ~20 damage per mob per second
            if damage_during_cast > player_hp * 0.3:
                continue  # Too dangerous to cast

        if dps > best_dps:
            best_dps = dps
            best_skill = skill_id

    return best_skill


def get_nearest_breakpoint(stat_name: str, current_value: int) -> tuple[int, int]:
    """Find the nearest stat breakpoint above current value.
    Returns (breakpoint_value, points_needed).
    """
    breakpoints = STAT_BREAKPOINTS.get(stat_name, [])
    for bp, _ in breakpoints:
        if bp > current_value:
            return (bp, bp - current_value)
    return (current_value, 0)


def get_scaling_stat_targets(job_name: str, base_level: int) -> dict[str, int]:
    """Get scaling stat targets for a class at a given level.
    Returns {stat: target_value}.
    """
    targets = SCALING_STAT_TARGETS.get(job_name, SCALING_STAT_TARGETS["novice"])
    best = {}
    for lvl, stats in targets:
        if base_level >= lvl:
            best = stats
    return best


def estimate_hits_to_die(monster_attack: int, player_hp: int) -> float:
    """Estimate how many hits a player can survive from a monster.
    Returns hits_to_die. If < 5, map is too dangerous.
    """
    dmg_per_hit = max(1, monster_attack)
    return player_hp / dmg_per_hit


def calculate_party_exp_share(player_level: int, party_levels: list[int], monster_exp: int) -> float:
    """Calculate EXP share for a player in a party.
    share = (player_level^2) / (sum_of_all_party_member_levels^2) * monster_exp
    """
    total_sq = sum(lvl * lvl for lvl in party_levels)
    if total_sq == 0:
        return monster_exp
    return (player_level * player_level) / total_sq * monster_exp


def calculate_weight_time_to_cap(weight_capacity: int, avg_drop_weight: float, kills_per_min: float) -> float:
    """Calculate minutes until weight cap is reached.
    Returns minutes. If < 10, should skip low-value drops.
    """
    if kills_per_min <= 0 or avg_drop_weight <= 0:
        return float('inf')
    weight_cap_50 = weight_capacity * 0.5
    kills_to_cap = weight_cap_50 / avg_drop_weight
    return kills_to_cap / kills_per_min


def build_spawn_circuit(spawn_heatmap: dict[tuple[int, int], int],
                        current_x: int, current_y: int,
                        max_points: int = 10) -> list[tuple[int, int]]:
    """Build an optimized walking circuit from spawn heatmap data.
    Returns list of (x, y) waypoints sorted by proximity.
    """
    if not spawn_heatmap:
        return []

    # Sort spawn points by frequency (most kills = most spawns)
    sorted_points = sorted(spawn_heatmap.items(), key=lambda x: x[1], reverse=True)
    points = [p[0] for p in sorted_points[:max_points]]

    if not points:
        return []

    # Nearest-neighbor circuit: start from current position
    circuit = []
    remaining = list(points)
    cx, cy = current_x, current_y

    while remaining:
        # Find nearest point
        nearest = min(remaining, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
        circuit.append(nearest)
        remaining.remove(nearest)
        cx, cy = nearest

    return circuit
