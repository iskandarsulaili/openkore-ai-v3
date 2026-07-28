from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path
import threading
from ai_sidecar.game_knowledge_db import GameKnowledgeDB
from ai_sidecar.anti_detection import BehaviorEngine, get_behavior_engine
from ai_sidecar.anti_detection.behavior_engine import BehaviorProfileType
from ai_sidecar.autonomy.ro_mechanics import (
    get_monster_stats, calculate_aspd, calculate_flee, calculate_hit_rate,
    calculate_monster_hit_rate, calculate_damage, calculate_profit_per_kill,
    calculate_skill_dps, get_best_skill, get_nearest_breakpoint,
    get_scaling_stat_targets, estimate_hits_to_die,
    calculate_party_exp_share, calculate_weight_time_to_cap,
    build_spawn_circuit, is_mvp, get_mvp_value, get_skill_element_level,
    get_optimal_element_for_map,
    ELEMENT_TABLE, SIZE_PENALTY, JOB_WEAPON_TYPE,
    WEAPON_BASE_ASPD, SKILL_DAMAGE, SKILL_SP_COSTS, FOOD_ITEMS,
    CARD_VALUES, STAT_BREAKPOINTS, SCALING_STAT_TARGETS,
    POTION_COST, POTION_HEAL, BLUE_POTION_COST, BLUE_POTION_SP, ARROW_COST,
    MVP_MONSTERS, ELEMENTAL_WEAPONS, JOB_CHANGE_TALK,
)

logger = logging.getLogger(__name__)


@dataclass
class HeuristicAction:
    kind: str  # "command" | "macro" | "reflex_override"
    command: str
    confidence: float
    domain: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HeuristicAssessment:
    horizon: str
    actions: list[HeuristicAction]
    confidence: float
    actionable: bool
    top_domain: str
    signals: dict[str, Any] = field(default_factory=dict)


# ── Town maps constant (used by HUNT and TOWN_STUCK states) ──
_HUNT_TOWNS = ("prontera", "izlude", "morocc", "payon", "geffen",
               "aldebaran", "comodo", "umbala", "niflheim",
               "rachel", "veins", "einbroch", "lighthalzen",
               "juno", "hugel", "yuno", "amatsu", "gonryun",
               "louyang", "ayothaya")

# ── Class-aware stat builds ──
# Each entry: (stat_priority_list, description)
CLASS_STAT_BUILDS: dict[str, list[tuple[str, int]]] = {
    # Pro RO builds: stat priority based on class mechanics
    # Archer: DEX to 50 for hit rate, then AGI for ASPD, then LUK for crits
    "novice":    [("dex", 20), ("str", 10), ("agi", 10)],
    "swordman":  [("str", 40), ("vit", 30), ("dex", 20)],  # Bash has 100% hit, STR first
    "mage":      [("int", 50), ("dex", 20)],                # INT for damage, DEX for cast time
    "archer":    [("dex", 50), ("agi", 30), ("luk", 20)],   # DEX for hit, AGI for ASPD, LUK for crit
    "acolyte":   [("int", 50), ("dex", 20), ("vit", 10)],   # INT for Heal damage, DEX for cast
    "merchant":  [("str", 50), ("vit", 30), ("dex", 10)],   # STR for damage, VIT for tank
    "thief":     [("agi", 50), ("dex", 20), ("str", 20)],   # AGI for ASPD+Double Attack proc, DEX for hit
    "taekwon":   [("str", 40), ("agi", 30), ("dex", 10)],
    "gunslinger":[("dex", 60), ("agi", 20)],                 # DEX for hit, AGI for ASPD
    "ninja":     [("int", 50), ("dex", 20)],
    "soul_linker":[("int", 60), ("dex", 20)],
}

# ── Job change NPC locations ──
JOB_CHANGE_NPCS: dict[str, tuple[str, int, int]] = {
    "novice": ("prontera", 160, 191),   # Archer Guild
    "archer": ("prontera", 160, 191),    # Bowman Guild
    "thief": ("prontera", 231, 38),      # Thief Guild
    "acolyte": ("prontera", 200, 170),   # Acolyte Guild (approximate)
    "mage": ("prontera", 180, 150),      # Mage Guild (approximate)
    "swordman": ("prontera", 140, 120),  # Swordman Guild (approximate)
    "merchant": ("prontera", 120, 200),  # Merchant Guild (approximate)
}

# ── Bot role assignments ──
BOT_ROLES: dict[str, str] = {
    # Dynamic: first bot in all_bots is leader, rest are dps
}

BOT_JOBS: dict[str, str] = {
    # Dynamic: class read from snapshot job_name field
}

# ── Class-aware hunting grounds ──
# (min_level, max_level, map_name, description)
CLASS_HUNTING_GROUNDS: dict[str, list[tuple[int, int, str, str]]] = {
    # Pro RO progression: dungeons for density, field maps only as fallback
    # Dungeons have 3-5x spawn density vs field maps
    "novice": [
        (1, 10,  "prt_fild05",   "Prontera Field — Porings, Lunatics, Pupa (safe, level 1-10)"),
        (10, 20, "pay_dun00",    "Payon Cave 1F — Skeletons, Zombies (undead, 3x density)"),
        (20, 35, "pay_dun01",    "Payon Cave 2F — Munak, Bongun, Ghoul (undead, 5x density)"),
        (35, 50, "gef_dun00",    "Geffen Dungeon 1F — Drainliar, Creamy, Flora (element advantage)"),
    ],
    "swordman": [
        (1, 15,  "pay_dun00",     "Payon Cave 1F — Skeletons (Bash one-shots undead)"),
        (15, 30, "orcsdun01",     "Orc Dungeon — Orc Warriors, Orc Archers (melee heaven, 5x density)"),
        (30, 50, "gef_dun01",     "Geffen Dungeon 2F — Anacondaq, Stapo, Alligator"),
    ],
    "mage": [
        (1, 15,  "gef_dun00",     "Geffen Dungeon 1F — Drainliar, Creamy (Fire Bolt one-shots)"),
        (15, 30, "gef_dun01",     "Geffen Dungeon 2F — Anacondaq, Stapo (element advantage)"),
        (30, 50, "mag_dun01",     "Magma Dungeon 1F — Kaho, Lava Golem (fire element)"),
    ],
    "archer": [
        (1, 15,  "pay_dun00",     "Payon Cave 1F — Skeletons, Zombies (kite from range)"),
        (15, 30, "pay_dun01",     "Payon Cave 2F — Munak, Bongun (Double Strafe burst)"),
        (30, 50, "iz_dun00",      "Byalan Dungeon 1F — Marine Sphere, Kukre (water element)"),
    ],
    "acolyte": [
        (1, 15,  "pay_dun00",     "Payon Cave 1F — Undead (Heal one-shots, 3x EXP)"),
        (15, 30, "pay_dun01",     "Payon Cave 2F — Munak, Bongun (Turn Undead)"),
        (30, 50, "iz_dun00",      "Byalan Dungeon 1F — Marine Sphere (Heal vs undead)"),
    ],
    "merchant": [
        (1, 15,  "pay_dun00",     "Payon Cave 1F — Skeletons (tanky, high HP)"),
        (15, 30, "orcsdun01",     "Orc Dungeon — Orc Warriors (tanky, good drops)"),
        (30, 50, "gef_dun01",     "Geffen Dungeon 2F — Anacondaq, Stapo"),
    ],
    "thief": [
        (1, 15,  "pay_dun00",     "Payon Cave 1F — Skeletons (Double Attack procs)"),
        (15, 30, "pay_dun01",     "Payon Cave 2F — Munak, Bongun (AGI build shines)"),
        (30, 50, "iz_dun00",      "Byalan Dungeon 1F — Marine Sphere, Kukre"),
    ],
    "taekwon": [
        (1, 15,  "pay_dun00",     "Payon Cave 1F — Skeletons (kick damage)"),
        (15, 30, "pay_dun01",     "Payon Cave 2F — Munak, Bongun"),
        (30, 50, "iz_dun00",      "Byalan Dungeon 1F — Marine Sphere"),
    ],
    "gunslinger": [
        (1, 15,  "pay_dun00",     "Payon Cave 1F — Skeletons (Single Action burst)"),
        (15, 30, "pay_dun01",     "Payon Cave 2F — Munak, Bongun"),
        (30, 50, "iz_dun00",      "Byalan Dungeon 1F — Marine Sphere"),
    ],
    "ninja": [
        (1, 15,  "pay_dun00",     "Payon Cave 1F — Skeletons (Kunai range)"),
        (15, 30, "pay_dun01",     "Payon Cave 2F — Munak, Bongun"),
        (30, 50, "iz_dun00",      "Byalan Dungeon 1F — Marine Sphere"),
    ],
    "soul_linker": [
        (1, 15,  "pay_dun00",     "Payon Cave 1F — Skeletons (Soul Strike)"),
        (15, 30, "pay_dun01",     "Payon Cave 2F — Munak, Bongun"),
        (30, 50, "iz_dun00",      "Byalan Dungeon 1F — Marine Sphere"),
    ],
}

# ── Class-aware skill training priorities ──
# (skill_id, max_level, description)
CLASS_SKILL_PRIORITIES: dict[str, list[tuple[str, int, str]]] = {
    "novice":    [("NV_BASIC", 1, "Basic Skill — sit to regen"), ("NV_FIRSTAID", 1, "First Aid — self-healing")],
    "swordman":  [("SM_BASH", 10, "Bash — core damage skill"), ("SM_RECOVERY", 5, "Increase HP Recovery")],
    "mage":      [("MG_SRECOVERY", 4, "SP Recovery — sustain"), ("MG_FIREBOLT", 10, "Fire Bolt — main nuke")],
    "archer":    [("AC_OWL", 1, "Owl's Eye — DEX boost"), ("AC_DOUBLE", 10, "Double Strafe — burst DPS")],
    "acolyte":   [("AL_HEAL", 10, "Heal — primary heal/nuke undead"), ("AL_DEMONBANE", 5, "Demon Bane — vs Undead")],
    "merchant":  [("MC_VENDING", 1, "Vending — sell items"), ("MC_DISCOUNT", 10, "Discount — cheaper buys")],
    "thief":     [("TF_DOUBLE", 10, "Double Attack — ASPD burst"), ("TF_HIDING", 5, "Hiding — escape")],
    "taekwon":   [("TK_PUNCH", 5, "Punch — kick damage"), ("TK_DODGE", 5, "Dodge — AGI synergy")],
    "gunslinger":[("GS_SINGLEACTION", 5, "Single Action — burst"), ("GS_CHAINACTION", 5, "Chain Action — multi-shot")],
    "ninja":     [("NJ_KUNAI", 5, "Kunai — ranged attack"), ("NJ_SYURIKEN", 5, "Syuriken — AoE")],
    "soul_linker":[("SL_SOULSTRIKE", 5, "Soul Strike — main nuke"), ("SL_SPIRIT", 5, "Spirit — buff")],
}


def _class_stat_allocation(
    job_name: str,
    current_stats: dict[str, int],
    stat_points: int,
    adaptive: AdaptiveDataStore | None = None,
    base_level: int = 1,
) -> list[tuple[str, int]]:
    """Determine which stats to allocate for a given class.

    Uses scaling stat targets with breakpoint awareness.
    Allocates to the nearest breakpoint (5/7/10) for maximum efficiency.
    Returns a list of (stat_name, points_to_add) tuples.
    """
    if stat_points <= 0:
        return []

    # Get scaling targets for this level
    if adaptive:
        targets = adaptive.get_scaling_stat_targets(job_name, base_level)
        if not targets:
            targets = dict(CLASS_STAT_BUILDS.get(job_name, CLASS_STAT_BUILDS["novice"]))
    else:
        targets = dict(CLASS_STAT_BUILDS.get(job_name, CLASS_STAT_BUILDS["novice"]))

    allocations: list[tuple[str, int]] = []
    remaining = stat_points

    # First pass: allocate to reach scaling targets
    for stat_name, target in targets.items():
        current = current_stats.get(stat_name, 1)
        needed = max(0, target - current)
        if needed > 0:
            add = min(needed, remaining)
            if add > 0:
                allocations.append((stat_name, add))
                remaining -= add
        if remaining <= 0:
            break

    # Second pass: allocate to nearest breakpoint for primary stats
    if remaining > 0 and adaptive:
        # Sort stats by priority (first in targets = highest priority)
        sorted_stats = sorted(targets.items(), key=lambda x: x[1], reverse=True)
        for stat_name, _ in sorted_stats:
            if remaining <= 0:
                break
            current = current_stats.get(stat_name, 1)
            # Find nearest breakpoint
            bp, needed = adaptive.get_nearest_breakpoint(stat_name, current)
            if needed > 0:
                add = min(needed, remaining)
                if add > 0:
                    allocations.append((stat_name, add))
                    remaining -= add

    # Dump remaining points into the first priority stat
    if remaining > 0 and targets:
        first_stat = list(targets.keys())[0]
        allocations.append((first_stat, remaining))

    return allocations


def _class_hunting_ground(
    job_name: str,
    base_level: int,
    current_map: str,
    adaptive: AdaptiveDataStore | None = None,
) -> tuple[str, str] | None:
    """Find the best hunting ground for a class at a given level.

    Uses adaptive data (kills, deaths, exp per visit) to rank maps.
    Falls back to hardcoded defaults for unknown maps.
    Returns (map_name, description) or None if already on the best map.
    """
    grounds = CLASS_HUNTING_GROUNDS.get(job_name, CLASS_HUNTING_GROUNDS["novice"])

    # Score each candidate map using adaptive data
    candidates = []
    for entry in grounds:
        min_lv, max_lv, map_name, desc = entry
        if min_lv <= base_level <= max_lv:
            if adaptive:
                map_score = adaptive.get_map_score(map_name)
                # Boost maps with good performance, penalize bad ones
                if map_score > 0:
                    candidates.append((map_score, entry))
                else:
                    candidates.append((0.5, entry))  # Default score for unknown maps
            else:
                candidates.append((0.5, entry))

    if not candidates:
        # Fallback: pick the last entry (highest level range)
        if grounds:
            candidates = [(0.5, grounds[-1])]

    if candidates:
        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, best = candidates[0]
        _, _, map_name, desc = best
        # Don't re-route if already on the correct map
        if current_map and map_name in current_map:
            return None
        return (map_name, desc)

    return None


def _class_skill_training(
    job_name: str,
    known_skills: list[str],
    skill_points: int,
    adaptive: AdaptiveDataStore | None = None,
) -> list[tuple[str, int, str]]:
    """Determine which skills to train next for a given class.

    Uses adaptive data (usage frequency, effectiveness) to prioritize.
    Falls back to hardcoded defaults.
    Returns a list of (skill_id, target_level, description) tuples.
    """
    if skill_points <= 0:
        return []

    priorities = CLASS_SKILL_PRIORITIES.get(job_name, CLASS_SKILL_PRIORITIES["novice"])

    # Re-prioritize based on adaptive data
    if adaptive and job_name in adaptive.skill_priority:
        skill_usage = adaptive.skill_priority[job_name]
        # Sort by usage frequency (most used = most important)
        sorted_skills = sorted(skill_usage.items(), key=lambda x: x[1], reverse=True)
        # Merge with hardcoded priorities: known used skills first, then hardcoded
        used_skills = {s[0] for s in sorted_skills}
        result: list[tuple[str, int, str]] = []
        for skill_id, _ in sorted_skills:
            if skill_id not in known_skills:
                result.append((skill_id, 1, f"Adaptive: used {skill_usage[skill_id]:.0f}x"))
                return result
        # If all used skills are known, fall through to hardcoded
        for skill_id, max_lv, desc in priorities:
            if skill_id not in known_skills:
                result.append((skill_id, 1, desc))
                return result
        return result

    # Hardcoded fallback
    result: list[tuple[str, int, str]] = []
    for skill_id, max_lv, desc in priorities:
        if skill_id not in known_skills:
            result.append((skill_id, 1, desc))
            return result  # Train one skill at a time

    return result


class AdaptiveDataStore:
    """Thread-safe data store that learns from outcomes.

    Tracks map performance, NPC locations, economy patterns.
    All public methods use RLock for concurrent bot access.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self.map_performance: dict[str, dict[str, float]] = {}
        self.stat_effectiveness: dict[str, dict[str, float]] = {}
        self.skill_priority: dict[str, dict[str, float]] = {}
        self.npc_locations: dict[str, dict[str, list[tuple[int, int, str]]]] = {}
        self.economy_data: dict[str, dict[str, float]] = {}
        self.death_analysis: dict[str, dict[str, Any]] = {}
        # Spawn heatmap: map_name -> {(x, y): count}
        self.spawn_heatmap: dict[str, dict[tuple[int, int], int]] = {}
        # Equipment progression: level -> weapon_id (rAthena-corrected IDs)
        # rAthena actual IDs: Knife=1201, Sword=1101, Bow=1701, Mace=1501, Rod=1601
        self.equipment_progression: dict[str, list[tuple[int, str, str]]] = {
            "novice": [(1, "1201", "Knife (ATK 17, 3 slots)")],
            "archer": [(1, "1701", "Bow (ATK 15, 3 slots)"), (15, "1704", "Composite Bow (ATK 25, 3 slots)"), (30, "1710", "Crossbow (ATK 65, 2 slots)")],
            "thief": [(1, "1201", "Knife (ATK 17, 3 slots)"), (15, "1207", "Main Gauche (ATK 30, 3 slots)"), (30, "1222", "Damascus (ATK 55, 1 slot)")],
            "swordman": [(1, "1101", "Sword (ATK 25, 3 slots)"), (15, "1107", "Blade (ATK 45, 3 slots)"), (30, "1113", "Scimitar (ATK 70, 2 slots)")],
            "mage": [(1, "1601", "Rod (ATK 15, 3 slots)"), (15, "1607", "Staff (ATK 30, 2 slots)"), (30, "1604", "Wand (ATK 50, 2 slots)")],
            "acolyte": [(1, "1501", "Mace (ATK 23, 3 slots)"), (15, "1504", "Mace (ATK 23, 3 slots)"), (30, "1510", "Flail (ATK 69, 2 slots)")],
        }
        # Loot value estimation: item_id -> estimated_vendor_price
        # Only pick up items worth more than the potion cost to kill the monster
        self.loot_values: dict[str, int] = {
            "909": 10,   # Jellopy (Poring drop) - cheap, skip if weight > 50%
            "938": 50,   # Memento - worth picking up
            "703": 500,  # Hinalle - valuable herb
            "704": 1000, # Aloe - valuable herb
            "705": 2000, # Master's Herb - very valuable
            "706": 5000, # Yggdrasil Seed - extremely valuable
            "707": 10000,# Yggdrasil Berry - jackpot
            "511": 100,  # Green Herb - common
            "512": 50,   # Apple - cheap
            "513": 30,   # Banana - cheap
            "514": 20,   # Grape - cheap
            "515": 10,   # Carrot - cheap
            "516": 5,    # Potato - very cheap
            "517": 3,    # Meat - very cheap
            "518": 2,    # Honey - very cheap
            "601": 1,    # Fly Wing - cheap but useful
            "602": 1,    # Butterfly Wing - cheap but useful
            "501": 10,   # Red Potion - useful
            "502": 20,   # Orange Potion - useful
            "503": 40,   # Yellow Potion - useful
            "504": 100,  # White Potion - useful
            "505": 200,  # Blue Potion - useful
            "506": 500,  # Awakening Potion - useful
        }
        # Monster stats: delegate to ro_mechanics (1,004 monsters from rAthena)
        # This replaces the old 26-monster hardcoded dict
        self.monster_stats: dict[str, dict] = {}  # Kept for backward compat, use get_monster_stats() instead
        # Spawn rotation prediction: map_name -> [(x, y, monster_name, respawn_time)]
        self.spawn_rotation: dict[str, list[tuple[int, int, str, float]]] = {}
        # Map connections: map_name -> [connected_map_names] (from rAthena warp scripts)
        self.map_connections: dict[str, list[str]] = {
            "pay_dun00": ["pay_arche", "pay_dun01"],
            "pay_dun01": ["pay_dun00", "pay_dun02"],
            "gef_dun00": ["gef_dun01", "gef_tower"],
            "orcsdun01": ["in_orcs01", "orcsdun02"],
            "iz_dun00": ["iz_dun01", "izlu2dun"],
            "prt_fild05": ["mjolnir_09", "prontera", "prt_sewb1"],
            "prt_fild04": [],
            "payon": ["pay_arche", "pay_fild01", "pay_fild08", "pay_gld", "payon_in01", "payon_in03"],
            "pay_arche": ["pay_dun00", "payon"],
            "prontera": ["prt_church", "prt_fild05", "prt_fild06", "prt_fild08", "prt_in"],
            "geffen": ["gef_fild00", "gef_fild04", "gef_fild07", "gef_tower", "geffen_in"],
            "morocc": ["moc_fild07", "moc_fild12", "moc_fild19", "moc_fild20", "moc_ruins", "morocc_in"],
            "izlude": [],
            "alberta": ["alb_ship", "alberta_in", "pay_fild03"],
            "aldebaran": ["alde_alche", "alde_gld", "aldeba_in", "c_tower1", "mjolnir_12", "xmas_fild01"],
        }
        # Map spawn data: map_name -> [(monster_name, count, respawn_ms)]
        # From rAthena pre-re mob spawn scripts
        self.map_spawns: dict[str, list[tuple[str, int, int]]] = {
            "pay_dun00": [("Familiar", 15, 0), ("Zombie", 20, 0), ("Skeleton", 35, 0), ("Poporing", 15, 0)],
            "pay_dun01": [("Munak", 20, 0), ("Bongun", 15, 0), ("Ghoul", 10, 0), ("Skeleton", 30, 0)],
            "gef_dun00": [("Hunter Fly", 30, 60000), ("Poporing", 15, 0), ("Poison Spore", 25, 0)],
            "orcsdun01": [("Steel Chonchon", 10, 0), ("Familiar", 15, 0), ("Drainliar", 5, 0), ("Orc Zombie", 80, 60000)],
            "iz_dun00": [("Plankton", 65, 0), ("Marina", 45, 0), ("Kukre", 15, 0), ("Hydra", 15, 0), ("Vadon", 15, 0)],
            "prt_fild05": [("Poring", 70, 0), ("Thief Bug Egg", 20, 0), ("Lunatic", 30, 0), ("Pupa", 30, 0), ("Thief Bug", 10, 0)],
            "prt_fild04": [("Rocker", 70, 0), ("Creamy", 40, 0), ("Pupa", 10, 0), ("Poring", 30, 0)],
        }

        # ── RO MECHANICS TABLES (imported from ro_mechanics.py) ──
        # Size penalty, element table, weapon types, stat breakpoints,
        # scaling targets, skill data, food items, card values are all
        # in the ro_mechanics module. These are kept for backward compatibility.
        self.size_penalty = SIZE_PENALTY
        self.element_table = ELEMENT_TABLE
        self.job_weapon_type = JOB_WEAPON_TYPE
        self.stat_breakpoints = STAT_BREAKPOINTS
        self.scaling_stat_targets = SCALING_STAT_TARGETS
        self.skill_sp_costs = SKILL_SP_COSTS
        self.skill_damage = SKILL_DAMAGE
        self.food_items = FOOD_ITEMS
        self.card_values = CARD_VALUES

    def record_kill(self, map_name: str, exp_gained: float, x: int = 0, y: int = 0, monster_name: str = "") -> None:
        with self._lock:
            self.map_performance.setdefault(map_name, {"kills": 0, "deaths": 0, "exp": 0, "visits": 0, "last_visit": 0})
            self.map_performance[map_name]["kills"] += 1
            self.map_performance[map_name]["exp"] += exp_gained
            self.map_performance[map_name]["last_visit"] = __import__("time").time()
            # Record spawn position for heatmap
            if x > 0 and y > 0:
                self.spawn_heatmap.setdefault(map_name, {})
                key = (x // 10 * 10, y // 10 * 10)  # Bucket to 10x10 cells
                self.spawn_heatmap[map_name][key] = self.spawn_heatmap[map_name].get(key, 0) + 1
                # Record spawn rotation (predict respawn time ~5s for most mobs)
                self.spawn_rotation.setdefault(map_name, [])
                _now = __import__("time").time()
                # Remove old entries (>30s old)
                self.spawn_rotation[map_name] = [(sx, sy, sm, st) for sx, sy, sm, st in self.spawn_rotation[map_name] if _now - st < 30]
                # Add this kill as a predicted respawn point
                self.spawn_rotation[map_name].append((x, y, monster_name, _now + 5.0))  # Predict respawn in ~5s

    def record_death(self, map_name: str, hp_at_death: float = 0) -> None:
        with self._lock:
            self.map_performance.setdefault(map_name, {"kills": 0, "deaths": 0, "exp": 0, "visits": 0, "last_visit": 0})
            self.map_performance[map_name]["deaths"] += 1
            self.death_analysis.setdefault(map_name, {"deaths": 0, "causes": {}, "avg_hp": 0})
            self.death_analysis[map_name]["deaths"] += 1
            old_avg = self.death_analysis[map_name]["avg_hp"]
            old_count = self.death_analysis[map_name]["deaths"] - 1
            self.death_analysis[map_name]["avg_hp"] = (old_avg * old_count + hp_at_death) / max(self.death_analysis[map_name]["deaths"], 1)

    def record_visit(self, map_name: str) -> None:
        with self._lock:
            self.map_performance.setdefault(map_name, {"kills": 0, "deaths": 0, "exp": 0, "visits": 0, "last_visit": 0})
            self.map_performance[map_name]["visits"] += 1
            self.map_performance[map_name]["last_visit"] = __import__("time").time()

    def estimate_kill_time(self, monster_name: str, attack_power: int = 25) -> float:
        """Estimate seconds to kill a monster given current attack power.
        Uses monster_stats cache. Returns 0 if unknown monster.
        Formula: (HP / max(1, ATK - DEF*0.5)) * attack_speed/1000
        """
        _mn = monster_name.lower().strip()
        stats = get_monster_stats(_mn)
        if not stats:
            return 0.0
        hp = stats["hp"]
        _def = stats.get("def", 0)
        _dmg_per_hit = max(1, attack_power - _def * 0.5)
        _hits_needed = hp / _dmg_per_hit
        _aspd = stats.get("attack_speed", 2000)
        _seconds_per_hit = _aspd / 1000.0
        return _hits_needed * _seconds_per_hit

    def get_best_target(self, monsters_nearby: list[dict], attack_power: int = 25) -> str | None:
        """Pick the best monster to attack: highest exp per second.
        Returns monster name or None.
        """
        best = None
        best_eps = 0.0
        for m in monsters_nearby:
            _name = m.get("name", "")
            _stats = get_monster_stats(_name.lower().strip())
            if not _stats:
                continue
            _kill_time = self.estimate_kill_time(_name, attack_power)
            if _kill_time <= 0:
                continue
            _exp = _stats.get("exp", 0)
            _eps = _exp / _kill_time
            if _eps > best_eps:
                best_eps = _eps
                best = _name
        return best

    def get_optimal_hunting_map(self, job_name: str, base_level: int, attack_power: int = 25) -> tuple[str, str]:
        """Find the best hunting map using rAthena data.
        Considers: monster density, monster HP vs player ATK, exp per kill, element advantage.
        Returns (map_name, reason).
        """
        grounds = CLASS_HUNTING_GROUNDS.get(job_name, CLASS_HUNTING_GROUNDS["novice"])
        best_map = None
        best_reason = ""
        best_score = -1.0

        for min_lv, max_lv, map_name, desc in grounds:
            if not (min_lv <= base_level <= max_lv):
                continue
            # Score this map using rAthena spawn data
            spawns = self.map_spawns.get(map_name, [])
            if not spawns:
                # No spawn data - use default score
                if best_map is None:
                    best_map = map_name
                    best_reason = desc
                    best_score = 0.5
                continue

            total_exp = 0
            total_hp = 0
            total_count = 0
            danger_level = 0  # Higher = more dangerous

            for m_name, count, _ in spawns:
                stats = get_monster_stats(m_name.lower().strip())
                if stats:
                    total_exp += stats["exp"] * count
                    total_hp += stats["hp"] * count
                    total_count += count
                    # Danger: monster ATK > player HP/2
                    if stats["attack"] > attack_power * 2:
                        danger_level += count
                    # Element advantage: undead vs holy (Acolyte Heal)
                    if job_name == "acolyte" and stats["element"] == "Undead":
                        total_exp *= 3  # Heal one-shots undead
                    # Element disadvantage: avoid elements that resist our attacks
                    if stats["element"] == "Water" and job_name in ("thief", "swordman"):
                        total_exp *= 0.5  # Physical vs water = reduced damage

            if total_count == 0:
                continue

            avg_exp = total_exp / total_count
            avg_hp = total_hp / total_count
            hits_to_kill = avg_hp / max(1, attack_power)
            danger_ratio = danger_level / max(total_count, 1)

            # Score: high exp, low hits-to-kill, low danger
            score = avg_exp / max(hits_to_kill, 1) * (1 - danger_ratio * 0.5)

            if score > best_score:
                best_score = score
                best_map = map_name
                best_reason = f"{desc} (score={score:.1f}, avg_hits={hits_to_kill:.1f}, danger={danger_ratio:.0%})"

        if best_map is None:
            return ("prt_fild05", "Fallback: Prontera Field (safe for all classes)")
        return (best_map, best_reason)

    def get_optimal_weapon(self, job_name: str, base_level: int, zeny: int) -> tuple[str, str] | None:
        """Find the best weapon to buy using rAthena-corrected IDs.
        Returns (weapon_id, description) or None if can't afford.
        """
        prog = self.equipment_progression.get(job_name, self.equipment_progression["novice"])
        best = None
        for lvl, wid, desc in prog:
            if base_level >= lvl:
                best = (wid, desc)
        if best and zeny >= 100:
            return best
        return None

    def estimate_survivability(self, map_name: str, base_level: int, attack_power: int = 25) -> float:
        """Estimate survivability on a map (0.0 = deadly, 1.0 = safe).
        Uses rAthena monster stats to check if monsters are too strong.
        """
        spawns = self.map_spawns.get(map_name, [])
        if not spawns:
            return 0.8  # Unknown map - assume moderately safe

        total_danger = 0
        total_count = 0
        for m_name, count, _ in spawns:
            stats = get_monster_stats(m_name.lower().strip())
            if stats:
                total_count += count
                # Danger if monster level > player level + 5
                if stats["level"] > base_level + 5:
                    total_danger += count * 2
                # Danger if monster ATK > player ATK * 3
                if stats["attack"] > attack_power * 3:
                    total_danger += count * 3
                # Danger if monster HP > player ATK * 20
                if stats["hp"] > attack_power * 20:
                    total_danger += count

        if total_count == 0:
            return 0.8

        danger_ratio = total_danger / total_count
        survivability = max(0.0, 1.0 - danger_ratio * 0.3)
        return survivability

    def calculate_aspd(self, agi: int = 1, dex: int = 1, base_aspd: int = 1560, skill_bonus: float = 0.0) -> float:
        """Delegate to ro_mechanics with weapon type from job.
        NOTE: weapon_type is hardcoded to 'dagger' because this delegate
        doesn't receive job_name. Callers should use ro_mechanics directly.
        """
        return calculate_aspd(agi, dex, "dagger", skill_bonus)

    def calculate_flee(self, agi: int = 1, base_level: int = 1, job_bonus: int = 0) -> int:
        """Delegate to ro_mechanics (with soft cap)."""
        return calculate_flee(agi, base_level, job_bonus)

    def calculate_hit_rate(self, dex: int = 1, base_level: int = 1, job_bonus: int = 0) -> int:
        """Delegate to ro_mechanics."""
        return calculate_hit_rate(dex, base_level, job_bonus)

    def calculate_damage(self, attack_power: int, monster_def: int, weapon_type: str = "dagger",
                         monster_size: str = "Medium", attack_element: str = "Neutral",
                         monster_element: str = "Neutral", monster_race: str = "Brute",
                         skill_mult: float = 1.0) -> int:
        """Delegate to ro_mechanics (uses 4-level element table, element_level from skill)."""
        return calculate_damage(attack_power, monster_def, weapon_type,
                               monster_size, attack_element, monster_element, monster_race,
                               1, skill_mult)

    def calculate_profit_per_kill(self, monster_name: str, attack_power: int, weapon_type: str = "dagger",
                                   agi: int = 1, dex: int = 1, base_level: int = 1,
                                   monster_hp: int = 50, monster_def: int = 0,
                                   monster_size: str = "Medium", monster_element: str = "Neutral",
                                   monster_race: str = "Brute", monster_attack: int = 7) -> float:
        """Delegate to ro_mechanics (full profit calc with SP/arrows/repair).
        Uses get_monster_stats() internally, so monster_hp/def/size/etc params are ignored.
        NOTE: _last_job is set in _assess_impl from signals. If not set, defaults to 'novice'.
        """
        _job = getattr(self, '_last_job', 'novice')
        _is_archer = _job == 'archer'
        _is_mage = _job == 'mage'
        return calculate_profit_per_kill(monster_name, attack_power, weapon_type,
                                         agi, dex, base_level, 100, 100,
                                         _is_archer, _is_mage)

    def get_nearest_breakpoint(self, stat_name: str, current_value: int) -> tuple[int, int]:
        """Delegate to ro_mechanics."""
        return get_nearest_breakpoint(stat_name, current_value)

    def get_scaling_stat_targets(self, job_name: str, base_level: int) -> dict[str, int]:
        """Delegate to ro_mechanics."""
        return get_scaling_stat_targets(job_name, base_level)

    def get_skill_rotation(self, job_name: str, current_sp: int, max_sp: int,
                           monster_element: str = "Neutral", monster_hp: int = 50,
                           attack_power: int = 25) -> list[dict]:
        """Deprecated - use get_best_skill() from ro_mechanics instead."""
        return []

    def estimate_hits_to_die(self, monster_attack: int, player_hp: int) -> float:
        """Delegate to ro_mechanics."""
        return estimate_hits_to_die(monster_attack, player_hp)

    def get_map_profit_score(self, map_name: str, job_name: str, base_level: int,
                              attack_power: int, agi: int = 1, dex: int = 1,
                              player_hp: int = 100) -> float:
        """Score a map by profit per hour, factoring death penalty and travel cost.
        Uses actual bot stats (agi, dex, player_hp) for accurate calculation.
        """
        spawns = self.map_spawns.get(map_name, [])
        if not spawns:
            return 0.0

        _weapon_type = self.job_weapon_type.get(job_name, "dagger")
        _total_profit = 0.0
        _total_time = 0.0
        _total_danger = 0

        for m_name, count, _ in spawns:
            stats = get_monster_stats(m_name.lower().strip())
            if not stats:
                continue

            _profit = calculate_profit_per_kill(
                m_name, attack_power, _weapon_type, agi, dex, base_level,
                player_hp, 100,
                job_name == 'archer', job_name == 'mage'
            )
            _hits_to_die = estimate_hits_to_die(stats.get("attack", 7), player_hp)
            _total_profit += _profit * count
            _total_time += (stats["hp"] / max(1, attack_power)) * count
            if _hits_to_die < 5:
                _total_danger += count

        if _total_time <= 0:
            return 0.0

        _profit_per_hour = _total_profit / _total_time * 3600
        _death_penalty = _total_danger / max(1, sum(c for _, c, _ in spawns)) * 0.01
        _travel_cost = 500 / max(1, _total_time / 3600)

        return _profit_per_hour - _death_penalty * 1000 - _travel_cost

    def get_map_score(self, map_name: str) -> float:
        with self._lock:
            perf = self.map_performance.get(map_name, {})
            kills = perf.get("kills", 0)
            deaths = perf.get("deaths", 0)
            exp = perf.get("exp", 0)
            visits = perf.get("visits", 1)
            if visits == 0:
                return 0.0
            kill_rate = kills / visits
            death_rate = deaths / max(visits, 1)
            exp_per_visit = exp / visits
            score = exp_per_visit * 0.01
            if death_rate > 0:
                score *= max(0.1, 1.0 - death_rate * 2)
            if kill_rate > 0:
                score *= min(2.0, 1.0 + kill_rate * 0.5)
            return score

    def get_best_map(self, bot_id: str, base_level: int) -> str | None:
        """Get the best hunting map for this bot's level from adaptive data."""
        with self._lock:
            if not self.map_performance:
                return None
            candidates = []
            for map_name, perf in self.map_performance.items():
                avg_level = perf.get("avg_level", base_level)
                if abs(avg_level - base_level) <= 5:
                    candidates.append((map_name, perf.get("kills", 0), perf.get("deaths", 1)))
            if not candidates:
                return None
            candidates.sort(key=lambda x: x[1] / max(x[2], 1), reverse=True)
            return candidates[0][0] if candidates else None

    def record_npc(self, service: str, map_name: str, x: int, y: int, name: str = "") -> None:
        with self._lock:
            self.npc_locations.setdefault(service, {})
            self.npc_locations[service].setdefault(map_name, [])
            for existing in self.npc_locations[service][map_name]:
                if existing[0] == x and existing[1] == y:
                    return
            self.npc_locations[service][map_name].append((x, y, name))

    def get_npc(self, service: str, map_name: str) -> tuple[int, int, str] | None:
        with self._lock:
            npcs = self.npc_locations.get(service, {}).get(map_name, [])
            if npcs:
                return npcs[0]
            return None

    def record_sale(self, item_name: str, price: float) -> None:
        with self._lock:
            self.economy_data.setdefault(item_name, {"avg_price": 0, "count": 0, "last_price": 0})
            entry = self.economy_data[item_name]
            old_avg = entry["avg_price"]
            old_count = entry["count"]
            entry["avg_price"] = (old_avg * old_count + price) / (old_count + 1)
            entry["count"] += 1
            entry["last_price"] = price



class HeuristicService:
    """Economy-first state machine for bot progression.

    Priority:
      1. SELL: In town with inventory -> sell all junk
      2. BUY: In town with zeny -> buy potions, arrows, weapon
      3. JOB_CHANGE: Level 10/10 Novice in town -> change job
      4. STATS: Have stat points -> allocate
      5. SKILLS: Have skill points -> learn
      6. PARTY: Not in party -> create/join
      7. HUNT: On hunting map -> kill monsters
      8. FLEE: Low HP -> teleport or use potion
    """

    def __init__(self):
        self._adaptive = AdaptiveDataStore()
        self._last_assessment: dict[str, HeuristicAssessment] = {}
        self._bot_state: dict[str, str] = {}
        self._state_since: dict[str, float] = {}
        self._last_progress: dict[str, dict] = {}
        self._last_sell_time: dict[str, float] = {}
        self._last_buy_time: dict[str, float] = {}
        self._bot_deaths: dict[str, int] = {}
        self._cold_start_fired: dict[str, bool] = {}
        self._cold_start_step: dict[str, int] = {}  # default 0 = needs cold start
        self._load_towns()
        self._town_entry_time: dict[str, float] = {}
        self._last_hunt_move: dict[str, float] = {}
        self._last_return_to_town: dict[str, float] = {}
        self._last_level: dict[str, int] = {}
        self._last_party_attempt: dict[str, float] = {}
        self._last_party_leave: dict[str, float] = {}
        self._last_party_members: dict[str, list] = {}
        self._last_party_seen: dict[str, float] = {}
        self._all_bots_cache: dict[str, list] = {}
        self._last_force_return: dict[str, float] = {}
        self._last_job_change_attempt: dict[str, float] = {}
        self._last_lockmap: dict[str, str] = {}
        # Config dedup cache: bot_id -> {config_key: last_set_value}
        # Prevents sending "set route_randomWalk 1" every cycle when it's already 1
        self._last_config_set: dict[str, dict[str, str]] = {}
        # Kills/hour tracking
        self._last_kills_count: dict[str, int] = {}
        self._last_kills_time: dict[str, float] = {}
        self._last_kills_log: dict[str, float] = {}
        # First-cycle flag: cleared on restart, forces re-send of critical configs
        self._first_cycle_done: dict[str, bool] = {}
        # Sitting detector: track when bot started sitting on hunting map
        self._sit_start_time: dict[str, float] = {}

    def _get_npc(self, task_type: str, map_name: str) -> dict | None:
        """Thread-safe NPC lookup - creates new DB connection per call."""
        try:
            gkd = GameKnowledgeDB()
            return gkd.find_npc_for_task(task_type, map_name)
        except Exception:
            return None

    def _load_towns(self) -> None:
        """Load town map names from database."""
        global _HUNT_TOWNS
        try:
            gkd = GameKnowledgeDB()
            conn = gkd._get_conn()
            rows = conn.execute("SELECT map_name FROM npc_interactions WHERE interaction_type='town_flag'").fetchall()
            _HUNT_TOWNS = {row['map_name'] for row in rows}
        except Exception:
            pass
        if not _HUNT_TOWNS:
            _HUNT_TOWNS = {"prontera", "morocc", "geffen", "payon", "aldebaran", "alberta", "izlude", "comodo", "umbala", "yuno", "einbroch", "einbech", "lighthalzen", "rachel", "veins", "niflheim", "manuk", "splendide", "brasilis", "moscovia", "amatsu", "kunlun", "louyang", "ayothaya", "jawaii", "gonryun", "hugel"}

    def _get_state(self, signals: dict, bot_id: str = "default") -> str:
        """Determine bot state from signals."""
        hp = signals.get("hp_ratio", 1.0)
        map_name = signals.get("map", "").lower()
        map_name = map_name.replace(".gat", "")
        is_town = map_name in _HUNT_TOWNS
        _prev_state = self._bot_state.get(bot_id, "UNKNOWN")
        _total_kills = signals.get("total_kills", 0) or 0
        _total_zeny = signals.get("zeny", 0) or 0
        # COLD_START: only on VERY FIRST spawn (never after death)
        _cold_fired = self._cold_start_fired.get(bot_id, False)
        if not _cold_fired and _prev_state == "UNKNOWN" and _total_kills == 0 and _total_zeny == 0:
            self._cold_start_fired[bot_id] = True
            return "COLD_START"
        # Stay in COLD_START until cold start sequence completes (step >= 4)
        if _prev_state == "COLD_START" and self._cold_start_step.get(bot_id, 0) < 4:
            return "COLD_START"
        # DEATH: if bot just died and respawned
        # Only trigger DEATH if bot actually died (HP was 0 or very low)
        # Not just because bot has 0 kills after selling starting gear
        if _cold_fired and _prev_state not in ("UNKNOWN", "COLD_START") and hp <= 0:
            return "DEATH"
        zeny = signals.get("zeny", 0) or 0
        # Weight: use actual game weight ratio from snapshot
        _inv_data = signals.get("inventory", {}) or {}
        weight = _inv_data.get("weight_pressure", 0) or 0.0
        base_level = signals.get("base_level", 1) or 1
        job_level = signals.get("job_level", 1) or 1
        job_name = signals.get("job_name", "novice").lower()
        stat_points = signals.get("stat_points", 0) or 0
        skill_points = signals.get("skill_points", 0) or 0
        in_party = signals.get("in_party", False)
        inventory = signals.get("inventory_items", []) or []

        # DEAD
        if hp <= 0:
            return "DEAD"

        # TOWN maps
        if is_town:
            # STUCK DETECTION: if in town > 120s with 0 kills, force hunting
            _town_start = self._town_entry_time.get(bot_id, 0)
            _now_t = __import__("time").time()
            if _town_start == 0:
                self._town_entry_time[bot_id] = _now_t
            _town_duration = _now_t - _town_start
            _kills_this_town = signals.get("kills_this_session", 0) or 0
            # Check if bot just warped (map changed in last 5s) - don't trigger TOWN_STUCK
            _last_map_change = signals.get("last_map_change", 0) or 0
            _just_warped = (_now_t - _last_map_change) < 5
            if _town_duration > 300 and _kills_this_town == 0 and not _just_warped:
                # Been in town too long with no kills - force hunting
                return "TOWN_STUCK"
            # Priority: SELL > WEAPON_BUY > BUY > JOB_CHANGE > STATS > SKILLS > PARTY > HUNT
            if weight > 0.05:
                return "SELL"
            if zeny > 0:
                _has_weapon = (signals.get("attack_power", 0) or 0) > 30
                if zeny >= 100 and not _has_weapon:
                    return "WEAPON_BUY"
                return "BUY"
            if base_level >= 10 and job_level >= 10 and job_name == "novice":
                return "JOB_CHANGE"
            if stat_points > 0:
                return "STATS"
            # If no stat points, skip STATS entirely to avoid wasted cycles
            if skill_points > 0:
                return "SKILLS"
            if not in_party:
                return "PARTY"
            return "TOWN_HUNT"

        # HUNTING
        return "HUNT"

    def _check_progress(self, signals: dict) -> bool:
        bot_id = signals.get("bot_id", "default")
        last = self._last_progress.get(bot_id, {})
        # Track kills separately - increment when monster dies
        _kills_sig = int(signals.get("last_monster_kill", 0) or 0)
        now = {
            "exp": signals.get("exp", 0) or signals.get("base_exp", 0) or 0,
            "zeny": signals.get("zeny", 0) or 0,
            "level": signals.get("base_level", 1) or 1,
            "kills": _kills_sig,
            "job_level": signals.get("job_level", 1) or 1,
            "items": len(signals.get("inventory_items", []) or []),
        }
        self._last_progress[bot_id] = now
        if not last:
            return True
        # Check multiple progress indicators
        for key in ("exp", "zeny", "level", "kills", "job_level", "items"):
            if now.get(key, 0) > last.get(key, 0):
                return True
        return False

    def _track_kills_per_hour(self, signals: dict, bot_id: str) -> float:
        """Track kills/hour and return the rate. Logs warning if 0 for 30+ minutes."""
        _now_t = __import__("time").time()
        _kills = int(signals.get("last_monster_kill", 0) or 0)
        _last_kills = self._last_kills_count.get(bot_id, 0)
        _last_kills_time = self._last_kills_time.get(bot_id, _now_t)
        _elapsed = _now_t - _last_kills_time
        if _elapsed < 60:
            return 0.0  # Not enough data yet
        _kills_gained = _kills - _last_kills
        _rate = (_kills_gained / _elapsed) * 3600  # kills/hour
        self._last_kills_count[bot_id] = _kills
        self._last_kills_time[bot_id] = _now_t
        # Log kills/hour every 5 minutes
        _last_log = self._last_kills_log.get(bot_id, 0)
        if _now_t - _last_log > 300:
            self._last_kills_log[bot_id] = _now_t
            logger.info(f"[kills_hour] {bot_id}: {_kills_gained} kills in {_elapsed:.0f}s = {_rate:.1f}/hour (total: {_kills})")
        # Escalate if 0 kills for 30+ minutes
        if _kills_gained == 0 and _elapsed > 1800:
            logger.warning(f"[kills_hour] {bot_id}: ZERO kills in {_elapsed:.0f}s! Escalating.")
        return _rate

    def set_domain_weights(self, weights: dict) -> None:
        pass

    def assess(self, signals: dict[str, Any], bot_id_override: str | None = None) -> HeuristicAssessment:
        try:
            return self._assess_impl(signals, bot_id_override)
        except Exception as e:
            logger.error(f"assess() crashed for {bot_id_override or 'unknown'}: {type(e).__name__}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            bot_id = bot_id_override or signals.get("bot_id", "default")
            return HeuristicAssessment(
                horizon=signals.get("horizon", "short_term"), actions=[], confidence=0.5,
                actionable=False, top_domain="survival", signals=dict(signals),
            )

    def _set_config_once(self, actions: list, bot_id: str, key: str, value: str, domain: str, reason: str, confidence: float = 0.95) -> None:
        """Emit a 'set' command only if the value has changed since last set.
        Prevents spamming 'set route_randomWalk 1' every cycle when it's already 1.
        On first cycle after restart, always sends (cache is empty)."""
        cache = self._last_config_set.setdefault(bot_id, {})
        last = cache.get(key)
        if last == value:
            return  # Already set to this value, skip
        cache[key] = value
        actions.append(HeuristicAction(
            kind="command", command=f"set {key} {value}",
            confidence=confidence, domain=domain,
            reason=reason,
        ))

    def _assess_impl(self, signals: dict[str, Any], bot_id_override: str | None = None) -> HeuristicAssessment:
        actions: list[HeuristicAction] = []
        bot_id = bot_id_override or signals.get("bot_id", "default")
        _now_t = __import__("time").time()
        state = self._get_state(signals, bot_id)
        prev_state = self._bot_state.get(bot_id, "UNKNOWN")

        if state != prev_state:
            self._bot_state[bot_id] = state
            self._state_since[bot_id] = __import__("time").time()
            logger.info(f"[heuristic] {bot_id} state: {prev_state} -> {state}")

        made_progress = self._check_progress(signals)
        state_duration = __import__("time").time() - self._state_since.get(bot_id, 0)
        is_stuck = not made_progress and state_duration > 120
        # Track kills/hour for monitoring
        self._track_kills_per_hour(signals, bot_id)

        hp = signals.get("hp_ratio", 1.0)
        map_name = signals.get("map", "").lower()
        zeny = signals.get("zeny", 0) or 0
        weight = signals.get("weight_ratio", 0) or 0
        base_level = signals.get("base_level", 1) or 1
        job_level = signals.get("job_level", 1) or 1
        job_name = signals.get("job_name", "novice").lower()
        self._adaptive._last_job = job_name  # Set for profit calculation delegate
        stat_points = signals.get("stat_points", 0) or 0
        skill_points = signals.get("skill_points", 0) or 0
        in_party = signals.get("in_party", False)
        inventory = signals.get("inventory_items", []) or []
        skills = signals.get("skills", []) or []
        bot_name = signals.get("bot_name", bot_id)
        horizon = signals.get("horizon", "short_term")
        _leader_map = ""
        # _profile_to_char is accessed via self._profile_to_char throughout

        # ── COLD START SEQUENCE: task-completion-triggered, not time-based ──
        # The OnboardingService exists but is disconnected from the heuristic.
        # This sequence fires each step when the PREVIOUS step is confirmed via signals.
        # Step 1: Buy Knife (item 1201) — confirmed when weapon in inventory_items
        # Step 2: Buy Red Potions (item 501) — confirmed when potions in inventory_items
        # Step 3: Set lockMap to prt_fild05 — confirmed when map changes
        # Step 4: Hunt — bot is on hunting map with weapon + potions
        # Define map check vars before use (config audit section defines them later)
        _cs_map = signals.get("map", "") or ""
        _cs_in_town = any(x in _cs_map for x in ["prontera", "morocc", "geffen", "payon", "aldebaran", "alberta", "izlude"])
        _cs_in_hunting = any(x in _cs_map for x in ["prt_fild", "pay_fild", "mjolnir", "gef_fild", "ra_fild", "moc_fild", "cmd_fild"])
        _cold_start_step = self._cold_start_step.get(bot_id, 0)
        _has_weapon = any(
            "knife" in str(item).lower() or "sword" in str(item).lower() or "mace" in str(item).lower() or
            "bow" in str(item).lower() or "dagger" in str(item).lower() or "rod" in str(item).lower()
            for item in inventory
        )
        _has_potions = any(
            "potion" in str(item).lower() or "red" in str(item).lower() or "orange" in str(item).lower() or "white" in str(item).lower()
            for item in inventory
        )
        if _cold_start_step == 0:
            # Step 0: Check if we need cold start at all
            if _has_weapon and _has_potions:
                # Already equipped — skip cold start
                self._cold_start_step[bot_id] = 4
            else:
                # Need cold start — ensure we're in Prontera first
                self._cold_start_step[bot_id] = 1
                _cold_start_step = 1  # Sync local var
                # Disable attack + go to portal exit. Re-enable attack after arrival.
                if not _cs_in_town:
                    actions.append(HeuristicAction(
                        kind="command", command="set attackAuto 0",
                        confidence=0.99, domain="economy",
                        reason="Cold start - disable attack to allow portal move",
                    ))
                    actions.append(HeuristicAction(
                        kind="command", command="move 367 205",
                        confidence=0.99, domain="economy",
                        reason="Cold start - go to Prontera portal to reach town",
                    ))
        if _cold_start_step == 1:
            # Step 1: Buy Knife (item 1201) if no weapon
            # Only queue actions when bot is in town (Prontera)
            if not _has_weapon:
                if zeny >= 50:
                    actions.append(HeuristicAction(
                        kind="command", command="buy 1201 1",
                        confidence=0.99, domain="economy",
                        reason="Cold start - buy Knife (no weapon detected)",
                    ))
                    actions.append(HeuristicAction(
                        kind="command", command="equip 1201",
                        confidence=0.99, domain="economy",
                        reason="Cold start - equip Knife after purchase",
                    ))
                else:
                    # 0 zeny — can't buy anything. Need to farm 50 zeny first.
                    # Go to prt_fild01 (safe map with Porings) and farm with fists.
                    # Only queue from Prontera — 'move prt_fild01' from Prontera
                    # is a direct route via portal 22 203.
                    if _cs_in_town:
                        actions.append(HeuristicAction(
                            kind="command", command="set attackAuto 3",
                            confidence=0.99, domain="economy",
                            reason="Cold start - re-enable attack for farming",
                        ))
                        actions.append(HeuristicAction(
                            kind="command", command="move prt_fild01",
                            confidence=0.99, domain="economy",
                            reason=f"Cold start - farm {50 - zeny}z on prt_fild01 (0 zeny, no weapon)",
                        ))
            else:
                # Weapon confirmed — move to step 2
                self._cold_start_step[bot_id] = 2
                _cold_start_step = 2
                logger.info(f"[cold_start] {bot_id}: weapon confirmed, step 1 -> 2")
        if _cold_start_step == 2:
            # Step 2: Buy Red Potions (item 501) if no potions
            if not _has_potions:
                if zeny >= 50:
                    _potion_qty = min(int(zeny / 50), 10)
                    if _potion_qty > 0:
                        actions.append(HeuristicAction(
                            kind="command", command=f"buy 501 {_potion_qty}",
                            confidence=0.99, domain="economy",
                            reason=f"Cold start - buy {_potion_qty} Red Potions",
                        ))
            else:
                # Potions confirmed — move to step 3
                self._cold_start_step[bot_id] = 3
                logger.info(f"[cold_start] {bot_id}: potions confirmed, step 2 -> 3")
        if _cold_start_step == 3:
            # Step 3: Return to hunting map with weapon and potions
            # 'move prt_fild05' rewrite handles lockMap + routing.
            if not _cs_in_hunting or _cs_map == "prt_fild01":
                actions.append(HeuristicAction(
                    kind="command", command="move prt_fild05",
                    confidence=0.99, domain="hunting",
                    reason="Cold start - return to hunting map with weapon and potions",
                ))
            elif _cs_in_hunting:
                # On hunting map with weapon + potions — cold start complete
                self._cold_start_step[bot_id] = 4
                logger.info(f"[cold_start] {bot_id}: on hunting map, cold start complete")
        # ── ZERO POTIONS ON HUNTING MAP: force return to town ──
        # Runs every cycle for ALL bots, not just COLD_START or hunting maps
        # Fixes: attackMaxDistance, attackDistance, attackAuto, avoidList settings
        # MUST run before state-specific returns to ensure config is always applied
        _audit_map = signals.get("map", "") or ""
        _audit_is_hunting = any(x in _audit_map for x in ["prt_fild", "pay_fild", "mjolnir", "gef_fild", "ra_fild", "moc_fild", "cmd_fild"])
        _audit_is_town = any(x in _audit_map for x in ["prontera", "morocc", "geffen", "payon", "aldebaran", "alberta", "izlude"])
        if _audit_is_hunting:
            # Ensure attack config is optimal for Novice-level combat
            self._set_config_once(actions, bot_id, "attackMaxDistance", "30", "hunting",
                "Config audit - increase chase distance to 30 for Novice attack range")
            self._set_config_once(actions, bot_id, "attackDistance", "5", "hunting",
                "Config audit - attack from 5 cells away")
            self._set_config_once(actions, bot_id, "attackAuto", "3", "hunting",
                "Config audit - enable aggressive auto-attack")
            self._set_config_once(actions, bot_id, "attackAuto_startOnSight", "1", "hunting",
                "Config audit - attack monsters as soon as they appear")
            self._set_config_once(actions, bot_id, "attackAuto_unstuck", "1", "hunting",
                "Config audit - don't give up mid-fight")
            # CRITICAL: Disable avoidList on hunting maps
            # The avoid system fires BEFORE attackAuto, causing bots to run away from monsters
            # instead of attacking them. This is the root cause of zero kills.
            # OpenKore uses 'avoidList' not 'avoidOutOfSight' for monster avoidance.
            # avoidList is a space-separated list of monster names. Setting to "0" adds "0"
            # to the list. Must set to empty string to truly disable it.
            self._set_config_once(actions, bot_id, "avoidList", "", "hunting",
                "Config audit - disable avoid system on hunting maps (prevents running from monsters)")
            self._set_config_once(actions, bot_id, "avoidList_inLockOnly", "", "hunting",
                "Config audit - disable avoid system in lockMap (prevents running from monsters)")
            # ── ZERO POTIONS ON HUNTING MAP: force return to town ──
            # If bot is on a hunting map with 0 potions, force return to town to buy potions.
            # This was previously handled by the bridge's Reflex #1, but that was stripped.
            # NOTE: inventory data is stored as 'inventory_items' in signals (from pdca_loop.py)
            # NOTE: No rate limit on emergency return — 0 potions on hunting map = return NOW
            # The bridge's auto-stand handles sitting bots before executing 'move prontera'.
            # The portal exit check (in bridge) breaks the loop if bot arrives and turns back.
            _audit_items = signals.get("inventory_items", []) or []
            _audit_has_potions = any(
                "potion" in str(item).lower() or "red" in str(item).lower() or "orange" in str(item).lower() or "white" in str(item).lower()
                for item in _audit_items
            )
            if not _audit_has_potions:
                # Only queue move prontera — bridge auto-stand handles sitting bots
                # Don't queue 'stand' or 'ai auto' as separate actions (bridge does both)
                # Rate limit: 30s between sends to let route to Prontera complete
                _audit_now = __import__("time").time()
                _audit_last_return = self._last_return_to_town.get(bot_id, 0)
                if _audit_now - _audit_last_return > 30:
                    self._last_return_to_town[bot_id] = _audit_now
                    actions.append(HeuristicAction(
                        kind="command", command="move prontera",
                        confidence=0.99, domain="economy",
                        reason="Zero potions on hunting map - return to town to buy potions",
                    ))
            # Anti-detection: randomize movement and command pacing per bot
            _audit_seed = hash(bot_id) & 0xFFFFFFFF
            _audit_rand = __import__("random").Random(_audit_seed)
            _audit_route_step = _audit_rand.randint(2, 5)
            _audit_route_walk = _audit_rand.choice(["1", "2"])
            _audit_attackAuto_pause = _audit_rand.randint(0, 2)
            self._set_config_once(actions, bot_id, "route_randomWalk", "1", "hunting",
                "Config audit - enable random walk for human-like movement")
            self._set_config_once(actions, bot_id, "route_randomWalk_inLockOnly", "1", "hunting",
                "Config audit - random walk only in lockMap")
            self._set_config_once(actions, bot_id, "route_randomWalk_maxRouteTime", str(_audit_route_step), "hunting",
                f"Config audit - random walk step {_audit_route_step} (per-bot variation)")
            self._set_config_once(actions, bot_id, "route_randomWalk_maxWalkTime", _audit_route_walk, "hunting",
                f"Config audit - random walk time {_audit_route_walk}s (per-bot variation)")
            self._set_config_once(actions, bot_id, "attackAuto_pause", str(_audit_attackAuto_pause), "hunting",
                f"Config audit - attack pause {_audit_attackAuto_pause}s (per-bot variation)")
        elif _audit_is_town:
            # In town — ensure bot is in auto mode and ready to move
            self._set_config_once(actions, bot_id, "attackAuto", "3", "hunting",
                "Config audit (town) - enable aggressive auto-attack for when we return to hunt")
            # Buy potions immediately if in town with 0 potions (no 30s wait)
            _audit_now = __import__("time").time()
            _audit_town_entry = self._town_entry_time.get(bot_id, _audit_now)
            _audit_town_time = _audit_now - _audit_town_entry
            # NOTE: inventory data is stored as 'inventory_items' in signals (from pdca_loop.py)
            _audit_items = signals.get("inventory_items", []) or []
            _audit_has_potions = any(
                "potion" in str(item).lower() or "red" in str(item).lower() or "orange" in str(item).lower() or "white" in str(item).lower()
                for item in _audit_items
            )
            if not _audit_has_potions:
                _audit_zeny = signals.get("zeny", 0) or 0
                if _audit_zeny >= 50:
                    _audit_potion_qty = min(int(_audit_zeny / 50), 10)
                    if _audit_potion_qty > 0:
                        actions.append(HeuristicAction(
                            kind="command", command=f"buy 501 {_audit_potion_qty}",
                            confidence=0.99, domain="economy",
                            reason=f"Town - buy {_audit_potion_qty} Red Potions (0 potions in inventory)",
                        ))
                    _audit_has_weapon = any(
                        "knife" in str(item).lower() or "sword" in str(item).lower() or "mace" in str(item).lower() or
                        "bow" in str(item).lower() or "dagger" in str(item).lower() or "rod" in str(item).lower()
                        for item in _audit_items
                    )
                    if not _audit_has_weapon and _audit_zeny >= 50:
                        actions.append(HeuristicAction(
                            kind="command", command="buy 1201 1",
                            confidence=0.99, domain="economy",
                            reason="Town stuck - buy Knife (no weapon detected)",
                        ))
                        actions.append(HeuristicAction(
                            kind="command", command="equip 1201",
                            confidence=0.99, domain="economy",
                            reason="Town stuck - equip Knife after purchase",
                        ))
                    # DO NOT queue move 367 205 here — bot must wait until potions confirmed
                    # Only queue portal exit when potions are actually in inventory
            if _audit_has_potions and _audit_zeny >= 50:
                # Have potions — return to hunt
                # ROUTE LOOP PREVENTION: only send move if bot isn't already at portal
                _audit_pos = signals.get("position", {}) or {}
                _audit_px = _audit_pos.get("x", 0) or 0
                _audit_py = _audit_pos.get("y", 0) or 0
                _audit_dist = ((_audit_px - 367)**2 + (_audit_py - 205)**2)**0.5
                if _audit_dist < 5:
                    # Already at portal - just enable auto mode
                    actions.append(HeuristicAction(
                        kind="command", command="ai auto",
                        confidence=0.99, domain="hunting",
                        reason="Already at portal - enable auto-attack",
                    ))
                else:
                    actions.append(HeuristicAction(
                        kind="command", command="move 367 205",
                        confidence=0.99, domain="hunting",
                        reason=f"Town stuck - return to hunt after {_audit_town_time:.0f}s in town",
                    ))
                    actions.append(HeuristicAction(
                        kind="command", command="ai auto",
                        confidence=0.95, domain="hunting",
                        reason="Enable auto-attack after returning to hunt",
                    ))

        # ── SITTING ON HUNTING MAP DETECTOR: force stand when sitting with 0 potions and HP > 50% ──
        # The bridge's stand-up reflex was stripped. The sidecar must handle this.
        # If bot is sitting on a hunting map with 0 potions and HP > 50%, force stand.
        # Also force stand if bot has been sitting for > 30s (stuck sitting).
        _audit_is_sitting = signals.get("is_sitting", False)
        logger.info(f"[sit_detector_debug] {bot_id}: hunting={_audit_is_hunting} sitting={_audit_is_sitting} map={_audit_map} hp={signals.get('hp_ratio', 1.0):.2f}")
        if _audit_is_hunting and _audit_is_sitting:
            _audit_now = __import__("time").time()
            _audit_sit_start = self._sit_start_time.get(bot_id, _audit_now)
            if not self._sit_start_time.get(bot_id):
                self._sit_start_time[bot_id] = _audit_now
            _audit_sit_duration = _audit_now - _audit_sit_start
            _audit_hp = signals.get("hp_ratio", 1.0)
            # Force stand if HP > 50% OR sitting > 30s
            if _audit_hp > 0.50 or _audit_sit_duration > 30:
                logger.info(f"[sit_detector] {bot_id}: sitting on {_audit_map} for {_audit_sit_duration:.0f}s, HP={_audit_hp:.0%}, forcing stand")
                actions.append(HeuristicAction(
                    kind="command", command="stand",
                    confidence=0.99, domain="survival",
                    reason=f"Sitting on hunting map for {_audit_sit_duration:.0f}s with HP={_audit_hp:.0%} - forcing stand",
                ))
                actions.append(HeuristicAction(
                    kind="command", command="ai auto",
                    confidence=0.99, domain="survival",
                    reason="Re-enable auto-attack after forced stand",
                ))
                self._sit_start_time[bot_id] = 0  # Reset
        else:
            # Reset sit timer when not sitting
            self._sit_start_time[bot_id] = 0

        # ── STATE: DEAD ──
        if state == "DEAD":
            actions.append(HeuristicAction(
                kind="command", command="stand",
                confidence=0.95, domain="survival",
                reason="Stand up after respawn",
            ))
            actions.append(HeuristicAction(
                kind="command", command="ai auto",
                confidence=0.95, domain="survival",
                reason="Just respawned - re-enable AI",
            ))
            total_confidence = 0.95
            top_domain = "survival"
            assessment = HeuristicAssessment(
                horizon=horizon, actions=actions, confidence=total_confidence,
                actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
            )
            self._last_assessment[bot_id] = assessment
            return assessment

        # ── DIRECT PARTY CHECK: Always check party before any state logic
        _party_in = signals.get("in_party", False)
        _party_members = signals.get("party_members", []) or []
        _all_bots = signals.get("all_bots", []) or []
        # Death/respawn flicker guard: cache party state for 30s to survive snapshot loss
        _now_t = __import__("time").time()
        _last_seen_party = self._last_party_seen.get(bot_id, 0)
        if (not _party_in or not _all_bots) and _last_seen_party > 0 and _now_t - _last_seen_party < 120:
            _party_in = True
            _party_members = self._last_party_members.get(bot_id, [])
            _all_bots = self._all_bots_cache.get(bot_id, [])
        if _party_in and _all_bots:
            self._last_party_seen[bot_id] = _now_t
            self._last_party_members[bot_id] = _party_members
            self._all_bots_cache[bot_id] = _all_bots
        _bot_profile = bot_id.split(":")[-1].split("/")[-1] if ":" in bot_id else bot_id
        _sorted_bots = sorted(_all_bots)
        _is_leader = len(_sorted_bots) > 0 and _bot_profile == _sorted_bots[0]
        # Compare by COUNT not by name (party_members has char names, all_bots has profile names)
        # party_members does NOT include the leader (OpenKore quirk)
        _expected_count = len(_all_bots)
        _actual_count = len(_party_members)
        _party_incomplete = (_actual_count + 1) < _expected_count  # +1 for leader (not in party_members list)
        logger.info("[party_check] " + str(bot_id) + " in_party=" + str(_party_in) + " members=" + str(_party_members) + " all_bots=" + str(_all_bots) + " expected=" + str(_expected_count) + " actual=" + str(_actual_count) + " incomplete=" + str(_party_incomplete))

        # Leader: check if party is incomplete AND level >= 40 (solo before 40 is faster)
        if _is_leader and _party_incomplete and state != "COLD_START" and state != "DEAD" and base_level >= 40:
            _now = __import__("time").time()
            _last_party = self._last_party_attempt.get(bot_id, 0)
            if _now - _last_party > 15:
                self._last_party_attempt[bot_id] = _now
                _ts = int(__import__("time").time())
                # If already in party with some members, just request missing ones
                # (don't leave+recreate - that destroys existing party)
                if _party_in and len(_party_members) > 0:
                    # Already have a party - just request missing members
                    # Build profile_to_char dynamically from all_bots
                    # Each bot's char name is read from its snapshot
                    for _other_bot in _all_bots:
                        if _other_bot != _bot_profile:
                            _char_name = _other_bot  # Fallback: use profile name
                            # Only request if not already in party
                            _already_in = any(_char_name.lower() in m.lower() for m in _party_members)
                            if not _already_in:
                                actions.append(HeuristicAction(
                                    kind="command", command=("party request " + str(_char_name)),
                                    confidence=0.95, domain="social",
                                    reason="Direct party check - request " + str(_other_bot) + " (" + str(_char_name) + ")",
                                ))
                            else:
                                # Even if already_in check says True, still try - party_members might be stale
                                # Only skip if we have 3+ members confirmed
                                if _actual_count < 3:
                                    actions.append(HeuristicAction(
                                        kind="command", command=("party request " + str(_char_name)),
                                        confidence=0.80, domain="social",
                                        reason="Direct party check - retry " + str(_other_bot) + " (" + str(_char_name) + ") - stale check",
                                    ))
                else:
                    # Not in party - move to town and create new one
                    actions.append(HeuristicAction(
                        kind="command", command="move prontera",
                        confidence=0.99, domain="social",
                        reason="Direct party check - move to town for party formation",
                    ))
                    actions.append(HeuristicAction(
                        kind="command", command=("party create AI" + str(_ts)),
                        confidence=0.95, domain="social",
                        reason="Direct party check - leader creates party",
                    ))
                    # Request ALL other bots using character names
                    for _other_bot in _all_bots:
                        if _other_bot != _bot_profile:
                            _char_name = _other_bot  # Fallback: use profile name
                            actions.append(HeuristicAction(
                                kind="command", command=("party request " + str(_char_name)),
                                confidence=0.95, domain="social",
                                reason="Direct party check - request " + str(_other_bot) + " (" + str(_char_name) + ")",
                            ))
                    actions.append(HeuristicAction(
                        kind="command", command="party share exp",
                        confidence=0.90, domain="social",
                        reason="Share experience in party",
                    ))
                actions.append(HeuristicAction(
                    kind="command", command="ai auto",
                    confidence=0.95, domain="hunting",
                    reason="Continue after party attempt",
                ))
                total_confidence = 0.95
                top_domain = "social"
                assessment = HeuristicAssessment(
                    horizon=horizon, actions=actions, confidence=total_confidence,
                    actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
                )
                self._last_assessment[bot_id] = assessment
                return assessment

        # Joiners: if not in party or in wrong party, leave stale party, set partyAuto, move to town
        # "Wrong party" = in a party that doesn't contain the leader's char name AND has only 1 member (self-only)
        _leader_char = (getattr(self, '_profile_to_char', {}) or {}).get(_sorted_bots[0], _sorted_bots[0]) if _sorted_bots else ""
        _joiner_in_wrong_party = _party_in and not _is_leader and len(_party_members) == 1 and _leader_char and _leader_char not in _party_members
        # Also: if joiner is on a town map while leader is on hunting map, force move
        _town_maps = ("prontera", "morocc", "geffen", "payon", "alberta", "izlude", "aldebaran", "comodo", "umbala", "niflheim", "louyang", "einbroch", "lighthalzen", "rachel", "veins", "juno", "yuno")
        # _leader_map already initialized at function start
        _joiner_stuck_in_town = not _is_leader and map_name and map_name in _town_maps and _leader_map and _leader_map not in _town_maps
        logger.info("[joiner_check] " + str(bot_id) + " party_in=" + str(_party_in) + " joiner_wrong=" + str(_joiner_in_wrong_party) + " stuck_town=" + str(_joiner_stuck_in_town) + " is_leader=" + str(_is_leader) + " state=" + str(state) + " members=" + str(_party_members) + " all_bots=" + str(_all_bots) + " leader_char=" + str(_leader_char) + " map=" + str(map_name) + " leader_map=" + str(_leader_map))
        # Only act if we have all_bots data - empty all_bots means flicker/no data
        # Only party at level 40+ (solo before 40 is faster)
        if (not _party_in or _joiner_in_wrong_party or _joiner_stuck_in_town) and not _is_leader and state != "COLD_START" and state != "DEAD" and _all_bots and base_level >= 40:
            if _party_in:
                actions.append(HeuristicAction(
                    kind="command", command="party leave",
                    confidence=0.99, domain="social",
                    reason="Direct party check - leave stale party",
                ))
            actions.append(HeuristicAction(
                kind="command", command="set partyAuto 2",
                confidence=0.99, domain="social",
                reason="Direct party check - set auto-accept",
            ))
            actions.append(HeuristicAction(
                kind="command", command="move prontera",
                confidence=0.95, domain="social",
                reason="Direct party check - move to town for party invite",
            ))
            actions.append(HeuristicAction(
                kind="command", command="ai auto",
                confidence=0.95, domain="hunting",
                reason="Continue after party attempt",
            ))
            total_confidence = 0.95
            top_domain = "social"
            assessment = HeuristicAssessment(
                horizon=horizon, actions=actions, confidence=total_confidence,
                actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
            )
            self._last_assessment[bot_id] = assessment
            return assessment

        # ── PARTY LEAVE: if in party but level < 40, leave party (solo is faster) ──
        # Force leave regardless of cached party state — the bridge may have stale data
        # Uses separate _last_party_leave dict (not _last_party_attempt) to prevent
        # cooldown reset from party creation code.
        _audit_base_level = signals.get("base_level", 1) or 1
        if _audit_base_level < 40 and state not in ("COLD_START", "DEAD"):
            _audit_now = __import__("time").time()
            _audit_last_party_leave = self._last_party_leave.get(bot_id, 0)
            if _audit_now - _audit_last_party_leave > 30:
                self._last_party_leave[bot_id] = _audit_now
                actions.append(HeuristicAction(
                    kind="command", command="party leave",
                    confidence=0.99, domain="social",
                    reason=f"Level {_audit_base_level} < 40 - force leave party (solo is faster)",
                ))

        # ── FORCE RETURN TO TOWN: 0 potions + 0 weapon + 5+ min on same map ──
        # Detects bots stuck on hunting map with no supplies
        _audit_inv = signals.get("inventory", {}) or {}
        _audit_items = _audit_inv.get("items", []) or []
        _audit_has_potions = any(
            "potion" in str(item).lower() or "red" in str(item).lower() or "orange" in str(item).lower() or "white" in str(item).lower()
            for item in _audit_items
        )
        _audit_has_weapon = any(
            "knife" in str(item).lower() or "sword" in str(item).lower() or "mace" in str(item).lower() or
            "bow" in str(item).lower() or "dagger" in str(item).lower() or "rod" in str(item).lower()
            for item in _audit_items
        )
        _audit_zeny = signals.get("zeny", 0) or 0
        if _audit_is_hunting and not _audit_has_potions and not _audit_has_weapon:
            # Bot has no potions and no weapon on hunting map — force return to town
            # Fires regardless of zeny (0-zeny bots need to go back and figure it out)
            _audit_now = __import__("time").time()
            _audit_last_force = self._last_force_return.get(bot_id, 0)
            if _audit_now - _audit_last_force > 120:  # 2 min cooldown
                self._last_force_return[bot_id] = _audit_now
                actions.append(HeuristicAction(
                    kind="command", command="move 367 205",
                    confidence=0.99, domain="emergency",
                    reason=f"Force return - no potions, no weapon, {_audit_zeny}z available",
                ))

        # ── DISABLE OPENCORE'S BUILT-IN POTION USE when 0 potions ──
        # OpenKore's useSelf_item system fires independently of the bridge.
        # When the bot has 0 potions, the bridge's survival mode (Reflex #22)
        # suppresses heal reflexes. This is handled at the bridge level.
        # Note: useSelf_item blocks use array syntax, not simple config keys,
        # so 'set useSelf_item_Red_Potion_timeout 300' won't work here.

        # ── JOB CHANGE: Novice with job_level >= 10 should job change ──
        # Force return to town and route to job change NPC
        _audit_job_name = signals.get("job_name", "") or ""
        _audit_job_level = signals.get("job_level", 0) or 0
        if _audit_job_name == "novice" and _audit_job_level >= 10:
            _audit_now = __import__("time").time()
            _audit_last_job_change = self._last_job_change_attempt.get(bot_id, 0)
            if _audit_now - _audit_last_job_change > 60:  # 1 min cooldown
                self._last_job_change_attempt[bot_id] = _audit_now
                # Route to job change NPC in Prontera
                _audit_job_npc = JOB_CHANGE_NPCS.get("novice", ("prontera", 160, 191))
                _audit_job_map, _audit_job_x, _audit_job_y = _audit_job_npc
                if _audit_map != _audit_job_map:
                    # Not in Prontera — return to town first
                    actions.append(HeuristicAction(
                        kind="command", command="move 367 205",
                        confidence=0.99, domain="emergency",
                        reason=f"Job change - Novice job level {_audit_job_level} >= 10, return to Prontera",
                    ))
                else:
                    # In Prontera — walk to job change NPC
                    actions.append(HeuristicAction(
                        kind="command", command=f"move {_audit_job_x} {_audit_job_y}",
                        confidence=0.99, domain="emergency",
                        reason=f"Job change - Novice job level {_audit_job_level} >= 10, walk to job NPC",
                    ))

        # ── STATE: COLD_START (fresh spawn - go hunt immediately) ──
        if state == "COLD_START":
            actions.append(HeuristicAction(
                kind="command", command="stand",
                confidence=0.99, domain="emergency",
                reason="Cold start - stand up",
            ))
            # Set ai auto FIRST so the bot starts moving immediately
            actions.append(HeuristicAction(
                kind="command", command="ai auto",
                confidence=0.99, domain="hunting",
                reason="Cold start - enable auto-attack before moving",
            ))
            # Combat config (deduped)
            self._set_config_once(actions, bot_id, "attackDistance", "7", "hunting",
                "Cold start - set attack distance")
            self._set_config_once(actions, bot_id, "attackMaxDistance", "20", "hunting",
                "Cold start - set chase distance")
            self._set_config_once(actions, bot_id, "attackAuto", "3", "hunting",
                "Enable aggressive auto-attack")
            self._set_config_once(actions, bot_id, "attackAuto_startOnSight", "1", "hunting",
                "Attack monsters as soon as they appear")
            self._set_config_once(actions, bot_id, "attackAuto_unstuck", "1", "hunting",
                "Cold start - don't give up mid-fight")
            # COLD START: Go to safe field first (prt_fild05 for level 1-10)
            # Then progress to dungeon at level 10+
            _cs_hunt_map = "prt_fild05"
            _cs_portal_coords = "22 203"  # Portal from Prontera (156, 164) -> prt_fild05 (22, 203)
            # Set route_randomWalk and lockMap
            self._set_config_once(actions, bot_id, "route_randomWalk", "1", "hunting",
                "Cold start - route_randomWalk 1 (walk within bounds)")
            self._set_config_once(actions, bot_id, "lockMap_randX", "100", "hunting",
                "Cold start - random walk radius X")
            self._set_config_once(actions, bot_id, "lockMap_randY", "100", "hunting",
                "Cold start - random walk radius Y")
            self._last_lockmap[bot_id] = _cs_hunt_map
            self._set_config_once(actions, bot_id, "lockMap", _cs_hunt_map, "hunting",
                "Cold start - set hunting map lock")
            # Economy: buy potions FIRST (before moving to hunting map)
            _cs_zeny = signals.get("zeny", 0) or 0
            if _cs_zeny >= 500:
                # Buy 10 Red Potions (item 501) for survival
                actions.append(HeuristicAction(
                    kind="command", command="buy 501 10",
                    confidence=0.99, domain="economy",
                    reason="Cold start - buy 10 Red Potions for survival",
                ))
            elif _cs_zeny >= 50:
                # Buy at least 1 potion if we can afford it
                _cs_potion_qty = int(_cs_zeny / 50)
                _cs_potion_qty = min(_cs_potion_qty, 10)
                if _cs_potion_qty > 0:
                    actions.append(HeuristicAction(
                        kind="command", command=f"buy 501 {_cs_potion_qty}",
                        confidence=0.99, domain="economy",
                        reason=f"Cold start - buy {_cs_potion_qty} Red Potions",
                    ))
            # Buy arrows if enough zeny
            if _cs_zeny >= 200:
                actions.append(HeuristicAction(
                    kind="command", command="buy 1750 200",
                    confidence=0.99, domain="economy",
                    reason="Buy 200 arrows (harmless for non-archers, critical for archers)",
                ))
            # Buy weapon if any zeny available
            _cs_job = signals.get("job_name", "novice") or "novice"
            if _cs_zeny >= 50:
                # Use rAthena-corrected weapon ID from equipment_progression
                _cs_weapon_id = "1201"  # Knife (ATK 17, 3 slots) — cheapest weapon, all classes can equip
                actions.append(HeuristicAction(
                    kind="command", command=f"buy {_cs_weapon_id} 1",
                    confidence=0.99, domain="economy",
                    reason=f"Cold start - buy weapon {_cs_weapon_id} for {_cs_job}",
                ))
                # Equip the weapon after buying — bare fists do 1-5 damage
                actions.append(HeuristicAction(
                    kind="command", command="equip 1201",
                    confidence=0.99, domain="economy",
                    reason="Cold start - equip weapon after purchase",
                ))
            # Attack config: increase chase distance, disable avoid system
            self._set_config_once(actions, bot_id, "attackMaxDistance", "30", "hunting",
                "Cold start - increase chase distance to 30 for Novice attack range")
            self._set_config_once(actions, bot_id, "attackDistance", "5", "hunting",
                "Cold start - attack from 5 cells away")
            self._set_config_once(actions, bot_id, "attackAuto", "3", "hunting",
                "Enable aggressive auto-attack")
            self._set_config_once(actions, bot_id, "attackAuto_startOnSight", "1", "hunting",
                "Attack monsters as soon as they appear")
            self._set_config_once(actions, bot_id, "attackAuto_unstuck", "1", "hunting",
                "Cold start - don't give up mid-fight")
            # Teleport config
            self._set_config_once(actions, bot_id, "teleportAuto_minAggressives", "8", "hunting",
                "Only teleport at 8+ mobs")
            self._set_config_once(actions, bot_id, "teleportAuto_hp", "0", "hunting",
                "Never teleport due to HP - use sit instead")
            self._set_config_once(actions, bot_id, "teleportAuto_deadly", "0", "hunting",
                "Disable deadly teleport")
            # Only send move if on a different map
            if map_name != _cs_hunt_map:
                actions.append(HeuristicAction(
                    kind="command", command=f"move {_cs_hunt_map}",
                    confidence=0.99, domain="hunting",
                    reason=f"Cold start - move to {_cs_hunt_map}",
                ))
            # Party creation for leader — only at level 40+ (solo before 40 is faster)
            _cs_bot_profile = bot_id.split(":")[-1].split("/")[-1] if ":" in bot_id else bot_id
            _cs_all_bots = signals.get("all_bots", []) or list(self._bot_roles.keys()) if hasattr(self, '_bot_roles') else []
            _cs_sorted = sorted(_cs_all_bots)
            _cs_is_leader = len(_cs_sorted) > 0 and _cs_bot_profile == _cs_sorted[0]
            _cs_base_level = signals.get("base_level", 1) or 1
            if _cs_is_leader and _cs_base_level >= 40:
                actions.append(HeuristicAction(
                    kind="command", command=f"party create AI{int(_now_t)}",
                    confidence=0.99, domain="social",
                    reason="Cold start - leader creates party with unique name",
                ))
                actions.append(HeuristicAction(
                    kind="command", command="party share exp",
                    confidence=0.95, domain="social",
                    reason="Share experience in party",
                ))
            total_confidence = 0.99
            top_domain = "emergency"
            assessment = HeuristicAssessment(
                horizon=horizon, actions=actions, confidence=total_confidence,
                actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
            )
            self._last_assessment[bot_id] = assessment
            return assessment

        # ── STATE: DEATH (respawned - sell items, buy potions, return to hunt) ──
        if state == "DEATH":
            # Try to sell items (NPC handles empty inventory gracefully)
            _has_items = (signals.get("inventory_items", 0) or 0) > 0
            _total_kills = signals.get("kills", 0) or 0
            # Sell if has items
            if _has_items:
                _sell_npc = self._get_npc("sell", map_name)
                if _sell_npc:
                    _sell_cmd = f"talknpc {_sell_npc['x']} {_sell_npc['y']} {' '.join(eval(_sell_npc['steps']))}"
                else:
                    _sell_cmd = "talknpc 147 175 c r1 n"
                actions.append(HeuristicAction(
                    kind="command", command=_sell_cmd,
                    confidence=0.99, domain="economy",
                    reason="Death recovery - sell items",
                ))
            # Buy potions after selling (every death, no exceptions)
            _death_zeny = signals.get("zeny", 0) or 0
            if _death_zeny >= 500:
                actions.append(HeuristicAction(
                    kind="command", command="buy 501 10",
                    confidence=0.99, domain="economy",
                    reason="Death recovery - buy 10 Red Potions",
                ))
            elif _death_zeny >= 50:
                _death_potion_qty = min(int(_death_zeny / 50), 10)
                if _death_potion_qty > 0:
                    actions.append(HeuristicAction(
                        kind="command", command=f"buy 501 {_death_potion_qty}",
                        confidence=0.99, domain="economy",
                        reason=f"Death recovery - buy {_death_potion_qty} Red Potions",
                    ))
            # Buy weapon if we don't have one (check inventory)
            _death_job = signals.get("job_name", "novice") or "novice"
            _death_inv = signals.get("inventory", {}) or {}
            _death_has_weapon = any(
                "knife" in str(item).lower() or "sword" in str(item).lower() or
                "mace" in str(item).lower() or "bow" in str(item).lower() or
                "dagger" in str(item).lower() or "rod" in str(item).lower()
                for item in (_death_inv.get("items", []) or [])
            )
            if not _death_has_weapon and _death_zeny >= 50:
                _death_weapon_id = "1201"  # Knife (ATK 17) — cheapest, all classes
                actions.append(HeuristicAction(
                    kind="command", command=f"buy {_death_weapon_id} 1",
                    confidence=0.99, domain="economy",
                    reason=f"Death recovery - buy weapon {_death_weapon_id} for {_death_job}",
                ))
            # Return to hunt via portal after 15s in town
            _town_time = __import__("time").time() - self._town_entry_time.get(bot_id, __import__("time").time())
            if _town_time > 15:
                self._set_config_once(actions, bot_id, "lockMap", "prt_fild05", "hunting",
                    "Lock to hunting map")
                self._set_config_once(actions, bot_id, "lockMap_randX", "30", "hunting",
                    "Random walk radius X")
                self._set_config_once(actions, bot_id, "lockMap_randY", "30", "hunting",
                    "Random walk radius Y")
                _portal = self._get_npc("portal_to_hunt", map_name)
                if _portal:
                    _portal_cmd = f"move {_portal['x']} {_portal['y']}"
                else:
                    _portal_cmd = "move 22 203"
                actions.append(HeuristicAction(
                    kind="command", command=_portal_cmd,
                    confidence=0.95, domain="hunting",
                    reason=f"In town {_town_time:.0f}s - return to hunt via portal",
                ))
                actions.append(HeuristicAction(
                    kind="command", command="ai auto",
                    confidence=0.95, domain="hunting",
                    reason="Enable auto-attack after returning to hunt",
                ))
            total_confidence = 0.99
            top_domain = "economy"
            assessment = HeuristicAssessment(
                horizon=horizon, actions=actions, confidence=total_confidence,
                actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
            )
            self._last_assessment[bot_id] = assessment
            return assessment

        # ── STATE: TOWN_STUCK (in town too long, force hunting) ──
        if state == "TOWN_STUCK":
            self._town_entry_time[bot_id] = __import__("time").time() + 300
            # If already on hunting map, just enable auto-attack
            if map_name not in _HUNT_TOWNS:
                actions.append(HeuristicAction(
                    kind="command", command="ai auto",
                    confidence=0.99, domain="hunting",
                    reason="Already on hunting map - enable auto-attack",
                ))
            else:
                _portal = self._get_npc("portal_to_hunt", map_name)
                if _portal:
                    _portal_cmd = f"move {_portal['x']} {_portal['y']}"
                else:
                    _portal_cmd = "move 22 203"  # fallback
                actions.append(HeuristicAction(
                    kind="command", command=_portal_cmd,
                    confidence=0.99, domain="emergency",
                    reason="Stuck in town > 300s with 0 kills - force portal to hunting map",
                ))
            total_confidence = 0.99
            top_domain = "emergency"
            assessment = HeuristicAssessment(
                horizon=horizon, actions=actions, confidence=total_confidence,
                actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
            )
            self._last_assessment[bot_id] = assessment
            return assessment

        # ── STATE: SELL ──
        if state == "SELL":
            # Cooldown: only sell every 60s to prevent tight loop
            _sell_now = __import__("time").time()
            _last_sell = self._last_sell_time.get(bot_id, 0)
            if _sell_now - _last_sell < 60:
                # Sell on cooldown - fall through to TOWN_HUNT
                pass
            else:
                self._last_sell_time[bot_id] = _sell_now
                # Stand up first
                actions.append(HeuristicAction(
                    kind="command", command="stand",
                    confidence=0.95, domain="economy",
                    reason="Stand up before walking to Tool Dealer",
                ))
                # Walk to Tool Dealer (290, 221) and sell
                actions.append(HeuristicAction(
                    kind="command", command="move 290 221",
                    confidence=0.95, domain="economy",
                    reason=f"Weight {weight:.0%} - walk to Tool Dealer to sell junk",
                ))
                actions.append(HeuristicAction(
                    kind="command", command="talknpc 290 221 c r1 n",
                    confidence=0.90, domain="economy",
                    reason="Open Tool Dealer and sell items (atomic dialog)",
                ))
                actions.append(HeuristicAction(
                    kind="command", command="talk cont",
                    confidence=0.80, domain="economy",
                    reason="Complete sell transaction",
                ))
            total_confidence = 0.90
            top_domain = "economy"
            assessment = HeuristicAssessment(
                horizon=horizon, actions=actions, confidence=total_confidence,
                actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
            )
            self._last_assessment[bot_id] = assessment
            return assessment

        # ── STATE: WEAPON_BUY (priority over potions) ──
        if state == "WEAPON_BUY":
            _map = signals.get("map", "") or ""
            _is_hunting = any(x in _map for x in ["prt_fild", "pay_fild", "mjolnir", "gef_fild", "ra_fild"])
            if _is_hunting:
                # On hunting map - go through portal to Prontera first
                actions.append(HeuristicAction(
                    kind="command", command="move 22 203",
                    confidence=0.99, domain="economy",
                    reason="Go through portal to Prontera to buy weapon",
                ))
            else:
                # In Prontera - walk to Weapon Shop
                actions.append(HeuristicAction(
                    kind="command", command="move 160 133",
                    confidence=0.95, domain="economy",
                    reason=f"Zeny {zeny} - walk to Weapon Shop to buy weapon",
                ))
                # Buy a bow (1701) or knife (1301) depending on class
                _weapon = "1701"  # Default: Bow
                if "thief" in job_name or "assassin" in job_name:
                    _weapon = "1301"  # Knife
                elif "sword" in job_name or "knight" in job_name:
                    _weapon = "1201"  # Sword
                elif "mage" in job_name or "wizard" in job_name:
                    _weapon = "1501"  # Rod
                elif "acolyte" in job_name or "priest" in job_name:
                    _weapon = "1501"  # Rod (Mace is 1301 but starts with Rod)
                # Atomic: walk to NPC, open shop, buy weapon in one cycle
                actions.append(HeuristicAction(
                    kind="command", command=f"move 160 133",
                    confidence=0.95, domain="economy",
                    reason=f"Walk to Weapon Shop to buy weapon {_weapon}",
                ))
                actions.append(HeuristicAction(
                    kind="command", command=f"talknpc 160 133 c r0 n",
                    confidence=0.90, domain="economy",
                    reason="Open Weapon Shop dialog",
                ))
                actions.append(HeuristicAction(
                    kind="command", command=f"buy {_weapon} 1",
                    confidence=0.85, domain="economy",
                    reason=f"Buy weapon {_weapon} for class {job_name}",
                ))
            total_confidence = 0.90
            top_domain = "economy"
            assessment = HeuristicAssessment(
                horizon=horizon, actions=actions, confidence=total_confidence,
                actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
            )
            self._last_assessment[bot_id] = assessment
            return assessment

        # ── STATE: BUY ──
        if state == "BUY":
            # Cooldown: only buy every 60s to prevent tight loop blocking TOWN_HUNT
            _buy_now = __import__("time").time()
            _last_buy = self._last_buy_time.get(bot_id, 0)
            if _buy_now - _last_buy < 60:
                # Buy on cooldown - fall through to TOWN_HUNT instead of doing nothing
                pass
            else:
                self._last_buy_time[bot_id] = _buy_now
                # Stand up first
                actions.append(HeuristicAction(
                    kind="command", command="stand",
                    confidence=0.95, domain="economy",
                    reason="Stand up before walking to Tool Dealer",
                ))
                # Buy Red Potions from Tool Dealer (290, 221 in Prontera)
                actions.append(HeuristicAction(
                    kind="command", command="move 290 221",
                    confidence=0.95, domain="economy",
                    reason=f"Zeny {zeny} - walk to Tool Dealer to buy potions",
                ))
                actions.append(HeuristicAction(
                    kind="command", command="talknpc 290 221",
                    confidence=0.90, domain="economy",
                    reason="Open Tool Dealer shop",
                ))
                actions.append(HeuristicAction(
                    kind="command", command="talk resp 1",
                    confidence=0.85, domain="economy",
                    reason="Select buy option",
                ))
                # Buy Red Potions (item 501) - as many as zeny allows
                max_buy = min(int(zeny / 50), 30)  # 50z each, max 30
                if max_buy > 0:
                    actions.append(HeuristicAction(
                        kind="command", command=f"buy 501 {max_buy}",
                        confidence=0.90, domain="economy",
                        reason=f"Buy {max_buy} Red Potions (50z each)",
                    ))
                actions.append(HeuristicAction(
                    kind="command", command="talk any",
                    confidence=0.80, domain="economy",
                    reason="Complete buy dialog",
                ))
            total_confidence = 0.90
            top_domain = "economy"
            assessment = HeuristicAssessment(
                horizon=horizon, actions=actions, confidence=total_confidence,
                actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
            )
            self._last_assessment[bot_id] = assessment
            return assessment

        # ── STATE: TOWN_HUNT (in town, ready to hunt — walk to hunting map every cycle) ──
        if state == "TOWN_HUNT":
            # Stand up and enable AI
            actions.append(HeuristicAction(
                kind="command", command="stand",
                confidence=0.99, domain="hunting",
                reason="Stand up before moving to hunting map",
            ))
            actions.append(HeuristicAction(
                kind="command", command="ai auto",
                confidence=0.99, domain="hunting",
                reason="Enable auto-attack before moving to hunting map",
            ))
            # Set lockMap to hunting map
            _th_hunt_map = self._adaptive.get_best_map(bot_id, base_level) or "prt_fild05"
            self._set_config_once(actions, bot_id, "lockMap", _th_hunt_map, "hunting",
                f"Lock to hunting map {_th_hunt_map}")
            self._set_config_once(actions, bot_id, "lockMap_randX", "100", "hunting",
                "Random walk radius X")
            self._set_config_once(actions, bot_id, "lockMap_randY", "100", "hunting",
                "Random walk radius Y")
            self._set_config_once(actions, bot_id, "route_randomWalk", "1", "hunting",
                "Walk within lockMap bounds")
            # Move to hunting map
            if map_name != _th_hunt_map:
                actions.append(HeuristicAction(
                    kind="command", command=f"move {_th_hunt_map}",
                    confidence=0.99, domain="hunting",
                    reason=f"Move to hunting map {_th_hunt_map}",
                ))
            total_confidence = 0.99
            top_domain = "hunting"
            assessment = HeuristicAssessment(
                horizon=horizon, actions=actions, confidence=total_confidence,
                actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
            )
            self._last_assessment[bot_id] = assessment
            return assessment

        # ── STATE: JOB_CHANGE ──
        if state == "JOB_CHANGE":
            actions.append(HeuristicAction(
                kind="command", command="stand",
                confidence=0.95, domain="progression",
                reason="Stand up before walking to Archer Guild",
            ))
            actions.append(HeuristicAction(
                kind="command", command="move 160 191",
                confidence=0.95, domain="progression",
                reason=f"Level {base_level}/{job_level} Novice - walk to Archer Guild",
            ))
            actions.append(HeuristicAction(
                kind="command", command="talknpc 160 191",
                confidence=0.90, domain="progression",
                reason="Start job change dialog with Archer Guild NPC",
            ))
            actions.append(HeuristicAction(
                kind="command", command="talk continue",
                confidence=0.85, domain="progression",
                reason="Continue through job change dialog",
            ))
            actions.append(HeuristicAction(
                kind="command", command="talk resp 0",
                confidence=0.80, domain="progression",
                reason="Select Archer job option",
            ))
            total_confidence = 0.90
            top_domain = "progression"
            assessment = HeuristicAssessment(
                horizon=horizon, actions=actions, confidence=total_confidence,
                actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
            )
            self._last_assessment[bot_id] = assessment
            return assessment

        # ── STATE: STATS ──
        if state == "STATS":
            # Check stat_points from signals
            _current_stat_points = signals.get("stat_points", 0) or 0
            if _current_stat_points <= 0:
                # No stat points available - skip to next state
                total_confidence = 0.50
                top_domain = "progression"
                assessment = HeuristicAssessment(
                    horizon=horizon, actions=[], confidence=total_confidence,
                    actionable=False, top_domain=top_domain, signals=dict(signals),
                )
                self._last_assessment[bot_id] = assessment
                return assessment
            _job_name = signals.get("job_name", "novice") or "novice"
            _current_stats = {
                "str": signals.get("str", 1) or 1,
                "agi": signals.get("agi", 1) or 1,
                "vit": signals.get("vit", 1) or 1,
                "int": signals.get("int", 1) or 1,
                "dex": signals.get("dex", 1) or 1,
                "luk": signals.get("luk", 1) or 1,
            }
            # Use scaling targets with breakpoint awareness
            _allocations = _class_stat_allocation(
                _job_name, _current_stats, _current_stat_points,
                self._adaptive, base_level
            )
            for _stat_name, _points in _allocations:
                for _ in range(min(_points, _current_stat_points)):
                    actions.append(HeuristicAction(
                        kind="command", command=f"stat_add {_stat_name}",
                        confidence=0.95, domain="progression",
                        reason=f"Allocate 1 {_stat_name.upper()} ({_job_name}, scaling target, breakpoint-aware)",
                    ))
                    _current_stat_points -= 1
                    if _current_stat_points <= 0:
                        break
            total_confidence = 0.95
            top_domain = "progression"
            assessment = HeuristicAssessment(
                horizon=horizon, actions=actions, confidence=total_confidence,
                actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
            )
            self._last_assessment[bot_id] = assessment
            return assessment

        # ── STATE: SKILLS ──
        if state == "SKILLS":
            if "NV_BASIC" not in skills:
                actions.append(HeuristicAction(
                    kind="command", command="add 1",
                    confidence=0.90, domain="progression",
                    reason="Learn Basic Skill to sit and regen",
                ))
            elif "NV_FIRSTAID" not in skills:
                actions.append(HeuristicAction(
                    kind="command", command="add 2",
                    confidence=0.85, domain="progression",
                    reason="Learn First Aid for emergency healing",
                ))
            total_confidence = 0.90
            top_domain = "progression"
            assessment = HeuristicAssessment(
                horizon=horizon, actions=actions, confidence=total_confidence,
                actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
            )
            self._last_assessment[bot_id] = assessment
            return assessment

        # ── STATE: PARTY ──
        if state == "PARTY":
            # Differentiate: leader creates, others join by leader name
            _bot_profile = bot_id.split(":")[-1].split("/")[-1] if ":" in bot_id else bot_id
            # Dynamic leader detection: first bot alphabetically is leader
            _all_bots = signals.get("all_bots", []) or []
            _sorted_bots = sorted(_all_bots)
            _is_leader = len(_sorted_bots) > 0 and _bot_profile == _sorted_bots[0]
            # Leader sends party commands with 30s cooldown (party request only works on same map)
            if _is_leader:
                _now = __import__("time").time()
                _last_party = self._last_party_attempt.get(bot_id, 0)
                if _now - _last_party > 30:
                    self._last_party_attempt[bot_id] = _now
                    actions.append(HeuristicAction(
                        kind="command", command=f"party create AI{int(_now_t)}",
                        confidence=0.90, domain="social",
                        reason="Leader - create party with unique name",
                    ))
                    # Request all known bots to join (dynamically detected)
                    _all_bots = signals.get("all_bots", []) or []
                    for _other_bot in _all_bots:
                        if _other_bot != _bot_profile:
                            actions.append(HeuristicAction(
                                kind="command", command=f"party request {_other_bot}",
                                confidence=0.90, domain="social",
                                reason=f"Leader - request {_other_bot} to join party",
                            ))
                    actions.append(HeuristicAction(
                        kind="command", command="party share exp",
                        confidence=0.85, domain="social",
                        reason="Share experience in party",
                    ))
            else:
                # Joiners: move to hunting map so leader can request them (same map required)
                actions.append(HeuristicAction(
                    kind="command", command="move 22 203",
                    confidence=0.90, domain="social",
                    reason="Joiners - move to hunting map to be on same map as leader",
                ))
                actions.append(HeuristicAction(
                    kind="command", command="ai auto",
                    confidence=0.90, domain="hunting",
                    reason="Continue after moving to hunting map",
                ))
            total_confidence = 0.85
            top_domain = "social"
            assessment = HeuristicAssessment(
                horizon=horizon, actions=actions, confidence=total_confidence,
                actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
            )
            self._last_assessment[bot_id] = assessment
            return assessment

        # ── STATE: TOWN_HUNT ──
        if state == "TOWN_HUNT":
            # Stand up and ensure auto mode
            actions.append(HeuristicAction(
                kind="command", command="stand",
                confidence=0.95, domain="combat",
                reason="Stand up before moving to hunting map",
            ))
            actions.append(HeuristicAction(
                kind="command", command="ai auto",
                confidence=0.95, domain="combat",
                reason="Ensure AI is in auto mode",
            ))
            # Move to hunting map - use adaptive data
            target_map = self._adaptive.get_best_map(bot_id, base_level)
            if not target_map:
                # Fallback based on level
                if base_level >= 20:
                    target_map = "pay_fild01"
                elif base_level >= 15:
                    target_map = "prt_fild08"
                else:
                    target_map = "prt_fild05"
            actions.append(HeuristicAction(
                kind="command", command=f"move {target_map}",
                confidence=0.90, domain="exploration",
                reason=f"Level {base_level} - move to {target_map} for grinding",
            ))
            total_confidence = 0.90
            top_domain = "exploration"
            assessment = HeuristicAssessment(
                horizon=horizon, actions=actions, confidence=total_confidence,
                actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
            )
            self._last_assessment[bot_id] = assessment
            return assessment

        # ── STATE: HUNT ──
        if state == "HUNT":
            # ── COMBAT CONFIG: Set once per value change (dedup via _set_config_once) ──
            _job_name = signals.get("job_name", "novice") or "novice"
            _class_lc = _job_name.lower()
            _atk_dist = 7
            _atk_max = 20
            _rw = 1
            self._set_config_once(actions, bot_id, "route_randomWalk", str(_rw), "hunting",
                "Walk within lockMap bounds (doesn't block AI, attacks anything it passes)")
            self._set_config_once(actions, bot_id, "lockMap_randX", "100", "hunting",
                "Random walk radius X")
            self._set_config_once(actions, bot_id, "lockMap_randY", "100", "hunting",
                "Random walk radius Y")
            self._set_config_once(actions, bot_id, "attackDistance", str(_atk_dist), "hunting",
                f"Class-appropriate attack distance for {_job_name}")
            self._set_config_once(actions, bot_id, "attackMaxDistance", str(_atk_max), "hunting",
                "Set max chase distance")
            self._set_config_once(actions, bot_id, "attackAuto", "3", "hunting",
                "Enable aggressive auto-attack")
            self._set_config_once(actions, bot_id, "attackAuto_followTarget", "1", "hunting",
                "Chase fleeing monsters")
            self._set_config_once(actions, bot_id, "attackAuto_noMove", "0", "hunting",
                "Allow movement during combat")
            self._set_config_once(actions, bot_id, "attackAuto_inLockOnly", "1", "hunting",
                "Only attack monsters in lockMap area")
            self._set_config_once(actions, bot_id, "attackAuto_onlyWhenSafe", "0", "hunting",
                "Attack even if not safe")
            self._set_config_once(actions, bot_id, "attackAuto_fleeToTarget", "0", "hunting",
                "Don't flee to target")
            self._set_config_once(actions, bot_id, "attackAuto_startDistance", "1", "hunting",
                "Start attacking from 1 cell away (immediate)")
            self._set_config_once(actions, bot_id, "attackAuto_keepDistance", "1", "hunting",
                "Keep distance while attacking")
            self._set_config_once(actions, bot_id, "attackAuto_maxDistance", "20", "hunting",
                "Keep attacking even if target moves")
            self._set_config_once(actions, bot_id, "attackAuto_unstuck", "1", "hunting",
                "Don't give up mid-fight")
            # ai auto is not a config set — always emit (not deduped)
            # Only force stand if HP is high enough to fight
            # If HP < 40%, let the bot sit to regen — don't force it into combat
            _hp_ratio_hunt = signals.get("hp_ratio", 1.0) or 1.0
            if _hp_ratio_hunt >= 0.40:
                actions.append(HeuristicAction(
                    kind="command", command="stand",
                    confidence=0.95, domain="hunting",
                    reason="Stand up before enabling auto-attack",
                ))
            actions.append(HeuristicAction(
                kind="command", command="ai auto",
                confidence=0.95, domain="hunting",
                reason="Ensure auto-attack mode is active",
            ))
            # Teleport config
            self._set_config_once(actions, bot_id, "teleportAuto", "0", "hunting",
                "No teleport in dungeons (density is 3-5x field maps)", confidence=0.99)
            self._set_config_once(actions, bot_id, "teleportAuto_minAggressives", "8", "survival",
                "Only teleport when 8+ mobs")
            self._set_config_once(actions, bot_id, "teleportAuto_hp", "0", "survival",
                "Never teleport due to HP")
            self._set_config_once(actions, bot_id, "teleportAuto_deadly", "0", "survival",
                "Disable deadly teleport")
            self._set_config_once(actions, bot_id, "teleportAuto_search", "0", "survival",
                "Disable search teleport")
            self._set_config_once(actions, bot_id, "teleportAuto_portal", "0", "survival",
                "Disable portal teleport")
            # Disable OpenKore's built-in sit logic — the bridge reflex handles sit
            # OpenKore's AI re-enables sitAuto between cycles.
            # Push aiSidecar_sitAutoHp=0 so the bridge's cycle override sets sitAuto_hp_lower=0 every cycle.
            self._set_config_once(actions, bot_id, "aiSidecar_sitAutoHp", "0", "survival",
                "Disable built-in sit — bridge reflex handles it")
            self._set_config_once(actions, bot_id, "aiSidecar_sitAutoHpMax", "0", "survival",
                "Disable built-in sit — bridge reflex handles it")
            # Loot config
            self._set_config_once(actions, bot_id, "itemsTakeAuto", "1", "economy",
                "Auto-pickup items")
            self._set_config_once(actions, bot_id, "itemsTakeWeight", "50", "economy",
                "Only pick up if weight < 50%")
            self._set_config_once(actions, bot_id, "itemsGatherAuto", "2", "economy",
                "Gather all items (weight filter prevents junk)")
            if map_name not in _HUNT_TOWNS:
                # On hunting map: check if we should return to town
                _hunt_duration = __import__("time").time() - self._state_since.get(bot_id, __import__("time").time())
                _hp_ratio = signals.get("hp_ratio", 1.0) or 1.0
                _sp_ratio = signals.get("sp_ratio", 1.0) or 1.0
                _current_sp = signals.get("sp", 0) or 0
                _max_sp = signals.get("max_sp", 100) or 100
                _agi = signals.get("agi", 1) or 1
                _dex = signals.get("dex", 1) or 1
                _str = signals.get("str", 1) or 1
                _player_hp = signals.get("hp", 100) or 100
                _attack_power = signals.get("attack_power", 25) or 25
                _has_items = (signals.get("inventory_items", 0) or 0) > 0
                _total_kills = signals.get("kills", 0) or 0
                _zeny = signals.get("zeny", 0) or 0
                # ── FLY WING ESCAPE: If surrounded by 3+ mobs, use Fly Wing ──
                _aggro_count = signals.get("aggressives", 0) or 0
                if _aggro_count >= 3 and _hp_ratio < 0.5:
                    actions.append(HeuristicAction(
                        kind="command", command="use 601",
                        confidence=0.99, domain="survival",
                        reason=f"Surrounded by {_aggro_count} mobs at {_hp_ratio:.0%} HP - Fly Wing escape",
                    ))
                # ── BUTTERFLY WING RETURN: If low HP + no items + been hunting 5+ min ──
                if _hp_ratio < 0.3 and not _has_items and _hunt_duration > 300:
                    actions.append(HeuristicAction(
                        kind="command", command="use 602",
                        confidence=0.95, domain="survival",
                        reason=f"Low HP ({_hp_ratio:.0%}) no items - Butterfly Wing to town",
                    ))
                # ── FOOD/BUFF SYSTEM: Use food if zeny > 1000 and been hunting 5+ min ──
                if _zeny > 1000 and _hunt_duration > 300:
                    _primary_stat = {"archer": "dex", "thief": "agi", "swordman": "str",
                                     "mage": "int", "acolyte": "int", "merchant": "str"}.get(_job_name, "str")
                    _food_id = {"str": "531", "agi": "532", "vit": "533", "int": "534", "dex": "535", "luk": "536"}.get(_primary_stat, "531")
                    actions.append(HeuristicAction(
                        kind="command", command=f"use {_food_id}",
                        confidence=0.80, domain="economy",
                        reason=f"Use {_primary_stat} food (+4 {_primary_stat.upper()}, 30 min)",
                    ))
                # ── DYNAMIC SKILL ROTATION: DPS-based, SP-aware, situation-aware ──
                _monster_element = signals.get("monster_element", "Neutral") or "Neutral"
                _monster_hp = signals.get("monster_hp", 50) or 50
                _monster_def = signals.get("monster_def", 0) or 0
                _monster_size = signals.get("monster_size", "Medium") or "Medium"
                _monster_race = signals.get("monster_race", "Brute") or "Brute"
                _monster_name = signals.get("monster_name", "") or ""
                _known_skills = signals.get("skills", []) or []
                _skill_levels = signals.get("skill_levels", {}) or {}
                _weapon_type = JOB_WEAPON_TYPE.get(_job_name, "dagger")
                _aggro_count = signals.get("aggressives", 0) or 0
                _best_skill = get_best_skill(
                    _known_skills, _skill_levels, _attack_power, _weapon_type,
                    _monster_def, _monster_size, _monster_element, _monster_race,
                    _current_sp, _max_sp, _agi, _dex, _aggro_count, _player_hp
                )
                if _best_skill:
                    actions.append(HeuristicAction(
                        kind="command", command=f"attack_skill {_best_skill}",
                        confidence=0.90, domain="combat",
                        reason=f"DPS skill: {_best_skill} (best DPS vs {_monster_element} monster)",
                    ))
                # ── MVP AWARENESS: If an MVP is nearby, prioritize it ──
                _nearby_monsters = signals.get("monsters", []) or []
                for _nm in _nearby_monsters:
                    _nm_name = _nm.get("name", "") if isinstance(_nm, dict) else str(_nm)
                    if is_mvp(_nm_name):
                        _mvp_value = get_mvp_value(_nm_name)
                        actions.append(HeuristicAction(
                            kind="command", command=f"attack {_nm_name}",
                            confidence=0.99, domain="hunting",
                            reason=f"MVP {_nm_name} nearby! (drop value ~{_mvp_value:,}z)",
                        ))
                        break
                # ── WEIGHT TIME-TO-CAP: Skip low-value drops if close to cap ──
                _weight_capacity = signals.get("weight_capacity", 1000) or 1000
                _kills_per_min = signals.get("kills_per_min", 5) or 5
                _avg_drop_weight = 1.0  # Average item weight
                _time_to_cap = calculate_weight_time_to_cap(_weight_capacity, _avg_drop_weight, _kills_per_min)
                if _time_to_cap < 10:
                    actions.append(HeuristicAction(
                        kind="command", command="set itemsTakeWeight 30",
                        confidence=0.80, domain="economy",
                        reason=f"Weight cap in {_time_to_cap:.0f} min - skip low-value drops",
                    ))
                # EQUIPMENT PROGRESSION: check if bot should upgrade weapon
                _eq_prog = self._adaptive.equipment_progression.get(job_name, [])
                _best_weapon = None
                for _lvl, _wid, _desc in _eq_prog:
                    if base_level >= _lvl:
                        _best_weapon = (_wid, _desc)
                if _best_weapon and zeny >= 100 and _hunt_duration > 60 and _total_kills > 5:
                    # Check if we have enough zeny and have been hunting long enough
                    # The actual buy happens in WEAPON_BUY state when in town
                    pass  # Will trigger WEAPON_BUY on next town visit
                # AT PORTAL EXIT: if bot is at (367, 205) on prt_fild05, move to center
                _x = signals.get("x", 0) or 0
                _y = signals.get("y", 0) or 0
                if abs(_x - 367) < 10 and abs(_y - 205) < 10 and map_name == "prt_fild05":
                    actions.append(HeuristicAction(
                        kind="command", command="move 200 200",
                        confidence=0.99, domain="hunting",
                        reason="At portal exit - move to center of hunting map",
                    ))
                    # Don't return early - continue to set combat config below
                # ── SPAWN CIRCUIT: Use heatmap to build optimized walking path ──
                _spawn_heatmap = self._adaptive.spawn_heatmap.get(map_name, {})
                if _spawn_heatmap and len(_spawn_heatmap) >= 3:
                    _circuit = build_spawn_circuit(_spawn_heatmap, _x, _y, 5)
                    if _circuit and len(_circuit) >= 2:
                        _next_wp = _circuit[0]
                        actions.append(HeuristicAction(
                            kind="command", command=f"move {_next_wp[0]} {_next_wp[1]}",
                            confidence=0.80, domain="hunting",
                            reason=f"Spawn circuit: walk to hot zone ({_next_wp[0]}, {_next_wp[1]})",
                        ))
                # JUST WARPED: if just arrived, sit to regen first
                if _hunt_duration < 15:
                    if _hp_ratio < 0.5:
                        actions.append(HeuristicAction(
                            kind="command", command="sit",
                            confidence=0.99, domain="survival",
                            reason=f"HP={_hp_ratio:.0%} just warped - sit to regen before hunting",
                        ))
                        # Don't return early - continue to set combat config below
                    # Don't return to town within first 30s - starting gear triggers weight check
                    if _hunt_duration < 30:
                        actions.append(HeuristicAction(
                            kind="command", command="ai auto",
                            confidence=0.95, domain="hunting",
                            reason=f"Just warped {_hunt_duration:.0f}s ago - hunt first, sell later",
                        ))
                        # Don't return early - continue to set combat config below
                # If HP < 30% and have items AND have killed something, sit to regen
                # Don't return to town from hunting map - let OpenKore's AI handle it
                if _hp_ratio < 0.3 and _has_items and _total_kills > 0 and _hunt_duration > 15:
                    actions.append(HeuristicAction(
                        kind="command", command="sit",
                        confidence=0.99, domain="survival",
                        reason=f"HP={_hp_ratio:.0%} items>0 - sit to regen on hunting map",
                    ))
                    total_confidence = 0.99
                    top_domain = "survival"
                    assessment = HeuristicAssessment(
                        horizon=horizon, actions=actions, confidence=total_confidence,
                        actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
                    )
                    self._last_assessment[bot_id] = assessment
                    return assessment
                # EMERGENCY: if HP < 15%, sit immediately regardless of items
                if _hp_ratio < 0.15:
                    actions.append(HeuristicAction(
                        kind="command", command="sit",
                        confidence=0.99, domain="survival",
                        reason=f"HP={_hp_ratio:.0%} CRITICAL - emergency sit",
                    ))
                    total_confidence = 0.99
                    top_domain = "survival"
                    assessment = HeuristicAssessment(
                        horizon=horizon, actions=actions, confidence=total_confidence,
                        actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
                    )
                    self._last_assessment[bot_id] = assessment
                    return assessment
                # If HP < 20% and no items, sit and regen instead of returning
                if _hp_ratio < 0.2 and not _has_items:
                    actions.append(HeuristicAction(
                        kind="command", command="sit",
                        confidence=0.99, domain="survival",
                        reason=f"HP={_hp_ratio:.0%} no items - sitting to regen",
                    ))
                    total_confidence = 0.99
                    top_domain = "survival"
                    assessment = HeuristicAssessment(
                        horizon=horizon, actions=actions, confidence=total_confidence,
                        actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
                    )
                    self._last_assessment[bot_id] = assessment
                    return assessment
                # If have items and been hunting > 120s, keep hunting
                # Don't return to town from hunting map - let sellAuto handle it
                if _has_items and _total_kills > 0 and _hunt_duration > 120:
                    actions.append(HeuristicAction(
                        kind="command", command="ai auto",
                        confidence=0.95, domain="hunting",
                        reason=f"Items>0 and hunted {_hunt_duration:.0f}s - keep hunting, sell later",
                    ))
                    total_confidence = 0.95
                    top_domain = "hunting"
                    assessment = HeuristicAssessment(
                        horizon=horizon, actions=actions, confidence=total_confidence,
                        actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
                    )
                    self._last_assessment[bot_id] = assessment
                    return assessment
                # If HP < 30% and no items, sit to regen
                if _hp_ratio < 0.3 and not _has_items:
                    actions.append(HeuristicAction(
                        kind="command", command="sit",
                        confidence=0.99, domain="survival",
                        reason=f"HP={_hp_ratio:.0%} no items - sitting to regen",
                    ))
                    total_confidence = 0.99
                    top_domain = "survival"
                    assessment = HeuristicAssessment(
                        horizon=horizon, actions=actions, confidence=total_confidence,
                        actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
                    )
                    self._last_assessment[bot_id] = assessment
                    return assessment
                # MAP PROGRESSION: Use rAthena data-driven optimal map selection
                # Character-aware: considers level, job, attack power, monster stats
                _attack_power = signals.get("attack_power", 25) or 25
                _optimal_map, _optimal_reason = self._adaptive.get_optimal_hunting_map(
                    job_name, base_level, _attack_power
                )
                _survivability = self._adaptive.estimate_survivability(
                    _optimal_map, base_level, _attack_power
                )
                # Skip if bot is already en route to a different map
                _last_set_lockmap = self._last_lockmap.get(bot_id, "")
                if not _last_set_lockmap:
                    _last_set_lockmap = _optimal_map
                    self._last_lockmap[bot_id] = _last_set_lockmap
                _current_lockmap = signals.get("lockMap", _last_set_lockmap) or _last_set_lockmap
                if map_name == _current_lockmap or _hunt_duration > 60:
                    # Use rAthena data-driven optimal map selection
                    _next_map = _optimal_map
                    _next_reason = _optimal_reason
                    # If current map is not the correct one for level, move
                    if map_name != _next_map and _hunt_duration > 30:
                        self._last_lockmap[bot_id] = _next_map
                        actions.append(HeuristicAction(
                            kind="command", command=f"set lockMap {_next_map}",
                            confidence=0.90, domain="hunting",
                            reason=f"Level {base_level} - {_next_reason}",
                        ))
                        actions.append(HeuristicAction(
                            kind="command", command=f"move {_next_map}",
                            confidence=0.90, domain="hunting",
                            reason=f"Level {base_level} - progressing to {_next_map}",
                        ))
                    total_confidence = 0.90
                    top_domain = "hunting"
                    assessment = HeuristicAssessment(
                        horizon=horizon, actions=actions, confidence=total_confidence,
                        actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
                    )
                    self._last_assessment[bot_id] = assessment
                    return assessment
                # STAT ALLOCATION: Use stat_points signal directly (most reliable)
                _stat_points = signals.get("stat_points", 0) or 0
                _job_name = signals.get("job_name", "novice") or "novice"
                if _stat_points > 0:
                    _stat_builds = {
                        "novice": ["dex", "str", "agi", "vit"],
                        "archer": ["dex", "agi", "str", "vit"],
                        "thief": ["dex", "agi", "str", "vit"],
                        "acolyte": ["dex", "int", "vit", "str"],
                        "swordman": ["dex", "str", "vit", "agi"],
                        "mage": ["dex", "int", "vit", "str"],
                    }
                    _build = _stat_builds.get(_job_name, ["dex", "str", "agi", "vit"])
                    _pts_to_alloc = _stat_points
                    for _stat_name in _build:
                        while _pts_to_alloc > 0:
                            actions.append(HeuristicAction(
                                kind="command", command=f"stat_add {_stat_name}",
                                confidence=0.99, domain="progression",
                                reason=f"Allocate 1 {_stat_name.upper()} ({_job_name}, {_stat_points} pts available)",
                            ))
                            _pts_to_alloc -= 1
                            if _pts_to_alloc <= 0:
                                break
                    total_confidence = 0.99
                    top_domain = "progression"
                    assessment = HeuristicAssessment(
                        horizon=horizon, actions=actions, confidence=total_confidence,
                        actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
                    )
                    self._last_assessment[bot_id] = assessment
                    return assessment
                # EQUIPMENT CHECK: If no weapon equipped and has zeny, buy gear
                _atk_power = signals.get("attack_power", 0) or 0
                _zeny = signals.get("zeny", 0) or 0
                _equip = signals.get("equipment", {}) or {}
                _has_weapon_equipped = any("weapon" in k.lower() for k in (_equip.keys() if isinstance(_equip, dict) else []))
                _no_weapon = not _has_weapon_equipped and _atk_power < 10
                if _no_weapon and _zeny >= 100:
                    self._state[bot_id] = "WEAPON_BUY"
                    self._state_since[bot_id] = _now_t
                    actions.append(HeuristicAction(
                        kind="command", command="ai auto",
                        confidence=0.95, domain="economy",
                        reason="No weapon detected - go buy one",
                    ))
                    total_confidence = 0.95
                    top_domain = "economy"
                    assessment = HeuristicAssessment(
                        horizon=horizon, actions=actions, confidence=total_confidence,
                        actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
                    )
                    self._last_assessment[bot_id] = assessment
                    return assessment
                # PARTY: Only handle party formation if level >= 20 (solo farm until then)
                _base_level = signals.get("level", 1) or 1
                if _base_level >= 20 and map_name in _HUNT_TOWNS:
                    _party_in = signals.get("in_party", False)
                    _party_members = signals.get("party_members", []) or []
                    _all_bots = signals.get("all_bots", []) or []
                    _bot_profile = bot_id.split(":")[-1].split("/")[-1] if ":" in bot_id else bot_id
                    _sorted_bots = sorted(_all_bots)
                    _is_leader = len(_sorted_bots) > 0 and _bot_profile == _sorted_bots[0]
                    _party_incomplete = _party_in and len(_party_members) + 1 < len(_all_bots)
                    if not _party_in or _party_incomplete:
                        if _is_leader:
                            _now = __import__("time").time()
                            _last_party = self._last_party_attempt.get(bot_id, 0)
                            if _now - _last_party > 5:  # Check every 5s
                                self._last_party_attempt[bot_id] = _now
                                actions.append(HeuristicAction(
                                    kind="command", command="party leave",
                                    confidence=0.99, domain="social",
                                    reason="Leader - leave old party to re-create",
                                ))
                                actions.append(HeuristicAction(
                                    kind="command", command=f"party create AI{int(_now)}",
                                    confidence=0.95, domain="social",
                                    reason="Leader - create party",
                                ))
                                for _other_bot in _all_bots:
                                    if _other_bot != _bot_profile:
                                        actions.append(HeuristicAction(
                                            kind="command", command=f"party request {_other_bot}",
                                            confidence=0.95, domain="social",
                                            reason=f"Leader - request {_other_bot} to join",
                                        ))
                                actions.append(HeuristicAction(
                                    kind="command", command="party share exp",
                                    confidence=0.90, domain="social",
                                    reason="Share experience in party",
                                ))
                        else:
                            actions.append(HeuristicAction(
                                kind="command", command="set partyAuto 2",
                                confidence=0.99, domain="social",
                                reason="Set partyAuto to auto-accept",
                            ))
                        total_confidence = 0.95
                        top_domain = "social"
                        assessment = HeuristicAssessment(
                            horizon=horizon, actions=actions, confidence=total_confidence,
                            actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
                        )
                        self._last_assessment[bot_id] = assessment
                        return assessment
                #                 # After party is formed, move to hunting map
                _map = signals.get("map", "") or ""
                if "prontera" in _map or "prt_in" in _map:
                    actions.append(HeuristicAction(
                        kind="command", command="move prt_fild05",
                        confidence=0.95, domain="hunting",
                        reason="Move to hunting map after party formation",
                    ))
                # ECONOMY CONFIG: Ensure sellAuto, itemsTakeAuto, buyAuto are set
                actions.append(HeuristicAction(
                    kind="command", command="set sellAuto 1",
                    confidence=0.99, domain="economy",
                    reason="Enable auto-sell",
                ))
                actions.append(HeuristicAction(
                    kind="command", command="set itemsTakeAuto 2",
                    confidence=0.99, domain="economy",
                    reason="Enable auto-loot",
                ))
                actions.append(HeuristicAction(
                    kind="command", command="set itemsTakeAuto_party 1",
                    confidence=0.99, domain="economy",
                    reason="Enable party loot sharing",
                ))
                # HP MANAGEMENT: Sit when low HP to prevent death (hunting map only)
                # Ranged classes (Archer/Mage) get lower threshold (30%) since they're at range
                _hp = signals.get("hp_ratio", 1.0) or 1.0
                _hp_job = signals.get("job_name", "novice") or "novice"
                _hp_map = signals.get("map", "") or ""
                _hp_on_hunting_map = "prt_fild" in _hp_map or "pay_fild" in _hp_map or "mjolnir" in _hp_map or "gef_fild" in _hp_map or "ra_fild" in _hp_map
                _hp_threshold = 0.30 if any(x in _hp_job.lower() for x in ["archer", "hunter", "mage", "wizard"]) else 0.50
                if _hp < _hp_threshold and _hp_on_hunting_map:
                    actions.append(HeuristicAction(
                        kind="command", command="sit",
                        confidence=0.99, domain="survival",
                        reason=f"HP {_hp*100:.0f}% < 50% - sit to regen",
                    ))
                    actions.append(HeuristicAction(
                        kind="command", command="ai auto",
                        confidence=0.99, domain="survival",
                        reason="Sit regen at low HP",
                    ))
                else:
                    # Already set combat config at top of HUNT handler
                    # Just ensure ai auto is set
                    actions.append(HeuristicAction(
                        kind="command", command="ai auto",
                        confidence=0.95, domain="hunting",
                        reason="On hunting map - enable auto-attack",
                    ))
                    total_confidence = 0.95
                    top_domain = "hunting"
                    assessment = HeuristicAssessment(
                        horizon=horizon, actions=actions, confidence=total_confidence,
                        actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
                    )
                    self._last_assessment[bot_id] = assessment
                    return assessment
            # AUTO-STATS: If bot has unspent stat points, transition to STATS state
            _current_stat_points = signals.get("stat_points", 0) or 0
            if _current_stat_points > 0:
                self._state[bot_id] = "STATS"
                self._state_since[bot_id] = __import__("time").time()
                actions.append(HeuristicAction(
                    kind="command", command="ai auto",
                    confidence=0.99, domain="progression",
                    reason=f"Has {_current_stat_points} unspent stat points in town - allocate via STATS state",
                ))
                total_confidence = 0.99
                top_domain = "progression"
                assessment = HeuristicAssessment(
                    horizon=horizon, actions=actions, confidence=total_confidence,
                    actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
                )
                self._last_assessment[bot_id] = assessment
                return assessment
            # In town: sell items, buy potions
            # IMPORTANT: ONLY generate shop commands, NOT "move" 
            # "move" interrupts NPC dialog - it comes on next cycle
            _has_items = (signals.get("inventory_items", 0) or 0) > 0
            if _has_items:
                # Sell first - talknpc opens NPC dialog, sellAuto handles the rest
                # Look up sell NPC from database
                _sell_npc = self._get_npc("sell", map_name)
                if _sell_npc:
                    _sell_cmd = f"talknpc {_sell_npc['x']} {_sell_npc['y']} {' '.join(eval(_sell_npc['steps']))}"
                else:
                    _sell_cmd = "talknpc 147 175 c r1 n"  # fallback
                actions.append(HeuristicAction(
                    kind="command", command=_sell_cmd,
                    confidence=0.99, domain="economy",
                    reason=f"In town - sell items",
                ))
            elif zeny >= 50:
                # No items to sell, but have zeny - buy potions
                _potions_to_buy = min(10, zeny // 50)
                if _potions_to_buy > 0:
                    # Check if near NPC - if so, buy directly. Otherwise walk to NPC first
                    _x = signals.get("x", 0) or 0
                    _y = signals.get("y", 0) or 0
                    _buy_npc = self._get_npc("buy_potion", map_name)
                    _buy_x = _buy_npc['x'] if _buy_npc else 126
                    _buy_y = _buy_npc['y'] if _buy_npc else 76
                    _dist_to_npc = abs(_x - _buy_x) + abs(_y - _buy_y)
                    if _dist_to_npc < 10:
                        actions.append(HeuristicAction(
                            kind="command", command=f"buy 501 {_potions_to_buy}",
                            confidence=0.99, domain="economy",
                            reason=f"In town - buy {_potions_to_buy} potions (zeny={zeny})",
                        ))
                    else:
                        _buy_npc = self._get_npc("buy_potion", map_name)
                        if _buy_npc:
                            _buy_cmd = f"move {_buy_npc['x']} {_buy_npc['y']}"
                        else:
                            _buy_cmd = "move 126 76"  # fallback
                        actions.append(HeuristicAction(
                            kind="command", command=_buy_cmd,
                            confidence=0.99, domain="economy",
                            reason=f"Walk to NPC to buy {_potions_to_buy} potions",
                        ))
                # After buying (or trying to), return to hunt
                # This fires on the NEXT cycle after buy command is generated
                # (buy command was generated this cycle, next cycle we return to hunt)
                _town_time = __import__("time").time() - self._town_entry_time.get(bot_id, __import__("time").time())
                if _town_time > 30:
                    _portal = self._get_npc("portal_to_hunt", map_name)
                    if _portal:
                        _portal_cmd = f"move {_portal['x']} {_portal['y']}"
                    else:
                        _portal_cmd = "move 22 203"  # fallback
                    actions.append(HeuristicAction(
                        kind="command", command=_portal_cmd,
                        confidence=0.95, domain="hunting",
                        reason=f"Been in town {_town_time:.0f}s with zeny - return to hunt",
                    ))
            # No move here - let next cycle handle it after shop dialog completes
            total_confidence = 0.95
            top_domain = "hunting"
            assessment = HeuristicAssessment(
                horizon=horizon, actions=actions, confidence=total_confidence,
                actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
            )
            self._last_assessment[bot_id] = assessment
            return assessment
        return assessment

    def confidence_for(self, horizon: str, signals: dict = None, bot_id: str = None) -> float:
        if signals:
            state = self._get_state(signals)
            if state in ("SELL", "BUY", "STATS", "SKILLS"):
                return 0.95  # High confidence - mechanical actions
            if state == "JOB_CHANGE":
                return 0.90
            if state == "DEAD":
                return 0.50
            if state in ("HUNT", "TOWN_HUNT"):
                return 0.85
            if state == "PARTY":
                return 0.85
            return 0.70
        return 0.70

    def _build_summary(self, assessment: HeuristicAssessment) -> str:
        if not assessment or not assessment.actions:
            return "no heuristic actions"
        parts = [f"{a.domain}:{a.command}" for a in assessment.actions[:5]]
        return f"conf={assessment.confidence:.2f} " + " | ".join(parts)
