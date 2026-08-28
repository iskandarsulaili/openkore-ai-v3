from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from ai_sidecar.actions import HeuristicAction as _HeuristicAction
from pathlib import Path
import threading
from ai_sidecar.game_knowledge_db import GameKnowledgeDB
from ai_sidecar.anti_detection import BehaviorEngine, get_behavior_engine
from ai_sidecar.anti_detection.behavior_engine import BehaviorProfileType
from ai_sidecar.autonomy.ro_mechanics import (
    PER_MAP_MON_CONTROL,
    JOB_CHANGE_2_1,
    JOB_2_1_CLASSES,
    CLASS_SKILL_TRAINING,

    get_monster_stats, calculate_aspd, calculate_flee, calculate_hit,
    calculate_monster_hit_rate, calculate_damage, calculate_profit_per_kill,
    calculate_skill_dps, get_best_skill, get_nearest_breakpoint,
    get_scaling_stat_targets, estimate_hits_to_die,
    calculate_party_exp_share, calculate_weight_time_to_cap,
    build_spawn_circuit, is_mvp, get_mvp_value, get_skill_element,
    get_optimal_element_for_map,
    ELEMENT_TABLE, SIZE_PENALTY, JOB_WEAPON_TYPE,
    WEAPON_BASE_ASPD, SKILL_DAMAGE, SKILL_SP_COSTS, FOOD_ITEMS,
    CARD_VALUES, STAT_BREAKPOINTS, SCALING_STAT_TARGETS,
    POTION_COST, POTION_HEAL_MIN, POTION_HEAL_MAX,
    BLUE_POTION_COST, BLUE_POTION_SP_MIN, BLUE_POTION_SP_MAX, ARROW_COST,
    MVP_MONSTERS, ELEMENTAL_WEAPONS, JOB_CHANGE_TALK,
    JOB_CHANGE_2_1, JOB_2_1_CLASSES,
    PER_MAP_MON_CONTROL, CLASS_SKILL_TRAINING,
)
from ai_sidecar.autonomy.domains import DomainRegistry
from ai_sidecar.domains.combat.dispatcher import TacticsDispatcher
from ai_sidecar.domains.npc.shop import NPCShop
from ai_sidecar.domains.npc.dialogue import NPCDialogueEngine
from ai_sidecar.domains.npc.storage import NPCStorage
from ai_sidecar.domains.quests.tracker import QuestTracker
from ai_sidecar.domains.quests.automation import QuestAutomation
from ai_sidecar.domains.equipment.manager import EquipmentManager
from ai_sidecar.domains.equipment.swapper import WeaponSwapper
from ai_sidecar.domains.crafting.alchemy import AlchemyCrafting
from ai_sidecar.domains.crafting.cooking import CookingCrafting
from ai_sidecar.domains.crafting.forging import ForgingCrafting
from ai_sidecar.domains.instances.registry import InstanceRegistry
from ai_sidecar.domains.instances.coordinator import InstanceCoordinator
from ai_sidecar.domains.consumables.buffs import AutoBuffManager
from ai_sidecar.domains.consumables.recovery import RecoveryManager
from ai_sidecar.domains.companions.pets import PetManager
from ai_sidecar.domains.companions.homunculus import HomunculusManager
from ai_sidecar.domains.companions.mercenary import MercenaryManager
from ai_sidecar.domains.environment.time import GameTimeTracker
from ai_sidecar.domains.navigation.portals import PortalDB
from ai_sidecar.domains.navigation.pathfinding import Pathfinder
from ai_sidecar.combat.map_knowledge import get_hunting_maps as _mk_get_hunting_maps
from ai_sidecar.combat.map_knowledge import reachable_hunting_maps as _mk_reachable_hunting_maps
from ai_sidecar.strategy.unified_consciousness import UnifiedConsciousness, DecisionDomain, DecisionUrgency
from ai_sidecar.domains.progression.lifecycle import LifecycleStateMachine
from ai_sidecar.domains.progression.advancement import AdvancementDomain
from ai_sidecar.domains.pvp.arenas import ArenaTactics
from ai_sidecar.domains.pvp.woe import WoETactics
from ai_sidecar.domains.learning.experience import ExperienceTracker
from ai_sidecar.domains.learning.adaptation import StrategyAdapter
from ai_sidecar.domains.planning.goals import GoalManager
from ai_sidecar.domains.planning.scheduler import TaskScheduler
from ai_sidecar.domains.planning.translator import TaskCommandTranslator
from ai_sidecar.domains.social.swarm import SwarmCoordinator
from ai_sidecar.state.collector import StateCollector
from ai_sidecar.domains.combat.safety import DangerPredictor, SafetyEvaluator
from ai_sidecar.domains.world.state import WorldState, get_world_state
from ai_sidecar.domains.combat.pressure import CombatPressureDomain
from ai_sidecar.domains.combat.tactics.kiting_v2 import TickBasedKiting
from ai_sidecar.domains.economy.map_policies import InventoryPolicies, SpawnNavigator
from ai_sidecar.domains.social.combo_protocol import ComboHandshakeProtocol
from ai_sidecar.runtime.event_bus import EventBus, post_death_event
from ai_sidecar.runtime.degradation import safe_assess, get_registry
from ai_sidecar.runtime.persistence import PersistentState
from ai_sidecar.domains.navigation.danger_pathfinding import DangerAwarePathfinder
from ai_sidecar.domains.equipment.loadout import ConsumableLoadoutPlanner, DurabilityMonitor, PostMortemAnalyzer
from ai_sidecar.domains.social.loot import LootDisciplineEngine, EventDetector, LiveMarketScanner
from ai_sidecar.domains.social.reputation import SocialReputationDomain
from ai_sidecar.domains.social.swarm.shm_ipc import SharedMemoryIPC, SharedMemoryCoordination
from ai_sidecar.domains.economy.farming_loop import FarmingLoopOptimizer
from ai_sidecar.domains.planning.rotation import MapRotationPlanner
from ai_sidecar.domains.world.game_sense import GameSenseEngine, get_game_sense_engine
from ai_sidecar.domains.world.monster_ai import (
    get_monster_ai, get_monster_tactic, should_one_shot,
    should_isolate, should_stun_first, is_night_aggro,
    is_mvp as monster_is_mvp,
)
from ai_sidecar.domains.social.party_synergy import PartySynergyEngine, get_party_synergy_engine
from ai_sidecar.woe.emperium_mechanics import get_emperium_mechanics
from ai_sidecar.woe.battlefield_awareness import get_battlefield_awareness
from ai_sidecar.woe.castle_intel import get_castle_intelligence
from ai_sidecar.prediction.self_learning_skill_predictor import SelfLearningSkillPredictor
from ai_sidecar.prediction.path_predictor import PathPredictor
from ai_sidecar.prediction.self_learning_spawn_tracker import SelfLearningSpawnTracker
from ai_sidecar.prediction.mvp_finisher import MvpFinisher
from ai_sidecar.prediction.server_tick_synchronizer import ServerTickSynchronizer
from ai_sidecar.anti_detection.bridge_wiring import get_bridge_wiring
from ai_sidecar.anti_detection.command_pacing import get_command_pacer
from ai_sidecar.anti_detection.session_profile import get_session_profiler

logger = logging.getLogger(__name__)


# Note: HeuristicAction is also defined in ai_sidecar.actions
@dataclass
class HeuristicAction(_HeuristicAction):
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

# ── Sellable junk items (name keyword -> item_id) ──
# Used by auto-sell system to identify and sell junk in town.
# Keys are lowercase substrings matched against inventory item names.
SELLABLE_JUNK: dict[str, str] = {
    "jellopy": "909",
    "sticky mucus": "938",
    "memento": "938",
    "green herb": "511",
    "apple": "512",
    "banana": "513",
    "grape": "514",
    "carrot": "515",
    "potato": "516",
    "meat": "517",
    "honey": "518",
    "empty bottle": "713",
    "feather": "715",
    "shell": "957",
    "dewdrop": "725",
    "sap": "948",
}

# ── Potion tier thresholds (min_level -> item_id, name, heal_amount) ──
POTION_TIERS: list[tuple[int, int, str, int]] = [
    (1,  501, "Red Potion", 45),
    (15, 502, "Orange Potion", 105),
    (30, 504, "White Potion", 250),
]

# ── Weight constants ──
NOVICE_WEIGHT_CAPACITY = 2000
POTION_WEIGHT = 1  # Each potion weighs 1
KNIFE_WEIGHT = 70

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
    Auto-persists state to disk via JSON for restart survival.
    """

    PERSISTENCE_DIR: str = "data/adaptive"

    def __init__(self, persistence_path: str | None = None):
        self._lock = threading.RLock()
        self._persistence_path = persistence_path or AdaptiveDataStore.PERSISTENCE_DIR
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

        # Load persisted state on startup
        self._load_state()

    # ── State persistence ──────────────────────────────────────────────

    def _persistence_file(self, name: str) -> str:
        """Return the full path for a persistence file."""
        p = Path(self._persistence_path)
        p.mkdir(parents=True, exist_ok=True)
        return str(p / name)

    def save_state(self) -> None:
        """Save adaptive data to disk so learning survives restarts."""
        try:
            import json, time
            with self._lock:
                data = {
                    "map_performance": self.map_performance,
                    "stat_effectiveness": self.stat_effectiveness,
                    "skill_priority": self.skill_priority,
                    "economy_data": self.economy_data,
                    "death_analysis": self.death_analysis,
                    # spawn_heatmap has tuple keys — convert to strings
                    "spawn_heatmap": {
                        m: {f"{x},{y}": c for (x, y), c in cells.items()}
                        for m, cells in self.spawn_heatmap.items()
                    },
                    # npc_locations: service -> map_name -> list of (x, y, name)
                    "npc_locations": self.npc_locations,
                    "_saved_at": time.time(),
                }
            path = self._persistence_file("adaptive_state.json")
            # Atomic write: write to temp, then rename
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2, default=str)
            Path(tmp).rename(path)
        except Exception as exc:
            logger.warning("AdaptiveDataStore.save_state failed: %s", exc)

    def load_state(self) -> None:
        """Load previously-saved adaptive state from disk."""
        self._load_state()

    def _load_state(self) -> None:
        """Internal: load state from disk on construction."""
        try:
            import json
            path = self._persistence_file("adaptive_state.json")
            if not Path(path).exists():
                return
            with open(path) as f:
                data = json.load(f)
            with self._lock:
                self.map_performance.update(data.get("map_performance", {}))
                self.stat_effectiveness.update(data.get("stat_effectiveness", {}))
                self.skill_priority.update(data.get("skill_priority", {}))
                self.economy_data.update(data.get("economy_data", {}))
                self.death_analysis.update(data.get("death_analysis", {}))
                # Restore spawn_heatmap from string keys back to tuples
                for m, cells in data.get("spawn_heatmap", {}).items():
                    self.spawn_heatmap.setdefault(m, {})
                    for skey, count in cells.items():
                        parts = skey.split(",")
                        if len(parts) == 2:
                            self.spawn_heatmap[m][(int(parts[0]), int(parts[1]))] = count
                # Restore npc_locations
                for service, maps in data.get("npc_locations", {}).items():
                    self.npc_locations.setdefault(service, {})
                    for map_name, npcs in maps.items():
                        self.npc_locations[service].setdefault(map_name, [])
                        for npc in npcs:
                            if isinstance(npc, list) and len(npc) >= 3:
                                self.npc_locations[service][map_name].append((npc[0], npc[1], npc[2]))
            logger.info("AdaptiveDataStore loaded from %s", path)
        except Exception as exc:
            logger.warning("AdaptiveDataStore._load_state failed: %s", exc)

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
        Uses renewal damage formula from ro_mechanics.
        """
        _mn = monster_name.lower().strip()
        stats = get_monster_stats(_mn)
        if not stats:
            return 0.0
        hp = stats["hp"]
        _def = stats.get("def", 0)
        _dmg_per_hit = calculate_damage(attack_power, defense=_def)
        _hits_needed = hp / max(_dmg_per_hit, 1)
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
        """Calculate character hit rate using renewal formula."""
        return calculate_hit(base_level, dex)

    def calculate_damage(self, attack_power: int, monster_def: int, weapon_type: str = "dagger",
                         monster_size: str = "Medium", attack_element: str = "Neutral",
                         monster_element: str = "Neutral", monster_race: str = "Brute",
                         skill_mult: float = 1.0) -> int:
        """Delegate to ro_mechanics (uses renewal formula, 4-level element table)."""
        return calculate_damage(
            attack_power,
            defense=monster_def,
            weapon_type=weapon_type,
            target_size=monster_size,
            attack_element=attack_element,
            target_element=monster_element,
            card_mod=skill_mult,
        )

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
        """Get the best hunting map for this bot's level from adaptive data.

        Only returns FARM-CAPABLE maps (fields/dungeons). Intro islands (int_land),
        towns, and other non-farm maps recorded in map_performance are excluded —
        otherwise adaptive data pollution makes the bot "farm" a no-mob intro map.
        """
        with self._lock:
            if not self.map_performance:
                return None
            candidates = []
            for map_name, perf in self.map_performance.items():
                _n = str(map_name or "").lower().replace(".gat", "")
                # FARM GATE: only actual hunting maps. Intro islands / towns /
                # rooms / dungeons with no farmable spawns are never valid targets.
                if "_fild" not in _n and "_dun" not in _n:
                    continue
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

    PERSISTENCE_DIR: str = "data/heuristic"

    def __init__(self, adaptive: AdaptiveDataStore | None = None, persistence_path: str | None = None):
        self._adaptive = adaptive or AdaptiveDataStore()
        self._persistence_path = persistence_path or HeuristicService.PERSISTENCE_DIR
        self._last_assessment: dict[str, HeuristicAssessment] = {}
        self._bot_state: dict[str, str] = {}
        self._state_since: dict[str, float] = {}
        self._last_progress: dict[str, dict] = {}
        self._last_sell_time: dict[str, float] = {}
        self._last_buy_time: dict[str, float] = {}
        self._last_economy_emit: dict[str, float] = {}  # rate-limit cold-start potion economy
        self._last_weapon_buy: dict[str, float] = {}  # rate-limit WEAPON_BUY shop-atom
        self._last_mon_emit: dict[str, float] = {}  # A2: rate-limit inline mon_control (resets client AI)
        self._last_academy_room_move: dict[str, float] = {}
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
        self._last_return_to_farm: dict[str, float] = {}  # return-to-farm after restock cooldown
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
        # Stuck detector: track last known position and last move time per bot
        self._last_position: dict[str, tuple[int, int, float]] = {}
        self._last_move_time: dict[str, float] = {}
        # Sell tracking: bot_id -> {item_id: last_sell_timestamp}
        self._sold_items: dict[str, dict[str, float]] = {}
        # Potion tier cache: bot_id -> current potion item_id being bought
        self._potion_tier: dict[str, int] = {}
        # Per-map mon_control tracking: bot_id -> {map_name: [(monster, attack, lvl, aggr)]}
        # Used to avoid re-sending mon_control for the same map
        self._last_mon_control_map: dict[str, str] = {}  # bot_id -> map_name
        # Secluded Island escape throttle: bot_id -> last escape-attempt epoch
        # The escape (move 49 57 + talknpc 49 57) fires at most once per ESCAPE
        # interval so the bot spends the rest of its cycles grinding the weak
        # island Porings (exp + 6008 quest items + Knife drops) instead of
        # constantly re-targeting the sail attempt every horizon.
        self._last_island_escape: dict[str, float] = {}
        # Academy registration attempt count: bot_id -> int. Once a bot has
        # attempted the Receptionist registration a few times, stop re-talking —
        # the snapshot's inventory/job signals can be stale (bot4 had the knife +
        # job_level 2 in the DB but the snapshot showed empty/1), so relying on
        # them alone left the bot looping at the receptionist. Cap the attempts
        # so the bot always proceeds to farming after a bounded number of tries.
        self._reg_attempts: dict[str, int] = {}
        # Academy-door walk throttle: bot_id -> last time we issued move 125 257.
        # The academy move (izlude -> iz_ac01 door) must be issued at most once
        # per ~10s, not every PDCA cycle, so OpenKore actually walks the long
        # route instead of continuously recalculating a fresh path each cycle.
        self._last_academy_move: dict[str, float] = {}
        # Per-bot weapon latch: once a weapon/kit is observed (or registered), it
        # is latched so an intermittent inventory snapshot never reverts the bot to
        # "weapon-less" (which ping-pongs it back to the academy door).
        self._weapon_latch: dict[str, bool] = {}
        # Per-bot cold-start hunt map. Was a SCALAR (audit finding #6): bot A
        # (izlude_c) set prt_fild08c, bot B (prt_fild05) overwrote it, so bot A's
        # transit-protection + step-1 authority used B's wrong map. Now keyed by
        # stable bot id so concurrent bots don't clobber each other's target.
        self._cold_start_hunt_map: dict[str, str] = {}
        # Save-point binding: bot_id -> set of town maps already bound there.
        # Once a bot visits a town's Kafra and binds its save point, it is NOT
        # re-bound on every visit (a one-shot per town). This keeps deaths/
        # respawns landing in a safe town, not a hostile start field.
        self._save_point_bound: dict[str, set[str]] = {}
        # Job change tracking: detect when job actually changes
        self._last_job_name: dict[str, str] = {}  # bot_id -> previous job_name
        # Post-job-change: track if we need to reset maps after job change
        self._post_job_change_reset: dict[str, bool] = {}  # bot_id -> True if job just changed
        # Mon_control dedup: bot_id -> set of (map, monster_tuple) already sent
        self._mon_control_sent: dict[str, set] = {}
        # Step timeout tracking: stable_key -> timestamp when step started
        self._cold_start_step_since: dict[str, float] = {}
        # Leader-decided team job assignments (persisted for crash recovery)
        self._team_levels: dict[str, int] = {}
        self._team_jobs_assigned: dict[str, bool] = {}
        self._last_job_change_time: dict[str, float] = {}
        self._assigned_jobs: dict[str, str] = {}  # profile -> assigned job class
        # Auto-save timer
        self._load_state()
        # ── Domain registry: loaded once, runs supplementary domains ──
        self._domain_registry: DomainRegistry | None = None
        self._state_collector: StateCollector | None = None
        self._new_domains_initialized: bool = False
        self._init_lock = threading.RLock()
        self._portal_db: PortalDB | None = None
        self._pathfinder: Pathfinder | None = None
        self._quest_tracker: QuestTracker | None = None
        self._equipment_manager: EquipmentManager | None = None
        self._swarm_coordinator: SwarmCoordinator | None = None
        self._lifecycle: LifecycleStateMachine | None = None
        self._experience_tracker: ExperienceTracker | None = None
        self._goal_manager: GoalManager | None = None
        self._task_scheduler: TaskScheduler | None = None
        # ── Unified Consciousness (conscious tier: novel-situation reasoning) ──
        self._consciousness: "UnifiedConsciousness | None" = None
        # ── Map rotation planner ──
        self._map_rotation: MapRotationPlanner | None = None
        # ── Danger predictor and safety evaluator ──
        self._danger_predictor: DangerPredictor | None = None
        # ── Game sense engine ──
        self._game_sense: GameSenseEngine = get_game_sense_engine()
        # ── Party synergy engine ──
        self._party_synergy: PartySynergyEngine = get_party_synergy_engine()

    # ── State persistence ──────────────────────────────────────────────

    def _persistence_file(self, name: str) -> str:
        """Return the full path for a persistence file."""
        p = Path(self._persistence_path)
        p.mkdir(parents=True, exist_ok=True)
        return str(p / name)

    def _serializable_dict(self, d: dict) -> dict:
        """Convert a dict with tuple values to JSON-safe format.
        Handles _last_position: str -> tuple[int, int, float] by
        serializing tuple values as lists.
        """
        result = {}
        for k, v in d.items():
            if isinstance(v, tuple):
                result[k] = list(v)
            elif isinstance(v, dict):
                result[k] = self._serializable_dict(v)
            else:
                result[k] = v
        return result

    def _deserialize_dict(self, d: dict) -> dict:
        """Reverse of _serializable_dict — restore tuples where needed."""
        result = {}
        for k, v in d.items():
            if isinstance(v, list) and k in ("_last_position",):
                result[k] = tuple(v)
            elif isinstance(v, dict):
                result[k] = self._deserialize_dict(v)
            else:
                result[k] = v
        return result

    def save_state(self) -> None:
        """Persist heuristic service state to disk so it survives restarts."""
        try:
            import json, time
            state = {
                "_bot_state": dict(self._bot_state),
                "_state_since": dict(self._state_since),
                "_last_progress": dict(self._last_progress),
                "_last_sell_time": dict(self._last_sell_time),
                "_last_buy_time": dict(self._last_buy_time),
                "_bot_deaths": dict(self._bot_deaths),
                "_cold_start_fired": dict(self._cold_start_fired),
                "_cold_start_step": dict(self._cold_start_step),
                "_town_entry_time": dict(self._town_entry_time),
                "_last_hunt_move": dict(self._last_hunt_move),
                "_last_return_to_town": dict(self._last_return_to_town),
                "_last_level": dict(self._last_level),
                "_last_party_attempt": dict(self._last_party_attempt),
                "_last_party_leave": dict(self._last_party_leave),
                "_last_party_seen": dict(self._last_party_seen),
                "_all_bots_cache": dict(self._all_bots_cache),
                "_last_force_return": dict(self._last_force_return),
                "_last_job_change_attempt": dict(self._last_job_change_attempt),
                "_last_lockmap": dict(self._last_lockmap),
                "_last_kills_count": dict(self._last_kills_count),
                "_last_kills_time": dict(self._last_kills_time),
                "_last_kills_log": dict(self._last_kills_log),
                "_first_cycle_done": dict(self._first_cycle_done),
                "_sit_start_time": dict(self._sit_start_time),
                "_last_position": self._serializable_dict(self._last_position),
                "_last_move_time": dict(self._last_move_time),
                "_potion_tier": dict(self._potion_tier),
                "_last_mon_control_map": dict(self._last_mon_control_map),
                "_last_job_name": dict(self._last_job_name),
                "_post_job_change_reset": dict(self._post_job_change_reset),
                "_team_levels": dict(self._team_levels),
                "_team_jobs_assigned": dict(self._team_jobs_assigned),
                "_last_job_change_time": dict(self._last_job_change_time),
                "_weapon_latch": dict(self._weapon_latch),
            "_assigned_jobs": dict(self._assigned_jobs),
                "_assigned_jobs": dict(self._assigned_jobs),
                "_saved_at": time.time(),
            }
            # Also persist the adaptive data store
            self._adaptive.save_state()
            path = self._persistence_file("heuristic_state.json")
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(state, f, indent=2, default=str)
            Path(tmp).rename(path)
        except Exception as exc:
            logger.warning("HeuristicService.save_state failed: %s", exc)

    def load_state(self) -> None:
        """Load previously-saved heuristic state from disk."""
        self._load_state()

    def _load_state(self) -> None:
        """Internal: load state from disk on construction."""
        try:
            import json
            path = self._persistence_file("heuristic_state.json")
            if not Path(path).exists():
                return
            with open(path) as f:
                data = json.load(f)
            mappings = [
                ("_bot_state",), ("_state_since",), ("_last_progress",),
                ("_last_sell_time",), ("_last_buy_time",), ("_bot_deaths",),
                ("_cold_start_fired",), ("_cold_start_step",),
                ("_town_entry_time",), ("_last_hunt_move",),
                ("_last_return_to_town",), ("_last_level",),
                ("_last_party_attempt",), ("_last_party_leave",),
                ("_last_party_seen",), ("_all_bots_cache",),
                ("_last_force_return",), ("_last_job_change_attempt",),
                ("_last_lockmap",), ("_last_kills_count",),
                ("_last_kills_time",), ("_last_kills_log",),
                ("_first_cycle_done",), ("_sit_start_time",),
                ("_last_move_time",), ("_potion_tier",),
                ("_last_mon_control_map",), ("_last_job_name",),
                ("_post_job_change_reset",),
                ("_weapon_latch",),
            ]
            for key in [m[0] for m in mappings]:
                if key in data:
                    setattr(self, key, dict(data[key]))
            # Restore _last_position (lists -> tuples)
            if "_last_position" in data:
                self._last_position = self._deserialize_dict(data["_last_position"])
            logger.info("HeuristicService loaded state from %s", path)
        except Exception as exc:
            logger.warning("HeuristicService._load_state failed: %s", exc)

    def _get_npc(self, task_type: str, map_name: str) -> dict | None:
        """Thread-safe NPC lookup - creates new DB connection per call."""
        try:
            gkd = GameKnowledgeDB()
            return gkd.find_npc_for_task(task_type, map_name)
        except Exception:
            return None

    def _resolve_safe_town(self) -> str:
        """Resolve a safe town to portal-walk to, AGNOSTICALLY (RC3a).

        Prefers the learned server_solutions safe_town, else a known town from the
        DB-loaded town set, else the true last-resort 'prontera'. Never emits a
        broken map prefix (e.g. the 'prt' of prt_fild05 is NOT a town).
        """
        _town = ""
        try:
            from ai_sidecar.server_adaptation import get_server_solutions_store
            _town = str(get_server_solutions_store().get("safe_town", None) or "")
        except Exception:
            _town = ""
        if not _town:
            # Use a real town from the DB-loaded set (server-agnostic), never a
            # field-map prefix.
            _towns = [t for t in _HUNT_TOWNS if isinstance(t, str)]
            _town = _towns[0] if _towns else "prontera"
        return _town

    def _load_towns(self) -> None:
        """Load town map names from database (server-agnostic).

        Uses the npc_interactions table's `task_type` column (the schema
        column; the old query wrongly referenced a non-existent
        `interaction_type` column, raising "no such column" every startup and
        silently falling back to the hardcoded set). Falls back to the known
        town prefixes if the DB has no town-flag rows yet.
        """
        global _HUNT_TOWNS
        try:
            gkd = GameKnowledgeDB()
            conn = gkd._get_conn()
            rows = conn.execute(
                "SELECT map_name FROM npc_interactions WHERE task_type='town_flag'"
            ).fetchall()
            if rows:
                _HUNT_TOWNS = {row["map_name"] for row in rows}
        except Exception:
            logger.debug("_load_towns DB query failed; using prefix fallback", exc_info=True)
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

        # JOB CHANGE DETECTION: track job transitions for post-job-change reset
        _prev_job = self._last_job_name.get(bot_id, "")
        if _prev_job and _prev_job != job_name:
            # Job changed! Set post-job-change flag so HUNT/JOB_CHANGE states can reset maps
            self._post_job_change_reset[bot_id] = True
            logger.info(f"[job_change_detect] {bot_id}: {_prev_job} -> {job_name} (post-job-change reset queued)")
        self._last_job_name[bot_id] = job_name

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
            # 2-1 JOB CHANGE: first class with job_level >= 50 => change to 2nd class
            _first_classes = {"swordman", "mage", "archer", "acolyte", "merchant", "thief", "taekwon", "gunslinger", "ninja", "soul_linker"}
            if job_name in _first_classes and job_level >= 50 and base_level >= 50:
                return "JOB_CHANGE"
            if stat_points > 0:
                return "STATS"
            # If no stat points, skip STATS entirely to avoid wasted cycles
            if skill_points > 0:
                return "SKILLS"
            # SOLO-BOT GUARD: a fleet with only ONE bot (no teammates to party
            # with) must NOT enter the PARTY state machine — it would emit party
            # ops forever and never reach TOWN_HUNT/HUNT (the academy/farm
            # progression). Only enter PARTY when there are other bots to join.
            _fleet_bots = signals.get("all_bots", []) or []
            if not in_party and len(_fleet_bots) <= 1:
                return "TOWN_HUNT"
            if not in_party:
                return "PARTY"
            return "TOWN_HUNT"

        # HUNTING
        return "HUNT"

    def _check_progress(self, signals: dict) -> bool:
        bot_id = signals.get("bot_id", "default")
        _stable = bot_id.split(":")[-1].split("/")[-1] if ":" in bot_id else bot_id
        last = self._last_progress.get(_stable, {})
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
        self._last_progress[_stable] = now
        if not last:
            return True
        # Check multiple progress indicators
        for key in ("exp", "zeny", "level", "kills", "job_level", "items"):
            if now.get(key, 0) > last.get(key, 0):
                return True
        return False

    def _get_potion_name(self, potion_id: int) -> str:
        """Return the display name for a potion item — AGNOSTIC (DB-backed)."""
        try:
            from ai_sidecar.knowledge_loader import get_items
            for _it in get_items():
                if str(_it.get("Id", "")) == str(potion_id):
                    _nm = str(_it.get("Name", "") or "")
                    if _nm:
                        return _nm.replace(" Potion", "")
        except Exception:
            pass
        return {501: "Red", 502: "Orange", 504: "White"}.get(potion_id, "Red")

    def _get_potion_id(self, base_level: int) -> int:
        """Return the best potion item ID for a given base level.
        Scales from Red (501) -> Orange (502) -> White (504) as level increases."""
        if base_level < 15:
            return 501  # Red Potion (heals 45 HP)
        elif base_level < 30:
            return 502  # Orange Potion (heals 105 HP)
        else:
            return 504  # White Potion (heals 250 HP)

    def _get_potion_cost(self, potion_id: int) -> int:
        """Return the cost of a potion by item ID — AGNOSTIC (DB-backed, RULE.md).

        Reads the REAL buy price from the server's knowledge DB; falls back to a
        conservative estimate only if the DB is unavailable. Never hardcodes prices
        (RAW real prices differ: Red 501=10z, Orange 502=50z, White 504=1200z).
        """
        try:
            from ai_sidecar.knowledge_loader import get_items
            for _it in get_items():
                if str(_it.get("Id", "")) == str(potion_id) or str(_it.get("id", "")) == str(potion_id):
                    _cost = int(_it.get("Buy", 0) or 0)
                    if _cost > 0:
                        return _cost
        except Exception:
            pass
        # Fallback only (server DB unavailable) — level-scaled estimates.
        return {501: 10, 502: 50, 504: 1200}.get(potion_id, 50)

    def _get_potion_max_buy(self, potion_id: int, zeny: int, weight: float, weight_capacity: int) -> int:
        """Calculate max potions to buy considering zeny and remaining weight capacity.
        weight is a ratio 0.0-1.0. weight_capacity is in raw units."""
        _cost_each = self._get_potion_cost(potion_id)
        _max_by_zeny = zeny // _cost_each if _cost_each > 0 else 0
        _remaining_weight_ratio = max(0.0, 1.0 - weight)
        _remaining_weight_units = _remaining_weight_ratio * weight_capacity
        _max_by_weight = int(_remaining_weight_units // POTION_WEIGHT)
        _max_buy = min(_max_by_zeny, _max_by_weight, 30)  # Cap at 30 total
        return max(0, _max_buy)

    def _emit_mon_control_for_map(self, actions: list, bot_id: str, map_name: str, base_level: int = 1) -> None:
        """Emit mon_control commands for the current map.
        
        Uses PER_MAP_MON_CONTROL table to set per-monster attack/ignore behavior.
        Only emits for maps that have entries in the table.
        Dedup: only emits when map changes, not every cycle.
        """
        _map = map_name.lower().replace(".gat", "")
        controls = PER_MAP_MON_CONTROL.get(_map)
        if not controls:
            return
        # EMIT-ON-MAP-CHANGE ONLY (completeness fix): the earlier GLOBAL 60s rate-limit
        # here re-emitted mon_control on a 60s timer even on a STABLE farm map — and every
        # mon_control apply makes the OpenKore client RESTART its AI (state 2), cancelling
        # in-progress kills. That periodic reset is what still slowed compounding (45
        # restarts). The CORRECT gate is the map-dedup below: emit mon_control ONLY when
        # the bot's actual map CHANGES (which genuinely needs a fresh set, e.g.
        # prt_fild08c<->prt_fild08). On a stable map, nothing re-emits -> zero AI resets.
        # (handled by _last_mon_control_map check below)
        # F3 (user directive): the attack decision must come from the GAME DB, not a
        # hardcoded 'always attack' tuple. Look up each monster in the real mob_db.yml
        # (via MonsterDB) — downgrade to ignore if it's a BOSS or far above the farmable
        # range, so mon_control reflects the game's actual data rather than literals.
        try:
            from ai_sidecar.combat.target_engine import MonsterDB
            _mdb = MonsterDB()
            _mdb.load()
        except Exception:
            _mdb = None
        _filtered: list = []
        for _monster, _attack, _lvl, _aggr in controls:
            _use = (_attack, _lvl, _aggr)
            if _mdb is not None:
                _md = _mdb.lookup(_monster)
                if _md is not None:
                    _boss = getattr(_md, "modes", []) and ("boss" in (_md.modes or []))
                    _lvl_ = int(getattr(_md, "level", 1) or 1)
                    _atk_ = int(getattr(_md, "attack", 0) or 0)
                    if _boss or _lvl_ > 40:
                        _use = (0, 0, 0)  # game data says too strong / boss — do NOT attack
                    else:
                        # SURVIVABILITY GATE (Task-1 ground truth): RAW has ZERO aggressive
                        # mobs, so the real danger model is mob ATK vs the bot's HP. A
                        # low-level bot (base_level <= 5) must only attack mobs it can
                        # out-trade (atk <= 8 = Poring/Lunatic atk1, Fabre atk17 is a
                        # one-hit risk barehanded). DB-backed, agnostic — no hardcoded
                        # mob names. Higher-level bots may attack stronger mobs.
                        _bl = int(base_level or 1)
                        if _bl <= 5 and _atk_ > 8:
                            _use = (0, 0, 0)  # too strong for a low-level bot — do NOT attack
                        else:
                            _use = (1, 0, 1)  # game data confirms farmable — attack
            _filtered.append((_monster, _use[0], _use[1], _use[2]))
        if not _filtered:
            return
        # Check if we already sent mon_control for this map
        # Normalize farm sub-map variants (prt_fild08c and prt_fild08 are the SAME farm)
        # so a snapshot map-flip between them does NOT re-emit mon_control and reset the
        # client's combat AI (which cancels in-progress kills). Only a TRUE map change
        # (different farm/field) triggers a fresh emission.
        _dedup_map = _map
        if _dedup_map.endswith(("c", "_c", "a", "_a")) and _dedup_map[:-1] in PER_MAP_MON_CONTROL:
            _dedup_map = _dedup_map[:-1]  # prt_fild08c -> prt_fild08 (same farm)
        _last_map = self._last_mon_control_map.get(bot_id, "")
        if _last_map == _dedup_map:
            return  # Already sent for this (farm) map
        # DIAG: log when re-emitting despite dedup (to catch the reset/flip)
        logger.debug("mon_control_reemit bot=%s map=%s dedup_key=%s last=%s last_mon_emit_age=%.0f",
                     bot_id, _map, _dedup_map, _last_map,
                     __import__("time").time() - self._last_mon_emit.get(bot_id, 0.0))
        self._last_mon_control_map[bot_id] = _dedup_map
        logger.info(f"[mon_control] {bot_id}: applying {len(_filtered)} entries for {_map}")
        for _monster, _attack, _lvl, _aggr in _filtered:
            actions.append(HeuristicAction(
                kind="command",
                command=f"mon_control {_monster}\t{_attack} {_lvl} {_aggr}",
                confidence=0.95, domain="hunting",
                reason=f"Per-map mon_control (game-DB checked): {_monster} -> attack={_attack} on {_map}",
            ))

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

    def _ensure_save_point(self, actions: list, bot_id: str, map_name: str) -> None:
        """Bind the bot's save point at the nearest town Kafra, one-shot per town.

        If the bot's current map is a town with a known Kafra and it has NOT yet
        bound its save point there, emit the Kafra save interaction so deaths /
        respawns (and Butterfly Wing returns) land in a SAFE town instead of a
        hostile start field. This is the highest-efficiency save-point policy: bind
        early (first town visit), never re-bind the same town, and never spam.

        Kafra save = `talknpc <x> <y>` then menu option "Save" (r0) — per
        ro_knowledge: this server's Kafra offers Save (r0) and Storage (r1).
        """
        _map = str(map_name or "").lower().replace(".gat", "")
        if not _map:
            return
        try:
            from ai_sidecar.kafra_teleport import KNOWN_KAFRA_LOCATIONS
        except Exception:
            KNOWN_KAFRA_LOCATIONS = {}
        coords = KNOWN_KAFRA_LOCATIONS.get(_map)
        if not coords:
            return  # not a Kafra town
        _bound = self._save_point_bound.get(bot_id, set())
        if _map in _bound:
            return  # already bound here — don't re-bind
        _bound = set(_bound)
        _bound.add(_map)
        self._save_point_bound[bot_id] = _bound
        # Also record into the Kafra teleport manager so the sidecar knows the
        # authoritative save point for return-to-save / butterfly logic.
        try:
            from ai_sidecar.kafra_teleport import get_kafra_manager
            get_kafra_manager().set_save_point(bot_id, _map, int(coords[0]), int(coords[1]))
        except Exception:
            pass
        _x, _y = int(coords[0]), int(coords[1])
        actions.append(HeuristicAction(
            kind="command",
            command=f"talknpc {_x} {_y}",
            confidence=0.95,
            reason=f"Adaptive: bind save point at {_map} Kafra ({_x},{_y}) so deaths respawn safely",
            domain="navigation",
        ))
        actions.append(HeuristicAction(
            kind="command",
            command="talk resp 0",
            confidence=0.90,
            reason="Adaptive: choose Kafra 'Save' to bind the respawn point",
            domain="navigation",
        ))
        logger.info("save_point: %s bound at %s Kafra (%d,%d)", bot_id, _map, _x, _y)

    def set_domain_weights(self, weights: dict) -> None:
        """Override domain execution priority weights.

        Delegates to the DomainRegistry's set_weights: effective priority =
        base_priority / weight, so heavier-weighted domains run earlier in the
        assessment chain. Invalid entries are ignored.
        """
        try:
            self.domain_registry.set_weights(dict(weights or {}))
        except Exception:
            logger.exception("set_domain_weights_failed")

    @property
    def domain_registry(self) -> DomainRegistry:
        """Lazy-load the domain registry on first access."""
        if self._domain_registry is None:
            self._domain_registry = DomainRegistry()
            self._domain_registry.load_all()
        return self._domain_registry


    def _init_new_domains(self) -> None:
        if self._new_domains_initialized:
            return
        with self._init_lock:
            # Double-check after acquiring lock
            if self._new_domains_initialized:
                return
            try:
                self._portal_db = PortalDB()
                self._pathfinder = Pathfinder()
                self._quest_tracker = QuestTracker(db_path="data/quests.db")
                self._equipment_manager = EquipmentManager()
                self._lifecycle = LifecycleStateMachine()
                self._experience_tracker = ExperienceTracker(db_path="data/learning.db")
                self._goal_manager = GoalManager(bot_id="default")
                self._task_scheduler = TaskScheduler()
                self._task_translator = TaskCommandTranslator()
                # Wire the goal manager into the scheduler (quest/job tasks)
                self._task_scheduler.set_goal_manager(self._goal_manager)
                self._swarm_coordinator = SwarmCoordinator(
                    bot_names=["kicapmasin", "kicapmasin2", "kicapmasin3"],
                    data_dir="data",
                )
                self._state_collector = StateCollector()
                self._map_rotation = MapRotationPlanner()
                self._danger_predictor = DangerPredictor()
                self._world_state = get_world_state()
                # Conscious tier: unified reasoning across the strategic sub-systems.
                # Instantiated lazily; if init fails (missing optional deps) the
                # heuristic continues with reflex rules only.
                self._consciousness = None
                try:
                    self._consciousness = UnifiedConsciousness()
                except Exception as _uc_err:
                    self._consciousness = None
                self._combat_pressure = CombatPressureDomain()
                self._kiting_v2 = TickBasedKiting()
                self._inventory_policies = InventoryPolicies()
                self._spawn_navigator = SpawnNavigator()
                self._combo_protocol = ComboHandshakeProtocol()
                self._danger_pathfinder = DangerAwarePathfinder()
                self._loadout_planner = ConsumableLoadoutPlanner()
                self._durability_monitor = DurabilityMonitor()
                self._post_mortem = PostMortemAnalyzer()
                self._loot_discipline = LootDisciplineEngine()
                self._event_detector = EventDetector()
                self._live_market = LiveMarketScanner()
                self._social_reputation = SocialReputationDomain()
                self._farming_loop = FarmingLoopOptimizer()
                # ── NEW DOMAIN MODULES ──
                for _mod_name, _import_path, _attr_name in [
                    ("ProfitabilityDomain", "ai_sidecar.domains.economy.profitability", "_profitability"),
                    ("WoEDomain", "ai_sidecar.domains.pvp.woe", "_woe"),
                    ("PartyDomain", "ai_sidecar.domains.social.party", "_party_domain"),
                    ("EquipmentOptimizer", "ai_sidecar.domains.equipment.optimizer", "_equipment_optimizer"),
                    ("MacroIntelligence", "ai_sidecar.autonomy.macro_intelligence", "_macro_intelligence"),
                    ("LifecycleManager", "ai_sidecar.domains.progression.lifecycle", "_connection_lifecycle"),
                    ("ColdStartManager", "ai_sidecar.domains.progression.cold_start", "_cold_start_manager"),
                    ("CompetitionAwareFarming", "ai_sidecar.domains.farming.competition", "_competition_planner"),
                    ("SelfLearningSkillPredictor", "ai_sidecar.prediction.self_learning_skill_predictor", "_skill_predictor_internal"),
                    ("PathPredictor", "ai_sidecar.prediction.path_predictor", "_path_predictor_internal"),
                    ("SelfLearningSpawnTracker", "ai_sidecar.prediction.self_learning_spawn_tracker", "_spawn_tracker_internal"),
                    ("MvpFinisher", "ai_sidecar.prediction.mvp_finisher", "_mvp_finisher_internal"),
                    ("ServerTickSynchronizer", "ai_sidecar.prediction.server_tick_synchronizer", "_tick_sync_internal"),
                ]:
                    try:
                        import importlib
                        _mod = importlib.import_module(_import_path)
                        _cls = getattr(_mod, _mod_name)
                        setattr(self, _attr_name, _cls())
                        logger.info(f"{_mod_name} initialized")
                    except Exception:
                        setattr(self, _attr_name, None)
                # BehaviorEngine: singleton, not instance-based
                try:
                    from ai_sidecar.anti_detection.behavior_engine import get_behavior_engine
                    self._behavior_engine = get_behavior_engine()
                    logger.info("BehaviorEngine initialized")
                except Exception:
                    self._behavior_engine = None
                self._new_domains_initialized = True
                logger.info("New domain modules initialized")
            except Exception as e:
                logger.warning(f"New domain init failed (non-fatal, will retry): {e}")
                self._new_domains_initialized = False  # Will retry on next assess cycle

    def _resolve_bot_id(self, signals: dict[str, Any]) -> str:
        """Extract a stable bot identifier from signals.
        
        Used by domain modules to access service state dicts.
        Strips account prefixes for stable cross-cycle tracking.
        """
        bot_id = signals.get("bot_id", "default")
        if ":" in bot_id:
            return bot_id.split(":")[-1].split("/")[-1]
        return bot_id

    def assess(self, signals: dict[str, Any], bot_id_override: str | None = None) -> HeuristicAssessment:
        try:
            assessment = self._assess_impl(signals, bot_id_override)
            # ── ACADEMY-ROOM EXIT GUARD (universal, sidecar decision) ──
            # iz_ac01_a is the academy tutorial ROOM with no portal to any field map,
            # so any farm-bound bot there spams "Cannot calculate a route". Regardless
            # of the state-gated sub-branch, a bot whose current map is iz_ac01_a must
            # walk to the room's EXIT warp (iz_ac01_a 100,24 -> izlude_a 127,253) first.
            # Emit that move (throttled 10s) so the bot always escapes the room. This is
            # a SIDECAR decision — the bridge only passes the command (RULE.md).
            try:
                _amap = str(signals.get("map", "") or "").lower().replace(".gat", "")
                if _amap == "iz_ac01_a":
                    _abid = bot_id_override or str(signals.get("bot_id", "default"))
                    _art = __import__("time").time()
                    if _art - self._last_academy_room_move.get(_abid, 0.0) >= 10.0:
                        self._last_academy_room_move[_abid] = _art
                        # Force deterministic routing (not wandering randomWalk) so the bot
                        # walks the direct path to the exit instead of looping the pathfinder.
                        _guard_norw = HeuristicAction(
                            kind="command", command="set route_randomWalk 0",
                            confidence=0.99, domain="hunting",
                            reason="Academy room (iz_ac01_a): deterministic route to exit, not randomWalk",
                        )
                        _guard = HeuristicAction(
                            kind="command", command="move 100 24",
                            confidence=0.99, domain="hunting",
                            reason="Academy room (iz_ac01_a): walk to room exit (100,24 -> izlude_a) so the farm map is reachable",
                        )
                        # Prepend BOTH so the exit walk is unambiguous and deterministic.
                        # Order: first disable randomWalk, THEN walk to the exit.
                        assessment.actions.insert(0, _guard)
                        assessment.actions.insert(0, _guard_norw)
            except Exception:
                pass
            # ── SUPPLEMENTARY DOMAINS: only run when in-game (not during login) ──
            # Bot is in-game only if map_known=True (from snapshot) or base_level > 1
            _bl = signals.get("base_level", 0) or 0
            _is_ingame = signals.get("map_known", False) or (isinstance(_bl, (int, float)) and _bl > 1)
            supplementary: list[HeuristicAction] = []
            if _is_ingame:
                self.domain_registry.assess_all(signals, supplementary, self)
            if supplementary:
                if not assessment.actions:
                    assessment.actions = supplementary
                else:
                    assessment.actions.extend(supplementary)
                assessment.actionable = True
                # Boost confidence from supplementary actions if higher
                sup_conf = max(a.confidence for a in supplementary)
                if sup_conf > assessment.confidence:
                    assessment.confidence = sup_conf
            # ── NEW DOMAIN MODULE DELEGATION (runs for ALL states) ──
            _bot_id = bot_id_override or signals.get("bot_id", "default")
            _actions = assessment.actions
            _bl = int(signals.get("base_level", 1) or 1)
            self._init_new_domains()
            if self._new_domains_initialized and self._state_collector:
                try:
                    _gs = self._state_collector.collect(signals)
                    _map = _gs.map_state.name if _gs.map_state else ""
                    if _map and ('prt_fild' in _map or 'pay_fild' in _map or 'gef_fild' in _map):
                        try:
                            _disp = TacticsDispatcher()
                            _disp.assess(signals, _actions, _bot_id)
                        except Exception:
                            pass
                    if self._quest_tracker:
                        _qa = self._quest_tracker.get_quests_near_completion(_bot_id)
                        if _qa:
                            _actions.append(HeuristicAction(kind="log", command=f"quests_near_complete={len(_qa)}", confidence=0.5, reason="Quests near completion", domain="quests"))
                    if self._experience_tracker:
                        self._experience_tracker.record_kill(signals.get("map",""), _bot_id)
                    # ── Connection Lifecycle: heartbeat every cycle ──
                    if hasattr(self, '_connection_lifecycle') and self._connection_lifecycle:
                        self._connection_lifecycle.heartbeat(_bot_id)
                    # ── Cold Start: create character if none exists ──
                    if hasattr(self, '_cold_start_manager') and self._cold_start_manager:
                        self._cold_start_manager.assess(signals, _actions, _bot_id)
                    if self._equipment_manager:
                        _eq = self._equipment_manager.assess_equipment(signals, _bot_id)
                        if _eq:
                            _actions.append(HeuristicAction(kind="log", command=f"equipment={_eq}", confidence=0.5, reason="Equipment assessment", domain="equipment"))
                    if self._portal_db and self._pathfinder and _gs.map_state:
                        _cm = _gs.map_state.name
                        if _gs.character and hasattr(_gs.character, 'base_level') and _gs.character.base_level:
                            # AGNOSTIC farm-target resolution (RC3a): prefer the learned
                            # server_solutions farm_map, else the portal-graph + level-scored
                            # reachable_hunting_maps resolver, else fall back to the adaptive
                            # get_best_map. No hardcoded map literal decides the farm.
                            _target = ""
                            _bl_c = int(_gs.character.base_level or 1)
                            try:
                                from ai_sidecar.server_adaptation import get_server_solutions_store
                                _learned = str(get_server_solutions_store().get("farm_map", None) or "")
                                if _learned:
                                    # Use the learned farm ONLY if it's reachable from the
                                    # current map (portal-graph verified). A learned farm that
                                    # is a 2-hop clone (e.g. prt_fild08c from izlude) is
                                    # unroutable and would spin "Cannot calculate a route".
                                    _reach_set = {m for m, _s in _mk_reachable_hunting_maps(str(_cm or ""), _bl_c)}
                                    if _learned in _reach_set:
                                        _target = _learned
                            except Exception:
                                _target = ""
                            if not _target:
                                try:
                                    _reach = _mk_reachable_hunting_maps(str(_cm or ""), _bl_c)
                                    if _reach:
                                        _target = _reach[0][0]
                                except Exception:
                                    _target = ""
                            if not _target:
                                try:
                                    _adaptive = getattr(self, "_adaptive", None)
                                    if _adaptive is not None and hasattr(_adaptive, "get_best_map"):
                                        _target = str(_adaptive.get_best_map(_bot_id, _bl_c) or "")
                                except Exception:
                                    _target = ""
                            # CONSCIOUS/REFLEX FALLBACK: a weapon-less cold-start novice
                            # that just escaped to izlude must complete ACADEMY FIRST
                            # (free Novice_Knife + 300 potions at iz_ac01 receptionist)
                            # before any field hunt. Sending it to prt_fild05 to farm
                            # zeny bare-handed is a losing suggestion the reflexive
                            # map-default would make; override the hunt target to the
                            # academy door so registration (and real gear) follows.
                            _cs_map_l = str(_cm or "").lower()
                            if _cs_map_l in ("izlude", "izlude_a") and not self._has_coldstart_weapon(signals):
                                _target = "iz_ac01"
                                # Phase 1 (server-agnostic): prefer the discovered portal
                                # graph for routing to the academy. The Pathfinder computes
                                # a route through whatever portals this server actually has,
                                # so it works on ANY server — not just rathena-ai-world.
                                # The literal academy-door coordinate is only a fallback for
                                # a fresh server where the portal graph isn't discovered yet.
                                _route_via_pathfinder = False
                                if self._pathfinder:
                                    try:
                                        _apath = self._pathfinder.find_path(_cm, "iz_ac01")
                                        if _apath:
                                            _route_via_pathfinder = True
                                            _actions.append(HeuristicAction(
                                                kind="command", command=f"navigate iz_ac01",
                                                confidence=0.90,
                                                reason=f"Pathfinder (server-agnostic): {_cm} -> iz_ac01 via discovered portals",
                                                domain="routing"))
                                    except Exception:
                                        _route_via_pathfinder = False
                                if not _route_via_pathfinder:
                                    # Fallback: resolve the academy-door warp from the
                                    # portal graph (FACT, data-driven — not a hardcoded
                                    # coordinate literal). Emit at most once per 10s.
                                    _ac_door = self._resolve_academy_door(_cm or "")
                                    _now = __import__("time").time()
                                    if _ac_door and _now - self._last_academy_move.get(_bot_id, 0.0) >= 10.0:
                                        self._last_academy_move[_bot_id] = _now
                                        _actions.append(HeuristicAction(
                                            kind="command", command=f"move {_ac_door}",
                                            confidence=0.95,
                                            reason="Cold start: walk to Academy door (warp to iz_ac01) — register for starter kit (resolved from portal graph)",
                                            domain="progression"))
                            if _cm and _target and _cm != _target and _target != "iz_ac01":
                                # S18 STAY-COMMITTED GUARD: if the bot is ALREADY on a farm
                                # map it can actually farm (a map with correct mon_control for
                                # its spawned mobs), do NOT re-route it to a different target —
                                # the cold-start default (prt_fild05 for level<10) would yank a
                                # working bot off its current farm and break the kill loop. A
                                # Pro picks a field and works it; commit to the farm it's on.
                                _cm_farmable = str(_cm or "").lower().replace(".gat", "")
                                # AGNOSTIC S18 stay-committed check: a bot is already on a
                                # farmable hunting map if the portal-graph hunting-map resolver
                                # classifies it as a hunting map (never a hardcoded set).
                                _is_farmable = False
                                try:
                                    _is_farmable = _cm_farmable in {m for m, _s in _mk_get_hunting_maps(_bl_c)}
                                except Exception:
                                    _is_farmable = _cm_farmable in {
                                        "prt_fild08c", "prt_fild08", "prt_fild05", "prt_fild04"}
                                if _is_farmable:
                                    _actions.append(HeuristicAction(
                                        kind="log", command=f"commit_farm={_cm_farmable}",
                                        confidence=0.99, domain="routing",
                                        reason=f"Already on farm {_cm_farmable} with combat config — stay and farm (S18)",
                                    ))
                                elif _cm_farmable == "iz_ac01_a":
                                    # ACADEMY-ROOM GATE (R3, server-agnostic): a bot inside the
                                    # academy room (iz_ac01_a) has NO portal to any field map —
                                    # routing to prt_fild05 from here is unroutable and just loops
                                    # "Cannot calculate a route". The exit guard's deterministic
                                    # 'move 100 24' handles leaving the room FIRST.
                                    _actions.append(HeuristicAction(
                                        kind="log", command=f"defer_hunt={_target}",
                                        confidence=0.99, domain="routing",
                                        reason=f"Academy room ({_cm}): defer hunting until bot exits the room (exit guard handles it)",
                                    ))
                                else:
                                    _path = self._pathfinder.find_path(_cm, _target)
                                    if _path:
                                        _actions.append(HeuristicAction(kind="command", command=f"navigate {_target}", confidence=0.7, reason=f"Pathfinder: {_cm} -> {_target}", domain="routing"))
                    if self._lifecycle:
                        _phase = self._lifecycle.get_phase(_bot_id)
                        if _phase:
                            _actions.append(HeuristicAction(kind="log", command=f"lifecycle_phase={_phase}", confidence=0.5, reason=f"Phase: {_phase}", domain="progression"))
                    # ORCHESTRATOR DECOUPLING (O3): fleet coordination must NOT run inside
                    # the per-bot cycle. The former `SwarmCoordinator.tick(...)` here
                    # polluted a farming bot with swarm-admin actions every cycle,
                    # starving its movement (bot10 was flooded with heuristic_*_swarm).
                    # Fleet intent is now decided by the SEPARATE coarse-cadence
                    # FleetOrchestrator (fleet/fleet_orchestrator.py) on a slow pass,
                    # decoupled from per-bot execution.
                    # ── Monster AI: per-monster behavior overrides ──
                    try:
                        _mon_name = str(signals.get("monster_name", "") or "")
                        if not _mon_name:
                            # Try to extract from actors list
                            _actors_m = signals.get("actors", []) or []
                            _combat_m = signals.get("combat", {}) or {}
                            _target_id = _combat_m.get('target_id')
                            if _target_id and _actors_m:
                                for _a in _actors_m:
                                    if isinstance(_a, dict) and _a.get("id") == _target_id:
                                        _mon_name = str(_a.get("name", "") or "")
                                        break
                        if _mon_name:
                            _tactic = get_monster_tactic(_mon_name)
                            if _tactic in ("isolate", "stun_first"):
                                _actions.append(HeuristicAction(kind="combat", command=f"monster_behavior:{_tactic}", confidence=0.8, reason=f"{_mon_name} needs {_tactic}", domain="monster_ai"))
                            if should_one_shot(_mon_name):
                                _actions.append(HeuristicAction(kind="combat", command=f"one_shot:{_mon_name}", confidence=0.85, reason=f"{_mon_name} should be one-shot", domain="monster_ai"))
                    except Exception:
                        pass
                    # ── Game Sense: run every cycle in-game ──
                    try:
                        self._game_sense.assess(signals, _actions, _bot_id)
                    except Exception:
                        pass
                    # ── Party Synergy: run every cycle in-party ──
                    try:
                        self._party_synergy.assess(signals, _actions, _bot_id)
                    except Exception:
                        pass
                    # ── Prediction: skill predictor every cycle in-combat ──
                    try:
                        _sp = getattr(self, '_skill_predictor_internal', None)
                        _combat = signals.get('combat', {}) or {}
                        if _sp and _combat.get('is_in_combat', False):
                            # Scan actors list for casting monsters/players
                            _actors = signals.get('actors', []) or []
                            for _actor in _actors:
                                _cast = _actor.get('casting') if isinstance(_actor, dict) else None
                                if _cast:
                                    _pred = _sp.predict_skill(
                                        _cast.get('caster_class', '') or str(_actor.get('name', '')),
                                        float(_cast.get('time', 0) if isinstance(_cast, dict) else 0),
                                    )
                                    if _pred and _pred.predicted_skill:
                                        _actions.append(HeuristicAction(
                                            kind="reflex_override",
                                            command=f"prediction:incoming={_pred.predicted_skill} confidence={_pred.confidence:.2f}",
                                            confidence=min(1.0, _pred.confidence * 1.5),
                                            reason=f"Predicted incoming skill: {_pred.predicted_skill}",
                                            domain="prediction",
                                        ))
                    except Exception:
                        pass
                    # ── Prediction: server tick sync ──
                    try:
                        _ts = getattr(self, '_tick_sync_internal', None)
                        if _ts:
                            _ts.record_packet_interval(20.0)  # Default, learned over time
                            if _ts.should_send_now():
                                _actions.append(HeuristicAction(
                                    kind="timing", command="tick_aligned_send",
                                    confidence=0.6,
                                    reason="Tick-aligned send ready",
                                    domain="prediction",
                                ))
                    except Exception:
                        pass
                    # ── Anti-detection: bridge wiring ──
                    try:
                        _bw = get_bridge_wiring()
                        _cfg = _bw.get_config_push(_bot_id)
                        if _cfg:
                            _actions.append(HeuristicAction(
                                kind="config_push", command=f"anti_detection_config",
                                confidence=0.9,
                                reason="Anti-detection config push",
                                domain="anti_detection",
                                metadata=_cfg,
                            ))
                    except Exception:
                        pass
                    # ── Anti-detection: periodic anti-afk (every 300 cycles ≈5min) ──
                    try:
                        _afk_cycle = getattr(self, '_afk_cycle', 0) + 1
                        self._afk_cycle = _afk_cycle
                        if _afk_cycle % 300 == 0:
                            from ai_sidecar.anti_detection.anti_afk import get_anti_afk_engine
                            _afk = get_anti_afk_engine()
                            _afk_action = _afk.get_next_action(_bot_id)
                            if _afk_action:
                                # Convert AntiAfkAction to a bridge-consumable command string
                                _cmd = f"anti_afk:{_afk_action.action_type}"
                                if _afk_action.parameters:
                                    _params = ";".join(f"{k}={v}" for k, v in _afk_action.parameters.items())
                                    _cmd += f":{_params}"
                                _actions.append(HeuristicAction(
                                    kind="anti_afk", command=_cmd,
                                    confidence=0.3, domain="anti_detection",
                                    reason=f"Anti-AFK: {_afk_action.action_type}",
                                ))
                    except Exception:
                        pass
                    # ── Session Manager: anti-24/7 logout scheduling ──
                    try:
                        from ai_sidecar.autonomy.session_manager import get_session_manager
                        _sm = get_session_manager()
                        # Ensure bot is registered
                        if not _sm.is_online(_bot_id):
                            _sm.start_session(_bot_id)
                        if _sm.should_logout(_bot_id):
                            _actions.append(HeuristicAction(
                                kind="command", command="quit",
                                confidence=0.95, domain="session",
                                reason="Session time limit reached",
                            ))
                        elif _sm.should_rotate_map(_bot_id, str(signals.get("map", "") or ""), 0):
                            _actions.append(HeuristicAction(
                                kind="command", command="move_random",
                                confidence=0.7, domain="session",
                                reason="Map rotation due",
                            ))
                    except Exception:
                        pass
                    # ── Farming Routes: level-based route suggestion ──
                    try:
                        from ai_sidecar.domains.navigation.farming_routes import route_for_level
                        _bl = int(signals.get("vitals", {}).get("base_level", 1) or 1)
                        _route = route_for_level(_bl)
                        if _route:
                            _actions.append(HeuristicAction(
                                kind="log", command=f"recommend_route:{_route.map_name}",
                                confidence=0.5, domain="navigation",
                                reason=f"Level {_bl} route: {_route.map_name} ({_route.zeny_per_hour_estimate} z/h)",
                            ))
                    except Exception:
                        pass
                    if getattr(self, '_cold_start_manager', None):
                        _bl = int(signals.get("base_level", 1) or 1)
                        _job = str(signals.get("job", "") or "").lower()
                        # Basic Skill: needed by ALL Novices to sit/regen (any level)
                        if _job == "novice":
                            # Guard the Red Potion buy (item 501) so it fires only
                            # when actually needed: no potions in inventory AND has
                            # zeny AND not within the shared 60s buy cooldown.
                            # Previously emitted unconditionally every cycle — the
                            # bot spammed "Error in function 'buy' / Store Item 501
                            # does not exist" (the shop window isn't open for a bare
                            # `buy`), blocking real progress.
                            _inv_items = signals.get("inventory_items") or signals.get("items") or signals.get("inventory", {}).get("items", []) if isinstance(signals.get("inventory", {}), dict) else (signals.get("inventory_items") or [])
                            _has_potion = any(
                                "potion" in str(i).lower() or "red" in str(i).lower()
                                for i in _inv_items
                            )
                            _zeny = int(signals.get("zeny", 0) or 0)
                            _pb_now = __import__("time").time()
                            _pb_last = self._last_buy_time.get(_bot_id, 0)
                            if not _has_potion and _zeny >= 500 and (_pb_now - _pb_last >= 60):
                                self._last_buy_time[_bot_id] = _pb_now
                                _actions.append(HeuristicAction(kind="command", command="buy 501 30", confidence=0.95, reason=f"Cold start: learn Basic Skill for sit/regen", domain="progression"))
                        if _bl <= 15:
                            if _bl <= 5:
                                # ── ACADEMY REGISTRATION (rathena-ai-world starter gear) ──
                                # New characters spawn in the Izlude Academy (iz_int,
                                # char_athena.conf start_point). The Academy
                                # Receptionist (iz_ac01 100,39) gives the starter kit
                                # on registration: Novice_Knife (1243), Novice_Plate,
                                # Novice_Hood, Novice_Boots, + 300 Novice_Potion.
                                # Bare-hands a level-1 bot cannot kill a 55-HP Poring
                                # (12 dmg/hit), so registration is the required first
                                # step before prt_fild08 hunting.
                                _cur_map = str(signals.get("map", "") or "").lower()
                                # ── ACADEMY-ROOM EXIT GUARD (sidecar decision) ──
                                # iz_ac01_a is the academy tutorial ROOM — it has no
                                # portal to any field map, so `move <farm>` is unroutable
                                # there. Any bot stuck in this room must first walk to the
                                # room's EXIT warp (iz_ac01_a 100,24 -> izlude_a 127,253)
                                # before it can reach the farm. Emit that move (at most
                                # once per 10s) so a farm-bound bot always escapes the room
                                # regardless of which sub-branch it is in. The bridge only
                                # passes this command — the decision is made here (sidecar).
                                if _cur_map == "iz_ac01_a":
                                    _ac_room_t = __import__("time").time()
                                    if _ac_room_t - self._last_academy_room_move.get(_bot_id, 0.0) >= 10.0:
                                        self._last_academy_room_move[_bot_id] = _ac_room_t
                                        _actions.append(HeuristicAction(
                                            kind="command", command="move 100 24",
                                            confidence=0.99, domain="hunting",
                                            reason="Academy room (iz_ac01_a): walk to room exit (100,24 -> izlude_a) so the farm map is reachable",
                                        ))
                                # ── SECLUDED ISLAND BAILOUT (rathena-ai-world intro) ──
                                # New characters spawn on int_land (Secluded Island),
                                # a disconnected intro map: char_athena.conf
                                # start_point + #ship_out set the save point there, so
                                # login/respawn always lands at int_land(77,101). The
                                # only way out is the WARPNPC `#intro_to_izlude` at
                                # (49,57): walking onto it fires OnTouch -> a menu,
                                # option 2 "Sail to Izlude!" sails to the academy.
                                # OpenKore cannot path from int_land to prt_fild08
                                # (no direct portal), so the bot must leave the island
                                # first — otherwise it spams "Cannot calculate a route
                                # from int_land ... to prt_fild08" forever.
                                # ── ADAPTIVE FIELDFALL-ABANDON (not island-specific) ──
                                # A disconnected/hostile intro field (e.g. the
                                # Secluded Island) can be a kill-trap for a fresh
                                # character: mobs that out-DPS a gear-less bot, no
                                # route out except one warp, zero progress. Instead
                                # of hardcoding a map, we DARE adapt on telemetry:
                                #   • If the bot IS making progress (kills/exp up)
                                #     -> keep fighting, escape opportunistically.
                                #   • If it is making NO progress for a sustained
                                #     window AND is under pressure -> ABANDON the
                                #     field: commit to the exit with full survival
                                #     and do not keep grinding a losing field.
                                # This is progress-aware and applies to any
                                # profitless/dangerous field, not just int_land.
                                _made_progress = self._check_progress(signals)
                                _esc_now = __import__("time").time()
                                _esc_last = self._last_island_escape.get(_bot_id, 0.0)
                                _danger_sig = float(signals.get("danger", signals.get("aggro_count", 0)) or 0)
                                _died_recent = bool(signals.get("died", False)) or self._bot_deaths.get(_bot_id, 0) > 0
                                if _cur_map.startswith("int_land"):
                                    # CLEAR the stale farm lockMap (e.g. prt_fild05 set by
                                    # an earlier cold-start econ step) so OpenKore's hunting
                                    # brain stops trying to path to a field it cannot reach
                                    # from the island ("Cannot calculate a route from
                                    # int_land to prt_fild05" loop). With lockMap cleared and
                                    # attackAuto limited, the ONLY winning directive left is
                                    # the (49,57) sailor escape emitted below.
                                    _lock_now = str(signals.get("lockMap", "") or "").lower()
                                    if _lock_now and _lock_now not in ("int_land", "int_land01") and "49" not in _lock_now:
                                        _actions.append(HeuristicAction(
                                            kind="command", command="set lockMap 0",
                                            confidence=0.95,
                                            reason="Island: clear unreachable farm lockMap so the bot escapes via the (49,57) sailor",
                                            domain="progression",
                                        ))
                                    # ── LEARNED SURVIVAL POLICY (no hardcoded fix) ──
                                    # The island field's monsters (Porings) are PASSIVE in
                                    # the live server (db/re/mob_db.yml: G_PORING Modes only
                                    # FixedItemDrop, no aggressive/chase/angry). They never
                                    # attack a bot that does not attack them first. So the
                                    # gearless bot must NOT fight — fighting a passive Poring
                                    # provokes retaliation that kills it (confirmed live:
                                    # every "Poring attacks you Dmg 5" death is retaliation
                                    # from the bot attacking first). The escape therefore
                                    # RUNS freely: attackAuto 0 (retaliate-only => never
                                    # attacks a passive mob) + not-in-lock-only, so the bot
                                    # walks unmolested to the escape warp. This is a learned
                                    # reactive policy keyed on the monster passivity signal,
                                    # generalizing to any non-hostile transit field.
                                    _live_hp = float(signals.get("hp", 100) or 100)
                                    _live_hp_max = float(signals.get("hp_max", 100) or 100) or 100.0
                                    _hp_frac = _live_hp / _live_hp_max if _live_hp_max > 0 else 1.0
                                    _hp_critical = _hp_frac <= 0.30
                                    _under_pressure = _danger_sig > 0 or _died_recent
                                    # Passive-field escape: never attack, just run. attackAuto 0
                                    # = retaliate-only; since the monsters are passive (don't
                                    # attack first), the bot stays unprovoked and walks free.
                                    _actions.append(HeuristicAction(
                                        kind="command", command="set attackAuto 0",
                                        confidence=0.99,
                                        reason="Cold start: passive-field transit — attackAuto 0 so the bot never provokes a passive Poring; run freely to the escape warp",
                                        domain="combat",
                                    ))
                                    _actions.append(HeuristicAction(
                                        kind="command", command="set attackAuto_inLockOnly 0",
                                        confidence=0.90,
                                        reason="Cold start: allow running anywhere while crossing a passive field",
                                        domain="combat",
                                    ))
                                    if _hp_critical and not _died_recent:
                                        _actions.append(HeuristicAction(
                                            kind="command", command="use Red Potion",
                                            confidence=0.90,
                                            reason=f"Cold start: HP critically low ({_hp_frac:.0%}) — heal to survive the transit run",
                                            domain="survival",
                                        ))
                                        _actions.append(HeuristicAction(
                                            kind="command", command="sit",
                                            confidence=0.60,
                                            reason="Cold start: critically low HP, sit to recover before resuming",
                                            domain="survival",
                                        ))
                                    # Danger flag for a disconnected field with no
                                    # progress -> force the exit (risk-taking),
                                    # firing FASTER (every 6s) than the default
                                    # periodic retry (20s) but never spamming.
                                    # For the Secluded-Island INTRO specifically, a
                                    # level<=5 bot CANNOT progress bare-handed (Porings
                                    # are 55 HP / 12 dmg per hit), so escape is the ONLY
                                    # winning move — force it regardless of the
                                    # progress/danger probes (a probe returning "making
                                    # progress" because the bot is TAKING hits would
                                    # otherwise suppress the escape forever and strand it).
                                    _is_isolated_intro = (_cur_map.startswith("int_land") and _bl <= 5)
                                    _force_exit = _is_isolated_intro or ((not _made_progress and _danger_sig > 0) or _died_recent)
                                    # Re-emission cadence. The MapRoute fix (3a2db163d)
                                    # made `move 49 57` actually walk, so re-issuing it too
                                    # often (6s) restarts the in-progress route before the
                                    # bot reaches the warp (30-tile trek through 40 Porings).
                                    # 25s ~ matches the bridge's move_int_land_sailor dedupe
                                    # (30s), so each fresh issue is not instantly deduped and
                                    # the bot gets a real uninterrupted window to walk. The
                                    # escape still fires reliably; it just isn't self-defeating.
                                    _exit_cooldown = 25.0 if _is_isolated_intro else (6.0 if _force_exit else 20.0)
                                    if _esc_now - _esc_last >= _exit_cooldown:
                                        self._last_island_escape[_bot_id] = _esc_now
                                        # The island Porings are PASSIVE (live server:
                                        # db/re/mob_db.yml G_PORING Modes = FixedItemDrop only,
                                        # no aggressive/chase/angry). A bot that never attacks
                                        # them is never provoked, so it runs freely to the
                                        # (49,57) escape warp — no buffs needed. Escape.
                                        _actions.append(HeuristicAction(
                                            kind="command",
                                            command="move 49 57",
                                            confidence=0.95,
                                            reason="Cold start: escape the Secluded Island (sailor warp 49 57) — forced reflex exit",
                                            domain="progression",
                                        ))
                                        _actions.append(HeuristicAction(
                                            kind="command",
                                            command="talk resp 2",
                                            confidence=0.95,
                                            reason="Cold start: pick 'Sail to Izlude!' (option 2) on the #intro_to_izlude warp select (walking onto 49 57 triggers the OnTouch)",
                                            domain="progression",
                                        ))
                                    else:
                                        logger.debug(
                                            "adaptive_field: profitable grind (progress=true) elapsed=%.0f/20s",
                                            _esc_now - _esc_last,
                                        )
                                if _cur_map in ("iz_int", "iz_ac01", "iz_int01", "iz_int02", "iz_int03", "iz_int04"):
                                    if _cur_map == "iz_ac01":
                                        # The Academy Receptionist (Novice_Knife + 300
                                        # potions) is on the iz_ac01 MAIN HALL at (100,39).
                                        # Fire the registration dialog ONLY while the bot
                                        # is actually on iz_ac01 — firing it from an iz_int*
                                        # room coordinates the talknpc at a spot with no
                                        # receptionist and the dialog silently fails.
                                        # ALSO: once registered, the bot has the Novice_Knife
                                        # (1243) in inventory and the Receptionist shows the
                                        # already-registered menu (no "Register" option) —
                                        # re-talking only lands in the "What's on the 1st
                                        # floor" submenu and loops forever. So gate the
                                        # registration on NOT yet having the knife.
                                        _has_knife = False
                                        _has_potions = False
                                        _reg_n = self._reg_attempts.get(_bot_id, 0)
                                        # A reliable registration signal even when the live
                                        # inventory snapshot lags: registration grants 100
                                        # base EXP (academy.txt getexp 100,100). If the bot's
                                        # base_exp >= 100 it has registered, so treat it as
                                        # having the kit and stop re-talking.
                                        if int(signals.get("base_exp", 0) or 0) >= 100:
                                            _has_knife = True
                                        _inv_items = signals.get("inventory_items", []) or []
                                        for _ii in _inv_items:
                                            if isinstance(_ii, dict):
                                                _ii_id = str(_ii.get("id", ""))
                                                _ii_name = str(_ii.get("name", ""))
                                                # Only the academy Novice_Knife (1243) counts as
                                                # "registered" — the basic Knife (1201) does NOT
                                                # (a bot with just 1201 still needs the academy
                                                # kit: Novice_Knife 1243 + 300 Novice_Potions 569,
                                                # which are what let a level-1 actually farm).
                                                if _ii_id == "1243" or "novice knife" in _ii_name.lower():
                                                    _has_knife = True
                                                if _ii_id == "569" or "novice potion" in _ii_name.lower() or "red potion" in _ii_name.lower():
                                                    _has_potions = True
                                        if not _has_knife:
                                            # Also cap registration attempts: the snapshot
                                            # can be stale (bot4 had the knife + job_level 2
                                            # in DB but the snapshot showed empty/1), so don't
                                            # loop forever if the knife signal is unreliable.
                                            # After a few tries, move on to farming regardless.
                                            _reg_n = self._reg_attempts.get(_bot_id, 0) + 1
                                            self._reg_attempts[_bot_id] = _reg_n
                                        # Fire registration ONLY while unregistered AND under
                                        # the attempt cap. Once capped, stop re-talking so the
                                        # bot proceeds to farming (the kit is likely already
                                        # granted server-side even if the snapshot is stale).
                                        # NOTE (founder 2026-08-25): NO hardcoded dialog
                                        # sequences. The CONSCIOUS-tier LLM responder
                                        # (_llm_npc_dialog_response) reads the ACTUAL menu
                                        # options from the bridge signal and selects the
                                        # Register option agnostically — RAW NPCs are
                                        # customized, some need free-text chat. This block only
                                        # ensures the bot STAYS on iz_ac01 near the receptionist
                                        # so the LLM can talk; the canned 'r0' sequence is gone.
                                        if not _has_knife and _reg_n <= 6:
                                            # Open the conversation with the Academy
                                            # Receptionist so the ACTUAL menu appears;
                                            # the conscious LLM dialog responder
                                            # (_llm_npc_dialog_response) reads it and
                                            # selects "Register" agnostically (no
                                            # hardcoded r<idx> sequence).
                                            _ac_reg = self._get_npc("academy_receptionist", "iz_ac01")
                                            _ac_x = int((_ac_reg or {}).get("x", 100) or 100)
                                            _ac_y = int((_ac_reg or {}).get("y", 39) or 39)
                                            _actions.append(HeuristicAction(
                                                kind="command",
                                                command=f"talknpc {_ac_x} {_ac_y} c",
                                                confidence=0.95,
                                                reason="Cold start: open Academy Receptionist dialog; LLM selects Register from the menu (agnostic)",
                                                domain="progression",
                                            ))
                                        else:
                                            # Registered / attempt-capped: the kit is granted
                                            # server-side (Novice_Knife + potions + 100 EXP).
                                            # STOP re-talking and LEAVE the academy so the farm
                                            # lockMap (set once the bot exits iz_ac01) can engage.
                                            # The academy hall exits north at the izlude portal
                                            # (iz_ac01 100 24 -> izlude 127 253). Walk there.
                                            _ac_move = self._last_academy_move.get(_bot_id, 0.0)
                                            _tnow = __import__("time").time()
                                            if _tnow - _ac_move >= 15.0:
                                                self._last_academy_move[_bot_id] = _tnow
                                                _actions.append(HeuristicAction(
                                                    kind="command",
                                                    command="move 100 24",
                                                    confidence=0.9,
                                                    reason="Registered: walk out of the Academy hall (north exit to izlude) so the farm lockMap engages",
                                                    domain="progression",
                                                ))
                                    else:
                                        # On an Academy interior ROOM (iz_int/01/02/03/04).
                                        # These are tutorial intro rooms with the `nowarpto`
                                        # mapflag (no portal/recall in), and OpenKore's static
                                        # field-route table doesn't know the custom iz_int* ->
                                        # iz_ac01 connection, so a map-name `move iz_ac01`
                                        # recalculates forever. Behind the room's #room_outXX /
                                        # intro_evt02 gate NPCs (which all duplicate at the same
                                        # coords, per npc/re/jobs/novice/academy.txt) the exit
                                        # funnels the player toward the iz_ac01 main hall. Issue a
                                        # concrete move to that forward gate so the bot physically
                                        # walks out of the room regardless of the route table.
                                        _actions.append(HeuristicAction(
                                            kind="command",
                                            command="move 51 30",
                                            confidence=0.95,
                                            reason="Cold start: walk from Academy room to forward gate toward iz_ac01 hall (Receptionist at 100,39)",
                                            domain="progression",
                                        ))
                                # (the izlude academy-walk lives in the pathfinder
                                # block, where an escaped izlude bot is detected and
                                # routed to the academy door).
                                # Level 1-5: Cryptura Academy (prt_fild08) — the
                                # rathena-ai-world starter field. Level-1 Porings/
                                # Lunatics (Atk 1, BaseExp 150) spawn here with a
                                # 10% Knife drop; the town map has no weapon vendor
                                # and its Porings offer no progression. Lock onto
                                # prt_fild08 (safe academy field), attack Porings/
                                # Lunatics only.
                                # NOTE: 'set lockMap' (not bare 'lockMap') so the
                                # bridge's _sidecar_set_lockMap marker shields it
                                # from the hunting-map stickiness override.
                                # Skip lockMap/mon_control while on the Secluded
                                # Island intro (int_land) — OpenKore can't route and
                                # there's nothing to fight there; the island bailout
                                # above moves the bot off it first.
                                if not _cur_map.startswith("int_land"):
                                    # Do NOT arm the academy-farm lockMap while the bot is
                                    # still inside an Academy room (iz_int*/iz_ac01) or on the
                                    # sail-arrival port (izlude_a) — the bot is still
                                    # registering for its free starter kit and a hunt lockMap
                                    # a hunt lockMap there only fights the registration walk (walks to the
                                    # iz_ac01 hall / gate, never arrives). Defer until the
                                    # bot has actually reached a playable field out of the
                                    # academy rooms / the port.
                                    _cs_defer_academy = _cur_map.startswith(("iz_int", "iz_ac", "izlude_a"))
                                    # ── ACADEMY-FIRST DEFER (2.27, 2026-08-25) ──
                                    # A weapon-less bot on the TOWN map (izlude/izlude_a/etc.) is
                                    # still headed to the academy door (2 tiles away) for its free
                                    # starter kit. Arming the farm lockMap HERE makes OpenKore
                                    # cross-map-route to the far field, overriding the academy walk
                                    # and leaving the bot stuck in town calculating routes forever
                                    # (verified live: EXP frozen at 547, 'Calculating lockMap route'
                                    # every cycle). Defer farm lockMap until the bot has a weapon
                                    # (i.e. has actually completed academy registration).
                                    _cs_weaponless_town = (
                                        not self._has_coldstart_weapon(signals)
                                        and _cur_map.startswith(("izlude", "prontera"))
                                    )
                                    if not _cs_defer_academy and not _cs_weaponless_town:
                                        # Phase 1 (server-agnostic): resolve the hunting
                                        # lockMap from the level-based map-knowledge layer,
                                        # not a hardcoded prt_fild08. get_hunting_maps(level)
                                        # returns the safest/most-suitable field for THIS
                                        # character on THIS server; fall back to prt_fild08
                                        # only if the knowledge layer has no level-1 field.
                                        _hunt_map = "prt_fild08"
                                        try:
                                            # REACHABLE-FARM resolution: pick the best
                                            # hunt map that is ACTUALLY routable from the
                                            # bot's current map (BFS over the portal
                                            # graph). Locking an unreachable farm
                                            # (e.g. prt_fild05 from izlude) makes OpenKore
                                            # A* fail -> 'Unable to calculate a route'.
                                            _hunts = _mk_reachable_hunting_maps(_cur_map, int(_bl or 1))
                                            if not _hunts:
                                                _hunts = _mk_get_hunting_maps(int(_bl or 1))
                                            if _hunts:
                                                _hunt_map = str(_hunts[0][0])
                                        except Exception:
                                            pass
                                        # If the bot is ALREADY on a playable field
                                        # (*_fild), farm THERE rather than forcing a
                                        # lockMap (prt_fild08) that may be unroutable
                                        # from the current location (e.g. prt_fild05 has
                                        # no direct portal to prt_fild08; a stranded bot
                                        # just toggles lockMap forever). A level-1 bot on
                                        # prt_fild05 can fight Porings/Lunatics right where
                                        # it stands. Server-agnostic: uses the bot's actual
                                        # current field, not a hardcoded destination.
                                        # BUT respect an explicit lockMap if the bot is
                                        # already routed to a specific target (e.g. the test
                                        # bot on prt_fild05 locked to prt_fild08 should RUN
                                        # there, not farm prt_fild05).
                                        _cur_field = str(_cur_map or "").lower()
                                        _existing_lock = str(signals.get("lockMap", "") or "").lower()
                                        _existing_lock = _existing_lock.replace(".gat", "")
                                        # Only override to the current field if the bot has
                                        # no real farm target: no lockMap, or a non-field
                                        # (town/default) lockMap. An explicit *_fild/_dun
                                        # lockMap means the bot is deliberately routed there
                                        # (e.g. prt_fild08) and must run, not farm here.
                                        _lock_is_townish = bool(_existing_lock) and "_fild" not in _existing_lock and "_dun" not in _existing_lock
                                        # Override to the current field if it's a farm map the
                                        # bot can actually fight on. NOTE (verified live
                                        # 2026-08-28): prt_fild08 IS accessible on this server
                                        # (present in 0x0840) and prt_fild08c is NOT (absent
                                        # from maps_athena.conf) — the old "prt_fild08
                                        # unroutable / real farm is prt_fild08c" assumption is
                                        # inverted vs the live server. A bot physically on any
                                        # farm variant of its lock target fights there rather
                                        # than toggle a mismatched lockMap forever.
                                        _cur_base = _cur_field[:-1] if _cur_field.endswith(("a", "b", "c")) else _cur_field
                                        _lock_base = _existing_lock[:-1] if _existing_lock.endswith(("a", "b", "c")) else _existing_lock
                                        _same_farm = bool(_cur_base) and _cur_base == _lock_base
                                        if "_fild" in _cur_field \
                                                and (not _existing_lock or _lock_is_townish or _same_farm):
                                            _hunt_map = _cur_field
                                        # Persist to a member so the field-transit block
                                        # below (a sibling block, out of lexical scope for
                                        # _hunt_map) can recognize the bot is on its farm
                                        # field even before the snapshot's lockMap syncs.
                                        self._cold_start_hunt_map[_bot_id] = _hunt_map
                                        # CONNECT-GRACE FIX (the reconnect-loop root cause):
                                        # sending 'set lockMap' the INSTANT a bot connects to the
                                        # map server triggers a client-side disconnect (observed:
                                        # connect -> set_lockMap -> Disconnecting, every ~37s, so
                                        # bot10 never left izlude_c and EXP stayed frozen at 782).
                                        # Suppress lockMap emission while the client is freshly
                                        # reconnected (reconnect_age_s < 15s) so it stabilizes
                                        # in-game first. Purely additive — no state mutation.
                                        _cs_reconn = float(signals.get("reconnect_age_s", 999.0) or 999.0)
                                        if _cs_reconn >= 15.0:
                                            _actions.append(HeuristicAction(kind="command", command=f"set lockMap {_hunt_map}", confidence=0.85, reason=f"Cold start: academy farm (lvl {_bl}, hunt={_hunt_map}) (post-connect grace)", domain="progression"))
                                        # ── ROOT-CAUSE FIX (A2): inline mon_control re-emission ──
                                        # These `mon_control` commands were emitted EVERY cycle this
                                        # block ran. Each emission makes the OpenKore client RESTART
                                        # its AI (state 2, target reselection) and CANCELS any
                                        # in-progress attack — so a bot could never sustain a kill
                                        # long enough to bank EXP (the plateau). Rate-limit to once
                                        # per 60s per bot: the initial set is enough; re-sending just
                                        # resets the client's combat AI.
                                        _now_mc = __import__("time").time()
                                        if _now_mc - self._last_mon_emit.get(bot_id, 0.0) >= 60.0:
                                            self._last_mon_emit[bot_id] = _now_mc
                                            _actions.append(HeuristicAction(kind="command", command="mon_control Poring 0 1 1", confidence=0.7, reason="Attack Porings only (rate-limited)", domain="progression"))
                                            _actions.append(HeuristicAction(kind="command", command="mon_control Lunatic 0 1 1", confidence=0.7, reason="Attack Lunatics too (rate-limited)", domain="progression"))
                                        # ── FARM-FIELD PORTAL-EXIT CENTERING ──
                                        # A bot that crosses INTO the academy farm (prt_fild08c)
                                        # at a portal edge lingers there (few mobs spawn at the
                                        # edge) and either gets pulled back through the portal or
                                        # spins "AI restarted for target reselection" with nothing
                                        # in range — never killing. Move it to the field interior
                                        # (200,200) where academy.txt spawns Porings/Lunatics.
                                        _px = int(signals.get("x", 0) or 0)
                                        _py = int(signals.get("y", 0) or 0)
                                        if "_fild" in _cur_map and (
                                            (str(_cur_map).startswith("prt_fild08") and
                                             ((abs(_px - 170) < 12 and abs(_py - 378) < 12) or
                                              (abs(_px - 367) < 12 and abs(_py - 212) < 12)))
                                            or (str(_cur_map) == "prt_fild05" and
                                                abs(_px - 367) < 12 and abs(_py - 205) < 12)
                                        ):
                                            _actions.append(HeuristicAction(
                                                kind="command", command="move 200 200",
                                                confidence=0.90, domain="hunting",
                                                reason=f"Cold start: at {_cur_map} portal exit — move to field interior to reach mob spawns",
                                            ))
                                # ── FIELD-TRANSIT PROTECTION (level 1-5) ──
                                # A level-1 Novice cannot survive the tougher NON-
                                # academy fields (prt_fild05 etc.) en route to the
                                # academy farm prt_fild08. If the bot is NOT yet on
                                # its lockMap, disable attacking entirely so it
                                # RUNS to the academy instead of fighting and dying. attackAuto 2 (auto
                                # everywhere) is the default — force 0 while in
                                # transit; OpenKore re-enables it once in town via
                                # attackAuto_inLockOnly semantics.
                                _cur_map = str(signals.get("map", "") or "").lower()
                                _lock = str(signals.get("lockMap", "") or "").lower()
                                # Keyed on map TYPE, not lock equality: the bridge's
                                # hunting-map stickiness overrides lockMap=current-map
                                # whenever a bot is on *_fild, so lock==cur is TRUE on
                                # every field — an equality-based "arrived" branch would
                                # re-enable attacking precisely where a level-1 bot must
                                # run, not fight. fild + low level ⇒ transit (run),
                                # UNLESS the field is the safe academy starter field
                                # prt_fild08 (where level-1 Porings/Lunatics are the
                                # intended target and fighting is correct).
                                # _cs_hunt is the resolved cold-start farm field (may be the
                                # bot's current field for a stranded bot). If the bot is on
                                # it, treat it as arrived (farm), not transit.
                                _cs_hunt = str(getattr(self, "_cold_start_hunt_map", {}).get(_bot_id, "") or "").lower()
                                # TRANSIT PROTECTION (level 1-5): a low-level bot must RUN to
                                # its farm, not fight en route. Covers BOTH fields AND towns —
                                # verified live: the bot died to Lunatic (atk9) on izlude (a
                                # town) because this only fired on *_fild maps, leaving
                                # attackAuto on in town. Now: if not on the lockMap/farm, run.
                                _on_farm = bool(_cur_map and (_cur_map == _lock or _cur_map == "prt_fild08" or (_cs_hunt and _cur_map == _cs_hunt)))
                                if _bl <= 5 and not _on_farm:
                                    _actions.append(HeuristicAction(kind="command", command="set attackAuto 0", confidence=0.90, reason=f"Cold start: transit through {_cur_map} at lvl {_bl} — run, don't fight", domain="survival"))
                                    _actions.append(HeuristicAction(kind="command", command="set attackAuto_inLockOnly 1", confidence=0.90, reason="Cold start: only attack on lockMap", domain="survival"))
                                elif _on_farm:
                                    # ── ARRIVED AT THE ACADEMY FARM: re-enable attacking ──
                                    # Reverse the transit protection so the bot can
                                    # actually farm Porings/Lunatics once it reaches
                                    # the academy field prt_fild08 (safe starter field
                                    # with level-1 mobs). Note prt_fild08 IS a *_fild
                                    # map, so match it explicitly — the generic
                                    # "arrived at town" check (non-fild) never fires
                                    # for a fild lockMap.
                                    _actions.append(HeuristicAction(kind="command", command="set attackAuto 3", confidence=0.90, reason=f"Cold start: at academy {_cur_map} lvl {_bl} — attack enabled", domain="survival"))
                                    _actions.append(HeuristicAction(kind="command", command="set attackAuto_inLockOnly 0", confidence=0.90, reason="Cold start: attack enabled on lockMap", domain="survival"))
                            else:
                                # Level 6-15: prt_fild05, farm the map's actual spawns
                                # (Pupa/Thief Bug give real EXP) — attack, not avoid.
                                # NOTE: 'set lockMap' so the bridge marker shields
                                # it from the hunting-map stickiness override.
                                _actions.append(HeuristicAction(kind="command", command="set lockMap prt_fild05", confidence=0.75, reason=f"Cold start: field at lvl {_bl}", domain="progression"))
                                _actions.append(HeuristicAction(kind="command", command="mon_control Pupa 0 1 1", confidence=0.6, reason="Farm Pupa (real EXP)", domain="progression"))
                                _actions.append(HeuristicAction(kind="command", command="mon_control Thief Bug 0 1 1", confidence=0.6, reason="Farm Thief Bug (real EXP)", domain="progression"))
                        if _bl <= 25:
                            _cs_mgr = getattr(self, '_cold_start_manager', None)
                            if _cs_mgr is not None:
                                _cs_mgr.assess(signals, _actions, _bot_id)
                    if self._npc_lookup and _bl >= 9:
                        if _job in ["novice"] and _bl >= 9:
                            _nj = "swordman" if int(signals.get("str", 0) or 0) > int(signals.get("int", 0) or 0) else "mage"
                            _jc = self._npc_lookup.get_job_change_dialogue(_job, _nj)
                            if _jc:
                                _cmd = self._npc_lookup.get_talk_command(_jc)
                                if _cmd:
                                    _actions.append(HeuristicAction(kind="command", command=_cmd, confidence=0.8, reason=f"Job change: {_job} -> {_nj}", domain="progression"))
                    safe_assess(self._party_engine, "party_engine", signals, _actions, _bot_id, get_registry())
                    safe_assess(self._danger_predictor, "danger_predictor", signals, _actions, _bot_id, get_registry())
                    safe_assess(self._map_rotation, "map_rotation", signals, _actions, _bot_id, get_registry())
                    safe_assess(self._world_state, "world_state", signals, _actions, _bot_id, get_registry())
                    safe_assess(self._combat_pressure, "combat_pressure", signals, _actions, _bot_id, get_registry())
                    safe_assess(self._kiting_v2, "kiting_v2", signals, _actions, _bot_id, get_registry())
                    safe_assess(self._inventory_policies, "inventory_policies", signals, _actions, _bot_id, get_registry())
                    safe_assess(self._spawn_navigator, "spawn_navigator", signals, _actions, _bot_id, get_registry())
                    safe_assess(self._combo_protocol, "combo_protocol", signals, _actions, _bot_id, get_registry())
                    safe_assess(self._loadout_planner, "loadout_planner", signals, _actions, _bot_id, get_registry())
                    safe_assess(self._durability_monitor, "durability_monitor", signals, _actions, _bot_id, get_registry())
                    safe_assess(self._post_mortem, "post_mortem", signals, _actions, _bot_id, get_registry())
                    safe_assess(self._loot_discipline, "loot_discipline", signals, _actions, _bot_id, get_registry())
                    safe_assess(self._event_detector, "event_detector", signals, _actions, _bot_id, get_registry())
                    safe_assess(self._live_market, "live_market", signals, _actions, _bot_id, get_registry())
                    safe_assess(self._danger_pathfinder, "danger_pathfinder", signals, _actions, _bot_id, get_registry())
                    safe_assess(self._social_reputation, "social_reputation", signals, _actions, _bot_id, get_registry())
                    safe_assess(self._farming_loop, "farming_loop", signals, _actions, _bot_id, get_registry())
                    safe_assess(self._competition_planner, "competition_planner", signals, _actions, _bot_id, get_registry())
                    # ── NEW DOMAIN MODULES: wired into assess cycle ──
                    # Macro Intelligence: evaluate triggers from current state
                    if hasattr(self, '_macro_intelligence') and self._macro_intelligence:
                        try:
                            _macro = self._macro_intelligence.process_triggers(
                                bot_state=signals, bot_id=_bot_id,
                            )
                            if _macro and hasattr(_macro, 'action_sequence'):
                                for _action in _macro.action_sequence:
                                    _actions.append(HeuristicAction(
                                        kind="command", command=getattr(_action, 'command', ''),
                                        confidence=0.85, domain="macro",
                                        reason=f"Macro: {_macro.description}",
                                    ))
                        except Exception:
                            pass
                    # Behavior Engine: human-like behavior simulation (used elsewhere in PDCA)
                    # The behavior engine is a singleton accessed via get_behavior_engine()
                    # and is consumed by the PDCA loop's anti-detection module — not here.
                    # Profitability: market-aware farming decisions
                    if hasattr(self, '_profitability') and self._profitability:
                        try:
                            self._profitability.assess(signals, _actions, _bot_id)
                        except Exception:
                            pass
                    # WoE: War of Emperium tactics
                    if hasattr(self, '_woe') and self._woe:
                        try:
                            self._woe.assess(signals, _actions, _bot_id)
                        except Exception:
                            pass
                    # Party: coordination and sync
                    if hasattr(self, '_party_domain') and self._party_domain:
                        try:
                            self._party_domain.assess(signals, _actions, _bot_id)
                        except Exception:
                            pass
                    # Equipment Optimizer: loadout recommendations
                    if hasattr(self, '_equipment_optimizer') and self._equipment_optimizer:
                        try:
                            _eq_results = self._equipment_optimizer.assess(signals, bot_id=_bot_id)
                            if _eq_results:
                                for _eq in _eq_results[:3]:
                                    _actions.append(HeuristicAction(
                                        kind="command", command=str(_eq),
                                        confidence=0.7, domain="equipment",
                                        reason="Equipment optimizer recommendation",
                                    ))
                        except Exception:
                            pass
                    if self._goal_manager and self._task_scheduler:
                        for _g in self._goal_manager.get_active_goals()[:2]:
                            _actions.append(HeuristicAction(kind="log", command=f"goal={_g}", confidence=0.5, reason=f"Goal: {_g}", domain="planning"))
                        # ── TASK SCHEDULER EXECUTION (was dormant) ──
                        # Run the scheduler and translate its semantic tasks
                        # into safe real commands / observable intents.
                        # NOTE: consumable_depletion_score is injected into
                        # signals by _emit_heuristic_actions (pdca_loop) from
                        # the enriched world projection before assess() runs.
                        try:
                            _schedule = self._task_scheduler.schedule_from_signals(signals)
                            _seen_tasks: set[tuple[str, str]] = set()
                            for _task in _schedule[:3]:  # top 3 by priority
                                for _ta in self._task_translator.translate(_task, signals, _bot_id):
                                    # Dedupe identical (kind, command) within one
                                    # cycle — repair/sell/grind all carry
                                    # return_town, which must emit once.
                                    _key = (_ta.kind, _ta.command)
                                    if _key in _seen_tasks:
                                        continue
                                    _seen_tasks.add(_key)
                                    _actions.append(_ta)
                        except Exception as _se:
                            logger.debug(f"[heuristic] task scheduler: {_se}")
                except Exception as _de:
                    logger.debug(f"[heuristic] domain delegation: {_de}")
            # ── CONSCIOUS-TIER FALLBACK (Phase 2, server-agnostic) ──
            # If the reflex/rule path produced NO progression directive and the bot is
            # in-game on a map the rules don't recognize (a novel / unknown server map),
            # delegate to the unified consciousness to reason the next action. This is
            # the "conscious solves novel situations" behavior: it works for ANY server
            # by reasoning from live observed state instead of matching literals. It
            # never overrides a normal rule-driven action.
            try:
                if self._consciousness is not None:
                    _had_progression = any(
                        getattr(a, "domain", "") in ("progression", "routing") for a in _actions
                    )
                    _m = str(signals.get("map", "") or "").lower()
                    _in_game = bool(signals.get("in_game", False))
                    if _in_game and not _had_progression and _m and not _m.startswith(("int_land", "iz_int", "iz_ac")):
                        _uc_ctx = {
                            "bot_id": _bot_id,
                            "map": _m,
                            "level": _bl,
                            "job": signals.get("job", "novice"),
                            "base_level": _bl,
                            "job_name": signals.get("job_name", "novice"),
                            "hp": signals.get("hp", 1),
                            "hp_max": signals.get("hp_max", 1),
                            "lockMap": signals.get("lockMap", ""),
                            "in_game": _in_game,
                        }
                        _ucd = self._consciousness.decide(
                            DecisionDomain.PROGRESSION, DecisionUrgency.SHORT_TERM, _uc_ctx
                        )
                        if _ucd is not None and getattr(_ucd, "action", ""):
                            _uc_act = str(_ucd.action)
                            _uc_cmd = _uc_act
                            if _uc_act.startswith("move "):
                                _uc_cmd = f"move {_uc_act[5:].strip()}"
                            _actions.append(HeuristicAction(
                                kind="command", command=_uc_cmd,
                                confidence=0.6,
                                reason=f"Conscious (server-agnostic): {getattr(_ucd,'reason','') or 'decide'} on map {_m}",
                                domain="progression",
                            ))
            except Exception as _uc_de:
                logger.debug(f"[heuristic] conscious fallback: {_uc_de}")
            return assessment
        except Exception as e:
            logger.error(f"assess() crashed for {bot_id_override or 'unknown'}: {type(e).__name__}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            bot_id = bot_id_override or signals.get("bot_id", "default")
            return HeuristicAssessment(
                horizon=signals.get("horizon", "short_term"), actions=[], confidence=0.5,
                actionable=False, top_domain="survival", signals=dict(signals),
            )

    def _resolve_academy_door(self, current_map: str) -> str:
        """Resolve the warp tile on `current_map` that leads to the Academy map
        (iz_ac01 / iz_ac01_a) by READING portals.txt — a FACT from the server's
        portal graph, never a hardcoded coordinate literal. Returns a 'x y'
        string or '' if the current map has no academy warp.

        This mirrors the LLM cold-start advisory's resolution (pdca_loop
        `_llm_cold_start_advisory`) so the reflex/fallback path uses the same
        data-driven source — honoring the founder rule that cold-start
        coordinates are FACTS from the portal graph, not baked literals.
        """
        _m = str(current_map or "").lower().replace(".gat", "")
        try:
            _here = Path(__file__)
            for _anc in [_here, *_here.parents]:
                _pf = _anc / "tables" / "portals.txt"
                if not _pf.exists():
                    continue
                for _ln in _pf.read_text(errors="ignore").splitlines():
                    _f = _ln.split()
                    if (len(_f) >= 5
                            and _f[0].lower().replace(".gat", "") == _m
                            and _f[3].lower() in ("iz_ac01", "iz_ac01_a")):
                        return f"{_f[1]} {_f[2]}"
                break
        except Exception:
            pass
        return ""

    def _has_coldstart_weapon(self, signals: dict) -> bool:
        """Whether the bot holds/carries any starter weapon (used to decide if the
        academy-first cold-start step is still needed). Robust to inventory-item
        digest (id+name), legacy name list, attack-power, and the bridge's
        has_weapon_in_inventory signal. Once a weapon is ever observed for a bot,
        it is latched so an intermittent inventory snapshot never reverts it."""
        _bot = self._resolve_bot_id(signals)
        try:
            _now = False
            # Direct bridge signal (top-level snapshot).
            if signals.get("has_weapon_in_inventory"):
                _now = True
            inv = signals.get("inventory_items", []) or signals.get("inventory", []) or []
            for item in inv:
                if isinstance(item, dict):
                    # Digest form: {name, item_id, quantity, type, equipped}
                    name = str(item.get("name", "") or "").lower()
                    _iid = item.get("item_id", 0) or item.get("id", 0)
                    _itype = item.get("type", 0) or 0
                else:
                    name = str(item).lower()
                    _iid = 0
                    _itype = 0
                # type 4 = weapon, type 5 = armor, type 8 = accessory
                if _itype in (4, 5, 8):
                    _now = True
                if any(w in name for w in ("knife", "sword", "mace", "bow", "dagger", "rod", "cutter")):
                    _now = True
                if _iid and str(_iid) in ("1201", "1243", "1244", "1301", "1302", "1303", "1601", "1701"):
                    _now = True
            if int(signals.get("attack_power", 0) or 0) > 30:
                _now = True
            return self._latch_weapon_seen(_bot, _now)
        except Exception:
            return self._latch_weapon_seen(_bot, False)

    def _latch_weapon_seen(self, bot_id: str, has_weapon: bool) -> bool:
        """Persist a per-bot weapon latch so an intermittent inventory snapshot
        (OpenKore parses the 2025 client's 0B09 item list inconsistently) never
        makes the bot 'forget' it already owns a weapon and ping-pong back to the
        academy door. A character does not lose its starter knife mid-session, and
        once registered the kit is granted server-side, so a positive observation
        is latched permanently for the process lifetime."""
        _stable = str(bot_id or "default").split(":")[-1].split("/")[-1] or "default"
        if has_weapon:
            self._weapon_latch[_stable] = True
            # Persist so a sidecar restart doesn't forget the bot already owns a weapon
            # (the intermittent 0B09 inventory parse may not re-report it for a long time).
            try:
                self.save_state()
            except Exception:
                pass
        return self._weapon_latch.get(_stable, False)

    def _set_config_once(self, actions: list, bot_id: str, key: str, value: str, domain: str, reason: str, confidence: float = 0.95) -> None:
        """Emit a 'set' command only if the value has changed since last set.
        Prevents spamming 'set route_randomWalk 1' every cycle when it's already 1.
        On first cycle after restart, always sends (cache is empty).
        Uses stable key (character name only) to handle bot_id prefix changes."""
        _stable = bot_id.split(":")[-1].split("/")[-1] if ":" in bot_id else bot_id
        cache = self._last_config_set.setdefault(_stable, {})
        last = cache.get(key)
        if last == value:
            return  # Already set to this value, skip
        cache[key] = value
        actions.append(HeuristicAction(
            kind="command", command=f"set {key} {value}",
            confidence=confidence, domain=domain,
            reason=reason,
        ))

    def _sell_config_once(self, bot_id: str, item_id: str, cooldown: float = 60.0) -> bool:
        """Rate-limited sell command tracker.
        Returns True if the sell command for this item should be queued
        (not recently sent within cooldown seconds).
        Similar to _set_config_once but for sell commands with a cooldown period."""
        cache = self._sold_items.setdefault(bot_id, {})
        _now = __import__("time").time()
        last = cache.get(item_id, 0.0)
        if _now - last < cooldown:
            return False
        cache[item_id] = _now
        return True

    def _check_stuck(self, signals: dict, bot_id: str) -> list[HeuristicAction]:
        """Detect if bot is stuck (hasn't moved >5 tiles in 30 seconds).
        Returns a list of unstuck actions if stuck is detected.
        Tracks position from signals 'x', 'y' and compares to last known position."""
        stuck_actions: list[HeuristicAction] = []
        _now = __import__("time").time()
        _x = signals.get("x", 0) or 0
        _y = signals.get("y", 0) or 0
        # ── UNKNOWN-POSITION GUARD (2026-08-25, leak/spin root cause) ──
        # When the snapshot has no position (x=0,y=0 = no signal yet), the
        # stuck-detector would fire 'move 0 0' (an invalid destination) every
        # cycle -> OpenKore A* fails -> route-fail spin -> the ~45MB/s leak.
        # Never treat an unknown position as "stuck". Also skip if the bot is
        # not in-game (no map).
        if (_x == 0 and _y == 0) or not signals.get("map_known", False):
            return stuck_actions
        _last_pos = self._last_position.get(bot_id)
        if _last_pos is not None:
            _lx, _ly, _ltime = _last_pos
            _dx = abs(_x - _lx)
            _dy = abs(_y - _ly)
            _dist = max(_dx, _dy)  # Chebyshev distance
            _elapsed = _now - _ltime
            if _dist <= 5 and _elapsed >= 30:
                # Bot hasn't moved significantly — queue unstuck actions
                logger.info(f"[stuck_detector] {bot_id}: position ({_lx},{_ly}) -> ({_x},{_y}) "
                           f"dist={_dist} over {_elapsed:.0f}s — sending unstuck")
                # Random target within ~20 tiles to break stuck — CLAMPED to the
                # map bounds: a negative/off-map target makes OpenKore's A* fail
                # from an invalid destination (move 0 -10 observed) and re-spins.
                _rand = __import__("random").Random(hash(bot_id + str(_now)) & 0xFFFFFFFF)
                _mw = int(signals.get("field_width", signals.get("map_width", 300)) or 300)
                _mh = int(signals.get("field_height", signals.get("map_height", 300)) or 300)
                _tx = max(0, min(_mw - 1, _x + _rand.randint(-15, 15)))
                _ty = max(0, min(_mh - 1, _y + _rand.randint(-15, 15)))
                stuck_actions.append(HeuristicAction(
                    kind="command", command="stand",
                    confidence=0.90, domain="survival",
                    reason=f"Stuck: position ({_x},{_y}) unchanged for {_elapsed:.0f}s — stand then random move to unstick (never 'ai manual' — it freezes auto-attack forever)",
                ))
                stuck_actions.append(HeuristicAction(
                    kind="command", command=f"move {_tx} {_ty}",
                    confidence=0.90, domain="survival",
                    reason=f"Random move to ({_tx},{_ty}) to break stuck",
                ))
                stuck_actions.append(HeuristicAction(
                    kind="command", command="ai auto",
                    confidence=0.90, domain="survival",
                    reason="Re-enable auto after unstuck move",
                ))
                self._last_position[bot_id] = (_tx, _ty, _now)
            else:
                # Position changed or not enough time — update record
                self._last_position[bot_id] = (_x, _y, _now)
        else:
            # First observation — set initial position
            self._last_position[bot_id] = (_x, _y, _now)
        return stuck_actions

    def _assess_impl(self, signals: dict[str, Any], bot_id_override: str | None = None) -> HeuristicAssessment:
        actions: list[HeuristicAction] = []
        bot_id = bot_id_override or signals.get("bot_id", "default")

        # ── NOT IN-GAME GUARD: skip all domains except cold start if bot hasn't entered game ──
        # Detect by checking if we have real snapshot data (not just default signals)
        _bl = signals.get("base_level", 0) or 0
        _mk = signals.get("map_known", False)
        if not _mk and (not isinstance(_bl, (int, float)) or _bl <= 1):
            return HeuristicAssessment(
                actions=[], actionable=False, confidence=1.0,
                top_domain="login", horizon=signals.get("horizon", "unknown"),
                signals={"reason": "Bot not in-game — waiting for login"}
            )

        # Normalize to stable key (character name only) — bridge sends different
        # account prefixes per cycle (openkoreai: vs Asgards Glory:). Reassign
        # bot_id so EVERY downstream keyed dict (_cold_start_step, _last_*,
        # _bot_state, _get_state read at 1277, writes at 4438) uses the SAME
        # stable key. The inconsistency (full-id reads vs stable-id writes) kept
        # COLD_START state permanently stuck at step 0 — the #1 live defect.
        _track_key = bot_id.split(":")[-1].split("/")[-1] if ":" in bot_id else bot_id
        bot_id = _track_key
        _now_t = __import__("time").time()
        # Auto-save state every 60 seconds
        try:
            _last_save = getattr(self, "_last_save_time", 0)
            if _now_t - _last_save > 60:
                self.save_state()
                object.__setattr__(self, "_last_save_time", _now_t)
        except Exception:
            pass
        state = self._get_state(signals, bot_id)
        prev_state = self._bot_state.get(_track_key, "UNKNOWN")

        if state != prev_state:
            self._bot_state[_track_key] = state
            self._state_since[_track_key] = __import__("time").time()
            logger.info(f"[heuristic] {_track_key} state: {prev_state} -> {state}")

        made_progress = self._check_progress(signals)
        state_duration = __import__("time").time() - self._state_since.get(_track_key, 0)
        is_stuck = not made_progress and state_duration > 120
        # Track kills/hour for monitoring
        self._track_kills_per_hour(signals, bot_id)
        # ── STUCK DETECTOR: check if bot hasn't moved in 30s ──
        _stuck_actions = self._check_stuck(signals, bot_id)
        if _stuck_actions:
            # Prepend stuck-detection actions so they execute immediately
            actions.extend(_stuck_actions)
            # Don't return early — let the state handler add its own actions too



        hp = signals.get("hp_ratio", 1.0)
        map_name = signals.get("map", "").lower()
        zeny = signals.get("zeny", 0) or 0
        weight = signals.get("weight_ratio", 0) or 0
        base_level = signals.get("base_level", 1) or 1
        job_level = signals.get("job_level", 1) or 1
        job_name = signals.get("job_name", "novice").lower()
        self._adaptive._last_job = job_name  # Set for profit calculation delegate
        stat_points = signals.get("stat_points", 0) or 0

        # ── SMART SAVE-POINT BINDING ──
        # Bind the respawn point at the nearest town Kafra (one-shot per town)
        # so deaths / Butterfly-Wing returns land in a SAFE town, never the
        # hostile start field. Highest-efficiency policy: bind on first town
        # visit, don't re-bind, don't spam.
        try:
            self._ensure_save_point(actions, bot_id, map_name)
        except Exception as _sp_err:
            logger.debug("ensure_save_point error: %s", _sp_err)
        skill_points = signals.get("skill_points", 0) or 0
        in_party = signals.get("in_party", False)
        inventory = signals.get("inventory_items", []) or []

        # ── STEP TIMEOUT ESCALATION: if cold start step stuck >5min, diagnose ──
        _cs_step = self._cold_start_step.get(_track_key, 0)
        _cs_since = self._cold_start_step_since.get(_track_key, 0)
        if _cs_since == 0:
            self._cold_start_step_since[_track_key] = _now_t
        elif _cs_step > 0 and _now_t - _cs_since > 300:
            # Step stuck for 5+ minutes — diagnose and recover
            logger.warning(f"[heuristic] {_track_key} step {_cs_step} stuck >5min — diagnosing")
            # Check what's blocking
            _blockers = []
            if hp < 0.3:
                _blockers.append("low_hp")
            if zeny < 50 and _cs_step >= 1:
                _blockers.append("no_zeny")
            if not any("knife" in str(i).lower() for i in inventory) and _cs_step >= 2:
                _blockers.append("no_weapon")
            if not any("potion" in str(i).lower() for i in inventory) and _cs_step >= 3:
                _blockers.append("no_potions")
            if _blockers:
                logger.warning(f"[heuristic] {_track_key} step {_cs_step} blockers: {_blockers}")
                # Emit diagnostic action
                actions.append(HeuristicAction(
                    kind="log",
                    command=f"Step {_cs_step} stuck: {', '.join(_blockers)}",
                    confidence=0.9,
                    reason=f"Step {_cs_step} stuck diagnosis: {', '.join(_blockers)}",
                ))
            # Reset timer to avoid spamming every cycle
            self._cold_start_step_since[_track_key] = _now_t
        skills = signals.get("skills", []) or []
        bot_name = signals.get("bot_name", bot_id)
        horizon = signals.get("horizon", "short_term")
        _leader_map = ""
        # _profile_to_char is accessed via self._profile_to_char throughout

        # ── COLD START SEQUENCE: task-completion-triggered, not time-based ──
        # The OnboardingService exists but is disconnected from the heuristic.
        # This sequence fires each step when the PREVIOUS step is confirmed via signals.
        # Step 1: Farm 50z on prt_fild05 — confirmed when zeny >= 50
        # Step 2: Buy Knife (item 1201) — confirmed when weapon in inventory_items
        # Step 3: Buy Red Potions (item 501) — confirmed when potions in inventory_items
        # Step 4: Return to hunting map with weapon + potions — cold start complete
        # Define map check vars before use (config audit section defines them later)
        _cs_map = signals.get("map", "") or ""
        _cs_in_town = any(x in _cs_map for x in ["prontera", "morocc", "geffen", "payon", "aldebaran", "alberta", "izlude"])
        _cs_in_hunting = any(x in _cs_map for x in ["prt_fild", "pay_fild", "mjolnir", "gef_fild", "ra_fild", "moc_fild", "cmd_fild"])
        # Use stable key based on character name only (not account prefix)
        _cs_stable_key = bot_id.split(":")[-1].split("/")[-1] if ":" in bot_id else bot_id
        _cold_start_step = self._cold_start_step.get(_cs_stable_key, 0)
        # AGNOSTIC + LATCHED weapon detection (RC3a / 0B09-parse gap): use the
        # latched decision so an intermittent inventory snapshot never reports a
        # weapon-less bot. Raw item-name matching below is a fallback only.
        _has_weapon = self._has_coldstart_weapon(signals)
        _has_potions = any(
            "potion" in str(item).lower() or "red" in str(item).lower() or "orange" in str(item).lower() or "white" in str(item).lower()
            for item in inventory
        )
        if _cold_start_step == 0:
            # Step 0: Check if we need cold start at all
            if _has_weapon and _has_potions:
                # Already equipped — skip cold start
                self._cold_start_step[_cs_stable_key] = 4
            else:
                # Need cold start
                if _cs_in_hunting:
                    # Already on hunting map with no weapon — advance directly to step 1 (farm)
                    self._cold_start_step[_cs_stable_key] = 1
                    _cold_start_step = 1
                elif not _cs_in_town and not str(signals.get("map", "") or "").lower().startswith("int_land"):
                    # On a different non-town map — portal walk to Prontera.
                    # NOTE: int_land (Secluded Island intro) is EXCLUDED — the
                    # island has no route to Prontera; the int_land bailout
                    # (sail to Izlude) handles it instead. Emitting `move
                    # prontera` here fights that bailout and errors
                    # ("Cannot calculate a route from int_land"), leaving the
                    # bot stuck at spawn.
                    actions.append(HeuristicAction(
                        kind="command", command="set attackAuto 0",
                        confidence=0.99, domain="economy",
                        reason="Cold start - disable attacking during portal walk (never 'ai manual' — that freezes auto-attack forever and blocks all EXP)",
                    ))
                    # AGNOSTIC town target (RC3a): resolve a safe town to portal-walk
                    # to via the shared helper (learned safe_town -> derived town ->
                    # last-resort). Never a hardcoded 'prontera' at decision time.
                    _town_target = self._resolve_safe_town()
                    actions.append(HeuristicAction(
                        kind="command", command=f"move {_town_target}",
                        confidence=0.99, domain="economy",
                        reason=f"Cold start - walk to safe town ({_town_target})",
                    ))
                else:
                    # In Prontera — advance to next step based on inventory
                    self._cold_start_step[_cs_stable_key] = 1
                    _cold_start_step = 1
        if _cold_start_step == 1:
            # Step 1: Farm 50z on prt_fild05 (no weapon, need zeny for knife)
            # Stay at step 1 until zeny >= 50
            if zeny >= 50:
                # Farmed enough — advance to step 2 (buy knife)
                self._cold_start_step[_cs_stable_key] = 2
                _cold_start_step = 2
                logger.info(f"[cold_start] {bot_id}: farmed 50z on prt_fild05, step 1 -> 2")
            else:
                if _cs_in_town:
                    # ── ACADEMY-FIRST (weapon-less bot in izlude) ──
                    # A level-1 novice that just escaped the island to izlude has no
                    # weapon and cannot farm prt_fild05 bare-handed. Before any
                    # zeny-farming, it must register at the Cryptura Academy
                    # receptionist (iz_ac01 100,39) for the free Novice_Knife + 300
                    # potions. Override the prt_fild05 step-1 walk to send it to the
                    # academy door (izlude 125,257 warp -> iz_ac01) instead.
                    if _cs_map == "izlude" and not self._has_coldstart_weapon(signals):
                        # Resolve the academy-door warp tile from the portal graph
                        # (FACT, data-driven — NOT a hardcoded coordinate literal).
                        # Throttle the academy-door move (at most once per 10s) so
                        # OpenKore walks the long izlude route instead of recalculating.
                        _ac_door = self._resolve_academy_door(_cs_map)
                        _now = __import__("time").time()
                        if _ac_door and _now - self._last_academy_move.get(bot_id, 0.0) >= 10.0:
                            self._last_academy_move[bot_id] = _now
                            actions.append(HeuristicAction(
                                kind="command", command=f"move {_ac_door}",
                                confidence=0.99, domain="progression",
                                reason="Cold start - walk to Academy door (warp to iz_ac01) for free starter kit (resolved from portal graph)",
                            ))
                        # Registration completes on arrival at iz_ac01 (handled by the
                        # academy block); do NOT advance the cold-start step here — the
                        # academy kit (knife+potions) will move the flow forward naturally.
                    else:
                        # In Prontera/other town. Prefer the academy-farm lockMap if the
                        # level-1-5 academy block already resolved one (prt_fild08 or the
                        # current field) — otherwise two competing cold-start blocks
                        # oscillate lockMap between prt_fild05 and prt_fild08 every cycle,
                        # leaving the bot stuck in town calculating routes and never
                        # farming (verified live). Only fall back to the legacy prt_fild05
                        # step-1 target when the academy block did NOT pick a hunt field.
                        _cs_authority_hunt = str(getattr(self, "_cold_start_hunt_map", {}).get(bot_id, "") or "").lower()
                        # ── SERVER-AGNOSTIC REACHABLE-FARM RESOLUTION (sidecar decision) ──
                        # A farm target that is UNROUTABLE from the bot's current map makes
                        # the bot loop forever ("Calculating route..."), never farming. Resolve
                        # the hunt target to the reachable variant for the current location.
                        # NOTE (verified live 2026-08-28): prt_fild08c exists in map_index
                        # + has a GAT, but is NOT loaded by the server's maps_athena.conf
                        # (only prt_fild08/08a/08b are) → it is absent from the 0x0840
                        # accessible-map list → OpenKore refuses to enter it ("Map server
                        # not ready" / routes back to town forever). The old izlude_c→
                        # prt_fild08c remap locked the bot to a map the server never serves.
                        # Always target the BASE farm map (prt_fild08), which IS accessible.
                        _cs_cur = str(signals.get("map", "") or "").lower().replace(".gat", "")
                        if _cs_cur == "izlude_c" and _cs_authority_hunt == "prt_fild08c":
                            _cs_authority_hunt = "prt_fild08"
                            self._cold_start_hunt_map[bot_id] = "prt_fild08"
                        if _cs_authority_hunt and "_fild" in _cs_authority_hunt:
                            actions.append(HeuristicAction(
                                kind="command", command=f"set lockMap {_cs_authority_hunt}",
                                confidence=0.99, domain="economy",
                                reason=f"Cold start step 1 - lockMap to academy hunt {_cs_authority_hunt}",
                            ))
                            actions.append(HeuristicAction(
                                kind="command", command=f"move {_cs_authority_hunt}",
                                confidence=0.99, domain="economy",
                                reason=f"Cold start step 1 - walk to academy hunt {_cs_authority_hunt}",
                            ))
                        else:
                            # SIDECAR routing decision (server-agnostic): the academy
                            # tutorial ROOM (iz_ac01_a) has no portal to any field map,
                            # so `move <farm>` is unroutable there. The bot must first
                            # walk to the room's EXIT warp (iz_ac01_a 100,24 ->
                            # izlude_a 127,253), from which the farm fields are reachable.
                            # Route a farm-bound bot in the academy room to the exit tile
                            # (a sidecar decision — the bridge only passes commands).
                            actions.append(HeuristicAction(
                                kind="command", command="set lockMap prt_fild05",
                                confidence=0.99, domain="economy",
                                reason=f"Cold start step 1 - set lockMap to prt_fild05, need {50 - zeny}z more",
                            ))
                            if str(_cs_map).lower() == "iz_ac01_a":
                                actions.append(HeuristicAction(
                                    kind="command", command="move 100 24",
                                    confidence=0.99, domain="economy",
                                    reason="Academy room (iz_ac01_a): walk to room exit (100,24 -> izlude_a) so the farm map is reachable",
                                ))
                            else:
                                actions.append(HeuristicAction(
                                    kind="command", command="move prt_fild05",
                                    confidence=0.99, domain="economy",
                                    reason=f"Cold start step 1 - walk to prt_fild05, need {50 - zeny}z more",
                                ))
                elif _cs_in_hunting:
                    # On prt_fild05 — enable AI and attack for farming
                    actions.append(HeuristicAction(
                        kind="command", command="ai auto",
                        confidence=0.99, domain="economy",
                        reason="Cold start step 1 - enable AI for farming Porings",
                    ))
                    actions.append(HeuristicAction(
                        kind="command", command="set attackAuto 3",
                        confidence=0.99, domain="economy",
                        reason="Cold start step 1 - enable attack for farming",
                    ))
                    # On the academy farm the map's actual spawns can include Thief Bug and
                    # Pupa — on many servers these give real EXP (Thief Bug ~188, Pupa ~157)
                    # and are low-HP, so ignoring them starves the bot of kills (0 EXP).
                    # Pragmatic + server-adaptive: attack whatever farmable mob actually
                    # spawns instead of hardcoding an ignore list. Only genuinely too-tough
                    # or 0-value mobs should be ignored — none here.
                    for _cs_attack in ["Pupa", "Thief Bug", "Thief Bug Egg"]:
                        actions.append(HeuristicAction(
                            kind="command", command=f"mon_control {_cs_attack}	0 1 1",
                            confidence=0.95, domain="economy",
                            reason=f"Cold start step 1 - attack {_cs_attack} if present (real EXP on this farm)",
                        ))
        if _cold_start_step == 2:
            # Step 2: Buy a starter weapon if none equipped — AGNOSTIC (gear planner picks
            # the best-affordable weapon by stat/zeny; fallback to a knife if planner empty).
            if not _has_weapon:
                _cs2_weapon_id = ""
                _cs2_weapon_name = ""
                try:
                    from ai_sidecar.gear_progression_planner import get_gear_progression_planner
                    _gpp_cs2 = get_gear_progression_planner()
                    _cs2_plan = _gpp_cs2.get_best_upgrade(base_level, zeny)
                    if _cs2_plan is not None and _cs2_plan.slot_name == "weapon" and _cs2_plan.is_affordable:
                        from ai_sidecar.knowledge_loader import get_weapons
                        for _w_cs2 in get_weapons():
                            if str(_w_cs2.get("Name", "") or _w_cs2.get("AegisName", "")).lower() == str(_cs2_plan.target_item).lower():
                                _cs2_weapon_id = str(_w_cs2.get("Id", ""))
                                _cs2_weapon_name = str(_cs2_plan.target_item)
                                break
                except Exception:
                    pass
                if not _cs2_weapon_id:
                    _cs2_opt = self._adaptive.get_optimal_weapon(job_name, base_level, zeny)
                    if _cs2_opt:
                        _cs2_weapon_id, _cs2_weapon_name = _cs2_opt
                if not _cs2_weapon_id:
                    _cs2_weapon_id = "1201"
                    _cs2_weapon_name = "Knife"
                if zeny >= 50:
                    actions.append(HeuristicAction(
                        kind="command", command=f"buy {_cs2_weapon_id} 1",
                        confidence=0.99, domain="economy",
                        reason=f"Cold start step 2 - buy {_cs2_weapon_name} (no weapon detected)",
                    ))
                    actions.append(HeuristicAction(
                        kind="command", command=f"equip {_cs2_weapon_id}",
                        confidence=0.99, domain="economy",
                        reason=f"Cold start step 2 - equip {_cs2_weapon_name} after purchase",
                    ))
            else:
                # Weapon confirmed — move to step 3 (buy potions)
                self._cold_start_step[_cs_stable_key] = 3
                _cold_start_step = 3
                logger.info(f"[cold_start] {bot_id}: weapon confirmed, step 2 -> 3")
        if _cold_start_step == 3:
            # Step 3: Buy potions (tiered by level) if no potions
            if not _has_potions:
                _cs_potion_id = self._get_potion_id(base_level)
                _cs_potion_cost = self._get_potion_cost(_cs_potion_id)
                _cs_potion_name = {501: "Red", 502: "Orange", 504: "White"}.get(_cs_potion_id, "Red")
                if zeny >= _cs_potion_cost:
                    # WEIGHT-AWARE BUYING: consider remaining weight capacity
                    _cs_weight_cap = NOVICE_WEIGHT_CAPACITY
                    _cs_max_by_weight = self._get_potion_max_buy(_cs_potion_id, zeny, weight, _cs_weight_cap)
                    _cs_max_by_zeny = int(zeny / _cs_potion_cost)
                    _cs_potion_qty = min(_cs_max_by_weight, _cs_max_by_zeny, 10)
                    if _cs_potion_qty > 0:
                        actions.append(HeuristicAction(
                            kind="command", command=f"buy {_cs_potion_id} {_cs_potion_qty}",
                            confidence=0.99, domain="economy",
                            reason=f"Cold start - buy {_cs_potion_qty} {_cs_potion_name} Potions "
                                   f"(level {base_level}, tier item {_cs_potion_id}, "
                                   f"weight={weight:.0%}, cap={_cs_weight_cap})",
                        ))
            else:
                # Potions confirmed — move to step 4
                self._cold_start_step[_cs_stable_key] = 4
                logger.info(f"[cold_start] {bot_id}: potions confirmed, step 3 -> 4")
        if _cold_start_step == 4:
            # Step 4: Return to hunting map with weapon and potions.
            # AGNOSTIC: use the SAME farm-map resolution as the routing domain
            # (get_best_map -> server_solutions farm_map) so cold-start and routing
            # never compete on different targets (verified live: step-4 hardcoded
            # prt_fild05 while routing emitted prt_fild08 -> oscillation, bot never
            # committed to one farm and died). Fall back to the learned farm_map.
            _cs4_farm = ""
            try:
                _cs4_farm = str(self._adaptive.get_best_map(bot_id, base_level) or "").lower()
            except Exception:
                pass
            if not _cs4_farm:
                try:
                    from ai_sidecar.server_adaptation import get_server_solutions_store
                    _cs4_farm = str((get_server_solutions_store().get("farm_map", None) or "") or "").strip().lower()
                except Exception:
                    pass
            if not _cs4_farm:
                _cs4_farm = "prt_fild05"
            if not _cs_in_hunting or _cs_map == "prt_fild01":
                actions.append(HeuristicAction(
                    kind="command", command=f"move {_cs4_farm}",
                    confidence=0.99, domain="hunting",
                    reason=f"Cold start - return to hunting map {_cs4_farm} with weapon and potions",
                ))
            elif _cs_in_hunting:
                # On hunting map with weapon + potions — cold start complete, advance to step 5
                if base_level >= 10:
                    self._cold_start_step[_cs_stable_key] = 5
                    logger.info(f"[cold_start] {bot_id}: on hunting map level {base_level}, step 4 -> 5")
                else:
                    self._cold_start_step[_cs_stable_key] = 4
                    logger.info(f"[cold_start] {bot_id}: on hunting map, cold start complete")
        # === BEYOND COLD START: progression pipeline steps 5+ ===
        if _cold_start_step == 5:
            # Step 5: Level to 10 on current hunting map
            if base_level >= 10:
                # Reached level 10 — advance to team evaluation
                self._cold_start_step[_cs_stable_key] = 6
                logger.info(f"[cold_start] {bot_id}: reached level {base_level}, step 5 -> 6")
            else:
                # Not level 10 yet — keep farming
                if _cs_in_hunting:
                    actions.append(HeuristicAction(
                        kind="command", command="ai auto",
                        confidence=0.99, domain="progression",
                        reason=f"Step 5 - farm to level 10 (currently {base_level})",
                    ))
                    actions.append(HeuristicAction(
                        kind="command", command="set attackAuto 3",
                        confidence=0.99, domain="progression",
                        reason=f"Step 5 - keep attacking until level 10 (currently {base_level})",
                    ))
                elif _cs_in_town:
                    actions.append(HeuristicAction(
                        kind="command", command="move prt_fild05",
                        confidence=0.99, domain="progression",
                        reason="Step 5 - return to hunting map to level",
                    ))
        if _cold_start_step == 6:
            # Step 6: Leader evaluates team, assigns jobs via LLM
            if self._team_jobs_assigned.get(_cs_stable_key, False):
                # Jobs already assigned — advance to step 7
                self._cold_start_step[_cs_stable_key] = 7
                logger.info(f"[cold_start] {bot_id}: jobs assigned, step 6 -> 7")
            else:
                # Track this bot's level in shared team state
                self._team_levels[_cs_stable_key] = base_level
                # Check if we're in town (must be in town for job change NPC access)
                if not _cs_in_town:
                    actions.append(HeuristicAction(
                        kind="command", command="move prontera",
                        confidence=0.99, domain="progression",
                        reason="Step 6 - return to town for job change",
                    ))
                elif _is_leader:
                    # Leader: check if ALL bots are level >= 10
                    # We need all bot levels. Use the shared _team_levels dict.
                    _all_ready = all(
                        self._team_levels.get(p, 0) >= 10
                        for p in _all_bots if p != _bot_profile
                    ) if _all_bots else False
                    _self_ready = base_level >= 10
                    if _all_ready and _self_ready:
                        # All bots ready — call team synergy API
                        try:
                            import json, requests
                            _payload = {
                                "bots": [
                                    {
                                        "bot_id": bot_id,
                                        "profile_name": _bot_profile,
                                        "base_level": base_level,
                                        "job_level": job_level,
                                        "current_job": job_name.title(),
                                    }
                                ]
                            }
                            # Add other bots from _team_levels
                            _resp = requests.post(
                                "http://127.0.0.1:18081/v1/conscious/team-synergy",
                                json=_payload, timeout=15.0,
                            )
                            if _resp.ok:
                                _data = _resp.json()
                                _assignments = _data.get("assignments", [])
                                for _a in _assignments:
                                    _prof = _a.get("profile_name", "")
                                    _job = _a.get("recommended_job", "Acolyte")
                                    actions.append(HeuristicAction(
                                        kind="command",
                                        command=f"job_change {_prof} {_job}",
                                        confidence=0.99, domain="progression",
                                        reason=f"Team synergy: {_prof} -> {_job} ({_a.get('role', '')})",
                                    ))
                                self._team_jobs_assigned[_cs_stable_key] = True
                                # Store each bot assigned job
                                for _a in _assignments:
                                    _prof = _a.get("profile_name", "")
                                    _job = _a.get("recommended_job", "Acolyte")
                                    self._assigned_jobs[_prof] = _job
                                logger.info(f"[team_synergy] {bot_id}: team jobs assigned via LLM")
                            else:
                                logger.warning(f"[team_synergy] API error: {_resp.status_code}")
                        except Exception as _e:
                            logger.warning(f"[team_synergy] call failed: {_e} — using knowledge fallback")
                            # Knowledge fallback: assign jobs based on position
                            _fallback_jobs = ["Acolyte", "Mage", "Swordsman", "Hunter", "Thief", "Merchant"]
                            for _i, _p in enumerate(_all_bots):
                                if _i < len(_fallback_jobs):
                                    actions.append(HeuristicAction(
                                        kind="command",
                                        command=f"job_change {_p} {_fallback_jobs[_i]}",
                                        confidence=0.95, domain="progression",
                                        reason=f"Knowledge fallback: {_p} -> {_fallback_jobs[_i]}",
                                    ))
                            self._team_jobs_assigned[_cs_stable_key] = True
                else:
                    # Follower: wait for leader to assign job
                    pass  # Leader will send job_change command
                # If not leader, just wait
        if _cold_start_step == 7:
            # Step 7: Execute job change (walk to NPC, talk)
            _is_novice = job_name == "novice"
            # Check if we have an assigned job (from leader or manual)
            _assigned_job = self._assigned_jobs.get(_bot_profile, "").lower()
            if not _is_novice:
                # Job change complete — advance
                self._cold_start_step[_cs_stable_key] = 8
                logger.info(f"[cold_start] {bot_id}: job changed to {job_name}, step 7 -> 8")
            elif _assigned_job:
                # Have an assigned job — look up NPC
                _jc_data = JOB_CHANGE_2_1.get(_assigned_job)
                if _jc_data:
                    _jc_map, _jc_x, _jc_y, _jc_talk_seq = _jc_data
                    if _cs_in_town:
                        # Walk to NPC talk spot
                        _talk_area_x = _jc_x + 1
                        _talk_area_y = _jc_y + 1
                        actions.append(HeuristicAction(
                            kind="command", command=f"move {_talk_area_x} {_talk_area_y}",
                            confidence=0.99, domain="progression",
                            reason=f"Step 7 - walk to {_assigned_job} job change NPC at ({_jc_x},{_jc_y})",
                        ))
                        # After walking, talk to NPC
                        # Use talknpc with the NPC coordinates
                        _talk_cmd = f"talknpc {_jc_x} {_jc_y}"
                        for _t in _jc_talk_seq:
                            _talk_cmd += " " + _t.replace("talk @npc@", "").strip()
                        actions.append(HeuristicAction(
                            kind="command", command=_talk_cmd,
                            confidence=0.99, domain="progression",
                            reason=f"Step 7 - talk to {_assigned_job} job change NPC",
                        ))
                    else:
                        _town7 = self._resolve_safe_town()
                        actions.append(HeuristicAction(
                            kind="command", command=f"move {_town7}",
                            confidence=0.99, domain="progression",
                            reason=f"Step 7 - go to {_town7} for job change",
                        ))
            else:
                # No assigned job yet — go to town and wait
                if not _cs_in_town:
                    _town7b = self._resolve_safe_town()
                    actions.append(HeuristicAction(
                        kind="command", command=f"move {_town7b}",
                        confidence=0.99, domain="progression",
                        reason=f"Step 7 - go to town {_town7b}, waiting for job assignment",
                    ))
        if _cold_start_step == 8:
            # Step 8: Post-job-change first hunt on appropriate map.
            # AGNOSTIC: resolve from game_knowledge_db by job class hint, else the
            # reachable_hunting_maps resolver — never a hardcoded job->map dict.
            _hunt_map = ""
            try:
                _gkd = GameKnowledgeDB()
                _zone = _gkd.get_hunting_zone(int(base_level or 1), class_hint=str(job_name or "all"))
                if _zone and _zone.get("map_name"):
                    _hunt_map = str(_zone["map_name"])
            except Exception:
                _hunt_map = ""
            if not _hunt_map:
                try:
                    _reach8 = _mk_reachable_hunting_maps(str(_cs_map or ""), int(base_level or 1))
                    if _reach8:
                        _hunt_map = _reach8[0][0]
                except Exception:
                    _hunt_map = ""
            if not _hunt_map:
                _hunt_map = "prt_fild05"  # true last-resort cold-start fallback
            if _cs_in_hunting:
                # On hunting map — farm
                actions.append(HeuristicAction(
                    kind="command", command="ai auto",
                    confidence=0.99, domain="progression",
                    reason=f"Step 8 - farm {_hunt_map} as {job_name}",
                ))
                actions.append(HeuristicAction(
                    kind="command", command="set attackAuto 3",
                    confidence=0.99, domain="progression",
                    reason=f"Step 8 - enable attack on {_hunt_map}",
                ))
            else:
                actions.append(HeuristicAction(
                    kind="command", command=f"move {_hunt_map}",
                    confidence=0.99, domain="progression",
                    reason=f"Step 8 - move to {_hunt_map} for post-job farming",
                ))
        # ── ZERO POTIONS ON HUNTING MAP: force return to town ──
        # Runs every cycle for ALL bots, not just COLD_START or hunting maps
        # Fixes: attackMaxDistance, attackDistance, attackAuto, avoidList settings
        # MUST run before state-specific returns to ensure config is always applied
        _audit_map = signals.get("map", "") or ""
        _audit_is_hunting = any(x in _audit_map for x in ["prt_fild", "pay_fild", "mjolnir", "gef_fild", "ra_fild", "moc_fild", "cmd_fild"])
        _audit_is_town = any(x in _audit_map for x in ["prontera", "morocc", "geffen", "payon", "aldebaran", "alberta", "izlude"])
        # ── COLD-START POTION ECONOMY (O3, runs for ALL bots incl. level 1-2) ──
        # A low-level bot that can't buy potions runs dry and flees/wanders to town. Point the
        # buyAuto potion at the TOWN shop and enable sellAuto so loot funds the potions. The
        # potion command + safe_town come from the DB-backed server_solutions store (server-
        # specific per RULE.md), NOT a hardcoded literal. RATE-LIMITED to once per 20s per bot
        # so it cannot flood the action queue and starve actual movement/routing.
        try:
            from ai_sidecar.server_adaptation import get_server_solutions_store
            _aum = str(_audit_map or "").lower().replace(".gat", "")
            if _aum not in ("iz_ac01_a", "iz_ac01"):  # must not override the academy-room exit
                _now_e = __import__("time").time()
                if _now_e - self._last_economy_emit.get(bot_id, 0.0) >= 20.0:
                    self._last_economy_emit[bot_id] = _now_e
                    _sstore = get_server_solutions_store()
                    try:
                        _potion_buy = str(_sstore.get_json("potion_solution", {}).get("buy_command") or "").strip()
                        _safe_town_buy = str(_sstore.get("safe_town", "") or "").strip() or "prontera"
                    except Exception:
                        _potion_buy, _safe_town_buy = "", ""
                    if not _potion_buy:
                        _potion_buy = "buy 501 30"
                    if _potion_buy:
                        self._set_config_once(actions, bot_id, "buyAuto_Red_Potion", _potion_buy, "economy",
                            f"Cold-start potion sustain: {_potion_buy} (from server_solutions DB)")
                    # Point the buy/sell NPC at the town shop (server-specific, DB-backed).
                    _town_npc = {"prontera": "prt_in 126 75", "izlude": "izlude 116 92"}.get(_safe_town_buy, f"{_safe_town_buy} 0 0")
                    self._set_config_once(actions, bot_id, "buyAuto_npc", _town_npc, "economy",
                        f"Buy potions at {_safe_town_buy} town shop (from server_solutions DB)")
                    self._set_config_once(actions, bot_id, "sellAuto", "1", "economy",
                        "Cold-start: auto-sell loot to fund potions")
        except Exception:
            pass
        if _audit_is_hunting:
            # ── PER-MAP MON_CONTROL (config audit) ──
            # Apply mon_control for hunting maps when bot is on a new map
            self._emit_mon_control_for_map(actions, bot_id, _audit_map, int(signals.get("base_level", 1) or 1))
            # Ensure attack config is optimal for Novice-level combat
            # RULE.md: NO hardcoded server-specific values. A Novice's melee
            # range is 1 cell (Knife) — the old 30/5 made the bot attack from
            # 5-30 cells (server melee range is still 1 -> EVERY attack missed
            # -> "in-range hit timeout" loop). Use the melee reality (1/2) —
            # the weapon's actual range comes from the game's own tables, not
            # a hardcoded chase distance.
            self._set_config_once(actions, bot_id, "attackMaxDistance", "2", "hunting",
                "Config audit - melee chase range (weapon range is 1 cell)")
            self._set_config_once(actions, bot_id, "attackDistance", "1", "hunting",
                "Config audit - melee attack distance (weapon range is 1 cell)")
            # Always enable attackAuto on hunting maps (includes cold start farming)
            _aa_val = "2" if base_level < 10 else "3"
            self._set_config_once(actions, bot_id, "attackAuto", _aa_val, "hunting",
                f"Config audit - attackAuto={_aa_val} (level {base_level})")
            # Proper attack targeting: only attack in lockMap, only when safe
            self._set_config_once(actions, bot_id, "attackAuto_inLockOnly", "1", "hunting",
                "Config audit - only attack monsters inside lockMap")
            self._set_config_once(actions, bot_id, "attackAuto_onlyWhenSafe", "1", "hunting",
                "Config audit - only attack when no aggressive monsters nearby")
            self._set_config_once(actions, bot_id, "attackAuto_onlyInSearch", "1", "hunting",
                "Config audit - only attack monsters in search area")
            self._set_config_once(actions, bot_id, "attackAuto_startOnSight", "0", "hunting",
                "Config audit - don't auto-start on sight (wait for search")
            # Ensure avoidList is disabled so we don't run from Porings
            self._set_config_once(actions, bot_id, "avoidList", "", "hunting",
                "Config audit - disable avoidList (prevents running from farm targets)")
            self._set_config_once(actions, bot_id, "avoidList_inLockOnly", "", "hunting",
                "Config audit - disable inLockOnly avoidList")
            # Enable aggressive attack on monsters within distance
            self._set_config_once(actions, bot_id, "attackDistanceAuto", "1", "hunting",
                "Config audit - auto-adjust attack distance")
            self._set_config_once(actions, bot_id, "attackMaxDistance", "30", "hunting",
                "Config audit - chase distance 30 cells")
            self._set_config_once(actions, bot_id, "attackDistance", "5", "hunting",
                "Config audit - start attacking from 5 cells away")
            self._set_config_once(actions, bot_id, "attackAuto_unstuck", "1", "hunting",
                "Config audit - don't give up mid-fight")
            # ── LOOTING CONFIG: auto-loot everything ──
            self._set_config_once(actions, bot_id, "itemsTakeAuto", "2", "hunting",
                "Config audit - auto-take all dropped items")
            self._set_config_once(actions, bot_id, "itemsGatherAuto", "2", "hunting",
                "Config audit - auto-gather all items")
            self._set_config_once(actions, bot_id, "itemsTakeAuto_party", "0", "hunting",
                "Config audit - don't take party members' drops")
            # ── SITTING CONFIG: let bot use sitAuto for HP regen ──
            self._set_config_once(actions, bot_id, "sitAuto_hp_lower", "20", "hunting",
                "Config audit - sit when HP < 20%")
            self._set_config_once(actions, bot_id, "sitAuto_hp_upper", "50", "hunting",
                "Config audit - stand when HP > 50%")
            self._set_config_once(actions, bot_id, "sitAuto_over_50", "0", "hunting",
                "Config audit - never sit due to weight")
            self._set_config_once(actions, bot_id, "sitAuto_idle", "0", "hunting",
                "Config audit - never sit idle")
            # ── SELL CONFIG: auto-sell when overweight ──
            self._set_config_once(actions, bot_id, "sellAuto", "1", "hunting",
                "Config audit - auto-sell loot when inventory full")
            self._set_config_once(actions, bot_id, "sellAuto_npc", "prt_in 126 75", "hunting",
                "Config audit - Tool Dealer in Prontera")
            self._set_config_once(actions, bot_id, "sellAuto_distance", "25", "hunting",
                "Config audit - walk up to 25 cells to sell")
            self._set_config_once(actions, bot_id, "sellAuto_maxWeight", "70", "hunting",
                "Config audit - sell when weight > 70%")
            self._set_config_once(actions, bot_id, "sellAuto_minZen", "0", "hunting",
                "Config audit - sell even with 0 zeny")
            # ── STORAGE CONFIG: deposit heavy items at Kafra for free ──
            self._set_config_once(actions, bot_id, "storageAuto", "1", "hunting",
                "Config audit - auto-deposit at Kafra")
            self._set_config_once(actions, bot_id, "storageAuto_distance", "5", "hunting",
                "Config audit - stand next to Kafra")
            self._set_config_once(actions, bot_id, "relogAfterStorage", "0", "hunting",
                "Config audit - don't relog after storage")
            self._set_config_once(actions, bot_id, "minStorageZeny", "0", "hunting",
                "Config audit - 0 zeny needed to use storage")
            # ── TELEPORT CONFIG: escape from danger ──
            self._set_config_once(actions, bot_id, "teleportAuto_hp", "10", "hunting",
                "Config audit - teleport when HP < 10%")
            self._set_config_once(actions, bot_id, "teleportAuto_deadly", "1", "hunting",
                "Config audit - teleport from deadly monsters")
            self._set_config_once(actions, bot_id, "attackAuto_startOnSight", "1", "hunting",
                "Config audit - attack monsters as soon as they appear")
            self._set_config_once(actions, bot_id, "attackAuto_unstuck", "1", "hunting",
                "Config audit - don't give up mid-fight")
            # ── LOOTING CONFIG: auto-loot everything ──
            self._set_config_once(actions, bot_id, "itemsTakeAuto", "2", "hunting",
                "Config audit - auto-take all dropped items")
            self._set_config_once(actions, bot_id, "itemsGatherAuto", "2", "hunting",
                "Config audit - auto-gather all items")
            self._set_config_once(actions, bot_id, "itemsTakeAuto_party", "0", "hunting",
                "Config audit - don't take party members' drops")
            # ── SITTING CONFIG: let bot use sitAuto for HP regen ──
            self._set_config_once(actions, bot_id, "sitAuto_hp_lower", "20", "hunting",
                "Config audit - sit when HP < 20%")
            self._set_config_once(actions, bot_id, "sitAuto_hp_upper", "50", "hunting",
                "Config audit - stand when HP > 50%")
            self._set_config_once(actions, bot_id, "sitAuto_over_50", "0", "hunting",
                "Config audit - never sit due to weight")
            self._set_config_once(actions, bot_id, "sitAuto_idle", "0", "hunting",
                "Config audit - never sit idle")
            # ── SELL CONFIG: auto-sell when overweight ──
            self._set_config_once(actions, bot_id, "sellAuto", "1", "hunting",
                "Config audit - auto-sell loot when inventory full")
            self._set_config_once(actions, bot_id, "sellAuto_npc", "prt_in 126 75", "hunting",
                "Config audit - Tool Dealer in Prontera")
            self._set_config_once(actions, bot_id, "sellAuto_distance", "25", "hunting",
                "Config audit - walk up to 25 cells to sell")
            self._set_config_once(actions, bot_id, "sellAuto_maxWeight", "70", "hunting",
                "Config audit - sell when weight > 70%")
            self._set_config_once(actions, bot_id, "sellAuto_minZen", "0", "hunting",
                "Config audit - sell even with 0 zeny")
            # ── TELEPORT CONFIG: escape from danger ──
            self._set_config_once(actions, bot_id, "teleportAuto_hp", "10", "hunting",
                "Config audit - teleport when HP < 10%")
            self._set_config_once(actions, bot_id, "teleportAuto_deadly", "1", "hunting",
                "Config audit - teleport from deadly monsters")
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
            # ── ZERO POTIONS ON HUNTING MAP ──
            # A bot on a hunting map with no potions used to be force-returned to town
            # unconditionally — that abandoned the farm on every potion gap (no commitment).
            # Now it STAYS and only returns to town on a real emergency (see gate below).
            # NOTE: inventory data is stored as 'inventory_items' in signals (from pdca_loop.py)
            # The bridge's auto-stand handles sitting bots before executing 'move prontera'.
            # The portal exit check (in bridge) breaks the loop if bot arrives and turns back.
            _audit_items = signals.get("inventory_items", []) or []
            _audit_has_potions = any(
                "potion" in str(item).lower() or "red" in str(item).lower() or "orange" in str(item).lower() or "white" in str(item).lower()
                for item in _audit_items
            )
            if not _audit_has_potions:
                # Only return to town if it is a REAL emergency — the bot is genuinely unable
                # to continue (HP critical or bag full). A level 1-3 novice can farm the
                # academy mobs (Poring/Lunatic/Fabre are low-HP and farmable bare-handed) and
                # loot to buy potions on the spot; abandoning the farm for every potion gap is
                # exactly the "no commitment" failure a Pro would never make. SO: leave the
                # farm only when HP is critical (<30%) or overweight — not on potion drought.
                _audit_hp = float(signals.get("hp_ratio", 1.0) or 1.0)
                _audit_weight = float(signals.get("weight_ratio", 0.0) or 0.0)
                _audit_map_low = str(signals.get("map", "") or "").lower()
                _audit_on_farm = any(x in _audit_map_low for x in ["prt_fild08", "prt_fild08c", "prt_fild05"])
                _real_emergency = (_audit_hp < 0.30) or (_audit_weight > 0.70)
                if _real_emergency and not _audit_on_farm:
                    # Only emergency-return when NOT on a farm map (a farm bot should restock
                    # in place / via buyAuto, not abandon the field unless truly critical).
                    _audit_now = __import__("time").time()
                    _audit_last_return = self._last_return_to_town.get(bot_id, 0)
                    if _audit_now - _audit_last_return > 60:
                        self._last_return_to_town[bot_id] = _audit_now
                        actions.append(HeuristicAction(
                            kind="command", command="move prontera",
                            confidence=0.99, domain="economy",
                            reason="Emergency: critical HP or bag full off-farm — return to town",
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
            _aa_val_t = "2" if base_level < 10 else "3"
            self._set_config_once(actions, bot_id, "attackAuto", _aa_val_t, "hunting",
                f"Config audit (town) - attackAuto={_aa_val_t} (level {base_level})")
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
            _audit_zeny = signals.get("zeny", 0) or 0  # init before any branch references it
            if not _audit_has_potions:
                _audit_potion_id = self._get_potion_id(base_level)
                _audit_potion_cost = self._get_potion_cost(_audit_potion_id)
                _audit_potion_name = {501: "Red", 502: "Orange", 504: "White"}.get(_audit_potion_id, "Red")
                if _audit_zeny >= _audit_potion_cost:
                    _audit_potion_qty = min(int(_audit_zeny / _audit_potion_cost), 10)
                    if _audit_potion_qty > 0:
                        actions.append(HeuristicAction(
                            kind="command", command=f"buy {_audit_potion_id} {_audit_potion_qty}",
                            confidence=0.99, domain="economy",
                            reason=f"Town - buy {_audit_potion_qty} {_audit_potion_name} Potions (0 potions in inventory, level {base_level})",
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
                    # Already have a party - report missing members
                    # OBSERVABILITY ONLY: the fleet coordinator is the party
                    # actor. Raw `party request` emissions produced frozen
                    # party-request spam (bridge gates could only LOG the
                    # block — Commands::run dispatches the handler with the
                    # ORIGINAL switch, so the hook cannot stop execution).
                    for _other_bot in _all_bots:
                        if _other_bot != _bot_profile:
                            _char_name = _other_bot  # Fallback: use profile name
                            _already_in = any(_char_name.lower() in m.lower() for m in _party_members)
                            if not _already_in:
                                actions.append(HeuristicAction(
                                    kind="log", command="party_member_missing",
                                    confidence=0.95, domain="social",
                                    reason="Direct party check - request " + str(_other_bot) + " (" + str(_char_name) + ")",
                                    metadata={"party_action": "request_pending", "member": _char_name},
                                ))
                            elif _actual_count < 3:
                                actions.append(HeuristicAction(
                                    kind="log", command="party_stale_check",
                                    confidence=0.80, domain="social",
                                    reason="Direct party check - retry " + str(_other_bot) + " (" + str(_char_name) + ") - stale check",
                                    metadata={"party_action": "retry_pending", "member": _char_name},
                                ))
                else:
                    # Not in party - report formation need (coordinator acts)
                    actions.append(HeuristicAction(
                        kind="log", command="party_formation_needed",
                        confidence=0.99, domain="social",
                        reason="Direct party check - move to town for party formation",
                        metadata={"party_action": "town_pending"},
                    ))
                    actions.append(HeuristicAction(
                        kind="log", command="party_create_pending",
                        confidence=0.95, domain="social",
                        reason="Direct party check - leader creates party",
                        metadata={"party_action": "create_pending", "suffix": "AI" + str(_ts)},
                    ))
                    # Report ALL other bots that would be invited
                    for _other_bot in _all_bots:
                        if _other_bot != _bot_profile:
                            _char_name = _other_bot  # Fallback: use profile name
                            actions.append(HeuristicAction(
                                kind="log", command="party_invite_pending",
                                confidence=0.95, domain="social",
                                reason="Direct party check - request " + str(_other_bot) + " (" + str(_char_name) + ")",
                                metadata={"party_action": "invite_pending", "member": _char_name},
                            ))
                    actions.append(HeuristicAction(
                        kind="log", command="party_share_pending",
                        confidence=0.90, domain="social",
                        reason="Share experience in party",
                        metadata={"party_action": "share_pending"},
                    ))
                actions.append(HeuristicAction(
                    kind="log", command="ai_mode_auto",
                    confidence=0.95, domain="planning",
                    reason="Continue after party attempt [log-only: config-audit owns AI mode]",
                    metadata={"ai_mode": "auto"},
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
            # OBSERVABILITY ONLY — fleet coordinator is the party actor.
            if _party_in:
                actions.append(HeuristicAction(
                    kind="log", command="party_stale_leave_pending",
                    confidence=0.99, domain="social",
                    reason="Direct party check - leave stale party",
                    metadata={"party_action": "leave_pending"},
                ))
            actions.append(HeuristicAction(
                kind="log", command="party_autojoin_pending",
                confidence=0.99, domain="social",
                reason="Direct party check - set auto-accept",
                metadata={"party_action": "autojoin_pending"},
            ))
            actions.append(HeuristicAction(
                kind="log", command="party_town_wait_pending",
                confidence=0.95, domain="social",
                reason="Direct party check - move to town for party invite",
                metadata={"party_action": "town_pending"},
            ))
            actions.append(HeuristicAction(
                kind="log", command="ai_mode_auto",
                confidence=0.95, domain="planning",
                reason="Continue after party attempt [log-only: config-audit owns AI mode]",
                metadata={"ai_mode": "auto"},
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
                    kind="log", command="party_leave_requested",
                    confidence=0.99, domain="social",
                    reason=f"Level {_audit_base_level} < 40 - force leave party (solo is faster)",
                    metadata={"party_action": "leave"},
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

        # ── RETURN-TO-FARM AFTER RESTOCK (COMPLETED per completeness mandate) ──
        # The inverse of force-return: a bot that went to town to restock (now HAS
        # potions + weapon) but is IDLE on a non-farm map must deterministically be
        # sent back to its farm — otherwise it parks in town forever (the compounding
        # blocker: bot10 farming to level 4 then parking in prontera). This closes the
        # farm -> restock -> return-to-farm loop that the orchestrator was meant to
        # cover but which must also work per-bot (no silent dependency on the captain).
        _audit_cs_hunt = getattr(self, "_cold_start_hunt_map", {}).get(bot_id, "prt_fild08c") or "prt_fild08c"
        _audit_farm_for_return = _audit_cs_hunt
        if _audit_is_town and _audit_has_potions and _audit_has_weapon and \
           _audit_map not in ("iz_ac01_a", "iz_ac01") and state != "DEAD":
            _audit_now2 = __import__("time").time()
            _audit_last_return = self._last_return_to_farm.get(bot_id, 0)
            if _audit_now2 - _audit_last_return > 45:  # 45s cooldown, then keep nudging home
                self._last_return_to_farm[bot_id] = _audit_now2
                actions.append(HeuristicAction(
                    kind="command", command=f"move {_audit_farm_for_return}",
                    confidence=0.95, domain="hunting",
                    reason=f"Return to farm {_audit_farm_for_return} after restock (has potions+weapon, idle in town)",
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
            # Skip during cold start step 0 — pipeline uses ai manual + move for portal walk
            _cs_map_str = str(signals.get("map", ""))
            _cs_in_town_str = any(x in _cs_map_str for x in ["prontera", "morocc", "geffen", "payon", "aldebaran", "alberta", "izlude"])
            _cs_z = int(signals.get("zeny", 0) or 0)
            _cs_has_weapon = bool(signals.get("weapon", {}).get("id", 0))
            if _cs_has_weapon or _cs_in_town_str or _cs_z >= 50:
                actions.append(HeuristicAction(
                    kind="command", command="ai auto",
                    confidence=0.99, domain="hunting",
                    reason="Cold start - enable auto-attack before moving",
                ))
            # Combat config (deduped) — skip attackAuto during pipeline step 0
            self._set_config_once(actions, bot_id, "attackDistance", "7", "hunting",
                "Cold start - set attack distance")
            self._set_config_once(actions, bot_id, "attackMaxDistance", "20", "hunting",
                "Cold start - set chase distance")
            # Only enable attack if bot has a weapon OR is in town (not pipeline step 0)
            _cs_map_str = str(signals.get("map", ""))
            _cs_in_town_str = any(x in _cs_map_str for x in ["prontera", "morocc", "geffen", "payon", "aldebaran", "alberta", "izlude"])
            _cs_z = int(signals.get("zeny", 0) or 0)
            _cs_has_weapon = bool(signals.get("weapon", {}).get("id", 0))
            if _cs_has_weapon or _cs_in_town_str or _cs_z >= 50:
                _aa_val_h = "2" if base_level < 10 else "3"
                self._set_config_once(actions, bot_id, "attackAuto", _aa_val_h, "hunting",
                    f"attackAuto={_aa_val_h} (level {base_level})")
            self._set_config_once(actions, bot_id, "attackAuto_startOnSight", "1", "hunting",
                "Attack monsters as soon as they appear")
            self._set_config_once(actions, bot_id, "attackAuto_unstuck", "1", "hunting",
                "Cold start - don't give up mid-fight")
            # COLD START: Go to safe field first (prt_fild05 for level 1-10)
            # Then progress to dungeon at level 10+
            _cs_hunt_map = "prt_fild05"
            _cs_portal_coords = "22 203"  # Portal from Prontera (156, 164) -> prt_fild05 (22, 203)
            # Set route_randomWalk and lockMap — skip during cold start step 0
            # (bot on hunting map, 0 potions, no weapon, routing to Prontera via bridge)
            # Also skip during pipeline step 1 (farming prt_fild01 for 50z)
            _cs_pipeline_active = _cold_start_step == 1 and not _cs_has_weapon and _cs_z < 50
            if (_cs_has_weapon or _cs_in_town_str or _cs_z >= 50) and not _cs_pipeline_active:
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
            _cs_potion_id = self._get_potion_id(base_level)
            _cs_potion_cost = self._get_potion_cost(_cs_potion_id)
            _cs_potion_name = self._get_potion_name(_cs_potion_id)
            if _cs_zeny >= _cs_potion_cost * 10:
                # Buy 10 potions for survival
                actions.append(HeuristicAction(
                    kind="command", command=f"buy {_cs_potion_id} 10",
                    confidence=0.99, domain="economy",
                    reason=f"Cold start - buy 10 {_cs_potion_name} Potions (item {_cs_potion_id}, level {base_level})",
                ))
            elif _cs_zeny >= _cs_potion_cost:
                # Buy at least 1 potion if we can afford it
                _cs_potion_qty = min(int(_cs_zeny / _cs_potion_cost), 10)
                if _cs_potion_qty > 0:
                    actions.append(HeuristicAction(
                        kind="command", command=f"buy {_cs_potion_id} {_cs_potion_qty}",
                        confidence=0.99, domain="economy",
                        reason=f"Cold start - buy {_cs_potion_qty} {_cs_potion_name} Potions (item {_cs_potion_id}, level {base_level})",
                    ))
            # Buy arrows if enough zeny
            if _cs_zeny >= 200:
                actions.append(HeuristicAction(
                    kind="command", command="buy 1750 200",
                    confidence=0.99, domain="economy",
                    reason="Buy 200 arrows (harmless for non-archers, critical for archers)",
                ))
            # Buy weapon if any zeny available — AGNOSTIC: pick the best-affordable
            # weapon from the DB-backed gear progression planner (stat/zeny ranked),
            # NOT a hardcoded item ID (RULE.md). Fall back to the level/job-aware
            # get_optimal_weapon only if the gear planner is unavailable.
            _cs_job = signals.get("job_name", "novice") or "novice"
            _cs_weapon_id = None
            _cs_weapon_name = ""
            try:
                from ai_sidecar.gear_progression_planner import get_gear_progression_planner
                _gpp_cs = get_gear_progression_planner()
                _cs_plan = _gpp_cs.get_best_upgrade(base_level, _cs_zeny)
                if _cs_plan is not None and _cs_plan.slot_name == "weapon" and _cs_plan.is_affordable:
                    # resolve the chosen item's real ID from the knowledge DB (by name)
                    from ai_sidecar.knowledge_loader import get_weapons
                    for _w_cs in get_weapons():
                        _w_nm = str(_w_cs.get("Name", "") or _w_cs.get("AegisName", "")).lower()
                        if _w_nm == str(_cs_plan.target_item).lower():
                            _cs_weapon_id = str(_w_cs.get("Id", ""))
                            _cs_weapon_name = str(_cs_plan.target_item)
                            break
            except Exception:
                pass
            if not _cs_weapon_id:
                _cs_opt = self._adaptive.get_optimal_weapon(_cs_job, base_level, _cs_zeny)
                if _cs_opt:
                    _cs_weapon_id, _cs_weapon_name = _cs_opt
            if _cs_weapon_id and _cs_zeny >= 50:
                actions.append(HeuristicAction(
                    kind="command", command=f"buy {_cs_weapon_id} 1",
                    confidence=0.99, domain="economy",
                    reason=f"Cold start - buy weapon {_cs_weapon_name} (id {_cs_weapon_id}) for {_cs_job}",
                ))
                # Equip the weapon after buying — bare fists do 1-5 damage
                actions.append(HeuristicAction(
                    kind="command", command=f"equip {_cs_weapon_id}",
                    confidence=0.99, domain="economy",
                    reason=f"Cold start - equip weapon {_cs_weapon_name} after purchase",
                ))
            # Attack config: increase chase distance, disable avoid system
            self._set_config_once(actions, bot_id, "attackMaxDistance", "30", "hunting",
                "Cold start - increase chase distance to 30 for Novice attack range")
            self._set_config_once(actions, bot_id, "attackDistance", "5", "hunting",
                "Cold start - attack from 5 cells away")
            _aa_val4 = "2" if base_level < 10 else "3"
            self._set_config_once(actions, bot_id, "attackAuto", _aa_val4, "hunting",
                f"attackAuto={_aa_val4} (level {base_level})")
            self._set_config_once(actions, bot_id, "attackAuto_startOnSight", "1", "hunting",
                "Attack monsters as soon as they appear")
            self._set_config_once(actions, bot_id, "attackAuto_unstuck", "1", "hunting",
                "Cold start - don't give up mid-fight")
            # Teleport config — use config audit defaults (teleportAuto_hp=10)
            self._set_config_once(actions, bot_id, "teleportAuto_minAggressives", "8", "hunting",
                "Only teleport at 8+ mobs")
            self._set_config_once(actions, bot_id, "teleportAuto_deadly", "0", "hunting",
                "Disable deadly teleport")
            # Only send move if on a different map
            if map_name != _cs_hunt_map:
                # RATE-LIMIT FIX (the park-in-town mechanism): this emitted `move <farm>`
                # EVERY cycle the bot was off-farm, flooding the client's command queue —
                # each move triggered a fresh route recalculation at the bot's spot (the
                # 94x 'Calculating route to prontera' storm), so bot10 never committed to
                # walking and stayed parked at 156,26 with EXP frozen. Send ONE move, let
                # the client walk; re-send at most once per 20s.
                _cs_mv_now = __import__("time").time()
                if _cs_mv_now - self._last_return_to_farm.get(bot_id, 0.0) >= 20.0:
                    self._last_return_to_farm[bot_id] = _cs_mv_now
                    actions.append(HeuristicAction(
                        kind="command", command=f"move {_cs_hunt_map}",
                        confidence=0.99, domain="hunting",
                        reason=f"Cold start - move to {_cs_hunt_map} (rate-limited once/20s)",
                    ))
            # Party creation for leader — only at level 40+ (solo before 40 is faster)
            _cs_bot_profile = bot_id.split(":")[-1].split("/")[-1] if ":" in bot_id else bot_id
            _cs_all_bots = signals.get("all_bots", []) or list(self._bot_roles.keys()) if hasattr(self, '_bot_roles') else []
            _cs_sorted = sorted(_cs_all_bots)
            _cs_is_leader = len(_cs_sorted) > 0 and _cs_bot_profile == _cs_sorted[0]
            _cs_base_level = signals.get("base_level", 1) or 1
            if _cs_is_leader and _cs_base_level >= 40:
                actions.append(HeuristicAction(
                    kind="log", command="party_create_pending",
                    confidence=0.99, domain="social",
                    reason="Cold start - leader creates party with unique name",
                    metadata={"party_action": "create_pending", "suffix": f"AI{int(_now_t)}"},
                ))
                actions.append(HeuristicAction(
                    kind="log", command="party_share_pending",
                    confidence=0.95, domain="social",
                    reason="Share experience in party",
                    metadata={"party_action": "share_pending"},
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
            _inv_count = signals.get("inventory_items", [])
            if isinstance(_inv_count, list):
                _has_items = len(_inv_count) > 0
            else:
                _has_items = int(_inv_count or 0) > 0
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
            _death_potion_id = self._get_potion_id(base_level)
            _death_potion_cost = self._get_potion_cost(_death_potion_id)
            _death_potion_name = {501: "Red", 502: "Orange", 504: "White"}.get(_death_potion_id, "Red")
            if _death_zeny >= _death_potion_cost * 10:
                actions.append(HeuristicAction(
                    kind="command", command=f"buy {_death_potion_id} 10",
                    confidence=0.99, domain="economy",
                    reason=f"Death recovery - buy 10 {_death_potion_name} Potions (item {_death_potion_id}, level {base_level})",
                ))
            elif _death_zeny >= _death_potion_cost:
                _death_potion_qty = min(int(_death_zeny / _death_potion_cost), 10)
                if _death_potion_qty > 0:
                    actions.append(HeuristicAction(
                        kind="command", command=f"buy {_death_potion_id} {_death_potion_qty}",
                        confidence=0.99, domain="economy",
                        reason=f"Death recovery - buy {_death_potion_qty} {_death_potion_name} Potions (item {_death_potion_id}, level {base_level})",
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
            # Return to hunt via portal after 15s in town.
            # AGNOSTIC (founder 2026-08-25): this block must NOT bake server-specific
            # facts (prt_fild05/22 203 are RAW-specific literals). It is skipped during
            # ANY cold-start phase (steps 0-3 = academy/starter plan owns movement) and
            # the farm portal is resolved from the knowledge DB (a FACT); when no fact
            # exists, the block emits NOTHING and the conscious LLM decides instead.
            _rth_step = self._cold_start_step.get(_cs_stable_key, 0)
            _rth_skip = _rth_step < 4
            if not _rth_skip and int(signals.get("zeny", 0) or 0) < 50 and not _has_weapon:
                _rth_skip = True
            _town_time = __import__("time").time() - self._town_entry_time.get(bot_id, __import__("time").time())
            # int_land (Secluded Island) has NO route to a field map; locking a hunt
            # map there makes OpenKore spin. The island escape must run instead.
            _is_island = str(signals.get("map", "") or "").lower().startswith("int_land")
            if not _rth_skip and _town_time > 15 and not _is_island:
                # Resolve the hunt map from the knowledge DB (FACT, server-agnostic).
                _rth_hunt_map = ""
                try:
                    _rth_hunt_map = str((self._get_npc("farm_map", map_name) or {}).get("map") or "") if False else ""
                except Exception:
                    pass
                if not _rth_hunt_map:
                    _rth_hunt_map = str(signals.get("hunt_map", "") or "")
                self._set_config_once(actions, bot_id, "lockMap_randX", "30", "hunting",
                    "Random walk radius X")
                self._set_config_once(actions, bot_id, "lockMap_randY", "30", "hunting",
                    "Random walk radius Y")
                _portal = self._get_npc("portal_to_hunt", map_name)
                # AGNOSTIC: only emit a move when the knowledge DB has a REAL
                # portal fact. NO hardcoded coordinate fallback — when absent,
                # the conscious LLM decides the destination (founder 2026-08-25).
                if _portal:
                    _portal_cmd = f"move {_portal['x']} {_portal['y']}"
                    actions.append(HeuristicAction(
                        kind="command", command=_portal_cmd,
                        confidence=0.95, domain="hunting",
                        reason=f"In town {_town_time:.0f}s - return to hunt via portal (knowledge-DB fact)",
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
                # AGNOSTIC: only emit with a REAL portal fact; no hardcoded coord.
                if _portal:
                    actions.append(HeuristicAction(
                        kind="command", command=f"move {_portal['x']} {_portal['y']}",
                        confidence=0.99, domain="emergency",
                        reason="Stuck in town > 300s with 0 kills - force portal to hunting map (knowledge-DB fact)",
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
                # ── AUTO-SELL KNOWN JUNK ITEMS ──
                # Search inventory for known sellable junk items and queue sell commands.
                # Uses _sell_config_once to avoid re-selling the same item within cooldown.
                _inv_items = signals.get("inventory_items", []) or []
                _junk_found = False
                for _item_entry in _inv_items:
                    _item_str = str(_item_entry).lower().strip()
                    for _junk_name, _junk_id in SELLABLE_JUNK.items():
                        if _junk_name in _item_str:
                            if self._sell_config_once(bot_id, _junk_id, cooldown=120.0):
                                actions.append(HeuristicAction(
                                    kind="command", command=f"sell {_junk_id} 0",
                                    confidence=0.85, domain="economy",
                                    reason=f"Sell {_junk_name} (item {_junk_id}) — junk from inventory",
                                ))
                                _junk_found = True
                            break  # Only match one junk name per item entry
                if _junk_found:
                    logger.info(f"[auto_sell] {bot_id}: queued sell commands for junk items in inventory")
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
            # SHIP FLOOD-FIX: WEAPON_BUY re-emits the same shop-routing commands EVERY
            # cycle, flooding the action queue and pinning a bot in a buy-loop at town
            # instead of farming. Rate-limit: run the shop-atom at most once per 15s per
            # bot; on cooldown return an EMPTY assessment (no extra actions) so the state
            # doesn't flood but the caller still gets a valid, actionable result.
            _now_wb = __import__("time").time()
            if _now_wb - self._last_weapon_buy.get(bot_id, 0.0) < 15.0:
                _empty_wb = HeuristicAssessment(
                    horizon=horizon, actions=[], confidence=0.1,
                    actionable=False, top_domain="economy", signals=dict(signals),
                )
                self._last_assessment[bot_id] = _empty_wb
                return _empty_wb
            self._last_weapon_buy[bot_id] = _now_wb
            if _is_hunting:
                # On hunting map - go through a portal to town to buy weapon.
                # AGNOSTIC: resolve the town portal from the knowledge DB; fallback
                # to the combat-safe hunt-map spirit (no hardcoded coord).
                _weap_portal = self._get_npc("portal_to_town", map_name)
                if _weap_portal:
                    actions.append(HeuristicAction(
                        kind="command", command=f"move {_weap_portal['x']} {_weap_portal['y']}",
                        confidence=0.99, domain="economy",
                        reason="Go through portal to town to buy weapon (knowledge-DB fact)",
                    ))
            else:
                # In town - walk to the Weapon Shop (data-driven from the knowledge
                # DB FACT store; seeded baseline + learned per server).
                _ws = self._get_npc("weapon_shop", "prontera")
                _ws_x, _ws_y = (int(_ws.get("x", 0)), int(_ws.get("y", 0))) if _ws else (0, 0)
                if _ws_x and _ws_y:
                    actions.append(HeuristicAction(
                        kind="command", command=f"move {_ws_x} {_ws_y}",
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
                if _ws_x and _ws_y:
                    actions.append(HeuristicAction(
                        kind="command", command=f"move {_ws_x} {_ws_y}",
                        confidence=0.95, domain="economy",
                        reason=f"Walk to Weapon Shop to buy weapon {_weapon}",
                    ))
                    # Menu selection for the weapon shop is handled by the conscious
                    # LLM dialog responder (reads the actual shop menu) — a hardcoded
                    # 'r0' would be a violation for customized shop menus.
                    actions.append(HeuristicAction(
                        kind="command", command=f"talknpc {_ws_x} {_ws_y} c",
                        confidence=0.90, domain="economy",
                        reason="Open Weapon Shop dialog (menu handled by LLM)",
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
                # ── POTION TIER SCALING ──
                _potion_id = self._get_potion_id(base_level)
                _potion_cost = self._get_potion_cost(_potion_id)
                _potion_name = {501: "Red", 502: "Orange", 504: "White"}.get(_potion_id, "Red")
                # ── WEIGHT MANAGEMENT ──
                _weight_cap = NOVICE_WEIGHT_CAPACITY
                _max_buy_weight = self._get_potion_max_buy(_potion_id, zeny, weight, _weight_cap)
                # Also cap by zeny and quantity
                _max_by_zeny = zeny // _potion_cost if _potion_cost > 0 else 0
                _max_buy = min(_max_buy_weight, _max_by_zeny, 30)
                # Stand up first
                actions.append(HeuristicAction(
                    kind="command", command="stand",
                    confidence=0.95, domain="economy",
                    reason="Stand up before walking to Tool Dealer",
                ))
                # Buy potions from Tool Dealer (290, 221 in Prontera)
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
                if _max_buy > 0:
                    actions.append(HeuristicAction(
                        kind="command", command=f"buy {_potion_id} {_max_buy}",
                        confidence=0.90, domain="economy",
                        reason=f"Buy {_max_buy} {_potion_name} Potions (item {_potion_id}, {_potion_cost}z each, "
                               f"level={base_level}, weight={weight:.0%})",
                    ))
                else:
                    # Can't afford any potions or weight full — log it
                    logger.info(f"[economy] {bot_id}: can't buy potions at level {base_level} — "
                               f"zeny={zeny}, cost={_potion_cost}, weight={weight:.0%}, "
                               f"weight_cap={_weight_cap}")
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
            # ── ACADEMY-FIRST DEFER (2026-08-25, founder rule) ──
            # A weapon-less town bot must register at its Academy for the free
            # starter kit BEFORE any field hunt. Skip the farm lockMap/move here
            # so the data-driven academy-door routing (routing domain) is the
            # single authority — otherwise this block emits `move int_land`
            # (a stale adaptive get_best_map result) that races the academy
            # door walk and leaves the bot stuck in town.
            _th_has_weapon = self._has_coldstart_weapon(signals)
            _th_academy_door = self._resolve_academy_door(map_name)
            if not _th_has_weapon and _th_academy_door:
                # NO stand here — emitting `stand` (even once) resets OpenKore's
                # AI and cancels the academy-door walk the routing domain just
                # dispatched (2.27: 'move 125 257' dispatched but never walked,
                # bot sat 4 tiles from the door). The routing domain's move IS
                # the action; this block adds NOTHING so the walk proceeds.
                _th_assessment = HeuristicAssessment(
                    horizon=horizon, actions=actions, confidence=0.99,
                    actionable=len(actions) > 0, top_domain="hunting", signals=dict(signals),
                )
                self._last_assessment[bot_id] = _th_assessment
                return _th_assessment
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
            # Skip during cold start pipeline step 1 (farming prt_fild01 for 50z)
            _th_cs_step = self._cold_start_step.get(_cs_stable_key, 0)
            if not (_th_cs_step == 1 and not _has_weapon and zeny < 50):
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
            # Determine which job change: Novice -> class, or class -> 2-1
            _jc_job = signals.get("job_name", "novice") or "novice"
            _jc_job_lower = _jc_job.lower()
            _jc_base_level = signals.get("base_level", 1) or 1
            _jc_job_level = signals.get("job_level", 1) or 1
            # Find the target class for each type
            _jc_target_class = ""
            _jc_npc_map = "prontera"
            _jc_npc_x = 160
            _jc_npc_y = 191
            _jc_talk_seq: list[str] = []
            _jc_is_2_1 = False
            if _jc_job_lower == "novice":
                # First job change: Novice -> first class
                # Default to Archer (most classes can be reached from Prontera)
                # The user can change this per-bot via job selection config
                _jc_target_class = "archer"
                _jc_npc = JOB_CHANGE_NPCS.get("novice", ("prontera", 160, 191))
                _jc_npc_map, _jc_npc_x, _jc_npc_y = _jc_npc
                _jc_talk_seq = JOB_CHANGE_TALK.get("archer", [
                    "talk continue", "talk resp 1", "talk resp 2", "talk resp 1"
                ])
                logger.info(f"[job_change] {bot_id}: Novice job Lv{_jc_job_level} -> {_jc_target_class}")
            else:
                # 2-1 job change: first class -> second class
                _jc_is_2_1 = True
                _jc_2_1_data = JOB_CHANGE_2_1.get(_jc_job_lower)
                if _jc_2_1_data:
                    _jc_npc_map, _jc_npc_x, _jc_npc_y, _jc_talk_seq = _jc_2_1_data
                else:
                    # Fallback if 2-1 data not found
                    _jc_npc_map, _jc_npc_x, _jc_npc_y = ("prontera", 160, 191)
                    _jc_talk_seq = ["talk continue", "talk resp 1", "talk resp 2", "talk resp 1"]
                _jc_target_class = JOB_2_1_CLASSES.get(_jc_job_lower, _jc_job_lower)
                logger.info(f"[job_change] {bot_id}: {_jc_job_lower} job Lv{_jc_job_level} -> {_jc_target_class} (2-1)")
            actions.append(HeuristicAction(
                kind="command", command="stand",
                confidence=0.95, domain="progression",
                reason="Stand up before walking to job change NPC",
            ))
            if map_name != _jc_npc_map:
                # Not on correct town map — move there first
                actions.append(HeuristicAction(
                    kind="command", command=f"move {_jc_npc_map}",
                    confidence=0.95, domain="progression",
                    reason=f"Move to {_jc_npc_map} for job change to {_jc_target_class}",
                ))
            else:
                # On correct map — walk to NPC and talk
                actions.append(HeuristicAction(
                    kind="command", command=f"move {_jc_npc_x} {_jc_npc_y}",
                    confidence=0.95, domain="progression",
                    reason=f"Walk to job change NPC for {_jc_target_class}",
                ))
                # Use JOB_CHANGE_TALK sequence
                for _step_idx, _step_cmd in enumerate(_jc_talk_seq):
                    _jc_confidence = max(0.70, 0.95 - (_step_idx * 0.03))
                    actions.append(HeuristicAction(
                        kind="command", command=_step_cmd,
                        confidence=_jc_confidence, domain="progression",
                        reason=f"Job change dialog step {_step_idx+1}/{len(_jc_talk_seq)}: {_jc_target_class}",
                    ))
                # After job change, reset mon_control tracking so new class gets fresh settings
                self._last_mon_control_map[bot_id] = ""
                # Clear lockmap cache so new class gets proper maps
                self._last_lockmap[bot_id] = ""
                # Clear cold start step for this bot (force re-evaluation with new class)
                self._cold_start_step[bot_id] = 4  # Mark as complete so we go to HUNT
                # Set post-job-change flag for HUNT state to reset maps on next cycle
                self._post_job_change_reset[bot_id] = True
                logger.info(f"[job_change] {bot_id}: change sequence sent for {_jc_target_class}, resetting map tracking")
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
            # Novice skills: Basic + First Aid (universal)
            _sk_job = signals.get("job_name", "novice") or "novice"
            _sk_job_lower = _sk_job.lower()
            _sk_known_skills = set(skills if isinstance(skills, list) else [])
            _sk_skill_points = signals.get("skill_points", 0) or 0
            _sk_skill_levels = signals.get("skill_levels", {}) or {}
            # Check CLASS_SKILL_TRAINING for the current job
            _sk_training = CLASS_SKILL_TRAINING.get(_sk_job_lower, CLASS_SKILL_TRAINING["novice"])
            _sk_found_skill = False
            for _sk_skill_id, _sk_target_level, _sk_desc in _sk_training:
                if _sk_skill_points <= 0:
                    break  # No more skill points
                # Check if we already have this skill at target level
                _sk_current_level = _sk_skill_levels.get(_sk_skill_id, 0) if isinstance(_sk_skill_levels, dict) else 0
                if _sk_current_level >= _sk_target_level:
                    continue  # Already at target level
                if _sk_skill_id in _sk_known_skills:
                    # Already have skill, level it up
                    _sk_next_level = _sk_current_level + 1
                    if _sk_next_level <= _sk_target_level:
                        actions.append(HeuristicAction(
                            kind="command", command=f"add {_sk_skill_id}",
                            confidence=0.90, domain="progression",
                            reason=f"Level up {_sk_skill_id} ({_sk_desc}) to Lv{_sk_next_level}/{_sk_target_level}",
                        ))
                        _sk_found_skill = True
                        _sk_skill_points -= 1
                else:
                    # Don't have skill yet - learn it
                    actions.append(HeuristicAction(
                        kind="command", command=f"add {_sk_skill_id}",
                        confidence=0.90, domain="progression",
                        reason=f"Learn {_sk_skill_id} ({_sk_desc}) Lv1/{_sk_target_level}",
                    ))
                    _sk_found_skill = True
                    _sk_skill_points -= 1
            if not _sk_found_skill:
                # All skills at target level or no skill points - nothing to do
                pass
            total_confidence = 0.90
            top_domain = "progression"
            assessment = HeuristicAssessment(
                horizon=horizon, actions=actions, confidence=total_confidence,
                actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
            )
            self._last_assessment[bot_id] = assessment
            return assessment

        # ── STATE: PARTY (step 0.5 between portal walk and farming) ──
        if state == "PARTY":
            # ── Map check: party ops only work on same town map ──
            _party_map = str(signals.get("map", "") or "").lower().replace(".gat", "")
            _party_in_town = _party_map in _HUNT_TOWNS
            _bot_profile = bot_id.split(":")[-1].split("/")[-1] if ":" in bot_id else bot_id
            # Dynamic leader detection: first bot alphabetically is leader
            _all_bots = signals.get("all_bots", []) or []
            _sorted_bots = sorted(_all_bots)
            _is_leader = len(_sorted_bots) > 0 and _bot_profile == _sorted_bots[0]
            if _party_in_town:
                # In town — safe to do party operations
                if _is_leader:
                    _now = __import__("time").time()
                    _last_party = self._last_party_attempt.get(bot_id, 0)
                    if _now - _last_party > 30:
                        self._last_party_attempt[bot_id] = _now
                        # OBSERVABILITY ONLY — fleet coordinator is the
                        # party actor (see direct party check above).
                        actions.append(HeuristicAction(
                            kind="log", command="party_create_pending",
                            confidence=0.90, domain="social",
                            reason="Leader - create party with unique name (all in town)",
                            metadata={"party_action": "create_pending", "suffix": f"AI{int(_now_t)}"},
                        ))
                        # Report all known bots that would be invited
                        for _other_bot in _all_bots:
                            if _other_bot != _bot_profile:
                                actions.append(HeuristicAction(
                                    kind="log", command="party_invite_pending",
                                    confidence=0.90, domain="social",
                                    reason=f"Leader - request {_other_bot} to join party (same town map)",
                                    metadata={"party_action": "invite_pending", "member": _other_bot},
                                ))
                        actions.append(HeuristicAction(
                            kind="log", command="party_share_pending",
                            confidence=0.85, domain="social",
                            reason="Share experience in party",
                            metadata={"party_action": "share_pending"},
                        ))
                else:
                    # Joiners: STAY in town — do NOT move to hunting map
                    # Party request only works on same map, so joiners must remain in Prontera
                    actions.append(HeuristicAction(
                        kind="log", command="party_waiting_town",
                        confidence=0.90, domain="social",
                        reason="Joiners - stay in town and wait for party invitation",
                        metadata={"party_action": "waiting"},
                    ))
            else:
                # Not in town — skip party formation (party request requires same map)
                # Bot should return to town first (TOWN_STUCK or TOWN_HUNT will handle this)
                logger.info("[party] bot=%s not in town (map=%s), skipping party formation", bot_id, _party_map)
                actions.append(HeuristicAction(
                    kind="log", command="party_skip_not_town",
                    confidence=0.90, domain="social",
                    reason="Not in town - skip party formation, continue via town/hunt state",
                    metadata={"party_action": "skipped_not_town"},
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
            # ── POST-JOB-CHANGE RESET ──
            # If we just changed job, clear lockmap and mon_control caches
            # so the bot re-evaluates maps and gets fresh mon_control for the new class
            if self._post_job_change_reset.pop(bot_id, False):
                self._last_lockmap[bot_id] = ""
                self._last_mon_control_map[bot_id] = ""
                logger.info(f"[hunt] {bot_id}: post-job-change reset applied (lockmap + mon_control cleared)")

            # ── PER-MAP MON_CONTROL ──
            # Emit mon_control commands when entering a new map
            self._emit_mon_control_for_map(actions, bot_id, map_name, int(signals.get("base_level", 1) or 1))

            # ── COMBAT CONFIG: Set once per value change (dedup via _set_config_once) ──
            _job_name = signals.get("job_name", "novice") or "novice"
            _class_lc = _job_name.lower()
            # ── PER-CLASS CONFIG ──
            if _class_lc == "swordman" or _class_lc == "knight":
                _atk_dist = 5; _atk_max = 20; _tele_min_agg = 8
            elif _class_lc == "thief" or _class_lc == "assassin":
                _atk_dist = 3; _atk_max = 15; _tele_min_agg = 6
            elif _class_lc == "acolyte" or _class_lc == "priest":
                _atk_dist = 7; _atk_max = 25; _tele_min_agg = 4
            elif _class_lc == "archer" or _class_lc == "hunter":
                _atk_dist = 10; _atk_max = 30; _tele_min_agg = 3
            elif _class_lc == "mage" or _class_lc == "wizard":
                _atk_dist = 8; _atk_max = 25; _tele_min_agg = 2
            else:  # novice or unknown
                _atk_dist = 3; _atk_max = 15; _tele_min_agg = 8
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
            _aa_val4 = "2" if base_level < 10 else "3"
            self._set_config_once(actions, bot_id, "attackAuto", _aa_val4, "hunting",
                f"attackAuto={_aa_val4} (level {base_level})")
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
            # Per-class teleport threshold
            self._set_config_once(actions, bot_id, "teleportAuto_minAggressives", str(_tele_min_agg), "hunting",
                f"Per-class teleport at {_tele_min_agg}+ mobs ({_job_name})")
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
            # Per RULE.md: use config audit defaults for teleportAuto_hp=10 and sitAuto=20
            # Don't override — the config audit sets these properly every cycle
            # sitAuto controlled by heuristic config audit (sitAuto_hp_lower=20)
            # aiSidecar_sitAutoHp is NOT set here — bridge uses _sidecar_set flag
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
                _inv_count = signals.get("inventory_items", [])
                if isinstance(_inv_count, list):
                    _has_items = len(_inv_count) > 0
                else:
                    _has_items = int(_inv_count or 0) > 0
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
                # Build available skills dict from known skills and levels
                _skill_dict: dict[str, int] = {}
                if _known_skills and _skill_levels:
                    for i, sk in enumerate(_known_skills[:20]):
                        lv = _skill_levels[i] if i < len(_skill_levels) else 1
                        _skill_dict[str(sk)] = int(lv)
                _best_skill, _best_lv, _best_dps = get_best_skill(
                    int(_attack_power or 25),
                    _skill_dict,
                    int(_current_sp or 10),
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
                                # OBSERVABILITY ONLY — fleet coordinator is
                                # the party actor (see direct party check).
                                actions.append(HeuristicAction(
                                    kind="log", command="party_recreate_pending",
                                    confidence=0.99, domain="social",
                                    reason="Leader - leave old party to re-create",
                                    metadata={"party_action": "recreate_pending"},
                                ))
                                actions.append(HeuristicAction(
                                    kind="log", command="party_create_pending",
                                    confidence=0.95, domain="social",
                                    reason="Leader - create party",
                                    metadata={"party_action": "create_pending", "suffix": f"AI{int(_now)}"},
                                ))
                                for _other_bot in _all_bots:
                                    if _other_bot != _bot_profile:
                                        actions.append(HeuristicAction(
                                            kind="log", command="party_invite_pending",
                                            confidence=0.95, domain="social",
                                            reason=f"Leader - request {_other_bot} to join",
                                            metadata={"party_action": "invite_pending", "member": _other_bot},
                                        ))
                                actions.append(HeuristicAction(
                                    kind="log", command="party_share_pending",
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
                # ── POTION BUY: point the buy NPC at the TOWN with a shop (server-specific, from
                # the DB-backed server_solutions store) so the bot actually buys potions — a buy
                # NPC on a field (no merchant) silently never fires, leaving the bot to run dry
                # and flee. The safe_town is learned/seeded per server (RULE.md: server-specific
                # solutions live in DB/memory, not hardcoded).
                try:
                    # Read the server's learned safe_town from the DB-backed store (singleton
                    # bound to the live DB at create_runtime). Server-specific: a different
                    # server's shop is elsewhere.
                    from ai_sidecar.server_adaptation import get_server_solutions_store
                    _sstore = get_server_solutions_store()
                    _safe_town_buy = ""
                    try:
                        _safe_town_buy = str(_sstore.get("safe_town", "") or "")
                    except Exception:
                        _safe_town_buy = ""
                    if not _safe_town_buy:
                        _safe_town_buy = "prontera"
                    # The buy NPC is the town Tool Dealer (prt_in 126 75 is Prontera's shop area).
                    _buy_npc_coords = {"prontera": "prt_in 126 75", "izlude": "izlude 116 92"}.get(_safe_town_buy, f"{_safe_town_buy} 0 0")
                    actions.append(HeuristicAction(
                        kind="command", command=f"set buyAuto_npc {_buy_npc_coords}",
                        confidence=0.99, domain="economy",
                        reason=f"Point potion buy at the town shop ({_safe_town_buy})",
                    ))
                except Exception:
                    pass
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
            _inv_count = signals.get("inventory_items", [])
            if isinstance(_inv_count, list):
                _has_items = len(_inv_count) > 0
            else:
                _has_items = int(_inv_count or 0) > 0
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
                # No items to sell, but have zeny - buy potions (tiered by level)
                _buy_potion_id = self._get_potion_id(base_level)
                _buy_potion_cost = self._get_potion_cost(_buy_potion_id)
                _buy_potion_name = {501: "Red", 502: "Orange", 504: "White"}.get(_buy_potion_id, "Red")
                _potions_to_buy = min(10, zeny // _buy_potion_cost)
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
                            kind="command", command=f"buy {_buy_potion_id} {_potions_to_buy}",
                            confidence=0.99, domain="economy",
                            reason=f"In town - buy {_potions_to_buy} {_buy_potion_name} Potions (item {_buy_potion_id}, zeny={zeny}, level={base_level})",
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
                        actions.append(HeuristicAction(
                            kind="command", command=f"move {_portal['x']} {_portal['y']}",
                            confidence=0.95, domain="hunting",
                            reason=f"Been in town {_town_time:.0f}s with zeny - return to hunt (knowledge-DB fact)",
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
        # ── def _build_summary(self, assessment: HeuristicAssessment) -> str:
        if not assessment or not assessment.actions:
            return "no heuristic actions"
        parts = [f"{a.domain}:{a.command}" for a in assessment.actions[:5]]
        return f"conf={assessment.confidence:.2f} " + " | ".join(parts)
