from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path
import threading
from ai_sidecar.game_knowledge_db import GameKnowledgeDB

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
    "novice":    [("dex", 20), ("str", 20), ("agi", 15), ("vit", 10)],
    "swordman":  [("str", 40), ("vit", 30), ("dex", 20)],
    "mage":      [("int", 40), ("dex", 30)],
    "archer":    [("dex", 50), ("agi", 30)],
    "acolyte":   [("int", 40), ("dex", 30)],
    "merchant":  [("str", 40), ("vit", 30), ("dex", 20)],
    "thief":     [("dex", 40), ("agi", 30)],
    "taekwon":   [("str", 30), ("agi", 30)],
    "gunslinger":[("dex", 50), ("agi", 30)],
    "ninja":     [("int", 40), ("dex", 30)],
    "soul_linker":[("int", 50), ("dex", 30)],
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
    "novice": [
        (1, 10,  "izlude",       "Poring Island — Porings, Lunatics, Fabres (safe, flat)"),
        (10, 20, "prt_fild08",   "South Prontera — Fabres, Peco Peco, Hornets"),
        (20, 35, "prt_fild04",   "West Prontera — Rockers, Spores, Wolves"),
    ],
    "swordman": [
        (1, 15,  "payon",        "Payon Fields — Peco Peco, Hornets (great job exp)"),
        (15, 30, "pay_fild01",   "Payon Forest — Wolves, Savage, Deniro"),
        (30, 50, "pay_fild04",   "Deep Payon — Ferus, Argiope, Mantis"),
    ],
    "mage": [
        (1, 15,  "gef_fild04",   "Geffen Fields — Rockers, Spores (Fire Bolt one-shots)"),
        (15, 30, "gef_fild06",   "Geffen West — Drainliar, Flora, Vitata"),
        (30, 50, "gef_fild10",   "Geffen Deep — Anacondaq, Stapo, Alligator"),
    ],
    "archer": [
        (1, 15,  "payon",        "Payon Cave 1F — Skeletons, Zombies (easy kite)"),
        (15, 30, "pay_dun00",    "Payon Cave 2F — Munak, Bongun, Ghoul"),
        (30, 50, "pay_dun02",    "Payon Cave 3F — Evil Druid, Wraith, Raydric"),
    ],
    "acolyte": [
        (1, 15,  "izlude",       "Izlude Fields — Willow, Familiar (Heal vs Undead)"),
        (15, 30, "pay_dun00",    "Payon Cave 1F — Undead (Heal one-shots)"),
        (30, 50, "pay_dun02",    "Payon Cave 2F — Munak, Bongun (Turn Undead)"),
    ],
    "merchant": [
        (1, 15,  "morocc",       "Morocco Plains — Peco Peco, Savage (tanky, high HP)"),
        (15, 30, "moc_fild01",   "Morocco Fields — Sandman, Pasana, Deviace"),
        (30, 50, "moc_fild03",   "Morocco Deep — Mimic, Myst, Alarm"),
    ],
    "thief": [
        (1, 15,  "payon",        "Payon Fields — Poison Spores, Wolves (AGI build)"),
        (15, 30, "pay_fild01",   "Payon Forest — Deniro, Savage, Hornet"),
        (30, 50, "pay_fild04",   "Deep Payon — Ferus, Argiope (Double Attack)"),
    ],
    "taekwon": [
        (1, 15,  "izlude",       "Izlude Fields — Muka, Yoyo (kick damage)"),
        (15, 30, "izlude",       "Izlude Deep — Side Winder, Alligator"),
        (30, 50, "ein_fild01",   "Einbroch Fields — Waste Stove, Kukre"),
    ],
    "gunslinger": [
        (1, 15,  "einbroch",     "Einbroch Fields — Poring, Poporing (Single Action)"),
        (15, 30, "ein_fild01",   "Einbroch Outer — Kukre, Waste Stove"),
        (30, 50, "ein_fild03",   "Einbroch Deep — Venatu, Dimik"),
    ],
    "ninja": [
        (1, 15,  "amatsu",       "Amatsu Fields — Muka, Savage (Kunai range)"),
        (15, 30, "ama_fild01",   "Amatsu Forest — Miyabi Doll, Tengu"),
        (30, 50, "ama_dun01",    "Amatsu Cave — Banshee, Bloody Butterfly"),
    ],
    "soul_linker": [
        (1, 15,  "lighthalzen",  "Lighthalzen — Estrun, Luciola Vespa (Soul Strike)"),
        (15, 30, "lhz_fild01",   "Lighthalzen Fields — Novus, Pinguicula"),
        (30, 50, "lhz_dun01",    "Lighthalzen Cave — Dimik, Venatu"),
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
) -> list[tuple[str, int]]:
    """Determine which stats to allocate for a given class.

    Uses adaptive data if available, falls back to hardcoded defaults.
    Returns a list of (stat_name, points_to_add) tuples.
    """
    if stat_points <= 0:
        return []

    # Try adaptive build first
    if adaptive:
        adaptive_build = adaptive.stat_effectiveness.get(job_name, {})
        if adaptive_build:
            # Sort by effectiveness score
            sorted_stats = sorted(adaptive_build.items(), key=lambda x: x[1], reverse=True)
            build = [(stat, 99) for stat, _ in sorted_stats]  # Push to 99
        else:
            build = CLASS_STAT_BUILDS.get(job_name, CLASS_STAT_BUILDS["novice"])
    else:
        build = CLASS_STAT_BUILDS.get(job_name, CLASS_STAT_BUILDS["novice"])

    allocations: list[tuple[str, int]] = []
    remaining = stat_points

    for stat_name, target in build:
        current = current_stats.get(stat_name, 1)
        needed = max(0, target - current)
        if needed > 0:
            add = min(needed, remaining)
            if add > 0:
                allocations.append((stat_name, add))
                remaining -= add
        if remaining <= 0:
            break

    # Dump remaining points into the first priority stat
    if remaining > 0 and build:
        first_stat = build[0][0]
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

    def record_kill(self, map_name: str, exp_gained: float) -> None:
        with self._lock:
            self.map_performance.setdefault(map_name, {"kills": 0, "deaths": 0, "exp": 0, "visits": 0, "last_visit": 0})
            self.map_performance[map_name]["kills"] += 1
            self.map_performance[map_name]["exp"] += exp_gained
            self.map_performance[map_name]["last_visit"] = __import__("time").time()

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
        self._load_towns()
        self._town_entry_time: dict[str, float] = {}
        self._last_hunt_move: dict[str, float] = {}
        self._last_return_to_town: dict[str, float] = {}
        self._last_level: dict[str, int] = {}
        self._last_party_attempt: dict[str, float] = {}
        self._last_party_members: dict[str, list] = {}
        self._last_party_seen: dict[str, float] = {}
        self._all_bots_cache: dict[str, list] = {}

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
        is_town = map_name in ("prontera", "izlude", "morocc", "payon", "geffen",
               "aldebaran", "comodo", "umbala", "niflheim",
               "rachel", "veins", "einbroch", "lighthalzen",
               "juno", "hugel", "yuno", "amatsu", "gonryun",
               "louyang", "ayothaya")
        _prev_state = self._bot_state.get(bot_id, "UNKNOWN")
        _total_kills = signals.get("total_kills", 0) or 0
        _total_zeny = signals.get("zeny", 0) or 0
        # COLD_START: only on VERY FIRST spawn (never after death)
        _cold_fired = self._cold_start_fired.get(bot_id, False)
        if not _cold_fired and _prev_state == "UNKNOWN" and _total_kills == 0 and _total_zeny == 0:
            self._cold_start_fired[bot_id] = True
            return "COLD_START"
        # DEATH: if bot just died and respawned
        # Only trigger DEATH if bot actually died (HP was 0 or very low)
        # Not just because bot has 0 kills after selling starting gear
        if _cold_fired and _prev_state not in ("UNKNOWN", "COLD_START") and hp <= 0:
            return "DEATH"
        zeny = signals.get("zeny", 0) or 0
        # Weight: compute from actual inventory items count in snapshot
        _inv_items = signals.get("inventory_items", []) or []
        # Count non-equipment items (stuff you can sell)
        _sellable = sum(1 for i in _inv_items if isinstance(i, dict) and not i.get("equipped", False))
        weight = _sellable * 0.02  # Each sellable item ~2% weight
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
                _has_weapon = signals.get("attack_power", 0) or 0 > 30
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

        hp = signals.get("hp_ratio", 1.0)
        map_name = signals.get("map", "").lower()
        zeny = signals.get("zeny", 0) or 0
        weight = signals.get("weight_ratio", 0) or 0
        base_level = signals.get("base_level", 1) or 1
        job_level = signals.get("job_level", 1) or 1
        job_name = signals.get("job_name", "novice").lower()
        stat_points = signals.get("stat_points", 0) or 0
        skill_points = signals.get("skill_points", 0) or 0
        in_party = signals.get("in_party", False)
        inventory = signals.get("inventory_items", []) or []
        skills = signals.get("skills", []) or []
        bot_name = signals.get("bot_name", bot_id)
        horizon = signals.get("horizon", "short_term")
        _leader_map = ""
        # _profile_to_char is accessed via self._profile_to_char throughout

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
        _expected_count = len(_all_bots)
        _actual_count = len(_party_members)
        _party_incomplete = _actual_count < _expected_count
        logger.info("[party_check] " + str(bot_id) + " in_party=" + str(_party_in) + " members=" + str(_party_members) + " all_bots=" + str(_all_bots) + " expected=" + str(_expected_count) + " actual=" + str(_actual_count) + " incomplete=" + str(_party_incomplete))

        # Leader: check if party is incomplete
        if _is_leader and _party_incomplete and state != "COLD_START" and state != "DEAD":
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
        if (not _party_in or _joiner_in_wrong_party or _joiner_stuck_in_town) and not _is_leader and state != "COLD_START" and state != "DEAD" and _all_bots:
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

        # ── STATE: COLD_START (fresh spawn - go hunt immediately) ──
        if state == "COLD_START":
            actions.append(HeuristicAction(
                kind="command", command="stand",
                confidence=0.99, domain="emergency",
                reason="Cold start - stand up",
            ))
            actions.append(HeuristicAction(
                kind="command", command="set attackDistance 2",
                confidence=0.99, domain="hunting",
                reason="Cold start - set attack distance",
            ))
            actions.append(HeuristicAction(
                kind="command", command="set attackMaxDistance 7",
                confidence=0.99, domain="hunting",
                reason="Cold start - set chase distance",
            ))
            actions.append(HeuristicAction(
                kind="command", command="set attackAuto 3",
                confidence=0.95, domain="hunting",
                reason="Enable aggressive auto-attack",
            ))
            actions.append(HeuristicAction(
                kind="command", command="set attackAuto_startOnSight 1",
                confidence=0.95, domain="hunting",
                reason="Attack monsters as soon as they appear",
            ))
            actions.append(HeuristicAction(
                kind="command", command="set attackAuto_maxDistance 10",
                confidence=0.99, domain="hunting",
                reason="Cold start - keep attacking even if target moves",
            ))
            actions.append(HeuristicAction(
                kind="command", command="set attackAuto_unstuck 1",
                confidence=0.99, domain="hunting",
                reason="Cold start - don't give up mid-fight",
            ))
            # COLD START: Minimal - keep weapon, go directly to hunt
            # Selling starting gear sells the weapon too (bot deals 0 damage)
            # Economy (sell loot, buy potions, buy weapon) handled by separate states
            _cs_hunt_map = "prt_fild05"
            _cs_portal_coords = "22 203"
            # Class-specific attack distance (critical for Archer with bow)
            _cs_job = signals.get("job_name", "novice") or "novice"
            if _cs_job.lower().startswith("archer") or _cs_job.lower().startswith("hunter"):
                _cs_atk_dist = 7
            elif _cs_job.lower().startswith("mage") or _cs_job.lower().startswith("wizard"):
                _cs_atk_dist = 7
            else:
                _cs_atk_dist = 2
            actions.append(HeuristicAction(
                kind="command", command=f"set attackDistance {_cs_atk_dist}",
                confidence=0.99, domain="hunting",
                reason=f"Cold start - set class attack distance {_cs_atk_dist} for {_cs_job}",
            ))
            # Set route_randomWalk 1 (walk within lockMap bounds) and lockMap_randX/Y 30
            actions.append(HeuristicAction(
                kind="command", command="set route_randomWalk 1",
                confidence=0.99, domain="hunting",
                reason="Cold start - walk within lockMap bounds to find monsters",
            ))
            actions.append(HeuristicAction(
                kind="command", command="set lockMap_randX 30",
                confidence=0.99, domain="hunting",
                reason="Cold start - small random walk radius",
            ))
            actions.append(HeuristicAction(
                kind="command", command="set lockMap_randY 30",
                confidence=0.99, domain="hunting",
                reason="Cold start - small random walk radius",
            ))
            # 1. Set lockMap first
            actions.append(HeuristicAction(
                kind="command", command=f"set lockMap {_cs_hunt_map}",
                confidence=0.99, domain="hunting",
                reason="Cold start - set hunting map lock",
            ))
            # 1b. All bots move to town first (party request requires same map)
            actions.append(HeuristicAction(
                kind="command", command="move prontera",
                confidence=0.99, domain="social",
                reason="Cold start - all bots move to town for party formation",
            ))
            # 1c. Party creation for leader - do this early so others can join
            _cs_bot_profile = bot_id.split(":")[-1].split("/")[-1] if ":" in bot_id else bot_id
            # Dynamic leader detection: first bot alphabetically is leader
            _cs_all_bots = signals.get("all_bots", []) or list(self._bot_roles.keys()) if hasattr(self, '_bot_roles') else []
            _cs_sorted = sorted(_cs_all_bots)
            _cs_is_leader = len(_cs_sorted) > 0 and _cs_bot_profile == _cs_sorted[0]
            if _cs_is_leader:
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
            # 1c. Comprehensive teleport config - DISABLE ALL teleport triggers
            # Only teleport when 8+ mobs aggressive (practically never on low-level maps)
            actions.append(HeuristicAction(
                kind="command", command="set teleportAuto_minAggressives 8",
                confidence=0.99, domain="hunting",
                reason="Only teleport at 8+ mobs",
            ))
            actions.append(HeuristicAction(
                kind="command", command="set teleportAuto_hp 0",
                confidence=0.99, domain="hunting",
                reason="Never teleport due to HP - use sit instead",
            ))
            actions.append(HeuristicAction(
                kind="command", command="set teleportAuto_deadly 0",
                confidence=0.99, domain="hunting",
                reason="Disable deadly teleport - prevents running from non-threats",  
            ))
            # (teleportAuto_idle/search/portal/minWeight set in HUNT state)
            # 1c. Buy arrows for Archer class (can't deal damage without them)
            if _cs_job.lower().startswith("archer") or _cs_job.lower().startswith("hunter"):
                actions.append(HeuristicAction(
                    kind="command", command="buy 1750 200",
                    confidence=0.99, domain="economy",
                    reason="Buy 200 arrows for Archer (need ammo to deal damage)",
                ))
            # 1d. Buy weapon if any zeny available (prevents 0 DMG from sold weapon)
            _cs_zeny = signals.get("zeny", 0) or 0
            if _cs_zeny >= 50:
                _cs_weapon_id = "1701"  # Default: Bow
                if "thief" in _cs_job or "assassin" in _cs_job:
                    _cs_weapon_id = "1301"  # Knife
                elif "sword" in _cs_job or "knight" in _cs_job:
                    _cs_weapon_id = "1201"  # Sword
                elif "mage" in _cs_job or "wizard" in _cs_job:
                    _cs_weapon_id = "1501"  # Rod
                elif "acolyte" in _cs_job or "priest" in _cs_job:
                    _cs_weapon_id = "1501"  # Rod
                actions.append(HeuristicAction(
                    kind="command", command=f"buy {_cs_weapon_id} 1",
                    confidence=0.99, domain="economy",
                    reason=f"Cold start - buy weapon {_cs_weapon_id} for {_cs_job}",
                ))
            # 2. Go directly to hunting map via portal (keep starting weapon!)
            actions.append(HeuristicAction(
                kind="command", command=f"move {_cs_portal_coords}",
                confidence=0.99, domain="emergency",
                reason=f"Cold start - go to {_cs_hunt_map} with starting gear",
            ))
            total_confidence = 0.99
            top_domain = "emergency"
            assessment = HeuristicAssessment(
                horizon=horizon, actions=actions, confidence=total_confidence,
                actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
            )
            self._last_assessment[bot_id] = assessment
            return assessment

        # ── STATE: DEATH (respawned - sell items, buy potions) ──
        if state == "DEATH":
            # Try to sell items (NPC handles empty inventory gracefully)
            # Sell first (talknpc opens NPC dialog, sellAuto handles selling)
            _has_items = (signals.get("inventory_items", 0) or 0) > 0
            _total_kills = signals.get("kills", 0) or 0
            # Sell if has items (starting gear or loot) - need to reduce weight to move
            if _has_items:
                _sell_npc = self._get_npc("sell", map_name)
                if _sell_npc:
                    _sell_cmd = f"talknpc {_sell_npc['x']} {_sell_npc['y']} {' '.join(eval(_sell_npc['steps']))}"
                else:
                    _sell_cmd = "talknpc 147 175 c r1 n"  # fallback
                actions.append(HeuristicAction(
                    kind="command", command=_sell_cmd,
                    confidence=0.99, domain="economy",
                    reason="Death recovery - sell items",
                ))
            # Buy potions on next cycle (after sell dialog completes)
            # Don't generate buy here - let next cycle handle it after sell
            # (sellAuto handles the actual selling after talknpc opens dialog)
            # Return to hunt via portal after 15s in town
            _town_time = __import__("time").time() - self._town_entry_time.get(bot_id, __import__("time").time())
            if _town_time > 15:
                actions.append(HeuristicAction(
                    kind="command", command="set lockMap prt_fild05",
                    confidence=0.95, domain="hunting",
                    reason="Lock to hunting map",
                ))
                actions.append(HeuristicAction(
                    kind="command", command="set lockMap_randX 30",
                    confidence=0.95, domain="hunting",
                    reason="Random walk radius X",
                ))
                actions.append(HeuristicAction(
                    kind="command", command="set lockMap_randY 30",
                    confidence=0.95, domain="hunting",
                    reason="Random walk radius Y",
                ))
                actions.append(HeuristicAction(
                    kind="command", command="set lockMap prt_fild05",
                    confidence=0.95, domain="hunting",
                    reason="Set hunting map lock before returning",
                ))
                _portal = self._get_npc("portal_to_hunt", map_name)
                if _portal:
                    _portal_cmd = f"move {_portal['x']} {_portal['y']}"
                else:
                    _portal_cmd = "move 22 203"  # fallback
                actions.append(HeuristicAction(
                    kind="command", command=_portal_cmd,
                    confidence=0.95, domain="hunting",
                    reason=f"In town {_town_time:.0f}s - return to hunt via portal",
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
            # Cooldown: only sell every 30s to prevent tight loop
            _sell_now = __import__("time").time()
            _last_sell = self._last_sell_time.get(bot_id, 0)
            if _sell_now - _last_sell < 30:
                # Sell on cooldown - skip and let next state handle
                pass
                total_confidence = 0.85
                top_domain = "hunting"
                assessment = HeuristicAssessment(
                    horizon=horizon, actions=actions, confidence=total_confidence,
                    actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
                )
                self._last_assessment[bot_id] = assessment
                return assessment
            self._last_sell_time[bot_id] = _sell_now
            # Stand up first
            actions.append(HeuristicAction(
                kind="command", command="stand",
                confidence=0.95, domain="economy",
                reason="Stand up before walking to Tool Dealer",
            ))
            # Walk to Special Dealer (147, 175) and sell
            actions.append(HeuristicAction(
                kind="command", command="move 147 175",
                confidence=0.95, domain="economy",
                reason=f"Weight {weight:.0f}% - walk to Special Dealer to sell junk",
            ))
            actions.append(HeuristicAction(
                kind="command", command="talknpc 147 175 c r1 n",
                confidence=0.90, domain="economy",
                reason="Open Special Dealer and sell items (atomic dialog)",
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
            # Stand up first
            actions.append(HeuristicAction(
                kind="command", command="stand",
                confidence=0.95, domain="economy",
                reason="Stand up before walking to Tool Dealer",
            ))
            # Buy Red Potions from Tool Dealer (126, 76)
            actions.append(HeuristicAction(
                kind="command", command="move 126 76",
                confidence=0.95, domain="economy",
                reason=f"Zeny {zeny} - walk to Tool Dealer to buy potions",
            ))
            actions.append(HeuristicAction(
                kind="command", command="talknpc 126 76",
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
            stat_builds = {
                "novice": ["dex", "str", "agi", "vit"],
                "archer": ["dex", "agi", "str", "vit"],
                "thief": ["agi", "dex", "str", "vit"],
                "acolyte": ["int", "dex", "vit", "str"],
                "swordman": ["str", "vit", "dex", "agi"],
                "mage": ["int", "dex", "vit", "str"],
            }
            _job_name = signals.get("job_name", "novice") or "novice"
            build = stat_builds.get(_job_name, ["dex", "str", "agi", "vit"])
            _sp = _current_stat_points
            for stat_name in build:
                if _sp > 0:
                    actions.append(HeuristicAction(
                        kind="command", command=f"stat_add {stat_name}",
                        confidence=0.95, domain="progression",
                        reason=f"Allocate 1 {stat_name.upper()} ({job_name} build)",
                    ))
                    _sp -= 1
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
            # ── COMBAT CONFIG: Set every cycle (before any early returns) ──
            _job_name = signals.get("job_name", "novice") or "novice"
            _class_lc = _job_name.lower()
            _atk_dist = 2  # melee default
            _atk_max = 20
            if _class_lc.startswith("archer") or _class_lc.startswith("hunter"):
                _atk_dist = 7
            elif _class_lc.startswith("mage") or _class_lc.startswith("wizard") or _class_lc.startswith("sorcerer"):
                _atk_dist = 7
            elif _class_lc.startswith("thief") or _class_lc.startswith("rogue") or _class_lc.startswith("assassin"):
                _atk_dist = 2
            elif _class_lc.startswith("acolyte") or _class_lc.startswith("priest") or _class_lc.startswith("monk"):
                _atk_dist = 2
            # route_randomWalk 1 (walk within lockMap bounds) + lockMap_randX/Y 30
            actions.append(HeuristicAction(
                kind="command", command="set route_randomWalk 1",
                confidence=0.95, domain="hunting",
                reason="Walk within lockMap bounds to find monsters",
            ))
            actions.append(HeuristicAction(
                kind="command", command="set lockMap_randX 30",
                confidence=0.95, domain="hunting",
                reason="Small random walk radius",
            ))
            actions.append(HeuristicAction(
                kind="command", command="set lockMap_randY 30",
                confidence=0.95, domain="hunting",
                reason="Small random walk radius",
            ))
            actions.append(HeuristicAction(
                kind="command", command=f"set attackDistance {_atk_dist}",
                confidence=0.95, domain="hunting",
                reason=f"Class-appropriate attack distance for {_job_name}",
            ))
            actions.append(HeuristicAction(
                kind="command", command=f"set attackMaxDistance {_atk_max}",
                confidence=0.95, domain="hunting",
                reason="Set max chase distance",
            ))
            actions.append(HeuristicAction(
                kind="command", command="set attackAuto 3",
                confidence=0.95, domain="hunting",
                reason="Enable aggressive auto-attack",
            ))
            actions.append(HeuristicAction(
                kind="command", command="set attackAuto_followTarget 1",
                confidence=0.95, domain="hunting",
                reason="Chase fleeing monsters",
            ))
            actions.append(HeuristicAction(
                kind="command", command="set attackAuto_noMove 0",
                confidence=0.95, domain="hunting",
                reason="Allow movement during combat",
            ))
            actions.append(HeuristicAction(
                kind="command", command="set attackAuto_inLockOnly 1",
                confidence=0.95, domain="hunting",
                reason="Only attack monsters in lockMap area",
            ))
            actions.append(HeuristicAction(
                kind="command", command="set attackAuto_onlyWhenSafe 0",
                confidence=0.95, domain="hunting",
                reason="Attack even if not safe",
            ))
            actions.append(HeuristicAction(
                kind="command", command="set attackAuto_fleeToTarget 0",
                confidence=0.95, domain="hunting",
                reason="Don't flee to target",
            ))
            actions.append(HeuristicAction(
                kind="command", command="set attackAuto_startDistance 1",
                confidence=0.95, domain="hunting",
                reason="Start attacking from 1 cell away (immediate)",
            ))
            actions.append(HeuristicAction(
                kind="command", command="set attackAuto_keepDistance 1",
                confidence=0.95, domain="hunting",
                reason="Keep distance while attacking",
            ))
            actions.append(HeuristicAction(
                kind="command", command="set attackAuto_maxDistance 20",
                confidence=0.95, domain="hunting",
                reason="Keep attacking even if target moves",
            ))
            actions.append(HeuristicAction(
                kind="command", command="set attackAuto_unstuck 1",
                confidence=0.95, domain="hunting",
                reason="Don't give up mid-fight",
            ))
            # Teleport config: disable all teleport triggers
            actions.append(HeuristicAction(
                kind="command", command="set teleportAuto 0",
                confidence=0.99, domain="hunting",
                reason="Disable teleporting to prevent town loop",
            ))
            actions.append(HeuristicAction(
                kind="command", command="set teleportAuto_minAggressives 8",
                confidence=0.95, domain="survival",
                reason="Only teleport when 8+ mobs",
            ))
            actions.append(HeuristicAction(
                kind="command", command="set teleportAuto_hp 0",
                confidence=0.95, domain="survival",
                reason="Never teleport due to HP",
            ))
            actions.append(HeuristicAction(
                kind="command", command="set teleportAuto_deadly 0",
                confidence=0.95, domain="survival",
                reason="Disable deadly teleport",
            ))
            actions.append(HeuristicAction(
                kind="command", command="set teleportAuto_search 0",
                confidence=0.95, domain="survival",
                reason="Disable search teleport",
            ))
            actions.append(HeuristicAction(
                kind="command", command="set teleportAuto_portal 0",
                confidence=0.95, domain="survival",
                reason="Disable portal teleport",
            ))
            if map_name not in _HUNT_TOWNS:
                # On hunting map: check if we should return to town
                _hunt_duration = __import__("time").time() - self._state_since.get(bot_id, __import__("time").time())
                _hp_ratio = signals.get("hp_ratio", 1.0) or 1.0
                _has_items = (signals.get("inventory_items", 0) or 0) > 0
                _total_kills = signals.get("kills", 0) or 0
                # AT PORTAL EXIT: if bot is at (367, 205) on prt_fild05, move to center
                _x = signals.get("x", 0) or 0
                _y = signals.get("y", 0) or 0
                if abs(_x - 367) < 10 and abs(_y - 205) < 10 and map_name == "prt_fild05":
                    actions.append(HeuristicAction(
                        kind="command", command="move 200 200",
                        confidence=0.99, domain="hunting",
                        reason="At portal exit - move to center of hunting map",
                    ))
                    total_confidence = 0.99
                    top_domain = "hunting"
                    assessment = HeuristicAssessment(
                        horizon=horizon, actions=actions, confidence=total_confidence,
                        actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
                    )
                    self._last_assessment[bot_id] = assessment
                    return assessment
                # JUST WARPED: if just arrived, sit to regen first
                if _hunt_duration < 15:
                    if _hp_ratio < 0.5:
                        actions.append(HeuristicAction(
                            kind="command", command="sit",
                            confidence=0.99, domain="survival",
                            reason=f"HP={_hp_ratio:.0%} just warped - sit to regen before hunting",
                        ))
                        total_confidence = 0.99
                        top_domain = "survival"
                        assessment = HeuristicAssessment(
                            horizon=horizon, actions=actions, confidence=total_confidence,
                            actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
                        )
                        self._last_assessment[bot_id] = assessment
                        return assessment
                    # Don't return to town within first 30s - starting gear triggers weight check
                    if _hunt_duration < 30:
                        actions.append(HeuristicAction(
                            kind="command", command="ai auto",
                            confidence=0.95, domain="hunting",
                            reason=f"Just warped {_hunt_duration:.0f}s ago - hunt first, sell later",
                        ))
                        total_confidence = 0.95
                        top_domain = "hunting"
                        assessment = HeuristicAssessment(
                            horizon=horizon, actions=actions, confidence=total_confidence,
                            actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
                        )
                        self._last_assessment[bot_id] = assessment
                        return assessment
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
                # MAP PROGRESSION: Check if bot should move to a better hunting map
                # Uses class-appropriate progression paths (1-99)
                # Different classes favor different maps based on mob elements and layout
                _base_level = signals.get("level", 1) or 1
                _progression_maps = [
                    (99, "gefen_fild01"),
                    (85, "gef_fild02"),
                    (70, "mjolnir_04"),
                    (60, "pay_fild01"),
                    (50, "gef_fild01"),
                    (40, "prt_fild08"),
                    (30, "pay_fild03"),
                    (20, "pay_fild01"),       # 20+: Porings/Poporings/Lunatics — better melee density
                    (10, "prt_fild05"),       # 10-20: Porings, Lunatics, Fabres
                    (1, "prt_fild04"),        # 1-10: Starter field (dense spawns for novices)
                ]
                _next_map = "prt_fild05"
                for _lvl, _map in _progression_maps:
                    if _base_level >= _lvl:
                        _next_map = _map
                        break
                # If current map is not the correct one for level, move
                if map_name != _next_map and _hunt_duration > 30:
                    actions.append(HeuristicAction(
                        kind="command", command=f"set lockMap {_next_map}",
                        confidence=0.90, domain="hunting",
                        reason=f"Level {_base_level} - moving to {_next_map}",
                    ))
                    actions.append(HeuristicAction(
                        kind="command", command=f"move {_next_map}",
                        confidence=0.90, domain="hunting",
                        reason=f"Level {_base_level} - progressing to {_next_map}",
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
