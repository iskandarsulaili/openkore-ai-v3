from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path
import threading

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


# ── Class-aware stat builds ──
# Each entry: (stat_priority_list, description)
CLASS_STAT_BUILDS: dict[str, list[tuple[str, int]]] = {
    "novice":    [("dex", 20), ("str", 20), ("agi", 15), ("vit", 10)],
    "swordman":  [("str", 40), ("vit", 30), ("dex", 20)],
    "mage":      [("int", 40), ("dex", 30)],
    "archer":    [("dex", 50), ("agi", 30)],
    "acolyte":   [("int", 40), ("dex", 30)],
    "merchant":  [("str", 40), ("vit", 30), ("dex", 20)],
    "thief":     [("agi", 40), ("dex", 30)],
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
    "kicapmasin": "leader",
    "kicapmasin2": "dps",
    "kicapmasin3": "support",
}

BOT_JOBS: dict[str, str] = {
    "kicapmasin": "archer",
    "kicapmasin2": "thief",
    "kicapmasin3": "acolyte",
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

    def get_best_map(self, candidates: list[str]) -> str | None:
        with self._lock:
            if not candidates:
                return None
            scored = [(self.get_map_score(m), m) for m in candidates]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]

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
    """State-machine-based heuristic service.

    Bot states:
      TOWN: In town → sell junk, buy potions, buy weapon, job change
      HUNTING: On hunting map → kill monsters, loot, level up
      DEAD: Just respawned → sell junk, buy supplies, go back
      SHOPPING: In town with supplies → buy potions, arrows, weapon
      JOB_CHANGE: Ready to job change → walk to NPC, complete dialog
      STUCK: No progress in 5min → lower confidence, let LLM handle

    Each state generates specific actions. Confidence drops when stuck.
    """

    def __init__(self):
        self._adaptive = AdaptiveDataStore()
        self._last_assessment: dict[str, HeuristicAssessment] = {}
        self._bot_state: dict[str, str] = {}  # bot_id -> state
        self._state_since: dict[str, float] = {}  # bot_id -> timestamp
        self._last_progress: dict[str, dict] = {}  # bot_id -> {exp, zeny, level, kills}

    def _get_state(self, signals: dict) -> str:
        """Determine bot state from signals."""
        bot_id = signals.get("bot_id", "default")
        hp = signals.get("hp_ratio", 1.0)
        sp = signals.get("sp_ratio", 1.0)
        map_name = signals.get("map", "").lower()
        zeny = signals.get("zeny", 0)
        weight = signals.get("weight_ratio", 0)
        base_level = signals.get("base_level", 1)
        job_level = signals.get("job_level", 1)
        job_name = signals.get("job_name", "novice").lower()
        stat_points = signals.get("stat_points", 0)
        skill_points = signals.get("skill_points", 0)
        in_party = signals.get("in_party", False)
        inventory = signals.get("inventory_items", [])
        skills = signals.get("skills", [])

        # DEAD: HP == 0
        if hp <= 0:
            return "DEAD"

        # TOWN: In a town map
        is_town = map_name in ("prontera", "izlude", "morocc", "payon", "geffen",
                               "aldebaran", "comodo", "umbala", "niflheim",
                               "rachel", "veins", "einbroch", "lighthalzen",
                               "juno", "hugel", "yuno", "amatsu", "gonryun",
                               "louyang", "ayothaya")

        if is_town:
            # JOB_CHANGE: Ready to change job
            if base_level >= 10 and job_level >= 10 and job_name == "novice":
                return "JOB_CHANGE"
            # SHOPPING: Need to buy supplies
            if zeny > 0 and weight < 50:
                return "SHOPPING"
            # TOWN: Default town state
            return "TOWN"

        # HUNTING: On a hunting map
        is_hunting = "_fild" in map_name or "_field" in map_name or "_01" in map_name or "_02" in map_name
        if is_hunting or not is_town:
            return "HUNTING"

        return "TOWN"

    def _check_progress(self, signals: dict) -> bool:
        """Check if bot made progress since last cycle."""
        bot_id = signals.get("bot_id", "default")
        last = self._last_progress.get(bot_id, {})
        now = {
            "exp": signals.get("exp", 0) or signals.get("base_exp", 0) or 0,
            "zeny": signals.get("zeny", 0) or 0,
            "level": signals.get("base_level", 1) or 1,
            "kills": signals.get("kills", 0) or 0,
        }
        self._last_progress[bot_id] = now

        if not last:
            return True  # First cycle, assume progress

        # Check if any metric improved
        for key in ("exp", "zeny", "level", "kills"):
            if now.get(key, 0) > last.get(key, 0):
                return True
        return False

    def set_domain_weights(self, weights: dict) -> None:
        """Set domain weights (kept for compatibility)."""
        pass

    def assess(self, signals: dict[str, Any]) -> HeuristicAssessment:
        """Produce heuristic actions based on bot state."""
        actions: list[HeuristicAction] = []
        bot_id = signals.get("bot_id", "default")
        state = self._get_state(signals)
        prev_state = self._bot_state.get(bot_id, "UNKNOWN")

        # Track state transitions
        if state != prev_state:
            self._bot_state[bot_id] = state
            self._state_since[bot_id] = __import__("time").time()
            logger.info(f"[heuristic] {bot_id} state: {prev_state} -> {state}")

        # Check for stuck condition
        made_progress = self._check_progress(signals)
        state_duration = __import__("time").time() - self._state_since.get(bot_id, 0)
        is_stuck = not made_progress and state_duration > 120  # 2min without progress

        # Extract signals
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
        inventory = signals.get("inventory_items", [])
        skills = signals.get("skills", [])
        bot_name = signals.get("bot_name", bot_id)
        horizon = signals.get("horizon", "short_term")

        # ── STATE: DEAD ──
        if state == "DEAD":
            # Just respawn - let the bridge's death handler handle it
            # The bridge will set respawn_ms and allow 15s economy window
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

        # ── STATE: JOB_CHANGE ──
        if state == "JOB_CHANGE":
            # Walk to job change NPC and complete dialog
            # Archer Guild: (160, 191) in Prontera
            # Thief Guild: (270, 220) in Prontera
            # Acolyte Guild: (270, 220) in Prontera
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
                reason="Select first job option (Archer)",
            ))
            total_confidence = 0.90
            top_domain = "progression"

            assessment = HeuristicAssessment(
                horizon=horizon, actions=actions, confidence=total_confidence,
                actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
            )
            self._last_assessment[bot_id] = assessment
            return assessment

        # ── STATE: TOWN ──
        if state == "TOWN":
            # Sell junk items
            if weight > 50:
                actions.append(HeuristicAction(
                    kind="command", command="ai auto",
                    confidence=0.90, domain="economy",
                    reason=f"Weight {weight:.0f}% - in Prontera, LLM should plan sell route",
                ))
            # Allocate stat points
            if stat_points > 0:
                # Determine which stat to raise based on class
                stat_builds = {
                    "novice": ["dex", "str", "agi", "vit"],
                    "archer": ["dex", "agi", "str", "vit"],
                    "thief": ["agi", "dex", "str", "vit"],
                    "acolyte": ["int", "dex", "vit", "str"],
                    "swordman": ["str", "vit", "dex", "agi"],
                    "mage": ["int", "dex", "vit", "str"],
                }
                build = stat_builds.get(job_name, ["dex", "str", "agi", "vit"])
                for stat_name in build:
                    if stat_points > 0:
                        actions.append(HeuristicAction(
                            kind="command", command=f"stat_add {stat_name}",
                            confidence=0.95, domain="progression",
                            reason=f"Allocate 1 {stat_name.upper()} ({job_name} build)",
                        ))
                        stat_points -= 1
            # Learn skills
            if skill_points > 0:
                if "NV_BASIC" not in skills:
                    actions.append(HeuristicAction(
                        kind="command", command="skills add 1",
                        confidence=0.90, domain="progression",
                        reason="Learn Basic Skill to sit and regen",
                    ))
                elif "NV_FIRSTAID" not in skills:
                    actions.append(HeuristicAction(
                        kind="command", command="skills add 2",
                        confidence=0.85, domain="progression",
                        reason="Learn First Aid for emergency healing",
                    ))
            # Party
            if not in_party and bot_name == "kicapmasin":
                actions.append(HeuristicAction(
                    kind="command", command="party create",
                    confidence=0.90, domain="social",
                    reason="Leader - create party for team play",
                ))
                actions.append(HeuristicAction(
                    kind="command", command="party share exp",
                    confidence=0.85, domain="social",
                    reason="Share experience in party",
                ))
            # Move to hunting map
            actions.append(HeuristicAction(
                kind="command", command="move prt_fild05",
                confidence=0.80, domain="exploration",
                reason=f"Level {base_level} - move to prt_fild05 for grinding",
            ))
            total_confidence = 0.90
            top_domain = "progression"

            assessment = HeuristicAssessment(
                horizon=horizon, actions=actions, confidence=total_confidence,
                actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
            )
            self._last_assessment[bot_id] = assessment
            return assessment

        # ── STATE: HUNTING ──
        if state == "HUNTING":
            # Ensure AI is in auto mode
            actions.append(HeuristicAction(
                kind="command", command="ai auto",
                confidence=0.95, domain="combat",
                reason="Ensure AI is in auto mode for hunting",
            ))
            # Allocate stat points if available
            if stat_points > 0:
                stat_builds = {
                    "novice": ["dex", "str", "agi", "vit"],
                    "archer": ["dex", "agi", "str", "vit"],
                    "thief": ["agi", "dex", "str", "vit"],
                    "acolyte": ["int", "dex", "vit", "str"],
                }
                build = stat_builds.get(job_name, ["dex", "str", "agi", "vit"])
                for stat_name in build:
                    if stat_points > 0:
                        actions.append(HeuristicAction(
                            kind="command", command=f"stat_add {stat_name}",
                            confidence=0.95, domain="progression",
                            reason=f"Allocate 1 {stat_name.upper()} ({job_name} build)",
                        ))
                        stat_points -= 1
            # Learn skills if available
            if skill_points > 0:
                if "NV_BASIC" not in skills:
                    actions.append(HeuristicAction(
                        kind="command", command="skills add 1",
                        confidence=0.90, domain="progression",
                        reason="Learn Basic Skill to sit and regen",
                    ))
            # If stuck, suggest moving to a different map
            if is_stuck:
                actions.append(HeuristicAction(
                    kind="command", command="move prt_fild08",
                    confidence=0.50, domain="exploration",
                    reason=f"Stuck on {map_name} for {state_duration:.0f}s - try different map",
                ))
            total_confidence = 0.90
            top_domain = "combat"

            assessment = HeuristicAssessment(
                horizon=horizon, actions=actions, confidence=total_confidence,
                actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
            )
            self._last_assessment[bot_id] = assessment
            return assessment

        # ── FALLBACK: Unknown state ──
        actions.append(HeuristicAction(
            kind="command", command="ai auto",
            confidence=0.50, domain="survival",
            reason=f"Unknown state '{state}' - fallback to auto",
        ))
        total_confidence = 0.50
        top_domain = "survival"

        assessment = HeuristicAssessment(
            horizon=horizon, actions=actions, confidence=total_confidence,
            actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
        )
        self._last_assessment[bot_id] = assessment
        return assessment

    def confidence_for(self, horizon: str, signals: dict = None, bot_id: str = None) -> float:
        """Return confidence for a given horizon."""
        if signals:
            state = self._get_state(signals)
            # Lower confidence for stuck states (let LLM handle)
            if state == "JOB_CHANGE":
                return 0.90  # High confidence - job change is mechanical
            if state == "DEAD":
                return 0.50  # Low confidence - let LLM handle respawn strategy
            if state == "TOWN":
                return 0.85  # Medium-high - town actions are routine
            if state == "HUNTING":
                return 0.85  # Medium-high - hunting is routine
            return 0.70
        return 0.70

    def _build_summary(self, assessment: HeuristicAssessment) -> str:
        """Build a summary string from an assessment."""
        if not assessment or not assessment.actions:
            return "no heuristic actions"
        parts = [f"{a.domain}:{a.command}" for a in assessment.actions[:5]]
        return f"conf={assessment.confidence:.2f} " + " | ".join(parts)
