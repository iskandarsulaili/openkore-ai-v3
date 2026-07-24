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
        self._last_hunt_move: dict[str, float] = {}

    def _get_state(self, signals: dict) -> str:
        """Determine bot state from signals."""
        hp = signals.get("hp_ratio", 1.0)
        map_name = signals.get("map", "").lower()
        map_name = map_name.replace(".gat", "")
        map_name = map_name.replace(".gat", "")
        zeny = signals.get("zeny", 0) or 0
        weight = signals.get("weight_ratio", 0) or 0
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
        is_town = map_name in ("prontera", "izlude", "morocc", "payon", "geffen",
                               "aldebaran", "comodo", "umbala", "niflheim",
                               "rachel", "veins", "einbroch", "lighthalzen",
                               "juno", "hugel", "yuno", "amatsu", "gonryun",
                               "louyang", "ayothaya")

        if is_town:
            # Priority: SELL > WEAPON_BUY > BUY > JOB_CHANGE > STATS > SKILLS > PARTY > HUNT
            if weight > 0.05:  # 5% weight threshold
                return "SELL"
            # Check if bot has a weapon by checking attack power > 30 (bare hands = 19)
            _has_weapon = signals.get("attack_power", 0) or 0 > 30
            if zeny >= 500 and not _has_weapon:
                return "WEAPON_BUY"
            if zeny > 0:
                return "BUY"
            if base_level >= 10 and job_level >= 10 and job_name == "novice":
                return "JOB_CHANGE"
            if stat_points > 0:
                return "STATS"
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
        actions: list[HeuristicAction] = []
        bot_id = bot_id_override or signals.get("bot_id", "default")
        state = self._get_state(signals)
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

        # ── STATE: SELL ──
        if state == "SELL":
            # Cooldown: only sell every 30s to prevent tight loop
            _sell_now = __import__("time").time()
            _last_sell = self._last_sell_time.get(bot_id, 0)
            if _sell_now - _last_sell < 30:
                # Skip sell, go straight to hunting
                actions.append(HeuristicAction(
                    kind="command", command="move prt_fild05",
                    confidence=0.85, domain="hunting",
                    reason="Sell on cooldown - go hunting instead",
                ))
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
                kind="command", command="talknpc 147 175 c r0 n",
                confidence=0.90, domain="economy",
                reason="Open Special Dealer and sell items (atomic dialog)",
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
            actions.append(HeuristicAction(
                kind="command", command="move 160 133",
                confidence=0.95, domain="economy",
                reason=f"Zeny {zeny} - walk to Weapon Shop to buy weapon",
            ))
            actions.append(HeuristicAction(
                kind="command", command="talknpc 160 133 c r0 n",
                confidence=0.90, domain="economy",
                reason="Open Weapon Shop and open buy menu (atomic dialog)",
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
            build = stat_builds.get(job_name, ["dex", "str", "agi", "vit"])
            for stat_name in build:
                if stat_points > 0:
                    actions.append(HeuristicAction(
                        kind="command", command=f"stat_add {stat_name}",
                        confidence=0.95, domain="progression",
                        reason=f"Allocate 1 {stat_name.upper()} ({job_name} build)",
                    ))
                    stat_points -= 1
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
            # Use bot_name from signals - no hardcoded names
            # The bridge rewrite handles party join syntax
            actions.append(HeuristicAction(
                kind="command", command="party create AI Team",
                confidence=0.90, domain="social",
                reason="Create party for team play",
            ))
            actions.append(HeuristicAction(
                kind="command", command="party share exp",
                confidence=0.85, domain="social",
                reason="Share experience in party",
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
            # Check if it's time to return to town (every 10 minutes)
            _hunt_start = self._state_since.get(bot_id, 0)
            _hunt_duration = __import__("time").time() - _hunt_start
            # Check if weight is high or any items to sell - return to town sooner
            _to_sell = weight > 0.3 or zeny == 0  # Sell if >30% weight or no zeny
            if _to_sell and _hunt_duration > 120:  # At least 2 min hunt
                actions.append(HeuristicAction(
                    kind="command", command="move prontera",
                    confidence=0.95, domain="exploration",
                    reason=f"Items to sell or need zeny - return to town (hunted {_hunt_duration:.0f}s)",
                ))
                self._state_since[bot_id] = __import__("time").time()  # Reset timer
            elif _hunt_duration > 1800:  # 30 minutes max
                actions.append(HeuristicAction(
                    kind="command", command="move prontera",
                    confidence=0.95, domain="exploration",
                    reason=f"Hunted for {_hunt_duration:.0f}s - return to town to sell/buy",
                ))
                self._state_since[bot_id] = __import__("time").time()  # Reset timer
                total_confidence = 0.95
                top_domain = "exploration"
                assessment = HeuristicAssessment(
                    horizon=horizon, actions=actions, confidence=total_confidence,
                    actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
                )
                self._last_assessment[bot_id] = assessment
                return assessment
            
            # Ensure AI is in auto mode and standing
            actions.append(HeuristicAction(
                kind="command", command="ai auto",
                confidence=0.95, domain="combat",
                reason="Ensure AI is in auto mode for hunting",
            ))
            # If hasn't moved in 30s, force move to find monsters (faster exploration)
            _last_hunt_move = self._last_hunt_move.get(bot_id, 0)
            _hunt_now = __import__("time").time()
            _hunt_towns = ("prontera", "izlude", "morocc", "payon", "geffen",
                          "aldebaran", "comodo", "umbala", "niflheim")
            if _hunt_now - _last_hunt_move > 30 and map_name not in _hunt_towns:
                self._last_hunt_move[bot_id] = _hunt_now
                # Smarter exploration: move toward center of map first, then spiral
                _map_size = 300  # Typical hunting map size
                # Use safer coordinates - center of map where monsters spawn
                _move_x = 150 + int(__import__("random").random() * 100)
                _move_y = 150 + int(__import__("random").random() * 100)
                actions.append(HeuristicAction(
                    kind="command", command=f"move {_move_x} {_move_y}",
                    confidence=0.70, domain="exploration",
                    reason=f"Explore to ({_move_x},{_move_y}) - find monsters",
                ))
            actions.append(HeuristicAction(
                kind="command", command="ai auto",
                confidence=0.95, domain="combat",
                reason="Ensure AI is in auto mode for hunting",
            ))
            # If stuck, suggest moving to a different map
            if is_stuck:
                target_map = self._adaptive.get_best_map(bot_id, base_level)
                if not target_map:
                    target_map = "prt_fild08" if base_level < 20 else "pay_fild01"
                actions.append(HeuristicAction(
                    kind="command", command=f"move {target_map}",
                    confidence=0.50, domain="exploration",
                    reason=f"Stuck on {map_name} for {state_duration:.0f}s - try {target_map}",
                ))
            total_confidence = 0.90
            top_domain = "combat"
            assessment = HeuristicAssessment(
                horizon=horizon, actions=actions, confidence=total_confidence,
                actionable=len(actions) > 0, top_domain=top_domain, signals=dict(signals),
            )
            self._last_assessment[bot_id] = assessment
            return assessment

        # ── FALLBACK ──
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
