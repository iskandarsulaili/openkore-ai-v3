from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path

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
    """Learns from outcomes and replaces hardcoded constants with adaptive data.

    Tracks: map performance, stat build effectiveness, skill priority,
    NPC locations, economy patterns. Improves over time based on actual results.
    """

    def __init__(self):
        # Map performance: {map_name: {kills, deaths, exp_gained, visits, last_visit}}
        self.map_performance: dict[str, dict[str, float]] = {}
        # Stat build effectiveness: {job_name: {stat: avg_level_at_success}}
        self.stat_effectiveness: dict[str, dict[str, float]] = {}
        # Skill priority: {job_name: {skill_id: times_used, avg_damage}}
        self.skill_priority: dict[str, dict[str, float]] = {}
        # NPC discovery: {service_type: {map: [(x, y, name)]}}
        self.npc_locations: dict[str, dict[str, list[tuple[int, int, str]]]] = {}
        # Economy: {item_name: {avg_sell_price, times_sold, last_sold}}
        self.economy_data: dict[str, dict[str, float]] = {}
        # Death analysis: {map_name: {deaths, causes, avg_hp_at_death}}
        self.death_analysis: dict[str, dict[str, Any]] = {}

    def record_kill(self, map_name: str, exp_gained: float) -> None:
        self.map_performance.setdefault(map_name, {"kills": 0, "deaths": 0, "exp": 0, "visits": 0, "last_visit": 0})
        self.map_performance[map_name]["kills"] += 1
        self.map_performance[map_name]["exp"] += exp_gained
        self.map_performance[map_name]["last_visit"] = __import__('time').time()

    def record_death(self, map_name: str, hp_at_death: float = 0) -> None:
        self.map_performance.setdefault(map_name, {"kills": 0, "deaths": 0, "exp": 0, "visits": 0, "last_visit": 0})
        self.map_performance[map_name]["deaths"] += 1
        self.death_analysis.setdefault(map_name, {"deaths": 0, "causes": {}, "avg_hp": 0})
        self.death_analysis[map_name]["deaths"] += 1
        old_avg = self.death_analysis[map_name]["avg_hp"]
        old_count = self.death_analysis[map_name]["deaths"] - 1
        self.death_analysis[map_name]["avg_hp"] = (old_avg * old_count + hp_at_death) / self.death_analysis[map_name]["deaths"]

    def record_visit(self, map_name: str) -> None:
        self.map_performance.setdefault(map_name, {"kills": 0, "deaths": 0, "exp": 0, "visits": 0, "last_visit": 0})
        self.map_performance[map_name]["visits"] += 1
        self.map_performance[map_name]["last_visit"] = __import__('time').time()

    def get_map_score(self, map_name: str) -> float:
        """Score a map based on actual performance. Higher = better."""
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
        # Score: exp per visit, penalized by deaths
        score = exp_per_visit * 0.01
        if death_rate > 0:
            score *= max(0.1, 1.0 - death_rate * 2)
        if kill_rate > 0:
            score *= min(2.0, 1.0 + kill_rate * 0.5)
        return score

    def get_best_map(self, candidates: list[str]) -> str | None:
        """Return the best map from candidates based on learned performance."""
        if not candidates:
            return None
        scored = [(self.get_map_score(m), m) for m in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def record_npc(self, service: str, map_name: str, x: int, y: int, name: str = "") -> None:
        self.npc_locations.setdefault(service, {})
        self.npc_locations[service].setdefault(map_name, [])
        # Avoid duplicates
        for existing in self.npc_locations[service][map_name]:
            if existing[0] == x and existing[1] == y:
                return
        self.npc_locations[service][map_name].append((x, y, name))

    def get_npc(self, service: str, map_name: str) -> tuple[int, int, str] | None:
        """Get NPC coordinates for a service on a map."""
        npcs = self.npc_locations.get(service, {}).get(map_name, [])
        if npcs:
            return npcs[0]
        return None

    def record_sale(self, item_name: str, price: float) -> None:
        self.economy_data.setdefault(item_name, {"avg_price": 0, "count": 0, "last_price": 0})
        entry = self.economy_data[item_name]
        old_avg = entry["avg_price"]
        old_count = entry["count"]
        entry["avg_price"] = (old_avg * old_count + price) / (old_count + 1)
        entry["count"] += 1
        entry["last_price"] = price


class HeuristicService:
    """Produces heuristic actions from game state signals without calling LLM.

    Uses AdaptiveDataStore to learn from outcomes and improve over time.
    The confidence score determines whether the PDCA loop skips the LLM entirely.
    Low confidence (< 0.7) means the LLM should make the decision instead.
    """

    def __init__(self):
        self._last_assessment: dict[str, HeuristicAssessment] = {}
        self._adaptive = AdaptiveDataStore()
        self._domain_weights: dict[str, float] = {
            "recovery": 0.15,
            "grind": 0.30,
            "economy": 0.25,
            "quest": 0.10,
            "exploration": 0.20,
        }
        # Track signals from previous cycle for feedback
        self._prev_signals: dict[str, dict[str, Any]] = {}

    def set_domain_weights(self, weights: dict[str, float]) -> None:
        self._domain_weights.update(weights)

    def assess(self, signals: dict[str, Any]) -> HeuristicAssessment:
        """Produce heuristic actions from game state signals.

        Only emits commands that are valid OpenKore commands AND pass the bridge
        policy allowlist (ai, move, macro, eventMacro, talknpc, take). For
        complex scenarios (vendor routing, respawn logic, etc.), emits metadata
        signals so the LLM conscious layer can generate proper action plans.
        """
        actions: list[HeuristicAction] = []
        total_confidence = 0.0
        weighted_domains: dict[str, float] = {}

        # Enriched state signals (emergent discovery)
        _enriched = signals.get("_enriched", None)

        # ── Survival: Stay in town if critically low HP ──
        hp_ratio = signals.get("hp_ratio", 1.0)
        map_name = signals.get("map", "")
        if hp_ratio < 0.3 and "prontera" not in map_name:
            actions.append(HeuristicAction(
                kind="command", command="move prontera",
                confidence=0.95, domain="survival",
                reason=f"Critically low HP ({hp_ratio:.0%}) — retreat to Prontera to regen safely",
            ))
            weighted_domains["survival"] = 0.95
            total_confidence = max(total_confidence, 0.95)

        # ── Progression: Learn skills ──
        skills = signals.get("skills", [])
        base_level = signals.get("base_level", 1)
        job_name = signals.get("job_name", "novice").lower()
        zeny = signals.get("zeny", 0)
        inventory = signals.get("inventory_items", {})
        stat_points = signals.get("stat_points", 0)
        skill_points = signals.get("skill_points", 0)

        # ── Class-aware cold start: Stat allocation ──
        if stat_points > 0:
            current_stats = {
                "str": signals.get("str", 1),
                "agi": signals.get("agi", 1),
                "vit": signals.get("vit", 1),
                "int": signals.get("int", 1),
                "dex": signals.get("dex", 1),
                "luk": signals.get("luk", 1),
            }
            allocations = _class_stat_allocation(job_name, current_stats, stat_points, self._adaptive)
            for stat_name, points in allocations:
                # OpenKore expects lowercase stat names (str, agi, dex, etc.)
                actions.append(HeuristicAction(
                    kind="command", command=f"stat_add {stat_name} {points}",
                    confidence=0.95, domain="progression",
                    reason=f"Allocate {points} {stat_name.upper()} ({job_name} build: {_build_summary(job_name)})",
                ))
                weighted_domains["progression"] = 0.95
                total_confidence = max(total_confidence, 0.95)

        # ── Class-aware cold start: Skill training ──
        if skill_points > 0:
            skill_training = _class_skill_training(job_name, skills, skill_points, self._adaptive)
            for skill_id, target_lv, desc in skill_training:
                actions.append(HeuristicAction(
                    kind="command", command=f"skills add {skill_id}",
                    confidence=0.90, domain="progression",
                    reason=f"Train {skill_id} ({desc}) for {job_name}",
                ))
                weighted_domains["progression"] = 0.90
                total_confidence = max(total_confidence, 0.90)

        # ── Class-aware cold start: Hunting ground routing ──
        hunting_ground = _class_hunting_ground(job_name, base_level, map_name, self._adaptive)
        if hunting_ground is not None:
            target_map, map_desc = hunting_ground
            actions.append(HeuristicAction(
                kind="command", command=f"move {target_map}",
                confidence=0.90, domain="exploration",
                reason=f"Level {base_level} {job_name} → {target_map} ({map_desc})",
            ))
            weighted_domains["exploration"] = 0.90
            total_confidence = max(total_confidence, 0.90)

        # Learn Basic Skill if not known (fallback for non-class skills)
        if "NV_BASIC" not in skills:
            actions.append(HeuristicAction(
                kind="command", command="skills add 1",
                confidence=0.95, domain="progression",
                reason="Learn Basic Skill to sit and regen",
            ))
            weighted_domains["progression"] = 0.95
            total_confidence = max(total_confidence, 0.95)

        # Learn First Aid if not known and Basic Skill is known
        elif "NV_FIRSTAID" not in skills:
            actions.append(HeuristicAction(
                kind="command", command="skills add 2",
                confidence=0.90, domain="progression",
                reason="Learn First Aid for self-healing",
            ))
            weighted_domains["progression"] = 0.90
            total_confidence = max(total_confidence, 0.90)

        # ── Economy: Restock potions ──
        has_potion = any("Potion" in str(k) for k in inventory) if isinstance(inventory, list) else False
        if not has_potion and zeny and zeny > 500 and signals.get("hp_ratio", 1.0) < 0.5:
            actions.append(HeuristicAction(
                kind="command", command="buy White Potion 30",
                confidence=0.85, domain="economy",
                reason="Restock healing potions (HP low, have zeny)",
            ))
            weighted_domains["economy"] = 0.85
            total_confidence = max(total_confidence, 0.85)

        # ── Economy: Sell junk when overweight ──
        weight_ratio = signals.get("weight_ratio", 0.0)
        if weight_ratio > 0.75:
            actions.append(HeuristicAction(
                kind="command", command="ai auto",
                confidence=0.70, domain="economy",
                reason=f"Near encumbered ({weight_ratio:.0%}) — LLM should plan vendor sell route",
                metadata={"needs_llm_vendor_route": True, "weight_ratio": weight_ratio},
            ))
            weighted_domains["economy"] = max(weighted_domains.get("economy", 0), 0.70)
            total_confidence = max(total_confidence, 0.70)

        # ── Map progression ──
        if base_level >= 10 and "prt_fild08" in map_name:
            actions.append(HeuristicAction(
                kind="command", command="move prt_fild04",
                confidence=0.60, domain="exploration",
                reason=f"Level {base_level} — move to better farming map",
            ))
            weighted_domains["exploration"] = 0.60
            total_confidence = max(total_confidence, 0.60)

        # Check recovery signal
        if signals.get("hp_ratio", 1.0) < 0.5:
            hp = signals["hp_ratio"]
            if hp < 0.2:
                # Critical HP — switch to manual so reflex rules can handle healing
                actions.append(HeuristicAction(
                    kind="reflex_override", command="ai manual",
                    confidence=0.95, domain="recovery",
                    reason=f"Critical HP ({hp:.0%}) — reflex healing should trigger",
                ))
                weighted_domains["recovery"] = 0.95
                total_confidence = max(total_confidence, 0.95)
            elif hp < 0.5:
                # Low HP — sit to recover
                actions.append(HeuristicAction(
                    kind="command", command="sit",
                    confidence=0.75, domain="recovery",
                    reason=f"Low HP ({hp:.0%})",
                ))
                weighted_domains["recovery"] = 0.75
                total_confidence = max(total_confidence, 0.75)

        # Check combat/aggro signal
        hostiles = signals.get("nearby_hostiles", 0)
        if hostiles > 0:
            if hostiles <= 3:
                actions.append(HeuristicAction(
                    kind="command", command="ai auto",
                    confidence=0.65, domain="grind",
                    reason=f"{hostiles} nearby hostiles (manageable)",
                ))
                weighted_domains["grind"] = 0.65
                total_confidence = max(total_confidence, 0.65)
            else:
                # Overwhelming — manual + flee handled by reflex
                actions.append(HeuristicAction(
                    kind="command", command="ai manual",
                    confidence=0.85, domain="recovery",
                    reason=f"{hostiles} nearby hostiles (overwhelming) — reflex flee should trigger",
                ))
                weighted_domains["recovery"] = max(weighted_domains.get("recovery", 0), 0.85)
                total_confidence = max(total_confidence, 0.85)

        # Check known map signal
        if signals.get("map_known", False):
            actions.append(HeuristicAction(
                kind="command", command="ai auto",
                confidence=0.55, domain="grind",
                reason="Known map, resuming auto mode",
            ))
            weighted_domains["grind"] = max(weighted_domains.get("grind", 0), 0.55)
            total_confidence = max(total_confidence, 0.55)

        # Check weight/encumbrance — let LLM plan vendor routing
        weight_ratio = signals.get("weight_ratio", 0)
        if weight_ratio and weight_ratio > 0.8:
            actions.append(HeuristicAction(
                kind="command", command="ai auto",
                confidence=0.70, domain="economy",
                reason=f"Near encumbered ({weight_ratio:.0%}) — LLM should plan vendor sell route",
                metadata={"needs_llm_vendor_route": True, "weight_ratio": weight_ratio},
            ))
            weighted_domains["economy"] = max(weighted_domains.get("economy", 0), 0.70)
            total_confidence = max(total_confidence, 0.70)

        # Check recent death — let LLM handle respawn logic
        if signals.get("recent_death", False):
            actions.append(HeuristicAction(
                kind="command", command="ai manual",
                confidence=0.90, domain="recovery",
                reason="Recent death detected — LLM should plan recovery",
                metadata={"needs_llm_recovery": True},
            ))
            weighted_domains["recovery"] = max(weighted_domains.get("recovery", 0), 0.90)
            total_confidence = max(total_confidence, 0.90)

        # ── JOB CHANGE: if base_lv >= 10 and job_lv >= 10 and still Novice ──
        job_level = signals.get("job_level", 0)
        if base_level >= 10 and job_level >= 10 and "novice" in job_name:
            # Route to job change NPC
            npc_info = JOB_CHANGE_NPCS.get("novice", ("prontera", 160, 191))
            npc_map, npc_x, npc_y = npc_info
            if "prontera" not in map_name:
                actions.append(HeuristicAction(
                    kind="command", command=f"move prontera",
                    confidence=0.98, domain="progression",
                    reason=f"Level {base_level}/{job_level} Novice — walk to Prontera for job change",
                ))
                weighted_domains["progression"] = 0.98
                total_confidence = max(total_confidence, 0.98)
            else:
                # Start job change dialog
                actions.append(HeuristicAction(
                    kind="command", command=f"talknpc {npc_x} {npc_y}",
                    confidence=0.98, domain="progression",
                    reason=f"Level {base_level}/{job_level} Novice — start job change at ({npc_x},{npc_y})",
                ))
                # Dialog responses: continue through intro, select job, confirm
                actions.append(HeuristicAction(
                    kind="command", command="talk continue",
                    confidence=0.95, domain="progression",
                    reason="Continue through job change NPC dialog",
                ))
                actions.append(HeuristicAction(
                    kind="command", command="talk resp 0",
                    confidence=0.95, domain="progression",
                    reason="Select first job option (Archer/Thief/Acolyte)",
                ))
                actions.append(HeuristicAction(
                    kind="command", command="talk resp 1",
                    confidence=0.90, domain="progression",
                    reason="Try second dialog option if first fails",
                ))
                actions.append(HeuristicAction(
                    kind="command", command="talk any",
                    confidence=0.85, domain="progression",
                    reason="Accept any dialog to complete job change",
                ))
                weighted_domains["progression"] = 0.98
                total_confidence = max(total_confidence, 0.98)

        # ── PARTY FORMATION (leader only): create party, invite siblings ──
        bot_name = signals.get("bot_name", "")
        is_leader = BOT_ROLES.get(bot_name, "") == "leader"
        in_party = signals.get("in_party", False)
        if is_leader and not in_party:
            actions.append(HeuristicAction(
                kind="command", command="party create",
                confidence=0.95, domain="social",
                reason=f"Leader {bot_name} — create party for team play",
            ))
            actions.append(HeuristicAction(
                kind="command", command="party share exp",
                confidence=0.95, domain="social",
                reason="Share experience in party",
            ))
            weighted_domains["social"] = 0.95
            total_confidence = max(total_confidence, 0.95)

        # ── PARTY INVITE: leader invites known sibling bots ──
        if is_leader and in_party:
            siblings = signals.get("nearby_players", [])
            party_members = signals.get("party_members", [])
            for sib_name in ["kicapmasin2", "kicapmasin3"]:
                if sib_name in siblings and sib_name not in party_members:
                    actions.append(HeuristicAction(
                        kind="command", command=f"party invite {sib_name}",
                        confidence=0.90, domain="social",
                        reason=f"Invite {sib_name} to party",
                    ))
                    weighted_domains["social"] = 0.90
                    total_confidence = max(total_confidence, 0.90)

        # ── FOLLOWER: follow leader's map ──
        if not is_leader:
            leader_map = signals.get("leader_map", "")
            if leader_map and leader_map != map_name and "prontera" not in map_name:
                actions.append(HeuristicAction(
                    kind="command", command=f"move {leader_map}",
                    confidence=0.85, domain="social",
                    reason=f"Follower — move to leader's map ({leader_map})",
                ))
                weighted_domains["social"] = 0.85
                total_confidence = max(total_confidence, 0.85)

        # ── ECONOMY: Walk to Prontera to sell/buy ──
        if weight_ratio > 0.70 and "prontera" not in map_name:
            actions.append(HeuristicAction(
                kind="command", command="move prontera",
                confidence=0.90, domain="economy",
                reason=f"Weight {weight_ratio:.0%} — return to Prontera to sell junk",
            ))
            weighted_domains["economy"] = max(weighted_domains.get("economy", 0), 0.90)
            total_confidence = max(total_confidence, 0.90)

        # ── ECONOMY: Sell junk when in Prontera ──
        if weight_ratio > 0.50 and "prontera" in map_name:
            actions.append(HeuristicAction(
                kind="command", command="ai auto",
                confidence=0.90, domain="economy",
                reason=f"Weight {weight_ratio:.0%} — in Prontera, LLM should plan sell route",
                metadata={"needs_llm_vendor_route": True, "weight_ratio": weight_ratio},
            ))
            weighted_domains["economy"] = max(weighted_domains.get("economy", 0), 0.90)
            total_confidence = max(total_confidence, 0.90)

        # ── ECONOMY: Buy potions when in Prontera ──
        has_potion = any("Potion" in str(k) for k in inventory) if isinstance(inventory, list) else False
        if not has_potion and zeny and zeny > 50 and "prontera" in map_name:
            actions.append(HeuristicAction(
                kind="command", command="talknpc 126 76",
                confidence=0.85, domain="economy",
                reason=f"Restock potions at Prontera Potion Shop (126,76) (zeny={zeny})",
            ))
            actions.append(HeuristicAction(
                kind="command", command="talk resp 0",
                confidence=0.80, domain="economy",
                reason="Select buy option at potion shop",
            ))
            actions.append(HeuristicAction(
                kind="command", command="talk continue",
                confidence=0.80, domain="economy",
                reason="Continue through potion shop dialog",
            ))
            weighted_domains["economy"] = max(weighted_domains.get("economy", 0), 0.85)
            total_confidence = max(total_confidence, 0.85)

        # ── ECONOMY: Buy arrows for Archer when in Prontera ──
        if "archer" in job_name and zeny > 10 and "prontera" in map_name:
            has_arrows = any("Arrow" in str(k) for k in inventory) if isinstance(inventory, list) else False
            if not has_arrows:
                actions.append(HeuristicAction(
                    kind="command", command="talknpc 160 133",
                    confidence=0.85, domain="economy",
                    reason=f"Buy arrows at Prontera Weapon Shop (160,133) (zeny={zeny})",
                ))
                actions.append(HeuristicAction(
                    kind="command", command="talk resp 0",
                    confidence=0.80, domain="economy",
                    reason="Select buy option at weapon shop",
                ))
                weighted_domains["economy"] = max(weighted_domains.get("economy", 0), 0.85)
                total_confidence = max(total_confidence, 0.85)

        # ── ECONOMY: Buy weapon if none equipped ──
        if zeny > 100 and "prontera" in map_name:
            has_weapon = any("Bow" in str(k) or "Knife" in str(k) or "Mace" in str(k) or "Sword" in str(k) or "Staff" in str(k) for k in inventory) if isinstance(inventory, list) else False
            if not has_weapon:
                actions.append(HeuristicAction(
                    kind="command", command="talknpc 160 133",
                    confidence=0.80, domain="economy",
                    reason=f"Buy weapon at Prontera Weapon Shop (160,133) (zeny={zeny})",
                ))
                actions.append(HeuristicAction(
                    kind="command", command="talk resp 0",
                    confidence=0.75, domain="economy",
                    reason="Select buy option at weapon shop",
                ))
                weighted_domains["economy"] = max(weighted_domains.get("economy", 0), 0.80)
                total_confidence = max(total_confidence, 0.80)

        # ── FEEDBACK: Check if previous commands succeeded ──
        # If stat_points > 0 and we already sent stat_add, something failed
        if stat_points > 0 and signals.get("_last_stat_points", 0) == stat_points:
            # stat_add didn't work - try with different syntax
            actions.append(HeuristicAction(
                kind="command", command="st add DEX 1",
                confidence=0.70, domain="progression",
                reason=f"stat_add failed, trying alternative syntax (st add)",
            ))
            weighted_domains["progression"] = max(weighted_domains.get("progression", 0), 0.70)
            total_confidence = max(total_confidence, 0.70)

        # ── MAP PROGRESSION: move to better hunting grounds ──
        hunting_ground = _class_hunting_ground(job_name, base_level, map_name)
        if hunting_ground is not None:
            target_map, map_desc = hunting_ground
            # Only queue if not already on this map
            if target_map not in map_name:
                actions.append(HeuristicAction(
                    kind="command", command=f"move {target_map}",
                    confidence=0.90, domain="exploration",
                    reason=f"Level {base_level} {job_name} → {target_map} ({map_desc})",
                ))
                weighted_domains["exploration"] = 0.90
                total_confidence = max(total_confidence, 0.90)

        # ── ADAPTIVE FEEDBACK: Record outcomes from previous cycle ──
        bot_id = signals.get("bot_id", "default")
        prev = self._prev_signals.get(bot_id, {})
        if prev:
            prev_map = prev.get("map", "")
            prev_hp = prev.get("hp_ratio", 1.0)
            curr_hp = signals.get("hp_ratio", 1.0)
            curr_map = signals.get("map", "")
            # Death detected: HP dropped to 0
            if prev_hp > 0.3 and curr_hp == 0:
                self._adaptive.record_death(prev_map, prev_hp)
            # Map change: record visit
            if prev_map and curr_map and prev_map != curr_map:
                self._adaptive.record_visit(curr_map)
            # Kill detected: exp gained
            prev_exp = prev.get("_last_exp", 0)
            curr_exp = signals.get("_last_exp", 0)
            if curr_exp > prev_exp:
                self._adaptive.record_kill(curr_map, curr_exp - prev_exp)
        # Store current signals for next cycle
        self._prev_signals[bot_id] = dict(signals)

        # ── ADAPTIVE CONFIDENCE: Lower confidence for decisions that should be LLM-driven ──
        # Map choice: low confidence if we have no data on this map
        if hunting_ground is not None:
            target_map = hunting_ground[0]
            map_score = self._adaptive.get_map_score(target_map)
            if map_score == 0:
                # Unknown map - let LLM decide
                for action in actions:
                    if action.domain == "exploration":
                        action.confidence = min(action.confidence, 0.5)
                        total_confidence = max(total_confidence, 0.5)

        # Stat allocation: lower confidence if previous attempt failed
        if stat_points > 0 and prev.get("stat_points", 0) == stat_points:
            for action in actions:
                if action.domain == "progression" and "stat_add" in action.command:
                    action.confidence = min(action.confidence, 0.6)
                    total_confidence = max(total_confidence, 0.6)

        # Determine top domain
        top_domain = "none"
        if weighted_domains:
            top_domain = str(max(weighted_domains, key=lambda k: float(weighted_domains.get(k, 0.0))))

        assessment = HeuristicAssessment(
            horizon=signals.get("horizon", "short_term"),
            actions=actions,
            confidence=total_confidence,
            actionable=len(actions) > 0,
            top_domain=top_domain,
            signals=dict(signals),
        )
        bot_id = signals.get("bot_id", "default")
        self._last_assessment[bot_id] = assessment
        return assessment

    def confidence_for(self, horizon: str, signals: dict | None = None, bot_id: str = "default") -> float:
        """Called by PDCA loop to check if heuristic can replace LLM for this horizon.

        Returns the confidence from the last assessment for this bot_id.
        """
        if signals is not None:
            sigs = dict(signals)
            sigs.setdefault("bot_id", bot_id)
            result = self.assess(sigs)
            return result.confidence
        last = self._last_assessment.get(bot_id) if hasattr(self, '_last_assessment') else None
        if last is not None:
            return last.confidence
        return 0.0


def _build_summary(job_name: str) -> str:
    """Return a short stat build summary for a class."""
    builds = {
        "novice":    "DEX>STR>AGI>VIT",
        "swordman":  "STR>VIT>DEX",
        "mage":      "INT>DEX",
        "archer":    "DEX>AGI",
        "acolyte":   "INT>DEX",
        "merchant":  "STR>VIT>DEX",
        "thief":     "AGI>DEX",
        "taekwon":   "STR>AGI",
        "gunslinger":"DEX>AGI",
        "ninja":     "INT>DEX",
        "soul_linker":"INT>DEX",
    }
    return builds.get(job_name, "DEX>STR")
