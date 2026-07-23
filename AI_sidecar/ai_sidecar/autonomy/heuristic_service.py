from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

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
) -> list[tuple[str, int]]:
    """Determine which stats to allocate for a given class.

    Returns a list of (stat_name, points_to_add) tuples.
    """
    if stat_points <= 0:
        return []

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
) -> tuple[str, str] | None:
    """Find the best hunting ground for a class at a given level.

    Returns (map_name, description) or None if already on the best map.
    """
    grounds = CLASS_HUNTING_GROUNDS.get(job_name, CLASS_HUNTING_GROUNDS["novice"])
    best: tuple[int, int, str, str] | None = None

    for entry in grounds:
        min_lv, max_lv, map_name, desc = entry
        if min_lv <= base_level <= max_lv:
            if best is None or min_lv > best[0]:
                best = entry

    if best is None:
        # Fallback: pick the last entry (highest level range)
        if grounds:
            best = grounds[-1]

    if best is not None:
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
) -> list[tuple[str, int, str]]:
    """Determine which skills to train next for a given class.

    Returns a list of (skill_id, target_level, description) tuples.
    """
    if skill_points <= 0:
        return []

    priorities = CLASS_SKILL_PRIORITIES.get(job_name, CLASS_SKILL_PRIORITIES["novice"])
    result: list[tuple[str, int, str]] = []

    for skill_id, max_lv, desc in priorities:
        if skill_id not in known_skills:
            result.append((skill_id, 1, desc))
            return result  # Train one skill at a time

    return result


class HeuristicService:
    """Produces heuristic actions from game state signals without calling LLM.

    Maps the existing decision_service opportunistic signals to executable actions.
    The confidence score determines whether the PDCA loop skips the LLM entirely.
    """

    def __init__(self):
        self._last_assessment: dict[str, HeuristicAssessment] = {}
        self._domain_weights: dict[str, float] = {
            "recovery": 0.15,
            "grind": 0.30,
            "economy": 0.25,
            "quest": 0.10,
            "exploration": 0.20,
        }

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
            allocations = _class_stat_allocation(job_name, current_stats, stat_points)
            for stat_name, points in allocations:
                stat_upper = stat_name.upper()
                actions.append(HeuristicAction(
                    kind="command", command=f"stat_add {stat_upper} {points}",
                    confidence=0.95, domain="progression",
                    reason=f"Allocate {points} {stat_upper} ({job_name} build: {_build_summary(job_name)})",
                ))
                weighted_domains["progression"] = 0.95
                total_confidence = max(total_confidence, 0.95)

        # ── Class-aware cold start: Skill training ──
        if skill_points > 0:
            skill_training = _class_skill_training(job_name, skills, skill_points)
            for skill_id, target_lv, desc in skill_training:
                actions.append(HeuristicAction(
                    kind="command", command=f"skills add {skill_id}",
                    confidence=0.90, domain="progression",
                    reason=f"Train {skill_id} ({desc}) for {job_name}",
                ))
                weighted_domains["progression"] = 0.90
                total_confidence = max(total_confidence, 0.90)

        # ── Class-aware cold start: Hunting ground routing ──
        hunting_ground = _class_hunting_ground(job_name, base_level, map_name)
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
