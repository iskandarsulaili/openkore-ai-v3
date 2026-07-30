"""Combat Intelligence — integrates PVP modules into combat decisions.

Wires the 4 self-learning PVP modules (GTB detection, elemental armor
checker, class counters, hit/flee analyzer) into the assess()-compatible
combat pipeline so they produce real HeuristicAction objects during
combat ticks instead of sitting unused.

Each module is lazily instantiated as a module-level singleton (matching
the pattern used by get_opponent_model(), get_tactics_dispatcher(), etc.).

Usage:
    from ai_sidecar.domains.combat.combat_intel import assess_combat_intel

    # Called each combat tick from the domain loop:
    assess_combat_intel(signals, actions, bot_id)

    # The assess() function appends HeuristicActions for:
    #   - "switch_to_magic" / "switch_to_physical"  (GTB)
    #   - "use_element"  (elemental armor recommendation)
    #   - "class_advantage"  (class counter heads-up)
    #   - "physical_viability"  (hit/flee recommendation)
"""

from __future__ import annotations

import logging
from typing import Any

from ai_sidecar.actions import HeuristicAction
from ai_sidecar.domains.pvp.gtb_detector import GtbDetector
from ai_sidecar.domains.pvp.self_learning_elemental_armor_checker import (
    SelfLearningElementalArmorChecker,
)
from ai_sidecar.domains.pvp.self_learning_class_counters import (
    SelfLearningClassCounters,
)
from ai_sidecar.domains.pvp.self_learning_hit_flee_analyzer import (
    SelfLearningHitFleeAnalyzer,
)

logger = logging.getLogger(__name__)

# ── Module-level singletons (lazy) ──────────────────────────────────

_gtb_detector: GtbDetector | None = None
_elemental_checker: SelfLearningElementalArmorChecker | None = None
_class_counters: SelfLearningClassCounters | None = None
_hit_flee_analyzer: SelfLearningHitFleeAnalyzer | None = None


def get_gtb_detector() -> GtbDetector:
    global _gtb_detector
    if _gtb_detector is None:
        _gtb_detector = GtbDetector()
    return _gtb_detector


def get_elemental_checker() -> SelfLearningElementalArmorChecker:
    global _elemental_checker
    if _elemental_checker is None:
        _elemental_checker = SelfLearningElementalArmorChecker()
    return _elemental_checker


def get_class_counters() -> SelfLearningClassCounters:
    global _class_counters
    if _class_counters is None:
        _class_counters = SelfLearningClassCounters()
    return _class_counters


def get_hit_flee_analyzer() -> SelfLearningHitFleeAnalyzer:
    global _hit_flee_analyzer
    if _hit_flee_analyzer is None:
        _hit_flee_analyzer = SelfLearningHitFleeAnalyzer()
    return _hit_flee_analyzer


# ── Helper: extract current target from signals ─────────────────────


def _extract_current_target(signals: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the current PVP target (player) from signals.

    Looks in:
      1. signals["combat"]["target"] — explicit PVP target dict
      2. signals["actors"] — first hostile player in range
      3. signals["target"] — flat target

    Returns a dict with at minimum 'name' and 'class' (if known),
    or None if no PVP target is found.
    """
    # Prefer explicit PVP target
    combat = signals.get("combat", {})
    target = combat.get("target") or signals.get("target")
    if isinstance(target, dict):
        name = target.get("name", "") or target.get("actor_name", "")
        if name:
            return {
                "name": name,
                "class": target.get("class", "") or target.get("job", ""),
                "level": target.get("level", 1),
                "hp": target.get("hp", 1),
                "max_hp": target.get("max_hp", 1),
                "is_player": True,
            }

    # Fallback: scan actors for a hostile player
    actors = signals.get("actors") or signals.get("monsters_around", [])
    for actor in actors:
        if actor.get("type") == "player" and actor.get("hp", 0) > 0:
            name = actor.get("name", "")
            if name:
                return {
                    "name": name,
                    "class": actor.get("class", "") or actor.get("job", ""),
                    "level": actor.get("level", 1),
                    "hp": actor.get("hp", 1),
                    "max_hp": actor.get("max_hp", 1),
                    "is_player": True,
                }

    return None


def _extract_my_stats(signals: dict[str, Any]) -> dict[str, Any]:
    """Extract 'my stats' from signals for hit/flee and class analysis."""
    vitals = signals.get("vitals", {}) or signals.get("stats", {})
    return {
        "base_level": int(vitals.get("base_level", vitals.get("level", 1))),
        "dex": int(vitals.get("dex", 1)),
        "agi": int(vitals.get("agi", 1)),
        "hit_bonus": int(vitals.get("hit_bonus", 0)),
        "class": str(vitals.get("job_name", vitals.get("class", "novice"))),
        "job_name": str(vitals.get("job_name", vitals.get("class", "novice"))),
    }


def _extract_target_stats(target: dict[str, Any]) -> dict[str, Any]:
    """Extract target stats dict for hit/flee analysis."""
    return {
        "base_level": int(target.get("level", 1)),
        "agi": int(target.get("agi", 1)),
        "flee_bonus": int(target.get("flee_bonus", 0)),
        "class": target.get("class", ""),
        "job_name": target.get("class", ""),
    }


# ── CombatIntel assess() ────────────────────────────────────────────


def assess_combat_intel(
    signals: dict[str, Any],
    actions: list[HeuristicAction],
    bot_id: str,
) -> None:
    """Assess PVP combat intelligence and append HeuristicActions.

    Call this once per combat tick from the DomainRegistry or the
    dispatcher's assess() chain.  It checks all 4 PVP modules and
    produces actions that the OpenKore bridge can execute.

    Appended action kinds:
      - "switch_to_magic" / "switch_to_physical" (GTB detection)
      - "use_element" (elemental armor recommendation)
      - "class_advantage" (class counter advantage score)
      - "physical_viability" (hit/flee recommend physical or magic)

    Args:
        signals: Raw state signals (same shape as dispatcher.assess()).
        actions: Mutable list to append HeuristicAction objects to.
        bot_id: Bot identifier string.
    """
    try:
        target = _extract_current_target(signals)
        if not target:
            return  # No PVP target — nothing to do

        my_stats = _extract_my_stats(signals)
        target_stats = _extract_target_stats(target)

        target_name: str = target.get("name", "unknown")
        target_class: str = target.get("class", "")
        my_class: str = my_stats.get("class", "novice")
        my_job: str = my_stats.get("job_name", my_class)

        # ── 1. GTB detection ─────────────────────────────────────
        _check_gtb(target_name, target_class, actions)

        # ── 2. Elemental armor recommendation ────────────────────
        _check_elemental(target_name, target_class, actions)

        # ── 3. Class counter advantage ───────────────────────────
        _check_class_counter(my_job, target_class, target_name, actions)

        # ── 4. Hit/flee physical viability ───────────────────────
        _check_hit_flee(my_stats, target_stats, target_name, actions)

    except Exception as e:
        logger.error("combat_intel.assess_combat_intel() failed: %s", e, exc_info=True)


# ── Individual module checks ────────────────────────────────────────


def _check_gtb(
    target_name: str,
    target_class: str,
    actions: list[HeuristicAction],
) -> None:
    """Check GTB status and append a switch-to-physical/magic action if needed."""
    gtb = get_gtb_detector()
    advice = gtb.get_engagement_advice(target_name, target_class)

    # Only act if we have reasonable confidence
    if advice.get("confidence", 0) < 0.3:
        return

    if advice.get("use_physical") and not advice.get("use_magic"):
        actions.append(HeuristicAction(
            kind="command",
            command="switch_to_physical",
            confidence=advice.get("confidence", 0.6),
            domain="combat_intel",
            reason=f"gtb_detected_{target_name}",
            metadata={
                "target": target_name,
                "gtb_probability": advice.get("gtb_probability", 0.0),
                "module": "gtb_detector",
            },
        ))
    elif advice.get("use_magic") and not advice.get("use_physical"):
        actions.append(HeuristicAction(
            kind="command",
            command="switch_to_magic",
            confidence=advice.get("confidence", 0.6),
            domain="combat_intel",
            reason=f"no_gtb_{target_name}",
            metadata={
                "target": target_name,
                "gtb_probability": advice.get("gtb_probability", 0.0),
                "module": "gtb_detector",
            },
        ))

    # If uncertain, append a log action suggesting a test
    if advice.get("re_test_suggested"):
        actions.append(HeuristicAction(
            kind="log",
            command="",
            confidence=0.5,
            domain="combat_intel",
            reason=f"gtb_retest_needed_{target_name}",
            metadata={
                "target": target_name,
                "gtb_probability": advice.get("gtb_probability", 0.0),
                "suggested_action": "cast_cheap_spell_to_test_gtb",
                "module": "gtb_detector",
            },
        ))


def _check_elemental(
    target_name: str,
    target_class: str,
    actions: list[HeuristicAction],
) -> None:
    """Check elemental armor and recommend best attack element."""
    checker = get_elemental_checker()
    rec = checker.recommend_attack_element(
        target_name=target_name,
        target_class=target_class,
    )

    if rec.get("confidence", 0) < 0.2:
        return

    best_element = rec.get("best_element", "neutral")
    multiplier = rec.get("expected_multiplier", 1.0)
    inferred_target_element = rec.get("inferred_target_element", "neutral")

    # Only recommend a non-neutral element if we have evidence
    if best_element != "neutral":
        actions.append(HeuristicAction(
            kind="command",
            command=f"use_element {best_element}",
            confidence=min(0.9, max(0.4, rec.get("confidence", 0.4))),
            domain="combat_intel",
            reason=f"elemental_advice_{target_name}_{best_element}",
            metadata={
                "target": target_name,
                "recommended_element": best_element,
                "expected_multiplier": multiplier,
                "inferred_target_element": inferred_target_element,
                "confidence": rec.get("confidence", 0.0),
                "module": "elemental_armor_checker",
            },
        ))


def _check_class_counter(
    my_class: str,
    target_class: str,
    target_name: str,
    actions: list[HeuristicAction],
) -> None:
    """Check class counter advantage and append advice."""
    if not target_class:
        return

    counters = get_class_counters()
    win_rate = counters.get_win_rate(my_class, target_class)

    # Always log the matchup for learning — even neutral info is useful
    advantage = "advantage" if win_rate > 0.55 else "disadvantage" if win_rate < 0.45 else "neutral"

    actions.append(HeuristicAction(
        kind="log",
        command="",
        confidence=0.7,
        domain="combat_intel",
        reason=f"class_counter_{my_class}_vs_{target_class}",
        metadata={
            "target": target_name,
            "my_class": my_class,
            "target_class": target_class,
            "predicted_win_rate": round(win_rate, 3),
            "advantage": advantage,
            "module": "class_counters",
        },
    ))

    # If we have a strong disadvantage, suggest a tactical change
    if win_rate < 0.35:
        # Try to find a better class
        best = counters.get_best_counter(target_class)
        if best and best.my_class != my_class and best.is_reliable:
            actions.append(HeuristicAction(
                kind="command",
                command=f"consider_class_switch {best.my_class}",
                confidence=0.5,
                domain="combat_intel",
                reason=f"class_disadvantage_{my_class}_vs_{target_class}_recommend_{best.my_class}",
                metadata={
                    "target": target_name,
                    "current_class": my_class,
                    "recommended_class": best.my_class,
                    "predicted_win_rate": round(win_rate, 3),
                    "improved_win_rate": round(best.predicted_win_rate, 3),
                    "advantage": advantage,
                    "module": "class_counters",
                },
            ))


def _check_hit_flee(
    my_stats: dict[str, Any],
    target_stats: dict[str, Any],
    target_name: str,
    actions: list[HeuristicAction],
) -> None:
    """Check hit/flee physical viability and append recommendation."""
    analyzer = get_hit_flee_analyzer()
    rec = analyzer.recommend_approach(my_stats, target_stats, target_name)

    if rec.get("confidence", 0) < 0.3:
        return

    estimated_hit_rate = rec.get("estimated_hit_rate", 0.95)
    use_magic = rec.get("use_magic", False)
    use_physical = rec.get("use_physical", True)

    # If physical is hopeless, recommend switching to magic
    if use_magic and not use_physical:
        actions.append(HeuristicAction(
            kind="command",
            command="switch_to_magic_approach",
            confidence=min(0.9, rec.get("confidence", 0.6)),
            domain="combat_intel",
            reason=f"physical_unviable_{target_name}_hitrate_{estimated_hit_rate:.0%}",
            metadata={
                "target": target_name,
                "estimated_hit_rate": round(estimated_hit_rate, 3),
                "calibration_offset": rec.get("calibration_offset", 0.0),
                "confidence": rec.get("confidence", 0.0),
                "samples": rec.get("samples", 0),
                "module": "hit_flee_analyzer",
            },
        ))
    elif estimated_hit_rate < 0.50:
        # Marginal — append a cautionary log
        actions.append(HeuristicAction(
            kind="log",
            command="",
            confidence=0.5,
            domain="combat_intel",
            reason=f"physical_marginal_{target_name}_hitrate_{estimated_hit_rate:.0%}",
            metadata={
                "target": target_name,
                "estimated_hit_rate": round(estimated_hit_rate, 3),
                "recommendation": "consider_magic_switch",
                "module": "hit_flee_analyzer",
            },
        ))


# ── Convenience: module-level assess (same signature as other domains) ──


def assess(signals: dict[str, Any], actions: list[HeuristicAction], bot_id: str) -> None:
    """Alias for assess_combat_intel — identical signature to dispatcher.assess()."""
    assess_combat_intel(signals, actions, bot_id)
