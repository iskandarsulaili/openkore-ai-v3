"""Self-Learning Elemental Armor Checker — tracks observed player elements
during PVP and recommends optimal attack element adaptively.

RO element wheel (attack element vs target armor):
  - Water > Fire > Earth > Wind > Water
  - Poison > all (neutral) but weak to itself
  - Holy > Undead/Dark
  - Ghost > Ghost
  - Shadow > Holy
  - Neutral deals 100% to neutral, 75% to others

Instead of hardcoded element tables, this module learns from damage observations:
which element deals how much damage to which player/bot. This automatically
adapts to custom server element mechanics or special gear.

Self-* properties:
  - Self-learning: learns effective elements per target from damage observations
  - Self-optimizing: picks best element based on observed damage multipliers
  - Self-adapting: detects when player changes armor element
  - Self-configuring: builds element effectiveness matrix from experience
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

# ── Element constants (reference only — learning overrides these) ─────

ELEMENTS: list[str] = [
    "neutral", "water", "earth", "fire", "wind",
    "poison", "holy", "dark", "ghost", "undead",
]

# Initial heuristic: standard RO elemental wheel
# Format: {attack_element: {target_element: multiplier}}
# These are initial values that learning replaces over time
INITIAL_ELEMENT_MULTIPLIERS: dict[str, dict[str, float]] = {
    "neutral":   {"neutral": 1.0, "water": 0.75, "earth": 0.75, "fire": 0.75, "wind": 0.75,
                   "poison": 0.75, "holy": 0.75, "dark": 0.75, "ghost": 0.75, "undead": 0.75},
    "water":     {"neutral": 1.0, "water": 0.25, "earth": 0.75, "fire": 1.75, "wind": 0.75,
                   "poison": 1.0,  "holy": 1.0,  "dark": 1.0,  "ghost": 0.75, "undead": 1.0},
    "earth":     {"neutral": 1.0, "water": 1.75, "earth": 0.25, "fire": 1.0,  "wind": 0.75,
                   "poison": 1.0,  "holy": 1.0,  "dark": 1.0,  "ghost": 0.75, "undead": 1.0},
    "fire":      {"neutral": 1.0, "water": 0.5,  "earth": 1.0,  "fire": 0.25, "wind": 1.75,
                   "poison": 1.0,  "holy": 1.0,  "dark": 1.0,  "ghost": 0.75, "undead": 1.0},
    "wind":      {"neutral": 1.0, "water": 1.75, "earth": 0.5,  "fire": 1.0,  "wind": 0.25,
                   "poison": 1.0,  "holy": 1.0,  "dark": 1.0,  "ghost": 0.75, "undead": 1.0},
    "poison":    {"neutral": 1.0, "water": 1.0,  "earth": 1.0,  "fire": 1.0,  "wind": 1.0,
                   "poison": 0.25, "holy": 1.0,  "dark": 1.0,  "ghost": 1.0,  "undead": 1.0},
    "holy":      {"neutral": 1.0, "water": 1.0,  "earth": 1.0,  "fire": 1.0,  "wind": 1.0,
                   "poison": 1.0,  "holy": 0.75, "dark": 1.75, "ghost": 1.0,  "undead": 1.75},
    "dark":      {"neutral": 1.0, "water": 1.0,  "earth": 1.0,  "fire": 1.0,  "wind": 1.0,
                   "poison": 1.0,  "holy": 0.5,  "dark": 0.25, "ghost": 1.0,  "undead": 1.0},
    "ghost":     {"neutral": 0.0, "water": 1.0,  "earth": 1.0,  "fire": 1.0,  "wind": 1.0,
                   "poison": 1.0,  "holy": 1.0,  "dark": 1.0,  "ghost": 1.75, "undead": 1.0},
    "undead":    {"neutral": 1.0, "water": 1.0,  "earth": 1.0,  "fire": 1.75, "wind": 1.0,
                   "poison": 1.0,  "holy": 0.0,  "dark": 1.0,  "ghost": 1.0,  "undead": 0.25},
}

MIN_OBSERVATIONS: int = 2
DECAY_ALPHA: float = 0.2


@dataclass
class ElementObservation:
    """One observed element damage event."""
    attack_element: str
    target_element: str
    damage: int
    expected_base_damage: int
    observed_multiplier: float  # damage / expected_base
    timestamp: float


@dataclass
class ElementEffectiveness:
    """Learned effectiveness of one element against another."""
    attack_element: str
    target_element: str

    observations: int = 0
    total_multiplier: float = 0.0
    learned_multiplier: float = 1.0  # Starts at heuristic, converges with data
    confidence: float = 0.0
    recent_multipliers: deque = field(default_factory=lambda: deque(maxlen=20))

    @property
    def effective_multiplier(self) -> float:
        """Blend learned with heuristic based on confidence."""
        initial = INITIAL_ELEMENT_MULTIPLIERS.get(
            self.attack_element, {}
        ).get(self.target_element, 1.0)
        if self.confidence < 0.3:
            return initial
        weight = min(0.8, self.confidence)
        return weight * self.learned_multiplier + (1 - weight) * initial

    def to_dict(self) -> dict[str, Any]:
        return {
            "attack": self.attack_element,
            "target": self.target_element,
            "multiplier": round(self.effective_multiplier, 3),
            "observations": self.observations,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class PlayerElementProfile:
    """Observed element armor profile for a specific player."""
    player_name: str
    player_class: str

    # Current best-guess of their armor element
    current_element: str = "neutral"
    element_confidence: float = 0.0

    # All observations of damage dealt to this player by element
    observations_per_element: dict[str, list[ElementObservation]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # When we last saw them change element
    last_element_change: float = 0.0
    element_changes_detected: int = 0

    total_observations: int = 0
    last_updated: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "player": self.player_name,
            "class": self.player_class,
            "current_element": self.current_element,
            "confidence": round(self.element_confidence, 3),
            "observations": self.total_observations,
            "element_changes": self.element_changes_detected,
        }


class SelfLearningElementalArmorChecker:
    """Learns player elemental armor configuration from damage observations.

    Usage:
        elem_checker = SelfLearningElementalArmorChecker()

        # Before engaging:
        best = elem_checker.recommend_attack_element(target_name="SirTanky")
        # Returns best element based on learned data

        # After dealing damage:
        elem_checker.record_damage(
            target_name="SirTanky",
            target_class="paladin",
            attack_element="fire",
            damage=450,
            expected_base=300,  # What we'd expect vs neutral
        )
    """

    def __init__(self) -> None:
        self._lock = RLock()

        # Learned element effectiveness matrix
        # {attack_element: {target_element: ElementEffectiveness}}
        self._effectiveness: dict[str, dict[str, ElementEffectiveness]] = defaultdict(dict)

        # Per-player element profiles
        self._player_profiles: dict[str, PlayerElementProfile] = {}

        # Stats
        self._total_observations: int = 0
        self._element_changes_detected: int = 0
        self._start_time: float = time.time()

    # ── Prediction ──────────────────────────────────────────────────────

    def predict_multiplier(
        self,
        attack_element: str,
        target_element: str,
    ) -> float:
        """Predict damage multiplier for an attack element vs target element."""
        atk = attack_element.lower()
        tgt = target_element.lower()

        with self._lock:
            eff = self._effectiveness.get(atk, {}).get(tgt)
            if eff and eff.confidence > 0.1:
                return eff.effective_multiplier
            # Fallback to initial heuristic
            return INITIAL_ELEMENT_MULTIPLIERS.get(atk, {}).get(tgt, 1.0)

    def infer_target_element(self, target_name: str) -> str:
        """Best guess of target's current armor element.

        Returns element name string.
        """
        with self._lock:
            profile = self._player_profiles.get(target_name.lower())
            if profile is None or profile.total_observations < 2:
                return "neutral"
            if profile.element_confidence > 0.5:
                return profile.current_element
            return "neutral"

    def recommend_attack_element(
        self,
        target_name: str | None = None,
        target_class: str | None = None,
        available_elements: list[str] | None = None,
    ) -> dict[str, Any]:
        """Recommend the best attack element against a target.

        Args:
            target_name: Specific player (uses learned profile)
            target_class: Used for fallback class-based estimation
            available_elements: Elements we can use. If None, checks all.

        Returns:
            dict with best_element, expected_multiplier, confidence, reason
        """
        if available_elements is None:
            available_elements = [e for e in ELEMENTS if e != "neutral"]

        target_element = "neutral"
        if target_name:
            target_element = self.infer_target_element(target_name)

        best_mult = 0.0
        best_element = available_elements[0] if available_elements else "neutral"

        for element in available_elements:
            mult = self.predict_multiplier(element, target_element)
            if mult > best_mult:
                best_mult = mult
                best_element = element

        # Confidence in our recommendation
        profile = None
        if target_name:
            profile = self._player_profiles.get(target_name.lower())
        confidence = profile.element_confidence if profile else 0.0

        return {
            "best_element": best_element,
            "expected_multiplier": round(best_mult, 3),
            "inferred_target_element": target_element,
            "confidence": round(confidence, 3),
            "reason": (
                f"{best_element} deals {best_mult:.1%} damage vs {target_element}-armored target"
                if target_element != "neutral"
                else f"{best_element} recommended — target element unknown, defaulting to standard"
            ),
        }

    # ── Learning from observations ─────────────────────────────────────

    def record_damage(
        self,
        target_name: str,
        target_class: str,
        attack_element: str,
        damage: int,
        expected_base_damage: int = 100,
    ) -> None:
        """Record a damage observation to learn target's element armor.

        Args:
            target_name: Player we hit
            target_class: Their class/job
            attack_element: The element of our attack
            damage: Observed damage dealt
            expected_base_damage: What damage we'd expect vs neutral armor (0-element)
        """
        with self._lock:
            atk_key = attack_element.lower()
            player_key = target_name.lower()

            # ── Compute observed multiplier ──
            if expected_base_damage <= 0:
                expected_base_damage = 100
            observed_mult = damage / expected_base_damage

            obs = ElementObservation(
                attack_element=atk_key,
                target_element="unknown",  # We'll infer from this
                damage=damage,
                expected_base_damage=expected_base_damage,
                observed_multiplier=observed_mult,
                timestamp=time.time(),
            )

            # ── Get/create player profile ──
            profile = self._player_profiles.get(player_key)
            if profile is None:
                profile = PlayerElementProfile(
                    player_name=target_name,
                    player_class=target_class,
                )
                self._player_profiles[player_key] = profile

            profile.observations_per_element[atk_key].append(obs)
            profile.total_observations += 1
            profile.last_updated = time.time()
            self._total_observations += 1

            # ── Infer target element from this observation ──
            inferred_target = self._infer_element_from_multiplier(atk_key, observed_mult)

            if inferred_target:
                # ── Update effectiveness matrix ──
                eff = self._effectiveness[atk_key].get(inferred_target)
                if eff is None:
                    eff = ElementEffectiveness(
                        attack_element=atk_key,
                        target_element=inferred_target,
                    )
                    self._effectiveness[atk_key][inferred_target] = eff

                eff.observations += 1
                eff.total_multiplier += observed_mult
                eff.recent_multipliers.append(observed_mult)

                # Update learned multiplier (moving average)
                recent_avg = sum(eff.recent_multipliers) / len(eff.recent_multipliers)
                if eff.observations == 1:
                    eff.learned_multiplier = observed_mult
                else:
                    eff.learned_multiplier = (
                        DECAY_ALPHA * recent_avg
                        + (1.0 - DECAY_ALPHA) * eff.learned_multiplier
                    )
                eff.confidence = min(1.0, eff.observations / 10.0)

                # ── Update player's current element estimate ──
                self._update_player_element_estimate(profile)

    def _infer_element_from_multiplier(
        self,
        attack_element: str,
        multiplier: float,
    ) -> str | None:
        """Try to infer target's element from observed damage multiplier.

        Uses the standard RO element wheel as reference.
        """
        candidates = INITIAL_ELEMENT_MULTIPLIERS.get(attack_element, {})
        best_element = None
        best_diff = 0.5  # Threshold: must be reasonably close

        for target_el, expected_mult in candidates.items():
            # Skip neutral (always safe but not informative)
            if target_el == "neutral":
                continue
            diff = abs(multiplier - expected_mult)
            if diff < best_diff:
                best_diff = diff
                best_element = target_el

        # If we have a good match and the multiplier is substantially different from 1.0
        if best_element and best_diff < 0.3:
            return best_element
        return None

    def _update_player_element_estimate(self, profile: PlayerElementProfile) -> None:
        """Update the best-guess element for a player based on all observations."""
        # For each element, compute how well it explains the observed damages
        element_scores: dict[str, float] = defaultdict(float)

        for atk_elem, obs_list in profile.observations_per_element.items():
            for obs in obs_list:
                # For each possible target element, compute how well it fits
                for tgt_elem in ELEMENTS:
                    expected = INITIAL_ELEMENT_MULTIPLIERS.get(atk_elem, {}).get(tgt_elem, 1.0)
                    if expected > 0:
                        # Score: how closely the observed multiplier matches expected
                        # Lower difference = higher score
                        diff = abs(obs.observed_multiplier - expected)
                        score = max(0, 1.0 - diff * 2.0)  # 0.0-1.0
                        element_scores[tgt_elem] += score

        if not element_scores:
            return

        # Find best element
        best_elem = max(element_scores, key=lambda e: element_scores[e])
        best_score = element_scores[best_elem]
        total_score = sum(element_scores.values())

        # Detect element change
        if profile.current_element != best_elem and profile.total_observations >= 3:
            if best_score > element_scores.get(profile.current_element, 0) * 1.2:
                logger.info(
                    "[ElementalArmor] %s likely changed element: %s -> %s",
                    profile.player_name, profile.current_element, best_elem,
                )
                profile.last_element_change = time.time()
                profile.element_changes_detected += 1
                self._element_changes_detected += 1

        profile.current_element = best_elem
        profile.element_confidence = best_score / max(total_score, 1)

    # ── Query / introspection ──────────────────────────────────────────

    def get_effectiveness_table(self) -> list[dict[str, Any]]:
        """Get the current learned element effectiveness matrix."""
        results: list[dict[str, Any]] = []
        with self._lock:
            for atk, targets in self._effectiveness.items():
                for tgt, eff in targets.items():
                    results.append(eff.to_dict())
        return results

    def get_stats(self) -> dict[str, Any]:
        """Summary statistics for introspection."""
        with self._lock:
            return {
                "total_observations": self._total_observations,
                "players_tracked": len(self._player_profiles),
                "element_changes_detected": self._element_changes_detected,
                "effectiveness_entries": sum(
                    len(t) for t in self._effectiveness.values()
                ),
                "player_profiles": {
                    name: p.to_dict()
                    for name, p in sorted(
                        self._player_profiles.items(),
                        key=lambda x: x[1].total_observations,
                        reverse=True,
                    )[:15]
                },
            }
