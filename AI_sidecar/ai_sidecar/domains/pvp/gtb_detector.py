"""GTB (Golden Thief Bug) Armor Detector — learns to identify GTB card users.

Golden Thief Bug card: grants complete magic immunity (all magic skills deal 0 damage).
Detecting GTB is critical — if the target wears GTB, the bot MUST switch to physical
attacks and NOT waste SP on magic.

Detection method:
  1. Cast a cheap, fast-cast spell (e.g. Napalm Beat, Fire Bolt Lv1)
  2. Observe damage:
     - 0 damage + no visual resist message → likely GTB
     - 0 damage + "resisted" message → high MDEF (no GTB)
     - Normal damage → no GTB
  3. Learn from each test and build player profiles

Self-* properties:
  - Self-learning: builds GTB probability model per player from damage observations
  - Self-optimizing: chooses cheapest detection spell based on SP cost vs. reliability
  - Self-adapting: detects when player might have switched cards (GTB on -> off)
  - Self-healing: if multiple low-damage tests give conflicting results, retests
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

# ── Constants ──────────────────────────────────────────────────────────

# Minimum MDEF to make 0-damage detection ambiguous
HIGH_MDEF_THRESHOLD: int = 80

# How many test observations before we reach high confidence
MIN_TESTS_FOR_CONFIDENCE: int = 3

# Decay factor for old observations
OBSERVATION_DECAY: float = 0.3

# GTB behavior: 0 damage when hit by magic, regardless of MDEF
# Non-GTB 0 damage: very high MDEF reduces damage but rarely to exactly 0


@dataclass
class DamageObservation:
    """One observation of a magic attack against a player."""
    timestamp: float
    damage: int
    spell_name: str
    spell_level: int = 1
    expected_min_damage: int = 100  # What we'd expect without GTB at this level
    is_zero_damage: bool = False

    @property
    def suggests_gtb(self) -> bool:
        """True if this observation is consistent with GTB.

        GTB = exactly 0 damage regardless of MDEF.
        Non-GTB very high MDEF = damage close to 0 but usually 1-10, not exactly 0.
        """
        return self.is_zero_damage


@dataclass
class GtbPlayerProfile:
    """Learned GTB profile for one player."""
    player_name: str
    player_class: str

    # Observations
    observations: list[DamageObservation] = field(default_factory=list)
    total_tests: int = 0
    zero_damage_count: int = 0
    non_zero_damage_count: int = 0
    nonzero_min_damage: int = 999999  # Lowest non-zero magic damage observed

    # Learned estimate
    gtb_probability: float = 0.3  # Default: somewhat likely (gambler's prior)
    confidence: float = 0.0       # 0.0-1.0

    # Time tracking
    last_tested: float = 0.0
    last_gtb_confirmed: float = 0.0
    last_no_gtb_confirmed: float = 0.0

    # Gear change suspicion
    suspected_gear_change: bool = False
    recent_outcomes: deque = field(default_factory=lambda: deque(maxlen=10))

    def to_dict(self) -> dict[str, Any]:
        return {
            "player": self.player_name,
            "class": self.player_class,
            "tests": self.total_tests,
            "zero_damage": self.zero_damage_count,
            "non_zero_damage": self.non_zero_damage_count,
            "gtb_probability": round(self.gtb_probability, 3),
            "confidence": round(self.confidence, 3),
            "suspected_gear_change": self.suspected_gear_change,
        }


class GtbDetector:
    """Self-learning GTB armor detector.

    Usage:
        detector = GtbDetector()

        # Before engaging:
        gtb = detector.get_gtb_probability("SirTanky", "paladin")

        # After casting a cheap spell:
        detector.record_magic_damage(
            target_name="SirTanky",
            target_class="paladin",
            damage=0,  # Observation
            spell_name="napalm_beat",
            expected_min=250,
        )
    """

    def __init__(self) -> None:
        self._lock = RLock()

        # Per-player GTB profiles
        self._profiles: dict[str, GtbPlayerProfile] = {}

        # Per-class GTB likelihood (some classes more likely to wear GTB)
        self._class_priors: dict[str, float] = defaultdict(lambda: 0.3)

        # Detection spells ranked by SP cost (for choosing cheapest test)
        # Format: (spell_name, min_level, sp_cost, expected_min_damage_per_level)
        self._detection_spells: list[tuple[str, int, int, int]] = [
            ("napalm_beat", 1, 10, 100),      # Mage cheap
            ("fire_bolt", 1, 15, 150),        # Mage cheap
            ("cold_bolt", 1, 15, 140),        # Mage cheap
            ("lightning_bolt", 1, 20, 160),   # Mage medium
            ("fire_ball", 1, 25, 200),        # Mage AoE
            ("soul_strike", 1, 20, 120),      # Acolyte cheap
        ]

        # Stats
        self._total_tests: int = 0
        self._total_gtb_detected: int = 0
        self._start_time: float = time.time()

    # ── Detection ──────────────────────────────────────────────────────

    def get_gtb_probability(
        self,
        player_name: str,
        player_class: str | None = None,
    ) -> float:
        """Get current GTB probability for a player (0.0 = no GTB, 1.0 = confirmed)."""
        with self._lock:
            profile = self._profiles.get(player_name.lower())
            if profile is None:
                # Return class prior if we have one
                if player_class:
                    return self._class_priors.get(player_class.lower(), 0.3)
                return 0.3  # Default prior
            return profile.gtb_probability

    def is_likely_gtb(
        self,
        player_name: str,
        player_class: str | None = None,
    ) -> bool:
        """Check if target likely has GTB (probability > 60%)."""
        return self.get_gtb_probability(player_name, player_class) > 0.6

    def recommend_spell(
        self,
        available_spells: list[str],
    ) -> tuple[str, int] | None:
        """Recommend the cheapest detection spell available.

        Returns (spell_name, level) or None if no detection spell available.
        """
        # Sort by SP cost ascending
        for spell_name, min_level, sp_cost, _ in sorted(
            self._detection_spells, key=lambda x: x[2]
        ):
            if spell_name in available_spells:
                return (spell_name, min_level)
        return None

    def should_test(self, player_name: str, confidence_threshold: float = 0.7) -> bool:
        """Check whether we should test this player for GTB.

        Skip testing if we're already confident enough.
        """
        prob = self.get_gtb_probability(player_name)
        confidence = 0.0
        profile = self._profiles.get(player_name.lower())
        if profile:
            confidence = profile.confidence
        # Test if not confident in either direction
        if not profile:
            return True
        return confidence < confidence_threshold or (
            0.2 < prob < 0.8 and profile.total_tests < 5
        )

    # ── Learning from observations ─────────────────────────────────────

    def record_magic_damage(
        self,
        target_name: str,
        target_class: str,
        damage: int,
        spell_name: str = "unknown",
        spell_level: int = 1,
        expected_min_damage: int = 100,
    ) -> None:
        """Record a magic damage observation against a player.

        Args:
            target_name: Player name
            target_class: Job/class
            damage: Observed damage (0 = no damage)
            spell_name: Which spell was used for testing
            spell_level: Level of the spell used
            expected_min_damage: Minimum damage expected against non-GTB target
        """
        with self._lock:
            player_key = target_name.lower()

            # Create or get profile
            profile = self._profiles.get(player_key)
            if profile is None:
                profile = GtbPlayerProfile(
                    player_name=target_name,
                    player_class=target_class,
                )
                self._profiles[player_key] = profile

            # Build observation
            is_zero = damage <= 0
            obs = DamageObservation(
                timestamp=time.time(),
                damage=max(0, damage),
                spell_name=spell_name,
                spell_level=spell_level,
                expected_min_damage=expected_min_damage,
                is_zero_damage=is_zero,
            )

            profile.observations.append(obs)
            profile.total_tests += 1
            profile.last_tested = time.time()
            self._total_tests += 1

            if is_zero:
                profile.zero_damage_count += 1
                profile.recent_outcomes.append(1)  # 1 = GTB-like
                profile.last_gtb_confirmed = time.time()
            else:
                profile.non_zero_damage_count += 1
                profile.recent_outcomes.append(0)  # 0 = non-GTB
                profile.nonzero_min_damage = min(
                    profile.nonzero_min_damage, max(0, damage)
                )
                profile.last_no_gtb_confirmed = time.time()

            # ── Update probability ──
            self._recalculate_probability(profile)

    def record_gear_change_suspicion(self, player_name: str) -> None:
        """Flag that a player may have changed their gear/cards.

        This triggers a re-test because GTB status may have changed.
        """
        with self._lock:
            profile = self._profiles.get(player_name.lower())
            if profile:
                profile.suspected_gear_change = True
                # Reduce confidence to trigger re-test
                profile.confidence *= 0.3
                # Keep some observations but weight them less
                logger.info(
                    "[GTB] %s may have changed gear — reducing GTB confidence for re-test",
                    player_name,
                )

    def _recalculate_probability(self, profile: GtbPlayerProfile) -> None:
        """Recalculate GTB probability from all observations.

        Uses a Bayesian-like approach:
        - P(GTB|obs) = P(obs|GTB) * P(GTB) / P(obs)
        - GTB: exactly 0 damage on magic
        - Non-GTB: very unlikely to hit exactly 0 (but possible with extreme MDEF)

        Simplified as: ratio of zero-damage observations, weighted by recency.
        """
        if profile.total_tests == 0:
            return

        # ── Empirical rate ──
        recent = list(profile.recent_outcomes)
        if recent:
            recent_gtb_rate = sum(recent) / len(recent)
        else:
            recent_gtb_rate = 0.0

        overall_gtb_rate = profile.zero_damage_count / profile.total_tests

        # ── Consider nonzero minimum damage ──
        # If we've seen non-zero damage, GTB becomes very unlikely
        if profile.non_zero_damage_count >= 2 and profile.nonzero_min_damage > 50:
            # Multiple non-zero hits > 50 = definitely no GTB
            profile.gtb_probability = 0.05
            profile.confidence = min(1.0, profile.total_tests / MIN_TESTS_FOR_CONFIDENCE)
            profile.suspected_gear_change = False
            return

        if profile.non_zero_damage_count >= 1 and profile.nonzero_min_damage > 200:
            # One solid non-zero hit = almost certainly no GTB
            # (GTB makes ALL magic 0, not just some)
            profile.gtb_probability = 0.1
            profile.confidence = min(1.0, profile.total_tests / MIN_TESTS_FOR_CONFIDENCE)
            profile.suspected_gear_change = False
            return

        # ── Blend recent and overall ──
        if profile.total_tests >= MIN_TESTS_FOR_CONFIDENCE:
            blend = 0.7 * recent_gtb_rate + 0.3 * overall_gtb_rate
            # GTB is impossible if we've seen clearly non-zero damage
            if profile.non_zero_damage_count > 0:
                # Still possible if target switched to GTB after test
                # But much less likely
                time_since_nonzero = time.time() - profile.last_no_gtb_confirmed
                if time_since_nonzero > 300:  # 5 minutes since last non-zero hit
                    # Could have switched — keep some probability
                    blend = blend * 0.5
                else:
                    # Recent non-zero damage = GTB unlikely
                    blend *= 0.3

            profile.gtb_probability = blend
            profile.confidence = min(1.0, profile.total_tests / 5.0)
        else:
            # Few observations — blend with class prior
            class_prior = self._class_priors.get(profile.player_class.lower(), 0.3)
            profile.gtb_probability = (
                0.6 * recent_gtb_rate + 0.4 * class_prior
            )
            profile.confidence = min(0.5, profile.total_tests / MIN_TESTS_FOR_CONFIDENCE)

        # ── Gear change adjustment ──
        if profile.suspected_gear_change and profile.total_tests >= 2:
            # We have new data after gear change — update
            if profile.zero_damage_count / profile.total_tests > 0.6:
                profile.gtb_probability = max(profile.gtb_probability, 0.7)

    # ── Recommendation engine ──────────────────────────────────────────

    def get_engagement_advice(
        self,
        player_name: str,
        player_class: str | None = None,
    ) -> dict[str, Any]:
        """Get engagement advice: use magic or physical?

        Returns:
            dict with recommendation and evidence.
        """
        prob = self.get_gtb_probability(player_name, player_class)

        if prob > 0.6:
            return {
                "use_magic": False,
                "use_physical": True,
                "gtb_probability": round(prob, 3),
                "reason": "Target likely has GTB — magic will deal 0 damage use physical attacks",
                "confidence": self._get_profile_confidence(player_name),
            }
        elif prob < 0.2:
            return {
                "use_magic": True,
                "use_physical": False,
                "gtb_probability": round(prob, 3),
                "reason": "Target unlikely to have GTB — magic attacks are viable",
                "confidence": self._get_profile_confidence(player_name),
            }
        else:
            return {
                "use_magic": True,
                "use_physical": False,
                "gtb_probability": round(prob, 3),
                "reason": f"GTB status uncertain (prob={prob:.0%}) — default to magic, re-test when possible",
                "confidence": self._get_profile_confidence(player_name),
                "re_test_suggested": True,
            }

    def _get_profile_confidence(self, player_name: str) -> float:
        profile = self._profiles.get(player_name.lower())
        return profile.confidence if profile else 0.0

    # ── Query / introspection ──────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Summary statistics for introspection/monitoring."""
        with self._lock:
            return {
                "total_tests": self._total_tests,
                "total_gtb_detected": self._total_gtb_detected,
                "players_tracked": len(self._profiles),
                "profiles": {
                    name: p.to_dict()
                    for name, p in sorted(
                        self._profiles.items(),
                        key=lambda x: x[1].total_tests,
                        reverse=True,
                    )[:20]
                },
            }
