"""Self-Learning Class Counter System — learns effective class counter matchups
from observed combat outcomes rather than hardcoded rules.

RO class counter theory (starting point that learning replaces):
  - Alchemist > Paladin (Acid Demo bypasses defense)
  - Assassin > Wizard (cloak + approach + kill before cast)
  - Wizard > Hunter (AoE bypasses flee)
  - Priest > Undead (Heal/Turn Undead)
  - Hunter > Assassin (trap reveals cloaked unit)
  - Sage > Mage (dispell + spider web)
  - Paladin > Assassin (reflect + tank through burst)
  - Monk > everything (Asura Strike burst)

But the actual effectiveness depends on gear, build, player skill, server rates.
This module learns from real combat data and continuously updates.

Self-* properties:
  - Self-learning: learns matchup win rates from actual PVP combat outcomes
  - Self-optimizing: finds the best class to counter a given opponent
  - Self-adapting: adjusts to server-specific meta (high-rate vs low-rate)
  - Self-configuring: builds the counter matrix from scratch through experience
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

MIN_OBSERVATIONS_FOR_RELIABLE: int = 5
CONFIDENCE_SCALE: int = 20  # Observations needed for full confidence weight
DECAY_ALPHA: float = 0.15
WINDOW_SIZE: int = 30

# Base class categories
MELEE_CLASSES: frozenset[str] = frozenset({
    "swordman", "knight", "lord knight", "rune knight", "royal guard",
    "thief", "assassin", "rogue", "stalker", "guillotine cross", "shadow chaser",
    "monk", "champion", "sura",
    "alchemist", "creator", "genetic",
    "blacksmith", "whitesmith", "mechanics",
    "crusader", "paladin", "royal guard",
})

MAGIC_CLASSES: frozenset[str] = frozenset({
    "mage", "wizard", "high wizard", "warlock",
    "sage", "professor", "sorcerer",
    "soul linker",
    "ninja", "kagerou", "oboro",
})

RANGED_CLASSES: frozenset[str] = frozenset({
    "archer", "hunter", "sniper", "ranger",
    "bard", "dancer", "minstrel", "wanderer",
    "gunslinger", "rebellion",
})

SUPPORT_CLASSES: frozenset[str] = frozenset({
    "acolyte", "priest", "high priest", "arch bishop",
    "monk", "champion",
})


@dataclass
class MatchupRecord:
    """Record of one class vs class combat outcome."""
    my_class: str
    opponent_class: str

    # Combat record
    engagements: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.5  # Default: even

    # Recent outcomes for recency-weighted calculation
    recent_outcomes: deque = field(default_factory=lambda: deque(maxlen=WINDOW_SIZE))

    # Kill/death details
    total_damage_dealt: int = 0
    total_damage_taken: int = 0
    avg_engagement_duration: float = 0.0  # seconds
    last_encounter_time: float = 0.0

    # Confidence
    confidence: float = 0.0

    # Trend detection
    win_rate_trend: float = 0.0  # Positive = improving, negative = declining

    def is_reliable(self) -> bool:
        return self.engagements >= MIN_OBSERVATIONS_FOR_RELIABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "matchup": f"{self.my_class} vs {self.opponent_class}",
            "engagements": self.engagements,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 3),
            "confidence": round(self.confidence, 3),
            "trend": round(self.win_rate_trend, 3),
            "reliable": self.is_reliable(),
        }


@dataclass
class CounterRecommendation:
    """A recommended counter for a specific opponent class."""
    my_class: str
    opponent_class: str
    predicted_win_rate: float
    confidence: float
    is_reliable: bool
    reason: str
    current_kdr: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "my_class": self.my_class,
            "vs": self.opponent_class,
            "predicted_win_rate": round(self.predicted_win_rate, 3),
            "confidence": round(self.confidence, 3),
            "reliable": self.is_reliable,
            "reason": self.reason,
        }


class SelfLearningClassCounters:
    """Learns effective class counter matchups from combat experience.

    Usage:
        counters = SelfLearningClassCounters()

        # Before a fight, check best class to use:
        best = counters.get_best_counter("wizard")
        # Returns CounterRecommendation for each available class

        # After a fight:
        counters.record_combat_outcome(
            my_class="rogue",
            opponent_class="wizard",
            won=True,
            damage_dealt=4500,
            damage_taken=2000,
            duration=12.5,
        )

        # Query win rate:
        wr = counters.get_win_rate("rogue", "wizard")
    """

    def __init__(self) -> None:
        self._lock = RLock()

        # Matchup matrix: {my_class: {opp_class: MatchupRecord}}
        self._matrix: dict[str, dict[str, MatchupRecord]] = defaultdict(dict)

        # Per-player records for personalized counter tracking
        # {player_name: {my_class: {opp_class: MatchupRecord}}}
        self._player_records: dict[str, dict[str, dict[str, MatchupRecord]]] = defaultdict(
            lambda: defaultdict(dict)
        )

        # Starting priors (not hardcoded — these decay as observations accumulate)
        # Represented as initial "virtual engagements" for Bayesian smoothing
        self._prior_strength: int = 2  # Virtual engagements weight

        # Stats
        self._total_engagements: int = 0
        self._total_wins: int = 0
        self._start_time: float = time.time()

    # ── Prediction ──────────────────────────────────────────────────────

    def get_win_rate(
        self,
        my_class: str,
        opponent_class: str,
    ) -> float:
        """Get current predicted win rate for this matchup.

        Returns 0.0-1.0.
        """
        my_key = my_class.lower()
        opp_key = opponent_class.lower()

        with self._lock:
            matchup = self._matrix.get(my_key, {}).get(opp_key)
            if matchup and matchup.is_reliable():
                return matchup.win_rate

            # Fall back to heuristic Bayesian prior
            return self._bayesian_prior(my_key, opp_key)

    def get_best_counter(
        self,
        opponent_class: str,
        available_classes: list[str] | None = None,
    ) -> CounterRecommendation:
        """Find the best counter class for a given opponent.

        Args:
            opponent_class: The opponent's class to counter
            available_classes: List of classes we can use. If None, all classes checked.

        Returns:
            CounterRecommendation with the best predicted matchup
        """
        opp_key = opponent_class.lower()
        best_win_rate = 0.0
        best_class = None
        best_record = None

        with self._lock:
            for my_class_key, matchups in self._matrix.items():
                if available_classes and my_class_key not in {
                    c.lower() for c in available_classes
                }:
                    continue

                matchup = matchups.get(opp_key)
                if matchup is None:
                    # Estimate from prior
                    wr = self._bayesian_prior(my_class_key, opp_key)
                    if wr > best_win_rate:
                        best_win_rate = wr
                        best_class = my_class_key
                        best_record = None
                elif matchup.is_reliable() and matchup.win_rate > best_win_rate:
                    best_win_rate = matchup.win_rate
                    best_class = my_class_key
                    best_record = matchup

        # Build reason
        if best_record:
            reason = (
                f"{best_class} counters {opponent_class} with "
                f"{best_record.win_rate:.0%} win rate over {best_record.engagements} engagements"
            )
            kdr = best_record.wins / max(best_record.losses, 1)
            return CounterRecommendation(
                my_class=best_class or "unknown",
                opponent_class=opponent_class,
                predicted_win_rate=best_win_rate,
                confidence=best_record.confidence if best_record else 0.0,
                is_reliable=best_record.is_reliable() if best_record else False,
                reason=reason,
                current_kdr=kdr,
            )
        else:
            reason = f"No data on countering {opponent_class} — {best_class or 'unknown'} has best estimated win rate ({best_win_rate:.0%})"
            return CounterRecommendation(
                my_class=best_class or "unknown",
                opponent_class=opponent_class,
                predicted_win_rate=best_win_rate,
                confidence=0.0,
                is_reliable=False,
                reason=reason,
                current_kdr=0.0,
            )

    def get_all_counters(
        self,
        opponent_class: str,
        min_observations: int = MIN_OBSERVATIONS_FOR_RELIABLE,
    ) -> list[CounterRecommendation]:
        """Get all classes that counter the opponent, sorted by win rate.

        Args:
            opponent_class: The opponent's class
            min_observations: Minimum engagements to consider

        Returns:
            List of CounterRecommendation sorted by predicted_win_rate descending
        """
        opp_key = opponent_class.lower()
        results: list[CounterRecommendation] = []

        with self._lock:
            for my_class_key, matchups in self._matrix.items():
                matchup = matchups.get(opp_key)
                if matchup and matchup.engagements >= min_observations:
                    kdr = matchup.wins / max(matchup.losses, 1)
                    result = CounterRecommendation(
                        my_class=my_class_key,
                        opponent_class=opponent_class,
                        predicted_win_rate=matchup.win_rate,
                        confidence=matchup.confidence,
                        is_reliable=matchup.is_reliable(),
                        reason=(
                            f"{matchup.wins}W/{matchup.losses}L "
                            f"({matchup.engagements} engagements)"
                        ),
                        current_kdr=kdr,
                    )
                    results.append(result)

        results.sort(key=lambda r: r.predicted_win_rate, reverse=True)
        return results

    # ── Learning from outcomes ─────────────────────────────────────────

    def record_combat_outcome(
        self,
        my_class: str,
        opponent_class: str,
        won: bool,
        damage_dealt: int = 0,
        damage_taken: int = 0,
        duration: float = 0.0,
        opponent_name: str | None = None,
    ) -> None:
        """Record a PVP combat outcome to improve counter predictions.

        Args:
            my_class: Your class/job
            opponent_class: Opponent's class/job
            won: True if you won the engagement
            damage_dealt: Total damage dealt to opponent
            damage_taken: Total damage taken from opponent
            duration: Duration of engagement in seconds
            opponent_name: Optional player name for personalized tracking
        """
        with self._lock:
            my_key = my_class.lower()
            opp_key = opponent_class.lower()

            # ── Update global matrix ──
            self._update_matchup(my_key, opp_key, won, damage_dealt, damage_taken, duration)

            # ── Update per-player tracking ──
            if opponent_name:
                player_key = opponent_name.lower()
                self._update_matchup(
                    my_key, opp_key, won,
                    damage_dealt, damage_taken, duration,
                    player_lookup=player_key,
                )

            self._total_engagements += 1
            if won:
                self._total_wins += 1

    def _update_matchup(
        self,
        my_class_key: str,
        opp_class_key: str,
        won: bool,
        damage_dealt: int,
        damage_taken: int,
        duration: float,
        player_lookup: str | None = None,
    ) -> None:
        """Update a specific matchup in the matrix or player records."""
        if player_lookup:
            records = self._player_records[player_lookup][my_class_key]
        else:
            records = self._matrix[my_class_key]

        matchup = records.get(opp_class_key)
        if matchup is None:
            matchup = MatchupRecord(
                my_class=my_class_key,
                opponent_class=opp_class_key,
            )
            records[opp_class_key] = matchup

        matchup.engagements += 1
        if won:
            matchup.wins += 1
            matchup.recent_outcomes.append(1)
        else:
            matchup.losses += 1
            matchup.recent_outcomes.append(0)

        matchup.total_damage_dealt += damage_dealt
        matchup.total_damage_taken += damage_taken
        matchup.last_encounter_time = time.time()

        # Update average duration with exponential moving average
        if matchup.engagements == 1:
            matchup.avg_engagement_duration = duration
        else:
            matchup.avg_engagement_duration = (
                DECAY_ALPHA * duration
                + (1.0 - DECAY_ALPHA) * matchup.avg_engagement_duration
            )

        # ── Recalculate win rate ──
        recent = list(matchup.recent_outcomes)
        if recent:
            recent_win_rate = sum(recent) / len(recent)
        else:
            recent_win_rate = 0.5

        overall_win_rate = matchup.wins / max(matchup.engagements, 1)

        # Blend recent and overall (weight recent more as we have more data)
        if matchup.engagements < MIN_OBSERVATIONS_FOR_RELIABLE:
            # Bayesian smoothing with prior
            prior = self._bayesian_prior(my_class_key, opp_class_key)
            virtual_wins = self._prior_strength * prior
            virtual_losses = self._prior_strength * (1.0 - prior)
            smoothed_wr = (
                (virtual_wins + matchup.wins)
                / (virtual_wins + virtual_losses + matchup.engagements)
            )
            recent_weight = 0.3
            matchup.win_rate = recent_weight * recent_win_rate + (1 - recent_weight) * smoothed_wr
            matchup.confidence = matchup.engagements / CONFIDENCE_SCALE
        else:
            # Trust data more
            recent_weight = min(0.6, matchup.engagements * 0.02)
            matchup.win_rate = recent_weight * recent_win_rate + (1 - recent_weight) * overall_win_rate
            matchup.confidence = min(1.0, matchup.engagements / CONFIDENCE_SCALE)

        # ── Track trend ──
        if matchup.engagements >= 3:
            recent_arr = list(matchup.recent_outcomes)
            half = len(recent_arr) // 2
            if half >= 1:
                recent_half = sum(recent_arr[half:]) / max(len(recent_arr[half:]), 1)
                old_half = sum(recent_arr[:half]) / max(len(recent_arr[:half]), 1)
                matchup.win_rate_trend = recent_half - old_half

    def _bayesian_prior(self, my_class: str, opponent_class: str) -> float:
        """Heuristic prior for class matchup.

        These are soft priors that get overwhelmed by real data.
        Starting point that learning replaces.
        """
        my = my_class.lower()
        opp = opponent_class.lower()

        # Check class categories for basic heuristics
        my_is_melee = any(m in my for m in ["assassin", "rogue", "stalker", "swordman", "knight", "monk", "champion"])

        # Assassin-like classes counter wizards (cloak + approach)
        if my in ("assassin", "rogue", "stalker", "guillotine cross", "shadow chaser") and opp in ("mage", "wizard", "high wizard", "warlock"):
            return 0.65

        # Wizards counter archers/hunters (AoE bypasses flee)
        if my in ("wizard", "high wizard", "warlock", "sorcerer") and opp in ("archer", "hunter", "sniper", "ranger"):
            return 0.60

        # Hunters counter assassins (trap reveals cloaked)
        if my in ("hunter", "sniper", "ranger") and opp in ("assassin", "rogue", "stalker", "guillotine cross"):
            return 0.60

        # Alchemist counters paladins (acid demo bypasses def)
        if my in ("alchemist", "creator", "genetic") and opp in ("paladin", "crusader", "royal guard"):
            return 0.65

        # Paladins counter assassins (reflect + tanky)
        if my in ("paladin", "crusader", "royal guard") and opp in ("assassin", "rogue", "stalker"):
            return 0.55

        # Monks counter everything with Asura (but slow)
        if my in ("monk", "champion", "sura") and opp not in ("hunter", "sniper", "ranger"):
            return 0.55

        # Default: slight advantage to melee vs magic (gap-close), slight disadvantage vs ranged
        if my_is_melee:
            if opp in MAGIC_CLASSES:
                return 0.55
            if opp in RANGED_CLASSES:
                return 0.45
        else:
            if my in MAGIC_CLASSES and opp in RANGED_CLASSES:
                return 0.55

        return 0.50  # Even

    # ── Query / introspection ──────────────────────────────────────────

    def get_best_class_against(
        self,
        target_class: str,
        our_class: str,
    ) -> float:
        """Get the predicted advantage score for our class vs a target class.

        Alias for get_win_rate() — returns 0.0-1.0 where >0.5 means
        our class has an advantage against the target.

        Args:
            target_class: The opponent's class
            our_class: Our class

        Returns:
            Advantage score (0.0-1.0), >0.5 = we have advantage
        """
        return self.get_win_rate(our_class, target_class)

    def get_all_matchups(self) -> list[MatchupRecord]:
        """Get all recorded matchups with sufficient data."""
        results: list[MatchupRecord] = []
        with self._lock:
            for my_class, matchups in self._matrix.items():
                for opp_class, record in matchups.items():
                    results.append(record)
        return results

    def get_stats(self) -> dict[str, Any]:
        """Summary statistics for introspection."""
        with self._lock:
            class_count = len(self._matrix)
            matchup_count = sum(
                len(matchups) for matchups in self._matrix.values()
            )
            reliable_matchups = sum(
                1 for m in self.get_all_matchups() if m.is_reliable()
            )
            overall_wr = (
                self._total_wins / max(self._total_engagements, 1)
                if self._total_engagements > 0 else 0.5
            )
            return {
                "total_engagements": self._total_engagements,
                "total_wins": self._total_wins,
                "overall_win_rate": round(overall_wr, 3),
                "classes_tracked": class_count,
                "total_matchups": matchup_count,
                "reliable_matchups": reliable_matchups,
                "player_records": sum(
                    len(classes) for players in self._player_records.values()
                    for classes in players.values()
                ),
                "top_matchups": [
                    m.to_dict() for m in sorted(
                        self.get_all_matchups(),
                        key=lambda x: x.engagements,
                        reverse=True,
                    )[:10]
                ],
            }
