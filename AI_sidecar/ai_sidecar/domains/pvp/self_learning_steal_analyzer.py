"""Self-Learning Steal-Before-Kill — learns which monsters are worth
stealing from based on historical success rates and item values.

RO mechanic: Thief/Stalker can use Steal skill to get items from monsters
without killing them. Some monsters have valuable steal-only loot.

Self-* properties:
  - Self-learning: tracks steal success rate per monster from actual attempts
  - Self-optimizing: prioritizes high-value monsters with good steal rates
  - Self-adapting: adjusts to server rates (some servers have modified steal rates)
  - Self-configuring: builds steal priority list from scratch through experience
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

DECAY_ALPHA: float = 0.15
WINDOW_SIZE: int = 30

# Steal is an on-kill attempt with ~40% chance on most monsters pre-renewal
# Chance is affected by DEX difference, monster level, and card bonuses


@dataclass
class StealRecord:
    """Learned steal record for one monster."""
    monster_name: str
    monster_level: int = 0

    # Attempts
    attempts: int = 0
    successes: int = 0
    success_rate: float = 0.4  # Start at ~40% (pre-renewal baseline)

    # Items stolen (name -> count)
    items_stolen: dict[str, int] = field(default_factory=dict)

    # Estimated total value of stolen items (zeny)
    total_value_stolen: int = 0

    # Value metrics
    average_kill_time: float = 0.0  # Seconds to kill without stealing
    value_per_attempt: float = 0.0
    steal_time_cost: float = 0.0  # Additional time spent stealing vs killing

    # Recent outcomes for recency-weighted calculation
    recent_outcomes: deque = field(default_factory=lambda: deque(maxlen=WINDOW_SIZE))

    # Confidence
    confidence: float = 0.0

    # Flags
    is_worth_stealing: bool = False
    steal_score: float = 0.0  # Composite score for prioritization

    last_attempt: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "monster": self.monster_name,
            "attempts": self.attempts,
            "successes": self.successes,
            "success_rate": round(self.success_rate, 3),
            "items_stolen": dict(self.items_stolen),
            "total_value": self.total_value_stolen,
            "value_per_attempt": round(self.value_per_attempt, 1),
            "worth_stealing": self.is_worth_stealing,
            "steal_score": round(self.steal_score, 3),
            "confidence": round(self.confidence, 3),
        }


@dataclass
class StealRecommendation:
    """Recommendation for steal-before-kill on a specific monster."""
    monster_name: str
    should_steal: bool
    expected_value: float
    success_probability: float
    confidence: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "monster": self.monster_name,
            "should_steal": self.should_steal,
            "expected_value": round(self.expected_value, 1),
            "success_probability": round(self.success_probability, 3),
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
        }


class SelfLearningStealAnalyzer:
    """Learns optimal steal-before-kill targets from experience.

    Usage:
        stealer = SelfLearningStealAnalyzer()

        # Before attacking:
        rec = stealer.evaluate_monster("Poring", level=5)
        if rec.should_steal:
            # Use steal skill first

        # After attempt:
        stealer.record_steal_attempt("Poring", success=True, item="Apple", value=50)

        # After kill:
        stealer.record_kill_time("Poring", kill_duration_seconds=2.5)
    """

    def __init__(self) -> None:
        self._lock = RLock()

        # Per-monster steal records
        self._records: dict[str, StealRecord] = {}

        # Value estimates for common items (learned from market data if available)
        self._item_value_estimates: dict[str, int] = {}

        # Global stats
        self._total_attempts: int = 0
        self._total_successes: int = 0
        self._total_value_stolen: int = 0
        self._start_time: float = time.time()

    # ── Prediction ──────────────────────────────────────────────────────

    def evaluate_monster(
        self,
        monster_name: str,
        monster_level: int = 0,
    ) -> StealRecommendation:
        """Evaluate whether a monster is worth stealing from.

        Args:
            monster_name: Monster name
            monster_level: Monster level (used for initial estimate)

        Returns:
            StealRecommendation with should_steal flag
        """
        with self._lock:
            rec = self._records.get(monster_name.lower())
            if rec is None:
                # No data — default: try stealing (learning phase)
                return StealRecommendation(
                    monster_name=monster_name,
                    should_steal=True,
                    expected_value=100.0,  # Conservative estimate
                    success_probability=0.4,
                    confidence=0.0,
                    reason="No data — steal to learn if this monster is valuable",
                )

            # Enough data for a recommendation
            if rec.confidence < 0.3:
                # Low confidence — recommend steal to gather data
                return StealRecommendation(
                    monster_name=monster_name,
                    should_steal=True,
                    expected_value=rec.value_per_attempt,
                    success_probability=rec.success_rate,
                    confidence=rec.confidence,
                    reason="Learning phase — gathering steal data for this monster",
                )

            return StealRecommendation(
                monster_name=monster_name,
                should_steal=rec.is_worth_stealing,
                expected_value=rec.value_per_attempt,
                success_probability=rec.success_rate,
                confidence=rec.confidence,
                reason=(
                    f"{'Worth' if rec.is_worth_stealing else 'Not worth'} stealing: "
                    f"{rec.value_per_attempt:.0f}z per attempt "
                    f"({rec.success_rate:.0%} success rate, {rec.attempts} attempts)"
                ),
            )

    def get_steal_priority_queue(
        self,
        monsters: list[dict[str, Any]],
    ) -> list[StealRecommendation]:
        """Get sorted steal priority for a list of nearby monsters.

        Args:
            monsters: List of {name, level, distance, hp} dicts

        Returns:
            Recommendations sorted by steal score descending
        """
        results: list[StealRecommendation] = []
        with self._lock:
            for m in monsters:
                name = str(m.get("name", ""))
                level = int(m.get("level", 0))
                rec = self.evaluate_monster(name, level)
                results.append(rec)

        results.sort(key=lambda r: r.expected_value * r.success_probability, reverse=True)
        return results

    # ── Learning from outcomes ─────────────────────────────────────────

    def record_steal_attempt(
        self,
        monster_name: str,
        success: bool,
        item: str | None = None,
        item_value: int = 0,
        monster_level: int = 0,
    ) -> None:
        """Record a steal attempt outcome.

        Args:
            monster_name: Monster name
            success: True if steal succeeded
            item: Item stolen (if success)
            item_value: Estimated market value of stolen item
            monster_level: Monster level (if known)
        """
        with self._lock:
            name_key = monster_name.lower()
            rec = self._records.get(name_key)
            if rec is None:
                rec = StealRecord(
                    monster_name=monster_name,
                    monster_level=monster_level,
                )
                self._records[name_key] = rec

            rec.attempts += 1
            rec.last_attempt = time.time()
            self._total_attempts += 1

            if success:
                rec.successes += 1
                rec.recent_outcomes.append(1)
                if item:
                    rec.items_stolen[item] = rec.items_stolen.get(item, 0) + 1
                rec.total_value_stolen += item_value
                self._total_successes += 1
                self._total_value_stolen += item_value

                # Update item value estimate (use highest seen as base)
                if item and item_value > 0:
                    current = self._item_value_estimates.get(item, 0)
                    if item_value > current:
                        self._item_value_estimates[item] = item_value
            else:
                rec.recent_outcomes.append(0)

            # ── Recalculate metrics ──
            self._recalculate_record(rec)

    def record_steal_value_observation(self, item_name: str, market_value: int) -> None:
        """Update estimated value of a stealable item based on market data."""
        with self._lock:
            current = self._item_value_estimates.get(item_name.lower(), 0)
            if market_value > current:
                self._item_value_estimates[item_name.lower()] = market_value
            elif current > 0:
                # Moving average
                self._item_value_estimates[item_name.lower()] = int(
                    0.7 * current + 0.3 * market_value
                )

    def record_kill_time(self, monster_name: str, kill_time_seconds: float) -> None:
        """Record how long it typically takes to kill this monster (without stealing).

        Used to calculate opportunity cost of stealing vs just killing.
        """
        with self._lock:
            rec = self._records.get(monster_name.lower())
            if rec is None:
                return

            if rec.average_kill_time == 0:
                rec.average_kill_time = kill_time_seconds
            else:
                rec.average_kill_time = (
                    DECAY_ALPHA * kill_time_seconds
                    + (1.0 - DECAY_ALPHA) * rec.average_kill_time
                )

            self._recalculate_record(rec)

    def _recalculate_record(self, rec: StealRecord) -> None:
        """Recalculate all derived metrics for a steal record."""
        rec.success_rate = rec.successes / max(rec.attempts, 1)
        rec.value_per_attempt = rec.total_value_stolen / max(rec.attempts, 1)
        rec.confidence = min(1.0, rec.attempts / 15.0)

        # ── Recent vs overall blend ──
        recent = list(rec.recent_outcomes)
        if recent:
            recent_success_rate = sum(recent) / len(recent)
            if rec.attempts >= 5:
                recent_weight = min(0.7, rec.attempts * 0.03)
            else:
                recent_weight = 0.3
            rec.success_rate = (
                recent_weight * recent_success_rate
                + (1.0 - recent_weight) * rec.success_rate
            )

        # ── Composite steal score ──
        # Higher is better:
        # - Value per attempt (higher = better)
        # - Success rate (higher = better)
        # - Penalize long kill times (stealing takes extra time)
        value_factor = min(1.0, rec.value_per_attempt / 5000.0)  # Normalize: 5k zeny = 1.0
        success_factor = rec.success_rate

        time_penalty = 1.0
        if rec.average_kill_time > 5.0:
            # Slow kills — stealing overhead is less significant
            time_penalty = 1.2
        elif rec.average_kill_time > 0 and rec.average_kill_time < 1.0:
            # Fast kills — stealing overhead is significant
            time_penalty = 0.7

        rec.steal_score = value_factor * success_factor * time_penalty
        rec.is_worth_stealing = rec.steal_score > 0.1 and rec.success_rate > 0.15

    # ── Query / introspection ──────────────────────────────────────────

    def get_top_steal_targets(self, n: int = 10) -> list[StealRecord]:
        """Get top N steal targets sorted by steal score."""
        with self._lock:
            sorted_records = sorted(
                self._records.values(),
                key=lambda r: r.steal_score,
                reverse=True,
            )
            return sorted_records[:n]

    def get_stats(self) -> dict[str, Any]:
        """Summary statistics for introspection."""
        with self._lock:
            return {
                "total_attempts": self._total_attempts,
                "total_successes": self._total_successes,
                "overall_success_rate": round(
                    self._total_successes / max(self._total_attempts, 1), 3
                ),
                "total_value_stolen": self._total_value_stolen,
                "monsters_tracked": len(self._records),
                "estimated_item_values": dict(self._item_value_estimates),
                "top_targets": [
                    r.to_dict() for r in self.get_top_steal_targets(10)
                ],
            }
