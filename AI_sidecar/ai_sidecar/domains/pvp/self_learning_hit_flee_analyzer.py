"""Self-Learning Hit/Flee Analyzer — learns actual hit/miss rates from combat experience.

Pre-renewal RO formula:
  HIT  = 175 + base_level + DEX + bonuses
  FLEE = 100 + base_level + AGI + bonuses

If attacker HIT < target FLEE: 95% physical evasion chance.
This module learns calibration offsets per class/server by observing actual
hit/miss events, then self-corrects the formula estimate over time.

Self-* properties:
  - Self-learning: updates hit_rate_estimates from every observed attack outcome
  - Self-optimizing: adjusts confidence intervals, narrows as data grows
  - Self-adapting: learns per-class and per-player calibration separately
  - Self-healing: if hit rate drops too low, signals that target has hidden
    flee bonuses (e.g. chicken hat, agi up, items)
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

# ── Default skill / constant lookup (pre-renewal) ─────────────────────

DEFAULT_PRE_RENEWAL_HIT_BASE: int = 175
DEFAULT_PRE_RENEWAL_FLEE_BASE: int = 100

MIN_OBSERVATIONS_BEFORE_CALIBRATE: int = 5
RECENT_WINDOW: int = 50  # recent hits/misses to track
DECAY_ALPHA: float = 0.15  # exponential decay weight (lower = smoother)


@dataclass
class HitFleeEstimate:
    """Learned estimate for one target's hit/flee relationship.

    Stores both the calculated formula estimate and the empirically
    observed hit rate so we can compare and calibrate.
    """
    # ── Formula-derived (input) ──
    estimated_hit: float = 0.0      # Our calculated HIT
    estimated_flee: float = 0.0     # Our calculated FLEE for target
    hit_over_flee: float = 0.0      # HIT - FLEE (positive = good)

    # ── Empirically observed ──
    total_attempts: int = 0
    total_hits: int = 0
    hit_rate: float = 0.0           # empirical hit rate 0.0-1.0
    hit_rate_samples: int = 0

    # ── Calibration ──
    calibration_offset: float = 0.0  # Correction to formula (negative = targets evade more than formula predicts)
    flee_bonus_estimate: float = 0.0 # Estimated extra flee (gear / buffs not in formula)

    # ── Decision ──
    use_magic: bool = False          # True if physical is hopeless (effective hit rate < 20%)
    confidence: float = 0.0          # 0.0-1.0 how much we trust this estimate

    # ── Recent history for pattern detection ──
    recent_hits: deque = field(default_factory=lambda: deque(maxlen=RECENT_WINDOW))

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimated_hit": round(self.estimated_hit, 1),
            "estimated_flee": round(self.estimated_flee, 1),
            "hit_over_flee": round(self.hit_over_flee, 1),
            "empirical_hit_rate": round(self.hit_rate, 3),
            "calibration_offset": round(self.calibration_offset, 2),
            "flee_bonus_estimate": round(self.flee_bonus_estimate, 2),
            "use_magic": self.use_magic,
            "confidence": round(self.confidence, 3),
            "samples": self.total_attempts,
        }


@dataclass
class ClassCalibration:
    """Per-class calibration model learned from observed combat.

    Some classes have innate flee bonuses (assassin, rogue) or
    heavy gear that reduces flee (knights in full plate).
    """
    class_name: str
    observations: int = 0
    avg_calibration_offset: float = 0.0    # Average offset for this class
    avg_flee_bonus: float = 0.0            # Average extra flee for this class
    hit_rate_std: float = 0.0              # Variability in hit rate
    recent_offsets: deque = field(default_factory=lambda: deque(maxlen=30))
    last_updated: float = 0.0


class SelfLearningHitFleeAnalyzer:
    """Learns hit/flee dynamics from actual combat outcomes.

    Usage:
        analyzer = SelfLearningHitFleeAnalyzer()
        estimate = analyzer.analyze(my_stats, target_stats)
        if estimate.use_magic:
            # Switch to magic attacks — physical won't land

        # Later, when we actually attack:
        analyzer.record_attack_outcome(target_name, target_class, hit=True)
    """

    def __init__(self) -> None:
        self._lock = RLock()

        # Per-player estimates {player_name: HitFleeEstimate}
        self._player_estimates: dict[str, HitFleeEstimate] = {}

        # Per-class calibration models
        self._class_calibrations: dict[str, ClassCalibration] = {}

        # Global calibration offset (server-wide)
        self._global_calibration_offset: float = 0.0
        self._global_observations: int = 0
        self._global_recent_rates: deque = deque(maxlen=100)

        # Metadata
        self._total_observations: int = 0
        self._start_time: float = time.time()

    # ── Core analysis ────────────────────────────────────────────────────

    def analyze(
        self,
        my_stats: dict[str, Any],
        target_stats: dict[str, Any],
        player_name: str | None = None,
    ) -> HitFleeEstimate:
        """Compute hit/flee analysis for a potential engagement.

        Uses formula first, then layers on learned calibration.

        Args:
            my_stats: dict with 'base_level', 'dex', 'hit_bonus'
            target_stats: dict with 'base_level', 'agi', 'flee_bonus', 'class'
            player_name: if known, loads learned model for this player

        Returns:
            HitFleeEstimate with use_magic recommendation
        """
        my_level = int(my_stats.get("base_level", 1))
        my_dex = int(my_stats.get("dex", 1))
        my_hit_bonus = int(my_stats.get("hit_bonus", 0))

        tgt_level = int(target_stats.get("base_level", 1))
        tgt_agi = int(target_stats.get("agi", 1))
        tgt_flee_bonus = int(target_stats.get("flee_bonus", 0))
        tgt_class = str(target_stats.get("class", target_stats.get("job_name", "unknown")))

        # ── Formula ──
        calculated_hit = DEFAULT_PRE_RENEWAL_HIT_BASE + my_level + my_dex + my_hit_bonus
        calculated_flee = DEFAULT_PRE_RENEWAL_FLEE_BASE + tgt_level + tgt_agi + tgt_flee_bonus

        # ── Start with default ──
        est = HitFleeEstimate(
            estimated_hit=float(calculated_hit),
            estimated_flee=float(calculated_flee),
            hit_over_flee=float(calculated_hit - calculated_flee),
        )

        # ── Apply class calibration ──
        class_cal = self._class_calibrations.get(tgt_class.lower())
        if class_cal and class_cal.observations >= MIN_OBSERVATIONS_BEFORE_CALIBRATE:
            est.calibration_offset += class_cal.avg_calibration_offset
            est.flee_bonus_estimate += class_cal.avg_flee_bonus
            est.calibration_offset += self._global_calibration_offset
        else:
            # Global calibration as fallback
            if self._global_observations >= MIN_OBSERVATIONS_BEFORE_CALIBRATE:
                est.calibration_offset += self._global_calibration_offset

        # ── Apply per-player calibration if known ──
        if player_name:
            player_est = self._player_estimates.get(player_name)
            if player_est and player_est.total_attempts >= MIN_OBSERVATIONS_BEFORE_CALIBRATE:
                # Blend player-specific with formula
                est.calibration_offset = (
                    0.7 * player_est.calibration_offset
                    + 0.3 * est.calibration_offset
                )
                est.flee_bonus_estimate = (
                    0.7 * player_est.flee_bonus_estimate
                    + 0.3 * est.flee_bonus_estimate
                )
                est.hit_rate = player_est.hit_rate
                est.hit_rate_samples = player_est.total_attempts
                est.total_attempts = player_est.total_attempts
                est.total_hits = player_est.total_hits

        # ── Effective hit rate estimate ──
        effective_flee = calculated_flee + est.flee_bonus_estimate
        # With calibration offset, our effective hit is adjusted
        effective_hit = calculated_hit + est.calibration_offset
        hit_delta = effective_hit - effective_flee

        # Pre-renewal: if HIT >= FLEE, 95% hit rate
        # If HIT < FLEE: hit_rate = 0.05 (95% miss)
        if hit_delta >= 0:
            base_hit_rate = 0.95
        else:
            # Each point of flee above hit reduces hit rate
            # Roughly: 95% - (flee - hit) * 0.5% per point
            miss_penalty = abs(hit_delta) * 0.005
            base_hit_rate = max(0.05, 0.95 - miss_penalty)

        # Blend empirical if we have it
        if est.hit_rate_samples >= MIN_OBSERVATIONS_BEFORE_CALIBRATE:
            # Trust empirical more as we get more samples
            empirical_weight = min(0.8, est.hit_rate_samples * 0.02)
            effective_hit_rate = (
                empirical_weight * est.hit_rate
                + (1.0 - empirical_weight) * base_hit_rate
            )
        else:
            effective_hit_rate = base_hit_rate

        est.hit_rate = effective_hit_rate
        est.use_magic = effective_hit_rate < 0.20
        est.confidence = min(1.0, self._total_observations / 50.0)

        return est

    # ── Learning from outcomes ──────────────────────────────────────────

    def record_attack_outcome(
        self,
        target_name: str,
        target_class: str | None = None,
        hit: bool = True,
        damage: int = 0,
    ) -> None:
        """Record a physical attack outcome to improve future estimates.

        Args:
            target_name: Player name identifier
            target_class: Job/class for cross-player learning
            hit: True if attack landed, False if missed
            damage: Damage dealt (0 for miss, >0 for hit)
        """
        with self._lock:
            name_key = target_name.lower()
            class_key = (target_class or "unknown").lower()

            # ── Update per-player estimate ──
            if name_key not in self._player_estimates:
                self._player_estimates[name_key] = HitFleeEstimate()
            player_est = self._player_estimates[name_key]

            player_est.total_attempts += 1
            if hit:
                player_est.total_hits += 1
                player_est.recent_hits.append(1)
            else:
                player_est.recent_hits.append(0)

            # Recalculate hit rate
            recent_arr = list(player_est.recent_hits)
            recent_hits = sum(recent_arr)
            recent_total = len(recent_arr)

            if recent_total > 0:
                recent_rate = recent_hits / recent_total
                # Exponential moving average for smoothing
                if player_est.hit_rate_samples == 0:
                    player_est.hit_rate = recent_rate
                else:
                    player_est.hit_rate = (
                        DECAY_ALPHA * recent_rate
                        + (1.0 - DECAY_ALPHA) * player_est.hit_rate
                    )
            player_est.hit_rate_samples += 1

            # If we observed a miss and estimate predicted hit, adjust calibration
            if not hit and player_est.estimated_hit >= player_est.estimated_flee + 10:
                # Our formula was too optimistic — target has hidden flee
                player_est.calibration_offset -= 3.0  # Adjust down
                player_est.flee_bonus_estimate += 2.0
            elif hit and player_est.estimated_hit < player_est.estimated_flee - 20:
                # Our formula was too pessimistic — target has less flee than calculated
                player_est.calibration_offset += 2.0
                player_est.flee_bonus_estimate = max(0, player_est.flee_bonus_estimate - 1.0)

            # ── Update global statistics ──
            self._global_observations += 1
            self._global_recent_rates.append(1.0 if hit else 0.0)
            if len(self._global_recent_rates) >= 10:
                global_rate = sum(self._global_recent_rates) / len(self._global_recent_rates)
                if global_rate < 0.5 and self._global_observations > 10:
                    # Server-wide, things miss more than expected
                    self._global_calibration_offset = (global_rate - 0.95) * 20

            # ── Update class calibration ──
            if class_key:
                self._update_class_calibration(class_key, hit)

            self._total_observations += 1

    def _update_class_calibration(self, class_name: str, hit: bool) -> None:
        """Update per-class calibration model from an observation."""
        cal = self._class_calibrations.get(class_name)
        if cal is None:
            cal = ClassCalibration(class_name=class_name)
            self._class_calibrations[class_name] = cal

        cal.observations += 1
        # 1.0 for hit, 0.0 for miss
        offset = 0.0 if hit else -5.0  # A miss suggests ~5 more flee than estimated
        cal.recent_offsets.append(offset)

        if cal.observations >= 3:
            offsets = list(cal.recent_offsets)
            cal.avg_calibration_offset = sum(offsets) / len(offsets)
            if len(offsets) > 1:
                mean = cal.avg_calibration_offset
                variance = sum((o - mean) ** 2 for o in offsets) / len(offsets)
                cal.hit_rate_std = math.sqrt(variance)
            cal.last_updated = time.time()

    # ── Query / introspection ───────────────────────────────────────────

    def get_player_estimate(self, player_name: str) -> HitFleeEstimate | None:
        """Get the current estimate for a specific player."""
        return self._player_estimates.get(player_name.lower())

    def get_class_calibration(self, class_name: str) -> ClassCalibration | None:
        """Get the current calibration for a class."""
        return self._class_calibrations.get(class_name.lower())

    def get_stats(self) -> dict[str, Any]:
        """Return summary stats for introspection/debugging."""
        with self._lock:
            return {
                "total_observations": self._total_observations,
                "players_tracked": len(self._player_estimates),
                "classes_tracked": len(self._class_calibrations),
                "global_calibration_offset": round(self._global_calibration_offset, 2),
                "global_observations": self._global_observations,
                "uptime_seconds": round(time.time() - self._start_time, 1),
                "players": {
                    name: est.to_dict()
                    for name, est in list(self._player_estimates.items())[:20]
                },
                "classes": {
                    name: {
                        "observations": cal.observations,
                        "avg_calibration_offset": round(cal.avg_calibration_offset, 2),
                        "avg_flee_bonus": round(cal.avg_flee_bonus, 2),
                        "hit_rate_std": round(cal.hit_rate_std, 3),
                    }
                    for name, cal in self._class_calibrations.items()
                },
            }

    def recommend_approach(
        self,
        my_stats: dict[str, Any],
        target_stats: dict[str, Any],
        target_name: str | None = None,
    ) -> dict[str, Any]:
        """Get a full engagement recommendation.

        Returns:
            dict with 'use_magic', 'estimated_hit_rate', 'reason', and metadata
        """
        estimate = self.analyze(my_stats, target_stats, target_name)

        if estimate.use_magic:
            reason = (
                f"Physical hit rate only {estimate.hit_rate:.0%} — "
                f"recommend switching to magic (always hits)"
            )
        else:
            reason = (
                f"Physical hit rate {estimate.hit_rate:.0%} — "
                f"viable to engage physically"
            )

        return {
            "use_magic": estimate.use_magic,
            "use_physical": not estimate.use_magic,
            "estimated_hit_rate": round(estimate.hit_rate, 3),
            "calibration_offset": round(estimate.calibration_offset, 2),
            "confidence": round(estimate.confidence, 3),
            "reason": reason,
            "samples": estimate.total_attempts,
        }
