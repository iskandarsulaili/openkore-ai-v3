"""Self-Learning Status Resistance Tracker — learns which status effects
actually land or fail against each class/build, then predicts resistance.

RO mechanic reference:
  - 100 VIT = stun immunity (exact threshold depends on server)
  - Status resistance is affected by VIT, INT (for some statuses),
    class innate bonuses, and equipment cards (Marc card = freeze immunity,
    Nightmare card = sleep immunity, etc.)

Self-* properties:
  - Self-learning: tracks every status attempt and outcome, builds per-class model
  - Self-optimizing: adjusts expected resistance as sample size grows
  - Self-adapting: detects when a player switches gear (gains/removes immunity)
  - Self-healing: if a normally-reliable spell keeps failing, flags possible
    card/gear change and updates model
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

# ── Status effect categories ─────────────────────────────────────────

# Status effects sorted by which stat or mechanic grants immunity / resistance
VIT_STATUSES: frozenset[str] = frozenset({
    "stun", "poison", "bleeding",
})
INT_STATUSES: frozenset[str] = frozenset({
    "silence", "confusion", "curse", "blind",
})
AGI_STATUSES: frozenset[str] = frozenset({
    "freeze",  # AGI affects freeze duration — AGI reduces frozen time
})
CARD_IMMUNE_STATUSES: dict[str, list[str]] = {
    "freeze": ["marc_card"],
    "sleep": ["nightmare_card"],
    "stun": ["orc_lord_card", "bloody_knight_card"],
    "silence": ["mummy_card"],
    "blind": ["sunglasses_blue", "skel_worker_card"],
    "stone_curse": ["rafflesia_card"],
    "poison": ["poisonous_ghost_card"],
}

# Default effect durations (seconds) — used when actual duration unknown
DEFAULT_STATUS_DURATIONS: dict[str, float] = {
    "stun": 3.0,
    "freeze": 5.0,
    "sleep": 7.0,
    "silence": 5.0,
    "confusion": 4.0,
    "curse": 10.0,
    "blind": 8.0,
    "poison": 15.0,
    "stone_curse": 6.0,
    "bleeding": 10.0,
}

MIN_OBSERVATIONS: int = 3
DECAY_ALPHA: float = 0.1


@dataclass
class StatusResistanceModel:
    """Learned resistance model for one status on one class/build."""
    status: str
    class_name: str

    # Attempt tracking
    attempts: int = 0
    successes: int = 0  # How many times the status LANDED
    failures: int = 0   # How many times it was resisted

    # Learned values
    resistance_probability: float = 0.5  # 0.0 = immune, 1.0 = always lands
    confidence: float = 0.0              # How reliable this estimate is
    last_attempt_time: float = 0.0

    # Recent outcomes for pattern detection
    recent_outcomes: deque = field(default_factory=lambda: deque(maxlen=20))

    # Gear-change detection
    gear_change_detected: bool = False
    consecutive_unexpected: int = 0  # Outcomes opposite to prediction

    def land_probability(self) -> float:
        """Probability that this status will land (1.0 = certain)."""
        return 1.0 - self.resistance_probability

    def is_reliable(self) -> bool:
        """Whether we have enough data to trust this model."""
        return self.attempts >= MIN_OBSERVATIONS and self.confidence > 0.3

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "class": self.class_name,
            "attempts": self.attempts,
            "successes": self.successes,
            "failures": self.failures,
            "resistance_pct": round(self.resistance_probability * 100, 1),
            "land_pct": round((1.0 - self.resistance_probability) * 100, 1),
            "confidence": round(self.confidence, 3),
            "reliable": self.is_reliable(),
            "gear_change_detected": self.gear_change_detected,
        }


@dataclass
class PlayerStatusProfile:
    """Holistic status resistance profile for a specific player."""
    player_name: str
    class_name: str

    # Known immunities per status
    immunities: set[str] = field(default_factory=set)
    partial_resistances: dict[str, float] = field(default_factory=dict)

    # Per-status models
    models: dict[str, StatusResistanceModel] = field(default_factory=dict)

    # Last-seen gear snapshot (if available)
    last_gear_hash: str | None = None

    # Observation count
    total_observations: int = 0
    last_updated: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "player": self.player_name,
            "class": self.class_name,
            "immunities": list(self.immunities),
            "partial_resistances": {k: f"{v:.0%}" for k, v in self.partial_resistances.items()},
            "total_observations": self.total_observations,
            "models": {s: m.to_dict() for s, m in self.models.items()},
        }


class SelfLearningStatusResistanceTracker:
    """Learns status resistance models from observed combat outcomes.

    Usage:
        tracker = SelfLearningStatusResistanceTracker()
        can_stun = tracker.predict_lands("stun", "paladin", "SirTanky")
        if can_stun:
            # Use stun spell/attack

        # After casting:
        tracker.record_status_attempt(
            status="stun", target="SirTanky",
            target_class="paladin", landed=True
        )
    """

    def __init__(self) -> None:
        self._lock = RLock()

        # Per-class + per-status models {class_key: {status: StatusResistanceModel}}
        self._class_resistance: dict[str, dict[str, StatusResistanceModel]] = defaultdict(dict)

        # Per-player profiles {player_key: PlayerStatusProfile}
        self._player_profiles: dict[str, PlayerStatusProfile] = {}

        # Global stats
        self._total_attempts: int = 0
        self._gear_change_alerts: int = 0
        self._start_time: float = time.time()

    # ── Prediction ──────────────────────────────────────────────────────

    def predict_lands(
        self,
        status: str,
        target_class: str,
        target_name: str | None = None,
    ) -> float:
        """Predict probability that *status* will land on this target.

        Returns 0.0-1.0 probability of successful application.
        """
        class_key = target_class.lower()
        status_key = status.lower()
        probability = 0.8  # Default optimism for untracked combos

        with self._lock:
            # 1. Check per-player first
            if target_name:
                player_key = target_name.lower()
                profile = self._player_profiles.get(player_key)
                if profile:
                    # If they have immunity, 0%
                    if status_key in profile.immunities:
                        return 0.0
                    if status_key in profile.partial_resistances:
                        return 1.0 - profile.partial_resistances[status_key]
                    # Check per-player model
                    model = profile.models.get(status_key)
                    if model and model.is_reliable():
                        probability = model.land_probability()

            # 2. Class-wide model as supplement
            class_models = self._class_resistance.get(class_key, {})
            class_model = class_models.get(status_key)
            if class_model and class_model.is_reliable():
                class_prob = class_model.land_probability()
                if target_name and class_key in self._player_profiles.get(target_name.lower(), {}).__dict__:
                    # Blend per-player (weight 0.6) with class-wide (weight 0.4)
                    probability = 0.6 * probability + 0.4 * class_prob
                else:
                    probability = class_prob

        return max(0.0, min(1.0, probability))

    def is_likely_immune(
        self,
        status: str,
        target_class: str,
        target_name: str | None = None,
    ) -> bool:
        """Check if the target is likely immune to this status.

        Returns True if resistance probability > 85%.
        """
        return self.predict_lands(status, target_class, target_name) < 0.15

    # ── Learning from outcomes ─────────────────────────────────────────

    def record_status_attempt(
        self,
        status: str,
        target_name: str,
        target_class: str,
        landed: bool,
        duration: float | None = None,
        target_gear_hash: str | None = None,
    ) -> None:
        """Record a status attempt outcome.

        Args:
            status: Status effect name (e.g. 'stun', 'freeze', 'silence')
            target_name: Player name
            target_class: Job/class
            landed: True if status was applied, False if resisted
            duration: How long the status lasted (None if unknown)
            target_gear_hash: Optional gear fingerprint for change detection
        """
        with self._lock:
            status_key = status.lower()
            player_key = target_name.lower()
            class_key = target_class.lower()

            self._total_attempts += 1

            # ── Update per-class model ──
            self._update_class_model(class_key, status_key, landed)

            # ── Update per-player profile ──
            if player_key not in self._player_profiles:
                self._player_profiles[player_key] = PlayerStatusProfile(
                    player_name=target_name,
                    class_name=target_class,
                )

            profile = self._player_profiles[player_key]
            profile.total_observations += 1
            profile.last_updated = time.time()

            # Gear change detection
            if target_gear_hash and profile.last_gear_hash:
                if target_gear_hash != profile.last_gear_hash:
                    logger.info(
                        "[StatusResistance] %s gear changed — resetting partial immunity suspicion",
                        target_name,
                    )
                    # Don't fully reset, but flag and reduce confidence
                    self._gear_change_alerts += 1
                    for model in profile.models.values():
                        model.confidence *= 0.5
                        model.gear_change_detected = True
            profile.last_gear_hash = target_gear_hash

            # Update per-player model for this status
            model = profile.models.get(status_key)
            if not model:
                model = StatusResistanceModel(
                    status=status_key,
                    class_name=class_key,
                )
                profile.models[status_key] = model

            model.attempts += 1
            model.last_attempt_time = time.time()

            if landed:
                model.successes += 1
                model.recent_outcomes.append(1)
                # If previously thought immune, lower resistance estimate
                if status_key in profile.immunities:
                    # Gear changed — remove from immunity
                    profile.immunities.discard(status_key)
                    logger.info(
                        "[StatusResistance] %s: %s landed despite prior immunity — gear likely changed",
                        target_name, status_key,
                    )
            else:
                model.failures += 1
                model.recent_outcomes.append(0)
                # Track consecutive failures for immunity detection
                model.consecutive_unexpected += 1

                # If we've failed N times in a row and this class is known to
                # not have class immunity, flag possible card/gear immunity
                if model.failures >= 3 and model.failures >= model.attempts * 0.8:
                    logger.info(
                        "[StatusResistance] %s appears immune to %s (%d/%d resisted) — marking likely immunity",
                        target_name, status_key, model.failures, model.attempts,
                    )
                    profile.immunities.add(status_key)

            # Recalculate resistance probability
            if model.attempts >= 3:
                recent = list(model.recent_outcomes)
                if recent:
                    recent_land_rate = sum(recent) / len(recent)
                    overall_land_rate = model.successes / model.attempts
                    # Blend: recent is weighted more
                    blend = 0.6 * recent_land_rate + 0.4 * overall_land_rate
                    model.resistance_probability = 1.0 - blend
                else:
                    model.resistance_probability = 1.0 - (model.successes / model.attempts)

                # Confidence scales with sample size
                model.confidence = min(1.0, model.attempts / 15.0)

            # Update partial resistances
            if model.is_reliable() and model.resistance_probability > 0.2:
                profile.partial_resistances[status_key] = model.resistance_probability
            elif status_key in profile.partial_resistances and model.resistance_probability < 0.2:
                profile.partial_resistances.pop(status_key, None)

    def _update_class_model(self, class_key: str, status_key: str, landed: bool) -> None:
        """Update the class-wide resistance model."""
        class_models = self._class_resistance[class_key]
        model = class_models.get(status_key)
        if not model:
            model = StatusResistanceModel(status=status_key, class_name=class_key)
            class_models[status_key] = model

        model.attempts += 1
        if landed:
            model.successes += 1
            model.recent_outcomes.append(1)
        else:
            model.failures += 1
            model.recent_outcomes.append(0)

        model.last_attempt_time = time.time()

        if model.attempts >= 3:
            recent = list(model.recent_outcomes)
            if recent:
                land_rate = sum(recent) / len(recent)
                model.resistance_probability = 1.0 - (
                    0.7 * land_rate + 0.3 * (model.successes / model.attempts)
                )
            else:
                model.resistance_probability = 1.0 - (model.successes / model.attempts)
            model.confidence = min(1.0, model.attempts / 20.0)

    # ── Class-level known resistances (baseline from formula) ──────────

    @staticmethod
    def get_base_resistance(status: str, vit: int, int_stat: int = 1) -> float:
        """Calculate base resistance probability from stats alone (pre-renewal).

        These aren't hardcoded rules that override learning — they're a starting
        estimate that learning replaces as observations accumulate.

        Return: 0.0 (immune) to 1.0 (always lands)
        """
        status_lower = status.lower()

        # VIT-based resistances
        if status_lower in VIT_STATUSES:
            if vit >= 100:
                return 0.0  # Immune at 100 VIT
            # Below 100 VIT: resistance scales with VIT, roughly 2% per point
            return max(0.0, 1.0 - (vit * 0.02))

        # INT-based resistances
        if status_lower in INT_STATUSES:
            if int_stat >= 100:
                return 0.0
            return max(0.0, 1.0 - (int_stat * 0.015))

        # For everything else, moderate resistance
        return 0.3

    # ── Query / introspection ──────────────────────────────────────────

    def get_recommended_statuses(
        self,
        target_class: str,
        target_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get the best statuses to use against a target, sorted by land probability."""
        candidates = []
        all_statuses = set(VIT_STATUSES | INT_STATUSES | AGI_STATUSES | set(CARD_IMMUNE_STATUSES.keys()))

        for status in all_statuses:
            prob = self.predict_lands(status, target_class, target_name)
            candidates.append({
                "status": status,
                "land_probability": round(prob, 3),
                "likely_immune": prob < 0.15,
            })

        candidates.sort(key=lambda x: x["land_probability"], reverse=True)
        return candidates

    def get_stats(self) -> dict[str, Any]:
        """Summary statistics for introspection."""
        with self._lock:
            class_count = len(self._class_resistance)
            total_class_models = sum(len(m) for m in self._class_resistance.values())
            return {
                "total_attempts": self._total_attempts,
                "players_tracked": len(self._player_profiles),
                "classes_tracked": class_count,
                "total_class_models": total_class_models,
                "gear_change_alerts": self._gear_change_alerts,
                "player_profiles": {
                    name: p.to_dict()
                    for name, p in list(self._player_profiles.items())[:15]
                },
            }
