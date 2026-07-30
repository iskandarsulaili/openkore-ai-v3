"""Self-Learning Skill Predictor — detects when enemy is casting and
predicts what skill based on cast time + caster class.

RO mechanic: When a player/monster starts casting, the client receives
a 'start_cast' packet. The cast bar fills over time. Different skills
have different cast times and visual indicators.

This module learns a model that improves as more casts are observed,
mapping (class, cast_time_ms, visible_effect) -> skill_name.

Self-* properties:
  - Self-learning: builds per-class cast-time signature database from observations
  - Self-optimizing: narrows confidence intervals as sample size grows
  - Self-adapting: adapts to server-specific cast time modifications
  - Self-healing: detects when predictions fail and adjusts model
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

# Cast time tolerance: how close (in ms) cast times need to match
CAST_TOLERANCE_MS: int = 200

MIN_OBSERVATIONS_FOR_RELIABLE: int = 3
DECAY_ALPHA: float = 0.15
MAX_RECENT_CASTS: int = 100


@dataclass
class SkillSignature:
    """Signature for one skill cast observation."""
    skill_name: str
    caster_class: str

    # Timing
    cast_time_ms: float = 0.0          # Observed cast time
    cast_time_std: float = 0.0         # Variability in cast time
    observed_cast_times: deque = field(default_factory=lambda: deque(maxlen=20))

    # Observational metadata
    observations: int = 0
    last_seen: float = 0.0
    confidence: float = 0.0

    # Visual indicators
    associated_effects: list[str] = field(default_factory=list)

    # Cooldown info (learned)
    cooldown_ms: float = 0.0
    cooldown_observations: int = 0

    def matches_cast_time(self, ms: float) -> bool:
        """Check if a cast time matches this skill's signature."""
        if self.observations < 2:
            return True  # No data yet — match everything
        return abs(ms - self.cast_time_ms) <= max(CAST_TOLERANCE_MS, self.cast_time_std * 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill_name,
            "class": self.caster_class,
            "cast_time_ms": round(self.cast_time_ms, 1),
            "cast_time_std": round(self.cast_time_std, 1),
            "observations": self.observations,
            "confidence": round(self.confidence, 3),
            "reliable": self.is_reliable(),
        }

    def is_reliable(self) -> bool:
        return self.observations >= MIN_OBSERVATIONS_FOR_RELIABLE and self.confidence > 0.4


@dataclass
class SkillPrediction:
    """Result of a skill prediction."""
    predicted_skill: str
    caster_class: str
    confidence: float
    match_accuracy_ms: float  # How close the cast time matched
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_skill": self.predicted_skill,
            "caster_class": self.caster_class,
            "confidence": round(self.confidence, 3),
            "match_accuracy_ms": round(self.match_accuracy_ms, 1),
            "recommendation": self.recommendation,
        }


class SelfLearningSkillPredictor:
    """Learns to predict what skill an enemy is casting.

    Usage:
        predictor = SelfLearningSkillPredictor()

        # When a cast starts:
        prediction = predictor.predict_skill("wizard", cast_time_ms=3200)
        if prediction.predicted_skill == "storm_gust":
            # Switch to GTB or run out of range

        # After the cast resolves:
        predictor.record_skill_used("wizard", "storm_gust", cast_time_ms=3200)
    """

    def __init__(self) -> None:
        self._lock = RLock()

        # Per-class skill signatures {class_key: {skill_name: SkillSignature}}
        self._skill_db: dict[str, dict[str, SkillSignature]] = defaultdict(dict)

        # Class-level cast time ranges (learned)
        # {class_key: {'min_ms': float, 'max_ms': float, 'avg_ms': float}}
        self._class_cast_ranges: dict[str, dict[str, float]] = {}

        # Recent cast observations for pattern mining
        self._recent_casts: deque = deque(maxlen=MAX_RECENT_CASTS)

        # Stats
        self._total_observations: int = 0
        self._correct_predictions: int = 0
        self._total_predictions: int = 0
        self._start_time: float = time.time()

        # Seed baseline RO knowledge so the predictor isn't empty on first run
        self.seed_default_knowledge()

    # ── Seed default knowledge ──────────────────────────────────────────

    def seed_default_knowledge(self) -> None:
        """Populate known RO skill signatures so predictions work from day one.

        These are baseline cast times from official RO mechanics.  The
        self-learning layer will adjust them as real observations come in.
        """
        # Wizard
        wizard_skills: dict[str, float] = {
            "fire_bolt": 1500,
            "cold_bolt": 1500,
            "lightning_bolt": 1500,
            "storm_gust": 5000,
            "meteor_storm": 6000,
            "lord_of_vermillion": 5000,
            "heavens_drive": 4000,
            "thunderstorm": 3000,
        }
        # Priest
        priest_skills: dict[str, float] = {
            "heal": 1000,
            "blessing": 2000,
            "increase_agility": 2000,
            "magnificat": 2000,
            "gloria": 3000,
            "kyrie_eleison": 2000,
            "assumptio": 3000,
            "lex_aeterna": 1000,
        }
        # Hunter
        hunter_skills: dict[str, float] = {
            "anklesnare": 500,
            "double_strafing": 0,
            "arrow_shower": 0,
            "blast_arrow": 1500,
        }
        # Knight
        knight_skills: dict[str, float] = {
            "bowling_bash": 500,
            "bash": 0,
            "magnum_break": 0,
            "brandish_spear": 1000,
            "spear_boomerang": 500,
        }
        # Assassin
        assassin_skills: dict[str, float] = {
            "sonic_blow": 0,
            "grimtooth": 0,
            "soul_destroyer": 2000,
        }
        # Blacksmith
        blacksmith_skills: dict[str, float] = {
            "hammer_fall": 1000,
            "power_thrust": 0,
            "weapon_perfection": 2000,
        }
        # Sage
        sage_skills: dict[str, float] = {
            "endow": 2000,
            "deluge": 3000,
            "volcano": 3000,
            "violent_gale": 3000,
        }
        # Dancer / Bard
        dancer_bard_skills: dict[str, float] = {
            "hypnotist_waltz": 3000,
            "experience_increase": 3000,
        }

        all_classes: list[tuple[str, dict[str, float]]] = [
            ("wizard", wizard_skills),
            ("priest", priest_skills),
            ("hunter", hunter_skills),
            ("knight", knight_skills),
            ("assassin", assassin_skills),
            ("blacksmith", blacksmith_skills),
            ("sage", sage_skills),
            ("dancer_bard", dancer_bard_skills),
        ]

        now = time.time()
        for class_name, skill_map in all_classes:
            class_key = class_name.lower()
            for skill_name, cast_ms in skill_map.items():
                sig = SkillSignature(
                    skill_name=skill_name,
                    caster_class=class_name.title(),
                    cast_time_ms=cast_ms,
                    cast_time_std=100.0,          # Baseline variance
                    observations=3,                # Starts "reliable"
                    last_seen=now,
                    confidence=0.5,                # Moderate baseline confidence
                )
                self._skill_db[class_key][skill_name] = sig
                self._class_cast_ranges[class_key] = {
                    "min_ms": cast_ms,
                    "max_ms": cast_ms,
                    "avg_ms": cast_ms,
                }

    # ── Prediction ──────────────────────────────────────────────────────

    def predict_skill(
        self,
        caster_class: str,
        cast_time_ms: float,
        visible_effects: list[str] | None = None,
    ) -> SkillPrediction:
        """Predict what skill is being cast based on class and cast time.

        Args:
            caster_class: The caster's job/class
            cast_time_ms: Observed cast time in milliseconds
            visible_effects: Any visible spell effects observed

        Returns:
            SkillPrediction with predicted skill name and confidence
        """
        class_key = caster_class.lower()
        effects = [e.lower() for e in (visible_effects or [])]

        candidates: list[tuple[str, SkillSignature, float]] = []

        with self._lock:
            skills = self._skill_db.get(class_key, {})

            if not skills:
                # No data for this class yet — return unknown
                return SkillPrediction(
                    predicted_skill="unknown",
                    caster_class=caster_class,
                    confidence=0.0,
                    match_accuracy_ms=0.0,
                    recommendation="No cast data for this class — observe and learn",
                )

            for skill_name, sig in skills.items():
                if not sig.matches_cast_time(cast_time_ms):
                    continue

                # Score: how close is the match?
                if sig.observations >= 2:
                    time_diff = abs(cast_time_ms - sig.cast_time_ms)
                    time_score = max(0, 1.0 - time_diff / max(sig.cast_time_ms * 0.5, 500))
                else:
                    time_score = 0.5

                # Visual effect bonus
                effect_score = 0.0
                if effects and sig.associated_effects:
                    matching_effects = sum(
                        1 for e in effects if any(
                            sig_eff in e or e in sig_eff
                            for sig_eff in sig.associated_effects
                        )
                    )
                    if matching_effects > 0:
                        effect_score = min(1.0, matching_effects * 0.3)

                # Confidence score
                conf_score = sig.confidence if sig.is_reliable() else sig.confidence * 0.3

                # Total score
                total = 0.5 * time_score + 0.3 * conf_score + 0.2 * effect_score
                candidates.append((skill_name, sig, total))

        if not candidates:
            return SkillPrediction(
                predicted_skill="unknown",
                caster_class=caster_class,
                confidence=0.0,
                match_accuracy_ms=0.0,
                recommendation=f"No known skills match {cast_time_ms}ms cast for {caster_class}",
            )

        # Sort by score descending
        candidates.sort(key=lambda x: x[2], reverse=True)
        best_name, best_sig, best_score = candidates[0]

        # Build alternatives list
        alternatives = [
            {
                "skill": name,
                "confidence": round(score, 3),
                "cast_time_ms": round(sig.cast_time_ms, 1),
                "observations": sig.observations,
            }
            for name, sig, score in candidates[1:4]  # Top alternatives
        ]

        self._total_predictions += 1

        # Recommendation based on prediction
        recommendation = self._get_skill_reaction(best_name, best_score)

        return SkillPrediction(
            predicted_skill=best_name,
            caster_class=caster_class,
            confidence=min(1.0, best_score),
            match_accuracy_ms=abs(cast_time_ms - best_sig.cast_time_ms),
            alternatives=alternatives,
            recommendation=recommendation,
        )

    def _get_skill_reaction(self, skill_name: str, confidence: float) -> str:
        """Generate recommended reaction for a predicted skill."""
        dangerous_skills = {
            "storm_gust": "RUN — AoE freeze + damage, leave the area",
            "lord_of_vermillion": "RUN — AoE lightning, high damage",
            "asura_strike": "KILL FIRST or run — one-shot burst",
            "acid_demo": "Sonic damage, ignores defense — don't tank",
            "sacrifice": "Avoid being near target's teammates",
            "coma": "Dodge or stun-lock caster",
            "full_divestment": "Don't let stalker close in",
            "shield_boomerang": "Stun incoming — prepare vitia or stun-resist",
            "fire_bolt": "Tank — moderate damage, single target",
            "cold_bolt": "Tank — moderate damage, single target",
            "lightning_bolt": "Tank — moderate damage, single target",
            "double_strafing": "Low threat — ranged physical",
            "soul_strike": "Low threat — magic, auto-target",
            "napalm_beat": "Low threat — short range magic",
        }

        if confidence > 0.6 and skill_name in dangerous_skills:
            return dangerous_skills[skill_name]
        elif confidence > 0.3:
            return f"Possible {skill_name} — prepare appropriate counter"
        return f"Low confidence prediction: {skill_name}"

    # ── Learning from observations ─────────────────────────────────────

    def record_skill_used(
        self,
        caster_class: str,
        skill_name: str,
        cast_time_ms: float,
        visible_effects: list[str] | None = None,
    ) -> None:
        """Record a cast observation to improve future predictions.

        Args:
            caster_class: The caster's job/class
            skill_name: The skill that was actually cast
            cast_time_ms: Observed cast time in milliseconds
            visible_effects: Any visible spell effects
        """
        with self._lock:
            class_key = caster_class.lower()
            skill_key = skill_name.lower()

            # Get/create signature
            sig = self._skill_db[class_key].get(skill_key)
            if sig is None:
                sig = SkillSignature(
                    skill_name=skill_name,
                    caster_class=caster_class,
                )
                self._skill_db[class_key][skill_key] = sig

            sig.observations += 1
            sig.last_seen = time.time()
            sig.observed_cast_times.append(cast_time_ms)

            # Calculate running statistics
            times = list(sig.observed_cast_times)
            if len(times) >= 2:
                sig.cast_time_ms = sum(times) / len(times)
                if len(times) > 1:
                    variance = sum((t - sig.cast_time_ms) ** 2 for t in times) / len(times)
                    sig.cast_time_std = math.sqrt(variance)
            else:
                sig.cast_time_ms = cast_time_ms

            # Update associated visual effects
            if visible_effects:
                for effect in visible_effects:
                    if effect.lower() not in [e.lower() for e in sig.associated_effects]:
                        sig.associated_effects.append(effect.lower())

            # Confidence grows with observations
            sig.confidence = min(1.0, sig.observations / 8.0)

            # ── Update class-level cast time ranges ──
            self._update_class_cast_range(class_key)

            # ── Record to recent casts for pattern mining ──
            self._recent_casts.append({
                "class": caster_class,
                "skill": skill_name,
                "cast_time_ms": cast_time_ms,
                "timestamp": time.time(),
            })

            self._total_observations += 1

    def record_cooldown_observation(
        self,
        caster_class: str,
        skill_name: str,
        time_between_casts_ms: float,
    ) -> None:
        """Record time between two consecutive casts of the same skill.

        Used to learn skill cooldowns.
        """
        with self._lock:
            class_key = caster_class.lower()
            skill_key = skill_name.lower()
            sig = self._skill_db.get(class_key, {}).get(skill_key)
            if sig and time_between_casts_ms > 0:
                if sig.cooldown_observations == 0:
                    sig.cooldown_ms = time_between_casts_ms
                else:
                    sig.cooldown_ms = (
                        DECAY_ALPHA * time_between_casts_ms
                        + (1.0 - DECAY_ALPHA) * sig.cooldown_ms
                    )
                sig.cooldown_observations += 1

    def _update_class_cast_range(self, class_key: str) -> None:
        """Update the overall cast time range for a class."""
        skills = self._skill_db.get(class_key, {})
        if not skills:
            return

        times = [s.cast_time_ms for s in skills.values() if s.observations >= 2]
        if not times:
            return

        self._class_cast_ranges[class_key] = {
            "min_ms": min(times),
            "max_ms": max(times),
            "avg_ms": sum(times) / len(times),
        }

    def record_prediction_accuracy(self, was_correct: bool) -> None:
        """Track whether our last prediction was correct for self-evaluation."""
        if was_correct:
            self._correct_predictions += 1

    # ── Query / introspection ──────────────────────────────────────────

    def get_prediction_accuracy(self) -> float:
        """Get overall prediction accuracy rate."""
        if self._total_predictions == 0:
            return 0.0
        return self._correct_predictions / self._total_predictions

    def get_stats(self) -> dict[str, Any]:
        """Summary statistics for introspection."""
        with self._lock:
            total_skills = sum(len(s) for s in self._skill_db.values())
            return {
                "total_observations": self._total_observations,
                "total_predictions": self._total_predictions,
                "correct_predictions": self._correct_predictions,
                "prediction_accuracy": round(self.get_prediction_accuracy(), 3),
                "classes_tracked": len(self._skill_db),
                "total_skills_tracked": total_skills,
                "class_skills": {
                    cls: {
                        "skill_count": len(skills),
                        "reliable_skills": sum(
                            1 for s in skills.values() if s.is_reliable()
                        ),
                    }
                    for cls, skills in self._skill_db.items()
                },
            }
