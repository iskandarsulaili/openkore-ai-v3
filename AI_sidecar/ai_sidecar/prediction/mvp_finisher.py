"""MVP Finisher — tracks MVP HP and triggers finisher attacks at optimal timing.

RO mechanic: MVP reward is based on:
  1. Most damage dealt (primary)
  2. Last hit (tiebreaker for equal damage)
  3. Party membership / proximity

This module learns the optimal HP threshold for finisher moves based on
observed MVP kill patterns, server lag, and typical race conditions.

Self-* properties:
  - Self-learning: learns optimal finisher HP thresholds from MVP encounters
  - Self-optimizing: adjusts finisher timing based on competition intensity
  - Self-adapting: adapts to server-specific MVP behavior (spawn timers, HP)
  - Self-healing: detects when a finisher failed and adjusts threshold
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

DEFAULT_FINISHER_THRESHOLD: float = 0.05  # 5% HP
MIN_MVP_OBSERVATIONS: int = 2
DECAY_ALPHA: float = 0.15
MAX_RECENT_MVPS: int = 20

# MVP HP estimation based on level and known HP patterns
BASE_MVP_HP_ESTIMATES: dict[str, int] = {
    "hatii": 1500000,
    "garm": 1200000,
    "kraken": 2000000,
    "thanatos": 2500000,
    "edga": 1800000,
    "baphomet": 2200000,
    "drake": 1400000,
    "doppelganger": 1600000,
    "gloom_under_night": 2800000,
    "turtle_general": 1300000,
    "moonlight_flower": 1100000,
    "osiris": 1900000,
    "phreeoni": 1200000,
    "orc_hero": 1500000,
    "orc_lord": 2100000,
    "maya": 1700000,
    "mistress": 1000000,
}


@dataclass
class MvpFinisherObservation:
    """Record of one MVP encounter."""
    mvp_name: str
    mvp_level: int

    # HP tracking
    estimated_max_hp: int = 0
    hp_when_entered: int = 0     # HP % when we started fighting
    hp_100pct: int = 0           # HP when we first see it (estimate of max)

    # Finisher
    finisher_threshold_used: float = 0.05  # % HP we aimed for
    actual_hp_when_killed: float = 0.0     # Actual % HP when it died
    we_got_last_hit: bool = False
    we_got_mvp: bool = False
    competition_count: int = 0   # How many other players were competing

    # Timing
    fight_duration: float = 0.0
    finisher_skill: str = ""
    finisher_damage: int = 0

    # Lag / race conditions
    server_tick_alignment_ms: float = 0.0
    latency_ms: float = 0.0

    timestamp: float = 0.0

    def __post_init__(self) -> None:
        self.timestamp = time.time()


@dataclass
class MvpModel:
    """Learned model for one specific MVP."""
    mvp_name: str
    encounters: int = 0

    # HP model
    estimated_max_hp: int = 0
    hp_estimates: list[int] = field(default_factory=list)

    # Finisher optimization
    success_rate: float = 0.0
    total_attempts: int = 0
    successful_finishes: int = 0
    optimal_threshold: float = DEFAULT_FINISHER_THRESHOLD
    observed_kill_hps: deque = field(default_factory=lambda: deque(maxlen=MAX_RECENT_MVPS))

    # Competition adjustment
    avg_competitors: float = 0.0
    last_encounter_time: float = 0.0

    # Per-MVP finisher recommendations
    best_finisher_skills: list[dict[str, Any]] = field(default_factory=list)

    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mvp": self.mvp_name,
            "encounters": self.encounters,
            "estimated_hp": self.estimated_max_hp,
            "optimal_threshold": f"{self.optimal_threshold:.1%}",
            "success_rate": round(self.success_rate, 3),
            "attempts": self.total_attempts,
            "successes": self.successful_finishes,
            "avg_competitors": round(self.avg_competitors, 1),
            "confidence": round(self.confidence, 3),
        }


class MvpFinisher:
    """Learns optimal MVP finisher timing from experience.

    Usage:
        finisher = MvpFinisher()

        # When we engage an MVP:
        finisher.start_tracking("Hatii", level=99, estimated_hp=1500000)

        # Periodically update HP:
        finisher.update_hp("Hatii", current_hp=750000)

        # Check if we should use finisher:
        advice = finisher.should_finish("Hatii")
        if advice["use_finisher"]:
            use(advice["recommended_skill"])

        # After the fight:
        finisher.record_encounter_outcome(
            mvp_name="Hatii",
            we_got_last_hit=True,
            we_got_mvp=True,
            finisher_hp_pct=0.04,
            competitors=3,
        )
    """

    def __init__(self) -> None:
        self._lock = RLock()

        # Per-MVP models
        self._mvp_models: dict[str, MvpModel] = {}

        # Active tracking for current MVP fights
        # {mvp_name: {field: value}}
        self._active_tracking: dict[str, dict[str, Any]] = {}

        # Server-wide meta
        self._total_encounters: int = 0
        self._total_last_hits: int = 0
        self._total_mvp_awards: int = 0
        self._optimal_threshold_adjustments: int = 0

        # Global latency estimate (for finisher timing adjustment)
        self._estimated_latency_ms: float = 0.0
        self._latency_samples: int = 0

        self._start_time: float = time.time()

        # Seed MVP knowledge so the finisher isn't empty on first run
        self.seed_mvp_knowledge()

    # ── Seed MVP knowledge ──────────────────────────────────────────────

    def seed_mvp_knowledge(self) -> None:
        """Populate known MVP behaviours so finisher decisions are informed
        from day one.

        Each entry records:
          - Where the MVP spawns (for path planning)
          - Key combat mechanics (element shifts, clones, summons, etc.)
          - Baseline HP estimate
        """
        mvp_data: list[dict[str, Any]] = [
            {
                "name": "Eddga",
                "spawn_map": "pay_fild03",
                "mechanics": "Shifts to Fire element below 25% HP",
                "hp": 1800000,
            },
            {
                "name": "Moonlight Flower",
                "spawn_map": "umbala_dun01",
                "mechanics": "Clones at 50% HP",
                "hp": 1100000,
            },
            {
                "name": "Drake",
                "spawn_map": "trev_dun02",
                "mechanics": "Summons skeletons",
                "hp": 1400000,
            },
            {
                "name": "Osiris",
                "spawn_map": "moc_pryd05",
                "mechanics": "Teleports",
                "hp": 1900000,
            },
            {
                "name": "Doppelganger",
                "spawn_map": "gef_dun01",
                "mechanics": "Clones at 75% HP",
                "hp": 1600000,
            },
            {
                "name": "Phreeoni",
                "spawn_map": "mjolnir_12",
                "mechanics": "Runs at low HP",
                "hp": 1200000,
            },
            {
                "name": "Orc Hero",
                "spawn_map": "orc_dun02",
                "mechanics": "Stuns with skill",
                "hp": 1500000,
            },
            {
                "name": "Orc Lord",
                "spawn_map": "orc_dun03",
                "mechanics": "Calls orcs",
                "hp": 2100000,
            },
            {
                "name": "Baphomet",
                "spawn_map": "nif_dun02",
                "mechanics": "Teleports + AoE",
                "hp": 2200000,
            },
            {
                "name": "Maya",
                "spawn_map": "ant_dun01",
                "mechanics": "Immune to physical when shell up",
                "hp": 1700000,
            },
            {
                "name": "Mistress",
                "spawn_map": "tur_dun01",
                "mechanics": "Flies away",
                "hp": 1000000,
            },
        ]

        now = time.time()
        for entry in mvp_data:
            key = entry["name"].lower()
            if key in self._mvp_models:
                continue  # Don't overwrite a model that already exists

            model = MvpModel(mvp_name=entry["name"])
            model.estimated_max_hp = entry["hp"]
            model.encounters = 1
            model.total_attempts = 1
            model.optimal_threshold = DEFAULT_FINISHER_THRESHOLD
            model.last_encounter_time = now
            model.confidence = 0.3  # Baseline — improved by real encounters

            self._mvp_models[key] = model

    # ── Active tracking ────────────────────────────────────────────────

    def start_tracking(
        self,
        mvp_name: str,
        level: int = 0,
        estimated_hp: int = 0,
    ) -> None:
        """Start tracking an MVP for finisher timing.

        Call when you first see or engage an MVP.
        """
        with self._lock:
            key = mvp_name.lower()
            self._active_tracking[key] = {
                "mvp_name": mvp_name,
                "level": level,
                "max_hp_estimate": estimated_hp or self._get_hp_estimate(key),
                "hp_100pct": self._get_hp_estimate(key),
                "last_hp": 0,
                "last_hp_pct": 1.0,
                "hp_history": [(time.time(), 1.0)],
                "start_time": time.time(),
                "damage_dealt": 0,
            }

    def update_hp(
        self,
        mvp_name: str,
        current_hp: int,
        max_hp: int = 0,
    ) -> None:
        """Update the current HP of a tracked MVP."""
        with self._lock:
            key = mvp_name.lower()
            tracking = self._active_tracking.get(key)
            if tracking is None:
                return

            if max_hp > 0:
                tracking["max_hp_estimate"] = max_hp

            max_hp_est = max(tracking["max_hp_estimate"], 1)
            hp_pct = max(0.0, current_hp / max_hp_est)

            tracking["last_hp"] = current_hp
            tracking["last_hp_pct"] = hp_pct
            tracking["hp_history"].append((time.time(), hp_pct))

            # Update max HP estimate if this is the highest we've seen
            if current_hp > tracking["hp_100pct"]:
                tracking["hp_100pct"] = current_hp

    def record_damage_dealt(
        self,
        mvp_name: str,
        damage: int,
    ) -> None:
        """Track damage we've dealt to the MVP."""
        with self._lock:
            key = mvp_name.lower()
            tracking = self._active_tracking.get(key)
            if tracking:
                tracking["damage_dealt"] += damage

    # ── Finisher decision ──────────────────────────────────────────────

    def should_finish(
        self,
        mvp_name: str,
        available_skills: list[str] | None = None,
    ) -> dict[str, Any]:
        """Check if we should use a finisher skill on the MVP.

        Returns:
            dict with use_finisher, threshold, recommended_skill, reason
        """
        with self._lock:
            key = mvp_name.lower()
            tracking = self._active_tracking.get(key)
            if tracking is None:
                return {"use_finisher": False, "reason": "MVP not tracked"}

            current_hp_pct = tracking["last_hp_pct"]
            model = self._mvp_models.get(key)

            # Determine optimal threshold
            if model and model.confidence > 0.3:
                threshold = model.optimal_threshold
            else:
                threshold = DEFAULT_FINISHER_THRESHOLD

            # Adjust threshold based on competition and latency
            adjusted_threshold = self._adjust_threshold(
                threshold, model, tracking
            )

            # Choose best finisher skill
            recommended_skill = self._recommend_finisher_skill(
                key, available_skills or []
            )

            should_finish = current_hp_pct <= adjusted_threshold

            # Calculate urgency
            # Lower HP = more urgent
            urgency = max(0.0, 1.0 - (current_hp_pct / adjusted_threshold))

            return {
                "use_finisher": should_finish,
                "current_hp_pct": round(current_hp_pct, 4),
                "threshold": round(adjusted_threshold, 4),
                "base_threshold": round(threshold, 4),
                "urgency": round(urgency, 3),
                "recommended_skill": recommended_skill,
                "estimated_hp_remaining": int(current_hp_pct * tracking["max_hp_estimate"]),
                "damage_dealt": tracking["damage_dealt"],
                "reason": (
                    f"MVP HP at {current_hp_pct:.1%} — ready for finisher!"
                    if should_finish
                    else f"Waiting for HP to drop below {adjusted_threshold:.1%} (currently {current_hp_pct:.1%})"
                ),
            }

    def _adjust_threshold(
        self,
        base_threshold: float,
        model: MvpModel | None,
        tracking: dict[str, Any],
    ) -> float:
        """Adjust finisher threshold based on learned factors.

        Higher threshold = finish earlier (compensates for lag / competition)
        """
        adjustment = 1.0

        # Competition: more competitors = need to finish earlier
        if model and model.avg_competitors > 1:
            comp_factor = 1.0 + (model.avg_competitors - 1) * 0.02
            adjustment = max(adjustment, comp_factor)

        # Latency: higher latency = need to trigger earlier
        if self._estimated_latency_ms > 50:
            latency_factor = 1.0 + (self._estimated_latency_ms / 1000.0)
            adjustment = max(adjustment, latency_factor)

        # Past failures: increase threshold if we've missed last hits
        if model and model.total_attempts > 2:
            success_rate = model.success_rate
            if success_rate < 0.3 and model.total_attempts >= 3:
                # We're missing too many — start earlier
                adjustment = max(adjustment, 1.3)

        return min(base_threshold * adjustment, 0.20)  # Cap at 20% HP

    def _recommend_finisher_skill(
        self,
        mvp_key: str,
        available_skills: list[str],
    ) -> str | None:
        """Pick the best finisher skill from available options."""
        if not available_skills:
            return None

        # Priority ordering for finisher skills (learned from experience)
        finisher_priority = [
            "asura_strike",      # Monk — highest single hit damage
            "acid_demo",         # Alchemist — ignores defense
            "sonic_blow",        # Assassin — multi-hit, high damage
            "spiral_pierce",     # Knight/Lord Knight — ranged pierce
            "soul_destroyer",    # Soul Linker — auto-hit magic
            "bolt_spam",         # Mage/Wizard — any bolt spell
            "double_strafing",   # Hunter — quick, ranged
            "arrow_shower",      # Archer — AoE, guaranteed hit
        ]

        for skill in finisher_priority:
            if skill in available_skills:
                return skill

        return available_skills[0]  # First available

    # ── Learning from outcomes ─────────────────────────────────────────

    def record_encounter_outcome(
        self,
        mvp_name: str,
        we_got_last_hit: bool,
        we_got_mvp: bool = False,
        finisher_hp_pct: float | None = None,
        competitors: int = 0,
        finisher_skill: str = "",
        finisher_damage: int = 0,
    ) -> None:
        """Record the outcome of an MVP encounter.

        This improves the finisher model for future encounters.
        """
        with self._lock:
            key = mvp_name.lower()
            model = self._mvp_models.get(key)
            if model is None:
                model = MvpModel(mvp_name=mvp_name)
                self._mvp_models[key] = model

            model.encounters += 1
            model.total_attempts += 1
            model.last_encounter_time = time.time()

            if we_got_last_hit:
                model.successful_finishes += 1
                self._total_last_hits += 1
            if we_got_mvp:
                self._total_mvp_awards += 1

            # Track HP where we actually killed
            if finisher_hp_pct is not None and finisher_hp_pct > 0:
                model.observed_kill_hps.append(finisher_hp_pct)

            # Success rate
            model.success_rate = model.successful_finishes / model.total_attempts

            # Competition
            if model.avg_competitors == 0:
                model.avg_competitors = float(competitors)
            else:
                model.avg_competitors = (
                    DECAY_ALPHA * competitors
                    + (1.0 - DECAY_ALPHA) * model.avg_competitors
                )

            # ── Optimize finisher threshold ──
            self._optimize_threshold(model)

            # ── Track best finisher skill ──
            if finisher_skill and finisher_damage > 0:
                # Find or create skill entry
                found = False
                for entry in model.best_finisher_skills:
                    if entry["skill"] == finisher_skill:
                        entry["uses"] += 1
                        entry["total_damage"] += finisher_damage
                        entry["avg_damage"] = entry["total_damage"] / entry["uses"]
                        found = True
                        break
                if not found:
                    model.best_finisher_skills.append({
                        "skill": finisher_skill,
                        "uses": 1,
                        "total_damage": finisher_damage,
                        "avg_damage": finisher_damage,
                    })

            # Confidence
            model.confidence = min(1.0, model.encounters / 8.0)

            self._total_encounters += 1

            # Clean up active tracking
            self._active_tracking.pop(key, None)

    def _optimize_threshold(self, model: MvpModel) -> None:
        """Learn the optimal finisher HP threshold for this MVP.

        If we consistently get last hits, we can afford to wait longer (lower threshold).
        If we keep missing, need to start earlier (raise threshold).
        """
        if model.total_attempts < 2:
            return

        # Current success rate vs target (85% success is ideal)
        success_rate = model.success_rate
        current_threshold = model.optimal_threshold

        if success_rate < 0.5:
            # Too many misses — start earlier
            model.optimal_threshold = min(current_threshold * 1.2, 0.20)
            self._optimal_threshold_adjustments += 1
        elif success_rate > 0.9 and model.successful_finishes >= 5:
            # Very successful — we can be more aggressive
            model.optimal_threshold = max(current_threshold * 0.9, 0.02)
            self._optimal_threshold_adjustments += 1

        # Also consider observed kill HP points
        if len(model.observed_kill_hps) >= 3:
            kill_hps = list(model.observed_kill_hps)
            avg_kill_hp = sum(kill_hps) / len(kill_hps)
            # Adjust toward the average of where kills actually happen
            blend = 0.3 * avg_kill_hp + 0.7 * model.optimal_threshold
            model.optimal_threshold = max(min(blend, 0.20), 0.02)

    def record_latency(self, latency_ms: float) -> None:
        """Record an observed latency measurement for timing adjustment."""
        with self._lock:
            if self._latency_samples == 0:
                self._estimated_latency_ms = latency_ms
            else:
                self._estimated_latency_ms = (
                    0.3 * latency_ms + 0.7 * self._estimated_latency_ms
                )
            self._latency_samples += 1

    # ── Helpers ─────────────────────────────────────────────────────────

    def _get_hp_estimate(self, mvp_key: str) -> int:
        """Get estimated max HP for an MVP."""
        # Check base estimates
        for name, hp in BASE_MVP_HP_ESTIMATES.items():
            if name in mvp_key or mvp_key in name:
                return hp

        # Check learned models
        model = self._mvp_models.get(mvp_key)
        if model and model.estimated_max_hp > 0:
            return model.estimated_max_hp

        return 1000000  # Default: 1M HP

    def record_hp_observation(self, mvp_name: str, max_hp: int) -> None:
        """Record a max HP observation for an MVP (from seeing it at full HP)."""
        with self._lock:
            key = mvp_name.lower()
            model = self._mvp_models.get(key)
            if model is None:
                model = MvpModel(mvp_name=mvp_name)
                self._mvp_models[key] = model

            if max_hp > 0:
                model.hp_estimates.append(max_hp)
                if model.estimated_max_hp == 0:
                    model.estimated_max_hp = max_hp
                else:
                    model.estimated_max_hp = max(model.estimated_max_hp, max_hp)

    # ── Query / introspection ──────────────────────────────────────────

    def get_active_trackings(self) -> list[dict[str, Any]]:
        """Get all currently tracked MVP fights."""
        with self._lock:
            return [
                {
                    "mvp": t["mvp_name"],
                    "hp_pct": round(t["last_hp_pct"], 3),
                    "damage_dealt": t["damage_dealt"],
                    "duration_seconds": round(time.time() - t["start_time"], 1),
                }
                for t in self._active_tracking.values()
            ]

    def get_stats(self) -> dict[str, Any]:
        """Summary statistics for introspection."""
        with self._lock:
            return {
                "total_encounters": self._total_encounters,
                "last_hits_secured": self._total_last_hits,
                "mvp_awards": self._total_mvp_awards,
                "mvp_models": len(self._mvp_models),
                "estimated_latency_ms": round(self._estimated_latency_ms, 1),
                "threshold_adjustments": self._optimal_threshold_adjustments,
                "active_trackings": len(self._active_tracking),
                "models": {
                    name: model.to_dict()
                    for name, model in self._mvp_models.items()
                },
            }
