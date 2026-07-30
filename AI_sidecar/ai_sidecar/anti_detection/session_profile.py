"""
Session Profile — session-length-dependent behavior scaling.
=============================================================

Detection signal: bots behave identically from minute 1 to minute 600.
Humans change over a session:
- First 15 min: Fresh, faster reactions, more attentive
- 15-60 min: Peak performance, consistent
- 60-120 min: Fatigue begins, slightly slower, more mistakes
- 120+ min: Tired, slower reactions, more variable
- 180+ min: Significant fatigue, frequent micro-breaks

This module:
1. Tracks session duration and computes a fatigue profile
2. Provides scaling factors for reaction delay, mistake rate, and social frequency
3. Supports session "warm-up" (faster at start) and "cool-down" (slower at end)
4. Integrates with the BehaviorEngine's SessionFatigueConfig
5. Can simulate multi-session patterns (player returning after break = fresh)

The session profile is per-bot and persists for the lifetime of the sidecar
process. On sidecar restart, session state is fresh (simulating the player
logging in fresh).

Integration:
- bridge_wiring.py calls get_profile() to get current session phase
- command_pacing.py uses fatigue_multiplier for delay scaling
- anti_afk.py uses social_frequency_scale to reduce idle actions when tired
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any

from ai_sidecar.anti_detection.behavior_engine import (
    BehaviorEngine,
    get_behavior_engine,
)

logger = logging.getLogger(__name__)


class SessionPhase(Enum):
    """Defined phases of a play session."""
    FRESH = "fresh"              # 0-15 min: Just logged in, sharp
    PEAK = "peak"                # 15-60 min: Optimal performance
    EARLY_FATIGUE = "fatigue"    # 60-120 min: Starting to tire
    TIRED = "tired"              # 120-180 min: Clearly fatigued
    EXHAUSTED = "exhausted"      # 180+ min: Very tired, high mistake rate


# Phase boundaries in minutes
_PHASE_BOUNDARIES = {
    SessionPhase.FRESH: (0, 15),
    SessionPhase.PEAK: (15, 60),
    SessionPhase.EARLY_FATIGUE: (60, 120),
    SessionPhase.TIRED: (120, 180),
    SessionPhase.EXHAUSTED: (180, float("inf")),
}


@dataclass
class SessionProfileConfig:
    """Configuration for session profiling."""
    enabled: bool = True

    # Fatigue onset
    fatigue_start_minutes: int = 60
    max_fatigue_multiplier: float = 2.0
    mistake_increase_rate: float = 0.5  # per hour

    # Warm-up (first N minutes = faster)
    warmup_enabled: bool = True
    warmup_duration_minutes: int = 15
    warmup_reaction_bonus: float = 0.80  # 80% of normal reaction time

    # Cool-down (last N actions before logout = slower)
    cooldown_enabled: bool = True
    cooldown_actions_before_logout: int = 5
    cooldown_delay_multiplier: float = 1.5

    # Micro-breaks (short pauses that increase with fatigue)
    micro_break_enabled: bool = True
    micro_break_start_minutes: int = 90
    micro_break_min_s: int = 3
    micro_break_max_s: int = 8
    micro_break_frequency_per_hour: float = 2.0  # At fatigue onset

    # Mistake scaling
    mistake_scale_fresh: float = 0.5      # Half normal mistakes when fresh
    mistake_scale_peak: float = 1.0        # Normal mistakes at peak
    mistake_scale_fatigued: float = 1.5    # 50% more mistakes when tired

    # Social frequency scaling
    social_frequency_fresh: float = 1.2    # More social when fresh
    social_frequency_peak: float = 1.0
    social_frequency_fatigued: float = 0.6  # Less social when tired


@dataclass
class SessionProfile:
    """Current session profile for a single bot."""
    bot_id: str
    session_start: float = field(default_factory=time.time)
    total_actions: int = 0
    total_idle_breaks: int = 0
    total_micro_breaks: int = 0

    # Rolling metrics
    recent_action_times: list[float] = field(default_factory=list)
    last_micro_break_time: float = 0.0

    def elapsed_minutes(self) -> float:
        """Minutes since session start."""
        return (time.time() - self.session_start) / 60.0

    @property
    def phase(self) -> SessionPhase:
        """Get the current session phase based on elapsed time."""
        minutes = self.elapsed_minutes()
        for phase, (start, end) in _PHASE_BOUNDARIES.items():
            if start <= minutes < end:
                return phase
        return SessionPhase.EXHAUSTED


class SessionProfiler:
    """Computes session-length-dependent behavior scaling factors.

    Usage::

        profiler = SessionProfiler(bot_id="bot1")
        profile = profiler.get_profile()  # Returns SessionProfile

        fatigue_mult = profiler.get_fatigue_multiplier()
        mistake_scale = profiler.get_mistake_scale()
        social_scale = profiler.get_social_frequency_scale()

        # Should we take a micro-break?
        if profiler.should_micro_break():
            duration = profiler.get_micro_break_duration()
            # pause for duration seconds
    """

    def __init__(
        self,
        bot_id: str = "default",
        config: SessionProfileConfig | None = None,
        engine: BehaviorEngine | None = None,
    ) -> None:
        self._bot_id = bot_id
        self._config = config or SessionProfileConfig()
        self._engine = engine or get_behavior_engine()
        self._profile = SessionProfile(bot_id=bot_id)
        self._lock = RLock()

    # ── Public API ───────────────────────────────────────────────────────────

    def get_profile(self) -> SessionProfile:
        """Get the current session profile (thread-safe)."""
        with self._lock:
            return self._profile

    def get_phase(self) -> SessionPhase:
        """Get the current session phase."""
        return self._profile.phase

    def get_fatigue_multiplier(self) -> float:
        """Get the fatigue multiplier for reaction delays.

        1.0 = normal, 2.0 = twice as slow (very tired).
        """
        cfg = self._config
        if not cfg.enabled:
            return 1.0

        minutes = self._profile.elapsed_minutes()
        if minutes <= cfg.fatigue_start_minutes:
            return 1.0

        fatigue_hours = (minutes - cfg.fatigue_start_minutes) / 60.0
        mult = 1.0 + fatigue_hours * cfg.mistake_increase_rate
        return min(cfg.max_fatigue_multiplier, mult)

    def get_warmup_multiplier(self) -> float:
        """Get the warm-up bonus multiplier.

        < 1.0 = faster reactions (warm-up bonus)
        1.0 = normal (warm-up over)
        """
        cfg = self._config
        if not cfg.warmup_enabled:
            return 1.0

        if self._profile.elapsed_minutes() < cfg.warmup_duration_minutes:
            return cfg.warmup_reaction_bonus
        return 1.0

    def get_reaction_multiplier(self) -> float:
        """Combined reaction time multiplier (warm-up × fatigue)."""
        return self.get_warmup_multiplier() * self.get_fatigue_multiplier()

    def get_mistake_scale(self) -> float:
        """Get the mistake rate scaling factor based on session phase."""
        phase = self._profile.phase
        cfg = self._config
        scales = {
            SessionPhase.FRESH: cfg.mistake_scale_fresh,
            SessionPhase.PEAK: cfg.mistake_scale_peak,
            SessionPhase.EARLY_FATIGUE: cfg.mistake_scale_fatigued,
            SessionPhase.TIRED: cfg.mistake_scale_fatigued * 1.3,
            SessionPhase.EXHAUSTED: cfg.mistake_scale_fatigued * 1.5,
        }
        return scales.get(phase, 1.0)

    def get_social_frequency_scale(self) -> float:
        """Get the social action frequency scaling factor.

        < 1.0 = less social (tired), > 1.0 = more social (fresh).
        """
        phase = self._profile.phase
        cfg = self._config
        scales = {
            SessionPhase.FRESH: cfg.social_frequency_fresh,
            SessionPhase.PEAK: cfg.social_frequency_peak,
            SessionPhase.EARLY_FATIGUE: cfg.social_frequency_fatigued,
            SessionPhase.TIRED: cfg.social_frequency_fatigued * 0.8,
            SessionPhase.EXHAUSTED: cfg.social_frequency_fatigued * 0.5,
        }
        return scales.get(phase, 1.0)

    def should_micro_break(self) -> bool:
        """Check if the bot should take a micro-break (short pause)."""
        cfg = self._config
        if not cfg.micro_break_enabled:
            return False

        minutes = self._profile.elapsed_minutes()
        if minutes < cfg.micro_break_start_minutes:
            return False

        minutes_since_last = (
            time.time() - self._profile.last_micro_break_time
        ) / 60.0

        # Expected frequency increases with fatigue
        fatigue_mult = self.get_fatigue_multiplier()
        expected_interval = 60.0 / (
            cfg.micro_break_frequency_per_hour * fatigue_mult
        )

        if minutes_since_last >= expected_interval and random.random() < 0.3:
            return True
        return False

    def get_micro_break_duration(self) -> int:
        """Get the duration of a micro-break in seconds."""
        with self._lock:
            self._profile.last_micro_break_time = time.time()
            self._profile.total_micro_breaks += 1
        return random.randint(
            self._config.micro_break_min_s,
            self._config.micro_break_max_s,
        )

    def record_action(self) -> None:
        """Record that an action was performed."""
        with self._lock:
            self._profile.total_actions += 1
            self._profile.recent_action_times.append(time.time())
            # Keep rolling window of 100 actions
            if len(self._profile.recent_action_times) > 100:
                self._profile.recent_action_times.pop(0)

    def get_actions_per_minute(self) -> float:
        """Get recent actions-per-minute rate."""
        with self._lock:
            recent = self._profile.recent_action_times
            if len(recent) < 2:
                return 0.0
            window = recent[-1] - recent[0]
            if window < 1.0:
                return 0.0
            return (len(recent) - 1) / (window / 60.0)

    def get_full_profile(self) -> dict[str, Any]:
        """Return the complete session profile as a dict (for telemetry)."""
        p = self._profile
        return {
            "bot_id": p.bot_id,
            "session_minutes": round(p.elapsed_minutes(), 1),
            "phase": p.phase.value,
            "total_actions": p.total_actions,
            "total_idle_breaks": p.total_idle_breaks,
            "total_micro_breaks": p.total_micro_breaks,
            "actions_per_minute": round(self.get_actions_per_minute(), 1),
            "fatigue_multiplier": round(self.get_fatigue_multiplier(), 2),
            "warmup_multiplier": round(self.get_warmup_multiplier(), 2),
            "reaction_multiplier": round(self.get_reaction_multiplier(), 2),
            "mistake_scale": round(self.get_mistake_scale(), 2),
            "social_frequency_scale": round(self.get_social_frequency_scale(), 2),
        }


# ── Global singleton registry ────────────────────────────────────────────────

_profilers: dict[str, SessionProfiler] = {}
_profilers_lock = RLock()


def get_session_profiler(bot_id: str = "default") -> SessionProfiler:
    """Get or create a SessionProfiler for a specific bot."""
    global _profilers
    with _profilers_lock:
        if bot_id not in _profilers:
            _profilers[bot_id] = SessionProfiler(bot_id=bot_id)
        return _profilers[bot_id]
