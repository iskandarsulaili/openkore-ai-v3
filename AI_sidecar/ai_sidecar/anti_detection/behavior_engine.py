"""
Behavior Engine — Human-like Imperfect Play
=============================================
Pro-botting insight: perfect play is the #1 detection signal.
This engine injects controlled imperfections that mimic a skilled
but fallible human player. Configurable per-profile via YAML.

Features
--------
1. Gaussian reaction times — μ=150-300ms, σ=50ms, with fatigue scaling
2. Movement deviation — Gaussian-noise-based path waypoint jitter (humans don't walk straight)
3. Movement noise — Perlin-noise-based route variation (smooth, natural-feeling offsets)
4. Click patterns — variable double-click timing, occasional misclick near target
5. Session fatigue — longer sessions = slower reactions, more mistakes
6. Idle breaks — 30-120s AFK every 20-45 minutes
7. Typing speed variation — variable WPM for chat messages
8. Bag/inventory irregular timing — non-uniform intervals
9. Logout/relog patterns — random 5-15s delay before reconnecting
10. Server tick alignment randomness — don't act on exact server ticks
11. Contextual behavior profiles — ACTIVE, AFK, TIRED, WATCHING cycles

|Integration
-----------
- Bridge anti-detection (`$ANTI_DETECTION_ENABLED`) is read from
  the sidecar config and amplified with richer per-profile settings.
- The engine exposes a ``get_behavior_modifier()`` dict that
  ``bridge_wiring.py`` polls on each action dispatch and applies
  to command delays.
- ``route_humanizer.py`` reads ``movement_deviation`` and
  ``movement_noise`` from the modifier to add waypoint jitter.
- ``command_pacing.py`` reads ``delay_ms`` for human-like timing.
- A ``human_likeness`` score (0.0–1.0) is computed from the active
  behavior mix; higher = more human-like.
- No LLM calls — everything is math + random.
"""

from __future__ import annotations

import enum
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_PROFILES_DIR = Path(os.environ.get(
    "BEHAVIOR_PROFILES_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "config" / "behavior_profiles"),
))
_DEFAULT_PROFILE_PATH = _PROFILES_DIR / "default.yaml"

# ── Contextual Behavior Profile Enum ─────────────────────────────────────────


class BehaviorProfileType(enum.Enum):
    """Contextual behavior profiles that mimic different human play states.

    Each profile adjusts reaction time, mistake rate, path quality, social
    frequency, and AFK chance to simulate a specific human-like state.
    """

    ACTIVE = "active"
    """Focused play — fast reactions, optimal pathing, few mistakes."""

    AFK = "afk"
    """Away-from-keyboard — slow reactions, frequent standing still, more mistakes."""

    TIRED = "tired"
    """Fatigued play — slower reactions, wrong targets, pathing errors."""

    WATCHING = "watching"
    """GM/spectator nearby — near-perfect play, minimal mistakes, fast reactions."""


# ── Contextual profile config (per-profile settings) ─────────────────────────


@dataclass
class ContextualProfileConfig:
    """Settings for a single contextual behavior profile."""

    reaction_delay_range: list[int] = field(default_factory=lambda: [200, 400])
    """[min_ms, max_ms] range for reaction delays."""

    mistake_rate: float = 0.03
    """Base probability of micro-mistakes (0.0–1.0)."""

    wrong_target_chance: float = 0.08
    """Probability of selecting a wrong target (0.0–1.0)."""

    path_quality: float = 0.85
    """Path quality factor (0.0=poor, 1.0=optimal)."""

    social_frequency: float = 0.05
    """Probability of social actions like sitting/emoting (0.0–1.0)."""

    afk_chance: float = 0.05
    """Probability of taking an AFK break (0.0–1.0)."""


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class ReactionTimeConfig:
    enabled: bool = True
    min_ms: int = 100
    max_ms: int = 900
    distribution: str = "gaussian"  # gaussian | log_normal | uniform
    gaussian_mu: float = 250.0     # mean reaction time in ms
    gaussian_sigma: float = 50.0   # standard deviation
    sigma: float = 0.55
    consecutive_multiplier: float = 0.08
    fatigue_threshold: int = 8


@dataclass
class BadPathConfig:
    enabled: bool = True
    probability: float = 0.12
    extra_distance_pct: float = 0.30
    detour_chance: float = 0.04
    detour_min_extra_pct: float = 0.50
    detour_max_extra_pct: float = 1.50
    recheck_interval_seconds: float = 15.0


@dataclass
class WrongTargetConfig:
    enabled: bool = True
    probability: float = 0.08
    range_pct: float = 0.35
    switch_delay_ms: int = 600
    max_wrong_targets: int = 3
    recheck_on_death: bool = True


@dataclass
class AfkBreakConfig:
    enabled: bool = True
    min_break_seconds: int = 30
    max_break_seconds: int = 120
    min_interval_minutes: int = 20
    max_interval_minutes: int = 45
    early_break_chance: float = 0.05
    resume_delay_ms: int = 400


@dataclass
class FavoriteSpotsConfig:
    enabled: bool = True
    probability: float = 0.55
    return_distance: int = 20
    stay_duration_minutes: list[int] = field(default_factory=lambda: [3, 12])
    spots: dict[str, list[tuple[int, int]]] = field(default_factory=dict)


@dataclass
class MicroMistakesConfig:
    enabled: bool = True
    walk_into_wall: dict[str, Any] = field(default_factory=lambda: {
        "probability": 0.015,
        "recovery_ms": 700,
        "stuck_duration_ms": 1200,
    })
    cancel_cast: dict[str, Any] = field(default_factory=lambda: {
        "probability": 0.008,
        "cast_time_pct": 0.5,
    })
    wrong_direction: dict[str, Any] = field(default_factory=lambda: {
        "probability": 0.025,
        "wrong_steps": [1, 3],
        "correction_delay_ms": 350,
    })
    double_click: dict[str, Any] = field(default_factory=lambda: {
        "probability": 0.02,
        "duplicate_only_if_queued": True,
    })
    skill_missclick: dict[str, Any] = field(default_factory=lambda: {
        "probability": 0.005,
        "recovery_delay_ms": 500,
    })
    inventory_stutter: dict[str, Any] = field(default_factory=lambda: {
        "probability": 0.01,
        "stutter_duration_ms": 400,
    })


@dataclass
class MovementDeviationConfig:
    """Gaussian-noise-based movement path deviation.
    Adds realistic jitter to movement waypoints so bots don't walk in perfect straight lines.
    Used by route_humanizer to inject human-like path variation."""
    enabled: bool = True
    deviation_strength: float = 1.0       # multiplier on Gaussian σ
    min_deviation_cells: float = 0.5       # minimum noise magnitude (cells)
    update_every_n_steps: int = 3          # recalculate every N steps in a route


@dataclass
class MovementNoiseConfig:
    """Perlin-noise-based movement noise for route variation.
    Produces smooth, natural-feeling offsets — not GPS drift, but
    the same math applied to in-game movement waypoints."""
    enabled: bool = True
    noise_amplitude: float = 3.0       # max cells of deviation
    noise_frequency: float = 0.1       # how often direction changes
    noise_update_interval_s: float = 2.0


@dataclass
class TypingSpeedConfig:
    """Variable typing speed for chat messages."""
    enabled: bool = True
    wpm_min: int = 40
    wpm_max: int = 80
    mistake_chance: float = 0.03
    correction_delay_ms: int = 300


@dataclass
class InventoryTimingConfig:
    """Irregular bag/inventory open timing."""
    enabled: bool = True
    min_interval_s: int = 30
    max_interval_s: int = 180
    open_duration_ms_min: int = 200
    open_duration_ms_max: int = 800


@dataclass
class LogoutRelogConfig:
    """Logout/relog delay patterns."""
    enabled: bool = True
    min_delay_s: int = 5
    max_delay_s: int = 15
    reconnect_stutter_chance: float = 0.10


@dataclass
class ServerTickConfig:
    """Server tick alignment randomness."""
    enabled: bool = True
    tick_jitter_ms: int = 50
    min_tick_offset_ms: int = 10
    max_tick_offset_ms: int = 200


@dataclass
class SessionFatigueConfig:
    """Session duration fatigue modeling."""
    enabled: bool = True
    fatigue_start_minutes: int = 60
    max_fatigue_multiplier: float = 2.0
    mistake_increase_rate: float = 0.5  # per hour of play


@dataclass
class ScoringWeights:
    reaction_time_weight: float = 0.20
    bad_path_weight: float = 0.15
    wrong_target_weight: float = 0.15
    afk_break_weight: float = 0.20
    favorite_spots_weight: float = 0.10
    micro_mistakes_weight: float = 0.20


@dataclass
class BehaviorProfile:
    """A loaded behavior profile — mirrors the YAML structure."""

    profile_name: str = "default"
    human_likeness_target: float = 0.78
    reaction_time: ReactionTimeConfig = field(default_factory=ReactionTimeConfig)
    bad_path: BadPathConfig = field(default_factory=BadPathConfig)
    wrong_target: WrongTargetConfig = field(default_factory=WrongTargetConfig)
    afk_breaks: AfkBreakConfig = field(default_factory=AfkBreakConfig)
    favorite_spots: FavoriteSpotsConfig = field(default_factory=FavoriteSpotsConfig)
    micro_mistakes: MicroMistakesConfig = field(default_factory=MicroMistakesConfig)
    movement_deviation: MovementDeviationConfig = field(default_factory=MovementDeviationConfig)
    movement_noise: MovementNoiseConfig = field(default_factory=MovementNoiseConfig)
    typing_speed: TypingSpeedConfig = field(default_factory=TypingSpeedConfig)
    inventory_timing: InventoryTimingConfig = field(default_factory=InventoryTimingConfig)
    logout_relog: LogoutRelogConfig = field(default_factory=LogoutRelogConfig)
    server_tick: ServerTickConfig = field(default_factory=ServerTickConfig)
    session_fatigue: SessionFatigueConfig = field(default_factory=SessionFatigueConfig)
    scoring: ScoringWeights = field(default_factory=ScoringWeights)
    # ── Contextual profile configs ──
    contextual_profiles: dict[str, ContextualProfileConfig] = field(default_factory=dict)


@dataclass
class BehaviorResult:
    """Result returned by a single behavior evaluation."""

    applied: bool = False
    description: str = ""
    delay_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Distribution helpers ────────────────────────────────────────────────────


def _gaussian_sample(mean_ms: float, sigma_ms: float, min_ms: float, max_ms: float) -> float:
    """Sample from a Gaussian (normal) distribution, clamped to [min_ms, max_ms]."""
    raw = random.gauss(mean_ms, sigma_ms)
    return max(min_ms, min(max_ms, raw))


def _log_normal_sample(mean_ms: float, sigma: float, min_ms: float, max_ms: float) -> float:
    """Sample from a log-normal distribution, clamped to [min_ms, max_ms]."""
    mu = math.log(mean_ms) - (sigma ** 2) / 2.0
    raw = random.lognormvariate(mu, sigma)
    return max(min_ms, min(max_ms, raw))


def _gaussian_deviation_2d(
    x: float, y: float, target_x: float, target_y: float, strength: float = 1.0
) -> tuple[float, float]:
    """Add Gaussian (normal) noise to a movement coordinate.

    Humans don't walk perfectly straight — they deviate slightly.
    Returns (noisy_x, noisy_y) where noise = N(0, σ) with σ proportional
    to distance × strength. Clamped so we don't drift past the target.

    Used by route_humanizer to add realistic path variation.
    """
    import math
    dx = target_x - x
    dy = target_y - y
    dist = math.sqrt(dx * dx + dy * dy) or 1.0
    # σ grows with distance but also has a random component per step
    sigma = max(0.5, dist * 0.05 * strength)
    nx = x + random.gauss(0, sigma)
    ny = y + random.gauss(0, sigma)
    # Clamp so we don't overshoot dramatically
    max_dev = max(3.0, dist * 0.15)
    nx = max(x - max_dev, min(x + max_dev, nx))
    ny = max(y - max_dev, min(y + max_dev, ny))
    return (nx, ny)


def _perlin_noise_1d(x: float, seed: float) -> float:
    """Simple 1D Perlin-like noise for movement path variation.
    (Formerly labelled "GPS drift" — same math, correct interpretation.)"""
    import hashlib
    # Hash-based pseudo-random gradient
    h = hashlib.md5(f"{x:.4f}:{seed:.4f}".encode()).digest()
    return (int.from_bytes(h[:4], 'big') / 2**32) * 2 - 1  # -1 to 1


def _smooth_noise(x: float, seed: float) -> float:
    """Smooth interpolated noise for movement path variation."""
    ix = math.floor(x)
    fx = x - ix
    # Smoothstep
    fx = fx * fx * (3 - 2 * fx)
    v1 = _perlin_noise_1d(ix, seed)
    v2 = _perlin_noise_1d(ix + 1, seed)
    return v1 + fx * (v2 - v1)


# ── HumanLikenessScorer ──────────────────────────────────────────────────────


class HumanLikenessScorer:
    """Computes a human-likeness score (0.0–1.0) from active behavior stats.

    The score reflects how well the current behavior mix matches real
    human play patterns. It is *not* a pass/fail — it is informational
    for tuning and reporting.
    """

    def __init__(self, profile: BehaviorProfile) -> None:
        self._profile = profile
        self._recent_scores: list[float] = []

    def compute(
        self,
        *,
        mean_reaction_ms: float,
        bad_path_rate: float,
        wrong_target_rate: float,
        afk_break_rate: float,
        fav_spot_rate: float,
        mistake_rate: float,
    ) -> float:
        """Calculate a blended human-likeness score.

        Each sub-score is a sigmoid around an empirically-derived
        "sweet spot" for that behavior. Weights come from the profile.
        """
        w = self._profile.scoring

        def _sigmoid(x: float, midpoint: float, steepness: float = 8.0) -> float:
            return 1.0 / (1.0 + math.exp(-steepness * (x - midpoint)))

        def _gaussian(x: float, mu: float, sigma: float) -> float:
            return math.exp(-0.5 * ((x - mu) / sigma) ** 2)

        r_reaction = _gaussian(mean_reaction_ms / 1000.0, mu=0.35, sigma=0.15)
        r_bad_path = _gaussian(bad_path_rate, mu=0.12, sigma=0.06)
        r_wrong_target = _gaussian(wrong_target_rate, mu=0.10, sigma=0.05)
        r_afk = _gaussian(afk_break_rate, mu=0.06, sigma=0.04)
        r_fav_spot = _gaussian(fav_spot_rate, mu=0.55, sigma=0.15)
        r_mistakes = _gaussian(mistake_rate, mu=0.035, sigma=0.02)

        blended = (
            w.reaction_time_weight * r_reaction
            + w.bad_path_weight * r_bad_path
            + w.wrong_target_weight * r_wrong_target
            + w.afk_break_weight * r_afk
            + w.favorite_spots_weight * r_fav_spot
            + w.micro_mistakes_weight * r_mistakes
        )

        score = max(0.0, min(1.0, blended))
        self._recent_scores.append(score)
        if len(self._recent_scores) > 100:
            self._recent_scores.pop(0)
        return score

    @property
    def rolling_average(self) -> float:
        if not self._recent_scores:
            return 0.0
        return sum(self._recent_scores) / len(self._recent_scores)


# ── BehaviorEngine ───────────────────────────────────────────────────────────


class BehaviorEngine:
    """Loads a YAML behavior profile and injects human-like imperfections.

    Thread-safe: all mutable state is guarded by ``_lock``.

    Supports contextual behavior profiles (ACTIVE, AFK, TIRED, WATCHING)
    that cycle automatically every 15–45 minutes. GM detection can
    override the current profile to WATCHING for near-perfect play.
    """

    def __init__(
        self,
        profile_path: str | Path | None = None,
        bridge_enabled: bool | None = None,
    ) -> None:
        self._lock = RLock()
        self._profile: BehaviorProfile = BehaviorProfile()
        self._scorer: HumanLikenessScorer = HumanLikenessScorer(self._profile)

        # Bridge integration
        self._bridge_enabled = bridge_enabled if bridge_enabled is not None else (
            os.environ.get("ANTI_DETECTION_ENABLED", "1") == "1"
        )

        # ── Contextual profile state ──
        self._current_profile: BehaviorProfileType = BehaviorProfileType.ACTIVE
        self._profile_override: BehaviorProfileType | None = None
        self._next_cycle_time: float = time.time() + random.uniform(900, 2700)

        # ── Per-bot state ──
        self._bot_state: dict[str, dict[str, Any]] = {}

        # ── Session tracking ──
        self._session_start_time: float = time.time()

        # ── Profile load ──
        load_path = Path(profile_path) if profile_path else _DEFAULT_PROFILE_PATH
        self._load_profile(load_path)

        logger.info(
            "BehaviorEngine initialised  profile=%s  bridge_enabled=%s  path=%s",
            self._profile.profile_name,
            self._bridge_enabled,
            load_path,
        )

    # ── Public API ───────────────────────────────────────────────────────────

    def get_behavior_modifier(self, bot_id: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Evaluate all behaviors and return a modifier dict for the bridge.

        The returned dict contains:
        - ``delay_ms``            — suggested reaction delay (ms)
        - ``bad_path``           — path override info
        - ``wrong_target``       — target override info
        - ``afk_break``          — break duration (0 = no break)
        - ``fav_spot``           — preferred coordinates
        - ``micro_mistake``      — mistake to inject
        - ``movement_deviation`` — Gaussian waypoint jitter
        - ``movement_noise``     — Perlin-noise movement offset
        - ``typing_speed``       — typing WPM for chat
        - ``inventory_delay``    — irregular inventory open delay
        - ``logout_delay``       — logout/relog delay
        - ``tick_jitter``        — server tick offset
        - ``human_likeness``     — current score
        - ``behavior_profile``   — current contextual profile name
        """
        with self._lock:
            self._check_cycle()

            ctx = context or {}
            state = self._ensure_bot_state(bot_id)
            result: dict[str, Any] = {
                "delay_ms": 0,
                "bad_path": {"enabled": False},
                "wrong_target": {"enabled": False},
                "afk_break": {"duration_s": 0},
                "fav_spot": {"enabled": False},
                "micro_mistake": {"type": None},
                "movement_deviation": {"enabled": False},
                "movement_noise": {"dx": 0, "dy": 0},
                "typing_speed": {"wpm": 60},
                "inventory_delay": {"delay_ms": 0},
                "logout_delay": {"delay_s": 0},
                "tick_jitter": {"offset_ms": 0},
                "human_likeness": self._scorer.rolling_average,
                "behavior_profile": self._current_profile.value,
            }

            profile_cfg = self._get_contextual_config()

            # ── 1. Reaction time ──
            delay_result = self._eval_reaction_time(bot_id, state, ctx)
            if delay_result.applied:
                result["delay_ms"] = delay_result.delay_ms
                state["last_reaction_delay_ms"] = delay_result.delay_ms

            # ── 2. Bad path ──
            path_result = self._eval_bad_path(bot_id, state, ctx)
            if path_result.applied:
                result["bad_path"] = {
                    "enabled": True,
                    "description": path_result.description,
                    "extra_pct": path_result.metadata.get("extra_pct", 0.0),
                }

            # ── 3. Wrong target ──
            target_result = self._eval_wrong_target(bot_id, state, ctx)
            if target_result.applied:
                result["wrong_target"] = {
                    "enabled": True,
                    "description": target_result.description,
                    "switch_delay_ms": self._profile.wrong_target.switch_delay_ms,
                }

            # ── 4. AFK break ──
            break_result = self._eval_afk_break(bot_id, state, ctx)
            if break_result.applied:
                result["afk_break"] = {
                    "duration_s": break_result.delay_ms / 1000.0,
                    "description": break_result.description,
                }

            # ── 5. Favorite spots ──
            spot_result = self._eval_favorite_spots(bot_id, state, ctx)
            if spot_result.applied:
                result["fav_spot"] = {
                    "enabled": True,
                    "target": spot_result.metadata.get("target"),
                    "description": spot_result.description,
                }

            # ── 6. Micro-mistakes ──
            mistake_result = self._eval_micro_mistakes(bot_id, state, ctx)
            if mistake_result.applied:
                result["micro_mistake"] = {
                    "type": mistake_result.description,
                    "delay_ms": mistake_result.delay_ms,
                    "details": mistake_result.metadata,
                }

            # ── 7. Movement deviation (Gaussian waypoint jitter) ──
            dev_result = self._eval_movement_deviation(bot_id, state, ctx)
            if dev_result.applied:
                result["movement_deviation"] = {
                    "enabled": True,
                    "original": dev_result.metadata.get("original", (0, 0)),
                    "deviated": dev_result.metadata.get("deviated", (0, 0)),
                    "delay_ms": dev_result.delay_ms,
                }

            # ── 8. Movement noise (Perlin-noise-based route variation) ──
            noise_result = self._eval_movement_noise(bot_id, state, ctx)
            if noise_result.applied:
                result["movement_noise"] = {
                    "dx": noise_result.metadata.get("dx", 0),
                    "dy": noise_result.metadata.get("dy", 0),
                }

            # ── 9. Typing speed ──
            typing_result = self._eval_typing_speed(bot_id, state, ctx)
            if typing_result.applied:
                result["typing_speed"] = {
                    "wpm": typing_result.metadata.get("wpm", 60),
                    "mistake": typing_result.metadata.get("mistake", False),
                }

            # ── 10. Inventory timing ──
            inv_result = self._eval_inventory_timing(bot_id, state, ctx)
            if inv_result.applied:
                result["inventory_delay"] = {
                    "delay_ms": inv_result.delay_ms,
                }

            # ── 11. Logout/relog delay ──
            logout_result = self._eval_logout_relog(bot_id, state, ctx)
            if logout_result.applied:
                result["logout_delay"] = {
                    "delay_s": logout_result.delay_ms / 1000.0,
                }

            # ── 12. Server tick jitter ──
            tick_result = self._eval_server_tick(bot_id, state, ctx)
            if tick_result.applied:
                result["tick_jitter"] = {
                    "offset_ms": tick_result.delay_ms,
                }

            # Update human likeness score
            likeliness = self._compute_likeness(state)
            result["human_likeness"] = round(likeliness, 4)

            # Bridge-level anti-detection delay
            if self._bridge_enabled:
                bridge_delay = random.randint(10, 50)
                result["delay_ms"] += bridge_delay

            return result

    def reload_profile(self, profile_path: str | Path | None = None) -> int:
        """Reload configuration from YAML. Returns number of errors (0 = ok)."""
        load_path = Path(profile_path) if profile_path else _DEFAULT_PROFILE_PATH
        if not load_path.exists():
            logger.error("Profile not found: %s", load_path)
            return 1
        self._load_profile(load_path)
        logger.info("Profile reloaded: %s", load_path)
        return 0

    @property
    def profile(self) -> BehaviorProfile:
        with self._lock:
            return self._profile

    @property
    def scorer(self) -> HumanLikenessScorer:
        return self._scorer

    # ── Contextual profile API ──────────────────────────────────────────────

    def cycle_profiles(self) -> BehaviorProfileType:
        """Randomly switch to a different contextual behavior profile."""
        with self._lock:
            if self._profile_override is not None:
                return self._current_profile

            candidates = [
                BehaviorProfileType.ACTIVE,
                BehaviorProfileType.AFK,
                BehaviorProfileType.TIRED,
            ]
            available = [p for p in candidates if p != self._current_profile]
            if not available:
                available = candidates

            new_profile = random.choice(available)
            self._current_profile = new_profile
            self._next_cycle_time = time.time() + random.uniform(900, 2700)

            logger.info(
                "Behavior profile cycled: %s  next_cycle_in=%.0fs",
                new_profile.value,
                self._next_cycle_time - time.time(),
            )
            return new_profile

    def set_profile(self, profile_name: str) -> BehaviorProfileType:
        """Override the current contextual profile (e.g. for GM detection)."""
        with self._lock:
            if not profile_name:
                self._profile_override = None
                logger.info("Behavior profile override cleared — resuming automatic cycling")
                return self._current_profile

            try:
                profile = BehaviorProfileType(profile_name.lower())
            except ValueError:
                logger.warning("Unknown behavior profile '%s' — ignoring", profile_name)
                return self._current_profile

            self._profile_override = profile
            self._current_profile = profile
            logger.info(
                "Behavior profile overridden: %s (GM detection override)",
                profile.value,
            )
            return profile

    def get_current_profile(self) -> dict[str, Any]:
        """Return diagnostic info about the current contextual profile."""
        with self._lock:
            cfg = self._get_contextual_config()
            return {
                "name": self._current_profile.value,
                "overridden": self._profile_override is not None,
                "config": {
                    "reaction_delay_range": cfg.reaction_delay_range,
                    "mistake_rate": cfg.mistake_rate,
                    "wrong_target_chance": cfg.wrong_target_chance,
                    "path_quality": cfg.path_quality,
                    "social_frequency": cfg.social_frequency,
                    "afk_chance": cfg.afk_chance,
                },
                "next_cycle_in": max(0.0, self._next_cycle_time - time.time()),
            }

    # ── Internal: profile cycling ────────────────────────────────────────────

    def _check_cycle(self) -> None:
        if self._profile_override is not None:
            return
        if time.time() >= self._next_cycle_time:
            self.cycle_profiles()

    def _get_contextual_config(self) -> ContextualProfileConfig:
        profile_name = self._current_profile.value
        configs = self._profile.contextual_profiles

        if profile_name in configs:
            return configs[profile_name]

        defaults: dict[str, ContextualProfileConfig] = {
            "active": ContextualProfileConfig(
                reaction_delay_range=[200, 400],
                mistake_rate=0.03,
                wrong_target_chance=0.08,
                path_quality=0.85,
                social_frequency=0.05,
                afk_chance=0.05,
            ),
            "afk": ContextualProfileConfig(
                reaction_delay_range=[800, 1500],
                mistake_rate=0.12,
                wrong_target_chance=0.20,
                path_quality=0.40,
                social_frequency=0.02,
                afk_chance=0.40,
            ),
            "tired": ContextualProfileConfig(
                reaction_delay_range=[500, 1000],
                mistake_rate=0.08,
                wrong_target_chance=0.15,
                path_quality=0.55,
                social_frequency=0.03,
                afk_chance=0.15,
            ),
            "watching": ContextualProfileConfig(
                reaction_delay_range=[100, 200],
                mistake_rate=0.005,
                wrong_target_chance=0.01,
                path_quality=0.98,
                social_frequency=0.01,
                afk_chance=0.0,
            ),
        }
        return defaults.get(profile_name, defaults["active"])

    # ── Profile loading ──────────────────────────────────────────────────────

    def _load_profile(self, path: Path) -> None:
        if not path.exists():
            logger.warning("Behavior profile not found at %s — using defaults", path)
            return

        try:
            import yaml
        except ImportError:
            logger.warning("PyYAML not installed — using default profile")
            return

        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw: dict[str, Any] = yaml.safe_load(fh) or {}
        except Exception:
            logger.exception("Failed to load behavior profile %s", path)
            return

        profile = BehaviorProfile()
        try:
            profile.profile_name = str(raw.get("profile_name", "default"))
            profile.human_likeness_target = float(raw.get("human_likeness_target", 0.78))

            if rt := raw.get("reaction_time"):
                profile.reaction_time.enabled = bool(rt.get("enabled", True))
                profile.reaction_time.min_ms = int(rt.get("min_ms", 100))
                profile.reaction_time.max_ms = int(rt.get("max_ms", 900))
                profile.reaction_time.distribution = str(rt.get("distribution", "gaussian"))
                profile.reaction_time.gaussian_mu = float(rt.get("gaussian_mu", 250.0))
                profile.reaction_time.gaussian_sigma = float(rt.get("gaussian_sigma", 50.0))
                profile.reaction_time.sigma = float(rt.get("sigma", 0.55))
                profile.reaction_time.consecutive_multiplier = float(rt.get("consecutive_multiplier", 0.08))
                profile.reaction_time.fatigue_threshold = int(rt.get("fatigue_threshold", 8))

            if bp := raw.get("bad_path"):
                profile.bad_path.enabled = bool(bp.get("enabled", True))
                profile.bad_path.probability = float(bp.get("probability", 0.12))
                profile.bad_path.extra_distance_pct = float(bp.get("extra_distance_pct", 0.30))
                profile.bad_path.detour_chance = float(bp.get("detour_chance", 0.04))
                profile.bad_path.detour_min_extra_pct = float(bp.get("detour_min_extra_pct", 0.50))
                profile.bad_path.detour_max_extra_pct = float(bp.get("detour_max_extra_pct", 1.50))
                profile.bad_path.recheck_interval_seconds = float(bp.get("recheck_interval_seconds", 15.0))

            if wt := raw.get("wrong_target"):
                profile.wrong_target.enabled = bool(wt.get("enabled", True))
                profile.wrong_target.probability = float(wt.get("probability", 0.08))
                profile.wrong_target.range_pct = float(wt.get("range_pct", 0.35))
                profile.wrong_target.switch_delay_ms = int(wt.get("switch_delay_ms", 600))
                profile.wrong_target.max_wrong_targets = int(wt.get("max_wrong_targets", 3))
                profile.wrong_target.recheck_on_death = bool(wt.get("recheck_on_death", True))

            if ab := raw.get("afk_breaks"):
                profile.afk_breaks.enabled = bool(ab.get("enabled", True))
                profile.afk_breaks.min_break_seconds = int(ab.get("min_break_seconds", 30))
                profile.afk_breaks.max_break_seconds = int(ab.get("max_break_seconds", 120))
                profile.afk_breaks.min_interval_minutes = int(ab.get("min_interval_minutes", 20))
                profile.afk_breaks.max_interval_minutes = int(ab.get("max_interval_minutes", 45))
                profile.afk_breaks.early_break_chance = float(ab.get("early_break_chance", 0.05))
                profile.afk_breaks.resume_delay_ms = int(ab.get("resume_delay_ms", 400))

            if fs := raw.get("favorite_spots"):
                profile.favorite_spots.enabled = bool(fs.get("enabled", True))
                profile.favorite_spots.probability = float(fs.get("probability", 0.55))
                profile.favorite_spots.return_distance = int(fs.get("return_distance", 20))
                profile.favorite_spots.stay_duration_minutes = list(
                    fs.get("stay_duration_minutes", [3, 12])
                )
                raw_spots: dict[str, list[list[int]]] = fs.get("spots", {})
                parsed_spots: dict[str, list[tuple[int, int]]] = {}
                for map_name, coords_list in raw_spots.items():
                    parsed_spots[map_name] = [
                        (int(c[0]), int(c[1])) for c in coords_list if len(c) >= 2
                    ]
                profile.favorite_spots.spots = parsed_spots

            if mm := raw.get("micro_mistakes"):
                profile.micro_mistakes.enabled = bool(mm.get("enabled", True))
                if ww := mm.get("walk_into_wall"):
                    profile.micro_mistakes.walk_into_wall.update(ww)
                if cc := mm.get("cancel_cast"):
                    profile.micro_mistakes.cancel_cast.update(cc)
                if wd := mm.get("wrong_direction"):
                    profile.micro_mistakes.wrong_direction.update(wd)
                if dc := mm.get("double_click"):
                    profile.micro_mistakes.double_click.update(dc)
                if sm := mm.get("skill_missclick"):
                    profile.micro_mistakes.skill_missclick.update(sm)
                if inv := mm.get("inventory_stutter"):
                    profile.micro_mistakes.inventory_stutter.update(inv)

            # ── New behavior configs ──
            if md_cfg := raw.get("movement_deviation"):
                profile.movement_deviation.enabled = bool(md_cfg.get("enabled", True))
                profile.movement_deviation.deviation_strength = float(md_cfg.get("deviation_strength", 1.0))
                profile.movement_deviation.min_deviation_cells = float(md_cfg.get("min_deviation_cells", 0.5))
                profile.movement_deviation.update_every_n_steps = int(md_cfg.get("update_every_n_steps", 3))

            if mn_cfg := raw.get("movement_noise"):
                profile.movement_noise.enabled = bool(mn_cfg.get("enabled", True))
                profile.movement_noise.noise_amplitude = float(mn_cfg.get("noise_amplitude", 3.0))
                profile.movement_noise.noise_frequency = float(mn_cfg.get("noise_frequency", 0.1))
                profile.movement_noise.noise_update_interval_s = float(mn_cfg.get("noise_update_interval_s", 2.0))

            if ts := raw.get("typing_speed"):
                profile.typing_speed.enabled = bool(ts.get("enabled", True))
                profile.typing_speed.wpm_min = int(ts.get("wpm_min", 40))
                profile.typing_speed.wpm_max = int(ts.get("wpm_max", 80))
                profile.typing_speed.mistake_chance = float(ts.get("mistake_chance", 0.03))
                profile.typing_speed.correction_delay_ms = int(ts.get("correction_delay_ms", 300))

            if inv_t := raw.get("inventory_timing"):
                profile.inventory_timing.enabled = bool(inv_t.get("enabled", True))
                profile.inventory_timing.min_interval_s = int(inv_t.get("min_interval_s", 30))
                profile.inventory_timing.max_interval_s = int(inv_t.get("max_interval_s", 180))
                profile.inventory_timing.open_duration_ms_min = int(inv_t.get("open_duration_ms_min", 200))
                profile.inventory_timing.open_duration_ms_max = int(inv_t.get("open_duration_ms_max", 800))

            if lr := raw.get("logout_relog"):
                profile.logout_relog.enabled = bool(lr.get("enabled", True))
                profile.logout_relog.min_delay_s = int(lr.get("min_delay_s", 5))
                profile.logout_relog.max_delay_s = int(lr.get("max_delay_s", 15))
                profile.logout_relog.reconnect_stutter_chance = float(lr.get("reconnect_stutter_chance", 0.10))

            if st := raw.get("server_tick"):
                profile.server_tick.enabled = bool(st.get("enabled", True))
                profile.server_tick.tick_jitter_ms = int(st.get("tick_jitter_ms", 50))
                profile.server_tick.min_tick_offset_ms = int(st.get("min_tick_offset_ms", 10))
                profile.server_tick.max_tick_offset_ms = int(st.get("max_tick_offset_ms", 200))

            if sf := raw.get("session_fatigue"):
                profile.session_fatigue.enabled = bool(sf.get("enabled", True))
                profile.session_fatigue.fatigue_start_minutes = int(sf.get("fatigue_start_minutes", 60))
                profile.session_fatigue.max_fatigue_multiplier = float(sf.get("max_fatigue_multiplier", 2.0))
                profile.session_fatigue.mistake_increase_rate = float(sf.get("mistake_increase_rate", 0.5))

            if sc := raw.get("scoring"):
                profile.scoring.reaction_time_weight = float(sc.get("reaction_time_weight", 0.20))
                profile.scoring.bad_path_weight = float(sc.get("bad_path_weight", 0.15))
                profile.scoring.wrong_target_weight = float(sc.get("wrong_target_weight", 0.15))
                profile.scoring.afk_break_weight = float(sc.get("afk_break_weight", 0.20))
                profile.scoring.favorite_spots_weight = float(sc.get("favorite_spots_weight", 0.10))
                profile.scoring.micro_mistakes_weight = float(sc.get("micro_mistakes_weight", 0.20))

            if cp := raw.get("contextual_profiles"):
                for profile_name, cfg_raw in cp.items():
                    if not isinstance(cfg_raw, dict):
                        continue
                    cfg = ContextualProfileConfig()
                    if "reaction_delay_range" in cfg_raw:
                        cfg.reaction_delay_range = list(cfg_raw["reaction_delay_range"])
                    if "mistake_rate" in cfg_raw:
                        cfg.mistake_rate = float(cfg_raw["mistake_rate"])
                    if "wrong_target_chance" in cfg_raw:
                        cfg.wrong_target_chance = float(cfg_raw["wrong_target_chance"])
                    if "path_quality" in cfg_raw:
                        cfg.path_quality = float(cfg_raw["path_quality"])
                    if "social_frequency" in cfg_raw:
                        cfg.social_frequency = float(cfg_raw["social_frequency"])
                    if "afk_chance" in cfg_raw:
                        cfg.afk_chance = float(cfg_raw["afk_chance"])
                    profile.contextual_profiles[profile_name] = cfg

        except Exception:
            logger.exception("Malformed YAML in %s — using defaults", path)
            return

        with self._lock:
            self._profile = profile
            self._scorer = HumanLikenessScorer(profile)

    # ── Per-bot state ────────────────────────────────────────────────────────

    def _ensure_bot_state(self, bot_id: str) -> dict[str, Any]:
        if bot_id not in self._bot_state:
            self._bot_state[bot_id] = {
                "consecutive_actions": 0,
                "last_action_time": 0.0,
                "last_reaction_delay_ms": 0,
                "last_break_time": 0.0,
                "last_path_recheck": 0.0,
                "wrong_target_count": 0,
                "current_wrong_target": False,
                "fav_spot_arrival_time": 0.0,
                "at_fav_spot": False,
                "mistake_counts": {},
                "total_actions": 0,
                "total_mistakes": 0,
                "total_bad_paths": 0,
                "total_wrong_targets": 0,
                "total_breaks": 0,
                "total_fav_spot_visits": 0,
                "cumulative_reaction_ms": 0.0,
                # New state
                "last_inventory_open": 0.0,
                "last_typing_time": 0.0,
                "last_noise_update": 0.0,
                "noise_offset_x": 0.0,
                "noise_offset_y": 0.0,
                "session_start": time.time(),
            }
        return self._bot_state[bot_id]

    # ── Behavior evaluators ──────────────────────────────────────────────────

    def _get_fatigue_multiplier(self, state: dict[str, Any]) -> float:
        """Calculate fatigue multiplier based on session duration."""
        cfg = self._profile.session_fatigue
        if not cfg.enabled:
            return 1.0
        elapsed_minutes = (time.time() - state.get("session_start", time.time())) / 60.0
        if elapsed_minutes <= cfg.fatigue_start_minutes:
            return 1.0
        fatigue_hours = (elapsed_minutes - cfg.fatigue_start_minutes) / 60.0
        mult = 1.0 + fatigue_hours * cfg.mistake_increase_rate
        return min(cfg.max_fatigue_multiplier, mult)

    def _eval_reaction_time(self, bot_id: str, state: dict[str, Any], ctx: dict[str, Any]) -> BehaviorResult:
        cfg = self._profile.reaction_time
        if not cfg.enabled:
            return BehaviorResult()

        now = time.time()
        elapsed = now - state["last_action_time"]

        if state["last_action_time"] > 0 and elapsed > 2.0:
            state["last_action_time"] = now
            return BehaviorResult()

        state["consecutive_actions"] += 1
        state["last_action_time"] = now

        profile_cfg = self._get_contextual_config()
        min_ms = max(10, profile_cfg.reaction_delay_range[0])
        max_ms = min(2000, profile_cfg.reaction_delay_range[1])
        if max_ms <= min_ms:
            max_ms = min_ms + 50

        # Gaussian distribution (default) — realistic human reaction times
        if cfg.distribution == "gaussian":
            mu = cfg.gaussian_mu
            sigma = cfg.gaussian_sigma
            delay_ms = _gaussian_sample(mu, sigma, float(min_ms), float(max_ms))
        elif cfg.distribution == "log_normal":
            mean_ms = (min_ms + max_ms) / 2.0
            delay_ms = _log_normal_sample(mean_ms, cfg.sigma, float(min_ms), float(max_ms))
        else:
            delay_ms = random.uniform(float(min_ms), float(max_ms))

        # Fatigue — longer delays after many consecutive actions
        consecutive = state["consecutive_actions"]
        if consecutive > cfg.fatigue_threshold:
            extra = (consecutive - cfg.fatigue_threshold) * cfg.consecutive_multiplier
            delay_ms *= 1.0 + extra

        # Session fatigue — longer sessions = slower reactions
        fatigue_mult = self._get_fatigue_multiplier(state)
        delay_ms *= fatigue_mult

        delay_ms = max(float(min_ms), min(float(max_ms), delay_ms))

        state["cumulative_reaction_ms"] += delay_ms
        state["total_actions"] += 1

        return BehaviorResult(
            applied=True,
            description="reaction_delay",
            delay_ms=int(round(delay_ms)),
        )

    def _eval_bad_path(self, bot_id: str, state: dict[str, Any], ctx: dict[str, Any]) -> BehaviorResult:
        cfg = self._profile.bad_path
        if not cfg.enabled:
            return BehaviorResult()

        now = time.time()
        last_check = state.get("last_path_recheck", 0.0)
        if last_check and (now - last_check) < cfg.recheck_interval_seconds:
            return BehaviorResult()

        state["last_path_recheck"] = now

        profile_cfg = self._get_contextual_config()
        adjusted_prob = cfg.probability * (1.0 - profile_cfg.path_quality) * 6.0
        adjusted_prob = max(0.0, min(1.0, adjusted_prob))

        # Session fatigue increases bad path probability
        fatigue_mult = self._get_fatigue_multiplier(state)
        adjusted_prob *= fatigue_mult

        if random.random() >= adjusted_prob:
            return BehaviorResult()

        if random.random() < cfg.detour_chance:
            extra = random.uniform(cfg.detour_min_extra_pct, cfg.detour_max_extra_pct)
        else:
            extra = random.uniform(0.0, cfg.extra_distance_pct)

        state["total_bad_paths"] += 1

        return BehaviorResult(
            applied=True,
            description="suboptimal_path",
            metadata={"extra_pct": round(extra, 3)},
        )

    def _eval_wrong_target(self, bot_id: str, state: dict[str, Any], ctx: dict[str, Any]) -> BehaviorResult:
        cfg = self._profile.wrong_target
        if not cfg.enabled:
            return BehaviorResult()

        if state["wrong_target_count"] >= cfg.max_wrong_targets:
            state["wrong_target_count"] = 0
            return BehaviorResult()

        profile_cfg = self._get_contextual_config()
        adjusted_prob = profile_cfg.wrong_target_chance

        # Session fatigue increases wrong target chance
        fatigue_mult = self._get_fatigue_multiplier(state)
        adjusted_prob *= fatigue_mult

        if random.random() >= adjusted_prob:
            return BehaviorResult()

        state["wrong_target_count"] += 1
        state["total_wrong_targets"] += 1
        state["current_wrong_target"] = True

        return BehaviorResult(
            applied=True,
            description=f"wrong_target_#{state['wrong_target_count']}",
        )

    def _eval_afk_break(self, bot_id: str, state: dict[str, Any], ctx: dict[str, Any]) -> BehaviorResult:
        cfg = self._profile.afk_breaks
        if not cfg.enabled:
            return BehaviorResult()

        now = time.time()
        elapsed = now - state["last_break_time"]

        profile_cfg = self._get_contextual_config()

        min_interval_s = cfg.min_interval_minutes * 60
        early_eligible = elapsed > min_interval_s and random.random() < profile_cfg.afk_chance

        max_interval_s = cfg.max_interval_minutes * 60
        due_for_break = elapsed > random.uniform(min_interval_s, max_interval_s)

        if not (early_eligible or due_for_break):
            return BehaviorResult()

        duration = random.randint(cfg.min_break_seconds, cfg.max_break_seconds)
        state["last_break_time"] = now
        state["total_breaks"] += 1

        return BehaviorResult(
            applied=True,
            description="afk_break",
            delay_ms=duration * 1000,
            metadata={"duration_s": duration, "resume_delay_ms": cfg.resume_delay_ms},
        )

    def _eval_favorite_spots(self, bot_id: str, state: dict[str, Any], ctx: dict[str, Any]) -> BehaviorResult:
        cfg = self._profile.favorite_spots
        if not cfg.enabled:
            return BehaviorResult()

        current_map: str | None = ctx.get("map")
        if not current_map or current_map not in cfg.spots:
            return BehaviorResult()

        if random.random() >= cfg.probability:
            return BehaviorResult()

        spots_for_map = cfg.spots[current_map]
        if not spots_for_map:
            return BehaviorResult()

        target = random.choice(spots_for_map)
        state["total_fav_spot_visits"] += 1

        return BehaviorResult(
            applied=True,
            description=f"fav_spot_{current_map}",
            metadata={"target": target, "map": current_map},
        )

    def _eval_micro_mistakes(self, bot_id: str, state: dict[str, Any], ctx: dict[str, Any]) -> BehaviorResult:
        cfg = self._profile.micro_mistakes
        if not cfg.enabled:
            return BehaviorResult()

        action_kind: str = ctx.get("action_kind", "unknown")

        profile_cfg = self._get_contextual_config()
        mistake_mult = profile_cfg.mistake_rate / 0.03

        # Session fatigue increases mistake rate
        fatigue_mult = self._get_fatigue_multiplier(state)
        mistake_mult *= fatigue_mult

        # ── Walk into wall ──
        if action_kind in ("move", "walk"):
            ww = cfg.walk_into_wall
            if random.random() < ww["probability"] * mistake_mult:
                state["total_mistakes"] += 1
                return BehaviorResult(
                    applied=True,
                    description="walk_into_wall",
                    delay_ms=int(ww["recovery_ms"]),
                    metadata={"stuck_ms": ww["stuck_duration_ms"]},
                )

        # ── Wrong direction ──
        if action_kind in ("move", "walk"):
            wd = cfg.wrong_direction
            if random.random() < wd["probability"] * mistake_mult:
                steps = random.randint(int(wd["wrong_steps"][0]), int(wd["wrong_steps"][1]))
                state["total_mistakes"] += 1
                return BehaviorResult(
                    applied=True,
                    description="wrong_direction",
                    delay_ms=int(wd["correction_delay_ms"]),
                    metadata={"steps": steps},
                )

        # ── Cancel cast ──
        if action_kind in ("skill", "cast"):
            cc = cfg.cancel_cast
            if random.random() < cc["probability"] * mistake_mult:
                state["total_mistakes"] += 1
                return BehaviorResult(
                    applied=True,
                    description="cancel_cast",
                    delay_ms=0,
                    metadata={"cast_time_pct": cc["cast_time_pct"]},
                )

        # ── Skill missclick ──
        if action_kind in ("skill", "cast"):
            sm = cfg.skill_missclick
            if random.random() < sm["probability"] * mistake_mult:
                state["total_mistakes"] += 1
                return BehaviorResult(
                    applied=True,
                    description="skill_missclick",
                    delay_ms=int(sm["recovery_delay_ms"]),
                )

        # ── Inventory stutter ──
        if action_kind in ("loot", "item", "inventory"):
            inv = cfg.inventory_stutter
            if random.random() < inv["probability"] * mistake_mult:
                state["total_mistakes"] += 1
                return BehaviorResult(
                    applied=True,
                    description="inventory_stutter",
                    delay_ms=int(inv["stutter_duration_ms"]),
                )

        # ── Double-click ──
        dc = cfg.double_click
        if random.random() < dc["probability"] * mistake_mult:
            state["total_mistakes"] += 1
            return BehaviorResult(
                applied=True,
                description="double_click",
                metadata={"duplicate_only_if_queued": dc.get("duplicate_only_if_queued", True)},
            )

        return BehaviorResult()

    # ── New behavior evaluators ──────────────────────────────────────────────

    def _eval_movement_deviation(self, bot_id: str, state: dict[str, Any], ctx: dict[str, Any]) -> BehaviorResult:
        """Generate Gaussian deviation for movement waypoints.
        
        Rather than Bézier curves (which are for mouse cursors on a GUI),
        this adds simple Gaussian noise to movement coordinates to simulate
        the slight path wandering humans exhibit. Used by route_humanizer.
        """
        cfg = self._profile.movement_deviation
        if not cfg.enabled:
            return BehaviorResult()

        target_x = ctx.get("target_x")
        target_y = ctx.get("target_y")
        if target_x is None or target_y is None:
            return BehaviorResult()

        current_x = ctx.get("x", 0)
        current_y = ctx.get("y", 0)

        # Apply Gaussian deviation
        noisy_x, noisy_y = _gaussian_deviation_2d(
            float(current_x), float(current_y),
            float(target_x), float(target_y),
            strength=cfg.deviation_strength,
        )

        # Add slight delay proportional to deviation magnitude — humans
        # are slightly slower when correcting a wandering path
        import math
        deviation_mag = math.sqrt(
            (noisy_x - target_x) ** 2 + (noisy_y - target_y) ** 2
        )
        delay_ms = int(deviation_mag * random.uniform(5, 15))

        return BehaviorResult(
            applied=True,
            description="movement_deviation",
            delay_ms=delay_ms,
            metadata={
                "original": (float(target_x), float(target_y)),
                "deviated": (round(noisy_x, 1), round(noisy_y, 1)),
                "deviation_strength": cfg.deviation_strength,
            },
        )

    def _eval_movement_noise(self, bot_id: str, state: dict[str, Any], ctx: dict[str, Any]) -> BehaviorResult:
        """Generate Perlin-noise-based movement noise for route variation.
        
        Produces smooth, natural-feeling offsets at each movement step.
        Unlike the old "GPS drift" name, this is clearly documented as
        movement path noise — same Perlin math, correct interpretation.
        """
        cfg = self._profile.movement_noise
        if not cfg.enabled:
            return BehaviorResult()

        now = time.time()
        last_update = state.get("last_noise_update", 0.0)
        if now - last_update < cfg.noise_update_interval_s:
            return BehaviorResult()

        state["last_noise_update"] = now

        # Use time-based seed for consistent noise per session
        seed = state.get("session_start", now)
        t = now * cfg.noise_frequency

        # Generate smooth noise for x and y offset
        dx = _smooth_noise(t, seed) * cfg.noise_amplitude
        dy = _smooth_noise(t + 1000, seed) * cfg.noise_amplitude

        state["noise_offset_x"] = dx
        state["noise_offset_y"] = dy

        return BehaviorResult(
            applied=True,
            description="movement_noise",
            metadata={"dx": round(dx, 1), "dy": round(dy, 1)},
        )

    def _eval_typing_speed(self, bot_id: str, state: dict[str, Any], ctx: dict[str, Any]) -> BehaviorResult:
        """Generate variable typing speed for chat messages."""
        cfg = self._profile.typing_speed
        if not cfg.enabled:
            return BehaviorResult()

        action_kind = ctx.get("action_kind", "")
        if action_kind not in ("chat", "message", "say", "whisper"):
            return BehaviorResult()

        # Variable WPM
        wpm = random.randint(cfg.wpm_min, cfg.wpm_max)

        # Occasional typing mistake
        mistake = random.random() < cfg.mistake_chance

        state["last_typing_time"] = time.time()

        return BehaviorResult(
            applied=True,
            description="typing_speed",
            delay_ms=cfg.correction_delay_ms if mistake else 0,
            metadata={"wpm": wpm, "mistake": mistake},
        )

    def _eval_inventory_timing(self, bot_id: str, state: dict[str, Any], ctx: dict[str, Any]) -> BehaviorResult:
        """Generate irregular bag/inventory open timing."""
        cfg = self._profile.inventory_timing
        if not cfg.enabled:
            return BehaviorResult()

        action_kind = ctx.get("action_kind", "")
        if action_kind not in ("inventory", "bag", "storage", "cart"):
            return BehaviorResult()

        now = time.time()
        last_open = state.get("last_inventory_open", 0.0)
        elapsed = now - last_open

        # Enforce minimum interval
        if elapsed < cfg.min_interval_s:
            return BehaviorResult()

        # Random delay for opening
        open_delay = random.randint(cfg.open_duration_ms_min, cfg.open_duration_ms_max)

        state["last_inventory_open"] = now

        return BehaviorResult(
            applied=True,
            description="inventory_timing",
            delay_ms=open_delay,
        )

    def _eval_logout_relog(self, bot_id: str, state: dict[str, Any], ctx: dict[str, Any]) -> BehaviorResult:
        """Generate random logout/relog delay."""
        cfg = self._profile.logout_relog
        if not cfg.enabled:
            return BehaviorResult()

        action_kind = ctx.get("action_kind", "")
        if action_kind not in ("logout", "relog", "quit", "exit"):
            return BehaviorResult()

        delay = random.randint(cfg.min_delay_s, cfg.max_delay_s)

        # Occasional reconnect stutter
        if random.random() < cfg.reconnect_stutter_chance:
            delay += random.randint(1, 3)

        return BehaviorResult(
            applied=True,
            description="logout_relog_delay",
            delay_ms=delay * 1000,
        )

    def _eval_server_tick(self, bot_id: str, state: dict[str, Any], ctx: dict[str, Any]) -> BehaviorResult:
        """Generate server tick alignment randomness."""
        cfg = self._profile.server_tick
        if not cfg.enabled:
            return BehaviorResult()

        # Don't apply to every action — only periodic actions
        action_kind = ctx.get("action_kind", "")
        tick_actions = {"attack", "skill", "move", "pickup", "sit", "stand"}
        if action_kind not in tick_actions:
            return BehaviorResult()

        # Only apply occasionally
        if random.random() > 0.3:
            return BehaviorResult()

        offset = random.randint(cfg.min_tick_offset_ms, cfg.max_tick_offset_ms)

        return BehaviorResult(
            applied=True,
            description="tick_jitter",
            delay_ms=offset,
        )

    # ── Scoring ──────────────────────────────────────────────────────────────

    def _compute_likeness(self, state: dict[str, Any]) -> float:
        total = state.get("total_actions", 0)
        if total < 1:
            return 0.5

        mean_reaction = state["cumulative_reaction_ms"] / total
        bad_path_rate = state.get("total_bad_paths", 0) / max(total, 1)
        wrong_target_rate = state.get("total_wrong_targets", 0) / max(total, 1)
        break_rate = state.get("total_breaks", 0) / max(total, 1)
        fav_spot_rate = state.get("total_fav_spot_visits", 0) / max(total, 1)
        mistake_rate = state.get("total_mistakes", 0) / max(total, 1)

        return self._scorer.compute(
            mean_reaction_ms=mean_reaction,
            bad_path_rate=bad_path_rate,
            wrong_target_rate=wrong_target_rate,
            afk_break_rate=break_rate,
            fav_spot_rate=fav_spot_rate,
            mistake_rate=mistake_rate,
        )

    # ── Stats for telemetry / API ────────────────────────────────────────────

    def get_stats(self, bot_id: str) -> dict[str, Any]:
        """Return diagnostic stats for a bot."""
        with self._lock:
            state = self._ensure_bot_state(bot_id)
            p = self._profile
            return {
                "profile_name": p.profile_name,
                "human_likeness_target": p.human_likeness_target,
                "human_likeness_score": self._scorer.rolling_average,
                "bridge_enabled": self._bridge_enabled,
                "behavior_profile": self._current_profile.value,
                "behavior_profile_overridden": self._profile_override is not None,
                "reaction_time": {
                    "enabled": p.reaction_time.enabled,
                    "distribution": p.reaction_time.distribution,
                    "gaussian_mu": p.reaction_time.gaussian_mu,
                    "gaussian_sigma": p.reaction_time.gaussian_sigma,
                    "min_ms": p.reaction_time.min_ms,
                    "max_ms": p.reaction_time.max_ms,
                },
                "session_fatigue": {
                    "enabled": p.session_fatigue.enabled,
                    "fatigue_start_minutes": p.session_fatigue.fatigue_start_minutes,
                    "max_fatigue_multiplier": p.session_fatigue.max_fatigue_multiplier,
                },
                "total_actions": state["total_actions"],
                "total_mistakes": state["total_mistakes"],
                "total_bad_paths": state["total_bad_paths"],
                "total_wrong_targets": state["total_wrong_targets"],
                "total_breaks": state["total_breaks"],
                "total_fav_spot_visits": state["total_fav_spot_visits"],
                "consecutive_actions": state["consecutive_actions"],
                "wrong_target_count": state["wrong_target_count"],
                "at_fav_spot": state["at_fav_spot"],
                "session_elapsed_minutes": round((time.time() - state.get("session_start", time.time())) / 60, 1),
            }


# ── Global singleton ─────────────────────────────────────────────────────────

_engine: BehaviorEngine | None = None
_engine_lock = RLock()


def get_behavior_engine(
    profile_path: str | Path | None = None,
    bridge_enabled: bool | None = None,
) -> BehaviorEngine:
    """Get or create the global BehaviorEngine singleton."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = BehaviorEngine(
                profile_path=profile_path,
                bridge_enabled=bridge_enabled,
            )
        elif profile_path is not None:
            _engine.reload_profile(profile_path)
        return _engine
