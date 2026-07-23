"""
Behavior Engine — Human-like Imperfect Play
=============================================
Pro-botting insight: perfect play is the #1 detection signal.
This engine injects controlled imperfections that mimic a skilled
but fallible human player.  Configurable per-profile via YAML.

Features
--------
1. Variable reaction times   — log-normal distribution (100–900 ms)
2. Bad path selection         — sub-optimal routes, detours
3. Wrong target selection     — attack non-optimal mob occasionally
4. AFK breaks                 — 30 s – 5 min every 30–90 min
5. Favorite spots             — prefer certain coords on known maps
6. Micro-mistakes             — walk into wall, cancel cast, etc.
7. Contextual behavior profiles — ACTIVE, AFK, TIRED, WATCHING
   cycles every 15–45 min with GM-detection override

Integration
-----------
- Bridge anti-detection (`$ANTI_DETECTION_ENABLED`) is read from
  the sidecar config and amplified with richer per-profile settings.
- The engine exposes a ``get_behavior_modifier()`` dict that the bridge
  polls on each action dispatch.
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
    distribution: str = "log_normal"  # log_normal | uniform
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
    max_break_seconds: int = 300
    min_interval_minutes: int = 30
    max_interval_minutes: int = 90
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


# ── Log-normal helper ────────────────────────────────────────────────────────


def _log_normal_sample(mean_ms: float, sigma: float, min_ms: float, max_ms: float) -> float:
    """Sample from a log-normal distribution, clamped to [min_ms, max_ms].

    The log-normal's scale parameter *μ* is chosen so that the median
    equals *mean_ms*.  The shape *σ* controls tail heaviness.
    """
    mu = math.log(mean_ms) - (sigma ** 2) / 2.0
    raw = random.lognormvariate(mu, sigma)
    return max(min_ms, min(max_ms, raw))


# ── HumanLikenessScorer ──────────────────────────────────────────────────────


class HumanLikenessScorer:
    """Computes a human-likeness score (0.0–1.0) from active behavior stats.

    The score reflects how well the current behavior mix matches real
    human play patterns.  It is *not* a pass/fail — it is informational
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
        "sweet spot" for that behavior.  Weights come from the profile.
        """
        w = self._profile.scoring

        def _sigmoid(x: float, midpoint: float, steepness: float = 8.0) -> float:
            """Logistic function peaked at *midpoint*."""
            return 1.0 / (1.0 + math.exp(-steepness * (x - midpoint)))

        def _gaussian(x: float, mu: float, sigma: float) -> float:
            """Gaussian peak at *mu* — used for metrics where *too much* or
            *too little* are both bad."""
            return math.exp(-0.5 * ((x - mu) / sigma) ** 2)

        # Reaction: optimal human reaction is 200–400 ms average
        r_reaction = _gaussian(mean_reaction_ms / 1000.0, mu=0.35, sigma=0.15)

        # Bad path: ~10–15% of navigation decisions
        r_bad_path = _gaussian(bad_path_rate, mu=0.12, sigma=0.06)

        # Wrong target: ~8–12% of target selections
        r_wrong_target = _gaussian(wrong_target_rate, mu=0.10, sigma=0.05)

        # AFK breaks: most humans take 1-2 breaks per hour, totaling 2-10 min
        r_afk = _gaussian(afk_break_rate, mu=0.06, sigma=0.04)

        # Favorite spots: ~50-70% adherence
        r_fav_spot = _gaussian(fav_spot_rate, mu=0.55, sigma=0.15)

        # Mistakes: ~2-5% of actions
        r_mistakes = _gaussian(mistake_rate, mu=0.035, sigma=0.02)

        blended = (
            w.reaction_time_weight * r_reaction
            + w.bad_path_weight * r_bad_path
            + w.wrong_target_weight * r_wrong_target
            + w.afk_break_weight * r_afk
            + w.favorite_spots_weight * r_fav_spot
            + w.micro_mistakes_weight * r_mistakes
        )

        # Clamp
        score = max(0.0, min(1.0, blended))
        self._recent_scores.append(score)
        # Keep last 100 scores for rolling average
        if len(self._recent_scores) > 100:
            self._recent_scores.pop(0)
        return score

    @property
    def rolling_average(self) -> float:
        """Average of the last 100 scores."""
        if not self._recent_scores:
            return 0.0
        return sum(self._recent_scores) / len(self._recent_scores)


# ── BehaviorEngine ───────────────────────────────────────────────────────────


class BehaviorEngine:
    """Loads a YAML behavior profile and injects human-like imperfections.

    Thread-safe: all mutable state is guarded by ``_lock``.

    Supports contextual behavior profiles (ACTIVE, AFK, TIRED, WATCHING)
    that cycle automatically every 15–45 minutes.  GM detection can
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

        # Bridge integration — read from env / passed value
        self._bridge_enabled = bridge_enabled if bridge_enabled is not None else (
            os.environ.get("ANTI_DETECTION_ENABLED", "1") == "1"
        )

        # ── Contextual profile state ──
        self._current_profile: BehaviorProfileType = BehaviorProfileType.ACTIVE
        self._profile_override: BehaviorProfileType | None = None
        self._next_cycle_time: float = time.time() + random.uniform(900, 2700)  # 15–45 min

        # ── Per-bot state ──
        self._bot_state: dict[str, dict[str, Any]] = {}

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
        """Evaluate all six behaviors and return a modifier dict for the bridge.

        The bridge (aiSidecarBridge.pl) calls this or polls the sidecar API
        before dispatching each action.  The returned dict contains:

        - ``delay_ms``        — suggested reaction delay (ms)
        - ``bad_path``        — path override info
        - ``wrong_target``    — target override info
        - ``afk_break``       — break duration (0 = no break)
        - ``fav_spot``        — preferred coordinates
        - ``micro_mistake``   — mistake to inject
        - ``human_likeness``  — current score
        - ``behavior_profile`` — current contextual profile name
        """
        with self._lock:
            # Check if it's time to cycle profiles
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
                "human_likeness": self._scorer.rolling_average,
                "behavior_profile": self._current_profile.value,
            }

            # Get contextual profile config for current profile
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

            # Update human likeness score
            likeliness = self._compute_likeness(state)
            result["human_likeness"] = round(likeliness, 4)

            # Mix in bridge-level anti-detection delay if the bridge flag is set
            if self._bridge_enabled:
                bridge_delay = random.randint(10, 50)  # Matches bridge defaults
                result["delay_ms"] += bridge_delay

            return result

    def reload_profile(self, profile_path: str | Path | None = None) -> int:
        """Reload configuration from YAML.  Returns number of errors (0 = ok)."""
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
        """Randomly switch to a different contextual behavior profile.

        Picks a random profile different from the current one (excluding
        WATCHING, which is only set via GM detection override).  Resets
        the cycle timer for the next 15–45 min interval.

        Returns the new profile.
        """
        with self._lock:
            # Don't cycle if GM override is active
            if self._profile_override is not None:
                return self._current_profile

            candidates = [
                BehaviorProfileType.ACTIVE,
                BehaviorProfileType.AFK,
                BehaviorProfileType.TIRED,
            ]
            # Remove current to avoid staying on same profile
            available = [p for p in candidates if p != self._current_profile]
            if not available:
                available = candidates

            new_profile = random.choice(available)
            self._current_profile = new_profile
            self._next_cycle_time = time.time() + random.uniform(900, 2700)  # 15–45 min

            logger.info(
                "Behavior profile cycled: %s  next_cycle_in=%.0fs",
                new_profile.value,
                self._next_cycle_time - time.time(),
            )
            return new_profile

    def set_profile(self, profile_name: str) -> BehaviorProfileType:
        """Override the current contextual profile (e.g. for GM detection).

        Accepts any value from BehaviorProfileType (case-insensitive).
        Pass ``None`` or an empty string to clear the override and resume
        automatic cycling.

        Returns the active profile after the change.
        """
        with self._lock:
            if not profile_name:
                # Clear override
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
        """Return diagnostic info about the current contextual profile.

        Returns a dict with:
        - ``name`` — current profile name
        - ``overridden`` — whether a GM detection override is active
        - ``config`` — the current profile's config values
        - ``next_cycle_in`` — seconds until next automatic cycle
        """
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
        """Check if it's time to cycle profiles and do so if needed."""
        if self._profile_override is not None:
            return  # Override active — don't auto-cycle
        if time.time() >= self._next_cycle_time:
            self.cycle_profiles()

    def _get_contextual_config(self) -> ContextualProfileConfig:
        """Get the ContextualProfileConfig for the current profile.

        Falls back to a sensible default if the profile isn't configured
        in the YAML.
        """
        profile_name = self._current_profile.value
        configs = self._profile.contextual_profiles

        if profile_name in configs:
            return configs[profile_name]

        # Fallback defaults per profile type
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
        """Load YAML from *path* and populate self._profile."""
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

            # ── reaction_time ──
            if rt := raw.get("reaction_time"):
                profile.reaction_time.enabled = bool(rt.get("enabled", True))
                profile.reaction_time.min_ms = int(rt.get("min_ms", 100))
                profile.reaction_time.max_ms = int(rt.get("max_ms", 900))
                profile.reaction_time.distribution = str(rt.get("distribution", "log_normal"))
                profile.reaction_time.sigma = float(rt.get("sigma", 0.55))
                profile.reaction_time.consecutive_multiplier = float(rt.get("consecutive_multiplier", 0.08))
                profile.reaction_time.fatigue_threshold = int(rt.get("fatigue_threshold", 8))

            # ── bad_path ──
            if bp := raw.get("bad_path"):
                profile.bad_path.enabled = bool(bp.get("enabled", True))
                profile.bad_path.probability = float(bp.get("probability", 0.12))
                profile.bad_path.extra_distance_pct = float(bp.get("extra_distance_pct", 0.30))
                profile.bad_path.detour_chance = float(bp.get("detour_chance", 0.04))
                profile.bad_path.detour_min_extra_pct = float(bp.get("detour_min_extra_pct", 0.50))
                profile.bad_path.detour_max_extra_pct = float(bp.get("detour_max_extra_pct", 1.50))
                profile.bad_path.recheck_interval_seconds = float(bp.get("recheck_interval_seconds", 15.0))

            # ── wrong_target ──
            if wt := raw.get("wrong_target"):
                profile.wrong_target.enabled = bool(wt.get("enabled", True))
                profile.wrong_target.probability = float(wt.get("probability", 0.08))
                profile.wrong_target.range_pct = float(wt.get("range_pct", 0.35))
                profile.wrong_target.switch_delay_ms = int(wt.get("switch_delay_ms", 600))
                profile.wrong_target.max_wrong_targets = int(wt.get("max_wrong_targets", 3))
                profile.wrong_target.recheck_on_death = bool(wt.get("recheck_on_death", True))

            # ── afk_breaks ──
            if ab := raw.get("afk_breaks"):
                profile.afk_breaks.enabled = bool(ab.get("enabled", True))
                profile.afk_breaks.min_break_seconds = int(ab.get("min_break_seconds", 30))
                profile.afk_breaks.max_break_seconds = int(ab.get("max_break_seconds", 300))
                profile.afk_breaks.min_interval_minutes = int(ab.get("min_interval_minutes", 30))
                profile.afk_breaks.max_interval_minutes = int(ab.get("max_interval_minutes", 90))
                profile.afk_breaks.early_break_chance = float(ab.get("early_break_chance", 0.05))
                profile.afk_breaks.resume_delay_ms = int(ab.get("resume_delay_ms", 400))

            # ── favorite_spots ──
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

            # ── micro_mistakes ──
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

            # ── scoring ──
            if sc := raw.get("scoring"):
                profile.scoring.reaction_time_weight = float(sc.get("reaction_time_weight", 0.20))
                profile.scoring.bad_path_weight = float(sc.get("bad_path_weight", 0.15))
                profile.scoring.wrong_target_weight = float(sc.get("wrong_target_weight", 0.15))
                profile.scoring.afk_break_weight = float(sc.get("afk_break_weight", 0.20))
                profile.scoring.favorite_spots_weight = float(sc.get("favorite_spots_weight", 0.10))
                profile.scoring.micro_mistakes_weight = float(sc.get("micro_mistakes_weight", 0.20))

            # ── contextual_profiles ──
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
            }
        return self._bot_state[bot_id]

    # ── Behavior evaluators ──────────────────────────────────────────────────

    def _eval_reaction_time(self, bot_id: str, state: dict[str, Any], ctx: dict[str, Any]) -> BehaviorResult:
        cfg = self._profile.reaction_time
        if not cfg.enabled:
            return BehaviorResult()

        now = time.time()
        elapsed = now - state["last_action_time"]

        # If we've been idle > 2 s, no artificial delay needed (already "thinking")
        if state["last_action_time"] > 0 and elapsed > 2.0:
            state["last_action_time"] = now
            return BehaviorResult()

        state["consecutive_actions"] += 1
        state["last_action_time"] = now

        # Apply contextual profile reaction delay range
        profile_cfg = self._get_contextual_config()
        min_ms = max(10, profile_cfg.reaction_delay_range[0])
        max_ms = min(2000, profile_cfg.reaction_delay_range[1])
        if max_ms <= min_ms:
            max_ms = min_ms + 50
        mean_ms = (min_ms + max_ms) / 2.0

        # Log-normal or uniform
        if cfg.distribution == "log_normal":
            delay_ms = _log_normal_sample(mean_ms, cfg.sigma, float(min_ms), float(max_ms))
        else:
            delay_ms = random.uniform(float(min_ms), float(max_ms))

        # Fatigue — longer delays after many consecutive actions
        consecutive = state["consecutive_actions"]
        if consecutive > cfg.fatigue_threshold:
            extra = (consecutive - cfg.fatigue_threshold) * cfg.consecutive_multiplier
            delay_ms *= 1.0 + extra
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

        # Don't re-evaluate too often
        now = time.time()
        last_check = state.get("last_path_recheck", 0.0)
        if last_check and (now - last_check) < cfg.recheck_interval_seconds:
            return BehaviorResult()

        state["last_path_recheck"] = now

        # Apply contextual profile path quality modifier
        profile_cfg = self._get_contextual_config()
        # Higher path_quality = lower bad path probability
        adjusted_prob = cfg.probability * (1.0 - profile_cfg.path_quality) * 6.0
        adjusted_prob = max(0.0, min(1.0, adjusted_prob))

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

        # If we already picked wrong targets recently, force-correct after limit
        if state["wrong_target_count"] >= cfg.max_wrong_targets:
            state["wrong_target_count"] = 0
            return BehaviorResult()

        # Apply contextual profile wrong target chance
        profile_cfg = self._get_contextual_config()
        adjusted_prob = profile_cfg.wrong_target_chance

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

        # Apply contextual profile afk chance
        profile_cfg = self._get_contextual_config()

        # Early break with small probability
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

        # Apply contextual profile mistake rate multiplier
        profile_cfg = self._get_contextual_config()
        mistake_mult = profile_cfg.mistake_rate / 0.03  # Normalize to base 3%

        # ── Walk into wall (movement actions) ──
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

        # ── Wrong direction (movement actions) ──
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

        # ── Cancel cast (skill actions) ──
        if action_kind in ("skill", "cast"):
            cc = cfg.cancel_cast
            if random.random() < cc["probability"] * mistake_mult:
                state["total_mistakes"] += 1
                return BehaviorResult(
                    applied=True,
                    description="cancel_cast",
                    delay_ms=0,  # Cancellation happens mid-cast; bridge handles
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

        # ── Double-click (any action) ──
        dc = cfg.double_click
        if random.random() < dc["probability"] * mistake_mult:
            state["total_mistakes"] += 1
            return BehaviorResult(
                applied=True,
                description="double_click",
                metadata={"duplicate_only_if_queued": dc.get("duplicate_only_if_queued", True)},
            )

        return BehaviorResult()

    # ── Scoring ──────────────────────────────────────────────────────────────

    def _compute_likeness(self, state: dict[str, Any]) -> float:
        total = state.get("total_actions", 0)
        if total < 1:
            return 0.5  # Neutral starting score

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
                    "min_ms": p.reaction_time.min_ms,
                    "max_ms": p.reaction_time.max_ms,
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
