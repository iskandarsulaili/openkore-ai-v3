"""
Command Pacing — ensures commands are sent with human-like timing jitter.
========================================================================

The problem: bots send commands with <50ms between identical commands.
The server logs time between packets. Humans have irregular timing
(200-800ms between actions). Bots have suspiciously uniform timing.

This module:
1. Provides per-bot pacing profiles (fast, average, cautious, slow)
2. Adds random jitter within human-normal ranges
3. Tracks recent command timestamps to ensure natural distribution
4. Prevents burst sequences of identical commands
5. Adds fatigue scaling for longer sessions (slower pacing over time)

Integration with the BehaviorEngine:
- Reads the ``delay_ms`` value from the engine's behavior modifier
- Applies contextual profile scaling (ACTIVE/AFK/TIRED/WATCHING)
- Can be used standalone or as a library by bridge_wiring.py
"""

from __future__ import annotations

import logging
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

# ── Command pacing tiers (ms between commands) ───────────────────────────────
# These approximate observed human behavior in RO:
#   Fast/aggressive:    200-400ms  (twitchy player, focused grinding)
#   Average/balanced:   300-600ms  (typical player)
#   Cautious:           400-800ms  (careful player, reads between actions)
#   Slow/lazy:          500-1000ms (distracted or tired player)
#   Inconsistent:       200-700ms  (variable, hard to profile)

_CMD_PACING_TIERS: dict[str, tuple[int, int]] = {
    "fast":       (200, 400),
    "average":    (300, 600),
    "cautious":   (400, 800),
    "slow":       (500, 1000),
    "inconsistent": (200, 700),
}

_DEFAULT_TIER = "average"


@dataclass
class PacingProfile:
    """Per-bot command pacing configuration."""
    tier: str = _DEFAULT_TIER
    min_delay_ms: int = 200
    max_delay_ms: int = 600

    # Fatigue scaling: increase delays as session progresses
    fatigue_enabled: bool = True
    fatigue_start_minutes: int = 60
    fatigue_max_multiplier: float = 1.8
    fatigue_rate_per_hour: float = 0.3

    # Burst protection: prevent too many commands in quick succession
    burst_window_seconds: float = 5.0
    max_commands_in_burst_window: int = 15

    # Identical command protection: don't send same command too fast
    identical_cmd_cooldown_ms: int = 800

    def get_delay_range(self) -> tuple[int, int]:
        """Get the base delay range for this profile."""
        return (self.min_delay_ms, self.max_delay_ms)


# ── Pre-defined pacing profiles ─────────────────────────────────────────────

_PACING_PROFILES: dict[str, PacingProfile] = {
    "fast": PacingProfile(
        tier="fast",
        min_delay_ms=200,
        max_delay_ms=400,
        max_commands_in_burst_window=20,
    ),
    "average": PacingProfile(
        tier="average",
        min_delay_ms=300,
        max_delay_ms=600,
    ),
    "cautious": PacingProfile(
        tier="cautious",
        min_delay_ms=400,
        max_delay_ms=800,
        max_commands_in_burst_window=10,
    ),
    "slow": PacingProfile(
        tier="slow",
        min_delay_ms=500,
        max_delay_ms=1000,
        max_commands_in_burst_window=8,
    ),
    "inconsistent": PacingProfile(
        tier="inconsistent",
        min_delay_ms=200,
        max_delay_ms=700,
    ),
}


class CommandPacer:
    """Ensures commands are sent with human-like timing jitter.

    Usage::

        pacer = CommandPacer(bot_id="bot1")
        delay_ms = pacer.get_delay("attack")
        # sleep(delay_ms / 1000)
        # send command

        # Or check if we should skip this command entirely
        if pacer.should_throttle("attack"):
            return  # skip this cycle
    """

    def __init__(
        self,
        bot_id: str = "default",
        profile: str | PacingProfile | None = None,
    ) -> None:
        self._bot_id = bot_id
        self._lock = RLock()

        # Load profile
        if isinstance(profile, PacingProfile):
            self._profile = profile
        elif isinstance(profile, str):
            self._profile = _PACING_PROFILES.get(profile, PacingProfile())
        else:
            # Deterministic per-bot
            self._profile = self._auto_select_profile(bot_id)

        # Command history (for burst detection)
        self._cmd_history: deque[tuple[float, str]] = deque(maxlen=50)
        self._last_identical_cmd_time: dict[str, float] = {}

        # Session tracking
        self._session_start: float = time.time()

    # ── Public API ───────────────────────────────────────────────────────────

    def get_delay(self, command_type: str = "general") -> int:
        """Get the recommended delay (ms) before sending a command.

        Args:
            command_type: Kind of command (attack, skill, move, heal, etc.)
                          Different types may have slightly different pacing.
        Returns:
            Delay in milliseconds.
        """
        with self._lock:
            min_ms, max_ms = self._profile.get_delay_range()

            # Fatigue scaling
            if self._profile.fatigue_enabled:
                fatigue_mult = self._get_fatigue_multiplier()
                max_ms = int(max_ms * fatigue_mult)

            # Pick a delay within range
            delay = random.randint(min_ms, max_ms)

            # Type-specific adjustments
            type_mult = _TYPE_MULTIPLIERS.get(command_type, 1.0)
            delay = int(delay * type_mult)

            # Log
            self._cmd_history.append((time.time(), command_type))

            return max(50, delay)

    def should_throttle(self, command_type: str) -> bool:
        """Check if a command should be throttled (skipped this cycle).

        Returns True if the bot would be sending too many commands too fast
        — in which case the caller should skip this cycle.
        """
        with self._lock:
            now = time.time()

            # Identical command cooldown
            last_time = self._last_identical_cmd_time.get(command_type, 0.0)
            elapsed_ms = (now - last_time) * 1000
            if elapsed_ms < self._profile.identical_cmd_cooldown_ms:
                return True

            # Burst window check
            window_start = now - self._profile.burst_window_seconds
            recent_count = sum(
                1 for t, _ in self._cmd_history if t >= window_start
            )
            if recent_count >= self._profile.max_commands_in_burst_window:
                return True

            # Record this command
            self._last_identical_cmd_time[command_type] = now
            return False

    def get_fatigue_description(self) -> dict[str, Any]:
        """Return current fatigue state for diagnostics."""
        elapsed_minutes = (time.time() - self._session_start) / 60.0
        mult = self._get_fatigue_multiplier()
        return {
            "session_minutes": round(elapsed_minutes, 1),
            "fatigue_multiplier": round(mult, 2),
            "pacing_tier": self._profile.tier,
            "delay_range_ms": list(self._profile.get_delay_range()),
        }

    def set_profile(self, profile: str | PacingProfile) -> None:
        """Change the pacing profile at runtime."""
        with self._lock:
            if isinstance(profile, PacingProfile):
                self._profile = profile
            else:
                self._profile = _PACING_PROFILES.get(
                    profile, _PACING_PROFILES[_DEFAULT_TIER]
                )

    # ── Internal ─────────────────────────────────────────────────────────────

    def _get_fatigue_multiplier(self) -> float:
        """Calculate fatigue multiplier based on session duration."""
        cfg = self._profile
        if not cfg.fatigue_enabled:
            return 1.0
        elapsed_minutes = (time.time() - self._session_start) / 60.0
        if elapsed_minutes <= cfg.fatigue_start_minutes:
            return 1.0
        fatigue_hours = (elapsed_minutes - cfg.fatigue_start_minutes) / 60.0
        mult = 1.0 + fatigue_hours * cfg.fatigue_rate_per_hour
        return min(cfg.fatigue_max_multiplier, mult)

    @staticmethod
    def _auto_select_profile(bot_id: str) -> PacingProfile:
        """Deterministically select a pacing profile based on bot name."""
        import hashlib
        seed = int(hashlib.sha256(bot_id.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        tier_names = list(_PACING_PROFILES.keys())
        tier = rng.choice(tier_names)
        profile = _PACING_PROFILES[tier]
        logger.info(
            "command_pacing: auto_selected bot=%s tier=%s range=%d-%dms",
            bot_id, tier, profile.min_delay_ms, profile.max_delay_ms,
        )
        return PacingProfile(
            tier=tier,
            min_delay_ms=profile.min_delay_ms,
            max_delay_ms=profile.max_delay_ms,
            max_commands_in_burst_window=profile.max_commands_in_burst_window,
        )


# ── Command type pacing multipliers ──────────────────────────────────────────
# Different command types have slightly different natural pacing.
# Heals are faster (reflex), movement is slower (deliberate).

_TYPE_MULTIPLIERS: dict[str, float] = {
    # Fast reactions
    "heal": 0.7,
    "emergency": 0.5,
    "flee": 0.6,
    "teleport": 0.6,
    # Normal
    "attack": 1.0,
    "skill": 1.0,
    "pickup": 1.0,
    "loot": 1.0,
    "sit": 1.0,
    "stand": 1.0,
    # Slower (deliberate actions)
    "move": 1.2,
    "walk": 1.2,
    "chat": 1.5,
    "whisper": 1.3,
    "trade": 1.3,
    "inventory": 1.2,
    "storage": 1.2,
    "npc": 1.4,
    # Slowest
    "macro": 1.1,
    "login": 2.0,
    "logout": 2.0,
    "relog": 3.0,
}


# ── Global singleton ─────────────────────────────────────────────────────────

_pacers: dict[str, CommandPacer] = {}
_pacers_lock = RLock()


def get_command_pacer(bot_id: str = "default") -> CommandPacer:
    """Get or create a CommandPacer for a specific bot."""
    with _pacers_lock:
        if bot_id not in _pacers:
            _pacers[bot_id] = CommandPacer(bot_id=bot_id)
        return _pacers[bot_id]
