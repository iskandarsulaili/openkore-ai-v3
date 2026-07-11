"""
Anti-Detection Module — Human-like behavior simulation.
======================================================
Pro players know: getting banned is the real cost. This module makes bots
behave like humans to avoid detection.

Features:
- Random delays between actions (human reaction time)
- Varied movement patterns (not straight-line)
- Session rotation (avoid 24/7 uptime patterns)
- Chat simulation (idle chatter)
- Weight simulation (not always optimal)
- Error simulation (occasional misclicks)
"""

from __future__ import annotations

import logging
from threading import RLock
import random
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HumanProfile:
    """A human player profile for behavior simulation."""
    reaction_time_ms: tuple[int, int] = (150, 400)  # Human reaction time
    typing_speed_cpm: tuple[int, int] = (200, 400)  # Characters per minute
    movement_variance: float = 0.3  # How much to deviate from optimal path
    idle_chance: float = 0.05  # Chance to idle briefly
    idle_duration_ms: tuple[int, int] = (2000, 8000)
    error_chance: float = 0.02  # Chance of "misclick"
    chat_frequency_minutes: tuple[int, int] = (5, 30)  # Chat every 5-30 min
    session_length_hours: tuple[float, float] = (2.0, 6.0)  # Session length
    break_frequency_minutes: tuple[int, int] = (45, 120)  # Break every 45-120 min
    break_duration_minutes: tuple[int, int] = (2, 10)  # Break for 2-10 min


class AntiDetection:
    """Human-like behavior simulation for bot anti-detection.

    Each bot gets its own profile with randomized parameters.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._lock = RLock()
        self._profiles: dict[str, HumanProfile] = {}
        self._session_start: dict[str, float] = {}
        self._last_chat: dict[str, float] = {}
        self._last_break: dict[str, float] = {}
        self._last_action: dict[str, float] = {}
        self._consecutive_actions: dict[str, int] = {}
        self._chat_messages = [
            "lol", "gg", "nice", "brb", "afk", "omw",
            "ty", "np", "wow", "omg", "lmao", "rip",
            "gl", "hf", "ez", "wp", "noob", "pro",
            "where to hunt?", "any party?", "selling stuff",
            "need buff pls", "ty for party", "gotta go soon",
            "one more run", "lagging", "dc'd", "back",
        ]

    def ensure_profile(self, bot_id: str) -> HumanProfile:
        """Create or get a human profile for a bot.

        Each bot gets randomized parameters for unique behavior.
        """
        if bot_id not in self._profiles:
            profile = HumanProfile(
                reaction_time_ms=(
                    random.randint(100, 300),
                    random.randint(300, 600),
                ),
                typing_speed_cpm=(
                    random.randint(150, 250),
                    random.randint(300, 500),
                ),
                movement_variance=random.uniform(0.2, 0.5),
                idle_chance=random.uniform(0.03, 0.08),
                idle_duration_ms=(
                    random.randint(1000, 3000),
                    random.randint(5000, 12000),
                ),
                error_chance=random.uniform(0.01, 0.04),
                chat_frequency_minutes=(
                    random.randint(3, 10),
                    random.randint(15, 45),
                ),
                session_length_hours=(
                    random.uniform(1.5, 3.0),
                    random.uniform(4.0, 8.0),
                ),
                break_frequency_minutes=(
                    random.randint(30, 60),
                    random.randint(90, 180),
                ),
                break_duration_minutes=(
                    random.randint(1, 3),
                    random.randint(5, 15),
                ),
            )
            self._profiles[bot_id] = profile
            self._session_start[bot_id] = time.time()
            self._last_chat[bot_id] = time.time()
            self._last_break[bot_id] = time.time()
            self._last_action[bot_id] = time.time()
            self._consecutive_actions[bot_id] = 0
            logger.info("Created human profile for bot %s", bot_id)

        return self._profiles[bot_id]

    def should_delay(self, bot_id: str) -> float:
        """Get a random delay in seconds before the next action.

        Simulates human reaction time.
        """
        if not self.enabled:
            return 0.0

        profile = self.ensure_profile(bot_id)
        now = time.time()
        elapsed = now - self._last_action.get(bot_id, now)

        # If we've been idle, no delay needed (already "thinking")
        if elapsed > 2.0:
            self._last_action[bot_id] = now
            return 0.0

        # Human reaction time
        min_ms, max_ms = profile.reaction_time_ms
        delay = random.randint(min_ms, max_ms) / 1000.0

        # Add variance for consecutive actions (fatigue)
        consecutive = self._consecutive_actions.get(bot_id, 0)
        if consecutive > 5:
            delay *= 1.0 + (consecutive - 5) * 0.1
        if consecutive > 20:
            delay *= 2.0  # Double delay after 20 consecutive actions

        self._last_action[bot_id] = now
        self._consecutive_actions[bot_id] = consecutive + 1
        return delay

    def should_idle(self, bot_id: str) -> float:
        """Check if the bot should idle briefly.

        Returns idle duration in seconds, or 0 if no idle.
        """
        if not self.enabled:
            return 0.0

        profile = self.ensure_profile(bot_id)
        if random.random() < profile.idle_chance:
            min_ms, max_ms = profile.idle_duration_ms
            return random.randint(min_ms, max_ms) / 1000.0
        return 0.0

    def should_take_break(self, bot_id: str) -> float:
        """Check if the bot should take a break.

        Returns break duration in seconds, or 0 if no break needed.
        """
        if not self.enabled:
            return 0.0

        profile = self.ensure_profile(bot_id)
        now = time.time()
        elapsed = now - self._last_break.get(bot_id, now)
        min_min, max_min = profile.break_frequency_minutes

        if elapsed > random.randint(min_min, max_min) * 60:
            min_dur, max_dur = profile.break_duration_minutes
            duration = random.randint(min_dur, max_dur) * 60
            self._last_break[bot_id] = now
            logger.info("Bot %s taking %ds break (anti-detection)", bot_id, duration)
            return duration
        return 0.0

    def should_end_session(self, bot_id: str) -> bool:
        """Check if the session should end (session rotation)."""
        if not self.enabled:
            return False

        profile = self.ensure_profile(bot_id)
        now = time.time()
        elapsed_hours = (now - self._session_start.get(bot_id, now)) / 3600
        min_hours, max_hours = profile.session_length_hours

        if elapsed_hours > random.uniform(min_hours, max_hours):
            logger.info("Bot %s session ended after %.1fh (anti-detection)", bot_id, elapsed_hours)
            return True
        return False

    def should_chat(self, bot_id: str) -> str | None:
        """Check if the bot should send a chat message.

        Returns the message or None.
        """
        if not self.enabled:
            return None

        profile = self.ensure_profile(bot_id)
        now = time.time()
        elapsed = now - self._last_chat.get(bot_id, now)
        min_min, max_min = profile.chat_frequency_minutes

        if elapsed > random.randint(min_min, max_min) * 60:
            self._last_chat[bot_id] = now
            return random.choice(self._chat_messages)
        return None

    def should_simulate_error(self) -> bool:
        """Check if we should simulate a human error."""
        if not self.enabled:
            return False
        return random.random() < 0.02  # 2% chance

    def get_movement_variance(self, bot_id: str) -> float:
        """Get movement path variance for this bot."""
        if not self.enabled:
            return 0.0
        profile = self.ensure_profile(bot_id)
        return profile.movement_variance

    def get_stats(self, bot_id: str) -> dict[str, Any]:
        """Get anti-detection stats for a bot."""
        profile = self.ensure_profile(bot_id)
        now = time.time()
        session_hours = (now - self._session_start.get(bot_id, now)) / 3600
        return {
            "enabled": self.enabled,
            "session_hours": round(session_hours, 1),
            "reaction_time_ms": profile.reaction_time_ms,
            "movement_variance": profile.movement_variance,
            "idle_chance": profile.idle_chance,
            "error_chance": profile.error_chance,
            "consecutive_actions": self._consecutive_actions.get(bot_id, 0),
        }
