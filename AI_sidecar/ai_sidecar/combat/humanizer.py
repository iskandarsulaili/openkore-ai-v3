"""
Human-like Behavior Patterns — randomized pathing, variable reaction times,
and natural-looking movement that mimics a skilled human, not a robot.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from threading import RLock

logger = logging.getLogger(__name__)


@dataclass
class HumanProfile:
    """Profile for human-like behavior."""
    reaction_time_min_ms: int = 50
    reaction_time_max_ms: int = 200
    path_randomization_pct: float = 0.15
    click_variance_px: int = 3
    movement_jitter_pct: float = 0.1
    skill_delay_variance_ms: int = 100
    look_around_interval_s: float = 8.0
    look_around_duration_ms: int = 300
    typo_chance: float = 0.02
    pause_chance: float = 0.05
    pause_duration_ms: int = 500


class Humanizer:
    """Adds human-like randomness to bot behavior."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._profile = HumanProfile()
        self._last_look_around: float = 0.0
        self._consecutive_perfect_moves: int = 0

    def get_reaction_delay(self) -> float:
        """Get a random reaction delay in seconds."""
        with self._lock:
            return random.uniform(
                self._profile.reaction_time_min_ms,
                self._profile.reaction_time_max_ms,
            ) / 1000.0

    def jitter_position(self, x: int, y: int) -> tuple[int, int]:
        """Add slight jitter to a position to avoid perfect straight lines."""
        with self._lock:
            variance = self._profile.click_variance_px
            jx = x + random.randint(-variance, variance)
            jy = y + random.randint(-variance, variance)
            return (jx, jy)

    def randomize_path(self, waypoints: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Add randomization to a path to avoid predictable routes."""
        with self._lock:
            if not waypoints:
                return waypoints
            result: list[tuple[int, int]] = []
            jitter = self._profile.path_randomization_pct
            for wx, wy in waypoints:
                dx = int(wx * jitter * random.uniform(-1, 1))
                dy = int(wy * jitter * random.uniform(-1, 1))
                result.append((wx + dx, wy + dy))
            return result

    def add_skill_delay_variance(self, base_delay_ms: int) -> int:
        """Add variance to skill timing."""
        with self._lock:
            variance = self._profile.skill_delay_variance_ms
            return base_delay_ms + random.randint(-variance, variance)

    def should_look_around(self) -> bool:
        """Check if we should simulate looking around."""
        with self._lock:
            now = time.time()
            if now - self._last_look_around > self._profile.look_around_interval_s:
                self._last_look_around = now
                return True
            return False

    def should_pause(self) -> bool:
        """Check if we should simulate a brief pause (like a human thinking)."""
        with self._lock:
            return random.random() < self._profile.pause_chance

    def get_pause_duration(self) -> float:
        """Get pause duration in seconds."""
        with self._lock:
            return self._profile.pause_duration_ms / 1000.0

    def should_make_typo(self) -> bool:
        """Check if we should simulate a typo in chat."""
        with self._lock:
            return random.random() < self._profile.typo_chance

    def make_typo(self, text: str) -> str:
        """Introduce a typo into text using keyboard proximity."""
        if not text or len(text) < 3:
            return text
        if random.random() > self._profile.typo_chance:
            return text
        # Keyboard proximity map for realistic typos
        proximity = {
            'q': 'w', 'w': 'qe', 'e': 'wr', 'r': 'et', 't': 'ry',
            'y': 'tu', 'u': 'yi', 'i': 'uo', 'o': 'ip', 'p': 'o',
            'a': 's', 's': 'ad', 'd': 'sf', 'f': 'dg', 'g': 'fh',
            'h': 'gj', 'j': 'hk', 'k': 'jl', 'l': 'k',
            'z': 'x', 'x': 'zc', 'c': 'xv', 'v': 'cb', 'b': 'vn',
            'n': 'bm', 'm': 'n',
        }
        pos = random.randint(0, len(text) - 1)
        char = text[pos].lower()
        if char in proximity:
            replacement = random.choice(proximity[char])
            if text[pos].isupper():
                replacement = replacement.upper()
            return text[:pos] + replacement + text[pos + 1:]
        # Fallback: double a character
        if random.random() < 0.3:
            return text[:pos] + text[pos] + text[pos:]
        return text

    def record_perfect_move(self) -> None:
        """Track consecutive perfect moves to increase randomization."""
        with self._lock:
            self._consecutive_perfect_moves += 1
            if self._consecutive_perfect_moves > 5:
                # Increase randomization after too many perfect moves
                self._profile.path_randomization_pct = min(
                    0.3, self._profile.path_randomization_pct * 1.2
                )

    def record_imperfect_move(self) -> None:
        """Reset perfect move counter."""
        with self._lock:
            self._consecutive_perfect_moves = 0
            self._profile.path_randomization_pct = 0.15

    def get_movement_speed_variance(self, base_speed: float) -> float:
        """Add variance to movement speed."""
        with self._lock:
            jitter = self._profile.movement_jitter_pct
            return base_speed * (1 + random.uniform(-jitter, jitter))

    def get_profile(self) -> HumanProfile:
        with self._lock:
            return self._profile

    def set_profile(self, profile: HumanProfile) -> None:
        with self._lock:
            self._profile = profile


# ── Global Singleton ──

_humanizer: Humanizer | None = None
_humanizer_lock = RLock()


def get_humanizer() -> Humanizer:
    global _humanizer
    with _humanizer_lock:
        if _humanizer is None:
            _humanizer = Humanizer()
        return _humanizer
