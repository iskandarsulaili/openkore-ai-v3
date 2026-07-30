"""
Route Humanizer — adds Gaussian noise to movement waypoints.
================================================================

The detection signal: bots walk in straight lines between exact coordinates.
Humans walk in slightly curved paths, overshoot targets slightly, correct back.

This module:
1. Takes a route (list of waypoints) and injects Gaussian noise
2. Reads deviation parameters from the BehaviorEngine's movement_deviation config
3. Also applies Perlin-noise-based movement noise for smooth, natural variation
4. NEVER uses Bézier curves (those are for mouse cursors on GUI applications)
5. Provides both per-waypoint and per-route humanization

Integration:
- bridge_wiring.py calls humanize_route() before sending movement commands
- route_humanizer.py reads movement_deviation and movement_noise from
  the BehaviorEngine's get_behavior_modifier() output
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from ai_sidecar.anti_detection.behavior_engine import (
    BehaviorEngine,
    _gaussian_deviation_2d,
    _smooth_noise,
    get_behavior_engine,
)

logger = logging.getLogger(__name__)

# ── Default noise seeds for Perlin-based path variation ──────────────────────

_DEFAULT_NOISE_SEED_X = 42.0
_DEFAULT_NOISE_SEED_Y = 137.0


@dataclass
class RouteHumanizerConfig:
    """Configuration for route humanization behavior."""
    enabled: bool = True
    deviation_strength: float = 1.0       # Multiplier on Gaussian noise
    noise_amplitude: float = 3.0          # Perlin noise amplitude (cells)
    noise_frequency: float = 0.1          # Perlin noise frequency
    overshoot_chance: float = 0.10        # Chance of overshooting target
    overshoot_cells: int = 2              # Max overshoot distance (cells)
    correction_delay_ms: int = 350        # Delay when correcting overshoot


@dataclass
class HumanizedRoute:
    """Result of humanizing a route."""
    original_waypoints: list[tuple[float, float]]
    humanized_waypoints: list[tuple[float, float]]
    deviation_count: int = 0
    overshoot_count: int = 0
    extra_delay_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class RouteHumanizer:
    """Adds human-like path variation to movement waypoints.

    Usage::

        humanizer = RouteHumanizer()
        route = [(100, 200), (150, 250), (200, 300)]
        result = humanizer.humanize("bot1", route, {"map": "prontera"})
        # Use result.humanized_waypoints instead of original_waypoints

        # Or humanize a single waypoint:
        noisy_x, noisy_y = humanizer.humanize_waypoint(
            "bot1", current_x, current_y, target_x, target_y
        )
    """

    def __init__(
        self,
        engine: BehaviorEngine | None = None,
        config: RouteHumanizerConfig | None = None,
    ) -> None:
        self._lock = RLock()
        self._engine = engine or get_behavior_engine()
        self._config = config or RouteHumanizerConfig()
        self._noise_offsets: dict[str, dict[str, float]] = {}

    # ── Public API ───────────────────────────────────────────────────────────

    def humanize(
        self,
        bot_id: str,
        waypoints: list[tuple[float, float]],
        context: dict[str, Any] | None = None,
    ) -> HumanizedRoute:
        """Take a list of waypoints and return humanized (noisy) waypoints.

        Args:
            bot_id: Bot identifier (for consistent noise per bot).
            waypoints: List of (x, y) tuples representing the route.
            context: Optional context dict (map, action_kind, etc.).

        Returns:
            HumanizedRoute with modified waypoints and metadata.
        """
        if not self._config.enabled:
            return HumanizedRoute(
                original_waypoints=waypoints,
                humanized_waypoints=list(waypoints),
            )

        if len(waypoints) < 2:
            return HumanizedRoute(
                original_waypoints=waypoints,
                humanized_waypoints=list(waypoints),
            )

        # Get behavior modifier for this bot
        modifier = self._engine.get_behavior_modifier(bot_id, context or {})
        deviation_cfg = modifier.get("movement_deviation", {})
        noise_cfg = modifier.get("movement_noise", {})

        humanized: list[tuple[float, float]] = []
        deviation_count = 0
        overshoot_count = 0

        for i, (x, y) in enumerate(waypoints):
            if i == 0:
                # First waypoint is current position — keep as-is
                humanized.append((x, y))
                continue

            prev_x, prev_y = humanized[-1]

            # 1. Apply Gaussian deviation (waypoint jitter)
            if deviation_cfg.get("enabled", self._config.enabled):
                noisy_x, noisy_y = _gaussian_deviation_2d(
                    prev_x, prev_y, x, y,
                    strength=self._config.deviation_strength,
                )
                deviation_count += 1
            else:
                noisy_x, noisy_y = x, y

            # 2. Apply Perlin movement noise (smooth path variation)
            if noise_cfg.get("dx", 0) != 0 or noise_cfg.get("dy", 0) != 0:
                dx = noise_cfg.get("dx", 0)
                dy = noise_cfg.get("dy", 0)
                noisy_x += dx * 0.1  # Small fraction of noise amplitude
                noisy_y += dy * 0.1

            # 3. Overshoot last waypoint sometimes
            is_last = (i == len(waypoints) - 1)
            if is_last and random.random() < self._config.overshoot_chance:
                overshoot_x = random.uniform(
                    -self._config.overshoot_cells, self._config.overshoot_cells
                )
                overshoot_y = random.uniform(
                    -self._config.overshoot_cells, self._config.overshoot_cells
                )
                # Overshoot past the target, then add correction point
                humanized.append((noisy_x + overshoot_x, noisy_y + overshoot_y))
                humanized.append((noisy_x, noisy_y))  # Correct back
                overshoot_count += 1
                continue

            humanized.append((noisy_x, noisy_y))

        # Calculate extra delay from deviations
        extra_delay_ms = deviation_count * random.randint(10, 30)
        extra_delay_ms += overshoot_count * self._config.correction_delay_ms

        return HumanizedRoute(
            original_waypoints=list(waypoints),
            humanized_waypoints=humanized,
            deviation_count=deviation_count,
            overshoot_count=overshoot_count,
            extra_delay_ms=extra_delay_ms,
            metadata={
                "deviation_strength": self._config.deviation_strength,
                "noise_amplitude": self._config.noise_amplitude,
            },
        )

    def humanize_waypoint(
        self,
        bot_id: str,
        current_x: float,
        current_y: float,
        target_x: float,
        target_y: float,
    ) -> tuple[float, float]:
        """Humanize a single waypoint — returns (noisy_x, noisy_y)."""
        if not self._config.enabled:
            return (target_x, target_y)

        modifier = self._engine.get_behavior_modifier(bot_id)
        deviation_cfg = modifier.get("movement_deviation", {})
        if not deviation_cfg.get("enabled", True):
            return (target_x, target_y)

        noisy_x, noisy_y = _gaussian_deviation_2d(
            current_x, current_y, target_x, target_y,
            strength=self._config.deviation_strength,
        )
        return (noisy_x, noisy_y)

    @property
    def config(self) -> RouteHumanizerConfig:
        return self._config

    @config.setter
    def config(self, value: RouteHumanizerConfig) -> None:
        with self._lock:
            self._config = value


# ── Global singleton ─────────────────────────────────────────────────────────

_humanizer: RouteHumanizer | None = None
_humanizer_lock = RLock()


def get_route_humanizer(
    engine: BehaviorEngine | None = None,
) -> RouteHumanizer:
    """Get or create the global RouteHumanizer singleton."""
    global _humanizer
    with _humanizer_lock:
        if _humanizer is None:
            _humanizer = RouteHumanizer(engine=engine)
        return _humanizer
