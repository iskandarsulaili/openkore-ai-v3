"""Path Predictor — predicts target position using velocity-based extrapolation.

Uses exponential smoothing over position history to estimate velocity,
handles acceleration detection, and provides confidence-based predictions.

Self-* properties:
  - Self-learning: builds velocity models from observed position history
  - Self-adapting: adjusts smoothing factor based on movement pattern consistency
  - Self-initializing: starts predicting from first 2 position observations
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

POSITION_HISTORY_MAXLEN: int = 10
VELOCITY_SMOOTHING_ALPHA: float = 0.4  # Exponential smoothing factor
ACCELERATION_THRESHOLD: float = 5.0  # Cells/s² change = acceleration detected
MIN_CONFIDENCE_POSITIONS: int = 3
MAX_PREDICTION_LOOKAHEAD_MS: float = 5000.0  # 5 sec max prediction
POSITION_STALE_TIMEOUT: float = 10.0  # seconds before dropping old data
MAP_BOUNDARY_MIN: float = 0.0
MAP_BOUNDARY_MAX: float = 512.0  # Default max RO map coordinate


@dataclass
class PositionSnapshot:
    """A single position observation with timestamp."""
    x: float
    y: float
    timestamp: float

    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class TargetMotionProfile:
    """Learned motion profile for a tracked target."""
    target_id: str

    # Position history (most recent at right)
    positions: deque[PositionSnapshot] = field(
        default_factory=lambda: deque(maxlen=POSITION_HISTORY_MAXLEN)
    )

    # Velocity (smoothed)
    vx: float = 0.0
    vy: float = 0.0
    speed: float = 0.0  # magnitude of velocity

    # Acceleration tracking
    prev_vx: float = 0.0
    prev_vy: float = 0.0
    ax: float = 0.0  # acceleration x
    ay: float = 0.0  # acceleration y
    acceleration_magnitude: float = 0.0

    # Movement consistency
    direction_changes: int = 0
    total_observations: int = 0
    avg_speed: float = 0.0
    max_speed: float = 0.0
    stopped_timestamp: float | None = None  # When target last stopped

    # Prediction confidence modifier
    confidence_modifier: float = 1.0

    def add_position(self, x: float, y: float, timestamp: float | None = None) -> None:
        """Add a new position observation and update motion model."""
        t = timestamp or time.time()
        obs = PositionSnapshot(x=x, y=y, timestamp=t)
        self.positions.append(obs)
        self.total_observations += 1

        if len(self.positions) < 2:
            return

        # Calculate instantaneous velocity
        prev = self.positions[-2]
        dt = max(t - prev.timestamp, 0.001)  # prevent div by zero
        inst_vx = (x - prev.x) / dt * 1000.0  # cells per second
        inst_vy = (y - prev.y) / dt * 1000.0

        # Store previous velocity before updating
        self.prev_vx = self.vx
        self.prev_vy = self.vy

        # Exponential smoothing
        if self.total_observations < 3:
            self.vx = inst_vx
            self.vy = inst_vy
        else:
            alpha = VELOCITY_SMOOTHING_ALPHA
            self.vx = alpha * inst_vx + (1.0 - alpha) * self.vx
            self.vy = alpha * inst_vy + (1.0 - alpha) * self.vy

        self.speed = math.sqrt(self.vx ** 2 + self.vy ** 2)

        # Track acceleration
        if self.total_observations >= 3:
            dvx = self.vx - self.prev_vx
            dvy = self.vy - self.prev_vy
            self.ax = dvx / dt * 1000.0
            self.ay = dvy / dt * 1000.0
            self.acceleration_magnitude = math.sqrt(self.ax ** 2 + self.ay ** 2)

        # Check for direction reversal
        if self.total_observations >= 3:
            old_dir = math.atan2(self.prev_vy, self.prev_vx)
            new_dir = math.atan2(self.vy, self.vx)
            dir_change = abs(new_dir - old_dir)
            # Normalize angle difference
            dir_change = min(dir_change, 2 * math.pi - dir_change)
            if dir_change > math.pi / 2:  # > 90° change
                self.direction_changes += 1
                self.confidence_modifier = max(0.2, self.confidence_modifier - 0.15)

        # Update average speed
        if self.total_observations <= 2:
            self.avg_speed = self.speed
        else:
            self.avg_speed = 0.3 * self.speed + 0.7 * self.avg_speed

        self.max_speed = max(self.max_speed, self.speed)

        # Detect stopped state
        if self.speed < 0.5:
            if self.stopped_timestamp is None:
                self.stopped_timestamp = t
        else:
            self.stopped_timestamp = None

    def is_stopped(self) -> bool:
        """Check if target appears to be stationary."""
        return self.stopped_timestamp is not None

    def is_confident(self) -> bool:
        """Check if we have enough data for reliable prediction."""
        return self.total_observations >= MIN_CONFIDENCE_POSITIONS

    def get_stationary_duration(self) -> float:
        """Get how long the target has been stopped (or 0 if moving)."""
        if self.stopped_timestamp is None:
            return 0.0
        return time.time() - self.stopped_timestamp


@dataclass
class PositionPrediction:
    """Result of position prediction for a target."""
    target_id: str
    predicted_x: float
    predicted_y: float
    confidence: float
    lookahead_ms: float
    speed: float
    direction_degrees: float
    is_stopped: bool
    accelerating: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "x": round(self.predicted_x, 1),
            "y": round(self.predicted_y, 1),
            "confidence": round(self.confidence, 3),
            "lookahead_ms": self.lookahead_ms,
            "speed": round(self.speed, 1),
            "direction": round(self.direction_degrees, 1),
            "stopped": self.is_stopped,
            "accelerating": self.accelerating,
        }


class PathPredictor:
    """Predicts target positions using velocity-based linear extrapolation.

    Usage:
        predictor = PathPredictor()

        # Track a target's position (call periodically):
        predictor.update_position("player_abc", 150, 200)

        # Predict where they'll be in 500ms:
        pred = predictor.predict_position("player_abc", lookahead_ms=500)
    """

    def __init__(self) -> None:
        self._lock = RLock()

        # Tracked targets: {target_id: TargetMotionProfile}
        self._targets: dict[str, TargetMotionProfile] = {}

        # Stale cleanup threshold
        self._max_stale_seconds: float = POSITION_STALE_TIMEOUT

    # ── Track updates ───────────────────────────────────────────────────

    def update_position(
        self,
        target_id: str,
        x: float,
        y: float,
        timestamp: float | None = None,
    ) -> None:
        """Record a position observation for a tracked target.

        Args:
            target_id: Unique identifier for the tracked entity
            x: X coordinate (cells)
            y: Y coordinate (cells)
            timestamp: Observation time (default: now)
        """
        t = timestamp or time.time()
        with self._lock:
            if target_id not in self._targets:
                self._targets[target_id] = TargetMotionProfile(target_id=target_id)
            self._targets[target_id].add_position(x, y, t)

    def update_positions_batch(
        self,
        positions: list[tuple[str, float, float]],
        timestamp: float | None = None,
    ) -> None:
        """Efficiently update many targets at once.

        Args:
            positions: List of (target_id, x, y) tuples
            timestamp: Observation time (default: now)
        """
        t = timestamp or time.time()
        with self._lock:
            for target_id, x, y in positions:
                if target_id not in self._targets:
                    self._targets[target_id] = TargetMotionProfile(target_id=target_id)
                self._targets[target_id].add_position(x, y, t)

    # ── Prediction ──────────────────────────────────────────────────────

    def predict_position(
        self,
        target_id: str,
        lookahead_ms: float = 500.0,
    ) -> PositionPrediction:
        """Predict where the target will be in *lookahead_ms* milliseconds.

        Uses velocity-based linear extrapolation with exponential smoothing.
        Confidence decreases with:
          - Fewer position observations
          - High acceleration (changin direction)
          - Long lookahead times
          - Recent direction changes

        Args:
            target_id: The target to predict
            lookahead_ms: How far ahead to predict (ms)

        Returns:
            PositionPrediction with coordinates and confidence
        """
        lookahead_ms = max(0.0, min(lookahead_ms, MAX_PREDICTION_LOOKAHEAD_MS))
        lookahead_s = lookahead_ms / 1000.0

        with self._lock:
            profile = self._targets.get(target_id)
            if profile is None:
                return PositionPrediction(
                    target_id=target_id,
                    predicted_x=0.0,
                    predicted_y=0.0,
                    confidence=0.0,
                    lookahead_ms=lookahead_ms,
                    speed=0.0,
                    direction_degrees=0.0,
                    is_stopped=False,
                    accelerating=False,
                    reason="unknown target",
                )

            if len(profile.positions) == 0:
                return PositionPrediction(
                    target_id=target_id,
                    predicted_x=0.0,
                    predicted_y=0.0,
                    confidence=0.0,
                    lookahead_ms=lookahead_ms,
                    speed=0.0,
                    direction_degrees=0.0,
                    is_stopped=profile.is_stopped(),
                    accelerating=False,
                    reason="no position data",
                )

            # Use most recent known position as base
            last_pos = profile.positions[-1]
            current_x = last_pos.x
            current_y = last_pos.y

            # If target appears stopped, predict current position
            if profile.is_stopped():
                direction = 0.0
                return PositionPrediction(
                    target_id=target_id,
                    predicted_x=current_x,
                    predicted_y=current_y,
                    confidence=0.95,
                    lookahead_ms=lookahead_ms,
                    speed=0.0,
                    direction_degrees=0.0,
                    is_stopped=True,
                    accelerating=False,
                    reason="target is stopped",
                )

            # Linear extrapolation
            pred_x = current_x + profile.vx * lookahead_s
            pred_y = current_y + profile.vy * lookahead_s

            # Clamp to map boundaries
            pred_x = max(MAP_BOUNDARY_MIN, min(MAP_BOUNDARY_MAX, pred_x))
            pred_y = max(MAP_BOUNDARY_MIN, min(MAP_BOUNDARY_MAX, pred_y))

            # ── Compute confidence ──
            confidence = 1.0

            # Few observations = low confidence
            obs_factor = min(1.0, profile.total_observations / 10.0)
            confidence *= obs_factor

            # Direction changes reduce confidence
            if profile.direction_changes > 0:
                dir_factor = max(0.3, 1.0 - profile.direction_changes * 0.1)
                confidence *= dir_factor

            # Acceleration reduces confidence
            accelerating = profile.acceleration_magnitude > ACCELERATION_THRESHOLD
            if accelerating:
                accel_factor = max(0.3, 1.0 - profile.acceleration_magnitude / 50.0)
                confidence *= accel_factor

            # Longer lookahead = lower confidence
            time_factor = max(0.2, 1.0 - lookahead_ms / MAX_PREDICTION_LOOKAHEAD_MS)
            confidence *= time_factor

            # Apply profile modifier
            confidence *= profile.confidence_modifier

            confidence = max(0.0, min(1.0, confidence))

            # Calculate direction
            if profile.speed > 0.1:
                direction = math.degrees(math.atan2(profile.vy, profile.vx))
            else:
                direction = 0.0

            # Build reason
            parts = []
            if profile.total_observations < MIN_CONFIDENCE_POSITIONS:
                parts.append(f"only {profile.total_observations} obs")
            if accelerating:
                parts.append("accelerating")
            if profile.direction_changes > 0:
                parts.append(f"{profile.direction_changes} direction changes")
            reason = "; ".join(parts) if parts else "stable prediction"

            return PositionPrediction(
                target_id=target_id,
                predicted_x=pred_x,
                predicted_y=pred_y,
                confidence=round(confidence, 3),
                lookahead_ms=lookahead_ms,
                speed=round(profile.speed, 1),
                direction_degrees=round(direction, 1),
                is_stopped=False,
                accelerating=accelerating,
                reason=reason,
            )

    def predict_positions_batch(
        self,
        target_ids: list[str],
        lookahead_ms: float = 500.0,
    ) -> dict[str, PositionPrediction]:
        """Predict positions for multiple targets at once."""
        results: dict[str, PositionPrediction] = {}
        with self._lock:
            for tid in target_ids:
                results[tid] = self.predict_position(tid, lookahead_ms)
        return results

    # ── Maintenance ─────────────────────────────────────────────────────

    def prune_stale(self, max_age_seconds: float = POSITION_STALE_TIMEOUT) -> int:
        """Remove targets with no recent position updates.

        Returns:
            Number of targets pruned
        """
        now = time.time()
        stale_ids: list[str] = []
        with self._lock:
            for tid, profile in self._targets.items():
                if profile.positions:
                    last = profile.positions[-1]
                    if now - last.timestamp > max_age_seconds:
                        stale_ids.append(tid)
            for tid in stale_ids:
                del self._targets[tid]
        return len(stale_ids)

    def clear(self) -> None:
        """Remove all tracked targets."""
        with self._lock:
            self._targets.clear()

    # ── Introspection ───────────────────────────────────────────────────

    def get_tracked_count(self) -> int:
        """Number of actively tracked targets."""
        with self._lock:
            return len(self._targets)

    def get_tracked_targets(self) -> list[dict[str, Any]]:
        """List tracked targets with motion stats."""
        with self._lock:
            results: list[dict[str, Any]] = []
            for tid, profile in self._targets.items():
                results.append({
                    "id": tid,
                    "observations": profile.total_observations,
                    "speed": round(profile.speed, 1),
                    "avg_speed": round(profile.avg_speed, 1),
                    "max_speed": round(profile.max_speed, 1),
                    "acceleration": round(profile.acceleration_magnitude, 1),
                    "direction_changes": profile.direction_changes,
                    "stopped": profile.is_stopped(),
                    "confidence": round(profile.confidence_modifier, 2),
                    "last_pos": (
                        round(profile.positions[-1].x, 1),
                        round(profile.positions[-1].y, 1),
                    ) if profile.positions else None,
                })
            return results
