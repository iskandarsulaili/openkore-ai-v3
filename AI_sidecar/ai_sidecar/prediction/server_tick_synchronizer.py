"""Server Tick Synchronizer — aligns action timing with RO server ticks.

RO Server operates in ~20ms ticks (50 ticks/sec). Actions sent between ticks
wait for the next tick, causing irregular latency (0-20ms jitter). This module
learns the actual server tick timing by measuring response pattern intervals
and provides optimal send timing for timing-critical skills.

Self-* properties:
  - Self-learning: learns server tick timing from observed response patterns
  - Self-adapting: adjusts to network jitter and server load variation
  - Self-optimizing: finds optimal send offset within tick window for critical skills
"""

from __future__ import annotations

import logging
import math
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

EXPECTED_TICK_MS: float = 20.0  # RO standard tick (50 Hz)
TICK_JITTER_TOLERANCE_MS: float = 5.0
MEASUREMENT_WINDOW: int = 100  # Number of samples for tick timing calculation
MIN_SAMPLES_FOR_RELIABLE: int = 10
DECAY_ALPHA: float = 0.15
LATENCY_SAMPLE_MAXLEN: int = 200


@dataclass
class TickMeasurement:
    """Measurement of one server tick interval."""
    interval_ms: float
    timestamp: float
    is_outlier: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "interval_ms": round(self.interval_ms, 2),
            "is_outlier": self.is_outlier,
        }


@dataclass
class LatencySample:
    """Measured round-trip latency for a packet."""
    rtt_ms: float
    packet_type: str
    timestamp: float
    server_tick_aligned: bool  # Was this sent aligned to a server tick?

    def to_dict(self) -> dict[str, Any]:
        return {
            "rtt_ms": round(self.rtt_ms, 1),
            "type": self.packet_type,
            "tick_aligned": self.server_tick_aligned,
        }


class ServerTickSynchronizer:
    """Learns server tick timing for optimal action alignment.

    RO servers process actions in ~20ms tick intervals. Actions arriving
    between ticks are queued for the next tick, introducing variable delay
    (0-20ms). By aligning sends to tick boundaries, timing-critical skills
    (Asura Strike, Storm Gust, interrupt attempts) get consistent latency.

    Usage:
        sync = ServerTickSynchronizer()

        # After receiving a server response, measure the interval:
        sync.record_packet_interval(interval_ms=19.8)

        # Before sending a critical action:
        estimated_latency = sync.get_estimated_latency()
        aligned = sync.should_send_now()  # Returns True if aligned to tick

        # For critical skills:
        wait_ms = sync.align_to_tick()  # How long to wait for optimal alignment
    """

    def __init__(self) -> None:
        self._lock = RLock()

        # Tick interval measurements
        self._tick_intervals: deque[TickMeasurement] = deque(
            maxlen=MEASUREMENT_WINDOW
        )

        # Learned tick rate (ms)
        self._avg_tick_ms: float = EXPECTED_TICK_MS
        self._std_tick_ms: float = 2.0
        self._min_tick_ms: float = 15.0
        self._max_tick_ms: float = 25.0
        self._tick_observations: int = 0

        # Latency samples
        self._latency_samples: deque[LatencySample] = deque(
            maxlen=LATENCY_SAMPLE_MAXLEN
        )

        # Smoothed latency estimates
        self._avg_rtt_ms: float = 50.0  # Reasonable default
        self._min_rtt_ms: float = float('inf')
        self._max_rtt_ms: float = 0.0
        self._jitter_ms: float = 0.0

        # Timing state
        self._last_packet_time: float = 0.0
        self._last_tick_time: float = 0.0
        self._tick_offset_ms: float = 0.0  # Phase offset within tick cycle

        # Tick phase tracking: we learn where in the tick cycle we are
        self._phase_observations: list[float] = []

        # Stats
        self._total_packets: int = 0
        self._aligned_sends: int = 0
        self._start_time: float = time.time()

    # ── Measurement ─────────────────────────────────────────────────────

    def record_packet_interval(self, interval_ms: float) -> None:
        """Record interval between consecutive server packets.

        Args:
            interval_ms: Time between receiving two server response packets
        """
        interval_ms = max(1.0, min(interval_ms, 200.0))  # Sanity clamp

        with self._lock:
            is_outlier = self._is_outlier_interval(interval_ms)
            measurement = TickMeasurement(
                interval_ms=interval_ms,
                timestamp=time.time(),
                is_outlier=is_outlier,
            )
            self._tick_intervals.append(measurement)

            if not is_outlier:
                self._tick_observations += 1
                self._update_tick_stats(interval_ms)

                # Track tick offset phase
                now = time.time()
                if self._last_packet_time > 0:
                    dt_ms = (now - self._last_packet_time) * 1000.0
                    # Phase = where we are in the tick cycle
                    phase = dt_ms % self._avg_tick_ms
                    self._phase_observations.append(phase)
                    if len(self._phase_observations) > 20:
                        self._phase_observations = self._phase_observations[-20:]

                self._last_packet_time = now

            self._total_packets += 1

    def record_latency_sample(
        self,
        rtt_ms: float,
        packet_type: str = "unknown",
    ) -> None:
        """Record a round-trip latency measurement.

        Args:
            rtt_ms: Round-trip time in milliseconds
            packet_type: Type of packet (e.g. 'attack', 'skill', 'move')
        """
        rtt_ms = max(0.0, min(rtt_ms, 5000.0))  # Sanity clamp

        with self._lock:
            sample = LatencySample(
                rtt_ms=rtt_ms,
                packet_type=packet_type,
                timestamp=time.time(),
                server_tick_aligned=self._was_send_aligned(),
            )
            self._latency_samples.append(sample)

            # Update running statistics
            if self._avg_rtt_ms == 0.0:
                self._avg_rtt_ms = rtt_ms
            else:
                self._avg_rtt_ms = (
                    DECAY_ALPHA * rtt_ms
                    + (1.0 - DECAY_ALPHA) * self._avg_rtt_ms
                )

            if rtt_ms < self._min_rtt_ms:
                self._min_rtt_ms = rtt_ms
            if rtt_ms > self._max_rtt_ms:
                self._max_rtt_ms = rtt_ms

            # Jitter = mean absolute deviation of recent samples
            recent = [s.rtt_ms for s in list(self._latency_samples)[-20:]]
            if len(recent) >= 5:
                mean = sum(recent) / len(recent)
                self._jitter_ms = sum(abs(r - mean) for r in recent) / len(recent)

    def _update_tick_stats(self, interval_ms: float) -> None:
        """Update learned tick rate statistics."""
        if self._tick_observations <= 1:
            self._avg_tick_ms = interval_ms
        else:
            # Weighted update — trust expected tick as prior initially
            prior_weight = max(0.0, 1.0 - self._tick_observations / 50.0)
            blended = (
                (1.0 - prior_weight) * self._avg_tick_ms
                + prior_weight * EXPECTED_TICK_MS
            )
            self._avg_tick_ms = (
                DECAY_ALPHA * interval_ms + (1.0 - DECAY_ALPHA) * blended
            )

        # Update bounds
        self._min_tick_ms = min(self._min_tick_ms, interval_ms)
        self._max_tick_ms = max(self._max_tick_ms, interval_ms)

        # Std from recent window
        recent_intervals = [m.interval_ms for m in self._tick_intervals if not m.is_outlier]
        if len(recent_intervals) >= MIN_SAMPLES_FOR_RELIABLE:
            self._std_tick_ms = statistics.stdev(recent_intervals[-20:])

    def _is_outlier_interval(self, interval_ms: float) -> bool:
        """Check if an interval is likely noise (not a real tick)."""
        if self._tick_observations < 5:
            # Too few samples — accept anything around expected tick
            return abs(interval_ms - EXPECTED_TICK_MS) > TICK_JITTER_TOLERANCE_MS * 4
        return abs(interval_ms - self._avg_tick_ms) > self._std_tick_ms * 3

    def _was_send_aligned(self) -> bool:
        """Check if the last send was aligned to a server tick."""
        # This returns whether we think we're currently aligned
        # Used for tracking alignment effectiveness
        with self._lock:
            now = time.time()
            if self._last_packet_time == 0:
                return False
            elapsed_ms = (now - self._last_packet_time) * 1000.0
            phase = elapsed_ms % self._avg_tick_ms
            return phase < 5.0 or phase > self._avg_tick_ms - 5.0

    # ── Timing alignment ────────────────────────────────────────────────

    def align_to_tick(self) -> float:
        """Calculate optimal wait time to align send with next server tick.

        Returns:
            Milliseconds to wait before sending for optimal tick alignment.
            Returns 0.0 if we're already aligned.
        """
        with self._lock:
            if self._tick_observations < MIN_SAMPLES_FOR_RELIABLE:
                # Don't know tick timing yet — don't delay
                return 0.0

            now = time.time()
            if self._last_packet_time == 0:
                return 0.0

            # Time since last packet
            elapsed_ms = (now - self._last_packet_time) * 1000.0

            # Where we are in the tick cycle
            phase_ms = elapsed_ms % self._avg_tick_ms

            # Time until next tick boundary
            wait_ms = self._avg_tick_ms - phase_ms

            # If we're within 2ms of a tick, send now (close enough)
            if wait_ms < 2.0 or wait_ms > self._avg_tick_ms - 2.0:
                return 0.0

            # Add a small random offset if we don't have good phase tracking
            # to avoid contending for the same tick slot as other clients
            if self._tick_observations < 20:
                wait_ms += (hash(str(now)) % 3) * 0.5

            return round(max(0.0, wait_ms), 1)

    def get_estimated_latency(self, packet_type: str = "any") -> float:
        """Get estimated current latency for timing-critical decisions.

        Args:
            packet_type: Optional packet type filter. If provided, returns
                         latency estimate specific to that packet type based
                         on historical samples.

        Returns:
            Estimated round-trip latency in milliseconds
        """
        with self._lock:
            if packet_type != "any":
                # Try to get type-specific latency
                recent = [
                    s for s in self._latency_samples
                    if s.packet_type == packet_type
                ][-10:]
                if recent:
                    return sum(s.rtt_ms for s in recent) / len(recent)

            # Use smoothed overall estimate
            return round(self._avg_rtt_ms + self._jitter_ms * 0.5, 1)

    def should_send_now(self) -> bool:
        """Check if current timing is aligned with a server tick.

        Returns True if we're within 3ms of a tick boundary.
        """
        wait_ms = self.align_to_tick()
        return wait_ms == 0.0

    # ── High-level utilities ────────────────────────────────────────────

    def record_send(self) -> None:
        """Record that we just sent a packet (for alignment tracking)."""
        with self._lock:
            self._last_packet_time = time.time()
            if self._was_send_aligned():
                self._aligned_sends += 1

    def estimate_skill_land_time(
        self,
        cast_time_ms: float,
        packet_type: str = "skill",
    ) -> float:
        """Estimate total time for a skill to land.

        Accounts for tick alignment + cast time + latency.

        Args:
            cast_time_ms: Skill cast time in milliseconds
            packet_type: Type of packet for latency estimation

        Returns:
            Estimated total milliseconds until skill lands
        """
        tick_wait = self.align_to_tick()
        latency = self.get_estimated_latency(packet_type)
        total_ms = tick_wait + cast_time_ms + latency
        return round(total_ms, 1)

    # ── Introspection ───────────────────────────────────────────────────

    def get_tick_info(self) -> dict[str, Any]:
        """Get learned server tick timing information."""
        with self._lock:
            return {
                "avg_tick_ms": round(self._avg_tick_ms, 2),
                "std_tick_ms": round(self._std_tick_ms, 2),
                "min_tick_ms": round(self._min_tick_ms, 2),
                "max_tick_ms": round(self._max_tick_ms, 2),
                "observations": self._tick_observations,
                "reliable": self._tick_observations >= MIN_SAMPLES_FOR_RELIABLE,
                "expected_tick_ms": EXPECTED_TICK_MS,
            }

    def get_latency_info(self) -> dict[str, Any]:
        """Get learned latency statistics."""
        with self._lock:
            return {
                "avg_rtt_ms": round(self._avg_rtt_ms, 1),
                "jitter_ms": round(self._jitter_ms, 1),
                "min_rtt_ms": round(self._min_rtt_ms, 1) if self._min_rtt_ms != float('inf') else 0.0,
                "max_rtt_ms": round(self._max_rtt_ms, 1),
                "samples": len(self._latency_samples),
                "aligned_send_ratio": round(
                    self._aligned_sends / max(self._total_packets, 1), 3
                ),
            }

    def get_stats(self) -> dict[str, Any]:
        """Summary statistics for introspection."""
        tick_info = self.get_tick_info()
        latency_info = self.get_latency_info()
        with self._lock:
            return {
                **tick_info,
                **latency_info,
                "total_packets": self._total_packets,
                "uptime_s": round(time.time() - self._start_time, 1),
            }
