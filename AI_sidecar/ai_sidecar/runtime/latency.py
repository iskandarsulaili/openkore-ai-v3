"""Latency Compensator — adaptive timing for God-tier Pro AI execution.

RO combat runs on 1-second ticks. Client-server latency determines:
- When to stand before moving (anti-sit delay = latency_ms + tick_buffer)
- When to expect skill cast results (cast_time + 2*latency_ms)
- When to retry a failed action (backoff = latency_ms * 1.5)
- When to interrupt a skill vs commit (cast_time_remaining > latency → commit)

This module observes actual round-trip times and adjusts all timing
parameters in real-time. No hardcoded delays.
"""
from __future__ import annotations
import time
import logging
from typing import Any

logger = logging.getLogger(__name__)

class LatencyTracker:
    """Tracks server latency using multiple observation methods.
    
    Methods:
    1. Snapshot→Action poll latency (end-to-end: bridge sends state → sidecar returns action)
    2. Command→ACK latency (bridge executes command → server ACK received)
    3. Estimated from movement (move command → position update ticks)
    """
    
    def __init__(self, window_size: int = 30):
        self._window_size = window_size
        self._samples: list[float] = []
        self._snapshot_ts: float = 0.0
        self._last_ping_ts: float = 0.0
        self._consecutive_timeouts = 0
    
    def record_snapshot_sent(self) -> None:
        """Called when the bridge sends a game state snapshot."""
        self._snapshot_ts = time.time()
    
    def record_action_received(self) -> float | None:
        """Called when an action is returned. Returns estimated RTT."""
        if self._snapshot_ts > 0:
            rtt = (time.time() - self._snapshot_ts) * 1000  # ms
            self._add_sample(rtt)
            self._consecutive_timeouts = 0
            return rtt
        return None
    
    def record_command_ack(self, cmd_latency_ms: float) -> None:
        """Called when a command is acknowledged by the server."""
        self._add_sample(cmd_latency_ms)
        self._consecutive_timeouts = 0
    
    def record_timeout(self) -> None:
        """Called when a request times out."""
        self._consecutive_timeouts += 1
        # On consecutive timeouts, add pessimistic estimate
        if self._consecutive_timeouts > 3:
            self._add_sample(1000.0)  # Assume 1s latency on repeated timeouts
    
    def _add_sample(self, ms: float) -> None:
        """Add a latency sample with bounds checking."""
        ms = max(20.0, min(5000.0, ms))  # Clamp 20ms-5000ms
        self._samples.append(ms)
        if len(self._samples) > self._window_size:
            self._samples.pop(0)
    
    def get_stats(self) -> dict[str, float]:
        """Get latency statistics."""
        if not self._samples:
            return {"avg_ms": 200.0, "min_ms": 200.0, "max_ms": 200.0, "p50_ms": 200.0, "p95_ms": 200.0}
        
        sorted_samples = sorted(self._samples)
        n = len(sorted_samples)
        return {
            "avg_ms": sum(sorted_samples) / n,
            "min_ms": sorted_samples[0],
            "max_ms": sorted_samples[-1],
            "p50_ms": sorted_samples[n // 2],
            "p95_ms": sorted_samples[int(n * 0.95)],
        }
    
    def get_latency_ms(self) -> float:
        """Get the best estimate of current server latency."""
        if not self._samples:
            return 200.0  # Default: 200ms (average international ping)
        stats = self.get_stats()
        return stats["p95_ms"]  # Use p95 to account for worst-case
    
    def get_tick_buffer(self) -> int:
        """Get the number of RO ticks (1s each) to buffer for latency."""
        latency = self.get_latency_ms()
        # At 50ms latency, buffer = 1 tick (just enough)
        # At 300ms latency, buffer = 3 ticks
        # At 1000ms latency, buffer = 8 ticks (needs extra)
        return max(1, int(latency / 150) + 1)
    
    def get_cast_compensation(self, cast_time_ms: float) -> float:
        """Get adjusted cast time factoring in latency.
        
        Returns new cast time in ms. If latency > cast_time_remaining,
        we should commit (cast will complete on server side before we
        see the result). If latency < cast_time_remaining, we can
        still interrupt.
        """
        latency = self.get_latency_ms()
        if latency >= cast_time_ms:
            # Server already considers the cast done
            return 0.0
        return cast_time_ms - latency * 0.5  # Half-latency adjustment
    
    def get_move_compensation(self) -> float:
        """Get move delay compensation in seconds.
        
        When bridge sends a move command, the server needs:
        - latency_ms to receive the command
        - 1 tick (~1s) to process the move
        - latency_ms to send the position update back
        
        Total: 2 * latency_ms + 1000ms
        
        We adjust the "stand before move" timing to account for this.
        """
        return (self.get_latency_ms() * 2 + 1000) / 1000  # seconds
    
    def get_backoff_ms(self, attempt: int = 0) -> float:
        """Get exponential backoff for retries (ms)."""
        base = self.get_latency_ms() * 0.5
        return min(base * (2 ** attempt), 10000)  # Cap at 10s


class SkillCastPlanner:
    """Plans skill cast timings with latency awareness.
    
    RO skill mechanics:
    - Cast time: time before skill fires (interruptible if hit)
    - Skill delay: cooldown after cast completes
    - After-cast delay: animation lock (can't act)
    
    Latency adds to all of these. This planner calculates when
    a skill WILL land on the server, not when we fire it locally.
    """
    
    # Pre-renewal cast time formula: cast_time = base_cast * (1 - DEX/150)
    # Minimum cast time: 0.2s (with 95+ DEX = instant)
    @staticmethod
    def calculate_effective_cast_time(
        base_cast_ms: float,
        dex: int,
        latency_ms: float,
        is_instant: bool = False
    ) -> float:
        """Calculate when a skill actually lands on the server.
        
        Args:
            base_cast_ms: Base cast time from skill data (ms)
            dex: Caster's DEX stat
            latency_ms: Current server latency
            is_instant: Whether skill is flagged as instant-cast
            
        Returns:
            Effective cast time in ms (when skill lands on server)
        """
        if is_instant:
            return latency_ms * 0.5  # Just network latency
        
        # Pre-renewal: cast_time = base * (1 - dex/150), min 200ms
        cast_reduction = max(0.0, min(1.0, dex / 150.0))
        cast_ms = max(200.0, base_cast_ms * (1.0 - cast_reduction))
        
        # Add latency: server needs to receive the cast request
        # before the skill starts casting on server-side
        effective = cast_ms + latency_ms * 0.3
        
        return effective
    
    @staticmethod
    def should_interrupt_cast(
        cast_start_time: float,
        effective_cast_ms: float,
        latency_ms: float,
        danger_level: float
    ) -> bool:
        """Determine if a skill cast should be interrupted.
        
        Args:
            cast_start_time: When cast started (time.time())
            effective_cast_ms: Effective cast duration (ms)
            latency_ms: Current latency
            danger_level: 0.0 = safe, 1.0 = critical
            
        Returns:
            True if cast should be interrupted
        """
        elapsed = (time.time() - cast_start_time) * 1000
        remaining = effective_cast_ms - elapsed
        
        if danger_level > 0.8 and remaining > latency_ms:
            # Critical danger AND we can still interrupt (remaining > latency)
            return True
        
        if remaining <= latency_ms:
            # Cast will complete before we can interrupt it
            # Commit to the cast
            return False
        
        # Default: don't interrupt if >50% complete
        if elapsed > effective_cast_ms * 0.5:
            return False
        
        return True


# Global instances
_latency_tracker: LatencyTracker | None = None
_cast_planner: SkillCastPlanner | None = None

def get_latency_tracker() -> LatencyTracker:
    global _latency_tracker
    if _latency_tracker is None:
        _latency_tracker = LatencyTracker()
    return _latency_tracker

def get_cast_planner() -> SkillCastPlanner:
    global _cast_planner
    if _cast_planner is None:
        _cast_planner = SkillCastPlanner()
    return _cast_planner
