"""
Gather-and-kill — runs past monsters to aggro them, then AoEs the pack.

More efficient than fighting one at a time. Gather a pack, then AoE them
all down. This module provides the decision logic for when to gather
vs when to attack.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GatherState:
    """State of the gather-and-kill cycle."""
    phase: str = "attack"  # gather, attack, transition
    aggro_count: int = 0
    target_count: int = 5  # How many to gather before attacking
    max_gather_distance: int = 15  # How far to run to gather
    gathered_at: float = 0.0
    last_attack_at: float = 0.0
    stuck_count: int = 0


@dataclass(slots=True)
class GatherAndKill:
    """Gather-and-kill kiting playstyle."""
    
    _lock: RLock = field(default_factory=RLock)
    _state: GatherState = field(default_factory=GatherState)
    _stats: dict[str, int] = field(default_factory=lambda: {
        "gather_cycles": 0, "aoe_kills": 0, "stuck_resets": 0,
    })
    
    def evaluate(self, aggro_count: int, has_aoe: bool, 
                 hp_pct: float, is_town: bool) -> dict[str, Any]:
        """Evaluate whether to gather or attack."""
        with self._lock:
            now = time.time()
            
            if is_town or hp_pct < 0.3:
                return {"action": "attack", "reason": "safe_mode"}
            
            if not has_aoe:
                return {"action": "attack", "reason": "no_aoe"}
            
            if self._state.phase == "gather":
                # We're gathering — check if we have enough
                if aggro_count >= self._state.target_count:
                    self._state.phase = "attack"
                    self._state.gathered_at = now
                    self._stats["gather_cycles"] += 1
                    return {"action": "attack", "reason": "enough_gathered", "count": aggro_count}
                
                # Check if we've been gathering too long
                if now - self._state.last_attack_at > 10:
                    self._state.stuck_count += 1
                    if self._state.stuck_count > 3:
                        self._state.phase = "attack"
                        self._state.stuck_count = 0
                        self._stats["stuck_resets"] += 1
                        return {"action": "attack", "reason": "stuck_reset"}
                    return {"action": "gather", "reason": "still_gathering", "count": aggro_count}
                
                return {"action": "gather", "reason": "gathering", "count": aggro_count}
            
            else:
                # We're attacking — check if we should gather again
                if aggro_count < 2:
                    # Clear area, start gathering again
                    self._state.phase = "gather"
                    self._state.last_attack_at = now
                    return {"action": "gather", "reason": "area_clear", "count": aggro_count}
                
                return {"action": "attack", "reason": "fighting", "count": aggro_count}
    
    def get_gather_context(self) -> str:
        """Get formatted gather context for LLM prompts."""
        with self._lock:
            return (
                f"── Gather-and-Kill ──\n"
                f"  Phase: {self._state.phase}\n"
                f"  Target count: {self._state.target_count}\n"
                f"  Stuck resets: {self._stats['stuck_resets']}\n"
                f"  Cycles: {self._stats['gather_cycles']}"
            )
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# Global instance
_gather: GatherAndKill | None = None
_gather_lock = RLock()


def get_gather_and_kill() -> GatherAndKill:
    global _gather
    with _gather_lock:
        if _gather is None:
            _gather = GatherAndKill()
        return _gather
