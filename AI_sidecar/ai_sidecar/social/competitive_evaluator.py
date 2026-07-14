"""
Competitive evaluation — decides when to compete, share, or flee.

When another farmer arrives at your spot, a real player evaluates them.
They check gear, level, class. They decide if they can out-farm them,
if it's worth fighting over the spot, or if they should move on.

This module evaluates competition and recommends responses.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Competitor:
    """Profile of a competing farmer."""
    name: str
    level: int = 0
    class_name: str = ""
    gear_score: float = 0.0  # Estimated based on visible info
    aggressive: bool = False  # Have they attacked us?
    map: str = ""
    spotted_at: float = 0.0
    encounters: int = 0


@dataclass(slots=True)
class CompetitiveEvaluator:
    """Evaluates competition and recommends responses."""
    
    _lock: RLock = field(default_factory=RLock)
    _competitors: dict[str, Competitor] = field(default_factory=dict)
    _stats: dict[str, int] = field(default_factory=lambda: {
        "evaluations": 0, "fights": 0, "retreats": 0, "shares": 0,
    })
    
    def spot_competitor(self, name: str, map: str, level: int = 0, class_name: str = "") -> None:
        """Record a competing farmer at our spot."""
        with self._lock:
            comp = self._competitors.setdefault(name, Competitor(name=name))
            comp.level = level or comp.level
            comp.class_name = class_name or comp.class_name
            comp.map = map
            comp.spotted_at = time.time()
            comp.encounters += 1
            self._stats["evaluations"] += 1
    
    def record_aggression(self, name: str) -> None:
        """Record that this competitor attacked us."""
        with self._lock:
            comp = self._competitors.get(name)
            if comp:
                comp.aggressive = True
    
    def evaluate(self, name: str) -> str:
        """Evaluate a competitor and recommend a response.
        
        Returns: 'fight', 'share', 'flee', or 'observe'
        """
        with self._lock:
            comp = self._competitors.get(name)
            if comp is None:
                return "observe"
            
            # If aggressive, always flee or fight
            if comp.aggressive:
                if comp.gear_score > 0.7:  # Stronger than us
                    self._stats["retreats"] += 1
                    return "flee"
                else:
                    self._stats["fights"] += 1
                    return "fight"
            
            # If we've seen them many times, share the spot
            if comp.encounters > 5:
                self._stats["shares"] += 1
                return "share"
            
            # Default: observe and see what they do
            return "observe"
    
    def get_competition_context(self) -> str:
        """Get formatted competition context for LLM prompts."""
        with self._lock:
            now = time.time()
            recent = {k: v for k, v in self._competitors.items() 
                     if now - v.spotted_at < 600}  # Last 10 minutes
            
            lines = ["── Competition Report ──"]
            if not recent:
                lines.append("  No recent competitors.")
                return "\n".join(lines)
            
            lines.append(f"  Active competitors: {len(recent)}")
            for comp in sorted(recent.values(), key=lambda c: -c.encounters)[:5]:
                rec = self.evaluate(comp.name)
                lines.append(
                    f"  {comp.name} (Lv.{comp.level} {comp.class_name}) "
                    f"on {comp.map} — encounters: {comp.encounters} "
                    f"→ recommend: {rec}"
                )
            
            return "\n".join(lines)
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# Global instance
_evaluator: CompetitiveEvaluator | None = None
_evaluator_lock = RLock()


def get_competitive_evaluator() -> CompetitiveEvaluator:
    global _evaluator
    with _evaluator_lock:
        if _evaluator is None:
            _evaluator = CompetitiveEvaluator()
        return _evaluator
