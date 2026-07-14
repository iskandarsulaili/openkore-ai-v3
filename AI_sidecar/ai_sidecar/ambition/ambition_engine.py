"""
Ambition engine — long-term goals and aspirations using LLM.

A top player has ambition. Dominate an MVP. Control a market. Build
the strongest guild. Be the richest player. Win Hall of Fame.

This module stores long-term ambitions, tracks progress toward them,
and uses the LLM to set priorities and adapt goals as conditions change.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class Ambition:
    """A long-term ambition with progress tracking."""
    name: str
    category: str  # wealth, pvp, economy, social, collection, mastery
    description: str
    target_value: float = 0.0
    current_value: float = 0.0
    importance: int = 5  # 1-10
    deadline_days: int = 0  # 0 = no deadline
    status: str = "active"  # active, paused, completed, abandoned
    created_at: float = 0.0
    completed_at: float = 0.0
    steps: list[str] = field(default_factory=list)
    next_step: str = ""


@dataclass(slots=True)
class AmbitionEngine:
    """Manages long-term goals using LLM for decision-making."""
    
    _lock: RLock = field(default_factory=RLock)
    _ambitions: list[Ambition] = field(default_factory=list)
    _current_focus: str = ""
    _stats: dict[str, int] = field(default_factory=lambda: {
        "proposed": 0, "completed": 0, "llm_calls": 0,
    })
    _llm_call: Callable | None = None  # Function to call LLM
    
    def propose(self, name: str, category: str, description: str, 
                target_value: float = 0.0, importance: int = 5) -> Ambition:
        """Propose a new ambition."""
        amb = Ambition(
            name=name,
            category=category,
            description=description,
            target_value=target_value,
            importance=importance,
            created_at=time.time(),
        )
        with self._lock:
            self._ambitions.append(amb)
            self._stats["proposed"] += 1
        logger.info("ambition_proposed: %s (%s) importance=%d", name, category, importance)
        return amb
    
    def set_focus(self, name: str) -> bool:
        """Set the current focus ambition."""
        with self._lock:
            for amb in self._ambitions:
                if amb.name == name and amb.status == "active":
                    self._current_focus = name
                    return True
        return False
    
    def update_progress(self, name: str, delta: float, step: str = "") -> None:
        """Update progress toward an ambition."""
        with self._lock:
            for amb in self._ambitions:
                if amb.name == name:
                    amb.current_value += delta
                    if step and step not in amb.steps:
                        amb.steps.append(step)
                        amb.next_step = ""
                    # Check completion
                    if amb.target_value > 0 and amb.current_value >= amb.target_value:
                        amb.status = "completed"
                        amb.completed_at = time.time()
                        self._stats["completed"] += 1
                        logger.info("ambition_completed: %s", amb.name)
                    break
    
    def get_ambition_context(self) -> str:
        """Get formatted ambition context for LLM prompts."""
        with self._lock:
            lines = ["── Ambitions & Long-Term Goals ──"]
            active = [a for a in self._ambitions if a.status == "active"]
            completed = [a for a in self._ambitions if a.status == "completed"]
            
            if self._current_focus:
                lines.append(f"  Current focus: {self._current_focus}")
            
            if active:
                lines.append("  Active ambitions:")
                for a in sorted(active, key=lambda x: -x.importance)[:5]:
                    progress = f"{a.current_value:.0f}/{a.target_value:.0f}" if a.target_value > 0 else "in progress"
                    lines.append(f"    [{a.importance}] {a.name} ({a.category}) — {progress}")
                    if a.next_step:
                        lines.append(f"      Next: {a.next_step}")
            
            if completed:
                lines.append(f"  Completed: {len(completed)}")
                for a in completed[-2:]:
                    lines.append(f"    ✓ {a.name}")
            
            return "\n".join(lines)
    
    def propose_from_llm(self, llm_output: str) -> None:
        """Parse LLM output to propose new ambitions.
        
        Expected format: AMBITION:name|category|description|target|importance
        """
        with self._lock:
            self._stats["llm_calls"] += 1
        for line in llm_output.split('\n'):
            line = line.strip()
            if line.startswith("AMBITION:"):
                parts = line[9:].split('|')
                if len(parts) >= 3:
                    name = parts[0].strip()
                    category = parts[1].strip()
                    description = parts[2].strip()
                    target = float(parts[3]) if len(parts) > 3 and parts[3].strip() else 0
                    importance = int(parts[4]) if len(parts) > 4 and parts[4].strip() else 5
                    self.propose(name, category, description, target, importance)
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# Global instance
_ambition: AmbitionEngine | None = None
_ambition_lock = RLock()


def get_ambition() -> AmbitionEngine:
    global _ambition
    with _ambition_lock:
        if _ambition is None:
            _ambition = AmbitionEngine()
        return _ambition
