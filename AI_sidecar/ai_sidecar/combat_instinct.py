"""
Combat instinct engine — reads combat context, not just HP numbers.

A pro player FEELS combat. They know WHY the HP dropped:
- Was it a skill cast? (read the cast bar)
- Was it an AoE? (check position)
- Was it a crit? (check damage spike pattern)
- Was it a DoT? (check debuff)

This module analyzes combat events to determine the CAUSE of damage,
not just the fact that damage occurred.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CombatEvent:
    timestamp: float
    event_type: str  # damage_taken, damage_dealt, skill_cast, skill_hit, debuff_applied, heal_received
    source: str  # monster name or skill name
    value: int
    element: str = "neutral"
    is_aoe: bool = False
    is_crit: bool = False
    is_dot: bool = False


@dataclass(slots=True)
class CombatInstinctEngine:
    """Reads combat context to determine WHY damage occurred."""
    
    _lock: RLock = field(default_factory=RLock)
    _event_history: dict[str, deque[CombatEvent]] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=50)))
    _monster_skill_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    _last_cast_seen: dict[str, str] = field(default_factory=dict)  # bot_id -> last skill cast by monster
    _stats: dict[str, int] = field(default_factory=lambda: {"events_processed": 0, "instinct_triggers": 0})
    
    def record_event(self, bot_id: str, event: CombatEvent) -> None:
        with self._lock:
            self._event_history[bot_id].append(event)
            self._stats["events_processed"] += 1
    
    def analyze_damage(self, bot_id: str, hp_drop: int, current_hp: int, max_hp: int,
                       nearby_monsters: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze WHY damage occurred and recommend response."""
        with self._lock:
            events = list(self._event_history.get(bot_id, []))
        
        result = {
            "cause": "unknown",
            "element": "neutral",
            "is_aoe": False,
            "is_crit": False,
            "is_dot": False,
            "threat_level": "low",
            "recommendation": "continue",
            "evasive_action": None,
        }
        
        # Check recent events for context
        recent = [e for e in events[-10:] if time.time() - e.timestamp < 3.0]
        
        for event in reversed(recent):
            if event.event_type == "skill_cast":
                # A monster just cast a skill — this is likely the cause
                result["cause"] = f"skill:{event.source}"
                result["element"] = event.element
                result["is_aoe"] = event.is_aoe
                result["threat_level"] = "high" if event.is_aoe else "medium"
                result["recommendation"] = "dodge" if event.is_aoe else "pot"
                result["evasive_action"] = "move" if event.is_aoe else "use_potion"
                self._stats["instinct_triggers"] += 1
                break
            elif event.event_type == "debuff_applied":
                result["cause"] = f"debuff:{event.source}"
                result["recommendation"] = "cure"
                result["evasive_action"] = "use_green_potion"
                self._stats["instinct_triggers"] += 1
                break
        
        # Check if damage is lethal
        hp_pct = current_hp / max(max_hp, 1)
        if hp_pct < 0.2 and result["recommendation"] != "dodge":
            result["recommendation"] = "flee"
            result["evasive_action"] = "fly_wing"
            result["threat_level"] = "critical"
        
        return result
    
    def should_interrupt(self, bot_id: str, current_action: str, 
                         monster_casting: str | None) -> bool:
        """Should the bot interrupt its current action to dodge?"""
        if monster_casting and monster_casting in ("WZ_STORMGUST", "WZ_METEORSTORM",
                                                     "WZ_HEAVENDRIVE", "MG_THUNDERSTORM"):
            return True
        return False
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)
