"""
Party leader coordination — wait/pause decisions for party movement.

A good party leader doesn't just run ahead. They wait for members to
catch up, pause when someone is in combat, check for missing members,
and coordinate movement. This module provides the decision logic.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PartyMemberStatus:
    """Status of a party member for leader coordination."""
    name: str
    map: str = ""
    x: int = 0
    y: int = 0
    hp_pct: float = 1.0
    is_in_combat: bool = False
    is_casting: bool = False
    is_dead: bool = False
    distance_from_leader: int = 0
    last_updated: float = 0.0


@dataclass(slots=True)
class PartyLeaderCoordinator:
    """Coordinates party movement as the leader."""
    
    _lock: RLock = field(default_factory=RLock)
    _members: dict[str, PartyMemberStatus] = field(default_factory=dict)
    _leader_name: str = ""
    _leader_x: int = 0
    _leader_y: int = 0
    _leader_map: str = ""
    _waiting: bool = False
    _wait_reason: str = ""
    _wait_until: float = 0.0
    _stats: dict[str, int] = field(default_factory=lambda: {
        "waits": 0, "pauses": 0, "member_checks": 0,
    })
    
    def set_leader(self, name: str, map: str = "", x: int = 0, y: int = 0) -> None:
        with self._lock:
            self._leader_name = name
            self._leader_map = map
            self._leader_x = x
            self._leader_y = y
    
    def update_member(self, name: str, **kwargs: Any) -> None:
        with self._lock:
            if name not in self._members:
                self._members[name] = PartyMemberStatus(name=name)
            m = self._members[name]
            for key, value in kwargs.items():
                if hasattr(m, key):
                    setattr(m, key, value)
            m.last_updated = time.time()
            # Compute distance from leader
            if m.map == self._leader_map:
                m.distance_from_leader = abs(m.x - self._leader_x) + abs(m.y - self._leader_y)
            else:
                m.distance_from_leader = 9999  # Different map
    
    def should_wait(self, max_distance: int = 15, 
                    max_dead_time: int = 10,
                    cast_wait_skills: list[str] | None = None) -> dict[str, Any]:
        """Should the leader wait for party members?"""
        with self._lock:
            now = time.time()
            self._stats["member_checks"] += 1
            
            if not self._members:
                return {"wait": False, "reason": "solo"}
            
            reasons = []
            
            for m in self._members.values():
                # Check if member is dead
                if m.is_dead:
                    if now - m.last_updated < max_dead_time:
                        reasons.append(f"{m.name} is dead")
                    continue
                
                # Check if member is in combat
                if m.is_in_combat:
                    reasons.append(f"{m.name} is in combat")
                
                # Check if member is casting a long skill
                if m.is_casting and cast_wait_skills:
                    if m.is_casting in cast_wait_skills:
                        reasons.append(f"{m.name} is casting {m.is_casting}")
                
                # Check distance
                if m.distance_from_leader > max_distance:
                    reasons.append(f"{m.name} is too far ({m.distance_from_leader} cells)")
            
            if reasons:
                self._waiting = True
                self._wait_reason = "; ".join(reasons[:3])
                self._wait_until = now + 5  # Re-check in 5s
                self._stats["waits"] += 1
                return {"wait": True, "reason": self._wait_reason, "duration_s": 5}
            
            self._waiting = False
            self._wait_reason = ""
            return {"wait": False, "reason": "all_ready"}
    
    def get_leader_context(self) -> str:
        """Get formatted leader context for LLM prompts."""
        with self._lock:
            lines = ["── Party Leader Status ──"]
            lines.append(f"  Leader: {self._leader_name}")
            lines.append(f"  Waiting: {'YES - ' + self._wait_reason if self._waiting else 'No'}")
            lines.append(f"  Members: {len(self._members)}")
            for m in self._members.values():
                status = "dead" if m.is_dead else "combat" if m.is_in_combat else "idle"
                lines.append(f"    {m.name}: {status} ({m.distance_from_leader} cells away)")
            return "\n".join(lines)
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# Global instance
_leader: PartyLeaderCoordinator | None = None
_leader_lock = RLock()


def get_party_leader() -> PartyLeaderCoordinator:
    global _leader
    with _leader_lock:
        if _leader is None:
            _leader = PartyLeaderCoordinator()
        return _leader
