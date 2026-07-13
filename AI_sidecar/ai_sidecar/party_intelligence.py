"""
Party intelligence — coordinates with human players, not just bots.

A pro player knows:
- When to buff, when to heal, when to DPS
- Which party composition works for which dungeon
- How to read human intent and predict their next move
- When to save a dying party member vs let them die
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PartyMember:
    name: str
    is_human: bool = True
    job: str = "novice"
    level: int = 1
    hp_pct: float = 1.0
    sp_pct: float = 1.0
    map: str = ""
    last_seen: float = 0.0
    role: str = "unknown"  # tank, healer, dps, support
    trust_score: float = 0.5  # 0.0 = unknown, 1.0 = trusted


@dataclass(slots=True)
class PartyIntelligence:
    """Coordinates with human players in a party."""
    
    _lock: RLock = field(default_factory=RLock)
    _party: dict[str, PartyMember] = field(default_factory=dict)
    _party_leader: str = ""
    _stats: dict[str, int] = field(default_factory=lambda: {"parties_joined": 0, "buffs_cast": 0, "heals_cast": 0, "saves": 0})
    
    def update_member(self, name: str, **kwargs: Any) -> None:
        with self._lock:
            if name not in self._party:
                self._party[name] = PartyMember(name=name)
            for key, value in kwargs.items():
                if hasattr(self._party[name], key):
                    setattr(self._party[name], key, value)
            self._party[name].last_seen = time.time()
    
    def get_party_composition(self) -> dict[str, list[str]]:
        """Analyze party composition and recommend roles."""
        with self._lock:
            roles: dict[str, list[str]] = {"tank": [], "healer": [], "dps": [], "support": [], "unknown": []}
            for member in self._party.values():
                roles[member.role].append(member.name)
            return roles
    
    def should_heal(self, member_name: str) -> bool:
        """Should we heal this party member?"""
        with self._lock:
            member = self._party.get(member_name)
            if not member:
                return False
            # Heal if below 40% HP, or below 60% and in combat
            if member.hp_pct < 0.4:
                return True
            if member.hp_pct < 0.6 and member.hp_pct < 0.8:
                return True
            return False
    
    def should_buff(self, member_name: str) -> bool:
        """Should we buff this party member?"""
        with self._lock:
            member = self._party.get(member_name)
            if not member:
                return False
            # Buff if we haven't seen them recently (likely just joined)
            if time.time() - member.last_seen > 60:
                return True
            return False
    
    def should_save(self, member_name: str) -> bool:
        """Should we risk our life to save this party member?"""
        with self._lock:
            member = self._party.get(member_name)
            if not member:
                return False
            # Save trusted members, let strangers die
            return member.trust_score > 0.7
    
    def assess_party_health(self) -> dict[str, Any]:
        """Assess overall party health."""
        with self._lock:
            if not self._party:
                return {"status": "no_party", "risk": "low"}
            
            low_hp = sum(1 for m in self._party.values() if m.hp_pct < 0.4)
            dead = sum(1 for m in self._party.values() if m.hp_pct <= 0)
            
            return {
                "status": "critical" if low_hp > 2 else "warning" if low_hp > 0 else "healthy",
                "members": len(self._party),
                "low_hp": low_hp,
                "dead": dead,
                "composition": self.get_party_composition(),
            }
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)
