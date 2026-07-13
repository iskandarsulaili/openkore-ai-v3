"""
WoE prediction system — predicts enemy behavior and sets traps.

A top player doesn't just fight in WoE. They predict. They know
the enemy guild always attacks from the south. They know the enemy's
best player will try to solo the emperium. They know when the enemy
will retreat. They set traps, fake retreats, and bait ambushes.

This module analyzes historical WoE data to predict enemy movements
and recommend counter-strategies.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GuildProfile:
    """Profile of a guild we've observed in WoE."""
    name: str
    alliance: str = ""  # ally, enemy, neutral
    strength: int = 0  # 1-10 estimated power
    preferred_entrance: str = ""  # Which castle entrance they favor
    preferred_time: str = ""  # When they usually attack
    emperium_breaker: str = ""  # Who usually breaks the emperium
    tactics: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    last_seen: float = 0.0
    encounter_count: int = 0


@dataclass
class CastleState:
    """Current state of a WoE castle."""
    name: str
    owner: str = ""
    defenders: int = 0
    siege_active: bool = False
    last_updated: float = 0.0


@dataclass(slots=True)
class WoEPredictor:
    """Predicts enemy behavior and recommends WoE strategies."""
    
    _lock: RLock = field(default_factory=RLock)
    _guilds: dict[str, GuildProfile] = field(default_factory=dict)
    _castles: dict[str, CastleState] = field(default_factory=dict)
    _battle_log: list[dict[str, Any]] = field(default_factory=list)
    _stats: dict[str, int] = field(default_factory=lambda: {
        "predictions": 0, "encounters": 0, "traps_set": 0,
    })
    
    def observe_guild(self, name: str, **kwargs) -> None:
        """Record an observation about a guild."""
        with self._lock:
            guild = self._guilds.setdefault(name, GuildProfile(name=name))
            guild.last_seen = time.time()
            guild.encounter_count += 1
            self._stats["encounters"] += 1
            
            for key, value in kwargs.items():
                if hasattr(guild, key):
                    setattr(guild, key, value)
    
    def predict_attack(self, castle_name: str) -> dict[str, Any]:
        """Predict the next attack on a castle."""
        with self._lock:
            self._stats["predictions"] += 1
            
            # Find guilds that might attack this castle
            potential = []
            for guild in self._guilds.values():
                if guild.alliance == "enemy":
                    potential.append(guild)
            
            if not potential:
                return {"confidence": 0.0, "prediction": "no_data"}
            
            # Sort by most likely to attack
            potential.sort(key=lambda g: -g.encounter_count)
            top = potential[0]
            
            return {
                "confidence": min(0.8, top.encounter_count * 0.1),
                "prediction": f"{top.name} may attack from {top.preferred_entrance or 'unknown'}",
                "guild": top.name,
                "entrance": top.preferred_entrance,
                "weaknesses": top.weaknesses,
            }
    
    def recommend_defense(self, castle_name: str) -> str:
        """Recommend a defense strategy for a castle."""
        prediction = self.predict_attack(castle_name)
        
        if prediction["confidence"] < 0.3:
            return "Standard defense — cover all entrances."
        
        guild_name = prediction.get("guild", "unknown")
        entrance = prediction.get("entrance", "unknown")
        weaknesses = prediction.get("weaknesses", [])
        
        lines = [f"── WoE Defense: {castle_name} ──"]
        lines.append(f"  Predicted attacker: {guild_name} (confidence: {prediction['confidence']*100:.0f}%)")
        lines.append(f"  Likely entrance: {entrance}")
        
        if weaknesses:
            lines.append("  Exploit weaknesses:")
            for w in weaknesses:
                lines.append(f"    - {w}")
        
        return "\n".join(lines)
    
    def get_woe_context(self) -> str:
        """Get formatted WoE context for LLM prompts."""
        with self._lock:
            lines = ["── WoE Intelligence ──"]
            lines.append(f"  Known guilds: {len(self._guilds)}")
            lines.append(f"  Castles tracked: {len(self._castles)}")
            
            for guild in sorted(self._guilds.values(), key=lambda g: -g.strength)[:5]:
                lines.append(
                    f"  {guild.name}: strength={guild.strength} "
                    f"alliance={guild.alliance} encounters={guild.encounter_count}"
                )
            
            return "\n".join(lines)
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# ── Global instance ──

_woe: WoEPredictor | None = None
_woe_lock = RLock()


def get_woe() -> WoEPredictor:
    global _woe
    with _woe_lock:
        if _woe is None:
            _woe = WoEPredictor()
        return _woe
