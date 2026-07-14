"""
Threat-based target selection — evaluates monsters and picks the optimal target.

A pro player doesn't attack the nearest monster. They attack the most
threatening one first: the one dealing the most damage, the one about
to cast a deadly skill, the one that's low HP and about to die.

This module evaluates all visible monsters and returns the optimal target.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MonsterThreat:
    """Threat assessment for a single monster."""
    monster_id: int = 0
    name: str = ""
    level: int = 1
    hp: int = 1
    max_hp: int = 1
    hp_pct: float = 1.0
    distance: int = 0
    element: str = "neutral"
    size: str = "medium"
    race: str = "formless"
    is_boss: bool = False
    is_aggressive: bool = False
    is_casting: bool = False
    casting_skill: str = ""
    damage_to_us: int = 0  # Total damage dealt to us
    damage_to_party: int = 0  # Total damage dealt to party
    damage_from_us: int = 0  # Total damage we've dealt to it
    is_low_hp: bool = False  # Below 30% HP
    threat_score: float = 0.0  # Computed score
    recommendation: str = "attack"  # attack, focus_fire, ignore, flee


@dataclass(slots=True)
class ThreatBasedTargeting:
    """Evaluates all visible monsters and returns the optimal target."""
    
    _lock: RLock = field(default_factory=RLock)
    _monsters: dict[int, MonsterThreat] = field(default_factory=dict)
    _stats: dict[str, int] = field(default_factory=lambda: {
        "evaluations": 0, "boss_targeted": 0, "caster_targeted": 0,
        "low_hp_finished": 0, "focus_fire": 0,
    })
    
    def update_monster(self, monster_id: int, **kwargs: Any) -> None:
        """Update or add a monster's threat data."""
        with self._lock:
            if monster_id not in self._monsters:
                self._monsters[monster_id] = MonsterThreat(monster_id=monster_id)
            m = self._monsters[monster_id]
            for key, value in kwargs.items():
                if hasattr(m, key):
                    setattr(m, key, value)
            # Auto-compute HP percentage
            if m.max_hp > 0:
                m.hp_pct = m.hp / m.max_hp
                m.is_low_hp = m.hp_pct < 0.3
    
    def record_damage_to_us(self, monster_id: int, damage: int) -> None:
        with self._lock:
            m = self._monsters.get(monster_id)
            if m:
                m.damage_to_us += damage
    
    def record_damage_to_party(self, monster_id: int, damage: int) -> None:
        with self._lock:
            m = self._monsters.get(monster_id)
            if m:
                m.damage_to_party += damage
    
    def record_damage_from_us(self, monster_id: int, damage: int) -> None:
        with self._lock:
            m = self._monsters.get(monster_id)
            if m:
                m.damage_from_us += damage

    def deprioritize_monster(self, monster_id: int) -> None:
        """Deprioritize a monster (e.g., because it killed us before)."""
        with self._lock:
            m = self._monsters.get(monster_id)
            if m:
                m.threat_score = max(-100, m.threat_score - 50)
                m.is_low_hp = False  # Don't try to finish it
    
    def get_best_target(self, player_class: str = "", 
                        party_size: int = 1,
                        has_aoe: bool = False) -> dict[str, Any] | None:
        """Evaluate all monsters and return the optimal target."""
        with self._lock:
            self._stats["evaluations"] += 1
            
            # Clean up dead monsters (memory leak prevention)
            dead_ids = [mid for mid, m in self._monsters.items() if m.hp <= 0]
            for mid in dead_ids:
                del self._monsters[mid]
            
            if not self._monsters:
                return None
            
            best = None
            best_score = -9999.0
            
            for mid, m in list(self._monsters.items()):
                if m.hp <= 0:
                    continue
                
                score = 0.0
                
                # 1. Boss priority: bosses get +50 base score
                if m.is_boss:
                    score += 50
                
                # 2. Casting priority: monsters casting deadly skills get +40
                if m.is_casting:
                    deadly_skills = ["WZ_STORMGUST", "WZ_METEORSTORM", "WZ_HEAVENDRIVE",
                                     "MG_THUNDERSTORM", "MG_FIREWALL", "AL_HEAL"]
                    if m.casting_skill in deadly_skills:
                        score += 40
                    else:
                        score += 20
                
                # 3. Damage threat: monsters that hurt us get +damage/100
                score += m.damage_to_us / 100.0
                score += m.damage_to_party / 100.0
                
                # 4. Low HP finish: monsters under 30% HP get +30 (easy kill)
                if m.is_low_hp and m.damage_from_us > 0:
                    score += 30
                
                # 5. Distance penalty: closer is better
                score -= m.distance * 0.5
                
                # 6. Aggressive monsters get +10
                if m.is_aggressive:
                    score += 10
                
                # 7. Focus fire bonus: if we've already damaged it, +15
                if m.damage_from_us > 0:
                    score += 15
                
                # 8. AoE bonus: if we have AoE, prefer clustered targets
                if has_aoe:
                    # Estimate cluster size (simplified)
                    nearby = sum(1 for om in self._monsters.values()
                                if om.monster_id != mid and om.hp > 0
                                and abs(om.distance - m.distance) < 5)
                    score += nearby * 5
                
                m.threat_score = score
                
                if score > best_score:
                    best_score = score
                    best = m
            
            if best is None:
                return None
            
            # Track stats
            if best.is_boss:
                self._stats["boss_targeted"] += 1
            if best.is_casting:
                self._stats["caster_targeted"] += 1
            if best.is_low_hp and best.damage_from_us > 0:
                self._stats["low_hp_finished"] += 1
            
            return {
                "monster_id": best.monster_id,
                "name": best.name,
                "threat_score": best_score,
                "is_boss": best.is_boss,
                "is_casting": best.is_casting,
                "is_low_hp": best.is_low_hp,
                "distance": best.distance,
                "recommendation": "attack",
            }
    
    def get_threat_context(self) -> str:
        """Get formatted threat context for LLM prompts."""
        with self._lock:
            lines = ["── Threat Assessment ──"]
            active = [m for m in self._monsters.values() if m.hp > 0]
            if not active:
                lines.append("  No active threats.")
                return "\n".join(lines)
            
            # Compute scores for context display (scores may not have been computed yet)
            for m in active:
                score = 0.0
                if m.is_boss: score += 50
                if m.is_casting: score += 40
                score += m.damage_to_us / 100.0
                score += m.damage_to_party / 100.0
                if m.is_low_hp and m.damage_from_us > 0: score += 30
                score -= m.distance * 0.5
                if m.is_aggressive: score += 10
                if m.damage_from_us > 0: score += 15
                m.threat_score = score
            
            # Sort by threat score
            sorted_m = sorted(active, key=lambda m: -m.threat_score)
            lines.append(f"  Active threats: {len(sorted_m)}")
            for m in sorted_m[:5]:
                flags = []
                if m.is_boss: flags.append("BOSS")
                if m.is_casting: flags.append(f"CASTING({m.casting_skill})")
                if m.is_low_hp: flags.append("LOW_HP")
                if m.is_aggressive: flags.append("AGGRO")
                flag_str = f" [{','.join(flags)}]" if flags else ""
                lines.append(f"    {m.name} (ID:{m.monster_id}) score={m.threat_score:.0f}{flag_str}")
            
            return "\n".join(lines)
    
    def cleanup_monsters(self, active_ids: set[int]) -> None:
        """Remove monsters not in the current snapshot (memory leak prevention)."""
        with self._lock:
            stale = [mid for mid in self._monsters if mid not in active_ids]
            for mid in stale:
                del self._monsters[mid]
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# Global instance
_targeting: ThreatBasedTargeting | None = None
_targeting_lock = RLock()


def get_threat_targeting() -> ThreatBasedTargeting:
    global _targeting
    with _targeting_lock:
        if _targeting is None:
            _targeting = ThreatBasedTargeting()
        return _targeting
