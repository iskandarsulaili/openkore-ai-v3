"""
Party follow positioning — maintains formation relative to party leader.

A coordinated party doesn't just follow the leader. They maintain
formation: melee up front, ranged in back, healers in the middle.
This module computes optimal follow positions based on party composition.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FormationPosition:
    """A position in the party formation."""
    role: str  # tank, melee_dps, ranged_dps, healer, support
    offset_x: int = 0  # Relative to leader
    offset_y: int = 0
    priority: int = 5  # Lower = closer to leader


@dataclass(slots=True)
class PartyFollowPositioning:
    """Computes optimal follow positions based on party composition."""
    
    _lock: RLock = field(default_factory=RLock)
    _formations: dict[str, list[FormationPosition]] = field(default_factory=lambda: {
        "balanced": [
            FormationPosition("tank", 0, -2, 1),
            FormationPosition("melee_dps", -2, -3, 3),
            FormationPosition("melee_dps", 2, -3, 3),
            FormationPosition("ranged_dps", -3, -5, 4),
            FormationPosition("ranged_dps", 3, -5, 4),
            FormationPosition("healer", 0, -4, 2),
            FormationPosition("support", -1, -4, 5),
        ],
        "melee_heavy": [
            FormationPosition("tank", 0, -2, 1),
            FormationPosition("melee_dps", -1, -3, 2),
            FormationPosition("melee_dps", 1, -3, 2),
            FormationPosition("melee_dps", -2, -4, 3),
            FormationPosition("melee_dps", 2, -4, 3),
            FormationPosition("healer", 0, -5, 4),
            FormationPosition("ranged_dps", 3, -6, 5),
        ],
        "ranged_heavy": [
            FormationPosition("tank", 0, -2, 1),
            FormationPosition("melee_dps", -1, -3, 3),
            FormationPosition("ranged_dps", -3, -5, 2),
            FormationPosition("ranged_dps", 3, -5, 2),
            FormationPosition("ranged_dps", -4, -6, 4),
            FormationPosition("healer", 0, -4, 3),
            FormationPosition("support", 2, -4, 5),
        ],
        "healer_protect": [
            FormationPosition("tank", 0, -2, 1),
            FormationPosition("melee_dps", -1, -3, 3),
            FormationPosition("melee_dps", 1, -3, 3),
            FormationPosition("healer", 0, -3, 2),
            FormationPosition("ranged_dps", -3, -5, 4),
            FormationPosition("ranged_dps", 3, -5, 4),
            FormationPosition("support", 0, -4, 5),
        ],
    })
    _stats: dict[str, int] = field(default_factory=lambda: {"positions_computed": 0})
    
    def get_position(self, my_role: str, party_roles: list[str], 
                     leader_x: int = 0, leader_y: int = 0,
                     formation: str = "balanced") -> dict[str, Any]:
        """Get the optimal follow position for a party member."""
        with self._lock:
            self._stats["positions_computed"] += 1
            
            form = self._formations.get(formation, self._formations["balanced"])
            
            # Find my position in formation
            my_idx = -1
            for i, pos in enumerate(form):
                if pos.role == my_role:
                    my_idx = i
                    break
            
            if my_idx < 0:
                # Role not in formation — assign based on priority
                assigned = [p for p in form if p.role not in party_roles]
                if assigned:
                    my_idx = form.index(assigned[0])
                else:
                    my_idx = len(form) - 1  # Last position
            
            pos = form[my_idx]
            
            return {
                "target_x": leader_x + pos.offset_x,
                "target_y": leader_y + pos.offset_y,
                "role": my_role,
                "formation": formation,
                "distance": max(abs(pos.offset_x), abs(pos.offset_y)),
            }
    
    def get_formation_context(self) -> str:
        """Get formatted formation context for LLM prompts."""
        with self._lock:
            lines = ["── Party Formation ──"]
            lines.append(f"  Available formations: {', '.join(self._formations.keys())}")
            for name, form in self._formations.items():
                roles = [f"{p.role}(d:{max(abs(p.offset_x), abs(p.offset_y))})" for p in form]
                lines.append(f"    {name}: {', '.join(roles)}")
            return "\n".join(lines)
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# Global instance
_follow: PartyFollowPositioning | None = None
_follow_lock = RLock()


def get_party_follow() -> PartyFollowPositioning:
    global _follow
    with _follow_lock:
        if _follow is None:
            _follow = PartyFollowPositioning()
        return _follow
