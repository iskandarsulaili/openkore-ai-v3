"""
War of Emperium intelligence — guild wars, castle defense, emperium breaking.

A pro player lives for WoE. This module handles:
- WoE schedule awareness
- Castle defense/attack tactics
- Emperium breaking strategy
- Guild coordination
- 50+ player battlefield awareness
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
class WoEIntelligence:
    """War of Emperium strategy engine."""
    
    _lock: RLock = field(default_factory=RLock)
    _woe_schedule: dict[str, list[int]] = field(default_factory=lambda: {
        "monday": [], "tuesday": [], "wednesday": [],
        "thursday": [], "friday": [], "saturday": [20, 22],
        "sunday": [20, 22],
    })
    _guild_info: dict[str, Any] = field(default_factory=dict)
    _castle_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    _stats: dict[str, int] = field(default_factory=lambda: {"woe_participations": 0, "emperium_breaks": 0, "deaths_in_woe": 0})
    
    def is_woe_active(self) -> bool:
        """Check if WoE is currently active."""
        now = time.localtime()
        day_name = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"][now.tm_wday]
        hour = now.tm_hour
        
        for start_hour in self._woe_schedule.get(day_name, []):
            if start_hour <= hour < start_hour + 2:
                return True
        return False
    
    def get_woe_status(self) -> dict[str, Any]:
        """Get current WoE status and recommendations."""
        active = self.is_woe_active()
        if not active:
            # Check if WoE is coming soon
            now = time.localtime()
            day_name = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"][now.tm_wday]
            next_woe = None
            for start_hour in self._woe_schedule.get(day_name, []):
                if start_hour > now.tm_hour:
                    next_woe = start_hour
                    break
            
            return {
                "active": False,
                "next_woe_hour": next_woe,
                "recommendation": "prepare" if next_woe and next_woe - now.tm_hour <= 2 else "ignore",
            }
        
        return {
            "active": True,
            "recommendation": "join_guild_war",
            "strategy": "defend_castle" if self._guild_info.get("has_castle") else "attack_castle",
            "priority": "emperium" if self._guild_info.get("has_castle") else "survival",
        }
    
    def recommend_equipment(self) -> dict[str, str]:
        """Recommend equipment for WoE."""
        return {
            "armor": "freyja_armor" if self.is_woe_active() else "normal",
            "weapon": "holy_weapon" if self.is_woe_active() else "normal",
            "shield": "valkyrie_shield" if self.is_woe_active() else "normal",
            "consumables": "woe_potions" if self.is_woe_active() else "normal",
        }
    
    def should_engage(self, enemy_count: int, ally_count: int, 
                      has_emperium: bool = False) -> dict[str, Any]:
        """Should we engage in PvP during WoE?"""
        if not self.is_woe_active():
            return {"engage": False, "reason": "not_woe_time"}
        
        if has_emperium:
            return {"engage": True, "reason": "emperium_break_priority", "target": "emperium"}
        
        if ally_count >= enemy_count:
            return {"engage": True, "reason": "numerical_advantage"}
        
        if ally_count < enemy_count * 0.5:
            return {"engage": False, "reason": "outnumbered", "action": "retreat"}
        
        return {"engage": True, "reason": "even_fight"}
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)
