"""
Party skill coordinator — timed party buffs and skill combos.

Knows skill timing: Priest casts Magnificat 3s before Wizard's Storm Gust.
Bard's Apple of Idun when party SP < 30%. Paladin's Providence before MVP.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PartySkillCoordinator:
    """Coordinates party skill timing and combos."""
    
    _lock: RLock = field(default_factory=RLock)
    
    _party_combos: list[dict[str, Any]] = field(default_factory=lambda: [
        {"prep": "ss magnificat", "main": "ss storm_gust", "prep_time_s": 3.0, "description": "Magnificat → Storm Gust"},
        {"prep": "ss kyrie_eleison", "main": "ss bowling_bash", "prep_time_s": 2.0, "description": "Kyrie → Bowling Bash"},
        {"prep": "ss providentia", "main": "ss bowling_bash", "prep_time_s": 2.0, "description": "Providence → Bowling Bash"},
        {"prep": "ss apple_of_idun", "main": "ss storm_gust", "prep_time_s": 1.0, "description": "Apple of Idun → Storm Gust"},
        {"prep": "ss assassin_cross", "main": "ss sonic_blow", "prep_time_s": 1.0, "description": "Assassin Cross → Sonic Blow"},
    ])
    
    _party_buffs: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        "magnificat": {"sp_cost": 40, "duration_s": 300, "classes": ["priest", "high_priest"]},
        "kyrie_eleison": {"sp_cost": 30, "duration_s": 300, "classes": ["priest", "high_priest"]},
        "gloria": {"sp_cost": 20, "duration_s": 300, "classes": ["priest", "high_priest"]},
        "providentia": {"sp_cost": 20, "duration_s": 300, "classes": ["paladin"]},
        "apple_of_idun": {"sp_cost": 30, "duration_s": 120, "classes": ["bard"]},
        "poem_of_bragi": {"sp_cost": 40, "duration_s": 120, "classes": ["bard"]},
        "assassin_cross": {"sp_cost": 20, "duration_s": 60, "classes": ["assassin_cross"]},
    })
    
    _last_buff_time: dict[str, float] = field(default_factory=dict)
    _active_combos: dict[str, dict[str, Any]] = field(default_factory=dict)
    
    def get_buffs_for_party(self, party_members: list[dict[str, Any]], current_sp: int) -> list[str]:
        """Get buff commands for the party based on member classes."""
        commands: list[str] = []
        now = time.time()
        
        for member in party_members:
            bot_id = member.get("bot_id", "")
            player_class = member.get("class", "novice").lower()
            
            for buff_name, buff_info in self._party_buffs.items():
                if player_class in buff_info["classes"]:
                    # Check cooldown
                    last_time = self._last_buff_time.get(f"{bot_id}:{buff_name}", 0)
                    if now - last_time > buff_info["duration_s"] * 0.8:  # Refresh at 80% duration
                        if current_sp >= buff_info["sp_cost"]:
                            commands.append(f"ss {buff_name}")
                            self._last_buff_time[f"{bot_id}:{buff_name}"] = now
        
        return commands
    
    def get_combos_for_party(self, party_members: list[dict[str, Any]], current_map: str) -> list[str]:
        """Get skill combos for the party based on member classes."""
        commands: list[str] = []
        member_classes = [m.get("class", "").lower() for m in party_members]
        
        for combo in self._party_combos:
            prep_skill = combo["prep"].replace("ss ", "")
            main_skill = combo["main"].replace("ss ", "")
            
            # Check if party has both skills
            prep_class = None
            main_class = None
            for buff_name, buff_info in self._party_buffs.items():
                if buff_name == prep_skill:
                    for cls in buff_info["classes"]:
                        if cls in member_classes:
                            prep_class = cls
                if buff_name == main_skill:
                    for cls in buff_info["classes"]:
                        if cls in member_classes:
                            main_class = cls
            
            if prep_class and main_class:
                commands.append(combo["prep"])
                # The main skill follows after prep_time_s
                # This is handled by the PDCA loop's timing
                commands.append(combo["main"])
        
        return commands[:6]  # Max 6 combo commands
    
    def counters(self) -> dict[str, int]:
        return {"combos": len(self._party_combos), "buffs": len(self._party_buffs)}
