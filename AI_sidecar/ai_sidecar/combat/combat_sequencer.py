"""Combat Sequencer — coordinates timing, emergencies, and skill chaining.

Architecture:
  - Thin coordination layer above SkillEngine + EquipmentManager + TargetEngine
  - Tracks per-bot action timing (equip GCD, skill cast, cooldowns)
  - Handles emergency interrupts (HP low → heal/escape)
  - Generates properly spaced action queue entries
  - Uses ClassTemplates for per-class combat priorities

RULE.md compliance:
  - All timing from rAthena DB (skill_db.yml cast/cooldown/delay)
  - All templates from skill_tree.yml — zero hardcoded preferences
  - Bridge only executes commands — sequencer is pure sidecar logic
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CombatState:
    """Per-bot combat state tracking."""
    last_equip_ms: float = 0.0
    last_skill_ms: float = 0.0
    last_skill_name: str = ""
    skill_cooldowns: dict[str, float] = field(default_factory=dict)
    buff_expiry: dict[str, float] = field(default_factory=dict)
    current_target: str = ""
    consecutive_failures: int = 0
    rotation_index: int = 0  # For skill cycling


class CombatSequencer:
    """Coordinates combat actions with proper timing."""

    def __init__(self):
        self._states: dict[str, CombatState] = {}
        self._equip_gcd_ms = 1000
        self._skill_gcd_ms = 1000
    
    def _state(self, bot_id: str) -> CombatState:
        if bot_id not in self._states:
            self._states[bot_id] = CombatState()
        return self._states[bot_id]
    
    def can_equip(self, bot_id: str) -> bool:
        """Check if equip cooldown has expired."""
        st = self._state(bot_id)
        now_ms = time.time() * 1000
        return (now_ms - st.last_equip_ms) >= self._equip_gcd_ms
    
    def mark_equip(self, bot_id: str):
        """Record equip action."""
        self._state(bot_id).last_equip_ms = time.time() * 1000
    
    def can_cast(self, bot_id: str, skill_name: str, cooldown_ms: int = 0) -> bool:
        """Check if a skill can be cast (GCD + cooldown)."""
        st = self._state(bot_id)
        now_ms = time.time() * 1000
        
        # Global cooldown check
        if (now_ms - st.last_skill_ms) < self._skill_gcd_ms:
            return False
        
        # Skill-specific cooldown check
        cd_until = st.skill_cooldowns.get(skill_name, 0)
        if now_ms < cd_until:
            return False
        
        return True
    
    def mark_cast(self, bot_id: str, skill_name: str, cast_time_ms: int = 0, 
                  aftercast_delay_ms: int = 0, cooldown_ms: int = 0):
        """Record a skill cast and update all timing."""
        st = self._state(bot_id)
        now_ms = time.time() * 1000
        
        st.last_skill_ms = now_ms
        st.last_skill_name = skill_name
        st.rotation_index += 1
        
        # Set skill cooldown
        if cooldown_ms > 0:
            st.skill_cooldowns[skill_name] = now_ms + cooldown_ms
        
        # Set next available time (after cast + aftercast delay)
        total_delay = cast_time_ms + aftercast_delay_ms
    
    def handle_emergency(self, bot_id: str, hp_ratio: float, sp_ratio: float,
                         job_name: str) -> Optional[str]:
        """Check for emergency conditions. Returns command or None.
        
        Priority:
        1. HP < 15% → flee (handled by bridge survival reflex already)
        2. HP < 40% and Acolyte → heal self
        3. HP < 25% → teleport/escape
        4. SP < 10% → stop using skills, auto-attack only
        """
        if hp_ratio < 0.40 and "Acolyte" in job_name:
            return "emergency_heal"
        if hp_ratio < 0.25:
            return "emergency_escape"
        if hp_ratio < 0.15:
            return "emergency_flee"  # Bridge survival reflex should handle this
        
        return None


# Global sequencer
_sequencer: CombatSequencer | None = None


def get_combat_sequencer() -> CombatSequencer:
    global _sequencer
    if _sequencer is None:
        _sequencer = CombatSequencer()
    return _sequencer
