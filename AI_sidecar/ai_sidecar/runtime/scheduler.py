"""Out-of-Combat Scheduler — buff cycles, auto-craft, auto-vend when idle.

When no monsters in range, bots should not stand still. Instead:
- Priest: Cast Blessing, Increase Agility, Kyrie Eleison (expire 2-4 min)
- Blacksmith: Auto-forge items if materials available
- Merchant: Auto-vend in town
- All: Sit to regen if HP/SP not full
- All: Consume food/drink if available
"""
from __future__ import annotations
from typing import Any
import logging
from datetime import datetime, timedelta
from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)

# Buff durations (approximate, in seconds)
BUFF_DURATIONS = {
    "AL_BLESSING": 240,     # 4 min
    "AL_INCAGI": 240,       # 4 min
    "PR_KYRIE": 120,        # 2 min
    "PR_MAGNIFICAT": 240,   # 4 min
    "PR_GLORIA": 180,       # 3 min
    "PR_ASSUMPTIO": 120,    # 2 min
    "PR_SUFFRAGIUM": 60,    # 1 min
    "MG_SRECOVERY": 0,      # passive
    "TF_HIDING": 0,         # until move
    "AS_CLOAKING": 0,       # until attack
}


class OutOfCombatScheduler:
    """Manages out-of-combat behavior cycles.
    
    Each bot gets a schedule of actions to perform when idle:
    - Buff rotation (reapply before expiry)
    - Regen (sit when below 80% HP/SP)
    - Crafting (when in town with materials)
    - Vending (when in town with items to sell)
    - Consumable use (awakening potions, food)
    """
    
    def __init__(self):
        self._last_buff: dict[str, dict[str, datetime]] = {}  # bot_id -> {skill_id: last_cast}
        self._tick_counters: dict[str, int] = {}
    
    def _should_rebuff(self, bot_id: str, skill_id: str) -> bool:
        """Check if a buff should be reapplied."""
        last = self._last_buff.get(bot_id, {}).get(skill_id)
        if not last:
            return True
        duration = BUFF_DURATIONS.get(skill_id, 120)
        elapsed = (datetime.now() - last).total_seconds()
        return elapsed > duration * 0.7  # Rebuff at 70% of duration
    
    def _record_buff(self, bot_id: str, skill_id: str) -> None:
        if bot_id not in self._last_buff:
            self._last_buff[bot_id] = {}
        self._last_buff[bot_id][skill_id] = datetime.now()
    
    def assess(self, signals: dict[str, Any], actions: list[HeuristicAction], bot_id: str) -> None:
        monsters = signals.get("monsters_around", []) or []
        in_combat = len(monsters) > 0
        
        self._tick_counters[bot_id] = self._tick_counters.get(bot_id, 0) + 1
        tick = self._tick_counters[bot_id]
        
        # Only run out-of-combat behavior every 5 ticks
        if in_combat or tick % 5 != 0:
            return
        
        job = str(signals.get("job", "") or "").lower()
        hp_pct = int(signals.get("hp", 100) or 100) / max(int(signals.get("hp_max", 100) or 100), 1) * 100
        sp_pct = int(signals.get("sp", 100) or 100) / max(int(signals.get("sp_max", 100) or 100), 1) * 100
        
        # 1. Sit to regen if HP or SP is low
        if hp_pct < 80 or sp_pct < 60:
            actions.append(HeuristicAction(
                kind="command",
                command="sit",
                confidence=0.8,
                reason=f"Out-of-combat: regen HP={hp_pct:.0f}% SP={sp_pct:.0f}%",
                domain="behavior",
            ))
        
        # 2. Buff rotation (Priest/Acolyte)
        if "priest" in job or "acolyte" in job or "monk" in job:
            buffs_to_cast = []
            if self._should_rebuff(bot_id, "AL_BLESSING"):
                buffs_to_cast.append("AL_BLESSING")
            if self._should_rebuff(bot_id, "AL_INCAGI"):
                buffs_to_cast.append("AL_INCAGI")
            if self._should_rebuff(bot_id, "PR_KYRIE"):
                buffs_to_cast.append("PR_KYRIE")
            
            for skill_id in buffs_to_cast:
                actions.append(HeuristicAction(
                    kind="command",
                    command=f"skill_cast {skill_id} 0",
                    confidence=0.9,
                    reason=f"Out-of-combat: rebuffing {skill_id}",
                    domain="behavior",
                ))
                self._record_buff(bot_id, skill_id)
        
        # 3. Sit-to-regen bonus (Novices regen 2x faster sitting)
        if "novice" in job and hp_pct < 100:
            actions.append(HeuristicAction(
                kind="command",
                command="sit",
                confidence=0.9,
                reason="Out-of-combat: Novice regen 2x faster sitting",
                domain="behavior",
            ))
        
        # 4. Log idle state
        actions.append(HeuristicAction(
            kind="log",
            command=f"idle job={job} hp={hp_pct:.0f}% sp={sp_pct:.0f}%",
            confidence=0.5,
            reason=f"Out-of-combat: idle state on {signals.get('map', 'unknown')}",
            domain="behavior",
        ))
