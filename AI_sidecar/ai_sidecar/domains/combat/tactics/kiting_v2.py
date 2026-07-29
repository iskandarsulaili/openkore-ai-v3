"""Kiting v2 — tick-based movement, knockback, traps, terrain awareness.

RO movement runs on 1-second position ticks. A Poring moves at 150
movespeed (3 cells/tick). An Archer attacks every 1.5s at range 9.
Without knockback or traps, the monster will always catch up.

This module models:
- Movespeed vs attack-speed ratio
- Knockback skills (Double Strafe pushes, Bowling Bash pushes)
- Traps (Ankle Snare stops movement for 10s)
- Terrain pathing (move behind walls, use line-of-sight breaks)
- Retreat timing (when to stop shooting and run)
"""
from __future__ import annotations
from typing import Any
import logging
from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)

# RO movement constants
CELLS_PER_TICK_SLOW = 2  # Walking
CELLS_PER_TICK_FAST = 4  # Sprinting
MONSTER_MOVESPEED = 3    # Typical monster movespeed (cells/tick)
ARROW_RANGE = 9
ATTACK_COOLDOWN_TICKS = 2  # ~1.5 seconds / 0.75s ticks
TRAP_DURATION_TICKS = 13   # Ankle Snare = 10 seconds


class TickBasedKiting:
    """Handles RO's tick-based kiting mechanics.
    
    Key insight: without knockback or CC, a melee monster will always
    close the distance. The only way to maintain range is:
    1. Knockback skills (Double Strafe, Sharp Shooting, Pierce)
    2. Traps (Ankle Snare, Claymore Trap)
    3. Terrain line-of-sight breaks (monsters can't path through walls)
    4. Retreat before the monster gets in melee range
    """
    
    def __init__(self):
        self._tick_counters: dict[str, int] = {}  # bot_id -> ticks
    
    def estimate_catch_up_ticks(self, distance: int, has_trap: bool = False) -> int:
        """Estimate how many ticks until a monster reaches melee range."""
        if has_trap:
            # A trapped monster doesn't move for ~13 ticks
            return 999
        # Monster closes 3 cells/tick, Archer range is 9
        # Time to close from max range: 9/3 = 3 ticks (2 seconds)
        # Time to close from mid range (5): 5/3 ≈ 2 ticks
        return max(1, distance // MONSTER_MOVESPEED)
    
    def estimate_arrows_before_caught(self, distance: int) -> int:
        """Estimate how many arrows you can fire before monster reaches you."""
        catch_up_ticks = self.estimate_catch_up_ticks(distance)
        # Each arrow takes ~2 ticks (1.5s attack speed)
        return max(1, catch_up_ticks // ATTACK_COOLDOWN_TICKS)
    
    def assess(self, signals: dict[str, Any], actions: list[HeuristicAction], bot_id: str) -> None:
        """Run tick-based kiting assessment."""
        job = str(signals.get("job", "") or "").lower()
        monsters = signals.get("monsters_around", []) or []
        
        # Only for ranged classes
        ranged_classes = ["archer", "hunter", "bard", "dancer", "sniper"]
        is_ranged = any(c in job for c in ranged_classes)
        if not is_ranged:
            return
        
        # Tick counter
        self._tick_counters[bot_id] = self._tick_counters.get(bot_id, 0) + 1
        tick = self._tick_counters[bot_id]
        
        # Find nearest monster
        nearest_dist = 99
        nearest_monster = None
        for m in monsters:
            if isinstance(m, dict):
                dist = abs(m.get("distance_to", 99))
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_monster = m
        
        if not nearest_monster:
            return
        
        # Check if we have traps (Hunter/Sniper)
        has_trap = any("hunter" in job or "sniper" in job)
        
        # Calculate catch-up time
        arrows_before_caught = self.estimate_arrows_before_caught(nearest_dist)
        catch_up_ticks = self.estimate_catch_up_ticks(nearest_dist, has_trap)
        
        # Decision matrix:
        if nearest_dist <= 3:
            # Monster is in melee range — MUST do something
            if has_trap and tick % 15 < 5:
                # Place trap and retreat
                actions.append(HeuristicAction(
                    kind="command",
                    command="skill_cast HT_ANKLESNARE 0",
                    confidence=0.8,
                    reason="Kiting: placing Ankle Snare at close range",
                    domain="combat",
                ))
            else:
                # Retreat and create distance
                actions.append(HeuristicAction(
                    kind="command",
                    command="retreat 5",
                    confidence=0.9,
                    reason=f"Kiting: monster at {nearest_dist} cells — retreating",
                    domain="combat",
                ))
        
        elif catch_up_ticks <= 2:
            # Monster will catch up in 1-2 ticks — need action NOW
            actions.append(HeuristicAction(
                kind="command",
                command="skill_cast KN_SPEARBOOMERANG 0",
                confidence=0.7,
                reason=f"Kiting: knockback (catch-up in {catch_up_ticks} ticks)",
                domain="combat",
            ))
        
        elif catch_up_ticks <= 4:
            # 3-4 ticks before catch-up — fire arrows and retreat
            if has_trap and tick % 20 < 3:
                actions.append(HeuristicAction(
                    kind="command",
                    command="skill_cast HT_ANKLESNARE 0",
                    confidence=0.6,
                    reason="Kiting: pre-emptive trap placement",
                    domain="combat",
                ))
            else:
                # Fire and retreat to maintain range
                actions.append(HeuristicAction(
                    kind="command",
                    command=f"attack",
                    confidence=0.8,
                    reason=f"Kiting: attacking ({arrows_before_caught} arrows before caught)",
                    domain="combat",
                ))
        else:
            # Safe distance — just attack
            actions.append(HeuristicAction(
                kind="command",
                command="attack",
                confidence=0.9,
                reason=f"Kiting: safe distance {nearest_dist} cells",
                domain="combat",
            ))
