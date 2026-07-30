"""
Kiting v2 — tick-based movement, knockback, traps, terrain awareness.

RO movement runs on 1-second position ticks. A Poring moves at 150
movespeed (3 cells/tick). An Archer attacks every 1.5s at range 9.
Without knockback or traps, the monster will always catch up.

This module models:
- Movespeed vs attack-speed ratio (catch-up math)
- Knockback skills (Double Strafe pushes, Bowling Bash pushes)
- Traps (Ankle Snare stops movement for 10s)
- Terrain pathing (move behind walls, use line-of-sight breaks)
- Retreat timing (when to stop shooting and run)
- Decision matrix based on distance threshold
"""

from __future__ import annotations
from typing import Any
import logging
from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)

# RO movement constants
CELLS_PER_TICK_SLOW = 2    # Walking
CELLS_PER_TICK_FAST = 4    # Sprinting
MONSTER_MOVESPEED = 3      # Typical monster movespeed (cells/tick)
ARROW_RANGE = 9
ATTACK_COOLDOWN_TICKS = 2  # ~1.5 seconds / 0.75s ticks
TRAP_DURATION_TICKS = 13   # Ankle Snare = 10 seconds (~13 ticks)

# Kiting decision thresholds
MELEE_THRESHOLD = 3        # Monster is in melee range
DANGER_THRESHOLD = 6       # Monster will catch up in ~2 ticks
SAFE_THRESHOLD = 7         # Safe distance for attacking
MAX_RANGE = 9              # Max bow range


class TickBasedKiting:
    """Handles RO's tick-based kiting mechanics.
    
    Key insight: without knockback or CC, a melee monster will always
    close the distance. The only way to maintain range is:
    1. Knockback skills (Double Strafe, Sharp Shooting, Pierce)
    2. Traps (Ankle Snare, Claymore Trap)
    3. Terrain line-of-sight breaks (monsters can't path through walls)
    4. Retreat before the monster gets in melee range
    
    Pro RO kiting:
    - At <3 cells: trap or retreat (must act NOW)
    - At 3-6 cells: knockback to reset distance or pre-place trap
    - At 7-9 cells: attack normally (safe)
    - Use Ankle Snare vs fast monsters (Bapho Jr., Agav, etc.)
    """
    
    def __init__(self):
        self._tick_counters: dict[str, int] = {}  # bot_id -> ticks
    
    def estimate_catch_up_ticks(self, distance: int, has_trap: bool = False) -> int:
        """Estimate how many ticks until a monster reaches melee range.
        
        RO formula:
        - Monster moves at 3 cells/tick (most monsters)
        - Distance to cover = current_distance - 1 (melee starts at 1)
        - Ticks required = distance_to_cover / monster_speed
        
        A trapped monster doesn't move for ~13 ticks (Ankle Snare).
        """
        if has_trap:
            # Trapped monster doesn't move for 10 seconds
            return 999
        cells_to_close = max(1, distance - 1)
        return max(1, cells_to_close // MONSTER_MOVESPEED)
    
    def estimate_arrows_before_caught(self, distance: int) -> int:
        """Estimate how many arrows you can fire before monster reaches you.
        
        Each arrow takes ~2 ticks (1.5s attack cooldown at 150 ASPD).
        At max range (9 cells): 3 ticks / 2 = 1 arrow before caught
        At optimal range (7 cells): 2 ticks / 2 = 1 arrow
        """
        catch_up_ticks = self.estimate_catch_up_ticks(distance)
        return max(1, catch_up_ticks // ATTACK_COOLDOWN_TICKS)

    def assess_kiting_decision(self, distance: int, has_trap: bool, job: str) -> dict[str, Any]:
        """Make a kiting decision based on distance and available tools.
        
        Decision matrix:
        | Range | Condition | Action |
        |-------|-----------|--------|
        | ≤3    | No trap   | Retreat immediately |
        | ≤3    | Has trap  | Place Ankle Snare, retreat |
        | 3-6   | Catch-up ≤ 2 ticks | Knockback (Double Strafe) |
        | 3-6   | Has trap  | Pre-emptive trap |
        | 7-9   | Safe      | Attack normally |
        | >9    | Too far   | Approach slightly |
        """
        if distance <= MELEE_THRESHOLD:
            # Monster is in melee range — MUST act NOW
            if has_trap:
                return {
                    "action": "place_trap_and_retreat",
                    "skill": "HT_ANKLESNARE",
                    "priority": 100,
                    "urgency": 1.0,
                    "reason": f"Kiting: monster at {distance}c — placing trap and retreating",
                }
            else:
                return {
                    "action": "retreat",
                    "skill": None,
                    "priority": 100,
                    "urgency": 1.0,
                    "reason": f"Kiting: monster at {distance}c — retreating immediately",
                }

        elif distance <= DANGER_THRESHOLD:
            # Monster will catch up in 1-2 ticks
            if has_trap:
                return {
                    "action": "place_trap",
                    "skill": "HT_ANKLESNARE",
                    "priority": 90,
                    "urgency": 0.8,
                    "reason": f"Kiting: monster at {distance}c — pre-emptive trap",
                }
            else:
                # Use knockback to reset distance
                return {
                    "action": "knockback_attack",
                    "skill": "AC_DOUBLE",  # Double Strafe pushes back
                    "priority": 85,
                    "urgency": 0.7,
                    "reason": f"Kiting: monster at {distance}c — knockback to create space",
                }

        elif distance <= SAFE_THRESHOLD:
            # Still within catch-up range but safe for a few ticks
            arrows = self.estimate_arrows_before_caught(distance)
            return {
                "action": "attack_with_retreat",
                "skill": "AC_DOUBLE",
                "priority": 70,
                "urgency": 0.4,
                "reason": f"Kiting: {distance}c, {arrows} arrows before retreat needed",
            }

        elif distance <= MAX_RANGE:
            # Safe distance
            return {
                "action": "attack",
                "skill": "AC_DOUBLE",
                "priority": 60,
                "urgency": 0.2,
                "reason": f"Kiting: safe at {distance}c",
            }

        else:
            # Too far — approach slightly
            return {
                "action": "approach",
                "skill": None,
                "priority": 40,
                "urgency": 0.3,
                "reason": f"Kiting: target at {distance}c — approaching to bow range",
            }

    def assess(self, signals: dict[str, Any], actions: list[HeuristicAction], bot_id: str) -> None:
        """Run tick-based kiting assessment.
        
        Evaluates:
        1. Current distance to nearest monster
        2. Catch-up time in ticks
        3. Available kiting tools (traps, knockback)
        4. Makes decision from kiting matrix
        """
        job = str(signals.get("job", "") or "").lower()
        monsters = signals.get("monsters_around", []) or []
        
        # Only for ranged classes
        ranged_classes = ["archer", "hunter", "bard", "dancer", "sniper", "ranger"]
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
        has_trap = any(c in job for c in ["hunter", "sniper", "ranger"])
        
        # Get kiting decision
        decision = self.assess_kiting_decision(nearest_dist, has_trap, job)
        
        arrows_before_caught = self.estimate_arrows_before_caught(nearest_dist)
        catch_up_ticks = self.estimate_catch_up_ticks(nearest_dist, has_trap)
        
        # Translate decision into HeuristicAction
        action = decision["action"]
        
        if action == "place_trap_and_retreat":
            actions.append(HeuristicAction(
                kind="command",
                command="skill_cast HT_ANKLESNARE 0",
                confidence=0.9,
                reason=decision["reason"],
                domain="combat",
            ))
            actions.append(HeuristicAction(
                kind="command",
                command="retreat 5",
                confidence=0.9,
                reason="Kiting: retreating after trap placement",
                domain="combat",
            ))
        
        elif action == "retreat":
            actions.append(HeuristicAction(
                kind="command",
                command="retreat 5",
                confidence=0.95,
                reason=decision["reason"],
                domain="combat",
            ))
        
        elif action == "place_trap":
            actions.append(HeuristicAction(
                kind="command",
                command="skill_cast HT_ANKLESNARE 0",
                confidence=0.8,
                reason=decision["reason"],
                domain="combat",
            ))
        
        elif action == "knockback_attack":
            actions.append(HeuristicAction(
                kind="command",
                command="skill_cast AC_DOUBLE 10",
                confidence=0.75,
                reason=decision["reason"],
                domain="combat",
            ))
            # Follow up with retreat to maintain distance
            if tick % 3 == 0:
                actions.append(HeuristicAction(
                    kind="command",
                    command="retreat 3",
                    confidence=0.6,
                    reason="Kiting: maintaining distance after knockback",
                    domain="combat",
                ))
        
        elif action == "attack_with_retreat":
            actions.append(HeuristicAction(
                kind="command",
                command="attack",
                confidence=0.8,
                reason=f"{decision['reason']} (catch-up in {catch_up_ticks} ticks)",
                domain="combat",
            ))
            # Pre-emptive retreat to avoid getting caught
            if catch_up_ticks <= 3 and tick % 2 == 0:
                actions.append(HeuristicAction(
                    kind="command",
                    command="retreat 2",
                    confidence=0.6,
                    reason="Kiting: pre-emptive retreat to avoid melee",
                    domain="combat",
                ))
        
        elif action == "attack":
            actions.append(HeuristicAction(
                kind="command",
                command="attack",
                confidence=0.9,
                reason=decision["reason"],
                domain="combat",
            ))
        
        elif action == "approach":
            actions.append(HeuristicAction(
                kind="command",
                command="follow 1",
                confidence=0.6,
                reason=decision["reason"],
                domain="combat",
            ))
