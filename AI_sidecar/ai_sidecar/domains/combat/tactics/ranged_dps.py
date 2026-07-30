"""
Ranged DPS tactics — distance maintenance, kiting, and Arrow Shower AoE.

Used by: Archer, Hunter, Sniper, and other ranged physical damage dealers.
Pro RO behavior: maintain distance, kite in circles, use knockback to reset distance.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from ai_sidecar.domains.combat.tactics.base import BaseTactics, TacticsContext, TargetInfo
from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)


class RangedDPSTactics(BaseTactics):
    """Ranged DPS tactics: maintain distance, kite, use AoE on groups.

    Priority 50 — default damage role.

    Pro RO behavior:
    - Maintain optimal range (7 cells for bow)
    - Use Arrow Shower knockback to reset distance
    - Kite in circles when approached
    - Use Improve Concentration for ASPD boost
    - Double Strafe as primary damage (380% at Lv10)
    """

    name: ClassVar[str] = "ranged_dps"
    priority: ClassVar[int] = 50
    description: ClassVar[str] = "Ranged damage dealer — kiting and distance management"
    role_type: ClassVar[str] = "dps"

    OPTIMAL_RANGE = 7  # Cells to maintain from target
    FLEE_RANGE = 4     # Distance at which to start backing up
    MAX_BOW_RANGE = 9  # Maximum bow range

    def select_target(self, ctx: TacticsContext) -> TargetInfo | None:
        """Ranged DPS target priority:
        1. Casting monsters (interrupt from safe distance).
        2. Low HP targets (finish kills).
        3. Monsters attacking party from range.
        4. Nearest monster that can be kited.
        """
        monsters = ctx.monsters
        if not monsters:
            return None

        def scorer(t: TargetInfo, c: TacticsContext) -> float:
            score = 0.0

            # Casting monsters — ranged interrupt
            if t.is_casting:
                score += 60.0

            # Low HP — rapid finish
            if t.hp_pct < 0.3:
                score += 50.0
            elif t.hp_pct < 0.5:
                score += 25.0

            # Element advantage (fire arrows vs undead, etc.)
            elem_mult = self._get_elem_multiplier(
                ctx.my_weapon_element, t.element
            )
            if elem_mult > 1.1:
                score += 20.0

            # Boss priority
            if t.is_boss:
                score += 30.0

            # Prefer targets at optimal range (7 cells)
            distance_score = 15 - abs(t.distance - self.OPTIMAL_RANGE)
            score += distance_score

            # Aggressive monsters
            if t.is_aggressive:
                score += 10.0

            # Value
            score += t.estimated_value * 0.05

            return score

        return self._find_best_by_score(ctx, scorer)

    def select_skill(self, ctx: TacticsContext, target: TargetInfo | None) -> tuple[str, int] | None:
        """Ranged DPS skill selection: Double Strafe primary, Arrow Shower AoE.

        Pro RO rotation:
        - Arrow Shower for knockback when too close
        - Arrow Shower for AoE on groups
        - Double Strafe spam for single-target DPS
        - Use Improve Concentration before engaging
        """
        if not target:
            return None

        available = set(s.upper() for s in ctx.available_skills)

        # Arrow Shower (knockback) when too close — reset distance
        if target.distance < self.FLEE_RANGE and "AC_SHOWER" in available:
            if ctx.cooldowns.get("ac_shower", 0) <= 0:
                return ("ac_shower", 5)

        # AoE on grouped enemies
        if ctx.aggro_count >= 3 and "AC_SHOWER" in available:
            if ctx.cooldowns.get("ac_shower", 0) <= 0:
                return ("ac_shower", 5)

        # Too close to target — back up first (don't shoot at melee range)
        if target.distance < self.OPTIMAL_RANGE - 2:
            return None  # Let positioning handle retreat

        # Double Strafe as primary damage (380% at Lv10)
        if "AC_DOUBLE" in available:
            if ctx.my_sp_pct > 0.2:
                return ("ac_double", 10)
            elif ctx.my_sp_pct > 0.1:
                return ("ac_double", 5)  # Lower level to conserve SP

        # Arrow Shower as AoE filler
        if "AC_SHOWER" in available:
            return ("ac_shower", 5)

        return None

    def evaluate_positioning(self, ctx: TacticsContext, target: TargetInfo | None) -> dict[str, Any] | None:
        """Ranged positioning: maintain optimal range, flee if too close.

        Pro RO kiting behavior:
        - Back up when target is within FLEE_RANGE (4 cells)
        - Approach slightly when target is beyond MAX_BOW_RANGE (9 cells)
        - Stay at OPTIMAL_RANGE (7 cells) for maximum efficiency
        """
        if target is None:
            return None

        # Too close — back up immediately (high urgency)
        if target.distance < self.FLEE_RANGE:
            return {
                "move_x": 0,
                "move_y": 0,
                "reason": f"backing_away_from_{target.name}",
                "urgency": 0.8,
            }

        # Slightly below optimal — back up slowly
        if target.distance < self.OPTIMAL_RANGE - 1:
            return {
                "move_x": 0,
                "move_y": 0,
                "reason": f"adjusting_range_for_{target.name}",
                "urgency": 0.4,
            }

        # Too far — approach slightly
        if target.distance > self.MAX_BOW_RANGE:
            return {
                "move_x": 0,
                "move_y": 0,
                "reason": f"approaching_{target.name}",
                "urgency": 0.3,
            }

        return None

    def assess_buffs(self, ctx: TacticsContext) -> list[str]:
        """Keep offensive/ASPD buffs active.

        Pro RO ranger:
        - Improve Concentration for ASPD (always on)
        - Owl's Eye for DEX passive
        - True Sight for HIT bonus
        """
        needed: list[str] = []
        available = set(s.upper() for s in ctx.available_skills)
        active = set(b.upper() for b in ctx.active_buffs)

        # Improve Concentration
        if "AC_CONCENTRATION" in available and "CONCENTRATION" not in active:
            needed.append("ac_concentration")
        elif "AC_OWL" in available and "OWLSEYE" not in active:
            needed.append("ac_owl")

        # True Sight
        if "HT_TRUESIGHT" in available and "TRUESIGHT" not in active:
            needed.append("ht_truesight")

        return needed

    def assess_emergency(self, ctx: TacticsContext) -> HeuristicAction | None:
        """Ranged emergency: flee if overwhelmed.

        Pro RO ranger:
        - Use Fly Wing if surrounded by 3+ at low HP
        - Use potions at higher threshold (ranged takes fewer hits)
        - Teleport if HP is critical
        """
        if ctx.my_hp_pct < 0.3 and ctx.aggro_count > 2:
            return self._make_action(
                command="flee_to_safe_spot",
                reason="ranged_overwhelmed_flee",
                confidence=0.9,
                hp_pct=ctx.my_hp_pct,
                aggro=ctx.aggro_count,
            )

        if ctx.my_hp_pct < 0.4:
            return self._make_action(
                command="use_potion_or_heal",
                reason="ranged_hp_low",
                confidence=0.8,
                hp_pct=ctx.my_hp_pct,
            )

        return None

    def build_rotation(self, ctx: TacticsContext, target: TargetInfo | None) -> list[tuple[str, int]]:
        """Ranged rotation: buff → Double Strafe spam → AoE as needed.

        Pro RO rotation for Hunter:
        1. Buff up (Improve Concentration)
        2. Double Strafe at Lv10 (380% damage)
        3. Arrow Shower if grouped (knockback + AoE)
        4. Maintain distance between shots
        """
        rotation: list[tuple[str, int]] = []
        available = set(s.upper() for s in ctx.available_skills)

        # Check range — don't shoot if too close
        if target and target.distance < self.OPTIMAL_RANGE - 2:
            return rotation  # Empty rotation = reposition

        # Double Strafe spam (Lv10 = 380%)
        if "AC_DOUBLE" in available and ctx.my_sp_pct > 0.2:
            rotation.append(("ac_double", 10))
        elif "AC_DOUBLE" in available:
            rotation.append(("ac_double", 5))  # Conserve SP

        # Arrow Shower for grouped mobs
        if ctx.aggro_count >= 3 and "AC_SHOWER" in available:
            rotation.append(("ac_shower", 5))

        # Fallback to Double Strafe
        if "AC_DOUBLE" in available:
            rotation.append(("ac_double", 10))

        return rotation

    @staticmethod
    def _get_elem_multiplier(attack_element: str, defense_element: str) -> float:
        """Quick element multiplier lookup."""
        table = {
            "neutral": {"neutral": 1.0, "water": 0.75, "earth": 0.75, "fire": 0.75,
                        "wind": 0.75, "poison": 0.75, "holy": 0.75, "dark": 0.75,
                        "ghost": 0.5, "undead": 0.5},
            "fire": {"neutral": 1.0, "water": 0.5, "earth": 1.25, "fire": 0.25,
                     "wind": 0.75, "poison": 0.75, "holy": 1.0, "dark": 1.0,
                     "ghost": 0.5, "undead": 1.25},
            "water": {"neutral": 1.0, "water": 0.25, "earth": 0.75, "fire": 1.25,
                      "wind": 0.5, "poison": 0.75, "holy": 1.0, "dark": 1.0,
                      "ghost": 0.5, "undead": 1.0},
            "wind": {"neutral": 1.0, "water": 1.25, "earth": 0.5, "fire": 1.25,
                     "wind": 0.25, "poison": 0.75, "holy": 1.0, "dark": 1.0,
                     "ghost": 0.5, "undead": 1.0},
            "earth": {"neutral": 1.0, "water": 1.25, "earth": 0.25, "fire": 0.75,
                      "wind": 1.25, "poison": 0.75, "holy": 1.0, "dark": 1.0,
                      "ghost": 0.5, "undead": 1.0},
        }
        return table.get(attack_element, {}).get(defense_element, 1.0)
