"""
Melee DPS tactics — burst damage, positioning, and back attacks.

Used by: Thief, Assassin, Rogue, and other melee damage dealers.
Pro RO behavior: close distance immediately, burst from behind, use positioning.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from ai_sidecar.domains.combat.tactics.base import BaseTactics, TacticsContext, TargetInfo
from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)


class MeleeDPSTactics(BaseTactics):
    """Melee DPS tactics: maximize damage output, burst priority.

    Priority 50 — default damage role.

    Pro RO behavior:
    - Close distance to 1 cell immediately
    - Use burst skills (Sonic Blow) on priority targets
    - Position behind target for backstab bonus
    - Hide to drop aggro when overwhelmed
    - Use Throw Sand to blind dangerous casters
    """

    name: ClassVar[str] = "melee_dps"
    priority: ClassVar[int] = 50
    description: ClassVar[str] = "Melee damage dealer — burst priority and positioning"
    role_type: ClassVar[str] = "dps"

    # Optimal range for melee — always 1 cell
    OPTIMAL_RANGE = 1

    # Burst skills (high SP cost, high damage)
    BURST_SKILLS = {"AS_SONICBLOW", "TF_POISON", "AS_VENOMDUST", "AS_GRIMTOOTH"}
    # Combo enablers
    COMBO_SKILLS = {"AS_RIGHT", "AS_LEFT", "AS_KATAR", "TF_DOUBLE"}
    # Utility
    UTILITY_SKILLS = {"TF_HIDING", "AS_CLOAKING"}

    def select_target(self, ctx: TacticsContext) -> TargetInfo | None:
        """Melee DPS target priority:
        1. Low HP targets (finish kills — prevent escape/heal).
        2. Casting monsters (interrupt before they cast).
        3. High value targets (cards, drops).
        4. Nearest monster (close distance).
        """
        monsters = ctx.monsters
        if not monsters:
            return None

        def scorer(t: TargetInfo, c: TacticsContext) -> float:
            score = 0.0

            # Low HP priority — finish kills (highest)
            if t.hp_pct < 0.3:
                score += 80.0
            elif t.hp_pct < 0.5:
                score += 40.0

            # Interrupt casting monsters — melee can interrupt with attack
            if t.is_casting:
                score += 50.0

            # Boss / MVP priority
            if t.is_boss:
                score += 30.0

            # Value (drops, cards)
            score += t.estimated_value * 0.1

            # Proximity — melee must be close, prefer nearest
            score += max(0, 15 - t.distance) * 2

            # Aggressive monster = higher priority
            if t.is_aggressive:
                score += 10.0

            # Element advantage (wind vs earth, etc.)
            elem_mult = self._get_elem_multiplier(
                self._get_weapon_element(ctx), t.element
            )
            if elem_mult > 1.1:
                score += 15.0
                
            # Prefer targets within melee range already
            if t.distance <= 2:
                score += 20.0

            return score

        return self._find_best_by_score(ctx, scorer)

    def select_skill(self, ctx: TacticsContext, target: TargetInfo | None) -> tuple[str, int] | None:
        """Melee DPS skill selection: burst → combo → basic.

        Pro RO rotation:
        1. Close distance first
        2. Blind casters with Throw Sand
        3. Poison dangerous targets (DoT)
        4. Sonic Blow on priority targets (8 hits)
        5. Hide when HP is critical
        """
        if not target:
            return None

        available = set(s.upper() for s in ctx.available_skills)

        # Out of range — must close distance first
        if target.distance > self.OPTIMAL_RANGE:
            return None  # Let positioning handle movement

        # Sonic Blow for burst damage (high SP, 8 hits)
        if "AS_SONICBLOW" in available and ctx.my_sp_pct > 0.3:
            if ctx.cooldowns.get("as_sonicblow", 0) <= 0:
                return ("as_sonicblow", 10)

        # Grimtooth as ranged opener (can hit through walls in PvP)
        if "AS_GRIMTOOTH" in available and target.distance > 2:
            if ctx.cooldowns.get("as_grimtooth", 0) <= 0:
                return ("as_grimtooth", 5)

        # Venom Dust for AoE poison on groups
        if "AS_VENOMDUST" in available and ctx.aggro_count >= 2:
            if ctx.cooldowns.get("as_venomdust", 0) <= 0:
                return ("as_venomdust", 5)

        # Poison single target (DoT opener)
        if "TF_POISON" in available and ctx.my_sp > 15:
            if ctx.cooldowns.get("tf_poison", 0) <= 0:
                return ("tf_poison", 5)

        # Throw Sand to blind casting monsters
        if target.is_casting and "TF_THROW_SAND" in available:
            if ctx.cooldowns.get("tf_throw_sand", 0) <= 0:
                return ("tf_throw_sand", 5)

        # Hiding for emergency (low HP, dangerous situation)
        if ctx.my_hp_pct < 0.25 and "TF_HIDING" in available:
            return ("tf_hiding", 5)

        # Double Attack is passive — just basic attack
        return None

    def evaluate_positioning(self, ctx: TacticsContext, target: TargetInfo | None) -> dict[str, Any] | None:
        """Melee DPS positioning: close distance to 1 cell.

        Pro RO melee behavior:
        - Close to 1 cell immediately for maximum damage
        - Back away only to use ranged skills (Grimtooth)
        - If surrounded, back toward a wall to reduce exposure
        """
        if target is None:
            return None

        if target.distance > 1:
            if target.distance > 8:
                # Very far — run toward target
                return {
                    "move_x": 0,
                    "move_y": 0,
                    "reason": f"sprinting_to_{target.name}",
                    "urgency": 0.9,
                }
            else:
                # Close distance
                return {
                    "move_x": 0,
                    "move_y": 0,
                    "reason": f"closing_to_{target.name}",
                    "urgency": 0.7 if target.is_casting else 0.3,
                }

        return None

    def assess_buffs(self, ctx: TacticsContext) -> list[str]:
        """Keep offensive buffs active."""
        needed: list[str] = []
        available = set(s.upper() for s in ctx.available_skills)
        active = set(b.upper() for b in ctx.active_buffs)

        # Improve Concentration if archer subclass
        if "AC_OWL" in available and "OWLSEYE" not in active:
            needed.append("ac_owl")

        return needed

    def assess_emergency(self, ctx: TacticsContext) -> HeuristicAction | None:
        """Melee DPS emergency: hide to drop aggro, or potion.

        Pro RO behavior:
        - Hide when HP < 20% to drop all aggro instantly
        - Use potions BEFORE hiding (heal first, then vanish)
        - If can't hide, flee
        """
        if ctx.my_hp_pct < 0.2 and "TF_HIDING" in set(s.upper() for s in ctx.available_skills):
            return self._make_action(
                command="use_skill tf_hiding",
                reason="dps_emergency_hide_drop_aggro",
                confidence=0.9,
                hp_pct=ctx.my_hp_pct,
                aggro=ctx.aggro_count,
            )

        if ctx.my_hp_pct < 0.3:
            return self._make_action(
                command="use_potion_or_heal",
                reason="dps_hp_low",
                confidence=0.8,
                hp_pct=ctx.my_hp_pct,
            )
            
        # Flee if overwhelmed (3+ aggro and HP < 50%)
        if ctx.my_hp_pct < 0.5 and ctx.aggro_count >= 3:
            return self._make_action(
                command="flee_to_safe_spot",
                reason="dps_overwhelmed_flee",
                confidence=0.85,
                hp_pct=ctx.my_hp_pct,
                aggro=ctx.aggro_count,
            )

        return None

    def build_rotation(self, ctx: TacticsContext, target: TargetInfo | None) -> list[tuple[str, int]]:
        """Melee DPS rotation: buff → poison → burst → basic.

        Pro RO rotation for Assassin:
        1. Poison opener (TF_POISON) for DoT
        2. Sonic Blow burst (8 hits)
        3. Basic attack until cooldowns reset
        """
        rotation: list[tuple[str, int]] = []
        available = set(s.upper() for s in ctx.available_skills)

        # Close distance check
        if target and target.distance > self.OPTIMAL_RANGE:
            return rotation  # Empty rotation = move closer

        # Poison opener for long fights
        if target and target.hp_pct > 0.7 and "TF_POISON" in available:
            rotation.append(("tf_poison", 5))

        # Sonic Blow burst (8 hits, high damage)
        if ctx.my_sp_pct > 0.4 and "AS_SONICBLOW" in available:
            if ctx.cooldowns.get("as_sonicblow", 0) <= 0:
                rotation.append(("as_sonicblow", 10))

        return rotation

    @staticmethod
    def _get_weapon_element(ctx: TacticsContext) -> str:
        """Determine the effective attack element."""
        return ctx.my_weapon_element.lower() or "neutral"

    @staticmethod
    def _get_elem_multiplier(attack_element: str, defense_element: str) -> float:
        """Quick element multiplier lookup (pre-renewal Level 1)."""
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
            "holy": {"neutral": 1.0, "water": 1.0, "earth": 1.0, "fire": 1.0,
                     "wind": 1.0, "poison": 1.0, "holy": 0.25, "dark": 2.0,
                     "ghost": 1.0, "undead": 2.0},
            "undead": {"neutral": 1.0, "water": 1.0, "earth": 1.0, "fire": 1.25,
                       "wind": 1.0, "poison": 0.5, "holy": 2.0, "dark": 0.5,
                       "ghost": 1.0, "undead": 0.25},
            "dark": {"neutral": 1.0, "water": 1.0, "earth": 1.0, "fire": 1.0,
                     "wind": 1.0, "poison": 1.0, "holy": 0.5, "dark": 0.25,
                     "ghost": 1.0, "undead": 1.0},
            "ghost": {"neutral": 0.0, "water": 1.0, "earth": 1.0, "fire": 1.0,
                      "wind": 1.0, "poison": 1.0, "holy": 1.0, "dark": 1.0,
                      "ghost": 0.75, "undead": 1.0},
            "poison": {"neutral": 1.0, "water": 1.0, "earth": 0.75, "fire": 1.0,
                       "wind": 0.5, "poison": 0.25, "holy": 0.5, "dark": 1.0,
                       "ghost": 0.5, "undead": 0.5},
        }
        return table.get(attack_element, {}).get(defense_element, 1.0)
