"""Melee DPS tactics — burst damage, positioning, and back attacks.

Used by: Thief, Assassin, Rogue, and other melee damage dealers.
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
    """

    name: ClassVar[str] = "melee_dps"
    priority: ClassVar[int] = 50
    description: ClassVar[str] = "Melee damage dealer — burst priority and positioning"
    role_type: ClassVar[str] = "dps"

    # Burst skills (high SP cost, high damage)
    BURST_SKILLS = {"AS_SONICBLOW", "TF_POISON", "AS_VENOMDUST", "AS_GRIMTOOTH"}
    # Combo enablers
    COMBO_SKILLS = {"AS_RIGHT", "AS_LEFT", "AS_KATAR", "TF_DOUBLE"}
    # Utility
    UTILITY_SKILLS = {"TF_HIDING", "AS_CLOAKING"}

    def select_target(self, ctx: TacticsContext) -> TargetInfo | None:
        """Melee DPS target priority:
        1. Low HP targets (finish kills).
        2. Casting monsters (interrupt).
        3. High value targets (cards, drops).
        4. Nearest monster.
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

            # Interrupt casting monsters
            if t.is_casting:
                score += 50.0

            # Boss / MVP priority
            if t.is_boss:
                score += 30.0

            # Value (drops, cards)
            score += t.estimated_value * 0.1

            # Proximity — melee must be close
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

            return score

        return self._find_best_by_score(ctx, scorer)

    def select_skill(self, ctx: TacticsContext, target: TargetInfo | None) -> tuple[str, int] | None:
        """Melee DPS skill selection: burst → combo → basic.

        Uses Sonic Blow / Grimtooth as primary, Double Attack as passive filler.
        """
        if not target:
            return None

        available = set(s.upper() for s in ctx.available_skills)

        # Sonic Blow for burst damage (high SP cost, save for important targets)
        if "AS_SONICBLOW" in available and ctx.my_sp_pct > 0.3:
            if ctx.cooldowns.get("as_sonicblow", 0) <= 0:
                return ("as_sonicblow", 10)

        # Grimtooth as ranged opener (can hit through walls in PvP)
        if "AS_GRIMTOOTH" in available and target.distance > 2:
            if ctx.cooldowns.get("as_grimtooth", 0) <= 0:
                return ("as_grimtooth", 5)

        # Venom Dust for AoE poison
        if "AS_VENOMDUST" in available and ctx.aggro_count >= 2:
            if ctx.cooldowns.get("as_venomdust", 0) <= 0:
                return ("as_venomdust", 5)

        # Poison single target
        if "TF_POISON" in available and ctx.my_sp > 15:
            if ctx.cooldowns.get("tf_poison", 0) <= 0:
                return ("tf_poison", 5)

        # Envenom as damage-over-time opener
        if "TF_POISON" in available and target.hp_pct > 0.8 and ctx.my_sp_pct > 0.5:
            return ("tf_poison", 5)

        # Hiding for emergency (low HP, dangerous situation)
        if ctx.my_hp_pct < 0.25 and "TF_HIDING" in available:
            return ("tf_hiding", 5)

        # Double Attack is passive — just basic attack
        return None

    def evaluate_positioning(self, ctx: TacticsContext, target: TargetInfo | None) -> dict[str, Any] | None:
        """Melee DPS positioning: get close (within 1 cell).

        Returns movement intent if too far.
        """
        if target is None:
            return None

        if target.distance > 1:
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
        """Melee DPS emergency: hide to drop aggro, or potion."""
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
        return None

    def build_rotation(self, ctx: TacticsContext, target: TargetInfo | None) -> list[tuple[str, int]]:
        """Melee DPS rotation: buff → poison → burst → basic."""
        rotation: list[tuple[str, int]] = []
        available = set(s.upper() for s in ctx.available_skills)

        # Poison opener for long fights
        if target and target.hp_pct > 0.7 and "TF_POISON" in available:
            rotation.append(("tf_poison", 5))

        # Sonic Blow burst
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
