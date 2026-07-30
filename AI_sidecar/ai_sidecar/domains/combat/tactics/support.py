"""
Support tactics — party healing, buffing, and survivability management.

Used by: Acolyte, Priest, Monk, and other support/healing classes.
Pro RO behavior: keep party buffed, heal BEFORE critical, stay behind party.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from ai_sidecar.domains.combat.tactics.base import BaseTactics, TacticsContext, TargetInfo
from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)


class SupportTactics(BaseTactics):
    """Support tactics: buff, heal, and utility casting.

    Priority 60 — medium-high priority (survival first).

    Pro RO Priest behavior:
    - Cast Blessing (Lv10) on party members for STR+DEX+INT+30
    - Cast Increase Agility (Lv10) for AGI+12 and Flee+12
    - Heal party members BEFORE they reach critical HP
    - Stay 5 cells behind the main party (safe position)
    - Use Holy Light against undead (2x damage)
    - Resurrect fallen party members when safe
    """

    name: ClassVar[str] = "support"
    priority: ClassVar[int] = 60
    description: ClassVar[str] = "Support — party healing and buffing"
    role_type: ClassVar[str] = "support"

    # Position — stay behind the party line
    OPTIMAL_RANGE = 5
    RETREAT_HP_THRESHOLD = 0.35  # Heal self before this

    def select_target(self, ctx: TacticsContext) -> TargetInfo | None:
        """Support target priority:
        1. Monsters attacking party members (defensive).
        2. Undead monsters (Holy Light = 2x damage).
        3. Nearest aggressive monster.
        """
        monsters = ctx.monsters
        if not monsters:
            return None

        def scorer(t: TargetInfo, c: TacticsContext) -> float:
            score = 0.0

            # Undead monster — support can kill with Holy Light
            if t.race.lower() == "undead":
                score += 40.0

            # Monster attacking a party member (defensive priority)
            if getattr(t, 'target', '') == "party_member":
                score += 60.0

            # Aggressive monster approaching
            if t.is_aggressive:
                score += 20.0

            # Distance — prefer closer monsters
            score += max(0, 15 - t.distance)

            return score

        return self._find_best_by_score(ctx, scorer)

    def select_skill(self, ctx: TacticsContext, target: TargetInfo | None) -> tuple[str, int] | None:
        """Support skill selection: heal → buff → utility.

        Pro RO Priest:
        - Heal party members with low HP
        - Use Holy Light on undead
        - Use Turn Undead for instant kill chance
        """
        available = set(s.upper() for s in ctx.available_skills)

        # Heal self if HP is low (support must stay alive)
        if ctx.my_hp_pct < self.RETREAT_HP_THRESHOLD and "AL_HEAL" in available:
            return ("al_heal", 10)

        # Heal party members if available
        party_hp = getattr(ctx, 'party_member_hp_pct', 1.0)
        if party_hp < 0.5 and "AL_HEAL" in available:
            return ("al_heal", 10)

        # Holy Light on undead
        if target and target.race.lower() == "undead":
            if "AL_HEAL" in available and ctx.my_sp > 15:
                # Heal damages undead too
                return ("al_heal", 10)
            if "AL_HOLYLIGHT" in available:
                return ("al_holy_light", 10)

        # Turn Undead for instant kill chance on strong undead
        if target and target.race.lower() == "undead" and target.hp_pct > 0.5:
            if "AL_TURNUNDEAD" in available and ctx.my_sp > 25:
                if ctx.cooldowns.get("al_turnundead", 0) <= 0:
                    return ("al_turnundead", 10)

        # Blessing buff on party (STR+DEX+INT+30 at Lv10)
        if "AL_BLESS" in available and ctx.my_sp > 15:
            if ctx.cooldowns.get("al_bless", 0) <= 0:
                return ("al_bless", 10)

        return None

    def evaluate_positioning(self, ctx: TacticsContext, target: TargetInfo | None) -> dict[str, Any] | None:
        """Support positioning: stay behind party at safe range.

        Pro RO Priest:
        - Stay at 5 cells from target (behind DPS)
        - Back up if being targeted by multiple mobs
        - Keep distance from undead (they hit hard)
        """
        if target is None:
            return None

        # Too close — back up (support is squishy)
        if target.distance < 3:
            return {
                "move_x": 0,
                "move_y": 0,
                "reason": f"backing_from_{target.name}",
                "urgency": 0.7,
            }

        # Too far to cast heals — approach slightly
        if target.distance > self.OPTIMAL_RANGE + 3:
            return {
                "move_x": 0,
                "move_y": 0,
                "reason": f"approaching_{target.name}_for_heal_range",
                "urgency": 0.3,
            }

        return None

    def assess_buffs(self, ctx: TacticsContext) -> list[str]:
        """Keep party buffs active.

        Pro RO Priest buff rotation:
        1. Blessing (Lv10) — +STR, +DEX, +INT
        2. Increase Agility (Lv10) — +AGI, +Flee
        3. Kyrie Eleison (Lv5+) — absorbs damage
        4. Gloria (Lv5) — +LUK for crit party
        5. Magnificat (Lv5) — fast SP regen
        """
        needed: list[str] = []
        available = set(s.upper() for s in ctx.available_skills)
        active = set(b.upper() for b in ctx.active_buffs)

        # Blessing
        if "AL_BLESS" in available and "BLESSING" not in active:
            needed.append("al_bless")

        # Increase Agility
        if "AL_INCAGI" in available and "INCAGI" not in active:
            needed.append("al_incagi")

        # Kyrie Eleison (shield)
        if "PR_KYRIE" in available and "KYRIE" not in active:
            needed.append("pr_kyrie")

        return needed

    def assess_emergency(self, ctx: TacticsContext) -> HeuristicAction | None:
        """Support emergency: heal self first, then party.

        Pro RO Priest:
        - Heal self when HP < 35% (must stay alive to support)
        - Heal party member when HP < 40%
        - Flee if overwhelmed (3+ aggro)
        """
        # Self heal on critical
        if ctx.my_hp_pct < self.RETREAT_HP_THRESHOLD:
            return self._make_action(
                command="use_skill al_heal",
                reason="support_self_heal",
                confidence=0.95,
                hp_pct=ctx.my_hp_pct,
            )

        # Party heal
        party_hp = getattr(ctx, 'party_member_hp_pct', 1.0)
        if party_hp < 0.4:
            return self._make_action(
                command="use_skill al_heal_target",
                reason="support_party_heal",
                confidence=0.9,
            )

        # Flee if overwhelmed
        if ctx.my_hp_pct < 0.4 and ctx.aggro_count > 3:
            return self._make_action(
                command="flee_to_safe_spot",
                reason="support_overwhelmed",
                confidence=0.85,
                hp_pct=ctx.my_hp_pct,
                aggro=ctx.aggro_count,
            )

        return None

    def build_rotation(self, ctx: TacticsContext, target: TargetInfo | None) -> list[tuple[str, int]]:
        """Support rotation: buff → heal → utility attack.

        Pro RO Priest rotation:
        1. Blessing (buff)
        2. Increase Agility (buff)
        3. Heal party members
        4. Holy Light on undead
        """
        rotation: list[tuple[str, int]] = []
        available = set(s.upper() for s in ctx.available_skills)

        # Blessing
        if "AL_BLESS" in available:
            rotation.append(("al_bless", 10))

        # Increase Agility
        if "AL_INCAGI" in available:
            rotation.append(("al_incagi", 10))

        # Heal (also damages undead)
        if "AL_HEAL" in available and ctx.my_sp > 15:
            rotation.append(("al_heal", 10))

        # Holy Light for offense
        if target and target.race.lower() == "undead" and "AL_HOLYLIGHT" in available:
            rotation.append(("al_holy_light", 10))

        return rotation
