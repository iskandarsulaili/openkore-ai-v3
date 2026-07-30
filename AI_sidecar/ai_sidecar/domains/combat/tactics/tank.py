"""
Tank tactics — aggro management, damage mitigation, and party protection.

Used by: Swordman, Knight, Crusader, and other tank classes.
Pro RO behavior: provoke to maintain aggro, Endure to prevent flinch, position between party and enemy.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from ai_sidecar.domains.combat.tactics.base import BaseTactics, TacticsContext, TargetInfo
from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)


class TankTactics(BaseTactics):
    """Tank tactics: hold aggro, mitigate damage, protect party.

    Priority 55 — high priority (party survival).

    Pro RO Knight/Crusader behavior:
    - Provoke (Lv10) every 5s to maintain aggro (-40% DEF for target)
    - Endure (Lv10) to prevent flinch from monster attacks
    - Position between party and enemy (body blocking)
    - Use Bowling Bash for AoE aggro
    - Use potions aggressively — tank must stay alive
    """

    name: ClassVar[str] = "tank"
    priority: ClassVar[int] = 55
    description: ClassVar[str] = "Tank — aggro management and damage mitigation"
    role_type: ClassVar[str] = "tank"

    OPTIMAL_RANGE = 1  # Melee range
    PROVOKE_RANGE = 9  # Can provoke from distance

    def select_target(self, ctx: TacticsContext) -> TargetInfo | None:
        """Tank target priority:
        1. Monsters attacking party members.
        2. Casting monsters (interrupt with provoke/attack).
        3. Boss/MVP monsters.
        4. Nearest aggressive monster.
        """
        monsters = ctx.monsters
        if not monsters:
            return None

        def scorer(t: TargetInfo, c: TacticsContext) -> float:
            score = 0.0

            # Monster attacking a party member — highest priority
            if getattr(t, 'target', '') == "party_member":
                score += 100.0

            # Interrupt casting monsters
            if t.is_casting:
                score += 60.0

            # Boss / MVP — tank must hold aggro
            if t.is_boss:
                score += 80.0

            # Nearest monster (get into aggro range)
            score += max(0, 15 - t.distance) * 3

            # Aggressive monsters
            if t.is_aggressive:
                score += 15.0

            # Monster with high ATK (needs to be controlled)
            if getattr(t, 'attack', 0) > 100:
                score += 20.0

            return score

        return self._find_best_by_score(ctx, scorer)

    def select_skill(self, ctx: TacticsContext, target: TargetInfo | None) -> tuple[str, int] | None:
        """Tank skill selection: provoke → endure → attack.

        Pro RO Tank rotation:
        1. Endure (Lv10) before engaging — prevents flinch
        2. Provoke (Lv10) to maintain aggro
        3. Bowling Bash for AoE aggro on groups
        4. Bash as filler
        """
        if not target:
            return None

        available = set(s.upper() for s in ctx.available_skills)

        # Endure first — prevents flinch from all attacks (Lv10 = 10 hits)
        if "SM_ENDURE" in available and "ENDURE" not in set(b.upper() for b in ctx.active_buffs):
            if ctx.cooldowns.get("sm_endure", 0) <= 0:
                return ("sm_endure", 10)

        # Provoke to maintain aggro (Lv10 = -40% DEF for target)
        if "SM_PROVOKE" in available:
            if ctx.cooldowns.get("sm_provoke", 0) <= 0:
                return ("sm_provoke", 10)

        # Bowling Bash for AoE aggro on groups
        if ctx.aggro_count >= 2 and "KN_BOWLINGBASH" in available:
            if ctx.cooldowns.get("kn_bowlingbash", 0) <= 0:
                return ("kn_bowlingbash", 10)

        # Bash as filler (Lv10 = 420% damage)
        if "SM_BASH" in available and ctx.my_sp > 10:
            return ("sm_bash", 10)

        return None

    def evaluate_positioning(self, ctx: TacticsContext, target: TargetInfo | None) -> dict[str, Any] | None:
        """Tank positioning: between party and monsters.

        Pro RO Tank:
        - Stay at melee range (1 cell)
        - Position between party and approaching monsters
        - Don't chase runners — let ranged DPS handle them
        """
        if target is None:
            return None

        if target.distance > 1:
            return {
                "move_x": 0,
                "move_y": 0,
                "reason": f"closing_to_{target.name}_tank_aggro",
                "urgency": 0.8,
            }

        return None

    def assess_buffs(self, ctx: TacticsContext) -> list[str]:
        """Keep tank buffs active.

        Pro RO Tank:
        - Endure (prevents flinch, allows uninterrupted attacks)
        - Increase Agility if available (more ASPD = more aggro)
        - Defender (Crusader) for DEF boost
        """
        needed: list[str] = []
        available = set(s.upper() for s in ctx.available_skills)
        active = set(b.upper() for b in ctx.active_buffs)

        # Endure — prevents flinch
        if "SM_ENDURE" in available and "ENDURE" not in active:
            needed.append("sm_endure")

        # Increase Agility (from support)
        if "AL_INCAGI" in available and "INCAGI" not in active:
            needed.append("al_incagi")

        # Defender (Crusader)
        if "CR_DEFENDER" in available and "DEFENDER" not in active:
            needed.append("cr_defender")

        return needed

    def assess_emergency(self, ctx: TacticsContext) -> HeuristicAction | None:
        """Tank emergency: use potions aggressively, hold position.

        Pro RO Tank:
        - Use potions at 50% HP (tank stays alive to hold aggro)
        - Endure before taking big hits
        - NEVER flee while party is alive (tank holds the line)
        """
        if ctx.my_hp_pct < 0.5:
            return self._make_action(
                command="use_potion_or_heal",
                reason="tank_hp_low_hold_position",
                confidence=0.9,
                hp_pct=ctx.my_hp_pct,
            )

        # Endure reactively if HP is dropping fast
        if ctx.my_hp_pct < 0.7 and "SM_ENDURE" in set(s.upper() for s in ctx.available_skills):
            if "ENDURE" not in set(b.upper() for b in ctx.active_buffs):
                return self._make_action(
                    command="use_skill sm_endure",
                    reason="tank_reactive_endure",
                    confidence=0.8,
                    hp_pct=ctx.my_hp_pct,
                )

        return None

    def build_rotation(self, ctx: TacticsContext, target: TargetInfo | None) -> list[tuple[str, int]]:
        """Tank rotation: Endure → Provoke → attack.

        Pro RO Tank rotation:
        1. Endure (flinch immunity)
        2. Provoke (aggro + DEF debuff)
        3. Bowling Bash (AoE aggro)
        4. Bash (single target filler)
        """
        rotation: list[tuple[str, int]] = []
        available = set(s.upper() for s in ctx.available_skills)

        # Endure for flinch immunity
        if "SM_ENDURE" in available:
            rotation.append(("sm_endure", 10))

        # Provoke for aggro
        if "SM_PROVOKE" in available:
            rotation.append(("sm_provoke", 10))

        # Bowling Bash for AoE
        if ctx.aggro_count >= 2 and "KN_BOWLINGBASH" in available:
            rotation.append(("kn_bowlingbash", 10))

        # Bash filler
        if "SM_BASH" in available:
            rotation.append(("sm_bash", 10))

        return rotation
