"""Tank tactics — threat management, aggro skills, party protection.

Used by: Swordsman, Knight, Paladin, Crusader, and other defensive roles.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from ai_sidecar.domains.combat.tactics.base import BaseTactics, TacticsContext, TargetInfo
from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)


class TankTactics(BaseTactics):
    """Tank tactics: hold aggro, protect party, absorb damage.

    Priority 10 — runs before DPS modules so tanks select targets first.
    """

    name: ClassVar[str] = "tank"
    priority: ClassVar[int] = 10
    description: ClassVar[str] = "Tank/defensive — threat management and party protection"
    role_type: ClassVar[str] = "tank"

    # Provoke / taunt skill IDs
    TAUNT_SKILLS = {"SM_PROVOKE", "CR_PROVOCATE", "LG_BANDING"}
    # AoE aggro skills
    AOE_AGGRO_SKILLS = {"SM_MAGNUM", "KN_BOWLINGBASH", "CR_SHIELDBOOMERANG"}
    # Defensive buffs
    DEFENSIVE_BUFFS = {"SM_ENDURE", "CR_AUTOGUARD", "CR_REFLECTSHIELD",
                       "KN_AURA", "LG_SHIELDSPELL"}

    def select_target(self, ctx: TacticsContext) -> TargetInfo | None:
        """Tank target priority:
        1. Monsters attacking party members (peel).
        2. Casting monsters (interrupt).
        3. Most dangerous monster (highest ATK).
        4. Nearest aggressive monster.
        """
        monsters = ctx.monsters
        if not monsters:
            return None

        # Check if any monster is attacking a party member
        if ctx.has_party:
            for m in monsters:
                target_id = int(m.get("target_id", m.get("attack_target", 0)))
                if target_id > 0 and target_id != ctx.current_target_id:
                    # Check if this target is a party member
                    for pm in ctx.party_members:
                        if int(pm.get("actor_id", pm.get("id", 0))) == target_id:
                            info = self._monster_to_info(m)
                            info.score = 100.0 + (1.0 - info.hp_pct) * 10
                            info.reason = f"peeling_for_{pm.get('name', 'ally')}"
                            return info

        # Interrupt casting monsters
        casting = [m for m in monsters if m.get("is_casting", False)]
        if casting:
            c = casting[0]
            info = self._monster_to_info(c)
            info.score = 90.0
            info.reason = "interrupting_cast"
            return info

        # Score by danger level
        def scorer(t: TargetInfo, c: TacticsContext) -> float:
            score = 0.0
            # Aggressive monsters first
            if t.is_aggressive:
                score += 30.0
            # Boss priority
            if t.is_boss:
                score += 50.0
            # Casting monsters
            if t.is_casting:
                score += 40.0
            # Low HP = easy kill
            score += (1.0 - t.hp_pct) * 20.0
            # Closer = more immediate threat
            score += max(0, 20 - t.distance)
            return score

        return self._find_best_by_score(ctx, scorer)

    def select_skill(self, ctx: TacticsContext, target: TargetInfo | None) -> tuple[str, int] | None:
        """Tank skill selection:
        1. Provoke if not taunted.
        2. AoE aggro skills if multiple enemies.
        3. Defensive stance if HP low.
        4. Damage skills as filler.
        """
        if not target:
            return None

        available = set(s.upper() for s in ctx.available_skills)

        # Provoke priority: use on boss or when party members nearby
        for taunt in self.TAUNT_SKILLS:
            if taunt in available:
                taunt_lower = taunt.lower()
                # Check cooldown
                if ctx.cooldowns.get(taunt_lower, 0) <= 0:
                    if target.is_boss or ctx.party_members_nearby > 0:
                        return (taunt_lower, 5)
                    if "provoke" in taunt_lower or "provocate" in taunt_lower:
                        return (taunt_lower, 5)

        # AoE aggro when surrounded
        if ctx.aggro_count >= 3:
            for aoe in self.AOE_AGGRO_SKILLS:
                if aoe in available and ctx.cooldowns.get(aoe.lower(), 0) <= 0:
                    return (aoe.lower(), 5)

        # Defensive buff if low HP
        if ctx.my_hp_pct < 0.5:
            if "SM_ENDURE" in available and ctx.cooldowns.get("endure", 0) <= 0:
                return ("endure", 5)

        # Damage filler: bash is low SP cost
        if "SM_BASH" in available:
            return ("sm_bash", 5)
        if "KN_BRANDISHSPEAR" in available:
            return ("kn_brandishspear", 5)

        return None

    def evaluate_positioning(self, ctx: TacticsContext, target: TargetInfo | None) -> dict[str, Any] | None:
        """Tank positioning: between party and monsters.

        Returns a movement intent to interpose between threats and allies.
        """
        if not ctx.has_party or not target:
            return None

        # If we have a target and party members, stay between them
        if target.distance > 3:
            return {
                "move_x": 0,
                "move_y": 0,
                "reason": f"approaching_{target.name}",
                "urgency": 0.6,
            }
        return None

    def assess_buffs(self, ctx: TacticsContext) -> list[str]:
        """Keep defensive buffs active."""
        needed: list[str] = []
        available = set(s.upper() for s in ctx.available_skills)
        active = set(b.upper() for b in ctx.active_buffs)

        if "SM_ENDURE" in available and "ENDURE" not in active:
            needed.append("endure")
        if "CR_AUTOGUARD" in available and "AUTOGUARD" not in active:
            needed.append("cr_autoguard")
        if "CR_REFLECTSHIELD" in available and "REFLECTSHIELD" not in active:
            needed.append("cr_reflectshield")

        return needed

    def assess_emergency(self, ctx: TacticsContext) -> HeuristicAction | None:
        """Tank emergency: use potion early, activate defensive cooldowns."""
        if ctx.my_hp_pct < 0.25 and ctx.aggro_count > 0:
            return self._make_action(
                command="use_potion_or_heal",
                reason="tank_hp_critical_with_aggro",
                confidence=0.95,
                hp_pct=ctx.my_hp_pct,
                aggro=ctx.aggro_count,
            )
        return None

    def build_rotation(self, ctx: TacticsContext, target: TargetInfo | None) -> list[tuple[str, int]]:
        """Tank rotation: buff → taunt → AoE → Bash spam."""
        rotation: list[tuple[str, int]] = []
        available = set(s.upper() for s in ctx.available_skills)

        # Always maintain provoke uptime
        if "SM_PROVOKE" in available and target and ctx.cooldowns.get("sm_provoke", 0) <= 0:
            rotation.append(("sm_provoke", 5))

        # AoE if grouped
        if ctx.aggro_count >= 3 and "SM_MAGNUM" in available:
            if ctx.cooldowns.get("sm_magnum", 0) <= 0:
                rotation.append(("sm_magnum", 5))

        # Damage filler
        if "SM_BASH" in available:
            rotation.append(("sm_bash", 10))

        return rotation
