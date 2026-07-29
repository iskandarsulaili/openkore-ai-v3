"""Support tactics — buff timing, healing priority, debuff removal.

Used by: Acolyte, Priest, Monk, and other support/healing roles.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from ai_sidecar.domains.combat.tactics.base import BaseTactics, TacticsContext, TargetInfo
from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)


class SupportTactics(BaseTactics):
    """Support tactics: heal party, maintain buffs, remove debuffs, assist DPS.

    Priority 20 — runs before DPS so support can heal/cleanse before others act.
    """

    name: ClassVar[str] = "support"
    priority: ClassVar[int] = 20
    description: ClassVar[str] = "Support/healer — buff maintenance, healing priority, debuff removal"
    role_type: ClassVar[str] = "support"

    # Healing thresholds
    HEAL_URGENT_THRESHOLD = 0.35
    HEAL_NORMAL_THRESHOLD = 0.60
    HEAL_TOPPING_THRESHOLD = 0.80

    # Buff skills
    BUFF_SKILLS = {
        "AL_BLESSING": 10,      # Blessing: STR/DEX/INT +10 at Lv10
        "AL_INCAGI": 10,        # Increase AGI: AGI +12 at Lv10
        "AL_ANGELUS": 5,       # Angelus: DEF +3 at Lv5
        "PR_GLORIA": 5,        # Gloria: LUK +30 at Lv5
        "PR_MAGNIFICAT": 5,    # Magnificat: SP regen
        "PR_IMPOSITIO": 5,     # Impositio Manus: ATK +25 at Lv5
        "PR_SUFFRAGIUM": 5,    # Suffragium: faster cast
        "PR_KYRIE": 10,        # Kyrie Eleison: damage absorb
        "PR_ASSUMPTIO": 5,     # Assumptio: DEF/MDEF +50%
    }

    # Debuff removal skills
    CLEANSE_SKILLS = {"PR_SANCTUARY", "PR_SLOWPOISON", "PR_BENEDICTIO"}

    def select_target(self, ctx: TacticsContext) -> TargetInfo | None:
        """Support target selection:
        1. If party needs healing → HeuristicAction (not a monster target).
        2. If safe, assist DPS on lowest HP monster.
        3. If undead, target with Heal (which damages undead).
        """
        # Priority 1: Check if any party member needs urgent healing
        if ctx.has_party:
            for pm in ctx.party_members:
                hp_pct = float(pm.get("hp_pct", pm.get("hp_ratio", 1.0)))
                if hp_pct < self.HEAL_URGENT_THRESHOLD:
                    # This should generate a heal action, not a target
                    # Return None to signal "no attack target needed"
                    return None

        monsters = ctx.monsters
        if not monsters:
            return None

        # Target undead (Heal nukes undead)
        undead = [m for m in monsters if str(m.get("element", "")).lower() == "undead"]
        if undead:
            info = self._monster_to_info(undead[0])
            info.score = 100.0 + (1.0 - info.hp_pct) * 10
            info.reason = "heal_nuke_undead"
            return info

        # Assist DPS: target lowest HP monster
        def scorer(t: TargetInfo, c: TacticsContext) -> float:
            score = 0.0
            # Low HP priority
            if t.hp_pct < 0.3:
                score += 60.0
            elif t.hp_pct < 0.5:
                score += 30.0
            # Darkness/undead — Holy Light bonus
            if t.element in ("undead", "dark"):
                score += 20.0
            # Aggressive
            if t.is_aggressive:
                score += 5.0
            return score

        return self._find_best_by_score(ctx, scorer)

    def select_skill(self, ctx: TacticsContext, target: TargetInfo | None) -> tuple[str, int] | None:
        """Support skill selection: heal > buff > damage undead > basic.

        Core mechanic: Heal deals 100%+ damage to undead at Lv1+.
        """
        available = set(s.upper() for s in ctx.available_skills)

        # ── Party healing priority ──
        if ctx.has_party:
            for pm in ctx.party_members:
                hp_pct = float(pm.get("hp_pct", pm.get("hp_ratio", 1.0)))
                if hp_pct < self.HEAL_URGENT_THRESHOLD and "AL_HEAL" in available:
                    return ("al_heal", 10)
                if hp_pct < self.HEAL_NORMAL_THRESHOLD and "AL_HEAL" in available:
                    return ("al_heal", 5)

        # ── Self-heal ──
        if ctx.my_hp_pct < self.HEAL_NORMAL_THRESHOLD and "AL_HEAL" in available:
            level = 10 if ctx.my_hp_pct < self.HEAL_URGENT_THRESHOLD else 5
            return ("al_heal", level)

        # ── Heal nuke undead ──
        if target and target.element == "undead" and "AL_HEAL" in available:
            return ("al_heal", 10)

        # ── Turn Undead (instant kill chance) ──
        if target and target.element == "undead" and "PR_TURNUNDEAD" in available:
            if ctx.cooldowns.get("pr_turnundead", 0) <= 0:
                return ("pr_turnundead", 10)

        # ── Holy Light damage ──
        if target and "AL_HOLYLIGHT" in available:
            return ("al_holylight", 5)

        return None

    def evaluate_positioning(self, ctx: TacticsContext, target: TargetInfo | None) -> dict[str, Any] | None:
        """Support positioning: stay near party, safe distance from threats."""
        if not ctx.has_party:
            return None

        # Stay at heal range (5-7 cells from party center)
        # If no target and party members are nearby, stay put
        return None

    def assess_buffs(self, ctx: TacticsContext) -> list[str]:
        """Keep party buffs active. Highest priority support function."""
        needed: list[str] = []
        available = set(s.upper() for s in ctx.available_skills)
        active = set(b.upper() for b in ctx.active_buffs)

        buff_priorities = ["AL_BLESSING", "AL_INCAGI", "PR_KYRIE",
                           "PR_ASSUMPTIO", "PR_GLORIA", "PR_MAGNIFICAT",
                           "PR_IMPOSITIO", "PR_SUFFRAGIUM"]

        for buff in buff_priorities:
            if buff in available:
                buff_name = buff.lower().replace("al_", "").replace("pr_", "")
                if buff_name.upper() not in active:
                    needed.append(buff)

        return needed

    def assess_emergency(self, ctx: TacticsContext) -> HeuristicAction | None:
        """Support emergency: mass heal, sanctuary, or Kyrie Eleison."""
        # Party emergency heal
        if ctx.has_party:
            low_hp_members = [
                pm for pm in ctx.party_members
                if float(pm.get("hp_pct", pm.get("hp_ratio", 1.0))) < self.HEAL_URGENT_THRESHOLD
            ]
            if low_hp_members and "AL_HEAL" in set(s.upper() for s in ctx.available_skills):
                return self._make_action(
                    command="use_skill al_heal",
                    reason="party_emergency_heal",
                    confidence=0.95,
                    target_party=low_hp_members[0].get("name", "ally"),
                    hp_pct=min(float(pm.get("hp_pct", 1.0)) for pm in low_hp_members),
                )

        # Self heal
        if ctx.my_hp_pct < 0.25:
            return self._make_action(
                command="use_skill al_heal",
                reason="support_emergency_self_heal",
                confidence=0.9,
                hp_pct=ctx.my_hp_pct,
            )

        return None

    def build_rotation(self, ctx: TacticsContext, target: TargetInfo | None) -> list[tuple[str, int]]:
        """Support rotation: emergency heal → buff upkeep → heal → damage.

        Prioritizes party survival over damage.
        """
        rotation: list[tuple[str, int]] = []
        available = set(s.upper() for s in ctx.available_skills)

        # Emergency self-heal
        if ctx.my_hp_pct < self.HEAL_URGENT_THRESHOLD and "AL_HEAL" in available:
            rotation.append(("al_heal", 10))

        # Party topping
        if ctx.has_party and "AL_HEAL" in available:
            rotation.append(("al_heal", 5))

        # Keep buffs active
        buffs = self.assess_buffs(ctx)
        for buff in buffs:
            # Find skill name
            for skill_id, _ in self.BUFF_SKILLS.items():
                if skill_id.lower().endswith(buff):
                    rotation.append((skill_id.lower(), 10))
                    break

        # Heal if HP not topped
        if ctx.my_hp_pct < self.HEAL_TOPPING_THRESHOLD and "AL_HEAL" in available:
            rotation.append(("al_heal", 5))

        # Damage undead if no healing needed
        if target and "AL_HOLYLIGHT" in available:
            rotation.append(("al_holylight", 5))

        return rotation
