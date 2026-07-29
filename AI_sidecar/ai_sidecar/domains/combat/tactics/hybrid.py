"""Hybrid tactics — mixed role, adapts based on party composition and needs.

Used by: mixed-role builds, solo players without clear role, classes that can
flex (e.g., Monk can DPS or support, Sage can support or DPS).
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from ai_sidecar.domains.combat.tactics.base import BaseTactics, TacticsContext, TargetInfo
from ai_sidecar.domains.combat.tactics.tank import TankTactics
from ai_sidecar.domains.combat.tactics.melee_dps import MeleeDPSTactics
from ai_sidecar.domains.combat.tactics.ranged_dps import RangedDPSTactics
from ai_sidecar.domains.combat.tactics.magic_dps import MagicDPSTactics
from ai_sidecar.domains.combat.tactics.support import SupportTactics
from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)


class HybridTactics(BaseTactics):
    """Hybrid tactics: adapts role based on party composition and context.

    Priority 90 — runs last, after specialized roles have their chance.
    Delegates to sub-tactics based on detected role need.

    Key adaptation rules:
      - Solo: self-sufficient DPS with occasional self-heal.
      - In party w/o tank: act as off-tank (melee hybrid).
      - In party w/o support: act as support (buff/heal priority).
      - In full party: flex to whatever gap exists.
    """

    name: ClassVar[str] = "hybrid"
    priority: ClassVar[int] = 90
    description: ClassVar[str] = "Hybrid/adaptable — adjusts role based on party needs"
    role_type: ClassVar[str] = "hybrid"

    def __init__(self) -> None:
        super().__init__()
        self._tank = TankTactics()
        self._melee_dps = MeleeDPSTactics()
        self._ranged_dps = RangedDPSTactics()
        self._magic_dps = MagicDPSTactics()
        self._support = SupportTactics()

    def _detect_primary_role(self, ctx: TacticsContext) -> str:
        """Detect which role the hybrid should play right now.

        Returns one of: "tank", "melee_dps", "ranged_dps", "magic_dps", "support"
        """
        available = set(s.upper() for s in ctx.available_skills)
        has_heals = "AL_HEAL" in available
        has_buffs = "AL_BLESSING" in available or "AL_INCAGI" in available
        has_taunts = "SM_PROVOKE" in available
        has_aoe = "SM_MAGNUM" in available or "KN_BOWLINGBASH" in available
        has_ranged = "AC_DOUBLE" in available or "AC_SHOWER" in available
        has_magic = "MG_FIREBOLT" in available or "MG_COLD" in available
        has_melee_burst = "AS_SONICBLOW" in available or "TF_DOUBLE" in available

        party_has_tank = any(
            pm.get("role", "") == "tank" or "knight" in str(pm.get("class", "")).lower()
            for pm in ctx.party_members
        )
        party_has_support = any(
            pm.get("role", "") == "support" or "priest" in str(pm.get("class", "")).lower()
            for pm in ctx.party_members
        )

        # Solo: self-sufficient
        if not ctx.has_party:
            if has_magic and ctx.my_sp_pct > 0.5:
                return "magic_dps"
            if has_ranged:
                return "ranged_dps"
            if has_melee_burst:
                return "melee_dps"
            if has_heals or has_buffs:
                return "support"
            return "melee_dps"

        # In party without tank
        if ctx.has_party and not party_has_tank and has_taunts:
            return "tank"

        # In party without support
        if ctx.has_party and not party_has_support and (has_heals or has_buffs):
            return "support"

        # Full party — flex based on skills
        if has_magic:
            return "magic_dps"
        if has_ranged:
            return "ranged_dps"
        if has_melee_burst:
            return "melee_dps"

        return "melee_dps"

    def _get_delegate(self, role: str) -> BaseTactics:
        mapping = {
            "tank": self._tank,
            "melee_dps": self._melee_dps,
            "ranged_dps": self._ranged_dps,
            "magic_dps": self._magic_dps,
            "support": self._support,
        }
        return mapping.get(role, self._melee_dps)

    def select_target(self, ctx: TacticsContext) -> TargetInfo | None:
        role = self._detect_primary_role(ctx)
        delegate = self._get_delegate(role)
        logger.debug("hybrid_tactics: active_role=%s delegate=%s", role, delegate.name)
        result = delegate.select_target(ctx)
        if result:
            result.reason = f"hybrid({role})_{result.reason}"
        return result

    def select_skill(self, ctx: TacticsContext, target: TargetInfo | None) -> tuple[str, int] | None:
        role = self._detect_primary_role(ctx)
        delegate = self._get_delegate(role)
        return delegate.select_skill(ctx, target)

    def evaluate_positioning(self, ctx: TacticsContext, target: TargetInfo | None) -> dict[str, Any] | None:
        role = self._detect_primary_role(ctx)
        delegate = self._get_delegate(role)
        result = delegate.evaluate_positioning(ctx, target)
        if result:
            result["reason"] = f"hybrid({role})_{result.get('reason', 'move')}"
        return result

    def assess_buffs(self, ctx: TacticsContext) -> list[str]:
        role = self._detect_primary_role(ctx)
        delegate = self._get_delegate(role)
        return delegate.assess_buffs(ctx)

    def assess_emergency(self, ctx: TacticsContext) -> HeuristicAction | None:
        role = self._detect_primary_role(ctx)
        delegate = self._get_delegate(role)
        action = delegate.assess_emergency(ctx)
        if action:
            action.reason = f"hybrid({role})_{action.reason}"
        return action or self._fallback_emergency(ctx)

    def _fallback_emergency(self, ctx: TacticsContext) -> HeuristicAction | None:
        """Universal fallback: potion if HP is low."""
        if ctx.my_hp_pct < 0.3:
            return self._make_action(
                command="use_potion_or_heal",
                reason="hybrid_fallback_hp_low",
                confidence=0.7,
                hp_pct=ctx.my_hp_pct,
            )
        return None

    def build_rotation(self, ctx: TacticsContext, target: TargetInfo | None) -> list[tuple[str, int]]:
        role = self._detect_primary_role(ctx)
        delegate = self._get_delegate(role)
        return delegate.build_rotation(ctx, target)
