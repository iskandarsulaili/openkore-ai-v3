"""Tactics package — combat behavior modules for each role."""
from ai_sidecar.domains.combat.tactics.base import BaseTactics, TacticsContext, TargetInfo
from ai_sidecar.domains.combat.tactics.tank import TankTactics
from ai_sidecar.domains.combat.tactics.melee_dps import MeleeDPSTactics
from ai_sidecar.domains.combat.tactics.ranged_dps import RangedDPSTactics
from ai_sidecar.domains.combat.tactics.magic_dps import MagicDPSTactics
from ai_sidecar.domains.combat.tactics.support import SupportTactics
from ai_sidecar.domains.combat.tactics.hybrid import HybridTactics
from ai_sidecar.domains.combat.tactics.kiting import KitingTactics

__all__ = [
    "BaseTactics", "TacticsContext", "TargetInfo",
    "TankTactics",
    "MeleeDPSTactics",
    "RangedDPSTactics",
    "MagicDPSTactics",
    "SupportTactics",
    "HybridTactics",
    "KitingTactics",
]
