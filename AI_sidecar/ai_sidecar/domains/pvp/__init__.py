"""PvP / War of Emporium domain — arena tactics, WoE castle warfare, and threat assessment.

Provides:
  - PvPDomain: Domain integration for the heuristic assessment loop
  - ArenaTactics: Target prioritization, buff/debuff tracking, squishy hunting
  - WoETactics: Castle detection, emperium break, guild war defense/offense
"""
from __future__ import annotations

from typing import Any

from ai_sidecar.actions import HeuristicAction
from ai_sidecar.domains import BaseDomain
from ai_sidecar.domains.pvp.arenas import ArenaTactics, ThreatScore, ThreatProfile
from ai_sidecar.domains.pvp.woe import WoETactics, CastleState, WoERole
from ai_sidecar.domains.pvp.self_learning_hit_flee_analyzer import SelfLearningHitFleeAnalyzer
from ai_sidecar.domains.pvp.self_learning_status_resistance_tracker import SelfLearningStatusResistanceTracker
from ai_sidecar.domains.pvp.gtb_detector import GtbDetector
from ai_sidecar.domains.pvp.self_learning_class_counters import SelfLearningClassCounters
from ai_sidecar.domains.pvp.self_learning_elemental_armor_checker import SelfLearningElementalArmorChecker
from ai_sidecar.domains.pvp.self_learning_steal_analyzer import SelfLearningStealAnalyzer

logger = __import__("logging").getLogger(__name__)

__all__ = [
    "PvPDomain",
    "ArenaTactics",
    "ThreatScore",
    "ThreatProfile",
    "WoETactics",
    "CastleState",
    "WoERole",
    "SelfLearningHitFleeAnalyzer",
    "SelfLearningStatusResistanceTracker",
    "GtbDetector",
    "SelfLearningClassCounters",
    "SelfLearningElementalArmorChecker",
    "SelfLearningStealAnalyzer",
]


class PvPDomain(BaseDomain):
    """Domain for PvP / War of Emporium decision-making.

    Activates only on PvP maps (pvp_*, arena_*, gld_*, sch_*) and war-time
    castles. Delegates to ArenaTactics or WoETactics depending on context.
    """

    name: str = "pvp"
    priority: int = 15  # Higher priority than combat (20) — overrides normal hunting

    # Map name fragments that trigger PvP mode
    PVP_MAP_PREFIXES: tuple[str, ...] = (
        "pvp_", "arena_", "gld_", "sch_",
    )

    def __init__(self) -> None:
        super().__init__()
        self.arena = ArenaTactics()
        self.woe = WoETactics()

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        bot_id: str,
    ) -> None:
        """Route to the correct PvP sub-tactics for the current map."""
        map_name = str(signals.get("map", "") or "").lower().replace(".gat", "")
        job_name = str(signals.get("job_name", "novice") or "novice").lower()
        base_level = int(signals.get("base_level", 1) or 1)

        # Only activate on PvP-capable maps
        if not self._is_pvp_map(map_name):
            return

        # Detect WoE — castles use gld_* maps
        if map_name.startswith("gld_") and self.woe.is_war_time():
            self.woe.assess(signals, actions, bot_id)
        elif map_name.startswith("gld_") and not self.woe.is_war_time():
            # Castle but not war time — back off
            actions.append(HeuristicAction(
                kind="command",
                command="move prontera",
                confidence=0.85,
                domain="pvp",
                reason=f"WoE not active on {map_name} — retreating",
                metadata={"map": map_name},
            ))
        else:
            # Arena / PvP room tactics
            self.arena.assess(signals, actions, bot_id)

    # ------------------------------------------------------------------
    def _is_pvp_map(self, map_name: str) -> bool:
        """Check whether *map_name* is a PvP-capable map."""
        return map_name.startswith(self.PVP_MAP_PREFIXES)


# Convenience factory used by DomainRegistry
def create_domain() -> PvPDomain:
    return PvPDomain()
