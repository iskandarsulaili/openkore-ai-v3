"""Social domain — party coordination, swarm intelligence."""
from __future__ import annotations
from typing import Any
import logging

from ai_sidecar.domains import BaseDomain
from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)


class SocialDomain(BaseDomain):
    """Social interactions: party, guild, chat, swarm coordination."""
    name = "social"
    priority = 40  # After combat (10), survival (20), economy (30)

    def __init__(self) -> None:
        super().__init__()
        self._swarm_coordinator = None
        self._party_cache: dict[str, bool] = {}

    def initialize(self) -> None:
        super().initialize()
        try:
            from ai_sidecar.domains.social.swarm import SwarmCoordinator
            self._swarm_coordinator = SwarmCoordinator(
                bot_names=["kicapmasin", "kicapmasin2", "kicapmasin3"],
                data_dir="data",
            )
            logger.info("SocialDomain: SwarmCoordinator initialized")
        except Exception as e:
            logger.warning(f"SocialDomain: SwarmCoordinator init failed: {e}")

    def assess(self, signals: dict[str, Any], actions: list[Any], bot_id: str) -> None:
        in_party = signals.get("in_party", False)
        
        # Party management
        if not in_party and bot_id not in self._party_cache:
            actions.append(HeuristicAction(
                kind="command", command="party create",
                confidence=0.8, reason="Create party",
                domain="social",
            ))
            self._party_cache[bot_id] = True
        
        # Swarm coordination
        if self._swarm_coordinator:
            try:
                swarm_actions = self._swarm_coordinator.generate_party_actions(bot_id, signals)
                actions.extend(swarm_actions)
            except Exception as e:
                logger.error(f"SocialDomain: swarm error: {e}")
