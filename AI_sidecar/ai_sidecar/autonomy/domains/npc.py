"""NPC domain — NPC interaction, shop, storage lookup.

Extracted from heuristic_service.py lines 1101-1107 (_get_npc helper),
2484-2562 (DEATH state NPC interaction), 3635-3694 (town sell/buy NPC).
"""

from __future__ import annotations

import logging
from typing import Any

from ai_sidecar.autonomy.domains import BaseDomain
from ai_sidecar.autonomy.heuristic_service import HeuristicAction
from ai_sidecar.game_knowledge_db import GameKnowledgeDB

logger = logging.getLogger(__name__)


class NPCDomain(BaseDomain):
    name: str = "npc"
    priority: int = 40

    # Fallback NPC coordinates
    SELL_NPC_FALLBACK = "talknpc 147 175 c r1 n"
    PORTAL_FALLBACK = "move 22 203"
    POTION_NPC_FALLBACK = (126, 76)

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        service: Any,
    ) -> None:
        """NPC interaction decisions are handled inline by other domains.

        This domain provides lookup helpers used by economy and routing domains.
        """
        pass

    def get_sell_npc_command(self, map_name: str) -> str:
        """Get the talknpc command for selling items on a given map."""
        try:
            gkd = GameKnowledgeDB()
            npc = gkd.find_npc_for_task("sell", map_name)
            if npc:
                return (
                    f"talknpc {npc['x']} {npc['y']} "
                    f"{' '.join(eval(npc['steps']))}"
                )
        except Exception:
            pass
        return self.SELL_NPC_FALLBACK

    def get_portal_command(self, map_name: str) -> str:
        """Get the move command for portal back to hunting map."""
        try:
            gkd = GameKnowledgeDB()
            npc = gkd.find_npc_for_task("portal_to_hunt", map_name)
            if npc:
                return f"move {npc['x']} {npc['y']}"
        except Exception:
            pass
        return self.PORTAL_FALLBACK

    def get_potion_npc_coords(self, map_name: str) -> tuple[int, int]:
        """Get NPC coordinates for buying potions."""
        try:
            gkd = GameKnowledgeDB()
            npc = gkd.find_npc_for_task("buy_potion", map_name)
            if npc:
                return (npc['x'], npc['y'])
        except Exception:
            pass
        return self.POTION_NPC_FALLBACK


def create_domain() -> NPCDomain:
    return NPCDomain()
