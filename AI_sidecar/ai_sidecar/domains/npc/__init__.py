"""NPC interaction domain."""
from __future__ import annotations

from ai_sidecar.domains.npc.dialogue import NPCDialogueEngine
from ai_sidecar.domains.npc.services import NPCService
from ai_sidecar.domains.npc.shop import NPCShop
from ai_sidecar.domains.npc.storage import NPCStorage

__all__ = [
    "NPCInteractionDomain",
    "NPCDialogueEngine",
    "NPCService",
    "NPCShop",
    "NPCStorage",
]


class NPCInteractionDomain:
    """Aggregate domain for all NPC interactions."""

    name = "npc"
    priority = 60

    def __init__(self) -> None:
        self.dialogue = NPCDialogueEngine()
        self.services = NPCService()
        self.shop = NPCShop()
        self.storage = NPCStorage()
