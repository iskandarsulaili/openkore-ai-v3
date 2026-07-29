"""NPC type identification and service lookup."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# NPC type classification keywords (matched against NPC names)
_NPC_TYPE_KEYWORDS: dict[str, list[str]] = {
    "merchant": [
        "tool", "dealer", "shop", "store", "mart", "vendor", "merchant",
        "item", "supply", "produce", "accessory",
    ],
    "weapon": [
        "weapon", "blade", "sword", "dagger", "bow", "smith", "blacksmith",
        "forge", "armor", "shield",
    ],
    "potion": [
        "potion", "apothecary", "pharmacy", "herb", "alchemist",
        "medicine", "drug",
    ],
    "storage": [
        "kafra", "storage", "bank", "warehouse", "vault",
    ],
    "repair": [
        "repair", "fix", "sharpen", "mend", "repairman",
    ],
    "healer": [
        "healer", "heal", "nun", "sister", "priest", "premium healer",
        "nurse", "doctor",
    ],
    "quest": [
        "quest", "mission", "task", "request", "notice", "board",
        "saga", "adventurer", "guild",
    ],
    "job": [
        "job", "class", "change", "master", "instructor", "trainer",
        "guildsman",
    ],
    "butcher": [
        "butcher", "meat", "food",
    ],
    "enchanter": [
        "enchanter", "enchant", "refine", "upgrade",
    ],
    "dye": [
        "dye", "hair", "stylish", "beauty", "cosmetic",
    ],
    "card": [
        "card", "token", "trader",
    ],
}


class NPCService:
    """Identify NPC type and provide service lookups."""

    def __init__(self, db: Any = None) -> None:
        self._gk_db = db
        if self._gk_db is None:
            from ai_sidecar.game_knowledge_db import GameKnowledgeDB
            self._gk_db = GameKnowledgeDB()

    def classify_npc(self, npc_name: str) -> str:
        """Classify what type of NPC (merchant, storage, quest, etc.) based on name.

        Returns:
            NPC type string, or 'unknown' if unclassifiable.
        """
        name_lower = npc_name.lower()
        scores: dict[str, int] = {}
        for npc_type, keywords in _NPC_TYPE_KEYWORDS.items():
            for kw in keywords:
                if kw in name_lower:
                    scores[npc_type] = scores.get(npc_type, 0) + 1

        if scores:
            return max(scores, key=scores.get)
        return "unknown"

    def find_npc_for_task(self, task_type: str, map_name: str = "prontera") -> dict | None:
        """Find an NPC that can fulfill a task on the given map.

        Args:
            task_type: Type of service needed (buy, sell, heal, storage, job_change, quest)
            map_name: Map to search on

        Returns:
            NPC interaction dict from GameKnowledgeDB, or None
        """
        return self._gk_db.find_npc_for_task(task_type, map_name)

    def get_merchant_map(self, map_name: str = "prontera") -> str:
        """Get the best map to find merchants (usually the town itself)."""
        return map_name if map_name else "prontera"

    def get_npcs_on_map(self, map_name: str) -> list[dict]:
        """Get all known NPC interactions on a given map from the DB."""
        return []  # Placeholder for future DB query

    def find_npc_by_name(self, npc_name: str, map_name: str | None = None) -> dict | None:
        """Find an NPC interaction by name (and optionally map) in the DB."""
        return self._gk_db.get_npc_interaction(npc_name, map_name)

    def service_to_interaction_type(self, service: str) -> str:
        """Convert a service name to the interaction_type used in GameKnowledgeDB."""
        mapping = {
            "merchant": "buy",
            "weapon": "buy",
            "potion": "buy",
            "storage": "storage",
            "repair": "repair",
            "healer": "heal",
            "quest": "quest",
            "job": "job_change",
            "card": "buy",
            "enchanter": "refine",
        }
        return mapping.get(service, service)

    def get_service_reason(self, npc_type: str) -> str:
        """Get a human-readable reason for visiting this NPC type."""
        reasons = {
            "merchant": "Buying supplies",
            "weapon": "Upgrading weapons/armor",
            "potion": "Restocking potions",
            "storage": "Accessing Kafra storage",
            "repair": "Repairing equipment durability",
            "healer": "Getting healed/blessed",
            "quest": "Quest interaction",
            "job": "Job advancement",
            "card": "Card trading",
            "enchanter": "Enchanting equipment",
            "dye": "Changing appearance",
        }
        return reasons.get(npc_type, f"Interacting with {npc_type} NPC")
