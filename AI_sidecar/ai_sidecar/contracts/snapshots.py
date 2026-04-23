from __future__ import annotations

"""Snapshot contract aliases for backward/forward compatibility.

Primary definitions live in ``ai_sidecar.contracts.state``.
"""

from ai_sidecar.contracts.state import (  # re-export
    ActorDigest,
    BotStateSnapshot,
    CombatState,
    InventoryDigest,
    InventoryItemDigest,
    MarketDigest,
    MarketQuoteDigest,
    NpcRelationshipDigest,
    Position,
    ProgressionDigest,
    QuestDigest,
    QuestObjectiveDigest,
    SkillDigest,
    SnapshotIngestResponse,
    Vitals,
)

__all__ = [
    "ActorDigest",
    "BotStateSnapshot",
    "CombatState",
    "InventoryDigest",
    "InventoryItemDigest",
    "MarketDigest",
    "MarketQuoteDigest",
    "NpcRelationshipDigest",
    "Position",
    "ProgressionDigest",
    "QuestDigest",
    "QuestObjectiveDigest",
    "SkillDigest",
    "SnapshotIngestResponse",
    "Vitals",
]

