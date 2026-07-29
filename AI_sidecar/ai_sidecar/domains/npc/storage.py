"""Kafra storage deposit and withdraw operations."""
from __future__ import annotations

import logging
from typing import Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Maximum number of items OpenKore can store per concept type
_MAX_STORAGE_SLOTS = 600


@dataclass
class StorageState:
    """Tracks storage interaction state."""
    is_open: bool = False
    items_in_storage: list[dict] = field(default_factory=list)
    items_to_deposit: list[dict] = field(default_factory=list)
    items_to_withdraw: list[dict] = field(default_factory=list)
    last_action: str = ""


class NPCStorage:
    """Handles Kafra storage (deposit/withdraw) operations."""

    def __init__(self, db: Any = None) -> None:
        self._storage_states: dict[str, StorageState] = {}
        self._gk_db = db
        if self._gk_db is None:
            from ai_sidecar.game_knowledge_db import GameKnowledgeDB
            self._gk_db = GameKnowledgeDB()

    def assess_storage_needs(
        self,
        signals: dict[str, Any],
        bot_id: str,
    ) -> list[dict]:
        """Check if storage is needed (inventory full or valuable items).

        Returns list of action dicts.
        """
        actions: list[dict] = []
        inventory = signals.get("inventory", []) or []
        inventory_weight = float(signals.get("weight", 0) or 0)
        max_weight = float(signals.get("max_weight", 10000) or 10000)
        weight_ratio = inventory_weight / max_weight if max_weight > 0 else 0
        zeny = int(signals.get("zeny", 0) or 0)

        # Check if inventory is getting full
        is_near_cap = weight_ratio > 0.75
        is_overweight = weight_ratio > 0.9

        # Identify items worth storing vs selling
        storeable_items = self._identify_storeable_items(inventory)
        valuable_to_store = self._identify_valuable_items(inventory)

        # If near weight capacity and has storeable items, suggest storage visit
        if (is_near_cap or is_overweight) and (storeable_items or valuable_to_store):
            reason_note = "overweight" if is_overweight else "near weight cap"
            actions.append({
                "type": "visit_storage",
                "priority": "high" if is_overweight else "medium",
                "reason": f"Inventory {reason_note} ({weight_ratio:.0%}) - deposit items",
                "storeable": len(storeable_items) + len(valuable_to_store),
                "items": storeable_items + valuable_to_store,
            })

        # If we have a lot of zeny, there might be items to withdraw
        if zeny > 500000:
            actions.append({
                "type": "check_storage_for_use",
                "priority": "low",
                "reason": f"High zeny ({zeny:,}z) - check if storage has useful equipment",
            })

        return actions

    def should_visit_storage(
        self,
        bot_id: str,
        signals: dict[str, Any],
    ) -> bool:
        """Quick check if storage visit is warranted."""
        needs = self.assess_storage_needs(signals, bot_id)
        return any(a["type"] == "visit_storage" for a in needs)

    def get_storage_open_command(self) -> str:
        """Generate the command to open Kafra storage."""
        return "talk @kafra@ storage"

    def get_storage_deposit_command(self, item_name: str, quantity: int = 1) -> str:
        """Generate command to deposit an item into storage."""
        return f"storage add {item_name} {quantity}"

    def get_storage_withdraw_command(self, item_name: str, quantity: int = 1) -> str:
        """Generate command to withdraw an item from storage."""
        return f"storage get {item_name} {quantity}"

    def get_storage_list_command(self) -> str:
        """Generate command to list storage contents."""
        return "storage list"

    def get_storage_close_command(self) -> str:
        """Generate command to close storage."""
        return "storage close"

    def optimize_deposit_list(
        self,
        inventory: list[dict],
        priority_items: set[str] | None = None,
    ) -> list[dict]:
        """Build an optimized list of items to deposit.

        Args:
            inventory: Current inventory list
            priority_items: Set of item names to prioritize for storage

        Returns:
            List of {item, quantity, reason} dicts
        """
        if priority_items is None:
            priority_items = set()

        deposit_list: list[dict] = []
        storeable = self._identify_storeable_items(inventory)

        for entry in storeable:
            name = entry.get("name", "")
            amount = entry.get("amount", 0)
            if amount > 0:
                reason = "valuable" if name in priority_items else "overflow"
                deposit_list.append({
                    "item": name,
                    "quantity": amount,
                    "reason": reason,
                })

        return deposit_list

    def _identify_storeable_items(self, inventory: list[dict]) -> list[dict]:
        """Identify non-equipment junk that should be stored."""
        storeable: list[dict] = []
        for item in inventory:
            name = (item.get("name", "") or "").lower()
            item_type = (item.get("type", "") or "").lower()
            # Equipment is worth storing; consumables should be used
            if item_type == "weapon" or item_type == "armor":
                storeable.append(item)
            elif "card" in name:
                storeable.append(item)
        return storeable

    def _identify_valuable_items(self, inventory: list[dict]) -> list[dict]:
        """Identify high-value items worth protecting."""
        valuable: list[dict] = []
        for item in inventory:
            name = (item.get("name", "") or "").lower()
            price = int(item.get("price", 0) or 0)
            # Items worth > 10k zeny should be stored
            if price > 10000:
                valuable.append(item)
            # Cards are always valuable
            elif "card" in name:
                valuable.append(item)
        return valuable

    def get_storage_count(self, signals: dict[str, Any]) -> int:
        """Get number of items in storage from signals."""
        return len(signals.get("storage_items", []) or [])

    def cleanup_bot(self, bot_id: str) -> None:
        """Remove storage state for a bot."""
        self._storage_states.pop(bot_id, None)
