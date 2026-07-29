"""NPC shop buy/sell operations with price negotiation awareness."""
from __future__ import annotations

import logging
import re
from typing import Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Default buy prices for common consumables (used when actual shop price unknown)
_DEFAULT_SHOP_PRICES: dict[str, int] = {
    "red potion": 50,
    "yellow potion": 500,
    "white potion": 1200,
    "green potion": 100,
    "blue potion": 2000,
    "fly wing": 300,
    "butterfly wing": 500,
    "holy water": 100,
    "empty bottle": 50,
    "iron": 400,
    "steel": 1500,
    "coal": 300,
    "poison bottle": 500,
}

# Items that should be sold automatically by vendor NPCs
_AUTO_SELL_TYPES: set[str] = {
    "jellopy", "sticky mucus", "feather", "shell", "clover",
    "scell", "memento", "gemstone", "rough ore", "emerald",
    "opal", "topaz", "amethyst", "sapphire", "garnet",
    "diamond", "zircon", "aquamarine",
}


@dataclass
class ShopState:
    """Tracks shop interaction state."""
    shop_open: bool = False
    current_tab: str = ""  # buy, sell, none
    items_for_sale: list[dict] = field(default_factory=list)
    selected_items: list[str] = field(default_factory=list)
    total_cost: int = 0
    offered_prices: dict[str, int] = field(default_factory=dict)


class NPCShop:
    """Handles buying from and selling to NPC shops."""

    def __init__(self, db: Any = None) -> None:
        self._shop_states: dict[str, ShopState] = {}
        self._gk_db = db
        if self._gk_db is None:
            from ai_sidecar.game_knowledge_db import GameKnowledgeDB
            self._gk_db = GameKnowledgeDB()

    def assess_shop_needs(
        self,
        signals: dict[str, Any],
        bot_id: str,
    ) -> list[dict]:
        """Assess whether the bot should visit a shop.

        Returns a list of action dicts (not yet HeuristicActions).
        Called from the parent domain's assess().
        """
        actions: list[dict] = []
        inventory = signals.get("inventory", []) or []
        zeny = int(signals.get("zeny", 0) or 0)
        hp_ratio = float(signals.get("hp_ratio", 1.0) or 1.0)
        sp_ratio = float(signals.get("sp_ratio", 1.0) or 1.0)
        map_name = str(signals.get("map", "") or "").lower().replace(".gat", "")

        # Check if we need potions (HP or SP low)
        needs_potions = hp_ratio < 0.4 or sp_ratio < 0.3
        # Check if we have junk to sell
        has_junk = any(
            any(junk in (item.get("name", "") or "").lower() for junk in _AUTO_SELL_TYPES)
            for item in inventory
        )

        if needs_potions and zeny > 5000 and map_name:
            actions.append({
                "type": "buy_potions",
                "priority": "high" if hp_ratio < 0.3 else "medium",
                "reason": f"HP={hp_ratio:.0%} SP={sp_ratio:.0%} - need potions",
                "map": map_name,
            })

        if has_junk:
            actions.append({
                "type": "sell_junk",
                "priority": "medium",
                "reason": "Inventory has sellable junk items",
                "map": map_name,
            })

        return actions

    def get_buy_command(
        self,
        item_name: str,
        quantity: int = 1,
        price: int | None = None,
    ) -> str:
        """Generate a buy command for a shop item.

        Args:
            item_name: Name or ID of the item
            quantity: How many to buy
            price: Expected price (for price validation)

        Returns:
            OpenKore buy command string
        """
        if price:
            return f"buy {item_name} {quantity} max_price {price}"
        return f"buy {item_name} {quantity}"

    def get_sell_command(self, item_name: str, quantity: int = 1) -> str:
        """Generate a sell command.

        Args:
            item_name: Name or ID of the item
            quantity: How many to sell

        Returns:
            OpenKore sell command string
        """
        return f"talk sell {item_name} {quantity}"

    def get_shop_list_command(self) -> str:
        """Generate command to list shop items."""
        return "talk shop"

    def get_buy_tab_command(self, tab_name: str = "") -> str:
        """Generate command to switch buy tab."""
        if tab_name:
            return f"talk buy {tab_name}"
        return "talk buy"

    def should_haggle(self, item_id: str, listed_price: int, zeny: int) -> bool:
        """Determine if the bot should try to negotiate price.

        Currently merchant-class bots with Discount skill get a small bonus.
        """
        return zeny > listed_price * 1.5  # Can afford with buffer

    def get_estimated_price(self, item_name: str) -> int:
        """Get the estimated shop price for an item."""
        return _DEFAULT_SHOP_PRICES.get(item_name.lower(), 1000)

    def build_shopping_list(
        self,
        signals: dict[str, Any],
        bot_id: str,
    ) -> list[dict]:
        """Build a shopping list of items to buy based on current needs.

        Returns list of {item, quantity, max_price} dicts.
        """
        shopping_list: list[dict] = []
        zeny = int(signals.get("zeny", 0) or 0)
        inventory = signals.get("inventory", []) or []
        hp_ratio = float(signals.get("hp_ratio", 1.0) or 1.0)
        sp_ratio = float(signals.get("sp_ratio", 1.0) or 1.0)
        job_name = str(signals.get("job_name", "novice") or "novice")
        base_level = int(signals.get("base_level", 1) or 1)

        # Count existing potions
        existing_hp_potions = 0
        existing_sp_potions = 0
        for item in inventory:
            name = (item.get("name", "") or "").lower()
            if "white potion" in name or "red potion" in name:
                existing_hp_potions += item.get("amount", 0) or 0
            if "blue potion" in name:
                existing_sp_potions += item.get("amount", 0) or 0

        # Buy HP potions if low
        if hp_ratio < 0.5 and existing_hp_potions < 10:
            if base_level < 40:
                potion = "red potion"
                price = 50
            else:
                potion = "white potion"
                price = 1200
            qty = min(30, zeny // price)
            if qty > 0:
                shopping_list.append({
                    "item": potion,
                    "quantity": qty,
                    "max_price": price,
                    "reason": f"HP={hp_ratio:.0%} restock potions",
                })

        # Buy SP potions if low
        if sp_ratio < 0.3 and existing_sp_potions < 5:
            qty = min(10, zeny // 2000)
            if qty > 0:
                shopping_list.append({
                    "item": "blue potion",
                    "quantity": qty,
                    "max_price": 2000,
                    "reason": f"SP={sp_ratio:.0%} restock SP potions",
                })

        # Archers need arrows
        if "archer" in job_name or "hunter" in job_name:
            has_arrows = any("arrow" in (item.get("name", "") or "").lower() for item in inventory)
            if not has_arrows:
                shopping_list.append({
                    "item": "arrow",
                    "quantity": 1000,
                    "max_price": 2,
                    "reason": "Archer class - restock arrows",
                })

        return shopping_list

    def cleanup_bot(self, bot_id: str) -> None:
        """Remove shop state for a bot."""
        self._shop_states.pop(bot_id, None)
