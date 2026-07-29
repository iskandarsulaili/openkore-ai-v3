"""
Item Worth Calculator — determines actual value of each item.

Calculates real zeny value considering:
  - NPC buy/sell prices
  - Player market prices (from vending data or estimates)
  - Quest turn-in value
  - Crafting utility value
  - Card rarity and usefulness
  - Whether item is equippable by current class
"""
from __future__ import annotations

import logging
from typing import Any

from ai_sidecar.domains.economy.database import (
    ItemValueDB,
    KEEP, SELL_NPC, SELL_PLAYER, SELL_ANY,
    DISCARD, CRAFTING, QUEST, POTION_FOOD, MATERIAL,
)

logger = logging.getLogger(__name__)


class ItemWorthCalculator:
    """Calculates the real value of items for decision-making.

    Provides three value axes:
      1. cash_value:  immediate zeny if sold (to NPC or player)
      2. utility_value: worth in terms of use (crafting, quest, equip)
      3. keep_value:  whether we should keep it regardless of price
    """

    def __init__(self, db: ItemValueDB | None = None) -> None:
        self._db = db or ItemValueDB()

    @property
    def db(self) -> ItemValueDB:
        return self._db

    # ── Core value calculation ────────────────────────────────────

    def calculate_worth(
        self,
        item_name: str,
        quantity: int = 1,
        zeny_on_hand: int = 0,
        inventory_weight_pct: float = 0.0,
        job_name: str = "novice",
    ) -> dict[str, Any]:
        """Full item worth assessment.

        Args:
            item_name: Name of the item.
            quantity: How many we have.
            zeny_on_hand: Current zeny (for affordability checks).
            inventory_weight_pct: Current weight percentage (0.0-1.0).
            job_name: Current class for equip checks.

        Returns:
            Dict with keys: cash_value, utility_value, keep_value,
            classification, reason, action (sell/discard/keep/buy).
        """
        data = self._db.get(item_name)
        if not data:
            return {
                "item": item_name,
                "cash_value": 0,
                "utility_value": 0,
                "total_value": 0,
                "classification": SELL_NPC,
                "keep": False,
                "action": "sell",
                "reason": "Unknown item — sell to NPC",
            }

        # Cash value: best price we can get right now
        npc_sell = int(data.get("npc_sell", 0))
        market_price = int(data.get("market_price", 0))
        cash_value = max(npc_sell, market_price)

        # Utility value: how much it's worth to us beyond cash
        classification = str(data.get("classification", SELL_NPC))
        utility_value = self._calc_utility_value(data, quantity, job_name)

        # Keep vs sell decision
        keep_reasons: list[str] = []
        sell_reasons: list[str] = []

        is_card = classification == KEEP or data.get("category") == "card"
        is_material = classification == MATERIAL
        is_crafting = classification == CRAFTING
        is_quest = classification == QUEST
        is_potion = classification == POTION_FOOD
        is_player_item = classification == SELL_PLAYER
        is_equip = data.get("category") in ("weapon", "armor", "accessory")

        if is_card:
            keep_reasons.append(f"Card — market_price={market_price}z")
        if is_material:
            keep_reasons.append(f"Upgrade material — {data.get('category', '')}")
        if is_crafting:
            keep_reasons.append(f"Crafting material — {data.get('category', '')}")
        if is_quest:
            keep_reasons.append("Quest item — needed for turn-ins")
        if is_potion and quantity <= self._db.get_keep_minimum(item_name):
            keep_reasons.append(f"Below keep_minimum ({data.get('keep_minimum', 0)})")
        if is_equip and self._is_player_equippable(data, job_name):
            keep_reasons.append("Equippable by current class")
            utility_value += max(npc_sell, market_price) * 2  # equip is 2x value

        # Decide action
        if keep_reasons:
            action = "keep"
            reason = "; ".join(keep_reasons)
            keep = True
        else:
            if is_potion and quantity > self._db.get_keep_minimum(item_name):
                excess = quantity - self._db.get_keep_minimum(item_name)
                action = "sell"
                reason = f"Sell excess ({excess}) above keep_minimum"
                keep = False
            elif is_player_item and market_price > npc_sell:
                action = "vendor"  # sell via vending for better price
                reason = f"Market ({market_price}z) > NPC ({npc_sell}z) — vendor it"
                keep = False
            elif classification == DISCARD:
                action = "discard"
                reason = "Junk — discard if weight critical"
                keep = False
            else:
                action = "sell"
                reason = f"NPC sell: {npc_sell}z — sell it"
                keep = False

        total_value = cash_value + utility_value

        return {
            "item": item_name,
            "item_id": data.get("id"),
            "quantity": quantity,
            "cash_value": cash_value,
            "utility_value": utility_value,
            "total_value": total_value,
            "classification": classification,
            "keep": keep,
            "action": action,
            "reason": reason,
            "market_price": market_price,
            "npc_sell_price": npc_sell,
        }

    # ── Internal helpers ──────────────────────────────────────────

    def _calc_utility_value(
        self,
        data: dict[str, Any],
        quantity: int,
        job_name: str,
    ) -> int:
        """Calculate utility (non-cash) value of an item.

        Factors:
          - Quest items: worth ~2x NPC sell (saves time)
          - Crafting materials: worth ~1.5x NPC sell (saves farm time)
          - Useful consumables: worth their NPC buy price
          - Equipment: worth 2x market if equippable
          - Cards: already reflected in market price
        """
        classification = data.get("classification", SELL_NPC)
        npc_sell = int(data.get("npc_sell", 0))
        npc_buy = int(data.get("npc_buy", 0))
        market_price = int(data.get("market_price", 0))

        if classification == QUEST:
            return max(npc_sell, market_price) * 2  # saves quest-running time

        if classification == CRAFTING:
            return max(npc_sell, market_price)  # saves farm time

        if classification == MATERIAL:
            return max(npc_sell, market_price)  # saves farm/buy time

        if classification == POTION_FOOD:
            # Potions are worth their buy price when we need them
            return max(0, npc_buy - npc_sell)  # convenience premium

        if data.get("category") == "card":
            # Cards have high utility for socketing
            return market_price * 2

        if data.get("category") in ("weapon", "armor", "accessory"):
            if self._is_player_equippable(data, job_name):
                return market_price * 2  # great find!
            return max(0, market_price - npc_sell)  # still worth more on market

        return 0

    def _is_player_equippable(
        self,
        data: dict[str, Any],
        job_name: str,
    ) -> bool:
        """Check if equipment is usable by the current class.

        Simple checks based on weapon category tags vs job.
        """
        tags = data.get("tags", [])
        category = data.get("category", "")

        if category == "weapon":
            # Basic class-weapon compatibility
            job_lower = job_name.lower()
            if "dagger" in tags and job_lower in ("thief", "assassin", "rogue", "novice"):
                return True
            if "sword" in tags and job_lower in ("swordman", "knight", "crusader", "novice"):
                return True
            if "bow" in tags and job_lower in ("archer", "hunter", "novice"):
                return True
            if "mace" in tags and job_lower in ("acolyte", "priest", "monk", "novice"):
                return True
            if "rod" in tags and job_lower in ("mage", "wizard", "sage", "novice"):
                return True
            # Generic weapon - anyone can use
            return True

        if category == "armor":
            return True  # anyone can wear most armors

        if category == "accessory":
            return True  # anyone can wear accessories

        return False

    # ── Batch assessment ──────────────────────────────────────────

    def assess_inventory(
        self,
        inventory: list[dict[str, Any]],
        zeny: int = 0,
        weight_pct: float = 0.0,
        job_name: str = "novice",
    ) -> list[dict[str, Any]]:
        """Assess entire inventory and return sorting recommendations.

        Returns items sorted by: keep=True first (priority order),
        then sell (by value descending).
        """
        results: list[dict[str, Any]] = []
        for item in inventory:
            name = str(item.get("name", item.get("item", "")))
            qty = int(item.get("amount", item.get("quantity", 1)))
            worth = self.calculate_worth(name, qty, zeny, weight_pct, job_name)
            results.append(worth)

        # Sort: keep first (by classification priority), then by value
        from ai_sidecar.domains.economy.database import CLASSIFICATION_PRIORITY

        def sort_key(w: dict[str, Any]) -> tuple:
            priority = CLASSIFICATION_PRIORITY.get(w["classification"], 99)
            return (0 if w["keep"] else 1, priority, -w["cash_value"])

        results.sort(key=sort_key)
        return results
