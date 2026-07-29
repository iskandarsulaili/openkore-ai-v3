"""
Smart Inventory Management — decides what to keep, what to sell, what to discard.

Policies:
  - Cards: NEVER sell to NPC
  - Valuable drops (Elunium, Oridecon, etc.): keep for player market or use
  - Quest items: keep (Jellopy, Fabric, etc.)
  - Crafting materials: keep for alchemy/forging
  - Everything else: sell to NPC when weight > 80%
  - Potions/food: always keep a minimum stock
  - Equipment: check if player-equippable, if yes keep
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ai_sidecar.domains.economy.database import (
    ItemValueDB,
    CLASSIFICATION_PRIORITY,
    KEEP, SELL_NPC, SELL_PLAYER, SELL_ANY,
    DISCARD, CRAFTING, QUEST, POTION_FOOD, MATERIAL,
)
from ai_sidecar.domains.economy.calculator import ItemWorthCalculator

logger = logging.getLogger(__name__)


@dataclass
class InventoryAction:
    """A recommended action for an inventory item."""
    item_name: str
    item_id: int
    quantity: int
    action: str  # "keep", "sell_npc", "vendor", "discard", "store", "move_to_cart"
    reason: str
    priority: int = 5  # 1=urgent, 10=low
    zeny_value: int = 0


@dataclass
class InventorySnapshot:
    """Current inventory state with recommendations."""
    items: list[dict[str, Any]]  # raw items
    weight_current: int
    weight_max: int
    weight_pct: float
    zeny: int
    actions: list[InventoryAction] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=lambda: {
        "keep": 0, "sell_npc": 0, "vendor": 0, "discard": 0, "total_value": 0,
    })


class InventoryManager:
    """Smart inventory manager with policy-based item disposition.

    Integrates with the item value DB and worth calculator to make
    intelligent keep/sell/discard decisions.
    """

    # Weight threshold to start selling aggressively
    SELL_THRESHOLD = 0.80  # 80% weight

    # Urgent sell threshold (in town, near capacity)
    URGENT_SELL_THRESHOLD = 0.95  # 95% weight

    # Discard threshold (weight critical, junk quality)
    DISCARD_WEIGHT_THRESHOLD = 0.98

    def __init__(
        self,
        db: ItemValueDB | None = None,
        calculator: ItemWorthCalculator | None = None,
    ) -> None:
        self._db = db or ItemValueDB()
        self._calc = calculator or ItemWorthCalculator(self._db)
        self._auto_sell_enabled = True

    # ── Core assessment ───────────────────────────────────────────

    def assess_inventory(
        self,
        signals: dict[str, Any],
        bot_id: str = "",
    ) -> InventorySnapshot:
        """Full inventory assessment with action recommendations.

        Args:
            signals: Bot state signals dict.
            bot_id: Bot identifier.

        Returns:
            InventorySnapshot with sorted action recommendations.
        """
        inventory = signals.get("inventory", []) or []
        zeny = int(signals.get("zeny", 0) or 0)
        weight_current = int(signals.get("weight", signals.get("weight_current", 0)) or 0)
        weight_max = int(signals.get("weight_max", 0) or 0)
        if weight_max == 0:
            weight_max = 2000  # default for novices
        weight_pct = weight_current / max(1, weight_max)
        job_name = str(signals.get("job_name", "novice") or "novice")

        snapshot = InventorySnapshot(
            items=inventory,
            weight_current=weight_current,
            weight_max=weight_max,
            weight_pct=weight_pct,
            zeny=zeny,
        )

        # Assess each item
        for item in inventory:
            name = str(item.get("name", item.get("item", "")) or "")
            quantity = int(item.get("amount", item.get("quantity", 1)) or 1)
            item_id = int(item.get("id", item.get("item_id", 0)) or 0)

            if not name:
                continue

            action = self._decide_item_action(
                name, quantity, weight_pct, zeny, job_name, item_id,
            )
            if action:
                snapshot.actions.append(action)
                snapshot.summary[action.action] = snapshot.summary.get(action.action, 0) + 1
                if action.action in ("sell_npc", "vendor"):
                    snapshot.summary["total_value"] += action.zeny_value

        # Sort actions by priority
        snapshot.actions.sort(key=lambda a: a.priority)

        return snapshot

    def _decide_item_action(
        self,
        name: str,
        quantity: int,
        weight_pct: float,
        zeny: int,
        job_name: str,
        item_id: int = 0,
    ) -> InventoryAction | None:
        """Decide what to do with a single inventory item.

        Decision logic (in priority order):
          1. Cards → NEVER sell to NPC
          2. Quest items → keep
          3. Crafting materials → keep
          4. Upgrade materials → keep or vendor
          5. Equipment → keep if equippable, else vendor
          6. Consumables → keep minimum stock, sell rest if heavy
          7. High-value player items → vendor
          8. Everything else → sell NPC if over weight threshold
          9. Junk → discard if weight critical
        """
        cls = self._db.get_classification(name)
        market_price = self._db.get_market_price(name)
        npc_sell = self._db.get_npc_sell_price(name)
        best_price = max(npc_sell, market_price)

        # 1. Cards — NEVER sell to NPC
        if cls == KEEP or self._db.is_card(name):
            return InventoryAction(
                item_name=name,
                item_id=item_id,
                quantity=quantity,
                action="keep",
                reason="Card or keep-classified — never sell to NPC",
                priority=1,
                zeny_value=best_price * quantity,
            )

        # 2. Quest items — keep
        if cls == QUEST:
            return InventoryAction(
                item_name=name,
                item_id=item_id,
                quantity=quantity,
                action="keep",
                reason=f"Quest item — keep for turn-ins ({best_price}z/value)",
                priority=2,
                zeny_value=best_price * quantity,
            )

        # 3. Crafting materials — keep
        if cls == CRAFTING:
            return InventoryAction(
                item_name=name,
                item_id=item_id,
                quantity=quantity,
                action="keep",
                reason=f"Crafting material — keep for alchemy/forging",
                priority=3,
                zeny_value=best_price * quantity,
            )

        # 4. Upgrade materials — keep (valuable)
        if cls == MATERIAL:
            return InventoryAction(
                item_name=name,
                item_id=item_id,
                quantity=quantity,
                action="keep",
                reason=f"Upgrade material — {name} worth {best_price}z each",
                priority=4,
                zeny_value=best_price * quantity,
            )

        # 5. Potions/consumables — keep minimum stock
        if cls == POTION_FOOD:
            keep_min = self._db.get_keep_minimum(name)
            if keep_min > 0 and quantity <= keep_min:
                return InventoryAction(
                    item_name=name,
                    item_id=item_id,
                    quantity=quantity,
                    action="keep",
                    reason=f"Below keep_minimum ({quantity}/{keep_min})",
                    priority=5,
                    zeny_value=npc_sell * quantity,
                )

        # Check for equipment that's player-equippable
        worth = self._calc.calculate_worth(name, quantity, zeny, weight_pct, job_name)
        if worth.get("action") == "keep":
            return InventoryAction(
                item_name=name,
                item_id=item_id,
                quantity=quantity,
                action="keep",
                reason=worth.get("reason", "Keep per calculator"),
                priority=3,
                zeny_value=worth.get("total_value", 0),
            )

        # ── Decisions when over weight threshold ──

        # 6. Potions: sell excess over keep_minimum when weight high
        if cls == POTION_FOOD and weight_pct >= self.SELL_THRESHOLD:
            keep_min = self._db.get_keep_minimum(name)
            excess = quantity - keep_min
            if excess > 0:
                return InventoryAction(
                    item_name=name,
                    item_id=item_id,
                    quantity=excess,
                    action="sell_npc",
                    reason=f"Excess potions ({excess}) — heavy ({weight_pct:.0%})",
                    priority=7,
                    zeny_value=npc_sell * excess,
                )
            return InventoryAction(
                item_name=name,
                item_id=item_id,
                quantity=quantity,
                action="keep",
                reason=f"Potions at minimum stock",
                priority=5,
                zeny_value=npc_sell * quantity,
            )

        # 7. High-value player items — vendor (if weight > 80% and in town)
        if cls in (SELL_PLAYER, SELL_ANY) and weight_pct >= self.SELL_THRESHOLD:
            if market_price > npc_sell * 1.5:  # 50%+ premium on player market
                return InventoryAction(
                    item_name=name,
                    item_id=item_id,
                    quantity=quantity,
                    action="vendor",
                    reason=f"Market ({market_price}z) > NPC ({npc_sell}z) — vendor it",
                    priority=8,
                    zeny_value=market_price * quantity,
                )

        # 8. Sell to NPC if heavy
        if weight_pct >= self.SELL_THRESHOLD:
            # If we're in town and heavy, sell it
            return InventoryAction(
                item_name=name,
                item_id=item_id,
                quantity=quantity,
                action="sell_npc",
                reason=f"Weight {weight_pct:.0%} — sell to NPC for {npc_sell}z each",
                priority=9,
                zeny_value=npc_sell * quantity,
            )

        # 9. Discard if weight critical and item is junk
        if cls == DISCARD and weight_pct >= self.DISCARD_WEIGHT_THRESHOLD:
            return InventoryAction(
                item_name=name,
                item_id=item_id,
                quantity=quantity,
                action="discard",
                reason=f"Junk — weight critical ({weight_pct:.0%})",
                priority=10,
                zeny_value=0,
            )

        # 10. Default: keep (we have space)
        return InventoryAction(
            item_name=name,
            item_id=item_id,
            quantity=quantity,
            action="keep",
            reason=f"Space available — keep ({best_price}z)",
            priority=6,
            zeny_value=best_price * quantity,
        )

    # ── Batch buy/sell commands ───────────────────────────────────

    def generate_sell_commands(self, snapshot: InventorySnapshot) -> list[str]:
        """Generate OpenKore sell commands for items flagged as sell_npc."""
        commands: list[str] = []
        for action in snapshot.actions:
            if action.action == "sell_npc" and action.quantity > 0:
                # Use the item name or ID
                item_ref = str(action.item_id) if action.item_id else action.item_name
                commands.append(f"talk sell {item_ref} {action.quantity}")
        return commands

    def generate_store_commands(self, snapshot: InventorySnapshot) -> list[str]:
        """Generate OpenKore storage commands for items worth storing."""
        commands: list[str] = []
        for action in snapshot.actions:
            if action.action in ("keep",) and action.zeny_value > 0:
                # Only store items above a value threshold
                if action.zeny_value >= 5000 and action.quantity > 1:
                    item_ref = str(action.item_id) if action.item_id else action.item_name
                    commands.append(f"talk store {item_ref} {action.quantity}")
        return commands

    # ── Quick assessment helpers ──────────────────────────────────

    def get_sell_value(self, inventory: list[dict[str, Any]]) -> int:
        """Get total NPC sell value of all sellable items."""
        total = 0
        for item in inventory:
            name = str(item.get("name", item.get("item", "")))
            qty = int(item.get("amount", item.get("quantity", 1)))
            cls = self._db.get_classification(name)
            if cls in (SELL_NPC, SELL_ANY, DISCARD):
                total += self._db.get_npc_sell_price(name) * qty
        return total

    def has_valuable_drops(self, inventory: list[dict[str, Any]]) -> bool:
        """Check if inventory has any valuable drops worth noting."""
        for item in inventory:
            name = str(item.get("name", item.get("item", "")))
            cls = self._db.get_classification(name)
            if cls in (MATERIAL, "keep"):
                return True
            market = self._db.get_market_price(name)
            if market >= 5000:
                return True
        return False

    def is_inventory_full(self, weight_pct: float) -> bool:
        """Check if inventory is near capacity."""
        return weight_pct >= self.SELL_THRESHOLD

    def toggle_auto_sell(self, enabled: bool | None = None) -> bool:
        """Toggle or check auto-sell status."""
        if enabled is not None:
            self._auto_sell_enabled = enabled
        return self._auto_sell_enabled
