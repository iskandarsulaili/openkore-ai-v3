"""
Vending/Buying Store Automation — sets up buying stores, checks vending stores,
and executes trades automatically. The market arbitrage engine actually EXECUTES
trades, not just identifies them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class VendingItem:
    """An item listed in a vending store."""
    item_name: str
    item_id: int = 0
    quantity: int = 1
    price: int = 0
    seller_name: str = ""
    shop_name: str = ""
    position_x: int = 0
    position_y: int = 0
    map_name: str = ""
    timestamp: float = 0.0


@dataclass
class BuyOrder:
    """An item we want to buy."""
    item_name: str
    item_id: int = 0
    max_price: int = 0
    quantity_needed: int = 1
    quantity_bought: int = 0
    priority: int = 50
    is_active: bool = True
    reason: str = ""


@dataclass
class SellOrder:
    """An item we want to sell."""
    item_name: str
    item_id: int = 0
    min_price: int = 0
    quantity: int = 1
    quantity_sold: int = 0
    priority: int = 50
    is_active: bool = True
    reason: str = ""


class VendingAutomation:
    """Automates vending store operations — buying and selling."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._buy_orders: list[BuyOrder] = []
        self._sell_orders: list[SellOrder] = []
        self._vending_items: list[VendingItem] = []
        self._max_vending_items: int = 1000
        self._max_buy_orders: int = 50
        self._max_sell_orders: int = 50
        self._total_bought: int = 0
        self._total_sold: int = 0
        self._total_spent: int = 0
        self._total_earned: int = 0
        self._vending_enabled: bool = True
        self._buying_enabled: bool = True
        self._enqueue_fn: Callable | None = None
        self._load_default_orders()

    def _load_default_orders(self) -> None:
        """Load default buy/sell orders based on common market needs."""
        # Items to buy (undervalued)
        self._buy_orders = [
            BuyOrder("White Potion", max_price=400, quantity_needed=500, priority=80, reason="Farming consumable"),
            BuyOrder("Blue Potion", max_price=1500, quantity_needed=100, priority=70, reason="SP recovery"),
            BuyOrder("Stem", max_price=80, quantity_needed=200, priority=50, reason="Crafting material"),
            BuyOrder("Iron Ore", max_price=200, quantity_needed=200, priority=50, reason="Crafting material"),
            BuyOrder("Oridecon", max_price=35000, quantity_needed=20, priority=60, reason="Upgrade material"),
            BuyOrder("Elunium", max_price=20000, quantity_needed=20, priority=60, reason="Upgrade material"),
        ]

        # Items to sell (at a markup)
        self._sell_orders = [
            SellOrder("White Potion", min_price=600, quantity=500, priority=80, reason="Farming profit"),
            SellOrder("Blue Potion", min_price=2500, quantity=100, priority=70, reason="Trading profit"),
            SellOrder("Stem", min_price=150, quantity=200, priority=50, reason="Crafting profit"),
        ]

    # ── Public API ──

    def record_vending_item(self, item: VendingItem) -> None:
        """Record an item seen in a vending store."""
        with self._lock:
            self._vending_items.append(item)
            if len(self._vending_items) > self._max_vending_items:
                self._vending_items = self._vending_items[-self._max_vending_items:]

    def check_vending_stores(self, items: list[dict]) -> list[VendingItem]:
        """Check vending stores for items we want to buy."""
        with self._lock:
            matches: list[VendingItem] = []
            for item_data in items:
                name = str(item_data.get("name", ""))
                price = int(item_data.get("price", 0))
                qty = int(item_data.get("quantity", 1))
                seller = str(item_data.get("seller", ""))

                if not name or price <= 0:
                    continue

                vi = VendingItem(
                    item_name=name,
                    quantity=qty,
                    price=price,
                    seller_name=seller,
                    timestamp=time.time(),
                )
                self._vending_items.append(vi)

                # Check if this matches a buy order
                for order in self._buy_orders:
                    if order.is_active and order.item_name == name and price <= order.max_price:
                        matches.append(vi)
                        break

            if len(self._vending_items) > self._max_vending_items:
                self._vending_items = self._vending_items[-self._max_vending_items:]

            return matches

    def execute_buy(self, item: VendingItem) -> bool:
        """Execute a buy from a vending store."""
        with self._lock:
            if not self._buying_enabled or not self._enqueue_fn:
                return False

            # Find matching buy order
            for order in self._buy_orders:
                if order.is_active and order.item_name == item.item_name and item.price <= order.max_price:
                    # Execute buy via enqueue
                    cmd = f"buy {item.item_name} {item.quantity}"
                    self._enqueue_fn("self", cmd)
                    order.quantity_bought += item.quantity
                    self._total_bought += item.quantity
                    self._total_spent += item.price * item.quantity
                    logger.info("vending_buy: %s x%d @ %dz from %s", item.item_name, item.quantity, item.price, item.seller_name)
                    return True
            return False

    def execute_sell(self, item_name: str, quantity: int, price: int) -> bool:
        """Execute a sell via vending store."""
        with self._lock:
            if not self._vending_enabled or not self._enqueue_fn:
                return False

            # Find matching sell order
            for order in self._sell_orders:
                if order.is_active and order.item_name == item_name and price >= order.min_price:
                    # Execute sell via enqueue
                    cmd = f"vend {item_name} {quantity} {price}"
                    self._enqueue_fn("self", cmd)
                    order.quantity_sold += quantity
                    self._total_sold += quantity
                    self._total_earned += price * quantity
                    logger.info("vending_sell: %s x%d @ %dz", item_name, quantity, price)
                    return True
            return False

    def set_vending(self, items: list[tuple[str, int, int]]) -> bool:
        """Set up a vending store with items to sell. Each tuple: (item_name, quantity, price)."""
        with self._lock:
            if not self._vending_enabled or not self._enqueue_fn:
                return False
            for name, qty, price in items:
                cmd = f"vend {name} {qty} {price}"
                self._enqueue_fn("self", cmd)
            logger.info("vending_setup: %d items listed", len(items))
            return True

    def set_buying_store(self, items: list[tuple[str, int, int]]) -> bool:
        """Set up a buying store. Each tuple: (item_name, quantity, max_price)."""
        with self._lock:
            if not self._buying_enabled or not self._enqueue_fn:
                return False
            for name, qty, price in items:
                cmd = f"buy {name} {qty} {price}"
                self._enqueue_fn("self", cmd)
            logger.info("buying_store_setup: %d items listed", len(items))
            return True

    def add_buy_order(self, order: BuyOrder) -> None:
        with self._lock:
            self._buy_orders.append(order)
            if len(self._buy_orders) > self._max_buy_orders:
                self._buy_orders.pop(0)

    def add_sell_order(self, order: SellOrder) -> None:
        with self._lock:
            self._sell_orders.append(order)
            if len(self._sell_orders) > self._max_sell_orders:
                self._sell_orders.pop(0)

    def get_buy_orders(self, active_only: bool = True) -> list[BuyOrder]:
        with self._lock:
            if active_only:
                return [o for o in self._buy_orders if o.is_active]
            return list(self._buy_orders)

    def get_sell_orders(self, active_only: bool = True) -> list[SellOrder]:
        with self._lock:
            if active_only:
                return [o for o in self._sell_orders if o.is_active]
            return list(self._sell_orders)

    def get_best_buy_opportunity(self) -> VendingItem | None:
        """Get the best buy opportunity from scanned vending stores."""
        with self._lock:
            if not self._vending_items:
                return None
            for item in reversed(self._vending_items):
                for order in self._buy_orders:
                    if order.is_active and order.item_name == item.item_name and item.price <= order.max_price:
                        return item
            return None

    def get_vending_summary(self) -> str:
        with self._lock:
            lines = [f"── Vending Automation Summary ──"]
            lines.append(f"Vending enabled: {self._vending_enabled}")
            lines.append(f"Buying enabled: {self._buying_enabled}")
            lines.append(f"Total bought: {self._total_bought} items ({self._total_spent:,}z)")
            lines.append(f"Total sold: {self._total_sold} items ({self._total_earned:,}z)")
            lines.append(f"Net profit: {self._total_earned - self._total_spent:,}z")
            lines.append(f"Active buy orders: {len(self.get_buy_orders())}")
            lines.append(f"Active sell orders: {len(self.get_sell_orders())}")
            lines.append(f"Vending items scanned: {len(self._vending_items)}")
            best = self.get_best_buy_opportunity()
            if best:
                lines.append(f"Best buy: {best.item_name} @ {best.price}z from {best.seller_name}")
            return "\n".join(lines)

    def set_enqueue_fn(self, fn: Callable) -> None:
        with self._lock:
            self._enqueue_fn = fn

    def set_vending_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._vending_enabled = enabled

    def set_buying_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._buying_enabled = enabled

    def reset(self) -> None:
        with self._lock:
            self._vending_items.clear()
            self._total_bought = 0
            self._total_sold = 0
            self._total_spent = 0
            self._total_earned = 0


# ── Global Singleton ──

_vending: VendingAutomation | None = None
_vending_lock = RLock()


def get_vending_automation() -> VendingAutomation:
    global _vending
    with _vending_lock:
        if _vending is None:
            _vending = VendingAutomation()
        return _vending
