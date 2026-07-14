"""
Automated Market Execution — sets up buying stores, scans vending stores,
executes trades automatically, and manages inventory across all bots.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class MarketOrder:
    """A buy or sell order to execute."""
    order_type: str  # buy, sell
    item_name: str
    quantity: int = 1
    unit_price: int = 0
    total_cost: int = 0
    priority: int = 50
    is_active: bool = True
    is_complete: bool = False
    created_at: float = 0.0
    expires_at: float = 0.0
    reason: str = ""


@dataclass
class VendingScan:
    """Result of scanning a vending store."""
    seller_name: str
    map_name: str
    x: int = 0
    y: int = 0
    items: list[dict] = field(default_factory=list)
    timestamp: float = 0.0


class MarketExecutor:
    """Executes market orders — buying and selling automatically."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._orders: list[MarketOrder] = []
        self._completed_orders: list[MarketOrder] = []
        self._vending_scans: list[VendingScan] = []
        self._max_orders: int = 100
        self._max_scans: int = 500
        self._total_spent: int = 0
        self._total_earned: int = 0
        self._trades_executed: int = 0
        self._enqueue_fn: Callable | None = None
        self._load_default_orders()

    def _load_default_orders(self) -> None:
        """Load default market orders."""
        now = time.time()
        self._orders = [
            MarketOrder("buy", "White Potion", 500, 400, 200000, 80, True, False, now, now + 86400, "Farming restock"),
            MarketOrder("buy", "Blue Potion", 100, 1500, 150000, 70, True, False, now, now + 86400, "SP restock"),
            MarketOrder("buy", "Stem", 1000, 80, 80000, 50, True, False, now, now + 86400, "Crafting material"),
            MarketOrder("buy", "Iron Ore", 1000, 200, 200000, 50, True, False, now, now + 86400, "Crafting material"),
            MarketOrder("sell", "White Potion", 500, 600, 300000, 80, True, False, now, now + 86400, "Farming profit"),
            MarketOrder("sell", "Blue Potion", 100, 2500, 250000, 70, True, False, now, now + 86400, "Trading profit"),
        ]

    # ── Public API ──

    def add_order(self, order: MarketOrder) -> None:
        with self._lock:
            self._orders.append(order)
            if len(self._orders) > self._max_orders:
                self._orders.pop(0)

    def scan_vending(self, scan: VendingScan) -> list[MarketOrder]:
        """Scan a vending store and return matching buy orders."""
        with self._lock:
            self._vending_scans.append(scan)
            if len(self._vending_scans) > self._max_scans:
                self._vending_scans = self._vending_scans[-self._max_scans:]

            matches: list[MarketOrder] = []
            for item in scan.items:
                name = str(item.get("name", ""))
                price = int(item.get("price", 0))
                qty = int(item.get("quantity", 1))

                for order in self._orders:
                    if order.is_active and not order.is_complete and order.order_type == "buy":
                        if order.item_name == name and price <= order.unit_price:
                            buy_order = MarketOrder(
                                "buy", name, min(qty, order.quantity),
                                price, price * min(qty, order.quantity),
                                order.priority, True, False, time.time(), time.time() + 60,
                                f"Vending buy from {scan.seller_name}"
                            )
                            matches.append(buy_order)
                            break
            return matches

    def execute_buy(self, item_name: str, quantity: int, unit_price: int, seller: str = "") -> bool:
        """Execute a buy order."""
        with self._lock:
            if not self._enqueue_fn:
                return False
            cmd = f"buy {item_name} {quantity}"
            self._enqueue_fn("self", cmd)
            self._total_spent += unit_price * quantity
            self._trades_executed += 1
            logger.info("market_buy: %s x%d @ %dz from %s", item_name, quantity, unit_price, seller)
            return True

    def execute_sell(self, item_name: str, quantity: int, unit_price: int) -> bool:
        """Execute a sell order."""
        with self._lock:
            if not self._enqueue_fn:
                return False
            cmd = f"vend {item_name} {quantity} {unit_price}"
            self._enqueue_fn("self", cmd)
            self._total_earned += unit_price * quantity
            self._trades_executed += 1
            logger.info("market_sell: %s x%d @ %dz", item_name, quantity, unit_price)
            return True

    def setup_buying_store(self, items: list[tuple[str, int, int]]) -> bool:
        """Set up a buying store. Each tuple: (item_name, quantity, max_price)."""
        with self._lock:
            if not self._enqueue_fn:
                return False
            for name, qty, price in items:
                self._enqueue_fn("self", f"buy {name} {qty} {price}")
            logger.info("buying_store_setup: %d items", len(items))
            return True

    def setup_selling_store(self, items: list[tuple[str, int, int]]) -> bool:
        """Set up a vending store. Each tuple: (item_name, quantity, price)."""
        with self._lock:
            if not self._enqueue_fn:
                return False
            for name, qty, price in items:
                self._enqueue_fn("self", f"vend {name} {qty} {price}")
            logger.info("selling_store_setup: %d items", len(items))
            return True

    def get_active_orders(self) -> list[MarketOrder]:
        with self._lock:
            now = time.time()
            return [o for o in self._orders if o.is_active and not o.is_complete and o.expires_at > now]

    def get_execution_summary(self) -> str:
        with self._lock:
            lines = [f"── Market Executor ──"]
            lines.append(f"Trades executed: {self._trades_executed}")
            lines.append(f"Total spent: {self._total_spent:,}z")
            lines.append(f"Total earned: {self._total_earned:,}z")
            lines.append(f"Net: {self._total_earned - self._total_spent:,}z")
            active = self.get_active_orders()
            if active:
                lines.append(f"Active orders: {len(active)}")
                for o in active[:5]:
                    lines.append(f"  {o.order_type} {o.item_name} x{o.quantity} @ {o.unit_price}z")
            return "\n".join(lines)

    def set_enqueue_fn(self, fn: Callable) -> None:
        with self._lock:
            self._enqueue_fn = fn

    def reset(self) -> None:
        with self._lock:
            self._orders.clear()
            self._completed_orders.clear()
            self._vending_scans.clear()
            self._total_spent = 0
            self._total_earned = 0
            self._trades_executed = 0
            self._load_default_orders()


# ── Global Singleton ──

_market_exec: MarketExecutor | None = None
_market_exec_lock = RLock()


def get_market_executor() -> MarketExecutor:
    global _market_exec
    with _market_exec_lock:
        if _market_exec is None:
            _market_exec = MarketExecutor()
        return _market_exec
