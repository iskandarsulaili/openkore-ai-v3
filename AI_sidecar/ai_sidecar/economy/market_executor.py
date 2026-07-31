"""
Automated Market Execution — handles complex dialog trees, out-of-stock,
dedicated merchant bot logic, buy/sell order management, and price negotiation.

A pro player doesn't just buy potions. They:
- Navigate complex NPC dialog trees (multiple menu levels)
- Handle out-of-stock gracefully (try next vendor)
- Run dedicated merchant bots for continuous vending
- Manage buy/sell orders with price negotiation
- Track market prices and adjust pricing dynamically
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
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
    # Advanced fields
    max_price: int = 0  # Maximum price we'll pay (for negotiation)
    min_price: int = 0  # Minimum price we'll accept (for negotiation)
    negotiation_rounds: int = 0
    vendor_preference: list[str] = field(default_factory=list)  # Preferred vendors
    fallback_items: list[str] = field(default_factory=list)  # Alternative items if out of stock


@dataclass
class VendingScan:
    """Result of scanning a vending store."""
    seller_name: str
    map_name: str
    x: int = 0
    y: int = 0
    items: list[dict] = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class DialogNode:
    """A node in an NPC dialog tree."""
    npc_name: str
    npc_map: str = ""
    npc_x: int = 0
    npc_y: int = 0
    dialog_sequence: list[str] = field(default_factory=list)  # Menu choices to navigate
    expected_responses: list[str] = field(default_factory=list)  # Expected NPC responses
    purpose: str = ""  # buy, sell, storage, quest, refine, identify


@dataclass
class MerchantBotConfig:
    """Configuration for a dedicated merchant bot."""
    bot_id: str = ""
    map_name: str = "pront_01"  # Prontera
    x: int = 150
    y: int = 150
    is_active: bool = False
    items_for_sale: list[dict] = field(default_factory=list)  # [{name, quantity, price}]
    items_to_buy: list[dict] = field(default_factory=list)  # [{name, quantity, max_price}]
    auto_restock: bool = True
    restock_threshold: float = 0.3  # Restock when inventory drops below 30%
    price_markup: float = 1.3  # 30% markup over buy price
    price_floor: float = 0.8  # Minimum price as fraction of market price
    last_restock_at: float = 0.0
    restock_interval: float = 1800.0  # 30 minutes


class MarketExecutor:
    """Executes market orders — buying and selling automatically.

    Features:
    - Handle out-of-stock items (try next vendor)
    - Handle complex dialog trees (multiple menu levels)
    - Dedicated merchant bot logic
    - Buy/sell order management
    - Price negotiation
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._orders: list[MarketOrder] = []
        self._completed_orders: list[MarketOrder] = []
        self._vending_scans: list[VendingScan] = []
        self._max_orders: int = 200
        self._max_scans: int = 1000
        self._total_spent: int = 0
        self._total_earned: int = 0
        self._trades_executed: int = 0
        self._enqueue_fn: Callable | None = None

        # Dialog tree knowledge base
        self._dialog_trees: dict[str, DialogNode] = {}  # npc_name -> DialogNode

        # Merchant bots
        self._merchant_bots: dict[str, MerchantBotConfig] = {}

        # Price history for negotiation
        self._price_history: dict[str, list[int]] = {}  # item_name -> [prices]

        # Out-of-stock tracking
        self._out_of_stock: dict[str, float] = {}  # item_name -> timestamp

        # Stats
        self._stats: dict[str, int] = {
            "orders_placed": 0,
            "orders_completed": 0,
            "trades_executed": 0,
            "out_of_stock_handled": 0,
            "dialog_navigations": 0,
            "price_negotiations": 0,
            "merchant_restocks": 0,
        }

        self._load_default_orders()

    def _load_default_orders(self) -> None:
        """Load default market orders."""
        now = time.time()
        self._orders = [
            MarketOrder("buy", "White Potion", 500, 400, 200000, 80, True, False, now, now + 86400, "Farming restock",
                        max_price=500, fallback_items=["Red Potion", "Orange Potion"]),
            MarketOrder("buy", "Blue Potion", 100, 1500, 150000, 70, True, False, now, now + 86400, "SP restock",
                        max_price=2000, fallback_items=["Yellow Potion"]),
            MarketOrder("buy", "Stem", 1000, 80, 80000, 50, True, False, now, now + 86400, "Crafting material",
                        max_price=120),
            MarketOrder("buy", "Iron Ore", 1000, 200, 200000, 50, True, False, now, now + 86400, "Crafting material",
                        max_price=300),
            MarketOrder("sell", "White Potion", 500, 600, 300000, 80, True, False, now, now + 86400, "Farming profit",
                        min_price=500),
            MarketOrder("sell", "Blue Potion", 100, 2500, 250000, 70, True, False, now, now + 86400, "Trading profit",
                        min_price=2000),
        ]

    # ── Dialog Tree Management ──

    def register_dialog_tree(self, node: DialogNode) -> None:
        """Register an NPC dialog tree for automated navigation."""
        with self._lock:
            self._dialog_trees[node.npc_name] = node
            logger.info("market_dialog_tree_registered: npc=%s purpose=%s steps=%d",
                        node.npc_name, node.purpose, len(node.dialog_sequence))

    def get_dialog_sequence(self, npc_name: str, purpose: str = "") -> list[str] | None:
        """Get the dialog sequence for an NPC."""
        with self._lock:
            node = self._dialog_trees.get(npc_name)
            if node and (not purpose or node.purpose == purpose):
                return node.dialog_sequence
            return None

    def navigate_dialog(self, npc_name: str, purpose: str) -> bool:
        """Navigate an NPC dialog tree. Returns True if successful."""
        with self._lock:
            sequence = self.get_dialog_sequence(npc_name, purpose)
            if not sequence:
                logger.warning("market_dialog_unknown: npc=%s purpose=%s", npc_name, purpose)
                return False

            if not self._enqueue_fn:
                return False

            # Queue dialog navigation commands
            for step in sequence:
                self._enqueue_fn("self", f"talknpc {npc_name}")
                self._enqueue_fn("self", f"menu {step}")

            self._stats["dialog_navigations"] += 1
            logger.info("market_dialog_navigated: npc=%s purpose=%s steps=%d",
                        npc_name, purpose, len(sequence))
            return True

    # ── Merchant Bot Management ──

    def register_merchant_bot(self, config: MerchantBotConfig) -> None:
        """Register a dedicated merchant bot."""
        with self._lock:
            self._merchant_bots[config.bot_id] = config
            logger.info("market_merchant_registered: bot=%s map=%s items=%d",
                        config.bot_id, config.map_name, len(config.items_for_sale))

    def get_merchant_bot(self, bot_id: str) -> MerchantBotConfig | None:
        with self._lock:
            return self._merchant_bots.get(bot_id)

    def setup_merchant_bot(self, bot_id: str) -> bool:
        """Set up a merchant bot for vending."""
        with self._lock:
            config = self._merchant_bots.get(bot_id)
            if not config or not self._enqueue_fn:
                return False

            # Move to vending location
            self._enqueue_fn(bot_id, f"move {config.map_name} {config.x} {config.y}")

            # Set up vending shop
            for item in config.items_for_sale:
                self._enqueue_fn(bot_id, f"vend {item['name']} {item['quantity']} {item['price']}")

            config.is_active = True
            logger.info("market_merchant_setup: bot=%s map=%s items=%d",
                        bot_id, config.map_name, len(config.items_for_sale))
            return True

    def restock_merchant_bot(self, bot_id: str) -> bool:
        """Restock a merchant bot's inventory."""
        with self._lock:
            config = self._merchant_bots.get(bot_id)
            if not config or not self._enqueue_fn:
                return False

            # Close shop, restock, reopen
            self._enqueue_fn(bot_id, "close_shop")
            self._enqueue_fn(bot_id, "ai auto")  # Go get items from storage
            self._enqueue_fn(bot_id, f"move {config.map_name} {config.x} {config.y}")

            for item in config.items_for_sale:
                self._enqueue_fn(bot_id, f"vend {item['name']} {item['quantity']} {item['price']}")

            config.last_restock_at = time.time()
            self._stats["merchant_restocks"] += 1
            logger.info("market_merchant_restocked: bot=%s", bot_id)
            return True

    def check_merchant_needs_restock(self, bot_id: str, inventory_pct: float) -> bool:
        """Check if a merchant bot needs restocking."""
        with self._lock:
            config = self._merchant_bots.get(bot_id)
            if not config or not config.auto_restock:
                return False
            if time.time() - config.last_restock_at < config.restock_interval:
                return False
            return inventory_pct < config.restock_threshold

    # ── Price Negotiation ──

    def negotiate_price(self, item_name: str, asking_price: int, order: MarketOrder) -> int | None:
        """Negotiate a price for an item.

        Returns the agreed price, or None if negotiation fails.
        """
        with self._lock:
            if order.order_type == "buy":
                # We're buying: want to pay less than asking
                if asking_price <= order.max_price:
                    return asking_price  # Accept immediately
                # Try to negotiate down
                counter = int(asking_price * 0.9)
                if counter >= order.max_price:
                    self._stats["price_negotiations"] += 1
                    return order.max_price  # Offer our max
                return None  # Too expensive
            else:
                # We're selling: want to get more than asking
                if asking_price >= order.min_price:
                    return asking_price  # Accept immediately
                # Try to negotiate up
                counter = int(asking_price * 1.1)
                if counter <= order.min_price:
                    self._stats["price_negotiations"] += 1
                    return order.min_price  # Accept our min
                return None  # Too cheap
            return None

    def record_price(self, item_name: str, price: int) -> None:
        """Record a price observation for market analysis."""
        with self._lock:
            if item_name not in self._price_history:
                self._price_history[item_name] = []
            self._price_history[item_name].append(price)
            # Keep last 100 prices
            if len(self._price_history[item_name]) > 100:
                self._price_history[item_name] = self._price_history[item_name][-100:]

    def get_average_price(self, item_name: str) -> int:
        """Get the average observed price for an item."""
        with self._lock:
            prices = self._price_history.get(item_name, [])
            if not prices:
                return 0
            return int(sum(prices) / len(prices))

    def get_market_price_trend(self, item_name: str) -> str:
        """Get the price trend for an item."""
        with self._lock:
            prices = self._price_history.get(item_name, [])
            if len(prices) < 5:
                return "unknown"
            recent = prices[-5:]
            if recent[-1] > recent[0] * 1.1:
                return "rising"
            elif recent[-1] < recent[0] * 0.9:
                return "falling"
            return "stable"

    # ── Out-of-Stock Handling ──

    def mark_out_of_stock(self, item_name: str) -> None:
        """Mark an item as out of stock from current vendor."""
        with self._lock:
            self._out_of_stock[item_name] = time.time()
            self._stats["out_of_stock_handled"] += 1
            logger.info("market_out_of_stock: item=%s", item_name)

    def find_alternative(self, item_name: str) -> str | None:
        """Find an alternative item when the desired one is out of stock."""
        with self._lock:
            for order in self._orders:
                if order.item_name == item_name and order.fallback_items:
                    for fallback in order.fallback_items:
                        # Check if fallback is also out of stock
                        if fallback not in self._out_of_stock or \
                           time.time() - self._out_of_stock.get(fallback, 0) > 3600:
                            return fallback
            return None

    def is_out_of_stock(self, item_name: str) -> bool:
        """Check if an item was recently marked out of stock."""
        with self._lock:
            ts = self._out_of_stock.get(item_name)
            if ts is None:
                return False
            return time.time() - ts < 3600  # Reset after 1 hour

    # ── Public API ──

    def add_order(self, order: MarketOrder) -> None:
        with self._lock:
            self._orders.append(order)
            self._stats["orders_placed"] += 1
            if len(self._orders) > self._max_orders:
                self._orders.pop(0)

    def scan_vending(self, scan: VendingScan) -> list[MarketOrder]:
        """Scan a vending store and return matching buy orders.

        Handles out-of-stock: if an item is out of stock, tries fallback items.
        """
        with self._lock:
            self._vending_scans.append(scan)
            if len(self._vending_scans) > self._max_scans:
                self._vending_scans = self._vending_scans[-self._max_scans:]

            matches: list[MarketOrder] = []
            for item in scan.items:
                name = str(item.get("name", ""))
                price = int(item.get("price", 0))
                qty = int(item.get("quantity", 1))

                # Skip if out of stock
                if self.is_out_of_stock(name):
                    alt = self.find_alternative(name)
                    if alt:
                        logger.info("market_out_of_stock_fallback: %s -> %s", name, alt)
                        name = alt
                    else:
                        continue

                for order in self._orders:
                    if order.is_active and not order.is_complete and order.order_type == "buy":
                        if order.item_name == name:
                            # Negotiate price
                            agreed_price = self.negotiate_price(name, price, order)
                            if agreed_price is None:
                                continue  # Price negotiation failed

                            buy_order = MarketOrder(
                                "buy", name, min(qty, order.quantity),
                                agreed_price, agreed_price * min(qty, order.quantity),
                                order.priority, True, False, time.time(), time.time() + 60,
                                f"Vending buy from {scan.seller_name}",
                                max_price=order.max_price,
                                fallback_items=order.fallback_items,
                            )
                            matches.append(buy_order)
                            self.record_price(name, agreed_price)
                            break

            return matches

    def execute_buy(self, item_name: str, quantity: int, unit_price: int, seller: str = "") -> bool:
        """Execute a buy order with out-of-stock handling."""
        with self._lock:
            if not self._enqueue_fn:
                return False

            # Check if item is out of stock
            if self.is_out_of_stock(item_name):
                alt = self.find_alternative(item_name)
                if alt:
                    logger.info("market_buy_fallback: %s -> %s", item_name, alt)
                    item_name = alt
                else:
                    logger.warning("market_buy_out_of_stock: no alternative for %s", item_name)
                    return False

            cmd = f"buy {item_name} {quantity}"
            self._enqueue_fn("self", cmd)
            self._total_spent += unit_price * quantity
            self._trades_executed += 1
            self._stats["trades_executed"] += 1
            self.record_price(item_name, unit_price)
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
            self._stats["trades_executed"] += 1
            self.record_price(item_name, unit_price)
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
            lines.append(f"Out-of-stock handled: {self._stats['out_of_stock_handled']}")
            lines.append(f"Dialog navigations: {self._stats['dialog_navigations']}")
            lines.append(f"Price negotiations: {self._stats['price_negotiations']}")
            lines.append(f"Merchant restocks: {self._stats['merchant_restocks']}")
            active = self.get_active_orders()
            if active:
                lines.append(f"Active orders: {len(active)}")
                for o in active[:5]:
                    lines.append(f"  {o.order_type} {o.item_name} x{o.quantity} @ {o.unit_price}z")
            # Merchant bots
            active_merchants = [c for c in self._merchant_bots.values() if c.is_active]
            if active_merchants:
                lines.append(f"Active merchant bots: {len(active_merchants)}")
                for mb in active_merchants[:3]:
                    lines.append(f"  {mb.bot_id} on {mb.map_name} ({len(mb.items_for_sale)} items)")
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
            self._dialog_trees.clear()
            self._merchant_bots.clear()
            self._price_history.clear()
            self._out_of_stock.clear()
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
