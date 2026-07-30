"""Vending Arbitrage — buy low, sell high with offline vending mules.

Pro players don't just farm and sell to NPC. They:
  1. Park mule characters at high-traffic market spots (Prontera, Alberta)
  2. Load them with items to vend at player prices
  3. Disconnect (mule stays vending via RO's offline shop system)
  4. Track sales, restock, and rotate inventory

This module manages:
  - Mule character registration and inventory tracking
  - Market spot selection (high traffic areas)
  - Pricing strategy (undercut, match market, premium during WoE)
  - Sales tracking and restock triggers
  - Buy low / sell high arbitrage detection

Thread-safe.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────

# Market spots ranked by traffic (best first)
HIGH_TRAFFIC_MARKETS: list[str] = [
    "prontera",       # Most active market hub on most servers
    "alberta",        # Second most active (port city, crafting hub)
    "morocc",         # Third (desert town, card market)
    "geffen",         # Magic item hub
    "payon",          # Archer/Thief item hub
    "izlude",         # Swordsman item hub
    "aldebaran",      # Potion/alchemy hub
]

# Vending pricing strategies
PRICING_UNDERCUT_PCT = 0.95     # 5% below current lowest
PRICING_MATCH_PCT = 1.0         # Match current lowest
PRICING_PREMIUM_PCT = 1.3       # 30% above during high demand

# Mule management
MAX_MULE_STOCK_ITEMS = 50       # Maximum different items per mule
RESTOCK_THRESHOLD = 10          # Restock when inventory drops below this
PROFIT_MIN_ZENY = 1000          # Minimum profit per item to bother
SALES_CHECK_INTERVAL = 3600     # Check sales every hour

# Restock temperature zones (how urgently restocking is needed)
class RestockUrgency(str, Enum):
    NONE = "none"           # Stock levels fine
    LOW = "low"             # Some items running low
    MEDIUM = "medium"       # Several items need restock
    HIGH = "high"           # Critical — multiple items depleted


# ── Data Models ────────────────────────────────────────────────────────────

@dataclass
class VendingMule:
    """A mule character parked at a market spot for offline vending."""
    name: str
    map_name: str
    x: int = 150               # Typical Prontera market coordinates
    y: int = 150
    is_active: bool = False     # Whether the mule is currently vending
    last_login: float = 0.0
    last_logout: float = 0.0
    total_sales: int = 0        # Total zeny earned
    total_transactions: int = 0 # Number of sales
    inventory: dict[str, int] = field(default_factory=dict)  # item_name -> quantity
    listed_items: dict[str, VendingListing] = field(default_factory=dict)


@dataclass
class VendingListing:
    """An item listed on a vending stall."""
    item_name: str
    quantity: int
    price_per_unit: int
    total_value: int  # quantity * price
    original_cost: int  # What we paid for it (buy price + fees)
    profit_per_unit: int  # selling price - cost
    listed_at: float
    sold: bool = False
    sold_at: float | None = None
    sold_quantity: int = 0


@dataclass
class MarketSpot:
    """A market spot with traffic data."""
    map_name: str
    traffic_rating: int   # 1-10
    typical_listings: int # Average number of vendors
    best_x: int
    best_y: int
    best_z: int = 0
    notes: str = ""


@dataclass
class ArbitrageOpportunity:
    """A buy-low / sell-high opportunity."""
    item_name: str
    buy_price: int           # Lowest price we can buy at
    buy_map: str             # Where to buy (market with lowest price)
    sell_price: int          # Highest price we can sell at
    sell_map: str            # Where to sell (market with highest price)
    profit_per_unit: int     # sell_price - buy_price
    profit_margin: float     # profit/buy as ratio
    confidence: float        # 0-1 how reliable this spread is
    max_quantity: int        # How many we can realistically flip
    estimated_total_profit: int  # profit * max_quantity


# Default market spots with known coordinates
MARKET_SPOTS: dict[str, MarketSpot] = {
    "prontera": MarketSpot(
        map_name="prontera", traffic_rating=10,
        typical_listings=50, best_x=150, best_y=150,
        notes="Main hub — best for general items, cards, potions",
    ),
    "alberta": MarketSpot(
        map_name="alberta", traffic_rating=8,
        typical_listings=30, best_x=130, best_y=130,
        notes="Port hub — good for marine-related items, seafood",
    ),
    "morocc": MarketSpot(
        map_name="morocc", traffic_rating=7,
        typical_listings=25, best_x=150, best_y=100,
        notes="Desert hub — card market, sand/mana items",
    ),
    "geffen": MarketSpot(
        map_name="geffen", traffic_rating=6,
        typical_listings=20, best_x=100, best_y=120,
        notes="Magic hub — wands, staffs, gemstones, converters",
    ),
}


# ── Vending Arbitrage Engine ──────────────────────────────────────────────

class VendingArbitrageEngine:
    """Manages offline vending mules and buy-low/sell-high arbitrage.

    Handles:
      - Mule creation and parking at market spots
      - Inventory loading and pricing strategy
      - Sales tracking and restock management
      - Cross-map price arbitrage detection
      - Profit/Loss tracking per mule and per item

    Thread-safe.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._mules: dict[str, VendingMule] = {}
        self._market_spots: dict[str, MarketSpot] = dict(MARKET_SPOTS)
        self._price_history: dict[str, list[dict[str, Any]]] = {}
        self._current_market_prices: dict[str, int] = {}
        self._last_sales_check: float = 0.0
        self._stats: dict[str, int | float] = {
            "mules_registered": 0,
            "active_mules": 0,
            "total_sales": 0,
            "total_profit": 0,
            "arbitrage_opportunities_found": 0,
            "restocks_triggered": 0,
        }

    # ── Mule Management ────────────────────────────────────────────────

    def register_mule(self, name: str, map_name: str = "prontera",
                       x: int = 150, y: int = 150) -> VendingMule:
        """Register a new vending mule.

        Args:
            name: Character name of the mule.
            map_name: Map to park the mule on.
            x, y: Coordinates to stand at.

        Returns:
            The newly registered VendingMule.
        """
        with self._lock:
            if name in self._mules:
                logger.warning("vending_arbitrage: mule '%s' already registered", name)
                return self._mules[name]

            spot = self._market_spots.get(map_name)
            if spot:
                x, y = spot.best_x, spot.best_y

            mule = VendingMule(
                name=name,
                map_name=map_name,
                x=x, y=y,
            )
            self._mules[name] = mule
            self._stats["mules_registered"] += 1  # type: ignore[assignment]
            logger.info(
                "vending_arbitrage: registered mule '%s' at %s (%d,%d)",
                name, map_name, x, y,
            )
            return mule

    def get_mule(self, name: str) -> VendingMule | None:
        """Get a registered mule by name."""
        with self._lock:
            return self._mules.get(name)

    def list_mules(self) -> list[VendingMule]:
        """List all registered mules."""
        with self._lock:
            return list(self._mules.values())

    def activate_mule(self, name: str) -> bool:
        """Mark a mule as actively vending.

        Returns True if the mule exists and was activated.
        """
        with self._lock:
            mule = self._mules.get(name)
            if mule is None:
                return False
            mule.is_active = True
            mule.last_login = time.time()
            self._stats["active_mules"] += 1  # type: ignore[assignment]
            logger.info("vending_arbitrage: mule '%s' activated for vending", name)
            return True

    def deactivate_mule(self, name: str) -> bool:
        """Mark a mule as no longer vending."""
        with self._lock:
            mule = self._mules.get(name)
            if mule is None:
                return False
            mule.is_active = False
            mule.last_logout = time.time()
            self._stats["active_mules"] = max(0, int(self._stats["active_mules"]) - 1)  # type: ignore[assignment]
            logger.info("vending_arbitrage: mule '%s' deactivated", name)
            return True

    def load_inventory(self, mule_name: str, items: dict[str, int]) -> None:
        """Load items onto a mule's inventory for vending.

        Args:
            mule_name: Name of the mule.
            items: Dict of item_name -> quantity.
        """
        with self._lock:
            mule = self._mules.get(mule_name)
            if mule is None:
                logger.warning("vending_arbitrage: cannot load inventory, mule '%s' unknown", mule_name)
                return

            for item_name, qty in items.items():
                mule.inventory[item_name] = mule.inventory.get(item_name, 0) + qty

            logger.info(
                "vending_arbitrage: loaded %d items onto mule '%s'",
                sum(items.values()), mule_name,
            )

    def list_item_for_sale(self, mule_name: str, item_name: str,
                            quantity: int, price_per_unit: int,
                            cost_per_unit: int = 0) -> bool:
        """List an item on a mule's vending stall.

        Args:
            mule_name: Name of the vending mule.
            item_name: Item name to list.
            quantity: How many to sell.
            price_per_unit: Selling price per unit.
            cost_per_unit: What we paid per unit (for P&L tracking).

        Returns:
            True if item was listed successfully.
        """
        with self._lock:
            mule = self._mules.get(mule_name)
            if mule is None:
                return False

            if len(mule.listed_items) >= MAX_MULE_STOCK_ITEMS:
                logger.warning(
                    "vending_arbitrage: mule '%s' at max %d listed items",
                    mule_name, MAX_MULE_STOCK_ITEMS,
                )
                return False

            # Check inventory
            available = mule.inventory.get(item_name, 0)
            if available < quantity:
                logger.warning(
                    "vending_arbitrage: mule '%s' only has %d of %s, need %d",
                    mule_name, available, item_name, quantity,
                )
                return False

            # Deduct from inventory
            mule.inventory[item_name] = available - quantity
            if mule.inventory[item_name] <= 0:
                del mule.inventory[item_name]

            # Create listing
            profit_per = price_per_unit - cost_per_unit
            listing = VendingListing(
                item_name=item_name,
                quantity=quantity,
                price_per_unit=price_per_unit,
                total_value=quantity * price_per_unit,
                original_cost=quantity * cost_per_unit,
                profit_per_unit=profit_per,
                listed_at=time.time(),
            )
            mule.listed_items[item_name] = listing
            return True

    def record_sale(self, mule_name: str, item_name: str,
                     quantity_sold: int = 1) -> bool:
        """Record that an item was sold from a mule's stall.

        Args:
            mule_name: Name of the vending mule.
            item_name: Name of the item sold.
            quantity_sold: How many units were sold.

        Returns:
            True if recorded successfully.
        """
        with self._lock:
            mule = self._mules.get(mule_name)
            if mule is None:
                return False

            listing = mule.listed_items.get(item_name)
            if listing is None:
                return False

            actual_sold = min(quantity_sold, listing.quantity)
            revenue = actual_sold * listing.price_per_unit
            profit = actual_sold * listing.profit_per_unit

            mule.total_sales += revenue
            mule.total_transactions += 1
            listing.sold_quantity += actual_sold
            listing.quantity -= actual_sold

            if listing.quantity <= 0:
                listing.sold = True
                listing.sold_at = time.time()
                del mule.listed_items[item_name]

            self._stats["total_sales"] += revenue  # type: ignore[assignment]
            self._stats["total_profit"] += profit  # type: ignore[assignment]

            logger.info(
                "vending_arbitrage: sold %dx %s from mule '%s' for %dz (+%dz profit)",
                actual_sold, item_name, mule_name, revenue, profit,
            )
            return True

    def check_restock_needed(self, mule_name: str) -> RestockUrgency:
        """Check if a mule needs restocking.

        Returns the urgency level based on remaining stock.
        """
        with self._lock:
            mule = self._mules.get(mule_name)
            if mule is None or not mule.is_active:
                return RestockUrgency.NONE

            total_listed = sum(l.quantity for l in mule.listed_items.values())
            total_inventory = sum(mule.inventory.values())

            if total_listed <= 5 and total_inventory <= 10:
                return RestockUrgency.HIGH
            if total_listed <= 10 and total_inventory <= 20:
                return RestockUrgency.MEDIUM
            if total_listed <= 20:
                return RestockUrgency.LOW
            return RestockUrgency.NONE

    def get_restock_list(self, mule_name: str) -> list[dict[str, Any]]:
        """Get a list of items that need restocking for a mule.

        Returns items that sold out or are running low, sorted by
        profit per unit descending.
        """
        with self._lock:
            mule = self._mules.get(mule_name)
            if mule is None:
                return []

            restock: list[dict[str, Any]] = []
            for item_name, listing in mule.listed_items.items():
                if listing.quantity < RESTOCK_THRESHOLD and listing.profit_per_unit > PROFIT_MIN_ZENY:
                    restock.append({
                        "item": item_name,
                        "remaining": listing.quantity,
                        "needed": RESTOCK_THRESHOLD - listing.quantity,
                        "profit_per_unit": listing.profit_per_unit,
                        "suggested_price": listing.price_per_unit,
                    })

            restock.sort(key=lambda r: -r["profit_per_unit"])
            return restock

    # ── Pricing Strategy ───────────────────────────────────────────────

    def suggest_price(self, item_name: str, base_market_price: int,
                       is_woe_period: bool = False,
                       competition_level: str = "medium") -> int:
        """Suggest a competitive selling price for an item.

        Strategy:
          - During WoE: premium pricing (+30% if demand is high)
          - High competition: undercut by 5%
          - Low competition: match market or slight premium
          - Default: match market price

        Args:
            item_name: Name of the item.
            base_market_price: Current market price reference.
            is_woe_period: Whether WoE is currently active.
            competition_level: "low", "medium", "high".

        Returns:
            Suggested selling price in zeny.
        """
        with self._lock:
            price = base_market_price

            if is_woe_period:
                # Premium pricing during WoE
                price = int(price * PRICING_PREMIUM_PCT)
            elif competition_level == "high":
                # Undercut to move inventory
                price = int(price * PRICING_UNDERCUT_PCT)
            elif competition_level == "low":
                # Keep at market (no need to discount)
                price = int(price * PRICING_MATCH_PCT)
            else:
                # Slight undercut for medium competition
                price = int(price * 0.97)

            return max(price, 1)

    # ── Market Spot Management ─────────────────────────────────────────

    def get_best_spot(self, item_category: str = "") -> MarketSpot:
        """Get the best market spot for a given item category.

        Args:
            item_category: Optional category to target specific market.

        Returns:
            The best MarketSpot for the category.
        """
        with self._lock:
            if not item_category:
                return list(self._market_spots.values())[0]  # Prontera (highest traffic)

            # Category-specific recommendations
            cat_lower = item_category.lower()
            if "magic" in cat_lower or "wand" in cat_lower:
                return self._market_spots.get("geffen", list(self._market_spots.values())[0])
            if "card" in cat_lower:
                return self._market_spots.get("morocc", list(self._market_spots.values())[0])
            if "potion" in cat_lower or "herb" in cat_lower:
                return self._market_spots.get("aldebaran", list(self._market_spots.values())[0])

            return self._market_spots.get("prontera", list(self._market_spots.values())[0])

    # ── Price Arbitrage ────────────────────────────────────────────────

    def update_market_price(self, item_name: str, price: int,
                              map_name: str | None = None) -> None:
        """Update the tracked market price for an item.

        If map_name is provided, tracks per-map prices for cross-map arbitrage.
        """
        with self._lock:
            self._current_market_prices[item_name] = price

            # Track price history
            if item_name not in self._price_history:
                self._price_history[item_name] = []
            self._price_history[item_name].append({
                "price": price,
                "map": map_name or "unknown",
                "timestamp": time.time(),
            })

            # Keep last 100 price points
            if len(self._price_history[item_name]) > 100:
                self._price_history[item_name] = self._price_history[item_name][-100:]

    def find_arbitrage_opportunities(self) -> list[ArbitrageOpportunity]:
        """Find buy-low/sell-high opportunities across the market.

        Looks for:
          1. Items that can be bought from NPC and sold to players (NPC arbitrage)
          2. Items with price spreads across different maps
          3. Items undervalued after maintenance (buy dip)

        Returns:
            List of ArbitrageOpportunity sorted by profit.
        """
        with self._lock:
            opportunities: list[ArbitrageOpportunity] = []

            # Simple arbitrage: items where we know base market price
            # and can estimate buy low / sell high windows
            for item_name, price_list in self._price_history.items():
                if len(price_list) < 2:
                    continue

                prices = [p["price"] for p in price_list]
                avg_price = sum(prices) / len(prices)
                min_price = min(prices)
                max_price = max(prices)

                spread = max_price - min_price
                if spread > PROFIT_MIN_ZENY and spread / max(avg_price, 1) > 0.1:
                    # Significant spread found
                    confidence = min(1.0, len(prices) / 20.0)
                    max_qty = min(100, int(1000 / max(avg_price, 1) * 10))
                    opp = ArbitrageOpportunity(
                        item_name=item_name,
                        buy_price=int(min_price),
                        buy_map="market",
                        sell_price=int(max_price),
                        sell_map="market",
                        profit_per_unit=int(spread),
                        profit_margin=round(spread / max(min_price, 1), 2),
                        confidence=round(confidence, 2),
                        max_quantity=max_qty,
                        estimated_total_profit=int(spread * max_qty),
                    )
                    opportunities.append(opp)

            opportunities.sort(key=lambda o: -o.estimated_total_profit)
            self._stats["arbitrage_opportunities_found"] = len(opportunities)  # type: ignore[assignment]
            return opportunities

    def get_mule_report(self, mule_name: str) -> str:
        """Get a formatted report for a specific mule."""
        with self._lock:
            mule = self._mules.get(mule_name)
            if mule is None:
                return f"Unknown mule: {mule_name}"

            lines = [f"── Vending Mule: {mule.name} ──"]
            lines.append(f"  Location: {mule.map_name} ({mule.x},{mule.y})")
            lines.append(f"  Active: {mule.is_active}")
            lines.append(f"  Total sales: {mule.total_sales:,}z")
            lines.append(f"  Total transactions: {mule.total_transactions}")
            lines.append("")
            lines.append("  Currently listed:")
            for item_name, listing in mule.listed_items.items():
                profit_str = f"+{listing.profit_per_unit}z/unit" if listing.profit_per_unit > 0 else "break-even"
                lines.append(f"    {item_name}: {listing.quantity}x @ {listing.price_per_unit:,}z ({profit_str})")
            lines.append("")
            lines.append("  Inventory (backstock):")
            for item_name, qty in mule.inventory.items():
                lines.append(f"    {item_name}: {qty}x")

            return "\n".join(lines)

    def get_all_mules_summary(self) -> str:
        """Get a formatted summary of all mules."""
        with self._lock:
            lines = ["── Vending Operations ──"]
            if not self._mules:
                lines.append("  No mules registered.")
                return "\n".join(lines)

            total_active = sum(1 for m in self._mules.values() if m.is_active)
            lines.append(f"  Mules: {len(self._mules)} total, {total_active} active")
            lines.append(f"  Total zeny earned: {int(self._stats['total_sales']):,}z")
            lines.append(f"  Estimated profit: {int(self._stats['total_profit']):,}z")
            lines.append("")

            for mule_name, mule in self._mules.items():
                status = "🟢" if mule.is_active else "⚪"
                listed = sum(l.quantity for l in mule.listed_items.values())
                lines.append(f"  {status} {mule_name}: {mule.map_name} ({listed} items listed, {mule.total_sales:,}z total)")

            return "\n".join(lines)

    def get_stats(self) -> dict[str, int | float]:
        with self._lock:
            return dict(self._stats)


# ── Global Singleton ─────────────────────────────────────────────────────

_vending_arbitrage: VendingArbitrageEngine | None = None
_vending_arbitrage_lock = RLock()


def get_vending_arbitrage() -> VendingArbitrageEngine:
    """Get the global VendingArbitrageEngine singleton."""
    global _vending_arbitrage
    with _vending_arbitrage_lock:
        if _vending_arbitrage is None:
            _vending_arbitrage = VendingArbitrageEngine()
        return _vending_arbitrage
