"""
Vending Detection — scan player shops, update market prices, identify arbitrage.

When in town:
  - Detects player shops ("Vending" or "@shop" signals)
  - Parses item listings and prices
  - Updates local market price estimates based on observed prices
  - Identifies items that sell for more on market than NPC (vending targets)
  - Recommends which items to vendor vs sell to NPC
"""
from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ai_sidecar.domains.economy.database import ItemValueDB, SELL_PLAYER, SELL_NPC

logger = logging.getLogger(__name__)


@dataclass
class VendingListing:
    """A single listing from a player shop."""
    shop_owner: str
    item_name: str
    item_id: int = 0
    price: int = 0
    quantity: int = 1
    shop_title: str = ""
    map_name: str = ""
    timestamp: float = 0.0


@dataclass
class VendingOpportunity:
    """An arbitrage opportunity detected between NPC and player prices."""
    item_name: str
    item_id: int
    npc_sell_price: int  # what NPC pays
    market_price: int     # estimated player market price
    our_quantity: int     # how many we have in inventory
    profit_potential: int  # (market_price - npc_sell) × quantity
    recommendation: str    # "vendor" or "sell_npc"


class VendingDetector:
    """Detects player shops, observes prices, and identifies vending opportunities.

    Features:
      - Parses shop listings from console/snapshot signals
      - Maintains rolling average of observed market prices
      - Flags items worth vending vs NPC-selling
      - Tracks which maps/npcs host active player markets
    """

    def __init__(self, db: ItemValueDB | None = None) -> None:
        self._db = db or ItemValueDB()
        # Observed market prices: item_name -> [price, ...]
        self._observed_prices: dict[str, list[int]] = defaultdict(list)
        # Active shops on current map
        self._active_shops: list[VendingListing] = []
        # Last scan time per map
        self._last_scan_time: dict[str, float] = {}
        # Shop owner cooldown (avoid re-scanning same shop too often)
        self._shop_cooldowns: dict[str, float] = {}

    # ── Signal parsing ────────────────────────────────────────────

    def scan_shops(self, signals: dict[str, Any]) -> list[VendingListing]:
        """Scan signals for player shop data.

        Looks for:
          - 'vending' or 'shops' key with shop listings
          - Console messages matching shop announce patterns
          - 'npcs' list entries with shop type

        Returns:
            List of parsed VendingListing objects.
        """
        listings: list[VendingListing] = []
        current_map = str(signals.get("map", "") or "").lower().replace(".gat", "")

        # Check for explicit shop data in signals
        raw_shops = signals.get("shops", signals.get("vending", []))
        if isinstance(raw_shops, list):
            for shop in raw_shops:
                listing = self._parse_shop_entry(shop, current_map)
                if listing:
                    listings.append(listing)

        # Check NPCs with shop type
        npcs = signals.get("npcs", [])
        if isinstance(npcs, list):
            for npc in npcs:
                npc_type = str(npc.get("type", npc.get("name", "")) or "")
                if "shop" in npc_type.lower() or "vending" in npc_type.lower():
                    listing = self._parse_shop_entry(npc, current_map)
                    if listing:
                        listings.append(listing)

        # Check console messages for shop announcements
        console = signals.get("console", signals.get("messages", []))
        if isinstance(console, list):
            for msg in console:
                text = str(msg.get("msg", msg) if isinstance(msg, dict) else msg)
                parsed = self._parse_shop_message(text, current_map)
                if parsed:
                    listings.append(parsed)

        # Update observed prices from fresh listings
        for listing in listings:
            if listing.price > 0:
                self._observed_prices[listing.item_name.lower()].append(listing.price)
                # Keep only last 20 observations per item
                prices = self._observed_prices[listing.item_name.lower()]
                if len(prices) > 20:
                    self._observed_prices[listing.item_name.lower()] = prices[-20:]

        if listings:
            self._active_shops = listings
            self._last_scan_time[current_map] = time.time()
            logger.debug("Scanned %d player shops on %s", len(listings), current_map)

        return listings

    def _parse_shop_entry(self, entry: dict[str, Any], current_map: str) -> VendingListing | None:
        """Parse a single shop entry from signals."""
        owner = str(entry.get("owner", entry.get("name", "")) or "")
        item_name = str(entry.get("item", entry.get("display", "")) or "")
        price = int(entry.get("price", entry.get("cost", 0)) or 0)
        quantity = int(entry.get("amount", entry.get("quantity", 1)) or 1)
        item_id = int(entry.get("id", entry.get("item_id", 0)) or 0)
        title = str(entry.get("title", entry.get("shop_title", "")) or "")

        if not item_name and not item_id:
            return None

        return VendingListing(
            shop_owner=owner,
            item_name=item_name.strip(),
            item_id=item_id,
            price=price,
            quantity=quantity,
            shop_title=title,
            map_name=current_map,
            timestamp=time.time(),
        )

    def _parse_shop_message(self, text: str, current_map: str) -> VendingListing | None:
        """Parse a shop announcement from console messages.

        Common patterns:
          - "Player: Buying/Selling item for Zz"
          - "Shop: item_name - pricez"
          - Vending/chat room titles with prices
        """
        text_clean = text.strip()

        # Pattern: "Player : Buying item_name for Zz (qty)"
        buy_match = re.search(
            r'(\w+)\s*:\s*(?:buying|selling)\s+(.+?)\s+for\s+(\d[\d,]*)\s*z',
            text_clean, re.IGNORECASE,
        )
        if buy_match:
            return VendingListing(
                shop_owner=buy_match.group(1),
                item_name=buy_match.group(2).strip(),
                price=int(buy_match.group(3).replace(",", "")),
                quantity=1,
                map_name=current_map,
                timestamp=time.time(),
            )

        # Pattern: "Shop: item_name - pricez"
        shop_match = re.search(
            r'shop[^:]*:\s*(.+?)\s*[–\-]\s*(\d[\d,]*)\s*z',
            text_clean, re.IGNORECASE,
        )
        if shop_match:
            return VendingListing(
                shop_owner="",
                item_name=shop_match.group(1).strip(),
                price=int(shop_match.group(2).replace(",", "")),
                quantity=1,
                map_name=current_map,
                timestamp=time.time(),
            )

        return None

    # ── Price observation ─────────────────────────────────────────

    def get_observed_price(self, item_name: str) -> int:
        """Get the rolling average of observed market prices for an item.

        Returns the median of last 20 observations, or 0 if none.
        """
        prices = self._observed_prices.get(item_name.lower(), [])
        if not prices:
            return 0
        sorted_prices = sorted(prices)
        return sorted_prices[len(sorted_prices) // 2]  # median

    def get_market_price(self, item_name: str) -> int:
        """Get best market price estimate for an item.

        Uses observed player prices if available, otherwise falls back
        to the item value database's market_price estimate.
        """
        observed = self.get_observed_price(item_name)
        if observed > 0:
            return observed
        return self._db.get_market_price(item_name)

    # ── Opportunity detection ─────────────────────────────────────

    def detect_opportunities(
        self,
        inventory: list[dict[str, Any]],
        zeny: int = 0,
    ) -> list[VendingOpportunity]:
        """Find items in inventory that should be vended vs NPC-sold.

        Returns:
            List of VendingOpportunity sorted by profit potential.
        """
        opportunities: list[VendingOpportunity] = []

        for item in inventory:
            name = str(item.get("name", item.get("item", "")))
            quantity = int(item.get("amount", item.get("quantity", 1)))
            item_id = int(item.get("id", item.get("item_id", 0)) or 0)

            npc_sell = self._db.get_npc_sell_price(name)
            market_price = self.get_market_price(name)

            if market_price <= npc_sell:
                continue  # no profit from vending vs NPC

            # Need at least 100z profit per unit to bother vending
            profit_per_unit = market_price - npc_sell
            if profit_per_unit < 100:
                continue

            # Only flag items with significant total profit
            total_profit = profit_per_unit * quantity
            if total_profit < 500:
                continue

            # Check classification — don't vendor keep/crafting/quest items
            classification = self._db.get_classification(name)
            if classification in ("keep", "crafting", "quest", "material"):
                continue

            recommendation = "vendor" if market_price > npc_sell * 2 else "sell_npc"
            db_entry = self._db.get(name)
            resolved_id = item_id or (db_entry.get("id", 0) if db_entry else 0)
            opportunities.append(VendingOpportunity(
                item_name=name,
                item_id=resolved_id,
                npc_sell_price=npc_sell,
                market_price=market_price,
                our_quantity=quantity,
                profit_potential=total_profit,
                recommendation=recommendation,
            ))

        # Sort by profit potential descending
        opportunities.sort(key=lambda o: o.profit_potential, reverse=True)
        return opportunities

    def should_vendor(self, item_name: str, quantity: int = 1) -> bool:
        """Check if an item is worth setting up a vendor for.

        Args:
            item_name: Item name.
            quantity: How many we have.

        Returns:
            True if we should vending instead of NPC-selling.
        """
        npc_sell = self._db.get_npc_sell_price(item_name)
        market_price = self.get_market_price(item_name)

        if market_price <= npc_sell:
            return False

        profit_per_unit = market_price - npc_sell
        total_profit = profit_per_unit * quantity

        # Only worth vending if profit > 1000z total
        return total_profit >= 1000 and profit_per_unit >= 100

    def get_vendor_recommendations(self) -> list[dict[str, Any]]:
        """Get items we should try to buy from player shops and resell.

        Returns:
            List of arbitrage recommendations.
        """
        recommendations: list[dict[str, Any]] = []

        # Look for items where player selling price is below market estimate
        for item_name, prices in self._observed_prices.items():
            if not prices:
                continue
            avg_price = sum(prices) / len(prices)
            market_est = self._db.get_market_price(item_name)

            # If players are selling below market, we could buy and resell
            if market_est > avg_price * 1.2:  # 20%+ margin
                recommendations.append({
                    "item_name": item_name,
                    "avg_buy_price": round(avg_price),
                    "estimated_resell": market_est,
                    "profit_per_unit": market_est - round(avg_price),
                    "confidence": min(1.0, len(prices) / 10),
                })

        return recommendations

    # ── Shop state ─────────────────────────────────────────────────

    def get_active_shops(self) -> list[VendingListing]:
        """Get currently active player shops on the map."""
        return self._active_shops

    def should_rescan(self, map_name: str, cooldown_seconds: int = 60) -> bool:
        """Check if we should scan shops on this map again."""
        last = self._last_scan_time.get(map_name.lower(), 0)
        return (time.time() - last) > cooldown_seconds
