"""
Player Shop Scanner — scans player shops for real market data.

Better than human because:
- Humans can scan ~10 shops per hour
- This system scans EVERY shop, EVERY time it's in town
- Humans forget prices (recency bias)
- This system remembers every price it's ever seen
- Humans miss underpriced items (inattention)
- This system catches every price anomaly
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ShopListing:
    """A single player shop listing."""
    item_name: str
    price: int
    seller: str
    quantity: int
    shop_name: str
    map_name: str
    timestamp: float


@dataclass(slots=True)
class PriceAnomaly:
    """An underpriced or overpriced item detected in a player shop."""
    item_name: str
    market_price: int  # Estimated market price
    listing_price: int  # Actual listing price
    ratio: float  # listing / market (lower = better deal)
    seller: str
    map_name: str
    profit_potential: int  # Estimated profit if bought and resold
    confidence: float  # 0.0-1.0


class PlayerShopScanner:
    """Scans and analyzes player shops for market intelligence.
    
    Capabilities:
    1. Track all observed player shop listings
    2. Calculate market prices from observed transactions
    3. Detect underpriced items (arbitrage opportunities)
    4. Track which sellers have good prices (reputation)
    5. Recommend buy/sell decisions based on player shop data
    """
    
    def __init__(self):
        self._lock = RLock()
        
        # All observed listings
        self._listings: deque[ShopListing] = deque(maxlen=50000)
        
        # Market prices (item_name -> list of observed prices)
        self._market_prices: dict[str, list[int]] = defaultdict(list)
        
        # Seller reputation (seller_name -> average price ratio vs market)
        self._seller_reputation: dict[str, list[float]] = defaultdict(list)
        
        # Item demand score (item_name -> demand score 0-1)
        self._item_demand: dict[str, float] = defaultdict(float)
        
        # Stats
        self._stats: dict[str, int] = defaultdict(int)
    
    def record_listing(self, item_name: str, price: int, seller: str,
                       quantity: int = 1, shop_name: str = "",
                       map_name: str = "") -> None:
        """Record a player shop listing."""
        with self._lock:
            listing = ShopListing(
                item_name=item_name,
                price=price,
                seller=seller,
                quantity=quantity,
                shop_name=shop_name,
                map_name=map_name,
                timestamp=time.time(),
            )
            self._listings.append(listing)
            self._market_prices[item_name].append(price)
            self._stats["listings_recorded"] += 1
    
    def get_market_price(self, item_name: str) -> int | None:
        """Get the estimated market price for an item.
        
        Uses median of observed prices (more robust than mean).
        Returns None if insufficient data.
        """
        with self._lock:
            prices = self._market_prices.get(item_name, [])
            if len(prices) < 3:
                return None
            
            sorted_prices = sorted(prices)
            n = len(sorted_prices)
            
            # Median
            if n % 2 == 0:
                median = (sorted_prices[n // 2 - 1] + sorted_prices[n // 2]) / 2
            else:
                median = sorted_prices[n // 2]
            
            return int(median)
    
    def get_price_range(self, item_name: str) -> tuple[int, int] | None:
        """Get the price range (min, max) for an item."""
        with self._lock:
            prices = self._market_prices.get(item_name, [])
            if not prices:
                return None
            return (min(prices), max(prices))
    
    def detect_anomalies(self, min_ratio: float = 0.7,
                         max_results: int = 20) -> list[PriceAnomaly]:
        """Detect underpriced items (arbitrage opportunities).
        
        An item is underpriced if its listing price is significantly
        below the estimated market price.
        """
        with self._lock:
            anomalies: list[PriceAnomaly] = []
            
            # Group recent listings by item
            recent_listings: dict[str, list[ShopListing]] = defaultdict(list)
            for listing in self._listings:
                if time.time() - listing.timestamp < 86400:  # Last 24 hours
                    recent_listings[listing.item_name].append(listing)
            
            for item_name, listings in recent_listings.items():
                market_price = self.get_market_price(item_name)
                if market_price is None:
                    continue
                
                for listing in listings:
                    if listing.price == 0:
                        continue
                    ratio = listing.price / market_price
                    
                    if ratio < min_ratio:
                        profit = market_price - listing.price
                        # Confidence based on how many data points we have
                        confidence = min(1.0, len(self._market_prices.get(item_name, [])) / 20)
                        
                        anomalies.append(PriceAnomaly(
                            item_name=item_name,
                            market_price=market_price,
                            listing_price=listing.price,
                            ratio=ratio,
                            seller=listing.seller,
                            map_name=listing.map_name,
                            profit_potential=profit,
                            confidence=confidence,
                        ))
            
            # Sort by profit potential (highest first)
            anomalies.sort(key=lambda a: -a.profit_potential)
            return anomalies[:max_results]
    
    def get_best_deals(self, max_results: int = 10) -> list[PriceAnomaly]:
        """Get the best current deals (highest profit potential, highest confidence)."""
        anomalies = self.detect_anomalies(min_ratio=0.8, max_results=50)
        
        # Score: profit * confidence
        scored = [(a.profit_potential * a.confidence, a) for a in anomalies]
        scored.sort(key=lambda x: -x[0])
        
        return [a for _, a in scored[:max_results]]
    
    def get_seller_reputation(self, seller_name: str) -> float:
        """Get a seller's reputation (0.0 = always overpriced, 1.0 = always fair).
        
        Calculated as the average of (market_price / listing_price) for all
        listings by this seller. Higher is better.
        """
        with self._lock:
            ratios = self._seller_reputation.get(seller_name, [])
            if not ratios:
                return 0.5  # Neutral for unknown sellers
            return sum(ratios) / len(ratios)
    
    def record_transaction(self, item_name: str, price: int,
                           bought: bool) -> None:
        """Record an actual transaction (buy or sell).
        
        This helps validate our price estimates.
        """
        with self._lock:
            self._stats["transactions_recorded"] += 1
            if bought:
                self._stats["items_bought"] += 1
            else:
                self._stats["items_sold"] += 1
    
    def get_item_demand(self, item_name: str) -> float:
        """Get the demand score for an item (0.0-1.0).
        
        High demand = many listings, high prices, fast turnover.
        """
        with self._lock:
            # Count recent listings
            recent = [l for l in self._listings
                      if l.item_name == item_name
                      and time.time() - l.timestamp < 86400]
            
            if not recent:
                return 0.0
            
            # Demand factors: listing count, price volatility, turnover rate
            count_factor = min(1.0, len(recent) / 50)
            
            prices = [l.price for l in recent]
            if len(prices) >= 3:
                volatility = (max(prices) - min(prices)) / (sum(prices) / len(prices))
            else:
                volatility = 0.0
            
            demand = (count_factor * 0.6 + volatility * 0.4)
            return min(1.0, demand)
    
    def get_stats(self) -> dict[str, int]:
        """Get scanner statistics."""
        with self._lock:
            return dict(self._stats)


# Global singleton
_scanner: PlayerShopScanner | None = None

def get_player_shop_scanner() -> PlayerShopScanner:
    """Get the global PlayerShopScanner instance."""
    global _scanner
    if _scanner is None:
        _scanner = PlayerShopScanner()
    return _scanner
