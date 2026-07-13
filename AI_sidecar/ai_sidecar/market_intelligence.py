"""
Market intelligence — tracks player-driven economy, not static prices.

A pro player knows:
- Yggdrasil Berry prices spike on WoE nights
- Certain cards are undervalued when meta shifts
- Arbitrage opportunities between towns
- Items worth hoarding for next patch
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MarketListing:
    item_name: str
    price: int
    quantity: int
    seller: str
    location: str  # prontera, morocc, etc.
    timestamp: float
    listing_type: str = "buy"  # buy or sell


@dataclass(slots=True)
class MarketIntelligence:
    """Tracks player-driven market prices and arbitrage opportunities."""
    
    _lock: RLock = field(default_factory=RLock)
    _listings: dict[str, list[MarketListing]] = field(default_factory=lambda: defaultdict(list))
    _price_history: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    _arbitrage_routes: list[dict[str, Any]] = field(default_factory=list)
    _stats: dict[str, int] = field(default_factory=lambda: {"listings_tracked": 0, "arbitrage_found": 0})
    
    def record_listing(self, listing: MarketListing) -> None:
        with self._lock:
            self._listings[listing.item_name].append(listing)
            self._price_history[listing.item_name].append({
                "price": listing.price,
                "timestamp": listing.timestamp,
                "location": listing.location,
            })
            self._stats["listings_tracked"] += 1
    
    def get_market_price(self, item_name: str, location: str | None = None) -> dict[str, Any]:
        """Get current market price for an item."""
        with self._lock:
            listings = self._listings.get(item_name, [])
            if location:
                listings = [l for l in listings if l.location == location]
            
            if not listings:
                return {"price": 0, "trend": "unknown", "confidence": 0}
            
            recent = [l for l in listings if time.time() - l.timestamp < 3600]
            if not recent:
                recent = listings[-20:]
            
            prices = [l.price for l in recent if l.price > 0]
            if not prices:
                return {"price": 0, "trend": "unknown", "confidence": 0}
            
            avg_price = sum(prices) / len(prices)
            min_price = min(prices)
            max_price = max(prices)
            
            # Trend analysis
            history = self._price_history.get(item_name, [])
            if len(history) > 10:
                recent_avg = sum(h["price"] for h in history[-5:]) / 5
                old_avg = sum(h["price"] for h in history[-10:-5]) / 5
                trend = "rising" if recent_avg > old_avg * 1.1 else "falling" if recent_avg < old_avg * 0.9 else "stable"
            else:
                trend = "unknown"
            
            return {
                "price": avg_price,
                "min": min_price,
                "max": max_price,
                "trend": trend,
                "confidence": min(1.0, len(recent) / 10),
                "listings": len(recent),
            }
    
    def find_arbitrage(self) -> list[dict[str, Any]]:
        """Find arbitrage opportunities between towns."""
        with self._lock:
            opportunities = []
            locations = set()
            for listings in self._listings.values():
                for l in listings:
                    locations.add(l.location)
            
            for item_name in self._listings:
                prices_by_loc: dict[str, list[int]] = defaultdict(list)
                for l in self._listings[item_name]:
                    if time.time() - l.timestamp < 3600:
                        prices_by_loc[l.location].append(l.price)
                
                if len(prices_by_loc) < 2:
                    continue
                
                for loc1, prices1 in prices_by_loc.items():
                    for loc2, prices2 in prices_by_loc.items():
                        if loc1 >= loc2:
                            continue
                        avg1 = sum(prices1) / len(prices1)
                        avg2 = sum(prices2) / len(prices2)
                        diff_pct = abs(avg1 - avg2) / max(avg1, avg2)
                        
                        if diff_pct > 0.2:  # 20%+ difference
                            buy_loc = loc1 if avg1 < avg2 else loc2
                            sell_loc = loc2 if avg1 < avg2 else loc1
                            profit_pct = abs(avg1 - avg2) / min(avg1, avg2)
                            
                            opportunities.append({
                                "item": item_name,
                                "buy_location": buy_loc,
                                "sell_location": sell_loc,
                                "buy_price": min(avg1, avg2),
                                "sell_price": max(avg1, avg2),
                                "profit_pct": profit_pct,
                            })
            
            opportunities.sort(key=lambda o: o["profit_pct"], reverse=True)
            self._arbitrage_routes = opportunities[:10]
            self._stats["arbitrage_found"] = len(opportunities)
            return self._arbitrage_routes
    
    def is_woe_time(self) -> bool:
        """Check if it's WoE time (prices spike)."""
        now = time.localtime()
        # WoE is typically 2-hour windows, e.g. 8-10pm
        hour = now.tm_hour
        return hour in (20, 21)  # 8-10pm
    
    def get_price_prediction(self, item_name: str) -> dict[str, Any]:
        """Predict price movement based on time and events."""
        market = self.get_market_price(item_name)
        prediction = {"direction": "stable", "confidence": 0.0, "reason": ""}
        
        if self.is_woe_time() and "berry" in item_name.lower():
            prediction["direction"] = "up"
            prediction["confidence"] = 0.8
            prediction["reason"] = "WoE demand spike"
        elif market["trend"] == "rising" and market["confidence"] > 0.5:
            prediction["direction"] = "up"
            prediction["confidence"] = 0.6
            prediction["reason"] = "Sustained upward trend"
        
        return prediction
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)
