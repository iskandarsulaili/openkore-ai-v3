"""
Market Arbitrage Engine — tracks item prices, identifies arbitrage opportunities,
and executes trades automatically. Buy low, hoard, sell high.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class PriceRecord:
    """A price observation for an item."""
    item_name: str
    item_id: int = 0
    buy_price: int = 0
    sell_price: int = 0
    quantity_available: int = 0
    timestamp: float = 0.0
    source: str = "vending"  # vending, buying_store, player_trade, npc


@dataclass
class ArbitrageOpportunity:
    """An arbitrage opportunity."""
    item_name: str
    item_id: int = 0
    buy_price: int = 0
    sell_price: int = 0
    profit_per_unit: int = 0
    expected_volume: int = 0
    total_profit: int = 0
    roi_pct: float = 0.0
    risk_level: str = "low"  # low, medium, high
    confidence: float = 0.0
    timestamp: float = 0.0
    strategy: str = ""  # flip, hoard, corner, arbitrage


@dataclass
class MarketTrend:
    """A market trend for an item."""
    item_name: str
    avg_buy_price_7d: float = 0.0
    avg_sell_price_7d: float = 0.0
    price_trend: str = "stable"  # rising, falling, stable
    volatility: float = 0.0
    volume_7d: int = 0
    best_time_to_sell: str = ""
    notes: str = ""


class MarketArbitrage:
    """Tracks item prices and identifies arbitrage opportunities."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._price_history: dict[str, list[PriceRecord]] = defaultdict(list)
        self._opportunities: list[ArbitrageOpportunity] = []
        self._trends: dict[str, MarketTrend] = {}
        self._max_history: int = 1000
        self._capital: int = 0
        self._inventory: dict[str, int] = {}  # item_name -> quantity held
        self._total_profit: int = 0
        self._trades_executed: int = 0
        self._enqueue_fn: Callable | None = None
        self._load_known_items()

    def _load_known_items(self) -> None:
        """Load known valuable items and their typical prices."""
        self._known_items: dict[str, dict] = {
            # Cards (high value, low volume)
            "Poring Card": {"category": "card", "typical_price": 10000, "volatility": "medium"},
            "Drops Card": {"category": "card", "typical_price": 8000, "volatility": "medium"},
            "Poporing Card": {"category": "card", "typical_price": 15000, "volatility": "medium"},
            "Lunatic Card": {"category": "card", "typical_price": 5000, "volatility": "medium"},
            "Savage Card": {"category": "card", "typical_price": 50000, "volatility": "medium"},
            "Thara Frog Card": {"category": "card", "typical_price": 200000, "volatility": "high"},
            "Hydra Card": {"category": "card", "typical_price": 150000, "volatility": "high"},
            "Vadon Card": {"category": "card", "typical_price": 100000, "volatility": "high"},
            "Drainliar Card": {"category": "card", "typical_price": 80000, "volatility": "high"},
            "Marc Card": {"category": "card", "typical_price": 120000, "volatility": "high"},
            "Mantis Card": {"category": "card", "typical_price": 60000, "volatility": "medium"},
            "Undead Card": {"category": "card", "typical_price": 100000, "volatility": "high"},
            "Ghost Card": {"category": "card", "typical_price": 300000, "volatility": "high"},
            "Demon Card": {"category": "card", "typical_price": 250000, "volatility": "high"},
            "Brute Card": {"category": "card", "typical_price": 50000, "volatility": "medium"},
            "Fish Card": {"category": "card", "typical_price": 40000, "volatility": "medium"},
            "Insect Card": {"category": "card", "typical_price": 30000, "volatility": "medium"},
            "Plant Card": {"category": "card", "typical_price": 20000, "volatility": "medium"},

            # Crafting Materials (high volume, stable)
            "Stem": {"category": "material", "typical_price": 100, "volatility": "low"},
            "Flower": {"category": "material", "typical_price": 50, "volatility": "low"},
            "Herb": {"category": "material", "typical_price": 200, "volatility": "low"},
            "Iron Ore": {"category": "material", "typical_price": 300, "volatility": "low"},
            "Coal": {"category": "material", "typical_price": 500, "volatility": "low"},
            "Steel": {"category": "material", "typical_price": 2000, "volatility": "low"},
            "Oridecon": {"category": "material", "typical_price": 50000, "volatility": "medium"},
            "Elunium": {"category": "material", "typical_price": 30000, "volatility": "medium"},

            # Potions (high volume, stable)
            "White Potion": {"category": "consumable", "typical_price": 500, "volatility": "low"},
            "Blue Potion": {"category": "consumable", "typical_price": 2000, "volatility": "low"},
            "Red Potion": {"category": "consumable", "typical_price": 50, "volatility": "low"},
            "Orange Potion": {"category": "consumable", "typical_price": 200, "volatility": "low"},
            "Yggdrasil Leaf": {"category": "consumable", "typical_price": 5000, "volatility": "medium"},
            "Yggdrasil Berry": {"category": "consumable", "typical_price": 50000, "volatility": "high"},
            "Condensed White Potion": {"category": "consumable", "typical_price": 2000, "volatility": "low"},
        }

    # ── Public API ──

    def record_price(self, record: PriceRecord) -> None:
        """Record a price observation."""
        with self._lock:
            self._price_history[record.item_name].append(record)
            if len(self._price_history[record.item_name]) > self._max_history:
                self._price_history[record.item_name] = self._price_history[record.item_name][-self._max_history:]
            self._update_trend(record.item_name)
            self._check_arbitrage(record.item_name)

    def _update_trend(self, item_name: str) -> None:
        """Update market trend for an item."""
        history = self._price_history.get(item_name, [])
        if len(history) < 5:
            return

        recent = history[-20:]
        prices = [r.sell_price for r in recent if r.sell_price > 0]
        if not prices:
            return

        avg_price = sum(prices) / len(prices)
        if len(prices) >= 10:
            first_half = sum(prices[:5]) / 5
            second_half = sum(prices[-5:]) / 5
            if second_half > first_half * 1.1:
                trend = "rising"
            elif second_half < first_half * 0.9:
                trend = "falling"
            else:
                trend = "stable"
        else:
            trend = "stable"

        volatility = (max(prices) - min(prices)) / avg_price if avg_price > 0 else 0

        self._trends[item_name] = MarketTrend(
            item_name=item_name,
            avg_buy_price_7d=avg_price,
            avg_sell_price_7d=avg_price,
            price_trend=trend,
            volatility=volatility,
            volume_7d=len(history),
        )

    def _check_arbitrage(self, item_name: str) -> None:
        """Check for arbitrage opportunities."""
        history = self._price_history.get(item_name, [])
        if len(history) < 3:
            return

        recent = history[-10:]
        buy_prices = [r.buy_price for r in recent if r.buy_price > 0]
        sell_prices = [r.sell_price for r in recent if r.sell_price > 0]

        if not buy_prices or not sell_prices:
            return

        avg_buy = sum(buy_prices) / len(buy_prices)
        avg_sell = sum(sell_prices) / len(sell_prices)
        min_buy = min(buy_prices)
        max_sell = max(sell_prices)

        # Check for flip opportunity (buy low, sell high)
        if max_sell > min_buy * 1.3:  # 30%+ profit
            profit = int(max_sell - min_buy)
            roi = (profit / min_buy) * 100
            opp = ArbitrageOpportunity(
                item_name=item_name,
                item_id=recent[0].item_id,
                buy_price=min_buy,
                sell_price=max_sell,
                profit_per_unit=profit,
                expected_volume=10,
                total_profit=profit * 10,
                roi_pct=roi,
                risk_level="low" if roi < 50 else "medium",
                confidence=min(1.0, len(recent) / 10),
                timestamp=time.time(),
                strategy="flip",
            )
            self._opportunities.append(opp)
            if len(self._opportunities) > 100:
                self._opportunities.pop(0)

        # Check for hoard opportunity (buy now, sell later when price rises)
        trend = self._trends.get(item_name)
        if trend and trend.price_trend == "rising" and avg_buy < avg_sell * 0.8:
            profit = int(avg_sell - avg_buy)
            roi = (profit / avg_buy) * 100
            opp = ArbitrageOpportunity(
                item_name=item_name,
                item_id=recent[0].item_id,
                buy_price=int(avg_buy),
                sell_price=int(avg_sell),
                profit_per_unit=profit,
                expected_volume=50,
                total_profit=profit * 50,
                roi_pct=roi,
                risk_level="medium",
                confidence=0.6,
                timestamp=time.time(),
                strategy="hoard",
            )
            self._opportunities.append(opp)

    def get_best_opportunity(self, min_roi: float = 20.0, max_risk: str = "medium") -> ArbitrageOpportunity | None:
        """Get the best arbitrage opportunity."""
        with self._lock:
            candidates = [o for o in self._opportunities if o.roi_pct >= min_roi]
            risk_order = {"low": 0, "medium": 1, "high": 2}
            candidates = [o for o in candidates if risk_order.get(o.risk_level, 0) <= risk_order.get(max_risk, 1)]
            if not candidates:
                return None
            candidates.sort(key=lambda o: -o.total_profit)
            return candidates[0]

    def get_opportunities(self, strategy: str | None = None) -> list[ArbitrageOpportunity]:
        with self._lock:
            if strategy:
                return [o for o in self._opportunities if o.strategy == strategy]
            return list(self._opportunities)

    def get_trend(self, item_name: str) -> MarketTrend | None:
        with self._lock:
            return self._trends.get(item_name)

    def get_trending_items(self, trend: str = "rising") -> list[MarketTrend]:
        with self._lock:
            return [t for t in self._trends.values() if t.price_trend == trend]

    def get_item_price(self, item_name: str) -> int:
        """Get the latest known price for an item."""
        with self._lock:
            history = self._price_history.get(item_name, [])
            if not history:
                info = self._known_items.get(item_name, {})
                return info.get("typical_price", 0)
            return history[-1].sell_price or history[-1].buy_price

    def record_trade(self, item_name: str, quantity: int, buy_price: int, sell_price: int) -> None:
        """Record a completed trade."""
        with self._lock:
            profit = (sell_price - buy_price) * quantity
            self._total_profit += profit
            self._trades_executed += 1
            if item_name in self._inventory:
                self._inventory[item_name] = self._inventory.get(item_name, 0) - quantity
            logger.info("trade_completed: %s x%d buy=%d sell=%d profit=%d",
                        item_name, quantity, buy_price, sell_price, profit)

    def set_capital(self, capital: int) -> None:
        with self._lock:
            self._capital = capital

    def get_capital(self) -> int:
        with self._lock:
            return self._capital

    def get_total_profit(self) -> int:
        with self._lock:
            return self._total_profit

    def get_trades_executed(self) -> int:
        with self._lock:
            return self._trades_executed

    def get_market_summary(self) -> str:
        with self._lock:
            lines = [f"── Market Arbitrage Summary ──"]
            lines.append(f"Capital: {self._capital:,}z")
            lines.append(f"Total profit: {self._total_profit:,}z")
            lines.append(f"Trades executed: {self._trades_executed}")
            lines.append(f"Items tracked: {len(self._price_history)}")
            lines.append(f"Active opportunities: {len(self._opportunities)}")
            rising = self.get_trending_items("rising")
            if rising:
                lines.append(f"Rising items: {', '.join(t.item_name for t in rising[:5])}")
            falling = self.get_trending_items("falling")
            if falling:
                lines.append(f"Falling items: {', '.join(t.item_name for t in falling[:5])}")
            best = self.get_best_opportunity()
            if best:
                lines.append(f"Best opportunity: {best.item_name} ({best.strategy}, ROI={best.roi_pct:.0f}%)")
            return "\n".join(lines)

    def set_enqueue_fn(self, fn: Callable) -> None:
        with self._lock:
            self._enqueue_fn = fn

    def reset(self) -> None:
        with self._lock:
            self._price_history.clear()
            self._opportunities.clear()
            self._trends.clear()
            self._inventory.clear()
            self._total_profit = 0
            self._trades_executed = 0


# ── Global Singleton ──

_market_arb: MarketArbitrage | None = None
_market_arb_lock = RLock()


def get_market_arbitrage() -> MarketArbitrage:
    global _market_arb
    with _market_arb_lock:
        if _market_arb is None:
            _market_arb = MarketArbitrage()
        return _market_arb
