"""
Price Trend Analyzer — 7/30-day moving averages, anomaly detection, signals, arbitrage.

Real economic analysis engine for RO markets. Tracks every item's price history
through PersistentState, computes moving averages, detects anomalies via Z-score,
and generates actionable buy/sell/hold/farm signals.

Data sources:
  - Observed trades (record_trade) from player shops, NPC, auctions
  - Seed historical data from AI_sidecar/data/price_history.yaml
  - PersistentState table 'market_prices' for ongoing storage

Signals:
  'buy'   — Price is low relative to trend, trending up → accumulate now
  'sell'  — Price is high relative to trend, trending down → dump now
  'hold'  — Price near moving average, stable → wait
  'avoid' — Price crashing, no bottom in sight → don't touch
"""
from __future__ import annotations

import json
import logging
import math
import random
import statistics
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from ai_sidecar.runtime.persistence import PersistentState

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────

PRICE_HISTORY_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "price_history.yaml"
)

Z_SCORE_ANOMALY_THRESHOLD = 2.0  # |Z| > this = price anomaly
SELL_SIGNAL_THRESHOLD = 0.08      # Price > MA_7d by 8% → sell zone
BUY_SIGNAL_THRESHOLD = 0.06       # Price < MA_7d by 6% → buy zone
MIN_TRADE_COUNT = 3               # Min trades before trend analysis is reliable
DEFAULT_FARM_PRODUCTIVITY: dict[str, float] = {
    # item → zeny_per_drop / difficulty_factor (higher = better farm target)
    "Jellopy": 0.5,
    "Sticky_Mucus": 1.2,
    "Immortal_Heart": 8.0,
    "Fabric": 5.0,
    "Spider_Silk": 4.0,
    "Flame_Heart": 12.0,
    "Elunium": 25.0,
    "Oridecon": 35.0,
    "Gold": 20.0,
}

# ── Helpers ───────────────────────────────────────────────────────


def _now_iso() -> str:
    """ISO-8601 timestamp for persistence."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _compute_moving_average(prices: list[int], window: int) -> float | None:
    """Simple moving average for the last *window* price points."""
    if len(prices) < window:
        return None
    return sum(prices[-window:]) / window


def _compute_z_score(value: float, mean: float, stdev: float) -> float:
    """Z-score: (value - mean) / stdev. Returns 0 if stdev is 0."""
    if stdev == 0:
        return 0.0
    return (value - mean) / stdev


def _is_anomaly(value: float, mean: float, stdev: float, threshold: float = Z_SCORE_ANOMALY_THRESHOLD) -> bool:
    """Check if a value is anomalous (|Z| > threshold)."""
    return abs(_compute_z_score(value, mean, stdev)) > threshold


# ═══════════════════════════════════════════════════════════════════
# TrendAnalyzer
# ═══════════════════════════════════════════════════════════════════


class TrendAnalyzer:
    """Tracks price trends, generates signals, and finds opportunities.

    Usage:
        analyzer = TrendAnalyzer()
        analyzer.record_trade("Elunium", 45000, 5, source="player_shop")
        trend = analyzer.get_price_trend("Elunium")
        signal = analyzer.get_signal("Elunium")
        targets = analyzer.get_best_farming_targets(top_n=3)
        arb = analyzer.get_arbitrage_opportunities()
    """

    def __init__(
        self,
        seed_path: str | Path = PRICE_HISTORY_PATH,
        seed_on_init: bool = True,
    ) -> None:
        # In-memory price series: item_name -> deque of (timestamp_iso, price, quantity, source)
        self._trades: dict[str, deque[tuple[str, int, int, str]]] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self._seed_loaded = False
        self._seed_path = Path(seed_path)

        # Cached computed values
        self._cache: dict[str, dict[str, Any]] = {}

        if seed_on_init:
            self._load_seed_data()

    # ── Public API ─────────────────────────────────────────────────

    def record_trade(
        self, item: str, price: int, quantity: int,
        source: str = "player_shop",
    ) -> None:
        """Log a trade observation.

        Args:
            item: Normalized item name (e.g. 'Elunium').
            price: Zeny price per unit.
            quantity: Number of units traded.
            source: 'player_shop', 'npc', or 'auction'.
        """
        if price <= 0 or quantity <= 0:
            return

        now = _now_iso()
        self._trades[item].append((now, price, quantity, source))

        # Persist to SQLite via PersistentState
        try:
            PersistentState.save_domain_state(
                "economy",
                f"trade:{item}:{now}",
                {
                    "item": item,
                    "price": price,
                    "quantity": quantity,
                    "source": source,
                    "timestamp": now,
                },
            )
            # Update rolling market prices table
            PersistentState.record_trade(item, price, price)
        except Exception as exc:
            logger.warning("Failed to persist trade for %s: %s", item, exc)

        # Invalidate cache for this item
        self._cache.pop(item, None)
        self._cache.pop("_farming_targets", None)
        self._cache.pop("_arbitrage", None)

    def get_price_trend(self, item: str) -> dict[str, Any]:
        """Compute price trend for an item.

        Returns dict:
            item            — normalized item name
            current_price   — most recent observed price (or None)
            ma_7d           — 7-day moving average (or None if < 7 data points)
            ma_30d          — 30-day moving average (or None if < 30 data points)
            trend_percent   — (current_price - ma_7d) / ma_7d * 100
            volatility      — std dev of recent prices / mean
            z_score         — current Z-score vs recent window
            is_anomaly      — True if |z_score| > threshold
            signal          — 'buy' / 'sell' / 'hold' / 'avoid'
            confidence      — 0.0 - 1.0 based on data sufficiency
            trade_count     — number of recorded trades
            last_updated    — ISO timestamp of most recent trade
        """
        cached = self._cache.get(item)
        if cached:
            return cached

        trades = list(self._trades.get(item, []))
        result: dict[str, Any] = {
            "item": item,
            "current_price": None,
            "ma_7d": None,
            "ma_30d": None,
            "trend_percent": None,
            "volatility": None,
            "z_score": None,
            "is_anomaly": False,
            "signal": "hold",
            "confidence": 0.0,
            "trade_count": len(trades),
            "last_updated": None,
        }

        if not trades:
            self._cache[item] = result
            return result

        # Extract price series
        prices = [t[1] for t in trades]
        result["current_price"] = prices[-1]
        result["last_updated"] = trades[-1][0]

        if len(prices) < MIN_TRADE_COUNT:
            self._cache[item] = result
            return result

        # Compute MAs
        ma_7d = _compute_moving_average(prices, min(7, len(prices)))
        ma_30d = _compute_moving_average(prices, min(30, len(prices)))
        result["ma_7d"] = round(ma_7d, 1) if ma_7d is not None else None
        result["ma_30d"] = round(ma_30d, 1) if ma_30d is not None else None

        # Trend percent
        if ma_7d is not None and ma_7d > 0:
            trend_pct = (prices[-1] - ma_7d) / ma_7d
            result["trend_percent"] = round(trend_pct * 100, 2)

        # Volatility (coefficient of variation)
        if len(prices) >= 3:
            try:
                std = statistics.stdev(prices)
                mean = statistics.mean(prices)
                result["volatility"] = round(std / mean, 4) if mean > 0 else 0
            except statistics.StatisticsError:
                result["volatility"] = 0

        # Z-score against recent 7-day window
        recent = prices[-min(7, len(prices)):]
        if len(recent) >= 3:
            try:
                r_mean = statistics.mean(recent)
                r_std = statistics.stdev(recent)
                z = _compute_z_score(prices[-1], r_mean, r_std)
                result["z_score"] = round(z, 2)
                result["is_anomaly"] = _is_anomaly(prices[-1], r_mean, r_std)
            except statistics.StatisticsError:
                pass

        # Generate signal
        result["signal"] = self._generate_signal(result)
        result["confidence"] = self._compute_confidence(result)

        self._cache[item] = result
        return result

    def get_signal(self, item: str) -> str:
        """Quick access to signal string for an item.

        Returns 'buy', 'sell', 'hold', or 'avoid'.
        """
        return self.get_price_trend(item).get("signal", "hold")

    def get_best_farming_targets(self, top_n: int = 3) -> list[dict[str, Any]]:
        """Find the best items to farm based on price-to-effort ratio.

        Ranks items by:
          - Current price trend (uptrend + buy signal = better)
          - Price-to-effort ratio (zeny per drop)
          - Market demand (volume)
          - Stability (avoid crash items)

        Returns top_n ranked items with recommendation metadata.
        """
        candidates: list[dict[str, Any]] = []

        # Collect all items that have trade data
        for item in self._trades:
            trend = self.get_price_trend(item)
            if trend["trade_count"] < MIN_TRADE_COUNT:
                continue
            if trend["current_price"] is None or trend["current_price"] <= 0:
                continue

            score = self._compute_farm_score(item, trend)
            candidates.append({
                "item": item,
                "current_price": trend["current_price"],
                "signal": trend["signal"],
                "trend_percent": trend["trend_percent"],
                "volatility": trend["volatility"],
                "confidence": trend["confidence"],
                "farm_score": round(score, 2),
            })

        # Sort by farm_score descending
        candidates.sort(key=lambda c: c["farm_score"], reverse=True)
        result = candidates[:top_n]

        self._cache["_farming_targets"] = {
            "timestamp": _now_iso(),
            "targets": result,
        }
        return result

    def get_arbitrage_opportunities(self) -> list[dict[str, Any]]:
        """Detect items with large buy/sell spreads.

        An arbitrage opportunity exists when:
          - Price is rising (upward trend)
          - Buy price (player_shop) is significantly below sell price (auction/market)
          - Volume is sufficient to execute

        Returns list of opportunities sorted by profit_potential descending.
        """
        opportunities: list[dict[str, Any]] = []

        for item in self._trades:
            trades = list(self._trades[item])
            if len(trades) < MIN_TRADE_COUNT:
                continue

            trend = self.get_price_trend(item)
            if trend["current_price"] is None:
                continue

            # Find min buy price (player_shop) and max sell price (auction)
            buy_prices = [t[1] for t in trades if t[3] == "player_shop"]
            sell_prices = [t[1] for t in trades if t[3] in ("auction", "npc")]

            if not buy_prices or not sell_prices:
                continue

            avg_buy = statistics.mean(buy_prices)
            avg_sell = statistics.mean(sell_prices)
            spread = avg_sell - avg_buy

            # Only flag if spread > 20% of buy price (minimum viable margin)
            if avg_buy > 0 and spread / avg_buy > 0.20:
                profit_potential = spread * 10  # assume we can flip 10 units
                opportunities.append({
                    "item": item,
                    "avg_buy_price": round(avg_buy, 1),
                    "avg_sell_price": round(avg_sell, 1),
                    "spread": round(spread, 1),
                    "spread_pct": round(spread / avg_buy * 100, 1),
                    "profit_potential": round(profit_potential, 0),
                    "signal": trend["signal"],
                    "buy_sources": len(buy_prices),
                    "sell_sources": len(sell_prices),
                })

        opportunities.sort(key=lambda o: o["profit_potential"], reverse=True)
        result = opportunities[:10]

        self._cache["_arbitrage"] = {
            "timestamp": _now_iso(),
            "opportunities": result,
        }
        return result

    def get_anomalies(self, threshold: float = Z_SCORE_ANOMALY_THRESHOLD) -> list[dict[str, Any]]:
        """Return all items currently flagged as price anomalies."""
        anomalies: list[dict[str, Any]] = []
        for item in self._trades:
            trend = self.get_price_trend(item)
            if trend.get("is_anomaly") and trend["z_score"] is not None:
                anomalies.append(trend)
        anomalies.sort(key=lambda a: abs(a["z_score"] or 0), reverse=True)
        return anomalies

    def get_market_summary(self) -> dict[str, Any]:
        """Get a high-level summary of market conditions.

        Returns dict with:
            total_items_tracked — number of items with trade data
            buy_signals         — items with 'buy' signal
            sell_signals        — items with 'sell' signal
            anomaly_count       — current anomaly count
            top_farming_targets — best targets (cached)
            top_arbitrage       — best arb opportunities (cached)
            last_updated        — ISO timestamp
        """
        buy = []
        sell = []
        hold = []
        avoid = []
        anomalies = 0

        for item in self._trades:
            trend = self.get_price_trend(item)
            sig = trend.get("signal", "hold")
            if sig == "buy":
                buy.append(item)
            elif sig == "sell":
                sell.append(item)
            elif sig == "avoid":
                avoid.append(item)
            else:
                hold.append(item)
            if trend.get("is_anomaly"):
                anomalies += 1

        return {
            "total_items_tracked": len(self._trades),
            "buy_signals": buy,
            "sell_signals": sell,
            "hold_signals": hold,
            "avoid_signals": avoid,
            "anomaly_count": anomalies,
            "top_farming_targets": self._cache.get("_farming_targets", {}).get("targets", [])[:3],
            "top_arbitrage": self._cache.get("_arbitrage", {}).get("opportunities", [])[:3],
            "last_updated": _now_iso(),
        }

    def clear_cache(self) -> None:
        """Reset computed caches. Call after bulk recording trades."""
        self._cache.clear()

    # ── Internal: Signal Generation ────────────────────────────────

    def _generate_signal(self, trend: dict[str, Any]) -> str:
        """Generate a trading signal from the trend data.

        Logic:
          'avoid' → Z-score < -3 (extreme crash) or volatility > 50%
          'sell'  → price > MA_7d by sell_threshold%, or Z-score > 2
          'buy'   → price < MA_7d by buy_threshold%, or Z-score < -2
          'hold'  → everything in between
        """
        price = trend.get("current_price")
        ma_7d = trend.get("ma_7d")
        z_score = trend.get("z_score")
        volatility = trend.get("volatility")
        trade_count = trend.get("trade_count", 0)

        # Need enough data
        if trade_count < MIN_TRADE_COUNT or price is None:
            return "hold"

        # Extreme crash detection
        if z_score is not None and z_score < -3.0:
            return "avoid"
        if volatility is not None and volatility > 0.50:
            # Too volatile to trade safely
            return "avoid"

        # Z-score based signals override threshold-based
        if z_score is not None:
            if z_score > 2.0:
                return "sell"
            if z_score < -2.0:
                return "buy"

        # MA-based signals
        if ma_7d is not None and ma_7d > 0:
            deviation = (price - ma_7d) / ma_7d
            if deviation > SELL_SIGNAL_THRESHOLD:
                return "sell"
            if deviation < -BUY_SIGNAL_THRESHOLD:
                return "buy"

        return "hold"

    def _compute_confidence(self, trend: dict[str, Any]) -> float:
        """Confidence in the signal (0.0 – 1.0).

        Factors:
          - Trade count (more = better)
          - Price recency (recent = better)
          - Volatility (too high = worse)
          - Data completeness (both MAs available = better)
        """
        score = 0.0
        trade_count = trend.get("trade_count", 0)
        volatility = trend.get("volatility")

        # Base: trade count
        if trade_count >= 50:
            score += 0.3
        elif trade_count >= 20:
            score += 0.2
        elif trade_count >= MIN_TRADE_COUNT:
            score += 0.1

        # MA data completeness
        if trend.get("ma_7d") is not None:
            score += 0.25
        if trend.get("ma_30d") is not None:
            score += 0.2

        # Volatility penalty
        if volatility is not None:
            if volatility < 0.05:
                score += 0.15  # very stable
            elif volatility < 0.15:
                score += 0.1
            elif volatility < 0.30:
                score += 0.05
            # >0.30 = no bonus (highly volatile)

        # Z-score availability bonus
        if trend.get("z_score") is not None:
            score += 0.1

        return min(1.0, max(0.0, score))

    def _compute_farm_score(self, item: str, trend: dict[str, Any]) -> float:
        """Score an item as a farming target (higher = better).

        Combines:
          - Signal weight: buy=+40, hold=0, sell=-20, avoid=-100
          - Trend momentum: positive trend_percent adds up to +30
          - Base productivity from DEFAULT_FARM_PRODUCTIVITY
          - Stability: high volatility penalizes
          - Volume proxy: more trades = more demand
        """
        score = 0.0

        # Signal weight
        signal = trend.get("signal", "hold")
        signal_weights = {"buy": 40, "hold": 0, "sell": -20, "avoid": -100}
        score += signal_weights.get(signal, 0)

        # Trend momentum
        trend_pct = trend.get("trend_percent")
        if trend_pct is not None:
            # Positive trend = more valuable to farm now
            score += max(-30, min(30, trend_pct))

        # Base productivity
        base_prod = DEFAULT_FARM_PRODUCTIVITY.get(item, 1.0)
        score += base_prod * 10

        # Stability bonus
        volatility = trend.get("volatility")
        if volatility is not None:
            if volatility < 0.10:
                score += 15  # stable = predictable farm income
            elif volatility < 0.25:
                score += 5
            else:
                score -= 10  # too volatile

        # Confidence bonus (more data = more reliable)
        conf = trend.get("confidence", 0)
        score += conf * 20

        return score

    # ── Internal: Seed Data Loading ────────────────────────────────

    def _load_seed_data(self) -> None:
        """Load historical price data from price_history.yaml.

        Each day's price gets recorded as a synthetic trade so the
        moving averages immediately have data to work from.
        """
        if self._seed_loaded:
            return

        path = self._seed_path
        if not path.exists():
            logger.warning(
                "Price history seed not found at %s — starting with empty history",
                path,
            )
            self._seed_loaded = True
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw: dict[str, Any] = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.error("Failed to load price history seed: %s", exc)
            self._seed_loaded = True
            return

        items_loaded = 0
        trades_loaded = 0

        for item_name, item_data in raw.items():
            if not isinstance(item_data, dict):
                continue
            prices_list = item_data.get("prices", [])
            if not prices_list:
                continue

            for entry in prices_list:
                date_str = entry.get("date", "")
                price = entry.get("price", 0)
                quantity = entry.get("volume", 1)
                source = entry.get("source", "player_shop")

                if price <= 0:
                    continue

                # Use date as timestamp for seed data
                self._trades[item_name].append((date_str, price, quantity, source))

            items_loaded += 1
            trades_loaded += len(prices_list)

        logger.info(
            "Loaded %d items (%d trades) from %s",
            items_loaded,
            trades_loaded,
            path,
        )
        self._seed_loaded = True

    # ── Internal: Persistence (restore session data) ───────────────

    def load_from_persistence(self) -> int:
        """Restore previously recorded trades from PersistentState.

        Reads domain_state entries with key prefix 'trade:' and loads
        them into the in-memory buffer. Returns number of trades restored.
        """
        try:
            all_state = PersistentState.load_all_domain_state("economy")
        except Exception as exc:
            logger.warning("Failed to load persistent state: %s", exc)
            return 0

        restored = 0
        for key, value in all_state.items():
            if not key.startswith("trade:"):
                continue
            if not isinstance(value, dict):
                continue

            item = value.get("item", "")
            price = value.get("price", 0)
            quantity = value.get("quantity", 1)
            source = value.get("source", "player_shop")
            timestamp = value.get("timestamp", _now_iso())

            if item and price > 0:
                self._trades[item].append((timestamp, price, quantity, source))
                restored += 1

        if restored:
            logger.info("Restored %d trades from persistent storage", restored)
        return restored


# ═══════════════════════════════════════════════════════════════════
# Convenience Factory
# ═══════════════════════════════════════════════════════════════════


def get_trend_analyzer(
    seed_path: str | Path = PRICE_HISTORY_PATH,
) -> TrendAnalyzer:
    """Get a fully initialized TrendAnalyzer with seed data + persisted trades."""
    analyzer = TrendAnalyzer(seed_path=seed_path, seed_on_init=True)
    analyzer.load_from_persistence()
    return analyzer
