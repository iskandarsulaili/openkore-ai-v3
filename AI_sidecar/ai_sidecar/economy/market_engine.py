"""
Market Engine — Complete economic intelligence for the bot fleet.

A pro player knows:
- What items are worth farming right now based on current market prices
- Where to buy low and sell high (arbitrage between towns)
- When prices spike (WOE nights, patch days, weekends)
- How to corner a market (buy out supply, resell at premium)
- What each farming spot yields in zeny/hour
- When to hold items vs sell immediately
- How to manage a merchant empire across 8 characters

This engine wires into:
- market_intelligence.py (existing price tracking)
- p2p_knowledge.py (shared price data across bots)
- item_value_db.py (item valuation)
- farming_selector.py (what to farm)
- opportunity_cost.py (value per hour)
"""

from __future__ import annotations

import logging
import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────────────────────


@dataclass
class FarmingSpot:
    """A farming location with estimated value."""
    map_name: str
    monster_name: str
    estimated_zeny_per_hour: float
    estimated_exp_per_hour: float
    estimated_danger_score: float  # 0.0 (safe) to 1.0 (deadly)
    confidence: float  # 0.0 to 1.0
    primary_drops: list[str] = field(default_factory=list)
    card_drops: list[str] = field(default_factory=list)
    competition_level: str = "unknown"  # low, medium, high
    last_assessed: float = 0.0


@dataclass
class ArbitrageOpportunity:
    """An arbitrage opportunity between two locations."""
    item_name: str
    buy_location: str
    sell_location: str
    buy_price: int
    sell_price: int
    profit_per_unit: int
    profit_pct: float
    estimated_volume: int  # how many can be moved
    confidence: float
    expires_at: float  # timestamp when this opportunity is stale


@dataclass
class MarketManipulationPlan:
    """A plan to manipulate a market segment."""
    item_name: str
    target_price: int
    current_price: int
    supply_to_buy: int
    estimated_cost: int
    estimated_profit: int
    risk_score: float  # 0.0 (safe) to 1.0 (risky)
    time_horizon_hours: float
    status: str = "proposed"  # proposed, active, completed, failed


@dataclass
class MerchantEmpire:
    """Track a merchant empire across multiple characters."""
    character_name: str
    role: str  # farmer, crafter, vendor, buyer, transporter
    current_map: str = ""
    zeny: int = 0
    inventory_value: int = 0
    active_vends: list[dict[str, Any]] = field(default_factory=list)
    last_profit_recorded: float = 0.0
    total_profit_today: int = 0


@dataclass
class ValuePerHour:
    """Value per hour calculation for an activity."""
    activity: str  # farm, craft, vend, arbitrage, quest
    location: str = ""
    zeny_per_hour: float = 0.0
    exp_per_hour: float = 0.0
    item_value_per_hour: float = 0.0
    total_value_per_hour: float = 0.0
    opportunity_cost: float = 0.0  # what you give up by doing this
    net_value: float = 0.0  # total - opportunity cost
    confidence: float = 0.0
    last_assessed: float = 0.0


@dataclass(slots=True)
class MarketEngine:
    """
    Complete market intelligence engine.

    Tracks prices, finds arbitrage, recommends farming spots,
    manages merchant empire, and wires into P2P knowledge sharing.
    """

    _lock: RLock = field(default_factory=RLock)
    _farming_spots: dict[str, FarmingSpot] = field(default_factory=dict)
    _arbitrage_opportunities: list[ArbitrageOpportunity] = field(default_factory=list)
    _manipulation_plans: list[MarketManipulationPlan] = field(default_factory=list)
    _merchant_empire: dict[str, MerchantEmpire] = field(default_factory=dict)
    _price_history: dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=168)))  # 168 hours = 1 week
    _hourly_prices: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _daily_prices: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _weekly_prices: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _value_per_hour_cache: dict[str, ValuePerHour] = field(default_factory=dict)
    _stats: dict[str, int] = field(default_factory=lambda: {
        "arbitrage_found": 0, "manipulations_attempted": 0,
        "farming_recommendations": 0, "price_updates": 0,
        "p2p_shares": 0, "merchant_trades": 0,
    })
    _p2p_node: Any = None  # P2PKnowledgeNode for sharing prices
    _market_intel: Any = None  # MarketIntelligence instance
    _item_value_db: Any = None  # ItemValueDB instance
    _farming_selector: Any = None  # FarmingTargetSelector instance
    _opportunity_cost: Any = None  # OpportunityCostEngine instance
    _last_cleanup: float = 0.0

    # ── Configuration ──

    WOE_HOURS: list[int] = field(default_factory=lambda: [20, 21])  # 8-10pm
    WEEKEND_DAYS: list[int] = field(default_factory=lambda: [5, 6])  # Saturday, Sunday
    ARBITRAGE_MIN_PROFIT_PCT: float = 0.15  # 15% minimum for arbitrage
    MANIPULATION_MIN_CAPITAL: int = 100000  # 100k zeny minimum for manipulation
    FARMING_SPOT_STALE_AFTER: float = 3600  # 1 hour
    PRICE_TREND_WINDOW: int = 5  # hours for trend calculation

    # ── Public API ──

    def set_p2p_node(self, node: Any) -> None:
        """Wire P2P knowledge node for price sharing."""
        self._p2p_node = node

    def set_market_intel(self, intel: Any) -> None:
        """Wire existing MarketIntelligence instance."""
        self._market_intel = intel

    def set_item_value_db(self, db: Any) -> None:
        """Wire ItemValueDB instance."""
        self._item_value_db = db

    def set_farming_selector(self, selector: Any) -> None:
        """Wire FarmingTargetSelector instance."""
        self._farming_selector = selector

    def set_opportunity_cost(self, oc: Any) -> None:
        """Wire OpportunityCostEngine instance."""
        self._opportunity_cost = oc

    # ── Price Tracking ──

    def record_price(self, item_name: str, price: int, location: str = "",
                     source: str = "vendor") -> None:
        """Record an observed price for an item."""
        with self._lock:
            now = time.time()
            if item_name not in self._price_history:
                self._price_history[item_name] = deque(maxlen=168)
            self._price_history[item_name].append({
                "price": price,
                "location": location,
                "source": source,
                "timestamp": now,
            })
            self._stats["price_updates"] += 1

            # Update hourly aggregation
            hour_key = time.strftime("%Y-%m-%d-%H", time.localtime(now))
            self._hourly_prices[item_name].append(price)

            # Share with P2P network
            if self._p2p_node is not None:
                try:
                    trend = self._get_trend(item_name)
                    self._p2p_node.broadcast_market_price(
                        item_name=item_name,
                        price=price,
                        listing_count=1,
                        trend=trend,
                    )
                    self._stats["p2p_shares"] += 1
                except Exception:
                    pass

    def get_current_price(self, item_name: str, location: str = "") -> dict[str, Any]:
        """Get current market price for an item with confidence."""
        with self._lock:
            history = self._price_history.get(item_name)
            if not history:
                # Fall back to market_intelligence
                if self._market_intel is not None:
                    try:
                        return self._market_intel.get_market_price(item_name, location or None)
                    except Exception:
                        pass
                return {"price": 0, "min": 0, "max": 0, "trend": "unknown",
                        "confidence": 0.0, "listings": 0}

            # Get recent prices (last hour)
            now = time.time()
            recent = [h for h in history if now - h["timestamp"] < 3600]
            if not recent:
                recent = list(history)[-20:]

            if not recent:
                return {"price": 0, "min": 0, "max": 0, "trend": "unknown",
                        "confidence": 0.0, "listings": 0}

            prices = [h["price"] for h in recent if h["price"] > 0]
            if not prices:
                return {"price": 0, "min": 0, "max": 0, "trend": "unknown",
                        "confidence": 0.0, "listings": 0}

            avg_price = sum(prices) / len(prices)
            min_price = min(prices)
            max_price = max(prices)

            # Trend analysis
            trend = self._get_trend(item_name)

            return {
                "price": avg_price,
                "min": min_price,
                "max": max_price,
                "trend": trend,
                "confidence": min(1.0, len(recent) / 10),
                "listings": len(recent),
            }

    def _get_trend(self, item_name: str) -> str:
        """Calculate price trend for an item."""
        history = self._price_history.get(item_name)
        if not history or len(history) < 10:
            return "unknown"

        recent = list(history)
        window = min(self.PRICE_TREND_WINDOW, len(recent) // 2)
        if window < 2:
            return "unknown"

        recent_prices = [h["price"] for h in recent[-window:] if h["price"] > 0]
        old_prices = [h["price"] for h in recent[-window*2:-window] if h["price"] > 0]

        if not recent_prices or not old_prices:
            return "unknown"

        recent_avg = sum(recent_prices) / len(recent_prices)
        old_avg = sum(old_prices) / len(old_prices)

        if old_avg == 0:
            return "unknown"

        ratio = recent_avg / old_avg
        if ratio > 1.1:
            return "rising"
        elif ratio < 0.9:
            return "falling"
        return "stable"

    def get_price_prediction(self, item_name: str) -> dict[str, Any]:
        """Predict short-term price movement."""
        market = self.get_current_price(item_name)
        prediction = {"direction": "stable", "confidence": 0.0,
                      "reason": "", "target_price": market["price"]}

        now = time.localtime()
        hour = now.tm_hour
        wday = now.tm_wday

        # WOE-aware pricing
        if hour in self.WOE_HOURS:
            if any(kw in item_name.lower() for kw in
                   ["berry", "yggdrasil", "potion", "white_potion",
                    "blue_potion", "seed", "flower"]):
                prediction["direction"] = "up"
                prediction["confidence"] = 0.8
                prediction["reason"] = "WOE demand spike"
                prediction["target_price"] = market["price"] * 1.3
                return prediction

        # Weekend pricing
        if wday in self.WEEKEND_DAYS:
            if any(kw in item_name.lower() for kw in
                   ["card", "elunium", "oridecon", "rough_elunium",
                    "rough_oridecon", "emerald", "sapphire"]):
                prediction["direction"] = "up"
                prediction["confidence"] = 0.6
                prediction["reason"] = "Weekend demand increase"
                prediction["target_price"] = market["price"] * 1.15
                return prediction

        # Trend-based prediction
        if market["trend"] == "rising" and market["confidence"] > 0.5:
            prediction["direction"] = "up"
            prediction["confidence"] = 0.6
            prediction["reason"] = "Sustained upward trend"
            prediction["target_price"] = market["price"] * 1.1
        elif market["trend"] == "falling" and market["confidence"] > 0.5:
            prediction["direction"] = "down"
            prediction["confidence"] = 0.5
            prediction["reason"] = "Sustained downward trend"
            prediction["target_price"] = market["price"] * 0.9

        return prediction

    # ── Arbitrage ──

    def find_arbitrage(self) -> list[ArbitrageOpportunity]:
        """Find arbitrage opportunities between towns."""
        with self._lock:
            opportunities = []
            items = list(self._price_history.keys())

            for item_name in items:
                history = self._price_history.get(item_name)
                if not history:
                    continue

                # Group prices by location (last 2 hours)
                now = time.time()
                prices_by_loc: dict[str, list[int]] = defaultdict(list)
                for h in history:
                    if now - h["timestamp"] < 7200 and h["location"]:
                        prices_by_loc[h["location"]].append(h["price"])

                if len(prices_by_loc) < 2:
                    continue

                for loc1, prices1 in prices_by_loc.items():
                    for loc2, prices2 in prices_by_loc.items():
                        if loc1 >= loc2:
                            continue
                        avg1 = sum(prices1) / len(prices1)
                        avg2 = sum(prices2) / len(prices2)

                        if avg1 <= 0 or avg2 <= 0:
                            continue

                        diff_pct = abs(avg1 - avg2) / max(avg1, avg2)
                        if diff_pct < self.ARBITRAGE_MIN_PROFIT_PCT:
                            continue

                        buy_loc = loc1 if avg1 < avg2 else loc2
                        sell_loc = loc2 if avg1 < avg2 else loc1
                        buy_price = min(avg1, avg2)
                        sell_price = max(avg1, avg2)
                        profit_pct = (sell_price - buy_price) / buy_price

                        # Estimate volume based on listing count
                        volume = min(
                            len([h for h in history if h["location"] == buy_loc]),
                            len([h for h in history if h["location"] == sell_loc]),
                        ) * 10  # rough multiplier

                        opportunities.append(ArbitrageOpportunity(
                            item_name=item_name,
                            buy_location=buy_loc,
                            sell_location=sell_loc,
                            buy_price=int(buy_price),
                            sell_price=int(sell_price),
                            profit_per_unit=int(sell_price - buy_price),
                            profit_pct=profit_pct,
                            estimated_volume=max(1, volume),
                            confidence=min(1.0, diff_pct / 0.5),
                            expires_at=now + 3600,
                        ))

            opportunities.sort(key=lambda o: o.profit_pct, reverse=True)
            self._arbitrage_opportunities = opportunities[:20]
            self._stats["arbitrage_found"] = len(opportunities)
            return self._arbitrage_opportunities

    # ── Farming Recommendations ──

    def recommend_farming(self, bot_level: int, bot_class: str,
                          current_map: str = "", zeny: int = 0,
                          party_size: int = 1) -> list[dict[str, Any]]:
        """Recommend what to farm based on current market conditions.

        Returns ranked list of farming recommendations with value/hour estimates.
        """
        with self._lock:
            recommendations = []

            # Try farming selector first
            if self._farming_selector is not None:
                try:
                    targets = self._farming_selector.select_targets(
                        level=bot_level, job_class=bot_class,
                        zeny=zeny, party_size=party_size,
                    )
                    for t in targets[:10]:
                        # Calculate value/hour based on current market prices
                        item_values = []
                        for drop in getattr(t, 'drops', []) or []:
                            price_info = self.get_current_price(drop)
                            if price_info["price"] > 0:
                                item_values.append(price_info["price"])

                        avg_drop_value = sum(item_values) / len(item_values) if item_values else 0
                        kills_per_hour = self._estimate_kills_per_hour(
                            bot_level, getattr(t, 'monster_level', bot_level),
                            party_size,
                        )
                        zeny_per_hour = avg_drop_value * kills_per_hour * 0.5  # 50% drop rate estimate
                        exp_per_hour = getattr(t, 'exp_per_kill', 0) * kills_per_hour

                        recommendations.append({
                            "map": getattr(t, 'map_name', ''),
                            "monster": getattr(t, 'monster_name', ''),
                            "zeny_per_hour": zeny_per_hour,
                            "exp_per_hour": exp_per_hour,
                            "danger_score": getattr(t, 'danger_score', 0.5),
                            "confidence": getattr(t, 'confidence', 0.5),
                            "drops": getattr(t, 'drops', []),
                            "source": "farming_selector",
                        })
                except Exception as e:
                    logger.debug("farming_selector failed: %s", e)

            # Fallback: use price history to find valuable items
            if not recommendations:
                valuable_items = self._find_valuable_items(bot_level)
                for item, price_info in valuable_items[:5]:
                    # Find which monsters drop this item
                    monster_info = self._find_monster_for_drop(item)
                    if monster_info:
                        kills_per_hour = self._estimate_kills_per_hour(
                            bot_level, monster_info.get("level", bot_level),
                            party_size,
                        )
                        zeny_per_hour = price_info["price"] * kills_per_hour * 0.3  # 30% drop rate

                        recommendations.append({
                            "map": monster_info.get("map", ""),
                            "monster": monster_info.get("name", item),
                            "zeny_per_hour": zeny_per_hour,
                            "exp_per_hour": monster_info.get("exp", 0) * kills_per_hour,
                            "danger_score": monster_info.get("danger", 0.5),
                            "confidence": price_info["confidence"],
                            "drops": [item],
                            "source": "market_driven",
                        })

            recommendations.sort(key=lambda r: r["zeny_per_hour"], reverse=True)
            self._stats["farming_recommendations"] += len(recommendations)
            return recommendations[:10]

    def _estimate_kills_per_hour(self, bot_level: int, monster_level: int,
                                  party_size: int) -> float:
        """Estimate kills per hour based on level difference and party size."""
        level_diff = bot_level - monster_level
        if level_diff >= 10:
            base_kph = 600  # Overleveled, fast kills
        elif level_diff >= 0:
            base_kph = 400  # On-level
        elif level_diff >= -10:
            base_kph = 250  # Slightly underleveled
        else:
            base_kph = 100  # Very underleveled

        # Party bonus (more party = more kills)
        party_mult = 1.0 + (party_size - 1) * 0.3
        return base_kph * party_mult

    def _find_valuable_items(self, bot_level: int) -> list[tuple[str, dict]]:
        """Find items that are valuable and farmable at this level."""
        valuable = []
        for item_name, history in self._price_history.items():
            if not history:
                continue
            recent = [h for h in history if time.time() - h["timestamp"] < 3600]
            if not recent:
                continue
            prices = [h["price"] for h in recent if h["price"] > 0]
            if not prices:
                continue
            avg_price = sum(prices) / len(prices)
            if avg_price > 1000:  # Minimum value threshold
                valuable.append((item_name, {
                    "price": avg_price,
                    "confidence": min(1.0, len(recent) / 10),
                }))
        valuable.sort(key=lambda x: x[1]["price"], reverse=True)
        return valuable

    def _find_monster_for_drop(self, item_name: str) -> dict[str, Any] | None:
        """Find which monster drops this item (from knowledge DB or item value DB)."""
        if self._item_value_db is not None:
            try:
                monsters = self._item_value_db.get_monsters_for_drop(item_name)
                if monsters:
                    return monsters[0]
            except Exception:
                pass
        return None

    # ── Value Per Hour ──

    def calculate_value_per_hour(self, activity: str, location: str = "",
                                 bot_level: int = 1, zeny: int = 0) -> ValuePerHour:
        """Calculate the value per hour of a given activity."""
        cache_key = f"{activity}:{location}:{bot_level}"
        cached = self._value_per_hour_cache.get(cache_key)
        if cached and time.time() - cached.last_assessed < 1800:  # 30 min cache
            return cached

        vph = ValuePerHour(activity=activity, location=location)

        if activity == "farm":
            # Use farming recommendations
            recs = self.recommend_farming(bot_level, "novice", location, zeny)
            if recs:
                best = recs[0]
                vph.zeny_per_hour = best["zeny_per_hour"]
                vph.exp_per_hour = best["exp_per_hour"]
                vph.item_value_per_hour = best["zeny_per_hour"] * 0.3
                vph.total_value_per_hour = vph.zeny_per_hour + vph.exp_per_hour * 0.01
                vph.confidence = best["confidence"]

        elif activity == "arbitrage":
            opps = self.find_arbitrage()
            if opps:
                best = opps[0]
                # Assume 10 trips per hour
                vph.zeny_per_hour = best.profit_per_unit * best.estimated_volume * 0.5
                vph.total_value_per_hour = vph.zeny_per_hour
                vph.confidence = best.confidence

        elif activity == "vend":
            # Vending income estimate
            vph.zeny_per_hour = zeny * 0.05 if zeny > 0 else 5000
            vph.total_value_per_hour = vph.zeny_per_hour
            vph.confidence = 0.3

        # Calculate opportunity cost
        if self._opportunity_cost is not None:
            try:
                vph.opportunity_cost = self._opportunity_cost.calculate(
                    current_activity=activity,
                    current_value=vph.total_value_per_hour,
                )
            except Exception:
                vph.opportunity_cost = vph.total_value_per_hour * 0.5

        vph.net_value = vph.total_value_per_hour - vph.opportunity_cost

        self._value_per_hour_cache[cache_key] = vph
        return vph

    # ── Market Manipulation ──

    def assess_manipulation(self, item_name: str, available_capital: int
                            ) -> MarketManipulationPlan | None:
        """Assess whether we can manipulate the market for an item."""
        market = self.get_current_price(item_name)
        if market["price"] <= 0 or market["confidence"] < 0.3:
            return None

        # Estimate supply based on listing count
        supply_estimate = market["listings"] * 10
        if supply_estimate <= 0:
            return None

        cost_to_buy = int(market["price"] * supply_estimate)
        if cost_to_buy > available_capital:
            return None

        # Can we resell at a premium?
        premium = 1.5  # 50% markup
        resell_price = int(market["price"] * premium)
        estimated_profit = (resell_price - market["price"]) * supply_estimate

        # Risk assessment
        risk = 0.5
        if market["trend"] == "falling":
            risk += 0.2
        if market["confidence"] < 0.5:
            risk += 0.2
        if supply_estimate > 100:
            risk += 0.1  # Too much supply to corner

        plan = MarketManipulationPlan(
            item_name=item_name,
            target_price=resell_price,
            current_price=int(market["price"]),
            supply_to_buy=supply_estimate,
            estimated_cost=cost_to_buy,
            estimated_profit=estimated_profit,
            risk_score=min(1.0, risk),
            time_horizon_hours=24,
        )

        with self._lock:
            self._manipulation_plans.append(plan)
            self._stats["manipulations_attempted"] += 1

        return plan

    # ── Merchant Empire ──

    def register_merchant(self, character_name: str, role: str) -> None:
        """Register a character in the merchant empire."""
        with self._lock:
            if character_name not in self._merchant_empire:
                self._merchant_empire[character_name] = MerchantEmpire(
                    character_name=character_name,
                    role=role,
                )
                logger.info("merchant_registered: %s as %s", character_name, role)

    def record_trade(self, character_name: str, profit: int) -> None:
        """Record a trade profit for a merchant character."""
        with self._lock:
            merchant = self._merchant_empire.get(character_name)
            if merchant:
                merchant.total_profit_today += profit
                merchant.last_profit_recorded = time.time()
                self._stats["merchant_trades"] += 1

    def get_empire_summary(self) -> str:
        """Get a summary of the merchant empire."""
        with self._lock:
            lines = ["── Merchant Empire ──"]
            total_profit = 0
            for name, m in self._merchant_empire.items():
                lines.append(f"  {name} ({m.role}): {m.zeny}z, "
                             f"inventory={m.inventory_value}z, "
                             f"profit_today={m.total_profit_today}z")
                total_profit += m.total_profit_today
            lines.append(f"Total profit today: {total_profit}z")
            lines.append(f"Active arbitrage: {len(self._arbitrage_opportunities)}")
            lines.append(f"Active manipulations: "
                         f"{len([p for p in self._manipulation_plans if p.status == 'active'])}")
            return "\n".join(lines)

    # ── WOE-Aware Pricing ──

    def is_woe_time(self) -> bool:
        """Check if it's WOE time."""
        now = time.localtime()
        return now.tm_hour in self.WOE_HOURS

    def get_woe_price_multiplier(self, item_name: str) -> float:
        """Get price multiplier during WOE for consumables."""
        if not self.is_woe_time():
            return 1.0

        item_lower = item_name.lower()
        if any(kw in item_lower for kw in ["berry", "yggdrasil", "seed", "flower"]):
            return 2.0  # Yggdrasil items double in price
        elif any(kw in item_lower for kw in ["white_potion", "blue_potion",
                                               "potion", "concentration"]):
            return 1.5
        elif any(kw in item_lower for kw in ["elunium", "oridecon"]):
            return 1.3
        return 1.0

    # ── P2P Integration ──

    def ingest_p2p_price(self, item_name: str, price: int, trend: str) -> None:
        """Ingest a price observation from the P2P network."""
        self.record_price(item_name, price, source="p2p")

    # ── Cycle Tick ──

    def tick(self) -> dict[str, Any]:
        """Called every PDCA cycle to update market state."""
        now = time.time()
        result = {
            "arbitrage_found": 0,
            "farming_recommendations": 0,
            "manipulations_active": 0,
            "merchants_active": 0,
            "woe_active": self.is_woe_time(),
        }

        # Find arbitrage opportunities
        opps = self.find_arbitrage()
        result["arbitrage_found"] = len(opps)

        # Clean up stale data every 10 minutes
        if now - self._last_cleanup > 600:
            self._cleanup()
            self._last_cleanup = now

        # Count active merchants
        with self._lock:
            result["merchants_active"] = len(self._merchant_empire)
            result["manipulations_active"] = len(
                [p for p in self._manipulation_plans if p.status == "active"]
            )

        return result

    def _cleanup(self) -> None:
        """Remove stale data."""
        with self._lock:
            now = time.time()
            # Clean stale arbitrage opportunities
            self._arbitrage_opportunities = [
                o for o in self._arbitrage_opportunities
                if o.expires_at > now
            ]
            # Clean failed manipulation plans
            self._manipulation_plans = [
                p for p in self._manipulation_plans
                if p.status != "failed" or now - getattr(p, 'last_updated', 0) < 86400
            ]

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def get_context(self) -> str:
        """Get formatted market context for LLM prompts."""
        with self._lock:
            lines = ["── Market Intelligence ──"]
            if self.is_woe_time():
                lines.append("  ⚠ WOE ACTIVE — consumable prices spiking!")

            # Top arbitrage opportunities
            if self._arbitrage_opportunities:
                lines.append("  Top arbitrage:")
                for opp in self._arbitrage_opportunities[:3]:
                    lines.append(f"    {opp.item_name}: buy {opp.buy_location} "
                                 f"({opp.buy_price}z) → sell {opp.sell_location} "
                                 f"({opp.sell_price}z) = {opp.profit_pct:.0%} profit")

            # Active manipulations
            active = [p for p in self._manipulation_plans if p.status == "active"]
            if active:
                lines.append("  Active manipulations:")
                for p in active[:2]:
                    lines.append(f"    {p.item_name}: target {p.target_price}z "
                                 f"(cost {p.estimated_cost}z, profit {p.estimated_profit}z)")

            # Merchant empire
            if self._merchant_empire:
                total_profit = sum(m.total_profit_today for m in self._merchant_empire.values())
                lines.append(f"  Merchant empire: {len(self._merchant_empire)} chars, "
                             f"{total_profit}z profit today")

            return "\n".join(lines)


# ── Global Singleton ──

_market_engine: MarketEngine | None = None
_market_engine_lock = RLock()


def get_market_engine() -> MarketEngine:
    global _market_engine
    with _market_engine_lock:
        if _market_engine is None:
            _market_engine = MarketEngine()
        return _market_engine
