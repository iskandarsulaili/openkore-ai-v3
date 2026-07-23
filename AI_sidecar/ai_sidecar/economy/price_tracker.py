"""
Real-time price tracking and economic intuition system.

Tracks item prices observed from NPC shops, player shops, and market data.
Provides price recommendations: buy low, sell high, hoard for spikes.
Detects market trends: WoE season spikes, bot farming crashes, new content rushes.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import math
import time
from ai_sidecar.economy.market_calendar import MarketCalendar, get_market_calendar


class PriceSource(Enum):
    NPC_SHOP = "npc_shop"           # Fixed NPC price
    PLAYER_SHOP = "player_shop"     # Player vendor (dynamic)
    MARKET_DATA = "market_data"     # External API data
    ESTIMATED = "estimated"         # Estimated from drop rate + demand


class TrendDirection(Enum):
    RISING = "rising"               # Price increasing
    FALLING = "falling"             # Price decreasing
    STABLE = "stable"               # Minimal change
    SPIKE = "spike"                 # Sudden increase (WoE season, patch)
    CRASH = "crash"                 # Sudden decrease (oversupply)


@dataclass
class PriceObservation:
    """A single price observation."""
    item_name: str
    price: int
    source: PriceSource
    timestamp: float
    quantity: int = 0
    location: str = ""
    seller: str = ""


@dataclass
class ItemPriceProfile:
    """Price history and recommendations for a single item."""
    item_name: str
    npc_buy_price: int = 0          # NPC buy-back price (floor)
    npc_sell_price: int = 0         # NPC selling price (ceiling)
    
    # Observed prices
    observations: list[PriceObservation] = field(default_factory=list)
    last_updated: float = 0.0
    
    # Computed stats
    min_observed_price: int = 0
    max_observed_price: int = 0
    avg_observed_price: int = 0
    observation_count: int = 0
    
    # Trend
    trend: TrendDirection = TrendDirection.STABLE
    trend_confidence: float = 0.0    # 0.0-1.0
    
    # Recommendations
    should_hoard: bool = False       # Store for future price spike
    should_sell_now: bool = True     # Sell immediately (no hoard value)
    hoard_until_event: str = ""      # "woe", "patch", "event"
    estimated_spike_multiplier: float = 1.0
    
    # Demand factors
    in_high_demand: bool = False
    demand_reason: str = ""
    
    # Vendor margins
    flip_profit_pct: float = 0.0    # NPC buy to player sell margin
    flip_profit_per_unit: int = 0
    
    # Usage tracking
    used_by_self: bool = False       # Bot uses this item (potions, ammo)
    daily_consumption: int = 0       # How many bot uses per day
    stockpile_target: int = 0        # How many to keep in inventory
    current_stock: int = 0
    restock_threshold: int = 0       # Re-buy when stock below this
    restock_amount: int = 0          # How many to buy when low


class PriceTracker:
    """Tracks item prices and provides economic recommendations."""
    
    def __init__(self, market_calendar: MarketCalendar | None = None):
        self.profiles: dict[str, ItemPriceProfile] = {}
        self._last_trend_calc: float = 0.0
        self._market_calendar: MarketCalendar = market_calendar or get_market_calendar()
        
        # Known NPC prices (pre-populated from rAthena data)
        self._init_npc_prices()
        
        # Event seasons and their price multipliers
        self._event_seasons: dict[str, dict] = {
            "woe": {
                "items": ["Blue Potion", "White Potion", "Convex Mirror", "Panacea",
                          "Holy Water", "Fire Arrow", "Silver Arrow", "Crystal Arrow",
                          "Trap", "Grenade", "Bomb"],
                "price_multiplier": 3.0,
                "duration_hours": 48,
                "predict_next": ["Saturday 20:00", "Wednesday 20:00"],
            },
            "patch_new_content": {
                "items": ["Convex Mirror", "Old Blue Box", "Old Violet Box",
                          "Elunium", "Oridecon"],
                "price_multiplier": 2.0,
                "duration_hours": 72,
            },
            "bot_crash": {
                "items": ["Stem", "Memento", "Iggdrasil Berry", "Honey",
                          "Flower", "Feather", "Bacsojin"],
                "price_multiplier": 0.3,  # Prices crash when bots flood market
            },
        }

        self.profiles: dict[str, ItemPriceProfile] = {}
        self._last_trend_calc: float = 0.0
        
        # Known NPC prices (pre-populated from rAthena data)
        self._init_npc_prices()
        
        # Event seasons and their price multipliers
        self._event_seasons: dict[str, dict] = {
            "woe": {
                "items": ["Blue Potion", "White Potion", "Convex Mirror", "Panacea",
                          "Holy Water", "Fire Arrow", "Silver Arrow", "Crystal Arrow",
                          "Trap", "Grenade", "Bomb"],
                "price_multiplier": 3.0,
                "duration_hours": 48,
                "predict_next": ["Saturday 20:00", "Wednesday 20:00"],
            },
            "patch_new_content": {
                "items": ["Convex Mirror", "Old Blue Box", "Old Violet Box",
                          "Elunium", "Oridecon"],
                "price_multiplier": 2.0,
                "duration_hours": 72,
            },
            "bot_crash": {
                "items": ["Stem", "Memento", "Iggdrasil Berry", "Honey",
                          "Flower", "Feather", "Bacsojin"],
                "price_multiplier": 0.3,  # Prices crash when bots flood market
            },
        }
    
    def _init_npc_prices(self):
        """Initialize known NPC shop prices."""
        base_prices = {
            "White Potion": {"npc_buy": 200, "npc_sell": 500, "flip_profit": 150.0},
            "Blue Potion": {"npc_buy": 600, "npc_sell": 1500, "flip_profit": 150.0},
            "Convex Mirror": {"npc_buy": 500, "npc_sell": 1000, "flip_profit": 100.0},
            "Panacea": {"npc_buy": 500, "npc_sell": 1000, "flip_profit": 100.0},
            "Holy Water": {"npc_buy": 100, "npc_sell": 500, "flip_profit": 400.0},
            "Fly Wing": {"npc_buy": 500, "npc_sell": 1000, "flip_profit": 100.0},
            "Butterfly Wing": {"npc_buy": 2000, "npc_sell": 3000, "flip_profit": 50.0},
            "Empty Bottle": {"npc_buy": 50, "npc_sell": 100, "flip_profit": 100.0},
            "Stem": {"npc_buy": 0, "npc_sell": 300, "flip_profit": 0.0},
            "Memento": {"npc_buy": 0, "npc_sell": 100, "flip_profit": 0.0},
            "Honey": {"npc_buy": 0, "npc_sell": 1000, "flip_profit": 0.0},
            "Iggdrasil Berry": {"npc_buy": 0, "npc_sell": 3000, "flip_profit": 0.0},
            "Feather": {"npc_buy": 0, "npc_sell": 50, "flip_profit": 0.0},
            "Elunium": {"npc_buy": 0, "npc_sell": 10000, "flip_profit": 0.0},
            "Oridecon": {"npc_buy": 0, "npc_sell": 15000, "flip_profit": 0.0},
            "Fire Arrow": {"npc_buy": 2, "npc_sell": 10, "flip_profit": 400.0},
            "Silver Arrow": {"npc_buy": 2, "npc_sell": 15, "flip_profit": 650.0},
            "Crystal Arrow": {"npc_buy": 2, "npc_sell": 20, "flip_profit": 900.0},
        }
        for name, data in base_prices.items():
            self.profiles[name.lower()] = ItemPriceProfile(
                item_name=name,
                npc_buy_price=data["npc_buy"],
                npc_sell_price=data["npc_sell"],
                flip_profit_pct=data["flip_profit"],
                min_observed_price=data["npc_buy"],
                max_observed_price=data["npc_sell"],
                avg_observed_price=data["npc_buy"],
                should_sell_now=True,
                should_hoard=False,
            )
    
    def record_observation(self, item_name: str, price: int, source: PriceSource,
                          quantity: int = 0, location: str = "", seller: str = ""):
        """Record a price observation."""
        key = item_name.lower()
        now = time.time()
        
        obs = PriceObservation(
            item_name=item_name, price=price, source=source,
            timestamp=now, quantity=quantity, location=location, seller=seller
        )
        
        if key not in self.profiles:
            self.profiles[key] = ItemPriceProfile(item_name=item_name)
        
        profile = self.profiles[key]
        profile.observations.append(obs)
        profile.last_updated = now
        profile.observation_count += 1
        
        # Update min/max
        if profile.min_observed_price == 0 or price < profile.min_observed_price:
            profile.min_observed_price = price
        if price > profile.max_observed_price:
            profile.max_observed_price = price
        
        # Running average
        total = profile.avg_observed_price * (profile.observation_count - 1) + price
        profile.avg_observed_price = total // profile.observation_count
        
        # Recalculate trend every 60 observations
        if profile.observation_count % 60 == 0:
            self._recalc_trend(profile)
    
    def _recalc_trend(self, profile: ItemPriceProfile):
        """Recalculate price trend for a profile."""
        if len(profile.observations) < 10:
            return
        
        recent = profile.observations[-10:]
        oldest = profile.observations[:10]
        
        recent_avg = sum(o.price for o in recent) / len(recent)
        oldest_avg = sum(o.price for o in oldest) / len(oldest)
        
        if oldest_avg == 0:
            return
        
        change_pct = (recent_avg - oldest_avg) / oldest_avg
        
        if change_pct > 0.30:
            profile.trend = TrendDirection.SPIKE
            profile.trend_confidence = min(abs(change_pct), 1.0)
        elif change_pct > 0.10:
            profile.trend = TrendDirection.RISING
            profile.trend_confidence = min(abs(change_pct), 0.8)
        elif change_pct < -0.30:
            profile.trend = TrendDirection.CRASH
            profile.trend_confidence = min(abs(change_pct), 1.0)
        elif change_pct < -0.10:
            profile.trend = TrendDirection.FALLING
            profile.trend_confidence = min(abs(change_pct), 0.8)
        else:
            profile.trend = TrendDirection.STABLE
            profile.trend_confidence = 0.5
    
    def get_sell_recommendation(self, item_name: str, current_price: int | None = None) -> dict:
        """Should we sell this item now, or hoard?
        
        Incorporates MarketCalendar timing: day-of-week, hour-of-day,
        and active event multipliers for timing-aware recommendations.
        """
        key = item_name.lower()
        profile = self.profiles.get(key)
        if not profile:
            return {"action": "sell", "reason": "unknown item, sell automatically"}
        
        # Get timing-aware multiplier from MarketCalendar
        timing_mul = self._market_calendar.get_combined_multiplier(item_name)
        
        # If price is above NPC sell, consider player shop
        if current_price and current_price > profile.npc_sell_price:
            multiplier = current_price / max(profile.npc_sell_price, 1)
            # Adjust threshold by timing — sell earlier during peak times
            adjusted_threshold = 2.0 / max(timing_mul, 0.5)
            if multiplier > adjusted_threshold and profile.trend in (TrendDirection.SPIKE, TrendDirection.RISING):
                return {
                    "action": "sell_now",
                    "reason": f"price {multiplier:.1f}x NPC (timing adj {timing_mul:.2f}x), sell during spike",
                    "timing_multiplier": timing_mul,
                }
            if multiplier > 1.5:
                return {
                    "action": "sell_player_shop",
                    "reason": f"price {multiplier:.1f}x NPC, player shop (timing {timing_mul:.2f}x)",
                    "timing_multiplier": timing_mul,
                }
        
        # Check event seasons (existing logic, now augmented with calendar)
        for event_name, event_data in self._event_seasons.items():
            if item_name in event_data["items"]:
                if profile.trend == TrendDirection.SPIKE:
                    return {
                        "action": "sell_now",
                        "reason": f"{event_name} spike, sell immediately (timing {timing_mul:.2f}x)",
                        "timing_multiplier": timing_mul,
                    }
                if profile.trend in (TrendDirection.RISING, TrendDirection.STABLE):
                    return {
                        "action": "hoard",
                        "reason": f"{event_name} expected, hoard for {event_data['price_multiplier']}x multiplier",
                        "timing_multiplier": timing_mul,
                    }
        
        # Timing-aware default: recommend hoarding if timing is favorable
        if timing_mul > 1.3:
            return {
                "action": "hoard",
                "reason": f"timing favorable ({timing_mul:.2f}x), wait for better price",
                "timing_multiplier": timing_mul,
            }
        
        # Default: sell at NPC price
        return {
            "action": "sell_npc",
            "reason": f"NPC price {profile.npc_sell_price}z (timing {timing_mul:.2f}x)",
            "timing_multiplier": timing_mul,
        }

        """Should we sell this item now, or hoard?"""
        key = item_name.lower()
        profile = self.profiles.get(key)
        if not profile:
            return {"action": "sell", "reason": "unknown item, sell automatically"}
        
        # If price is above NPC sell, consider player shop
        if current_price and current_price > profile.npc_sell_price:
            multiplier = current_price / max(profile.npc_sell_price, 1)
            if multiplier > 2.0 and profile.trend in (TrendDirection.SPIKE, TrendDirection.RISING):
                return {"action": "sell_now", "reason": f"price {multiplier:.1f}x NPC, sell during spike"}
            if multiplier > 1.5:
                return {"action": "sell_player_shop", "reason": f"price {multiplier:.1f}x NPC, player shop"}
        
        # Check event seasons
        for event_name, event_data in self._event_seasons.items():
            if item_name in event_data["items"]:
                if profile.trend == TrendDirection.SPIKE:
                    return {"action": "sell_now", "reason": f"{event_name} spike, sell immediately"}
                if profile.trend in (TrendDirection.RISING, TrendDirection.STABLE):
                    return {"action": "hoard", "reason": f"{event_name} expected, hoard for {event_data['price_multiplier']}x multiplier"}
        
        # Default: sell at NPC price
        return {"action": "sell_npc", "reason": f"NPC price {profile.npc_sell_price}z"}
    
    def get_buy_recommendation(self, item_name: str) -> dict:
        """Should we buy this item now, or wait?
        
        Incorporates MarketCalendar timing: buy during off-peak (low day factor × low hour factor)
        for best prices, wait during peak demand.
        """
        key = item_name.lower()
        profile = self.profiles.get(key)
        if not profile:
            return {"action": "buy", "reason": "item needed, buy now"}
        
        # Get timing-aware multiplier from MarketCalendar
        timing_mul = self._market_calendar.get_combined_multiplier(item_name)
        
        if profile.trend == TrendDirection.SPIKE:
            return {
                "action": "wait",
                "reason": f"price spiking (trend={profile.trend.value}), wait for crash",
                "timing_multiplier": timing_mul,
            }
        
        if profile.trend == TrendDirection.CRASH:
            return {
                "action": "buy",
                "reason": f"price crashing, buy at bottom (trend={profile.trend.value})",
                "timing_multiplier": timing_mul,
            }
        
        # Timing-aware: if timing is unfavorable (high multiplier), wait
        if timing_mul > 1.2 and profile.trend != TrendDirection.FALLING:
            return {
                "action": "wait",
                "reason": f"timing unfavorable ({timing_mul:.2f}x), wait for off-peak",
                "timing_multiplier": timing_mul,
            }
        
        if profile.current_stock < profile.restock_threshold:
            return {
                "action": "buy",
                "reason": f"stock low ({profile.current_stock}/{profile.restock_threshold}), re{profile.restock_amount}",
                "timing_multiplier": timing_mul,
            }
        
        return {
            "action": "wait",
            "reason": "sufficient stock, no rush",
            "timing_multiplier": timing_mul,
        }

        """Should we buy this item now, or wait?"""
        key = item_name.lower()
        profile = self.profiles.get(key)
        if not profile:
            return {"action": "buy", "reason": "item needed, buy now"}
        
        if profile.trend == TrendDirection.SPIKE:
            return {"action": "wait", "reason": f"price spiking (trend={profile.trend.value}), wait for crash"}
        
        if profile.trend == TrendDirection.CRASH:
            return {"action": "buy", "reason": f"price crashing, buy at bottom (trend={profile.trend.value})"}
        
        if profile.current_stock < profile.restock_threshold:
            return {"action": "buy", "reason": f"stock low ({profile.current_stock}/{profile.restock_threshold}), re{profile.restock_amount}"}
        
        return {"action": "wait", "reason": "sufficient stock, no rush"}
    
    def detect_economic_opportunity(self) -> list[dict]:
        """Find arbitrage opportunities (buy low at NPC, sell high at player shop)."""
        opportunities = []
        for profile in self.profiles.values():
            if profile.flip_profit_pct > 100 and profile.observation_count > 0:
                opportunities.append({
                    "item": profile.item_name,
                    "buy_price": profile.npc_sell_price,
                    "sell_price": profile.max_observed_price,
                    "profit_per_unit": profile.max_observed_price - profile.npc_sell_price,
                    "profit_pct": profile.flip_profit_pct,
                    "confidence": min(profile.observation_count / 100, 0.9),
                })
        
        # Sort by profit per unit, descending
        opportunities.sort(key=lambda x: x["profit_per_unit"], reverse=True)
        return opportunities[:10]
    
    def set_usage_profile(self, item_name: str, consumed_daily: int = 0,
                          stockpile_target: int = 0, restock_threshold: int = 0,
                          restock_amount: int = 0):
        """Set bot's usage profile for a consumable item."""
        key = item_name.lower()
        if key not in self.profiles:
            self.profiles[key] = ItemPriceProfile(item_name=item_name)
        
        profile = self.profiles[key]
        profile.used_by_self = True
        profile.daily_consumption = consumed_daily
        profile.stockpile_target = stockpile_target
        profile.restock_threshold = restock_threshold
        profile.restock_amount = restock_amount


# ── Singleton instance ──
_price_tracker: PriceTracker | None = None


def get_price_tracker() -> PriceTracker:
    """Get the global price tracker instance."""
    global _price_tracker
    if _price_tracker is None:
        _price_tracker = PriceTracker()
    return _price_tracker
