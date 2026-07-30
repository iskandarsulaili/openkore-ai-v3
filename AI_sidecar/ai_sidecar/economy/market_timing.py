"""Market Timing — price fluctuation tracking with seasonal, daily, and event-driven patterns.

Pro players know exactly WHEN prices peak. This module tracks:
  - WoE price surges (potions +50%, converters +200% before WoE)
  - Post-maintenance price dips (NPC overflow items)
  - Card value decay on old servers vs new servers
  - Day-of-week and hour-of-day price patterns
  - Special event pricing (Christmas, Valentine's, MVP card drops)

All prices in zeny. Thread-safe.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


# ── Enums ──────────────────────────────────────────────────────────────────

class ServerAge(str, Enum):
    """Server age classification affects all card and rare item values."""
    NEW = "new"          # 0-3 months: card prices at peak
    YOUNG = "young"      # 3-12 months: still inflated
    MATURE = "mature"    # 1-2 years: stabilized
    OLD = "old"          # 2+ years: card prices at floor


class DayPhase(str, Enum):
    """Phases of the RO week that affect market prices."""
    NORMAL = "normal"
    PRE_WOE = "pre_woe"      # 2 hours before WoE
    WOE = "woe"              # War of Emporium active
    POST_WOE = "post_woe"    # 2 hours after WoE
    POST_MAINT = "post_maint"  # After server maintenance
    EVENT = "event"          # Special event active


# ── Price Pattern Data ─────────────────────────────────────────────────────

@dataclass
class PriceMultiplier:
    """A price multiplier with time window and certainty."""
    multiplier: float  # 1.0 = normal, 2.0 = double, 0.5 = half
    confidence: float  # 0.0-1.0 how reliable this pattern is
    reason: str        # Human-readable explanation


@dataclass
class ItemPricePattern:
    """Known price pattern for an item category or specific item."""
    item_name: str | None          # None if category-wide pattern
    category: str                   # Card, Potion, Elemental Converter, etc.
    item_glob: str | None          # Partial name match pattern
    base_price: int                # Normal market price
    woe_multiplier: float          # Price multiplier during WoE
    pre_woe_multiplier: float      # Price multiplier 2h before WoE
    post_maint_multiplier: float   # Price multiplier right after maintenance
    event_multiplier: float        # During special events
    server_age_decay: float        # How much value decays per month on old servers
    weekly_pattern: dict[int, float] = field(default_factory=dict)  # day_of_week -> multiplier
    hourly_pattern: dict[int, float] = field(default_factory=dict)  # hour_of_day -> multiplier


# ── Known Price Patterns (Real RO Economy Data) ────────────────────────────

# These are based on actual RO economy behavior across multiple servers.
# Prices represent mid-population server estimates circa 2023-2025.

KNOWN_PRICE_PATTERNS: list[ItemPricePattern] = [
    # ── Potions ──
    ItemPricePattern(
        item_name=None, category="Potion", item_glob="*Potion",
        base_price=500, woe_multiplier=2.0, pre_woe_multiplier=1.5,
        post_maint_multiplier=0.5, event_multiplier=1.3,
        server_age_decay=0.0,  # Potions don't decay — always needed
        weekly_pattern={5: 1.4, 6: 2.0, 0: 1.3},  # Fri+Sat+Sun peak
        hourly_pattern={20: 2.0, 21: 2.2, 22: 2.0, 18: 1.5, 19: 1.8},
    ),
    # ── White Potions ──
    ItemPricePattern(
        item_name="White Potion", category="Healing", item_glob=None,
        base_price=1000, woe_multiplier=2.0, pre_woe_multiplier=1.5,
        post_maint_multiplier=0.6, event_multiplier=1.2,
        server_age_decay=0.0,
        weekly_pattern={5: 1.3, 6: 2.0, 0: 1.5},
        hourly_pattern={20: 2.0, 21: 2.5, 22: 2.0},
    ),
    # ── Elemental Converters ──
    ItemPricePattern(
        item_name=None, category="Elemental Converter", item_glob="*Converter",
        base_price=15000, woe_multiplier=1.5, pre_woe_multiplier=3.0,
        post_maint_multiplier=0.4, event_multiplier=1.5,
        server_age_decay=0.0,
        weekly_pattern={5: 2.5, 6: 3.0},  # Heavy pre-WoE demand
        hourly_pattern={17: 2.5, 18: 3.0, 19: 2.8, 20: 2.0},
    ),
    # ── Cards ──
    ItemPricePattern(
        item_name=None, category="Card", item_glob="*Card",
        base_price=100000, woe_multiplier=1.2, pre_woe_multiplier=1.1,
        post_maint_multiplier=0.8, event_multiplier=1.0,
        server_age_decay=0.08,  # 8% value loss per month on old servers
        weekly_pattern={},
        hourly_pattern={},
    ),
    # ── Poring Card ──
    ItemPricePattern(
        item_name="Poring Card", category="Card", item_glob=None,
        base_price=500000, woe_multiplier=1.1, pre_woe_multiplier=1.0,
        post_maint_multiplier=0.9, event_multiplier=1.0,
        server_age_decay=0.10,  # 10% loss per month (old server = 50K)
        weekly_pattern={},
        hourly_pattern={},
    ),
    # ── Savage Babe Card ──
    ItemPricePattern(
        item_name="Savage Babe Card", category="Card", item_glob=None,
        base_price=200000, woe_multiplier=1.1, pre_woe_multiplier=1.0,
        post_maint_multiplier=0.9, event_multiplier=1.0,
        server_age_decay=0.08,
        weekly_pattern={},
        hourly_pattern={},
    ),
    # ── MVP Cards ──
    ItemPricePattern(
        item_name=None, category="Card", item_glob="*MVP*Card",
        base_price=5000000, woe_multiplier=1.3, pre_woe_multiplier=1.2,
        post_maint_multiplier=0.5, event_multiplier=1.0,
        server_age_decay=0.12,  # 12% per month on old servers
        weekly_pattern={6: 1.2},
        hourly_pattern={},
    ),
    # ── Empty Bottles ──
    ItemPricePattern(
        item_name="Empty Bottle", category="Usable", item_glob=None,
        base_price=2000, woe_multiplier=1.5, pre_woe_multiplier=1.3,
        post_maint_multiplier=0.5, event_multiplier=1.0,
        server_age_decay=0.0,
        weekly_pattern={5: 1.2, 6: 1.5, 0: 1.2},
        hourly_pattern={19: 1.3, 20: 1.5, 21: 1.4},
    ),
    # ── White Herbs ──
    ItemPricePattern(
        item_name="White Herb", category="Healing", item_glob=None,
        base_price=200, woe_multiplier=2.5, pre_woe_multiplier=1.8,
        post_maint_multiplier=0.4, event_multiplier=1.3,
        server_age_decay=0.0,
        weekly_pattern={5: 1.5, 6: 2.5, 0: 2.0},
        hourly_pattern={19: 2.0, 20: 2.5, 21: 2.2},
    ),
]

# Server age -> card value multiplier
SERVER_AGE_CARD_MULTIPLIERS: dict[ServerAge, float] = {
    ServerAge.NEW: 1.5,     # New server: cards 1.5x base
    ServerAge.YOUNG: 1.2,   # Young server: cards 1.2x base
    ServerAge.MATURE: 1.0,  # Mature server: 1.0x (baseline)
    ServerAge.OLD: 0.5,     # Old server: cards 0.5x base
}

# Server age -> material value multiplier
SERVER_AGE_MATERIAL_MULTIPLIERS: dict[ServerAge, float] = {
    ServerAge.NEW: 2.0,     # New server: materials scarce, 2x price
    ServerAge.YOUNG: 1.5,   # Young server: still inflated
    ServerAge.MATURE: 1.0,  # Mature: baseline
    ServerAge.OLD: 0.8,     # Old: materials abundant
}


# ── Market Timing Engine ──────────────────────────────────────────────────

class MarketTimingEngine:
    """Tracks price fluctuations across time patterns for RO economy.

    Provides current price multipliers for any item based on:
      - WoE schedule (Saturday/Sunday 20:00-22:00)
      - Day of week and hour of day
      - Server age (card value decay)
      - Known special events
      - Post-maintenance price dips

    Thread-safe.
    """

    def __init__(self, server_age: ServerAge = ServerAge.MATURE) -> None:
        self._lock = RLock()
        self._server_age = server_age
        self._server_age_seconds: float = 0.0  # Seconds since server started
        self._active_events: set[str] = set()
        self._custom_price_overrides: dict[str, int] = {}
        self._last_maintenance_time: float = 0.0
        self._stats: dict[str, int | float] = {
            "multiplier_queries": 0,
            "pattern_matches": 0,
        }

        # WoE schedule: Saturday and Sunday 20:00-22:00 server time
        self._woe_days = {5, 6}  # Saturday=5, Sunday=6
        self._woe_start_hour = 20
        self._woe_end_hour = 22

    # ── Public API ─────────────────────────────────────────────────────

    def get_current_price_multiplier(
        self,
        item_name: str | None = None,
        category: str | None = None,
    ) -> PriceMultiplier:
        """Get the current price multiplier for an item based on temporal patterns.

        Args:
            item_name: Specific item name (e.g., "White Potion").
            category: Item category (e.g., "Card", "Potion").

        Returns:
            PriceMultiplier with the current multiplier and explanation.
        """
        with self._lock:
            self._stats["multiplier_queries"] += 1  # type: ignore[assignment]
            now = datetime.now(timezone.utc)
            weekday = now.weekday()  # 0=Monday, 6=Sunday
            hour = now.hour

            # Find best matching pattern
            pattern = self._find_best_pattern(item_name, category)
            if pattern is None:
                return PriceMultiplier(1.0, 0.5, "No pattern known for this item")

            multiplier = 1.0
            reasons: list[str] = []

            # 1. Day-of-week pattern
            if pattern.weekly_pattern and weekday in pattern.weekly_pattern:
                m = pattern.weekly_pattern[weekday]
                if m != 1.0:
                    multiplier *= m
                    reasons.append(f"weekday={weekday}: x{m}")

            # 2. Hourly pattern
            if pattern.hourly_pattern and hour in pattern.hourly_pattern:
                m = pattern.hourly_pattern[hour]
                if m != 1.0:
                    multiplier *= m
                    reasons.append(f"hour={hour}: x{m}")

            # 3. Phase detection
            phase = self._determine_current_phase(weekday, hour)
            if phase == DayPhase.WOE:
                multiplier *= pattern.woe_multiplier
                reasons.append(f"WoE active: x{pattern.woe_multiplier}")
            elif phase == DayPhase.PRE_WOE:
                multiplier *= pattern.pre_woe_multiplier
                reasons.append(f"Pre-WoE: x{pattern.pre_woe_multiplier}")
            elif phase == DayPhase.POST_MAINT:
                multiplier *= pattern.post_maint_multiplier
                reasons.append(f"Post-maintenance: x{pattern.post_maint_multiplier}")
            elif phase == DayPhase.EVENT:
                multiplier *= pattern.event_multiplier
                reasons.append(f"Event active: x{pattern.event_multiplier}")

            # 4. Server age decay for cards
            if category == "Card" or (pattern.item_glob and "Card" in pattern.item_glob):
                age_mult = SERVER_AGE_CARD_MULTIPLIERS.get(self._server_age, 1.0)
                if age_mult != 1.0:
                    multiplier *= age_mult
                    reasons.append(f"Server age ({self._server_age.value}): x{age_mult}")

            # 5. Material pricing by server age
            if category in ("Material", "Usable", "Healing"):
                age_mult = SERVER_AGE_MATERIAL_MULTIPLIERS.get(self._server_age, 1.0)
                if age_mult != 1.0:
                    multiplier *= age_mult
                    reasons.append(f"Materials ({self._server_age.value}): x{age_mult}")

            confidence = 0.7 if len(reasons) > 0 else 0.3
            self._stats["pattern_matches"] += 1  # type: ignore[assignment]

            return PriceMultiplier(
                multiplier=round(multiplier, 4),
                confidence=round(confidence, 2),
                reason=" * ".join(reasons) if reasons else "No temporal adjustment",
            )

    def get_adjusted_market_price(self, item_name: str, base_market_price: int,
                                   category: str | None = None) -> int:
        """Get the time-adjusted market price for an item.

        Takes the base market price and applies current temporal multipliers.
        """
        mult = self.get_current_price_multiplier(item_name, category)
        adjusted = int(base_market_price * mult.multiplier)
        logger.debug(
            "market_timing: %s base=%d adjusted=%d (x%s: %s)",
            item_name, base_market_price, adjusted, mult.multiplier, mult.reason,
        )
        return max(1, adjusted)

    def get_server_age_adjustment(self, item_name: str, base_price: int,
                                   category: str) -> int:
        """Apply server age adjustment to item price.

        New servers: cards and materials are worth more.
        Old servers: cards are worth much less, materials slightly less.
        """
        with self._lock:
            if category == "Card":
                mult = SERVER_AGE_CARD_MULTIPLIERS.get(self._server_age, 1.0)
            elif category in ("Material", "Usable", "Healing"):
                mult = SERVER_AGE_MATERIAL_MULTIPLIERS.get(self._server_age, 1.0)
            else:
                mult = 1.0
            return int(base_price * mult)

    def is_woe_window(self) -> bool:
        """Check if WoE is currently active."""
        now = datetime.now(timezone.utc)
        weekday = now.weekday()
        hour = now.hour
        phase = self._determine_current_phase(weekday, hour)
        return phase == DayPhase.WOE

    def is_pre_woe_window(self) -> bool:
        """Check if we're in the pre-WoE preparation window."""
        now = datetime.now(timezone.utc)
        weekday = now.weekday()
        hour = now.hour
        phase = self._determine_current_phase(weekday, hour)
        return phase == DayPhase.PRE_WOE

    def get_seconds_to_next_woe(self) -> float:
        """Get seconds until the next WoE period starts.

        Returns 0 if WoE is currently active.
        """
        now = datetime.now(timezone.utc)
        weekday = now.weekday()
        hour = now.hour
        minute = now.minute

        phase = self._determine_current_phase(weekday, hour)
        if phase == DayPhase.WOE:
            return 0.0

        # Check each WoE day (Saturday, Sunday)
        for day_offset in range(7):
            check_day = (weekday + day_offset) % 7
            if check_day in self._woe_days:
                woe_start = now.replace(hour=self._woe_start_hour, minute=0, second=0, microsecond=0)
                woe_start += timedelta(days=day_offset)
                if day_offset == 0 and hour < self._woe_start_hour:
                    # Later today
                    delta = (woe_start - now).total_seconds()
                    return max(0, delta)
                elif day_offset > 0:
                    delta = (woe_start - now).total_seconds()
                    return max(0, delta)

        # Fallback: next Saturday
        days_ahead = (5 - weekday) % 7
        if days_ahead == 0:
            days_ahead = 7
        next_sat = now + timedelta(days=days_ahead)
        next_sat = next_sat.replace(hour=self._woe_start_hour, minute=0, second=0, microsecond=0)
        return (next_sat - now).total_seconds()

    def set_server_age(self, age: ServerAge) -> None:
        """Set the server age for price adjustment calculations."""
        with self._lock:
            self._server_age = age
            logger.info("market_timing: server age set to %s", age.value)

    def set_server_start_time(self, timestamp: float) -> None:
        """Set the server start timestamp for age calculations."""
        with self._lock:
            self._server_age_seconds = timestamp

    def set_custom_price(self, item_name: str, price: int) -> None:
        """Override the market price for a specific item."""
        with self._lock:
            self._custom_price_overrides[item_name] = price

    def register_event(self, event_name: str) -> None:
        """Register an active special event that affects prices."""
        with self._lock:
            self._active_events.add(event_name)
            logger.info("market_timing: event '%s' registered", event_name)

    def clear_event(self, event_name: str) -> None:
        """Clear a registered event."""
        with self._lock:
            self._active_events.discard(event_name)

    def record_maintenance(self) -> None:
        """Record that server maintenance just happened.

        Post-maintenance prices are volatile — many items dip for ~2 hours.
        """
        with self._lock:
            self._last_maintenance_time = time.time()
            logger.info("market_timing: maintenance recorded")

    def get_buy_low_windows(self, item_name: str, category: str | None = None) -> list[dict[str, Any]]:
        """Get the best times to buy an item (when prices are lowest).

        Returns list of (day_of_week, hour) tuples sorted by lowest multiplier.
        """
        with self._lock:
            pattern = self._find_best_pattern(item_name, category)
            if pattern is None:
                return []

            windows: list[dict[str, Any]] = []

            # Check all day/hour combinations for lowest multiplier
            for day in range(7):
                day_mult = pattern.weekly_pattern.get(day, 1.0)
                for hour in range(24):
                    hour_mult = pattern.hourly_pattern.get(hour, 1.0)
                    total_mult = day_mult * hour_mult

                    # Apply phase adjustment
                    phase = self._determine_current_phase(day, hour)
                    if phase == DayPhase.POST_MAINT:
                        total_mult *= pattern.post_maint_multiplier

                    if total_mult < 0.85:  # Only return significant discounts
                        windows.append({
                            "day": day,
                            "hour": hour,
                            "multiplier": round(total_mult, 3),
                            "effective_price": int(pattern.base_price * total_mult),
                            "reason": f"Day {day} hour {hour}: x{total_mult:.2f}",
                        })

            windows.sort(key=lambda w: w["multiplier"])
            return windows[:20]

    def get_sell_high_windows(self, item_name: str, category: str | None = None) -> list[dict[str, Any]]:
        """Get the best times to sell an item (when prices are highest).

        Returns list of (day_of_week, hour) tuples sorted by highest multiplier.
        """
        with self._lock:
            pattern = self._find_best_pattern(item_name, category)
            if pattern is None:
                return []

            windows: list[dict[str, Any]] = []

            for day in range(7):
                day_mult = pattern.weekly_pattern.get(day, 1.0)
                for hour in range(24):
                    hour_mult = pattern.hourly_pattern.get(hour, 1.0)
                    total_mult = day_mult * hour_mult

                    phase = self._determine_current_phase(day, hour)
                    if phase == DayPhase.PRE_WOE:
                        total_mult *= pattern.pre_woe_multiplier
                    elif phase == DayPhase.WOE:
                        total_mult *= pattern.woe_multiplier

                    if total_mult > 1.15:
                        windows.append({
                            "day": day,
                            "hour": hour,
                            "multiplier": round(total_mult, 3),
                            "effective_price": int(pattern.base_price * total_mult),
                            "reason": f"Day {day} hour {hour}: x{total_mult:.2f}",
                        })

            windows.sort(key=lambda w: -w["multiplier"])
            return windows[:20]

    def get_stats(self) -> dict[str, int | float]:
        with self._lock:
            return dict(self._stats)

    # ── Internal ───────────────────────────────────────────────────────

    def _find_best_pattern(
        self, item_name: str | None, category: str | None,
    ) -> ItemPricePattern | None:
        """Find the best matching price pattern for an item."""
        exact_match: ItemPricePattern | None = None
        category_match: ItemPricePattern | None = None
        glob_match: ItemPricePattern | None = None

        for pattern in KNOWN_PRICE_PATTERNS:
            # Exact name match
            if item_name and pattern.item_name and pattern.item_name.lower() == item_name.lower():
                exact_match = pattern
            # Category match (fallback)
            if category and pattern.category and pattern.category.lower() == category.lower():
                if pattern.item_name is None and pattern.item_glob is None:
                    category_match = pattern
            # Glob match
            if item_name and pattern.item_glob:
                glob_pattern = pattern.item_glob.replace("*", "").lower()
                if glob_pattern in item_name.lower():
                    glob_match = pattern

        return exact_match or glob_match or category_match

    def _determine_current_phase(self, weekday: int, hour: int) -> DayPhase:
        """Determine the current market phase."""
        now_ts = time.time()

        # Check post-maintenance window (2 hours after maintenance)
        if (self._last_maintenance_time > 0
                and now_ts - self._last_maintenance_time < 7200):
            return DayPhase.POST_MAINT

        # Check active events
        if self._active_events:
            return DayPhase.EVENT

        # Check WoE schedule (Saturday/Sunday)
        if weekday in self._woe_days:
            if self._woe_start_hour <= hour < self._woe_end_hour:
                return DayPhase.WOE
            if hour == self._woe_start_hour - 2:
                return DayPhase.PRE_WOE
            if hour == self._woe_start_hour - 1:
                return DayPhase.PRE_WOE

        return DayPhase.NORMAL


# ── Global Singleton ─────────────────────────────────────────────────────

_market_timing: MarketTimingEngine | None = None
_market_timing_lock = RLock()


def get_market_timing(server_age: ServerAge = ServerAge.MATURE) -> MarketTimingEngine:
    """Get the global MarketTimingEngine singleton."""
    global _market_timing
    with _market_timing_lock:
        if _market_timing is None:
            _market_timing = MarketTimingEngine(server_age)
        return _market_timing
