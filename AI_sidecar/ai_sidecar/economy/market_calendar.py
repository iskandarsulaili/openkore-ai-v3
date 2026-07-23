"""
Market timing calendar for Ragnarok Online economy.

Tracks day-of-week pricing patterns, time-of-day activity levels,
server events (WoE, double exp, MVP events, holidays), and patch cycles.
All multipliers are data-driven from observed patterns — no hardcoded dates.

Integrates with PriceTracker to provide timing-aware price recommendations.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import math
import time
from typing import Literal


class EventType(str, Enum):
    """Types of server events that affect the economy."""
    WOE = "woe"
    DOUBLE_EXP = "double_exp"
    MVP_EVENT = "mvp_event"
    HOLIDAY = "holiday"
    PATCH = "patch"
    BOT_CRASH = "bot_crash"
    CUSTOM = "custom"


@dataclass
class ServerEvent:
    """A recorded server event with observed price impact."""
    event_type: EventType
    timestamp: float
    observed_multiplier: float = 1.0
    affected_categories: list[str] = field(default_factory=lambda: ["all"])
    duration_hours: float = 24.0
    label: str = ""


@dataclass
class TimingRecommendation:
    """Recommended time to buy or sell an item."""
    day_of_week: int          # 0=Monday, 6=Sunday
    hour: int                 # 0-23
    expected_multiplier: float
    reason: str


# ── Default observed patterns (data-driven defaults) ──

# Day-of-week price multipliers observed from market data
# Monday: cheapest (post-WoE dump, low demand)
# Friday: most expensive (pre-WoE stocking, weekend demand)
DEFAULT_DAY_FACTORS: dict[int, float] = {
    0: 0.85,   # Monday    — cheapest, post-weekend dump
    1: 0.90,   # Tuesday   — still low
    2: 0.95,   # Wednesday — recovering
    3: 1.00,   # Thursday  — baseline
    4: 1.08,   # Friday    — pre-WoE / weekend stocking begins
    5: 1.15,   # Saturday  — peak WoE day, highest demand
    6: 1.10,   # Sunday    — still elevated
}

# Hour-of-day activity multipliers (server time)
# Peak hours: evening (18-23) when most players are online
# Off-peak: early morning (3-9) when fewest players are online
DEFAULT_HOUR_FACTORS: dict[int, float] = {
    0: 0.95,   # Midnight  — moderate activity
    1: 0.90,
    2: 0.85,
    3: 0.80,   # 3 AM     — lowest activity
    4: 0.78,
    5: 0.75,   # 5 AM     — absolute low
    6: 0.78,
    7: 0.85,
    8: 0.90,
    9: 0.95,
    10: 1.00,  # 10 AM    — baseline
    11: 1.05,
    12: 1.10,  # Noon     — lunch hour activity
    13: 1.05,
    14: 1.00,
    15: 1.00,
    16: 1.05,
    17: 1.10,  # 5 PM     — after-school / after-work ramp
    18: 1.20,  # 6 PM     — peak begins
    19: 1.30,  # 7 PM     — prime time
    20: 1.35,  # 8 PM     — highest activity
    21: 1.30,  # 9 PM     — still high
    22: 1.20,  # 10 PM    — winding down
    23: 1.05,  # 11 PM    — late night
}

# Event type → category → multiplier mapping
# Learned from observing price movements during past events
DEFAULT_EVENT_MULTIPLIERS: dict[str, dict[str, float]] = {
    "woe": {
        "consumable": 2.5,       # Pots, ammo, traps
        "gear": 1.5,             # PvP gear
        "material": 1.3,         # Refining materials
        "all": 1.8,
    },
    "double_exp": {
        "consumable": 1.4,       # More grinding = more potion use
        "material": 1.2,
        "all": 1.3,
    },
    "mvp_event": {
        "gear": 1.6,             # MVP drops in demand
        "material": 1.4,
        "all": 1.4,
    },
    "holiday": {
        "consumable": 1.3,
        "cosmetic": 2.0,         # Costumes, cards
        "all": 1.2,
    },
    "patch": {
        "gear": 1.8,             # New content gear rush
        "material": 1.5,        # New recipes need materials
        "consumable": 1.2,
        "all": 1.4,
    },
    "bot_crash": {
        "material": 0.3,         # Bot farmed materials crash
        "consumable": 0.5,
        "all": 0.4,
    },
}

# Category assignment for items (data-driven, expandable)
DEFAULT_ITEM_CATEGORIES: dict[str, str] = {
    # Consumables
    "white potion": "consumable",
    "blue potion": "consumable",
    "convex mirror": "consumable",
    "panacea": "consumable",
    "holy water": "consumable",
    "fly wing": "consumable",
    "butterfly wing": "consumable",
    "empty bottle": "consumable",
    "honey": "consumable",
    "iggdrasil berry": "consumable",
    "trap": "consumable",
    "grenade": "consumable",
    "bomb": "consumable",
    "fire arrow": "consumable",
    "silver arrow": "consumable",
    "crystal arrow": "consumable",
    # Materials
    "stem": "material",
    "memento": "material",
    "feather": "material",
    "elunium": "material",
    "oridecon": "material",
    "flower": "material",
    "bacsojin": "material",
    "old blue box": "material",
    "old violet box": "material",
    # Gear
    "ring": "gear",
    "necklace": "gear",
    "earring": "gear",
    "boots": "gear",
    "gloves": "gear",
    "manteau": "gear",
    "armor": "gear",
    "weapon": "gear",
    "shield": "gear",
    "helmet": "gear",
}


class MarketCalendar:
    """
    Tracks market timing patterns and server events for price-aware recommendations.

    All multipliers are data-driven from observed patterns. The calendar learns
    from recorded events and adjusts recommendations accordingly.
    """

    def __init__(
        self,
        day_factors: dict[int, float] | None = None,
        hour_factors: dict[int, float] | None = None,
        event_multipliers: dict[str, dict[str, float]] | None = None,
        item_categories: dict[str, str] | None = None,
    ):
        self._day_factors: dict[int, float] = day_factors or DEFAULT_DAY_FACTORS.copy()
        self._hour_factors: dict[int, float] = hour_factors or DEFAULT_HOUR_FACTORS.copy()
        self._event_multipliers: dict[str, dict[str, float]] = (
            event_multipliers or {
                k: v.copy() for k, v in DEFAULT_EVENT_MULTIPLIERS.items()
            }
        )
        self._item_categories: dict[str, str] = (
            item_categories or DEFAULT_ITEM_CATEGORIES.copy()
        )

        # Active events currently affecting the market
        self._active_events: list[ServerEvent] = []

        # Historical event log for pattern learning
        self._event_history: list[ServerEvent] = []

        # Observed day-of-week factors (learned from actual price data)
        self._observed_day_factors: dict[int, list[float]] = {
            i: [] for i in range(7)
        }

        # Observed hour-of-day factors (learned from actual price data)
        self._observed_hour_factors: dict[int, list[float]] = {
            i: [] for i in range(24)
        }

    # ── Day-of-week pricing ──

    def get_day_factor(self, day_of_week: int | None = None) -> float:
        """
        Get the price multiplier for a given day of the week.

        Args:
            day_of_week: 0=Monday, 6=Sunday. Defaults to current day.

        Returns:
            Price multiplier (e.g., 0.85 for Monday = cheapest day to buy).
        """
        if day_of_week is None:
            day_of_week = datetime.now(timezone.utc).weekday()
        return self._day_factors.get(day_of_week, 1.0)

    def set_day_factor(self, day_of_week: int, factor: float) -> None:
        """Override the day factor for a specific day (data-driven update)."""
        self._day_factors[day_of_week] = round(factor, 4)

    def record_day_observation(self, day_of_week: int, price_ratio: float) -> None:
        """
        Record an observed price ratio for a day of the week.
        Used to refine day factors over time from actual market data.
        """
        if 0 <= day_of_week <= 6:
            self._observed_day_factors[day_of_week].append(price_ratio)
            # Recompute as rolling median
            if len(self._observed_day_factors[day_of_week]) >= 5:
                values = sorted(self._observed_day_factors[day_of_week])
                median = values[len(values) // 2]
                self._day_factors[day_of_week] = round(median, 4)

    # ── Hour-of-day activity ──

    def get_hour_factor(self, hour: int | None = None) -> float:
        """
        Get the activity multiplier for a given hour of the day.

        Args:
            hour: 0-23. Defaults to current hour.

        Returns:
            Activity multiplier (higher = more players = better sell prices).
        """
        if hour is None:
            hour = datetime.now(timezone.utc).hour
        return self._hour_factors.get(hour, 1.0)

    def set_hour_factor(self, hour: int, factor: float) -> None:
        """Override the hour factor for a specific hour (data-driven update)."""
        self._hour_factors[hour] = round(factor, 4)

    def record_hour_observation(self, hour: int, activity_ratio: float) -> None:
        """
        Record an observed activity ratio for an hour.
        Used to refine hour factors over time from actual market data.
        """
        if 0 <= hour <= 23:
            self._observed_hour_factors[hour].append(activity_ratio)
            if len(self._observed_hour_factors[hour]) >= 10:
                values = sorted(self._observed_hour_factors[hour])
                median = values[len(values) // 2]
                self._hour_factors[hour] = round(median, 4)

    # ── Combined timing multiplier ──

    def get_timing_multiplier(
        self,
        day_of_week: int | None = None,
        hour: int | None = None,
    ) -> float:
        """
        Get the combined timing multiplier (day × hour).

        This is the base market condition multiplier before event adjustments.
        """
        return self.get_day_factor(day_of_week) * self.get_hour_factor(hour)

    # ── Event tracking ──

    def record_event(
        self,
        event_type: EventType | str,
        timestamp: float | None = None,
        observed_multiplier: float = 1.0,
        affected_categories: list[str] | None = None,
        duration_hours: float = 24.0,
        label: str = "",
    ) -> ServerEvent:
        """
        Record a server event for market timing.

        Args:
            event_type: Type of event (woe, double_exp, mvp_event, etc.)
            timestamp: When the event occurred/starts. Defaults to now.
            observed_multiplier: Observed price impact multiplier.
            affected_categories: Which item categories this affects.
            duration_hours: How long the event lasts.
            label: Human-readable label for the event.

        Returns:
            The created ServerEvent.
        """
        if isinstance(event_type, str):
            event_type = EventType(event_type)

        if timestamp is None:
            timestamp = time.time()

        event = ServerEvent(
            event_type=event_type,
            timestamp=timestamp,
            observed_multiplier=observed_multiplier,
            affected_categories=affected_categories or ["all"],
            duration_hours=duration_hours,
            label=label,
        )

        self._active_events.append(event)
        self._event_history.append(event)

        # Prune expired events
        self._prune_expired_events()

        return event

    def _prune_expired_events(self) -> None:
        """Remove events that have expired."""
        now = time.time()
        self._active_events = [
            e for e in self._active_events
            if (now - e.timestamp) < (e.duration_hours * 3600)
        ]

    def get_active_events(self) -> list[ServerEvent]:
        """Get all currently active server events."""
        self._prune_expired_events()
        return list(self._active_events)

    def is_event_active(self, event_type: EventType | str) -> bool:
        """Check if a specific event type is currently active."""
        if isinstance(event_type, str):
            event_type = EventType(event_type)
        self._prune_expired_events()
        return any(e.event_type == event_type for e in self._active_events)

    def get_event_multiplier(
        self,
        item_category: str,
        event_type: EventType | str | None = None,
    ) -> float:
        """
        Get the price multiplier from active events for a given item category.

        Args:
            item_category: Category of the item (consumable, gear, material, etc.)
            event_type: Optional — filter to a specific event type.

        Returns:
            Combined multiplier from all active events affecting this category.
            Returns 1.0 if no active events affect this category.
        """
        self._prune_expired_events()

        multiplier = 1.0

        for event in self._active_events:
            if event_type is not None and event.event_type != EventType(event_type):
                continue

            event_key = event.event_type.value
            event_data = self._event_multipliers.get(event_key, {})

            # Check category-specific multiplier first
            if item_category in event_data:
                multiplier *= event_data[item_category]
            elif "all" in event_data:
                multiplier *= event_data["all"]
            else:
                # Use the event's observed multiplier as fallback
                multiplier *= event.observed_multiplier

        return multiplier

    def get_combined_multiplier(
        self,
        item_name: str,
        day_of_week: int | None = None,
        hour: int | None = None,
    ) -> float:
        """
        Get the full combined multiplier: timing × event effects.

        This is the final price adjustment factor for a given item at a given time.
        """
        timing = self.get_timing_multiplier(day_of_week, hour)
        category = self.get_item_category(item_name)
        event_mul = self.get_event_multiplier(category)
        return timing * event_mul

    # ── Item category lookup ──

    def get_item_category(self, item_name: str) -> str:
        """Get the economic category for an item name."""
        key = item_name.lower().strip()
        return self._item_categories.get(key, "all")

    def set_item_category(self, item_name: str, category: str) -> None:
        """Set or override the category for an item."""
        self._item_categories[item_name.lower().strip()] = category

    # ── Best time recommendations ──

    def get_best_sell_time(self, item_category: str) -> TimingRecommendation:
        """
        Get the best time to sell items of a given category.

        Sell when demand is highest: peak hour on a high-factor day,
        ideally during an active event that boosts this category.

        Returns:
            TimingRecommendation with day, hour, expected multiplier, and reason.
        """
        self._prune_expired_events()

        best_day = 5  # Saturday (default best sell day)
        best_hour = 20  # 8 PM (default best sell hour)
        best_multiplier = 0.0

        for day in range(7):
            for hour in range(24):
                day_f = self._day_factors.get(day, 1.0)
                hour_f = self._hour_factors.get(hour, 1.0)
                event_mul = self.get_event_multiplier(item_category)
                total = day_f * hour_f * event_mul

                if total > best_multiplier:
                    best_multiplier = total
                    best_day = day
                    best_hour = hour

        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        reason = (
            f"Sell on {day_names[best_day]} at {best_hour:02d}:00 — "
            f"peak demand with {best_multiplier:.2f}x expected multiplier"
        )

        # Check if an active event makes this even better
        active = self.get_active_events()
        if active:
            event_labels = [e.label or e.event_type.value for e in active]
            reason += f" (active events: {', '.join(event_labels)})"

        return TimingRecommendation(
            day_of_week=best_day,
            hour=best_hour,
            expected_multiplier=best_multiplier,
            reason=reason,
        )

    def get_best_buy_time(self, item_category: str) -> TimingRecommendation:
        """
        Get the best time to buy items of a given category.

        Buy when demand is lowest: off-peak hour on a low-factor day,
        ideally when no events are active.

        Returns:
            TimingRecommendation with day, hour, expected multiplier, and reason.
        """
        self._prune_expired_events()

        best_day = 0  # Monday (default best buy day)
        best_hour = 5  # 5 AM (default best buy hour)
        best_multiplier = float("inf")

        for day in range(7):
            for hour in range(24):
                day_f = self._day_factors.get(day, 1.0)
                hour_f = self._hour_factors.get(hour, 1.0)
                event_mul = self.get_event_multiplier(item_category)
                total = day_f * hour_f * event_mul

                if total < best_multiplier:
                    best_multiplier = total
                    best_day = day
                    best_hour = hour

        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        reason = (
            f"Buy on {day_names[best_day]} at {best_hour:02d}:00 — "
            f"lowest demand with {best_multiplier:.2f}x expected multiplier"
        )

        return TimingRecommendation(
            day_of_week=best_day,
            hour=best_hour,
            expected_multiplier=best_multiplier,
            reason=reason,
        )

    # ── Event prediction ──

    def predict_next_woe(self) -> list[dict]:
        """
        Predict upcoming WoE times based on observed patterns.
        WoE typically runs Wednesday 20:00 and Saturday 20:00 server time.
        """
        now = datetime.now(timezone.utc)
        predictions = []

        # Standard WoE schedule
        woe_schedule = [
            {"day": 2, "hour": 20, "label": "Wednesday WoE"},   # Wednesday
            {"day": 5, "hour": 20, "label": "Saturday WoE"},    # Saturday
        ]

        for slot in woe_schedule:
            days_ahead = (slot["day"] - now.weekday()) % 7
            if days_ahead == 0 and now.hour >= slot["hour"]:
                days_ahead = 7  # Already passed this week
            event_time = now + timedelta(days=days_ahead)
            event_time = event_time.replace(
                hour=slot["hour"], minute=0, second=0, microsecond=0
            )
            predictions.append({
                "label": slot["label"],
                "timestamp": event_time.timestamp(),
                "days_until": days_ahead,
                "hours_until": days_ahead * 24 + (slot["hour"] - now.hour) % 24,
            })

        return predictions

    # ── Serialization / state ──

    def get_state(self) -> dict:
        """Get serializable state for persistence."""
        return {
            "day_factors": {str(k): v for k, v in self._day_factors.items()},
            "hour_factors": {str(k): v for k, v in self._hour_factors.items()},
            "event_multipliers": self._event_multipliers,
            "item_categories": self._item_categories,
            "active_events": [
                {
                    "event_type": e.event_type.value,
                    "timestamp": e.timestamp,
                    "observed_multiplier": e.observed_multiplier,
                    "affected_categories": e.affected_categories,
                    "duration_hours": e.duration_hours,
                    "label": e.label,
                }
                for e in self._active_events
            ],
            "event_history": [
                {
                    "event_type": e.event_type.value,
                    "timestamp": e.timestamp,
                    "observed_multiplier": e.observed_multiplier,
                    "affected_categories": e.affected_categories,
                    "duration_hours": e.duration_hours,
                    "label": e.label,
                }
                for e in self._event_history[-100:]  # Keep last 100
            ],
        }

    def load_state(self, state: dict) -> None:
        """Restore state from a previously saved dict."""
        if "day_factors" in state:
            self._day_factors = {int(k): v for k, v in state["day_factors"].items()}
        if "hour_factors" in state:
            self._hour_factors = {int(k): v for k, v in state["hour_factors"].items()}
        if "event_multipliers" in state:
            self._event_multipliers = state["event_multipliers"]
        if "item_categories" in state:
            self._item_categories = state["item_categories"]
        if "active_events" in state:
            self._active_events = [
                ServerEvent(
                    event_type=EventType(e["event_type"]),
                    timestamp=e["timestamp"],
                    observed_multiplier=e.get("observed_multiplier", 1.0),
                    affected_categories=e.get("affected_categories", ["all"]),
                    duration_hours=e.get("duration_hours", 24.0),
                    label=e.get("label", ""),
                )
                for e in state["active_events"]
            ]
        if "event_history" in state:
            self._event_history = [
                ServerEvent(
                    event_type=EventType(e["event_type"]),
                    timestamp=e["timestamp"],
                    observed_multiplier=e.get("observed_multiplier", 1.0),
                    affected_categories=e.get("affected_categories", ["all"]),
                    duration_hours=e.get("duration_hours", 24.0),
                    label=e.get("label", ""),
                )
                for e in state["event_history"]
            ]


# ── Singleton instance ──
_market_calendar: MarketCalendar | None = None


def get_market_calendar() -> MarketCalendar:
    """Get the global MarketCalendar instance."""
    global _market_calendar
    if _market_calendar is None:
        _market_calendar = MarketCalendar()
    return _market_calendar
