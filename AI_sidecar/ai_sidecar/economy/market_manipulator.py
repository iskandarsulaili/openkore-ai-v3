"""
Market Manipulation Engine — corners markets, creates artificial scarcity,
times buy/sell cycles, and cross-references server events with price trends.
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
class MarketCorner:
    """A market corner operation — buying out all of an item."""
    item_name: str
    target_quantity: int = 0
    bought_quantity: int = 0
    max_unit_price: int = 0
    target_sell_price: int = 0
    start_time: float = 0.0
    is_active: bool = False
    is_complete: bool = False
    estimated_profit: int = 0
    risk_level: str = "medium"


@dataclass
class PriceSpike:
    """A predicted price spike event."""
    item_name: str
    predicted_price: int = 0
    current_price: int = 0
    spike_reason: str = ""
    confidence: float = 0.0
    time_to_spike_hours: float = 0.0
    recommended_action: str = ""


class MarketManipulator:
    """Manipulates the market for maximum profit."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._corners: list[MarketCorner] = []
        self._price_history: dict[str, list[tuple[float, int]]] = defaultdict(list)
        self._spike_predictions: list[PriceSpike] = []
        self._max_history: int = 1000
        self._capital: int = 0
        self._total_profit: int = 0
        self._enqueue_fn: Callable | None = None
        self._load_corner_targets()

    def _load_corner_targets(self) -> None:
        """Load items worth cornering."""
        self._corner_targets: dict[str, dict] = {
            "Oridecon": {"max_buy": 40000, "target_sell": 80000, "qty": 100, "risk": "low"},
            "Elunium": {"max_buy": 25000, "target_sell": 50000, "qty": 100, "risk": "low"},
            "Poring Card": {"max_buy": 8000, "target_sell": 20000, "qty": 50, "risk": "medium"},
            "Thara Frog Card": {"max_buy": 150000, "target_sell": 300000, "qty": 10, "risk": "high"},
            "Hydra Card": {"max_buy": 100000, "target_sell": 250000, "qty": 10, "risk": "high"},
            "Yggdrasil Berry": {"max_buy": 30000, "target_sell": 80000, "qty": 50, "risk": "medium"},
            "White Potion": {"max_buy": 400, "target_sell": 800, "qty": 1000, "risk": "low"},
            "Blue Potion": {"max_buy": 1500, "target_sell": 3000, "qty": 500, "risk": "low"},
            "Stem": {"max_buy": 80, "target_sell": 200, "qty": 2000, "risk": "low"},
            "Iron Ore": {"max_buy": 200, "target_sell": 500, "qty": 2000, "risk": "low"},
        }

    # ── Public API ──

    def record_price(self, item_name: str, price: int) -> None:
        """Record a price observation."""
        with self._lock:
            self._price_history[item_name].append((time.time(), price))
            if len(self._price_history[item_name]) > self._max_history:
                self._price_history[item_name] = self._price_history[item_name][-self._max_history:]
            self._check_spike(item_name)

    def _check_spike(self, item_name: str) -> None:
        """Check if a price spike is predicted."""
        history = self._price_history.get(item_name, [])
        if len(history) < 10:
            return

        recent = [p for _, p in history[-10:]]
        older = [p for _, p in history[-20:-10]]
        if not older:
            return

        avg_recent = sum(recent) / len(recent)
        avg_older = sum(older) / len(older)

        if avg_recent > avg_older * 1.3:
            spike = PriceSpike(
                item_name=item_name,
                predicted_price=int(avg_recent * 1.2),
                current_price=int(avg_recent),
                spike_reason="Price rising rapidly",
                confidence=0.6,
                time_to_spike_hours=24,
                recommended_action="buy_now",
            )
            self._spike_predictions.append(spike)
            if len(self._spike_predictions) > 100:
                self._spike_predictions.pop(0)

    def start_corner(self, item_name: str) -> bool:
        """Start cornering a market."""
        with self._lock:
            target = self._corner_targets.get(item_name)
            if not target:
                return False
            if any(c.is_active and c.item_name == item_name for c in self._corners):
                return False

            corner = MarketCorner(
                item_name=item_name,
                target_quantity=target["qty"],
                max_unit_price=target["max_buy"],
                target_sell_price=target["target_sell"],
                start_time=time.time(),
                is_active=True,
                estimated_profit=(target["target_sell"] - target["max_buy"]) * target["qty"],
                risk_level=target["risk"],
            )
            self._corners.append(corner)
            logger.info("market_corner_started: %s (target=%d, profit=%dz)", item_name, target["qty"], corner.estimated_profit)
            return True

    def record_buy(self, item_name: str, quantity: int, unit_price: int) -> None:
        """Record a buy for an active corner."""
        with self._lock:
            for corner in self._corners:
                if corner.is_active and corner.item_name == item_name:
                    corner.bought_quantity += quantity
                    if corner.bought_quantity >= corner.target_quantity:
                        corner.is_complete = True
                        corner.is_active = False
                        logger.info("market_corner_complete: %s (bought=%d/%d)", item_name, corner.bought_quantity, corner.target_quantity)
                    break

    def get_active_corners(self) -> list[MarketCorner]:
        with self._lock:
            return [c for c in self._corners if c.is_active]

    def get_completed_corners(self) -> list[MarketCorner]:
        with self._lock:
            return [c for c in self._corners if c.is_complete]

    def get_best_corner_opportunity(self) -> str | None:
        """Get the best item to corner right now."""
        with self._lock:
            for name, target in sorted(self._corner_targets.items(), key=lambda x: -x[1]["qty"]):
                if not any(c.is_active and c.item_name == name for c in self._corners):
                    return name
            return None

    def get_spike_predictions(self, min_confidence: float = 0.5) -> list[PriceSpike]:
        with self._lock:
            return [s for s in self._spike_predictions if s.confidence >= min_confidence]

    def get_market_summary(self) -> str:
        with self._lock:
            lines = [f"── Market Manipulator ──"]
            lines.append(f"Capital: {self._capital:,}z")
            lines.append(f"Total profit: {self._total_profit:,}z")
            lines.append(f"Items tracked: {len(self._price_history)}")
            active = self.get_active_corners()
            if active:
                lines.append(f"Active corners: {', '.join(f'{c.item_name}({c.bought_quantity}/{c.target_quantity})' for c in active)}")
            completed = self.get_completed_corners()
            if completed:
                lines.append(f"Completed corners: {len(completed)}")
            spikes = self.get_spike_predictions()
            if spikes:
                lines.append(f"Predicted spikes: {', '.join(f'{s.item_name}({s.confidence:.0%})' for s in spikes[:3])}")
            return "\n".join(lines)

    def set_capital(self, capital: int) -> None:
        with self._lock:
            self._capital = capital

    def set_enqueue_fn(self, fn: Callable) -> None:
        with self._lock:
            self._enqueue_fn = fn

    def reset(self) -> None:
        with self._lock:
            self._corners.clear()
            self._price_history.clear()
            self._spike_predictions.clear()
            self._total_profit = 0


# ── Global Singleton ──

_market_manip: MarketManipulator | None = None
_market_manip_lock = RLock()


def get_market_manipulator() -> MarketManipulator:
    global _market_manip
    with _market_manip_lock:
        if _market_manip is None:
            _market_manip = MarketManipulator()
        return _market_manip
