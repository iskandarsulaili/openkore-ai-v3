"""
Market manipulator — shapes the economy, doesn't just participate.

A top player doesn't just buy and sell at market price. They control
the market. Buy out cheap listings, undercut competitors, crash prices,
corner supply. This module gives the bot active market influence.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class MarketItem:
    """Tracked market item."""
    name: str
    current_price: int = 0
    lowest_price: int = 0
    volume_24h: int = 0
    supply_estimate: int = 0
    demand_estimate: int = 0
    trend: str = "stable"  # rising, falling, stable, volatile
    last_updated: float = 0.0
    our_inventory: int = 0
    our_cost_basis: int = 0


@dataclass
class MarketAction:
    """A market manipulation action."""
    action_type: str  # buyout, undercut, dump, corner, arbitrage
    item: str
    quantity: int = 0
    price: int = 0
    reason: str = ""
    executed: bool = False
    result: str = ""
    timestamp: float = 0.0


@dataclass(slots=True)
class MarketManipulator:
    """Actively shapes the server economy."""
    
    _lock: RLock = field(default_factory=RLock)
    _items: dict[str, MarketItem] = field(default_factory=dict)
    _actions: list[MarketAction] = field(default_factory=list)
    _capital: int = 0
    _stats: dict[str, int] = field(default_factory=lambda: {
        "buyouts": 0, "undercuts": 0, "dumps": 0, "corners": 0, "arbitrages": 0,
        "profit": 0, "loss": 0,
    })
    _enqueue_fn: Callable | None = None
    
    def observe_price(self, item_name: str, price: int, volume: int = 0) -> None:
        """Record a price observation."""
        with self._lock:
            item = self._items.setdefault(item_name, MarketItem(name=item_name))
            if item.current_price > 0:
                if price < item.current_price * 0.9:
                    item.trend = "falling"
                elif price > item.current_price * 1.1:
                    item.trend = "rising"
                else:
                    item.trend = "stable"
            item.current_price = price
            if item.lowest_price == 0 or price < item.lowest_price:
                item.lowest_price = price
            item.volume_24h = max(item.volume_24h, volume)
            item.last_updated = time.time()
    
    def set_capital(self, amount: int) -> None:
        with self._lock:
            self._capital = amount
    
    def evaluate_opportunities(self) -> list[MarketAction]:
        """Evaluate current market conditions and return recommended actions."""
        with self._lock:
            now = time.time()
            actions: list[MarketAction] = []
            
            for name, item in self._items.items():
                if now - item.last_updated > 86400:
                    continue  # Stale data
                
                # 1. Buyout opportunity: item trending up, we have capital
                if item.trend == "rising" and self._capital > item.current_price * 10:
                    if item.volume_24h > 0 and item.current_price < item.lowest_price * 1.5:
                        actions.append(MarketAction(
                            action_type="buyout",
                            item=name,
                            quantity=min(10, self._capital // item.current_price),
                            price=item.current_price,
                            reason=f"Buying {name} while cheap, trend is rising",
                            timestamp=now,
                        ))
                
                # 2. Undercut opportunity: we have inventory, competitors selling
                if item.our_inventory > 0 and item.current_price > item.our_cost_basis * 1.2:
                    actions.append(MarketAction(
                        action_type="undercut",
                        item=name,
                        quantity=item.our_inventory,
                        price=item.current_price - 1,
                        reason=f"Undercutting {name} at {item.current_price - 1}z",
                        timestamp=now,
                    ))
                
                # 3. Dump opportunity: trend falling, cut losses
                if item.trend == "falling" and item.our_inventory > 5:
                    actions.append(MarketAction(
                        action_type="dump",
                        item=name,
                        quantity=item.our_inventory,
                        price=int(item.current_price * 0.95),
                        reason=f"Dumping {name} before price crashes further",
                        timestamp=now,
                    ))
            
            return actions
    
    def execute_action(self, action: MarketAction) -> bool:
        """Execute a market action."""
        with self._lock:
            action.executed = True
            action.timestamp = time.time()
            self._actions.append(action)
            
            if action.action_type == "buyout":
                cost = action.quantity * action.price
                if cost <= self._capital:
                    self._capital -= cost
                    item = self._items.get(action.item)
                    if item:
                        item.our_inventory += action.quantity
                        item.our_cost_basis = int((item.our_cost_basis * (item.our_inventory - action.quantity) + cost) / max(item.our_inventory, 1))
                    self._stats["buyouts"] += 1
                    action.result = f"Bought {action.quantity}x {action.item} for {cost}z"
                    logger.info("market_buyout: %s x%d for %d", action.item, action.quantity, cost)
                    return True
            
            elif action.action_type == "undercut":
                if self._enqueue_fn:
                    self._enqueue_fn("default", f"chat selling {action.item} {action.price}z each")
                self._stats["undercuts"] += 1
                action.result = f"Listed {action.quantity}x {action.item} at {action.price}z"
                logger.info("market_undercut: %s at %d", action.item, action.price)
                return True
            
            elif action.action_type == "dump":
                if self._enqueue_fn:
                    self._enqueue_fn("default", f"chat selling {action.item} cheap {action.price}z")
                self._stats["dumps"] += 1
                action.result = f"Dumping {action.quantity}x {action.item} at {action.price}z"
                logger.info("market_dump: %s x%d at %d", action.item, action.quantity, action.price)
                return True
            
            return False
    
    def get_market_context(self) -> str:
        """Get formatted market context for LLM prompts."""
        with self._lock:
            lines = ["── Market Intelligence ──"]
            lines.append(f"  Capital: {self._capital:,}z")
            lines.append(f"  Buyouts: {self._stats['buyouts']} | Undercuts: {self._stats['undercuts']}")
            lines.append(f"  Profit: {self._stats['profit']:,}z | Loss: {self._stats['loss']:,}z")
            
            tracked = [i for i in self._items.values() if i.last_updated > 0]
            if tracked:
                lines.append(f"  Tracked items: {len(tracked)}")
                for item in sorted(tracked, key=lambda x: x.last_updated, reverse=True)[:5]:
                    lines.append(f"    {item.name}: {item.current_price}z ({item.trend}) inv={item.our_inventory}")
            
            return "\n".join(lines)
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# Global instance
_market: MarketManipulator | None = None
_market_lock = RLock()


def get_market_manipulator() -> MarketManipulator:
    global _market
    with _market_lock:
        if _market is None:
            _market = MarketManipulator()
        return _market
