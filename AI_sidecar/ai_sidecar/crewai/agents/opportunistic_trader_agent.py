from __future__ import annotations

from typing import Any
from .base_agent import BehaviorProfile



class OpportunisticTraderProfile(BehaviorProfile):
    """Market watching, price arbitrage."""

    agent_id = "opportunistic_trader"
    role = "Opportunistic Trader"
    goal = "Identify profitable market opportunities and execute trades"
    backstory = (
        "A sharp-eyed market opportunist who lives for price differences. "
        "This agent watches market boards and NPC buy/sell spreads, "
        "identifies arbitrage opportunities, and executes trades at "
        "the perfect moment."
    )

    def can_handle(self, signals: dict[str, Any]) -> float:
        market_data = signals.get("market_data", [])
        if not market_data:
            return 0.0
        opportunities = [
            m for m in market_data
            if m.get("buy_price", 0) < m.get("sell_price", 999999)
        ]
        if opportunities:
            return min(0.5 + 0.3 * len(opportunities) / 5.0, 1.0)
        return 0.1

    def get_action(self, signals: dict[str, Any]) -> dict[str, Any] | None:
        market = signals.get("market_data", [])
        opportunities = [
            m for m in market
            if m.get("buy_price", 0) < m.get("sell_price", 999999)
            and m.get("buy_price", 0) > 0
        ]
        if not opportunities:
            return None

        best = max(opportunities, key=lambda x: x.get("sell_price", 0) - x.get("buy_price", 0))
        item = best.get("item", "unknown")
        profit = best.get("sell_price", 0) - best.get("buy_price", 0)
        return {
            "kind": "trade",
            "command": f"buy {best.get('item_id', '')} {best.get('buy_price', 0)}; sell {best.get('item_id', '')} {best.get('sell_price', 0)}",
            "confidence": 0.75,
            "reason": f"Arbitrage: buy {item} at {best.get('buy_price', 0)}, sell at {best.get('sell_price', 0)} (profit: {profit})",
        }
