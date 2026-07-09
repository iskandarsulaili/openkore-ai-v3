"""EconomyAgent — vending, buying, trading, price checking, storage, refine, cards."""

from __future__ import annotations

import time
from typing import Any

from ai_sidecar.fleet.behavior_profiles import BehaviorProfile


class EconomyAgent(BehaviorProfile):
    """Handles all RO economy mechanics — shop, trade, storage, refine, cards."""

    def vending_action(self, inventory_items: list[dict[str, Any]], zeny: int,
                       map_name: str) -> dict[str, Any]:
        sellable = [i for i in inventory_items if i.get("vendor_price", 0) > 0
                    and not i.get("equipped", False)]
        if not sellable:
            return {"action": "no_action", "reason": "nothing_to_vendor"}
        if zeny > self._signals.get("zeny_target", 50000):
            return {"action": "skip", "reason": "zeny_sufficient"}
        items_to_sell = sorted(sellable, key=lambda x: x.get("vendor_price", 0),
                               reverse=True)[:5]
        return {"action": "vendor", "items": items_to_sell, "location": map_name}

    def buying_action(self, consumables: list[dict[str, Any]], inventory_qty: dict[str, int],
                      zeny: int) -> dict[str, Any]:
        needed = []
        for item in consumables:
            name = item.get("name", "")
            min_qty = item.get("min_qty", 0)
            have = inventory_qty.get(name, 0)
            if have < min_qty:
                needed.append({"item": name, "buy_qty": min_qty - have,
                               "max_price": item.get("max_price", 0)})
        if needed:
            return {"action": "buy", "items": needed, "zeny_available": zeny}
        return {"action": "no_action", "reason": "stocked"}

    def trade_check(self, offer_item: str, offer_qty: int, want_item: str,
                    want_qty: int) -> dict[str, Any]:
        best, score = self.best_action("economy")
        if best and "trade" in best and score > 0.5:
            return {"action": "accept_trade", "confidence": score}
        return {"action": "evaluate", "offer": f"{offer_qty}x{offer_item}",
                "request": f"{want_qty}x{want_item}"}

    def storage_decision(self, inventory_weight: int, max_weight: int,
                         storage_capacity: int, valuable_items: list[str]) -> dict[str, Any]:
        weight_pct = inventory_weight / max_weight
        if weight_pct > 0.8:
            return {"action": "visit_kafra", "deposit": valuable_items,
                    "reason": "weight_limit"}
        return {"action": "no_action"}

    def refine_decision(self, item_refine_lvl: int, has_enriched: bool,
                        zeny: int) -> dict[str, Any]:
        risk_table = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.6, 5: 0.4,
                      6: 0.4, 7: 0.2, 8: 0.2, 9: 0.2}
        succeed_chance = risk_table.get(item_refine_lvl, 0.0)
        if has_enriched:
            succeed_chance = min(1.0, succeed_chance * 1.4)
        if item_refine_lvl >= 4 and succeed_chance < 0.5 and zeny < 50000:
            return {"action": "wait", "reason": "low_zeny_or_high_risk"}
        if succeed_chance > 0.3:
            return {"action": "refine", "current_lvl": item_refine_lvl,
                    "success_rate": succeed_chance}
        return {"action": "hold", "reason": "too_risky"}

    def card_management(self, owned_cards: list[dict[str, Any]],
                        equipped_cards: list[str]) -> dict[str, Any]:
        keep = []
        sell = []
        use = []
        for card in owned_cards:
            name = card.get("name", "")
            if name in equipped_cards:
                continue
            value = card.get("estimated_value", 0)
            if value > 500000:
                keep.append(name)
            elif value > 10000:
                use.append(name)
            else:
                sell.append(name)
        return {"keep": keep, "sell": sell, "use": use}

    def record_outcome(self, action: str, success: bool, zeny_earned: float = 0.0) -> None:
        self._record_experience("economy", action, success, reward=zeny_earned,
                                zeny_earned=zeny_earned)
