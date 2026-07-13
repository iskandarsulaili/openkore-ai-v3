"""
Economy optimizer — player vendor prices, farm targets, storage strategy.

The LLM decides strategy; the optimizer provides market data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

# NPC buy prices for common items
NPC_BUY_PRICES: dict[str, int] = {
    "red_potion": 500,
    "orange_potion": 1000,
    "yellow_potion": 2000,
    "white_potion": 5000,
    "blue_potion": 3000,
    "fly_wing": 500,
    "butterfly_wing": 2000,
}

# NPC sell prices for common farm items
NPC_SELL_PRICES: dict[str, int] = {
    "stem": 100,
    "scorpion_tail": 200,
    "hollow": 50,
    "jellopy": 30,
    "memento": 500,
    "immortal_heart": 1000,
    "evil_horn": 300,
    "skull": 100,
    "decayed_nail": 200,
    "stiff_leaf": 50,
}

# Profitable farming targets per level range
FARM_TARGETS: list[dict[str, Any]] = [
    {"level_min": 1, "level_max": 20, "monster": "poring", "drops": ["jellopy", "sticky_moss"], "value_per_hour": 5000},
    {"level_min": 1, "level_max": 20, "monster": "lunatic", "drops": ["hollow", "fur"], "value_per_hour": 4000},
    {"level_min": 10, "level_max": 30, "monster": "drops", "drops": ["sticky_moss", "jellopy"], "value_per_hour": 6000},
    {"level_min": 10, "level_max": 30, "monster": "pupa", "drops": ["stiff_leaf"], "value_per_hour": 8000},
    {"level_min": 15, "level_max": 35, "monster": "spore", "drops": ["mushroom_spore", "stiff_leaf"], "value_per_hour": 10000},
    {"level_min": 20, "level_max": 40, "monster": "savage", "drops": ["savage_robe", "fur"], "value_per_hour": 15000},
    {"level_min": 25, "level_max": 45, "monster": "argiope", "drops": ["scorpion_tail"], "value_per_hour": 20000},
    {"level_min": 30, "level_max": 50, "monster": "drainliar", "drops": ["decayed_nail", "skull"], "value_per_hour": 25000},
    {"level_min": 35, "level_max": 55, "monster": "rafflesia", "drops": ["stem"], "value_per_hour": 30000},
    {"level_min": 40, "level_max": 60, "monster": "myst", "drops": ["immortal_heart"], "value_per_hour": 40000},
    {"level_min": 50, "level_max": 70, "monster": "injustice", "drops": ["evil_horn"], "value_per_hour": 50000},
]


@dataclass(slots=True)
class EconomyOptimizer:
    """Economy optimization — buy/sell decisions, farm targets."""
    
    _lock: RLock = field(default_factory=RLock)
    _stats: dict[str, int] = field(default_factory=lambda: {"buys": 0, "sells": 0, "storage": 0})
    
    def get_buy_price(self, item_name: str) -> int:
        """Get the NPC buy price for an item."""
        return NPC_BUY_PRICES.get(item_name.lower(), 99999)
    
    def get_sell_price(self, item_name: str) -> int:
        """Get the NPC sell price for an item."""
        return NPC_SELL_PRICES.get(item_name.lower(), 10)
    
    def should_sell_to_npc(self, item_name: str, player_vendor_price: int | None = None) -> bool:
        """Decide whether to sell to NPC or player vendor."""
        npc_price = self.get_sell_price(item_name)
        if player_vendor_price is not None and player_vendor_price > npc_price * 1.3:
            return False  # Player vendor pays more
        return True
    
    def get_farm_recommendation(self, level: int) -> dict[str, Any] | None:
        """Get the best farming target for a given level."""
        best = None
        best_value = 0
        for target in FARM_TARGETS:
            if target["level_min"] <= level <= target["level_max"]:
                if target["value_per_hour"] > best_value:
                    best_value = target["value_per_hour"]
                    best = target
        return best
    
    def get_potion_budget(self, zeny: int) -> dict[str, int]:
        """Get recommended potion purchase based on zeny."""
        if zeny < 1000:
            return {"red_potion": 0, "orange_potion": 0}
        if zeny < 5000:
            return {"red_potion": 10, "orange_potion": 0}
        if zeny < 20000:
            return {"red_potion": 30, "orange_potion": 10}
        return {"red_potion": 50, "orange_potion": 20, "white_potion": 10}
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)
