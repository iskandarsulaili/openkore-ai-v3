"""Market Seeder — seed realistic market prices based on server type and age.

Pro-botting insight: knowing market prices is essential for:
  - Deciding what to farm (cards vs materials vs consumables)
  - Deciding what to buy/sell at NPC vs player market
  - Estimating profit per hour for different farming strategies
  - Adapting to server economy (new vs old, low-rate vs high-rate)

This module provides realistic seed prices for common RO items,
adjustable by server type (low/mid/high rate) and server age
(fresh/established/old).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ── Enums ───────────────────────────────────────────────────────────────────


class ServerRate(str, Enum):
    """Server experience/drop rate classification."""
    LOW_RATE = "low_rate"       # 1x-5x rates
    MID_RATE = "mid_rate"       # 5x-25x rates
    HIGH_RATE = "high_rate"     # 25x+ rates


class ServerAge(str, Enum):
    """Server age classification affects item scarcity."""
    FRESH = "fresh"             # 0-3 months: items scarce, prices high
    ESTABLISHED = "established" # 3-12 months: stable economy
    OLD = "old"                 # 12+ months: items abundant, prices low


# ── Price multipliers by server type ────────────────────────────────────────

SERVER_RATE_MULTIPLIERS: dict[ServerRate, float] = {
    ServerRate.LOW_RATE: 2.5,    # Low rate: items are scarce, 2.5x base
    ServerRate.MID_RATE: 1.0,    # Mid rate: baseline
    ServerRate.HIGH_RATE: 0.4,   # High rate: items are abundant, 0.4x base
}

SERVER_AGE_MULTIPLIERS: dict[ServerAge, float] = {
    ServerAge.FRESH: 1.5,         # Fresh server: items scarce, 1.5x
    ServerAge.ESTABLISHED: 1.0,  # Established: baseline
    ServerAge.OLD: 0.6,          # Old server: items abundant, 0.6x
}


# ── Seed price data ─────────────────────────────────────────────────────────

# Base prices for mid-rate, established server
# Format: item_name -> {"buy": NPC buy price, "sell": NPC sell price}

CONSUMABLES: dict[str, dict[str, int]] = {
    "White Potion": {"buy": 500, "sell": 1000},
    "Blue Potion": {"buy": 2000, "sell": 4000},
    "Fly Wing": {"buy": 100, "sell": 200},
    "Butterfly Wing": {"buy": 500, "sell": 1000},
    "Red Potion": {"buy": 50, "sell": 100},
    "Orange Potion": {"buy": 200, "sell": 400},
    "Yellow Potion": {"buy": 350, "sell": 700},
    "Green Potion": {"buy": 100, "sell": 200},
    "Awakening Potion": {"buy": 1000, "sell": 2000},
    "Concentration Potion": {"buy": 2000, "sell": 4000},
}

# Low-rate prices override for consumables
CONSUMABLES_LOW_RATE: dict[str, dict[str, int]] = {
    "White Potion": {"buy": 500, "sell": 2500},
    "Blue Potion": {"buy": 2000, "sell": 10000},
    "Fly Wing": {"buy": 100, "sell": 500},
    "Butterfly Wing": {"buy": 500, "sell": 2000},
}

# High-rate prices override for consumables
CONSUMABLES_HIGH_RATE: dict[str, dict[str, int]] = {
    "White Potion": {"buy": 500, "sell": 200},
    "Blue Potion": {"buy": 2000, "sell": 500},
}

CARDS: dict[str, int] = {
    # Common cards (mid-rate established)
    "Poring Card": 50000,
    "Savage Babe Card": 200000,
    "Poporing Card": 100000,
    "Drops Card": 30000,
    "Lunatic Card": 80000,
    "Fabre Card": 40000,
    "Pupa Card": 25000,
    "Thief Bug Card": 500000,
    "Chonchon Card": 35000,
    "Rocker Card": 60000,
    "Spore Card": 45000,
    "Creamy Card": 150000,
    "Willows Card": 100000,
    "Vadon Card": 120000,
    "Hornet Card": 70000,
    "Familiar Card": 80000,
    "Skeleton Card": 100000,
    "Orc Warrior Card": 300000,
    "Orc Archer Card": 200000,
    "Orc Lady Card": 800000,
    "Geographer Card": 500000,
    "Soldier Skeleton Card": 1000000,
    "Munak Card": 150000,
    "Bongun Card": 120000,
    "Ghoul Card": 250000,
    "Zombie Card": 90000,
    "Marina Card": 60000,
    "Plankton Card": 30000,
    "Kukre Card": 50000,
    "Hydra Card": 200000,
    "Drainliar Card": 80000,
    "Stapo Card": 40000,
    "Anacondaq Card": 70000,
    "Alligator Card": 100000,
    "Peco Peco Card": 120000,
    "Pickie Card": 50000,
    "Steel Chonchon Card": 60000,
    "Orc Zombie Card": 180000,
    "Mummy Card": 200000,
    "Myst Card": 150000,
    "Argos Card": 80000,
    "Deniro Card": 40000,
    "Piere Card": 50000,
    "Andre Card": 45000,
    "Vitata Card": 35000,
    "Metaller Card": 30000,
    "Poison Spore Card": 70000,
    "Magnolia Card": 90000,
    "Martin Card": 55000,
    "Thief Bug Egg Card": 20000,
}

MATERIALS: dict[str, int] = {
    "Jellopy": 1,          # Vendor trash
    "Sticky Mucus": 50,
    "Memento": 100,
    "Empty Bottle": 200,
    "White Herb": 300,
    "Blue Herb": 500,
    "Red Herb": 100,
    "Green Herb": 150,
    "Yellow Herb": 200,
    "Iron": 500,
    "Coal": 2000,
    "Steel": 3000,
    "Feather": 10,
    "Shell": 5,
    "Dewdrop": 20,
    "Sap": 30,
    "Hinalle": 500,
    "Aloe": 1000,
    "Mastela Fruit": 2000,
    "Yggdrasil Seed": 5000,
    "Yggdrasil Berry": 10000,
    "Rough Oridecon": 5000,
    "Oridecon": 15000,
    "Rough Elunium": 3000,
    "Elunium": 10000,
    "Mithril": 8000,
    "Gold": 5000,
    "Emveretarcon": 2000,
    "Brigan": 1000,
    "Cacatua": 500,
    "Poppy": 300,
    "Apple": 50,
    "Banana": 30,
    "Grape": 20,
    "Carrot": 10,
    "Potato": 5,
    "Meat": 3,
    "Honey": 2,
    "Milk": 100,
    "Cheese": 200,
    "Bitter Herb": 50,
    "Aloe Leaflet": 100,
    "Garlet": 200,
    "Izidor": 300,
    "Kafra Leaf": 500,
    "Mora Leaf": 1000,
    "Mora Seed": 2000,
    "Mora Berry": 5000,
    "Mora Root": 10000,
}

EQUIPMENT: dict[str, dict[str, int]] = {
    # Slot weapons (varies by type and slots)
    "Knife [3]": {"buy": 1000, "sell": 50000},
    "Sword [3]": {"buy": 2000, "sell": 100000},
    "Bow [3]": {"buy": 2000, "sell": 100000},
    "Mace [3]": {"buy": 1500, "sell": 75000},
    "Rod [3]": {"buy": 1500, "sell": 75000},
    "Main Gauche [3]": {"buy": 5000, "sell": 150000},
    "Blade [3]": {"buy": 8000, "sell": 200000},
    "Composite Bow [3]": {"buy": 8000, "sell": 200000},
    "Scimitar [2]": {"buy": 15000, "sell": 300000},
    "Crossbow [2]": {"buy": 15000, "sell": 300000},
    "Damascus [1]": {"buy": 30000, "sell": 500000},
    "Flail [2]": {"buy": 10000, "sell": 250000},
    "Staff [2]": {"buy": 10000, "sell": 250000},
    "Wand [2]": {"buy": 20000, "sell": 400000},
    # Non-slot weapons
    "Knife": {"buy": 500, "sell": 5000},
    "Sword": {"buy": 1000, "sell": 10000},
    "Bow": {"buy": 1000, "sell": 10000},
    "Mace": {"buy": 800, "sell": 8000},
    "Rod": {"buy": 800, "sell": 8000},
    # Armor
    "Cotton Shirt": {"buy": 1000, "sell": 10000},
    "Manteau": {"buy": 5000, "sell": 50000},
    "Boots": {"buy": 3000, "sell": 30000},
    "Buckler": {"buy": 2000, "sell": 20000},
    "Guard [1]": {"buy": 10000, "sell": 100000},
    "Hood [1]": {"buy": 8000, "sell": 80000},
    "Shoes [1]": {"buy": 12000, "sell": 120000},
    "Muffler [1]": {"buy": 15000, "sell": 150000},
    "Plate Armor": {"buy": 20000, "sell": 200000},
    "Chain Mail": {"buy": 30000, "sell": 300000},
    "Full Plate": {"buy": 50000, "sell": 500000},
    "Robe of Cast": {"buy": 40000, "sell": 400000},
    "Silk Robe": {"buy": 25000, "sell": 250000},
    "Wooden Mail": {"buy": 15000, "sell": 150000},
    "Tights [1]": {"buy": 60000, "sell": 600000},
    "Pantie [1]": {"buy": 10000, "sell": 100000},
    "Biretta": {"buy": 5000, "sell": 50000},
    "Hat": {"buy": 3000, "sell": 30000},
    "Helm [1]": {"buy": 20000, "sell": 200000},
    "Goggles": {"buy": 8000, "sell": 80000},
    "Sunglasses": {"buy": 5000, "sell": 50000},
    "Ribbon [1]": {"buy": 10000, "sell": 100000},
    "Filigree F. Hat [1]": {"buy": 15000, "sell": 150000},
    "Bunny Band": {"buy": 20000, "sell": 200000},
    "Cat Ear Beret": {"buy": 50000, "sell": 500000},
    "Angel Wing": {"buy": 100000, "sell": 1000000},
    "Devil Wing": {"buy": 100000, "sell": 1000000},
    "Santa Hat": {"buy": 30000, "sell": 300000},
    "Poring Hat": {"buy": 50000, "sell": 500000},
    "Drooping Cat": {"buy": 80000, "sell": 800000},
    "Majestic Goat": {"buy": 200000, "sell": 2000000},
    "Crown": {"buy": 500000, "sell": 5000000},
    "Tiara": {"buy": 300000, "sell": 3000000},
    "Circlet": {"buy": 100000, "sell": 1000000},
    "Elven Ears": {"buy": 50000, "sell": 500000},
    "Spirit Chain": {"buy": 80000, "sell": 800000},
    "Necklace": {"buy": 20000, "sell": 200000},
    "Ring": {"buy": 30000, "sell": 300000},
    "Earring": {"buy": 25000, "sell": 250000},
    "Brooch": {"buy": 40000, "sell": 400000},
    "Glove": {"buy": 35000, "sell": 350000},
    "Rosary": {"buy": 15000, "sell": 150000},
    "Skull Ring": {"buy": 50000, "sell": 500000},
    "Safety Ring": {"buy": 60000, "sell": 600000},
    "Vesper Core 01": {"buy": 100000, "sell": 1000000},
    "Vesper Core 02": {"buy": 100000, "sell": 1000000},
    "Vesper Core 03": {"buy": 150000, "sell": 1500000},
    "Vesper Core 04": {"buy": 150000, "sell": 1500000},
}


# ── Market Seeder ────────────────────────────────────────────────────────────


@dataclass
class MarketSeeder:
    """Seeds realistic market prices based on server type and age.

    Provides:
      - seed_prices(): Returns a dict of item_name -> market_price
      - apply_to_db(db): Updates an ItemValueDB with seeded prices
      - Price adjustments for server rate (low/mid/high)
      - Price adjustments for server age (fresh/established/old)

    Default: mid_rate, established.
    """

    server_rate: ServerRate = ServerRate.MID_RATE
    server_age: ServerAge = ServerAge.ESTABLISHED

    # ── Public API ───────────────────────────────────────────────────────────

    def seed_prices(self) -> dict[str, int]:
        """Generate a complete price dictionary for all known items.

        Returns:
            dict of item_name -> market_price (in zeny), adjusted for
            server rate and age.
        """
        prices: dict[str, int] = {}

        # Get multipliers
        rate_mult = SERVER_RATE_MULTIPLIERS.get(self.server_rate, 1.0)
        age_mult = SERVER_AGE_MULTIPLIERS.get(self.server_age, 1.0)

        # Consumables
        for name, prices_dict in CONSUMABLES.items():
            # Check for rate-specific overrides
            if self.server_rate == ServerRate.LOW_RATE and name in CONSUMABLES_LOW_RATE:
                sell_price = CONSUMABLES_LOW_RATE[name]["sell"]
            elif self.server_rate == ServerRate.HIGH_RATE and name in CONSUMABLES_HIGH_RATE:
                sell_price = CONSUMABLES_HIGH_RATE[name]["sell"]
            else:
                sell_price = prices_dict["sell"]

            # Apply multipliers
            adjusted = int(sell_price * rate_mult * age_mult)
            prices[name] = max(1, adjusted)

        # Cards
        for name, base_price in CARDS.items():
            # Cards are heavily affected by server rate and age
            adjusted = int(base_price * rate_mult * age_mult)
            prices[name] = max(100, adjusted)

        # Materials
        for name, base_price in MATERIALS.items():
            # Materials are affected by server age (scarcity)
            adjusted = int(base_price * age_mult)
            prices[name] = max(1, adjusted)

        # Equipment
        for name, prices_dict in EQUIPMENT.items():
            sell_price = prices_dict["sell"]
            adjusted = int(sell_price * rate_mult * age_mult)
            prices[name] = max(100, adjusted)

        return prices

    def seed_consumables(self) -> dict[str, int]:
        """Get only consumable prices."""
        prices: dict[str, int] = {}
        rate_mult = SERVER_RATE_MULTIPLIERS.get(self.server_rate, 1.0)
        age_mult = SERVER_AGE_MULTIPLIERS.get(self.server_age, 1.0)

        for name, prices_dict in CONSUMABLES.items():
            if self.server_rate == ServerRate.LOW_RATE and name in CONSUMABLES_LOW_RATE:
                sell_price = CONSUMABLES_LOW_RATE[name]["sell"]
            elif self.server_rate == ServerRate.HIGH_RATE and name in CONSUMABLES_HIGH_RATE:
                sell_price = CONSUMABLES_HIGH_RATE[name]["sell"]
            else:
                sell_price = prices_dict["sell"]
            adjusted = int(sell_price * rate_mult * age_mult)
            prices[name] = max(1, adjusted)

        return prices

    def seed_cards(self) -> dict[str, int]:
        """Get only card prices."""
        prices: dict[str, int] = {}
        rate_mult = SERVER_RATE_MULTIPLIERS.get(self.server_rate, 1.0)
        age_mult = SERVER_AGE_MULTIPLIERS.get(self.server_age, 1.0)

        for name, base_price in CARDS.items():
            adjusted = int(base_price * rate_mult * age_mult)
            prices[name] = max(100, adjusted)

        return prices

    def seed_materials(self) -> dict[str, int]:
        """Get only material prices."""
        prices: dict[str, int] = {}
        age_mult = SERVER_AGE_MULTIPLIERS.get(self.server_age, 1.0)

        for name, base_price in MATERIALS.items():
            adjusted = int(base_price * age_mult)
            prices[name] = max(1, adjusted)

        return prices

    def seed_equipment(self) -> dict[str, int]:
        """Get only equipment prices."""
        prices: dict[str, int] = {}
        rate_mult = SERVER_RATE_MULTIPLIERS.get(self.server_rate, 1.0)
        age_mult = SERVER_AGE_MULTIPLIERS.get(self.server_age, 1.0)

        for name, prices_dict in EQUIPMENT.items():
            sell_price = prices_dict["sell"]
            adjusted = int(sell_price * rate_mult * age_mult)
            prices[name] = max(100, adjusted)

        return prices

    def apply_to_db(self, db: Any) -> None:
        """Update an ItemValueDB with seeded market prices.

        This method updates the market_value field of items in the database
        to match the seeded prices. It's designed to work with the existing
        ItemValueDB class from ai_sidecar.economy.item_value_db.

        Args:
            db: An ItemValueDB instance (or any object with an
                ``items`` dict of ``{name: ItemValuation}``).
        """
        prices = self.seed_prices()

        # Try to update via the items dict
        if hasattr(db, "items") and isinstance(db.items, dict):
            for item_name, market_price in prices.items():
                # Try exact match
                if item_name in db.items:
                    db.items[item_name].market_value = market_price
                    continue

                # Try case-insensitive match
                for db_name, valuation in db.items.items():
                    if db_name.lower() == item_name.lower():
                        valuation.market_value = market_price
                        break

        # Try to update via a method if available
        if hasattr(db, "set_market_price"):
            for item_name, market_price in prices.items():
                try:
                    db.set_market_price(item_name, market_price)
                except Exception:
                    pass

        logger.info(
            "MarketSeeder applied %d prices to DB (rate=%s, age=%s)",
            len(prices),
            self.server_rate.value,
            self.server_age.value,
        )

    def get_price(self, item_name: str) -> int | None:
        """Get the seeded market price for a single item.

        Args:
            item_name: Name of the item.

        Returns:
            Market price in zeny, or None if item is not in the seed data.
        """
        prices = self.seed_prices()
        return prices.get(item_name)

    def get_buy_price(self, item_name: str) -> int | None:
        """Get the NPC buy price for a single item (what you'd pay at NPC shop).

        Args:
            item_name: Name of the item.

        Returns:
            NPC buy price in zeny, or None if item is not in seed data.
        """
        # Check consumables
        if item_name in CONSUMABLES:
            return CONSUMABLES[item_name]["buy"]

        # Check equipment
        if item_name in EQUIPMENT:
            return EQUIPMENT[item_name]["buy"]

        # Cards and materials don't have NPC buy prices
        return None

    def get_sell_price(self, item_name: str) -> int | None:
        """Get the NPC sell price for a single item (what NPC pays you).

        Args:
            item_name: Name of the item.

        Returns:
            NPC sell price in zeny, or None if item is not in seed data.
        """
        # Check consumables
        if item_name in CONSUMABLES:
            return CONSUMABLES[item_name]["sell"]

        # Check equipment
        if item_name in EQUIPMENT:
            return EQUIPMENT[item_name]["sell"]

        # Cards
        if item_name in CARDS:
            return CARDS[item_name]

        # Materials
        if item_name in MATERIALS:
            return MATERIALS[item_name]

        return None

    def get_item_category(self, item_name: str) -> str | None:
        """Get the category of an item.

        Returns:
            'consumable', 'card', 'material', 'equipment', or None.
        """
        if item_name in CONSUMABLES:
            return "consumable"
        if item_name in CARDS:
            return "card"
        if item_name in MATERIALS:
            return "material"
        if item_name in EQUIPMENT:
            return "equipment"
        return None

    def get_top_items_by_value(self, limit: int = 20) -> list[tuple[str, int, str]]:
        """Get the most valuable items from the seed data.

        Args:
            limit: Maximum number of items to return.

        Returns:
            List of (item_name, market_price, category) tuples, sorted
            by price descending.
        """
        prices = self.seed_prices()
        items_with_cat: list[tuple[str, int, str]] = []

        for name, price in prices.items():
            cat = self.get_item_category(name) or "unknown"
            items_with_cat.append((name, price, cat))

        items_with_cat.sort(key=lambda x: x[1], reverse=True)
        return items_with_cat[:limit]

    def get_cheapest_items(self, limit: int = 20) -> list[tuple[str, int, str]]:
        """Get the cheapest items from the seed data.

        Args:
            limit: Maximum number of items to return.

        Returns:
            List of (item_name, market_price, category) tuples, sorted
            by price ascending.
        """
        prices = self.seed_prices()
        items_with_cat: list[tuple[str, int, str]] = []

        for name, price in prices.items():
            cat = self.get_item_category(name) or "unknown"
            items_with_cat.append((name, price, cat))

        items_with_cat.sort(key=lambda x: x[1])
        return items_with_cat[:limit]

    def get_vendor_trash_items(self) -> list[str]:
        """Get items that are vendor trash (worth < 100z).

        Returns:
            List of item names that are essentially worthless.
        """
        prices = self.seed_prices()
        return [name for name, price in prices.items() if price < 100]

    def get_profitable_items(self, min_profit: int = 1000) -> list[tuple[str, int, str]]:
        """Get items worth farming (market price significantly above vendor price).

        Args:
            min_profit: Minimum profit margin in zeny.

        Returns:
            List of (item_name, profit_margin, category) tuples.
        """
        profitable: list[tuple[str, int, str]] = []

        for name in CONSUMABLES:
            buy = CONSUMABLES[name]["buy"]
            sell = CONSUMABLES[name]["sell"]
            profit = sell - buy
            if profit >= min_profit:
                profitable.append((name, profit, "consumable"))

        for name in EQUIPMENT:
            buy = EQUIPMENT[name]["buy"]
            sell = EQUIPMENT[name]["sell"]
            profit = sell - buy
            if profit >= min_profit:
                profitable.append((name, profit, "equipment"))

        profitable.sort(key=lambda x: x[1], reverse=True)
        return profitable


# ── Singleton factory ───────────────────────────────────────────────────────

_market_seeder: MarketSeeder | None = None


def get_market_seeder(
    server_rate: ServerRate = ServerRate.MID_RATE,
    server_age: ServerAge = ServerAge.ESTABLISHED,
) -> MarketSeeder:
    """Get or create the singleton MarketSeeder.

    Args:
        server_rate: Server rate classification.
        server_age: Server age classification.

    Returns:
        MarketSeeder instance.
    """
    global _market_seeder
    if _market_seeder is None:
        _market_seeder = MarketSeeder(server_rate=server_rate, server_age=server_age)
    return _market_seeder
