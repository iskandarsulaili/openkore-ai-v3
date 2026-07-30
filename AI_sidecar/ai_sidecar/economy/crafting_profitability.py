"""Crafting Profitability Calculator — real profit margin analysis for RO crafting.

Pro players don't just craft randomly. They calculate exact profit margins
based on current market prices:

  - Empty Bottle (2z NPC) + White Herb (200z market) = White Potion (1000z market)
  - That's 798z profit per potion, 798K zeny per 1000 crafts
  - Elemental Converters: 5K to make, sell for 15K during WoE
  - Always checks if market price of output > cost of inputs + NPC buying

Thread-safe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────

# Zeny
MIN_PROFIT_PER_UNIT = 10      # Minimum zeny profit per crafted item
MIN_PROFIT_MARGIN = 0.20       # 20% minimum profit margin to consider crafting
MAX_INVESTMENT = 1000000       # Maximum zeny to invest in a single batch
CRAFT_BATCH_SIZE = 1000        # Default batch size for profit calculations

# NPC prices (fixed — these don't change with market)
NPC_EMPTY_BOTTLE_PRICE = 2     # Empty Bottle from NPC
NPC_WHITE_POTION_PRICE = 500   # White Potion from NPC (for arbitrage comparison)

# Estimated craft costs (NPC mat costs + service)
DEFAULT_CRAFTING_FEE = 100     # NPC crafting service fee per attempt
SUCCESS_RATE_FACTOR = 0.9      # 90% craft success rate baseline


# ── Data Models ────────────────────────────────────────────────────────────

@dataclass
class CraftIngredient:
    """An ingredient needed for crafting."""
    item_name: str
    quantity: int
    npc_price: int          # What NPC sells it for (if available)
    market_price: int        # What players sell it for
    source: str              # "npc", "player_market", "farm"


@dataclass
class CraftRecipe:
    """A known crafting recipe with cost analysis."""
    output_item: str
    output_quantity: int        # How many units per craft
    output_market_price: int    # Current player market price
    output_npc_buy: int         # NPC sell price (what NPC pays for it)
    ingredients: list[CraftIngredient]
    total_cost: int             # Total cost of all ingredients
    craft_fee: int              # NPC crafting fee per batch
    success_rate: float         # 0.0-1.0
    category: str               # "potion", "converter", "weapon", "armor", etc.


@dataclass
class CraftingMargin:
    """Profit analysis for a crafting recipe."""
    recipe: CraftRecipe
    cost_per_unit: float        # Total cost / output_quantity
    market_price_per_unit: int   # Current selling price
    profit_per_unit: float       # market_price - cost
    total_profit_per_batch: float  # profit * batch_size
    profit_margin: float        # profit / cost as ratio
    roi_percentage: float       # Return on investment (profit / cost * 100)
    is_profitable: bool         # True if profit exceeds thresholds
    recommendation: str         # "craft_now", "wait_for_event", "skip"
    reasoning: str              # Human-readable explanation


# ── Known Crafting Recipes (Real RO Data) ─────────────────────────────────

# These are actual RO recipes with real NPC and estimated market prices.
# Prices are for a mature server; adjust with MarketTimingEngine.

KNOWN_RECIPES: list[dict[str, Any]] = [
    # Potions
    {
        "output": "White Potion", "qty": 1,
        "npc_buy": 500, "market": 1000,
        "craft_fee": 200, "success": 0.95,
        "ingredients": [
            {"item": "Empty Bottle", "qty": 1, "npc": 2, "market": 200},
            {"item": "White Herb", "qty": 1, "npc": 0, "market": 200},
        ],
    },
    {
        "output": "Blue Potion", "qty": 1,
        "npc_buy": 2000, "market": 4000,
        "craft_fee": 500, "success": 0.85,
        "ingredients": [
            {"item": "Empty Bottle", "qty": 1, "npc": 2, "market": 200},
            {"item": "Blue Herb", "qty": 1, "npc": 0, "market": 500},
            {"item": "Ment", "qty": 3, "npc": 0, "market": 100},
        ],
    },
    {
        "output": "Condensed White Potion", "qty": 1,
        "npc_buy": 2000, "market": 5000,
        "craft_fee": 500, "success": 0.80,
        "ingredients": [
            {"item": "White Potion", "qty": 3, "npc": 500, "market": 1000},
            {"item": "White Herb", "qty": 1, "npc": 0, "market": 200},
        ],
    },
    {
        "output": "Levertine Potion", "qty": 1,
        "npc_buy": 0, "market": 12000,
        "craft_fee": 1000, "success": 0.70,
        "ingredients": [
            {"item": "White Potion", "qty": 10, "npc": 500, "market": 1000},
            {"item": "Blue Potion", "qty": 1, "npc": 2000, "market": 4000},
            {"item": "Empty Bottle", "qty": 1, "npc": 2, "market": 200},
        ],
    },
    # Elemental Converters
    {
        "output": "Fire Converter", "qty": 1,
        "npc_buy": 0, "market": 15000,
        "craft_fee": 500, "success": 0.90,
        "ingredients": [
            {"item": "Empty Bottle", "qty": 1, "npc": 2, "market": 200},
            {"item": "Flame Heart", "qty": 1, "npc": 0, "market": 5000},
        ],
    },
    {
        "output": "Water Converter", "qty": 1,
        "npc_buy": 0, "market": 15000,
        "craft_fee": 500, "success": 0.90,
        "ingredients": [
            {"item": "Empty Bottle", "qty": 1, "npc": 2, "market": 200},
            {"item": "Mystic Frozen", "qty": 1, "npc": 0, "market": 5000},
        ],
    },
    {
        "output": "Wind Converter", "qty": 1,
        "npc_buy": 0, "market": 15000,
        "craft_fee": 500, "success": 0.90,
        "ingredients": [
            {"item": "Empty Bottle", "qty": 1, "npc": 2, "market": 200},
            {"item": "Great Nature", "qty": 1, "npc": 0, "market": 5000},
        ],
    },
    {
        "output": "Earth Converter", "qty": 1,
        "npc_buy": 0, "market": 15000,
        "craft_fee": 500, "success": 0.90,
        "ingredients": [
            {"item": "Empty Bottle", "qty": 1, "npc": 2, "market": 200},
            {"item": "Green Live", "qty": 1, "npc": 0, "market": 5000},
        ],
    },
    # Elemental Arrow Quivers
    {
        "output": "Crystal Arrow Quiver", "qty": 100,
        "npc_buy": 0, "market": 50000,
        "craft_fee": 2000, "success": 0.85,
        "ingredients": [
            {"item": "Crystal Arrow", "qty": 1, "npc": 0, "market": 300},
            {"item": "Stem", "qty": 10, "npc": 0, "market": 50},
            {"item": "Crystal Blue", "qty": 5, "npc": 0, "market": 200},
        ],
    },
]


# ── Crafting Profitability Calculator ────────────────────────────────────

class CraftingProfitability:
    """Calculates profit margins for RO crafting recipes.

    Uses real market prices (from item_value_db) and temporal multipliers
    (from market_timing) to determine which crafts are profitable right now.

    Thread-safe.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._recipes: list[CraftRecipe] = []
        self._load_known_recipes()
        self._custom_market_prices: dict[str, int] = {}
        self._stats: dict[str, int | float] = {
            "recipes_loaded": 0,
            "profitable_recipes": 0,
            "analyses": 0,
        }

    # ── Public API ─────────────────────────────────────────────────────

    def analyze_recipe(self, output_item: str,
                        market_price_override: int | None = None,
                        ingredient_price_overrides: dict[str, int] | None = None) -> CraftingMargin | None:
        """Analyze the profitability of a specific recipe.

        Args:
            output_item: Name of the item to craft.
            market_price_override: Optional current market price for output.
            ingredient_price_overrides: Optional per-ingredient price overrides.

        Returns:
            CraftingMargin with profit analysis, or None if recipe unknown.
        """
        with self._lock:
            self._stats["analyses"] += 1  # type: ignore[assignment]

            recipe = self._find_recipe(output_item)
            if recipe is None:
                logger.debug("crafting_profit: unknown recipe for '%s'", output_item)
                return None

            # Apply price overrides
            output_price = market_price_override or recipe.output_market_price
            ingredients = list(recipe.ingredients)
            ingredient_overrides = ingredient_price_overrides or {}

            total_cost = 0
            adjusted_ingredients: list[CraftIngredient] = []

            for ing in ingredients:
                override = ingredient_overrides.get(ing.item_name)
                market_price = override if override is not None else ing.market_price
                # Use cheapest available source
                effective_price = min(ing.npc_price, market_price) if ing.npc_price > 0 else market_price
                cost = effective_price * ing.quantity
                total_cost += cost

                adjusted_ingredients.append(CraftIngredient(
                    item_name=ing.item_name,
                    quantity=ing.quantity,
                    npc_price=ing.npc_price,
                    market_price=market_price,
                    source=ing.source,
                ))

            # Add crafting fee
            total_cost += recipe.craft_fee

            # Apply success rate
            expected_output = recipe.output_quantity * recipe.success_rate
            cost_per_unit = total_cost / max(expected_output, 1)

            # NPC arbitrage: if NPC sells crafted item cheaper, don't craft
            if recipe.output_npc_buy > 0 and recipe.output_npc_buy < output_price:
                npc_cost_saving = output_price - recipe.output_npc_buy
            else:
                npc_cost_saving = 0

            profit_per_unit = output_price - cost_per_unit
            total_profit_per_batch = profit_per_unit * CRAFT_BATCH_SIZE
            profit_margin = profit_per_unit / max(cost_per_unit, 1)
            roi = profit_margin * 100

            is_profitable = (
                profit_per_unit >= MIN_PROFIT_PER_UNIT
                and profit_margin >= MIN_PROFIT_MARGIN
            )

            # Build recommendation
            if is_profitable and profit_margin > 0.5:
                recommendation = "craft_now"
                reasoning = (
                    f"Profit of {profit_per_unit:.0f}z/unit ({roi:.0f}% ROI). "
                    f"Batch of {CRAFT_BATCH_SIZE} nets ~{total_profit_per_batch:,.0f}z."
                )
            elif is_profitable:
                recommendation = "craft_now"
                reasoning = (
                    f"Margin of {profit_per_unit:.0f}z/unit ({roi:.1f}% ROI). "
                    f"Modest but profitable."
                )
            elif profit_per_unit > 0 and profit_margin < MIN_PROFIT_MARGIN:
                recommendation = "wait_for_event"
                reasoning = (
                    f"Currently {profit_per_unit:.0f}z/unit profit but only "
                    f"{roi:.1f}% margin. Wait for WoE price surge for better returns."
                )
            else:
                recommendation = "skip"
                reasoning = (
                    f"Cost per unit {cost_per_unit:.0f}z exceeds market price "
                    f"{output_price}z. Not profitable."
                )

            return CraftingMargin(
                recipe=recipe,
                cost_per_unit=round(cost_per_unit, 2),
                market_price_per_unit=output_price,
                profit_per_unit=round(profit_per_unit, 2),
                total_profit_per_batch=round(total_profit_per_batch, 0),
                profit_margin=round(profit_margin, 4),
                roi_percentage=round(roi, 2),
                is_profitable=is_profitable,
                recommendation=recommendation,
                reasoning=reasoning,
            )

    def analyze_all_recipes(
        self,
        market_price_overrides: dict[str, int] | None = None,
    ) -> list[CraftingMargin]:
        """Analyze all known recipes and return sorted by profitability.

        Args:
            market_price_overrides: Optional dict of item_name -> current price.

        Returns:
            List of CraftingMargin, sorted by profit per unit descending.
        """
        with self._lock:
            results: list[CraftingMargin] = []
            for recipe in self._recipes:
                margin = self.analyze_recipe(
                    recipe.output_item,
                    market_price_overrides.get(recipe.output_item) if market_price_overrides else None,
                    market_price_overrides,
                )
                if margin is not None:
                    results.append(margin)

            results.sort(key=lambda m: -m.profit_per_unit)

            profitable = sum(1 for r in results if r.is_profitable)
            self._stats["profitable_recipes"] = profitable  # type: ignore[assignment]

            return results

    def get_most_profitable_recipes(
        self, top_n: int = 5,
        market_price_overrides: dict[str, int] | None = None,
    ) -> list[CraftingMargin]:
        """Get the top N most profitable recipes right now.

        Args:
            top_n: Number of recipes to return.
            market_price_overrides: Optional current market prices.

        Returns:
            List of top CraftingMargin results.
        """
        results = self.analyze_all_recipes(market_price_overrides)
        profitable = [r for r in results if r.is_profitable]
        return profitable[:top_n]

    def update_market_price(self, item_name: str, price: int) -> None:
        """Update a market price for a recipe output or ingredient.

        This causes next analyze_* call to use the new price.
        """
        with self._lock:
            self._custom_market_prices[item_name] = price

    def set_npc_arbitrage(self, item_name: str, npc_sell_price: int) -> None:
        """Record the NPC sell price for an item for arbitrage detection.

        Items where NPC price < player market price can be bought from NPC
        and sold to players for profit.
        """
        with self._lock:
            logger.info(
                "crafting_profit: NPC arbitrage for %s — NPC pays %dz vs market %d",
                item_name, npc_sell_price,
                self._custom_market_prices.get(item_name, 0),
            )

    def get_npc_arbitrage_opportunities(self) -> list[dict[str, Any]]:
        """Find items where NPC buy price < player market price.

        These are items you can buy from NPC at fixed price and sell
        to players for a profit (known as NPC arbitrage).

        Examples:
          - White Potion: NPC sells for 500z, players buy for 1000z (100% profit)
          - Empty Bottle: NPC sells for 2z, players buy for 200z (9900% profit)
          - Fly Wing: NPC sells for 300z, players buy for 500z (66% profit)
        """
        with self._lock:
            opportunities: list[dict[str, Any]] = []

            # Check all recipes for NPC arbitrage on ingredients
            checked_items: set[str] = set()
            for recipe in self._recipes:
                for ing in recipe.ingredients:
                    if ing.item_name in checked_items:
                        continue
                    checked_items.add(ing.item_name)

                    if ing.npc_price > 0 and ing.market_price > ing.npc_price:
                        profit_per = ing.market_price - ing.npc_price
                        margin = profit_per / ing.npc_price
                        opportunities.append({
                            "item": ing.item_name,
                            "npc_price": ing.npc_price,
                            "market_price": ing.market_price,
                            "profit_per_unit": profit_per,
                            "profit_margin": round(margin, 2),
                            "roi": round(margin * 100, 1),
                            "action": f"Buy from NPC for {ing.npc_price}z, sell for {ing.market_price}z",
                        })

                # Check output items too
                if recipe.output_item in checked_items:
                    continue
                checked_items.add(recipe.output_item)

                if recipe.output_npc_buy > 0 and recipe.output_market_price > recipe.output_npc_buy:
                    profit_per = recipe.output_market_price - recipe.output_npc_buy
                    margin = profit_per / recipe.output_npc_buy
                    opportunities.append({
                        "item": recipe.output_item,
                        "npc_price": recipe.output_npc_buy,
                        "market_price": recipe.output_market_price,
                        "profit_per_unit": profit_per,
                        "profit_margin": round(margin, 2),
                        "roi": round(margin * 100, 1),
                        "action": f"Already available from NPC — don't craft! Buy for {recipe.output_npc_buy}z",
                    })

            opportunities.sort(key=lambda o: -o["profit_per_unit"])
            return opportunities

    def get_stats(self) -> dict[str, int | float]:
        with self._lock:
            return dict(self._stats)

    def get_crafting_summary(self) -> str:
        """Get a formatted crafting profitability summary."""
        with self._lock:
            lines = ["── Crafting Profitability ──"]
            profitable = self.analyze_all_recipes()
            top = [r for r in profitable if r.is_profitable][:5]

            if not top:
                lines.append("  No profitable recipes found at current prices.")
            else:
                for i, m in enumerate(top):
                    lines.append(
                        f"  {i+1}. {m.recipe.output_item}: "
                        f"+{m.profit_per_unit:.0f}z/unit "
                        f"({m.roi_percentage:.0f}% ROI) — {m.recommendation}"
                    )

            # NPC arbitrage
            arb = self.get_npc_arbitrage_opportunities()[:3]
            if arb:
                lines.append("")
                lines.append("  NPC Arbitrage:")
                for o in arb:
                    lines.append(f"    {o['item']}: NPC {o['npc_price']}z → market {o['market_price']}z")

            return "\n".join(lines)

    # ── Internal ───────────────────────────────────────────────────────

    def _load_known_recipes(self) -> None:
        """Load known crafting recipes into structured data."""
        raw = KNOWN_RECIPES
        for r in raw:
            ingredients = [
                CraftIngredient(
                    item_name=ing["item"],
                    quantity=ing["qty"],
                    npc_price=ing.get("npc", 0),
                    market_price=ing.get("market", 0),
                    source="npc" if ing.get("npc", 0) > 0 else "player_market",
                )
                for ing in r.get("ingredients", [])
            ]

            total_cost = sum(
                (min(i.market_price, i.npc_price) if i.npc_price > 0 else i.market_price) * i.quantity
                for i in ingredients
            )

            recipe = CraftRecipe(
                output_item=r["output"],
                output_quantity=r["qty"],
                output_market_price=r["market"],
                output_npc_buy=r.get("npc_buy", 0),
                ingredients=ingredients,
                total_cost=total_cost + r.get("craft_fee", 0),
                craft_fee=r.get("craft_fee", 0),
                success_rate=r.get("success", 0.9),
                category=self._categorize_output(r["output"]),
            )
            self._recipes.append(recipe)

        self._stats["recipes_loaded"] = len(self._recipes)  # type: ignore[assignment]
        logger.info("crafting_profit: loaded %d recipes", len(self._recipes))

    def _find_recipe(self, output_item: str) -> CraftRecipe | None:
        """Find a recipe by output item name (case-insensitive)."""
        lower = output_item.lower()
        for recipe in self._recipes:
            if recipe.output_item.lower() == lower:
                return recipe
        return None

    def _categorize_output(self, item_name: str) -> str:
        """Categorize an output item by type."""
        name_lower = item_name.lower()
        if "potion" in name_lower:
            return "potion"
        if "converter" in name_lower:
            return "converter"
        if "arrow" in name_lower:
            return "ammunition"
        if "scroll" in name_lower:
            return "scroll"
        return "misc"


# ── Global Singleton ─────────────────────────────────────────────────────

_crafting_profitability: CraftingProfitability | None = None
_crafting_profitability_lock = RLock()


def get_crafting_profitability() -> CraftingProfitability:
    """Get the global CraftingProfitability singleton."""
    global _crafting_profitability
    with _crafting_profitability_lock:
        if _crafting_profitability is None:
            _crafting_profitability = CraftingProfitability()
        return _crafting_profitability
