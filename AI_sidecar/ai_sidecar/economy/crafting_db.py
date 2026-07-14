"""Crafting for Profit — potions, converters, arrows, and more.

A pro player doesn't just farm raw zeny. They craft:
- Potions from herbs and bottles (2-3x profit margin)
- Elemental converters from materials (5-10x profit margin)
- Arrows from materials (steady demand)
- Food/scrolls for WoE prep

This module catalogs profitable crafting recipes and computes profit margins.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from ai_sidecar.economy.item_value_db import get_item_value_db, ItemValueDB
from ai_sidecar.economy.npc_shop_db import get_npc_shop_db, NPCShopDB

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────

# Minimum profit margin to consider a recipe worthwhile
MIN_CRAFT_MARGIN = 0.5  # 50%

# Minimum absolute profit per craft
MIN_CRAFT_PROFIT = 500  # 500z

# Maximum risk for crafting (0.0-1.0)
MAX_CRAFT_RISK = 0.5


# ── Dataclasses ───────────────────────────────────────────────────────────


@dataclass
class CraftingIngredient:
    """A single ingredient in a crafting recipe."""
    item_name: str
    quantity: int
    estimated_cost: int  # Per unit
    source: str  # npc, farm, market


@dataclass
class CraftingRecipe:
    """A crafting recipe with profit analysis."""
    output_item: str
    output_quantity: int  # How many units produced
    output_market_value: int  # Estimated market price per unit
    ingredients: list[CraftingIngredient]
    total_cost: int  # Total cost of all ingredients
    profit_per_craft: int  # Total profit for one craft cycle
    profit_margin_pct: float  # (revenue - cost) / cost * 100
    required_level: int  # Minimum level/class requirement
    required_skill: str  # Skill needed to craft
    difficulty: str  # easy, medium, hard
    risk_level: str  # low, medium, high
    demand: str  # low, medium, high (player demand)
    category: str  # potion, converter, arrow, food, scroll


# ── Crafting Database ─────────────────────────────────────────────────────


@dataclass(slots=True)
class CraftingDB:
    """Database of profitable crafting recipes.

    Uses ItemValueDB for item valuations and NPCShopDB for ingredient costs.
    Thread-safe.
    """

    _lock: RLock = field(default_factory=RLock)
    _item_db: ItemValueDB = field(default_factory=get_item_value_db)
    _npc_db: NPCShopDB = field(default_factory=get_npc_shop_db)
    _recipes: list[CraftingRecipe] = field(default_factory=list)
    _profitable_recipes: list[CraftingRecipe] = field(default_factory=list)
    _stats: dict[str, int] = field(default_factory=lambda: {"recipes_loaded": 0, "profitable": 0})

    def __post_init__(self) -> None:
        self._load_recipes()

    def _load_recipes(self) -> None:
        """Load all known crafting recipes and compute profitability."""
        recipes: list[CraftingRecipe] = []

        # ── Potions (Alchemist/Creator) ──
        recipes.extend(self._potion_recipes())

        # ── Elemental Converters ──
        recipes.extend(self._converter_recipes())

        # ── Arrows ──
        recipes.extend(self._arrow_recipes())

        # ── Food (Cook) ──
        recipes.extend(self._food_recipes())

        # ── Scrolls (Sage/Professor) ──
        recipes.extend(self._scroll_recipes())

        self._recipes = recipes
        self._stats["recipes_loaded"] = len(recipes)

        # Compute profitability for each recipe
        profitable = []
        for recipe in recipes:
            if recipe.profit_per_craft >= MIN_CRAFT_PROFIT and recipe.profit_margin_pct >= MIN_CRAFT_MARGIN * 100:
                profitable.append(recipe)

        profitable.sort(key=lambda r: -r.profit_per_craft)
        self._profitable_recipes = profitable
        self._stats["profitable"] = len(profitable)

        logger.info("crafting_db_loaded: %d recipes, %d profitable", len(recipes), len(profitable))

    def _get_item_value(self, item_name: str) -> int:
        """Get the market value of an item."""
        valuation = self._item_db.get_item(item_name)
        if valuation:
            return valuation.market_value
        return 0

    def _get_npc_cost(self, item_name: str) -> int:
        """Get the NPC shop cost of an item."""
        price = self._npc_db.get_npc_price(item_name)
        if price:
            return price
        # Fallback: use buy price from item DB
        valuation = self._item_db.get_item(item_name)
        if valuation and valuation.buy_price > 0:
            return valuation.buy_price
        return 0

    def _potion_recipes(self) -> list[CraftingRecipe]:
        """Potion crafting recipes (Alchemist)."""
        recipes = []

        # White Potion: Empty Bottle (10z) + Herbs
        white_potion_val = self._get_item_value("White Potion")
        empty_bottle_cost = self._get_npc_cost("Empty Bottle")
        if white_potion_val > 0 and empty_bottle_cost > 0:
            # Simplified: 1 Empty Bottle + materials → 1 White Potion
            total_cost = empty_bottle_cost + 200  # Estimated material cost
            profit = white_potion_val - total_cost
            margin = (profit / max(total_cost, 1)) * 100
            recipes.append(CraftingRecipe(
                output_item="White Potion",
                output_quantity=1,
                output_market_value=white_potion_val,
                ingredients=[
                    CraftingIngredient("Empty Bottle", 1, empty_bottle_cost, "npc"),
                    CraftingIngredient("Herbs/Materials", 1, 200, "farm"),
                ],
                total_cost=total_cost,
                profit_per_craft=profit,
                profit_margin_pct=round(margin, 1),
                required_level=40,
                required_skill="Potion Creation",
                difficulty="easy",
                risk_level="low",
                demand="high",
                category="potion",
            ))

        # Blue Potion: Empty Bottle + Blue Herbs
        blue_potion_val = self._get_item_value("Blue Potion")
        if blue_potion_val > 0 and empty_bottle_cost > 0:
            total_cost = empty_bottle_cost + 800  # Estimated material cost
            profit = blue_potion_val - total_cost
            margin = (profit / max(total_cost, 1)) * 100
            recipes.append(CraftingRecipe(
                output_item="Blue Potion",
                output_quantity=1,
                output_market_value=blue_potion_val,
                ingredients=[
                    CraftingIngredient("Empty Bottle", 1, empty_bottle_cost, "npc"),
                    CraftingIngredient("Blue Herbs/Materials", 1, 800, "farm"),
                ],
                total_cost=total_cost,
                profit_per_craft=profit,
                profit_margin_pct=round(margin, 1),
                required_level=50,
                required_skill="Potion Creation",
                difficulty="easy",
                risk_level="low",
                demand="high",
                category="potion",
            ))

        return recipes

    def _converter_recipes(self) -> list[CraftingRecipe]:
        """Elemental converter recipes."""
        recipes = []

        # Elemental Converters are made from materials + elemental ores
        # Flame Converter: Coal + Red Gemstone + materials
        coal_val = self._get_item_value("Coal")
        red_gem_cost = self._get_npc_cost("Red Gemstone")
        flame_val = self._get_item_value("Flame Converter")

        if flame_val > 0 and coal_val > 0 and red_gem_cost > 0:
            total_cost = coal_val + red_gem_cost + 1000  # Estimated additional materials
            profit = flame_val - total_cost
            margin = (profit / max(total_cost, 1)) * 100
            recipes.append(CraftingRecipe(
                output_item="Flame Converter",
                output_quantity=1,
                output_market_value=flame_val,
                ingredients=[
                    CraftingIngredient("Coal", 1, coal_val, "farm"),
                    CraftingIngredient("Red Gemstone", 1, red_gem_cost, "npc"),
                    CraftingIngredient("Materials", 1, 1000, "farm"),
                ],
                total_cost=total_cost,
                profit_per_craft=profit,
                profit_margin_pct=round(margin, 1),
                required_level=60,
                required_skill="Elemental Converter Creation",
                difficulty="medium",
                risk_level="low",
                demand="high",
                category="converter",
            ))

        # Frost Converter: materials
        frost_val = self._get_item_value("Frost Converter")
        blue_gem_cost = self._get_npc_cost("Blue Gemstone")
        if frost_val > 0 and blue_gem_cost > 0:
            total_cost = blue_gem_cost + 1200
            profit = frost_val - total_cost
            margin = (profit / max(total_cost, 1)) * 100
            recipes.append(CraftingRecipe(
                output_item="Frost Converter",
                output_quantity=1,
                output_market_value=frost_val,
                ingredients=[
                    CraftingIngredient("Blue Gemstone", 1, blue_gem_cost, "npc"),
                    CraftingIngredient("Materials", 1, 1200, "farm"),
                ],
                total_cost=total_cost,
                profit_per_craft=profit,
                profit_margin_pct=round(margin, 1),
                required_level=60,
                required_skill="Elemental Converter Creation",
                difficulty="medium",
                risk_level="low",
                demand="high",
                category="converter",
            ))

        return recipes

    def _arrow_recipes(self) -> list[CraftingRecipe]:
        """Arrow crafting recipes."""
        recipes = []

        # Basic Arrows: materials → stack of arrows
        arrow_val = self._get_item_value("Arrow")
        if arrow_val > 0:
            # 1 material → 100 arrows typically
            total_cost = 100  # Estimated material cost
            profit = (arrow_val * 100) - total_cost
            margin = (profit / max(total_cost, 1)) * 100
            recipes.append(CraftingRecipe(
                output_item="Arrow",
                output_quantity=100,
                output_market_value=arrow_val,
                ingredients=[
                    CraftingIngredient("Materials", 1, 100, "farm"),
                ],
                total_cost=total_cost,
                profit_per_craft=profit,
                profit_margin_pct=round(margin, 1),
                required_level=1,
                required_skill="Arrow Crafting",
                difficulty="easy",
                risk_level="low",
                demand="high",
                category="arrow",
            ))

        # Silver Arrows: materials + silver
        silver_arrow_val = self._get_item_value("Silver Arrow")
        if silver_arrow_val > 0:
            total_cost = 300
            profit = (silver_arrow_val * 100) - total_cost
            margin = (profit / max(total_cost, 1)) * 100
            recipes.append(CraftingRecipe(
                output_item="Silver Arrow",
                output_quantity=100,
                output_market_value=silver_arrow_val,
                ingredients=[
                    CraftingIngredient("Silver/Materials", 1, 300, "farm"),
                ],
                total_cost=total_cost,
                profit_per_craft=profit,
                profit_margin_pct=round(margin, 1),
                required_level=30,
                required_skill="Arrow Crafting",
                difficulty="easy",
                risk_level="low",
                demand="medium",
                category="arrow",
            ))

        return recipes

    def _food_recipes(self) -> list[CraftingRecipe]:
        """Food crafting recipes (Cook)."""
        recipes = []
        # Food recipes would go here — requires knowledge.json food data
        return recipes

    def _scroll_recipes(self) -> list[CraftingRecipe]:
        """Scroll crafting recipes (Sage/Professor)."""
        recipes = []
        # Scroll recipes would go here
        return recipes

    # ── Public API ─────────────────────────────────────────────────────

    def get_profitable_recipes(self, min_margin: float = MIN_CRAFT_MARGIN,
                                 category: str | None = None) -> list[CraftingRecipe]:
        """Get profitable crafting recipes, optionally filtered by category."""
        with self._lock:
            candidates = [
                r for r in self._profitable_recipes
                if r.profit_margin_pct >= min_margin * 100
            ]
            if category:
                candidates = [r for r in candidates if r.category == category]
            return candidates

    def get_best_recipe(self, min_margin: float = MIN_CRAFT_MARGIN) -> CraftingRecipe | None:
        """Get the single most profitable recipe."""
        with self._lock:
            candidates = [
                r for r in self._profitable_recipes
                if r.profit_margin_pct >= min_margin * 100
            ]
            if not candidates:
                return None
            return candidates[0]

    def get_crafting_summary(self) -> str:
        """Get a formatted summary of profitable recipes."""
        with self._lock:
            lines = ["── Crafting for Profit ──"]
            lines.append(f"Recipes loaded: {self._stats['recipes_loaded']}")
            lines.append(f"Profitable: {self._stats['profitable']}")
            lines.append("")

            for recipe in self._profitable_recipes[:10]:
                lines.append(
                    f"  {recipe.output_item} x{recipe.output_quantity}: "
                    f"cost={recipe.total_cost:,}z → "
                    f"value={recipe.output_market_value * recipe.output_quantity:,}z "
                    f"(+{recipe.profit_per_craft:,}z, {recipe.profit_margin_pct:.0f}%) "
                    f"[{recipe.difficulty}, {recipe.demand} demand]"
                )

            return "\n".join(lines)

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# ── Global Singleton ─────────────────────────────────────────────────────

_crafting_db: CraftingDB | None = None
_crafting_db_lock = RLock()


def get_crafting_db() -> CraftingDB:
    """Get the global CraftingDB singleton."""
    global _crafting_db
    with _crafting_db_lock:
        if _crafting_db is None:
            _crafting_db = CraftingDB()
        return _crafting_db
