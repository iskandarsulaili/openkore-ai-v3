"""Supply Chain Awareness — what items are needed to craft what, and what to farm.

A pro player doesn't just farm random monsters. They think in supply chains:
- "I need Empty Bottles to make potions"
- "I need herbs to make potions"
- "I need to farm materials for converters"
- "I should farm X because it drops Y which crafts into Z which sells for 5x"

This module connects farming → crafting → selling into a complete supply chain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from ai_sidecar.economy.item_value_db import get_item_value_db, ItemValueDB
from ai_sidecar.economy.npc_shop_db import get_npc_shop_db, NPCShopDB
from ai_sidecar.economy.crafting_db import get_crafting_db, CraftingDB, CraftingRecipe

logger = logging.getLogger(__name__)


# ── Dataclasses ───────────────────────────────────────────────────────────


@dataclass
class SupplyChainNode:
    """A node in a supply chain."""
    item_name: str
    quantity: int
    source: str  # farm, npc, craft
    cost_per_unit: int
    total_cost: int
    is_bottleneck: bool  # True if this is the hardest-to-get item


@dataclass
class SupplyChain:
    """A complete supply chain from raw materials to final product."""
    final_product: str
    final_value: int  # Market value of final product
    total_cost: int  # Total cost of all inputs
    profit: int  # final_value - total_cost
    profit_margin_pct: float
    nodes: list[SupplyChainNode]
    steps: int  # Number of steps in the chain
    complexity: str  # simple, moderate, complex
    time_estimate_minutes: int  # Estimated time to complete one cycle


# ── Supply Chain Analyzer ─────────────────────────────────────────────────


@dataclass(slots=True)
class SupplyChainAnalyzer:
    """Analyzes supply chains from raw materials to finished goods.

    Thread-safe. Uses ItemValueDB, NPCShopDB, and CraftingDB.
    """

    _lock: RLock = field(default_factory=RLock)
    _item_db: ItemValueDB = field(default_factory=get_item_value_db)
    _npc_db: NPCShopDB = field(default_factory=get_npc_shop_db)
    _crafting_db: CraftingDB = field(default_factory=get_crafting_db)
    _chains: list[SupplyChain] = field(default_factory=list)
    _stats: dict[str, int] = field(default_factory=lambda: {"chains_analyzed": 0})

    def __post_init__(self) -> None:
        self._analyze_chains()

    def _analyze_chains(self) -> None:
        """Analyze all supply chains from profitable recipes."""
        chains: list[SupplyChain] = []

        recipes = self._crafting_db.get_profitable_recipes()
        for recipe in recipes:
            chain = self._build_chain(recipe)
            if chain:
                chains.append(chain)

        chains.sort(key=lambda c: -c.profit)
        self._chains = chains
        self._stats["chains_analyzed"] = len(chains)

        logger.info("supply_chain_analyzed: %d chains", len(chains))

    def _build_chain(self, recipe: CraftingRecipe) -> SupplyChain | None:
        """Build a supply chain from a crafting recipe."""
        nodes: list[SupplyChainNode] = []
        total_cost = 0
        bottlenecks = 0

        for ingredient in recipe.ingredients:
            # Determine if this ingredient is a bottleneck
            # (hard to farm, expensive, or limited supply)
            is_bottleneck = False
            cost = ingredient.estimated_cost * ingredient.quantity

            # Check if ingredient is NPC-buyable (easy)
            npc_price = self._npc_db.get_npc_price(ingredient.item_name)
            if npc_price is None:
                # Must be farmed — potential bottleneck
                is_bottleneck = True
                bottlenecks += 1

            node = SupplyChainNode(
                item_name=ingredient.item_name,
                quantity=ingredient.quantity,
                source=ingredient.source,
                cost_per_unit=ingredient.estimated_cost,
                total_cost=cost,
                is_bottleneck=is_bottleneck,
            )
            nodes.append(node)
            total_cost += cost

        # Determine complexity
        if len(nodes) <= 2:
            complexity = "simple"
        elif len(nodes) <= 4:
            complexity = "moderate"
        else:
            complexity = "complex"

        # Time estimate
        time_est = len(nodes) * 30  # 30 min per ingredient

        final_value = recipe.output_market_value * recipe.output_quantity
        profit = final_value - total_cost
        margin = (profit / max(total_cost, 1)) * 100

        return SupplyChain(
            final_product=recipe.output_item,
            final_value=final_value,
            total_cost=total_cost,
            profit=profit,
            profit_margin_pct=round(margin, 1),
            nodes=nodes,
            steps=len(nodes),
            complexity=complexity,
            time_estimate_minutes=time_est,
        )

    # ── Public API ─────────────────────────────────────────────────────

    def get_best_chains(self, top_n: int = 5) -> list[SupplyChain]:
        """Get the best supply chains sorted by profit."""
        with self._lock:
            return self._chains[:top_n]

    def get_simple_chains(self, top_n: int = 5) -> list[SupplyChain]:
        """Get simple supply chains (good for beginners)."""
        with self._lock:
            simple = [c for c in self._chains if c.complexity == "simple"]
            return simple[:top_n]

    def get_chain_for_product(self, product_name: str) -> SupplyChain | None:
        """Get the supply chain for a specific product."""
        with self._lock:
            for chain in self._chains:
                if chain.final_product.lower() == product_name.lower():
                    return chain
            return None

    def get_supply_chain_summary(self) -> str:
        """Get a formatted summary of supply chains."""
        with self._lock:
            lines = ["── Supply Chain Analysis ──"]
            lines.append(f"Chains analyzed: {self._stats['chains_analyzed']}")
            lines.append("")

            for chain in self._chains[:5]:
                lines.append(f"  {chain.final_product}:")
                lines.append(f"    Cost: {chain.total_cost:,}z → Value: {chain.final_value:,}z")
                lines.append(f"    Profit: {chain.profit:,}z ({chain.profit_margin_pct:.0f}%)")
                lines.append(f"    Steps: {chain.steps} ({chain.complexity})")
                lines.append(f"    Time: ~{chain.time_estimate_minutes} min")
                for node in chain.nodes:
                    bottleneck = " ⚠ BOTTLENECK" if node.is_bottleneck else ""
                    lines.append(f"      {node.quantity}x {node.item_name} ({node.source}){bottleneck}")
                lines.append("")

            return "\n".join(lines)

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# ── Global Singleton ─────────────────────────────────────────────────────

_supply_chain: SupplyChainAnalyzer | None = None
_supply_chain_lock = RLock()


def get_supply_chain_analyzer() -> SupplyChainAnalyzer:
    """Get the global SupplyChainAnalyzer singleton."""
    global _supply_chain
    with _supply_chain_lock:
        if _supply_chain is None:
            _supply_chain = SupplyChainAnalyzer()
        return _supply_chain
