"""Economy Orchestrator — wires all economy subsystems into the PDCA loop.

This is the top-level coordinator that:
1. Uses ItemValueDB to know what items are worth farming
2. Uses FarmingTargetSelector to pick the most profitable monsters
3. Uses NPCShopDB for arbitrage opportunities
4. Uses CraftingDB for profitable recipes
5. Uses SupplyChainAnalyzer for end-to-end supply chains
6. Integrates with the existing EconomicEngine for map-level tracking

The orchestrator provides a unified get_economy_context() method that the
PDCA loop can inject into LLM prompts, giving the AI real economic awareness.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from ai_sidecar.economy.item_value_db import (
    get_item_value_db, ItemValueDB, ItemValuation,
)
from ai_sidecar.economy.farming_selector import (
    get_farming_selector, FarmingTargetSelector, FarmingTarget, LootFilter,
)
from ai_sidecar.economy.npc_shop_db import (
    get_npc_shop_db, NPCShopDB, NPCArbitrage,
)
from ai_sidecar.economy.crafting_db import (
    get_crafting_db, CraftingDB, CraftingRecipe,
)
from ai_sidecar.economy.supply_chain import (
    get_supply_chain_analyzer, SupplyChainAnalyzer, SupplyChain,
)

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────

# How often to refresh economic context (seconds)
CONTEXT_REFRESH_INTERVAL = 60  # 1 minute

# Minimum zeny to consider for market operations
MIN_MARKET_CAPITAL = 10000

# Maximum number of items in context output
MAX_CONTEXT_ITEMS = 10


# ── Economy Orchestrator ───────────────────────────────────────────────────


@dataclass(slots=True)
class EconomyOrchestrator:
    """Top-level economy coordinator.

    Wires all economy subsystems together and provides unified context
    for the PDCA loop. Thread-safe.
    """

    _lock: RLock = field(default_factory=RLock)
    _item_db: ItemValueDB = field(default_factory=get_item_value_db)
    _farming_selector: FarmingTargetSelector = field(default_factory=get_farming_selector)
    _npc_db: NPCShopDB = field(default_factory=get_npc_shop_db)
    _crafting_db: CraftingDB = field(default_factory=get_crafting_db)
    _supply_chain: SupplyChainAnalyzer = field(default_factory=get_supply_chain_analyzer)
    _last_context_refresh: float = 0.0
    _cached_context: str = ""
    _stats: dict[str, int] = field(default_factory=lambda: {
        "context_refreshes": 0, "farming_recommendations": 0,
    })

    # ── Public API ─────────────────────────────────────────────────────

    def get_economy_context(self, level: int = 50, zeny: int = 0,
                             current_map: str = "") -> str:
        """Get a unified economy context string for LLM prompts.

        This is the main entry point for the PDCA loop. Returns a formatted
        string with:
        - Best farming targets
        - NPC arbitrage opportunities
        - Profitable crafting recipes
        - Supply chain analysis
        - Loot filter advice
        """
        with self._lock:
            now = time.time()
            if (self._cached_context
                    and now - self._last_context_refresh < CONTEXT_REFRESH_INTERVAL):
                return self._cached_context

            self._last_context_refresh = now
            self._stats["context_refreshes"] += 1  # type: ignore[assignment]

            lines = ["── Economy Orchestrator ──"]
            lines.append(f"Level: {level} | Zeny: {zeny:,}z | Map: {current_map or 'unknown'}")
            lines.append("")

            # 1. Best farming targets
            lines.append("▶ FARMING TARGETS:")
            target = self._farming_selector.get_best_target(level, zeny, current_map)
            if target:
                lines.append(
                    f"  Best: {target.monster_name} (Lv{target.monster_level}) "
                    f"→ {target.expected_zeny_per_kill:.0f}z/kill "
                    f"({target.expected_zeny_per_hour:,.0f}z/hr)"
                )
                lines.append(f"  Best drop: {target.best_drop}")
                lines.append(f"  Competition: {target.competition_risk}")
            else:
                lines.append("  No profitable targets found for your level.")

            all_targets = self._farming_selector.get_all_targets()
            if all_targets:
                lines.append("  Top targets:")
                for t in all_targets[:5]:
                    lines.append(
                        f"    #{t.priority}: {t.monster_name} "
                        f"({t.expected_zeny_per_kill:.0f}z/kill)"
                    )
            lines.append("")

            # 2. NPC arbitrage
            lines.append("▶ NPC ARBITRAGE:")
            arbitrage = self._npc_db.get_best_arbitrage()
            if arbitrage:
                for opp in arbitrage[:5]:
                    lines.append(
                        f"  {opp.item_name}: buy={opp.npc_buy_price:,}z "
                        f"→ sell≈{opp.estimated_market_price:,}z "
                        f"(+{opp.profit_per_unit:,}z, {opp.margin_pct:.0f}%)"
                    )
            else:
                lines.append("  No arbitrage opportunities found.")
            lines.append("")

            # 3. Profitable crafting
            lines.append("▶ CRAFTING FOR PROFIT:")
            recipes = self._crafting_db.get_profitable_recipes()
            if recipes:
                for recipe in recipes[:5]:
                    lines.append(
                        f"  {recipe.output_item} x{recipe.output_quantity}: "
                        f"cost={recipe.total_cost:,}z → "
                        f"value={recipe.output_market_value * recipe.output_quantity:,}z "
                        f"(+{recipe.profit_per_craft:,}z, {recipe.profit_margin_pct:.0f}%)"
                    )
            else:
                lines.append("  No profitable recipes found.")
            lines.append("")

            # 4. Supply chain analysis
            lines.append("▶ SUPPLY CHAINS:")
            chains = self._supply_chain.get_best_chains(3)
            if chains:
                for chain in chains:
                    lines.append(
                        f"  {chain.final_product}: "
                        f"cost={chain.total_cost:,}z → "
                        f"value={chain.final_value:,}z "
                        f"(+{chain.profit:,}z, {chain.profit_margin_pct:.0f}%) "
                        f"[{chain.complexity}, ~{chain.time_estimate_minutes}min]"
                    )
            else:
                lines.append("  No supply chains available.")
            lines.append("")

            # 5. Loot filter advice
            lines.append("▶ LOOT FILTER:")
            lines.append(self._farming_selector.get_loot_filter_advice())

            # 6. Top valuable items
            lines.append("")
            lines.append("▶ TOP VALUABLE ITEMS (worth picking up):")
            valuable = self._item_db.get_valuable_items()
            if valuable:
                for v in valuable[:5]:
                    lines.append(
                        f"  {v.name}: {v.market_value:,}z "
                        f"(density={v.value_density:.0f}z/wt, "
                        f"action={v.recommendation})"
                    )

            self._cached_context = "\n".join(lines)
            return self._cached_context

    def get_farming_recommendation(self, level: int, zeny: int = 0,
                                    weight_capacity: int = 100) -> dict[str, Any]:
        """Get a complete farming recommendation with all details."""
        with self._lock:
            self._stats["farming_recommendations"] += 1  # type: ignore[assignment]

            # Get best target
            target = self._farming_selector.get_best_target(level, zeny)

            # Get arbitrage opportunities
            arbitrage = self._npc_db.get_best_arbitrage()

            # Get best recipe
            recipe = self._crafting_db.get_best_recipe()

            return {
                "best_target": {
                    "monster": target.monster_name if target else None,
                    "zeny_per_kill": target.expected_zeny_per_kill if target else 0,
                    "zeny_per_hour": target.expected_zeny_per_hour if target else 0,
                    "best_drop": target.best_drop if target else None,
                } if target else None,
                "arbitrage": [
                    {
                        "item": opp.item_name,
                        "buy": opp.npc_buy_price,
                        "sell": opp.estimated_market_price,
                        "profit": opp.profit_per_unit,
                        "margin": opp.margin_pct,
                    }
                    for opp in arbitrage[:3]
                ],
                "best_recipe": {
                    "item": recipe.output_item if recipe else None,
                    "cost": recipe.total_cost if recipe else 0,
                    "value": recipe.output_market_value * recipe.output_quantity if recipe else 0,
                    "profit": recipe.profit_per_craft if recipe else 0,
                } if recipe else None,
            }

    def should_pickup_item(self, item_name: str, item_value: int,
                            item_weight: int, item_category: str) -> bool:
        """Determine if an item is worth picking up."""
        return self._farming_selector.should_pickup(
            item_name, item_value, item_weight, item_category
        )

    def get_item_value(self, item_name: str) -> int:
        """Get the market value of an item."""
        valuation = self._item_db.get_item(item_name)
        if valuation:
            return valuation.market_value
        return 0

    def get_economy_summary(self) -> str:
        """Get a short summary of economy system status."""
        with self._lock:
            lines = ["── Economy System Status ──"]
            lines.append(f"Items in DB: {self._item_db.counters().get('items_loaded', 0)}")
            lines.append(f"Monsters analyzed: {self._item_db.counters().get('monsters_loaded', 0)}")
            lines.append(f"NPC items cataloged: {self._npc_db.counters().get('items_cataloged', 0)}")
            lines.append(f"Arbitrage opportunities: {self._npc_db.counters().get('arbitrage_found', 0)}")
            lines.append(f"Profitable recipes: {self._crafting_db.counters().get('profitable', 0)}")
            lines.append(f"Supply chains: {self._supply_chain.counters().get('chains_analyzed', 0)}")
            lines.append(f"Farming targets: {len(self._farming_selector.get_all_targets())}")
            return "\n".join(lines)

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# ── Global Singleton ─────────────────────────────────────────────────────

_economy_orchestrator: EconomyOrchestrator | None = None
_economy_orchestrator_lock = RLock()


def get_economy_orchestrator() -> EconomyOrchestrator:
    """Get the global EconomyOrchestrator singleton."""
    global _economy_orchestrator
    with _economy_orchestrator_lock:
        if _economy_orchestrator is None:
            _economy_orchestrator = EconomyOrchestrator()
        return _economy_orchestrator
