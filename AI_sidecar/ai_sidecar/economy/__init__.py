"""Economic engine package — real money-making economy system.

Modules:
  - item_value_db: Knows what items are worth farming vs vendor trash
  - farming_selector: Picks the most profitable monsters/maps
  - npc_shop_db: NPC shop prices and arbitrage opportunities
  - crafting_db: Profitable crafting recipes
  - supply_chain: End-to-end supply chain analysis
  - orchestrator: Wires everything together for the PDCA loop
  - economic_engine: Original map-level profit tracking (legacy)
  - market_arbitrage: Original market price tracking (legacy)
  - market_executor: Original trade execution (legacy)
  - market_manipulator: Original market cornering (legacy)
  - vending_automation: Original vending automation (legacy)
  - opportunity_cost: Original opportunity cost analysis (legacy)
"""

from ai_sidecar.economy.item_value_db import (
    ItemValueDB, ItemValuation, MonsterDropValue,
    get_item_value_db,
)
from ai_sidecar.economy.farming_selector import (
    FarmingTargetSelector, FarmingTarget, LootFilter,
    get_farming_selector,
)
from ai_sidecar.economy.npc_shop_db import (
    NPCShopDB, NPCShopEntry, NPCArbitrage,
    get_npc_shop_db,
)
from ai_sidecar.economy.crafting_db import (
    CraftingDB, CraftingRecipe, CraftingIngredient,
    get_crafting_db,
)
from ai_sidecar.economy.supply_chain import (
    SupplyChainAnalyzer, SupplyChain, SupplyChainNode,
    get_supply_chain_analyzer,
)
from ai_sidecar.economy.orchestrator import (
    EconomyOrchestrator,
    get_economy_orchestrator,
)

__all__ = [
    "ItemValueDB", "ItemValuation", "MonsterDropValue", "get_item_value_db",
    "FarmingTargetSelector", "FarmingTarget", "LootFilter", "get_farming_selector",
    "NPCShopDB", "NPCShopEntry", "NPCArbitrage", "get_npc_shop_db",
    "CraftingDB", "CraftingRecipe", "CraftingIngredient", "get_crafting_db",
    "SupplyChainAnalyzer", "SupplyChain", "SupplyChainNode", "get_supply_chain_analyzer",
    "EconomyOrchestrator", "get_economy_orchestrator",
]
