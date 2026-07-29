"""
RO Economy Domain — smart item management, farming profitability, vending detection.

This package provides a complete economy management system for RO bots:

  - ItemValueDB:   Item value database loaded from AI_sidecar/data/item_values.yaml
  - ItemWorthCalculator: Calculates real value (cash + utility) for any item
  - ProfitabilityCalculator: Finds best zeny/hour farming spots
  - VendingDetector: Scans player shops, updates market prices, finds arbitrage
  - InventoryManager: Smart keep/sell/discard decisions based on policy

Usage:
    from ai_sidecar.domains.economy import EconomyDomain

    economy = EconomyDomain()
    actions = economy.assess(signals, [], bot_id)
"""
from __future__ import annotations

import logging
from typing import Any

from ai_sidecar.domains.economy.database import ItemValueDB
from ai_sidecar.domains.economy.calculator import ItemWorthCalculator
from ai_sidecar.domains.economy.profitability import ProfitabilityCalculator
from ai_sidecar.domains.economy.vending import VendingDetector
from ai_sidecar.domains.economy.inventory import InventoryManager

logger = logging.getLogger(__name__)

__all__ = [
    "EconomyDomain",
    "ItemValueDB",
    "ItemWorthCalculator",
    "ProfitabilityCalculator",
    "VendingDetector",
    "InventoryManager",
]


class EconomyDomain:
    """Aggregate domain for all economy activities.

    Integrates inventory management, vending detection, profitability
    analysis, and item worth calculation into a single assess() call
    for the heuristic system.
    """

    name = "economy"
    priority = 50  # After combat (60) and survival (40), before routing

    def __init__(self) -> None:
        self.db = ItemValueDB()
        self.calculator = ItemWorthCalculator(self.db)
        self.profitability = ProfitabilityCalculator(self.db)
        self.vending = VendingDetector(self.db)
        self.inventory = InventoryManager(self.db, self.calculator)
        self._initialized = False
        self._last_assessment: dict[str, Any] = {}

    def initialize(self) -> None:
        """Called once when the domain is registered."""
        self._initialized = True
        logger.info(
            "Economy domain initialized: %d items in DB, %d maps with spawn data",
            len(self.db),
            len(self.profitability.get_spawns("prt_fild05") or []),
        )

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[Any],
        bot_id: str,
    ) -> None:
        """Assess economy state and produce actions.

        Evaluates:
          1. Is inventory getting full? → sell junk to NPC
          2. Are there vending opportunities? → vendor items
          3. Is the current farm spot profitable? → recommend better spots
          4. Do we need potions? → buy from NPC

        Args:
            signals: Bot state signals dict.
            actions: List to append HeuristicAction objects to.
            bot_id: Normalized bot identifier.
        """
        if not self._initialized:
            self.initialize()

        # Gather state
        inventory = signals.get("inventory", []) or []
        zeny = int(signals.get("zeny", 0) or 0)
        weight_current = int(signals.get("weight", signals.get("weight_current", 0)) or 0)
        weight_max = int(signals.get("weight_max", 0) or 0)
        if weight_max == 0:
            weight_max = 2000
        weight_pct = weight_current / max(1, weight_max)
        current_map = str(signals.get("map", "") or "").lower().replace(".gat", "")
        job_name = str(signals.get("job_name", "novice") or "novice")
        in_town = self._is_town_map(current_map)

        # ── 1. Inventory assessment ──
        snapshot = self.inventory.assess_inventory(signals, bot_id)

        # ── 2. Sell junk if inventory is full ──
        if weight_pct >= 0.80 and in_town:
            sell_items = [a for a in snapshot.actions if a.action == "sell_npc"]
            if sell_items:
                total_value = sum(a.zeny_value for a in sell_items)
                from ai_sidecar.actions import HeuristicAction
                actions.append(HeuristicAction(
                    kind="command",
                    command="talk sell",
                    confidence=0.95,
                    reason=f"Sell {len(sell_items)} junk items worth ~{total_value}z (weight {weight_pct:.0%})",
                    domain=self.name,
                    metadata={
                        "action": "sell_junk",
                        "items": [
                            {"name": a.item_name, "qty": a.quantity, "value": a.zeny_value}
                            for a in sell_items
                        ],
                        "total_value": total_value,
                    },
                ))

        # ── 3. Vendor valuable items (in town) ──
        if in_town:
            vendor_items = [a for a in snapshot.actions if a.action == "vendor"]
            if vendor_items:
                total_vendor_value = sum(a.zeny_value for a in vendor_items)
                from ai_sidecar.actions import HeuristicAction
                actions.append(HeuristicAction(
                    kind="command",
                    command="talk shop",
                    confidence=0.85,
                    reason=f"{len(vendor_items)} items worth {total_vendor_value}z on player market",
                    domain=self.name,
                    metadata={
                        "action": "vendor_items",
                        "items": [
                            {"name": a.item_name, "qty": a.quantity, "value": a.zeny_value}
                            for a in vendor_items
                        ],
                        "total_value": total_vendor_value,
                        "setup_vendor": True,
                    },
                ))

        # ── 4. Scan player shops for prices (in town, periodically) ──
        if in_town and self.vending.should_rescan(current_map, cooldown_seconds=120):
            shop_listings = self.vending.scan_shops(signals)
            if shop_listings:
                logger.debug(
                    "Found %d player shops on %s",
                    len(shop_listings), current_map,
                )
                # Check for arbitrage opportunities
                opportunities = self.vending.detect_opportunities(inventory, zeny)
                if opportunities:
                    top = opportunities[0]
                    from ai_sidecar.actions import HeuristicAction
                    actions.append(HeuristicAction(
                        kind="command",
                        command=f"talk shop {top.item_name}",
                        confidence=0.75,
                        reason=f"Arbitrage: {top.item_name} — buy @ ~{top.npc_sell_price}z, sell @ ~{top.market_price}z",
                        domain=self.name,
                        metadata={
                            "action": "arbitrage",
                            "opportunities": [
                                {"item": o.item_name, "profit": o.profit_potential,
                                 "market": o.market_price, "npc": o.npc_sell_price}
                                for o in opportunities[:3]
                            ],
                        },
                    ))

        # ── 5. Assess farm spot profitability ──
        # Calculate profitability for current map if we have spawn data
        if current_map and self.profitability.get_spawns(current_map):
            try:
                # Estimate kills per minute from signals
                kills_per_minute = float(signals.get("kills_per_minute", signals.get("kpm", 15)) or 15)
                result = self.profitability.calculate_map_profitability(
                    current_map, kills_per_minute,
                    int(signals.get("base_level", 1) or 1),
                    job_name,
                )
                if result.zeny_per_hour > 0:
                    from ai_sidecar.actions import HeuristicAction
                    actions.append(HeuristicAction(
                        kind="log",
                        command="",
                        confidence=0.9,
                        reason=(
                            f"Map {current_map}: ~{result.zeny_per_hour}z/hr "
                            f"({result.zeny_per_kill}z/kill, "
                            f"{result.kills_per_hour}kills/hr)"
                        ),
                        domain=self.name,
                        metadata={
                            "action": "profitability_report",
                            "map": current_map,
                            "zeny_per_hour": result.zeny_per_hour,
                            "kills_per_hour": result.kills_per_hour,
                            "zeny_per_kill": result.zeny_per_kill,
                            "confidence": result.confidence,
                        },
                    ))

                    # If profitability is low, suggest a better map
                    if result.zeny_per_hour < 5000 and result.confidence > 0.3:
                        from ai_sidecar.actions import HeuristicAction
                        actions.append(HeuristicAction(
                            kind="command",
                            command=f"move",
                            confidence=0.6,
                            reason=f"Low profit ({result.zeny_per_hour}z/hr) — consider better farming spot",
                            domain=self.name,
                            metadata={
                                "action": "low_profit_warning",
                                "current_profit": result.zeny_per_hour,
                                "suggestion": "Try a higher-density or higher-level map",
                            },
                        ))
            except Exception as exc:
                logger.warning("Failed to assess map profitability: %s", exc)

        # ── 6. Potion stock check ──
        hp_ratio = float(signals.get("hp_ratio", 1.0) or 1.0)
        sp_ratio = float(signals.get("sp_ratio", 1.0) or 1.0)
        needs_potions = hp_ratio < 0.4 or sp_ratio < 0.3
        if needs_potions and zeny > 5000 and in_town:
            from ai_sidecar.actions import HeuristicAction
            actions.append(HeuristicAction(
                kind="command",
                command="talk buy",
                confidence=0.9,
                reason=f"HP={hp_ratio:.0%} SP={sp_ratio:.0%} — restock potions",
                domain=self.name,
                metadata={
                    "action": "buy_potions",
                    "hp_ratio": hp_ratio,
                    "sp_ratio": sp_ratio,
                },
            ))

        self._last_assessment = {
            "weight_pct": weight_pct,
            "in_town": in_town,
            "actions_count": len(actions),
            "zeny": zeny,
        }

    def _is_town_map(self, map_name: str) -> bool:
        """Check if the current map is a town/safe zone."""
        towns = (
            "prontera", "izlude", "morocc", "payon", "geffen",
            "aldebaran", "comodo", "umbala", "niflheim",
            "rachel", "veins", "einbroch", "lighthalzen",
            "juno", "hugel", "yuno", "amatsu", "gonryun",
            "louyang", "ayothaya", "alberta",
        )
        return any(t in map_name for t in towns)

    def cleanup_bot(self, bot_id: str) -> None:
        """Clean up per-bot state."""
        pass
