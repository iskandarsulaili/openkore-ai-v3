"""NPC Shop Price Database — knows what items NPCs sell and buy, enabling arbitrage.

A pro player knows:
- NPC shops sell potions, arrows, and basic equipment at fixed prices
- Players often pay 2-3x the NPC price for convenience
- Some items can be bought from NPCs and resold to players for profit
- NPC buy prices set a floor for item values

This module catalogs NPC shop prices and identifies arbitrage opportunities
between NPC shops and the player market.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────

# Minimum profit margin to consider an arbitrage worthwhile
MIN_ARBITRAGE_MARGIN = 0.3  # 30%

# Minimum absolute profit per unit
MIN_ARBITRAGE_PROFIT = 100  # 100z

# Maximum risk for arbitrage (0.0-1.0)
MAX_ARBITRAGE_RISK = 0.5


# ── Dataclasses ───────────────────────────────────────────────────────────


@dataclass
class NPCShopEntry:
    """An item sold by an NPC shop."""
    item_name: str
    aegis_name: str
    npc_buy_price: int  # What the NPC sells it for
    npc_sell_price: int  # What the NPC buys it for (typically 0 for most items)
    npc_name: str  # Name of the NPC
    map_name: str  # Map where the NPC is located
    town: str  # Town name
    category: str  # potion, arrow, equipment, material, etc.
    stock: int  # -1 for unlimited, >0 for limited


@dataclass
class NPCArbitrage:
    """An arbitrage opportunity between NPC and player market."""
    item_name: str
    aegis_name: str
    npc_buy_price: int
    estimated_market_price: int  # What players typically pay
    profit_per_unit: int
    margin_pct: float
    risk_level: str  # low, medium, high
    volume_potential: str  # low, medium, high
    npc_location: str
    strategy: str  # flip, bulk, repeat
    confidence: float  # 0.0-1.0


# ── NPC Shop Database ─────────────────────────────────────────────────────


@dataclass(slots=True)
class NPCShopDB:
    """Database of NPC shop prices and arbitrage opportunities.

    Uses knowledge.json data plus hardcoded known NPC shop entries.
    Thread-safe.
    """

    _lock: RLock = field(default_factory=RLock)
    _npc_items: dict[str, list[NPCShopEntry]] = field(default_factory=dict)  # item_name -> entries
    _arbitrage_opportunities: list[NPCArbitrage] = field(default_factory=list)
    _stats: dict[str, int] = field(default_factory=lambda: {"items_cataloged": 0, "arbitrage_found": 0})

    def __post_init__(self) -> None:
        self._load_known_npc_shops()

    def _load_known_npc_shops(self) -> None:
        """Load known NPC shop data.

        Uses knowledge.json for item prices and hardcoded NPC locations.
        """
        # Load item prices from knowledge.json
        item_prices: dict[str, dict[str, int]] = {}
        for candidate in [
            str(Path(__file__).parent.parent.parent.parent / "knowledge" / "knowledge.json"),
            str(Path(__file__).parent.parent.parent / "knowledge" / "knowledge.json"),
            "knowledge/knowledge.json",
        ]:
            if candidate and Path(candidate).exists():
                try:
                    data = json.loads(Path(candidate).read_text(encoding="utf-8"))
                    all_items = data.get("items", {}).get("all", [])
                    for item in all_items:
                        if not isinstance(item, dict):
                            continue
                        aegis = item.get("AegisName", "")
                        name = item.get("Name", "")
                        buy = int(item.get("Buy", 0) or 0)
                        sell = int(item.get("Sell", 0) or 0)
                        if aegis and buy > 0:
                            item_prices[aegis] = {"buy": buy, "sell": sell, "name": name}
                    break
                except Exception:
                    continue

        # Known NPC shops in major towns
        # Format: (item_name, aegis_name, npc_buy_price, npc_name, map_name, town, category, stock)
        known_shops: list[tuple[str, str, int, str, str, str, str, int]] = [
            # ── Prontera ──
            # Potion Shop
            ("Red Potion", "Red_Potion", 50, "Potion Shop", "prontera", "Prontera", "potion", -1),
            ("Orange Potion", "Orange_Potion", 200, "Potion Shop", "prontera", "Prontera", "potion", -1),
            ("Yellow Potion", "Yellow_Potion", 400, "Potion Shop", "prontera", "Prontera", "potion", -1),
            ("White Potion", "White_Potion", 500, "Potion Shop", "prontera", "Prontera", "potion", -1),
            ("Blue Potion", "Blue_Potion", 2000, "Potion Shop", "prontera", "Prontera", "potion", -1),
            ("Green Potion", "Green_Potion", 200, "Potion Shop", "prontera", "Prontera", "potion", -1),

            # Arrow Shop
            ("Arrow", "Arrow", 2, "Arrow Shop", "prontera", "Prontera", "arrow", -1),
            ("Silver Arrow", "Silver_Arrow", 5, "Arrow Shop", "prontera", "Prontera", "arrow", -1),
            ("Fire Arrow", "Fire_Arrow", 5, "Arrow Shop", "prontera", "Prontera", "arrow", -1),

            # Weapon Shop
            ("Katana", "Katana", 20000, "Weapon Shop", "prontera", "Prontera", "weapon", 10),
            ("Saber", "Saber", 15000, "Weapon Shop", "prontera", "Prontera", "weapon", 10),
            ("Blade", "Blade", 12000, "Weapon Shop", "prontera", "Prontera", "weapon", 10),

            # ── Morocc ──
            ("Red Potion", "Red_Potion", 50, "Potion Shop", "morocc", "Morocc", "potion", -1),
            ("Orange Potion", "Orange_Potion", 200, "Potion Shop", "morocc", "Morocc", "potion", -1),
            ("White Potion", "White_Potion", 500, "Potion Shop", "morocc", "Morocc", "potion", -1),
            ("Blue Potion", "Blue_Potion", 2000, "Potion Shop", "morocc", "Morocc", "potion", -1),

            # ── Payon ──
            ("Red Potion", "Red_Potion", 50, "Potion Shop", "payon", "Payon", "potion", -1),
            ("Orange Potion", "Orange_Potion", 200, "Potion Shop", "payon", "Payon", "potion", -1),
            ("White Potion", "White_Potion", 500, "Potion Shop", "payon", "Payon", "potion", -1),
            ("Blue Potion", "Blue_Potion", 2000, "Potion Shop", "payon", "Payon", "potion", -1),
            ("Arrow", "Arrow", 2, "Arrow Shop", "payon", "Payon", "arrow", -1),

            # ── Geffen ──
            ("Red Potion", "Red_Potion", 50, "Potion Shop", "geffen", "Geffen", "potion", -1),
            ("Orange Potion", "Orange_Potion", 200, "Potion Shop", "geffen", "Geffen", "potion", -1),
            ("White Potion", "White_Potion", 500, "Potion Shop", "geffen", "Geffen", "potion", -1),
            ("Blue Potion", "Blue_Potion", 2000, "Potion Shop", "geffen", "Geffen", "potion", -1),
            ("Green Potion", "Green_Potion", 200, "Potion Shop", "geffen", "Geffen", "potion", -1),

            # ── Aldebaran ──
            ("Red Potion", "Red_Potion", 50, "Potion Shop", "aldebaran", "Aldebaran", "potion", -1),
            ("White Potion", "White_Potion", 500, "Potion Shop", "aldebaran", "Aldebaran", "potion", -1),
            ("Blue Potion", "Blue_Potion", 2000, "Potion Shop", "aldebaran", "Aldebaran", "potion", -1),

            # ── Universal: Kafra Storage ──
            ("Wing of Fly", "Wing_Of_Fly", 500, "Kafra", "all", "All", "consumable", -1),
            ("Butterfly Wing", "Butterfly_Wing", 500, "Kafra", "all", "All", "consumable", -1),

            # ── Universal: Tool Dealer ──
            ("Empty Bottle", "Empty_Bottle", 10, "Tool Dealer", "all", "All", "material", -1),
            ("Red Gemstone", "Red_Gemstone", 500, "Tool Dealer", "all", "All", "material", -1),
            ("Blue Gemstone", "Blue_Gemstone", 1000, "Tool Dealer", "all", "All", "material", -1),
            ("Yellow Gemstone", "Yellow_Gemstone", 1000, "Tool Dealer", "all", "All", "material", -1),
        ]

        # Build entries, using knowledge.json prices when available
        for name, aegis, price, npc, map_name, town, category, stock in known_shops:
            # Use knowledge.json price if available (more accurate)
            if aegis in item_prices:
                actual_price = item_prices[aegis]["buy"]
            else:
                actual_price = price

            entry = NPCShopEntry(
                item_name=name,
                aegis_name=aegis,
                npc_buy_price=actual_price,
                npc_sell_price=item_prices.get(aegis, {}).get("sell", 0),
                npc_name=npc,
                map_name=map_name,
                town=town,
                category=category,
                stock=stock,
            )

            if name not in self._npc_items:
                self._npc_items[name] = []
            self._npc_items[name].append(entry)

        self._stats["items_cataloged"] = len(self._npc_items)
        self._find_arbitrage()
        logger.info("npc_shop_db_loaded: %d items, %d arbitrage opportunities",
                     self._stats["items_cataloged"], self._stats["arbitrage_found"])

    def _find_arbitrage(self) -> None:
        """Find arbitrage opportunities between NPC shops and player market.

        Player market prices are estimated at 2-3x NPC buy price for commonly
        traded items.
        """
        opportunities: list[NPCArbitrage] = []

        # Items with known player market multipliers
        market_multipliers: dict[str, float] = {
            "potion": 2.0,  # Potions sell for 2x NPC price
            "arrow": 3.0,  # Arrows sell for 3x NPC price
            "consumable": 2.5,  # Consumables sell for 2.5x
            "material": 2.0,  # Materials sell for 2x
            "weapon": 1.5,  # Weapons sell for 1.5x (less margin)
        }

        for item_name, entries in self._npc_items.items():
            if not entries:
                continue

            # Get the cheapest NPC price for this item
            cheapest = min(entries, key=lambda e: e.npc_buy_price)
            npc_price = cheapest.npc_buy_price

            # Estimate market price
            multiplier = market_multipliers.get(cheapest.category, 1.5)
            market_price = int(npc_price * multiplier)

            profit = market_price - npc_price
            margin = profit / max(npc_price, 1)

            if profit >= MIN_ARBITRAGE_PROFIT and margin >= MIN_ARBITRAGE_MARGIN:
                # Determine risk
                if margin > 1.0:
                    risk = "low"
                elif margin > 0.5:
                    risk = "low"
                else:
                    risk = "medium"

                # Volume potential
                if cheapest.category in ("potion", "arrow", "consumable"):
                    volume = "high"
                elif cheapest.category == "material":
                    volume = "medium"
                else:
                    volume = "low"

                # Strategy
                if cheapest.category in ("potion", "arrow"):
                    strategy = "bulk"
                elif cheapest.category == "material":
                    strategy = "repeat"
                else:
                    strategy = "flip"

                # Confidence
                confidence = min(1.0, margin / 2.0)

                opp = NPCArbitrage(
                    item_name=item_name,
                    aegis_name=cheapest.aegis_name,
                    npc_buy_price=npc_price,
                    estimated_market_price=market_price,
                    profit_per_unit=profit,
                    margin_pct=round(margin * 100, 1),
                    risk_level=risk,
                    volume_potential=volume,
                    npc_location=f"{cheapest.town} ({cheapest.npc_name})",
                    strategy=strategy,
                    confidence=round(confidence, 2),
                )
                opportunities.append(opp)

        opportunities.sort(key=lambda o: -o.profit_per_unit * (3 if o.volume_potential == "high" else 1))
        self._arbitrage_opportunities = opportunities
        self._stats["arbitrage_found"] = len(opportunities)

    # ── Public API ─────────────────────────────────────────────────────

    def get_npc_price(self, item_name: str) -> int | None:
        """Get the cheapest NPC buy price for an item."""
        with self._lock:
            entries = self._npc_items.get(item_name)
            if not entries:
                return None
            return min(e.npc_buy_price for e in entries)

    def get_npc_locations(self, item_name: str) -> list[dict[str, Any]]:
        """Get all NPC locations where an item can be bought."""
        with self._lock:
            entries = self._npc_items.get(item_name, [])
            return [
                {
                    "npc": e.npc_name,
                    "map": e.map_name,
                    "town": e.town,
                    "price": e.npc_buy_price,
                    "stock": e.stock,
                }
                for e in entries
            ]

    def get_best_arbitrage(self, min_margin: float = MIN_ARBITRAGE_MARGIN,
                            max_risk: str = "medium") -> list[NPCArbitrage]:
        """Get the best NPC arbitrage opportunities."""
        with self._lock:
            risk_order = {"low": 0, "medium": 1, "high": 2}
            max_risk_val = risk_order.get(max_risk, 1)
            candidates = [
                o for o in self._arbitrage_opportunities
                if o.margin_pct >= min_margin * 100
                and risk_order.get(o.risk_level, 0) <= max_risk_val
            ]
            return candidates[:10]

    def get_arbitrage_summary(self) -> str:
        """Get a formatted summary of arbitrage opportunities."""
        with self._lock:
            lines = ["── NPC Shop Arbitrage ──"]
            lines.append(f"Items cataloged: {self._stats['items_cataloged']}")
            lines.append(f"Arbitrage opportunities: {self._stats['arbitrage_found']}")
            lines.append("")

            for opp in self._arbitrage_opportunities[:10]:
                lines.append(
                    f"  {opp.item_name}: "
                    f"buy={opp.npc_buy_price:,}z → sell≈{opp.estimated_market_price:,}z "
                    f"(+{opp.profit_per_unit:,}z, {opp.margin_pct:.0f}%) "
                    f"[{opp.risk_level} risk, {opp.volume_potential} volume]"
                )

            return "\n".join(lines)

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# ── Global Singleton ─────────────────────────────────────────────────────

_npc_shop_db: NPCShopDB | None = None
_npc_shop_db_lock = RLock()


def get_npc_shop_db() -> NPCShopDB:
    """Get the global NPCShopDB singleton."""
    global _npc_shop_db
    with _npc_shop_db_lock:
        if _npc_shop_db is None:
            _npc_shop_db = NPCShopDB()
        return _npc_shop_db
