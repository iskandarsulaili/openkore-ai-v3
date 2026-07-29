"""EconomyEngine — self-aware economy management for RO bot.

A top-tier bot doesn't just grind. It knows the *value* of every item it picks up,
plans its budget intelligently, restocks consumables before they run dry, tracks
its own earnings to learn what works, and makes smart merchant decisions. All
data-driven from YAML — no hardcoded prices.

Features:
  - loot_value(item_name, item_id) -> classification + zeny estimate
  - budget_planning(zeny, level, job, inventory) -> buy/sell recommendations
  - restock_needed(inventory, item_name, keep_minimum) -> bool + quantity
  - Session earnings/spending tracking for learning feedback
  - Multi-currency awareness (zeny, tokens, event currency)
  - Merchant interaction decisions (which NPC to buy/sell from)
  - Thread-safe via RLock
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

# Default data paths relative to the project root
_DEFAULT_DATA_DIR = str(
    Path(__file__).parent.parent.parent.parent / "data"
)

# Internal value thresholds (used only as fallbacks when YAML data is incomplete)
_SELL_NPC_PREMIUM_THRESHOLD = 0.85  # if market_price / npc_sell < this, sell to NPC instead of player
_MARKET_PREMIUM_THRESHOLD = 1.3  # if market_price / npc_buy > this, consider selling to player

# Valid classifications
VALID_CLASSIFICATIONS = frozenset({
    "keep", "sell_npc", "sell_player", "sell_any",
    "discard", "crafting", "quest", "potion_food", "material",
})


# ── Dataclasses ───────────────────────────────────────────────────────────


@dataclass
class LootValuation:
    """Valuation result for a single loot item."""
    item_name: str
    item_id: int
    classification: str  # keep / sell_npc / sell_player / sell_any / discard / crafting / quest / potion_food / material
    npc_sell_price: int  # What NPC pays for it
    market_price: int  # Estimated player market price
    estimated_zeny: int  # Best zeny estimate (max of npc_sell, market_price if sellable)
    estimated_zeny_per_weight: float  # Value density
    keep_minimum: int  # Recommended minimum stock
    category: str
    tags: list[str]
    recommendation: str  # Human-readable action


@dataclass
class BudgetRecommendation:
    """Budget planning recommendation."""
    buy: list[dict[str, Any]]  # Items to buy (name, quantity, estimated_cost, priority)
    sell_to_npc: list[dict[str, Any]]  # Items to sell to NPC (name, quantity, estimated_zeny)
    sell_to_player: list[dict[str, Any]]  # Items to sell on player market (name, quantity, estimated_zeny)
    restock: list[dict[str, Any]]  # Items to restock (name, quantity_needed, estimated_cost)
    priority_items: list[dict[str, Any]]  # High-priority purchases (gear, etc.)
    total_estimated_cost: int  # Total zeny needed for recommended purchases
    total_estimated_income: int  # Total zeny from recommended sales
    net_impact: int  # income - cost
    assessment: str  # Human-readable summary


@dataclass
class RestockResult:
    """Result of a restock check."""
    needs_restock: bool
    current_count: int
    keep_minimum: int
    restock_quantity: int
    estimated_cost: int
    item_name: str
    item_id: int | None


@dataclass
class MerchantDecision:
    """Decision about which merchant/NPC to interact with."""
    npc_name: str
    npc_map: str
    npc_coords: tuple[int, int]
    transaction_type: str  # buy / sell
    items: list[dict[str, Any]]
    total_value: int
    priority: int  # 1-10 (higher = more urgent)
    reason: str


@dataclass
class SessionLedger:
    """Per-session earnings and spending tracking."""
    session_id: str
    zeny_earned: int = 0
    zeny_spent: int = 0
    items_sold: list[dict[str, Any]] = field(default_factory=list)
    items_bought: list[dict[str, Any]] = field(default_factory=list)
    items_discarded: list[dict[str, Any]] = field(default_factory=list)
    tokens_earned: int = 0
    tokens_spent: int = 0
    total_kills: int = 0
    total_drops: int = 0
    trips_to_vendor: int = 0
    trips_to_market: int = 0

    @property
    def net_zeny(self) -> int:
        return self.zeny_earned - self.zeny_spent

    @property
    def net_tokens(self) -> int:
        return self.tokens_earned - self.tokens_spent

    def summary(self) -> dict[str, Any]:
        """Get a summary dict for learning feedback."""
        return {
            "session_id": self.session_id,
            "net_zeny": self.net_zeny,
            "net_tokens": self.net_tokens,
            "total_zeny_flow": self.zeny_earned + self.zeny_spent,
            "total_kills": self.total_kills,
            "total_drops": self.total_drops,
            "items_sold_count": len(self.items_sold),
            "items_bought_count": len(self.items_bought),
            "items_discarded_count": len(self.items_discarded),
            "trips_to_vendor": self.trips_to_vendor,
            "trips_to_market": self.trips_to_market,
            "kills_per_zeny": round(self.total_kills / max(self.zeny_earned, 1), 4),
            "drop_rate": round(self.total_drops / max(self.total_kills, 1), 4),
        }


# ── Economy Engine ────────────────────────────────────────────────────────


@dataclass(slots=True)
class EconomyEngine:
    """Self-aware economy management engine.

    All data driven from YAML files. Thread-safe.
    """

    _lock: RLock = field(default_factory=RLock)
    _item_values_path: str = ""
    _market_prices_path: str = ""

    # YAML-loaded data
    _item_values: dict[str, dict[str, Any]] = field(default_factory=dict)  # item_name -> item data
    _market_prices: dict[str, dict[str, Any]] = field(default_factory=dict)  # item_name -> market data
    _npc_merchants: dict[str, dict[str, Any]] = field(default_factory=dict)  # npc_name -> merchant data

    # Item lookups by ID
    _items_by_id: dict[int, str] = field(default_factory=dict)  # item_id -> item_name

    # Session tracking
    _current_session: SessionLedger | None = None

    # Loaded flag
    _loaded: bool = False

    # ── Initialization ──────────────────────────────────────────────────

    def load(self, data_path: str | None = None) -> None:
        """Load all YAML data files.

        Args:
            data_path: Path to the data directory. If None, uses default.
        """
        with self._lock:
            self._load_item_values(data_path)
            self._load_market_prices(data_path)
            self._load_npc_merchants(data_path)
            self._build_id_index()
            self._loaded = True
            logger.info(
                "EconomyEngine loaded: %d items, %d market prices",
                len(self._item_values), len(self._market_prices),
            )

    def _resolve_data_path(self, data_path: str | None) -> str:
        """Resolve the data directory path."""
        if data_path:
            return data_path
        return _DEFAULT_DATA_DIR

    def _load_item_values(self, data_path: str | None) -> None:
        """Load item_values.yaml."""
        path = os.path.join(self._resolve_data_path(data_path), "item_values.yaml")
        if not os.path.exists(path):
            logger.warning("item_values.yaml not found at %s", path)
            self._item_values = {}
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                logger.warning("item_values.yaml root is not a mapping")
                self._item_values = {}
                return

            # Filter out comments-only entries and validate
            cleaned: dict[str, dict[str, Any]] = {}
            for key, val in data.items():
                if isinstance(val, dict) and "id" not in val and "classification" not in val:
                    continue  # Skip section headers / comments
                if isinstance(val, dict) and "classification" in val:
                    cleaned[key] = val
                elif isinstance(val, dict) and "id" in val:
                    cleaned[key] = val

            self._item_values = cleaned
            self._item_values_path = path
            logger.debug("Loaded %d items from item_values.yaml", len(self._item_values))

        except (yaml.YAMLError, OSError) as exc:
            logger.warning("Failed to load item_values.yaml: %s", exc)
            self._item_values = {}

    def _load_market_prices(self, data_path: str | None) -> None:
        """Load market_prices.yaml."""
        path = os.path.join(self._resolve_data_path(data_path), "market_prices.yaml")
        if not os.path.exists(path):
            logger.warning("market_prices.yaml not found at %s", path)
            self._market_prices = {}
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                logger.warning("market_prices.yaml root is not a mapping")
                self._market_prices = {}
                return

            cleaned: dict[str, dict[str, Any]] = {}
            for key, val in data.items():
                if isinstance(val, dict) and "market_buy" in val:
                    cleaned[key] = val

            self._market_prices = cleaned
            self._market_prices_path = path
            logger.debug("Loaded %d market prices from market_prices.yaml", len(self._market_prices))

        except (yaml.YAMLError, OSError) as exc:
            logger.warning("Failed to load market_prices.yaml: %s", exc)
            self._market_prices = {}

    def _load_npc_merchants(self, data_path: str | None) -> None:
        """Load NPC merchant data embedded in item_values (npc_buy price > 0)."""
        merchants: dict[str, dict[str, Any]] = {}
        with self._lock:
            for item_name, item_data in self._item_values.items():
                npc_buy = item_data.get("npc_buy", 0) or 0
                if npc_buy > 0:
                    # These items can be bought from NPCs
                    category = item_data.get("category", "general")
                    if category not in merchants:
                        merchants[category] = {
                            "npc_name": f"NPC_{category}_shop",
                            "category": category,
                            "items": [],
                        }
                    merchants[category]["items"].append({
                        "item_name": item_name,
                        "item_id": item_data.get("id", 0),
                        "npc_buy": npc_buy,
                        "classification": item_data.get("classification", "sell_npc"),
                    })

            # Sort items by price within each merchant
            for merchant in merchants.values():
                merchant["items"].sort(key=lambda x: x["npc_buy"])

            self._npc_merchants = merchants
            logger.debug("Built %d NPC merchant categories", len(merchants))

    def _build_id_index(self) -> None:
        """Build a lookup index mapping item IDs to item names."""
        index: dict[int, str] = {}
        for name, data in self._item_values.items():
            item_id = data.get("id")
            if item_id:
                # Normalize to int
                try:
                    index[int(item_id)] = name
                except (ValueError, TypeError):
                    pass
        self._items_by_id = index

    # ── Session Management ──────────────────────────────────────────────

    def start_session(self, session_id: str | None = None) -> SessionLedger:
        """Start a new tracking session.

        Args:
            session_id: Optional custom session ID. Auto-generated if None.

        Returns:
            The new SessionLedger.
        """
        import uuid
        sid = session_id or str(uuid.uuid4())[:8]
        ledger = SessionLedger(session_id=sid)
        with self._lock:
            self._current_session = ledger
        logger.info("EconomyEngine session started: %s", sid)
        return ledger

    def get_session(self) -> SessionLedger | None:
        """Get the current session ledger, if any."""
        with self._lock:
            return self._current_session

    def record_earnings(self, amount: int, currency: str = "zeny",
                        source: str = "drop", item: dict[str, Any] | None = None) -> None:
        """Record earnings in the current session.

        Args:
            amount: Amount earned.
            currency: Currency type ("zeny", "tokens", etc.).
            source: Source of earnings ("drop", "sell", "quest", etc.).
            item: Optional item details.
        """
        with self._lock:
            session = self._current_session
            if session is None:
                return
            if currency == "zeny":
                session.zeny_earned += amount
            elif currency == "tokens":
                session.tokens_earned += amount
            if item:
                item_data = dict(item)
                item_data["source"] = source
                session.items_sold.append(item_data)

    def record_spending(self, amount: int, currency: str = "zeny",
                        purpose: str = "potion", item: dict[str, Any] | None = None) -> None:
        """Record spending in the current session.

        Args:
            amount: Amount spent.
            currency: Currency type ("zeny", "tokens", etc.).
            purpose: Purpose of spending ("potion", "warp", "gear", etc.).
            item: Optional item details.
        """
        with self._lock:
            session = self._current_session
            if session is None:
                return
            if currency == "zeny":
                session.zeny_spent += amount
            elif currency == "tokens":
                session.tokens_spent += amount
            if item:
                item_data = dict(item)
                item_data["purpose"] = purpose
                session.items_bought.append(item_data)

    def record_kills(self, count: int = 1) -> None:
        """Record kills for session tracking."""
        with self._lock:
            if self._current_session:
                self._current_session.total_kills += count

    def record_drops(self, count: int = 1) -> None:
        """Record drops for session tracking."""
        with self._lock:
            if self._current_session:
                self._current_session.total_drops += count

    def end_session(self) -> dict[str, Any] | None:
        """End the current session and return its summary.

        Returns:
            Session summary dict, or None if no active session.
        """
        with self._lock:
            session = self._current_session
            if session is None:
                return None
            summary = session.summary()
            self._current_session = None
            logger.info(
                "EconomyEngine session ended: net_zeny=%+d, net_tokens=%+d",
                summary["net_zeny"], summary["net_tokens"],
            )
            return summary

    # ── Core Public API ─────────────────────────────────────────────────

    def loot_value(self, item_name: str, item_id: int | None = None) -> LootValuation | None:
        """Classify a loot item and estimate its value.

        Args:
            item_name: Name of the item (can be display name or internal name).
            item_id: Optional item ID for more reliable lookup.

        Returns:
            LootValuation with classification and zeny estimate, or None if unknown.
        """
        with self._lock:
            item_data = self._find_item_by_name_or_id(item_name, item_id)
            if item_data is None:
                return self._loot_value_unknown(item_name, item_id)

            name = item_data["_resolved_name"]
            data = item_data["_resolved_data"]

            item_id_val = data.get("id", item_id or 0) or 0
            classification = data.get("classification", "sell_npc")
            npc_sell = data.get("npc_sell", 0) or 0
            market_price = data.get("market_price", 0) or 0
            keep_minimum = data.get("keep_minimum", 0) or 0
            category = data.get("category", "unknown")
            tags = data.get("tags", []) or []

            # Determine best zeny estimate
            estimated_zeny = self._estimate_zeny(classification, npc_sell, market_price)

            # Weight estimate (default 1 if not available, since RO items vary)
            weight = data.get("weight", 1) or 1
            zeny_per_weight = round(estimated_zeny / max(weight, 1), 2)

            # Generate human-readable recommendation
            recommendation = self._recommend_loot_action(classification, estimated_zeny)

            return LootValuation(
                item_name=name,
                item_id=int(item_id_val),
                classification=classification,
                npc_sell_price=npc_sell,
                market_price=market_price,
                estimated_zeny=estimated_zeny,
                estimated_zeny_per_weight=zeny_per_weight,
                keep_minimum=keep_minimum,
                category=category,
                tags=tags,
                recommendation=recommendation,
            )

    def _find_item_by_name_or_id(self, item_name: str, item_id: int | None) -> dict[str, Any] | None:
        """Find an item in the loaded YAML data by name or ID."""
        # Try exact name match first
        if item_name in self._item_values:
            return {
                "_resolved_name": item_name,
                "_resolved_data": self._item_values[item_name],
            }

        # Try name variations (lowercase, underscores vs spaces, etc.)
        name_lower = item_name.lower().replace(" ", "_").replace("-", "_")
        for key, data in self._item_values.items():
            if key.lower() == name_lower:
                return {
                    "_resolved_name": key,
                    "_resolved_data": data,
                }

        # Try by ID
        if item_id is not None:
            resolved_name = self._items_by_id.get(item_id)
            if resolved_name and resolved_name in self._item_values:
                return {
                    "_resolved_name": resolved_name,
                    "_resolved_data": self._item_values[resolved_name],
                }

        # Try fuzzy match on name (partial match)
        for key, data in self._item_values.items():
            if name_lower in key.lower() or key.lower() in name_lower:
                return {
                    "_resolved_name": key,
                    "_resolved_data": data,
                }

        return None

    def _loot_value_unknown(self, item_name: str, item_id: int | None) -> LootValuation | None:
        """Handle unknown items — return a best-effort valuation."""
        # Check if it's in market_prices but not item_values
        market_data = self._market_prices.get(item_name)
        if market_data:
            market_buy = market_data.get("market_buy", 0) or 0
            market_sell = market_data.get("market_sell", 0) or 0
            return LootValuation(
                item_name=item_name,
                item_id=item_id or 0,
                classification="sell_player",
                npc_sell_price=0,
                market_price=market_buy,
                estimated_zeny=market_sell or market_buy,
                estimated_zeny_per_weight=0.0,
                keep_minimum=0,
                category="unknown",
                tags=[],
                recommendation="Unknown item — check market price before selling",
            )

        # Try name variations in market_prices
        name_lower = item_name.lower().replace(" ", "_").replace("-", "_")
        for key, data in self._market_prices.items():
            if key.lower() == name_lower:
                market_buy = data.get("market_buy", 0) or 0
                market_sell = data.get("market_sell", 0) or 0
                return LootValuation(
                    item_name=item_name,
                    item_id=item_id or 0,
                    classification="sell_player",
                    npc_sell_price=0,
                    market_price=market_buy,
                    estimated_zeny=market_sell or market_buy,
                    estimated_zeny_per_weight=0.0,
                    keep_minimum=0,
                    category="unknown",
                    tags=[],
                    recommendation="Market-unknown item — worth investigating",
                )

        return None

    def _estimate_zeny(self, classification: str, npc_sell: int, market_price: int) -> int:
        """Estimate the best zeny value for an item given its classification."""
        if classification in ("keep", "crafting", "quest", "material", "potion_food"):
            # These are for use, not for sale — value is what you'd save by not buying
            return max(npc_sell, market_price)
        elif classification == "discard":
            return 0
        elif classification == "sell_player":
            # Prefer market price if available and higher
            return max(market_price, npc_sell)
        elif classification == "sell_any":
            return max(market_price, npc_sell)
        else:  # sell_npc or default
            return npc_sell

    def _recommend_loot_action(self, classification: str, estimated_zeny: int) -> str:
        """Generate a human-readable action recommendation for a loot item."""
        if classification == "keep":
            return "Keep — item is useful equipment or card"
        elif classification == "crafting":
            return "Keep for crafting (alchemy/forging/cooking)"
        elif classification == "quest":
            return "Keep for quest turn-in"
        elif classification == "potion_food":
            return f"Keep for personal use (worth ~{estimated_zeny:,}z)"
        elif classification == "material":
            return f"Keep for upgrades (worth ~{estimated_zeny:,}z)"
        elif classification == "discard":
            return "Discard — literal junk, not worth inventory space"
        elif classification == "sell_player":
            return f"Sell to players for ~{estimated_zeny:,}z (market premium)"
        elif classification == "sell_any":
            return f"Sell to NPC for {estimated_zeny:,}z or player market"
        else:  # sell_npc
            return f"Sell to NPC vendor for {estimated_zeny:,}z"

    # ── Budget Planning ─────────────────────────────────────────────────

    def budget_planning(self, zeny: int, level: int, job: str,
                        inventory: list[dict[str, Any]]) -> BudgetRecommendation:
        """Generate buy/sell recommendations based on current finances.

        Args:
            zeny: Current zeny balance.
            level: Character level.
            job: Character job/class.
            inventory: List of inventory items, each with at minimum:
                       {"name": str, "quantity": int, "id": int (optional)}.

        Returns:
            BudgetRecommendation with categorized buy/sell/restock advice.
        """
        with self._lock:
            buy: list[dict[str, Any]] = []
            sell_to_npc: list[dict[str, Any]] = []
            sell_to_player: list[dict[str, Any]] = []
            restock: list[dict[str, Any]] = []
            priority_items: list[dict[str, Any]] = []

            total_cost = 0
            total_income = 0

            # Analyze each inventory item
            for inv_item in inventory:
                inv_name = inv_item.get("name", "")
                inv_qty = inv_item.get("quantity", 1)
                inv_id = inv_item.get("id")

                if not inv_name:
                    continue

                valuation = self.loot_value(inv_name, inv_id)
                if valuation is None:
                    continue

                # Check if this is a keep/potion item that needs minimum stock
                if valuation.classification in ("potion_food", "crafting", "quest", "material", "keep"):
                    keep_min = valuation.keep_minimum
                    if keep_min > 0 and inv_qty > keep_min:
                        surplus = inv_qty - keep_min
                        # Sell surplus
                        est_value = valuation.estimated_zeny * surplus
                        rec = {
                            "name": inv_name,
                            "item_id": valuation.item_id,
                            "quantity": surplus,
                            "estimated_zeny": est_value,
                            "classification": valuation.classification,
                        }
                        if valuation.classification in ("potion_food",):
                            # Potions: keep minimum, maybe sell surplus to NPC
                            sell_to_npc.append(rec)
                            total_income += est_value
                        elif valuation.classification in ("crafting", "quest"):
                            # Keep these, but if huge surplus sell some
                            if inv_qty > keep_min * 3:
                                sell_to_player.append(rec)
                                total_income += est_value
                elif valuation.classification in ("sell_npc",):
                    est_value = valuation.estimated_zeny * inv_qty
                    sell_to_npc.append({
                        "name": inv_name,
                        "item_id": valuation.item_id,
                        "quantity": inv_qty,
                        "estimated_zeny": est_value,
                        "classification": valuation.classification,
                    })
                    total_income += est_value
                elif valuation.classification in ("sell_player", "sell_any"):
                    est_value = valuation.estimated_zeny * inv_qty
                    # Check if market premium justifies player selling
                    if valuation.market_price > valuation.npc_sell_price * _MARKET_PREMIUM_THRESHOLD:
                        sell_to_player.append({
                            "name": inv_name,
                            "item_id": valuation.item_id,
                            "quantity": inv_qty,
                            "estimated_zeny": est_value,
                            "market_price": valuation.market_price,
                            "classification": valuation.classification,
                        })
                    else:
                        sell_to_npc.append({
                            "name": inv_name,
                            "item_id": valuation.item_id,
                            "quantity": inv_qty,
                            "estimated_zeny": valuation.npc_sell_price * inv_qty,
                            "classification": valuation.classification,
                        })
                        total_income += valuation.npc_sell_price * inv_qty
                    total_income += est_value
                elif valuation.classification == "discard":
                    # Just note it
                    pass
                elif valuation.classification in ("keep",):
                    # Check if we should sell it (duplicates, etc.)
                    if inv_qty > 1 and valuation.estimated_zeny > 1000:
                        sell_qty = inv_qty - 1  # keep one, sell rest
                        est_value = valuation.estimated_zeny * sell_qty
                        sell_to_player.append({
                            "name": inv_name,
                            "item_id": valuation.item_id,
                            "quantity": sell_qty,
                            "estimated_zeny": est_value,
                            "classification": "sell_player",
                        })
                        total_income += est_value

            # Check if potions need restocking
            potion_shortfall = self._check_potion_restock(inventory)
            for item_name, needed, cost in potion_shortfall:
                restock.append({
                    "name": item_name,
                    "quantity_needed": needed,
                    "estimated_cost": cost,
                })
                total_cost += cost

            # Level-appropriate priority buys
            priority_items = self._get_priority_purchases(zeny, level, job)
            for pri in priority_items:
                total_cost += pri.get("estimated_cost", 0)

            # Generate assessment
            net = total_income - total_cost
            assessment = self._generate_budget_assessment(zeny, net, len(sell_to_npc), len(restock))

            return BudgetRecommendation(
                buy=buy,
                sell_to_npc=sell_to_npc,
                sell_to_player=sell_to_player,
                restock=restock,
                priority_items=priority_items,
                total_estimated_cost=total_cost,
                total_estimated_income=total_income,
                net_impact=net,
                assessment=assessment,
            )

    def _check_potion_restock(self, inventory: list[dict[str, Any]]) -> list[tuple[str, int, int]]:
        """Check which consumables need restocking.

        Returns:
            List of (item_name, quantity_needed, estimated_cost) tuples.
        """
        shortfalls: list[tuple[str, int, int]] = []

        # Build a quick lookup of current inventory quantities
        inv_qty: dict[str, int] = {}
        for item in inventory:
            name = item.get("name", "")
            qty = item.get("quantity", 0)
            if name:
                inv_qty[name.lower().replace(" ", "_")] = qty

        # Check all potion_food items
        for item_name, data in self._item_values.items():
            classification = data.get("classification", "")
            if classification not in ("potion_food",):
                continue

            keep_min = data.get("keep_minimum", 0) or 0
            if keep_min <= 0:
                continue

            current = inv_qty.get(item_name.lower().replace(" ", "_"), 0)
            if current < keep_min:
                needed = keep_min - current
                npc_buy = data.get("npc_buy", 0) or 0
                market_price = data.get("market_price", 0) or 0
                cost = max(npc_buy, market_price) * needed
                shortfalls.append((item_name, needed, cost))

        return shortfalls

    def _get_priority_purchases(self, zeny: int, level: int, job: str) -> list[dict[str, Any]]:
        """Suggest priority gear/equipment purchases for the character's level.

        Returns:
            List of purchase recommendations with cost estimates.
        """
        priorities: list[dict[str, Any]] = []

        # Check market_prices for level-appropriate gear
        weapon_priority = self._find_best_affordable_weapon(zeny, level, job)
        if weapon_priority:
            priorities.append(weapon_priority)

        armor_priority = self._find_best_affordable_armor(zeny, level)
        if armor_priority:
            priorities.append(armor_priority)

        return priorities

    def _find_best_affordable_weapon(self, zeny: int, level: int, job: str) -> dict[str, Any] | None:
        """Find the best weapon affordable for current zeny."""
        # Use market_prices data to find weapons in the right price range
        weapons: list[dict[str, Any]] = []
        for name, data in self._market_prices.items():
            market_buy = data.get("market_buy", 0) or 0
            notes = (data.get("notes", "") or "").lower()
            # Check notes for weapon keywords
            weapon_keywords = ["dagger", "sword", "mace", "staff", "bow", "spear",
                                "katar", "blade", "rod", "wand", "knife"]
            is_weapon = any(kw in notes for kw in weapon_keywords) or name.lower() in [
                "knife", "sword", "mace", "bow", "staff", "rod",
                "main_gauche", "cutter", "composite_bow", "blade",
                "scimitar", "damascus", "wand", "flail", "jur",
                "katar", "spear", "glaive", "chain",
            ]
            if is_weapon and market_buy <= zeny:
                weapons.append({
                    "name": name,
                    "estimated_cost": market_buy,
                    "type": "weapon",
                    "notes": data.get("notes", ""),
                    "priority": 8,  # Weapons are high priority
                })

        if not weapons:
            return None

        # Pick the best affordable (most expensive within budget)
        weapons.sort(key=lambda w: -w["estimated_cost"])
        return weapons[0]

    def _find_best_affordable_armor(self, zeny: int, level: int) -> dict[str, Any] | None:
        """Find the best armor affordable for current zeny."""
        armors: list[dict[str, Any]] = []
        for name, data in self._market_prices.items():
            market_buy = data.get("market_buy", 0) or 0
            notes = (data.get("notes", "") or "").lower()
            is_armor = any(kw in notes for kw in ["def", "armor", "shield", "robe",
                                                    "garment", "shoes", "boots",
                                                    "mdef"])
            if is_armor and market_buy <= zeny and market_buy > 0:
                armors.append({
                    "name": name,
                    "estimated_cost": market_buy,
                    "type": "armor",
                    "notes": data.get("notes", ""),
                    "priority": 7,
                })

        if not armors:
            return None

        armors.sort(key=lambda w: -w["estimated_cost"])
        return armors[0]

    def _generate_budget_assessment(self, zeny: int, net_impact: int,
                                     sell_count: int, restock_count: int) -> str:
        """Generate a human-readable budget assessment."""
        parts: list[str] = []

        if zeny < 1000:
            parts.append("CRITICAL: Very low on zeny — prioritize selling vendor trash")
        elif zeny < 10000:
            parts.append("Low on zeny — focus on farming and selling drops")
        elif zeny < 100000:
            parts.append("Adequate funds — consider upgrading gear")
        elif zeny < 1000000:
            parts.append("Comfortable — you can afford mid-range gear upgrades")
        else:
            parts.append("Wealthy — consider premium gear or investing in market")

        if sell_count > 0:
            parts.append(f"Found {sell_count} item(s) ready for sale")
        if restock_count > 0:
            parts.append(f"Need to restock {restock_count} consumable type(s)")

        if net_impact > 0:
            parts.append(f"Net positive: ~{net_impact:,}z after recommended actions")
        else:
            parts.append(f"Net spend: ~{abs(net_impact):,}z after recommended purchases")

        return " | ".join(parts)

    # ── Restock Check ───────────────────────────────────────────────────

    def restock_needed(self, inventory: list[dict[str, Any]], item_name: str,
                       keep_minimum: int | None = None) -> RestockResult:
        """Check if a specific item needs restocking.

        Args:
            inventory: List of inventory items, each with {"name": str, "quantity": int}.
            item_name: Name of the item to check.
            keep_minimum: Override the keep_minimum from YAML. If None, uses YAML value.

        Returns:
            RestockResult with restock quantity and cost estimate.
        """
        with self._lock:
            # Find current inventory count
            current_count = 0
            for item in inventory:
                inv_name = item.get("name", "")
                if not inv_name:
                    continue
                # Normalize comparison
                if inv_name.lower().replace(" ", "_") == item_name.lower().replace(" ", "_"):
                    current_count = item.get("quantity", 0)
                    break

            # Get the item's keep_minimum from YAML
            data = self._item_values.get(item_name)
            if data is None:
                # Try normalized name
                normalized = item_name.lower().replace(" ", "_")
                for key, val in self._item_values.items():
                    if key.lower() == normalized:
                        data = val
                        break

            resolved_keep_min: int = keep_minimum if keep_minimum is not None else 0
            if keep_minimum is None and data is not None:
                resolved_keep_min = int(data.get("keep_minimum", 0) or 0)

            item_id: int | None = int(data.get("id", 0) or 0) if data else None
            npc_buy: int = int(data.get("npc_buy", 0) or 0) if data else 0
            market_price: int = int(data.get("market_price", 0) or 0) if data else 0
            cost_per: int = max(npc_buy, market_price, 1)

            needs_restock = current_count < resolved_keep_min
            restock_qty: int = max(0, resolved_keep_min - current_count) if needs_restock else 0
            estimated_cost = cost_per * restock_qty

            return RestockResult(
                needs_restock=needs_restock,
                current_count=current_count,
                keep_minimum=resolved_keep_min,
                restock_quantity=restock_qty,
                estimated_cost=estimated_cost,
                item_name=item_name,
                item_id=item_id,
            )

    # ── Merchant Interaction ────────────────────────────────────────────

    def merchant_decision(self, inventory: list[dict[str, Any]],
                          zeny: int, current_map: str = "unknown") -> MerchantDecision | None:
        """Decide if and where to go to a merchant.

        Analyzes inventory for items to sell and consumables to buy, then
        recommends the best NPC/category merchant interaction.

        Args:
            inventory: Current inventory items.
            zeny: Current zeny balance.
            current_map: Current map name for proximity-based decisions.

        Returns:
            MerchantDecision with NPC/category recommendation, or None if no action needed.
        """
        with self._lock:
            # First, check if we need to sell or restock urgently
            urgent_sell: list[dict[str, Any]] = []
            urgent_buy: list[dict[str, Any]] = []
            sell_value = 0
            buy_cost = 0

            # Check for items to sell
            for inv_item in inventory:
                inv_name = inv_item.get("name", "")
                inv_qty = inv_item.get("quantity", 1)

                valuation = self.loot_value(inv_name, inv_item.get("id"))
                if valuation is None:
                    continue

                if valuation.classification in ("sell_npc",) and inv_qty > 0:
                    val = valuation.npc_sell_price * inv_qty
                    urgent_sell.append({
                        "name": inv_name,
                        "quantity": inv_qty,
                        "value": val,
                    })
                    sell_value += val

                elif valuation.classification in ("sell_player",) and inv_qty > 0:
                    # Only recommend selling to player market if premium is significant
                    if valuation.market_price > valuation.npc_sell_price * _MARKET_PREMIUM_THRESHOLD:
                        # Needs player market, not NPC
                        pass
                    else:
                        val = valuation.npc_sell_price * inv_qty
                        urgent_sell.append({
                            "name": inv_name,
                            "quantity": inv_qty,
                            "value": val,
                        })
                        sell_value += val

            # Check for items to buy (low on consumables)
            for item_name, data in self._item_values.items():
                classification = data.get("classification", "")
                if classification not in ("potion_food",):
                    continue
                keep_min = data.get("keep_minimum", 0) or 0
                if keep_min <= 0:
                    continue

                # Find in inventory
                current_qty = 0
                for inv_item in inventory:
                    if inv_item.get("name", "").lower().replace(" ", "_") == item_name.lower():
                        current_qty = inv_item.get("quantity", 0)
                        break

                if current_qty < keep_min:
                    needed = keep_min - current_qty
                    npc_buy = data.get("npc_buy", 0) or 0
                    cost = npc_buy * needed
                    urgent_buy.append({
                        "name": item_name,
                        "quantity_needed": needed,
                        "cost": cost,
                    })
                    buy_cost += cost

            # Determine priority and action
            can_afford_restock = zeny >= buy_cost
            has_items_to_sell = len(urgent_sell) > 0
            needs_restock = len(urgent_buy) > 0 and can_afford_restock

            if not has_items_to_sell and not needs_restock:
                return None

            # Determine best NPC category
            if needs_restock and has_items_to_sell:
                transaction_type = "both"
                priority = 9 if buy_cost > zeny * 0.3 else 7
                reason = f"Have {len(urgent_sell)} item(s) to sell and {len(urgent_buy)} consumable(s) to restock"
            elif needs_restock:
                transaction_type = "buy"
                priority = 8
                reason = f"Low on consumables — need {len(urgent_buy)} restock(s)"
            else:
                transaction_type = "sell"
                priority = 6
                reason = f"Have {len(urgent_sell)} item(s) to sell worth ~{sell_value:,}z"

            # Find the right NPC category
            # Determine what categories are needed
            categories_needed: set[str] = set()
            for item in urgent_buy:
                name = item["name"]
                item_data = self._item_values.get(name, {})
                categories_needed.add(item_data.get("category", "general"))

            # Map categories to NPC names
            if "potion" in str(categories_needed) or not categories_needed:
                npc_name = "NPC_potion_shop"
                npc_map = "prontera"
                npc_coords = (135, 130)
            elif "weapon" in str(categories_needed):
                npc_name = "NPC_weapon_shop"
                npc_map = "prontera"
                npc_coords = (140, 125)
            elif "armor" in str(categories_needed):
                npc_name = "NPC_armor_shop"
                npc_map = "prontera"
                npc_coords = (130, 120)
            else:
                npc_name = "NPC_general_shop"
                npc_map = "prontera"
                npc_coords = (128, 126)

            all_items = urgent_sell + urgent_buy
            total_value = sell_value if transaction_type == "sell" else buy_cost
            # For "both", use the larger value
            if transaction_type == "both":
                total_value = max(sell_value, buy_cost)

            return MerchantDecision(
                npc_name=npc_name,
                npc_map=npc_map,
                npc_coords=npc_coords,
                transaction_type=transaction_type,
                items=all_items,
                total_value=total_value,
                priority=priority,
                reason=reason,
            )

    # ── Market Price Lookup ─────────────────────────────────────────────

    def get_market_price(self, item_name: str) -> dict[str, Any] | None:
        """Look up player-market pricing for an item.

        Args:
            item_name: Name of the item.

        Returns:
            Market price dict with market_buy, market_sell, volatility, or None.
        """
        with self._lock:
            # Direct lookup first
            if item_name in self._market_prices:
                return dict(self._market_prices[item_name])

            # Normalized lookup
            normalized = item_name.lower().replace(" ", "_").replace("-", "_")
            for key, data in self._market_prices.items():
                if key.lower() == normalized:
                    return dict(data)

            # Check item_values for market_price field
            item_data = self._item_values.get(item_name)
            if item_data is None:
                for key, data in self._item_values.items():
                    if key.lower() == normalized:
                        item_data = data
                        break

            if item_data and "market_price" in item_data:
                mp = item_data["market_price"] or 0
                return {
                    "market_buy": mp,
                    "market_sell": int(mp * 0.8),
                    "volatility": "low",
                    "source": "item_values",
                }

            return None

    def get_market_opportunities(self, min_profit: int = 1000) -> list[dict[str, Any]]:
        """Find arbitrage/selling opportunities by comparing item values with market prices.

        Args:
            min_profit: Minimum zeny profit to consider an opportunity.

        Returns:
            List of opportunity dicts with item, buy, sell, profit info.
        """
        opportunities: list[dict[str, Any]] = []

        with self._lock:
            for item_name, item_data in self._item_values.items():
                npc_sell = item_data.get("npc_sell", 0) or 0
                market_price = item_data.get("market_price", 0) or 0
                classification = item_data.get("classification", "")

                # Check for player-market premium
                if market_price > npc_sell and market_price > min_profit:
                    profit = market_price - npc_sell
                    if profit >= min_profit and classification in ("sell_player", "sell_any"):
                        opportunities.append({
                            "item_name": item_name,
                            "item_id": item_data.get("id", 0),
                            "npc_sell_price": npc_sell,
                            "market_price": market_price,
                            "profit_per_item": profit,
                            "strategy": "sell_to_player",
                            "classification": classification,
                        })

            # Sort by profit descending
            opportunities.sort(key=lambda o: -o["profit_per_item"])

        return opportunities

    # ── Bulk Classification ─────────────────────────────────────────────

    def classify_inventory(self, inventory: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        """Classify an entire inventory into action buckets.

        Args:
            inventory: List of inventory items.

        Returns:
            Dict mapping action bucket names to lists of items.
        """
        buckets: dict[str, list[dict[str, Any]]] = {
            "keep": [],
            "sell_npc": [],
            "sell_player": [],
            "discard": [],
            "crafting": [],
            "quest": [],
            "potion_food": [],
            "unknown": [],
        }

        for item in inventory:
            name = item.get("name", "")
            qty = item.get("quantity", 1)
            valuation = self.loot_value(name, item.get("id"))
            if valuation is None:
                buckets["unknown"].append(item)
            else:
                cls = valuation.classification
                bucket = buckets.get(cls, buckets["unknown"])
                bucket.append({
                    "name": name,
                    "quantity": qty,
                    "estimated_zeny": valuation.estimated_zeny * qty,
                    "classification": cls,
                    "recommendation": valuation.recommendation,
                })

        return buckets

    # ── Info / Summary ──────────────────────────────────────────────────

    def get_item_summary(self, item_name: str) -> dict[str, Any] | None:
        """Get a comprehensive summary of an item.

        Args:
            item_name: Name of the item.

        Returns:
            Dict with item values, market price, and recommendations, or None.
        """
        with self._lock:
            valuation = self.loot_value(item_name)
            if valuation is None:
                return None

            market = self.get_market_price(item_name)

            return {
                "name": valuation.item_name,
                "id": valuation.item_id,
                "classification": valuation.classification,
                "npc_sell_price": valuation.npc_sell_price,
                "market_price": valuation.market_price,
                "estimated_zeny": valuation.estimated_zeny,
                "estimated_zeny_per_weight": valuation.estimated_zeny_per_weight,
                "keep_minimum": valuation.keep_minimum,
                "category": valuation.category,
                "tags": valuation.tags,
                "recommendation": valuation.recommendation,
                "market_detail": market,
            }

    def engine_status(self) -> dict[str, Any]:
        """Get status info about the economy engine."""
        with self._lock:
            return {
                "loaded": self._loaded,
                "items_loaded": len(self._item_values),
                "market_prices_loaded": len(self._market_prices),
                "npc_merchant_categories": len(self._npc_merchants),
                "item_values_path": self._item_values_path,
                "market_prices_path": self._market_prices_path,
                "active_session": self._current_session is not None,
            }

    def get_session_summary(self) -> dict[str, Any] | None:
        """Get a summary of the current session (if active)."""
        session = self.get_session()
        if session is None:
            return None
        return session.summary()


# ── Factory Function ──────────────────────────────────────────────────────


def create_economy_engine(data_path: str) -> EconomyEngine:
    """Create and fully load an EconomyEngine instance.

    Args:
        data_path: Path to the directory containing item_values.yaml and
                   market_prices.yaml.

    Returns:
        A fully initialized EconomyEngine.

    Example:
        >>> engine = create_economy_engine("path/to/data")
        >>> val = engine.loot_value("red_potion", 501)
        >>> val.classification
        'potion_food'
    """
    engine = EconomyEngine()
    engine.load(data_path)
    return engine