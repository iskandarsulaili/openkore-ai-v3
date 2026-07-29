"""
RO Item Value Database — loaded from AI_sidecar/data/item_values.yaml.

Provides efficient lookups by item name, ID, classification, category, and tags.
All values are thread-safe (read-only after load).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ── Default path ──
_DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "item_values.yaml"


# ── Classification constants ──
KEEP = "keep"
SELL_NPC = "sell_npc"
SELL_PLAYER = "sell_player"
SELL_ANY = "sell_any"
DISCARD = "discard"
CRAFTING = "crafting"
QUEST = "quest"
POTION_FOOD = "potion_food"
MATERIAL = "material"

# Priority order for classification (used by inventory management)
CLASSIFICATION_PRIORITY: dict[str, int] = {
    KEEP: 1,         # never sell
    MATERIAL: 2,     # never sell
    CRAFTING: 3,     # never sell (used for alchemy/forging)
    QUEST: 4,        # never sell (turn-in value)
    POTION_FOOD: 5,  # sell excess above keep_minimum
    SELL_PLAYER: 6,  # sell to players only, NPC as last resort
    SELL_ANY: 7,     # sell to whoever pays more
    SELL_NPC: 8,     # sell to NPC shop
    DISCARD: 9,      # drop on ground if needed
}


class ItemValueDB:
    """Thread-safe (read-only) item value database.

    Loads from item_values.yaml on init. Provides flexible lookups.
    """

    def __init__(self, yaml_path: str | Path | None = None) -> None:
        self._yaml_path = Path(yaml_path) if yaml_path else _DEFAULT_DATA_PATH
        self._items: dict[str, dict[str, Any]] = {}  # normalized_name -> data
        self._by_id: dict[int, list[dict[str, Any]]] = {}  # item_id -> matches
        self._loaded = False
        self._load()

    # ── Loading ───────────────────────────────────────────────────

    def _load(self) -> None:
        """Load and index item values from YAML."""
        path = self._yaml_path
        if not path.exists():
            logger.warning("Item value DB not found at %s, using empty DB", path)
            self._loaded = True
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        except Exception as exc:
            logger.error("Failed to load item values from %s: %s", path, exc)
            self._loaded = True
            return

        if not isinstance(raw, dict):
            logger.warning("Item values YAML has no top-level keys")
            self._loaded = True
            return

        for norm_name, data in raw.items():
            if not isinstance(data, dict):
                continue
            data["_key"] = norm_name
            self._items[norm_name] = data

            # Index by item ID
            item_id = data.get("id")
            if item_id is not None:
                self._by_id.setdefault(int(item_id), []).append(data)

        logger.info(
            "Loaded %d items from item value DB (%d indexed by ID)",
            len(self._items),
            len(self._by_id),
        )
        self._loaded = True

    def reload(self) -> None:
        """Reload from disk (for hot-reload scenarios)."""
        self._items.clear()
        self._by_id.clear()
        self._load()

    # ── Lookups ───────────────────────────────────────────────────

    def get(self, item_name: str) -> dict[str, Any] | None:
        """Look up an item by its normalized name.

        Args:
            item_name: Display name, normalized name, or partial match.

        Returns:
            Item data dict or None.
        """
        if not item_name:
            return None

        norm = item_name.strip().lower().replace(" ", "_").replace("-", "_")

        # Exact match first
        if norm in self._items:
            return self._items[norm]

        # Try direct key (YAML key)
        if item_name in self._items:
            return self._items[item_name]

        # Partial match: search all keys
        for key, data in self._items.items():
            if norm in key or key in norm:
                return data

        return None

    def get_by_id(self, item_id: int) -> list[dict[str, Any]]:
        """Look up items by their numeric ID.

        Args:
            item_id: rAthena item ID (e.g. 909 for Jellopy).

        Returns:
            List of matching item data dicts (multiple entries possible
            for slotted variants).
        """
        return self._by_id.get(item_id, [])

    def search(self, **filters: Any) -> list[dict[str, Any]]:
        """Search items by classification, category, tag, etc.

        Args:
            **filters: Key-value pairs to match against item data.
                       Special key 'tags' matches any of the listed tags.

        Returns:
            List of matching item data dicts.
        """
        results: list[dict[str, Any]] = []
        tags_filter = filters.pop("tags", None)
        if isinstance(tags_filter, str):
            tags_filter = [tags_filter]

        for data in self._items.values():
            match = True
            for key, value in filters.items():
                if key == "tags" and tags_filter:
                    continue  # handled below
                if data.get(key) != value:
                    match = False
                    break
            if not match:
                continue
            if tags_filter:
                item_tags = data.get("tags", [])
                if not any(t in item_tags for t in tags_filter):
                    continue
            results.append(data)

        return results

    # ── Value methods ─────────────────────────────────────────────

    def get_npc_sell_price(self, item_name: str) -> int:
        """Get the NPC sell price (what you get when selling to NPC)."""
        data = self.get(item_name)
        if not data:
            return 0
        return int(data.get("npc_sell", 0))

    def get_npc_buy_price(self, item_name: str) -> int:
        """Get the NPC buy price (what it costs from NPC)."""
        data = self.get(item_name)
        if not data:
            return 0
        return int(data.get("npc_buy", 0))

    def get_market_price(self, item_name: str) -> int:
        """Get the estimated player market price (vending / player shop)."""
        data = self.get(item_name)
        if not data:
            return 0
        return int(data.get("market_price", 0))

    def get_best_price(self, item_name: str) -> int:
        """Get the best possible price (max of npc_sell and market_price)."""
        data = self.get(item_name)
        if not data:
            return 0
        return max(int(data.get("npc_sell", 0)), int(data.get("market_price", 0)))

    def get_classification(self, item_name: str) -> str:
        """Get item classification (keep / sell_npc / sell_player / etc.)."""
        data = self.get(item_name)
        if not data:
            return SELL_NPC  # default: sell to NPC
        return str(data.get("classification", SELL_NPC))

    def is_card(self, item_name: str) -> bool:
        """Check if an item is a card (always keep)."""
        data = self.get(item_name)
        return data is not None and data.get("category") == "card"

    def is_quest_item(self, item_name: str) -> bool:
        """Check if an item is needed for quests."""
        data = self.get(item_name)
        if not data:
            return False
        return data.get("classification") == QUEST

    def is_crafting_material(self, item_name: str) -> bool:
        """Check if an item is used for crafting."""
        data = self.get(item_name)
        if not data:
            return False
        return data.get("classification") == CRAFTING

    def is_upgrade_material(self, item_name: str) -> bool:
        """Check if an item is used for refining/upgrading."""
        data = self.get(item_name)
        if not data:
            return False
        return data.get("classification") == MATERIAL

    def get_keep_minimum(self, item_name: str) -> int:
        """Get the minimum stock to keep for consumable items."""
        data = self.get(item_name)
        if not data:
            return 0
        return int(data.get("keep_minimum", 0))

    def get_all_item_names(self) -> list[str]:
        """Get all known item names."""
        return list(self._items.keys())

    def get_all_items(self) -> dict[str, dict[str, Any]]:
        """Get the full items dict (read-only)."""
        return dict(self._items)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, item_name: str) -> bool:
        return self.get(item_name) is not None
