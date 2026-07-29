"""Quest Items Database — loaded from AI_sidecar/data/quest_items.yaml.

Provides the inventory system with a list of items that should NEVER
be sold because they are needed for quest turn-ins, job changes,
crafting, or pet taming.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_QUEST_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "quest_items.yaml"


class QuestItemsDB:
    """Quest items database — items the bot must keep for quests.

    Loads from quest_items.yaml on init. Provides lookup methods
    to check if an item is quest-critical.
    """

    def __init__(self, yaml_path: str | Path | None = None) -> None:
        self._yaml_path = Path(yaml_path) if yaml_path else _DEFAULT_QUEST_PATH
        self._items: dict[str, dict[str, Any]] = {}
        self._by_id: dict[int, list[dict[str, Any]]] = {}
        self._loaded = False
        self._load()

    def _load(self) -> None:
        """Load and index quest items from YAML."""
        path = self._yaml_path
        if not path.exists():
            logger.warning("Quest items DB not found at %s, using empty DB", path)
            self._loaded = True
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        except Exception as exc:
            logger.error("Failed to load quest items from %s: %s", path, exc)
            self._loaded = True
            return

        if not isinstance(raw, dict):
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
            "Loaded %d quest item entries from %s",
            len(self._items),
            path,
        )
        self._loaded = True

    def reload(self) -> None:
        """Reload from disk."""
        self._items.clear()
        self._by_id.clear()
        self._load()

    def is_quest_item_name(self, item_name: str) -> bool:
        """Check if an item name matches a quest item entry."""
        if not item_name:
            return False
        norm = item_name.strip().lower().replace(" ", "_").replace("-", "_")

        # Exact match on key
        if norm in self._items:
            return True

        # Partial match
        for key, _data in self._items.items():
            if norm in key or key in norm:
                return True
            item_name_field = _data.get("item_name", "").lower()
            if norm in item_name_field or item_name_field in norm:
                return True

        return False

    def is_quest_item_id(self, item_id: int) -> bool:
        """Check if an item ID is a quest item."""
        return item_id in self._by_id

    def get_quest_info(self, item_name: str) -> dict[str, Any] | None:
        """Get quest details for an item name."""
        norm = item_name.strip().lower().replace(" ", "_").replace("-", "_")
        if norm in self._items:
            return self._items[norm]

        for key, data in self._items.items():
            if norm in key or key in norm:
                return data
            item_name_field = data.get("item_name", "").lower()
            if norm in item_name_field or item_name_field in norm:
                return data
        return None

    def get_quest_info_by_id(self, item_id: int) -> list[dict[str, Any]]:
        """Get quest details for an item ID."""
        return self._by_id.get(item_id, [])

    def get_all_quest_items(self) -> dict[str, dict[str, Any]]:
        """Get the full quest items dict."""
        return dict(self._items)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def __len__(self) -> int:
        return len(self._items)


# Singleton for quest items DB
_quest_db: QuestItemsDB | None = None


def get_quest_items_db() -> QuestItemsDB:
    """Get the global QuestItemsDB instance."""
    global _quest_db
    if _quest_db is None:
        _quest_db = QuestItemsDB()
    return _quest_db


def is_quest_item(item_name_or_id: str | int) -> bool:
    """Convenience: check if an item is a quest item by name or ID."""
    db = get_quest_items_db()
    if isinstance(item_name_or_id, int):
        return db.is_quest_item_id(item_name_or_id)
    return db.is_quest_item_name(item_name_or_id)
