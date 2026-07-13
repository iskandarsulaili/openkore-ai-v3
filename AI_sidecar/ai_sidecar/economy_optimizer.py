"""
Economy optimizer — uses knowledge.json for farm targets, potion budgets, and item values.

Replaces hardcoded FARM_TARGETS with dynamic data from 35,525 items in knowledge.json.
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


@dataclass(slots=True)
class EconomyOptimizer:
    """Dynamic economy optimizer using knowledge.json data."""
    
    _lock: RLock = field(default_factory=RLock)
    _knowledge: dict[str, Any] = field(default_factory=dict)
    _stats: dict[str, int] = field(default_factory=lambda: {"queries": 0})
    
    def __post_init__(self) -> None:
        self._load_knowledge()
    
    def _load_knowledge(self) -> None:
        for candidate in [
            str(Path(__file__).parent.parent.parent / "knowledge" / "knowledge.json"),
            str(Path(__file__).parent.parent / "knowledge" / "knowledge.json"),
            "knowledge/knowledge.json",
        ]:
            if candidate and Path(candidate).exists():
                try:
                    self._knowledge = json.loads(Path(candidate).read_text(encoding="utf-8"))
                    logger.info("economy_optimizer_loaded: %d items", len(self._knowledge.get("items", {}).get("all", [])))
                    return
                except Exception:
                    continue
        logger.warning("economy_optimizer: knowledge.json not found")
    
    def get_farm_targets(self, level: int) -> list[dict[str, Any]]:
        """Get valuable farm targets for a given level from knowledge.json."""
        targets = []
        items_all = self._knowledge.get("items", {}).get("all", [])
        for item in items_all:
            buy = int(item.get("Buy", 0) or 0)
            sell = int(item.get("Sell", 0) or 0)
            weight = int(item.get("Weight", 0) or 0)
            value = max(buy, sell)
            if value > 500 and weight < 100:  # Valuable and light
                targets.append({
                    "name": item.get("Name", item.get("AegisName", "?")),
                    "value": value,
                    "weight": weight,
                    "type": item.get("Type", ""),
                })
        targets.sort(key=lambda t: t["value"], reverse=True)
        return targets[:20]
    
    def get_potion_budgets(self, level: int, zeny: int) -> list[dict[str, Any]]:
        """Get affordable potions for a given level and zeny."""
        potions = []
        items_all = self._knowledge.get("items", {}).get("all", [])
        for item in items_all:
            if not isinstance(item, dict):
                continue
            name = str(item.get("Name", "")).lower()
            item_type = str(item.get("Type", "")).strip()
            buy = int(item.get("Buy", 0) or 0)
            if buy <= 0 or buy > zeny:
                continue
            # Healing type items are always potions
            if item_type == "Healing":
                potions.append({
                    "name": item.get("Name", item.get("AegisName", "?")),
                    "buy": buy,
                    "weight": int(item.get("Weight", 0) or 0),
                })
                continue
            # Usable items: only include if name suggests healing/restoration
            if item_type in ("Usable", "Usable_Delayed", "DelayConsume", "Delayconsume"):
                name_norm = name.replace(" ", "_").replace("-", "_")
                if any(kw in name_norm for kw in ["potion", "berry", "panacea", "herb",
                                                     "healing", "recovery", "cure", "antidote",
                                                     "vitamin", "medicine", "elixir", "tonic",
                                                     "syrup", "condensed"]):
                    potions.append({
                        "name": item.get("Name", item.get("AegisName", "?")),
                        "buy": buy,
                        "weight": int(item.get("Weight", 0) or 0),
                    })
        potions.sort(key=lambda p: p["buy"])
        return potions[:20]
    
    def get_item_value(self, item_name: str) -> dict[str, Any]:
        """Get the value of a specific item."""
        items_all = self._knowledge.get("items", {}).get("all", [])
        for item in items_all:
            if not isinstance(item, dict):
                continue
            if str(item.get("Name", "")).lower() == item_name.lower() or \
               str(item.get("AegisName", "")).lower() == item_name.lower():
                return {
                    "name": item.get("Name", item.get("AegisName", "?")),
                    "buy": int(item.get("Buy", 0) or 0),
                    "sell": int(item.get("Sell", 0) or 0),
                    "weight": int(item.get("Weight", 0) or 0),
                    "type": item.get("Type", ""),
                }
        return {"name": item_name, "buy": 0, "sell": 0, "weight": 0, "type": "unknown"}
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)
