"""Per-Map Inventory Policies — different keep/sell rules per map.

A pro player configures per-map:
- prt_fild05: keep Jellopy (quest), sell everything else
- pay_dun00: keep Bat Wings (quest), keep all cards, sell rest
- mjolnir_01: focus on Ores and materials

Policies auto-switch when the bot moves to a different map.
"""
from __future__ import annotations
from typing import Any
import logging
from pathlib import Path

from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    yaml = None

_DATA_DIR = Path(__file__).parent.parent.parent / "data"


class MapInventoryPolicy:
    """Keep/sell policy for a single map."""
    
    def __init__(self, data: dict):
        self.keep_items: set[str] = set(i.lower() for i in data.get("keep_items", []))
        self.sell_items: set[str] = set(i.lower() for i in data.get("sell_items", []))
        self.discard_items: set[str] = set(i.lower() for i in data.get("discard_items", []))
        self.keep_cards: bool = data.get("keep_cards", True)
        self.keep_quest_items: bool = data.get("keep_quest_items", True)
        self.sell_at_weight: int = data.get("sell_at_weight", 80)
        self.keep_potions_min: int = data.get("keep_potions_min", 10)
        self.description: str = data.get("description", "")
    
    def should_keep(self, item_name: str, item_class: str = "") -> bool:
        name_lower = item_name.lower()
        if name_lower in self.keep_items:
            return True
        if name_lower in self.sell_items:
            return False
        if name_lower in self.discard_items:
            return False
        if self.keep_cards and ("card" in name_lower or item_class == "CARD"):
            return True
        if self.keep_quest_items and ("jellopy" in name_lower or "fabric" in name_lower or "sticky" in name_lower or "feather" in name_lower or "bat wing" in name_lower or "hollow" in name_lower):
            return True
        # Default: keep potions, sell everything else
        if "potion" in name_lower or "juice" in name_lower:
            return True
        return False


class InventoryPolicies:
    """Per-map inventory policies manager."""
    
    _DATA_PATH = _DATA_DIR / "map_inventory_policies.yaml"
    
    def __init__(self):
        self._policies: dict[str, MapInventoryPolicy] = {}
        self._default = MapInventoryPolicy({
            "keep_items": [],
            "sell_items": [],
            "discard_items": [],
            "keep_cards": True,
            "keep_quest_items": True,
            "sell_at_weight": 80,
            "keep_potions_min": 10,
            "description": "Default: sell everything at 80% weight",
        })
        self._load()
    
    def _load(self) -> None:
        if yaml is None:
            return
        path = self._DATA_PATH
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            policies = data.get("map_policies", {})
            for map_name, policy_data in policies.items():
                self._policies[map_name] = MapInventoryPolicy(policy_data)
            logger.info(f"Loaded {len(self._policies)} per-map inventory policies")
    
    def get_policy(self, map_name: str) -> MapInventoryPolicy:
        """Get policy for a map, falling back to default."""
        return self._policies.get(map_name, self._default)
    
    def assess(self, signals: dict[str, Any], actions: list[HeuristicAction], bot_id: str) -> None:
        """Apply per-map inventory policy."""
        current_map = str(signals.get("map", "") or "")
        policy = self.get_policy(current_map)
        
        weight = int(signals.get("weight", 0) or 0)
        weight_max = int(signals.get("weight_max", 100) or 100)
        weight_pct = weight / max(weight_max, 1) * 100
        
        # Check if we need to sell
        if weight_pct >= policy.sell_at_weight:
            actions.append(HeuristicAction(
                kind="command",
                command="sellAuto 1",
                confidence=0.9,
                reason=f"Inventory {weight_pct:.0f}% — selling per {current_map} policy",
                domain="economy",
            ))
        
        # Log current policy
        actions.append(HeuristicAction(
            kind="log",
            command=f"inventory_policy map={current_map} sell_at={policy.sell_at_weight}% keep_cards={policy.keep_cards}",
            confidence=0.5,
            reason=f"Per-map inventory: {policy.description}",
            domain="economy",
        ))


class SpawnNavigator:
    """Navigates to spawn hotspots for maximum kills per hour.
    
    Instead of walking through the map portal and attacking randomly,
    walks between known spawn clusters in a loop.
    """
    
    _DATA_PATH = _DATA_DIR / "spawn_hotspots.yaml"
    
    def __init__(self):
        self._hotspots: dict[str, list[list[int]]] = {}
        self._current_hotspot_index: dict[str, int] = {}  # bot_id -> index
        self._load()
    
    def _load(self) -> None:
        if yaml is None:
            return
        path = self._DATA_PATH
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            for map_name, map_data in data.items():
                hotspots = map_data.get("hotspots", []) if isinstance(map_data, dict) else []
                self._hotspots[map_name] = hotspots
    
    def get_next_hotspot(self, map_name: str, bot_id: str) -> tuple[int, int] | None:
        """Get the next spawn hotspot to walk to."""
        hotspots = self._hotspots.get(map_name)
        if not hotspots:
            return None
        
        idx = self._current_hotspot_index.get(bot_id, 0)
        hotspot = hotspots[idx % len(hotspots)]
        self._current_hotspot_index[bot_id] = (idx + 1) % len(hotspots)
        return tuple(hotspot)  # type: ignore
    
    def assess(self, signals: dict[str, Any], actions: list[HeuristicAction], bot_id: str) -> None:
        """Navigate to spawn hotspots for efficient farming."""
        current_map = str(signals.get("map", "") or "")
        hotspot = self.get_next_hotspot(current_map, bot_id)
        
        if hotspot:
            x, y = hotspot
            actions.append(HeuristicAction(
                kind="command",
                command=f"move {x} {y}",
                confidence=0.7,
                reason=f"Spawn routing: moving to hotspot ({x},{y}) on {current_map}",
                domain="navigation",
            ))
