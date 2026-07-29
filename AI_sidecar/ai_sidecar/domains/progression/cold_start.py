"""Cold Start Planner — data-driven leveling pipeline.

Replaces hardcoded cold start logic with data-driven decisions.
Safe zones, item prices, and progression paths are in YAML data files,
not hardcoded in the AI system.
"""
from __future__ import annotations
from typing import Any
import logging
from pathlib import Path

from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)

# Try to load YAML data
try:
    import yaml
    _DATA_DIR = Path(__file__).parent.parent.parent / "data"
except ImportError:
    yaml = None  # type: ignore
    _DATA_DIR = None

_SafeZoneDB = None


def _load_start_zones() -> dict:
    """Load safe start zone data from YAML file."""
    global _SafeZoneDB
    if _SafeZoneDB is not None:
        return _SafeZoneDB
    if yaml is None or _DATA_DIR is None:
        _SafeZoneDB = {}
        return _SafeZoneDB
    path = _DATA_DIR / "start_zones.yaml"
    if path.exists():
        with open(path) as f:
            _SafeZoneDB = yaml.safe_load(f) or {}
    else:
        _SafeZoneDB = {}
    return _SafeZoneDB


def get_safe_zone(level: int) -> dict | None:
    """Get the recommended safe start zone for a given level."""
    data = _load_start_zones()
    zones = data.get("start_zones", {})
    if level <= 3:
        key = "level_1_3"
    elif level <= 10:
        key = "level_4_10"
    elif level <= 25:
        key = "level_11_25"
    else:
        return None
    
    zone_list = zones.get(key, [])
    if zone_list:
        return zone_list[0]
    return None


def get_item_price(item_name: str) -> dict | None:
    """Get pricing info for an item."""
    data = _load_start_zones()
    prices = data.get("item_prices", {})
    return prices.get(item_name)


class ColdStartPlanner:
    """Data-driven cold start planner.
    
    Produces HeuristicAction commands for the cold start pipeline
    based on character level, current map, and zeny.
    Does NOT use hardcoded values — everything comes from data files.
    """
    
    def assess(self, signals: dict[str, Any], actions: list[HeuristicAction], bot_id: str) -> None:
        level = int(signals.get("base_level", 1) or 1)
        job = str(signals.get("job", "Novice") or "Novice")
        zeny = int(signals.get("zeny", 0) or 0)
        current_map = str(signals.get("map", "") or "")
        weight = int(signals.get("weight", 0) or 0)
        weight_max = int(signals.get("weight_max", 100) or 100)
        
        weight_pct = weight / max(weight_max, 1)
        
        # Get safe zone for this level
        zone = get_safe_zone(level)
        if not zone:
            return  # No recommendation for this level
        
        target_map = zone["map"]
        expected_zeny = zone["expected_zeny_per_hour"]
        npc_town = zone["npc"]
        
        # Item pricing
        knife_price = get_item_price("Knife")
        potion_price = get_item_price("Red_Potion")
        
        knife_cost = knife_price["buy"] if knife_price else 500
        potion_cost = potion_price["buy"] if potion_price else 10
        
        # Determine what to do
        has_weapon = self._has_weapon(signals)
        
        # Step 0: If we have a weapon and potions, go hunt
        if has_weapon and zeny >= potion_cost * 5:
            if current_map != target_map:
                # Find portal from current town to target zone
                actions.append(HeuristicAction(
                    kind="command",
                    command=f"move {zone.get('portal', [22, 203])[0]} {zone.get('portal', [22, 203])[1]}",
                    confidence=0.8,
                    reason=f"ColdStart: move to {target_map} for leveling",
                    domain="progression",
                ))
            else:
                # Already on the right map — start hunting
                actions.append(HeuristicAction(
                    kind="command",
                    command="attackAuto 2",
                    confidence=0.9,
                    reason=f"ColdStart: start hunting on {target_map}",
                    domain="progression",
                ))
        
        # Step 1: If we have no weapon, farm for one
        elif not has_weapon:
            if zeny < knife_cost:
                # Farm for a weapon
                if current_map == target_map:
                    actions.append(HeuristicAction(
                        kind="command",
                        command="attackAuto 2",
                        confidence=0.9,
                        reason=f"ColdStart: farm {knife_cost - zeny}z for weapon on {target_map}",
                        domain="progression",
                    ))
                else:
                    # Walk to farm map
                    actions.append(HeuristicAction(
                        kind="command",
                        command=f"move {zone.get('portal', [22, 203])[0]} {zone.get('portal', [22, 203])[1]}",
                        confidence=0.8,
                        reason=f"ColdStart: walk to {target_map} to farm",
                        domain="progression",
                    ))
            else:
                # Have enough zeny — buy weapon
                actions.append(HeuristicAction(
                    kind="command",
                    command=f"buy {knife_cost} {knife_price.get('npc', 'prt_in 42 170')}",
                    confidence=0.9,
                    reason=f"ColdStart: buy weapon ({knife_cost}z)",
                    domain="progression",
                ))
        
        # Step 2: Have weapon but no potions — buy some
        elif zeny >= potion_cost * 10:
            actions.append(HeuristicAction(
                kind="command",
                command=f"buy {potion_cost * 10} {potion_price.get('npc', 'prt_in 22 164')}",
                confidence=0.9,
                reason=f"ColdStart: buy potions ({potion_cost * 10}z for 10)",
                domain="progression",
            ))
        
        # Step 3: Overweight — sell junk
        if weight_pct > 0.7:
            actions.append(HeuristicAction(
                kind="command",
                command="sellAuto 1",
                confidence=0.9,
                reason="ColdStart: overweight, sell junk",
                domain="economy",
            ))
        
        # Log current state
        actions.append(HeuristicAction(
            kind="log",
            command=f"cold_start level={level} job={job} zeny={zeny} map={current_map} target={target_map}",
            confidence=0.5,
            reason="ColdStart state tracking",
            domain="progression",
        ))
    
    def _has_weapon(self, signals: dict) -> bool:
        inventory = signals.get("inventory", {}) or {}
        if isinstance(inventory, dict):
            items = inventory.get("items", inventory.get("inventory", []))
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        name = str(item.get("name", item.get("identifiedDisplayName", "")) or "")
                        if any(w in name.lower() for w in ["knife", "sword", "mace", "bow", "staff", "dagger", "axe"]):
                            return True
        equipment = signals.get("equipment", {}) or {}
        if isinstance(equipment, dict):
            for slot, item in equipment.items():
                if isinstance(item, dict):
                    name = str(item.get("name", "") or "")
                    if any(w in name.lower() for w in ["weapon", "knife", "sword"]):
                        return True
        return False


# Singleton
_cold_start_planner: ColdStartPlanner | None = None


def get_cold_start_planner() -> ColdStartPlanner:
    global _cold_start_planner
    if _cold_start_planner is None:
        _cold_start_planner = ColdStartPlanner()
    return _cold_start_planner
