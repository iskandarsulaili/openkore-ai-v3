"""Consumable Loadout Planner — per-map/activity consumable recommendations.

A real RO player at level 60 carries:
- 20 White Potions
- 10 Awakening Potions (+attack speed)
- 5 Concentrated Potions (+damage)
- 2 Elemental Converters
- 3 Fly Wings
- 1 Butterfly Wing
- 10 Magnifiers
- Antidotes for poison maps
- Traps for Hunter
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

DATA_DIR = Path(__file__).parent.parent.parent / "data"


class ConsumableLoadout:
    """Per-map consumable loadout recommendation.
    
    Different maps need different consumable sets:
    - Poison map (Drainliar): carry Antidotes
    - Undead map (Culverts): carry Holy Water
    - Fire map (Magma Dungeon): carry Fire Armor Scrolls
    """
    
    @staticmethod
    def get_loadout(map_name: str, job: str, level: int) -> list[dict]:
        """Get recommended consumable loadout for a map/job/level."""
        loadout = []
        
        # Base loadout (everyone)
        base_potions = {
            "Red_Potion": min(15, 5 + level // 5),
            "Fly_Wing": max(3, level // 15),
            "Butterfly_Wing": 1,
        }
        
        # Potion quality improves with level
        if level >= 15:
            base_potions["Orange_Potion"] = 10
            del base_potions["Red_Potion"]
        if level >= 30:
            base_potions["White_Potion"] = 10
            del base_potions["Orange_Potion"]
        if level >= 50:
            base_potions["White_Potion"] = 20
        
        for name, qty in base_potions.items():
            loadout.append({"item": name, "quantity": qty, "priority": "essential"})
        
        # Job-specific
        job_lower = job.lower()
        if "hunter" in job_lower or "sniper" in job_lower or "ranger" in job_lower:
            loadout.append({"item": "Trap", "quantity": 5, "priority": "high"})
            loadout.append({"item": "Arrow", "quantity": 500, "priority": "essential"})
        
        if "knight" in job_lower or "paladin" in job_lower or "crusader" in job_lower:
            if level >= 30:
                loadout.append({"item": "Awakening_Potion", "quantity": 5, "priority": "medium"})
        
        # Map-specific
        map_lower = map_name.lower()
        if "drain" in map_lower or "poison" in map_lower:
            loadout.append({"item": "Antidote", "quantity": 5, "priority": "high"})
        
        if "undead" in map_lower or "culvert" in map_lower or "ghost" in map_lower:
            loadout.append({"item": "Holy_Water", "quantity": 5, "priority": "medium"})
        
        if "magma" in map_lower or "fire" in map_lower:
            loadout.append({"item": "Fire_Resist_Potion", "quantity": 3, "priority": "high"})
        
        return loadout


class ConsumableLoadoutPlanner:
    """Plans consumable loadouts per bot per activity."""
    
    def __init__(self):
        self._loadout_cache: dict[str, list] = {}
    
    def assess(self, signals: dict[str, Any], actions: list[HeuristicAction], bot_id: str) -> None:
        current_map = str(signals.get("map", "") or "")
        job = str(signals.get("job", "") or "")
        level = int(signals.get("base_level", 1) or 1)
        inventory = signals.get("inventory", {}) or {}
        zeny = int(signals.get("zeny", 0) or 0)
        
        loadout = ConsumableLoadout.get_loadout(current_map, job, level)
        
        # Check what we're missing
        items_have = set()
        if isinstance(inventory, dict):
            inv_items = inventory.get("items", inventory.get("inventory", []))
            if isinstance(inv_items, list):
                for item in inv_items:
                    if isinstance(item, dict):
                        name = str(item.get("name", item.get("identifiedDisplayName", "")) or "").lower()
                        name = name.replace(" ", "_")
                        items_have.add(name)
        
        missing = [l for l in loadout if l["item"].lower().replace(" ", "_") not in items_have]
        
        if missing and zeny > 500:
            for item in missing:
                actions.append(HeuristicAction(
                    kind="command",
                    command=f"buy {item['item']} {item['quantity']}",
                    confidence=0.7,
                    reason=f"Loadout: missing {item['item']}x{item['quantity']} for {current_map}",
                    domain="economy",
                ))
        
        # Track loadout state
        actions.append(HeuristicAction(
            kind="log",
            command=f"loadout map={current_map} have={len(items_have)} need={len(loadout)} missing={len(missing)}",
            confidence=0.5,
            reason=f"Consumable loadout: {len(missing)} items missing for {current_map}",
            domain="economy",
        ))


class DurabilityMonitor:
    """Tracks equipment durability and recommends repair.
    
    Equipment degrades with use. At 0 durability, deals 50% damage.
    A pro player repairs when durability drops below 80%.
    """
    
    def __init__(self):
        self._repair_threshold = 0.8  # Repair at 80% durability
    
    def assess(self, signals: dict[str, Any], actions: list[HeuristicAction], bot_id: str) -> None:
        equipment = signals.get("equipment", {}) or {}
        in_town = bool(signals.get("in_town", False))
        
        if isinstance(equipment, dict):
            for slot, item in equipment.items():
                if isinstance(item, dict):
                    durability = item.get("durability", item.get("condition", 100))
                    if isinstance(durability, str):
                        try:
                            durability = float(durability.replace("%", ""))
                        except ValueError:
                            durability = 100
                    
                    if isinstance(durability, (int, float)):
                        pct = durability / 100.0 if durability > 1 else durability
                        
                        if pct < self._repair_threshold and in_town:
                            actions.append(HeuristicAction(
                                kind="command",
                                command=f"repair {slot}",
                                confidence=0.9,
                                reason=f"Durability: {slot} at {pct*100:.0f}% — repairing",
                                domain="equipment",
                            ))
        
        # Log overall durability
        min_durability = 100
        if isinstance(equipment, dict):
            for item in equipment.values():
                if isinstance(item, dict):
                    dur = item.get("durability", item.get("condition", 100))
                    if isinstance(dur, str):
                        try:
                            dur = float(dur.replace("%", ""))
                        except ValueError:
                            dur = 100
                    if isinstance(dur, (int, float)):
                        min_durability = min(min_durability, dur / 100.0 if dur > 1 else dur)
        
        if min_durability < self._repair_threshold:
            actions.append(HeuristicAction(
                kind="log",
                command=f"durability min={min_durability*100:.0f}%",
                confidence=0.5,
                reason=f"Low durability: {min_durability*100:.0f}% — find repair NPC",
                domain="equipment",
            ))


class PostMortemAnalyzer:
    """Records death causes and adjusts behavior.
    
    When a bot dies:
    1. Record the map, monster, cause, and build state
    2. Adjust danger thresholds for that map/monster
    3. If same death happens 3+ times, flag map as too dangerous
    4. Recommend countermeasures (antidotes for poison, shield for Back Stab)
    """
    
    def __init__(self):
        self._death_records: dict[str, list[dict]] = {}  # bot_id -> [death events]
        self._adjusted_thresholds: dict[str, dict] = {}  # bot_id -> {map: danger_mult}
    
    def record_death(self, bot_id: str, map_name: str, monster_name: str, 
                     cause: str, hp_at_death: int = 0) -> None:
        """Record a death event for analysis."""
        from datetime import datetime
        if bot_id not in self._death_records:
            self._death_records[bot_id] = []
        
        self._death_records[bot_id].append({
            "map": map_name,
            "monster": monster_name,
            "cause": cause,
            "hp_at_death": hp_at_death,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Adjust threshold if same map/monster death
        self._adjust_threshold(bot_id, map_name, monster_name, cause)
    
    def _adjust_threshold(self, bot_id: str, map_name: str, monster_name: str, cause: str) -> None:
        """Learn from death by adjusting danger thresholds."""
        if bot_id not in self._adjusted_thresholds:
            self._adjusted_thresholds[bot_id] = {}
        
        # Count deaths on this map
        map_deaths = [d for d in self._death_records.get(bot_id, []) if d["map"] == map_name]
        death_count = len(map_deaths)
        
        # Increase danger multiplier for this map
        mult = 1.0 + (death_count * 0.3)  # Each death adds 30% danger
        self._adjusted_thresholds[bot_id][map_name] = {
            "mult": min(mult, 3.0),  # Cap at 3x
            "cause": cause,
            "monster": monster_name,
            "counter": self._suggest_countermeasure(cause),
        }
    
    @staticmethod
    def _suggest_countermeasure(cause: str) -> str:
        """Suggest a countermeasure based on death cause."""
        cause_lower = cause.lower()
        if "back stab" in cause_lower or "ignores def" in cause_lower:
            return "equip_shield"
        if "poison" in cause_lower:
            return "carry_antidote"
        if "stun" in cause_lower:
            return "vit_100_for_immunity"
        if "arrow" in cause_lower or "ranged" in cause_lower:
            return "maintain_distance"
        if "surrounded" in cause_lower or "aoe" in cause_lower:
            return "flee_when_surrounded"
        return "raise_pot_threshold"
    
    def get_map_danger_mult(self, bot_id: str, map_name: str) -> float:
        """Get learned danger multiplier for a map."""
        records = self._adjusted_thresholds.get(bot_id, {})
        return records.get(map_name, {}).get("mult", 1.0)
    
    def assess(self, signals: dict[str, Any], actions: list[HeuristicAction], bot_id: str) -> None:
        """Check if bot just died and analyze."""
        is_dead = bool(signals.get("dead", False) or signals.get("is_dead", False))
        current_map = str(signals.get("map", "") or "")
        
        if is_dead:
            last_monster = signals.get("last_attacked_monster", signals.get("last_monster", "unknown"))
            cause = f"died to {last_monster}"
            self.record_death(bot_id, current_map, str(last_monster), cause)
            
            counter = self._suggest_countermeasure(cause)
            actions.append(HeuristicAction(
                kind="command",
                command=f"countermeasure {counter}",
                confidence=0.8,
                reason=f"Post-mortem: {cause} — recommended: {counter}",
                domain="learning",
            ))
        
        # Log adjusted thresholds
        danger_mult = self.get_map_danger_mult(bot_id, current_map)
        if danger_mult > 1.0:
            actions.append(HeuristicAction(
                kind="log",
                command=f"danger_multiplier {current_map}={danger_mult:.1f}x",
                confidence=0.5,
                reason=f"Learned: {current_map} has {danger_mult:.1f}x danger multiplier",
                domain="learning",
            ))
