"""
Knowledge graph — connects all rAthena data into a queryable graph structure.

Maps relationships between:
- Maps → monsters (which monsters spawn where)
- Monsters → items (which monsters drop what)
- Items → classes (who can equip what)
- Classes → skills (skill trees)
- Elements → elements (damage multipliers)
- Maps → maps (portal connections)

Enables graph-based queries like:
"Find the safest farming spot for a level 30 Mage with fire skills"
"What's the best weapon upgrade path for a Swordman at level 40?"
"Which map has the highest density of water-element monsters?"
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class KnowledgeGraph:
    """Graph database connecting all rAthena knowledge."""
    
    knowledge_path: Path | None = None
    _lock: RLock = field(default_factory=RLock)
    
    # Graph nodes
    _maps: dict[str, dict[str, Any]] = field(default_factory=dict)
    _monsters: dict[str, dict[str, Any]] = field(default_factory=dict)
    _items: dict[str, dict[str, Any]] = field(default_factory=dict)
    _classes: dict[str, dict[str, Any]] = field(default_factory=dict)
    _skills: dict[str, dict[str, Any]] = field(default_factory=dict)
    
    # Graph edges
    _map_monsters: dict[str, list[str]] = field(default_factory=dict)  # map -> [monster_names]
    _monster_drops: dict[str, list[str]] = field(default_factory=dict)  # monster -> [item_names]
    _class_skills: dict[str, list[str]] = field(default_factory=dict)  # class -> [skill_names]
    _item_classes: dict[str, list[str]] = field(default_factory=dict)  # item -> [class_names]
    _map_connections: dict[str, list[str]] = field(default_factory=dict)  # map -> [connected_maps]
    _element_chart: dict[str, dict[str, float]] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        if self.knowledge_path and self.knowledge_path.exists():
            self._load()
    
    def _load(self) -> None:
        """Load knowledge.json and build the graph."""
        try:
            data = json.loads(self.knowledge_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("knowledge_graph_load_failed: %s", e)
            return
        
        # Build monster index
        for mob in data.get("mobs", []):
            if isinstance(mob, dict):
                name = str(mob.get("AegisName", mob.get("Name", ""))).lower()
                if name:
                    self._monsters[name] = {
                        "name": name,
                        "level": mob.get("Level", 1),
                        "hp": mob.get("Hp", 0),
                        "element": str(mob.get("Element", "Neutral")).lower(),
                        "element_level": mob.get("ElementLevel", 1),
                        "race": str(mob.get("Race", "Formless")).lower(),
                        "size": str(mob.get("Size", "Medium")).lower(),
                        "defense": mob.get("Defense", 0),
                        "mdef": mob.get("MagicDefense", 0),
                        "base_exp": mob.get("BaseExp", 0),
                        "job_exp": mob.get("JobExp", 0),
                        "attack": mob.get("Attack", 0),
                        "attack2": mob.get("Attack2", 0),
                    }
                    # Track drops
                    drops = mob.get("Drops", [])
                    if isinstance(drops, list):
                        self._monster_drops[name] = [str(d.get("Item", "")).lower() for d in drops if isinstance(d, dict)]
        
        # Build item index
        for category in ["weapons", "armors", "cards", "usable", "etc"]:
            for item in data.get("items", {}).get(category, []):
                if isinstance(item, dict):
                    aegis = str(item.get("AegisName", "")).lower()
                    name = str(item.get("Name", "")).lower()
                    key = aegis or name
                    if key:
                        self._items[key] = {
                            "name": name or key,
                            "aegis_name": aegis,
                            "type": str(item.get("Type", "")),
                            "subtype": str(item.get("SubType", "")),
                            "buy": item.get("Buy", 0),
                            "sell": item.get("Sell", 0),
                            "weight": item.get("Weight", 0),
                            "atk": item.get("Attack", 0),
                            "matk": item.get("Matk", 0),
                            "defense": item.get("Defense", 0),
                            "slots": item.get("Slots", 0),
                            "equip_level_min": item.get("EquipLevelMin", 0),
                            "weapon_level": item.get("WeaponLevel", 0),
                        }
                        # Track equipable classes
                        jobs = item.get("Jobs", {})
                        if isinstance(jobs, dict):
                            self._item_classes[key] = [str(j).lower() for j, v in jobs.items() if v]
        
        # Build element chart from attr_fix data (rAthena format: Level + element->element->damage%)
        elements_raw = data.get("elements", {})
        if isinstance(elements_raw, dict):
            # Try re/pre-re keys first
            for mode_key in ["re", "pre-re"]:
                mode_data = elements_raw.get(mode_key, {})
                if isinstance(mode_data, dict):
                    body = mode_data.get("Body", [])
                    if isinstance(body, list):
                        for entry in body:
                            if isinstance(entry, dict):
                                level = entry.get("Level", 1)
                                if level != 1:
                                    continue
                                for atk_ele, def_map in entry.items():
                                    if atk_ele == "Level":
                                        continue
                                    if isinstance(def_map, dict):
                                        for def_ele, dmg_pct in def_map.items():
                                            dmg = float(dmg_pct) / 100.0
                                            atk_key = str(atk_ele).lower()
                                            def_key = str(def_ele).lower()
                                            if atk_key not in self._element_chart:
                                                self._element_chart[atk_key] = {}
                                            self._element_chart[atk_key][def_key] = dmg
            # Try Body array directly (old format)
            body = elements_raw.get("Body", [])
            if isinstance(body, list) and not self._element_chart:
                for entry in body:
                    if isinstance(entry, dict):
                        level = entry.get("Level", 1)
                        if level != 1:
                            continue
                        for atk_ele, def_map in entry.items():
                            if atk_ele == "Level":
                                continue
                            if isinstance(def_map, dict):
                                for def_ele, dmg_pct in def_map.items():
                                    dmg = float(dmg_pct) / 100.0
                                    atk_key = str(atk_ele).lower()
                                    def_key = str(def_ele).lower()
                                    if atk_key not in self._element_chart:
                                        self._element_chart[atk_key] = {}
                                    self._element_chart[atk_key][def_key] = dmg
        
        # Build class-skill mapping from skill trees (rAthena format: Job + Tree)
        for entry in data.get("skill_trees", []):
            if isinstance(entry, dict):
                class_name = str(entry.get("Job", entry.get("Class", ""))).lower()
                tree = entry.get("Tree", [])
                if isinstance(tree, list):
                    for skill_entry in tree:
                        if isinstance(skill_entry, dict):
                            skill = str(skill_entry.get("Name", "")).lower()
                            if class_name and skill:
                                if class_name not in self._class_skills:
                                    self._class_skills[class_name] = []
                                if skill not in self._class_skills[class_name]:
                                    self._class_skills[class_name].append(skill)
        
        # Build map connections from map_knowledge defaults
        from ai_sidecar.map_knowledge import DEFAULT_MAP_DATA
        for map_name, map_data in DEFAULT_MAP_DATA.items():
            connections = map_data.get("portal_connections", [])
            if connections:
                self._map_connections[map_name] = list(connections)
                for conn in connections:
                    if conn not in self._map_connections:
                        self._map_connections[conn] = []
                    if map_name not in self._map_connections[conn]:
                        self._map_connections[conn].append(map_name)
        
        logger.info("knowledge_graph_loaded: %d monsters, %d items, %d classes, %d skills",
                    len(self._monsters), len(self._items), len(self._class_skills),
                    sum(len(v) for v in self._class_skills.values()))
    
    def find_farming_spot(self, level: int, element: str = "neutral", max_risk: float = 0.5) -> list[dict[str, Any]]:
        """Find the best farming spots for a given level and element preference."""
        results = []
        for mname, mob in self._monsters.items():
            mob_level = int(mob.get("level", 1))
            mob_element = str(mob.get("element", "neutral"))
            
            # Check level range (within 15 levels)
            if abs(mob_level - level) > 15:
                continue
            
            # Check element advantage
            element_mult = 1.0
            if element in self._element_chart and mob_element in self._element_chart[element]:
                element_mult = self._element_chart[element][mob_element]
            
            # Calculate score
            exp_score = (int(mob.get("base_exp", 0)) + int(mob.get("job_exp", 0))) / max(mob_level, 1)
            risk_score = int(mob.get("attack", 0)) / max(int(mob.get("hp", 1)), 1)
            
            if risk_score <= max_risk:
                results.append({
                    "monster": mname,
                    "level": mob_level,
                    "element": mob_element,
                    "exp_score": round(exp_score, 1),
                    "risk_score": round(risk_score, 3),
                    "element_advantage": round(element_mult, 2),
                    "drops": self._monster_drops.get(mname, [])[:5],
                })
        
        results.sort(key=lambda r: r["exp_score"], reverse=True)
        return results[:20]
    
    def find_gear_upgrade(self, player_class: str, level: int, zeny: int, slot: str = "weapon") -> list[dict[str, Any]]:
        """Find the best gear upgrade for a class at a given level."""
        results = []
        for iname, item in self._items.items():
            equip_min = int(item.get("equip_level_min", 0))
            item_type = str(item.get("type", ""))
            price = int(item.get("buy", 0))
            
            # Check if equippable
            if equip_min > level or equip_min == 0:
                continue
            if price > zeny:
                continue
            
            # Check slot
            if slot == "weapon" and item_type == "Weapon":
                pass
            elif slot == "armor" and item_type == "Armor":
                pass
            else:
                continue
            
            # Check class
            classes = self._item_classes.get(iname, [])
            if classes and player_class.lower() not in classes:
                continue
            
            atk = int(item.get("atk", 0))
            matk = int(item.get("matk", 0))
            defense = int(item.get("defense", 0))
            
            results.append({
                "name": item.get("name", iname),
                "aegis_name": iname,
                "level_req": equip_min,
                "price": price,
                "atk": atk,
                "matk": matk,
                "defense": defense,
                "slots": item.get("slots", 0),
                "score": (atk + matk + defense) / max(price, 1) * 1000,
            })
        
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:10]
    
    def find_route(self, from_map: str, to_map: str) -> list[str]:
        """Find the shortest path between two maps using BFS."""
        if from_map == to_map:
            return [from_map]
        
        visited = {from_map}
        queue = [[from_map]]
        
        while queue:
            path = queue.pop(0)
            current = path[-1]
            
            for neighbor in self._map_connections.get(current, []):
                if neighbor == to_map:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        
        return [from_map]  # No path found
    
    def get_element_advantage(self, attack_element: str, defense_element: str) -> float:
        """Get elemental damage multiplier."""
        return self._element_chart.get(attack_element, {}).get(defense_element, 1.0)
    
    def counters(self) -> dict[str, int]:
        return {
            "monsters": len(self._monsters),
            "items": len(self._items),
            "classes": len(self._class_skills),
            "skills": sum(len(v) for v in self._class_skills.values()),
            "maps": len(self._map_connections),
        }
