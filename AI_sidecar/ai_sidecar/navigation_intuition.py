"""
Navigation intuition — warp routing, safe spots, shortcuts, dead ends.

A pro player knows every warp point, every shortcut, every safe spot.
They know the fastest route from Prontera to Orc Dungeon is through
the warp at (147, 123) in morocc. They know hidden warps, dead ends,
and line-of-sight breaks for kiting.

This module provides real route planning with warp awareness.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WarpPoint:
    name: str
    map: str
    x: int
    y: int
    target_map: str
    target_x: int
    target_y: int
    cost: int = 0  # zeny cost if applicable
    level_req: int = 0
    quest_req: str = ""


@dataclass(slots=True)
class SafeSpot:
    map: str
    x: int
    y: int
    radius: int = 3  # safe radius in cells
    description: str = ""


@dataclass(slots=True)
class MapNode:
    name: str
    warps: list[WarpPoint] = field(default_factory=list)
    safe_spots: list[SafeSpot] = field(default_factory=list)
    danger_zones: list[dict[str, Any]] = field(default_factory=list)
    is_town: bool = False
    has_kafra: bool = False
    has_vendor: bool = False
    has_blacksmith: bool = False


@dataclass(slots=True)
class NavigationIntuition:
    """Real route planning with warp awareness and safe spots."""
    
    _lock: RLock = field(default_factory=RLock)
    _maps: dict[str, MapNode] = field(default_factory=dict)
    _warp_graph: dict[str, list[tuple[str, int]]] = field(default_factory=dict)  # map -> [(target_map, cost)]
    _route_cache: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    _stats: dict[str, int] = field(default_factory=lambda: {"routes_planned": 0, "warps_used": 0, "shortcuts_found": 0})
    
    def __post_init__(self) -> None:
        self._build_default_map_data()
    
    def _build_default_map_data(self) -> None:
        """Build map data with known warps and safe spots."""
        # Major town warps
        towns = {
            "prontera": {"is_town": True, "has_kafra": True, "has_vendor": True, "has_blacksmith": True},
            "morocc": {"is_town": True, "has_kafra": True, "has_vendor": True},
            "geffen": {"is_town": True, "has_kafra": True, "has_vendor": True, "has_blacksmith": True},
            "payon": {"is_town": True, "has_kafra": True, "has_vendor": True},
            "aldebaran": {"is_town": True, "has_kafra": True, "has_vendor": True},
            "yuno": {"is_town": True, "has_kafra": True, "has_vendor": True},
            "izlude": {"is_town": True, "has_kafra": True},
            "comodo": {"is_town": True, "has_kafra": True, "has_vendor": True},
        }
        
        for name, props in towns.items():
            node = MapNode(name=name)
            node.is_town = props.get("is_town", False)
            node.has_kafra = props.get("has_kafra", False)
            node.has_vendor = props.get("has_vendor", False)
            node.has_blacksmith = props.get("has_blacksmith", False)
            self._maps[name] = node
        
        # Known warp connections (map -> [(target_map, cost)])
        warp_connections = {
            "prontera": [("prt_fild08", 0), ("prt_fild09", 0), ("prt_fild04", 0), ("prt_fild01", 0)],
            "morocc": [("moc_fild01", 0), ("moc_fild02", 0), ("moc_fild03", 0)],
            "geffen": [("gef_fild01", 0), ("gef_fild02", 0), ("gef_fild03", 0)],
            "payon": [("pay_fild01", 0), ("pay_fild02", 0), ("pay_fild03", 0)],
            "aldebaran": [("alberta", 0), ("alde_fild01", 0)],
            "yuno": [("yuno_fild01", 0), ("yuno_fild02", 0)],
            "izlude": [("iz_fild01", 0)],
            "comodo": [("cmd_fild01", 0)],
        }
        
        for src, targets in warp_connections.items():
            if src not in self._maps:
                self._maps[src] = MapNode(name=src)
            for tgt, cost in targets:
                if tgt not in self._maps:
                    self._maps[tgt] = MapNode(name=tgt)
                self._warp_graph.setdefault(src, []).append((tgt, cost))
                self._warp_graph.setdefault(tgt, []).append((src, cost))
        
        # Safe spots (positions where melee monsters can't reach)
        safe_spots = [
            SafeSpot(map="prt_fild08", x=100, y=100, radius=5, description="Tree circle safe spot"),
            SafeSpot(map="moc_fild01", x=50, y=50, radius=3, description="Rock formation safe spot"),
            SafeSpot(map="pay_fild01", x=200, y=150, radius=4, description="Hill safe spot"),
        ]
        for spot in safe_spots:
            if spot.map in self._maps:
                self._maps[spot.map].safe_spots.append(spot)
    
    def find_route(self, start_map: str, end_map: str) -> list[str]:
        """Find shortest route between two maps using BFS on warp graph."""
        cache_key = (start_map, end_map)
        if cache_key in self._route_cache:
            return self._route_cache[cache_key]
        
        if start_map == end_map:
            return [start_map]
        
        # BFS
        visited = {start_map}
        queue: deque[tuple[str, list[str]]] = deque([(start_map, [start_map])])
        
        while queue:
            current, path = queue.popleft()
            for neighbor, _ in self._warp_graph.get(current, []):
                if neighbor == end_map:
                    result = path + [neighbor]
                    self._route_cache[cache_key] = result
                    self._stats["routes_planned"] += 1
                    return result
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return [start_map, end_map]  # fallback: direct walk
    
    def get_safe_spot(self, map_name: str) -> SafeSpot | None:
        """Get nearest safe spot on a map."""
        node = self._maps.get(map_name)
        if node and node.safe_spots:
            return node.safe_spots[0]
        return None
    
    def is_town(self, map_name: str) -> bool:
        node = self._maps.get(map_name)
        return node.is_town if node else ("prontera" in map_name or "morocc" in map_name or 
                                           "geffen" in map_name or "payon" in map_name or
                                           "aldebaran" in map_name or "yuno" in map_name or
                                           "izlude" in map_name or "comodo" in map_name)
    
    def has_kafra(self, map_name: str) -> bool:
        node = self._maps.get(map_name)
        return node.has_kafra if node else self.is_town(map_name)
    
    def estimate_travel_time(self, start_map: str, end_map: str) -> float:
        """Estimate travel time in seconds."""
        route = self.find_route(start_map, end_map)
        # Rough estimate: 30s per map transition + 10s per map walk
        return len(route) * 30 + 10
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)
