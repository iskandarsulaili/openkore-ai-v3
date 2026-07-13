"""
Map knowledge database — geometry, warps, danger zones, spawn data.

Loaded from knowledge.json at startup. The LLM reviews and can add
new map entries based on observed bot behavior (kaizen discovery).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MapEntry:
    name: str
    safe_x: int = 0
    safe_y: int = 0
    is_town: bool = False
    is_pvp: bool = False
    is_dungeon: bool = False
    min_level: int = 1
    max_level: int = 999
    warps: list[dict[str, Any]] = field(default_factory=list)
    danger_zones: list[dict[str, Any]] = field(default_factory=list)
    spawn_density: float = 1.0  # 0.0-2.0 multiplier
    portal_connections: list[str] = field(default_factory=list)
    dead_ends: list[tuple[int, int]] = field(default_factory=list)
    recommended_elements: list[str] = field(default_factory=list)
    notes: str = ""


# Default map data for common maps (expanded from rAthena knowledge)
DEFAULT_MAP_DATA: dict[str, dict[str, Any]] = {
    "prontera": {"safe_x": 156, "safe_y": 191, "is_town": True, "portal_connections": ["prt_fild08", "prt_in"]},
    "prt_in": {"safe_x": 227, "safe_y": 18, "is_town": True, "portal_connections": ["prontera"]},
    "prt_fild08": {"min_level": 1, "max_level": 25, "spawn_density": 1.2, "portal_connections": ["prontera"], "recommended_elements": ["neutral"]},
    "prt_fild01": {"min_level": 1, "max_level": 15, "spawn_density": 0.8, "portal_connections": ["prontera"]},
    "prt_fild04": {"min_level": 8, "max_level": 30, "spawn_density": 1.0, "portal_connections": ["prontera"]},
    "prt_fild11": {"min_level": 15, "max_level": 40, "spawn_density": 1.0, "portal_connections": ["prontera"]},
    "morocc": {"safe_x": 150, "safe_y": 100, "is_town": True, "portal_connections": ["moc_fild01", "moc_fild02", "moc_pryd01"]},
    "moc_fild01": {"min_level": 10, "max_level": 30, "spawn_density": 1.0, "portal_connections": ["morocc"]},
    "moc_fild02": {"min_level": 15, "max_level": 35, "spawn_density": 1.1, "portal_connections": ["morocc"]},
    "moc_pryd01": {"min_level": 20, "max_level": 50, "is_dungeon": True, "spawn_density": 1.3, "portal_connections": ["morocc"]},
    "geffen": {"safe_x": 120, "safe_y": 85, "is_town": True, "portal_connections": ["gef_fild01", "gef_fild02", "gef_fild03"]},
    "gef_fild01": {"min_level": 20, "max_level": 45, "spawn_density": 1.0, "portal_connections": ["geffen"]},
    "gef_fild02": {"min_level": 25, "max_level": 50, "spawn_density": 1.0, "portal_connections": ["geffen"]},
    "gef_fild03": {"min_level": 30, "max_level": 55, "spawn_density": 1.1, "portal_connections": ["geffen"]},
    "payon": {"safe_x": 210, "safe_y": 120, "is_town": True, "portal_connections": ["pay_fild01", "pay_fild02", "pay_dun00"]},
    "pay_fild01": {"min_level": 15, "max_level": 35, "spawn_density": 1.0, "portal_connections": ["payon"]},
    "pay_fild02": {"min_level": 20, "max_level": 40, "spawn_density": 1.0, "portal_connections": ["payon"]},
    "pay_dun00": {"min_level": 30, "max_level": 60, "is_dungeon": True, "spawn_density": 1.4, "portal_connections": ["payon"]},
    "aldebaran": {"safe_x": 140, "safe_y": 130, "is_town": True, "portal_connections": ["alde_fild01", "alde_fild02"]},
    "alde_fild01": {"min_level": 25, "max_level": 50, "spawn_density": 1.0, "portal_connections": ["aldebaran"]},
    "alde_fild02": {"min_level": 30, "max_level": 55, "spawn_density": 1.0, "portal_connections": ["aldebaran"]},
    "yuno": {"safe_x": 180, "safe_y": 150, "is_town": True, "portal_connections": ["yuno_fild01", "yuno_fild02"]},
    "yuno_fild01": {"min_level": 50, "max_level": 80, "spawn_density": 1.0, "portal_connections": ["yuno"]},
    "yuno_fild02": {"min_level": 55, "max_level": 85, "spawn_density": 1.0, "portal_connections": ["yuno"]},
    "einbroch": {"safe_x": 200, "safe_y": 180, "is_town": True},
    "lighthalzen": {"safe_x": 150, "safe_y": 120, "is_town": True},
    "hugel": {"safe_x": 100, "safe_y": 80, "is_town": True},
    "rachel": {"safe_x": 130, "safe_y": 100, "is_town": True},
}


@dataclass(slots=True)
class MapKnowledge:
    """Map geometry and danger zone database."""
    
    knowledge_path: Path | None = None
    _lock: RLock = field(default_factory=RLock)
    _maps: dict[str, MapEntry] = field(default_factory=dict)
    _llm_discovered: list[dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        self._load()
    
    def _load(self) -> None:
        """Load map data from knowledge.json, fall back to defaults."""
        # Start with defaults
        for name, data in DEFAULT_MAP_DATA.items():
            self._maps[name] = MapEntry(name=name, **data)
        
        # Override with knowledge.json if available
        if self.knowledge_path is not None and self.knowledge_path.exists():
            try:
                data = json.loads(self.knowledge_path.read_text(encoding="utf-8"))
                map_data = data.get("map_data", data.get("maps", {}))
                if isinstance(map_data, dict):
                    for name, entry in map_data.items():
                        if name in self._maps:
                            for key, value in entry.items():
                                if hasattr(self._maps[name], key):
                                    setattr(self._maps[name], key, value)
                        else:
                            self._maps[name] = MapEntry(name=name, **entry)
                logger.info("map_knowledge_loaded: %d maps", len(self._maps))
            except Exception as e:
                logger.debug("map_knowledge_load_skipped: %s", e)
    
    def get_map(self, name: str) -> MapEntry | None:
        with self._lock:
            return self._maps.get(name.lower())
    
    def is_safe(self, map_name: str, x: int, y: int) -> bool:
        """Check if coordinates are in a safe zone (not a dead end or danger zone)."""
        entry = self.get_map(map_name)
        if entry is None:
            return True  # Unknown map — assume safe
        for dz in entry.danger_zones:
            dx, dy = dz.get("x", 0), dz.get("y", 0)
            radius = dz.get("radius", 5)
            if abs(x - dx) <= radius and abs(y - dy) <= radius:
                return False
        return True
    
    def get_recommended_maps(self, level: int) -> list[dict[str, Any]]:
        """Get maps suitable for a given level range."""
        results = []
        for entry in self._maps.values():
            if entry.min_level <= level <= entry.max_level and not entry.is_town:
                results.append({
                    "name": entry.name,
                    "min_level": entry.min_level,
                    "max_level": entry.max_level,
                    "spawn_density": entry.spawn_density,
                    "is_dungeon": entry.is_dungeon,
                    "connections": entry.portal_connections,
                })
        results.sort(key=lambda m: abs(m["min_level"] - level))
        return results[:10]
    
    def discover_map(self, map_name: str, observed_data: dict[str, Any]) -> None:
        """LLM-discovered map data from bot experience."""
        with self._lock:
            if map_name not in self._maps:
                self._maps[map_name] = MapEntry(name=map_name)
            entry = self._maps[map_name]
            for key, value in observed_data.items():
                if hasattr(entry, key):
                    setattr(entry, key, value)
            self._llm_discovered.append({
                "map": map_name,
                "data": observed_data,
                "timestamp": __import__("time").time(),
            })
    
    def get_discoveries(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._llm_discovered[-50:])
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return {"maps": len(self._maps), "discoveries": len(self._llm_discovered)}
