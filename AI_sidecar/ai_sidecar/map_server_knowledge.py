"""
Map Server Knowledge — tracks which maps belong to which map server.

In Ragnarok Online, maps are grouped into map servers. When a bot crosses
a map-server boundary, it may experience:
  - Brief disconnect/reconnect
  - Different spawn rules
  - Different warp portal availability

This module tracks map-server assignments so the pathfinder can:
1. Anticipate disconnects when crossing boundaries
2. Use warp portals (NPC teleportation) within the same map server
3. Prefer staying on the same map server when possible
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MapServerInfo:
    """Information about a map server."""
    name: str
    maps: set[str] = field(default_factory=set)
    has_warp_portal: bool = False
    warp_portal_npc: str = ""
    warp_portal_location: tuple[int, int] = (0, 0)
    warp_destinations: dict[str, tuple[str, int, int, int]] = field(default_factory=dict)
    # destination_map -> (npc_name, x, y, zeny_cost)


# Known map-server assignments (based on iRO/rAthena classic layout)
# Maps are grouped by the map-server they run on.
# Warp portals (NPC teleporters) are listed per server.

MAP_SERVER_ASSIGNMENTS: dict[str, dict[str, Any]] = {
    "prontera_server": {
        "maps": {
            "prontera", "prt_in", "prt_fild01", "prt_fild02", "prt_fild03",
            "prt_fild04", "prt_fild05", "prt_fild06", "prt_fild07", "prt_fild08",
            "prt_fild09", "prt_fild10", "prt_fild11",
            "prt_maze01", "prt_maze02", "prt_maze03",
            "mjolnir_01", "mjolnir_02", "mjolnir_03", "mjolnir_04",
            "mjolnir_05", "mjolnir_06", "mjolnir_07", "mjolnir_08",
            "mjolnir_09", "mjolnir_10", "mjolnir_11", "mjolnir_12",
        },
        "has_warp_portal": True,
        "warp_portal_npc": "Warp Portal",
        "warp_portal_location": (147, 123),
        "warp_destinations": {
            "geffen": ("Warp Portal", 118, 57, 0),
            "morocc": ("Warp Portal", 156, 90, 0),
            "alberta": ("Warp Portal", 46, 119, 0),
            "payon": ("Warp Portal", 187, 190, 0),
        },
    },
    "geffen_server": {
        "maps": {
            "geffen", "gef_fild00", "gef_fild01", "gef_fild02", "gef_fild03",
            "gef_fild04", "gef_fild05", "gef_fild06", "gef_fild07", "gef_fild08",
            "gef_fild09", "gef_fild10", "gef_fild11", "gef_fild12", "gef_fild13",
            "gef_fild14",
            "gef_dun00", "gef_dun01", "gef_dun02", "gef_dun03",
            "orcsdun01", "orcsdun02",
            "tur_dun01", "tur_dun02", "tur_dun03", "tur_dun04",
        },
        "has_warp_portal": True,
        "warp_portal_npc": "Warp Portal",
        "warp_portal_location": (120, 50),
        "warp_destinations": {
            "prontera": ("Warp Portal", 161, 180, 0),
        },
    },
    "morocc_server": {
        "maps": {
            "morocc", "moc_fild01", "moc_fild02", "moc_fild03",
            "moc_fild04", "moc_fild05", "moc_fild06", "moc_fild07",
            "moc_fild08", "moc_fild09", "moc_fild10", "moc_fild11",
            "moc_fild12", "moc_fild13", "moc_fild14", "moc_fild15",
            "moc_fild16", "moc_fild17", "moc_fild18", "moc_fild19",
            "moc_fild20", "moc_fild21", "moc_fild22",
            "moc_dun01", "moc_dun02", "moc_dun03", "moc_dun04",
            "moc_pryd01", "moc_pryd02", "moc_pryd03", "moc_pryd04",
            "moc_pryd05", "moc_pryd06",
        },
        "has_warp_portal": True,
        "warp_portal_npc": "Warp Portal",
        "warp_portal_location": (155, 88),
        "warp_destinations": {
            "prontera": ("Warp Portal", 294, 216, 0),
        },
    },
    "payon_server": {
        "maps": {
            "payon", "pay_fild01", "pay_fild02", "pay_fild03", "pay_fild04",
            "pay_fild05", "pay_fild06", "pay_fild07", "pay_fild08", "pay_fild09",
            "pay_fild10", "pay_fild11",
            "pay_dun00", "pay_dun01", "pay_dun02", "pay_dun03", "pay_dun04",
        },
        "has_warp_portal": True,
        "warp_portal_npc": "Warp Portal",
        "warp_portal_location": (185, 190),
        "warp_destinations": {
            "prontera": ("Warp Portal", 278, 325, 0),
        },
    },
    "aldebaran_server": {
        "maps": {
            "aldebaran", "alberta", "alde_fild01", "alde_fild02", "alde_fild03",
            "alde_fild04", "alde_fild05", "alde_fild06", "alde_fild07", "alde_fild08",
            "alde_fild09", "alde_fild10",
            "ama_fild01", "ama_dun01",
            "cmd_fild01", "cmd_fild02", "cmd_fild03", "cmd_fild04", "cmd_fild05",
            "cmd_fild06", "cmd_fild07", "cmd_fild08", "cmd_fild09",
            "comodo", "comodo_fild01",
        },
        "has_warp_portal": True,
        "warp_portal_npc": "Warp Portal",
        "warp_portal_location": (44, 119),
        "warp_destinations": {
            "prontera": ("Warp Portal", 34, 329, 0),
        },
    },
    "yuno_server": {
        "maps": {
            "yuno", "yuno_fild01", "yuno_fild02", "yuno_fild03", "yuno_fild04",
            "yuno_fild05", "yuno_fild06", "yuno_fild07", "yuno_fild08",
            "einbroch", "ein_fild01", "ein_fild02", "ein_fild03", "ein_fild04",
            "ein_fild05", "ein_fild06", "ein_fild07", "ein_fild08", "ein_fild09",
            "ein_fild10",
            "lighthalzen", "lhz_fild01", "lhz_fild02", "lhz_fild03",
            "lhz_dun01", "lhz_dun02", "lhz_dun03",
        },
        "has_warp_portal": False,
    },
    "izlude_server": {
        "maps": {
            "izlude", "iz_fild01", "iz_fild02", "iz_fild03", "iz_fild04",
            "iz_dun00", "iz_dun01", "iz_dun02", "iz_dun03", "iz_dun04",
        },
        "has_warp_portal": False,
    },
    "xmas_server": {
        "maps": {
            "xmas", "xmas_fild01", "xmas_dun01",
            "ice_dun01", "ice_dun02", "ice_dun03",
        },
        "has_warp_portal": False,
    },
    "hugel_server": {
        "maps": {
            "hugel", "hu_fild01", "hu_fild02", "hu_fild03", "hu_fild04",
            "hu_fild05", "hu_fild06", "hu_fild07",
            "abyss_01", "abyss_02", "abyss_03",
        },
        "has_warp_portal": False,
    },
    "rachel_server": {
        "maps": {
            "rachel", "ra_fild01", "ra_fild02", "ra_fild03", "ra_fild04",
            "ra_fild05", "ra_fild06", "ra_fild07", "ra_fild08", "ra_fild09",
            "ra_fild10", "ra_fild11", "ra_fild12",
            "ra_san01", "ra_san02", "ra_san03", "ra_san04",
            "ra_temple", "ra_temple2",
        },
        "has_warp_portal": False,
    },
    "veins_server": {
        "maps": {
            "veins", "ve_fild01", "ve_fild02", "ve_fild03", "ve_fild04",
            "ve_fild05", "ve_fild06", "ve_fild07",
            "thor_v01", "thor_v02", "thor_v03",
        },
        "has_warp_portal": False,
    },
}


class MapServerKnowledge:
    """Knows which maps belong to which map server.

    Thread-safe singleton. Provides map-server lookups, boundary detection,
    and warp portal information.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._servers: dict[str, MapServerInfo] = {}
        self._map_to_server: dict[str, str] = {}  # map_name -> server_name
        self._load_servers()

    def _load_servers(self) -> None:
        """Load map-server assignments from the known data."""
        for server_name, data in MAP_SERVER_ASSIGNMENTS.items():
            info = MapServerInfo(
                name=server_name,
                maps=set(data.get("maps", [])),
                has_warp_portal=data.get("has_warp_portal", False),
                warp_portal_npc=data.get("warp_portal_npc", ""),
                warp_portal_location=tuple(data.get("warp_portal_location", (0, 0))),
                warp_destinations=dict(data.get("warp_destinations", {})),
            )
            self._servers[server_name] = info
            for map_name in info.maps:
                self._map_to_server[map_name] = server_name

    # ── Public API ───────────────────────────────────────────────────

    def get_server_for_map(self, map_name: str) -> str | None:
        """Get the map server name for a given map."""
        with self._lock:
            return self._map_to_server.get(map_name)

    def get_server_info(self, server_name: str) -> MapServerInfo | None:
        """Get full info about a map server."""
        with self._lock:
            return self._servers.get(server_name)

    def get_server_info_for_map(self, map_name: str) -> MapServerInfo | None:
        """Get map server info for a given map."""
        server_name = self.get_server_for_map(map_name)
        if server_name is None:
            return None
        return self.get_server_info(server_name)

    def is_same_server(self, map_a: str, map_b: str) -> bool:
        """Check if two maps are on the same map server."""
        server_a = self.get_server_for_map(map_a)
        server_b = self.get_server_for_map(map_b)
        if server_a is None or server_b is None:
            return False
        return server_a == server_b

    def crossing_boundary(self, current_map: str, next_map: str) -> bool:
        """Check if moving from current_map to next_map crosses a map-server boundary."""
        return not self.is_same_server(current_map, next_map)

    def get_boundary_crossings(self, path_maps: list[str]) -> list[tuple[str, str, int]]:
        """Find all map-server boundary crossings in a path.

        Returns list of (from_map, to_map, index_in_path) for each crossing.
        """
        crossings: list[tuple[str, str, int]] = []
        for i in range(len(path_maps) - 1):
            if self.crossing_boundary(path_maps[i], path_maps[i + 1]):
                crossings.append((path_maps[i], path_maps[i + 1], i))
        return crossings

    def has_warp_portal(self, map_name: str) -> bool:
        """Check if a map has a warp portal (NPC teleporter)."""
        info = self.get_server_info_for_map(map_name)
        if info is None:
            return False
        return info.has_warp_portal

    def get_warp_portal_location(self, map_name: str) -> tuple[int, int] | None:
        """Get the location of the warp portal NPC on a map."""
        info = self.get_server_info_for_map(map_name)
        if info is None or not info.has_warp_portal:
            return None
        return info.warp_portal_location

    def get_warp_destinations(self, map_name: str) -> dict[str, tuple[str, int, int, int]]:
        """Get available warp destinations from a map's warp portal.

        Returns dict of destination_map -> (npc_name, x, y, zeny_cost).
        """
        info = self.get_server_info_for_map(map_name)
        if info is None or not info.has_warp_portal:
            return {}
        return dict(info.warp_destinations)

    def can_warp_to(self, from_map: str, to_map: str) -> bool:
        """Check if a warp portal can take us from from_map to to_map."""
        destinations = self.get_warp_destinations(from_map)
        return to_map in destinations

    def get_warp_cost(self, from_map: str, to_map: str) -> int:
        """Get the zeny cost to warp from one map to another."""
        destinations = self.get_warp_destinations(from_map)
        if to_map in destinations:
            return destinations[to_map][3]
        return 0

    def get_all_maps_on_server(self, map_name: str) -> set[str]:
        """Get all maps on the same server as the given map."""
        server_name = self.get_server_for_map(map_name)
        if server_name is None:
            return {map_name}
        info = self.get_server_info(server_name)
        if info is None:
            return {map_name}
        return set(info.maps)

    def get_server_names(self) -> list[str]:
        """Get all known map server names."""
        with self._lock:
            return list(self._servers.keys())

    def get_map_count(self) -> int:
        """Get total number of known maps across all servers."""
        with self._lock:
            return len(self._map_to_server)

    def add_map_to_server(self, map_name: str, server_name: str) -> None:
        """Add a map to a server at runtime (dynamic discovery)."""
        with self._lock:
            if server_name not in self._servers:
                self._servers[server_name] = MapServerInfo(name=server_name)
            self._servers[server_name].maps.add(map_name)
            self._map_to_server[map_name] = server_name

    def add_warp_destination(self, from_map: str, to_map: str,
                              npc_name: str = "Warp Portal",
                              x: int = 0, y: int = 0, cost: int = 0) -> None:
        """Add a warp destination at runtime."""
        server_name = self.get_server_for_map(from_map)
        if server_name is None:
            return
        with self._lock:
            info = self._servers.get(server_name)
            if info is None:
                return
            info.has_warp_portal = True
            info.warp_portal_npc = npc_name
            info.warp_destinations[to_map] = (npc_name, x, y, cost)


# ── Global Singleton ──

_map_server_knowledge: MapServerKnowledge | None = None
_map_server_knowledge_lock = RLock()


def get_map_server_knowledge() -> MapServerKnowledge:
    global _map_server_knowledge
    with _map_server_knowledge_lock:
        if _map_server_knowledge is None:
            _map_server_knowledge = MapServerKnowledge()
        return _map_server_knowledge
