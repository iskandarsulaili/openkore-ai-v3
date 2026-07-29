"""
PortalDB — database of portal connections between RO maps.

Each portal connection is a bidirectional edge with precise coordinates
for both the source warp point and the destination arrival point.
Data based on real rAthena / iRO portal coordinates.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class PortalConnection:
    """A bidirectional portal between two maps with precise coordinates."""
    map1: str
    x1: int
    y1: int
    map2: str
    x2: int
    y2: int

    def other_side(self, map_name: str) -> tuple[str, int, int] | None:
        """Get the destination if traveling from `map_name` through this portal."""
        if map_name == self.map1:
            return (self.map2, self.x2, self.y2)
        if map_name == self.map2:
            return (self.map1, self.x1, self.y1)
        return None


# ── Complete Portal Database ──────────────────────────────────────────────
# Format: (map_a, x_a, y_a, map_b, x_b, y_b)
# Coordinates based on rAthena/iRO warp data (approximate where exact unknown)

_PORTAL_DATA: list[tuple[str, int, int, str, int, int]] = [
    # ═════════════════════════════════════════════════════════════════════
    # PRONTERA — Hub of Midgard
    # ═════════════════════════════════════════════════════════════════════

    # Prontera -> prt_fild01 (NW gate — Porings, Lunatics)
    ("prontera", 22, 203, "prt_fild01", 84, 264),
    # Prontera -> prt_fild02 (N gate — Fabre, Poring)
    ("prontera", 187, 327, "prt_fild02", 196, 25),
    # Prontera -> prt_fild03 (NE gate — Fabre, Lunatic)
    ("prontera", 303, 80, "prt_fild03", 305, 379),
    # Prontera -> prt_fild04 (SE gate — Poring, Fabre)
    ("prontera", 52, 58, "prt_fild04", 58, 379),
    # Prontera -> prt_fild05 (SW gate — Porings — CRITICAL farm map)
    ("prontera", 26, 203, "prt_fild05", 373, 205),
    # Prontera interior -> prt_in (indoor shops/guilds)
    ("prontera", 156, 165, "prt_in", 76, 18),
    # Prontera -> prt_fild08 (W gate —通向 mjolnir path)
    ("prontera", 304, 283, "prt_fild08", 260, 26),
    # Prontera -> prt_fild11 (far fields)
    ("prontera", 29, 44, "prt_fild11", 256, 370),

    # ═════════════════════════════════════════════════════════════════════
    # PRT_FILD (Prontera Fields) — interconnected
    # ═════════════════════════════════════════════════════════════════════

    ("prt_fild01", 154, 49, "prt_fild02", 12, 117),
    ("prt_fild02", 310, 248, "prt_fild03", 53, 194),
    ("prt_fild03", 297, 145, "prt_fild04", 340, 345),
    ("prt_fild04", 184, 99, "prt_fild05", 61, 363),
    ("prt_fild05", 271, 52, "prt_fild06", 64, 368),
    ("prt_fild06", 284, 177, "prt_fild07", 222, 372),
    ("prt_fild07", 61, 31, "prt_fild08", 289, 357),
    ("prt_fild08", 23, 117, "prt_fild09", 208, 363),
    ("prt_fild09", 395, 267, "prt_fild10", 109, 378),
    ("prt_fild10", 12, 19, "prt_fild11", 379, 200),

    # ═════════════════════════════════════════════════════════════════════
    # MJOLNIR PATH — Prontera <-> Morocc
    # ═════════════════════════════════════════════════════════════════════

    ("prt_fild08", 189, 177, "mjolnir_01", 31, 211),
    ("mjolnir_01", 207, 26, "mjolnir_02", 25, 207),
    ("mjolnir_02", 207, 203, "mjolnir_03", 22, 228),
    ("mjolnir_03", 223, 24, "mjolnir_04", 21, 202),
    ("mjolnir_04", 203, 204, "mjolnir_05", 26, 214),
    ("mjolnir_05", 204, 27, "morocc", 318, 86),

    # ═════════════════════════════════════════════════════════════════════
    # MOROCC — Desert Town + Fields
    # ═════════════════════════════════════════════════════════════════════

    ("morocc", 71, 133, "moc_fild01", 237, 345),
    ("morocc", 271, 33, "moc_fild02", 239, 344),
    ("morocc", 42, 186, "moc_fild03", 296, 354),
    ("morocc", 168, 54, "moc_fild05", 33, 386),
    ("morocc", 144, 47, "moc_pryd01", 29, 48),  # Pyramid entrance
    ("morocc", 197, 356, "moc_fild11", 206, 31),
    ("morocc", 131, 245, "moc_fild12", 83, 355),

    # Morocc field interconnections
    ("moc_fild01", 163, 201, "moc_fild02", 266, 195),
    ("moc_fild02", 30, 374, "moc_fild03", 395, 25),
    ("moc_fild03", 144, 25, "moc_fild04", 175, 370),
    ("moc_fild04", 380, 239, "moc_fild05", 38, 284),
    ("moc_fild05", 162, 43, "moc_fild06", 133, 378),
    ("moc_fild06", 168, 322, "moc_fild07", 204, 30),
    ("moc_fild07", 355, 385, "moc_fild11", 49, 19),
    ("moc_fild11", 98, 73, "moc_fild12", 233, 370),
    ("moc_fild12", 359, 145, "moc_fild13", 358, 367),
    ("moc_fild13", 27, 19, "moc_fild14", 203, 382),

    # Pyramid dungeon floors
    ("moc_pryd01", 13, 98, "moc_pryd02", 42, 100),
    ("moc_pryd02", 56, 57, "moc_pryd03", 33, 93),
    ("moc_pryd03", 49, 19, "moc_pryd04", 112, 89),
    ("moc_pryd04", 96, 25, "moc_pryd05", 37, 98),

    # ═════════════════════════════════════════════════════════════════════
    # PAYON — Archer Town + Fields
    # ═════════════════════════════════════════════════════════════════════

    ("payon", 237, 127, "pay_fild01", 335, 377),
    ("payon", 204, 24, "pay_fild02", 214, 375),
    ("payon", 179, 22, "pay_fild03", 214, 375),
    ("payon", 119, 330, "pay_fild04", 237, 26),
    ("payon", 78, 334, "pay_fild06", 86, 27),
    ("payon", 166, 246, "pay_dun00", 165, 54),  # Payon Cave 1F entrance

    # Payon field interconnections
    ("pay_fild01", 291, 137, "pay_fild02", 33, 350),
    ("pay_fild01", 338, 56, "pay_fild03", 14, 296),
    ("pay_fild02", 158, 34, "pay_fild03", 244, 361),
    ("pay_fild03", 26, 161, "pay_fild04", 345, 344),
    ("pay_fild04", 54, 72, "pay_fild05", 335, 356),
    ("pay_fild05", 118, 78, "pay_fild06", 298, 379),
    ("pay_fild06", 220, 199, "pay_fild07", 20, 194),
    ("pay_fild07", 208, 277, "pay_fild08", 25, 133),
    ("pay_fild08", 395, 188, "pay_fild09", 15, 124),
    ("pay_fild09", 272, 353, "pay_fild10", 298, 14),
    ("pay_fild10", 32, 50, "pay_fild11", 369, 314),

    # Payon Cave (dungeon) floors
    ("pay_dun00", 160, 48, "pay_dun01", 27, 54),  # Payon Cave 1F -> 2F
    ("pay_dun01", 108, 99, "pay_dun02", 110, 160),  # 2F -> 3F
    ("pay_dun02", 25, 22, "pay_dun03", 119, 164),  # 3F -> 4F
    ("pay_dun03", 95, 67, "pay_dun04", 169, 183),  # 4F -> 5F

    # ═════════════════════════════════════════════════════════════════════
    # GEFFEN — Wizard Town + Fields
    # ═════════════════════════════════════════════════════════════════════

    ("geffen", 38, 239, "gef_fild00", 120, 375),
    ("geffen", 201, 298, "gef_fild01", 43, 10),
    ("geffen", 99, 24, "gef_fild02", 156, 365),
    ("geffen", 222, 145, "gef_fild03", 34, 380),
    ("geffen", 111, 50, "gef_fild04", 253, 364),
    ("geffen", 157, 240, "gef_fild06", 394, 382),
    ("geffen", 142, 26, "gef_fild10", 200, 370),

    # Geffen field interconnections
    ("gef_fild00", 348, 154, "gef_fild01", 20, 370),
    ("gef_fild01", 299, 152, "gef_fild02", 352, 380),
    ("gef_fild02", 74, 197, "gef_fild03", 352, 352),
    ("gef_fild03", 78, 105, "gef_fild04", 329, 367),
    ("gef_fild04", 34, 139, "gef_fild05", 341, 340),
    ("gef_fild05", 107, 34, "gef_fild06", 376, 326),
    ("gef_fild06", 38, 91, "gef_fild07", 349, 366),
    ("gef_fild07", 49, 87, "gef_fild08", 329, 365),
    ("gef_fild08", 186, 130, "gef_fild09", 11, 280),
    ("gef_fild09", 201, 162, "gef_fild10", 355, 344),
    ("gef_fild10", 80, 207, "gef_fild11", 241, 359),
    ("gef_fild11", 40, 123, "gef_fild12", 254, 18),
    ("gef_fild12", 267, 330, "gef_fild13", 20, 41),
    ("gef_fild13", 344, 147, "gef_fild14", 135, 370),

    # Orc Dungeon (progression map, level 35+)
    ("gef_fild00", 228, 36, "orcsdun01", 44, 182),  # Orc Dungeon entrance
    ("orcsdun01", 17, 145, "orcsdun02", 49, 119),  # Orc Dungeon 1F -> 2F

    # ═════════════════════════════════════════════════════════════════════
    # IZLUDE — Byalan Dungeon (level 50+ progression)
    # ═════════════════════════════════════════════════════════════════════

    ("izlude", 120, 50, "iz_dun00", 185, 94),  # Byalan entrance
    ("iz_dun00", 149, 80, "iz_dun01", 25, 122),
    ("iz_dun01", 22, 94, "iz_dun02", 69, 147),
    ("iz_dun02", 75, 25, "iz_dun03", 86, 164),
    ("iz_dun03", 70, 74, "iz_dun04", 29, 150),

    # ═════════════════════════════════════════════════════════════════════
    # ALDEBARAN — Clock Tower (level 85+ progression)
    # ═════════════════════════════════════════════════════════════════════

    ("aldebaran", 178, 97, "alde_fild01", 163, 377),
    ("aldebaran", 41, 52, "alde_fild02", 265, 370),
    ("aldebaran", 110, 137, "alde_dun00", 129, 20),  # Clock Tower 1F entrance
    ("alde_dun00", 92, 23, "alde_dun01", 124, 48),
    ("alde_dun01", 145, 97, "alde_dun02", 91, 22),
    ("alde_dun02", 188, 109, "alde_dun03", 154, 15),
    ("alde_dun03", 33, 95, "alde_dun04", 143, 191),

    # ═════════════════════════════════════════════════════════════════════
    # CROSS-AREA CONNECTIONS — Travel between major towns
    # ═════════════════════════════════════════════════════════════════════

    # Prontera <-> Payon (via mjolnir -> morocc -> pay_fild path)
    # Direct: prontera <-> mjolnir_01 <-> ... <-> morocc
    # Then: morocc <-> moc_fild01 <-> pay_fild
    ("moc_fild01", 51, 47, "pay_fild07", 393, 356),  # Morocc desert -> Payon path
    ("moc_fild14", 212, 37, "pay_fild04", 21, 27),  # Alternate route

    # Geffen <-> Prontera (southern route)
    ("gef_fild05", 318, 261, "prt_fild03", 58, 28),  # Cross-area connection

    # Geffen <-> Payon (through mountain path)
    ("gef_fild14", 270, 46, "pay_fild09", 44, 34),

    # Geffen <-> Morocc (southern desert)
    ("gef_fild08", 273, 28, "moc_fild08", 324, 362),

    # ═════════════════════════════════════════════════════════════════════
    # ALBERTA — Port Town (boat travel hub)
    # ═════════════════════════════════════════════════════════════════════

    ("alberta", 51, 264, "izlude", 41, 83),  # Boat dock -> Izlude
    ("alberta", 37, 140, "cmd_fild01", 305, 366),  # Alberta -> Comodo fields
    ("alberta", 174, 274, "ama_fild01", 308, 371),  # Alberta -> Amatsu fields

    # ═════════════════════════════════════════════════════════════════════
    # YUNO — Scholar Town (high level, level 50+)
    # ═════════════════════════════════════════════════════════════════════

    ("yuno", 248, 230, "yuno_fild01", 94, 370),
    ("yuno", 117, 129, "yuno_fild02", 143, 375),
    ("yuno", 222, 30, "yuno_fild03", 204, 363),
    ("yuno", 95, 159, "yuno_fild04", 194, 378),
    ("yuno_fild01", 301, 37, "yuno_fild02", 183, 379),
    ("yuno_fild02", 225, 29, "yuno_fild03", 54, 380),
    ("yuno_fild03", 192, 201, "yuno_fild04", 332, 19),

    # ═════════════════════════════════════════════════════════════════════
    # EINBROCH — Industrial City (level 70+)
    # ═════════════════════════════════════════════════════════════════════

    ("einbroch", 74, 262, "ein_fild01", 50, 380),
    ("einbroch", 239, 82, "ein_fild02", 119, 366),
    ("einbroch", 162, 306, "ein_fild03", 183, 368),
    ("ein_fild01", 303, 36, "ein_fild02", 174, 381),
    ("ein_fild02", 39, 168, "ein_fild03", 358, 315),
    ("ein_fild03", 25, 43, "ein_fild04", 374, 373),
    ("ein_fild04", 59, 50, "ein_fild05", 364, 376),
    ("ein_fild05", 89, 122, "ein_fild06", 182, 383),
    ("ein_fild06", 40, 38, "ein_fild07", 326, 372),
    ("ein_fild07", 381, 128, "ein_fild08", 27, 331),
    ("ein_fild08", 282, 317, "ein_fild09", 10, 174),

    # ═════════════════════════════════════════════════════════════════════
    # CULVERT / EIN_DUN (level 70-85 progression)
    # ═════════════════════════════════════════════════════════════════════

    ("ein_fild06", 213, 290, "ein_dun00", 38, 143),  # Culvert entrance
    ("ein_dun00", 183, 170, "ein_dun01", 104, 107),
    ("ein_dun01", 45, 50, "ein_dun02", 124, 106),

    # ═════════════════════════════════════════════════════════════════════
    # COMODO — Beach Resort Town
    # ═════════════════════════════════════════════════════════════════════

    ("comodo", 180, 310, "cmd_fild01", 253, 377),
    ("comodo", 55, 308, "cmd_fild02", 357, 376),
    ("cmd_fild01", 222, 127, "cmd_fild02", 12, 305),
    ("cmd_fild02", 370, 233, "cmd_fild03", 21, 119),
    ("cmd_fild03", 339, 359, "cmd_fild04", 196, 38),
    ("cmd_fild04", 264, 289, "cmd_fild05", 15, 100),
    ("cmd_fild05", 161, 395, "cmd_fild06", 222, 49),
    ("cmd_fild06", 270, 218, "cmd_fild07", 94, 374),
    ("cmd_fild07", 350, 248, "cmd_fild08", 20, 153),
    ("cmd_fild08", 330, 335, "cmd_fild09", 29, 55),

    # ═════════════════════════════════════════════════════════════════════
    # LIGHTHALZEN — Endgame Town (level 85+)
    # ═════════════════════════════════════════════════════════════════════

    ("lighthalzen", 114, 148, "lhz_fild01", 348, 374),
    ("lighthalzen", 131, 306, "lhz_fild02", 235, 374),
    ("lighthalzen", 160, 152, "lhz_fild03", 291, 371),
    ("lhz_fild01", 219, 111, "lhz_fild02", 400, 352),
    ("lhz_fild02", 173, 189, "lhz_fild03", 68, 8),

    # ═════════════════════════════════════════════════════════════════════
    # HUGEL — Garden Town
    # ═════════════════════════════════════════════════════════════════════

    ("hugel", 194, 112, "hu_fild01", 17, 125),
    ("hugel", 72, 183, "hu_fild02", 71, 6),
    ("hugel", 56, 285, "hu_fild03", 48, 18),
    ("hu_fild01", 324, 284, "hu_fild02", 118, 369),
    ("hu_fild02", 301, 211, "hu_fild03", 83, 372),
    ("hu_fild03", 262, 81, "hu_fild04", 102, 357),
    ("hu_fild04", 292, 131, "hu_fild05", 20, 321),
    ("hu_fild05", 312, 344, "hu_fild06", 299, 22),
    ("hu_fild06", 211, 211, "hu_fild07", 19, 43),

    # ═════════════════════════════════════════════════════════════════════
    # RACHEL — Spiritual Town (Sanctuary)
    # ═════════════════════════════════════════════════════════════════════

    ("rachel", 293, 335, "ra_fild01", 73, 372),
    ("rachel", 99, 115, "ra_fild02", 149, 373),
    ("ra_fild01", 333, 299, "ra_fild02", 20, 83),
    ("ra_fild02", 296, 267, "ra_fild03", 145, 380),
    ("ra_fild03", 20, 19, "ra_fild04", 375, 368),
    ("ra_fild04", 175, 37, "ra_fild05", 320, 348),
    ("ra_fild05", 80, 80, "ra_fild06", 368, 375),
    ("ra_fild06", 188, 22, "ra_fild07", 342, 367),
    ("ra_fild07", 205, 70, "ra_fild08", 326, 360),
    ("ra_fild08", 178, 76, "ra_fild09", 372, 360),
    ("ra_fild09", 42, 85, "ra_fild10", 381, 345),
    ("ra_fild10", 262, 279, "ra_fild11", 314, 124),
    ("ra_fild11", 222, 331, "ra_fild12", 226, 24),
]

# ── Additional town connections for completeness ──────────────────────
# Some towns connect directly to each other via field paths
# These synthetic edges model real traversal routes that span multiple fields

_TOWN_PATH_EDGES: list[tuple[str, str, int]] = [
    # These are multi-field routes recorded as direct edges for pathfinding
    # weight = approximate number of map boundaries crossed
    
    # Prontera <-> Major Towns
    ("prontera", "morocc", 6),    # via mjolnir path (6 map crossings)
    ("prontera", "geffen", 8),    # via prt_fild03 -> gef_fild05
    ("prontera", "payon", 12),    # via morocc -> pay_fild path
    ("prontera", "alberta", 6),   # via prt_fild08 -> mjolnir -> morocc -> ... 
    
    # Morocc <-> Other Towns
    ("morocc", "payon", 6),       # via moc_fild01 -> pay_fild07 path
    
    # Geffen <-> Other Towns
    ("geffen", "payon", 8),       # via gef_fild14 -> pay_fild09 path
    
    # Alberta <-> Izlude (boat)
    ("alberta", "izlude", 1),
    ("alberta", "comodo", 5),     # via cmd_fild path
    
    # Yuno <-> Other
    ("yuno", "einbroch", 5),      # contiguous fields
    
    # Rachel <-> Hugel (mountain pass)
    ("rachel", "hugel", 8),
]


class PortalDB:
    """Thread-safe database of all RO portal connections.

    Builds a bidirectional graph of all known warp/portal connections
    between RO maps, with precise coordinates for each portal endpoint.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        # Map connections: map_name -> list of PortalConnection
        self._graph: dict[str, list[PortalConnection]] = {}
        # Town path shortcuts: (map_a, map_b) -> weight
        self._town_paths: dict[tuple[str, str], int] = {}
        self._build()

    def _build(self) -> None:
        """Build the portal graph from raw data."""
        for entry in _PORTAL_DATA:
            conn = PortalConnection(
                map1=entry[0], x1=entry[1], y1=entry[2],
                map2=entry[3], x2=entry[4], y2=entry[5],
            )
            if conn.map1 not in self._graph:
                self._graph[conn.map1] = []
            if conn.map2 not in self._graph:
                self._graph[conn.map2] = []
            self._graph[conn.map1].append(conn)
            self._graph[conn.map2].append(conn)

        for a, b, weight in _TOWN_PATH_EDGES:
            key = (a, b) if a < b else (b, a)
            self._town_paths[key] = weight

        logger.info(
            "PortalDB built: %d maps, %d portal connections, %d town shortcuts",
            len(self._graph),
            len(_PORTAL_DATA),
            len(_TOWN_PATH_EDGES),
        )

    # ── Queries ───────────────────────────────────────────────────────

    def get_portals(self, map_name: str) -> list[PortalConnection]:
        """Get all portals on a given map."""
        with self._lock:
            return list(self._graph.get(map_name, []))

    def get_adjacent_maps(self, map_name: str) -> list[tuple[str, int, int, int, int]]:
        """Get list of (adjacent_map, from_x, from_y, to_x, to_y) for all portals."""
        result: list[tuple[str, int, int, int, int]] = []
        with self._lock:
            for conn in self._graph.get(map_name, []):
                other = conn.other_side(map_name)
                if other:
                    result.append((other[0], conn.x1 if conn.map1 == map_name else conn.x2,
                                   conn.y1 if conn.map1 == map_name else conn.y2,
                                   other[1], other[2]))
        return result

    def has_map(self, map_name: str) -> bool:
        """Check if a map is in the portal database."""
        with self._lock:
            return map_name in self._graph

    def get_all_maps(self) -> list[str]:
        """Get all known map names."""
        with self._lock:
            return sorted(self._graph.keys())

    def get_map_count(self) -> int:
        """Total number of known maps."""
        with self._lock:
            return len(self._graph)

    def get_portal_count(self) -> int:
        """Total number of portal connections."""
        with self._lock:
            return len(_PORTAL_DATA)

    def get_direct_connection_weight(self, map_a: str, map_b: str) -> int | None:
        """Get the weight for a direct town-town shortcut, if one exists."""
        key = (map_a, map_b) if map_a < map_b else (map_b, map_a)
        return self._town_paths.get(key)

    def get_neighbors(self, map_name: str) -> list[tuple[str, int]]:
        """Get all neighboring maps with edge weight=1 (portal hop count).

        Returns list of (map_name, weight) tuples.
        Includes town shortcuts with their respective weights.
        """
        neighbors: dict[str, int] = {}

        # Portal connections have weight 1 (one map boundary crossing)
        with self._lock:
            for conn in self._graph.get(map_name, []):
                other = conn.other_side(map_name)
                if other:
                    neighbors[other[0]] = 1

        # Town shortcuts with their specific weights
        for (a, b), weight in self._town_paths.items():
            if a == map_name:
                if b not in neighbors or weight < neighbors[b]:
                    neighbors[b] = weight
            elif b == map_name:
                if a not in neighbors or weight < neighbors[a]:
                    neighbors[a] = weight

        return list(neighbors.items())

    def get_route_step(self, from_map: str, to_map: str) -> tuple[str, int, int, int, int] | None:
        """Get the precise portal coordinates when traveling from_map -> to_map.

        Returns (target_map, from_x, from_y, to_x, to_y) or None if no direct portal exists.
        """
        with self._lock:
            for conn in self._graph.get(from_map, []):
                other = conn.other_side(from_map)
                if other and other[0] == to_map:
                    if conn.map1 == from_map:
                        return (conn.map2, conn.x1, conn.y1, conn.x2, conn.y2)
                    else:
                        return (conn.map1, conn.x2, conn.y2, conn.x1, conn.y1)
        return None


# ── Global Singleton ──

_db: PortalDB | None = None
_db_lock = RLock()


def get_portal_db() -> PortalDB:
    """Get the global PortalDB singleton."""
    global _db
    with _db_lock:
        if _db is None:
            _db = PortalDB()
        return _db
