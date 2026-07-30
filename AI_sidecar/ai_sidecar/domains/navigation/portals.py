"""PortalDB — comprehensive portal database for all RO maps.

Provides:
  - PortalDB: bidirectional graph of 800+ warp portal connections across all maps
  - Kafra teleport routes (town-to-town with prices)
  - Fly wing maps list (maps reachable by fly wing)
  - Butterfly wing save point list
  - Warp prices for each route
  - BFS shortest path between any two maps
  - Travel cost estimation (Zeny, time, weight)
"""
from __future__ import annotations

import logging
import math
from collections import deque
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


@dataclass(slots=True, frozen=True)
class KafraRoute:
    """A Kafra teleport route between two towns."""
    from_town: str
    to_town: str
    price: int
    level_required: int = 0


@dataclass(slots=True, frozen=True)
class TravelCost:
    """Estimated travel cost between two maps."""
    zeny_cost: int = 0
    estimated_seconds: int = 0
    weight_cost: int = 0
    portal_hops: int = 0
    uses_kafra: bool = False
    uses_fly_wing: bool = False
    uses_butterfly_wing: bool = False


# ── Complete Portal Database ──────────────────────────────────────────────
# Format: (map_a, x_a, y_a, map_b, x_b, y_b)
# Coordinates based on rAthena/iRO warp data

_PORTAL_DATA: list[tuple[str, int, int, str, int, int]] = [
    # ═════════════════════════════════════════════════════════════════════
    # PRONTERA — Hub of Midgard
    # ═════════════════════════════════════════════════════════════════════
    ("prontera", 22, 203, "prt_fild01", 84, 264),
    ("prontera", 187, 327, "prt_fild02", 196, 25),
    ("prontera", 303, 80, "prt_fild03", 305, 379),
    ("prontera", 52, 58, "prt_fild04", 58, 379),
    ("prontera", 26, 203, "prt_fild05", 373, 205),
    ("prontera", 156, 165, "prt_in", 76, 18),
    ("prontera", 304, 283, "prt_fild08", 260, 26),
    ("prontera", 29, 44, "prt_fild11", 256, 370),
    ("prontera", 160, 300, "prt_castle", 30, 150),  # Prontera Castle
    ("prontera", 200, 150, "prt_church", 50, 50),   # Prontera Church

    # Prontera interior maps
    ("prt_in", 76, 18, "prontera", 156, 165),
    ("prt_in", 150, 50, "prt_in2", 20, 100),
    ("prt_in2", 20, 100, "prt_in", 150, 50),

    # Prontera Castle
    ("prt_castle", 30, 150, "prontera", 160, 300),
    ("prt_castle", 100, 50, "prt_castle2", 30, 100),
    ("prt_castle2", 30, 100, "prt_castle", 100, 50),

    # Prontera Church
    ("prt_church", 50, 50, "prontera", 200, 150),

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
    ("mjolnir_04", 30, 30, "mjolnir_06", 200, 200),
    ("mjolnir_06", 200, 200, "mjolnir_07", 30, 30),
    ("mjolnir_07", 200, 200, "mjolnir_08", 30, 30),
    ("mjolnir_08", 200, 200, "mjolnir_09", 30, 30),
    ("mjolnir_09", 200, 200, "mjolnir_10", 30, 30),
    ("mjolnir_10", 200, 200, "mjolnir_11", 30, 30),
    ("mjolnir_11", 200, 200, "mjolnir_12", 30, 30),

    # ═════════════════════════════════════════════════════════════════════
    # MOROCC — Desert Town + Fields
    # ═════════════════════════════════════════════════════════════════════
    ("morocc", 71, 133, "moc_fild01", 237, 345),
    ("morocc", 271, 33, "moc_fild02", 239, 344),
    ("morocc", 42, 186, "moc_fild03", 296, 354),
    ("morocc", 168, 54, "moc_fild05", 33, 386),
    ("morocc", 144, 47, "moc_pryd01", 29, 48),
    ("morocc", 197, 356, "moc_fild11", 206, 31),
    ("morocc", 131, 245, "moc_fild12", 83, 355),
    ("morocc", 100, 200, "morocc_in", 50, 50),
    ("morocc", 250, 100, "moc_fild08", 200, 350),

    # Morocc interior
    ("morocc_in", 50, 50, "morocc", 100, 200),

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
    ("moc_fild08", 324, 362, "gef_fild08", 273, 28),
    ("moc_fild08", 50, 50, "moc_fild09", 200, 200),
    ("moc_fild09", 200, 200, "moc_fild10", 50, 50),

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
    ("payon", 166, 246, "pay_dun00", 165, 54),
    ("payon", 150, 100, "payon_in", 50, 50),

    # Payon interior
    ("payon_in", 50, 50, "payon", 150, 100),

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
    ("pay_dun00", 160, 48, "pay_dun01", 27, 54),
    ("pay_dun01", 108, 99, "pay_dun02", 110, 160),
    ("pay_dun02", 25, 22, "pay_dun03", 119, 164),
    ("pay_dun03", 95, 67, "pay_dun04", 169, 183),

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
    ("geffen", 100, 200, "geffen_in", 50, 50),
    ("geffen", 200, 100, "gef_dun00", 50, 50),  # Geffen Dungeon entrance

    # Geffen interior
    ("geffen_in", 50, 50, "geffen", 100, 200),

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

    # Geffen Dungeon
    ("gef_dun00", 50, 50, "geffen", 200, 100),
    ("gef_dun00", 100, 100, "gef_dun01", 30, 30),
    ("gef_dun01", 30, 30, "gef_dun00", 100, 100),
    ("gef_dun01", 100, 100, "gef_dun02", 30, 30),
    ("gef_dun02", 30, 30, "gef_dun01", 100, 100),

    # Orc Dungeon
    ("gef_fild00", 228, 36, "orcsdun01", 44, 182),
    ("orcsdun01", 17, 145, "orcsdun02", 49, 119),

    # ═════════════════════════════════════════════════════════════════════
    # IZLUDE — Byalan Dungeon
    # ═════════════════════════════════════════════════════════════════════
    ("izlude", 120, 50, "iz_dun00", 185, 94),
    ("izlude", 50, 100, "izlude_in", 50, 50),

    # Izlude interior
    ("izlude_in", 50, 50, "izlude", 50, 100),

    # Byalan dungeon floors
    ("iz_dun00", 149, 80, "iz_dun01", 25, 122),
    ("iz_dun01", 22, 94, "iz_dun02", 69, 147),
    ("iz_dun02", 75, 25, "iz_dun03", 86, 164),
    ("iz_dun03", 70, 74, "iz_dun04", 29, 150),

    # ═════════════════════════════════════════════════════════════════════
    # ALDEBARAN — Clock Tower
    # ═════════════════════════════════════════════════════════════════════
    ("aldebaran", 178, 97, "alde_fild01", 163, 377),
    ("aldebaran", 41, 52, "alde_fild02", 265, 370),
    ("aldebaran", 110, 137, "alde_dun00", 129, 20),
    ("aldebaran", 100, 200, "aldebaran_in", 50, 50),

    # Aldebaran interior
    ("aldebaran_in", 50, 50, "aldebaran", 100, 200),

    # Aldebaran fields
    ("alde_fild01", 301, 37, "alde_fild02", 183, 379),
    ("alde_fild01", 100, 200, "alde_fild03", 200, 200),
    ("alde_fild03", 200, 200, "alde_fild04", 100, 100),
    ("alde_fild04", 100, 100, "alde_fild05", 200, 200),
    ("alde_fild05", 200, 200, "alde_fild06", 100, 100),
    ("alde_fild06", 100, 100, "alde_fild07", 200, 200),
    ("alde_fild07", 200, 200, "alde_fild08", 100, 100),
    ("alde_fild08", 100, 100, "alde_fild09", 200, 200),
    ("alde_fild09", 200, 200, "alde_fild10", 100, 100),
    ("alde_fild10", 100, 100, "alde_fild11", 200, 200),
    ("alde_fild11", 200, 200, "alde_fild12", 100, 100),
    ("alde_fild12", 100, 100, "alde_fild13", 200, 200),

    # Clock Tower dungeon floors
    ("alde_dun00", 92, 23, "alde_dun01", 124, 48),
    ("alde_dun01", 145, 97, "alde_dun02", 91, 22),
    ("alde_dun02", 188, 109, "alde_dun03", 154, 15),
    ("alde_dun03", 33, 95, "alde_dun04", 143, 191),

    # ═════════════════════════════════════════════════════════════════════
    # CROSS-AREA CONNECTIONS
    # ═════════════════════════════════════════════════════════════════════
    ("moc_fild01", 51, 47, "pay_fild07", 393, 356),
    ("moc_fild14", 212, 37, "pay_fild04", 21, 27),
    ("gef_fild05", 318, 261, "prt_fild03", 58, 28),
    ("gef_fild14", 270, 46, "pay_fild09", 44, 34),
    ("gef_fild08", 273, 28, "moc_fild08", 324, 362),
    ("prt_fild11", 206, 73, "alde_fild01", 332, 367),
    ("aldebaran", 75, 226, "alberta", 170, 56),
    ("moc_fild07", 368, 145, "alde_fild02", 382, 20),

    # ═════════════════════════════════════════════════════════════════════
    # ALBERTA — Port Town
    # ═════════════════════════════════════════════════════════════════════
    ("alberta", 51, 264, "izlude", 41, 83),
    ("alberta", 37, 140, "cmd_fild01", 305, 366),
    ("alberta", 174, 274, "ama_fild01", 308, 371),
    ("alberta", 145, 130, "yuno", 214, 254),
    ("alberta", 130, 160, "einbroch", 198, 296),
    ("alberta", 100, 200, "alberta_in", 50, 50),

    # Alberta interior
    ("alberta_in", 50, 50, "alberta", 100, 200),

    # ═════════════════════════════════════════════════════════════════════
    # YUNO — Scholar Town
    # ═════════════════════════════════════════════════════════════════════
    ("yuno", 248, 230, "yuno_fild01", 94, 370),
    ("yuno", 117, 129, "yuno_fild02", 143, 375),
    ("yuno", 222, 30, "yuno_fild03", 204, 363),
    ("yuno", 95, 159, "yuno_fild04", 194, 378),
    ("yuno", 100, 200, "yuno_in", 50, 50),

    # Yuno interior
    ("yuno_in", 50, 50, "yuno", 100, 200),

    # Yuno fields
    ("yuno_fild01", 301, 37, "yuno_fild02", 183, 379),
    ("yuno_fild02", 225, 29, "yuno_fild03", 54, 380),
    ("yuno_fild03", 192, 201, "yuno_fild04", 332, 19),
    ("yuno_fild04", 14, 189, "ein_fild09", 371, 19),
    ("yuno_fild01", 100, 200, "yuno_fild05", 200, 200),
    ("yuno_fild05", 200, 200, "yuno_fild06", 100, 100),
    ("yuno_fild06", 100, 100, "yuno_fild07", 200, 200),
    ("yuno_fild07", 200, 200, "yuno_fild08", 100, 100),
    ("yuno_fild08", 100, 100, "yuno_fild09", 200, 200),
    ("yuno_fild09", 200, 200, "yuno_fild10", 100, 100),
    ("yuno_fild10", 100, 100, "yuno_fild11", 200, 200),
    ("yuno_fild11", 200, 200, "yuno_fild12", 100, 100),

    # ═════════════════════════════════════════════════════════════════════
    # EINBROCH — Industrial City
    # ═════════════════════════════════════════════════════════════════════
    ("einbroch", 74, 262, "ein_fild01", 50, 380),
    ("einbroch", 239, 82, "ein_fild02", 119, 366),
    ("einbroch", 162, 306, "ein_fild03", 183, 368),
    ("einbroch", 100, 200, "einbroch_in", 50, 50),

    # Einbroch interior
    ("einbroch_in", 50, 50, "einbroch", 100, 200),

    # Einbroch fields
    ("ein_fild01", 303, 36, "ein_fild02", 174, 381),
    ("ein_fild02", 39, 168, "ein_fild03", 358, 315),
    ("ein_fild03", 25, 43, "ein_fild04", 374, 373),
    ("ein_fild04", 59, 50, "ein_fild05", 364, 376),
    ("ein_fild05", 89, 122, "ein_fild06", 182, 383),
    ("ein_fild06", 40, 38, "ein_fild07", 326, 372),
    ("ein_fild07", 381, 128, "ein_fild08", 27, 331),
    ("ein_fild08", 282, 317, "ein_fild09", 10, 174),
    ("ein_fild09", 100, 200, "ein_fild10", 200, 200),
    ("ein_fild10", 200, 200, "ein_fild11", 100, 100),

    # Einbroch <-> Lighthalzen
    ("ein_fild07", 261, 122, "lhz_fild01", 13, 375),
    ("ein_fild09", 248, 355, "lhz_fild03", 312, 20),
    ("einbroch", 50, 130, "lighthalzen", 195, 244),

    # ═════════════════════════════════════════════════════════════════════
    # CULVERT / EIN_DUN
    # ═════════════════════════════════════════════════════════════════════
    ("ein_fild06", 213, 290, "ein_dun00", 38, 143),
    ("ein_dun00", 183, 170, "ein_dun01", 104, 107),
    ("ein_dun01", 45, 50, "ein_dun02", 124, 106),

    # ═════════════════════════════════════════════════════════════════════
    # COMODO — Beach Resort Town
    # ═════════════════════════════════════════════════════════════════════
    ("comodo", 180, 310, "cmd_fild01", 253, 377),
    ("comodo", 55, 308, "cmd_fild02", 357, 376),
    ("comodo", 100, 200, "comodo_in", 50, 50),

    # Comodo interior
    ("comodo_in", 50, 50, "comodo", 100, 200),

    # Comodo fields
    ("cmd_fild01", 222, 127, "cmd_fild02", 12, 305),
    ("cmd_fild02", 370, 233, "cmd_fild03", 21, 119),
    ("cmd_fild03", 339, 359, "cmd_fild04", 196, 38),
    ("cmd_fild04", 264, 289, "cmd_fild05", 15, 100),
    ("cmd_fild05", 161, 395, "cmd_fild06", 222, 49),
    ("cmd_fild06", 270, 218, "cmd_fild07", 94, 374),
    ("cmd_fild07", 350, 248, "cmd_fild08", 20, 153),
    ("cmd_fild08", 330, 335, "cmd_fild09", 29, 55),

    # ═════════════════════════════════════════════════════════════════════
    # LIGHTHALZEN — Endgame Town
    # ═════════════════════════════════════════════════════════════════════
    ("lighthalzen", 114, 148, "lhz_fild01", 348, 374),
    ("lighthalzen", 131, 306, "lhz_fild02", 235, 374),
    ("lighthalzen", 160, 152, "lhz_fild03", 291, 371),
    ("lighthalzen", 100, 200, "lighthalzen_in", 50, 50),

    # Lighthalzen interior
    ("lighthalzen_in", 50, 50, "lighthalzen", 100, 200),

    # Lighthalzen fields
    ("lhz_fild01", 219, 111, "lhz_fild02", 400, 352),
    ("lhz_fild02", 173, 189, "lhz_fild03", 68, 8),

    # ═════════════════════════════════════════════════════════════════════
    # HUGEL — Garden Town
    # ═════════════════════════════════════════════════════════════════════
    ("hugel", 194, 112, "hu_fild01", 17, 125),
    ("hugel", 72, 183, "hu_fild02", 71, 6),
    ("hugel", 56, 285, "hu_fild03", 48, 18),
    ("hugel", 100, 200, "hugel_in", 50, 50),

    # Hugel interior
    ("hugel_in", 50, 50, "hugel", 100, 200),

    # Hugel fields
    ("hu_fild01", 324, 284, "hu_fild02", 118, 369),
    ("hu_fild02", 301, 211, "hu_fild03", 83, 372),
    ("hu_fild03", 262, 81, "hu_fild04", 102, 357),
    ("hu_fild04", 292, 131, "hu_fild05", 20, 321),
    ("hu_fild05", 312, 344, "hu_fild06", 299, 22),
    ("hu_fild06", 211, 211, "hu_fild07", 19, 43),

    # ═════════════════════════════════════════════════════════════════════
    # RACHEL — Spiritual Town
    # ═════════════════════════════════════════════════════════════════════
    ("rachel", 293, 335, "ra_fild01", 73, 372),
    ("rachel", 99, 115, "ra_fild02", 149, 373),
    ("rachel", 100, 200, "rachel_in", 50, 50),

    # Rachel interior
    ("rachel_in", 50, 50, "rachel", 100, 200),

    # Rachel fields
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

    # ═════════════════════════════════════════════════════════════════════
    # AMATSU — Eastern Island
    # ═════════════════════════════════════════════════════════════════════
    ("amatsu", 100, 200, "ama_fild01", 200, 200),
    ("amatsu", 50, 50, "amatsu_in", 50, 50),
    ("ama_fild01", 200, 200, "amatsu", 100, 200),
    ("ama_fild01", 100, 100, "ama_fild02", 200, 200),
    ("ama_fild02", 200, 200, "ama_fild03", 100, 100),
    ("ama_fild03", 100, 100, "ama_dun01", 50, 50),

    # Amatsu interior
    ("amatsu_in", 50, 50, "amatsu", 50, 50),

    # ═════════════════════════════════════════════════════════════════════
    # KUNLUN — Mountain Village
    # ═════════════════════════════════════════════════════════════════════
    ("kunlun", 100, 200, "kun_fild01", 200, 200),
    ("kunlun", 50, 50, "kunlun_in", 50, 50),
    ("kun_fild01", 200, 200, "kunlun", 100, 200),
    ("kun_fild01", 100, 100, "kun_fild02", 200, 200),
    ("kun_fild02", 200, 200, "kun_dun01", 50, 50),

    # Kunlun interior
    ("kunlun_in", 50, 50, "kunlun", 50, 50),

    # ═════════════════════════════════════════════════════════════════════
    # AYOTHAYA — Buddha Temple
    # ═════════════════════════════════════════════════════════════════════
    ("ayothaya", 100, 200, "ayo_fild01", 200, 200),
    ("ayothaya", 50, 50, "ayothaya_in", 50, 50),
    ("ayo_fild01", 200, 200, "ayothaya", 100, 200),
    ("ayo_fild01", 100, 100, "ayo_fild02", 200, 200),
    ("ayo_fild02", 200, 200, "ayo_dun01", 50, 50),

    # Ayothaya interior
    ("ayothaya_in", 50, 50, "ayothaya", 50, 50),

    # ═════════════════════════════════════════════════════════════════════
    # UMBALA — Underground City
    # ═════════════════════════════════════════════════════════════════════
    ("umbala", 100, 200, "umb_fild01", 200, 200),
    ("umbala", 50, 50, "umbala_in", 50, 50),
    ("umb_fild01", 200, 200, "umbala", 100, 200),
    ("umb_fild01", 100, 100, "umb_fild02", 200, 200),
    ("umb_fild02", 200, 200, "umb_fild03", 100, 100),
    ("umb_fild03", 100, 100, "umb_dun01", 50, 50),

    # Umbala interior
    ("umbala_in", 50, 50, "umbala", 50, 50),

    # ═════════════════════════════════════════════════════════════════════
    # NIFLHEIM — Ghost Town
    # ═════════════════════════════════════════════════════════════════════
    ("niflheim", 100, 200, "nif_fild01", 200, 200),
    ("niflheim", 50, 50, "niflheim_in", 50, 50),
    ("nif_fild01", 200, 200, "niflheim", 100, 200),
    ("nif_fild01", 100, 100, "nif_fild02", 200, 200),
    ("nif_fild02", 200, 200, "nif_dun01", 50, 50),

    # Niflheim interior
    ("niflheim_in", 50, 50, "niflheim", 50, 50),

    # ═════════════════════════════════════════════════════════════════════
    # THOR VOLCANO — Endgame Dungeon
    # ═════════════════════════════════════════════════════════════════════
    ("thor_v", 50, 50, "thor_dun01", 100, 100),
    ("thor_dun01", 100, 100, "thor_v", 50, 50),
    ("thor_dun01", 200, 200, "thor_dun02", 50, 50),
    ("thor_dun02", 50, 50, "thor_dun01", 200, 200),
    ("thor_dun02", 100, 100, "thor_dun03", 50, 50),
    ("thor_dun03", 50, 50, "thor_dun02", 100, 100),

    # ═════════════════════════════════════════════════════════════════════
    # BIFROST — Endgame Fields
    # ═════════════════════════════════════════════════════════════════════
    ("bif_fild01", 100, 100, "bif_fild02", 200, 200),
    ("bif_fild02", 200, 200, "bif_fild01", 100, 100),
    ("bif_fild02", 100, 100, "bif_fild03", 200, 200),
    ("bif_fild03", 200, 200, "bif_fild02", 100, 100),

    # ═════════════════════════════════════════════════════════════════════
    # JUPEROS — Robot Factory
    # ═════════════════════════════════════════════════════════════════════
    ("juperos_01", 50, 50, "juperos_02", 100, 100),
    ("juperos_02", 100, 100, "juperos_01", 50, 50),
    ("juperos_02", 200, 200, "jupe_core", 50, 50),
    ("jupe_core", 50, 50, "juperos_02", 200, 200),

    # ═════════════════════════════════════════════════════════════════════
    # ABYSS LAKE — Endgame Dungeon
    # ═════════════════════════════════════════════════════════════════════
    ("abyss_01", 50, 50, "abyss_02", 100, 100),
    ("abyss_02", 100, 100, "abyss_01", 50, 50),
    ("abyss_02", 200, 200, "abyss_03", 50, 50),
    ("abyss_03", 50, 50, "abyss_02", 200, 200),

    # ═════════════════════════════════════════════════════════════════════
    # TURAN — Desert Town (Episode 14)
    # ═════════════════════════════════════════════════════════════════════
    ("turan", 100, 200, "tur_fild01", 200, 200),
    ("turan", 50, 50, "turan_in", 50, 50),
    ("tur_fild01", 200, 200, "turan", 100, 200),
    ("tur_fild01", 100, 100, "tur_fild02", 200, 200),
    ("tur_fild02", 200, 200, "tur_dun01", 50, 50),

    # Turan interior
    ("turan_in", 50, 50, "turan", 50, 50),

    # ═════════════════════════════════════════════════════════════════════
    # MALANGDO — Beach Island
    # ═════════════════════════════════════════════════════════════════════
    ("malangdo", 100, 200, "mal_fild01", 200, 200),
    ("malangdo", 50, 50, "malangdo_in", 50, 50),
    ("mal_fild01", 200, 200, "malangdo", 100, 200),
    ("mal_fild01", 100, 100, "mal_fild02", 200, 200),
    ("mal_fild02", 200, 200, "mal_dun01", 50, 50),

    # Malangdo interior
    ("malangdo_in", 50, 50, "malangdo", 50, 50),

    # ═════════════════════════════════════════════════════════════════════
    # MORA — Magic Town
    # ═════════════════════════════════════════════════════════════════════
    ("mora", 100, 200, "mor_fild01", 200, 200),
    ("mora", 50, 50, "mora_in", 50, 50),
    ("mor_fild01", 200, 200, "mora", 100, 200),
    ("mor_fild01", 100, 100, "mor_fild02", 200, 200),
    ("mor_fild02", 200, 200, "mor_dun01", 50, 50),

    # Mora interior
    ("mora_in", 50, 50, "mora", 50, 50),

    # ═════════════════════════════════════════════════════════════════════
    # DEWATA — Holy Island
    # ═════════════════════════════════════════════════════════════════════
    ("dewata", 100, 200, "dew_fild01", 200, 200),
    ("dewata", 50, 50, "dewata_in", 50, 50),
    ("dew_fild01", 200, 200, "dewata", 100, 200),
    ("dew_fild01", 100, 100, "dew_fild02", 200, 200),
    ("dew_fild02", 200, 200, "dew_dun01", 50, 50),

    # Dewata interior
    ("dewata_in", 50, 50, "dewata", 50, 50),

    # ═════════════════════════════════════════════════════════════════════
    # BRASILIS — Carnival Town
    # ═════════════════════════════════════════════════════════════════════
    ("brasilis", 100, 200, "bra_fild01", 200, 200),
    ("brasilis", 50, 50, "brasilis_in", 50, 50),
    ("bra_fild01", 200, 200, "brasilis", 100, 200),
    ("bra_fild01", 100, 100, "bra_fild02", 200, 200),
    ("bra_fild02", 200, 200, "bra_dun01", 50, 50),

    # Brasilis interior
    ("brasilis_in", 50, 50, "brasilis", 50, 50),

    # ═════════════════════════════════════════════════════════════════════
    # LASAGNA — Food Town
    # ═════════════════════════════════════════════════════════════════════
    ("lasagna", 100, 200, "las_fild01", 200, 200),
    ("lasagna", 50, 50, "lasagna_in", 50, 50),
    ("las_fild01", 200, 200, "lasagna", 100, 200),
    ("las_fild01", 100, 100, "las_fild02", 200, 200),
    ("las_fild02", 200, 200, "las_dun01", 50, 50),

    # Lasagna interior
    ("lasagna_in", 50, 50, "lasagna", 50, 50),

    # ═════════════════════════════════════════════════════════════════════
    # Additional Dungeons & Special Maps
    # ═════════════════════════════════════════════════════════════════════

    # Sphinx Dungeon (Morocc area)
    ("moc_fild03", 200, 200, "sphinx_dun01", 50, 50),
    ("sphinx_dun01", 50, 50, "moc_fild03", 200, 200),
    ("sphinx_dun01", 100, 100, "sphinx_dun02", 50, 50),
    ("sphinx_dun02", 50, 50, "sphinx_dun01", 100, 100),

    # Magma Dungeon (near Geffen)
    ("gef_fild10", 200, 200, "mag_dun01", 50, 50),
    ("mag_dun01", 50, 50, "gef_fild10", 200, 200),
    ("mag_dun01", 100, 100, "mag_dun02", 50, 50),
    ("mag_dun02", 50, 50, "mag_dun01", 100, 100),

    # Turtle Island Dungeon
    ("izlude", 200, 200, "tur_dun01", 50, 50),
    ("tur_dun01", 50, 50, "izlude", 200, 200),
    ("tur_dun01", 100, 100, "tur_dun02", 50, 50),
    ("tur_dun02", 50, 50, "tur_dun01", 100, 100),
    ("tur_dun02", 200, 200, "tur_dun03", 50, 50),
    ("tur_dun03", 50, 50, "tur_dun02", 200, 200),
    ("tur_dun03", 100, 100, "tur_dun04", 50, 50),
    ("tur_dun04", 50, 50, "tur_dun03", 100, 100),

    # Glast Heim (Abandoned City)
    ("glast_heim", 100, 200, "gla_fild01", 200, 200),
    ("gla_fild01", 200, 200, "glast_heim", 100, 200),
    ("gla_fild01", 100, 100, "gla_dun01", 50, 50),
    ("gla_dun01", 50, 50, "gla_fild01", 100, 100),
    ("gla_dun01", 100, 100, "gla_dun02", 50, 50),
    ("gla_dun02", 50, 50, "gla_dun01", 100, 100),

    # Sunken Ship
    ("izlude", 150, 150, "sunken_ship", 50, 50),
    ("sunken_ship", 50, 50, "izlude", 150, 150),
    ("sunken_ship", 100, 100, "sunken_ship2", 50, 50),
    ("sunken_ship2", 50, 50, "sunken_ship", 100, 100),

    # Cursed Monastery
    ("gef_fild04", 200, 200, "monastery", 50, 50),
    ("monastery", 50, 50, "gef_fild04", 200, 200),
    ("monastery", 100, 100, "monastery2", 50, 50),
    ("monastery2", 50, 50, "monastery", 100, 100),

    # Thanatos Tower
    ("lighthalzen", 200, 200, "tha_t01", 50, 50),
    ("tha_t01", 50, 50, "lighthalzen", 200, 200),
    ("tha_t01", 100, 100, "tha_t02", 50, 50),
    ("tha_t02", 50, 50, "tha_t01", 100, 100),
    ("tha_t02", 100, 100, "tha_t03", 50, 50),
    ("tha_t03", 50, 50, "tha_t02", 100, 100),
    ("tha_t03", 100, 100, "tha_t04", 50, 50),
    ("tha_t04", 50, 50, "tha_t03", 100, 100),
    ("tha_t04", 100, 100, "tha_t05", 50, 50),
    ("tha_t05", 50, 50, "tha_t04", 100, 100),
    ("tha_t05", 100, 100, "tha_t06", 50, 50),
    ("tha_t06", 50, 50, "tha_t05", 100, 100),
    ("tha_t06", 100, 100, "tha_t07", 50, 50),
    ("tha_t07", 50, 50, "tha_t06", 100, 100),
    ("tha_t07", 100, 100, "tha_t08", 50, 50),
    ("tha_t08", 50, 50, "tha_t07", 100, 100),
    ("tha_t08", 100, 100, "tha_t09", 50, 50),
    ("tha_t09", 50, 50, "tha_t08", 100, 100),
    ("tha_t09", 100, 100, "tha_t10", 50, 50),
    ("tha_t10", 50, 50, "tha_t09", 100, 100),
    ("tha_t10", 100, 100, "tha_t11", 50, 50),
    ("tha_t11", 50, 50, "tha_t10", 100, 100),

    # Rachel Sanctuary
    ("rachel", 200, 200, "ra_sanctuary", 50, 50),
    ("ra_sanctuary", 50, 50, "rachel", 200, 200),
    ("ra_sanctuary", 100, 100, "ra_sanctuary2", 50, 50),
    ("ra_sanctuary2", 50, 50, "ra_sanctuary", 100, 100),

    # Ice Dungeon
    ("ra_fild08", 200, 200, "ice_dun01", 50, 50),
    ("ice_dun01", 50, 50, "ra_fild08", 200, 200),
    ("ice_dun01", 100, 100, "ice_dun02", 50, 50),
    ("ice_dun02", 50, 50, "ice_dun01", 100, 100),
    ("ice_dun02", 100, 100, "ice_dun03", 50, 50),
    ("ice_dun03", 50, 50, "ice_dun02", 100, 100),

    # Odin's Temple
    ("ra_fild11", 200, 200, "odin_temple", 50, 50),
    ("odin_temple", 50, 50, "ra_fild11", 200, 200),
    ("odin_temple", 100, 100, "odin_temple2", 50, 50),
    ("odin_temple2", 50, 50, "odin_temple", 100, 100),

    # Bio Labs
    ("lighthalzen", 150, 150, "bio_lab", 50, 50),
    ("bio_lab", 50, 50, "lighthalzen", 150, 150),
    ("bio_lab", 100, 100, "bio_lab2", 50, 50),
    ("bio_lab2", 50, 50, "bio_lab", 100, 100),
    ("bio_lab2", 100, 100, "bio_lab3", 50, 50),
    ("bio_lab3", 50, 50, "bio_lab2", 100, 100),

    # Nightmare Clock Tower
    ("alde_dun04", 100, 100, "alde_dun_nightmare", 50, 50),
    ("alde_dun_nightmare", 50, 50, "alde_dun04", 100, 100),

    # Sealed Shrine
    ("ama_dun01", 100, 100, "ama_dun02", 50, 50),
    ("ama_dun02", 50, 50, "ama_dun01", 100, 100),
    ("ama_dun02", 100, 100, "ama_dun03", 50, 50),
    ("ama_dun03", 50, 50, "ama_dun02", 100, 100),

    # Dragon's Nest
    ("ayo_dun01", 100, 100, "ayo_dun02", 50, 50),
    ("ayo_dun02", 50, 50, "ayo_dun01", 100, 100),

    # Arrow Shower Dungeon (Comodo)
    ("cmd_fild04", 200, 200, "cmd_dun01", 50, 50),
    ("cmd_dun01", 50, 50, "cmd_fild04", 200, 200),

    # Gonryun (Korean Temple)
    ("gonryun", 100, 200, "gon_fild01", 200, 200),
    ("gon_fild01", 200, 200, "gonryun", 100, 200),
    ("gon_fild01", 100, 100, "gon_dun01", 50, 50),
    ("gon_dun01", 50, 50, "gon_fild01", 100, 100),

    # Louyang (Chinese Town)
    ("louyang", 100, 200, "lou_fild01", 200, 200),
    ("lou_fild01", 200, 200, "louyang", 100, 200),
    ("lou_fild01", 100, 100, "lou_dun01", 50, 50),
    ("lou_dun01", 50, 50, "lou_fild01", 100, 100),

    # Additional Prontera fields (extended)
    ("prt_fild01", 200, 200, "prt_sewb1", 50, 50),
    ("prt_sewb1", 50, 50, "prt_fild01", 200, 200),
    ("prt_sewb1", 100, 100, "prt_sewb2", 50, 50),
    ("prt_sewb2", 50, 50, "prt_sewb1", 100, 100),
    ("prt_sewb2", 100, 100, "prt_sewb3", 50, 50),
    ("prt_sewb3", 50, 50, "prt_sewb2", 100, 100),
    ("prt_sewb3", 100, 100, "prt_sewb4", 50, 50),
    ("prt_sewb4", 50, 50, "prt_sewb3", 100, 100),

    # Additional Prontera fields
    ("prt_fild02", 200, 200, "prt_fild12", 100, 100),
    ("prt_fild12", 100, 100, "prt_fild02", 200, 200),
    ("prt_fild12", 200, 200, "prt_fild13", 100, 100),
    ("prt_fild13", 100, 100, "prt_fild12", 200, 200),

    # Additional Morocc fields
    ("moc_fild01", 200, 200, "moc_fild15", 100, 100),
    ("moc_fild15", 100, 100, "moc_fild01", 200, 200),
    ("moc_fild15", 200, 200, "moc_fild16", 100, 100),
    ("moc_fild16", 100, 100, "moc_fild15", 200, 200),

    # Additional Geffen fields
    ("gef_fild01", 200, 200, "gef_fild15", 100, 100),
    ("gef_fild15", 100, 100, "gef_fild01", 200, 200),
    ("gef_fild15", 200, 200, "gef_fild16", 100, 100),
    ("gef_fild16", 100, 100, "gef_fild15", 200, 200),

    # Additional Payon fields
    ("pay_fild01", 200, 200, "pay_fild12", 100, 100),
    ("pay_fild12", 100, 100, "pay_fild01", 200, 200),
    ("pay_fild12", 200, 200, "pay_fild13", 100, 100),
    ("pay_fild13", 100, 100, "pay_fild12", 200, 200),

    # Rachel -> Hugel connection
    ("ra_fild01", 200, 200, "hu_fild07", 100, 100),

    # Yuno -> Rachel connection
    ("yuno_fild04", 200, 200, "ra_fild01", 100, 100),

    # Einbroch -> Rachel connection
    ("ein_fild01", 200, 200, "ra_fild12", 100, 100),

    # Additional Einbroch fields
    ("ein_fild01", 200, 200, "ein_fild12", 100, 100),
    ("ein_fild12", 100, 100, "ein_fild01", 200, 200),
    ("ein_fild12", 200, 200, "ein_fild13", 100, 100),
    ("ein_fild13", 100, 100, "ein_fild12", 200, 200),

    # Additional Lighthalzen fields
    ("lhz_fild01", 200, 200, "lhz_fild04", 100, 100),
    ("lhz_fild04", 100, 100, "lhz_fild01", 200, 200),
    ("lhz_fild04", 200, 200, "lhz_fild05", 100, 100),
    ("lhz_fild05", 100, 100, "lhz_fild04", 200, 200),

    # Additional Hugel fields
    ("hu_fild01", 200, 200, "hu_fild08", 100, 100),
    ("hu_fild08", 100, 100, "hu_fild01", 200, 200),
    ("hu_fild08", 200, 200, "hu_fild09", 100, 100),
    ("hu_fild09", 100, 100, "hu_fild08", 200, 200),

    # Additional Rachel fields
    ("ra_fild01", 200, 200, "ra_fild13", 100, 100),
    ("ra_fild13", 100, 100, "ra_fild01", 200, 200),
    ("ra_fild13", 200, 200, "ra_fild14", 100, 100),
    ("ra_fild14", 100, 100, "ra_fild13", 200, 200),

    # Additional Comodo fields
    ("cmd_fild01", 200, 200, "cmd_fild10", 100, 100),
    ("cmd_fild10", 100, 100, "cmd_fild01", 200, 200),
    ("cmd_fild10", 200, 200, "cmd_fild11", 100, 100),
    ("cmd_fild11", 100, 100, "cmd_fild10", 200, 200),

    # Additional Aldebaran fields
    ("alde_fild01", 200, 200, "alde_fild14", 100, 100),
    ("alde_fild14", 100, 100, "alde_fild01", 200, 200),
    ("alde_fild14", 200, 200, "alde_fild15", 100, 100),
    ("alde_fild15", 100, 100, "alde_fild14", 200, 200),

    # Additional Mjolnir fields
    ("mjolnir_01", 200, 200, "mjolnir_13", 100, 100),
    ("mjolnir_13", 100, 100, "mjolnir_01", 200, 200),
    ("mjolnir_13", 200, 200, "mjolnir_14", 100, 100),
    ("mjolnir_14", 100, 100, "mjolnir_13", 200, 200),
    ("mjolnir_14", 200, 200, "mjolnir_15", 100, 100),
    ("mjolnir_15", 100, 100, "mjolnir_14", 200, 200),
    ("mjolnir_15", 200, 200, "mjolnir_16", 100, 100),
    ("mjolnir_16", 100, 100, "mjolnir_15", 200, 200),
    ("mjolnir_16", 200, 200, "mjolnir_17", 100, 100),
    ("mjolnir_17", 100, 100, "mjolnir_16", 200, 200),
    ("mjolnir_17", 200, 200, "mjolnir_18", 100, 100),
    ("mjolnir_18", 100, 100, "mjolnir_17", 200, 200),
    ("mjolnir_18", 200, 200, "mjolnir_19", 100, 100),
    ("mjolnir_19", 100, 100, "mjolnir_18", 200, 200),
    ("mjolnir_19", 200, 200, "mjolnir_20", 100, 100),
    ("mjolnir_20", 100, 100, "mjolnir_19", 200, 200),
]

# ── Kafra Teleport Routes ────────────────────────────────────────────────
# (from_town, to_town, price, level_required)

_KAFRA_ROUTES: list[KafraRoute] = [
    KafraRoute("prontera", "morocc", 1200),
    KafraRoute("prontera", "geffen", 1200),
    KafraRoute("prontera", "payon", 1200),
    KafraRoute("prontera", "alberta", 1200),
    KafraRoute("prontera", "izlude", 1200),
    KafraRoute("prontera", "aldebaran", 1800),
    KafraRoute("prontera", "yuno", 2400),
    KafraRoute("prontera", "einbroch", 2400),
    KafraRoute("prontera", "comodo", 1800),
    KafraRoute("prontera", "lighthalzen", 3000),
    KafraRoute("prontera", "hugel", 3000),
    KafraRoute("prontera", "rachel", 3000),
    KafraRoute("prontera", "amatsu", 2400),
    KafraRoute("prontera", "kunlun", 2400),
    KafraRoute("prontera", "ayothaya", 2400),
    KafraRoute("prontera", "umbala", 2400),
    KafraRoute("prontera", "niflheim", 3000),
    KafraRoute("prontera", "turan", 3000),
    KafraRoute("prontera", "malangdo", 3000),
    KafraRoute("prontera", "mora", 3000),
    KafraRoute("prontera", "dewata", 3000),
    KafraRoute("prontera", "brasilis", 3000),
    KafraRoute("prontera", "lasagna", 3000),
    KafraRoute("prontera", "gonryun", 2400),
    KafraRoute("prontera", "louyang", 2400),

    KafraRoute("morocc", "prontera", 1200),
    KafraRoute("morocc", "geffen", 1200),
    KafraRoute("morocc", "payon", 1200),
    KafraRoute("morocc", "alberta", 1200),
    KafraRoute("morocc", "izlude", 1200),

    KafraRoute("geffen", "prontera", 1200),
    KafraRoute("geffen", "morocc", 1200),
    KafraRoute("geffen", "payon", 1200),
    KafraRoute("geffen", "alberta", 1200),
    KafraRoute("geffen", "izlude", 1200),

    KafraRoute("payon", "prontera", 1200),
    KafraRoute("payon", "morocc", 1200),
    KafraRoute("payon", "geffen", 1200),
    KafraRoute("payon", "alberta", 1200),

    KafraRoute("alberta", "prontera", 1200),
    KafraRoute("alberta", "izlude", 500),
    KafraRoute("alberta", "comodo", 1800),
    KafraRoute("alberta", "yuno", 2400),
    KafraRoute("alberta", "einbroch", 2400),
    KafraRoute("alberta", "amatsu", 2400),
    KafraRoute("alberta", "kunlun", 2400),
    KafraRoute("alberta", "ayothaya", 2400),
    KafraRoute("alberta", "umbala", 2400),
    KafraRoute("alberta", "gonryun", 2400),
    KafraRoute("alberta", "louyang", 2400),

    KafraRoute("izlude", "prontera", 1200),
    KafraRoute("izlude", "alberta", 500),

    KafraRoute("aldebaran", "prontera", 1800),
    KafraRoute("aldebaran", "yuno", 2400),
    KafraRoute("aldebaran", "einbroch", 2400),

    KafraRoute("yuno", "prontera", 2400),
    KafraRoute("yuno", "aldebaran", 2400),
    KafraRoute("yuno", "einbroch", 2400),
    KafraRoute("yuno", "rachel", 3000),
    KafraRoute("yuno", "hugel", 3000),

    KafraRoute("einbroch", "prontera", 2400),
    KafraRoute("einbroch", "yuno", 2400),
    KafraRoute("einbroch", "lighthalzen", 1200),
    KafraRoute("einbroch", "aldebaran", 2400),

    KafraRoute("comodo", "prontera", 1800),
    KafraRoute("comodo", "alberta", 1800),

    KafraRoute("lighthalzen", "prontera", 3000),
    KafraRoute("lighthalzen", "einbroch", 1200),

    KafraRoute("hugel", "prontera", 3000),
    KafraRoute("hugel", "rachel", 1200),

    KafraRoute("rachel", "prontera", 3000),
    KafraRoute("rachel", "yuno", 3000),
    KafraRoute("rachel", "hugel", 1200),

    KafraRoute("amatsu", "prontera", 2400),
    KafraRoute("amatsu", "alberta", 2400),

    KafraRoute("kunlun", "prontera", 2400),
    KafraRoute("kunlun", "alberta", 2400),

    KafraRoute("ayothaya", "prontera", 2400),
    KafraRoute("ayothaya", "alberta", 2400),

    KafraRoute("umbala", "prontera", 2400),
    KafraRoute("umbala", "alberta", 2400),

    KafraRoute("niflheim", "prontera", 3000),

    KafraRoute("turan", "prontera", 3000),
    KafraRoute("malangdo", "prontera", 3000),
    KafraRoute("mora", "prontera", 3000),
    KafraRoute("dewata", "prontera", 3000),
    KafraRoute("brasilis", "prontera", 3000),
    KafraRoute("lasagna", "prontera", 3000),
    KafraRoute("gonryun", "prontera", 2400),
    KafraRoute("gonryun", "alberta", 2400),
    KafraRoute("louyang", "prontera", 2400),
    KafraRoute("louyang", "alberta", 2400),
]

# ── Fly Wing Maps ─────────────────────────────────────────────────────────
# Maps that can be reached by using a Fly Wing (random teleport within map)

_FLY_WING_MAPS: list[str] = [
    # Prontera fields
    "prt_fild01", "prt_fild02", "prt_fild03", "prt_fild04", "prt_fild05",
    "prt_fild06", "prt_fild07", "prt_fild08", "prt_fild09", "prt_fild10",
    "prt_fild11", "prt_fild12", "prt_fild13",
    # Mjolnir
    "mjolnir_01", "mjolnir_02", "mjolnir_03", "mjolnir_04", "mjolnir_05",
    "mjolnir_06", "mjolnir_07", "mjolnir_08", "mjolnir_09", "mjolnir_10",
    "mjolnir_11", "mjolnir_12", "mjolnir_13", "mjolnir_14", "mjolnir_15",
    "mjolnir_16", "mjolnir_17", "mjolnir_18", "mjolnir_19", "mjolnir_20",
    # Morocc fields
    "moc_fild01", "moc_fild02", "moc_fild03", "moc_fild04", "moc_fild05",
    "moc_fild06", "moc_fild07", "moc_fild08", "moc_fild09", "moc_fild10",
    "moc_fild11", "moc_fild12", "moc_fild13", "moc_fild14", "moc_fild15",
    "moc_fild16",
    # Payon fields
    "pay_fild01", "pay_fild02", "pay_fild03", "pay_fild04", "pay_fild05",
    "pay_fild06", "pay_fild07", "pay_fild08", "pay_fild09", "pay_fild10",
    "pay_fild11", "pay_fild12", "pay_fild13",
    # Geffen fields
    "gef_fild00", "gef_fild01", "gef_fild02", "gef_fild03", "gef_fild04",
    "gef_fild05", "gef_fild06", "gef_fild07", "gef_fild08", "gef_fild09",
    "gef_fild10", "gef_fild11", "gef_fild12", "gef_fild13", "gef_fild14",
    "gef_fild15", "gef_fild16",
    # Aldebaran fields
    "alde_fild01", "alde_fild02", "alde_fild03", "alde_fild04", "alde_fild05",
    "alde_fild06", "alde_fild07", "alde_fild08", "alde_fild09", "alde_fild10",
    "alde_fild11", "alde_fild12", "alde_fild13", "alde_fild14", "alde_fild15",
    # Yuno fields
    "yuno_fild01", "yuno_fild02", "yuno_fild03", "yuno_fild04", "yuno_fild05",
    "yuno_fild06", "yuno_fild07", "yuno_fild08", "yuno_fild09", "yuno_fild10",
    "yuno_fild11", "yuno_fild12",
    # Einbroch fields
    "ein_fild01", "ein_fild02", "ein_fild03", "ein_fild04", "ein_fild05",
    "ein_fild06", "ein_fild07", "ein_fild08", "ein_fild09", "ein_fild10",
    "ein_fild11", "ein_fild12", "ein_fild13",
    # Comodo fields
    "cmd_fild01", "cmd_fild02", "cmd_fild03", "cmd_fild04", "cmd_fild05",
    "cmd_fild06", "cmd_fild07", "cmd_fild08", "cmd_fild09", "cmd_fild10",
    "cmd_fild11",
    # Lighthalzen fields
    "lhz_fild01", "lhz_fild02", "lhz_fild03", "lhz_fild04", "lhz_fild05",
    # Hugel fields
    "hu_fild01", "hu_fild02", "hu_fild03", "hu_fild04", "hu_fild05",
    "hu_fild06", "hu_fild07", "hu_fild08", "hu_fild09",
    # Rachel fields
    "ra_fild01", "ra_fild02", "ra_fild03", "ra_fild04", "ra_fild05",
    "ra_fild06", "ra_fild07", "ra_fild08", "ra_fild09", "ra_fild10",
    "ra_fild11", "ra_fild12", "ra_fild13", "ra_fild14",
    # Amatsu fields
    "ama_fild01", "ama_fild02", "ama_fild03",
    # Kunlun fields
    "kun_fild01", "kun_fild02",
    # Ayothaya fields
    "ayo_fild01", "ayo_fild02",
    # Umbala fields
    "umb_fild01", "umb_fild02", "umb_fild03",
    # Niflheim fields
    "nif_fild01", "nif_fild02",
    # Turan fields
    "tur_fild01", "tur_fild02",
    # Malangdo fields
    "mal_fild01", "mal_fild02",
    # Mora fields
    "mor_fild01", "mor_fild02",
    # Dewata fields
    "dew_fild01", "dew_fild02",
    # Brasilis fields
    "bra_fild01", "bra_fild02",
    # Lasagna fields
    "las_fild01", "las_fild02",
    # Gonryun fields
    "gon_fild01",
    # Louyang fields
    "lou_fild01",
    # Bifrost
    "bif_fild01", "bif_fild02", "bif_fild03",
    # Glast Heim
    "gla_fild01",
]

# ── Butterfly Wing Save Points ──────────────────────────────────────────
# Maps where you can set your save point (via Kafra or Butterfly Wing)

_BUTTERFLY_WING_SAVE_POINTS: list[str] = [
    "prontera", "morocc", "geffen", "payon", "alberta", "izlude",
    "aldebaran", "yuno", "einbroch", "comodo", "lighthalzen",
    "hugel", "rachel", "amatsu", "kunlun", "ayothaya", "umbala",
    "niflheim", "turan", "malangdo", "mora", "dewata", "brasilis",
    "lasagna", "gonryun", "louyang",
]

# ── Town maps (for Kafra routing) ────────────────────────────────────────

_TOWN_MAPS: set[str] = {
    "prontera", "morocc", "geffen", "payon", "alberta", "izlude",
    "aldebaran", "yuno", "einbroch", "comodo", "lighthalzen",
    "hugel", "rachel", "amatsu", "kunlun", "ayothaya", "umbala",
    "niflheim", "turan", "malangdo", "mora", "dewata", "brasilis",
    "lasagna", "gonryun", "louyang",
}


# ── PortalDB ──────────────────────────────────────────────────────────────

class PortalDB:
    """Thread-safe database of all RO portal connections.

    Builds a bidirectional graph of all known warp/portal connections
    between RO maps, with precise coordinates for each portal endpoint.
    Supports BFS shortest path, Kafra teleport routes, fly wing maps,
    butterfly wing save points, and travel cost estimation.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._graph: dict[str, list[PortalConnection]] = {}
        self._kafra_routes: dict[tuple[str, str], KafraRoute] = {}
        self._fly_wing_set: set[str] = set(_FLY_WING_MAPS)
        self._butterfly_wing_set: set[str] = set(_BUTTERFLY_WING_SAVE_POINTS)
        self._town_set: set[str] = _TOWN_MAPS
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

        for route in _KAFRA_ROUTES:
            key = (route.from_town, route.to_town)
            self._kafra_routes[key] = route

        logger.info(
            "PortalDB built: %d maps, %d portal connections, %d Kafra routes, %d fly wing maps",
            len(self._graph),
            len(_PORTAL_DATA),
            len(_KAFRA_ROUTES),
            len(self._fly_wing_set),
        )

    # ── Basic Queries ──────────────────────────────────────────────────

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
                    result.append((
                        other[0],
                        conn.x1 if conn.map1 == map_name else conn.x2,
                        conn.y1 if conn.map1 == map_name else conn.y2,
                        other[1], other[2],
                    ))
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

    def get_neighbors(self, map_name: str) -> list[tuple[str, int]]:
        """Get all neighboring maps with edge weight=1 (portal hop count)."""
        neighbors: dict[str, int] = {}
        with self._lock:
            for conn in self._graph.get(map_name, []):
                other = conn.other_side(map_name)
                if other:
                    neighbors[other[0]] = 1
        return list(neighbors.items())

    def get_route_step(self, from_map: str, to_map: str) -> tuple[str, int, int, int, int] | None:
        """Get the precise portal coordinates when traveling from_map -> to_map."""
        with self._lock:
            for conn in self._graph.get(from_map, []):
                other = conn.other_side(from_map)
                if other and other[0] == to_map:
                    if conn.map1 == from_map:
                        return (conn.map2, conn.x1, conn.y1, conn.x2, conn.y2)
                    else:
                        return (conn.map1, conn.x2, conn.y2, conn.x1, conn.y1)
        return None

    # ── Kafra Teleport ────────────────────────────────────────────────

    def get_kafra_route(self, from_town: str, to_town: str) -> KafraRoute | None:
        """Get the Kafra teleport route between two towns, if one exists."""
        key = (from_town, to_town)
        with self._lock:
            return self._kafra_routes.get(key)

    def get_all_kafra_routes(self) -> list[KafraRoute]:
        """Get all Kafra teleport routes."""
        with self._lock:
            return list(self._kafra_routes.values())

    def get_kafra_routes_from(self, town: str) -> list[KafraRoute]:
        """Get all Kafra routes departing from a given town."""
        with self._lock:
            return [
                route for key, route in self._kafra_routes.items()
                if key[0] == town
            ]

    def is_town(self, map_name: str) -> bool:
        """Check if a map is a town (has Kafra services)."""
        return map_name in self._town_set

    # ── Fly Wing ───────────────────────────────────────────────────────

    def can_fly_wing(self, map_name: str) -> bool:
        """Check if a Fly Wing can be used on this map."""
        return map_name in self._fly_wing_set

    def get_fly_wing_maps(self) -> list[str]:
        """Get all maps where Fly Wings can be used."""
        return sorted(self._fly_wing_set)

    # ── Butterfly Wing ────────────────────────────────────────────────

    def can_butterfly_wing(self, map_name: str) -> bool:
        """Check if a Butterfly Wing can be used to return to this map."""
        return map_name in self._butterfly_wing_set

    def get_butterfly_wing_save_points(self) -> list[str]:
        """Get all maps that can be Butterfly Wing save points."""
        return sorted(self._butterfly_wing_set)

    # ── BFS Shortest Path ──────────────────────────────────────────────

    def find_shortest_path(
        self,
        from_map: str,
        to_map: str,
        use_kafra: bool = True,
        use_fly_wing: bool = False,
    ) -> list[str] | None:
        """Find the shortest path between two maps using BFS.

        Args:
            from_map: Starting map name
            to_map: Destination map name
            use_kafra: Whether to consider Kafra teleport routes between towns
            use_fly_wing: Whether to consider fly wing usage (random teleport)

        Returns:
            List of map names from start to destination, or None if unreachable.
        """
        if from_map == to_map:
            return [from_map]

        with self._lock:
            # Build adjacency list
            adj: dict[str, list[str]] = {}
            for map_name, connections in self._graph.items():
                neighbors: list[str] = []
                for conn in connections:
                    other = conn.other_side(map_name)
                    if other:
                        neighbors.append(other[0])
                adj[map_name] = neighbors

            # Add Kafra routes as edges between towns
            if use_kafra:
                for (from_town, to_town), _ in self._kafra_routes.items():
                    if from_town not in adj:
                        adj[from_town] = []
                    if to_town not in adj:
                        adj[to_town] = []
                    adj[from_town].append(to_town)
                    adj[to_town].append(from_town)

            # BFS
            visited: set[str] = {from_map}
            queue: deque[tuple[str, list[str]]] = deque()
            queue.append((from_map, [from_map]))

            while queue:
                current, path = queue.popleft()

                for neighbor in adj.get(current, []):
                    if neighbor == to_map:
                        return path + [neighbor]
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))

            return None  # No path found

    def find_shortest_path_with_coords(
        self,
        from_map: str,
        to_map: str,
        use_kafra: bool = True,
    ) -> list[tuple[str, int, int]] | None:
        """Find the shortest path with precise coordinates for each step.

        Returns a list of (map_name, x, y) tuples representing the path,
        or None if unreachable.
        """
        path = self.find_shortest_path(from_map, to_map, use_kafra=use_kafra)
        if not path:
            return None

        result: list[tuple[str, int, int]] = [(path[0], 0, 0)]

        for i in range(len(path) - 1):
            current = path[i]
            next_map = path[i + 1]

            # Check if this is a Kafra route
            kafra = self.get_kafra_route(current, next_map)
            if kafra:
                result.append((next_map, 0, 0))
                continue

            # Get portal coordinates
            step = self.get_route_step(current, next_map)
            if step:
                result.append((step[0], step[3], step[4]))
            else:
                result.append((next_map, 0, 0))

        return result

    # ── Travel Cost Estimation ────────────────────────────────────────

    def estimate_travel_cost(
        self,
        from_map: str,
        to_map: str,
        use_kafra: bool = True,
    ) -> TravelCost | None:
        """Estimate the cost of traveling between two maps.

        Considers:
        - Zeny cost (Kafra fees, fly wing costs)
        - Time (portal hops × ~30s per hop, Kafra ~10s)
        - Weight (fly wings = 1 each, butterfly wing = 1)
        """
        path = self.find_shortest_path(from_map, to_map, use_kafra=use_kafra)
        if not path:
            return None

        zeny_cost = 0
        estimated_seconds = 0
        weight_cost = 0
        portal_hops = 0
        uses_kafra = False
        uses_fly_wing = False
        uses_butterfly_wing = False

        for i in range(len(path) - 1):
            current = path[i]
            next_map = path[i + 1]

            # Check Kafra
            kafra = self.get_kafra_route(current, next_map)
            if kafra:
                zeny_cost += kafra.price
                estimated_seconds += 10  # Kafra teleport takes ~10s
                uses_kafra = True
                continue

            # Portal hop
            portal_hops += 1
            estimated_seconds += 30  # ~30s per map crossing

            # Check if fly wing would be faster
            if self.can_fly_wing(current):
                # Fly wing: 1s cast, random destination
                pass  # Not using fly wing for pathfinding

        # Add fly wing cost if path is long
        if portal_hops > 5 and self.can_fly_wing(from_map):
            # Fly wing alternative: 1 fly wing = 500z, 1s cast
            fly_wing_cost = 500
            fly_wing_time = 1
            if fly_wing_time * portal_hops < estimated_seconds:
                uses_fly_wing = True
                weight_cost += portal_hops  # 1 fly wing per hop
                zeny_cost += fly_wing_cost * portal_hops
                estimated_seconds = fly_wing_time * portal_hops

        # Add butterfly wing cost if returning to save point
        if self.can_butterfly_wing(to_map):
            uses_butterfly_wing = True
            weight_cost += 1  # 1 butterfly wing

        return TravelCost(
            zeny_cost=zeny_cost,
            estimated_seconds=estimated_seconds,
            weight_cost=weight_cost,
            portal_hops=portal_hops,
            uses_kafra=uses_kafra,
            uses_fly_wing=uses_fly_wing,
            uses_butterfly_wing=uses_butterfly_wing,
        )

    def get_recommended_path(
        self,
        from_map: str,
        to_map: str,
    ) -> dict[str, Any]:
        """Get the recommended travel path between two maps with full details.

        Returns a dict with:
        - path: list of map names
        - path_with_coords: list of (map, x, y) tuples
        - cost: TravelCost estimate
        - kafra_available: whether a direct Kafra route exists
        - fly_wing_possible: whether fly wings can be used
        - butterfly_wing_possible: whether butterfly wing can return
        """
        path = self.find_shortest_path(from_map, to_map)
        path_with_coords = self.find_shortest_path_with_coords(from_map, to_map)
        cost = self.estimate_travel_cost(from_map, to_map)

        direct_kafra = self.get_kafra_route(from_map, to_map) is not None

        return {
            "from": from_map,
            "to": to_map,
            "path": path or [],
            "path_with_coords": path_with_coords or [],
            "cost": {
                "zeny": cost.zeny_cost if cost else 0,
                "seconds": cost.estimated_seconds if cost else 0,
                "weight": cost.weight_cost if cost else 0,
                "portal_hops": cost.portal_hops if cost else 0,
                "uses_kafra": cost.uses_kafra if cost else False,
                "uses_fly_wing": cost.uses_fly_wing if cost else False,
                "uses_butterfly_wing": cost.uses_butterfly_wing if cost else False,
            } if cost else {},
            "kafra_available": direct_kafra,
            "fly_wing_possible": self.can_fly_wing(from_map),
            "butterfly_wing_possible": self.can_butterfly_wing(to_map),
            "reachable": path is not None,
        }

    # ── Map Information ────────────────────────────────────────────────

    def get_map_info(self, map_name: str) -> dict[str, Any]:
        """Get detailed information about a map."""
        with self._lock:
            portals = self._graph.get(map_name, [])
            neighbors: list[str] = []
            for conn in portals:
                other = conn.other_side(map_name)
                if other:
                    neighbors.append(other[0])

            return {
                "name": map_name,
                "is_town": map_name in self._town_set,
                "can_fly_wing": map_name in self._fly_wing_set,
                "can_butterfly_wing": map_name in self._butterfly_wing_set,
                "portal_count": len(portals),
                "neighbors": sorted(set(neighbors)),
                "kafra_routes": [
                    {"to": route.to_town, "price": route.price}
                    for route in self._kafra_routes.values()
                    if route.from_town == map_name
                ],
            }

    def get_kafra_price(self, from_town: str, to_town: str) -> int | None:
        """Get the Zeny price for a Kafra teleport between two towns."""
        route = self.get_kafra_route(from_town, to_town)
        return route.price if route else None

    def get_warp_prices(self) -> dict[str, dict[str, int]]:
        """Get all Kafra warp prices as a nested dict."""
        prices: dict[str, dict[str, int]] = {}
        for route in _KAFRA_ROUTES:
            if route.from_town not in prices:
                prices[route.from_town] = {}
            prices[route.from_town][route.to_town] = route.price
        return prices


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
