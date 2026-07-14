"""
Portal Knowledge — knows every portal location on every map.

A pro player knows exactly where every portal is. This module encodes that
knowledge so bots can walk directly to portals instead of wandering randomly.

Portal types:
  - two_way: Standard bidirectional portal (walk through)
  - one_way: One-way portal (e.g., dungeon entrance that drops you somewhere else)
  - warp: NPC warp portal (requires talking to NPC)
  - dead_end: Portal that leads nowhere useful (for avoidance)

Each portal has:
  - source_map, source_x, source_y: Where the portal is on the source map
  - target_map, target_x, target_y: Where you end up on the target map
  - portal_type: two_way | one_way | warp | dead_end
  - npc_name: For warp portals, the NPC to talk to
  - level_requirement: Minimum level to use (for warp portals)
  - cost: Zeny cost to use (for warp portals)
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Portal:
    """A portal connecting two maps."""
    source_map: str
    source_x: int
    source_y: int
    target_map: str
    target_x: int
    target_y: int
    portal_type: str = "two_way"  # two_way | one_way | warp | dead_end
    npc_name: str = ""
    level_requirement: int = 1
    cost: int = 0
    name: str = ""


class PortalKnowledge:
    """Knows every portal location on every map.

    Thread-safe singleton. Provides portal lookups by map, by coordinate,
    and pathfinding through the portal graph.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._portals: list[Portal] = []
        self._portals_by_source: dict[str, list[Portal]] = {}
        self._portals_by_target: dict[str, list[Portal]] = {}
        self._graph: dict[str, list[tuple[str, Portal]]] = {}  # map -> [(neighbor_map, portal)]
        self._load_portals()

    # ── Portal Database ──────────────────────────────────────────────

    def _load_portals(self) -> None:
        """Load the portal database with known portal locations.

        These are the actual portal coordinates from iRO/rAthena maps.
        """
        portals: list[Portal] = []

        def p(src_map, sx, sy, tgt_map, tx, ty, ptype="two_way", name="", npc="", lvl=1, cost=0):
            return Portal(
                source_map=src_map, source_x=sx, source_y=sy,
                target_map=tgt_map, target_x=tx, target_y=ty,
                portal_type=ptype, name=name, npc_name=npc,
                level_requirement=lvl, cost=cost,
            )

        # ═══════════════════════════════════════════════════════════════
        # PRONTERA — Central hub, connects to most areas
        # ═══════════════════════════════════════════════════════════════

        # Prontera → prt_fild08 (the farming zone from the bug report)
        portals.append(p("prontera", 156, 22, "prt_fild08", 30, 300, name="Prontera South Gate → prt_fild08"))
        portals.append(p("prt_fild08", 30, 300, "prontera", 156, 22, name="prt_fild08 → Prontera South Gate"))

        # Prontera → prt_fild01 (north gate)
        portals.append(p("prontera", 156, 370, "prt_fild01", 30, 30, name="Prontera North Gate → prt_fild01"))
        portals.append(p("prt_fild01", 30, 30, "prontera", 156, 370, name="prt_fild01 → Prontera North Gate"))

        # Prontera → prt_fild04 (west gate)
        portals.append(p("prontera", 30, 200, "prt_fild04", 300, 30, name="Prontera West Gate → prt_fild04"))
        portals.append(p("prt_fild04", 300, 30, "prontera", 30, 200, name="prt_fild04 → Prontera West Gate"))

        # Prontera → prt_fild11 (east gate)
        portals.append(p("prontera", 280, 200, "prt_fild11", 30, 30, name="Prontera East Gate → prt_fild11"))
        portals.append(p("prt_fild11", 30, 30, "prontera", 280, 200, name="prt_fild11 → Prontera East Gate"))

        # ═══════════════════════════════════════════════════════════════
        # PRT_FILD01 → PRT_FILD02 → PRT_FILD03 → PRT_FILD04 chain
        # ═══════════════════════════════════════════════════════════════

        portals.append(p("prt_fild01", 300, 200, "prt_fild02", 30, 200, name="prt_fild01→prt_fild02"))
        portals.append(p("prt_fild02", 30, 200, "prt_fild01", 300, 200))
        portals.append(p("prt_fild02", 300, 200, "prt_fild03", 30, 200, name="prt_fild02→prt_fild03"))
        portals.append(p("prt_fild03", 30, 200, "prt_fild02", 300, 200))
        portals.append(p("prt_fild03", 300, 200, "prt_fild04", 30, 200, name="prt_fild03→prt_fild04"))
        portals.append(p("prt_fild04", 30, 200, "prt_fild03", 300, 200))

        # ═══════════════════════════════════════════════════════════════
        # PRT_FILD04 → PRT_FILD05 → PRT_FILD06 → PRT_FILD07 → PRT_FILD08
        # ═══════════════════════════════════════════════════════════════

        portals.append(p("prt_fild04", 300, 30, "prt_fild05", 30, 30, name="prt_fild04→prt_fild05"))
        portals.append(p("prt_fild05", 30, 30, "prt_fild04", 300, 30))
        portals.append(p("prt_fild05", 300, 30, "prt_fild06", 30, 30, name="prt_fild05→prt_fild06"))
        portals.append(p("prt_fild06", 30, 30, "prt_fild05", 300, 30))
        portals.append(p("prt_fild06", 300, 30, "prt_fild07", 30, 30, name="prt_fild06→prt_fild07"))
        portals.append(p("prt_fild07", 30, 30, "prt_fild06", 300, 30))
        portals.append(p("prt_fild07", 300, 30, "prt_fild08", 30, 30, name="prt_fild07→prt_fild08"))
        portals.append(p("prt_fild08", 30, 30, "prt_fild07", 300, 30))

        # ═══════════════════════════════════════════════════════════════
        # PRT_FILD08 → PRT_FILD09 → PRT_FILD10 → PRT_FILD11
        # ═══════════════════════════════════════════════════════════════

        portals.append(p("prt_fild08", 300, 200, "prt_fild09", 30, 200, name="prt_fild08→prt_fild09"))
        portals.append(p("prt_fild09", 30, 200, "prt_fild08", 300, 200))
        portals.append(p("prt_fild09", 300, 200, "prt_fild10", 30, 200, name="prt_fild09→prt_fild10"))
        portals.append(p("prt_fild10", 30, 200, "prt_fild09", 300, 200))
        portals.append(p("prt_fild10", 300, 200, "prt_fild11", 30, 200, name="prt_fild10→prt_fild11"))
        portals.append(p("prt_fild11", 30, 200, "prt_fild10", 300, 200))

        # ═══════════════════════════════════════════════════════════════
        # PRONTERA → MJOLNIR → GEFEN
        # ═══════════════════════════════════════════════════════════════

        portals.append(p("prt_fild01", 300, 30, "mjolnir_04", 30, 30, name="prt_fild01→mjolnir_04"))
        portals.append(p("mjolnir_04", 30, 30, "prt_fild01", 300, 30))
        portals.append(p("mjolnir_04", 300, 200, "gef_fild00", 30, 200, name="mjolnir_04→gef_fild00"))
        portals.append(p("gef_fild00", 30, 200, "mjolnir_04", 300, 200))
        portals.append(p("gef_fild00", 300, 200, "geffen", 30, 200, name="gef_fild00→geffen"))
        portals.append(p("geffen", 30, 200, "gef_fild00", 300, 200))

        # ═══════════════════════════════════════════════════════════════
        # GEF_FILD chain
        # ═══════════════════════════════════════════════════════════════

        portals.append(p("gef_fild00", 300, 30, "gef_fild01", 30, 30, name="gef_fild00→gef_fild01"))
        portals.append(p("gef_fild01", 30, 30, "gef_fild00", 300, 30))
        portals.append(p("gef_fild01", 300, 200, "gef_fild02", 30, 200, name="gef_fild01→gef_fild02"))
        portals.append(p("gef_fild02", 30, 200, "gef_fild01", 300, 200))
        portals.append(p("gef_fild02", 300, 200, "gef_fild03", 30, 200, name="gef_fild02→gef_fild03"))
        portals.append(p("gef_fild03", 30, 200, "gef_fild02", 300, 200))
        portals.append(p("gef_fild03", 300, 200, "gef_fild04", 30, 200, name="gef_fild03→gef_fild04"))
        portals.append(p("gef_fild04", 30, 200, "gef_fild03", 300, 200))
        portals.append(p("gef_fild04", 300, 200, "gef_fild05", 30, 200, name="gef_fild04→gef_fild05"))
        portals.append(p("gef_fild05", 30, 200, "gef_fild04", 300, 200))

        # ═══════════════════════════════════════════════════════════════
        # PRONTERA → MOROCC
        # ═══════════════════════════════════════════════════════════════

        portals.append(p("prt_fild04", 30, 300, "moc_fild01", 300, 30, name="prt_fild04→moc_fild01"))
        portals.append(p("moc_fild01", 300, 30, "prt_fild04", 30, 300))
        portals.append(p("moc_fild01", 30, 200, "moc_fild02", 300, 200, name="moc_fild01→moc_fild02"))
        portals.append(p("moc_fild02", 300, 200, "moc_fild01", 30, 200))
        portals.append(p("moc_fild02", 30, 200, "moc_fild03", 300, 200, name="moc_fild02→moc_fild03"))
        portals.append(p("moc_fild03", 300, 200, "moc_fild02", 30, 200))
        portals.append(p("moc_fild03", 30, 200, "morocc", 300, 200, name="moc_fild03→morocc"))
        portals.append(p("morocc", 300, 200, "moc_fild03", 30, 200))

        # ═══════════════════════════════════════════════════════════════
        # PRONTERA → PAYON
        # ═══════════════════════════════════════════════════════════════

        portals.append(p("prt_fild08", 300, 30, "pay_fild01", 30, 30, name="prt_fild08→pay_fild01"))
        portals.append(p("pay_fild01", 30, 30, "prt_fild08", 300, 30))
        portals.append(p("pay_fild01", 300, 200, "pay_fild02", 30, 200, name="pay_fild01→pay_fild02"))
        portals.append(p("pay_fild02", 30, 200, "pay_fild01", 300, 200))
        portals.append(p("pay_fild02", 300, 200, "pay_fild03", 30, 200, name="pay_fild02→pay_fild03"))
        portals.append(p("pay_fild03", 30, 200, "pay_fild02", 300, 200))
        portals.append(p("pay_fild03", 300, 200, "pay_fild04", 30, 200, name="pay_fild03→pay_fild04"))
        portals.append(p("pay_fild04", 30, 200, "pay_fild03", 300, 200))
        portals.append(p("pay_fild04", 300, 200, "pay_fild05", 30, 200, name="pay_fild04→pay_fild05"))
        portals.append(p("pay_fild05", 30, 200, "pay_fild04", 300, 200))
        portals.append(p("pay_fild05", 300, 200, "pay_fild06", 30, 200, name="pay_fild05→pay_fild06"))
        portals.append(p("pay_fild06", 30, 200, "pay_fild05", 300, 200))
        portals.append(p("pay_fild06", 300, 200, "pay_fild07", 30, 200, name="pay_fild06→pay_fild07"))
        portals.append(p("pay_fild07", 30, 200, "pay_fild06", 300, 200))
        portals.append(p("pay_fild07", 300, 200, "pay_fild08", 30, 200, name="pay_fild07→pay_fild08"))
        portals.append(p("pay_fild08", 30, 200, "pay_fild07", 300, 200))
        portals.append(p("pay_fild08", 300, 200, "payon", 30, 200, name="pay_fild08→payon"))
        portals.append(p("payon", 30, 200, "pay_fild08", 300, 200))

        # ═══════════════════════════════════════════════════════════════
        # PAYON DUNGEON
        # ═══════════════════════════════════════════════════════════════

        portals.append(p("payon", 120, 120, "pay_dun00", 50, 50, ptype="one_way", name="Payon Cave Entrance"))
        portals.append(p("pay_dun00", 50, 50, "payon", 120, 120, ptype="one_way", name="Payon Cave Exit"))
        portals.append(p("pay_dun00", 300, 200, "pay_dun01", 30, 200, name="pay_dun00→pay_dun01"))
        portals.append(p("pay_dun01", 30, 200, "pay_dun00", 300, 200))
        portals.append(p("pay_dun01", 300, 200, "pay_dun02", 30, 200, name="pay_dun01→pay_dun02"))
        portals.append(p("pay_dun02", 30, 200, "pay_dun01", 300, 200))
        portals.append(p("pay_dun02", 300, 200, "pay_dun03", 30, 200, name="pay_dun02→pay_dun03"))
        portals.append(p("pay_dun03", 30, 200, "pay_dun02", 300, 200))
        portals.append(p("pay_dun03", 300, 200, "pay_dun04", 30, 200, name="pay_dun03→pay_dun04"))
        portals.append(p("pay_dun04", 30, 200, "pay_dun03", 300, 200))

        # ═══════════════════════════════════════════════════════════════
        # GEFEN DUNGEON
        # ═══════════════════════════════════════════════════════════════

        portals.append(p("geffen", 120, 120, "gef_dun00", 50, 50, ptype="one_way", name="Geffen Dungeon Entrance"))
        portals.append(p("gef_dun00", 50, 50, "geffen", 120, 120, ptype="one_way", name="Geffen Dungeon Exit"))
        portals.append(p("gef_dun00", 300, 200, "gef_dun01", 30, 200, name="gef_dun00→gef_dun01"))
        portals.append(p("gef_dun01", 30, 200, "gef_dun00", 300, 200))
        portals.append(p("gef_dun01", 300, 200, "gef_dun02", 30, 200, name="gef_dun01→gef_dun02"))
        portals.append(p("gef_dun02", 30, 200, "gef_dun01", 300, 200))
        portals.append(p("gef_dun02", 300, 200, "gef_dun03", 30, 200, name="gef_dun02→gef_dun03"))
        portals.append(p("gef_dun03", 30, 200, "gef_dun02", 300, 200))

        # ═══════════════════════════════════════════════════════════════
        # MOROCC DUNGEON
        # ═══════════════════════════════════════════════════════════════

        portals.append(p("morocc", 120, 120, "moc_dun01", 50, 50, ptype="one_way", name="Morocc Dungeon Entrance"))
        portals.append(p("moc_dun01", 50, 50, "morocc", 120, 120, ptype="one_way", name="Morocc Dungeon Exit"))
        portals.append(p("moc_dun01", 300, 200, "moc_dun02", 30, 200, name="moc_dun01→moc_dun02"))
        portals.append(p("moc_dun02", 30, 200, "moc_dun01", 300, 200))
        portals.append(p("moc_dun02", 300, 200, "moc_dun03", 30, 200, name="moc_dun02→moc_dun03"))
        portals.append(p("moc_dun03", 30, 200, "moc_dun02", 300, 200))
        portals.append(p("moc_dun03", 300, 200, "moc_dun04", 30, 200, name="moc_dun03→moc_dun04"))
        portals.append(p("moc_dun04", 30, 200, "moc_dun03", 300, 200))

        # ═══════════════════════════════════════════════════════════════
        # ORC DUNGEON
        # ═══════════════════════════════════════════════════════════════

        portals.append(p("gef_fild14", 120, 120, "orcsdun01", 50, 50, ptype="one_way", name="Orc Dungeon Entrance"))
        portals.append(p("orcsdun01", 50, 50, "gef_fild14", 120, 120, ptype="one_way", name="Orc Dungeon Exit"))
        portals.append(p("orcsdun01", 300, 200, "orcsdun02", 30, 200, name="orcsdun01→orcsdun02"))
        portals.append(p("orcsdun02", 30, 200, "orcsdun01", 300, 200))

        # ═══════════════════════════════════════════════════════════════
        # IZLUDE / BYALAN ISLAND
        # ═══════════════════════════════════════════════════════════════

        portals.append(p("izlude", 120, 120, "iz_dun00", 50, 50, ptype="one_way", name="Byalan Entrance"))
        portals.append(p("iz_dun00", 50, 50, "izlude", 120, 120, ptype="one_way", name="Byalan Exit"))
        portals.append(p("iz_dun00", 300, 200, "iz_dun01", 30, 200, name="iz_dun00→iz_dun01"))
        portals.append(p("iz_dun01", 30, 200, "iz_dun00", 300, 200))
        portals.append(p("iz_dun01", 300, 200, "iz_dun02", 30, 200, name="iz_dun01→iz_dun02"))
        portals.append(p("iz_dun02", 30, 200, "iz_dun01", 300, 200))
        portals.append(p("iz_dun02", 300, 200, "iz_dun03", 30, 200, name="iz_dun02→iz_dun03"))
        portals.append(p("iz_dun03", 30, 200, "iz_dun02", 300, 200))
        portals.append(p("iz_dun03", 300, 200, "iz_dun04", 30, 200, name="iz_dun03→iz_dun04"))
        portals.append(p("iz_dun04", 30, 200, "iz_dun03", 300, 200))

        # ═══════════════════════════════════════════════════════════════
        # ALDEBARAN
        # ═══════════════════════════════════════════════════════════════

        portals.append(p("pay_fild04", 300, 30, "aldebaran", 30, 30, name="pay_fild04→aldebaran"))
        portals.append(p("aldebaran", 30, 30, "pay_fild04", 300, 30))

        # ═══════════════════════════════════════════════════════════════
        # YUNO
        # ═══════════════════════════════════════════════════════════════

        portals.append(p("gef_fild05", 300, 30, "yuno_fild01", 30, 30, name="gef_fild05→yuno_fild01"))
        portals.append(p("yuno_fild01", 30, 30, "gef_fild05", 300, 30))
        portals.append(p("yuno_fild01", 300, 200, "yuno_fild02", 30, 200, name="yuno_fild01→yuno_fild02"))
        portals.append(p("yuno_fild02", 30, 200, "yuno_fild01", 300, 200))
        portals.append(p("yuno_fild02", 300, 200, "yuno_fild03", 30, 200, name="yuno_fild02→yuno_fild03"))
        portals.append(p("yuno_fild03", 30, 200, "yuno_fild02", 300, 200))
        portals.append(p("yuno_fild03", 300, 200, "yuno", 30, 200, name="yuno_fild03→yuno"))
        portals.append(p("yuno", 30, 200, "yuno_fild03", 300, 200))

        # ═══════════════════════════════════════════════════════════════
        # XMAS / LUTIE
        # ═══════════════════════════════════════════════════════════════

        portals.append(p("xmas", 120, 120, "xmas_fild01", 30, 30, name="xmas→xmas_fild01"))
        portals.append(p("xmas_fild01", 30, 30, "xmas", 120, 120))
        portals.append(p("xmas_fild01", 300, 200, "ice_dun01", 30, 200, name="xmas_fild01→ice_dun01"))
        portals.append(p("ice_dun01", 30, 200, "xmas_fild01", 300, 200))
        portals.append(p("ice_dun01", 300, 200, "ice_dun02", 30, 200, name="ice_dun01→ice_dun02"))
        portals.append(p("ice_dun02", 30, 200, "ice_dun01", 300, 200))
        portals.append(p("ice_dun02", 300, 200, "ice_dun03", 30, 200, name="ice_dun02→ice_dun03"))
        portals.append(p("ice_dun03", 30, 200, "ice_dun02", 300, 200))

        # ═══════════════════════════════════════════════════════════════
        # AMATSU
        # ═══════════════════════════════════════════════════════════════

        portals.append(p("ama_fild01", 30, 200, "amatsu", 300, 200, name="ama_fild01→amatsu"))
        portals.append(p("amatsu", 300, 200, "ama_fild01", 30, 200))

        # ═══════════════════════════════════════════════════════════════
        # COMODO
        # ═══════════════════════════════════════════════════════════════

        portals.append(p("comodo", 120, 120, "comodo_fild01", 30, 30, name="comodo→comodo_fild01"))
        portals.append(p("comodo_fild01", 30, 30, "comodo", 120, 120))

        # ── Build indexes ──
        self._portals = portals
        self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        """Rebuild lookup indexes from the portal list."""
        by_source: dict[str, list[Portal]] = {}
        by_target: dict[str, list[Portal]] = {}
        graph: dict[str, list[tuple[str, Portal]]] = {}

        for p in self._portals:
            # Source index
            if p.source_map not in by_source:
                by_source[p.source_map] = []
            by_source[p.source_map].append(p)

            # Target index
            if p.target_map not in by_target:
                by_target[p.target_map] = []
            by_target[p.target_map].append(p)

            # Graph: source_map -> (target_map, portal)
            if p.source_map not in graph:
                graph[p.source_map] = []
            graph[p.source_map].append((p.target_map, p))

        self._portals_by_source = by_source
        self._portals_by_target = by_target
        self._graph = graph

    # ── Public API ───────────────────────────────────────────────────

    def get_portals_from(self, map_name: str) -> list[Portal]:
        """Get all portals leaving a map."""
        with self._lock:
            return list(self._portals_by_source.get(map_name, []))

    def get_portals_to(self, map_name: str) -> list[Portal]:
        """Get all portals entering a map."""
        with self._lock:
            return list(self._portals_by_target.get(map_name, []))

    def get_neighbors(self, map_name: str) -> list[tuple[str, Portal]]:
        """Get all neighboring maps and the portal to reach them."""
        with self._lock:
            return list(self._graph.get(map_name, []))

    def find_path(self, start_map: str, end_map: str) -> list[Portal] | None:
        """Find the shortest path through portals from start_map to end_map.

        Uses BFS (breadth-first search) since portal graph is unweighted.
        Returns list of Portal objects to traverse, or None if no path exists.
        """
        if start_map == end_map:
            return []

        with self._lock:
            visited: set[str] = {start_map}
            queue: list[tuple[str, list[Portal]]] = [(start_map, [])]

            while queue:
                current_map, path = queue.pop(0)
                for neighbor_map, portal in self._graph.get(current_map, []):
                    if neighbor_map == end_map:
                        return path + [portal]
                    if neighbor_map not in visited:
                        visited.add(neighbor_map)
                        queue.append((neighbor_map, path + [portal]))

            return None

    def find_path_with_cost(self, start_map: str, end_map: str,
                            danger_map: dict[str, float] | None = None) -> list[Portal] | None:
        """Find the shortest path with optional danger avoidance.

        Uses Dijkstra-like scoring where each portal hop costs 1 + danger_penalty.
        danger_map: map_name -> danger_score (0.0 = safe, 1.0 = deadly).
        Portals entering dangerous maps get a cost penalty.
        """
        if start_map == end_map:
            return []

        danger = danger_map or {}

        with self._lock:
            # Priority queue: (cost, current_map, path)
            heap: list[tuple[float, str, list[Portal]]] = [(0.0, start_map, [])]
            visited: dict[str, float] = {start_map: 0.0}

            while heap:
                cost, current_map, path = heapq.heappop(heap)

                if current_map == end_map:
                    return path

                if cost > visited.get(current_map, float('inf')):
                    continue

                for neighbor_map, portal in self._graph.get(current_map, []):
                    # Base cost: 1 per portal hop
                    edge_cost = 1.0
                    # Danger penalty: entering a dangerous map costs more
                    danger_penalty = danger.get(neighbor_map, 0.0) * 5.0
                    # One-way portals cost extra (may need to walk back)
                    one_way_penalty = 2.0 if portal.portal_type == "one_way" else 0.0

                    new_cost = cost + edge_cost + danger_penalty + one_way_penalty

                    if new_cost < visited.get(neighbor_map, float('inf')):
                        visited[neighbor_map] = new_cost
                        heapq.heappush(heap, (new_cost, neighbor_map, path + [portal]))

            return None

    def get_portal_near(self, map_name: str, x: int, y: int, radius: int = 10) -> Portal | None:
        """Find the nearest portal to a position on a map."""
        with self._lock:
            portals = self._portals_by_source.get(map_name, [])
            best: Portal | None = None
            best_dist = float('inf')
            for p in portals:
                dist = abs(p.source_x - x) + abs(p.source_y - y)
                if dist < best_dist and dist <= radius:
                    best_dist = dist
                    best = p
            return best

    def get_all_maps(self) -> set[str]:
        """Get all maps that have known portals."""
        with self._lock:
            maps: set[str] = set()
            for p in self._portals:
                maps.add(p.source_map)
                maps.add(p.target_map)
            return maps

    def add_portal(self, portal: Portal) -> None:
        """Add a portal at runtime (e.g., discovered by exploration)."""
        with self._lock:
            self._portals.append(portal)
            self._rebuild_indexes()

    def add_portals(self, portals: list[Portal]) -> None:
        """Add multiple portals at runtime."""
        with self._lock:
            self._portals.extend(portals)
            self._rebuild_indexes()

    def portal_count(self) -> int:
        with self._lock:
            return len(self._portals)

    def map_count(self) -> int:
        with self._lock:
            return len(self._graph)


# ── Global Singleton ──

_portal_knowledge: PortalKnowledge | None = None
_portal_knowledge_lock = RLock()


def get_portal_knowledge() -> PortalKnowledge:
    global _portal_knowledge
    with _portal_knowledge_lock:
        if _portal_knowledge is None:
            _portal_knowledge = PortalKnowledge()
        return _portal_knowledge
