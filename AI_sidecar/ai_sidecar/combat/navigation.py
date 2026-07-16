"""
Map routing / navigation module — map connection graph, BFS pathfinder,
and bridge-executable command builder for multi-map navigation.

Enables auto-restock (go to town → buy pots → return) and MvP hunting
(go to MvP map) by routing the bot through the RO map topology.

Uses the bridge commands:
  - Commands::run("move prontera 156 129")  – within-map movement
  - Commands::run("warp prt_fild01")         – map change
  - Commands::run("go save")                 – return to save point
  - Commands::run("tele")                    – random teleport
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)


# ── Pre-renewal (classic) RO map adjacency graph ──────────────────────────
# Each entry: map_name -> list of directly-connected maps (fields, dungeons).
# Bidirectional: if A lists B, B should also list A.

MAP_CONNECTIONS: dict[str, list[str]] = {
    # ═════════════════════════════════════════════════════════════════════
    #  TOWNS
    # ═════════════════════════════════════════════════════════════════════
    "prontera": [
        "prt_fild01", "prt_fild02", "prt_fild03", "prt_fild04",
        "prt_fild05", "prt_fild06", "prt_fild07", "prt_fild08",
        "prt_fild09", "prt_fild10", "prt_fild11", "prt_in",
    ],
    "morocc": [
        "moc_fild01", "moc_fild02", "moc_fild03", "moc_fild04",
        "moc_fild05", "moc_fild06", "moc_fild07", "moc_fild08",
        "moc_fild09", "moc_pryd01", "in_moc_161",
    ],
    "geffen": [
        "gef_fild01", "gef_fild02", "gef_fild03", "gef_fild04",
        "gef_fild05", "gef_fild06", "gef_fild07",
        "gef_tower01", "gef_dun01",
    ],
    "payon": [
        "pay_fild01", "pay_fild02", "pay_fild03", "pay_fild04",
        "pay_fild05", "pay_fild06", "pay_fild07", "pay_fild08",
        "pay_fild09", "pay_fild10", "pay_fild11", "pay_dun00",
    ],
    "aldebaran": [
        "alde_fild01", "alde_fild02", "alde_fild03", "alde_fild04",
        "alde_dun01",
    ],
    "izlude": [
        "iz_fild01", "iz_fild02", "iz_fild03", "iz_fild04",
        "izlude_in", "iz_dun00",
    ],
    "alberta": [
        "iz_fild03", "alberta_in",
    ],
    "comodo": [
        "cmd_fild01", "cmd_fild02", "cmd_fild03", "cmd_fild04",
        "cmd_fild05", "cmd_fild06", "cmd_fild07", "cmd_fild08",
    ],
    "yuno": [
        "yuno_fild01", "yuno_fild02",
    ],
    "xmas": [
        "xmas_fild01", "xmas_dun01", "xmas_in",
    ],
    "einbroch": [
        "ein_fild01", "ein_fild02", "ein_fild03", "ein_fild04",
        "ein_fild05", "ein_fild06", "ein_fild07", "ein_fild08",
        "ein_fild09", "ein_fild10", "ein_dun01",
    ],
    "lighthalzen": [
        "lhz_fild01", "lhz_fild02", "lhz_fild03",
        "lhz_dun01",
    ],
    "hugel": [
        "hu_fild01", "hu_fild02", "hu_fild03", "hu_fild04",
        "hu_fild05", "hu_fild06", "hu_fild07", "hu_dun01",
    ],
    "rachel": [
        "ra_fild01", "ra_fild02", "ra_fild03", "ra_fild04",
        "ra_fild05", "ra_fild06", "ra_fild07", "ra_fild08",
        "ra_fild09", "ra_dun01",
    ],
    "gonryun": [
        "gon_fild01", "gon_dun01",
    ],
    "amatsu": [
        "ama_fild01", "ama_dun01",
    ],
    "ayothaya": [
        "ayo_fild01", "ayo_dun01",
    ],
    "louyang": [
        "lou_fild01", "lou_dun01",
    ],
    "umbala": [
        "um_fild01", "um_fild02", "um_dun01",
    ],
    "niflheim": [
        "nif_fild01", "nif_dun01",
    ],

    # ═════════════════════════════════════════════════════════════════════
    #  PRONTERA FIELDS  (ringed around Prontera)
    # ═════════════════════════════════════════════════════════════════════
    "prt_fild01":  ["prontera", "prt_fild11", "prt_sewb1", "prt_maze01"],
    "prt_fild02":  ["prontera", "prt_fild03", "gef_fild06"],
    "prt_fild03":  ["prontera", "prt_fild02", "prt_fild04"],
    "prt_fild04":  ["prontera", "prt_fild03", "prt_fild05"],
    "prt_fild05":  ["prontera", "prt_fild04", "prt_fild06", "moc_fild01"],
    "prt_fild06":  ["prontera", "prt_fild05", "prt_fild07"],
    "prt_fild07":  ["prontera", "prt_fild06", "iz_fild01"],
    "prt_fild08":  ["prontera", "prt_fild09", "prt_fild11"],
    "prt_fild09":  ["prontera", "prt_fild08", "prt_fild10"],
    "prt_fild10":  ["prontera", "prt_fild09", "prt_fild11"],
    "prt_fild11":  ["prontera", "prt_fild01", "prt_fild08", "prt_fild10"],

    # Prontera indoor
    "prt_in": ["prontera"],

    # ── Prontera Sewers ──
    "prt_sewb1": ["prt_fild01", "prt_sewb2"],
    "prt_sewb2": ["prt_sewb1", "prt_sewb3"],
    "prt_sewb3": ["prt_sewb2", "prt_sewb4"],
    "prt_sewb4": ["prt_sewb3"],

    # ── Prontera Maze ──
    "prt_maze01": ["prt_fild01", "prt_maze02", "prt_maze03"],
    "prt_maze02": ["prt_maze01"],
    "prt_maze03": ["prt_maze01"],

    # ═════════════════════════════════════════════════════════════════════
    #  MOROCC DESERT FIELDS
    # ═════════════════════════════════════════════════════════════════════
    "moc_fild01": ["morocc", "moc_fild02", "moc_fild07", "prt_fild05"],
    "moc_fild02": ["morocc", "moc_fild01", "moc_fild03"],
    "moc_fild03": ["morocc", "moc_fild02", "gef_fild04"],
    "moc_fild04": ["morocc", "moc_fild05"],
    "moc_fild05": ["morocc", "moc_fild04", "moc_fild06"],
    "moc_fild06": ["morocc", "moc_fild05"],
    "moc_fild07": ["morocc", "moc_fild01", "moc_fild08"],
    "moc_fild08": ["morocc", "moc_fild07", "moc_fild09"],
    "moc_fild09": ["morocc", "moc_fild08", "gef_fild05"],
    "in_moc_161": ["morocc"],

    # ── Pyramids ──
    "moc_pryd01": ["morocc", "moc_pryd02", "moc_pryd03", "moc_pryd04"],
    "moc_pryd02": ["moc_pryd01"],
    "moc_pryd03": ["moc_pryd01"],
    "moc_pryd04": ["moc_pryd01", "moc_pryd05"],
    "moc_pryd05": ["moc_pryd04", "moc_pryd06"],
    "moc_pryd06": ["moc_pryd05"],

    # ═════════════════════════════════════════════════════════════════════
    #  GEFFEN FIELDS
    # ═════════════════════════════════════════════════════════════════════
    "gef_fild01": ["geffen", "gef_fild02", "alde_fild01"],
    "gef_fild02": ["geffen", "gef_fild01", "gef_fild03"],
    "gef_fild03": ["geffen", "gef_fild02", "gef_fild04", "pay_fild03"],
    "gef_fild04": ["geffen", "gef_fild03", "gef_fild05", "moc_fild03"],
    "gef_fild05": ["geffen", "gef_fild04", "gef_fild06", "moc_fild09"],
    "gef_fild06": ["geffen", "gef_fild05", "gef_fild07", "prt_fild02"],
    "gef_fild07": ["geffen", "gef_fild06"],

    # ── Geffen Tower ──
    "gef_tower01": ["geffen", "gef_tower02"],
    "gef_tower02": ["gef_tower01", "gef_tower03"],
    "gef_tower03": ["gef_tower02", "gef_tower04"],
    "gef_tower04": ["gef_tower03"],

    # ── Geffen Dungeon ──
    "gef_dun01": ["geffen", "gef_dun02"],
    "gef_dun02": ["gef_dun01", "gef_dun03"],
    "gef_dun03": ["gef_dun02"],

    # ═════════════════════════════════════════════════════════════════════
    #  PAYON FIELDS
    # ═════════════════════════════════════════════════════════════════════
    "pay_fild01": ["payon", "pay_fild02", "pay_fild03"],
    "pay_fild02": ["payon", "pay_fild01"],
    "pay_fild03": ["payon", "pay_fild01", "gef_fild03"],
    "pay_fild04": ["payon"],
    "pay_fild05": ["payon"],
    "pay_fild06": ["payon"],
    "pay_fild07": ["payon"],
    "pay_fild08": ["payon"],
    "pay_fild09": ["payon"],
    "pay_fild10": ["payon"],
    "pay_fild11": ["payon"],

    # ── Payon Dungeon (Culvert / Cave) ──
    "pay_dun00": ["payon", "pay_dun01"],
    "pay_dun01": ["pay_dun00", "pay_dun02"],
    "pay_dun02": ["pay_dun01", "pay_dun03"],
    "pay_dun03": ["pay_dun02", "pay_dun04"],
    "pay_dun04": ["pay_dun03"],

    # ═════════════════════════════════════════════════════════════════════
    #  ALDEBARAN FIELDS
    # ═════════════════════════════════════════════════════════════════════
    "alde_fild01": ["aldebaran", "alde_fild02", "gef_fild01"],
    "alde_fild02": ["aldebaran", "alde_fild01", "alde_fild03"],
    "alde_fild03": ["aldebaran", "alde_fild02", "alde_fild04"],
    "alde_fild04": ["aldebaran", "alde_fild03"],

    # ── Clock Tower ──
    "alde_dun01": ["aldebaran", "alde_dun02"],
    "alde_dun02": ["alde_dun01", "alde_dun03"],
    "alde_dun03": ["alde_dun02", "alde_dun04"],
    "alde_dun04": ["alde_dun03"],

    # ═════════════════════════════════════════════════════════════════════
    #  IZLUDE FIELDS
    # ═════════════════════════════════════════════════════════════════════
    "iz_fild01": ["izlude", "prt_fild07"],
    "iz_fild02": ["izlude", "iz_fild03", "iz_fild04"],
    "iz_fild03": ["izlude", "iz_fild02", "alberta"],
    "iz_fild04": ["izlude", "iz_fild02", "iz_fild05"],
    "iz_fild05": ["iz_fild04"],
    "izlude_in": ["izlude"],

    # ── Byalan Dungeon ──
    "iz_dun00": ["izlude", "iz_dun01"],
    "iz_dun01": ["iz_dun00", "iz_dun02"],
    "iz_dun02": ["iz_dun01", "iz_dun03"],
    "iz_dun03": ["iz_dun02", "iz_dun04"],
    "iz_dun04": ["iz_dun03"],

    # ── Alberta ──
    "alberta_in": ["alberta"],

    # ═════════════════════════════════════════════════════════════════════
    #  COMODO FIELDS
    # ═════════════════════════════════════════════════════════════════════
    "cmd_fild01": ["comodo", "cmd_fild02", "cmd_fild04"],
    "cmd_fild02": ["comodo", "cmd_fild01", "cmd_fild03"],
    "cmd_fild03": ["comodo", "cmd_fild02", "cmd_fild06"],
    "cmd_fild04": ["comodo", "cmd_fild01", "cmd_fild05"],
    "cmd_fild05": ["comodo", "cmd_fild04"],
    "cmd_fild06": ["comodo", "cmd_fild03", "cmd_fild07"],
    "cmd_fild07": ["comodo", "cmd_fild06", "cmd_fild08"],
    "cmd_fild08": ["comodo", "cmd_fild07"],

    # ═════════════════════════════════════════════════════════════════════
    #  YUNO FIELDS
    # ═════════════════════════════════════════════════════════════════════
    "yuno_fild01": ["yuno"],
    "yuno_fild02": ["yuno"],

    # ═════════════════════════════════════════════════════════════════════
    #  XMAS / LUTIE
    # ═════════════════════════════════════════════════════════════════════
    "xmas_fild01": ["xmas"],
    "xmas_dun01":  ["xmas", "xmas_dun02"],
    "xmas_dun02":  ["xmas_dun01"],
    "xmas_in":     ["xmas"],

    # ═════════════════════════════════════════════════════════════════════
    #  EINBROCH FIELDS
    # ═════════════════════════════════════════════════════════════════════
    "ein_fild01": ["einbroch"],
    "ein_fild02": ["einbroch"],
    "ein_fild03": ["einbroch"],
    "ein_fild04": ["einbroch", "ein_fild05"],
    "ein_fild05": ["einbroch", "ein_fild04", "ein_fild06"],
    "ein_fild06": ["einbroch", "ein_fild05", "ein_fild07"],
    "ein_fild07": ["einbroch", "ein_fild06", "ein_fild08"],
    "ein_fild08": ["einbroch", "ein_fild07", "ein_fild09"],
    "ein_fild09": ["einbroch", "ein_fild08", "ein_fild10"],
    "ein_fild10": ["einbroch", "ein_fild09"],
    "ein_dun01":  ["einbroch", "ein_dun02"],
    "ein_dun02":  ["ein_dun01"],

    # ═════════════════════════════════════════════════════════════════════
    #  LIGHTHALZEN
    # ═════════════════════════════════════════════════════════════════════
    "lhz_fild01": ["lighthalzen"],
    "lhz_fild02": ["lighthalzen"],
    "lhz_fild03": ["lighthalzen"],
    "lhz_dun01":  ["lighthalzen", "lhz_dun02"],
    "lhz_dun02":  ["lhz_dun01", "lhz_dun03"],
    "lhz_dun03":  ["lhz_dun02"],

    # ═════════════════════════════════════════════════════════════════════
    #  HUGEL FIELDS
    # ═════════════════════════════════════════════════════════════════════
    "hu_fild01": ["hugel"],
    "hu_fild02": ["hugel", "hu_fild03"],
    "hu_fild03": ["hugel", "hu_fild02", "hu_fild04"],
    "hu_fild04": ["hugel", "hu_fild03", "hu_fild05"],
    "hu_fild05": ["hugel", "hu_fild04", "hu_fild06"],
    "hu_fild06": ["hugel", "hu_fild05", "hu_fild07"],
    "hu_fild07": ["hugel", "hu_fild06"],
    "hu_dun01":  ["hugel", "hu_dun02"],
    "hu_dun02":  ["hu_dun01"],

    # ═════════════════════════════════════════════════════════════════════
    #  RACHEL FIELDS
    # ═════════════════════════════════════════════════════════════════════
    "ra_fild01": ["rachel"],
    "ra_fild02": ["rachel", "ra_fild03"],
    "ra_fild03": ["rachel", "ra_fild02", "ra_fild04"],
    "ra_fild04": ["rachel", "ra_fild03", "ra_fild05"],
    "ra_fild05": ["rachel", "ra_fild04", "ra_fild06"],
    "ra_fild06": ["rachel", "ra_fild05", "ra_fild07"],
    "ra_fild07": ["rachel", "ra_fild06", "ra_fild08"],
    "ra_fild08": ["rachel", "ra_fild07", "ra_fild09"],
    "ra_fild09": ["rachel", "ra_fild08"],
    "ra_dun01":  ["rachel", "ra_dun02"],
    "ra_dun02":  ["ra_dun01"],

    # ═════════════════════════════════════════════════════════════════════
    #  GONRYUN (Kuno)
    # ═════════════════════════════════════════════════════════════════════
    "gon_fild01": ["gonryun"],
    "gon_dun01":  ["gonryun", "gon_dun02"],
    "gon_dun02":  ["gon_dun01", "gon_dun03"],
    "gon_dun03":  ["gon_dun02"],

    # ═════════════════════════════════════════════════════════════════════
    #  AMATSU
    # ═════════════════════════════════════════════════════════════════════
    "ama_fild01": ["amatsu"],
    "ama_dun01":  ["amatsu", "ama_dun02"],
    "ama_dun02":  ["ama_dun01"],

    # ═════════════════════════════════════════════════════════════════════
    #  AYOTHAYA
    # ═════════════════════════════════════════════════════════════════════
    "ayo_fild01": ["ayothaya"],
    "ayo_dun01":  ["ayothaya", "ayo_dun02"],
    "ayo_dun02":  ["ayo_dun01"],

    # ═════════════════════════════════════════════════════════════════════
    #  LOUYANG
    # ═════════════════════════════════════════════════════════════════════
    "lou_fild01": ["louyang"],
    "lou_dun01":  ["louyang", "lou_dun02"],
    "lou_dun02":  ["lou_dun01", "lou_dun03"],
    "lou_dun03":  ["lou_dun02"],

    # ═════════════════════════════════════════════════════════════════════
    #  UMBALA
    # ═════════════════════════════════════════════════════════════════════
    "um_fild01": ["umbala"],
    "um_fild02": ["umbala"],
    "um_dun01":  ["umbala", "um_dun02"],
    "um_dun02":  ["um_dun01"],

    # ═════════════════════════════════════════════════════════════════════
    #  NIFLHEIM
    # ═════════════════════════════════════════════════════════════════════
    "nif_fild01": ["niflheim"],
    "nif_dun01":  ["niflheim", "nif_dun02"],
    "nif_dun02":  ["nif_dun01"],
}


# ── Router ────────────────────────────────────────────────────────────────

class Router:
    """BFS-based map-to-map route planner for RO map topology.

    Usage::

        router = Router()
        path = router.find_path("prontera", "payon")
        next_map = router.get_next_map("prontera", "payon")
        commands = router.get_navigate_commands("prontera", "payon")
    """

    def __init__(self, connections: dict[str, list[str]] | None = None) -> None:
        self._connections = connections or MAP_CONNECTIONS
        self._route_cache: dict[tuple[str, str], list[str]] = {}

    # ── public API ────────────────────────────────────────────────────────

    def find_path(self, from_map: str, to_map: str) -> list[str]:
        """BFS shortest path between two maps.

        Returns ordered list of map names (including start and end).
        Returns empty list if no path exists.
        """
        from_map = from_map.lower()
        to_map = to_map.lower()

        if from_map == to_map:
            return [from_map]

        cache_key = (from_map, to_map)
        if cache_key in self._route_cache:
            return self._route_cache[cache_key]

        visited: set[str] = {from_map}
        queue: deque[tuple[str, list[str]]] = deque()
        queue.append((from_map, [from_map]))

        while queue:
            current, path = queue.popleft()
            for neighbor in self._connections.get(current, []):
                if neighbor == to_map:
                    result = path + [neighbor]
                    self._route_cache[cache_key] = result
                    return result
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return []

    def get_next_map(self, current_map: str, target_map: str) -> str | None:
        """Return the *first* map to warp to when navigating.

        Returns None if already at target or unreachable.
        """
        path = self.find_path(current_map, target_map)
        if len(path) < 2:
            return None
        return path[1]

    def get_navigate_commands(
        self,
        current_map: str,
        target_map: str,
        *,
        town_safe_coords: tuple[int, int] | None = None,
        use_teleport: bool = False,
    ) -> list[dict[str, Any]]:
        """Build a list of bridge-executable action commands for navigation.

        Parameters
        ----------
        current_map, target_map :
            Source and destination map names (case-insensitive).
        town_safe_coords :
            (x, y) safe spot to move to on arrival.  Falls back to
            per-town defaults if omitted.
        use_teleport :
            If True, emit a ``tele`` command at the start (random teleport
            to break line-of-sight / gather mobs).

        Returns
        -------
        List of command dicts consumable by the bridge::

            {"kind": "navigate", "command": "warp prt_fild01", "map": "prt_fild01"}
        """
        if current_map.lower() == target_map.lower():
            return [self._make_cmd("arrived", target_map, "go save")]

        path = self.find_path(current_map, target_map)
        if not path:
            logger.warning("nav_unreachable: %s -> %s", current_map, target_map)
            # Unreachable — fall back to go save + tele
            return [
                self._make_cmd("retreat", current_map, "go save"),
                self._make_cmd("emergency", current_map, "tele"),
            ]

        commands: list[dict[str, Any]] = []

        if use_teleport:
            commands.append(self._make_cmd("teleport", current_map, "tele"))

        # If still on the starting map, issue a move to the town centre
        # so the bot stands near the warp point.
        if path[0] == current_map.lower():
            tx, ty = town_safe_coords or self._default_coords(path[0])
            commands.append(
                self._make_cmd("move", path[0], f"move {path[0]} {tx} {ty}")
            )

        # Generate warp commands for each hop (skip index 0 = start)
        for i, hop in enumerate(path):
            if i == 0:
                continue
            commands.append(self._make_cmd("warp", hop, f"warp {hop}"))
            # Move to town centre on arrival if it's a town map
            if self._is_town(hop):
                tx, ty = town_safe_coords or self._default_coords(hop)
                commands.append(
                    self._make_cmd("move", hop, f"move {hop} {tx} {ty}")
                )

        return commands

    # ── internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _make_cmd(action: str, map_name: str, command: str) -> dict[str, Any]:
        return {
            "kind": "navigate",
            "action": action,
            "map": map_name,
            "command": command,
        }

    @staticmethod
    def _is_town(map_name: str) -> bool:
        """Heuristic: most town maps don't start with a prefix like prt_, gef_, etc."""
        known_towns = {
            "prontera", "morocc", "geffen", "payon", "aldebaran",
            "izlude", "alberta", "comodo", "yuno", "xmas",
            "einbroch", "lighthalzen", "hugel", "rachel",
            "gonryun", "amatsu", "ayothaya", "louyang", "umbala",
            "niflheim",
        }
        return map_name in known_towns

    @staticmethod
    def _default_coords(map_name: str) -> tuple[int, int]:
        """Return safe town-centre coordinates for known towns."""
        coords: dict[str, tuple[int, int]] = {
            "prontera": (156, 191),
            "morocc": (150, 100),
            "geffen": (120, 85),
            "payon": (210, 120),
            "aldebaran": (140, 130),
            "izlude": (109, 132),
            "alberta": (60, 240),
            "comodo": (210, 150),
            "yuno": (180, 150),
            "xmas": (194, 139),
            "einbroch": (200, 180),
            "lighthalzen": (150, 120),
            "hugel": (100, 80),
            "rachel": (130, 100),
            "gonryun": (150, 100),
            "amatsu": (150, 100),
            "ayothaya": (150, 100),
            "louyang": (150, 100),
            "umbala": (150, 100),
            "niflheim": (150, 100),
        }
        return coords.get(map_name, (150, 150))

    def clear_cache(self) -> None:
        self._route_cache.clear()
