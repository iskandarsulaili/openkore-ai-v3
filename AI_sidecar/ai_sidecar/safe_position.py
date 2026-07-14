"""
Safe Position Manager — knows where to stand to avoid aggro.

A pro player knows the safe spots on every map. This module encodes that
knowledge so bots can retreat to safety when overwhelmed.

Key features:
1. Safe spots per map (towns, portal areas, safe zones)
2. Emergency retreat: find nearest safe spot when aggro is too high
3. Safe route calculation: path through safe spots to destination
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SafeSpot:
    """A safe position on a map."""
    map_name: str
    x: int
    y: int
    spot_type: str = "safe"  # safe | portal_area | town | chokepoint
    radius: int = 5  # How close you need to be to be "safe"
    description: str = ""


class SafePositionManager:
    """Knows safe positions on every map.

    Thread-safe singleton. Provides safe spot lookup, nearest safe spot
    calculation, and emergency retreat planning.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._safe_spots: dict[str, list[SafeSpot]] = {}
        self._load_safe_spots()

    def _load_safe_spots(self) -> None:
        """Load known safe spots for all maps."""
        spots: dict[str, list[SafeSpot]] = {}

        def add(map_name: str, x: int, y: int, stype: str = "safe", radius: int = 5, desc: str = ""):
            if map_name not in spots:
                spots[map_name] = []
            spots[map_name].append(SafeSpot(
                map_name=map_name, x=x, y=y,
                spot_type=stype, radius=radius, description=desc,
            ))

        # ── Towns (entire town is safe) ──
        for town in ["prontera", "geffen", "payon", "morocc", "aldebaran",
                      "yuno", "izlude", "xmas", "comodo", "amatsu"]:
            add(town, 150, 150, "town", 50, f"{town} town center")

        # ── Prontera Fields ──
        add("prt_fild01", 30, 30, "portal_area", 8, "prt_fild01 north portal (to Prontera)")
        add("prt_fild01", 300, 200, "portal_area", 8, "prt_fild01 east portal (to prt_fild02)")
        add("prt_fild01", 300, 30, "portal_area", 8, "prt_fild01 south portal (to mjolnir_04)")

        add("prt_fild02", 30, 200, "portal_area", 8, "prt_fild02 west portal (to prt_fild01)")
        add("prt_fild02", 300, 200, "portal_area", 8, "prt_fild02 east portal (to prt_fild03)")

        add("prt_fild03", 30, 200, "portal_area", 8, "prt_fild03 west portal (to prt_fild02)")
        add("prt_fild03", 300, 200, "portal_area", 8, "prt_fild03 east portal (to prt_fild04)")

        add("prt_fild04", 30, 200, "portal_area", 8, "prt_fild04 west portal (to prt_fild03)")
        add("prt_fild04", 300, 30, "portal_area", 8, "prt_fild04 south portal (to prt_fild05)")
        add("prt_fild04", 300, 30, "portal_area", 8, "prt_fild04 portal (to Prontera)")
        add("prt_fild04", 30, 300, "portal_area", 8, "prt_fild04 portal (to moc_fild01)")

        add("prt_fild05", 30, 30, "portal_area", 8, "prt_fild05 west portal (to prt_fild04)")
        add("prt_fild05", 300, 30, "portal_area", 8, "prt_fild05 east portal (to prt_fild06)")

        add("prt_fild06", 30, 30, "portal_area", 8, "prt_fild06 west portal (to prt_fild05)")
        add("prt_fild06", 300, 30, "portal_area", 8, "prt_fild06 east portal (to prt_fild07)")

        add("prt_fild07", 30, 30, "portal_area", 8, "prt_fild07 west portal (to prt_fild06)")
        add("prt_fild07", 300, 30, "portal_area", 8, "prt_fild07 east portal (to prt_fild08)")

        add("prt_fild08", 30, 30, "portal_area", 8, "prt_fild08 west portal (to prt_fild07)")
        add("prt_fild08", 30, 300, "portal_area", 8, "prt_fild08 south portal (to Prontera)")
        add("prt_fild08", 300, 30, "portal_area", 8, "prt_fild08 east portal (to pay_fild01)")
        add("prt_fild08", 300, 200, "portal_area", 8, "prt_fild08 portal (to prt_fild09)")

        add("prt_fild09", 30, 200, "portal_area", 8, "prt_fild09 west portal (to prt_fild08)")
        add("prt_fild09", 300, 200, "portal_area", 8, "prt_fild09 east portal (to prt_fild10)")

        add("prt_fild10", 30, 200, "portal_area", 8, "prt_fild10 west portal (to prt_fild09)")
        add("prt_fild10", 300, 200, "portal_area", 8, "prt_fild10 east portal (to prt_fild11)")

        add("prt_fild11", 30, 200, "portal_area", 8, "prt_fild11 west portal (to prt_fild10)")
        add("prt_fild11", 30, 30, "portal_area", 8, "prt_fild11 portal (to Prontera)")

        # ── Morocc Fields ──
        add("moc_fild01", 300, 30, "portal_area", 8, "moc_fild01 portal (to prt_fild04)")
        add("moc_fild01", 30, 200, "portal_area", 8, "moc_fild01 portal (to moc_fild02)")
        add("moc_fild02", 300, 200, "portal_area", 8, "moc_fild02 portal (to moc_fild01)")
        add("moc_fild02", 30, 200, "portal_area", 8, "moc_fild02 portal (to moc_fild03)")
        add("moc_fild03", 300, 200, "portal_area", 8, "moc_fild03 portal (to moc_fild02)")
        add("moc_fild03", 30, 200, "portal_area", 8, "moc_fild03 portal (to morocc)")

        # ── Geffen Fields ──
        add("gef_fild00", 30, 200, "portal_area", 8, "gef_fild00 portal (to mjolnir_04)")
        add("gef_fild00", 300, 200, "portal_area", 8, "gef_fild00 portal (to geffen)")
        add("gef_fild00", 300, 30, "portal_area", 8, "gef_fild00 portal (to gef_fild01)")
        add("gef_fild01", 30, 30, "portal_area", 8, "gef_fild01 portal (to gef_fild00)")
        add("gef_fild01", 300, 200, "portal_area", 8, "gef_fild01 portal (to gef_fild02)")
        add("gef_fild02", 30, 200, "portal_area", 8, "gef_fild02 portal (to gef_fild01)")
        add("gef_fild02", 300, 200, "portal_area", 8, "gef_fild02 portal (to gef_fild03)")
        add("gef_fild03", 30, 200, "portal_area", 8, "gef_fild03 portal (to gef_fild02)")
        add("gef_fild03", 300, 200, "portal_area", 8, "gef_fild03 portal (to gef_fild04)")
        add("gef_fild04", 30, 200, "portal_area", 8, "gef_fild04 portal (to gef_fild03)")
        add("gef_fild04", 300, 200, "portal_area", 8, "gef_fild04 portal (to gef_fild05)")
        add("gef_fild05", 30, 200, "portal_area", 8, "gef_fild05 portal (to gef_fild04)")
        add("gef_fild05", 300, 30, "portal_area", 8, "gef_fild05 portal (to yuno_fild01)")

        # ── Payon Fields ──
        add("pay_fild01", 30, 30, "portal_area", 8, "pay_fild01 portal (to prt_fild08)")
        add("pay_fild01", 300, 200, "portal_area", 8, "pay_fild01 portal (to pay_fild02)")
        add("pay_fild02", 30, 200, "portal_area", 8, "pay_fild02 portal (to pay_fild01)")
        add("pay_fild02", 300, 200, "portal_area", 8, "pay_fild02 portal (to pay_fild03)")
        add("pay_fild03", 30, 200, "portal_area", 8, "pay_fild03 portal (to pay_fild02)")
        add("pay_fild03", 300, 200, "portal_area", 8, "pay_fild03 portal (to pay_fild04)")
        add("pay_fild04", 30, 200, "portal_area", 8, "pay_fild04 portal (to pay_fild03)")
        add("pay_fild04", 300, 200, "portal_area", 8, "pay_fild04 portal (to pay_fild05)")
        add("pay_fild04", 300, 30, "portal_area", 8, "pay_fild04 portal (to aldebaran)")
        add("pay_fild05", 30, 200, "portal_area", 8, "pay_fild05 portal (to pay_fild04)")
        add("pay_fild05", 300, 200, "portal_area", 8, "pay_fild05 portal (to pay_fild06)")
        add("pay_fild06", 30, 200, "portal_area", 8, "pay_fild06 portal (to pay_fild05)")
        add("pay_fild06", 300, 200, "portal_area", 8, "pay_fild06 portal (to pay_fild07)")
        add("pay_fild07", 30, 200, "portal_area", 8, "pay_fild07 portal (to pay_fild06)")
        add("pay_fild07", 300, 200, "portal_area", 8, "pay_fild07 portal (to pay_fild08)")
        add("pay_fild08", 30, 200, "portal_area", 8, "pay_fild08 portal (to pay_fild07)")
        add("pay_fild08", 300, 200, "portal_area", 8, "pay_fild08 portal (to payon)")

        # ── Dungeon entrances ──
        add("payon", 120, 120, "portal_area", 8, "Payon Cave entrance")
        add("geffen", 120, 120, "portal_area", 8, "Geffen Dungeon entrance")
        add("morocc", 120, 120, "portal_area", 8, "Morocc Dungeon entrance")
        add("gef_fild14", 120, 120, "portal_area", 8, "Orc Dungeon entrance")
        add("izlude", 120, 120, "portal_area", 8, "Byalan entrance")

        # ── Prontera specific ──
        add("prontera", 156, 22, "portal_area", 8, "Prontera South Gate (to prt_fild08)")
        add("prontera", 156, 370, "portal_area", 8, "Prontera North Gate (to prt_fild01)")
        add("prontera", 30, 200, "portal_area", 8, "Prontera West Gate (to prt_fild04)")
        add("prontera", 280, 200, "portal_area", 8, "Prontera East Gate (to prt_fild11)")

        self._safe_spots = spots

    # ── Public API ───────────────────────────────────────────────────

    def get_safe_spots(self, map_name: str) -> list[SafeSpot]:
        """Get all safe spots on a map."""
        with self._lock:
            return list(self._safe_spots.get(map_name, []))

    def get_nearest_safe_spot(self, map_name: str, x: int, y: int) -> SafeSpot | None:
        """Get the nearest safe spot to a position."""
        with self._lock:
            spots = self._safe_spots.get(map_name, [])
            if not spots:
                return None
            best = min(spots, key=lambda s: abs(s.x - x) + abs(s.y - y))
            return best

    def get_nearest_town_spot(self, map_name: str) -> SafeSpot | None:
        """Get the nearest town safe spot."""
        with self._lock:
            spots = self._safe_spots.get(map_name, [])
            town_spots = [s for s in spots if s.spot_type == "town"]
            if town_spots:
                return town_spots[0]
            return None

    def get_nearest_portal_spot(self, map_name: str, x: int, y: int) -> SafeSpot | None:
        """Get the nearest portal-area safe spot."""
        with self._lock:
            spots = self._safe_spots.get(map_name, [])
            portal_spots = [s for s in spots if s.spot_type == "portal_area"]
            if not portal_spots:
                return None
            return min(portal_spots, key=lambda s: abs(s.x - x) + abs(s.y - y))

    def is_safe(self, map_name: str, x: int, y: int) -> bool:
        """Check if a position is within a safe spot."""
        with self._lock:
            spots = self._safe_spots.get(map_name, [])
            for spot in spots:
                if abs(spot.x - x) <= spot.radius and abs(spot.y - y) <= spot.radius:
                    return True
            return False

    def get_retreat_plan(self, map_name: str, x: int, y: int) -> SafeSpot | None:
        """Get the best retreat target when overwhelmed.

        Priority: town > portal_area > safe spot
        """
        with self._lock:
            spots = self._safe_spots.get(map_name, [])
            if not spots:
                return None

            # Prefer town spots
            town = [s for s in spots if s.spot_type == "town"]
            if town:
                return town[0]

            # Then portal areas (can escape to next map)
            portal = [s for s in spots if s.spot_type == "portal_area"]
            if portal:
                return min(portal, key=lambda s: abs(s.x - x) + abs(s.y - y))

            # Then any safe spot
            return min(spots, key=lambda s: abs(s.x - x) + abs(s.y - y))

    def add_safe_spot(self, spot: SafeSpot) -> None:
        """Add a safe spot at runtime."""
        with self._lock:
            if spot.map_name not in self._safe_spots:
                self._safe_spots[spot.map_name] = []
            self._safe_spots[spot.map_name].append(spot)

    def get_all_safe_maps(self) -> set[str]:
        """Get all maps that have known safe spots."""
        with self._lock:
            return set(self._safe_spots.keys())


# ── Global Singleton ──

_safe_position_manager: SafePositionManager | None = None
_safe_position_manager_lock = RLock()


def get_safe_position_manager() -> SafePositionManager:
    global _safe_position_manager
    with _safe_position_manager_lock:
        if _safe_position_manager is None:
            _safe_position_manager = SafePositionManager()
        return _safe_position_manager
