"""
Kafra Teleportation — Kafra warp service, save point management, and return.

In Ragnarok Online, Kafra NPCs provide:
1. Warp service: Pay zeny to teleport to major cities
2. Save point: Set your respawn location
3. Butterfly Wing return: Return to your save point

This module handles:
- Kafra warp service (pay zeny to teleport between cities)
- Save point management (set and query save points)
- Return to save point (via butterfly wing or death)
- Cost tracking for Kafra services
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class KafraLocation:
    """Location of a Kafra NPC on a map."""
    map_name: str
    x: int
    y: int
    npc_name: str = "Kafra Employee"


@dataclass(slots=True)
class SavePoint:
    """A player's save point (respawn location)."""
    map_name: str
    x: int
    y: int
    bot_id: str = ""


# Known Kafra locations in major towns
KNOWN_KAFRA_LOCATIONS: dict[str, tuple[int, int]] = {
    "prontera": (158, 180),
    "geffen": (120, 50),
    "morocc": (155, 88),
    "payon": (185, 190),
    "aldebaran": (126, 112),
    "alberta": (44, 119),
    "izlude": (108, 138),
    "yuno": (150, 150),
    "comodo": (100, 100),
    "xmas": (80, 80),
    "einbroch": (200, 180),
    "lighthalzen": (150, 120),
    "hugel": (100, 80),
    "rachel": (130, 100),
}

# Kafra warp destinations and costs (zeny)
# Format: from_city -> [(to_city, zeny_cost)]
KAFRA_WARP_ROUTES: dict[str, list[tuple[str, int]]] = {
    "prontera": [
        ("geffen", 200), ("morocc", 200), ("payon", 200),
        ("aldebaran", 300), ("alberta", 200), ("izlude", 200),
        ("yuno", 500), ("comodo", 400),
    ],
    "geffen": [
        ("prontera", 200), ("morocc", 300), ("payon", 300),
        ("aldebaran", 400), ("alberta", 300),
    ],
    "morocc": [
        ("prontera", 200), ("geffen", 300), ("payon", 300),
        ("aldebaran", 400), ("alberta", 300),
    ],
    "payon": [
        ("prontera", 200), ("geffen", 300), ("morocc", 300),
        ("aldebaran", 400), ("alberta", 300),
    ],
    "aldebaran": [
        ("prontera", 300), ("geffen", 400), ("morocc", 400),
        ("payon", 400), ("alberta", 200), ("izlude", 200),
        ("yuno", 500),
    ],
    "alberta": [
        ("prontera", 200), ("geffen", 300), ("morocc", 300),
        ("payon", 300), ("aldebaran", 200), ("izlude", 200),
    ],
    "izlude": [
        ("prontera", 200), ("aldebaran", 200), ("alberta", 200),
    ],
    "yuno": [
        ("prontera", 500), ("aldebaran", 500), ("einbroch", 300),
        ("lighthalzen", 300),
    ],
    "comodo": [
        ("prontera", 400), ("aldebaran", 300), ("alberta", 300),
    ],
}


class KafraTeleportManager:
    """Manages Kafra teleportation services.

    Thread-safe singleton. Handles warp service, save points,
    and return-to-save logic.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._save_points: dict[str, SavePoint] = {}  # bot_id -> SavePoint
        self._kafra_locations: dict[str, KafraLocation] = {}
        self._load_kafra_locations()

    def _load_kafra_locations(self) -> None:
        """Load known Kafra NPC locations."""
        for map_name, (x, y) in KNOWN_KAFRA_LOCATIONS.items():
            self._kafra_locations[map_name] = KafraLocation(
                map_name=map_name, x=x, y=y,
            )

    # ── Save Point Management ───────────────────────────────────────

    def set_save_point(self, bot_id: str, map_name: str, x: int, y: int) -> None:
        """Set a bot's save point (respawn location)."""
        with self._lock:
            self._save_points[bot_id] = SavePoint(
                map_name=map_name, x=x, y=y, bot_id=bot_id,
            )
            logger.info("kafra_save_point_set: bot=%s map=%s (%d,%d)",
                        bot_id, map_name, x, y)

    def get_save_point(self, bot_id: str) -> SavePoint | None:
        """Get a bot's save point."""
        with self._lock:
            return self._save_points.get(bot_id)

    def get_default_save_point(self) -> SavePoint:
        """Get the default save point (Prontera)."""
        return SavePoint(map_name="prontera", x=156, y=191)

    # ── Kafra Location Queries ──────────────────────────────────────

    def has_kafra(self, map_name: str) -> bool:
        """Check if a map has a Kafra NPC."""
        with self._lock:
            return map_name in self._kafra_locations

    def get_kafra_location(self, map_name: str) -> KafraLocation | None:
        """Get the Kafra NPC location on a map."""
        with self._lock:
            return self._kafra_locations.get(map_name)

    def get_kafra_coords(self, map_name: str) -> tuple[int, int] | None:
        """Get the coordinates of the Kafra NPC on a map."""
        loc = self.get_kafra_location(map_name)
        if loc is None:
            return None
        return (loc.x, loc.y)

    def get_nearest_kafra_map(self, current_map: str,
                               portal_knowledge: Any = None) -> str | None:
        """Find the nearest map with a Kafra NPC.

        Uses portal knowledge to find the closest town with Kafra.
        """
        if self.has_kafra(current_map):
            return current_map

        if portal_knowledge is None:
            # Fallback: check all known Kafra maps
            for map_name in self._kafra_locations:
                if map_name:
                    return map_name
            return "prontera"

        # BFS through portal graph to find nearest Kafra
        visited = {current_map}
        queue = [current_map]

        while queue:
            map_name = queue.pop(0)
            if self.has_kafra(map_name):
                return map_name
            for neighbor, _ in portal_knowledge.get_neighbors(map_name):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return "prontera"  # ultimate fallback

    # ── Warp Service ──────────────────────────────────────────────────

    def can_warp_to(self, from_city: str, to_city: str) -> bool:
        """Check if Kafra can warp from one city to another."""
        with self._lock:
            routes = KAFRA_WARP_ROUTES.get(from_city, [])
            return any(dest == to_city for dest, _ in routes)

    def get_warp_cost(self, from_city: str, to_city: str) -> int:
        """Get the zeny cost to warp between two cities."""
        with self._lock:
            routes = KAFRA_WARP_ROUTES.get(from_city, [])
            for dest, cost in routes:
                if dest == to_city:
                    return cost
            return 0

    def get_available_warp_destinations(self, from_city: str) -> list[tuple[str, int]]:
        """Get all available warp destinations from a city with costs."""
        with self._lock:
            return list(KAFRA_WARP_ROUTES.get(from_city, []))

    def get_warp_command(self, bot_id: str, from_city: str,
                          to_city: str) -> dict[str, Any]:
        """Get a bridge command to use Kafra warp service.

        Returns a command that tells the bridge to:
        1. Walk to Kafra NPC
        2. Talk to Kafra
        3. Select warp destination
        4. Pay zeny
        """
        cost = self.get_warp_cost(from_city, to_city)
        kafra_loc = self.get_kafra_location(from_city)

        return {
            "action": "kafra_warp",
            "from_city": from_city,
            "to_city": to_city,
            "cost": cost,
            "kafra_location": {
                "x": kafra_loc.x if kafra_loc else 0,
                "y": kafra_loc.y if kafra_loc else 0,
            } if kafra_loc else None,
            "npc_name": kafra_loc.npc_name if kafra_loc else "Kafra Employee",
        }

    def get_save_command(self, bot_id: str, map_name: str) -> dict[str, Any]:
        """Get a bridge command to set save point at Kafra."""
        kafra_loc = self.get_kafra_location(map_name)
        return {
            "action": "kafra_save",
            "map": map_name,
            "kafra_location": {
                "x": kafra_loc.x if kafra_loc else 0,
                "y": kafra_loc.y if kafra_loc else 0,
            } if kafra_loc else None,
            "npc_name": kafra_loc.npc_name if kafra_loc else "Kafra Employee",
        }

    def get_return_command(self, bot_id: str) -> dict[str, Any]:
        """Get a bridge command to return to save point.

        Uses butterfly wing if available, otherwise walks.
        """
        save = self.get_save_point(bot_id)
        if save is None:
            save = self.get_default_save_point()

        return {
            "action": "return_to_save",
            "save_point": {
                "map": save.map_name,
                "x": save.x,
                "y": save.y,
            },
            "method": "butterfly_wing",  # bridge will fall back to walking
        }

    def get_respawn_command(self, bot_id: str) -> dict[str, Any]:
        """Get a bridge command for respawn handling after death."""
        save = self.get_save_point(bot_id)
        if save is None:
            save = self.get_default_save_point()

        return {
            "action": "respawn",
            "save_point": {
                "map": save.map_name,
                "x": save.x,
                "y": save.y,
            },
        }

    # ── Status ───────────────────────────────────────────────────────

    def get_status_summary(self, bot_id: str) -> str:
        """Get a human-readable status summary for a bot."""
        save = self.get_save_point(bot_id)
        lines = [
            f"── Kafra Teleportation ──",
        ]
        if save:
            lines.append(f"Save Point: {save.map_name} ({save.x}, {save.y})")
        else:
            lines.append(f"Save Point: Not set (default: Prontera)")
        lines.append(f"Kafra locations known: {len(self._kafra_locations)}")
        lines.append(f"Warp routes available: {sum(len(v) for v in KAFRA_WARP_ROUTES.values())}")
        return "\n".join(lines)


# ── Global Singleton ──

_kafra_manager: KafraTeleportManager | None = None
_kafra_manager_lock = RLock()


def get_kafra_manager() -> KafraTeleportManager:
    global _kafra_manager
    with _kafra_manager_lock:
        if _kafra_manager is None:
            _kafra_manager = KafraTeleportManager()
        return _kafra_manager
