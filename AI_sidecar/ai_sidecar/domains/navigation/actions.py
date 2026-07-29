"""
Navigation actions — convert computed paths to HeuristicAction commands.

Provides the NavigationDomain that integrates with the PDCA assessment loop,
converting pathfinding results into move commands the bridge can execute.
"""
from __future__ import annotations

import logging
from typing import Any

from ai_sidecar.actions import HeuristicAction
from ai_sidecar.domains.navigation.pathfinding import (
    PathWaypoint,
    get_pathfinder,
    find_path,
)
from ai_sidecar.domains.navigation.portals import get_portal_db

logger = logging.getLogger(__name__)


def path_to_move_commands(
    path: list[PathWaypoint],
    bot_id: str = "",
) -> list[HeuristicAction]:
    """Convert a computed path to a sequence of move commands.

    Each waypoint generates a move command to the portal location
    on the current map. After crossing, the next command moves to
    the next portal.
    """
    if not path:
        return []

    actions: list[HeuristicAction] = []
    for i, wp in enumerate(path):
        action = HeuristicAction(
            kind="command",
            command=f"move {wp.from_x} {wp.from_y}",
            confidence=0.95,
            reason=(
                f"navigation step {i + 1}/{len(path)}: "
                f"move to portal on {wp.map_name} "
                f"({wp.from_x},{wp.from_y}) -> {wp.to_x},{wp.to_y}"
            ),
            domain="navigation",
            metadata={
                "step": i,
                "total_steps": len(path),
                "current_map": wp.map_name,
                "portal_x": wp.from_x,
                "portal_y": wp.from_y,
                "arrival_x": wp.to_x,
                "arrival_y": wp.to_y,
                "bot_id": bot_id,
            },
        )
        actions.append(action)

    return actions


def build_navigation_route(
    from_map: str,
    to_map: str,
    bot_id: str = "",
) -> list[HeuristicAction]:
    """Find shortest path from -> to and return move commands.

    Returns list of HeuristicAction commands. Empty list if no path found.
    """
    path = find_path(from_map, to_map)
    if not path:
        logger.info(
            "navigation: no route from '%s' to '%s' for bot '%s'",
            from_map, to_map, bot_id or "?",
        )
        return []

    logger.info(
        "navigation: route %s -> %s: %d hops for bot '%s'",
        from_map, to_map, len(path), bot_id or "?",
    )
    return path_to_move_commands(path, bot_id=bot_id)


def format_path_for_log(path: list[PathWaypoint]) -> str:
    """Format a path as a readable string for logging."""
    if not path:
        return "(empty path)"

    segments: list[str] = []
    for i, wp in enumerate(path):
        segments.append(
            f"{i + 1}. {wp.map_name} ({wp.from_x},{wp.from_y}) -> "
            f"({wp.to_x},{wp.to_y})"
        )
    return " | ".join(segments)


def nearest_portal(map_name: str) -> HeuristicAction | None:
    """Generate a 'move to nearest portal' action for town-orientation."""
    portals = get_portal_db().get_portals(map_name)
    if not portals:
        return None

    # Pick the first portal as the "nearest" (typically the one closest to center)
    portal = portals[0]
    if portal.map1 == map_name:
        return HeuristicAction(
            kind="command",
            command=f"move {portal.x1} {portal.y1}",
            confidence=0.8,
            reason=f"move to portal on {map_name} at ({portal.x1},{portal.y1})",
            domain="navigation",
            metadata={
                "portal_x": portal.x1,
                "portal_y": portal.y1,
                "target_map": portal.map2,
            },
        )
    else:
        return HeuristicAction(
            kind="command",
            command=f"move {portal.x2} {portal.y2}",
            confidence=0.8,
            reason=f"move to portal on {map_name} at ({portal.x2},{portal.y2})",
            domain="navigation",
            metadata={
                "portal_x": portal.x2,
                "portal_y": portal.y2,
                "target_map": portal.map1,
            },
        )


# ── NavigationDomain — integrates with PDCA assessment loop ──────────

class NavigationDomain:
    """Navigation domain — pathfinding and routing for the PDCA loop.

    This domain does NOT directly set lockMap (that's the heuristic's job).
    Instead, it provides pathfinding services that other domains and the
    heuristic can use to generate waypoint move commands.

    In assess(): checks if the bot needs to navigate to a different map
    and emits route commands.
    """

    name = "navigation"
    priority = 60  # Runs after economy (50), before general (100)

    def __init__(self) -> None:
        self._pathfinder = get_pathfinder()
        self._portal_db = get_portal_db()
        self._last_destination: dict[str, str] = {}  # bot_id -> target_map
        self._route_in_progress: dict[str, bool] = {}  # bot_id -> bool

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        bot_id: str,
    ) -> None:
        """Assess navigation needs from current signals.

        Currently a passive domain — provides pathfinding services.
        Heuristic or other domains call build_navigation_route() directly.
        This assess() is reserved for future auto-routing scenarios
        (e.g., dungeon retreat, emergency town return).
        """
        _ = signals  # Future: detect map changes, stuck detection
        _ = actions
        _ = bot_id
        pass  # Passive for now — routing decisions come from heuristic

    # ── Public domain services ───────────────────────────────────────

    def route_to(
        self,
        target_map: str,
        current_map: str,
        bot_id: str = "",
    ) -> list[HeuristicAction]:
        """Compute and return move commands from current_map to target_map."""
        return build_navigation_route(current_map, target_map, bot_id=bot_id)

    def get_pathfinder(self):
        """Access the underlying pathfinder."""
        return self._pathfinder

    def get_portal_db(self):
        """Access the underlying portal database."""
        return self._portal_db

    def set_destination(self, bot_id: str, target_map: str) -> None:
        """Record a destination for a bot (used by heuristic)."""
        self._last_destination[bot_id] = target_map

    def get_destination(self, bot_id: str) -> str | None:
        """Get the last recorded destination for a bot."""
        return self._last_destination.get(bot_id)

    def clear_destination(self, bot_id: str) -> None:
        """Clear navigation state for a bot."""
        self._last_destination.pop(bot_id, None)
        self._route_in_progress.pop(bot_id, None)

    def counters(self) -> dict[str, int]:
        """Return diagnostic counters."""
        return {
            "maps_in_db": self._portal_db.get_map_count(),
            "portals_in_db": self._portal_db.get_portal_count(),
            "cached_paths": self._pathfinder.get_cache_stats()["cached_paths"],
            "active_routes": len(self._last_destination),
        }

    def __repr__(self) -> str:
        return (
            f"<NavigationDomain: {self._portal_db.get_map_count()} maps, "
            f"{self._portal_db.get_portal_count()} portals>"
        )
