"""
Pathfinding module — Dijkstra shortest-path routing on the RO portal graph.

Computes optimal routes between any two RO maps using:
1. Direct portal connections (edge weight = 1)
2. Town path shortcuts (weighted by approximate map count)
3. Optional warp portal routing (NPC teleporters)
"""
from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from typing import Any

from ai_sidecar.domains.navigation.portals import PortalDB, get_portal_db

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PathWaypoint:
    """A single waypoint in a computed path.

    When traveling through a portal, the bot moves from (from_x, from_y)
    on the current map, through a portal, to (to_x, to_y) on the next map.
    """
    map_name: str
    from_x: int
    from_y: int
    to_x: int
    to_y: int

    def move_command(self) -> str:
        """Generate an OpenKore move command for this waypoint."""
        return f"move {self.from_x} {self.from_y}"

    def as_tuple(self) -> tuple[str, int, int, int, int]:
        return (self.map_name, self.from_x, self.from_y, self.to_x, self.to_y)


# ── LRU-style cache for computed paths ──────────────────────────────

@dataclass(slots=True)
class PathCache:
    _cache: dict[tuple[str, str], list[PathWaypoint]] = field(default_factory=dict)
    _max_size: int = 256

    def get(self, from_map: str, to_map: str) -> list[PathWaypoint] | None:
        return self._cache.get((from_map, to_map))

    def put(self, from_map: str, to_map: str, path: list[PathWaypoint]) -> None:
        key = (from_map, to_map)
        if key in self._cache:
            return  # Already cached
        if len(self._cache) >= self._max_size:
            # Evict oldest entry (simple FIFO eviction)
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = path

    def invalidate(self, map_name: str | None = None) -> None:
        """Invalidate cache — all entries, or entries involving a specific map."""
        if map_name is None:
            self._cache.clear()
        else:
            self._cache = {
                k: v for k, v in self._cache.items()
                if map_name not in (k[0], k[1])
            }

    def __len__(self) -> int:
        return len(self._cache)


# ── Dijkstra Pathfinder ──────────────────────────────────────────────

class PathNotFoundError(ValueError):
    """Raised when no path exists between two maps."""
    pass


class Pathfinder:
    """Dijkstra shortest-path on the RO portal graph.

    Computes the shortest path between any two RO maps using portal
    connections. Returns a list of waypoints with precise coordinates
    for each portal crossing.

    Thread-safe when used with PortalDB (which is thread-safe).
    """

    def __init__(self, portal_db: PortalDB | None = None) -> None:
        self._db = portal_db or get_portal_db()
        self._cache = PathCache()

    # ── Public API ───────────────────────────────────────────────────

    def find_path(self, from_map: str, to_map: str) -> list[PathWaypoint]:
        """Find the shortest path from from_map to to_map.

        Returns list of PathWaypoints describing each portal crossing,
        or empty list if no path exists.
        """
        if from_map == to_map:
            return []

        # Check cache first
        cached = self._cache.get(from_map, to_map)
        if cached is not None:
            return list(cached)

        # Run Dijkstra
        path = self._dijkstra(from_map, to_map)

        # Cache the result
        self._cache.put(from_map, to_map, path)
        return list(path)

    def find_path_with_coords(
        self,
        from_map: str,
        to_map: str,
    ) -> list[tuple[str, int, int, int, int]]:
        """Find path and return as list of (map, from_x, from_y, to_x, to_y) tuples."""
        waypoints = self.find_path(from_map, to_map)
        return [wp.as_tuple() for wp in waypoints]

    def path_exists(self, from_map: str, to_map: str) -> bool:
        """Check if a path exists between two maps (fast check using Dijkstra)."""
        try:
            path = self.find_path(from_map, to_map)
            return len(path) > 0
        except PathNotFoundError:
            return False

    def get_path_distance(self, from_map: str, to_map: str) -> int:
        """Get the number of portal crossings (hops) on the shortest path."""
        path = self.find_path(from_map, to_map)
        return len(path)

    def invalidate_cache(self, map_name: str | None = None) -> None:
        """Clear cached paths."""
        self._cache.invalidate(map_name)

    def get_cache_stats(self) -> dict[str, int]:
        return {
            "cached_paths": len(self._cache),
            "max_cache_size": self._cache._max_size,
        }

    # ── Dijkstra Implementation ──────────────────────────────────────

    def _dijkstra(self, start: str, goal: str) -> list[PathWaypoint]:
        """Standard Dijkstra on the portal graph.

        Returns a list of PathWaypoints describing the sequence of portal
        crossings from start to goal.

        If goal is unreachable, returns empty list.
        """
        if not self._db.has_map(start):
            logger.warning("pathfinding: unknown start map '%s'", start)
            return []
        if not self._db.has_map(goal):
            logger.warning("pathfinding: unknown goal map '%s'", goal)
            return []

        # Priority queue entries: (total_cost, current_map, path_so_far_in_portal_steps)
        # Using list of tuples for the path so we can reconstruct
        # Format: [(map_name, from_x, from_y, to_x, to_y), ...]
        # But we store the portal coordinates for each step

        # Distances: map_name -> cost_to_reach
        distances: dict[str, int] = {start: 0}

        # Previous step for reconstruction: map_name -> (prev_map, portal_info)
        # portal_info = (from_x, from_y, to_x, to_y) for the step that arrived at this map
        previous: dict[str, tuple[str, int, int, int, int]] = {}

        # Priority queue: (cost, map_name)
        pq: list[tuple[int, str]] = [(0, start)]
        visited: set[str] = set()

        while pq:
            current_cost, current = heapq.heappop(pq)

            if current in visited:
                continue

            if current == goal:
                # Reconstruct the path
                return self._reconstruct_path(previous, start, goal)

            visited.add(current)

            # Explore neighbors
            neighbors = self._db.get_neighbors(current)
            for neighbor_map, edge_weight in neighbors:
                if neighbor_map in visited:
                    continue

                new_cost = current_cost + edge_weight

                if neighbor_map not in distances or new_cost < distances[neighbor_map]:
                    distances[neighbor_map] = new_cost

                    # Get portal coordinates for this step
                    route_step = self._db.get_route_step(current, neighbor_map)
                    if route_step is not None:
                        # route_step = (target_map, from_x, from_y, to_x, to_y)
                        _, from_x, from_y, to_x, to_y = route_step
                        previous[neighbor_map] = (current, from_x, from_y, to_x, to_y)
                    else:
                        # HARDENED (completeness sweep 2026-08-10): a graph edge
                        # with NO known portal coordinates cannot be routed — the
                        # old code stored (0,0) as a "placeholder", which later
                        # produced `move 0 0` commands sending the bot to a map
                        # corner. get_neighbors() and get_route_step() currently
                        # iterate the SAME _graph conns, so this branch is
                        # unreachable today, but if the graph ever gains edges
                        # without portal coords (e.g. Kafra/fly-wing shortcuts),
                        # silently routing through (0,0) would strand the bot.
                        # Skip the edge instead: Dijkstra finds an alternate
                        # route, or honestly reports the goal unreachable.
                        logger.debug(
                            "pathfinding: edge %s -> %s has no portal coords; skipping (route may be lost)",
                            current, neighbor_map,
                        )
                        continue

                    heapq.heappush(pq, (new_cost, neighbor_map))

        # Goal not reachable
        logger.info(
            "pathfinding: no path from '%s' to '%s' (visited %d maps, %d reachable)",
            start, goal, len(visited), len(distances),
        )
        return []

    def _reconstruct_path(
        self,
        previous: dict[str, tuple[str, int, int, int, int]],
        start: str,
        goal: str,
    ) -> list[PathWaypoint]:
        """Reconstruct the path from Dijkstra's previous map."""
        waypoints: list[PathWaypoint] = []
        current = goal

        # Walk backwards from goal to start
        while current != start:
            if current not in previous:
                logger.warning(
                    "pathfinding: broken path reconstruction at '%s'", current
                )
                return []

            prev_map, from_x, from_y, to_x, to_y = previous[current]

            # The portal from prev_map -> current: we arrive at (to_x, to_y) on 'current'
            # From 'current', we need to walk to the portal at (from_x, from_y) on 'prev_map'
            # Wait — let me think about this carefully.
            #
            # When traveling from prev_map to current:
            #   - On prev_map, the step starts from (from_x, from_y) (the portal on prev_map)
            #   - On current, the step ends at (to_x, to_y) (arrival coords on current)
            #
            # But waypoints represent each step: on map X, go to portal (X_x, X_y)
            # then warp to map Y arriving at (Y_x, Y_y).
            #
            # So for the step prev_map -> current:
            #   Waypoint: map=prev_map, from_x=portal_on_prev, from_y=portal_on_prev,
            #             to_x=arrival_on_current, to_y=arrival_on_current

            waypoints.append(PathWaypoint(
                map_name=prev_map,
                from_x=from_x,
                from_y=from_y,
                to_x=to_x,
                to_y=to_y,
            ))

            current = prev_map

        # The path was built backwards (goal -> start), so reverse it
        waypoints.reverse()
        return waypoints


# ── Convenience Functions ───────────────────────────────────────────

def find_path(from_map: str, to_map: str) -> list[PathWaypoint]:
    """Find shortest path between two maps using the global Pathfinder."""
    return get_pathfinder().find_path(from_map, to_map)


def path_exists(from_map: str, to_map: str) -> bool:
    """Check if a path exists between two maps."""
    return get_pathfinder().path_exists(from_map, to_map)


# ── Global Singleton ──

_pathfinder: Pathfinder | None = None
_pathfinder_lock = object()


def get_pathfinder() -> Pathfinder:
    """Get the global Pathfinder singleton."""
    global _pathfinder
    if _pathfinder is None:
        _pathfinder = Pathfinder()
    return _pathfinder
