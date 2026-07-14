"""
Aggro-Aware Pathfinding — Weighted A* with dynamic cost weighting.

Uses OpenKore's existing walkability data and adds aggro monster cost weighting.
A pro player knows to take the long way around if the short path goes through
5 aggressive monsters. This module finds the SAFE path, not just the short path.

Algorithm: Weighted A* with Dynamic Cost Map
- Base cost: 1.0 per walkable tile
- Aggro cost: Σ(threat / distance²) for each monster within range
- Wall cost: infinity (non-walkable)
- Heuristic: octile distance (faster than Manhattan for 8-directional movement)
- Weighting: f(n) = g(n) + 1.5 * h(n) (weighted A* for faster search)
- Recompute: timer-based every 500ms, or on demand when cost delta exceeds threshold
"""

from __future__ import annotations

import heapq
import logging
import math
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class PathNode:
    """A node in the A* search."""
    x: int
    y: int
    g: float = 0.0  # Cost from start
    h: float = 0.0  # Heuristic to goal
    f: float = 0.0  # Total cost (g + w * h)
    parent: Any | None = None  # Parent node for path reconstruction

    def __lt__(self, other: "PathNode") -> bool:
        return self.f < other.f

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PathNode):
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))


@dataclass
class AggroThreat:
    """A threat source that affects path cost."""
    x: int
    y: int
    threat_score: float = 1.0  # 0-10, higher = more dangerous
    aggro_range: int = 10  # Tiles within which this monster adds cost
    is_boss: bool = False
    is_casting: bool = False


@dataclass
class PathResult:
    """Result of a pathfinding query."""
    path: list[tuple[int, int]] = field(default_factory=list)
    found: bool = False
    total_cost: float = 0.0
    computation_time_ms: float = 0.0
    nodes_expanded: int = 0
    max_threat_on_path: float = 0.0
    avg_threat_on_path: float = 0.0
    safe_path: bool = True


class AggroPathfinder:
    """Weighted A* pathfinder with dynamic aggro cost weighting."""

    # 8-directional movement
    DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    # Cost multiplier for diagonal movement
    DIAGONAL_COST = math.sqrt(2)

    def __init__(self) -> None:
        self._lock = RLock()
        self._walkable: list[list[bool]] = []  # walkable[y][x]
        self._map_width: int = 0
        self._map_height: int = 0
        self._threats: list[AggroThreat] = []
        self._cost_cache: dict[tuple[int, int], float] = {}
        self._cost_cache_time: float = 0.0
        self._cost_cache_ttl: float = 0.5  # 500ms cache TTL
        self._max_threat_range: int = 15  # Max range to consider for threat cost
        self._aggro_cost_multiplier: float = 5.0  # How much aggro affects path cost
        self._heuristic_weight: float = 1.5  # Weighted A* factor
        self._max_nodes: int = 50000  # Safety limit
        self._stats: dict[str, int] = {
            "queries": 0, "found": 0, "not_found": 0, "nodes_expanded": 0,
        }

    # ── Public API ──

    def set_walkable(self, walkable: list[list[bool]], width: int, height: int) -> None:
        """Set the walkability grid from OpenKore's $field->{walkable}."""
        with self._lock:
            self._walkable = walkable
            self._map_width = width
            self._map_height = height
            self._cost_cache.clear()
            logger.info("aggro_pathfinder: walkable grid set (%dx%d)", width, height)

    def update_threats(self, threats: list[AggroThreat]) -> None:
        """Update the list of active threats (monsters)."""
        with self._lock:
            self._threats = threats
            self._cost_cache.clear()  # Invalidate cost cache on threat change

    def find_path(self, start_x: int, start_y: int, goal_x: int, goal_y: int) -> PathResult:
        """Find the safest path from start to goal using Weighted A*."""
        start_time = time.time()
        result = PathResult()

        with self._lock:
            self._stats["queries"] += 1

            # Validate coordinates
            if not self._is_walkable(start_x, start_y):
                logger.warning("aggro_pathfinder: start (%d,%d) is not walkable", start_x, start_y)
                result.found = False
                return result
            if not self._is_walkable(goal_x, goal_y):
                logger.warning("aggro_pathfinder: goal (%d,%d) is not walkable", goal_x, goal_y)
                result.found = False
                return result

            # A* search
            start_node = PathNode(x=start_x, y=start_y)
            goal_node = PathNode(x=goal_x, y=goal_y)

            open_set: list[PathNode] = []
            closed_set: set[tuple[int, int]] = set()
            g_scores: dict[tuple[int, int], float] = {}

            start_node.h = self._heuristic(start_x, start_y, goal_x, goal_y)
            start_node.f = start_node.g + self._heuristic_weight * start_node.h
            heapq.heappush(open_set, start_node)
            g_scores[(start_x, start_y)] = 0.0

            nodes_expanded = 0

            while open_set and nodes_expanded < self._max_nodes:
                current = heapq.heappop(open_set)
                current_key = (current.x, current.y)

                if current_key in closed_set:
                    continue

                closed_set.add(current_key)
                nodes_expanded += 1

                # Goal check
                if current.x == goal_x and current.y == goal_y:
                    # Reconstruct path
                    path: list[tuple[int, int]] = []
                    node = current
                    while node:
                        path.append((node.x, node.y))
                        node = node.parent
                    path.reverse()

                    result.found = True
                    result.path = path
                    result.total_cost = current.g
                    result.computation_time_ms = (time.time() - start_time) * 1000
                    result.nodes_expanded = nodes_expanded

                    # Compute threat metrics on path
                    total_threat = 0.0
                    max_threat = 0.0
                    for px, py in path:
                        threat = self._get_tile_cost(px, py) - 1.0  # Subtract base cost
                        total_threat += threat
                        max_threat = max(max_threat, threat)
                    result.max_threat_on_path = max_threat
                    result.avg_threat_on_path = total_threat / len(path) if path else 0
                    result.safe_path = max_threat < 3.0  # Arbitrary threshold

                    self._stats["found"] += 1
                    self._stats["nodes_expanded"] += nodes_expanded
                    return result

                # Expand neighbors
                for dx, dy in self.DIRECTIONS:
                    nx, ny = current.x + dx, current.y + dy

                    if not self._is_walkable(nx, ny):
                        continue

                    neighbor_key = (nx, ny)
                    if neighbor_key in closed_set:
                        continue

                    # Movement cost
                    if dx != 0 and dy != 0:
                        move_cost = self.DIAGONAL_COST
                    else:
                        move_cost = 1.0

                    # Tile cost (base + aggro threat)
                    tile_cost = self._get_tile_cost(nx, ny)
                    step_cost = move_cost * tile_cost

                    new_g = current.g + step_cost

                    if neighbor_key in g_scores and new_g >= g_scores[neighbor_key]:
                        continue

                    g_scores[neighbor_key] = new_g
                    neighbor = PathNode(
                        x=nx, y=ny,
                        g=new_g,
                        h=self._heuristic(nx, ny, goal_x, goal_y),
                        parent=current,
                    )
                    neighbor.f = neighbor.g + self._heuristic_weight * neighbor.h
                    heapq.heappush(open_set, neighbor)

            # No path found
            result.found = False
            result.computation_time_ms = (time.time() - start_time) * 1000
            result.nodes_expanded = nodes_expanded
            self._stats["not_found"] += 1
            self._stats["nodes_expanded"] += nodes_expanded
            return result

    def find_safe_path(self, start_x: int, start_y: int, goal_x: int, goal_y: int,
                       max_threat_threshold: float = 3.0) -> PathResult:
        """Find a path that stays below a maximum threat threshold."""
        result = self.find_path(start_x, start_y, goal_x, goal_y)
        if result.found and result.max_threat_on_path > max_threat_threshold:
            # Try with higher aggro cost multiplier to penalize threats more
            old_mult = self._aggro_cost_multiplier
            self._aggro_cost_multiplier = old_mult * 2
            self._cost_cache.clear()
            result2 = self.find_path(start_x, start_y, goal_x, goal_y)
            self._aggro_cost_multiplier = old_mult
            self._cost_cache.clear()
            if result2.found and result2.max_threat_on_path < result.max_threat_on_path:
                return result2
        return result

    def get_safe_direction(self, x: int, y: int, goal_x: int, goal_y: int) -> tuple[int, int] | None:
        """Get the next step direction toward the goal, avoiding threats."""
        result = self.find_path(x, y, goal_x, goal_y)
        if result.found and len(result.path) > 1:
            next_step = result.path[1]
            return (next_step[0] - x, next_step[1] - y)
        return None

    def is_safe_to_move(self, x: int, y: int, goal_x: int, goal_y: int) -> bool:
        """Check if it's safe to move from current position toward goal."""
        result = self.find_path(x, y, goal_x, goal_y)
        return result.found and result.safe_path

    # ── Internal Methods ──

    def _is_walkable(self, x: int, y: int) -> bool:
        """Check if a tile is walkable."""
        if not self._walkable:
            return True  # No walkability data = assume walkable
        if x < 0 or x >= self._map_width or y < 0 or y >= self._map_height:
            return False
        return self._walkable[y][x]

    def _get_tile_cost(self, x: int, y: int) -> float:
        """Get the total cost of a tile (base + aggro threat)."""
        # Check cache
        key = (x, y)
        now = time.time()
        if key in self._cost_cache and now - self._cost_cache_time < self._cost_cache_ttl:
            return self._cost_cache[key]

        # Base cost
        cost = 1.0

        # Aggro threat cost
        for threat in self._threats:
            dist = math.sqrt((x - threat.x) ** 2 + (y - threat.y) ** 2)
            if dist <= threat.aggro_range and dist > 0:
                # Threat cost: threat_score / distance²
                # Bosses and casting monsters have higher threat
                threat_weight = threat.threat_score
                if threat.is_boss:
                    threat_weight *= 2.0
                if threat.is_casting:
                    threat_weight *= 1.5
                threat_cost = (threat_weight * self._aggro_cost_multiplier) / (dist * dist)
                cost += threat_cost

        # Cache the result
        self._cost_cache[key] = cost
        self._cost_cache_time = now
        return cost

    def _heuristic(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Octile distance heuristic (faster than Manhattan for 8-directional)."""
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        return max(dx, dy) + (self.DIAGONAL_COST - 1) * min(dx, dy)

    def get_stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def get_path_summary(self, result: PathResult) -> str:
        if not result.found:
            return "No path found"
        return (
            f"Path: {len(result.path)} tiles, "
            f"cost={result.total_cost:.1f}, "
            f"time={result.computation_time_ms:.1f}ms, "
            f"nodes={result.nodes_expanded}, "
            f"max_threat={result.max_threat_on_path:.1f}, "
            f"avg_threat={result.avg_threat_on_path:.2f}, "
            f"safe={result.safe_path}"
        )

    def reset(self) -> None:
        with self._lock:
            self._walkable = []
            self._map_width = 0
            self._map_height = 0
            self._threats.clear()
            self._cost_cache.clear()
            self._stats = {"queries": 0, "found": 0, "not_found": 0, "nodes_expanded": 0}


# ── Global Singleton ──

_aggro_pathfinder: AggroPathfinder | None = None
_aggro_pathfinder_lock = RLock()


def get_aggro_pathfinder() -> AggroPathfinder:
    global _aggro_pathfinder
    with _aggro_pathfinder_lock:
        if _aggro_pathfinder is None:
            _aggro_pathfinder = AggroPathfinder()
        return _aggro_pathfinder
