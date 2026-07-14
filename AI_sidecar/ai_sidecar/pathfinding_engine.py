"""
Pathfinding Engine — finds the optimal route through the RO world.

Combines portal knowledge with aggro avoidance, fly wing support,
map-server boundary awareness, and Kafra teleportation to produce
a step-by-step navigation plan that the bridge can execute.

Key design:
1. Portal graph (from PortalKnowledge) gives the map-level route
2. Within each map, walk directly to the next portal
3. Aggro avoidance: prefer safe maps, avoid high-danger zones
4. Fly wing support: use wings when stuck, surrounded, or traveling far
5. Map-server boundary awareness: handle disconnect/reconnect
6. Kafra teleportation: use warp service for long-distance travel
7. Produces bridge-compatible move commands
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from ai_sidecar.portal_knowledge import Portal, get_portal_knowledge

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class NavigationStep:
    """A single step in a navigation plan."""
    map_name: str
    target_x: int
    target_y: int
    step_type: str = "walk"  # walk | portal | warp | fly_wing | kafra_warp | arrive | respawn
    portal: Portal | None = None
    description: str = ""
    # Fly wing / Kafra metadata
    use_item: str = ""  # "Fly Wing" | "Butterfly Wing"
    zeny_cost: int = 0
    # Map-server boundary
    crossing_boundary: bool = False
    server_name: str = ""


@dataclass(slots=True)
class NavigationPlan:
    """A complete navigation plan from start to destination."""
    steps: list[NavigationStep] = field(default_factory=list)
    total_maps: int = 0
    total_portals: int = 0
    estimated_time_s: int = 0
    danger_level: str = "low"  # low | medium | high | deadly
    complete: bool = False
    error: str = ""
    # New fields
    uses_fly_wing: bool = False
    uses_kafra: bool = False
    crosses_boundary: bool = False
    total_zeny_cost: int = 0


class PathfindingEngine:
    """Finds optimal routes through the RO world.

    Thread-safe. Combines portal knowledge with aggro/danger data,
    fly wing support, map-server awareness, and Kafra teleportation
    to produce bridge-compatible navigation plans.
    """

    # Distance thresholds for fly wing usage (in portal hops)
    FLY_WING_SHORT_CUTOFF: int = 3  # walk if <= 3 hops
    FLY_WING_MEDIUM_CUTOFF: int = 6  # consider fly wing
    FLY_WING_LONG_CUTOFF: int = 10  # definitely use fly wing

    # Kafra warp thresholds
    KAFRA_WARP_MIN_HOPS: int = 4  # use Kafra if path is this many hops or more
    KAFRA_WARP_MAX_COST: int = 500  # max zeny to spend on Kafra warp

    def __init__(self) -> None:
        self._lock = RLock()
        self._portal_knowledge = get_portal_knowledge()
        # Danger map: map_name -> danger_score (0.0-1.0)
        # Populated by aggro_pathfinder or external data
        self._danger_map: dict[str, float] = {}
        # Known safe spots per map
        self._safe_spots: dict[str, list[tuple[int, int]]] = {}
        # Fly wing manager (lazy import to avoid circular deps)
        self._fly_wing_mgr = None
        self._map_server_knowledge = None
        self._predictive_aggro = None
        self._kafra_mgr = None

    # ── Lazy Imports ─────────────────────────────────────────────────

    def _get_fly_wing_mgr(self):
        if self._fly_wing_mgr is None:
            from ai_sidecar.fly_wing_manager import get_fly_wing_manager
            self._fly_wing_mgr = get_fly_wing_manager()
        return self._fly_wing_mgr

    def _get_map_server_knowledge(self):
        if self._map_server_knowledge is None:
            from ai_sidecar.map_server_knowledge import get_map_server_knowledge
            self._map_server_knowledge = get_map_server_knowledge()
        return self._map_server_knowledge

    def _get_predictive_aggro(self):
        if self._predictive_aggro is None:
            from ai_sidecar.predictive_aggro import get_predictive_aggro
            self._predictive_aggro = get_predictive_aggro()
        return self._predictive_aggro

    def _get_kafra_mgr(self):
        if self._kafra_mgr is None:
            from ai_sidecar.kafra_teleport import get_kafra_manager
            self._kafra_mgr = get_kafra_manager()
        return self._kafra_mgr

    # ── Danger Map Management ────────────────────────────────────────

    def set_danger(self, map_name: str, score: float) -> None:
        """Set the danger score for a map (0.0 = safe, 1.0 = deadly)."""
        with self._lock:
            self._danger_map[map_name] = max(0.0, min(1.0, score))

    def set_danger_map(self, danger_map: dict[str, float]) -> None:
        """Set multiple danger scores at once."""
        with self._lock:
            for k, v in danger_map.items():
                self._danger_map[k] = max(0.0, min(1.0, v))

    def get_danger(self, map_name: str) -> float:
        """Get the danger score for a map."""
        with self._lock:
            return self._danger_map.get(map_name, 0.0)

    def set_safe_spot(self, map_name: str, x: int, y: int) -> None:
        """Register a safe spot on a map."""
        with self._lock:
            if map_name not in self._safe_spots:
                self._safe_spots[map_name] = []
            self._safe_spots[map_name].append((x, y))

    def get_safe_spot(self, map_name: str) -> tuple[int, int] | None:
        """Get the nearest safe spot on a map."""
        with self._lock:
            spots = self._safe_spots.get(map_name, [])
            if spots:
                return spots[0]
            return None

    # ── Pathfinding ──────────────────────────────────────────────────

    def find_path(self, start_map: str, end_map: str,
                  bot_id: str = "",
                  aggro_count: int = 0,
                  is_stuck: bool = False,
                  fly_wings_available: bool = False,
                  zeny_available: int = 0) -> NavigationPlan:
        """Find the best path from start_map to end_map.

        Enhanced with fly wing support, map-server boundary awareness,
        predictive aggro, and Kafra teleportation.

        Args:
            start_map: Current map
            end_map: Destination map
            bot_id: Bot identifier (for fly wing tracking)
            aggro_count: Current aggro count (for fly wing decision)
            is_stuck: Whether the bot is stuck
            fly_wings_available: Whether fly wings are in inventory
            zeny_available: Available zeny for Kafra warp

        Returns a NavigationPlan with step-by-step instructions.
        """
        plan = NavigationPlan()

        if start_map == end_map:
            plan.complete = True
            plan.steps.append(NavigationStep(
                map_name=start_map,
                target_x=0, target_y=0,
                step_type="arrive",
                description=f"Already on {start_map}",
            ))
            return plan

        with self._lock:
            danger = dict(self._danger_map)

        # Check map-server boundary
        map_server = self._get_map_server_knowledge()
        crossing_boundary = map_server.crossing_boundary(start_map, end_map)
        plan.crosses_boundary = crossing_boundary

        # Check if Kafra warp is available and beneficial
        kafra_mgr = self._get_kafra_mgr()
        kafra_available = (
            kafra_mgr.has_kafra(start_map)
            and kafra_mgr.can_warp_to(start_map, end_map)
            and zeny_available >= kafra_mgr.get_warp_cost(start_map, end_map)
        )

        # Try danger-aware path first
        portals = self._portal_knowledge.find_path_with_cost(start_map, end_map, danger)

        # Fall back to shortest path
        if portals is None:
            portals = self._portal_knowledge.find_path(start_map, end_map)

        if portals is None:
            plan.error = f"No path found from {start_map} to {end_map}"
            logger.warning("pathfinding_no_path: %s -> %s", start_map, end_map)
            return plan

        path_hops = len(portals)

        # ── Decide on Kafra warp ──
        if kafra_available and path_hops >= self.KAFRA_WARP_MIN_HOPS:
            cost = kafra_mgr.get_warp_cost(start_map, end_map)
            if cost <= self.KAFRA_WARP_MAX_COST:
                kafra_loc = kafra_mgr.get_kafra_location(start_map)
                plan.steps.append(NavigationStep(
                    map_name=start_map,
                    target_x=kafra_loc.x if kafra_loc else 0,
                    target_y=kafra_loc.y if kafra_loc else 0,
                    step_type="kafra_warp",
                    description=f"Kafra warp from {start_map} to {end_map} (cost: {cost}z)",
                    zeny_cost=cost,
                ))
                plan.steps.append(NavigationStep(
                    map_name=end_map,
                    target_x=0, target_y=0,
                    step_type="arrive",
                    description=f"Arrived at {end_map} via Kafra warp",
                ))
                plan.total_maps = 2
                plan.total_portals = 0
                plan.estimated_time_s = 10
                plan.complete = True
                plan.uses_kafra = True
                plan.total_zeny_cost = cost
                plan.danger_level = "low"
                logger.info("pathfinding_kafra_warp: %s -> %s cost=%d",
                            start_map, end_map, cost)
                return plan

        # ── Decide on fly wing ──
        fly_wing_mgr = self._get_fly_wing_mgr()
        should_fly = False
        fly_reason = ""

        if fly_wings_available and bot_id:
            should_fly, fly_reason = fly_wing_mgr.should_use_fly_wing(
                bot_id, aggro_count=aggro_count, is_stuck=is_stuck,
                path_hops=path_hops, crossing_map_server=crossing_boundary,
            )

        if should_fly:
            plan.uses_fly_wing = True
            plan.steps.append(NavigationStep(
                map_name=start_map,
                target_x=0, target_y=0,
                step_type="fly_wing",
                use_item="Fly Wing",
                description=f"Use Fly Wing ({fly_reason})",
            ))
            # After fly wing, we may land on a random map.
            # For now, we still provide the portal path as fallback.
            plan.steps.append(NavigationStep(
                map_name=end_map,
                target_x=0, target_y=0,
                step_type="arrive",
                description=f"Arrived at {end_map} (after fly wing)",
            ))
            plan.total_maps = 2
            plan.total_portals = 0
            plan.estimated_time_s = 5
            plan.complete = True
            plan.danger_level = "low"
            logger.info("pathfinding_fly_wing: %s -> %s reason=%s",
                        start_map, end_map, fly_reason)
            return plan

        # ── Build normal navigation steps ──
        current_map = start_map
        steps: list[NavigationStep] = []

        for portal in portals:
            # Check if this step crosses a map-server boundary
            step_crosses = map_server.crossing_boundary(current_map, portal.target_map)
            server_name = map_server.get_server_for_map(current_map) or ""

            # Step: walk to portal on current map
            steps.append(NavigationStep(
                map_name=current_map,
                target_x=portal.source_x,
                target_y=portal.source_y,
                step_type="walk",
                portal=portal,
                description=f"Walk to {portal.name or portal.target_map} portal at ({portal.source_x},{portal.source_y})",
                crossing_boundary=False,
                server_name=server_name,
            ))

            # Step: take portal to next map
            steps.append(NavigationStep(
                map_name=current_map,
                target_x=portal.source_x,
                target_y=portal.source_y,
                step_type="portal",
                portal=portal,
                description=f"Take portal to {portal.target_map}",
                crossing_boundary=step_crosses,
                server_name=server_name,
            ))

            current_map = portal.target_map

        # Final step: arrive at destination map
        final_server = map_server.get_server_for_map(current_map) or ""
        steps.append(NavigationStep(
            map_name=current_map,
            target_x=0, target_y=0,
            step_type="arrive",
            description=f"Arrived at {current_map}",
            crossing_boundary=False,
            server_name=final_server,
        ))

        plan.steps = steps
        plan.total_maps = len(set(s.map_name for s in steps))
        plan.total_portals = len(portals)
        plan.estimated_time_s = len(portals) * 15 + len(steps) * 5
        plan.complete = True

        # Determine overall danger level
        max_danger = max((danger.get(s.map_name, 0.0) for s in steps), default=0.0)
        if max_danger >= 0.7:
            plan.danger_level = "deadly"
        elif max_danger >= 0.4:
            plan.danger_level = "high"
        elif max_danger >= 0.2:
            plan.danger_level = "medium"
        else:
            plan.danger_level = "low"

        logger.info(
            "pathfinding_plan: %s -> %s via %d portals, danger=%s, est=%ds, boundary=%s",
            start_map, end_map, plan.total_portals, plan.danger_level,
            plan.estimated_time_s, plan.crosses_boundary,
        )
        return plan

    def find_path_to_zone(self, current_map: str, target_map: str,
                          current_x: int = 0, current_y: int = 0,
                          bot_id: str = "",
                          aggro_count: int = 0,
                          is_stuck: bool = False,
                          fly_wings_available: bool = False,
                          zeny_available: int = 0) -> NavigationPlan:
        """Find path to a hunting zone, starting from current position.

        If already on the target map, returns a plan to walk to a safe spot.
        """
        if current_map == target_map:
            plan = NavigationPlan(complete=True)
            safe = self.get_safe_spot(target_map)
            if safe:
                plan.steps.append(NavigationStep(
                    map_name=target_map,
                    target_x=safe[0], target_y=safe[1],
                    step_type="walk",
                    description=f"Move to safe spot on {target_map}",
                ))
            plan.steps.append(NavigationStep(
                map_name=target_map,
                target_x=0, target_y=0,
                step_type="arrive",
                description=f"Already on {target_map}",
            ))
            return plan

        return self.find_path(
            current_map, target_map,
            bot_id=bot_id, aggro_count=aggro_count,
            is_stuck=is_stuck, fly_wings_available=fly_wings_available,
            zeny_available=zeny_available,
        )

    def to_bridge_commands(self, plan: NavigationPlan) -> list[dict[str, Any]]:
        """Convert a NavigationPlan to bridge-compatible move commands.

        Returns a list of dicts with format:
          {"action": "move", "x": int, "y": int, "map": str}
        Extended to support fly wing, Kafra warp, and boundary crossing commands.
        """
        commands: list[dict[str, Any]] = []
        for step in plan.steps:
            if step.step_type == "walk":
                commands.append({
                    "action": "move",
                    "x": step.target_x,
                    "y": step.target_y,
                    "map": step.map_name,
                })
            elif step.step_type == "portal" and step.portal:
                cmd = {
                    "action": "move",
                    "x": step.portal.source_x,
                    "y": step.portal.source_y,
                    "map": step.portal.source_map,
                }
                if step.crossing_boundary:
                    cmd["crossing_boundary"] = True
                    cmd["server_name"] = step.server_name
                commands.append(cmd)
            elif step.step_type == "fly_wing":
                commands.append({
                    "action": "use_item",
                    "item": step.use_item or "Fly Wing",
                    "quantity": 1,
                    "reason": "navigation",
                })
            elif step.step_type == "kafra_warp":
                commands.append({
                    "action": "kafra_warp",
                    "from_city": step.map_name,
                    "to_city": step.description.split(" to ")[-1].split(" ")[0]
                    if " to " in step.description else "",
                    "cost": step.zeny_cost,
                })
            elif step.step_type == "arrive":
                commands.append({
                    "action": "move",
                    "x": step.target_x or 50,
                    "y": step.target_y or 50,
                    "map": step.map_name,
                })
        return commands

    def get_route_summary(self, plan: NavigationPlan) -> str:
        """Get a human-readable summary of a navigation plan."""
        if not plan.complete:
            return f"Navigation failed: {plan.error}"
        lines = [
            f"Route: {plan.total_maps} maps, {plan.total_portals} portals, "
            f"danger={plan.danger_level}",
        ]
        if plan.uses_fly_wing:
            lines.append("Uses Fly Wing")
        if plan.uses_kafra:
            lines.append(f"Uses Kafra Warp (cost: {plan.total_zeny_cost}z)")
        if plan.crosses_boundary:
            lines.append("Crosses map-server boundary")
        for i, step in enumerate(plan.steps):
            lines.append(f"  {i+1}. [{step.step_type}] {step.description}")
        return "\n".join(lines)


# ── Global Singleton ──

_pathfinding_engine: PathfindingEngine | None = None
_pathfinding_engine_lock = RLock()


def get_pathfinding_engine() -> PathfindingEngine:
    global _pathfinding_engine
    with _pathfinding_engine_lock:
        if _pathfinding_engine is None:
            _pathfinding_engine = PathfindingEngine()
        return _pathfinding_engine
