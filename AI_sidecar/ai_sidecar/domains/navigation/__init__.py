"""Navigation domain — portal database, Dijkstra pathfinding, route actions,
and RO fast-travel (Kafra, Fly Wing, Butterfly Wing, Airship).

Provides:
  - PortalDB: Thread-safe database of 115+ real RO portal connections
  - Pathfinder: Dijkstra shortest-path with LRU caching
  - NavigationDomain: Domain integration with the PDCA assessment loop
  - TravelRecommender: Optimal travel method selection (walk, Kafra, wings, airship)
  - KafraDestination / AirshipRoute: Data classes for fast-travel systems
  - Convenience functions: find_path(), path_to_move_commands(), route_to()
"""
from __future__ import annotations

from ai_sidecar.domains.navigation.portals import PortalDB, PortalConnection, get_portal_db
from ai_sidecar.domains.navigation.pathfinding import (
    Pathfinder,
    PathWaypoint,
    find_path,
    path_exists,
    get_pathfinder,
)
from ai_sidecar.domains.navigation.actions import (
    NavigationDomain,
    build_navigation_route,
    path_to_move_commands,
    nearest_portal,
)
from ai_sidecar.domains.navigation.travel_recommender import (
    TravelRecommender,
    TravelRecommendation,
    KafraDestination,
    AirshipRoute,
    get_travel_recommender,
)

__all__ = [
    # Domain
    "NavigationDomain",
    # Portal database
    "PortalDB",
    "PortalConnection",
    "get_portal_db",
    # Pathfinding
    "Pathfinder",
    "PathWaypoint",
    "find_path",
    "path_exists",
    "get_pathfinder",
    # Actions / route building
    "build_navigation_route",
    "path_to_move_commands",
    "nearest_portal",
    # Travel recommender
    "TravelRecommender",
    "TravelRecommendation",
    "KafraDestination",
    "AirshipRoute",
    "get_travel_recommender",
]
