"""
NavigationEngine — Kafra / Fly Wing / Butterfly Wing / Walk routing.

Provides intelligent RO navigation using all available travel methods:
  - Kafra Warp: NPC teleport between major towns (data-driven from YAML)
  - Fly Wing: Item-based random teleport within current map (danger-based)
  - Butterfly Wing: Item-based return to save point (HP-based)
  - Walk: Portal-based pathfinding as fallback

All route data is loaded from YAML — no hardcoded paths.
Thread-safe via RLock on all public methods.

Factory function:
    engine = create_nav_engine("data/kafra_warp.yaml")
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

# Item IDs
FLY_WING_ITEM_ID = "601"
BUTTERFLY_WING_ITEM_ID = "602"

# Danger thresholds
DANGER_HIGH = 0.8       # Emergency — use fly wing immediately
DANGER_MEDIUM = 0.5     # Consider fly wing
DANGER_LOW = 0.2        # Safe to walk

# HP thresholds
HP_CRITICAL = 0.30      # <30% HP — emergency butterfly wing
HP_DANGER = 0.50        # <50% HP — return to save point
HP_LOW = 0.70           # <70% HP — caution, monitor
HP_SAFE = 0.95          # >=95% HP — healthy


# ── NavigationEngine ───────────────────────────────────────────────────────

class NavigationEngine:
    """Intelligent RO navigation using Kafra, Fly Wings, Butterfly Wings, and walking.

    All routing data is loaded from YAML. The engine is thread-safe via an RLock
    that protects all public methods.

    Typical usage::

        engine = NavigationEngine("data/kafra_warp.yaml")

        # Direct Kafra warp
        route = engine.kafra_route("prontera", "payon")
        # => [{"command": "warp payon", "cost": 200, "map": "payon", ...}]

        # Multi-hop via Kafra network
        route = engine.kafra_route("einbroch", "payon")
        # => [{"command": "warp prontera", ...}, {"command": "warp payon", ...}]

        # Fly Wing decision
        decision = engine.fly_wing_usage("pay_fild01", danger_level=0.7)
        # => {"action": "use_fly_wing", "confidence": 0.85, ...}

        # Butterfly Wing decision
        decision = engine.butterfly_wing_return(hp_pct=0.25)
        # => {"action": "use_butterfly_wing", "confidence": 0.99, ...}

        # Best overall route
        result = engine.best_route(
            "prontera", "pay_dun00",
            zeny=5000, has_fly_wings=True, has_butterfly_wings=True,
            hp_pct=0.9, danger_level=0.3,
        )
        # => {"method": "kafra_warp", "route": [...], "cost": 200, "reason": "..."}
    """

    def __init__(self, data_path: str | Path) -> None:
        self._lock = threading.RLock()
        self._data_path = Path(data_path)
        self._kafra_data: dict[str, Any] = {}
        # index: source_map -> {dest_map: {map, cost, label}}
        self._kafra_index: dict[str, dict[str, dict[str, Any]]] = {}
        self._load_data()

    # ── Data loading ──────────────────────────────────────────────────────

    def _load_data(self) -> None:
        """Load Kafra warp data from the configured YAML file."""
        path = self._data_path
        if not path.exists():
            logger.warning("Kafra data file not found: %s", path)
            self._kafra_data = {"kafras": {}}
            self._kafra_index = {}
            return

        try:
            with open(path, "r") as f:
                self._kafra_data = yaml.safe_load(f) or {"kafras": {}}
            self._build_index()
            logger.info(
                "NavigationEngine: loaded %d Kafra locations from %s",
                len(self._kafra_index),
                path,
            )
        except Exception as exc:
            logger.error("Failed to load Kafra data from %s: %s", path, exc)
            self._kafra_data = {"kafras": {}}
            self._kafra_index = {}

    def _build_index(self) -> None:
        """Build the Kafra destination lookup index from raw YAML data."""
        index: dict[str, dict[str, dict[str, Any]]] = {}
        kafras = self._kafra_data.get("kafras", {})
        for src_map, info in kafras.items():
            dests: dict[str, dict[str, Any]] = {}
            for dest_key, dest_info in info.get("destinations", {}).items():
                dests[dest_info["map"]] = {
                    "map": dest_info["map"],
                    "cost": dest_info.get("cost", 200),
                    "label": dest_info.get("label", dest_info["map"]),
                }
            index[src_map] = dests
        self._kafra_index = index

    # ── Kafra warp routing ────────────────────────────────────────────────

    def kafra_route(self, from_map: str, to_map: str) -> list[dict[str, Any]]:
        """Find a Kafra warp route from *from_map* to *to_map*.

        Supports direct single-hop warps and multi-hop routing via BFS
        through the Kafra NPC network.

        Returns:
            A list of warp command dictionaries, one per hop.
            Empty list if no route exists.

        Each dict contains::

            {"command": "warp payon", "map": "payon",
             "cost": 200, "label": "Payon (200z)",
             "hop": 0, "source": "prontera"}
        """
        from_map = from_map.lower().strip()
        to_map = to_map.lower().strip()

        with self._lock:
            # Direct single-hop warp
            if from_map in self._kafra_index:
                dests = self._kafra_index[from_map]
                if to_map in dests:
                    dest = dests[to_map]
                    return [
                        {
                            "command": f"warp {to_map}",
                            "map": to_map,
                            "cost": dest["cost"],
                            "label": dest["label"],
                            "hop": 0,
                            "source": from_map,
                        }
                    ]

            # Multi-hop via BFS
            return self._bfs_kafra_route(from_map, to_map)

    def _bfs_kafra_route(
        self, from_map: str, to_map: str
    ) -> list[dict[str, Any]]:
        """BFS through the Kafra network to find a multi-hop route.

        Uses breadth-first search to find the path with the fewest
        Kafra warp hops between two maps.
        """
        if from_map not in self._kafra_index:
            return []

        visited: set[str] = {from_map}
        queue: list[tuple[str, list[dict[str, Any]]]] = [(from_map, [])]

        while queue:
            current, path = queue.pop(0)

            if current not in self._kafra_index:
                continue

            for dest_map, dest in self._kafra_index[current].items():
                if dest_map in visited:
                    continue

                step = {
                    "command": f"warp {dest_map}",
                    "map": dest_map,
                    "cost": dest["cost"],
                    "label": dest["label"],
                    "hop": len(path),
                    "source": current,
                }
                new_path = path + [step]

                if dest_map == to_map:
                    return new_path

                # Only enqueue if we can continue from the destination
                if dest_map in self._kafra_index:
                    visited.add(dest_map)
                    queue.append((dest_map, new_path))

            visited.add(current)

        return []

    # ── Fly Wing decisions ────────────────────────────────────────────────

    def fly_wing_usage(
        self, current_map: str, danger_level: float
    ) -> dict[str, Any]:
        """Determine whether to use a Fly Wing based on the danger level.

        Args:
            current_map: The map the bot is currently on (reserved for
                future map-specific logic).
            danger_level: A float from 0.0 (completely safe) to 1.0
                (extremely dangerous), typically derived from monster
                density, aggro radius, and HP.

        Returns:
            A dict with keys:

            - ``action``: ``"use_fly_wing"``, ``"consider_fly_wing"``,
              or ``"continue_walking"``
            - ``confidence``: 0.0–1.0 confidence in the recommendation
            - ``reason``: Human-readable explanation
            - ``command``: The action command (``"flywing"`` or ``""``)
        """
        _ = current_map  # Reserved for future map-specific danger data

        with self._lock:
            if danger_level >= DANGER_HIGH:
                return {
                    "action": "use_fly_wing",
                    "confidence": 0.85,
                    "reason": (
                        f"Danger level is {danger_level:.0%} — high threat. "
                        "Using Fly Wing to immediately reposition away from danger."
                    ),
                    "command": "flywing",
                }

            if danger_level >= DANGER_MEDIUM:
                return {
                    "action": "consider_fly_wing",
                    "confidence": 0.60,
                    "reason": (
                        f"Danger level is {danger_level:.0%} — moderate threat. "
                        "Fly Wing recommended to skip monster packs and reposition."
                    ),
                    "command": "flywing",
                }

            return {
                "action": "continue_walking",
                "confidence": 0.90,
                "reason": (
                    f"Danger level is {danger_level:.0%} — safe to walk."
                ),
                "command": "",
            }

    # ── Butterfly Wing decisions ──────────────────────────────────────────

    def butterfly_wing_return(
        self,
        hp_pct: float,
        is_lost: bool = False,
    ) -> dict[str, Any]:
        """Determine whether to use a Butterfly Wing to return to the save point.

        Args:
            hp_pct: Current HP as a fraction of maximum HP (0.0 to 1.0).
            is_lost: Whether the bot is lost (no path to target found).

        Returns:
            A dict with keys:

            - ``action``: ``"use_butterfly_wing"``, ``"return_to_save"``,
              ``"consider_return"``, or ``"continue"``
            - ``confidence``: 0.0–1.0 confidence in the recommendation
            - ``reason``: Human-readable explanation
            - ``command``: The action command (``"butterfly"`` or ``""``)
        """
        with self._lock:
            # Emergency: critically low HP
            if hp_pct < HP_CRITICAL:
                return {
                    "action": "use_butterfly_wing",
                    "confidence": 0.99,
                    "reason": (
                        f"HP critically low ({hp_pct:.0%}). "
                        "Emergency return to save point via Butterfly Wing."
                    ),
                    "command": "butterfly",
                }

            # Danger: low HP, recommend return to recover
            if hp_pct < HP_DANGER:
                return {
                    "action": "use_butterfly_wing",
                    "confidence": 0.80,
                    "reason": (
                        f"HP is low ({hp_pct:.0%}). "
                        "Returning to save point via Butterfly Wing to recover safely."
                    ),
                    "command": "butterfly",
                }

            # Lost: no path to target exists
            if is_lost:
                return {
                    "action": "return_to_save",
                    "confidence": 0.70,
                    "reason": (
                        "Bot is lost (no path to target). "
                        "Returning to save point via Butterfly Wing to re-orient."
                    ),
                    "command": "butterfly",
                }

            # Caution: moderately low HP
            if hp_pct < HP_LOW:
                return {
                    "action": "consider_return",
                    "confidence": 0.40,
                    "reason": (
                        f"HP is moderate ({hp_pct:.0%}). "
                        "Continue but monitor HP — Butterfly Wing available if needed."
                    ),
                    "command": "",
                }

            # Safe
            return {
                "action": "continue",
                "confidence": 0.95,
                "reason": (
                    f"HP is healthy ({hp_pct:.0%}). No Butterfly Wing needed."
                ),
                "command": "",
            }

    # ── Walk routing (fallback) ───────────────────────────────────────────

    def walk_route(self, from_map: str, to_map: str) -> list[dict[str, Any]]:
        """Generate a walking route as a fallback navigation strategy.

        For full portal-based pathfinding with Dijkstra over 115+ real RO
        portal connections, use ``Pathfinder`` from
        ``ai_sidecar.domains.navigation.pathfinding`` instead.

        This method provides a basic route indication: if a Kafra exists
        on the source map it suggests moving there first, then walking
        to the target from the nearest hub.

        Returns:
            A list of move command dicts. Returns a single "already at
            target" entry when *from_map* equals *to_map*.
        """
        from_map = from_map.lower().strip()
        to_map = to_map.lower().strip()

        with self._lock:
            if from_map == to_map:
                return [
                    {
                        "command": "",
                        "map": from_map,
                        "action": "already_at_target",
                        "reason": "Already at target map.",
                    }
                ]

            route: list[dict[str, Any]] = []

            # If there is a Kafra on the source map, suggest moving to it
            if from_map in self._kafra_index:
                route.append(
                    {
                        "command": f"move {from_map}",
                        "map": from_map,
                        "action": "move_to_kafra",
                        "reason": f"Move to Kafra NPC on {from_map}.",
                    }
                )

            # Check if Kafra network can get us closer
            kafra_approach = self.kafra_route(from_map, to_map)
            if kafra_approach:
                via = kafra_approach[-1]["map"]
                route.append(
                    {
                        "command": f"walk {to_map}",
                        "map": to_map,
                        "action": "walk_to_target",
                        "reason": f"Walk from {via} to {to_map}.",
                        "via_kafra": via,
                    }
                )
            else:
                route.append(
                    {
                        "command": f"move {to_map}",
                        "map": to_map,
                        "action": "walk_to_target",
                        "reason": f"Walk from {from_map} to {to_map}.",
                    }
                )

            return route

    # ── Best-route selection ──────────────────────────────────────────────

    def best_route(
        self,
        from_map: str,
        to_map: str,
        zeny: int = 0,
        has_fly_wings: bool = False,
        has_butterfly_wings: bool = False,
        hp_pct: float = 1.0,
        danger_level: float = 0.0,
    ) -> dict[str, Any]:
        """Select the best travel method considering all options and constraints.

        Decision order:
        1. Already at target → walk (no-op)
        2. HP critical + has Butterfly Wing → butterfly wing escape
        3. High danger + has Fly Wing → fly wing reposition
        4. Kafra warp available and affordable → kafra warp
        5. Fallback → walk route

        Args:
            from_map: Current map name (e.g. ``"prontera"``).
            to_map: Target map name (e.g. ``"pay_dun00"``).
            zeny: Current zeny balance.
            has_fly_wings: Whether the bot has Fly Wings in inventory.
            has_butterfly_wings: Whether the bot has Butterfly Wings in inventory.
            hp_pct: Current HP as a fraction of max HP (0.0 to 1.0).
            danger_level: Current danger level on the map (0.0 to 1.0).

        Returns:
            A dict with keys:

            - ``method``: One of ``"kafra_warp"``, ``"fly_wing"``,
              ``"butterfly_wing"``, or ``"walk"``
            - ``route``: List of command dicts to execute
            - ``cost``: Total zeny cost (0 for free methods)
            - ``reason``: Human-readable explanation of the decision
        """
        from_map = from_map.lower().strip()
        to_map = to_map.lower().strip()

        with self._lock:
            # ── 1. Already at target ──────────────────────────────────────
            if from_map == to_map:
                return {
                    "method": "walk",
                    "route": [
                        {
                            "command": "",
                            "map": from_map,
                            "action": "already_at_target",
                        }
                    ],
                    "cost": 0,
                    "reason": "Already at target map.",
                }

            # ── 2. Emergency: critically low HP → Butterfly Wing ──────────
            if hp_pct < HP_CRITICAL and has_butterfly_wings:
                bw = self.butterfly_wing_return(hp_pct)
                return {
                    "method": "butterfly_wing",
                    "route": [
                        {
                            "command": bw["command"],
                            "action": "butterfly_wing",
                            "reason": bw["reason"],
                        }
                    ],
                    "cost": 0,
                    "reason": bw["reason"],
                }

            # ── 3. High danger with Fly Wing available ────────────────────
            if danger_level >= DANGER_MEDIUM and has_fly_wings:
                fw = self.fly_wing_usage(from_map, danger_level)
                if fw["action"] in ("use_fly_wing", "consider_fly_wing"):
                    return {
                        "method": "fly_wing",
                        "route": [
                            {
                                "command": fw["command"],
                                "action": "fly_wing",
                                "reason": fw["reason"],
                            }
                        ],
                        "cost": 0,
                        "reason": fw["reason"],
                    }

            # ── 4. Kafra warp (if available and affordable) ───────────────
            kafra = self.kafra_route(from_map, to_map)
            if kafra:
                total_cost = sum(step["cost"] for step in kafra)
                if total_cost <= zeny:
                    return {
                        "method": "kafra_warp",
                        "route": kafra,
                        "cost": total_cost,
                        "reason": (
                            f"Kafra warp from {from_map} to {to_map} "
                            f"({total_cost}z) — fastest available method."
                        ),
                    }

                # Kafra available but not affordable — note the shortfall
                return {
                    "method": "kafra_warp",
                    "route": kafra,
                    "cost": total_cost,
                    "reason": (
                        f"Kafra warp from {from_map} to {to_map} "
                        f"costs {total_cost}z but you have {zeny}z. "
                        f"Shortfall: {total_cost - zeny}z."
                    ),
                }

            # ── 5. Fallback: walk ─────────────────────────────────────────
            walk = self.walk_route(from_map, to_map)
            return {
                "method": "walk",
                "route": walk,
                "cost": 0,
                "reason": (
                    f"Walking from {from_map} to {to_map} — "
                    "no faster method available or affordable."
                ),
            }

    # ── Utilities ─────────────────────────────────────────────────────────

    def reload(self) -> None:
        """Reload Kafra warp data from the YAML file (hot-reload support)."""
        self._load_data()

    def get_kafra_locations(self) -> list[str]:
        """Return a sorted list of all maps that have a Kafra NPC."""
        with self._lock:
            return sorted(self._kafra_index.keys())

    def get_kafra_destinations(self, map_name: str) -> list[dict[str, Any]]:
        """Return all destinations reachable from a Kafra on *map_name*.

        Returns an empty list if there is no Kafra on that map.
        """
        with self._lock:
            map_name = map_name.lower().strip()
            dests = self._kafra_index.get(map_name, {})
            return [
                {"map": d["map"], "cost": d["cost"], "label": d["label"]}
                for d in dests.values()
            ]

    def is_kafra_map(self, map_name: str) -> bool:
        """Check if *map_name* has a Kafra NPC."""
        with self._lock:
            return map_name.lower().strip() in self._kafra_index

    def __repr__(self) -> str:
        return (
            f"<NavigationEngine: {len(self._kafra_index)} Kafra locations, "
            f"data={self._data_path}>"
        )


# ── Factory function ───────────────────────────────────────────────────────

def create_nav_engine(data_path: str | Path) -> NavigationEngine:
    """Create and return a fully initialised NavigationEngine.

    Args:
        data_path: Path to the Kafra warp YAML file (e.g. ``"data/kafra_warp.yaml"``).

    Returns:
        A ready-to-use NavigationEngine instance.
    """
    return NavigationEngine(data_path)
