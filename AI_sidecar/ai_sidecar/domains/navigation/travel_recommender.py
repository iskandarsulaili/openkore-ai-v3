"""
TravelRecommender — optimal RO travel method selection.

Integrates all real RO fast-travel systems:
  - Walk (portal-based pathfinding via PortalDB + Pathfinder)
  - Kafra Warp (NPC teleport between major towns)
  - Fly Wing (Item 601 — random teleport within current map)
  - Butterfly Wing (Item 602 — teleport to save point)
  - Airship / Ferry / Train (intercontinental connections)

Decision logic considers distance, cost, urgency (HP), inventory,
and current location to recommend the best travel method.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ai_sidecar.actions import HeuristicAction
from ai_sidecar.domains.navigation.pathfinding import Pathfinder, get_pathfinder

logger = logging.getLogger(__name__)

# ── Item constants ──────────────────────────────────────────────────────
FLY_WING_ITEM_ID = "601"
FLY_WING_ITEM_NAME = "Fly Wing"
BUTTERFLY_WING_ITEM_ID = "602"
BUTTERFLY_WING_ITEM_NAME = "Butterfly Wing"

# ── NPC price defaults (iRO / rAthena standard) ─────────────────────────
FLY_WING_NPC_PRICE = 500     # Most tool dealers / general stores
BUTTERFLY_WING_NPC_PRICE = 500  # General stores
WING_BUY_MINIMUM = 5        # Always keep at least this many wings on hand
WING_BUY_TARGET = 20        # Restock up to this many wings
WING_BUY_THRESHOLD = 10     # Buy more when below this count

# ── Escape thresholds (HP ratio) ────────────────────────────────────────
HP_CRITICAL = 0.30    # < 30% HP — emergency butterfly wing escape
HP_DANGER = 0.50      # < 50% HP — consider butterfly wing
HP_CAUTION = 0.80     # < 80% HP — consider fly wing
HP_SAFE = 0.95        # >= 95% HP — no HP concern

# ── Distance thresholds (map hops via Pathfinder) ───────────────────────
SAME_MAP_HOPS = 0          # Already on target map
ADJACENT_MAP_HOPS = 1      # One portal crossing
SHORT_DISTANCE_HOPS = 2    # Two portal crossings
MEDIUM_DISTANCE_HOPS = 5   # 3-5 hops: Kafra warp viable
LONG_DISTANCE_HOPS = 10    # 6+ hops: consider airship

# ── Cost-benefit thresholds ─────────────────────────────────────────────
MAX_AFFORDABLE_FRACTION = 0.10  # Don't spend more than 10% of zeny on warp
MIN_ZENY_FOR_WARP = 2000       # Minimum zeny to consider paid travel
AIRSHIP_MIN_ZENY = 5000        # Minimum zeny to consider airship

# ── DATA FILE PATHS ─────────────────────────────────────────────────────
_KAFRA_DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "kafra_warp.yaml"
_AIRSHIP_DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "airship_routes.yaml"

# Fallback: also check relative to AI_sidecar/
_ALT_KAFRA_PATH = Path(os.getcwd()) / "data" / "kafra_warp.yaml"
_ALT_AIRSHIP_PATH = Path(os.getcwd()) / "data" / "airship_routes.yaml"


# ── Data classes ────────────────────────────────────────────────────────

@dataclass(slots=True, frozen=True)
class KafraDestination:
    """A destination reachable from a specific Kafra NPC."""
    map: str
    cost: int
    label: str


@dataclass(slots=True, frozen=True)
class AirshipRoute:
    """An airship/ferry/train route between two towns."""
    from_map: str
    to_map: str
    cost: int
    label: str
    route_type: str  # "airship" | "ferry" | "train"


@dataclass(slots=True)
class TravelRecommendation:
    """The recommended travel method with supporting info."""
    method: str          # "walk" | "kafra_warp" | "fly_wing" | "butterfly_wing" | "airship"
    command: str         # The actual command to execute
    confidence: float    # 0.0 - 1.0
    reason: str          # Human-readable explanation
    cost: int            # Zeny cost (0 for free methods)
    metadata: dict[str, Any] | None = None

    def to_heuristic_action(self) -> HeuristicAction:
        """Convert this recommendation to a HeuristicAction for the PDCA loop."""
        if self.method == "walk" and not self.command:
            # No-op (already at target)
            return HeuristicAction(
                kind="command", command="",
                confidence=self.confidence, domain="navigation",
                reason=self.reason, metadata=self.metadata or {},
            )
        return HeuristicAction(
            kind="command",
            command=self.command,
            confidence=self.confidence,
            domain="navigation",
            reason=self.reason,
            metadata={
                "travel_method": self.method,
                "cost": self.cost,
                **(self.metadata or {}),
            },
        )



# ── TravelRecommender ───────────────────────────────────────────────────

class TravelRecommender:
    """Recommends the optimal RO travel method given current state.

    Integrates walking (Pathfinder), Kafra warp, Fly Wing, Butterfly Wing,
    and airship travel into a single recommendation system.

    Usage:
        recommender = TravelRecommender()
        recommendation = recommender.recommend(
            current_map="prontera",
            target_map="pay_dun00",
            current_hp=450,
            max_hp=500,
            zeny=15000,
            inventory={"601": 5, "602": 1},
            job_name="archer",
            base_level=25,
        )
        # => TravelRecommendation(method="kafra_warp", command="warp payon", ...)
    """

    def __init__(
        self,
        pathfinder: Pathfinder | None = None,
        kafra_data_path: str | Path | None = None,
        airship_data_path: str | Path | None = None,
    ) -> None:
        self._pathfinder = pathfinder or get_pathfinder()
        self._kafra_data: dict[str, Any] = {}
        self._airship_data: dict[str, Any] = {}

        # Load Kafra warp data
        kafra_path = self._resolve_data_path(
            kafra_data_path, _KAFRA_DATA_PATH, _ALT_KAFRA_PATH
        )
        self._load_kafra_data(kafra_path)

        # Load airship data
        airship_path = self._resolve_data_path(
            airship_data_path, _AIRSHIP_DATA_PATH, _ALT_AIRSHIP_PATH
        )
        self._load_airship_data(airship_path)

        # Build lookup indexes
        self._kafra_destinations: dict[str, dict[str, KafraDestination]] = {}
        self._build_kafra_index()

        self._airship_routes: dict[str, dict[str, AirshipRoute]] = {}
        self._build_airship_index()

        logger.info(
            "TravelRecommender initialized: %d Kafra locations, %d airship routes",
            len(self._kafra_destinations),
            sum(len(r) for r in self._airship_routes.values()),
        )

    # ── Data loading ─────────────────────────────────────────────────

    @staticmethod
    def _resolve_data_path(
        explicit: str | Path | None,
        primary: Path,
        fallback: Path,
    ) -> Path:
        """Resolve the data file path from explicit arg, primary path, or fallback."""
        if explicit:
            return Path(explicit)
        if primary.exists():
            return primary
        if fallback.exists():
            return fallback
        # Last resort: use primary even if missing (will produce clear error later)
        return primary

    def _load_kafra_data(self, path: Path) -> None:
        """Load Kafra warp data from YAML."""
        try:
            if not path.exists():
                logger.warning("Kafra data file not found: %s", path)
                self._kafra_data = {"kafras": {}}
                return
            with open(path, "r") as f:
                self._kafra_data = yaml.safe_load(f) or {"kafras": {}}
            count = len(self._kafra_data.get("kafras", {}))
            logger.info("Loaded Kafra data: %d Kafra NPCs from %s", count, path)
        except Exception as e:
            logger.error("Failed to load Kafra data from %s: %s", path, e)
            self._kafra_data = {"kafras": {}}

    def _load_airship_data(self, path: Path) -> None:
        """Load airship route data from YAML."""
        try:
            if not path.exists():
                logger.warning("Airship data file not found: %s", path)
                self._airship_data = {"airship_routes": {}, "port_cities": {}}
                return
            with open(path, "r") as f:
                self._airship_data = yaml.safe_load(f) or {"airship_routes": {}, "port_cities": {}}
            count = len(self._airship_data.get("airship_routes", {}))
            logger.info("Loaded airship data: %d routes from %s", count, path)
        except Exception as e:
            logger.error("Failed to load airship data from %s: %s", path, e)
            self._airship_data = {"airship_routes": {}, "port_cities": {}}

    def _build_kafra_index(self) -> None:
        """Build a lookup index from source_map -> {dest_map: KafraDestination}."""
        kafras = self._kafra_data.get("kafras", {})
        for src_map, info in kafras.items():
            dests: dict[str, KafraDestination] = {}
            for dest_key, dest_info in info.get("destinations", {}).items():
                dests[dest_info["map"]] = KafraDestination(
                    map=dest_info["map"],
                    cost=dest_info.get("cost", 0),
                    label=dest_info.get("label", dest_info["map"]),
                )
            self._kafra_destinations[src_map] = dests

    def _build_airship_index(self) -> None:
        """Build lookup index from departure_map -> {arrival_map: AirshipRoute}."""
        routes = self._airship_data.get("airship_routes", {})
        for route_key, info in routes.items():
            from_map = info["from"]
            to_map = info["to"]
            route = AirshipRoute(
                from_map=from_map,
                to_map=to_map,
                cost=info.get("cost", 0),
                label=info.get("label", f"{from_map} -> {to_map}"),
                route_type=info.get("route_type", "airship"),
            )
            if from_map not in self._airship_routes:
                self._airship_routes[from_map] = {}
            self._airship_routes[from_map][to_map] = route

    # ── Public API ───────────────────────────────────────────────────

    def recommend(
        self,
        current_map: str,
        target_map: str,
        current_hp: int = 1,
        max_hp: int = 1,
        zeny: int = 0,
        inventory: dict[str, int] | None = None,
        job_name: str = "novice",
        base_level: int = 1,
    ) -> TravelRecommendation:
        """Recommend the optimal travel method from current_map to target_map.

        Args:
            current_map: Map the bot is currently on (e.g., "prontera").
            target_map: Map the bot wants to reach (e.g., "pay_dun00").
            current_hp: Current HP value.
            max_hp: Maximum HP value.
            zeny: Current zeny amount.
            inventory: Dict mapping item_id -> count (e.g., {"601": 5, "602": 1}).
            job_name: Character's job class (for context/logging).
            base_level: Character's base level.

        Returns:
            TravelRecommendation with the optimal method.
        """
        inv = inventory or {}
        current_map = current_map.lower().strip()
        target_map = target_map.lower().strip()
        hp_ratio = current_hp / max_hp if max_hp > 0 else 1.0

        # ── 1. EMERGENCY: Check if we need to escape ──
        if hp_ratio < HP_CRITICAL:
            bwing_count = inv.get(BUTTERFLY_WING_ITEM_ID, 0)
            if bwing_count > 0:
                return TravelRecommendation(
                    method="butterfly_wing",
                    command="butterfly",
                    confidence=0.99,
                    reason=(
                        f"HP critically low ({current_hp}/{max_hp}, {hp_ratio:.0%}). "
                        "Emergency escape to save point via Butterfly Wing."
                    ),
                    cost=0,
                    metadata={
                        "item_id": BUTTERFLY_WING_ITEM_ID,
                        "inventory_after": bwing_count - 1,
                    },
                )
            # No butterfly wing — try fly wing as emergency
            fwing_count = inv.get(FLY_WING_ITEM_ID, 0)
            if fwing_count > 0:
                return TravelRecommendation(
                    method="fly_wing",
                    command="flywing",
                    confidence=0.90,
                    reason=(
                        f"HP critically low ({current_hp}/{max_hp}, {hp_ratio:.0%}). "
                        "No Butterfly Wing — using Fly Wing to escape danger."
                    ),
                    cost=0,
                    metadata={
                        "item_id": FLY_WING_ITEM_ID,
                        "inventory_after": fwing_count - 1,
                    },
                )

        # ── 2. SAME MAP: Already at target ──
        if current_map == target_map:
            rec = TravelRecommendation(
                method="walk",
                command="",
                confidence=0.99,
                reason="Already at target map.",
                cost=0,
            )
            # Check if wings need restocking while in town
            if self._is_town_map(current_map):
                wing_rec = self._check_wing_supplies(inv, zeny)
                if wing_rec:
                    rec.metadata = {"restock_wings": {
                        "command": wing_rec.command,
                        "cost": wing_rec.cost,
                        "reason": wing_rec.reason,
                    }}
                    rec.reason += f" {wing_rec.reason}"
            return rec

        # ── 3. ESTIMATE WALK DISTANCE ──
        walk_hops = self._estimate_walk_hops(current_map, target_map)

        # ── 4. EMERGENCY (DANGER): HP < 50%, consider butterfly wing ──
        if hp_ratio < HP_DANGER:
            bwing_count = inv.get(BUTTERFLY_WING_ITEM_ID, 0)
            if bwing_count > 0 and walk_hops > ADJACENT_MAP_HOPS:
                return TravelRecommendation(
                    method="butterfly_wing",
                    command="butterfly",
                    confidence=0.80,
                    reason=(
                        f"HP below 50% ({current_hp}/{max_hp}). "
                        f"{walk_hops} map hops to target — using Butterfly Wing "
                        "to return to save point and recover."
                    ),
                    cost=0,
                    metadata={
                        "item_id": BUTTERFLY_WING_ITEM_ID,
                        "inventory_after": bwing_count - 1,
                    },
                )

        # ── 5. CHECK AIRSHIP ROUTES (long distance, or adjacent to port hub) ──
        if walk_hops >= LONG_DISTANCE_HOPS and zeny >= AIRSHIP_MIN_ZENY:
            airship_rec = self._check_airship(current_map, target_map, zeny)
            if airship_rec:
                return airship_rec

        # ── 6. CHECK KAFRA WARP (short and medium distance) ──
        # Let _check_kafra_warp decide affordability internally
        if walk_hops >= ADJACENT_MAP_HOPS:
            kafra_rec = self._check_kafra_warp(current_map, target_map, zeny)
            if kafra_rec:
                return kafra_rec

        # ── 7. CHECK FLY WING (short range, skip packs) ──
        if hp_ratio < HP_CAUTION and walk_hops <= SHORT_DISTANCE_HOPS:
            fwing_count = inv.get(FLY_WING_ITEM_ID, 0)
            if fwing_count > 0:
                return TravelRecommendation(
                    method="fly_wing",
                    command="flywing",
                    confidence=0.60,
                    reason=(
                        f"HP at {hp_ratio:.0%} with {walk_hops} map hop(s) to target. "
                        "Using Fly Wing to skip monster packs and reach portal."
                    ),
                    cost=0,
                    metadata={"item_id": FLY_WING_ITEM_ID},
                )

        # ── 8. DEFAULT: Walk using Pathfinder ──
        walk_actions = self._build_walk_actions(current_map, target_map)
        rec = TravelRecommendation(
            method="walk",
            command=walk_actions[0].command if walk_actions else f"move {target_map}",
            confidence=0.80,
            reason=(
                f"Walking from {current_map} to {target_map} "
                f"({walk_hops} map hops) — no faster method available."
            ),
            cost=0,
            metadata={
                "walk_hops": walk_hops,
                "total_actions": len(walk_actions),
            },
        )

        # ── 9. SUPPLEMENTARY: Check if wings need restocking (in town) ──
        if self._is_town_map(current_map):
            wing_rec = self._check_wing_supplies(inv, zeny)
            if wing_rec:
                # Prepend wing restock as metadata — caller can execute before travel
                rec.metadata = dict(rec.metadata or {})
                rec.metadata["restock_wings"] = {
                    "command": wing_rec.command,
                    "cost": wing_rec.cost,
                    "reason": wing_rec.reason,
                }
                rec.reason += f" Also: {wing_rec.reason}"

        return rec

    # ── Travel method checks ─────────────────────────────────────────

    def _check_kafra_warp(
        self,
        current_map: str,
        target_map: str,
        zeny: int,
    ) -> TravelRecommendation | None:
        """Check if Kafra warp is available and affordable.

        Returns a recommendation if a multi-hop Kafra chain can reach the target,
        or if a single Kafra hop is useful for getting closer.
        """
        if current_map not in self._kafra_destinations:
            return None

        available = self._kafra_destinations[current_map]

        # Direct Kafra warp to target
        if target_map in available:
            dest = available[target_map]
            if zeny >= dest.cost and dest.cost <= zeny * MAX_AFFORDABLE_FRACTION or dest.cost == 0:
                return TravelRecommendation(
                    method="kafra_warp",
                    command=f"warp {target_map}",
                    confidence=0.95,
                    reason=(
                        f"Kafra warp from {current_map} to {target_map} "
                        f"({dest.label}) — affordable at {dest.cost}z."
                    ),
                    cost=dest.cost,
                    metadata={
                        "source_map": current_map,
                        "target_map": target_map,
                        "cost": dest.cost,
                    },
                )

        # Check if Kafra can get us closer — intermediate hop
        # Find the Kafra destination that's closest to target_map
        best_dest = None
        best_remaining_hops = 999
        for dest_map, dest in available.items():
            if dest.cost > zeny * MAX_AFFORDABLE_FRACTION and dest.cost > 0:
                continue
            if dest.cost > zeny:
                continue
            remaining = self._estimate_walk_hops(dest_map, target_map)
            if remaining < best_remaining_hops:
                best_remaining_hops = remaining
                best_dest = dest

        if best_dest and best_dest.map != target_map:
            current_remaining = self._estimate_walk_hops(current_map, target_map)
            if best_remaining_hops < current_remaining:
                return TravelRecommendation(
                    method="kafra_warp",
                    command=f"warp {best_dest.map}",
                    confidence=0.75,
                    reason=(
                        f"Kafra warp from {current_map} to {best_dest.map} "
                        f"({best_dest.cost}z) reduces walk from "
                        f"{current_remaining} to {best_remaining_hops} map hops."
                    ),
                    cost=best_dest.cost,
                    metadata={
                        "source_map": current_map,
                        "intermediate_map": best_dest.map,
                        "target_map": target_map,
                        "cost": best_dest.cost,
                        "remaining_hops": best_remaining_hops,
                    },
                )

        return None

    def _check_airship(
        self,
        current_map: str,
        target_map: str,
        zeny: int,
    ) -> TravelRecommendation | None:
        """Check if an airship/ferry/train route is available.

        Checks direct routes from current port, then walks to nearest port
        if on a nearby map.
        """
        # Direct route from current port
        if current_map in self._airship_routes:
            routes = self._airship_routes[current_map]
            if target_map in routes:
                route = routes[target_map]
                if zeny >= route.cost:
                    return TravelRecommendation(
                        method="airship",
                        command=f"airship {target_map}",
                        confidence=0.95,
                        reason=(
                            f"Airship from {current_map} to {target_map} "
                            f"({route.label}) — {route.cost}z, {route.route_type}."
                        ),
                        cost=route.cost,
                        metadata={
                            "source_map": current_map,
                            "target_map": target_map,
                            "cost": route.cost,
                            "route_type": route.route_type,
                        },
                    )

            # Check if going via airship hub is better
            # e.g., if on alberta and need to reach einbroch but only to yuno exists
            for dest_map, route in routes.items():
                if zeny < route.cost:
                    continue
                remaining = self._estimate_walk_hops(dest_map, target_map)
                current_remaining = self._estimate_walk_hops(current_map, target_map)
                if remaining < current_remaining:
                    return TravelRecommendation(
                        method="airship",
                        command=f"airship {dest_map}",
                        confidence=0.70,
                        reason=(
                            f"Airship from {current_map} to {dest_map} "
                            f"({route.label}) reduces walk from "
                            f"{current_remaining} to {remaining} map hops."
                        ),
                        cost=route.cost,
                        metadata={
                            "source_map": current_map,
                            "intermediate": dest_map,
                            "target_map": target_map,
                            "cost": route.cost,
                            "route_type": route.route_type,
                            "remaining_hops": remaining,
                        },
                    )

        # Check if walking to a nearby port city is worthwhile
        port_cities_data = self._airship_data.get("port_cities", {})
        best_port_rec: TravelRecommendation | None = None
        best_total_hops = 999

        for port_map, port_info in port_cities_data.items():
            if port_map == current_map:
                continue
            # How far to walk to this port?
            walk_to_port = self._estimate_walk_hops(current_map, port_map)
            if walk_to_port > MEDIUM_DISTANCE_HOPS:
                continue  # Too far to walk to port

            # Check if port has a route that helps
            port_routes = self._airship_routes.get(port_map, {})
            for dest_map, route in port_routes.items():
                if zeny < route.cost:
                    continue
                remaining = self._estimate_walk_hops(dest_map, target_map)
                total_hops = walk_to_port + remaining
                if total_hops < best_total_hops:
                    current_remaining = self._estimate_walk_hops(current_map, target_map)
                    if total_hops < current_remaining:
                        best_total_hops = total_hops
                        best_port_rec = TravelRecommendation(
                            method="walk",
                            command=f"move {port_map}",
                            confidence=0.65,
                            reason=(
                                f"Walk from {current_map} to {port_map} port "
                                f"({walk_to_port} hops), then airship to {dest_map} "
                                f"({route.label}) — total {total_hops} hops "
                                f"vs {current_remaining} walking all the way."
                            ),
                            cost=route.cost,
                            metadata={
                                "source_map": current_map,
                                "port_map": port_map,
                                "airship_destination": dest_map,
                                "target_map": target_map,
                                "cost": route.cost,
                                "walk_to_port": walk_to_port,
                                "remaining_after_airship": remaining,
                                "total_hops": total_hops,
                            },
                        )

        return best_port_rec

    def _check_wing_supplies(
        self,
        inventory: dict[str, int],
        zeny: int,
    ) -> TravelRecommendation | None:
        """Check if we need to buy more Fly Wings or Butterfly Wings."""
        fly_wings = inventory.get(FLY_WING_ITEM_ID, 0)
        bfly_wings = inventory.get(BUTTERFLY_WING_ITEM_ID, 0)

        # Need both types stocked
        needs_fly = fly_wings < WING_BUY_THRESHOLD
        needs_bfly = bfly_wings < WING_BUY_THRESHOLD

        if not needs_fly and not needs_bfly:
            return None

        # Check if we have enough zeny
        cost = 0
        buy_items: list[str] = []
        if needs_fly:
            buy_count = WING_BUY_TARGET - fly_wings
            cost += buy_count * FLY_WING_NPC_PRICE
            buy_items.append(f"{buy_count}x Fly Wing")
        if needs_bfly:
            buy_count = WING_BUY_TARGET - bfly_wings
            cost += buy_count * BUTTERFLY_WING_NPC_PRICE
            buy_items.append(f"{buy_count}x Butterfly Wing")

        if zeny < cost:
            # Can't afford full restock — buy what we can
            affordable_fly = min(
                WING_BUY_TARGET - fly_wings,
                max(0, (zeny - (WING_BUY_MINIMUM - bfly_wings) * BUTTERFLY_WING_NPC_PRICE) // FLY_WING_NPC_PRICE)
            ) if needs_fly else 0
            if affordable_fly > 0:
                return TravelRecommendation(
                    method="buy_wings",
                    command=f"buy {FLY_WING_ITEM_ID} {affordable_fly}",
                    confidence=0.50,
                    reason=(
                        f"Low on Fly Wings ({fly_wings}/{WING_BUY_TARGET}). "
                        f"Only {zeny}z — buying {affordable_fly} Fly Wings "
                        f"({affordable_fly * FLY_WING_NPC_PRICE}z)."
                    ),
                    cost=affordable_fly * FLY_WING_NPC_PRICE,
                    metadata={
                        "item_id": FLY_WING_ITEM_ID,
                        "buy_count": affordable_fly,
                        "cost_per_item": FLY_WING_NPC_PRICE,
                    },
                )
            return None  # Can't afford anything

        # Can afford full restock
        command_parts = []
        if needs_fly:
            command_parts.append(f"buy {FLY_WING_ITEM_ID} {WING_BUY_TARGET - fly_wings}")
        if needs_bfly:
            command_parts.append(f"buy {BUTTERFLY_WING_ITEM_ID} {WING_BUY_TARGET - bfly_wings}")

        return TravelRecommendation(
            method="buy_wings",
            command="; ".join(command_parts),
            confidence=0.70,
            reason=(
                f"Restocking wings — have {fly_wings} Fly Wings, "
                f"{bfly_wings} Butterfly Wings ({cost}z total)."
            ),
            cost=cost,
            metadata={
                "fly_wings_current": fly_wings,
                "bfly_wings_current": bfly_wings,
                "fly_wings_to_buy": max(0, WING_BUY_TARGET - fly_wings) if needs_fly else 0,
                "bfly_wings_to_buy": max(0, WING_BUY_TARGET - bfly_wings) if needs_bfly else 0,
            },
        )

    # ── Walking ──────────────────────────────────────────────────────

    def _has_free_warp(self, from_map: str, to_map: str) -> bool:
        """Check if there's a direct free (0 cost) Kafra warp between these maps."""
        dests = self._kafra_destinations.get(from_map, {})
        return to_map in dests and dests[to_map].cost == 0

    _TOWN_MAP_PREFIXES = (
        "prontera", "payon", "morocc", "geffen", "alberta", "izlude",
        "aldebaran", "comodo", "yuno", "einbroch", "lighthalzen",
        "hugel", "rachel", "veins", "umbala", "niflheim",
        "amatsu", "gonryun", "louyang", "ayothaya",
    )

    def _is_town_map(self, map_name: str) -> bool:
        """Check if a map is a town (has NPCs, shops, Kafra)."""
        mn = map_name.lower().strip()
        if mn in self._kafra_destinations:
            return True
        return any(mn.startswith(p) and len(mn) == len(p) for p in self._TOWN_MAP_PREFIXES)

    def _estimate_walk_hops(self, from_map: str, to_map: str) -> int:
        """Estimate the number of map hops when walking.

        Uses Pathfinder to compute portal crossings.
        Returns 999 if no path exists (effectively infinite cost).
        """
        try:
            path = self._pathfinder.find_path(from_map, to_map)
            return len(path)
        except Exception:
            return 999

    def _build_walk_actions(
        self,
        from_map: str,
        to_map: str,
    ) -> list[HeuristicAction]:
        """Build walk commands using Pathfinder.

        Returns list of HeuristicAction move commands.
        """
        from ai_sidecar.domains.navigation.actions import build_navigation_route

        actions = build_navigation_route(from_map, to_map)
        if not actions:
            # Fallback: simple map move command
            return [
                HeuristicAction(
                    kind="command",
                    command=f"move {to_map}",
                    confidence=0.60,
                    reason=f"No portal path found — direct move to {to_map}",
                    domain="navigation",
                )
            ]
        return actions

    # ── Utility ──────────────────────────────────────────────────────

    def get_kafra_destinations(self, town_map: str) -> list[KafraDestination]:
        """Get all Kafra warp destinations available from a given town."""
        dests = self._kafra_destinations.get(town_map.lower(), {})
        return list(dests.values())

    def get_airship_destinations(self, port_map: str) -> list[AirshipRoute]:
        """Get all airship destinations available from a given port."""
        routes = self._airship_routes.get(port_map.lower(), {})
        return list(routes.values())

    def list_all_ports(self) -> list[str]:
        """List all port cities in the airship network."""
        return list(self._airship_data.get("port_cities", {}).keys())

    def list_all_kafra_towns(self) -> list[str]:
        """List all towns with Kafra service."""
        return list(self._kafra_destinations.keys())

    def kafra_storage_fee(self) -> int:
        """Get the per-slot Kafra storage fee."""
        storage = self._kafra_data.get("kafra_storage", {})
        return storage.get("fee_per_slot", 30)

    def estimate_route_cost(
        self,
        current_map: str,
        target_map: str,
        zeny: int = 0,
        inventory: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Estimate cost of all available travel methods.

        Returns dict mapping method name to cost and feasibility info.
        Useful for the heuristic to make cost-benefit decisions.
        """
        inv = inventory or {}
        result: dict[str, Any] = {
            "walk": {"cost": 0, "hops": self._estimate_walk_hops(current_map, target_map)},
            "kafra_warp": None,
            "airship": None,
            "fly_wing": {"available": inv.get(FLY_WING_ITEM_ID, 0) > 0},
            "butterfly_wing": {"available": inv.get(BUTTERFLY_WING_ITEM_ID, 0) > 0},
        }

        # Kafra
        if current_map in self._kafra_destinations and target_map in self._kafra_destinations[current_map]:
            dest = self._kafra_destinations[current_map][target_map]
            result["kafra_warp"] = {"cost": dest.cost, "affordable": zeny >= dest.cost}

        # Airship
        airship_rec = self._check_airship(current_map, target_map, zeny)
        if airship_rec:
            result["airship"] = {
                "cost": airship_rec.cost,
                "route_type": airship_rec.metadata.get("route_type", "airship") if airship_rec.metadata else "airship",
                "affordable": zeny >= airship_rec.cost,
            }

        return result


# ── Global singleton ─────────────────────────────────────────────────

_recommender: TravelRecommender | None = None


def get_travel_recommender(
    pathfinder: Pathfinder | None = None,
) -> TravelRecommender:
    """Get the global TravelRecommender singleton."""
    global _recommender
    if _recommender is None:
        _recommender = TravelRecommender(pathfinder=pathfinder)
    return _recommender
