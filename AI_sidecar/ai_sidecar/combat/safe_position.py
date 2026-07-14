"""
Dynamic Safe Position Computation — computes safe positions in real-time.

A pro player knows: "If I stand on tile (120, 85) in Payon Cave, only 2
monsters can path to me because the walls funnel them." This module computes
safe positions dynamically using monster positions and aggro status.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SafePosition:
    """A computed safe position."""
    x: int
    y: int
    safety_score: float = 0.0  # 0-100, higher = safer
    monsters_can_reach: int = 0
    distance_from_aggro: float = 0.0
    is_chokepoint: bool = False
    is_corner: bool = False
    is_near_wall: bool = False
    reason: str = ""


@dataclass
class PositionSituation:
    """Current position situation."""
    my_x: int = 0
    my_y: int = 0
    map_name: str = ""
    monsters: list[dict] = field(default_factory=list)
    aggro_count: int = 0
    nearest_monster_distance: float = 999.0
    monsters_within_10: int = 0
    monsters_within_5: int = 0
    is_surrounded: bool = False
    safe_direction: tuple[int, int] = (0, 0)
    best_safe_spot: SafePosition | None = None


class SafePositionComputer:
    """Computes safe positions in real-time."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._known_corners: dict[str, list[tuple[int, int]]] = {}
        self._known_chokepoints: dict[str, list[tuple[int, int]]] = {}
        self._known_walls: dict[str, list[tuple[int, int, int, int]]] = {}  # map -> [(x1,y1,x2,y2)]
        self._load_known_geometry()

    def _load_known_geometry(self) -> None:
        """Load known map geometry for common farming maps."""
        # Prontera fields — corners and chokepoints
        self._known_corners["prt_fild01"] = [(50, 50), (350, 50), (50, 350), (350, 350)]
        self._known_corners["prt_fild02"] = [(30, 30), (370, 30), (30, 370), (370, 370)]
        self._known_corners["prt_fild03"] = [(40, 40), (360, 40), (40, 360), (360, 360)]
        self._known_corners["prt_fild04"] = [(20, 20), (380, 20), (20, 380), (380, 380)]
        self._known_corners["prt_fild05"] = [(60, 60), (340, 60), (60, 340), (340, 340)]
        self._known_corners["prt_fild06"] = [(10, 10), (390, 10), (10, 390), (390, 390)]
        self._known_corners["prt_fild07"] = [(25, 25), (375, 25), (25, 375), (375, 375)]
        self._known_corners["prt_fild08"] = [(35, 35), (365, 35), (35, 365), (365, 365)]
        self._known_corners["prt_fild09"] = [(45, 45), (355, 45), (45, 355), (355, 355)]
        self._known_corners["prt_fild10"] = [(55, 55), (345, 55), (55, 345), (345, 345)]
        self._known_corners["prt_fild11"] = [(15, 15), (385, 15), (15, 385), (385, 385)]

        # Payon Cave — tight corridors, many chokepoints
        self._known_chokepoints["pay_dun00"] = [(100, 50), (200, 50), (150, 100)]
        self._known_chokepoints["pay_dun01"] = [(80, 80), (180, 80), (120, 120), (200, 120)]
        self._known_chokepoints["pay_dun02"] = [(60, 60), (160, 60), (100, 100), (180, 100)]
        self._known_chokepoints["pay_dun03"] = [(50, 50), (150, 50), (90, 90), (170, 90)]
        self._known_chokepoints["pay_dun04"] = [(40, 40), (140, 40), (80, 80), (160, 80)]

        # Geffen Dungeon
        self._known_chokepoints["gef_dun00"] = [(120, 60), (220, 60), (170, 110)]
        self._known_chokepoints["gef_dun01"] = [(90, 70), (190, 70), (140, 120)]
        self._known_chokepoints["gef_dun02"] = [(70, 50), (170, 50), (120, 100)]
        self._known_chokepoints["gef_dun03"] = [(50, 40), (150, 40), (100, 90)]

        # Orc Dungeon
        self._known_chokepoints["orcsdun01"] = [(100, 100), (200, 100), (150, 150)]
        self._known_chokepoints["orcsdun02"] = [(80, 80), (180, 80), (130, 130)]

        # Byalan Island
        self._known_chokepoints["iz_dun00"] = [(150, 50), (250, 50), (200, 100)]
        self._known_chokepoints["iz_dun01"] = [(100, 60), (200, 60), (150, 110)]
        self._known_chokepoints["iz_dun02"] = [(80, 40), (180, 40), (130, 90)]
        self._known_chokepoints["iz_dun03"] = [(60, 30), (160, 30), (110, 80)]
        self._known_chokepoints["iz_dun04"] = [(40, 20), (140, 20), (90, 70)]
        self._known_chokepoints["iz_dun05"] = [(20, 10), (120, 10), (70, 60)]

        # Morocc fields
        self._known_corners["moc_fild17"] = [(30, 30), (370, 30), (30, 370), (370, 370)]
        self._known_corners["moc_fild18"] = [(40, 40), (360, 40), (40, 360), (360, 360)]
        self._known_corners["moc_fild19"] = [(50, 50), (350, 50), (50, 350), (350, 350)]
        self._known_corners["moc_fild20"] = [(20, 20), (380, 20), (20, 380), (380, 380)]
        self._known_corners["moc_fild21"] = [(60, 60), (340, 60), (60, 340), (340, 340)]
        self._known_corners["moc_fild22"] = [(10, 10), (390, 10), (10, 390), (390, 390)]

    # ── Public API ──

    def assess_position(self, my_x: int, my_y: int, map_name: str, monsters: list[dict]) -> PositionSituation:
        """Assess the current position situation."""
        with self._lock:
            situation = PositionSituation(
                my_x=my_x,
                my_y=my_y,
                map_name=map_name,
                monsters=monsters,
            )

            aggro = 0
            nearest_dist = 999.0
            within_10 = 0
            within_5 = 0

            for m in monsters:
                mx = m.get("x", 0) or 0
                my = m.get("y", 0) or 0
                dist = math.sqrt((mx - my_x) ** 2 + (my - my_y) ** 2)
                if dist < nearest_dist:
                    nearest_dist = dist
                if dist <= 10:
                    within_10 += 1
                if dist <= 5:
                    within_5 += 1
                if m.get("relation") == "aggressive":
                    aggro += 1

            situation.aggro_count = aggro
            situation.nearest_monster_distance = nearest_dist
            situation.monsters_within_10 = within_10
            situation.monsters_within_5 = within_5
            situation.is_surrounded = within_5 >= 3

            # Compute safe direction (away from nearest monsters)
            if monsters:
                avg_x = sum(m.get("x", 0) or 0 for m in monsters) / len(monsters)
                avg_y = sum(m.get("y", 0) or 0 for m in monsters) / len(monsters)
                dx = my_x - avg_x
                dy = my_y - avg_y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > 0:
                    situation.safe_direction = (int(dx / dist * 10), int(dy / dist * 10))

            # Find best safe spot
            situation.best_safe_spot = self._find_best_safe_spot(my_x, my_y, map_name, monsters)

            return situation

    def _find_best_safe_spot(self, my_x: int, my_y: int, map_name: str, monsters: list[dict]) -> SafePosition | None:
        """Find the best safe position to move to."""
        candidates: list[SafePosition] = []

        # Check known corners
        corners = self._known_corners.get(map_name, [])
        for cx, cy in corners:
            dist = math.sqrt((cx - my_x) ** 2 + (cy - my_y) ** 2)
            monsters_near = sum(1 for m in monsters if math.sqrt((m.get("x", 0) - cx) ** 2 + (m.get("y", 0) - cy) ** 2) < 10)
            score = 100 - monsters_near * 20 - dist * 0.5
            candidates.append(SafePosition(
                x=cx, y=cy,
                safety_score=max(0, score),
                monsters_can_reach=monsters_near,
                distance_from_aggro=dist,
                is_corner=True,
                reason=f"Corner at ({cx},{cy})",
            ))

        # Check known chokepoints
        chokepoints = self._known_chokepoints.get(map_name, [])
        for cx, cy in chokepoints:
            dist = math.sqrt((cx - my_x) ** 2 + (cy - my_y) ** 2)
            monsters_near = sum(1 for m in monsters if math.sqrt((m.get("x", 0) - cx) ** 2 + (m.get("y", 0) - cy) ** 2) < 10)
            score = 100 - monsters_near * 15 - dist * 0.3
            candidates.append(SafePosition(
                x=cx, y=cy,
                safety_score=max(0, score),
                monsters_can_reach=monsters_near,
                distance_from_aggro=dist,
                is_chokepoint=True,
                reason=f"Chokepoint at ({cx},{cy})",
            ))

        # Compute dynamic safe spot (away from all monsters)
        if monsters:
            avg_x = sum(m.get("x", 0) or 0 for m in monsters) / len(monsters)
            avg_y = sum(m.get("y", 0) or 0 for m in monsters) / len(monsters)
            dx = my_x - avg_x
            dy = my_y - avg_y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 0:
                safe_x = int(my_x + dx / dist * 15)
                safe_y = int(my_y + dy / dist * 15)
                safe_x = max(5, min(395, safe_x))
                safe_y = max(5, min(395, safe_y))
                monsters_near = sum(1 for m in monsters if math.sqrt((m.get("x", 0) - safe_x) ** 2 + (m.get("y", 0) - safe_y) ** 2) < 10)
                score = 100 - monsters_near * 20
                candidates.append(SafePosition(
                    x=safe_x, y=safe_y,
                    safety_score=max(0, score),
                    monsters_can_reach=monsters_near,
                    distance_from_aggro=dist,
                    reason=f"Dynamic safe spot at ({safe_x},{safe_y})",
                ))

        if not candidates:
            return None

        candidates.sort(key=lambda s: -s.safety_score)
        return candidates[0]

    def get_safe_direction(self, my_x: int, my_y: int, monsters: list[dict]) -> tuple[int, int]:
        """Get the safest direction to move."""
        with self._lock:
            if not monsters:
                return (0, 0)
            avg_x = sum(m.get("x", 0) or 0 for m in monsters) / len(monsters)
            avg_y = sum(m.get("y", 0) or 0 for m in monsters) / len(monsters)
            dx = my_x - avg_x
            dy = my_y - avg_y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist == 0:
                return (1, 0)
            return (int(dx / dist * 10), int(dy / dist * 10))

    def is_surrounded(self, my_x: int, my_y: int, monsters: list[dict], threshold: int = 3) -> bool:
        """Check if surrounded by monsters within 5 tiles."""
        with self._lock:
            within_5 = sum(1 for m in monsters if math.sqrt((m.get("x", 0) - my_x) ** 2 + (m.get("y", 0) - my_y) ** 2) <= 5)
            return within_5 >= threshold

    def get_nearest_corner(self, my_x: int, my_y: int, map_name: str) -> tuple[int, int] | None:
        """Get the nearest corner on the map."""
        with self._lock:
            corners = self._known_corners.get(map_name, [])
            if not corners:
                return None
            best = min(corners, key=lambda c: math.sqrt((c[0] - my_x) ** 2 + (c[1] - my_y) ** 2))
            return best

    def get_nearest_chokepoint(self, my_x: int, my_y: int, map_name: str) -> tuple[int, int] | None:
        """Get the nearest chokepoint on the map."""
        with self._lock:
            chokepoints = self._known_chokepoints.get(map_name, [])
            if not chokepoints:
                return None
            best = min(chokepoints, key=lambda c: math.sqrt((c[0] - my_x) ** 2 + (c[1] - my_y) ** 2))
            return best

    def register_corner(self, map_name: str, x: int, y: int) -> None:
        with self._lock:
            if map_name not in self._known_corners:
                self._known_corners[map_name] = []
            self._known_corners[map_name].append((x, y))

    def register_chokepoint(self, map_name: str, x: int, y: int) -> None:
        with self._lock:
            if map_name not in self._known_chokepoints:
                self._known_chokepoints[map_name] = []
            self._known_chokepoints[map_name].append((x, y))

    def get_position_summary(self, situation: PositionSituation) -> str:
        """Get a human-readable position summary."""
        lines = [f"── Position Summary ──"]
        lines.append(f"Position: ({situation.my_x}, {situation.my_y}) on {situation.map_name}")
        lines.append(f"Aggro: {situation.aggro_count}")
        lines.append(f"Nearest monster: {situation.nearest_monster_distance:.1f} tiles")
        lines.append(f"Monsters within 10: {situation.monsters_within_10}")
        lines.append(f"Monsters within 5: {situation.monsters_within_5}")
        lines.append(f"Surrounded: {situation.is_surrounded}")
        if situation.best_safe_spot:
            lines.append(f"Best safe spot: {situation.best_safe_spot.reason} (score: {situation.best_safe_spot.safety_score:.0f})")
        return "\n".join(lines)


# ── Global Singleton ──

_safe_pos_computer: SafePositionComputer | None = None
_safe_pos_lock = RLock()


def get_safe_position_computer() -> SafePositionComputer:
    global _safe_pos_computer
    with _safe_pos_lock:
        if _safe_pos_computer is None:
            _safe_pos_computer = SafePositionComputer()
        return _safe_pos_computer
