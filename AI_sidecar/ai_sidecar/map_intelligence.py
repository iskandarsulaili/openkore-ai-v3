"""
Map Intelligence Layer — provides map-level awareness for RO farming.

A pro player knows every map: where the safe spots are, where monsters spawn,
which chokepoints limit incoming damage, and the optimal farming route.
This module encodes that knowledge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MapData:
    """Data about a single map."""
    name: str
    safe_spots: list[tuple[int, int]] = field(default_factory=list)
    chokepoints: list[tuple[int, int]] = field(default_factory=list)
    spawn_points: list[tuple[int, int]] = field(default_factory=list)
    respawn_timers: dict[str, int] = field(default_factory=dict)
    recommended_level_range: tuple[int, int] = (1, 99)
    danger_zones: list[tuple[int, int, int]] = field(default_factory=list)
    is_town: bool = False
    is_dungeon: bool = False
    is_woe_map: bool = False
    adjacent_maps: list[str] = field(default_factory=list)
    difficulty: str = "easy"  # easy, medium, hard, deadly
    monster_density: str = "low"  # low, medium, high, very_high
    recommended_party_size: int = 1
    has_teleport: bool = True
    warp_cost: int = 0


@dataclass
class FarmingRoute:
    """A farming route on a map."""
    map_name: str
    waypoints: list[tuple[int, int]] = field(default_factory=list)
    estimated_zeny_per_hour: int = 0
    estimated_exp_per_hour: int = 0
    primary_mobs: list[str] = field(default_factory=list)
    loop_time_seconds: int = 60
    danger_level: str = "low"


class MapIntelligence:
    """Provides map-level awareness for RO farming."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._maps: dict[str, MapData] = {}
        self._routes: dict[str, list[FarmingRoute]] = {}
        self._load_default_maps()

    def _load_default_maps(self) -> None:
        """Load data for common farming maps."""
        # ── Prontera Fields ──
        self._maps["prt_fild01"] = MapData(
            name="prt_fild01",
            safe_spots=[(100, 100), (200, 200)],
            chokepoints=[(150, 150)],
            spawn_points=[(50, 50), (100, 150), (200, 100), (150, 200)],
            respawn_timers={"Poring": 5, "Lunatic": 5, "Fabre": 5, "Peco Peco": 8},
            recommended_level_range=(1, 25),
            danger_zones=[(150, 150, 30)],
            is_town=False, is_dungeon=False,
            adjacent_maps=["prt_fild02", "prt_fild03", "prontera"],
            difficulty="easy", monster_density="low",
        )
        self._maps["prt_fild02"] = MapData(
            name="prt_fild02",
            safe_spots=[(100, 100)],
            chokepoints=[(120, 120)],
            spawn_points=[(50, 50), (150, 100), (200, 150)],
            respawn_timers={"Poring": 5, "Lunatic": 5, "Chonchon": 5},
            recommended_level_range=(1, 30),
            is_town=False, is_dungeon=False,
            adjacent_maps=["prt_fild01", "prt_fild03", "prt_fild04"],
            difficulty="easy", monster_density="low",
        )
        self._maps["prt_fild03"] = MapData(
            name="prt_fild03",
            safe_spots=[(100, 100)],
            chokepoints=[(130, 130)],
            spawn_points=[(50, 50), (100, 150), (200, 100)],
            respawn_timers={"Poring": 5, "Lunatic": 5, "Peco Peco": 8, "Chonchon": 5},
            recommended_level_range=(10, 35),
            is_town=False, is_dungeon=False,
            adjacent_maps=["prt_fild01", "prt_fild02", "prt_fild04", "prt_fild05"],
            difficulty="easy", monster_density="medium",
        )
        self._maps["prt_fild04"] = MapData(
            name="prt_fild04",
            safe_spots=[(100, 100)],
            chokepoints=[(120, 120)],
            spawn_points=[(50, 50), (150, 100), (200, 150)],
            respawn_timers={"Peco Peco": 8, "Chonchon": 5, "Spore": 6},
            recommended_level_range=(15, 40),
            is_town=False, is_dungeon=False,
            adjacent_maps=["prt_fild03", "prt_fild05", "prt_fild06"],
            difficulty="easy", monster_density="medium",
        )
        self._maps["prt_fild05"] = MapData(
            name="prt_fild05",
            safe_spots=[(100, 100)],
            chokepoints=[(130, 130)],
            spawn_points=[(50, 50), (100, 150), (200, 100)],
            respawn_timers={"Peco Peco": 8, "Spore": 6, "Savage": 10},
            recommended_level_range=(20, 50),
            is_town=False, is_dungeon=False,
            adjacent_maps=["prt_fild04", "prt_fild06", "prt_fild07"],
            difficulty="medium", monster_density="medium",
        )
        self._maps["prt_fild06"] = MapData(
            name="prt_fild06",
            safe_spots=[(100, 100)],
            chokepoints=[(120, 120)],
            spawn_points=[(50, 50), (150, 100), (200, 150)],
            respawn_timers={"Savage": 10, "Elder Willow": 7, "Thief Bug": 6},
            recommended_level_range=(25, 55),
            is_town=False, is_dungeon=False,
            adjacent_maps=["prt_fild05", "prt_fild07", "prt_fild08"],
            difficulty="medium", monster_density="medium",
        )
        self._maps["prt_fild07"] = MapData(
            name="prt_fild07",
            safe_spots=[(100, 100)],
            chokepoints=[(130, 130)],
            spawn_points=[(50, 50), (100, 150), (200, 100)],
            respawn_timers={"Savage": 10, "Elder Willow": 7, "Thief Bug": 6, "Mantis": 8},
            recommended_level_range=(30, 60),
            is_town=False, is_dungeon=False,
            adjacent_maps=["prt_fild06", "prt_fild08", "prt_fild09"],
            difficulty="medium", monster_density="high",
        )
        self._maps["prt_fild08"] = MapData(
            name="prt_fild08",
            safe_spots=[(100, 100)],
            chokepoints=[(120, 120)],
            spawn_points=[(50, 50), (150, 100), (200, 150)],
            respawn_timers={"Mantis": 8, "Thief Bug": 6, "Dustiness": 7},
            recommended_level_range=(35, 65),
            is_town=False, is_dungeon=False,
            adjacent_maps=["prt_fild07", "prt_fild09", "prt_fild10"],
            difficulty="medium", monster_density="high",
        )
        self._maps["prt_fild09"] = MapData(
            name="prt_fild09",
            safe_spots=[(100, 100)],
            chokepoints=[(130, 130)],
            spawn_points=[(50, 50), (100, 150), (200, 100)],
            respawn_timers={"Mantis": 8, "Dustiness": 7, "Horn": 8},
            recommended_level_range=(40, 70),
            is_town=False, is_dungeon=False,
            adjacent_maps=["prt_fild08", "prt_fild10", "prt_fild11"],
            difficulty="medium", monster_density="high",
        )
        self._maps["prt_fild10"] = MapData(
            name="prt_fild10",
            safe_spots=[(100, 100)],
            chokepoints=[(120, 120)],
            spawn_points=[(50, 50), (150, 100), (200, 150)],
            respawn_timers={"Horn": 8, "Dustiness": 7, "Mantis": 8},
            recommended_level_range=(45, 75),
            is_town=False, is_dungeon=False,
            adjacent_maps=["prt_fild09", "prt_fild11", "gef_fild00"],
            difficulty="hard", monster_density="high",
        )
        self._maps["prt_fild11"] = MapData(
            name="prt_fild11",
            safe_spots=[(100, 100)],
            chokepoints=[(130, 130)],
            spawn_points=[(50, 50), (100, 150), (200, 100)],
            respawn_timers={"Horn": 8, "Mantis": 8, "Petite": 10},
            recommended_level_range=(50, 80),
            is_town=False, is_dungeon=False,
            adjacent_maps=["prt_fild10", "gef_fild00", "gef_fild01"],
            difficulty="hard", monster_density="high",
        )

        # ── Payon Cave ──
        self._maps["pay_dun00"] = MapData(
            name="pay_dun00",
            safe_spots=[(50, 50)],
            chokepoints=[(80, 80), (100, 100)],
            spawn_points=[(30, 30), (60, 60), (90, 90), (120, 120)],
            respawn_timers={"Zombie": 6, "Skeleton": 6, "Ghoul": 8},
            recommended_level_range=(20, 45),
            danger_zones=[(80, 80, 40)],
            is_town=False, is_dungeon=True,
            adjacent_maps=["payon", "pay_dun01"],
            difficulty="medium", monster_density="medium",
        )
        self._maps["pay_dun01"] = MapData(
            name="pay_dun01",
            safe_spots=[(50, 50)],
            chokepoints=[(80, 80), (100, 100)],
            spawn_points=[(30, 30), (60, 60), (90, 90), (120, 120)],
            respawn_timers={"Zombie": 6, "Skeleton": 6, "Ghoul": 8, "Mummy": 10},
            recommended_level_range=(30, 55),
            danger_zones=[(80, 80, 40)],
            is_town=False, is_dungeon=True,
            adjacent_maps=["pay_dun00", "pay_dun02"],
            difficulty="medium", monster_density="high",
        )
        self._maps["pay_dun02"] = MapData(
            name="pay_dun02",
            safe_spots=[(50, 50)],
            chokepoints=[(80, 80)],
            spawn_points=[(30, 30), (60, 60), (90, 90)],
            respawn_timers={"Mummy": 10, "Myst": 8, "Archer Skeleton": 7},
            recommended_level_range=(40, 65),
            danger_zones=[(80, 80, 40)],
            is_town=False, is_dungeon=True,
            adjacent_maps=["pay_dun01", "pay_dun03"],
            difficulty="hard", monster_density="high",
        )
        self._maps["pay_dun03"] = MapData(
            name="pay_dun03",
            safe_spots=[(50, 50)],
            chokepoints=[(80, 80)],
            spawn_points=[(30, 30), (60, 60), (90, 90)],
            respawn_timers={"Mummy": 10, "Myst": 8, "Evil Druid": 12},
            recommended_level_range=(50, 75),
            danger_zones=[(80, 80, 40)],
            is_town=False, is_dungeon=True,
            adjacent_maps=["pay_dun02", "pay_dun04"],
            difficulty="hard", monster_density="very_high",
        )
        self._maps["pay_dun04"] = MapData(
            name="pay_dun04",
            safe_spots=[(50, 50)],
            chokepoints=[(80, 80)],
            spawn_points=[(30, 30), (60, 60), (90, 90)],
            respawn_timers={"Evil Druid": 12, "Wraith": 10, "Nightmare": 10},
            recommended_level_range=(60, 85),
            danger_zones=[(80, 80, 40)],
            is_town=False, is_dungeon=True,
            adjacent_maps=["pay_dun03"],
            difficulty="hard", monster_density="very_high",
        )

        # ── Geffen Dungeon ──
        self._maps["gef_dun00"] = MapData(
            name="gef_dun00",
            safe_spots=[(50, 50)],
            chokepoints=[(80, 80)],
            spawn_points=[(30, 30), (60, 60), (90, 90)],
            respawn_timers={"Drainliar": 5, "Punk": 7, "Myst": 8},
            recommended_level_range=(30, 55),
            is_town=False, is_dungeon=True,
            adjacent_maps=["geffen", "gef_dun01"],
            difficulty="medium", monster_density="medium",
        )
        self._maps["gef_dun01"] = MapData(
            name="gef_dun01",
            safe_spots=[(50, 50)],
            chokepoints=[(80, 80)],
            spawn_points=[(30, 30), (60, 60), (90, 90)],
            respawn_timers={"Punk": 7, "Myst": 8, "Medusa": 10},
            recommended_level_range=(40, 65),
            is_town=False, is_dungeon=True,
            adjacent_maps=["gef_dun00", "gef_dun02"],
            difficulty="medium", monster_density="high",
        )
        self._maps["gef_dun02"] = MapData(
            name="gef_dun02",
            safe_spots=[(50, 50)],
            chokepoints=[(80, 80)],
            spawn_points=[(30, 30), (60, 60), (90, 90)],
            respawn_timers={"Medusa": 10, "Injustice": 8, "Raydric": 8},
            recommended_level_range=(50, 75),
            is_town=False, is_dungeon=True,
            adjacent_maps=["gef_dun01", "gef_dun03"],
            difficulty="hard", monster_density="high",
        )
        self._maps["gef_dun03"] = MapData(
            name="gef_dun03",
            safe_spots=[(50, 50)],
            chokepoints=[(80, 80)],
            spawn_points=[(30, 30), (60, 60), (90, 90)],
            respawn_timers={"Raydric": 8, "Injustice": 8, "Succubus": 12},
            recommended_level_range=(60, 85),
            is_town=False, is_dungeon=True,
            adjacent_maps=["gef_dun02"],
            difficulty="hard", monster_density="very_high",
        )

        # ── Orc Dungeon ──
        self._maps["orcsdun01"] = MapData(
            name="orcsdun01",
            safe_spots=[(50, 50)],
            chokepoints=[(80, 80)],
            spawn_points=[(30, 30), (60, 60), (90, 90)],
            respawn_timers={"Orc Warrior": 6, "Orc Archer": 6, "Orc Zombie": 7},
            recommended_level_range=(40, 70),
            is_town=False, is_dungeon=True,
            adjacent_maps=["gef_fild14", "orcsdun02"],
            difficulty="medium", monster_density="high",
        )
        self._maps["orcsdun02"] = MapData(
            name="orcsdun02",
            safe_spots=[(50, 50)],
            chokepoints=[(80, 80)],
            spawn_points=[(30, 30), (60, 60), (90, 90)],
            respawn_timers={"Orc Warrior": 6, "Orc Archer": 6, "Orc Lady": 10, "Orc Hero": 120},
            recommended_level_range=(50, 80),
            is_town=False, is_dungeon=True,
            adjacent_maps=["orcsdun01"],
            difficulty="hard", monster_density="very_high",
        )

        # ── Byalan Island ──
        self._maps["iz_dun00"] = MapData(
            name="iz_dun00",
            safe_spots=[(50, 50)],
            chokepoints=[(80, 80)],
            spawn_points=[(30, 30), (60, 60), (90, 90)],
            respawn_timers={"Vadon": 6, "Marc": 7, "Marina": 5},
            recommended_level_range=(20, 45),
            is_town=False, is_dungeon=True,
            adjacent_maps=["izlude", "iz_dun01"],
            difficulty="medium", monster_density="medium",
        )
        self._maps["iz_dun01"] = MapData(
            name="iz_dun01",
            safe_spots=[(50, 50)],
            chokepoints=[(80, 80)],
            spawn_points=[(30, 30), (60, 60), (90, 90)],
            respawn_timers={"Vadon": 6, "Marc": 7, "Marina": 5, "Kukre": 6},
            recommended_level_range=(25, 50),
            is_town=False, is_dungeon=True,
            adjacent_maps=["iz_dun00", "iz_dun02"],
            difficulty="medium", monster_density="medium",
        )
        self._maps["iz_dun02"] = MapData(
            name="iz_dun02",
            safe_spots=[(50, 50)],
            chokepoints=[(80, 80)],
            spawn_points=[(30, 30), (60, 60), (90, 90)],
            respawn_timers={"Kukre": 6, "Marc": 7, "Strouf": 8},
            recommended_level_range=(30, 55),
            is_town=False, is_dungeon=True,
            adjacent_maps=["iz_dun01", "iz_dun03"],
            difficulty="medium", monster_density="high",
        )
        self._maps["iz_dun03"] = MapData(
            name="iz_dun03",
            safe_spots=[(50, 50)],
            chokepoints=[(80, 80)],
            spawn_points=[(30, 30), (60, 60), (90, 90)],
            respawn_timers={"Strouf": 8, "Marc": 7, "Obeaune": 8},
            recommended_level_range=(40, 65),
            is_town=False, is_dungeon=True,
            adjacent_maps=["iz_dun02", "iz_dun04"],
            difficulty="hard", monster_density="high",
        )
        self._maps["iz_dun04"] = MapData(
            name="iz_dun04",
            safe_spots=[(50, 50)],
            chokepoints=[(80, 80)],
            spawn_points=[(30, 30), (60, 60), (90, 90)],
            respawn_timers={"Obeaune": 8, "Strouf": 8, "Mysteltainn": 12},
            recommended_level_range=(50, 75),
            is_town=False, is_dungeon=True,
            adjacent_maps=["iz_dun03"],
            difficulty="hard", monster_density="very_high",
        )

        # ── Sograt Desert ──
        self._maps["moc_fild17"] = MapData(
            name="moc_fild17",
            safe_spots=[(100, 100)],
            chokepoints=[(120, 120)],
            spawn_points=[(50, 50), (150, 100), (200, 150)],
            respawn_timers={"Scorpion": 6, "Savage": 10, "Hornet": 5},
            recommended_level_range=(15, 40),
            is_town=False, is_dungeon=False,
            adjacent_maps=["moc_fild16", "moc_fild18", "moc_fild19"],
            difficulty="easy", monster_density="medium",
        )
        self._maps["moc_fild18"] = MapData(
            name="moc_fild18",
            safe_spots=[(100, 100)],
            chokepoints=[(120, 120)],
            spawn_points=[(50, 50), (150, 100), (200, 150)],
            respawn_timers={"Scorpion": 6, "Hornet": 5, "Mantis": 8},
            recommended_level_range=(20, 45),
            is_town=False, is_dungeon=False,
            adjacent_maps=["moc_fild17", "moc_fild19", "moc_fild20"],
            difficulty="medium", monster_density="medium",
        )
        self._maps["moc_fild19"] = MapData(
            name="moc_fild19",
            safe_spots=[(100, 100)],
            chokepoints=[(120, 120)],
            spawn_points=[(50, 50), (150, 100), (200, 150)],
            respawn_timers={"Mantis": 8, "Hornet": 5, "Scorpion": 6},
            recommended_level_range=(25, 50),
            is_town=False, is_dungeon=False,
            adjacent_maps=["moc_fild17", "moc_fild18", "moc_fild20", "moc_fild21"],
            difficulty="medium", monster_density="high",
        )
        self._maps["moc_fild20"] = MapData(
            name="moc_fild20",
            safe_spots=[(100, 100)],
            chokepoints=[(120, 120)],
            spawn_points=[(50, 50), (150, 100), (200, 150)],
            respawn_timers={"Mantis": 8, "Scorpion": 6, "Sandman": 10},
            recommended_level_range=(30, 55),
            is_town=False, is_dungeon=False,
            adjacent_maps=["moc_fild19", "moc_fild21", "moc_fild22"],
            difficulty="medium", monster_density="high",
        )
        self._maps["moc_fild21"] = MapData(
            name="moc_fild21",
            safe_spots=[(100, 100)],
            chokepoints=[(120, 120)],
            spawn_points=[(50, 50), (150, 100), (200, 150)],
            respawn_timers={"Sandman": 10, "Mantis": 8, "Scorpion King": 12},
            recommended_level_range=(35, 60),
            is_town=False, is_dungeon=False,
            adjacent_maps=["moc_fild19", "moc_fild20", "moc_fild22"],
            difficulty="hard", monster_density="high",
        )
        self._maps["moc_fild22"] = MapData(
            name="moc_fild22",
            safe_spots=[(100, 100)],
            chokepoints=[(120, 120)],
            spawn_points=[(50, 50), (150, 100), (200, 150)],
            respawn_timers={"Sandman": 10, "Scorpion King": 12, "Mantis": 8},
            recommended_level_range=(40, 65),
            is_town=False, is_dungeon=False,
            adjacent_maps=["moc_fild20", "moc_fild21"],
            difficulty="hard", monster_density="high",
        )

        # ── Towns ──
        for town in ["prontera", "geffen", "payon", "morocc", "aldebaran", "yuno", "izlude", "xmas", "comodo"]:
            self._maps[town] = MapData(
                name=town,
                safe_spots=[(150, 150)],
                is_town=True,
                adjacent_maps=[f"{town}_fild01"] if town != "izlude" else ["iz_dun00"],
                difficulty="easy", monster_density="low",
            )

        # ── Build farming routes ──
        self._build_routes()

    def _build_routes(self) -> None:
        """Build default farming routes for each map."""
        for map_name, md in self._maps.items():
            if md.is_town:
                continue
            route = FarmingRoute(
                map_name=map_name,
                waypoints=md.spawn_points[:4] if md.spawn_points else [(50, 50), (100, 100)],
                primary_mobs=list(md.respawn_timers.keys())[:3],
                loop_time_seconds=60,
                danger_level=md.difficulty,
            )
            self._routes[map_name] = [route]

    # ── Public API ──

    def get_map_data(self, map_name: str) -> MapData | None:
        with self._lock:
            return self._maps.get(map_name.lower().strip())

    def get_safe_spot(self, map_name: str, x: int = 0, y: int = 0) -> tuple[int, int] | None:
        """Get the nearest safe spot on a map."""
        md = self.get_map_data(map_name)
        if not md or not md.safe_spots:
            return None
        if not md.safe_spots:
            return None
        best = md.safe_spots[0]
        best_dist = abs(best[0] - x) + abs(best[1] - y)
        for spot in md.safe_spots[1:]:
            dist = abs(spot[0] - x) + abs(spot[1] - y)
            if dist < best_dist:
                best_dist = dist
                best = spot
        return best

    def get_farming_route(self, map_name: str, player_level: int = 1) -> list[tuple[int, int]] | None:
        """Get the best farming route for a map."""
        md = self.get_map_data(map_name)
        if not md:
            return None
        routes = self._routes.get(map_name.lower().strip())
        if not routes:
            return md.spawn_points[:4] if md.spawn_points else [(50, 50), (100, 100)]
        return routes[0].waypoints

    def get_chokepoint_near(self, map_name: str, x: int, y: int) -> tuple[int, int] | None:
        """Get the nearest chokepoint to a position."""
        md = self.get_map_data(map_name)
        if not md or not md.chokepoints:
            return None
        best = min(md.chokepoints, key=lambda cp: abs(cp[0] - x) + abs(cp[1] - y))
        return best

    def is_safe_position(self, map_name: str, x: int, y: int) -> bool:
        """Check if a position is in a safe zone."""
        md = self.get_map_data(map_name)
        if not md:
            return False
        for sx, sy in md.safe_spots:
            if abs(sx - x) < 5 and abs(sy - y) < 5:
                return True
        return False

    def get_recommended_maps(self, player_level: int, job_class: str = "") -> list[MapData]:
        """Get maps recommended for a player's level."""
        result: list[MapData] = []
        with self._lock:
            for md in self._maps.values():
                if md.is_town:
                    continue
                min_lv, max_lv = md.recommended_level_range
                if min_lv <= player_level <= max_lv:
                    result.append(md)
        result.sort(key=lambda m: m.recommended_level_range[0])
        return result

    def get_spawn_density(self, map_name: str, x: int, y: int, radius: int = 20) -> int:
        """Estimate spawn density near a position."""
        md = self.get_map_data(map_name)
        if not md or not md.spawn_points:
            return 0
        count = 0
        for sx, sy in md.spawn_points:
            if abs(sx - x) <= radius and abs(sy - y) <= radius:
                count += 1
        return count

    def get_respawn_time(self, monster_name: str) -> int:
        """Get the respawn time for a monster."""
        with self._lock:
            for md in self._maps.values():
                if monster_name in md.respawn_timers:
                    return md.respawn_timers[monster_name]
        return 5  # default

    def get_map_difficulty(self, map_name: str) -> str:
        md = self.get_map_data(map_name)
        if not md:
            return "unknown"
        return md.difficulty

    def get_adjacent_maps(self, map_name: str) -> list[str]:
        md = self.get_map_data(map_name)
        if not md:
            return []
        return md.adjacent_maps

    def get_all_maps(self) -> list[MapData]:
        with self._lock:
            return list(self._maps.values())

    def get_farming_maps(self) -> list[MapData]:
        with self._lock:
            return [m for m in self._maps.values() if not m.is_town]

    def get_dungeon_maps(self) -> list[MapData]:
        with self._lock:
            return [m for m in self._maps.values() if m.is_dungeon]

    def get_woe_maps(self) -> list[MapData]:
        with self._lock:
            return [m for m in self._maps.values() if m.is_woe_map]


# ── Global Singleton ──

_map_intel: MapIntelligence | None = None
_map_intel_lock = RLock()


def get_map_intelligence() -> MapIntelligence:
    global _map_intel
    with _map_intel_lock:
        if _map_intel is None:
            _map_intel = MapIntelligence()
        return _map_intel
