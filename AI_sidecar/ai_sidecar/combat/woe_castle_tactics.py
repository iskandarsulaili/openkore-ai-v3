"""
Castle-Specific WoE Tactics — per-castle chokepoint maps, guardian kill order,
emperium break strategies, and role-specific positioning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CastleTactic:
    """Tactical data for a specific castle."""
    castle_name: str
    map_name: str
    chokepoints: list[tuple[int, int, int, int]] = field(default_factory=list)  # (x1,y1,x2,y2) corridor
    guardian_positions: list[tuple[int, int]] = field(default_factory=list)
    emperium_position: tuple[int, int] = (0, 0)
    defender_positions: list[tuple[int, int]] = field(default_factory=list)
    attacker_entrance: tuple[int, int] = (0, 0)
    recommended_guardian_order: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class WoEBattlefield:
    """Current WoE battlefield state."""
    castle_name: str = ""
    map_name: str = ""
    allies_in_castle: int = 0
    enemies_in_castle: int = 0
    guardians_alive: int = 0
    emperium_hp_pct: float = 1.0
    our_flag_status: str = "safe"  # safe, under_attack, lost
    enemy_flag_status: str = "unknown"
    recommended_tactic: str = ""


class WoECastleTactics:
    """Castle-specific WoE tactics."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._castles: dict[str, CastleTactic] = {}
        self._load_castles()

    def _load_castles(self) -> None:
        """Load tactical data for all castles."""
        # Prontera castles
        self._castles["prt_gld01"] = CastleTactic(
            castle_name="Prt Castle 1",
            map_name="prt_gld01",
            chokepoints=[(50, 50, 50, 100), (100, 30, 150, 30)],
            guardian_positions=[(80, 60), (120, 60)],
            emperium_position=(100, 80),
            defender_positions=[(50, 50), (100, 30)],
            attacker_entrance=(10, 50),
            recommended_guardian_order=["left_guardian", "right_guardian"],
            notes="Narrow entrance, good for defense. Hold the stairs.",
        )
        self._castles["prt_gld02"] = CastleTactic(
            castle_name="Prt Castle 2",
            map_name="prt_gld02",
            chokepoints=[(60, 40, 60, 90), (120, 20, 120, 70)],
            guardian_positions=[(70, 50), (110, 50)],
            emperium_position=(90, 70),
            defender_positions=[(60, 40), (120, 20)],
            attacker_entrance=(20, 50),
            recommended_guardian_order=["right_guardian", "left_guardian"],
            notes="Two entrances, split defense needed.",
        )
        self._castles["prt_gld03"] = CastleTactic(
            castle_name="Prt Castle 3",
            map_name="prt_gld03",
            chokepoints=[(40, 60, 80, 60), (100, 40, 100, 80)],
            guardian_positions=[(60, 70), (90, 50)],
            emperium_position=(75, 60),
            defender_positions=[(40, 60), (100, 40)],
            attacker_entrance=(15, 60),
            recommended_guardian_order=["left_guardian", "right_guardian"],
            notes="Open layout, good for AoE defense.",
        )
        self._castles["prt_gld04"] = CastleTactic(
            castle_name="Prt Castle 4",
            map_name="prt_gld04",
            chokepoints=[(30, 30, 30, 80), (80, 20, 80, 60)],
            guardian_positions=[(50, 40), (70, 40)],
            emperium_position=(60, 55),
            defender_positions=[(30, 30), (80, 20)],
            attacker_entrance=(5, 50),
            recommended_guardian_order=["right_guardian", "left_guardian"],
            notes="Tight corridors, excellent for small groups.",
        )

        # Geffen castles
        self._castles["gef_gld01"] = CastleTactic(
            castle_name="Gef Castle 1",
            map_name="gef_gld01",
            chokepoints=[(50, 30, 50, 80), (100, 50, 150, 50)],
            guardian_positions=[(70, 40), (120, 60)],
            emperium_position=(95, 50),
            defender_positions=[(50, 30), (100, 50)],
            attacker_entrance=(10, 50),
            recommended_guardian_order=["left_guardian", "right_guardian"],
            notes="Long corridor approach, good for ranged defense.",
        )
        self._castles["gef_gld02"] = CastleTactic(
            castle_name="Gef Castle 2",
            map_name="gef_gld02",
            chokepoints=[(40, 50, 90, 50), (70, 20, 70, 70)],
            guardian_positions=[(60, 40), (80, 60)],
            emperium_position=(70, 50),
            defender_positions=[(40, 50), (70, 20)],
            attacker_entrance=(10, 50),
            recommended_guardian_order=["right_guardian", "left_guardian"],
            notes="Cross-shaped layout, central emperium room.",
        )
        self._castles["gef_gld03"] = CastleTactic(
            castle_name="Gef Castle 3",
            map_name="gef_gld03",
            chokepoints=[(30, 40, 30, 90), (90, 30, 90, 80)],
            guardian_positions=[(50, 50), (80, 50)],
            emperium_position=(65, 60),
            defender_positions=[(30, 40), (90, 30)],
            attacker_entrance=(5, 50),
            recommended_guardian_order=["left_guardian", "right_guardian"],
            notes="Two parallel corridors, split push required.",
        )
        self._castles["gef_gld04"] = CastleTactic(
            castle_name="Gef Castle 4",
            map_name="gef_gld04",
            chokepoints=[(60, 20, 60, 70), (100, 50, 150, 50)],
            guardian_positions=[(80, 30), (120, 50)],
            emperium_position=(100, 40),
            defender_positions=[(60, 20), (100, 50)],
            attacker_entrance=(15, 50),
            recommended_guardian_order=["right_guardian", "left_guardian"],
            notes="Elevated emperium room, defenders have height advantage.",
        )

        # Payon castles
        self._castles["pay_gld01"] = CastleTactic(
            castle_name="Pay Castle 1",
            map_name="pay_gld01",
            chokepoints=[(40, 30, 40, 80), (80, 50, 130, 50)],
            guardian_positions=[(60, 40), (100, 60)],
            emperium_position=(80, 50),
            defender_positions=[(40, 30), (80, 50)],
            attacker_entrance=(5, 50),
            recommended_guardian_order=["left_guardian", "right_guardian"],
            notes="Forest theme, lots of line-of-sight blockers.",
        )
        self._castles["pay_gld02"] = CastleTactic(
            castle_name="Pay Castle 2",
            map_name="pay_gld02",
            chokepoints=[(50, 20, 50, 70), (90, 30, 90, 80)],
            guardian_positions=[(70, 30), (80, 60)],
            emperium_position=(75, 45),
            defender_positions=[(50, 20), (90, 30)],
            attacker_entrance=(10, 50),
            recommended_guardian_order=["right_guardian", "left_guardian"],
            notes="Underground theme, tight spaces.",
        )
        self._castles["pay_gld03"] = CastleTactic(
            castle_name="Pay Castle 3",
            map_name="pay_gld03",
            chokepoints=[(30, 50, 80, 50), (60, 20, 60, 70)],
            guardian_positions=[(50, 40), (70, 60)],
            emperium_position=(60, 50),
            defender_positions=[(30, 50), (60, 20)],
            attacker_entrance=(5, 50),
            recommended_guardian_order=["left_guardian", "right_guardian"],
            notes="Symmetrical layout, balanced attack/defense.",
        )
        self._castles["pay_gld04"] = CastleTactic(
            castle_name="Pay Castle 4",
            map_name="pay_gld04",
            chokepoints=[(40, 40, 40, 90), (100, 30, 100, 80)],
            guardian_positions=[(60, 50), (90, 50)],
            emperium_position=(75, 60),
            defender_positions=[(40, 40), (100, 30)],
            attacker_entrance=(5, 50),
            recommended_guardian_order=["right_guardian", "left_guardian"],
            notes="Long approach, good for kiting defense.",
        )

        # Aldebaran castles
        self._castles["alde_gld01"] = CastleTactic(
            castle_name="Alde Castle 1",
            map_name="alde_gld01",
            chokepoints=[(50, 30, 50, 80), (100, 40, 150, 40)],
            guardian_positions=[(70, 40), (120, 50)],
            emperium_position=(95, 45),
            defender_positions=[(50, 30), (100, 40)],
            attacker_entrance=(10, 50),
            recommended_guardian_order=["left_guardian", "right_guardian"],
            notes="Clock tower theme, vertical layout.",
        )
        self._castles["alde_gld02"] = CastleTactic(
            castle_name="Alde Castle 2",
            map_name="alde_gld02",
            chokepoints=[(40, 50, 90, 50), (70, 20, 70, 70)],
            guardian_positions=[(60, 40), (80, 60)],
            emperium_position=(70, 50),
            defender_positions=[(40, 50), (70, 20)],
            attacker_entrance=(10, 50),
            recommended_guardian_order=["right_guardian", "left_guardian"],
            notes="Library theme, lots of bookshelves for cover.",
        )
        self._castles["alde_gld03"] = CastleTactic(
            castle_name="Alde Castle 3",
            map_name="alde_gld03",
            chokepoints=[(30, 30, 30, 80), (80, 40, 130, 40)],
            guardian_positions=[(50, 40), (100, 50)],
            emperium_position=(75, 45),
            defender_positions=[(30, 30), (80, 40)],
            attacker_entrance=(5, 50),
            recommended_guardian_order=["left_guardian", "right_guardian"],
            notes="Gingerbread theme, open center room.",
        )
        self._castles["alde_gld04"] = CastleTactic(
            castle_name="Alde Castle 4",
            map_name="alde_gld04",
            chokepoints=[(50, 20, 50, 70), (90, 50, 140, 50)],
            guardian_positions=[(70, 30), (110, 60)],
            emperium_position=(90, 40),
            defender_positions=[(50, 20), (90, 50)],
            attacker_entrance=(10, 50),
            recommended_guardian_order=["right_guardian", "left_guardian"],
            notes="Toy factory theme, multi-level layout.",
        )

        # Schuttgart castles
        self._castles["sch_gld01"] = CastleTactic(
            castle_name="Sch Castle 1",
            map_name="sch_gld01",
            chokepoints=[(40, 40, 40, 90), (100, 30, 100, 80)],
            guardian_positions=[(60, 50), (90, 50)],
            emperium_position=(75, 60),
            defender_positions=[(40, 40), (100, 30)],
            attacker_entrance=(5, 50),
            recommended_guardian_order=["left_guardian", "right_guardian"],
            notes="Ice theme, slippery floors, careful positioning.",
        )
        self._castles["sch_gld02"] = CastleTactic(
            castle_name="Sch Castle 2",
            map_name="sch_gld02",
            chokepoints=[(50, 30, 50, 80), (90, 40, 140, 40)],
            guardian_positions=[(70, 40), (110, 50)],
            emperium_position=(90, 45),
            defender_positions=[(50, 30), (90, 40)],
            attacker_entrance=(10, 50),
            recommended_guardian_order=["right_guardian", "left_guardian"],
            notes="Dwarf theme, narrow tunnels.",
        )
        self._castles["sch_gld03"] = CastleTactic(
            castle_name="Sch Castle 3",
            map_name="sch_gld03",
            chokepoints=[(30, 50, 80, 50), (60, 20, 60, 70)],
            guardian_positions=[(50, 40), (70, 60)],
            emperium_position=(60, 50),
            defender_positions=[(30, 50), (60, 20)],
            attacker_entrance=(5, 50),
            recommended_guardian_order=["left_guardian", "right_guardian"],
            notes="Mountain theme, elevation changes.",
        )
        self._castles["sch_gld04"] = CastleTactic(
            castle_name="Sch Castle 4",
            map_name="sch_gld04",
            chokepoints=[(40, 30, 40, 80), (80, 50, 130, 50)],
            guardian_positions=[(60, 40), (100, 60)],
            emperium_position=(80, 50),
            defender_positions=[(40, 30), (80, 50)],
            attacker_entrance=(5, 50),
            recommended_guardian_order=["right_guardian", "left_guardian"],
            notes="Ruins theme, open areas with pillars.",
        )

    # ── Public API ──

    def get_castle_tactic(self, map_name: str) -> CastleTactic | None:
        """Get tactical data for a specific castle map."""
        with self._lock:
            return self._castles.get(map_name)

    def get_chokepoints(self, map_name: str) -> list[tuple[int, int, int, int]]:
        """Get chokepoints for a castle map."""
        with self._lock:
            castle = self._castles.get(map_name)
            return castle.chokepoints if castle else []

    def get_guardian_positions(self, map_name: str) -> list[tuple[int, int]]:
        """Get guardian positions for a castle map."""
        with self._lock:
            castle = self._castles.get(map_name)
            return castle.guardian_positions if castle else []

    def get_emperium_position(self, map_name: str) -> tuple[int, int]:
        """Get emperium position for a castle map."""
        with self._lock:
            castle = self._castles.get(map_name)
            return castle.emperium_position if castle else (0, 0)

    def get_defender_positions(self, map_name: str) -> list[tuple[int, int]]:
        """Get recommended defender positions."""
        with self._lock:
            castle = self._castles.get(map_name)
            return castle.defender_positions if castle else []

    def get_attacker_entrance(self, map_name: str) -> tuple[int, int]:
        """Get the attacker entrance point."""
        with self._lock:
            castle = self._castles.get(map_name)
            return castle.attacker_entrance if castle else (0, 0)

    def get_recommended_tactic(self, map_name: str, role: str = "attacker") -> str:
        """Get the recommended tactic for a castle and role."""
        with self._lock:
            castle = self._castles.get(map_name)
            if not castle:
                return "No tactical data for this castle"

            if role == "attacker":
                return (
                    f"Enter from {castle.attacker_entrance}. "
                    f"Kill guardians in order: {', '.join(castle.recommended_guardian_order)}. "
                    f"Then push to emperium at {castle.emperium_position}. "
                    f"Watch for defenders at chokepoints: {castle.chokepoints}. "
                    f"Notes: {castle.notes}"
                )
            elif role == "defender":
                return (
                    f"Hold chokepoints: {castle.chokepoints}. "
                    f"Defender positions: {castle.defender_positions}. "
                    f"Protect guardians at {castle.guardian_positions}. "
                    f"Last stand at emperium ({castle.emperium_position}). "
                    f"Notes: {castle.notes}"
                )
            elif role == "support":
                return (
                    f"Position behind defenders at {castle.defender_positions}. "
                    f"Keep buffs on defenders. "
                    f"Watch for flanks at chokepoints: {castle.chokepoints}. "
                    f"Notes: {castle.notes}"
                )
            return castle.notes

    def assess_battlefield(self, map_name: str, allies: int = 0, enemies: int = 0,
                           guardians: int = 0, emperium_hp: float = 1.0) -> WoEBattlefield:
        """Assess the current battlefield state."""
        with self._lock:
            bf = WoEBattlefield(
                castle_name=map_name,
                map_name=map_name,
                allies_in_castle=allies,
                enemies_in_castle=enemies,
                guardians_alive=guardians,
                emperium_hp_pct=emperium_hp,
            )

            if enemies == 0:
                bf.recommended_tactic = "push_emperium"
            elif allies > enemies * 1.5:
                bf.recommended_tactic = "aggressive_push"
            elif allies < enemies * 0.5:
                bf.recommended_tactic = "defensive_hold"
            elif emperium_hp < 0.3:
                bf.recommended_tactic = "final_push"
            else:
                bf.recommended_tactic = "balanced"

            return bf

    def get_all_castles(self) -> list[CastleTactic]:
        with self._lock:
            return list(self._castles.values())

    def get_castle_names(self) -> list[str]:
        with self._lock:
            return sorted(self._castles.keys())

    def get_woe_tactics_summary(self) -> str:
        with self._lock:
            lines = [f"── WoE Castle Tactics ──"]
            lines.append(f"Castles loaded: {len(self._castles)}")
            for name, castle in sorted(self._castles.items()):
                lines.append(f"  {name}: {castle.castle_name} — {castle.notes[:60]}")
            return "\n".join(lines)


# ── Global Singleton ──

_woe_tactics: WoECastleTactics | None = None
_woe_tactics_lock = RLock()


def get_woe_castle_tactics() -> WoECastleTactics:
    global _woe_tactics
    with _woe_tactics_lock:
        if _woe_tactics is None:
            _woe_tactics = WoECastleTactics()
        return _woe_tactics
