"""
MVP Respawn Tracker — records kill times, predicts respawns,
routes bots to due MVPs, and coordinates multi-bot MVP hunting.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class MVPRecord:
    """A record of an MVP kill or sighting."""
    monster_id: int
    monster_name: str
    map_name: str
    kill_time: float = 0.0
    sighting_time: float = 0.0
    respawn_time: float = 0.0
    respawn_window_minutes: int = 120  # Default respawn window
    is_due: bool = False
    is_up: bool = False
    killed_by_us: bool = False
    strategy_used: str = ""


@dataclass
class MVPHuntTarget:
    """An MVP that's worth hunting right now."""
    monster_id: int
    monster_name: str
    map_name: str
    time_until_respawn_min: float = 0.0
    priority: int = 50
    estimated_value: int = 0
    difficulty: str = "medium"
    is_worth_hunting: bool = True


class MVPTracker:
    """Tracks MVP respawns and coordinates hunting."""

    # Known MVPs with their respawn windows (minutes) and estimated card value
    KNOWN_MVPS: dict[int, dict] = {
        1038: {"name": "Osiris", "respawn": 120, "value": 50000000, "difficulty": "hard"},
        1046: {"name": "Phreeoni", "respawn": 120, "value": 30000000, "difficulty": "hard"},
        1049: {"name": "Maya", "respawn": 120, "value": 40000000, "difficulty": "hard"},
        1059: {"name": "Moonlight Flower", "respawn": 60, "value": 20000000, "difficulty": "medium"},
        1086: {"name": "Dracula", "respawn": 120, "value": 35000000, "difficulty": "hard"},
        1087: {"name": "Doppelganger", "respawn": 120, "value": 45000000, "difficulty": "hard"},
        1088: {"name": "Orc Hero", "respawn": 60, "value": 15000000, "difficulty": "medium"},
        1112: {"name": "Orc Lord", "respawn": 120, "value": 25000000, "difficulty": "hard"},
        1115: {"name": "Mistress", "respawn": 60, "value": 20000000, "difficulty": "medium"},
        1147: {"name": "Baphomet", "respawn": 120, "value": 60000000, "difficulty": "hard"},
        1150: {"name": "Eddga", "respawn": 60, "value": 15000000, "difficulty": "medium"},
        1159: {"name": "Golden Thief Bug", "respawn": 120, "value": 50000000, "difficulty": "hard"},
        1205: {"name": "Kraken", "respawn": 120, "value": 30000000, "difficulty": "hard"},
        1272: {"name": "Turtle General", "respawn": 120, "value": 40000000, "difficulty": "hard"},
        1312: {"name": "Atroce", "respawn": 120, "value": 35000000, "difficulty": "hard"},
        1313: {"name": "Gloom Under Night", "respawn": 120, "value": 30000000, "difficulty": "hard"},
        1511: {"name": "Kiel-D-01", "respawn": 120, "value": 25000000, "difficulty": "hard"},
        1630: {"name": "Ktullanux", "respawn": 120, "value": 30000000, "difficulty": "hard"},
        1639: {"name": "Valkyrie Randgris", "respawn": 120, "value": 40000000, "difficulty": "hard"},
        1719: {"name": "Beelzebub", "respawn": 120, "value": 50000000, "difficulty": "hard"},
        1751: {"name": "Ifrit", "respawn": 120, "value": 35000000, "difficulty": "hard"},
        1871: {"name": "Amdarais", "respawn": 120, "value": 30000000, "difficulty": "hard"},
        1874: {"name": "Bacon Lord", "respawn": 60, "value": 10000000, "difficulty": "medium"},
    }

    def __init__(self) -> None:
        self._lock = RLock()
        self._records: dict[int, MVPRecord] = {}
        self._hunt_targets: list[MVPHuntTarget] = []
        self._enqueue_fn: Callable | None = None
        self._load_known_mvps()

    def _load_known_mvps(self) -> None:
        """Initialize records for all known MVPs."""
        for mid, data in self.KNOWN_MVPS.items():
            self._records[mid] = MVPRecord(
                monster_id=mid,
                monster_name=data["name"],
                map_name="unknown",
                respawn_window_minutes=data["respawn"],
            )

    # ── Public API ──

    def record_kill(self, monster_id: int, map_name: str, killed_by_us: bool = False, strategy: str = "") -> None:
        """Record an MVP kill."""
        with self._lock:
            data = self.KNOWN_MVPS.get(monster_id)
            if not data:
                return
            now = time.time()
            self._records[monster_id] = MVPRecord(
                monster_id=monster_id,
                monster_name=data["name"],
                map_name=map_name,
                kill_time=now,
                respawn_time=now + (data["respawn"] * 60),
                respawn_window_minutes=data["respawn"],
                is_due=False,
                is_up=False,
                killed_by_us=killed_by_us,
                strategy_used=strategy,
            )
            logger.info("mvp_kill_recorded: %s on %s (respawn at %s)", data["name"], map_name,
                        time.strftime("%H:%M", time.localtime(now + data["respawn"] * 60)))

    def record_sighting(self, monster_id: int, map_name: str) -> None:
        """Record an MVP sighting (it's alive and on this map)."""
        with self._lock:
            data = self.KNOWN_MVPS.get(monster_id)
            if not data:
                return
            now = time.time()
            if monster_id in self._records:
                self._records[monster_id].sighting_time = now
                self._records[monster_id].map_name = map_name
                self._records[monster_id].is_up = True
                self._records[monster_id].is_due = False
            else:
                self._records[monster_id] = MVPRecord(
                    monster_id=monster_id,
                    monster_name=data["name"],
                    map_name=map_name,
                    sighting_time=now,
                    respawn_window_minutes=data["respawn"],
                    is_up=True,
                )

    def update_hunt_targets(self) -> list[MVPHuntTarget]:
        """Update and return the list of MVPs worth hunting right now."""
        with self._lock:
            now = time.time()
            targets: list[MVPHuntTarget] = []

            for mid, record in self._records.items():
                data = self.KNOWN_MVPS.get(mid)
                if not data:
                    continue

                # Check if MVP is due for respawn
                if record.respawn_time > 0 and now >= record.respawn_time:
                    record.is_due = True
                    record.is_up = False

                # Check if MVP is currently up (sighted recently)
                if record.sighting_time > 0 and now - record.sighting_time < 300:
                    record.is_up = True

                if record.is_due or record.is_up:
                    time_until = max(0, record.respawn_time - now) / 60.0 if record.respawn_time > 0 else 0
                    priority = 100 if record.is_up else max(10, 100 - int(time_until * 2))
                    targets.append(MVPHuntTarget(
                        monster_id=mid,
                        monster_name=data["name"],
                        map_name=record.map_name,
                        time_until_respawn_min=time_until,
                        priority=priority,
                        estimated_value=data["value"],
                        difficulty=data["difficulty"],
                        is_worth_hunting=priority >= 30,
                    ))

            targets.sort(key=lambda t: -t.priority)
            self._hunt_targets = targets
            return targets

    def get_best_hunt_target(self) -> MVPHuntTarget | None:
        """Get the best MVP to hunt right now."""
        targets = self.update_hunt_targets()
        return targets[0] if targets else None

    def get_mvp_status(self, monster_id: int) -> str:
        with self._lock:
            record = self._records.get(monster_id)
            if not record:
                return "Unknown"
            if record.is_up:
                return f"UP on {record.map_name}"
            if record.is_due:
                return f"DUE (respawned on {record.map_name})"
            if record.respawn_time > 0:
                remaining = max(0, record.respawn_time - time.time())
                return f"Respawning in {int(remaining/60)}m"
            return "Unknown"

    def get_mvp_summary(self) -> str:
        with self._lock:
            lines = [f"── MVP Tracker ──"]
            targets = self.update_hunt_targets()
            up = [t for t in targets if t.is_worth_hunting and any(
                r.is_up for r in self._records.values() if r.monster_id == t.monster_id
            )]
            due = [t for t in targets if t.is_worth_hunting and any(
                r.is_due for r in self._records.values() if r.monster_id == t.monster_id
            )]
            if up:
                lines.append(f"Currently UP: {', '.join(f'{t.monster_name}({t.map_name})' for t in up[:5])}")
            if due:
                lines.append(f"Due to respawn: {', '.join(f'{t.monster_name}({t.map_name})' for t in due[:5])}")
            if not up and not due:
                lines.append("No MVPs currently up or due")
            best = self.get_best_hunt_target()
            if best:
                lines.append(f"Best target: {best.monster_name} on {best.map_name} (value={best.estimated_value:,}z)")
            return "\n".join(lines)

    def set_enqueue_fn(self, fn: Callable) -> None:
        with self._lock:
            self._enqueue_fn = fn

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._hunt_targets.clear()
            self._load_known_mvps()


# ── Global Singleton ──

_mvp_tracker: MVPTracker | None = None
_mvp_tracker_lock = RLock()


def get_mvp_tracker() -> MVPTracker:
    global _mvp_tracker
    with _mvp_tracker_lock:
        if _mvp_tracker is None:
            _mvp_tracker = MVPTracker()
        return _mvp_tracker
