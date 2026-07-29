"""World State Model — contextual awareness for RO bots.

Tracks everything a human player absorbs without thinking:
- Time of day (undead 1.5x damage at night)
- Weather (rain reduces fire damage)
- Server events (double EXP, double drops, WoE schedules)
- Guild relations (allies, enemies, war status)
- Recent deaths on map (danger signal)
- MVP spawn timers
"""
from __future__ import annotations
from typing import Any
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta

from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    yaml = None

DATA_DIR = Path(__file__).parent.parent.parent / "data"


class TimeOfDay:
    """Tracks RO time of day.
    
    RO has its own day/night cycle (1 real hour = 1 RO day).
    Day: 6:00-18:00  Night: 18:00-6:00
    
    Night effects:
    - Undead monsters deal 1.5x damage
    - Some monsters only spawn at night
    - Some skills have different effects
    """
    
    DAY_HOURS = range(6, 18)  # 6 AM to 6 PM
    
    @classmethod
    def is_night(cls, ro_hour: int | None = None) -> bool:
        """Check if it's night in RO time."""
        if ro_hour is None:
            ro_hour = cls.get_ro_hour()
        return ro_hour not in cls.DAY_HOURS
    
    @classmethod
    def get_ro_hour(cls) -> int:
        """Get current RO hour (0-23). RO runs 24x real time."""
        now = datetime.now(timezone.utc)
        # RO day = 1 real hour, starting at server reset
        ro_hour = (now.minute + 6) % 24  # approximate
        return ro_hour
    
    @classmethod
    def undead_damage_mult(cls, ro_hour: int | None = None) -> float:
        """Undead damage multiplier based on time of day."""
        return 1.5 if cls.is_night(ro_hour) else 1.0


class WeatherSystem:
    """Tracks RO weather effects.
    
    Some maps have permanent weather:
    - Comodo: always raining (Fire -25%)
    - Mount Mjolnir: always windy (Wind +25%)
    
    Some weather changes dynamically.
    """
    
    MAP_WEATHER = {
        "comodo": "rain",
        "cmd_fild01": "rain",
        "cmd_fild02": "rain",
        "mjolnir_01": "wind",
        "mjolnir_02": "wind",
        "mjolnir_03": "wind",
        "mjolnir_04": "wind",
        "umbala": "snow",
        "um_fild01": "snow",
        "um_fild02": "snow",
        "um_fild03": "snow",
    }
    
    # Element damage modifiers by weather
    WEATHER_MODIFIERS = {
        "rain": {"Fire": 0.75, "Water": 1.25, "Wind": 0.9, "Earth": 1.1},
        "wind": {"Wind": 1.25, "Earth": 0.75, "Water": 0.9, "Fire": 1.1},
        "snow": {"Water": 1.25, "Fire": 0.75, "Wind": 1.1, "Earth": 0.9},
    }
    
    @classmethod
    def get_weather(cls, map_name: str) -> str | None:
        """Get weather for a given map."""
        return cls.MAP_WEATHER.get(map_name)
    
    @classmethod
    def get_element_modifier(cls, map_name: str, element: str) -> float:
        """Get element damage modifier from weather."""
        weather = cls.get_weather(map_name)
        if weather and weather in cls.WEATHER_MODIFIERS:
            return cls.WEATHER_MODIFIERS[weather].get(element, 1.0)
        return 1.0


class GuildRelations:
    """Tracks guild relations — allies, enemies, war status.
    
    Populated from signals sent by the bridge plugin.
    """
    
    def __init__(self):
        self._allies: set[str] = set()
        self._enemies: set[str] = set()
        self._war_zones: set[str] = set()
        self._castle_owners: dict[str, str] = {}  # castle_name -> guild_name
    
    def update(self, signals: dict[str, Any]) -> None:
        guild_data = signals.get("guild", {})
        if isinstance(guild_data, dict):
            self._allies = set(guild_data.get("allies", []))
            self._enemies = set(guild_data.get("enemies", []))
            self._war_zones = set(guild_data.get("war_zones", []))
        woe_data = signals.get("woe", {})
        if isinstance(woe_data, dict):
            self._castle_owners = woe_data.get("castles", {})
    
    def is_ally(self, player_name: str) -> bool:
        return player_name in self._allies
    
    def is_enemy(self, player_name: str) -> bool:
        return player_name in self._enemies
    
    def is_war_zone(self, map_name: str) -> bool:
        return map_name in self._war_zones
    
    def is_woe_active(self) -> bool:
        now = datetime.now(timezone.utc)
        weekday = now.weekday()  # 0=Monday, 6=Sunday
        hour = now.hour
        # WoE: Wed/Thu/Sat 20:00-22:00 (server time, typically UTC+8)
        if weekday in [2, 3, 5]:  # Wed, Thu, Sat
            if 12 <= hour <= 14:  # 20:00-22:00 UTC+8 = 12:00-14:00 UTC
                return True
        return False


class DeathTracker:
    """Tracks recent deaths on the current map.
    
    If multiple players die on the same map in quick succession,
    something dangerous is there.
    """
    
    def __init__(self, window_minutes: int = 10):
        self._window = timedelta(minutes=window_minutes)
        self._deaths: list[tuple[str, datetime]] = []
    
    def report_death(self, map_name: str) -> None:
        self._deaths.append((map_name, datetime.now(timezone.utc)))
        self._prune()
    
    def recent_deaths_on_map(self, map_name: str) -> int:
        self._prune()
        now = datetime.now(timezone.utc)
        return sum(1 for m, t in self._deaths if m == map_name and (now - t) <= self._window)
    
    def is_dangerous(self, map_name: str, threshold: int = 3) -> bool:
        """Check if recent deaths exceed danger threshold."""
        return self.recent_deaths_on_map(map_name) >= threshold
    
    def _prune(self) -> None:
        now = datetime.now(timezone.utc)
        self._deaths = [(m, t) for m, t in self._deaths if (now - t) <= self._window]


class MVPTracker:
    """Tracks MVP spawn timers and locations.
    
    Each MVP has a respawn timer (typically 60-120 minutes).
    Knowing when and where an MVP spawns lets the bot camp or avoid it.
    """
    
    _DATA_PATH = DATA_DIR / "mvp_data.yaml"
    
    def __init__(self):
        self._mvps: dict[str, dict] = {}
        self._sightings: dict[str, datetime] = {}  # mvp_name -> last_kill_time
        self._load_data()
    
    def _load_data(self) -> None:
        if yaml is None:
            return
        path = self._DATA_PATH
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            self._mvps = data.get("mvps", {})
            logger.info(f"Loaded {len(self._mvps)} MVP entries")
    
    def get_mvp(self, name: str) -> dict | None:
        return self._mvps.get(name)
    
    def report_kill(self, mvp_name: str) -> None:
        self._sightings[mvp_name] = datetime.now(timezone.utc)
    
    def time_until_respawn(self, mvp_name: str) -> float | None:
        """Minutes until MVP respawns, or None if not tracked."""
        if mvp_name not in self._mvps:
            return None
        if mvp_name not in self._sightings:
            return 0.0  # Available now (no known kill)
        respawn_minutes = self._mvps[mvp_name].get("respawn_minutes", 120)
        elapsed = (datetime.now(timezone.utc) - self._sightings[mvp_name]).total_seconds() / 60
        return max(0.0, respawn_minutes - elapsed)
    
    def can_kill(self, mvp_name: str, party_dps: float) -> bool:
        """Check if the party can kill this MVP based on DPS check."""
        mvp = self._mvps.get(mvp_name)
        if not mvp:
            return False
        mvp_hp = mvp.get("hp", 100000)
        enrage_minutes = mvp.get("enrage_minutes", 10)
        # Total damage needed = HP / party DPS per minute
        minutes_to_kill = mvp_hp / max(party_dps * 60, 1)
        return minutes_to_kill < enrage_minutes
    
    def is_near_spawn(self, mvp_name: str, current_map: str, x: int, y: int, distance: int = 15) -> bool:
        """Check if the bot is near an MVP spawn point."""
        mvp = self._mvps.get(mvp_name)
        if not mvp:
            return False
        spawn_map = mvp.get("map")
        if spawn_map != current_map:
            return False
        sx, sy = mvp.get("spawn", [0, 0])
        return abs(x - sx) <= distance and abs(y - sy) <= distance

    def get_mvps_on_map(self, map_name: str) -> list[dict]:
        """Get all MVPs that spawn on the given map."""
        return [
            {"name": k, **v}
            for k, v in self._mvps.items()
            if v.get("map") == map_name
        ]


class WorldState:
    """Aggregate world state — everything a human player knows without thinking."""
    
    def __init__(self):
        self.time_of_day = TimeOfDay()
        self.weather = WeatherSystem()
        self.guilds = GuildRelations()
        self.deaths = DeathTracker()
        self.mvps = MVPTracker()
    
    def assess(self, signals: dict[str, Any], actions: list[HeuristicAction], bot_id: str) -> None:
        """Emit world state awareness as HeuristicActions."""
        current_map = str(signals.get("map", "") or "")
        ro_hour = TimeOfDay.get_ro_hour()
        
        # Night check
        if TimeOfDay.is_night(ro_hour):
            actions.append(HeuristicAction(
                kind="log",
                command=f"world_state night={ro_hour} undead_mult=1.5",
                confidence=0.8,
                reason=f"Night time: undead deal 1.5x damage",
                domain="world",
            ))
        
        # Weather check
        weather = self.weather.get_weather(current_map)
        if weather:
            actions.append(HeuristicAction(
                kind="log",
                command=f"world_state weather={weather} map={current_map}",
                confidence=0.8,
                reason=f"Weather effect: {weather} on {current_map}",
                domain="world",
            ))
        
        # Death danger check
        death_count = self.deaths.recent_deaths_on_map(current_map)
        if death_count >= 3:
            actions.append(HeuristicAction(
                kind="command",
                command=f"mon_control * 0 0 1",
                confidence=0.8,
                reason=f"Danger: {death_count} recent deaths on {current_map}",
                domain="safety",
            ))
        
        # MVP proximity check (if current map has MVP spawn)
        mvps = self.mvps.get_mvps_on_map(current_map)
        for mvp in mvps:
            if self.mvps.time_until_respawn(mvp["name"]) == 0.0:
                # MVP is available
                actions.append(HeuristicAction(
                    kind="log",
                    command=f"mvp_available {mvp['name']} on {current_map}",
                    confidence=0.7,
                    reason=f"MVP {mvp['name']} is alive on this map!",
                    domain="mvp",
                ))
        
        # Party registration
        self.guilds.update(signals)
        if self.guilds.is_woe_active():
            actions.append(HeuristicAction(
                kind="log",
                command=f"woe_active war_zones={list(self.guilds._war_zones)[:3]}",
                confidence=0.9,
                reason="War of Emperium is active",
                domain="pvp",
            ))


# Singleton
_world_state: WorldState | None = None


def get_world_state() -> WorldState:
    global _world_state
    if _world_state is None:
        _world_state = WorldState()
    return _world_state
