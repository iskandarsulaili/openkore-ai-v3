"""EnvironmentState — map time, weather, and environment properties."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EnvironmentState(BaseModel):
    """Current map environment — time, weather, and properties."""

    model_config = ConfigDict(extra="ignore")

    # ── Time ──
    map_time: int = 0  # In-game time (0-1440 minutes)
    time_of_day: str = "day"  # dawn | day | dusk | night
    is_night: bool = False

    # ── Weather ──
    weather: str = "none"  # none | rain | snow | fog | sandstorm
    has_weather_effect: bool = False

    # ── Map flags ──
    is_pvp: bool = False
    is_gvg: bool = False  # Guild vs Guild zone
    is_vs: bool = False  # PvP or GvG
    is_town: bool = False
    is_dungeon: bool = False
    is_field: bool = False
    is_indoor: bool = False
    is_deadly: bool = False  # Map deals damage (e.g. magma, poison zone)

    # ── Party sharing ──
    share_exp_range: int | None = None
    share_level_range: int | None = None

    # ── Misc ──
    map_id: int | None = None
    map_type: int | None = None  # rAthena map type bitfield
    raw: dict[str, Any] = Field(default_factory=dict)


def collect_environment(signals: dict[str, Any]) -> EnvironmentState:
    """Parse map environment properties from the bridge signal dict.

    Handles:
      - ``signals['environment']`` — dict with env info
      - ``signals['map_time']``, ``signals['is_night']`` — flat keys
      - ``signals['weather']`` — weather string
      - ``signals['map_type']`` — map type bitfield
    """
    e_dict: dict[str, Any] = signals.get("environment") or {}

    map_time = int(signals.get("map_time", e_dict.get("map_time", 0)))
    time_of_day = str(signals.get("time_of_day", e_dict.get("time_of_day", "day")))
    is_night = bool(signals.get("is_night", e_dict.get("is_night", time_of_day == "night")))
    weather = str(signals.get("weather", e_dict.get("weather", "none")))

    # Map type bitfield flags (rAthena map_types)
    map_type = int(signals.get("map_type", e_dict.get("map_type", 0)))
    is_pvp = bool(signals.get("is_pvp", e_dict.get("is_pvp", False)))
    is_gvg = bool(signals.get("is_gvg", e_dict.get("is_gvg", False)))
    is_town = bool(signals.get("is_town", e_dict.get("is_town", False)))
    is_dungeon = bool(signals.get("is_dungeon", e_dict.get("is_dungeon", False)))
    is_field = bool(signals.get("is_field", e_dict.get("is_field", False)))
    is_indoor = bool(signals.get("is_indoor", e_dict.get("is_indoor", False)))
    is_deadly = bool(signals.get("is_deadly", e_dict.get("is_deadly", False)))

    # If map_type bitfield is available, decode it
    if map_type > 0:
        is_pvp = is_pvp or bool(map_type & 0x0200)
        is_gvg = is_gvg or bool(map_type & 0x0400)

    return EnvironmentState(
        map_time=map_time,
        time_of_day=time_of_day,
        is_night=is_night,
        weather=weather,
        has_weather_effect=bool(signals.get("has_weather", e_dict.get("has_weather", weather != "none"))),
        is_pvp=is_pvp,
        is_gvg=is_gvg,
        is_vs=is_pvp or is_gvg,
        is_town=is_town,
        is_dungeon=is_dungeon,
        is_field=is_field,
        is_indoor=is_indoor,
        is_deadly=is_deadly,
        share_exp_range=int(signals.get("share_exp_range", e_dict.get("share_exp_range", 15))),
        share_level_range=int(signals.get("share_level_range", e_dict.get("share_level_range", 15))),
        map_id=int(signals.get("map_id", e_dict.get("map_id", 0))) or None,
        map_type=map_type or None,
    )
