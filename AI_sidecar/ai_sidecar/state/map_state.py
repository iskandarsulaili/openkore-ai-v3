"""MapState — current map, field info, portals in range, spawn data."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PortalInfo(BaseModel):
    """A portal/warp within range on the current map."""

    model_config = ConfigDict(extra="ignore")

    x: int = 0
    y: int = 0
    dest_map: str | None = None
    dest_x: int | None = None
    dest_y: int | None = None
    name: str | None = None


class MonsterSpawn(BaseModel):
    """A monster known to spawn on this map."""

    model_config = ConfigDict(extra="ignore")

    name: str = ""
    count: int = 0
    respawn_ms: int = 0


class MapState(BaseModel):
    """Current map information — name, field properties, nearby portals, spawns."""

    model_config = ConfigDict(extra="ignore")

    name: str = ""
    name_raw: str = ""
    is_town: bool = False
    is_dungeon: bool = False
    is_field: bool = False
    x: int = 0
    y: int = 0
    portals: list[PortalInfo] = Field(default_factory=list)
    spawns: list[MonsterSpawn] = Field(default_factory=list)
    connected_maps: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


# ── Well-known town map prefixes ──
_TOWN_PREFIXES: set[str] = {
    "prontera", "morocc", "geffen", "payon", "aldebaran",
    "alberta", "izlude", "comodo", "umbala", "niflheim",
    "rachel", "veins", "einbroch", "einbech", "lighthalzen",
    "juno", "hugel", "yuno", "amatsu", "gonryun",
    "louyang", "ayothaya", "jawaii", "brasilis", "moscovia",
    "manuk", "splendide",
}


def collect_map(signals: dict[str, Any]) -> MapState:
    """Parse map/position data from the bridge signal dict.

    Handles:
      - ``signals['map']`` — map name string (may include ``.gat`` suffix)
      - ``signals['x']``, ``signals['y']`` — current coordinates
      - ``signals['portals']`` — list of portal dicts
      - ``signals['position']`` — structured position dict
    """
    raw_map: str = str(signals.get("map", signals.get("position", {}).get("map", "")) or "")
    clean_map = raw_map.lower().replace(".gat", "")

    x = int(signals.get("x", signals.get("position", {}).get("x", 0)))
    y = int(signals.get("y", signals.get("position", {}).get("y", 0)))

    # Determine map type
    is_town = any(clean_map.startswith(p) and len(clean_map) <= len(p) + 3 for p in _TOWN_PREFIXES)
    is_dungeon = "_dun" in clean_map or "_01" in clean_map[-3:] or "_02" in clean_map[-3:]
    is_field = "_fild" in clean_map or "fild" in clean_map

    # Parse portals
    portals_raw: list[dict] = list(signals.get("portals", signals.get("portals_nearby", [])) or [])
    portals: list[PortalInfo] = []
    for p in portals_raw:
        if isinstance(p, dict):
            portals.append(PortalInfo(**{k: v for k, v in p.items() if k in PortalInfo.model_fields}))
        elif isinstance(p, (list, tuple)) and len(p) >= 3:
            portals.append(PortalInfo(x=int(p[0]), y=int(p[1]), dest_map=str(p[2])))

    # Parse spawns (provided by knowledge layer, not bridge signals)
    spawns_raw: list[dict] = list(signals.get("spawns", signals.get("map_spawns", [])) or [])
    spawns: list[MonsterSpawn] = []
    for s in spawns_raw:
        if isinstance(s, dict):
            spawns.append(MonsterSpawn(**{k: v for k, v in s.items() if k in MonsterSpawn.model_fields}))

    return MapState(
        name=clean_map,
        name_raw=raw_map,
        is_town=is_town,
        is_dungeon=is_dungeon,
        is_field=is_field,
        x=x,
        y=y,
        portals=portals,
        spawns=spawns,
    )
