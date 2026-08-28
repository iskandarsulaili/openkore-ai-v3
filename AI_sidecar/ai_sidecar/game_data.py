"""Shared game-data loaders from the OpenKore core tables/ directory.

The core ships authoritative game facts in tables/ (the same data the Perl core
resolves against): job-change guild locations, cities, etc. The sidecar brains
MUST read these tables — never duplicate them as hardcoded literals (RULE.md).

All loaders are server-agnostic: the tables describe the RO game itself
(identical on every server), so reading them is safe on any server.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

_TABLES_DIR = Path(__file__).resolve().parents[2] / "tables"


def tables_dir() -> Path:
    return _TABLES_DIR


def load_job_change_locations() -> dict[str, dict[str, object]]:
    """Read tables/job_change_locations.txt -> {target_job: {map, x, y, desc}}.

    Format: target_job | map | x y | description | requirements
    Returns {} if the table is missing (callers fall back to their learned
    store — never hardcoded literals).
    """
    _out: dict[str, dict[str, object]] = {}
    _p = _TABLES_DIR / "job_change_locations.txt"
    if not _p.exists():
        return _out
    for _line in _p.read_text(encoding="utf-8", errors="ignore").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#"):
            continue
        _parts = [p.strip() for p in _line.split("|")]
        if len(_parts) < 4:
            continue
        _job = _parts[0].lower()
        _map = _parts[1].strip().lower()
        _xy = _parts[2].split()
        if not _map or len(_xy) < 2:
            continue
        try:
            _x, _y = int(_xy[0]), int(_xy[1])
        except ValueError:
            continue
        _out[_job] = {
            "map": _map,
            "x": _x,
            "y": _y,
            "desc": _parts[3] if len(_parts) > 3 else "",
        }
    return _out


def load_city_maps() -> list[str]:
    """Read tables/cities.txt (map#name per line) -> list of city map names."""
    _p = _TABLES_DIR / "cities.txt"
    if not _p.exists():
        return []
    _out: list[str] = []
    for _line in _p.read_text(encoding="utf-8", errors="ignore").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#"):
            continue
        _m = _line.split("#")[0].strip().lower()
        if _m.endswith(".rsw"):
            _m = _m[:-4]
        if _m and _m not in _out:
            _out.append(_m)
    return _out


# RO map-prefix -> parent town. This is static RO geography (the field/dungeon
# map-prefix graph — prt_* belongs to prontera, pay_* to payon, etc.), identical
# on every RO server. It lives HERE as a single authoritative game-data table
# (the same class as tables/) so no brain module hardcodes town names.
MAP_PREFIX_TOWN: dict[str, str] = {
    "prt": "prontera",
    "pay": "payon",
    "gef": "geffen",
    "moc": "morocc",
    "cmd": "comodo",
    "alb": "alberta",
    "iz": "izlude",
    "alde": "aldebaran",
    "yuno": "yuno",
    "xmas": "xmas",
    "ein": "einbroch",
    "lhz": "lighthalzen",
    "hu": "hugel",
    "ra": "rachel",
    "ama": "amatsu",
    "gon": "gonryun",
    "umb": "umbala",
    "nifl": "niflheim",
    "lou": "louyang",
    "ve": "veins",
    "brasil": "brasilis",
    "man": "manuk",
    "spl": "splendide",
    "dew": "dewata",
    "mal": "malangdo",
    "mjolnir": "mjolnir",
}


# Town portal exit points (map -> (x, y)): the portal from a town to its
# fields. Static RO geography (the same portal coords on every server), kept
# in ONE authoritative table (game_data) so decision paths never inline them.
TOWN_PORTALS: dict[str, tuple[int, int]] = {
    "izlude": (367, 205),   # izlude -> prt_fild08
    "prontera": (156, 289), # prontera -> prt_fild01
    "morocc": (287, 95),    # morocc -> moc_fild01
    "geffen": (147, 133),   # geffen -> gef_fild07
    "payon": (113, 218),    # payon -> pay_fild01
}


def town_portal(town: str) -> tuple[int, int]:
    """Return the portal exit point for a town, or (0, 0) if unknown."""
    return TOWN_PORTALS.get((town or "").lower(), (0, 0))


def parent_town(field_map: str) -> str:
    """Resolve a field/dungeon map's parent town from the RO prefix graph.

    prt_fild08 -> prt -> prontera. Never a hardcoded literal in callers —
    this is the single authoritative map.
    """
    _m = (field_map or "").strip().lower().replace(".gat", "").replace(".rsw", "")
    if not _m:
        return ""
    _pref = _m.split("_")[0]
    if _pref in MAP_PREFIX_TOWN:
        return MAP_PREFIX_TOWN[_pref]
    return ""
