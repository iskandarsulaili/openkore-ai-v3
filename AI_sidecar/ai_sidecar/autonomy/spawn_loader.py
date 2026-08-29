"""Agnostic spawn loader — parses the LIVE rAthena server's mob spawn scripts.

RULE.md compliance: no hardcoded map/monster literals. The server's own
`npc/re/mobs/**/*.txt` (and classic `npc/mobs/**/*.txt`) are the single source
of truth for what spawns on each map. The result replaces the hardcoded
map_spawns dicts in heuristic_service/ro_mechanics.

Spawn line format (rAthena):
  map,x,y<tab>monster<tab>Name<tab>AegisID,Count,RespawnMs[,event]

Returns: {map_name: [(monster_name, count, respawn_ms), ...]}
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# monster<tab>Name<tab>ID,Count,Respawn
_MONSTER_RE = re.compile(
    r"^\s*([\w@\-]+),(\d+),(\d+)\s+monster\s+([\w ']+?)\s+(\d+),(\d+),(\d+)"
)

_DEFAULT_ROOTS = (
    ("rathena-AI-world", "npc", "re", "mobs"),
    ("rathena-AI-world", "npc", "mobs"),
    ("rathena", "npc", "re", "mobs"),
    ("rathena", "npc", "mobs"),
)


def _find_spawn_root() -> str | None:
    """Locate the server's mob spawn scripts without hardcoding a repo path."""
    home = os.path.expanduser("~")
    for rel in _DEFAULT_ROOTS:
        p = os.path.join(home, *rel)
        if os.path.isdir(p):
            return p
    return None


def load_map_spawns(root: str | None = None) -> dict[str, list[tuple[str, int, int]]]:
    """Parse all mob spawn scripts into {map: [(monster, count, respawn_ms)]}."""
    root = root or _find_spawn_root()
    if not root or not os.path.isdir(root):
        logger.warning("spawn_loader: no server spawn root found (tried %s)", _DEFAULT_ROOTS)
        return {}

    spawns: dict[str, list[tuple[str, int, int]]] = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".txt"):
                continue
            path = os.path.join(dirpath, fn)
            try:
                with open(path, "r", errors="replace") as f:
                    for line in f:
                        m = _MONSTER_RE.match(line)
                        if not m:
                            continue
                        map_name, x, y, mob_name, aegis_id, count, respawn = m.groups()
                        try:
                            count_i = int(count)
                        except ValueError:
                            count_i = 0
                        try:
                            respawn_i = int(respawn)
                        except ValueError:
                            respawn_i = 0
                        spawns.setdefault(map_name, []).append(
                            (mob_name.strip(), count_i, respawn_i)
                        )
            except OSError as e:
                logger.warning("spawn_loader: cannot read %s: %s", path, e)
    return spawns


def merge_spawns(
    learned: dict[str, list[tuple[str, int, int]]],
    fallback: dict[str, list[tuple[str, int, int]]] | None = None,
) -> dict[str, list[tuple[str, int, int]]]:
    """Learned spawns win; fallback (if any) only fills maps the server doesn't define."""
    merged = dict(learned)
    for map_name, entries in (fallback or {}).items():
        if map_name not in merged:
            merged[map_name] = entries
    return merged
