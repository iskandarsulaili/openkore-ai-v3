"""
NPC Auto-Discovery Engine — Finds service NPCs dynamically from game state.
===========================================================================
No hardcoded coordinates. The AI reads NPC data from the live snapshot,
matches NPC names against known service patterns, and routes to the
nearest matching NPC. Works on any server without configuration.
"""

from __future__ import annotations

import logging
from threading import RLock
import math
from typing import Any

logger = logging.getLogger(__name__)


# NPC service patterns — name fragments that identify service NPCs
# These are server-agnostic name patterns, not hardcoded NPC IDs
NPC_SERVICE_PATTERNS = {
    "storage": [
        "kafra", "storage", "keeper", "warehouse", "bank", "vault",
        "保管", "倉庫", "银行", "보관",
    ],
    "vendor": [
        "tool", "dealer", "shop", "item", "mart", "store", "merchant",
        "trade", "barter", "pawn", "exchange",
        "道具", "商人", "商店", "物品", "상점",
    ],
    "buyer": [
        "buyer", "purchase", "collect", "回收", "购买", "구매",
    ],
    "healer": [
        "heal", "nun", "nurse", "priest", "monk", "recovery",
        "治愈", "治疗", "회복",
    ],
    "refiner": [
        "refine", "smith", "forge", "upgrade", "enchant",
        "精炼", "强化", "강화",
    ],
    "identifier": [
        "identify", "appraise", "鉴定", "감정",
    ],
    "quest": [
        "quest", "mission", "notice", "board", "guide", "eden",
        "任务", "公告", "퀘스트",
    ],
    "job_change": [
        "job", "class", "master", "guild", "association",
        "转职", "职业", "전직",
    ],
}


class NPCDiscoveryEngine:
    """Discovers NPC positions dynamically from game state snapshots.

    The bot reads NPC data from the live snapshot, matches names against
    known service patterns, and provides the nearest matching NPC's
    position for route commands. Works on any server without configuration.

    Key insight: instead of hardcoding "prontera 181 186", the AI scans
    the actor list for NPCs whose names match service patterns, then
    routes to the nearest one.
    """

    def __init__(self):
        self._lock = RLock()
        self._discovered_npcs: dict[str, dict[str, Any]] = {}  # map_name -> npc info
        self._last_scan_map: str = ""

    def discover_vendor_npc(
        self,
        snapshot: Any,
        current_map: str,
        service: str = "vendor",
    ) -> dict[str, Any] | None:
        """Find the nearest NPC providing a service in the current map.

        Scans the snapshot's actor list for NPCs whose names match
        the given service pattern. Returns the closest NPC's position
        and name, or None if no matching NPC is found.

        This replaces hardcoded 'sellAuto_npc prontera 181 186'.
        """
        if not snapshot:
            return None

        patterns = NPC_SERVICE_PATTERNS.get(service, [])
        if not patterns:
            return None

        # Get bot position from snapshot
        bot_x = 0
        bot_y = 0
        if isinstance(snapshot, dict):
            pos = snapshot.get("position", {}) or {}
            bot_x = int(pos.get("x", 0) or 0)
            bot_y = int(pos.get("y", 0) or 0)
        else:
            pos = getattr(snapshot, "position", None) or {}
            bot_x = int(getattr(pos, "x", 0) or 0)
            bot_y = int(getattr(pos, "y", 0) or 0)

        # Scan actors for matching NPCs
        actors = []
        if isinstance(snapshot, dict):
            actors = snapshot.get("actors", []) or []
        else:
            actors = getattr(snapshot, "actors", []) or []

        best_npc = None
        best_dist = float("inf")

        for actor in actors:
            actor_type = ""
            actor_name = ""
            actor_x = 0
            actor_y = 0

            if isinstance(actor, dict):
                actor_type = str(actor.get("actor_type", "") or "")
                actor_name = str(actor.get("name", "") or "")
                actor_x = int(actor.get("x", 0) or 0)
                actor_y = int(actor.get("y", 0) or 0)
            else:
                actor_type = str(getattr(actor, "actor_type", "") or "")
                actor_name = str(getattr(actor, "name", "") or "")
                actor_x = int(getattr(actor, "x", 0) or 0)
                actor_y = int(getattr(actor, "y", 0) or 0)

            if actor_type.lower() != "npc":
                continue

            # Check if NPC name matches any service pattern
            name_lower = actor_name.lower()
            for pattern in patterns:
                if pattern in name_lower:
                    dist = math.sqrt((actor_x - bot_x) ** 2 + (actor_y - bot_y) ** 2)
                    if dist < best_dist:
                        best_dist = dist
                        best_npc = {
                            "name": actor_name,
                            "x": actor_x,
                            "y": actor_y,
                            "distance": int(dist),
                            "service": service,
                            "map": current_map,
                        }
                    break

        if best_npc:
            # Cache the discovered NPC
            self._discovered_npcs[f"{current_map}:{service}"] = best_npc
            self._last_scan_map = current_map
            logger.info(
                "npc_discovered: map=%s service=%s npc=%s x=%d y=%d dist=%d",
                current_map, service, best_npc["name"], best_npc["x"],
                best_npc["y"], best_npc["distance"],
            )

        return best_npc

    def get_command_for_service(
        self,
        snapshot: Any,
        current_map: str,
        service: str = "vendor",
    ) -> str | None:
        """Generate an OpenKore command to interact with a service NPC.

        Returns a talknpc command with the discovered NPC's position,
        or a move command to approach the NPC. Returns None if no NPC
        is found.
        """
        npc = self.discover_vendor_npc(snapshot, current_map, service)
        if not npc:
            return None

        # If far from NPC, move closer first
        if npc["distance"] > 5:
            return f"move {npc['x']} {npc['y']}"

        # Close enough to talk
        return f"talknpc {npc['x']} {npc['y']}"

    def get_nearest_town_for_map(self, map_name: str) -> str:
        """Determine the nearest town for a given map name.

        Uses map name prefixes to infer the town. Returns the town name
        that OpenKore can route to.
        """
        map_lower = map_name.lower()
        if "payon" in map_lower or "pay_" in map_lower:
            return "payon"
        elif "morocc" in map_lower or "moc_" in map_lower:
            return "morocc"
        elif "geffen" in map_lower or "gef_" in map_lower:
            return "geffen"
        elif "aldebaran" in map_lower or "alde_" in map_lower:
            return "aldebaran"
        elif "yuno" in map_lower:
            return "yuno"
        elif "xmas" in map_lower:
            return "xmas"
        elif "amatsu" in map_lower or "ama_" in map_lower:
            return "amatsu"
        elif "prontera" in map_lower or "prt_" in map_lower or "mjolnir" in map_lower:
            return "prontera"
        return "prontera"  # Safe default

    def get_stats(self) -> dict[str, Any]:
        """Get discovery stats."""
        return {
            "discovered_npcs": len(self._discovered_npcs),
            "last_scan_map": self._last_scan_map,
            "services": list(NPC_SERVICE_PATTERNS.keys()),
        }
# ── Map Access Requirements ──────────────────────────────────────────────
# Maps that require NPC interaction to enter/leave.
MAP_ACCESS_REQUIREMENTS: dict[str, dict[str, object]] = {
    "pay_dun00": {"town": "payon", "npc_patterns": ["dungeon", "portal", "man", "payon", "cave"],
                  "description": "Payon Dungeon entrance via NPC near (115, 175) in payon"},
    "pay_dun01": {"town": "payon", "npc_patterns": ["dungeon", "portal", "man", "payon", "cave"],
                  "description": "Payon Dungeon 2F via NPC in payon"},
    "gef_dun00": {"town": "geffen", "npc_patterns": ["dungeon", "portal", "guard", "geffen", "tower"],
                  "description": "Geffen Dungeon entrance via NPC in geffen"},
    "moc_dun00": {"town": "morocc", "npc_patterns": ["dungeon", "pyramid", "portal", "morocc"],
                  "description": "Morroc Pyramid entrance via NPC in morocc"},
    "amatsu": {"town": "alberta", "npc_patterns": ["ship", "captain", "ferry", "sail", "amatsu"],
               "description": "Amatsu requires ship NPC in alberta"},
    "gonryun": {"town": "yuno", "npc_patterns": ["portal", "gate", "temple", "monk"],
                "description": "Gonryun requires NPC interaction in yuno"},
    "tur_dun01": {"town": "aldebaran", "npc_patterns": ["ship", "captain", "turtle", "island"],
                  "description": "Turtle Island requires ship NPC in aldebaran"},
    "c_tower1": {"town": "aldebaran", "npc_patterns": ["clock", "tower", "portal", "aldebaran"],
                 "description": "Clock Tower entrance via NPC in aldebaran"},
}


def requires_npc_access(map_name: str) -> dict[str, object] | None:
    """Check if a map requires NPC interaction to access."""
    map_lower = map_name.lower().replace(".gat", "")
    for dungeon_map, info in MAP_ACCESS_REQUIREMENTS.items():
        if dungeon_map in map_lower or map_lower in dungeon_map:
            return info
    return None


def get_town_for_gated_map(map_name: str) -> str | None:
    """Get the town where the access NPC lives for a gated map."""
    access = requires_npc_access(map_name)
    if access:
        return str(access["town"])
    return None
