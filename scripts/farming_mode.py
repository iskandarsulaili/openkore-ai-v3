#!/usr/bin/env python3
"""
Farming Mode — Configure OpenKore for reliable out-of-the-box farming.
===============================================================
Sets up all 3 bot profiles with proper farming config so they
work immediately without any manual configuration.

The secret: OpenKore's built-in AI is already mature (auto-attack,
auto-loot, auto-move, auto-sell). The sidecar just needs to configure
it correctly, then get out of the way.
"""

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Recommended hunting maps by level range (server-agnostic — common rAthena maps)
FARMING_ZONES = {
    "1-10":  ["prt_fild08", "prt_fild04", "moc_fild01", "pay_fild01"],
    "11-20": ["prt_fild04", "pay_fild08", "moc_fild02", "gef_fild01"],
    "21-35": ["pay_fild04", "moc_fild03", "gef_fild02", "mjolnir_04"],
    "36-50": ["pay_fild01", "moc_fild17", "gef_fild05", "mjolnir_07"],
    "51-70": ["gef_fild10", "moc_fild07", "pay_fild11", "xmas_fild01"],
    "71-90": ["gef_fild14", "moc_fild10", "pay_fild09", "yuno_fild07"],
    "91+":   ["gef_fild14", "moc_fild11", "yuno_fild08", "ama_fild01"],
}

FARMING_CONFIG = {
    # ── Combat ──
    "attackAuto": "2",
    "attackAuto_routeToLock": "1",
    "attackAuto_outOfLock": "1",
    "attackAuto_party": "1",
    "attackAuto_onlyWhenSafe": "0",
    "attackAuto_followTarget": "0",
    "attackAuto_notInTown": "1",
    "attackAuto_notWhile_storageAuto": "1",
    "attackAuto_notWhile_buyAuto": "1",
    "attackAuto_notWhile_sellAuto": "1",
    "attackAuto_considerDamagedAggressive": "0",
    "attackAuto_considerAggressiveIfCastOnCastSensor": "0",

    # ── Looting ──
    "itemsTakeAuto": "2",
    "itemsTakeAuto_party": "0",
    "items_gather_auto": "2",
    "items_maxWeight": "49",
    "items_gather_autoDelay": "100",

    # ── Movement ──
    "route_randomWalk": "1",
    "route_randomWalk_inLockOnly": "1",
    "route_randomWalk_inTown": "0",
    "route_randomWalk_maxRouteTime": "75",
    "route_maxWarpFee": "0",
    "route_teleport": "0",
    "route_escape_unknownMap": "0",
    "route_escape_reachedNoPortal": "0",
    "route_escape_randomWalk": "0",
    "route_escape_shout": "0",

    # ── Teleport ──
    "teleportAuto_hp": "10",
    "teleportAuto_sp": "0",
    "teleportAuto_idle": "0",
    "teleportAuto_portal": "0",
    "teleportAuto_search": "0",
    "teleportAuto_minAggressives": "0",
    "teleportAuto_minAggressivesInLock": "0",
    "teleportAuto_onlyWhenSafe": "0",
    "teleportAuto_maxDmg": "0",
    "teleportAuto_maxDmgInLock": "0",
    "teleportAuto_deadly": "0",
    "teleportAuto_useSkillForDailyWarp": "0",
    "teleportAuto_useSkill": "3",
    "teleportAuto_useSkill_forLock": "0",

    # ── Death Handling ──
    "autoMoveOnDeath": "1",
    "autoMoveOnDeath_x": "156",
    "autoMoveOnDeath_y": "191",
    "autoMoveOnDeath_map": "prontera",

    # ── Auto-Sell ──
    "sellAuto": "1",
    "sellAuto_npc": "prt_in 5 108",
    "sellAuto_npc_steps": "n",
    "sellAuto_distance": "5",
    "sellAuto_standby": "0",

    # ── Auto-Store ──
    "storageAuto": "1",
    "storageAuto_npc": "prt_in 6 110",
    "storageAuto_npc_distance": "5",
    "storageAuto_npc_steps": "n",
    "storageAuto_standby": "0",
    "storage_useFlash": "1",
    "storageAuto_notAfterDeath": "0",

    # ── Auto-Buy ──
    "buyAuto": "0",

    # ── Party ──
    "partyAuto": "1",
    "partyAutoShare": "1",

    # ── Misc ──
    "equipAuto": "0",
    "expAuto": "0",
}


def generate_farming_config(profile_path: Path, bot_level: int = 1, zone: str = "") -> str:
    """Generate a complete farming config.txt for a bot profile."""
    lines = []

    # Read existing config to preserve account info
    existing = {}
    if profile_path.exists():
        with open(profile_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if " " in line:
                    key, _, val = line.partition(" ")
                    existing[key] = val

    # Preserve account credentials
    if "username" in existing:
        lines.append(f"username {existing['username']}")
    if "password" in existing:
        lines.append(f"password {existing['password']}")
    lines.append("")

    # Apply farming config
    for key, val in FARMING_CONFIG.items():
        lines.append(f"{key} {val}")
    lines.append("")

    # Set lockMap to the recommended zone
    if zone:
        lines.append(f"lockMap {zone}")
    else:
        # Auto-detect based on level
        ranges = sorted(FARMING_ZONES.keys())
        for r in ranges:
            low, high = r.split("-")
            if low <= str(bot_level) <= high:
                lines.append(f"lockMap {FARMING_ZONES[r][0]}")
                break
        else:
            lines.append("lockMap prt_fild08")
    lines.append("lockMap_x")
    lines.append("lockMap_y")
    lines.append("lockMap_randX 0")
    lines.append("lockMap_randY 0")
    lines.append("")

    # Preserve any other existing keys that aren't in our config
    preserved_keys = {"username", "password", "char", "charSelectTimeout",
                      "loginPinCode", "poseidonServer", "poseidonPort",
                      "poseidonTimeout", "ignoreInvalidLogin", "messageLength",
                      "XKore"}
    for key, val in existing.items():
        if key not in FARMING_CONFIG and key not in preserved_keys:
            lines.append(f"{key} {val}")

    return "\n".join(lines)


def recommend_zone_for_level(level: int, bot_count: int = 1, bot_index: int = 0) -> str:
    """Recommend a hunting zone, distributing across bots for swarm play."""
    ranges = sorted(FARMING_ZONES.keys())
    for r in ranges:
        low_str, high_str = r.split("-")
        low, high = int(low_str), int(high_str)
        if low <= level <= high:
            zones = FARMING_ZONES[r]
            if bot_count > 1 and len(zones) >= bot_count:
                # Assign each bot a different zone
                return zones[bot_index % len(zones)]
            return zones[0]

    # Fallback for high levels
    zones = FARMING_ZONES["91+"]
    if bot_count > 1 and len(zones) >= bot_count:
        return zones[bot_index % len(zones)]
    return zones[0]


def setup_farming_profiles(base_dir: str, bot_levels: dict[str, int]) -> list[str]:
    """Set up farming config for all bot profiles."""
    base = Path(base_dir)
    profiles = []

    bot_names = list(bot_levels.keys())
    for idx, (bot_name, level) in enumerate(bot_levels.items()):
        profile_dir = base / f".bot_profiles/{bot_name}/control"
        config_path = profile_dir / "config.txt"

        if not profile_dir.exists():
            logger.warning("Profile directory not found: %s", profile_dir)
            continue

        zone = recommend_zone_for_level(level, len(bot_levels), idx)
        config = generate_farming_config(config_path, level, zone)

        # Backup original config
        if config_path.exists():
            backup = config_path.with_suffix(".txt.original")
            if not backup.exists():
                config_path.rename(backup)

        with open(config_path, "w") as f:
            f.write(config)

        profiles.append({
            "bot_id": bot_name,
            "level": level,
            "zone": zone,
            "config": str(config_path),
        })
        logger.info("Farming config generated for %s → %s (zone: %s)", bot_name, config_path, zone)

    return profiles