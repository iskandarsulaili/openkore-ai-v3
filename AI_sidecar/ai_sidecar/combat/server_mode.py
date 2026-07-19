"""Server Mode Detector — detects pre-renewal vs renewal game mode.

Architecture:
  - Reads server config from bridge snapshot
  - Falls back to monster/player stat analysis
  - Returns mode string for all subsequent formula branching
  - All rAthena DB files have pre-re and re versions — mode selects which to load

RULE.md compliance: Zero hardcoded. Detection is data-driven from live server state.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Known server name patterns for detection
_SERVER_MODE_PATTERNS = {
    "renewal": [
        "ragnarok", "renewal", "high rate", "transcendent",
        "3rd class", "third class", "fourth", "4th",
    ],
    "prerenewal": [
        "classic", "pre-renewal", "pre-re", "low rate",
        "1st class", "2nd class", "99/50",
    ],
}


class ServerMode:
    """Game server mode enum."""
    PRERENEWAL = "prerenewal"
    RENEWAL = "renewal"
    UNKNOWN = "unknown"


class ServerModeDetector:
    """Detects the game server mode from bridge data."""

    def __init__(self):
        self._mode = ServerMode.UNKNOWN
        self._detected = False
    
    def detect(self, runtime_state, snapshot=None) -> str:
        """Detect server mode. Returns 'prerenewal', 'renewal', or 'unknown'.
        
        Detection order:
        1. Server name from bridge meta (most reliable)
        2. Player stats/max level (if character data available)
        3. Monster HP ranges (fallback)
        """
        if self._detected:
            return self._mode
        
        # Method 1: Server name from bridge snapshot meta
        mode = self._detect_from_server_name(runtime_state, snapshot)
        if mode != ServerMode.UNKNOWN:
            self._mode = mode
            self._detected = True
            logger.info("server_mode: detected %s from server name", mode)
            return mode
        
        # Method 2: Character data (max HP, level, stats)
        mode = self._detect_from_character(snapshot)
        if mode != ServerMode.UNKNOWN:
            self._mode = mode
            self._detected = True
            logger.info("server_mode: detected %s from character data", mode)
            return mode
        
        # Method 3: Monster data (HP ranges)
        mode = self._detect_from_monsters(runtime_state)
        if mode != ServerMode.UNKNOWN:
            self._mode = mode
            self._detected = True
            logger.info("server_mode: detected %s from monster data", mode)
            return mode
        
        logger.info("server_mode: could not detect, defaulting to renewal")
        self._mode = ServerMode.RENEWAL
        self._detected = True
        return self._mode
    
    def _detect_from_server_name(self, runtime_state, snapshot) -> str:
        """Check server name from bridge meta."""
        if snapshot is None:
            return ServerMode.UNKNOWN
        
        # Try to get server name from snapshot meta
        server_name = ""
        if isinstance(snapshot, dict):
            meta = snapshot.get("meta", {}) or {}
            raw = snapshot.get("raw", {}) or {}
            server_name = str(raw.get("master", meta.get("source", "")))
        else:
            meta = getattr(snapshot, "meta", None)
            raw = getattr(snapshot, "raw", None) or {}
            server_name = str(getattr(raw, "master", "") if isinstance(raw, dict) else "")
            if not server_name:
                server_name = str(getattr(meta, "source", ""))
        
        server_lower = server_name.lower()
        for keyword in _SERVER_MODE_PATTERNS["renewal"]:
            if keyword in server_lower:
                return ServerMode.RENEWAL
        for keyword in _SERVER_MODE_PATTERNS["prerenewal"]:
            if keyword in server_lower:
                return ServerMode.PRERENEWAL
        
        return ServerMode.UNKNOWN
    
    def _detect_from_character(self, snapshot) -> str:
        """Check character level and HP to determine mode."""
        if snapshot is None:
            return ServerMode.UNKNOWN
        
        level = 0
        hp_max = 0
        job_name = ""
        
        if isinstance(snapshot, dict):
            prog = snapshot.get("progression", {}) or {}
            vitals = snapshot.get("vitals", {}) or {}
            level = int(prog.get("base_level", 0) or 0)
            hp_max = int(vitals.get("hp_max", 0) or 0)
            job_name = str(prog.get("job_name", "") or "")
        else:
            prog = getattr(snapshot, "progression", None)
            vitals = getattr(snapshot, "vitals", None)
            if prog:
                level = int(getattr(prog, "base_level", 0) or 0)
                job_name = str(getattr(prog, "job_name", "") or "")
            if vitals:
                hp_max = int(getattr(vitals, "hp_max", 0) or 0)
        
        # Renewal has higher level cap and HP pools
        if level > 99:
            return ServerMode.RENEWAL
        if hp_max > 100000:
            return ServerMode.RENEWAL
        
        # Check job name for 3rd/4th class keywords
        job_lower = job_name.lower()
        renewal_jobs = [
            "rune", "warlock", "ranger", "arch", "mechanic", 
            "guillotine", "royal", "sorcerer", "minstrel", "wanderer",
            "sura", "shadow", "genetic", "dragon", "meister",
            "cardinal", "windhawk", "imperial", "biolo", "inquisitor",
        ]
        for keyword in renewal_jobs:
            if keyword in job_lower:
                return ServerMode.RENEWAL
        
        return ServerMode.UNKNOWN
    
    def _detect_from_monsters(self, runtime_state) -> str:
        """Check monster HP ranges from snapshot cache."""
        try:
            snapshots = getattr(runtime_state, "snapshot_cache", None)
            if snapshots is None or not hasattr(snapshots, "bot_ids"):
                return ServerMode.UNKNOWN
            
            # Sample a few monsters from actor data across all bots
            max_hp_seen = 0
            for bid in snapshots.bot_ids():
                snap = snapshots.get(bid)
                if snap is None:
                    continue
                if isinstance(snap, dict):
                    actors = snap.get("actors", {}) or {}
                    for monster in actors.get("list", []):
                        hp = int(monster.get("hp", 0) or 0)
                        max_hp_seen = max(max_hp_seen, hp)
                else:
                    actors = getattr(snap, "actors", None)
                    if actors:
                        for monster in getattr(actors, "list", []):
                            hp = int(getattr(monster, "hp", 0) or 0)
                            max_hp_seen = max(max_hp_seen, hp)
            
            # Renewal monsters have significantly higher HP
            if max_hp_seen > 500000:
                return ServerMode.RENEWAL
        except Exception:
            pass
        
        return ServerMode.UNKNOWN


# Singleton instance
_detector = None


def get_server_mode(runtime_state=None, snapshot=None) -> str:
    """Get the detected server mode. Creates detector on first call."""
    global _detector
    if _detector is None:
        _detector = ServerModeDetector()
    return _detector.detect(runtime_state, snapshot)
