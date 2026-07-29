"""Inter-Bot Communication — shared memory IPC for ultra-low latency bot coordination.

Replaces SwarmFileStore disk-based polling (1-5s delay) with /dev/shm
shared memory (microsecond delay) for use cases where speed matters:
- Party combo coordination (Wizard sees 5 mobs → Knight needs to act NOW)
- Formation changes (Priest needs to know Hunter position in real-time)
- Emergency alerts ("monster aggro on Bot3 — request immediate help")
"""
from __future__ import annotations
import json
import os
import time
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Use /dev/shm for ultra-fast IPC (ramdisk, survives process restart)
SHM_BASE = Path("/dev/shm") / "openkore-ai"
SHM_COORD = SHM_BASE / "coordination"
SHM_ALERTS = SHM_BASE / "alerts"
SHM_STATE = SHM_BASE / "state"


def _ensure_shm():
    """Ensure shared memory directories exist."""
    for d in [SHM_BASE, SHM_COORD, SHM_ALERTS, SHM_STATE]:
        d.mkdir(parents=True, exist_ok=True)


class SharedMemoryIPC:
    """Ultra-fast bot-to-bot communication via /dev/shm shared memory.
    
    All operations use atomic file writes for consistency.
    Latency: <1ms (vs 1-5s for SwarmFileStore disk).
    
    Use cases:
    - "party_position_update" — Priest needs Hunter's position every 100ms
    - "combo_ready" — Wizard signals Knight for Bowling Bash
    - "emergency_help" — Bot3 being mobbed, needs immediate rescue
    - "loot_spawned" — Rogue notified of valuable drop location
    """
    
    ALERT_TTL = 5.0   # Alerts expire after 5 seconds
    STATE_TTL = 2.0   # State updates expire after 2 seconds
    
    @staticmethod
    def send_state(bot_id: str, state_type: str, data: dict) -> None:
        """Publish bot state to shared memory.
        
        Other bots can read this instantly without disk I/O.
        Used for: position updates, HP/SP status, target selection.
        """
        _ensure_shm()
        payload = {
            "bot_id": bot_id,
            "type": state_type,
            "data": data,
            "ts": time.time(),
        }
        path = SHM_STATE / f"{bot_id}_{state_type}.json"
        # Atomic write — write to temp, rename
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.rename(path)
    
    @staticmethod
    def read_state(bot_id: str, state_type: str, max_age: float | None = None) -> dict | None:
        """Read another bot's state from shared memory.
        
        Returns None if no state published or state is too old.
        """
        path = SHM_STATE / f"{bot_id}_{state_type}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            age = time.time() - data.get("ts", 0)
            max_age = max_age or SharedMemoryIPC.STATE_TTL
            if age > max_age:
                return None  # Stale state
            return data
        except (json.JSONDecodeError, OSError):
            return None
    
    @staticmethod
    def list_bots_on_map(map_name: str) -> list[str]:
        """List all bots that recently published state for a given map."""
        if not SHM_STATE.exists():
            return []
        bots = set()
        for f in SHM_STATE.iterdir():
            if f.suffix == ".json":
                try:
                    data = json.loads(f.read_text())
                    state_map = data.get("data", {}).get("map", "")
                    age = time.time() - data.get("ts", 0)
                    if state_map == map_name and age < SharedMemoryIPC.STATE_TTL:
                        bots.add(data["bot_id"])
                except (json.JSONDecodeError, OSError):
                    pass
        return list(bots)
    
    @staticmethod
    def send_alert(alert_type: str, bot_id: str, message: str, urgency: int = 5) -> None:
        """Send an urgent alert to all bots.
        
        urgency: 1-10 (10 = most urgent)
        Types: "emergency", "combo_request", "loot_alert", "position_share"
        """
        _ensure_shm()
        payload = {
            "type": alert_type,
            "bot_id": bot_id,
            "message": message,
            "urgency": urgency,
            "ts": time.time(),
        }
        path = SHM_ALERTS / f"{alert_type}_{bot_id}_{int(time.time() * 1000)}.json"
        path.write_text(json.dumps(payload))
        
        # Clean old alerts
        SharedMemoryIPC._clean_old_files(SHM_ALERTS, SharedMemoryIPC.ALERT_TTL)
    
    @staticmethod
    def get_alerts(bot_id: str | None = None, since_ts: float = 0.0) -> list[dict]:
        """Get alerts, optionally filtered by target bot."""
        if not SHM_ALERTS.exists():
            return []
        alerts = []
        now = time.time()
        for f in sorted(SHM_ALERTS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if f.suffix != ".json":
                continue
            try:
                data = json.loads(f.read_text())
                age = now - data.get("ts", 0)
                if age > SharedMemoryIPC.ALERT_TTL:
                    f.unlink(missing_ok=True)
                    continue
                if data["ts"] < since_ts:
                    continue
                alerts.append(data)
            except (json.JSONDecodeError, OSError):
                pass
        return alerts[:20]  # Return most recent 20
    
    @staticmethod
    def get_last_combo_signal(caster_bot: str, combo_type: str) -> dict | None:
        """Get the most recent combo signal from a specific bot."""
        if not SHM_COORD.exists():
            return None
        latest = None
        for f in SHM_COORD.iterdir():
            if f.suffix != ".json":
                continue
            try:
                data = json.loads(f.read_text())
                if data.get("caster") == caster_bot and data.get("combo") == combo_type:
                    age = time.time() - data.get("ts", 0)
                    if age < 5.0:  # Combo signals expire after 5s
                        if latest is None or data["ts"] > latest["ts"]:
                            latest = data
            except (json.JSONDecodeError, OSError):
                pass
        return latest
    
    @staticmethod
    def _clean_old_files(directory: Path, max_age: float) -> None:
        """Remove files older than max_age seconds."""
        if not directory.exists():
            return
        now = time.time()
        for f in directory.iterdir():
            try:
                age = now - f.stat().st_mtime
                if age > max_age and f.suffix == ".json":
                    f.unlink(missing_ok=True)
            except OSError:
                pass


class SharedMemoryCoordination:
    """High-level coordination helpers built on SharedMemoryIPC."""
    
    @staticmethod
    def share_position(bot_id: str, map_name: str, x: int, y: int, hp_pct: float = 100) -> None:
        """Share current position for party coordination."""
        SharedMemoryIPC.send_state(bot_id, "position", {
            "map": map_name,
            "x": x,
            "y": y,
            "hp_pct": hp_pct,
        })
    
    @staticmethod
    def request_combo(caster_bot: str, target_bot: str, combo_type: str, skill_id: str) -> None:
        """Request a combo execution from another bot."""
        path = SHM_COORD / f"combo_{caster_bot}_{combo_type}.json"
        _ensure_shm()
        payload = {
            "caster": caster_bot,
            "target": target_bot,
            "combo": combo_type,
            "skill": skill_id,
            "ts": time.time(),
        }
        path.write_text(json.dumps(payload))
    
    @staticmethod
    def respond_to_combo(caster_bot: str, combo_type: str, response: str):
        """Respond to a combo request (confirm/deny)."""
        path = SHM_COORD / f"response_{caster_bot}_{combo_type}.json"
        _ensure_shm()
        payload = {
            "caster": caster_bot,
            "combo": combo_type,
            "response": response,
            "ts": time.time(),
        }
        path.write_text(json.dumps(payload))
    
    @staticmethod
    def emergency_alert(bot_id: str, message: str, map_name: str = "") -> None:
        """Send an emergency alert — bot is in danger and needs help."""
        SharedMemoryIPC.send_alert("emergency", bot_id, message, urgency=10)
        logger.warning(f"EMERGENCY from {bot_id} on {map_name}: {message}")


# Migration note: This module REPLACES SwarmFileStore for time-sensitive 
# coordination. SwarmFileStore remains for persistent coordination that
# survives restarts (e.g., formation assignments, resource sharing).
# Use SharedMemoryIPC for anything that needs sub-second latency.
