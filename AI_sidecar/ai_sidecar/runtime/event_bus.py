"""Event Bus — cross-domain communication blackboard.

Any domain module can post events to the blackboard.
Any domain module can read current state from the blackboard.
This replaces the silo architecture with shared awareness.

Events types:
- "combat:critical_hp" → Economy: adjust map recommendation
- "inventory:high_value_drop" → Loot: designate looter
- "navigation:arrived_map" → WorldState: check map danger
- "economy:market_shift" → Combat: adjust gear
- "learning:death" → Planning: adjust build
"""
from __future__ import annotations
import logging
import threading
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


class EventBus:
    """Thread-safe shared blackboard for cross-domain communication.
    
    Usage:
        EventBus.post("combat:critical_hp", {"hp_pct": 15, "map": "prt_fild05"})
        danger = EventBus.get("safety:danger_level")
    """
    
    _lock = threading.Lock()
    _board: dict[str, Any] = {}
    _history: list[dict] = []
    _MAX_HISTORY = 100
    
    @classmethod
    def post(cls, key: str, value: Any) -> None:
        """Post a value to the blackboard.
        
        Key convention: "<domain>:<event_name>"
        Examples:
          "combat:critical_hp" — bot at critical HP
          "economy:market_shift" — market prices changed
          "safety:danger_level" — danger level changed
          "navigation:arrived_map" — map changed
          "learning:death_recorded" — death recorded
        """
        with cls._lock:
            cls._board[key] = value
            cls._history.append({
                "key": key,
                "value": value,
                "ts": datetime.now().isoformat(),
            })
            if len(cls._history) > cls._MAX_HISTORY:
                cls._history = cls._history[-cls._MAX_HISTORY:]
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Read a value from the blackboard."""
        with cls._lock:
            return cls._board.get(key, default)
    
    @classmethod
    def get_many(cls, prefix: str) -> dict[str, Any]:
        """Get all keys matching a prefix.
        
        Example: get_many("combat:") returns all combat events.
        """
        with cls._lock:
            return {k: v for k, v in cls._board.items() if k.startswith(prefix)}
    
    @classmethod
    def clear(cls, key: str | None = None) -> None:
        """Clear specific key or entire board."""
        with cls._lock:
            if key:
                cls._board.pop(key, None)
            else:
                cls._board.clear()
    
    @classmethod
    def get_recent(cls, limit: int = 10) -> list[dict]:
        """Get most recent events."""
        with cls._lock:
            return cls._history[-limit:]
    
    @classmethod
    def summarize(cls) -> dict[str, Any]:
        """Get a summary of current blackboard state."""
        with cls._lock:
            return {
                "active_keys": len(cls._board),
                "keys": list(cls._board.keys()),
                "history_count": len(cls._history),
                "recent": cls._history[-5:] if cls._history else [],
            }


# Convenience wrappers for common event types
def post_danger_event(danger_level: float, map_name: str, reason: str) -> None:
    EventBus.post("safety:danger_level", {"level": danger_level, "map": map_name})
    if danger_level > 0.8:
        EventBus.post("safety:evacuate", {"map": map_name, "reason": reason, "urgency": "high"})

def post_death_event(map_name: str, monster_name: str) -> None:
    EventBus.post("learning:death", {"map": map_name, "monster": monster_name})
    EventBus.post("combat:critical_hp", {"hp_pct": 0, "map": map_name})

def post_market_event(item_name: str, price: int, trend: str) -> None:
    EventBus.post(f"economy:market_{item_name}", {"price": price, "trend": trend})

def post_loot_event(item_name: str, value: int, map_name: str) -> None:
    EventBus.post(f"loot:high_value", {"item": item_name, "value": value, "map": map_name})

def post_arrival_event(map_name: str, bot_id: str) -> None:
    EventBus.post("navigation:arrived_map", {"map": map_name, "bot_id": bot_id})
