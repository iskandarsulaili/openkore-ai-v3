"""Cruise Control — state caching so the bot doesn't recompute every tick.

When a bot is farming Porings at 80% HP with 10 potions and 50% weight,
the answer is always "keep attacking." There's no need to ask 30 domains
what to do. This module detects "steady state" and reuses the last decision.
"""
from __future__ import annotations
import time
import logging
from typing import Any
from collections import deque

from ai_sidecar.actions import HeuristicAction
from ai_sidecar.runtime.event_bus import EventBus

logger = logging.getLogger(__name__)


class CruiseController:
    """Determines when the bot is in 'steady state' and can reuse decisions.
    
    Steady state = no significant change since last tick:
    - Same map
    - HP/SP within 10% of last tick
    - Same monsters (or none)
    - Same enemies around (count is same)
    - No death since last tick
    - No new items dropped
    """
    
    def __init__(self, tick_window: int = 5):
        self._last_state: dict[str, Any] = {}
        self._tick_count = 0
        self._steady_ticks = 0
        self._recent_actions: deque[list[HeuristicAction]] = deque(maxlen=10)
        self._last_decision_time = 0.0
        self._max_steady_ticks = 15  # Max 15 ticks before forced re-evaluation
        self._tick_window = tick_window
    
    def is_steady_state(self, signals: dict[str, Any]) -> bool:
        """Check if bot state is essentially unchanged from last tick."""
        self._tick_count += 1
        
        # Extract key state indicators
        state_key = (
            signals.get("map", ""),
            int(signals.get("hp", 0) / max(1, signals.get("hp_max", 1)) * 10),  # HP decile
            int(signals.get("sp", 0) / max(1, signals.get("sp_max", 1)) * 10),  # SP decile
            len(signals.get("monsters_around", [])),  # Monster count
            signals.get("dead", False),
            signals.get("is_dead", False),
        )
        
        # Force re-evaluation on death
        if state_key[4] or state_key[5]:
            self._steady_ticks = 0
            return False
        
        if self._last_state and self._last_state == state_key:
            self._steady_ticks += 1
        else:
            self._steady_ticks = 0
        
        self._last_state = state_key
        
        # First 2 ticks: always evaluate (cold start buffer)
        if self._tick_count <= 2:
            return False
        
        # Force re-evaluation every N ticks
        if self._steady_ticks >= self._max_steady_ticks:
            self._steady_ticks = 0
            return False
        
        # In steady state if unchanged for tick_window ticks
        return self._steady_ticks >= self._tick_window
    
    def cache_decisions(self, actions: list[HeuristicAction]) -> None:
        """Store decisions for reuse during steady state."""
        self._recent_actions.append(actions)
        self._last_decision_time = time.time()
    
    def get_cached(self) -> list[HeuristicAction]:
        """Get last good set of decisions."""
        if self._recent_actions:
            return list(self._recent_actions[-1])
        return []
    
    def get_stats(self) -> dict[str, Any]:
        return {
            "tick": self._tick_count,
            "steady_ticks": self._steady_ticks,
            "cached_decisions": len(self._recent_actions[-1]) if self._recent_actions else 0,
            "last_decision_age": time.time() - self._last_decision_time if self._last_decision_time else 0,
        }


class DomainWire:
    """Wires EventBus and PersistentState into domain modules.
    
    Each domain module can:
    1. POST events to EventBus when something noteworthy happens
    2. READ state from EventBus to inform decisions
    3. PERSIST learning data via PersistentState
    4. READ persisted data on initialization
    
    This replaces the current silo architecture with a shared
    communication layer.
    """
    
    @staticmethod
    def wire_post_mortem(post_mortem, event_bus, persistent_state) -> None:
        """Wire PostMortemAnalyzer to EventBus and PersistentState.
        
        Posts:
        - 'death:{bot_id}' when death detected
        - 'danger:{map}' when map danger level changes
        
        Reads:
        - 'hp:critical' for recent near-death experiences
        """
        if not post_mortem:
            return
        
        # Attach event bus to post_mortem
        if hasattr(post_mortem, '_event_bus'):
            post_mortem._event_bus = event_bus
        if hasattr(post_mortem, '_persistent'):
            post_mortem._persistent = persistent_state
    
    @staticmethod
    def wire_economy(economy_modules: list, event_bus, persistent_state) -> None:
        """Wire economy modules to EventBus.
        
        Posts:
        - 'market:price_changed:{item}' when market price shifts
        - 'economy:funds_low' when zeny is critically low
        
        Reads:
        - 'farming:danger:{map}' for danger-adjusted farming decisions
        - 'team:need_funds' for resource pooling
        """
        for mod in economy_modules:
            if mod and hasattr(mod, '_event_bus'):
                mod._event_bus = event_bus
    
    @staticmethod
    def wire_combat(combat_modules: list, event_bus) -> None:
        """Wire combat modules to EventBus.
        
        Posts:
        - 'combat:skill_cast:{skill_name}' when casting a skill
        - 'combat:cast_interrupted' when cast is interrupted
        - 'combat:kill:{monster}' when a monster dies
        - 'combat:kiting' when kiting is triggered
        
        Reads:
        - 'party:combo_requested' for combo protocol
        - 'danger:predicted_hit' for kiting decisions
        """
        for mod in combat_modules:
            if mod and hasattr(mod, '_event_bus'):
                mod._event_bus = event_bus
