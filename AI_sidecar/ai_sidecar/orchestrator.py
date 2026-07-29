"""BotOrchestrator — top-level runtime with event bus, persistence, and scheduling.

Consolidates 50+ domain modules into a single runtime with:
- Event bus for cross-domain communication
- SQLite persistence for learning data
- Resource pooling across bots
- Out-of-combat behavior scheduling
- Error alerting and recovery
"""
from __future__ import annotations
import logging
from typing import Any

from ai_sidecar.autonomy.heuristic_service import HeuristicService
from ai_sidecar.actions import HeuristicAction
from ai_sidecar.runtime.event_bus import EventBus
from ai_sidecar.runtime.persistence import PersistentState
from ai_sidecar.runtime.pool import ResourcePool
from ai_sidecar.runtime.scheduler import OutOfCombatScheduler
from ai_sidecar.runtime.action_filter import filter_actions, get_filter_logger, BridgeActionLogger, BatchActionQueue
from ai_sidecar.runtime.latency import get_latency_tracker
from ai_sidecar.runtime.cruise import CruiseController

logger = logging.getLogger(__name__)


class BotRuntime:
    """Production runtime that wraps HeuristicService with operational layers.
    
    Features:
    - think() -> actions -> act() -> commands (standard flow)
    - EventBus for cross-domain communication
    - PersistentState for crash-resistant learning
    - ResourcePool for cross-bot economy
    - Scheduler for out-of-combat behavior
    - Error alerting (all errors logged to ERROR level)
    """

    def __init__(self):
        self._hs: HeuristicService | None = None
        self._initialized = False
        self._resource_pool = ResourcePool()
        self._out_of_combat = OutOfCombatScheduler()
        self._last_error: str | None = None
        self._error_count = 0
        self._cruise = CruiseController()
        self._latency = get_latency_tracker()
        self._batch_queue = BatchActionQueue()

    def initialize(self) -> None:
        if self._initialized:
            return
        try:
            self._hs = HeuristicService()
            self._initialized = True
            # Initialize persistence
            stats = PersistentState.get_stats()
            logger.info(f"BotRuntime initialized. Persistent state: {stats}")
        except Exception as e:
            logger.error(f"BotRuntime initialization failed: {e}")
            self._last_error = str(e)
            self._error_count += 1
            raise

    def think(self, signals: dict[str, Any]) -> list[HeuristicAction]:
        """Process signals and return actions.
        
        This method orchestrates:
        1. HeuristicService.assess() — the main AI decision engine
        2. ResourcePool — check cross-bot resource needs
        3. OutOfCombatScheduler — idle behavior
        4. EventBus — cross-domain communication
        5. PersistentState — save learning data
        6. Error alerting — log all failures at ERROR level
        """
        if not self._initialized:
            self.initialize()

        actions: list[HeuristicAction] = []
        if not self._hs:
            return actions

        bot_id = signals.get("bot_id", str(signals.get("id", "unknown")))

        # Track snapshot sent for latency measurement
        self._latency.record_snapshot_sent()

        # Check cruise control (steady state decision caching)
        if self._cruise.is_steady_state(signals):
            cached = self._cruise.get_cached()
            if cached:
                logger.debug(f"[{bot_id}] Steady state: reusing {len(cached)} cached actions")
                return cached

        # 1. Main AI decision engine
        try:
            assessment = self._hs.assess(signals)
            if assessment and assessment.actions:
                # Record latency from snapshot to action
                self._latency.record_action_received()

                # Apply action filter: reduce 72+ actions to top real commands
                filtered = filter_actions(assessment.actions, max_commands=5)
                actions.extend(filtered)
                # Cache for cruise control
                self._cruise.cache_decisions(actions)
                # Add to batch queue for multi-action polling
                self._batch_queue.add_actions(filtered)
                # Log bridge actions for verification
                _logger = get_filter_logger()
                for a in filtered:
                    if a.kind == "command" and a.command and not a.command.startswith("goal="):
                        _logger.log_action(a, source="heuristic")
            else:
                logger.warning(f"[{bot_id}] HeuristicService returned no actions")
        except Exception as e:
            logger.error(f"[{bot_id}] HeuristicService.assess() failed: {e}")
            self._last_error = str(e)
            self._error_count += 1
            # Fallback: basic survival actions
            actions.append(HeuristicAction(
                kind="command", command="attackAuto 2",
                confidence=0.5, reason=f"Fallback: AI failed ({e})", domain="safety"
            ))

        # 2. Post events to the blackboard
        map_name = str(signals.get("map", "") or "")
        hp = int(signals.get("hp", 100) or 100)
        hp_max = int(signals.get("hp_max", 100) or 100)
        hp_pct = hp / max(hp_max, 1) * 100

        if hp_pct < 30:
            EventBus.post(f"combat:critical_hp:{bot_id}", {"hp_pct": hp_pct, "map": map_name})

        # 3. Resource pooling
        try:
            zeny = int(signals.get("zeny", 0) or 0)
            pool_actions: list[HeuristicAction] = []
            self._resource_pool.assess(signals, pool_actions, bot_id)
            actions.extend(pool_actions)
        except Exception as e:
            logger.error(f"[{bot_id}] ResourcePool failed: {e}")

        # 4. Out-of-combat behavior
        try:
            ooc_actions: list[HeuristicAction] = []
            self._out_of_combat.assess(signals, ooc_actions, bot_id)
            actions.extend(ooc_actions)
        except Exception as e:
            logger.error(f"[{bot_id}] OutOfCombatScheduler failed: {e}")

        # 5. Save bot state to persistence
        try:
            PersistentState.save_bot_state(bot_id, "last_signals", {
                "map": map_name,
                "hp_pct": hp_pct,
                "zeny": zeny,
                "level": signals.get("base_level", 0),
                "job": signals.get("job", ""),
                "timestamp": __import__("datetime").datetime.now().isoformat(),
            })
        except Exception as e:
            logger.debug(f"[{bot_id}] Persistence save failed: {e}")

        return actions

    def act(self, actions: list[HeuristicAction]) -> list[str]:
        """Convert actions to bridge commands."""
        commands = []
        for action in actions:
            if action.kind == "command" and action.command:
                commands.append(action.command)
        return commands

    def get_status(self) -> dict[str, Any]:
        return {
            "initialized": self._initialized,
            "healthy": self._hs is not None,
            "error_count": self._error_count,
            "last_error": self._last_error,
            "persistence": PersistentState.get_stats(),
            "bridge_actions": get_filter_logger().get_stats(),
            "cruise": self._cruise.get_stats() if hasattr(self, '_cruise') else {},
            "latency": self._latency.get_stats() if hasattr(self, '_latency') else {},
            "action_batch": {"queue_size": self._batch_queue.queue_size()} if hasattr(self, '_batch_queue') else {},
        }

    def get_event_summary(self) -> dict:
        return EventBus.summarize()


# Global singleton
_runtime: BotRuntime | None = None

def get_runtime() -> BotRuntime:
    global _runtime
    if _runtime is None:
        _runtime = BotRuntime()
    return _runtime
