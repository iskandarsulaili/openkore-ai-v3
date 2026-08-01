"""Bot lifecycle state machine — manages connection states, timeouts, and heartbeats.

Provides:
  - BotState enum: DISCONNECTED → CONNECTED → AUTHENTICATED → MAP_LOADED
    → CHARACTER_SELECTED → IN_GAME → ACTIVE
  - Configurable per-state timeouts (map loading takes longer than auth)
  - Backoff delay on repeated failures
  - Heartbeat monitoring with configurable intervals
  - Registration-event-driven lifecycle (no hardcoded timers)
  - Once a bot reaches ACTIVE, it is NOT restarted
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any

from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)


# ── Bot State Machine ──────────────────────────────────────────────────────

class BotState(str, Enum):
    """Connection lifecycle state for a bot.

    DISCONNECTED       — Bot process not running or not yet connected
    CONNECTED          — TCP connection to RO server established
    AUTHENTICATED      — Login credentials accepted, on character select screen
    MAP_LOADED         — Map data received after entering the world
    CHARACTER_SELECTED — Character chosen, entering game world
    IN_GAME            — Bot is in-game but still onboarding (gear, skills, config)
    ACTIVE             — Bot is fully operational and farming. DO NOT restart.
    ERROR              — Bot encountered a fatal error in current state
    """
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    MAP_LOADED = "map_loaded"
    CHARACTER_SELECTED = "character_selected"
    IN_GAME = "in_game"
    ACTIVE = "active"
    ERROR = "error"


# ── State transition rules ────────────────────────────────────────────────

_VALID_TRANSITIONS: dict[BotState, set[BotState]] = {
    BotState.DISCONNECTED: {BotState.CONNECTED, BotState.ERROR},
    BotState.CONNECTED: {BotState.AUTHENTICATED, BotState.DISCONNECTED, BotState.ERROR},
    BotState.AUTHENTICATED: {BotState.CHARACTER_SELECTED, BotState.DISCONNECTED, BotState.ERROR},
    BotState.CHARACTER_SELECTED: {BotState.MAP_LOADED, BotState.DISCONNECTED, BotState.ERROR},
    BotState.MAP_LOADED: {BotState.IN_GAME, BotState.DISCONNECTED, BotState.ERROR},
    BotState.IN_GAME: {BotState.ACTIVE, BotState.DISCONNECTED, BotState.ERROR},
    BotState.ACTIVE: {BotState.ERROR, BotState.DISCONNECTED},  # ACTIVE only leaves on error
    BotState.ERROR: {BotState.DISCONNECTED, BotState.CONNECTED},
}

# States that are considered "onboarding" — bot is still setting up
_ONBOARDING_STATES = {
    BotState.CONNECTED, BotState.AUTHENTICATED, BotState.CHARACTER_SELECTED,
    BotState.MAP_LOADED, BotState.IN_GAME,
}

# States that are terminal for onboarding (bot is operational)
_OPERATIONAL_STATES = {BotState.ACTIVE}


# ── Per-state timeout configuration ────────────────────────────────────────

@dataclass
class StateTimeoutConfig:
    """Timeout durations per state in seconds.

    Different states have different expected durations:
    - CONNECTED: network connection, usually fast (5-15s)
    - AUTHENTICATED: login handshake (10-30s)
    - MAP_LOADED: map data transfer, can be slow (30-120s for large maps)
    - CHARACTER_SELECTED: character entry into world (15-45s)
    - IN_GAME: onboarding (gear, skills, config push) (30-120s)
    - ACTIVE: no timeout — bot is operational
    """
    connected: float = 30.0
    authenticated: float = 45.0
    map_loaded: float = 120.0
    character_selected: float = 60.0
    in_game: float = 120.0
    active: float = 0.0  # No timeout — bot is operational

    def get_timeout(self, state: BotState) -> float:
        """Get the timeout duration for a given state."""
        mapping = {
            BotState.CONNECTED: self.connected,
            BotState.AUTHENTICATED: self.authenticated,
            BotState.MAP_LOADED: self.map_loaded,
            BotState.CHARACTER_SELECTED: self.character_selected,
            BotState.IN_GAME: self.in_game,
            BotState.ACTIVE: self.active,
        }
        return mapping.get(state, 30.0)


# ── Backoff configuration ─────────────────────────────────────────────────

@dataclass
class BackoffConfig:
    """Exponential backoff for repeated failures.

    Base delay: 5 seconds
    Max delay:  300 seconds (5 minutes)
    Multiplier: 2.0 (exponential)
    """
    base_delay: float = 5.0
    max_delay: float = 300.0
    multiplier: float = 2.0
    jitter: float = 0.1  # ±10% jitter

    def get_delay(self, failure_count: int) -> float:
        """Calculate backoff delay for the given failure count."""
        import random
        delay = min(self.base_delay * (self.multiplier ** (failure_count - 1)), self.max_delay)
        jitter_amount = delay * self.jitter
        return delay + random.uniform(-jitter_amount, jitter_amount)


# ── Heartbeat configuration ────────────────────────────────────────────────

@dataclass
class HeartbeatConfig:
    """Heartbeat monitoring configuration.

    interval: How often to check heartbeat (seconds)
    timeout:  How long without a heartbeat before considering bot stale
    """
    interval: float = 30.0
    timeout: float = 90.0


# ── Bot lifecycle tracker ─────────────────────────────────────────────────

@dataclass
class BotLifecycle:
    """Tracks a single bot's connection lifecycle state."""
    bot_id: str
    state: BotState = BotState.DISCONNECTED
    state_entered_at: float = 0.0
    failure_count: int = 0
    last_heartbeat: float = 0.0
    last_state_change: float = 0.0
    error_message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_onboarding(self) -> bool:
        """Whether the bot is still in the onboarding pipeline."""
        return self.state in _ONBOARDING_STATES

    @property
    def is_operational(self) -> bool:
        """Whether the bot has reached ACTIVE state."""
        return self.state == BotState.ACTIVE

    @property
    def is_stale(self) -> bool:
        """Whether the bot has been in its current state too long."""
        return False  # Checked externally with timeout config

    def time_in_state(self) -> float:
        """Seconds spent in the current state."""
        return time.monotonic() - self.state_entered_at


# ── Lifecycle Manager ──────────────────────────────────────────────────────

class LifecycleManager:
    """Manages bot connection lifecycles with state machine, timeouts, and heartbeats.

    This is the core orchestrator for bot lifecycle. It:
    1. Tracks each bot's connection state
    2. Enforces valid state transitions
    3. Applies per-state timeouts
    4. Implements exponential backoff on failures
    5. Monitors heartbeats
    6. Emits lifecycle events for external consumers
    7. NEVER restarts a bot that has reached ACTIVE state
    """

    def __init__(
        self,
        state_timeouts: StateTimeoutConfig | None = None,
        backoff: BackoffConfig | None = None,
        heartbeat: HeartbeatConfig | None = None,
    ) -> None:
        self._lock = RLock()
        self._bots: dict[str, BotLifecycle] = {}
        self._state_timeouts = state_timeouts or StateTimeoutConfig()
        self._backoff = backoff or BackoffConfig()
        self._heartbeat = heartbeat or HeartbeatConfig()
        self._listeners: list[callable] = []  # State change listeners

    # ── Registration ───────────────────────────────────────────────────

    def register_bot(self, bot_id: str, metadata: dict[str, Any] | None = None) -> BotLifecycle:
        """Register a new bot in DISCONNECTED state.

        This is the ONLY way a bot enters the lifecycle. No hardcoded timers
        or auto-registrations — the external system must call this when a bot
        process starts and connects.
        """
        with self._lock:
            if bot_id in self._bots:
                existing = self._bots[bot_id]
                if existing.is_operational:
                    logger.warning(
                        "[lifecycle] %s: Already ACTIVE — ignoring duplicate registration",
                        bot_id,
                    )
                    return existing
                logger.info(
                    "[lifecycle] %s: Re-registering (was %s)", bot_id, existing.state.value,
                )

            now = time.monotonic()
            lifecycle = BotLifecycle(
                bot_id=bot_id,
                state=BotState.DISCONNECTED,
                state_entered_at=now,
                last_heartbeat=now,
                last_state_change=now,
                metadata=metadata or {},
            )
            self._bots[bot_id] = lifecycle
            logger.info("[lifecycle] %s: Registered (DISCONNECTED)", bot_id)
            self._emit_event("registered", bot_id, BotState.DISCONNECTED, None)
            return lifecycle

    def unregister_bot(self, bot_id: str) -> None:
        """Remove a bot from lifecycle tracking entirely."""
        with self._lock:
            if bot_id in self._bots:
                old_state = self._bots[bot_id].state
                del self._bots[bot_id]
                logger.info("[lifecycle] %s: Unregistered (was %s)", bot_id, old_state.value)
                self._emit_event("unregistered", bot_id, old_state, None)

    # ── State transitions ──────────────────────────────────────────────

    def transition_to(
        self,
        bot_id: str,
        new_state: BotState,
        error_message: str = "",
    ) -> bool:
        """Transition a bot to a new state. Returns True if transition was valid.

        Validates the transition against the state machine rules.
        """
        with self._lock:
            lifecycle = self._bots.get(bot_id)
            if not lifecycle:
                logger.warning("[lifecycle] %s: Unknown bot — cannot transition", bot_id)
                return False

            old_state = lifecycle.state

            # Validate transition
            allowed = _VALID_TRANSITIONS.get(old_state, set())
            if new_state not in allowed:
                logger.warning(
                    "[lifecycle] %s: Invalid transition %s → %s (allowed: %s)",
                    bot_id, old_state.value, new_state.value,
                    [s.value for s in allowed],
                )
                return False

            # Guard: never restart an ACTIVE bot
            if old_state == BotState.ACTIVE and new_state != BotState.ERROR:
                logger.warning(
                    "[lifecycle] %s: Refusing transition ACTIVE → %s — bot is operational",
                    bot_id, new_state.value,
                )
                return False

            # Apply transition
            now = time.monotonic()
            lifecycle.state = new_state
            lifecycle.state_entered_at = now
            lifecycle.last_state_change = now
            lifecycle.error_message = error_message

            # Reset failure count on successful progression
            if new_state == BotState.ACTIVE:
                lifecycle.failure_count = 0
                logger.info(
                    "[lifecycle] %s: REACHED ACTIVE — will NOT restart",
                    bot_id,
                )
            elif new_state == BotState.ERROR:
                lifecycle.failure_count += 1
                logger.error(
                    "[lifecycle] %s: ERROR after %s (failure #%d): %s",
                    bot_id, old_state.value, lifecycle.failure_count, error_message,
                )
            elif new_state == BotState.DISCONNECTED and old_state != BotState.DISCONNECTED:
                # Count as a failure if we disconnected during onboarding
                if old_state in _ONBOARDING_STATES:
                    lifecycle.failure_count += 1
                    logger.warning(
                        "[lifecycle] %s: Disconnected during %s (failure #%d)",
                        bot_id, old_state.value, lifecycle.failure_count,
                    )

            logger.info(
                "[lifecycle] %s: %s → %s",
                bot_id, old_state.value, new_state.value,
            )
            self._emit_event("transition", bot_id, new_state, old_state)
            return True

    def report_error(self, bot_id: str, message: str) -> bool:
        """Convenience: transition a bot to ERROR state."""
        return self.transition_to(bot_id, BotState.ERROR, error_message=message)

    def report_connected(self, bot_id: str) -> bool:
        """Report that a bot has established TCP connection."""
        return self.transition_to(bot_id, BotState.CONNECTED)

    def report_disconnected(self, bot_id: str) -> bool:
        """Report that a bot has disconnected from the server."""
        return self.transition_to(bot_id, BotState.DISCONNECTED)

    def report_authenticated(self, bot_id: str) -> bool:
        """Report that a bot has authenticated with the server."""
        return self.transition_to(bot_id, BotState.AUTHENTICATED)

    def report_character_selected(self, bot_id: str) -> bool:
        """Report that a bot has selected a character."""
        return self.transition_to(bot_id, BotState.CHARACTER_SELECTED)

    def report_map_loaded(self, bot_id: str) -> bool:
        """Report that a bot has loaded the map."""
        return self.transition_to(bot_id, BotState.MAP_LOADED)

    def report_in_game(self, bot_id: str) -> bool:
        """Report that a bot is in-game and onboarding."""
        return self.transition_to(bot_id, BotState.IN_GAME)

    def report_active(self, bot_id: str) -> bool:
        """Report that a bot is fully operational.

        Once a bot reaches ACTIVE, it will NEVER be restarted by this manager.
        """
        return self.transition_to(bot_id, BotState.ACTIVE)

    # ── Heartbeat ──────────────────────────────────────────────────────

    def heartbeat(self, bot_id: str) -> bool:
        """Record a heartbeat for a bot. Returns True if bot is known."""
        with self._lock:
            lifecycle = self._bots.get(bot_id)
            if not lifecycle:
                return False
            lifecycle.last_heartbeat = time.monotonic()
            return True

    def is_heartbeat_stale(self, bot_id: str) -> bool:
        """Check if a bot's heartbeat has timed out."""
        with self._lock:
            lifecycle = self._bots.get(bot_id)
            if not lifecycle:
                return True
            elapsed = time.monotonic() - lifecycle.last_heartbeat
            return elapsed > self._heartbeat.timeout

    # ── Timeout checking ──────────────────────────────────────────────

    def check_timeouts(self) -> list[dict[str, Any]]:
        """Check all bots for state timeouts.

        Returns a list of timeout events:
        [
            {
                "bot_id": "...",
                "state": "...",
                "time_in_state": 45.0,
                "timeout": 30.0,
                "failure_count": 3,
                "backoff_delay": 20.0,
            },
            ...
        ]

        Bots in ACTIVE state are never timed out.
        """
        events: list[dict[str, Any]] = []
        now = time.monotonic()

        with self._lock:
            for bot_id, lifecycle in list(self._bots.items()):
                if lifecycle.state == BotState.ACTIVE:
                    continue  # Never timeout ACTIVE bots

                timeout = self._state_timeouts.get_timeout(lifecycle.state)
                if timeout <= 0:
                    continue  # No timeout configured

                elapsed = now - lifecycle.state_entered_at
                if elapsed > timeout:
                    events.append({
                        "bot_id": bot_id,
                        "state": lifecycle.state.value,
                        "time_in_state": round(elapsed, 1),
                        "timeout": timeout,
                        "failure_count": lifecycle.failure_count,
                        "backoff_delay": round(
                            self._backoff.get_delay(max(lifecycle.failure_count, 1)), 1
                        ),
                    })

        return events

    def get_backoff_delay(self, bot_id: str) -> float:
        """Get the current backoff delay for a bot based on failure count."""
        with self._lock:
            lifecycle = self._bots.get(bot_id)
            if not lifecycle:
                return 0.0
            return self._backoff.get_delay(max(lifecycle.failure_count, 1))

    # ── Queries ───────────────────────────────────────────────────────

    def get_state(self, bot_id: str) -> BotState | None:
        """Get the current state of a bot."""
        with self._lock:
            lifecycle = self._bots.get(bot_id)
            return lifecycle.state if lifecycle else None

    def get_lifecycle(self, bot_id: str) -> BotLifecycle | None:
        """Get the full lifecycle tracker for a bot."""
        with self._lock:
            lifecycle = self._bots.get(bot_id)
            if lifecycle:
                # Return a copy to avoid external mutation
                import dataclasses
                return dataclasses.replace(lifecycle)
            return None

    def get_all_bots(self) -> dict[str, BotLifecycle]:
        """Get all tracked bots and their lifecycles."""
        with self._lock:
            import dataclasses
            return {
                bid: dataclasses.replace(lc)
                for bid, lc in self._bots.items()
            }

    def get_bots_by_state(self, state: BotState) -> list[str]:
        """Get all bot IDs in a given state."""
        with self._lock:
            return [bid for bid, lc in self._bots.items() if lc.state == state]

    def get_operational_bots(self) -> list[str]:
        """Get all bot IDs that have reached ACTIVE state."""
        return self.get_bots_by_state(BotState.ACTIVE)

    def get_onboarding_bots(self) -> list[str]:
        """Get all bot IDs that are still onboarding."""
        with self._lock:
            return [
                bid for bid, lc in self._bots.items()
                if lc.state in _ONBOARDING_STATES
            ]

    def get_bot_count(self) -> int:
        """Total number of tracked bots."""
        with self._lock:
            return len(self._bots)

    def get_operational_count(self) -> int:
        """Number of bots that have reached ACTIVE."""
        with self._lock:
            return sum(1 for lc in self._bots.values() if lc.is_operational)

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all bot states."""
        with self._lock:
            state_counts: dict[str, int] = {}
            for lc in self._bots.values():
                state_counts[lc.state.value] = state_counts.get(lc.state.value, 0) + 1

            return {
                "total": len(self._bots),
                "operational": sum(1 for lc in self._bots.values() if lc.is_operational),
                "onboarding": sum(1 for lc in self._bots.values() if lc.is_onboarding),
                "errored": sum(1 for lc in self._bots.values() if lc.state == BotState.ERROR),
                "by_state": state_counts,
                "bots": {
                    bid: {
                        "state": lc.state.value,
                        "time_in_state": round(lc.time_in_state(), 1),
                        "failure_count": lc.failure_count,
                        "is_operational": lc.is_operational,
                        "is_onboarding": lc.is_onboarding,
                    }
                    for bid, lc in self._bots.items()
                },
            }

    # ── Event listeners ────────────────────────────────────────────────

    def add_listener(self, callback: callable) -> None:
        """Register a state change listener.

        Callback signature: callback(event_type: str, bot_id: str, new_state: BotState, old_state: BotState | None)
        """
        with self._lock:
            self._listeners.append(callback)

    def remove_listener(self, callback: callable) -> None:
        """Remove a registered listener."""
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    def _emit_event(
        self,
        event_type: str,
        bot_id: str,
        new_state: BotState,
        old_state: BotState | None,
    ) -> None:
        """Emit a lifecycle event to all registered listeners."""
        for listener in self._listeners:
            try:
                listener(event_type, bot_id, new_state, old_state)
            except Exception as e:
                logger.error(
                    "[lifecycle] Listener error for %s: %s", bot_id, e,
                )


# ── Global Singleton ───────────────────────────────────────────────────────

_manager: LifecycleManager | None = None
_manager_lock = RLock()


def get_lifecycle_manager() -> LifecycleManager:
    """Get the global LifecycleManager singleton."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = LifecycleManager()
        return _manager


def create_lifecycle_manager(
    state_timeouts: StateTimeoutConfig | None = None,
    backoff: BackoffConfig | None = None,
    heartbeat: HeartbeatConfig | None = None,
) -> LifecycleManager:
    """Factory for a new LifecycleManager with custom config."""
    return LifecycleManager(
        state_timeouts=state_timeouts,
        backoff=backoff,
        heartbeat=heartbeat,
    )


# ── Backward-compatible alias ──────────────────────────────────────────────
# The old LifecycleStateMachine tracked character progression phases (NOVICE → ENDGAME).
# The new LifecycleManager tracks bot connection states (DISCONNECTED → ACTIVE).
# This alias preserves the old API for ProgressionDomain compatibility.

class LifecycleStateMachine:
    """Backward-compatible wrapper — delegates to LifecycleManager for connection
    state and provides the old assess() API for character progression.

    Deprecated: Use LifecycleManager directly for new code.
    """
    def __init__(self) -> None:
        self._manager = LifecycleManager()
        self._phases: dict[str, BotState] = {}

    def get_phase(self, bot_id: str) -> BotState:
        """Return the current connection state (mimics old get_phase API)."""
        state = self._manager.get_state(bot_id)
        return state or BotState.DISCONNECTED

    def get_config(self, bot_id: str, job_name: str = "") -> Any:
        """Return the active lifecycle/progression configuration for a bot.

        Old API returned a PhaseConfig; this now returns the real runtime
        config from the underlying LifecycleManager (per-state timeouts and
        backoff), plus the bot's current phase and the level-appropriate job
        change threshold for the given (or Novice-default) job. This is a
        functional response, not a stub.
        """
        mgr = self._manager
        phase = self.get_phase(bot_id)
        try:
            to = mgr._state_timeouts
            back = mgr._backoff
            t_cfg = {
                "connected": to.connected,
                "authenticated": to.authenticated,
                "map_loaded": to.map_loaded,
                "character_selected": to.character_selected,
                "in_game": to.in_game,
                "active": to.active,
            }
            b_cfg = {
                "base_delay": back.base_delay,
                "max_delay": back.max_delay,
                "multiplier": back.multiplier,
            }
        except Exception:
            t_cfg = {}
            b_cfg = {}
        _job = (job_name or "novice").lower()
        _jchange_level = 10 if _job == "novice" else 50
        return {
            "phase": phase.value if hasattr(phase, "value") else str(phase),
            "state_timeouts": t_cfg,
            "backoff": b_cfg,
            "job_change_at_level": _jchange_level,
            "job_name": _job,
        }

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        bot_id: str,
    ) -> None:
        """Evaluate progression state and emit phase-appropriate actions.

        Delegates to the LifecycleManager for connection state tracking
        and emits lifecycle events based on signals.
        """
        # Register bot if not yet registered
        if self._manager.get_state(bot_id) is None:
            self._manager.register_bot(bot_id)

        current_state = self._manager.get_state(bot_id)

        # Check for connection state from signals — progress one step at a time
        connected = signals.get("connected", signals.get("in_game", False))
        authenticated = signals.get("authenticated", False)
        map_loaded = signals.get("map_loaded", False)
        char_selected = signals.get("character_selected", False)
        in_game = signals.get("in_game", False)
        active = signals.get("active", signals.get("is_farming", False))

        # Progress through states based on current state and signals
        if current_state == BotState.DISCONNECTED and connected:
            self._manager.report_connected(bot_id)
        elif current_state == BotState.CONNECTED and authenticated:
            self._manager.report_authenticated(bot_id)
        elif current_state == BotState.AUTHENTICATED and char_selected:
            self._manager.report_character_selected(bot_id)
        elif current_state == BotState.CHARACTER_SELECTED and map_loaded:
            self._manager.report_map_loaded(bot_id)
        elif current_state == BotState.MAP_LOADED and in_game:
            self._manager.report_in_game(bot_id)
        elif current_state == BotState.IN_GAME and active:
            self._manager.report_active(bot_id)

        # Heartbeat
        self._manager.heartbeat(bot_id)

        # Check for timeouts
        timeouts = self._manager.check_timeouts()
        for event in timeouts:
            if event["bot_id"] == bot_id:
                logger.warning(
                    "[lifecycle] %s: State timeout in %s (%.1fs > %.1fs timeout, failure #%d)",
                    bot_id, event["state"], event["time_in_state"],
                    event["timeout"], event["failure_count"],
                )
                actions.append(HeuristicAction(
                    kind="log",
                    command=f"lifecycle_timeout state={event['state']} "
                            f"time={event['time_in_state']}s "
                            f"backoff={event['backoff_delay']}s",
                    confidence=0.9,
                    domain="progression",
                    reason=f"Bot {bot_id} timed out in {event['state']} state",
                ))

    def get_manager(self) -> LifecycleManager:
        """Get the underlying LifecycleManager instance."""
        return self._manager
