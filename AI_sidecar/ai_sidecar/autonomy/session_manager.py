"""Session Manager — anti-24/7 session scheduling to avoid bot detection.

Pro-botting insight: the #1 detection signal is 24/7 uptime.
This module enforces human-like session patterns:
  - Max session duration (default 3h)
  - Min/max offline breaks (2-6h)
  - Daily play cap (8h)
  - Map rotation (avoid >2h on same map)
  - Peak hour avoidance (server prime time 20:00-23:00)
  - Gaussian jitter on login delays

All state persisted to disk via JSON for restart survival.
Thread-safe.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SESSION_STATE_FILE = "session_state.json"

# ── Dataclasses ─────────────────────────────────────────────────────────────


@dataclass
class SessionConfig:
    """Configuration for session scheduling behaviour."""

    max_session_hours: float = 3.0
    """Maximum continuous play session before forced logout (hours)."""

    min_offline_hours: float = 2.0
    """Minimum offline break between sessions (hours)."""

    max_offline_hours: float = 6.0
    """Maximum offline break between sessions (hours)."""

    daily_play_cap_hours: float = 8.0
    """Total play time per rolling 24h window before forced logout (hours)."""

    max_time_on_map_hours: float = 2.0
    """Maximum continuous time on the same map before rotation (hours)."""

    peak_hour_start: int = 20
    """Server peak hour start (24h)."""

    peak_hour_end: int = 23
    """Server peak hour end (24h)."""

    avoid_peak_hours: bool = True
    """If True, avoid botting during server peak hours."""

    gaussian_jitter_sigma: float = 0.3
    """Sigma for Gaussian jitter on login delay (fraction of base delay)."""


@dataclass
class BotSessionState:
    """Per-bot session tracking state."""

    bot_id: str = ""
    session_start: float = 0.0
    """Unix timestamp when the current session started."""

    total_today: float = 0.0
    """Total play time in the current rolling 24h window (seconds)."""

    last_logout: float = 0.0
    """Unix timestamp of the last logout."""

    current_map: str = ""
    """Current map the bot is on."""

    map_enter_time: float = 0.0
    """Unix timestamp when the bot entered the current map."""

    sessions_today: int = 0
    """Number of sessions started in the current 24h window."""

    is_logged_out: bool = False
    """Whether the bot is currently logged out (scheduled)."""

    scheduled_login_at: float = 0.0
    """Unix timestamp when the bot should log back in."""


# ── Session Manager ─────────────────────────────────────────────────────────


class SessionManager:
    """Manages per-bot session scheduling to avoid 24/7 detection patterns.

    Features:
      - Enforces max session duration (default 3h)
      - Enforces min/max offline breaks (2-6h)
      - Enforces daily play cap (8h per rolling 24h)
      - Enforces map rotation (>2h on same map triggers rotation)
      - Avoids server peak hours (20:00-23:00) when configured
      - Gaussian jitter on login delays for natural variation
      - All state persisted to JSON for restart survival

    Integration:
      Call ``should_logout(bot_id)`` from the PDCA assess() flow.
      If True, enqueue a ``quit`` action via HeuristicAction.
      Call ``get_next_login_delay(bot_id)`` to determine when to reconnect.
    """

    def __init__(
        self,
        config: SessionConfig | None = None,
        data_dir: str | Path | None = None,
    ) -> None:
        self._lock = RLock()
        self._config = config or SessionConfig()
        self._data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self._state_file = self._data_dir / SESSION_STATE_FILE

        # Per-bot session state: bot_id -> BotSessionState
        self._bot_states: dict[str, BotSessionState] = {}

        # Load persisted state
        self._load_state()

        logger.info(
            "SessionManager initialised  max_session=%.1fh  daily_cap=%.1fh  "
            "offline=[%.1f-%.1f]h  peak_avoid=%s",
            self._config.max_session_hours,
            self._config.daily_play_cap_hours,
            self._config.min_offline_hours,
            self._config.max_offline_hours,
            self._config.avoid_peak_hours,
        )

    # ── Public API ───────────────────────────────────────────────────────────

    def register_bot(self, bot_id: str) -> None:
        """Register a bot for session tracking.

        Creates a new BotSessionState if one doesn't exist.
        Safe to call multiple times.
        """
        with self._lock:
            if bot_id not in self._bot_states:
                self._bot_states[bot_id] = BotSessionState(bot_id=bot_id)
                logger.info("SessionManager: registered bot '%s'", bot_id)

    def start_session(self, bot_id: str, current_map: str = "") -> None:
        """Record the start of a new play session for a bot.

        Args:
            bot_id: Unique bot identifier.
            current_map: Map the bot is on at session start.
        """
        with self._lock:
            self.register_bot(bot_id)
            state = self._bot_states[bot_id]
            now = time.time()

            # If this is a new session (was logged out), reset session_start
            if state.is_logged_out or state.session_start == 0:
                state.session_start = now
                state.sessions_today += 1
                state.is_logged_out = False
                state.scheduled_login_at = 0.0
                logger.info(
                    "SessionManager: bot '%s' started session #%d",
                    bot_id, state.sessions_today,
                )

            if current_map:
                state.current_map = current_map
                state.map_enter_time = now

            self._save_state()

    def end_session(self, bot_id: str) -> None:
        """Record the end of a play session for a bot.

        Updates total_today and last_logout, then schedules the next login.
        """
        with self._lock:
            state = self._bot_states.get(bot_id)
            if state is None or state.is_logged_out:
                return

            now = time.time()
            session_duration = now - state.session_start

            # Accumulate total play time in rolling 24h window
            self._prune_old_sessions(bot_id)
            state.total_today += session_duration
            state.last_logout = now
            state.is_logged_out = True

            # Schedule next login
            delay = self._compute_login_delay(bot_id)
            state.scheduled_login_at = now + delay

            logger.info(
                "SessionManager: bot '%s' ended session (duration=%.1fh, "
                "total_today=%.1fh, next_login_in=%.1fh)",
                bot_id,
                session_duration / 3600,
                state.total_today / 3600,
                delay / 3600,
            )

            self._save_state()

    def should_logout(self, bot_id: str) -> bool:
        """Check if a bot should log out now.

        Returns True if any of these conditions are met:
          - Session duration > max_session_hours
          - Total play time today > daily_play_cap_hours
          - Currently in peak hours and avoid_peak_hours is True
          - Bot is already marked as logged out

        Args:
            bot_id: Unique bot identifier.

        Returns:
            True if the bot should log out / disconnect.
        """
        with self._lock:
            state = self._bot_states.get(bot_id)
            if state is None:
                return False

            # Already logged out
            if state.is_logged_out:
                return False

            # Session not started yet
            if state.session_start <= 0:
                return False

            now = time.time()
            session_duration = (now - state.session_start) / 3600  # hours

            # Condition 1: Max session duration exceeded
            if session_duration >= self._config.max_session_hours:
                logger.info(
                    "SessionManager: bot '%s' session=%.1fh > max=%.1fh — logout",
                    bot_id, session_duration, self._config.max_session_hours,
                )
                return True

            # Condition 2: Daily play cap exceeded
            self._prune_old_sessions(bot_id)
            total_hours = state.total_today / 3600
            if total_hours >= self._config.daily_play_cap_hours:
                logger.info(
                    "SessionManager: bot '%s' total_today=%.1fh > cap=%.1fh — logout",
                    bot_id, total_hours, self._config.daily_play_cap_hours,
                )
                return True

            # Condition 3: Peak hour avoidance
            if self._config.avoid_peak_hours:
                current_hour = time.localtime(now).tm_hour
                if self._config.peak_hour_start <= current_hour < self._config.peak_hour_end:
                    logger.info(
                        "SessionManager: bot '%s' peak hour %d-%d — logout",
                        bot_id, self._config.peak_hour_start, self._config.peak_hour_end,
                    )
                    return True

            return False

    def should_rotate_map(
        self,
        bot_id: str,
        current_map: str,
        time_on_map: float | None = None,
    ) -> bool:
        """Check if a bot should rotate to a different map.

        Returns True if the bot has been on the same map longer than
        max_time_on_map_hours.

        Args:
            bot_id: Unique bot identifier.
            current_map: Current map name.
            time_on_map: Seconds spent on this map. If None, uses internal
                         tracking from map_enter_time.

        Returns:
            True if the bot should move to a different map.
        """
        with self._lock:
            state = self._bot_states.get(bot_id)
            if state is None:
                return False

            # If map changed, update tracking
            if current_map != state.current_map:
                state.current_map = current_map
                state.map_enter_time = time.time()
                self._save_state()
                return False

            # Calculate time on map
            if time_on_map is not None:
                elapsed_hours = time_on_map / 3600
            else:
                elapsed_hours = (time.time() - state.map_enter_time) / 3600

            if elapsed_hours >= self._config.max_time_on_map_hours:
                logger.info(
                    "SessionManager: bot '%s' on map '%s' for %.1fh > max=%.1fh — rotate",
                    bot_id, current_map, elapsed_hours, self._config.max_time_on_map_hours,
                )
                return True

            return False

    def get_next_login_delay(self, bot_id: str) -> float:
        """Get the delay in seconds before the bot should log back in.

        Returns a random delay between min_offline_hours and max_offline_hours,
        with Gaussian jitter applied for natural variation.

        Args:
            bot_id: Unique bot identifier.

        Returns:
            Delay in seconds.
        """
        with self._lock:
            state = self._bot_states.get(bot_id)
            if state and state.scheduled_login_at > 0:
                remaining = state.scheduled_login_at - time.time()
                if remaining > 0:
                    return remaining

            return self._compute_login_delay(bot_id)

    def is_online(self, bot_id: str) -> bool:
        """Check if a bot is currently online (not scheduled-logged-out)."""
        with self._lock:
            state = self._bot_states.get(bot_id)
            if state is None:
                return False
            return not state.is_logged_out

    def get_session_info(self, bot_id: str) -> dict[str, Any]:
        """Get human-readable session info for a bot.

        Returns:
            Dict with session stats (duration, total_today, next_login, etc.).
        """
        with self._lock:
            state = self._bot_states.get(bot_id)
            if state is None:
                return {"bot_id": bot_id, "registered": False}

            now = time.time()
            session_duration = (now - state.session_start) / 3600 if not state.is_logged_out else 0

            return {
                "bot_id": bot_id,
                "registered": True,
                "is_online": not state.is_logged_out,
                "session_duration_hours": round(session_duration, 2),
                "total_today_hours": round(state.total_today / 3600, 2),
                "sessions_today": state.sessions_today,
                "current_map": state.current_map,
                "last_logout": state.last_logout,
                "next_login_at": state.scheduled_login_at,
                "next_login_in_hours": round(
                    max(0, state.scheduled_login_at - now) / 3600, 2
                ) if state.scheduled_login_at > 0 else 0,
            }

    def get_all_bot_info(self) -> dict[str, dict[str, Any]]:
        """Get session info for all registered bots."""
        with self._lock:
            return {bid: self.get_session_info(bid) for bid in self._bot_states}

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        bot_id: str,
    ) -> None:
        """Run session management assessment — call every PDCA cycle.

        Checks if the bot should log out and enqueues a quit action if so.

        Args:
            signals: Current bot state signals dict.
            actions: List to append HeuristicActions to.
            bot_id: Unique bot identifier.
        """
        if not signals or not bot_id:
            return

        self.register_bot(bot_id)

        # Extract current state from signals
        current_map = str(signals.get("map", signals.get("current_map", "")) or "")
        is_online = bool(signals.get("online", signals.get("is_online", True)))

        # If the bot is online, check session state
        if is_online:
            # Start session if not started
            state = self._bot_states.get(bot_id)
            if state and state.session_start == 0:
                self.start_session(bot_id, current_map)
            elif state and current_map and current_map != state.current_map:
                # Map changed — update tracking
                state.current_map = current_map
                state.map_enter_time = time.time()

            # Check if we should log out
            if self.should_logout(bot_id):
                actions.append(HeuristicAction(
                    kind="command",
                    command="quit",
                    confidence=0.95,
                    domain="session",
                    reason=(
                        f"SessionManager: session limit reached — "
                        f"enforcing anti-24/7 logout"
                    ),
                    metadata={
                        "action": "scheduled_logout",
                        "bot_id": bot_id,
                    },
                ))
                self.end_session(bot_id)

            # Check if we should rotate map
            if current_map and self.should_rotate_map(bot_id, current_map):
                actions.append(HeuristicAction(
                    kind="command",
                    command=f"move random",  # Bridge should resolve to a different map
                    confidence=0.80,
                    domain="session",
                    reason=(
                        f"SessionManager: on map '{current_map}' too long — "
                        f"rotating to avoid detection"
                    ),
                    metadata={
                        "action": "map_rotation",
                        "current_map": current_map,
                        "bot_id": bot_id,
                    },
                ))
        else:
            # Bot is offline — check if it's time to log back in
            state = self._bot_states.get(bot_id)
            if state and state.is_logged_out:
                now = time.time()
                if state.scheduled_login_at > 0 and now >= state.scheduled_login_at:
                    # Time to log back in
                    actions.append(HeuristicAction(
                        kind="command",
                        command="relog",
                        confidence=0.90,
                        domain="session",
                        reason="SessionManager: scheduled offline period ended — reconnecting",
                        metadata={
                            "action": "scheduled_relog",
                            "bot_id": bot_id,
                        },
                    ))
                    state.is_logged_out = False
                    state.scheduled_login_at = 0.0
                    self._save_state()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _compute_login_delay(self, bot_id: str) -> float:
        """Compute a random login delay with Gaussian jitter.

        Base delay is uniform random between min_offline and max_offline.
        Gaussian jitter (sigma fraction of base) is added for natural variation.
        Peak hour extension is applied if configured.

        Returns:
            Delay in seconds.
        """
        cfg = self._config
        base_hours = random.uniform(cfg.min_offline_hours, cfg.max_offline_hours)

        # Gaussian jitter
        jitter = random.gauss(0, base_hours * cfg.gaussian_jitter_sigma)
        delay_hours = base_hours + jitter

        # Clamp to configured range
        delay_hours = max(cfg.min_offline_hours, min(cfg.max_offline_hours, delay_hours))

        # Peak hour extension: if the scheduled login would land in peak hours,
        # extend the delay to push past peak
        if cfg.avoid_peak_hours:
            now = time.time()
            scheduled_time = now + delay_hours * 3600
            scheduled_hour = time.localtime(scheduled_time).tm_hour
            if cfg.peak_hour_start <= scheduled_hour < cfg.peak_hour_end:
                # Extend to just after peak hours
                target_hour = cfg.peak_hour_end
                now_struct = time.localtime(now)
                target_time = time.mktime((
                    now_struct.tm_year, now_struct.tm_mon, now_struct.tm_mday,
                    target_hour, 0, 0,
                    now_struct.tm_wday, now_struct.tm_yday, now_struct.tm_isdst,
                ))
                if target_time <= now:
                    target_time += 86400  # Next day
                delay_hours = (target_time - now) / 3600 + random.uniform(0.1, 0.5)
                logger.debug(
                    "SessionManager: extended login delay to avoid peak hours "
                    "(scheduled would land at %d:00)",
                    scheduled_hour,
                )

        return delay_hours * 3600

    def _prune_old_sessions(self, bot_id: str) -> None:
        """Prune session time older than 24 hours from total_today.

        This implements the rolling 24h window for daily play cap.
        Since we only track total_today as an accumulator, we decay it
        based on time since last logout.
        """
        state = self._bot_states.get(bot_id)
        if state is None:
            return

        now = time.time()
        # If the bot has been offline for more than 24h, reset
        if state.last_logout > 0 and (now - state.last_logout) > 86400:
            state.total_today = 0.0
            state.sessions_today = 0
            logger.debug("SessionManager: reset daily counters for bot '%s' (24h+ offline)", bot_id)

    # ── State persistence ────────────────────────────────────────────────────

    def _save_state(self) -> None:
        """Persist session state to disk as JSON."""
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            data: dict[str, Any] = {
                "config": {
                    "max_session_hours": self._config.max_session_hours,
                    "min_offline_hours": self._config.min_offline_hours,
                    "max_offline_hours": self._config.max_offline_hours,
                    "daily_play_cap_hours": self._config.daily_play_cap_hours,
                    "max_time_on_map_hours": self._config.max_time_on_map_hours,
                    "avoid_peak_hours": self._config.avoid_peak_hours,
                    "peak_hour_start": self._config.peak_hour_start,
                    "peak_hour_end": self._config.peak_hour_end,
                },
                "bot_states": {
                    bid: {
                        "bot_id": s.bot_id,
                        "session_start": s.session_start,
                        "total_today": s.total_today,
                        "last_logout": s.last_logout,
                        "current_map": s.current_map,
                        "map_enter_time": s.map_enter_time,
                        "sessions_today": s.sessions_today,
                        "is_logged_out": s.is_logged_out,
                        "scheduled_login_at": s.scheduled_login_at,
                    }
                    for bid, s in self._bot_states.items()
                },
                "_saved_at": time.time(),
            }

            tmp = str(self._state_file) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            Path(tmp).rename(self._state_file)

        except Exception as exc:
            logger.warning("SessionManager._save_state failed: %s", exc)

    def _load_state(self) -> None:
        """Load previously-persisted session state from disk."""
        try:
            if not self._state_file.exists():
                return

            with open(self._state_file) as f:
                data = json.load(f)

            # Restore config
            cfg_data = data.get("config", {})
            self._config.max_session_hours = cfg_data.get("max_session_hours", self._config.max_session_hours)
            self._config.min_offline_hours = cfg_data.get("min_offline_hours", self._config.min_offline_hours)
            self._config.max_offline_hours = cfg_data.get("max_offline_hours", self._config.max_offline_hours)
            self._config.daily_play_cap_hours = cfg_data.get("daily_play_cap_hours", self._config.daily_play_cap_hours)
            self._config.max_time_on_map_hours = cfg_data.get("max_time_on_map_hours", self._config.max_time_on_map_hours)
            self._config.avoid_peak_hours = cfg_data.get("avoid_peak_hours", self._config.avoid_peak_hours)
            self._config.peak_hour_start = cfg_data.get("peak_hour_start", self._config.peak_hour_start)
            self._config.peak_hour_end = cfg_data.get("peak_hour_end", self._config.peak_hour_end)

            # Restore bot states
            for bid, sdata in data.get("bot_states", {}).items():
                self._bot_states[bid] = BotSessionState(
                    bot_id=sdata.get("bot_id", bid),
                    session_start=sdata.get("session_start", 0.0),
                    total_today=sdata.get("total_today", 0.0),
                    last_logout=sdata.get("last_logout", 0.0),
                    current_map=sdata.get("current_map", ""),
                    map_enter_time=sdata.get("map_enter_time", 0.0),
                    sessions_today=sdata.get("sessions_today", 0),
                    is_logged_out=sdata.get("is_logged_out", False),
                    scheduled_login_at=sdata.get("scheduled_login_at", 0.0),
                )

            logger.info(
                "SessionManager loaded from %s (%d bots)",
                self._state_file, len(self._bot_states),
            )

        except Exception as exc:
            logger.warning("SessionManager._load_state failed: %s", exc)


# ── Singleton factory ───────────────────────────────────────────────────────

_session_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get or create the singleton SessionManager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
