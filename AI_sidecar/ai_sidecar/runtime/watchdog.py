"""Watchdog Supervisor — manages bot process lifecycle with state machine integration.

Architecture:
  The watchdog manages the OS-level bot PROCESS lifecycle (start/stop/restart).
  The LifecycleManager from progression/lifecycle.py manages the CONNECTION state
  (DISCONNECTED → ACTIVE). The watchdog uses the lifecycle state to decide whether
  a bot should be restarted or left alone:
  
  - If a bot is in ACTIVE state and its process dies → it's HARDWARE/SERVER CRASH,
    don't restart blindly (server may be down)
  - If a bot never reached ACTIVE and dies → restart with backoff
  - If a bot has been in onboarding too long → it's stale, restart
  
  On startup, the watchdog creates characters via cold_start.py if none exist.
  Cold start creates a level 1 Novice with starter gear and farms Porings for
  starting zeny before progressing to the configured hunting build.
"""

from __future__ import annotations
import os
import sys
import time
import signal
import logging
import subprocess
import threading
from typing import Any
from collections import defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Try importing lifecycle manager — it may or may not be installed
try:
    from ai_sidecar.domains.progression.lifecycle import (
        LifecycleManager, BotState, StateTimeoutConfig, BackoffConfig,
    )
    HAS_LIFECYCLE = True
except ImportError:
    HAS_LIFECYCLE = False
    logger.warning("[watchdog] LifecycleManager not available — running in legacy mode")

try:
    from ai_sidecar.domains.progression.cold_start import ColdStartManager
    HAS_COLD_START = True
except ImportError:
    HAS_COLD_START = False
    logger.warning("[watchdog] ColdStartManager not available — skipping auto character creation")


class BotProcess:
    """Tracks a single bot's OS-level process lifecycle.

    This is the lowest-level tracking: is the process running?
    Higher-level connection state tracking is done by LifecycleManager.
    """

    def __init__(self, name: str, profile_dir: str):
        self.name = name
        self.profile_dir = profile_dir
        self.pid: int | None = None
        self.process: subprocess.Popen | None = None
        self.last_restart: datetime | None = None
        self.restart_count: int = 0
        self.restart_window: list[datetime] = []
        self.max_restarts_per_hour = 5
        self.console_log: str = ""
        self.started_at: datetime | None = None
        self.last_log_write: datetime | None = None
        self.exit_code: int | None = None
        self._consecutive_crash_count: int = 0  # Crashes within 5min of start

    def can_restart(self) -> bool:
        """Check if bot can be restarted (circuit breaker)."""
        now = datetime.now()
        self.restart_window = [t for t in self.restart_window if now - t < timedelta(hours=1)]
        # If we crashed 3+ times within 5 minutes of starting, increase backoff
        if self._consecutive_crash_count >= 3:
            return False  # Too many quick crashes — server may be down
        return len(self.restart_window) < self.max_restarts_per_hour

    def _get_start_command(self) -> list[str]:
        """Build the OpenKore command line."""
        return [
            "perl", "-I", f"{PROJECT_ROOT}/src",
            f"{PROJECT_ROOT}/openkore.pl",
            "--plugins=plugins",
            f"--control={self.profile_dir}",
        ]

    def start(self) -> bool:
        """Start the bot process."""
        if not self.can_restart():
            logger.error(f"[watchdog] {self.name}: Circuit breaker tripped (max {self.max_restarts_per_hour}/hour)")
            return False

        log_file = f"/tmp/bot_watchdog_{self.name}.log"
        cmd = self._get_start_command()

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=open(log_file, 'w'),
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,  # never inherit tty stdin: if
                # ErrorHandler::showError hits <STDIN> on a die during
                # shutdown, it must return EOF immediately, not hang forever.
                cwd=PROJECT_ROOT,
            )
            self.pid = self.process.pid
            self.started_at = datetime.now()
            self.last_log_write = datetime.now()
            self.restart_count += 1
            self.restart_window.append(datetime.now())
            logger.warning(
                f"[watchdog] {self.name}: Started (PID {self.pid}, restart #{self.restart_count})"
            )
            return True
        except Exception as e:
            logger.error(f"[watchdog] {self.name}: Start failed: {e}")
            return False

    def is_alive(self) -> bool:
        """Check if bot process is still running."""
        if not self.process or not self.pid:
            return False
        try:
            alive = self.process.poll() is None
            if not alive:
                self.exit_code = self.process.returncode
                # Track consecutive quick crashes
                if self.started_at and (datetime.now() - self.started_at).total_seconds() < 300:
                    self._consecutive_crash_count += 1
                else:
                    self._consecutive_crash_count = 0
            return alive
        except Exception:
            return False

    def is_stale(self, max_idle_seconds: int = 300) -> bool:
        """Check if bot has stopped writing to its console log.

        Increased from 120s to 300s to account for map loading + character creation.
        """
        if not self.last_log_write:
            return False
        return (datetime.now() - self.last_log_write).total_seconds() > max_idle_seconds

    def stop(self) -> None:
        """Stop the bot process."""
        if self.process and self.pid:
            try:
                os.kill(self.pid, signal.SIGTERM)
                time.sleep(2)
                if self.process.poll() is None:
                    os.kill(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process = None
            self.pid = None
            self.exit_code = None


class WatchdogSupervisor:
    """Monitors all bots and manages their lifecycles.

    Integrates with LifecycleManager for connection state tracking and
    ColdStartManager for automatic character creation.
    """

    def __init__(self, check_interval: int = 15):
        self._bots: dict[str, BotProcess] = {}
        self._check_interval = check_interval
        self._running = False
        self._sidecar_url = "http://127.0.0.1:18081"

        # Initialize lifecycle manager
        if HAS_LIFECYCLE:
            self._lifecycle = LifecycleManager(
                state_timeouts=StateTimeoutConfig(
                    connected=30.0,
                    authenticated=60.0,
                    map_loaded=180.0,  # 3 min for 8000+ maps
                    character_selected=60.0,
                    in_game=180.0,  # 3 min for onboarding
                    active=0.0,
                ),
                backoff=BackoffConfig(base_delay=5.0, max_delay=300.0, multiplier=2.0),
            )
        else:
            self._lifecycle = None

        # Initialize cold start manager
        if HAS_COLD_START:
            self._cold_start = ColdStartManager()
        else:
            self._cold_start = None

        # Track bot connection state from snapshots
        self._bot_connection_state: dict[str, dict] = {}

    def add_bot(self, name: str, profile_dir: str) -> None:
        """Register a bot for monitoring."""
        full_dir = os.path.join(PROJECT_ROOT, profile_dir)
        if not os.path.isdir(full_dir):
            logger.error(f"[watchdog] Profile dir not found: {full_dir}")
            return
        self._bots[name] = BotProcess(name, full_dir)
        if self._lifecycle:
            self._lifecycle.register_bot(name, {"profile_dir": full_dir})
        logger.info(f"[watchdog] Registered bot: {name} (profile: {full_dir})")

    def _validate_sidecar_accessible(self) -> bool:
        """Check if the sidecar is running."""
        import urllib.request
        try:
            resp = urllib.request.urlopen(f"{self._sidecar_url}/health", timeout=3)
            return resp.status == 200
        except Exception:
            return False

    def _process_snapshot(self, bot_name: str, snapshot: dict) -> BotState | None:
        """Process a snapshot to determine the bot's connection state.

        Returns the new state or None if unchanged.
        """
        if not self._lifecycle:
            return None

        # Extract connection indicators from snapshot
        actor = snapshot.get("actor", {})
        net = snapshot.get("net", {})
        config = snapshot.get("config", {})

        # Determine connection state from snapshot fields
        is_connected = net.get("serverConnected", False)
        is_in_game = actor.get("inGame", False)
        char_name = actor.get("name", "")
        map_name = actor.get("map", "")
        hp = actor.get("hp", 0)
        max_hp = actor.get("hp_max", 1)

        if not is_connected:
            return self._lifecycle.report_disconnected(bot_name)

        if is_in_game and char_name:
            # In game — check if maps are loaded and we have HP data
            if map_name and hp > 0:
                # Fully operational
                return self._lifecycle.report_active(bot_name)
            else:
                # In game but still loading
                return self._lifecycle.report_in_game(bot_name)

        # Connected but not in game — check sub-states
        has_char_select = config.get("char", "") != ""
        maps_loaded = config.get("maps_loaded", False)

        if maps_loaded:
            return self._lifecycle.report_map_loaded(bot_name)
        if has_char_select:
            return self._lifecycle.report_character_selected(bot_name)
        if is_connected:
            return self._lifecycle.report_authenticated(bot_name)

        return None

    def _fetch_bot_connection_states(self) -> dict[str, BotState]:
        """Fetch bot states from sidecar /v1/fleet endpoint."""
        import urllib.request
        import json

        states: dict[str, BotState] = {}
        try:
            resp = urllib.request.urlopen(f"{self._sidecar_url}/v1/fleet/status", timeout=3)
            data = json.loads(resp.read())
            for bot_id, info in data.get("bots", {}).items():
                state_str = info.get("state", "disconnected")
                try:
                    states[bot_id] = BotState(state_str)
                except ValueError:
                    pass
        except Exception:
            pass
        return states

    def _on_startup_character_check(self) -> None:
        """On startup, check if characters need to be created.

        The cold_start manager creates characters when the server is available.
        This is called once at startup.
        """
        if not self._cold_start:
            return
        # The cold start runs asynchronously — it monitors snapshots and
        # creates characters when it detects the bot is on the character
        # select screen. No action needed here.

    def _check_bots(self) -> None:
        """Single check cycle — make lifecycle-aware restart decisions."""
        now = datetime.now()

        # Fetch connection states from sidecar
        connection_states = {}
        if self._lifecycle:
            try:
                connection_states = self._fetch_bot_connection_states()
            except Exception:
                pass

        for name, bot in list(self._bots.items()):
            bot_is_alive = bot.is_alive()

            # Update lifecycle from sidecar state
            if self._lifecycle and name in connection_states:
                lc_state = connection_states[name]
                # Only transition if the sidecar reports a different state
                if lc_state == BotState.ACTIVE:
                    self._lifecycle.report_active(name)

            if bot_is_alive:
                bot.last_log_write = now
                continue

            # Bot process is dead — consult lifecycle state
            should_restart = True
            if self._lifecycle:
                lc = self._lifecycle._bots.get(name)
                if lc and lc.is_operational:
                    # Bot was ACTIVE but process died — server or network issue
                    # Don't restart blindly; let the operator handle it
                    logger.warning(
                        f"[watchdog] {name}: ACTIVE bot process died (exit={bot.exit_code}). "
                        f"Server may be down. Not restarting."
                    )
                    should_restart = False
                elif lc and lc.failure_count >= 5:
                    logger.warning(
                        f"[watchdog] {name}: Too many onboarding failures ({lc.failure_count}). "
                        f"Waiting for manual intervention."
                    )
                    should_restart = False

            if should_restart:
                uptime = ""
                if bot.started_at:
                    uptime = f" after {(now - bot.started_at).total_seconds():.0f}s uptime"
                logger.warning(
                    f"[watchdog] {name}: Process died{uptime} (exit={bot.exit_code}, "
                    f"restart #{bot.restart_count})"
                )
                if bot.can_restart():
                    logger.warning(f"[watchdog] {name}: Restarting...")
                    self._lifecycle and self._lifecycle.register_bot(name)
                    bot.start()
                else:
                    logger.error(f"[watchdog] {name}: Circuit breaker tripped — skipping restart")

    def _start_single_bot(self, name: str, retries: int = 3) -> bool:
        """Start a single bot with retries."""
        bot = self._bots.get(name)
        if not bot:
            return False
        for attempt in range(retries):
            if bot.start():
                if self._lifecycle:
                    self._lifecycle.register_bot(name)
                return True
            if attempt < retries - 1:
                logger.warning(f"[watchdog] {name}: Start attempt {attempt + 1} failed, retrying...")
                time.sleep(2)
        return False

    def start_all(self) -> None:
        """Start all registered bots."""
        logger.warning(f"[watchdog] Starting {len(self._bots)} bots...")
        # Character check is done by cold_start during snapshot processing
        self._on_startup_character_check()
        for name, bot in self._bots.items():
            if not bot.is_alive():
                self._start_single_bot(name)

    def stop_all(self) -> None:
        """Stop all bots."""
        logger.warning(f"[watchdog] Stopping all bots...")
        for name, bot in self._bots.items():
            bot.stop()
        if self._lifecycle:
            for name in list(self._lifecycle._bots.keys()):
                self._lifecycle.unregister_bot(name)

    def run(self) -> None:
        """Main loop — monitor and restart bots with lifecycle awareness."""
        if not self._bots:
            logger.error("[watchdog] No bots registered. Add bots with add_bot()")
            return

        self._running = True
        self.start_all()
        logger.warning(f"[watchdog] Running — checking every {self._check_interval}s")

        try:
            while self._running:
                time.sleep(self._check_interval)
                self._check_bots()
        except KeyboardInterrupt:
            logger.warning("[watchdog] Shutting down...")
        finally:
            self.stop_all()

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive status of all bots."""
        status = {
            "bots": {},
            "alive_count": 0,
            "total_count": len(self._bots),
        }
        for name, bot in self._bots.items():
            lc_state = None
            lc_failures = None
            if self._lifecycle:
                lc = self._lifecycle._bots.get(name)
                if lc:
                    lc_state = lc.state.value
                    lc_failures = lc.failure_count

            bot_status = {
                "alive": bot.is_alive(),
                "pid": bot.pid,
                "uptime": str(datetime.now() - bot.started_at).split('.')[0] if bot.started_at and bot.is_alive() else "N/A",
                "restarts": bot.restart_count,
                "exit_code": bot.exit_code,
            }
            if lc_state:
                bot_status["lifecycle_state"] = lc_state
            if lc_failures is not None:
                bot_status["failures"] = lc_failures

            status["bots"][name] = bot_status
            if bot.is_alive():
                status["alive_count"] += 1

        return status


def run_daemon():
    """Entry point for running as a background daemon."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{PROJECT_ROOT}/logs/watchdog.log"),
        ],
    )

    supervisor = WatchdogSupervisor(check_interval=15)

    # Register bots from bot_profiles
    profiles_dir = os.path.join(PROJECT_ROOT, ".bot_profiles")
    if os.path.isdir(profiles_dir):
        for name in sorted(os.listdir(profiles_dir)):
            profile_dir = os.path.join(profiles_dir, name, "control")
            if os.path.isdir(profile_dir):
                supervisor.add_bot(name, os.path.join(".bot_profiles", name, "control"))

    if not supervisor._bots:
        # Fallback: register default bots
        for name in ["kicapmasin", "kicapmasin2", "kicapmasin3"]:
            supervisor.add_bot(name, f".bot_profiles/{name}/control")

    supervisor.run()


if __name__ == "__main__":
    run_daemon()
