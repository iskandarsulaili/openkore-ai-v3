#!/usr/bin/env python3
"""
Multi-Account Manager — Start/stop 3 bots, stagger reconnects, avoid dual-login.
=============================================================================
Solves the #1 problem: bots keep killing each other's sessions.
"""

import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from threading import Lock, Thread
from typing import Any

logger = logging.getLogger(__name__)


class MultiAccountManager:
    """Manages multiple OpenKore bot processes with staggered reconnects.

    Each bot gets its own profile directory. The manager ensures:
    - Bots connect one at a time (staggered by 10s)
    - Old sessions are properly cleaned before reconnect
    - Bot processes are monitored and auto-restarted
    - All bots are killed cleanly on shutdown
    """

    def __init__(self, base_dir: str, profiles: list[str],
                 config: dict[str, Any] | None = None):
        self._base = Path(base_dir)
        self._profiles = profiles  # ["kicapmasin", "kicapmasin2", "kicapmasin3"]
        self._config = config or {}
        self._lock = Lock()
        self._processes: dict[str, subprocess.Popen] = {}
        self._monitor_thread: Thread | None = None
        self._running = False
        self._stagger_delay = int(self._config.get("stagger_delay", 10))
        self._server_timeout = int(self._config.get("server_timeout", 120))

    def start_all(self) -> dict[str, int]:
        """Start all bots with staggered delays."""
        pids = {}

        for i, profile in enumerate(self._profiles):
            if i == 0:
                # First bot: wait for server timeout to clear old sessions
                logger.info("Waiting %ds for server session timeout...", self._server_timeout)
                time.sleep(self._server_timeout)
            else:
                # Subsequent bots: stagger by delay
                logger.info("Staggering %s by %ds...", profile, self._stagger_delay)
                time.sleep(self._stagger_delay)

            pid = self._start_bot(profile)
            if pid:
                pids[profile] = pid
                logger.info("Started %s (PID %d)", profile, pid)

        return pids

    def start_one(self, profile: str) -> int | None:
        """Start a single bot."""
        return self._start_bot(profile)

    def _start_bot(self, profile: str) -> int | None:
        """Start one OpenKore process."""
        control_dir = self._base / ".bot_profiles" / profile / "control"
        if not control_dir.exists():
            logger.error("Profile directory not found: %s", control_dir)
            return None

        try:
            proc = subprocess.Popen(
                ["perl", "-I", "src", "openkore.pl",
                 "--plugins=plugins", f"--control={control_dir}"],
                cwd=str(self._base),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid,
            )
            with self._lock:
                self._processes[profile] = proc
            return proc.pid
        except Exception as e:
            logger.error("Failed to start bot %s: %s", profile, e)
            return None

    def stop_all(self) -> dict[str, int]:
        """Stop all bots gracefully, then force kill remaining."""
        results = {}

        # Phase 1: Graceful SIGTERM
        with self._lock:
            for profile, proc in list(self._processes.items()):
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    results[profile] = "sigterm"
                except Exception as e:
                    logger.warning("Failed to SIGTERM %s: %s", profile, e)

        time.sleep(3)

        # Phase 2: Force SIGKILL
        with self._lock:
            for profile, proc in list(self._processes.items()):
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                    results[profile] = "sigkill"
                except Exception:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        results[profile] = "nuclear"
                    except Exception:
                        results[profile] = "failed"

        with self._lock:
            self._processes.clear()

        return results

    def stop_one(self, profile: str) -> bool:
        """Stop a single bot."""
        with self._lock:
            proc = self._processes.pop(profile, None)
        if not proc:
            return False
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5)
            return True
        except Exception:
            try:
                proc.kill()
                return True
            except Exception:
                return False

    def is_running(self, profile: str) -> bool:
        """Check if a bot is running."""
        with self._lock:
            proc = self._processes.get(profile)
        if not proc:
            return False
        return proc.poll() is None

    def running_count(self) -> int:
        """Count running bots."""
        with self._lock:
            return sum(1 for p in self._processes.values() if p.poll() is None)

    def status(self) -> dict[str, Any]:
        """Return status of all bots."""
        result = {}
        for profile in self._profiles:
            result[profile] = {
                "running": self.is_running(profile),
                "pid": self._processes.get(profile).pid if self._processes.get(profile) else None,
            }
        return result

    def start_monitor(self, interval_s: int = 30) -> None:
        """Start background thread to monitor bot health."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return

        self._running = True
        self._monitor_thread = Thread(target=self._monitor_loop, args=(interval_s,), daemon=True)
        self._monitor_thread.start()
        logger.info("Bot monitor started (interval=%ds)", interval_s)

    def stop_monitor(self) -> None:
        self._running = False

    def _monitor_loop(self, interval_s: int) -> None:
        """Monitor bot health and restart dead bots."""
        while self._running:
            time.sleep(interval_s)
            with self._lock:
                for profile, proc in list(self._processes.items()):
                    if proc.poll() is not None:
                        logger.warning("Bot %s died (PID %d, exit %d), restarting...",
                                       profile, proc.pid, proc.returncode)
                        # Remove dead process
                        del self._processes[profile]

            # Restart dead bots outside the lock
            # (We need to re-check because _start_bot acquires the lock)
            self._restart_dead_bots()

    def _restart_dead_bots(self) -> None:
        """Restart any dead bots."""
        for profile in self._profiles:
            with self._lock:
                proc = self._processes.get(profile)
                if proc and proc.poll() is None:
                    continue  # Still alive

            # Dead or not running — restart
            logger.info("Restarting bot %s...", profile)
            time.sleep(5)  # Wait before reconnecting
            self._start_bot(profile)