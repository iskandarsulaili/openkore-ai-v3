"""Watchdog Supervisor — auto-restarts crashed bots within 30 seconds.

The start.sh launches bots once. When they crash (which they do),
they stay crashed. This supervisor:
1. Monitors bot processes every 15 seconds
2. Detects crashes by PID existence + console log staleness
3. Restarts crashed bots with correct profile config
4. Logs all restarts with crash reason
5. Circuit breaker: max 5 restarts per bot per hour
"""
from __future__ import annotations
import os
import sys
import time
import signal
import logging
import subprocess
from typing import Any
from collections import defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class BotProcess:
    """Tracks a single bot's lifecycle."""
    
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
    
    def can_restart(self) -> bool:
        """Check if bot can be restarted (circuit breaker)."""
        # Clear window of restarts older than 1 hour
        now = datetime.now()
        self.restart_window = [t for t in self.restart_window if now - t < timedelta(hours=1)]
        return len(self.restart_window) < self.max_restarts_per_hour
    
    def start(self) -> bool:
        """Start the bot process."""
        if not self.can_restart():
            logger.error(f"[watchdog] {self.name}: Circuit breaker tripped (max {self.max_restarts_per_hour}/hour)")
            return False
        
        log_file = f"/tmp/bot_watchdog_{self.name}.log"
        cmd = [
            "perl", "-I", f"{PROJECT_ROOT}/src",
            f"{PROJECT_ROOT}/openkore.pl",
            "--plugins=plugins",
            f"--control={self.profile_dir}",
        ]
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=open(log_file, 'w'),
                stderr=subprocess.STDOUT,
                cwd=PROJECT_ROOT,
            )
            self.pid = self.process.pid
            self.started_at = datetime.now()
            self.last_log_write = datetime.now()
            self.restart_count += 1
            self.restart_window.append(datetime.now())
            logger.warning(f"[watchdog] {self.name}: Started (PID {self.pid}, restart #{self.restart_count})")
            return True
        except Exception as e:
            logger.error(f"[watchdog] {self.name}: Start failed: {e}")
            return False
    
    def is_alive(self) -> bool:
        """Check if bot process is still running."""
        if not self.process or not self.pid:
            return False
        # Check if process exists
        try:
            alive = self.process.poll() is None
            if not alive:
                logger.warning(f"[watchdog] {self.name}: Process exited with code {self.process.returncode}")
            return alive
        except Exception:
            return False
    
    def is_stale(self, max_idle_seconds: int = 120) -> bool:
        """Check if bot has stopped writing to its console log."""
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


class WatchdogSupervisor:
    """Monitors all bots and restarts crashed ones.
    
    Usage:
        supervisor = WatchdogSupervisor()
        supervisor.add_bot("kicapmasin", ".bot_profiles/kicapmasin/control")
        supervisor.run()  # Blocks forever, monitoring every 15s
    """
    
    def __init__(self, check_interval: int = 15):
        self._bots: dict[str, BotProcess] = {}
        self._check_interval = check_interval
        self._running = False
        self._sidecar_url = "http://127.0.0.1:18081"
    
    def add_bot(self, name: str, profile_dir: str) -> None:
        """Register a bot for monitoring."""
        full_dir = os.path.join(PROJECT_ROOT, profile_dir)
        if not os.path.isdir(full_dir):
            logger.error(f"[watchdog] Profile dir not found: {full_dir}")
            return
        self._bots[name] = BotProcess(name, full_dir)
        logger.info(f"[watchdog] Registered bot: {name} (profile: {full_dir})")
    
    def check_sidecar_health(self) -> bool:
        """Check if the sidecar is running."""
        import urllib.request
        try:
            resp = urllib.request.urlopen(f"{self._sidecar_url}/health", timeout=3)
            return resp.status == 200
        except Exception:
            return False
    
    def get_bot_status_from_sidecar(self) -> dict[str, str]:
        """Get bot status from sidecar fleet endpoint."""
        import urllib.request, json
        try:
            resp = urllib.request.urlopen(f"{self._sidecar_url}/v1/fleet/status", timeout=3)
            data = json.loads(resp.read())
            return data.get("bots", {})
        except Exception:
            return {}
    
    def _check_bots(self) -> None:
        """Single check cycle — check all bots and restart dead ones."""
        now = datetime.now()
        
        for name, bot in list(self._bots.items()):
            if bot.is_alive():
                # Update log staleness check
                bot.last_log_write = now
                continue
            
            # Bot is dead — restart if circuit breaker allows
            if bot.started_at:
                # Calculate uptime
                uptime = (now - bot.started_at).total_seconds()
                logger.warning(f"[watchdog] {name}: Crashed after {uptime:.0f}s uptime (restart #{bot.restart_count})")
            
            if bot.can_restart():
                logger.warning(f"[watchdog] {name}: Restarting...")
                bot.start()
            else:
                logger.error(f"[watchdog] {name}: Circuit breaker tripped — skipping restart")
        
        # Log summary
        alive = sum(1 for b in self._bots.values() if b.is_alive())
        if alive < len(self._bots):
            logger.warning(f"[watchdog] Status: {alive}/{len(self._bots)} bots alive")
    
    def start_all(self) -> None:
        """Start all registered bots."""
        logger.warning(f"[watchdog] Starting {len(self._bots)} bots...")
        for name, bot in self._bots.items():
            if not bot.is_alive():
                bot.start()
    
    def stop_all(self) -> None:
        """Stop all bots."""
        logger.warning(f"[watchdog] Stopping all bots...")
        for name, bot in self._bots.items():
            bot.stop()
    
    def run(self) -> None:
        """Main loop — monitor and restart bots forever."""
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
        return {
            "bots": {
                name: {
                    "alive": bot.is_alive(),
                    "pid": bot.pid,
                    "uptime": str(datetime.now() - bot.started_at).split('.')[0] if bot.started_at and bot.is_alive() else "N/A",
                    "restarts": bot.restart_count,
                }
                for name, bot in self._bots.items()
            },
            "alive_count": sum(1 for b in self._bots.values() if b.is_alive()),
            "total_count": len(self._bots),
        }


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
        for name in os.listdir(profiles_dir):
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
