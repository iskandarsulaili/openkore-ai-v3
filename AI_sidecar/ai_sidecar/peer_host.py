"""RAW peer-host supervisor for openkore-ai-v3.

Complementarity rule (founder directive 2026-08-26): the bot's "serves maps"
capacity must COMPLEMENT — never conflict with — the launcher's existing P2P
stack (peer-host map-server.exe, P2P relay, in-game WebRTC mesh).

Key invariants enforced here:
1. REUSE the launcher's `linux-host-map-server` binary (manifest-pinned, sha256
   verified, self-updating) — we NEVER ship a second/competing host binary.
2. SINGLE-WRITER EVE model: this module ONLY spawns the host on a box that does
   NOT already run the central map-server (or another host) on the map port.
   On the central box it refuses to start (no conflict). A peer host claims
   empty maps via the char JIT assigner — never fights another host.
3. The host uses the 2-file standalone mode: `id.conf` (operator SSO/host token)
   beside the binary -> it self-fetches ephemeral DB creds + self-reports
   host-seconds to p2p_hall_of_fame (same reward as a launcher-spawned host).
4. Lifecycle: boot (download-on-demand + verify), heartbeat supervision, kill
   on quit. Default OFF unless `p2p_host_enabled` + id.conf present + box is
   not the central owner.

The in-game transport mesh (0x035F/0x0361) and relay membership are the
LAUNCHER's DLL's job — this module never touches them.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# The manifest path for the raw Linux peer-host binary (single-file, embedded
# conf/db/npc, E5 attestation, self-updating). Manifest-pinned sha256.
HOST_MANIFEST_PATH = "linux-host-map-server"

# Map server internal listen port (also the central's). If a listener is on
# this port (central owner or another host), the bot must NOT spawn a host.
MAP_LISTEN_PORT = 5121

# Central's own map-server process name (on the box that hosts the world).
CENTRAL_MAP_EXE = "map-server"


class PeerHostSupervisor:
    """Supervises the RAW peer-host map-server as a bot capacity contribution.

    Thread-safe: spawn/supervise/kill guarded by a lock. The host process runs
    detached; a supervisor thread polls liveness + re-spawns on unexpected exit
    (bounded), and kills it cleanly on stop()/shutdown.
    """

    def __init__(
        self,
        *,
        data_dir: Path,
        manifest_base: str = "https://rathena-ai.openkore-ai.com",
        enabled: bool = False,
        host_manifest_path: str = HOST_MANIFEST_PATH,
        map_listen_port: int = MAP_LISTEN_PORT,
        central_map_exe: str = CENTRAL_MAP_EXE,
        respawn_delay_s: int = 60,
        max_respawns: int = 3,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.host_dir = self.data_dir / "peer-host"
        self.host_bin = self.host_dir / "map-server"
        self.id_conf = self.host_dir / "id.conf"
        self.host_log = self.host_dir / "host-boot.log"
        self._manifest_base = manifest_base.rstrip("/")
        self._host_manifest_path = host_manifest_path
        self._map_listen_port = map_listen_port
        self._central_map_exe = central_map_exe
        self._enabled = bool(enabled)
        self._respawn_delay_s = respawn_delay_s
        self._max_respawns = max_respawns
        self._lock = threading.RLock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._supervisor_thread: threading.Thread | None = None
        self._running = False
        self._respawn_count = 0
        self._host_started_ts: float = 0.0

    # ── identity / capability helpers ────────────────────────────────────

    def box_is_central(self) -> bool:
        """True if THIS box already owns the map port (central or another host).

        Single-writer EVE guard: if a map-server is listening on the map port,
        spawning a second host would conflict -> the bot must NOT host here.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.3)
                return s.connect_ex(("127.0.0.1", self._map_listen_port)) == 0
        except OSError:
            return False

    def has_token(self) -> bool:
        """True if id.conf holds a usable SSO/host token (bare or token=)."""
        if not self.id_conf.is_file():
            return False
        try:
            for line in self.id_conf.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("//"):
                    return True
            return False
        except OSError:
            return False

    def can_host(self) -> tuple[bool, str]:
        """Check all preconditions. Returns (ok, reason)."""
        if not self._enabled:
            return False, "p2p_host_enabled=false"
        if self.box_is_central():
            return False, f"port {self._map_listen_port} owned by central/another host (single-writer EVE)"
        if not self.has_token():
            return False, "no token in id.conf (anonymous host earns no reward; configure id.conf to enable)"
        return True, "ready"

    # ── download / verify ────────────────────────────────────────────────

    def _fetch(self, url: str, timeout: int = 120) -> bytes:
        req = Request(url, headers={"User-Agent": "openkore-ai-v3-peer-host/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()

    def _sha256(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _latest_sha(self) -> tuple[str, str]:
        """Fetch the manifest, return (url, sha256) for the host binary."""
        man_url = f"{self._manifest_base}/api/ads/manifest"
        data = self._fetch(man_url)
        import json

        d = json.loads(data)
        for f in d.get("data", {}).get("files", []):
            if f and isinstance(f, dict) and f.get("path") == self._host_manifest_path:
                return (f.get("url", ""), f.get("sha256", ""))
        raise RuntimeError(f"host binary {self._host_manifest_path} not in manifest")

    def ensure_binary(self) -> bool:
        """Download the host binary if missing/stale (manifest sha256 pinned)."""
        try:
            url, want_sha = self._latest_sha()
        except Exception as e:  # noqa: BLE001
            logger.warning("peer_host_manifest_fetch_failed: %s", e)
            return False
        self.host_dir.mkdir(parents=True, exist_ok=True)
        current = self.host_bin.read_bytes() if self.host_bin.is_file() else b""
        if current and self._sha256(current) == want_sha:
            return True  # already current
        logger.info("peer_host_downloading: size-pending sha=%s...", want_sha[:16])
        body = self._fetch(url)
        if self._sha256(body) != want_sha:
            logger.warning("peer_host_sha_mismatch: got=%s want=%s", self._sha256(body)[:16], want_sha[:16])
            return False
        # atomic write
        tmp = self.host_bin.with_suffix(".bin.tmp")
        tmp.write_bytes(body)
        os.replace(tmp, self.host_bin)
        os.chmod(self.host_bin, 0o755)
        logger.info("peer_host_ready: %d bytes sha=%s", len(body), want_sha[:16])
        return True

    # ── spawn / supervise / kill ─────────────────────────────────────────

    def start(self) -> tuple[bool, str]:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                return True, "already running"
            ok, reason = self.can_host()
            if not ok:
                logger.info("peer_host_skip: %s", reason)
                return False, reason
            if not self.ensure_binary():
                return False, "host binary unavailable (download/sha)"
            env = os.environ.copy()
            # Do NOT set RAW_HOST_TOKEN/RAW_HOST_SESSION here — the 2-file
            # standalone host self-bootstraps by reading id.conf beside the
            # binary at Core::start() (hosted_reward_apply_env). Passing empty
            # values would defeat that bootstrap.
            env["HOST_LOG"] = str(self.host_log)
            self.host_dir.mkdir(parents=True, exist_ok=True)
            logf = open(self.host_log, "ab")
            try:
                self._proc = subprocess.Popen(
                    [str(self.host_bin)],
                    cwd=str(self.host_dir),
                    env=env,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            finally:
                logf.close()
            self._host_started_ts = time.time()
            self._running = True
            self._respawn_count = 0
            self._supervisor_thread = threading.Thread(
                target=self._supervise, daemon=True, name="peer-host-supervisor"
            )
            self._supervisor_thread.start()
            logger.info("peer_host_spawned: pid=%d log=%s", self._proc.pid, self.host_log)
            return True, f"spawned pid={self._proc.pid}"

    def _supervise(self) -> None:
        while self._running:
            time.sleep(5)
            with self._lock:
                if not self._running or self._proc is None:
                    return
                rc = self._proc.poll()
                if rc is None:
                    continue  # alive
                # exited unexpectedly
                if self._respawn_count >= self._max_respawns:
                    logger.warning("peer_host_gave_up: rc=%d after %d respawns", rc, self._respawn_count)
                    return
                self._respawn_count += 1
                logger.info("peer_host_exited: rc=%d respawn %d/%d in %ds", rc, self._respawn_count, self._max_respawns, self._respawn_delay_s)
                time.sleep(self._respawn_delay_s)
                ok, reason = self.can_host()
                if not ok:
                    logger.info("peer_host_respawn_skip: %s", reason)
                    return
                if not self.ensure_binary():
                    logger.warning("peer_host_respawn_binary_missing")
                    return
                logf = open(self.host_log, "ab")
                try:
                    self._proc = subprocess.Popen(
                        [str(self.host_bin)],
                        cwd=str(self.host_dir),
                        env=os.environ.copy(),
                        stdout=logf,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                finally:
                    logf.close()
                self._host_started_ts = time.time()

    def stop(self) -> None:
        with self._lock:
            self._running = False
            p = self._proc
            if p and p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    try:
                        p.terminate()
                    except Exception:  # noqa: BLE001
                        pass
                try:
                    p.wait(timeout=8)
                except Exception:  # noqa: BLE001
                    try:
                        p.kill()
                    except Exception:  # noqa: BLE001
                        pass
            self._proc = None

    def status(self) -> dict[str, Any]:
        with self._lock:
            running = bool(self._proc and self._proc.poll() is None)
            return {
                "enabled": self._enabled,
                "running": running,
                "pid": self._proc.pid if running else None,
                "uptime_s": int(time.time() - self._host_started_ts) if self._host_started_ts else 0,
                "respawns": self._respawn_count,
                "can_host": self.can_host(),
                "host_dir": str(self.host_dir),
                "has_token": self.has_token(),
                "box_is_central": self.box_is_central(),
                "log": str(self.host_log),
            }
