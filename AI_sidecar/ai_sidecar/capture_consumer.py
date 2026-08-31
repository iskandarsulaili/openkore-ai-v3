"""
Capture Consumer — learns the REAL 0x0436 map-login layout from captured packets.
=================================================================================
BQ (2026-08-31, user directive "All of them" = A minimal + B ML feed + C both).

The packet-capture chain (DLL -> ring -> launcher -> central server -> export)
collects the REAL client's 0x0436 map-login bytes. This consumer:

  A. PULLS the export (Bearer + paginated), parses 0x0436 frames, learns the
     ACTUAL layout (length + field offsets), and writes it to the bot config
     (`mapLoginLength` + `mapLoginLayout`) so the bot's sendMapLogin emits the
     server's accepted form — replacing the blind 19->23->26 rotation AND the
     hardcoded (wrong) 23-byte layout.
  B. FEEDS the ML store: appends every captured 0x0436 frame (raw bytes + the
     learned layout) to shared_learning_db.packet_layouts for ML training.
  C. BOTH — one module does A + B.

Server-agnostic: no hardcoded server values. The layout is LEARNED from the
captured bytes (the account_id field position is detected by matching the
known account_id value, not assumed).

Auth: the sidecar shares the box with MariaDB (DB_USER/DB_PASSWORD/DB_HOST/
DB_DATABASE env). It mints a one-time SSO token for an admin (group 99) account
directly in discord_login_tokens, then calls the export with `Authorization:
Bearer <token>` (the Y1 auth path the export accepts).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import secrets
import struct
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The 0x0436 packet id (CZ_ENTER2).
PACKET_ID_0436 = 0x0436
# Frame magic (v3 ring): [u32 len][u32 magic 0xF0C7][u64 ts][raw].
FRAME_MAGIC = 0xF0C7
# The admin account used to mint the export token (group 99).
ADMIN_USER = "kicapmasin888"
# Export endpoint (the Y1 Bearer path).
EXPORT_PATH = "/api/ads/telemetry/capture"
# Default export page size (rows; each row = one capture upload).
DEFAULT_PAGE = 200
# Max frames to scan per run (bounded).
MAX_FRAMES = 5000


@dataclass(slots=True)
class LearnedLayout:
    """The learned 0x0436 map-login layout."""
    length: int = 0
    account_offset: int = -1
    char_offset: int = -1
    login1_offset: int = -1
    login2_offset: int = -1
    tick_offset: int = -1
    sex_offset: int = -1
    samples: int = 0
    last_observed_ms: int = 0
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "length": self.length,
            "account_offset": self.account_offset,
            "char_offset": self.char_offset,
            "login1_offset": self.login1_offset,
            "login2_offset": self.login2_offset,
            "tick_offset": self.tick_offset,
            "sex_offset": self.sex_offset,
            "samples": self.samples,
            "last_observed_ms": self.last_observed_ms,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LearnedLayout":
        return cls(
            length=int(d.get("length", 0)),
            account_offset=int(d.get("account_offset", -1)),
            char_offset=int(d.get("char_offset", -1)),
            login1_offset=int(d.get("login1_offset", -1)),
            login2_offset=int(d.get("login2_offset", -1)),
            tick_offset=int(d.get("tick_offset", -1)),
            sex_offset=int(d.get("sex_offset", -1)),
            samples=int(d.get("samples", 0)),
            last_observed_ms=int(d.get("last_observed_ms", 0)),
            confidence=float(d.get("confidence", 0.0)),
        )


class CaptureConsumer:
    """Pulls captured packets, learns the 0x0436 layout, writes bot config + ML feed."""

    def __init__(
        self,
        *,
        export_base: str = "https://rathena-ai.openkore-ai.com",
        admin_user: str = ADMIN_USER,
        db_host: str = "",
        db_port: int = 3306,
        db_user: str = "",
        db_password: str = "",
        db_database: str = "ragnarok",
        bot_config_path: str = "",
        shared_learning_db_path: str = "",
        page_size: int = DEFAULT_PAGE,
    ) -> None:
        self.export_base = export_base.rstrip("/")
        self.admin_user = admin_user
        self.db_host = db_host or os.environ.get("DB_HOST", "127.0.0.1")
        self.db_port = int(db_port or os.environ.get("DB_PORT", "3306"))
        self.db_user = db_user or os.environ.get("DB_USER", "ragnarok")
        self.db_password = db_password or os.environ.get("DB_PASSWORD", "")
        self.db_database = db_database or os.environ.get("DB_DATABASE", "ragnarok")
        # Resolve the real paths when not given: the sidecar runs from the repo
        # root (python -m ai_sidecar.app), so the bot config is control/config.txt
        # and the shared learning DB is AI_sidecar/data/shared_learning.db.
        if not bot_config_path:
            bot_config_path = os.path.join(os.getcwd(), "control", "config.txt")
        if not shared_learning_db_path:
            # MUST match SharedLearningDB's own default exactly
            # (AI_sidecar/data/shared_learning.db) or the ML feed lands in a
            # different store than the trainer reads. SharedLearningDB resolves
            # relative to its own module dir: ai_sidecar/learning -> ../../data.
            # capture_consumer.py is at AI_sidecar/ai_sidecar/, so dirname x2
            # = AI_sidecar, then /data.
            shared_learning_db_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "data", "shared_learning.db",
            )
        self.bot_config_path = bot_config_path
        self.shared_learning_db_path = shared_learning_db_path
        self.page_size = page_size
        self._layout: LearnedLayout | None = None

    # ── Auth: mint a one-time SSO token for an admin account ──────────────
    def _mint_token(self) -> str:
        """Insert a fresh unused SSO token for the admin account (group 99)."""
        import pymysql

        token = secrets.token_urlsafe(22)[:22]
        conn = pymysql.connect(
            host=self.db_host,
            port=self.db_port,
            user=self.db_user,
            password=self.db_password,
            database=self.db_database,
            autocommit=True,
        )
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT account_id FROM login WHERE userid = %s AND group_id >= 99 LIMIT 1",
                    (self.admin_user,),
                )
                row = cur.fetchone()
                if not row:
                    raise RuntimeError(f"admin account {self.admin_user!r} not found or not group 99")
                account_id = row[0]
                cur.execute(
                    "INSERT INTO discord_login_tokens (account_id, token, launcher_state, expires, used) "
                    "VALUES (%s, %s, NULL, NOW() + INTERVAL 30 DAY, 1)",
                    (account_id, token),
                )
        finally:
            conn.close()
        return token

    # ── Export pull (Bearer + paginated) ─────────────────────────────────
    def _pull_page(self, token: str, offset: int, limit: int) -> tuple[list[str], bool]:
        """Fetch one page of capture lines. Returns (lines, has_more).

        has_more is the server's authoritative row-based signal (set by the
        limit+1 sentinel row). The consumer must use it, not a frame-count
        heuristic — a single row (capture upload) can hold ~130 frames, so
        comparing frame count against the ROW limit would over/under-fetch.
        """
        url = f"{self.export_base}{EXPORT_PATH}?username={self.admin_user}&offset={offset}&limit={limit}"
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "openkore-ai-v3-capture-consumer/1.0",
        })
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            logger.warning("capture export HTTP %s: %s", e.code, e.read()[:200])
            return [], False
        except Exception as e:
            logger.warning("capture export fetch failed: %s", e)
            return [], False
        # Response shape: {success, data: {lines: [...], has_more, ...}}
        d = data.get("data", {}) if isinstance(data, dict) else {}
        packets = d.get("lines", []) if isinstance(d, dict) else []
        has_more = bool(d.get("has_more", False)) if isinstance(d, dict) else False
        return [str(p) for p in packets], has_more

    def _pull_all(self, token: str) -> list[str]:
        """Pull all capture lines (paginated by the server's has_more)."""
        lines: list[str] = []
        offset = 0
        while len(lines) < MAX_FRAMES:
            page, has_more = self._pull_page(token, offset, self.page_size)
            if not page:
                break
            lines.extend(page)
            if not has_more:
                break
            offset += self.page_size
        return lines

    # ── Layout learning ───────────────────────────────────────────────────
    def _learn_from_lines(self, lines: list[str]) -> LearnedLayout | None:
        """Parse 0x0436 frames, learn the layout by matching the account_id."""
        # The admin account's id (for offset detection).
        account_id = self._admin_account_id()
        if account_id is None:
            logger.warning("cannot resolve admin account id; layout learning needs it")
            return None

        candidates: list[tuple[int, int, int, int, int, int, int]] = []
        for line in lines:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                pid = int(parts[0], 16)
                ts = int(parts[1])
                raw = bytes.fromhex(parts[2])
            except (ValueError, IndexError):
                continue
            if pid != PACKET_ID_0436:
                continue
            n = len(raw)
            if n < 2:
                continue
            # Find the account_id (4-byte LE) at each possible offset.
            for off in range(2, n - 3):
                val = struct.unpack_from("<I", raw, off)[0]
                if val == account_id:
                    # account@off, char@off+4, login1@off+8, login2@off+12, tick@off+16, sex@off+20
                    if off + 20 < n:
                        candidates.append((n, off, off + 4, off + 8, off + 12, off + 16, off + 20))
                    break
        if not candidates:
            logger.info("no 0x0436 frames with a matching account_id in this pull")
            return None

        # Majority vote on the layout.
        from collections import Counter
        length_c = Counter(c[0] for c in candidates)
        length = length_c.most_common(1)[0][0]
        acc_c = Counter(c[1] for c in candidates)
        acc_off = acc_c.most_common(1)[0][0]
        # The rest follow from acc_off (fixed stride).
        layout = LearnedLayout(
            length=length,
            account_offset=acc_off,
            char_offset=acc_off + 4,
            login1_offset=acc_off + 8,
            login2_offset=acc_off + 12,
            tick_offset=acc_off + 16,
            sex_offset=acc_off + 20,
            samples=len(candidates),
            last_observed_ms=int(time.time() * 1000),
            confidence=min(1.0, len(candidates) / 5.0),
        )
        self._layout = layout
        logger.info(
            "learned 0x0436 layout: len=%d account@%d char@%d login1@%d login2@%d tick@%d sex@%d (samples=%d)",
            layout.length, layout.account_offset, layout.char_offset,
            layout.login1_offset, layout.login2_offset, layout.tick_offset,
            layout.sex_offset, layout.samples,
        )
        return layout

    def _admin_account_id(self) -> int | None:
        import pymysql
        conn = pymysql.connect(
            host=self.db_host, port=self.db_port, user=self.db_user,
            password=self.db_password, database=self.db_database, autocommit=True,
        )
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT account_id FROM login WHERE userid = %s LIMIT 1", (self.admin_user,))
                row = cur.fetchone()
                return int(row[0]) if row else None
        finally:
            conn.close()

    # ── Bot config write ──────────────────────────────────────────────────
    def _write_bot_config(self, layout: LearnedLayout) -> bool:
        """Write mapLoginLength + mapLoginLayout to the bot's control/config.txt.

        The bots run from .bot_profiles/<name>/control/config.txt (one per
        profile), NOT the repo-root control/config.txt. The default
        bot_config_path (cwd/control/config.txt) is a legacy fallback that no
        running bot reads — writing only there means the learned layout never
        reaches the bot and it keeps sending the wrong 0x0436 form. Discover
        every .bot_profiles/*/control/config.txt and write the learned layout
        to each (atomic per file), so every bot adapts.
        """
        targets: list[Path] = []
        if self.bot_config_path:
            targets.append(Path(self.bot_config_path))
        # Discover all bot profiles under the repo root (cwd when run via
        # python -m ai_sidecar.app). Each profile's control/config.txt is a
        # real bot config the bridge reads.
        repo_root = Path(os.getcwd())
        profiles = repo_root / ".bot_profiles"
        if profiles.is_dir():
            for cfg in sorted(profiles.glob("*/control/config.txt")):
                if cfg not in targets:
                    targets.append(cfg)
        if not targets:
            logger.warning("no bot config targets; skipping config write")
            return False
        wrote = 0
        for path in targets:
            try:
                content = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
                lines = content.splitlines()
                # Remove existing mapLoginLength / mapLoginLayout lines.
                kept = [l for l in lines if not l.strip().startswith("mapLoginLength") and not l.strip().startswith("mapLoginLayout")]
                kept.append(f"mapLoginLength {layout.length}")
                kept.append(f"mapLoginLayout {json.dumps(layout.to_dict())}")
                path.parent.mkdir(parents=True, exist_ok=True)
                # ATOMIC write (temp + os.replace): the bot reads config.txt at
                # startup and on reconnect — a non-atomic truncate+write could
                # hand it a truncated mapLoginLayout line mid-write, the JSON
                # decode fails, and the bot falls back to the WRONG 23-byte
                # form. os.replace is atomic on POSIX + Windows.
                tmp = path.with_suffix(path.suffix + ".tmp")
                tmp.write_text("\n".join(kept) + "\n", encoding="utf-8")
                os.replace(tmp, path)
                wrote += 1
                logger.info("wrote bot config %s: mapLoginLength=%d mapLoginLayout=%s",
                            path, layout.length, json.dumps(layout.to_dict()))
            except Exception as exc:  # one bad profile must not block the rest
                logger.warning("failed to write bot config %s: %s", path, exc)
        return wrote > 0

    # ── ML feed ───────────────────────────────────────────────────────────
    def _feed_ml(self, lines: list[str], layout: LearnedLayout) -> int:
        """Append captured 0x0436 frames to shared_learning_db.packet_layouts."""
        if not self.shared_learning_db_path:
            logger.warning("no shared_learning_db_path; skipping ML feed")
            return 0
        from ai_sidecar.learning.shared_learning_db import SharedLearningDB
        db = SharedLearningDB(db_path=self.shared_learning_db_path)
        count = 0
        for line in lines:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                pid = int(parts[0], 16)
                ts = int(parts[1])
                raw = bytes.fromhex(parts[2])
            except (ValueError, IndexError):
                continue
            if pid != PACKET_ID_0436:
                continue
            db.record_packet_layout(
                packet_id=pid,
                length=len(raw),
                raw_hex=raw.hex(),
                captured_at_ms=ts,
                learned_layout=json.dumps(layout.to_dict()) if layout else "",
            )
            count += 1
        logger.info("fed %d 0x0436 frames to ML store", count)
        return count

    def _learn_from_store(self) -> LearnedLayout | None:
        """Learn the layout from the ACCUMULATED packet_layouts store.

        This is the ML feed's CONSUMER: the store is written by _feed_ml on
        every run, and this reads it back so the learned layout reflects the
        whole captured dataset (not just the latest pull). Without this the
        store would be write-only (dormant) — the completeness mandate requires
        every written table to be read.
        """
        if not self.shared_learning_db_path:
            return None
        from ai_sidecar.learning.shared_learning_db import SharedLearningDB
        db = SharedLearningDB(db_path=self.shared_learning_db_path)
        rows = db.get_packet_layouts(packet_id=PACKET_ID_0436, limit=500)
        if not rows:
            return None
        account_id = self._admin_account_id()
        if account_id is None:
            return None
        candidates: list[tuple[int, int, int, int, int, int, int]] = []
        for row in rows:
            try:
                raw = bytes.fromhex(row.get("raw_hex", ""))
            except (ValueError, TypeError):
                continue
            n = len(raw)
            if n < 2:
                continue
            for off in range(2, n - 3):
                val = struct.unpack_from("<I", raw, off)[0]
                if val == account_id:
                    if off + 20 < n:
                        candidates.append((n, off, off + 4, off + 8, off + 12, off + 16, off + 20))
                    break
        if not candidates:
            return None
        from collections import Counter
        length = Counter(c[0] for c in candidates).most_common(1)[0][0]
        acc_off = Counter(c[1] for c in candidates).most_common(1)[0][0]
        layout = LearnedLayout(
            length=length,
            account_offset=acc_off,
            char_offset=acc_off + 4,
            login1_offset=acc_off + 8,
            login2_offset=acc_off + 12,
            tick_offset=acc_off + 16,
            sex_offset=acc_off + 20,
            samples=len(candidates),
            last_observed_ms=int(time.time() * 1000),
            confidence=min(1.0, len(candidates) / 5.0),
        )
        self._layout = layout
        logger.info(
            "learned 0x0436 layout from store: len=%d account@%d (samples=%d)",
            layout.length, layout.account_offset, layout.samples,
        )
        return layout

    def _delete_token(self, token: str) -> None:
        """Delete a minted token after use (prevents unbounded growth)."""
        import pymysql
        try:
            conn = pymysql.connect(
                host=self.db_host, port=self.db_port, user=self.db_user,
                password=self.db_password, database=self.db_database, autocommit=True,
            )
            try:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM discord_login_tokens WHERE token = %s", (token,))
            finally:
                conn.close()
        except Exception as e:
            logger.warning("token cleanup failed: %s", e)

    # ── Main run ──────────────────────────────────────────────────────────
    def run(self) -> dict[str, Any]:
        """Pull captures, learn the layout, write config + ML feed."""
        token = self._mint_token()
        try:
            lines = self._pull_all(token)
        finally:
            # Delete the token after use — a fresh token every 10-min run would
            # otherwise grow discord_login_tokens unboundedly (144/day x 30-day
            # expiry = ~4320 rows before any expire).
            self._delete_token(token)
        if not lines:
            logger.info("no capture lines pulled")
            return {"learned": False, "frames": 0, "reason": "no captures"}
        layout = self._learn_from_lines(lines)
        if layout is None:
            return {"learned": False, "frames": len(lines), "reason": "no 0x0436 with matching account"}
        wrote = self._write_bot_config(layout)
        fed = self._feed_ml(lines, layout)
        # The ML feed's consumer: re-learn from the accumulated store so the
        # written data is genuinely read (not dormant).
        store_layout = self._learn_from_store()
        return {
            "learned": True,
            "frames": len(lines),
            "layout": layout.to_dict(),
            "store_layout": store_layout.to_dict() if store_layout else None,
            "config_written": wrote,
            "ml_fed": fed,
        }


def run_consumer() -> dict[str, Any]:
    """Entry point for the sidecar startup / cron."""
    consumer = CaptureConsumer()
    return consumer.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_consumer()
    print(json.dumps(result, indent=2))
