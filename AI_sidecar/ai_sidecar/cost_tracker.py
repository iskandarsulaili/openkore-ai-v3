from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CostSnapshot:
    daily_tokens_used: int = 0
    daily_calls_made: int = 0
    hourly_calls_made: int = 0
    monthly_cost_usd: float = 0.0
    tier: str = "standard"
    budget_exhausted: bool = False
    budget_reason: str = ""


class CostTracker:
    """Tracks LLM usage per-bot and enforces budget limits.

    Supports two modes controlled by per_bot_budget:
      - per_bot_budget=True:  each bot has its own daily budget
      - per_bot_budget=False: all bots share one daily budget (default)

    State is in-memory with optional SQLite persistence.
    The PDCA loop and model router check this before making LLM calls.
    """

    def __init__(self, per_bot_budget: bool = False):
        self._lock = RLock()
        self._per_bot = per_bot_budget
        # Fleet-wide totals, always accumulated regardless of per_bot mode. This is the
        # true fleet gate: per_bot_budget=True bounds each bot, but N bots × per-bot
        # budget would otherwise be unbounded at the fleet level. check() consults
        # fleet totals + the per-bot totals so a single shared cap applies fleet-wide.
        self._fleet_daily_tokens = 0
        self._fleet_daily_calls = 0
        self._fleet_hourly_calls: list[float] = []
        # Shared counters (used when per_bot_budget=False)
        self._daily_tokens = 0
        self._daily_calls = 0
        self._hourly_calls: list[float] = []
        self._monthly_cost = 0.0
        self._day_start = int(time.time())
        # Per-bot counters (used when per_bot_budget=True)
        self._bot_tokens: dict[str, int] = {}
        self._bot_calls: dict[str, int] = {}
        self._bot_hourly: dict[str, list[float]] = {}
        self._bot_monthly_cost: dict[str, float] = {}

    def _get_bot_state(self, bot_id: str) -> dict[str, Any]:
        """Get or create per-bot counters."""
        if not self._per_bot:
            return {}  # Use shared counters
        if bot_id not in self._bot_tokens:
            self._bot_tokens[bot_id] = 0
            self._bot_calls[bot_id] = 0
            self._bot_hourly[bot_id] = []
            self._bot_monthly_cost[bot_id] = 0.0
        return {
            "tokens": self._bot_tokens[bot_id],
            "calls": self._bot_calls[bot_id],
            "hourly": self._bot_hourly[bot_id],
            "cost": self._bot_monthly_cost[bot_id],
        }

    def record_call(self, tokens: int, *, model: str, tier: str, bot_id: str = "default") -> None:
        """Record an LLM call and its token usage for a specific bot."""
        with self._lock:
            self._check_day_rollover()
            # Fleet-wide totals always accumulate (the true fleet gate).
            self._fleet_daily_tokens += tokens
            self._fleet_daily_calls += 1
            self._fleet_hourly_calls.append(time.time())
            cutoff = time.time() - 3600
            self._fleet_hourly_calls = [t for t in self._fleet_hourly_calls if t > cutoff]
            if self._per_bot:
                self._bot_tokens[bot_id] = self._bot_tokens.get(bot_id, 0) + tokens
                self._bot_calls[bot_id] = self._bot_calls.get(bot_id, 0) + 1
                hourly = self._bot_hourly.setdefault(bot_id, [])
                hourly.append(time.time())
                cutoff = time.time() - 3600
                self._bot_hourly[bot_id] = [t for t in hourly if t > cutoff]
                pricing = {"off": 0.0, "economy": 0.15, "standard": 0.30, "premium": 0.60}
                price_per_m = pricing.get(tier, 0.30)
                self._bot_monthly_cost[bot_id] = self._bot_monthly_cost.get(bot_id, 0.0) + (tokens / 1_000_000) * price_per_m
            else:
                self._daily_tokens += tokens
                self._daily_calls += 1
                self._hourly_calls.append(time.time())
                cutoff = time.time() - 3600
                self._hourly_calls = [t for t in self._hourly_calls if t > cutoff]
                pricing = {"off": 0.0, "economy": 0.15, "standard": 0.30, "premium": 0.60}
                price_per_m = pricing.get(tier, 0.30)
                self._monthly_cost += (tokens / 1_000_000) * price_per_m

    def check(self, *, daily_budget_tokens: int, max_calls_per_hour: int, tier: str, bot_id: str = "default") -> tuple[bool, str]:
        """Returns (allowed: bool, reason: str) for a specific bot or globally.

        Enforces BOTH the fleet-wide cap (always) AND the per-bot cap (when
        per_bot_budget=True). Fleet-first: a shared daily/hourly budget applies across
        all bots so N bots cannot collectively exceed it.
        """
        with self._lock:
            self._check_day_rollover()
            if tier == "off":
                return False, "cost_tier_off"

            # Fleet-wide gate (always).
            if daily_budget_tokens > 0 and self._fleet_daily_tokens >= daily_budget_tokens:
                return False, f"fleet_daily_token_budget_exceeded:{self._fleet_daily_tokens}/{daily_budget_tokens}"
            if max_calls_per_hour > 0:
                cutoff = time.time() - 3600
                fleet_recent = sum(1 for t in self._fleet_hourly_calls if t > cutoff)
                if fleet_recent >= max_calls_per_hour:
                    return False, f"fleet_hourly_call_limit_exceeded:{fleet_recent}/{max_calls_per_hour}"

            # Per-bot gate (when enabled).
            if self._per_bot:
                tokens = self._bot_tokens.get(bot_id, 0)
                calls = self._bot_calls.get(bot_id, 0)
                hourly = self._bot_hourly.get(bot_id, [])
            else:
                tokens = self._daily_tokens
                calls = self._daily_calls
                hourly = self._hourly_calls

            if daily_budget_tokens > 0 and tokens >= daily_budget_tokens:
                return False, f"daily_token_budget_exceeded:{tokens}/{daily_budget_tokens}"

            if max_calls_per_hour > 0:
                cutoff = time.time() - 3600
                recent = sum(1 for t in hourly if t > cutoff)
                if recent >= max_calls_per_hour:
                    return False, f"hourly_call_limit_exceeded:{recent}/{max_calls_per_hour}"

            return True, "ok"

    def snapshot(self, bot_id: str = "default") -> CostSnapshot:
        with self._lock:
            self._check_day_rollover()
            if self._per_bot:
                tokens = self._bot_tokens.get(bot_id, 0)
                calls = self._bot_calls.get(bot_id, 0)
                hourly = self._bot_hourly.get(bot_id, [])
                cost = self._bot_monthly_cost.get(bot_id, 0.0)
            else:
                tokens = self._daily_tokens
                calls = self._daily_calls
                hourly = self._hourly_calls
                cost = self._monthly_cost
            cutoff = time.time() - 3600
            return CostSnapshot(
                daily_tokens_used=tokens,
                daily_calls_made=calls,
                hourly_calls_made=sum(1 for t in hourly if t > cutoff),
                monthly_cost_usd=round(cost, 4),
                tier="standard",
                budget_exhausted=False,
                budget_reason="ok",
            )

    def _check_day_rollover(self) -> None:
        now = int(time.time())
        if now - self._day_start >= 86400:
            self._daily_tokens = 0
            self._daily_calls = 0
            self._fleet_daily_tokens = 0
            self._fleet_daily_calls = 0
            self._fleet_hourly_calls = []
            self._day_start = now
            if self._per_bot:
                self._bot_tokens.clear()
                self._bot_calls.clear()
                self._bot_hourly.clear()
                self._bot_monthly_cost.clear()
            logger.info("cost_tracker: daily budget reset")

    def persist(self, sqlite_path: str | None = None) -> None:
        """Persist current budget state to SQLite."""
        if not sqlite_path:
            return
        try:
            import sqlite3
            db = sqlite3.connect(sqlite_path, timeout=5.0)
            db.execute("CREATE TABLE IF NOT EXISTS cost_budget (bot_id TEXT, key TEXT, value INTEGER, updated_at REAL, PRIMARY KEY(bot_id, key))")
            if self._per_bot:
                for bot_id in self._bot_tokens:
                    db.execute("INSERT OR REPLACE INTO cost_budget (bot_id, key, value, updated_at) VALUES (?, ?, ?, ?)",
                               (bot_id, "daily_tokens", self._bot_tokens[bot_id], time.time()))
                    db.execute("INSERT OR REPLACE INTO cost_budget (bot_id, key, value, updated_at) VALUES (?, ?, ?, ?)",
                               (bot_id, "daily_calls", self._bot_calls[bot_id], time.time()))
            else:
                db.execute("INSERT OR REPLACE INTO cost_budget (bot_id, key, value, updated_at) VALUES (?, ?, ?, ?)",
                           ("_shared", "daily_tokens", self._daily_tokens, time.time()))
                db.execute("INSERT OR REPLACE INTO cost_budget (bot_id, key, value, updated_at) VALUES (?, ?, ?, ?)",
                           ("_shared", "daily_calls", self._daily_calls, time.time()))
            # Fleet totals persist so a restart does not silently reset the fleet budget.
            db.execute("INSERT OR REPLACE INTO cost_budget (bot_id, key, value, updated_at) VALUES (?, ?, ?, ?)",
                       ("_fleet", "daily_tokens", self._fleet_daily_tokens, time.time()))
            db.execute("INSERT OR REPLACE INTO cost_budget (bot_id, key, value, updated_at) VALUES (?, ?, ?, ?)",
                       ("_fleet", "daily_calls", self._fleet_daily_calls, time.time()))
            db.commit()
            db.close()
        except Exception:
            logger.warning("cost_tracker_persist_failed")

    def restore(self, sqlite_path: str | None = None) -> None:
        """Restore budget state from SQLite on startup."""
        if not sqlite_path:
            return
        try:
            import sqlite3
            db = sqlite3.connect(sqlite_path, timeout=5.0)
            cursor = db.execute("SELECT bot_id, key, value, updated_at FROM cost_budget")
            now = time.time()
            for bot_id, key, value, updated_at in cursor.fetchall():
                if now - updated_at < 86400:
                    if bot_id == "_fleet":
                        if key == "daily_tokens":
                            self._fleet_daily_tokens = int(value)
                        elif key == "daily_calls":
                            self._fleet_daily_calls = int(value)
                        self._day_start = int(updated_at)
                    elif bot_id == "_shared":
                        if key == "daily_tokens":
                            self._daily_tokens = int(value)
                        elif key == "daily_calls":
                            self._daily_calls = int(value)
                        self._day_start = int(updated_at)
                    elif self._per_bot:
                        if key == "daily_tokens":
                            self._bot_tokens[bot_id] = int(value)
                        elif key == "daily_calls":
                            self._bot_calls[bot_id] = int(value)
            db.close()
            if self._daily_tokens > 0 or self._bot_tokens:
                logger.info("cost_tracker: restored budget state")
        except Exception:
            logger.warning("cost_tracker_restore_failed")
