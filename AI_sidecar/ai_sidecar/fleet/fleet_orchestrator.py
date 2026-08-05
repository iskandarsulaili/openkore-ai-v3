"""Separate Fleet Orchestrator (user-directed, re-tier).

The fleet has a stateful FleetCoordinator (roles/orders/threats) but NO dedicated
entity that *decides* fleet-wide intent on a coarse cadence. Fleet coordination was
historically tangled into the per-bot cycle (SwarmCoordinator.tick ran inside every
bot's loop), which polluted a farming bot with swarm-admin actions — the same anti-
pattern as before the human-cognition re-tier.

This module is the TRUE SEPARATE ORCHESTRATOR:
  - RUNS ON A COARSE CADENCE (every ~15s), fully decoupled from per-bot cycles.
  - CONSCIOUS-ONLY: it issues high-level INTENT directives to bots; it does NOT
    micro-manage per-cycle skilled action (that belongs to each bot's subconscious/
    reflex). It reads the whole fleet and only intervening when a bot is clearly
    stuck, off-role, or idle.
  - DIRECTIVE-LEVEL: emits e.g. "reassign bot to farm", never per-cycle commands.

This mirrors a human fleet captain: slow, whole-field, intent-bearing — while the
individual players execute their trained roles.
"""

from __future__ import annotations

import logging
import time
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)

# A farm map the fleet drives bots toward (server-agnostic default; real farm comes
# from the per-bot heuristic / server_solutions store, not hardcoded here).
_DEFAULT_TARGET = "prt_fild08c"
_DEFAULT_TICK_SECONDS = 15.0


class FleetOrchestrator:
    """Coarse-cadence, conscious-only, fleet-wide directive entity.

    It does NOT run per-bot. The host calls `tick(all_bot_states)` on a slow cadence.
    """

    def __init__(
        self,
        *,
        tick_seconds: float = _DEFAULT_TICK_SECONDS,
        coordinator: Any | None = None,
        enqueue_fn: Callable | None = None,
    ) -> None:
        self._lock = RLock()
        self._tick_seconds = float(tick_seconds)
        self._last_tick = float("-inf")  # first tick runs immediately (then coarse cadence)
        self._coordinator = coordinator
        self._enqueue_fn = enqueue_fn  # (bot_id, command_str) -> enqueues an order
        self._stats: dict[str, int] = {"ticks": 0, "orders": 0, "directives": 0}

    @staticmethod
    def _get(st: Any, key: str, default: Any = None) -> Any:
        """Read a field from a dict-like or an object-like snapshot."""
        if isinstance(st, dict):
            return st.get(key, default)
        return getattr(st, key, default)

    def _is_farm_ready(self, st: Any) -> bool:
        """True if a bot is in_game, on a field map, and not dead.

        Conscious-level judgment only — no per-cycle control.
        """
        raw = self._get(st, "raw", {}) or {}
        if not isinstance(raw, dict):
            raw = {}
        in_game = bool(raw.get("in_game", False))
        map_name = str(self._get(st, "map", "") or "").replace(".gat", "")
        is_field = any(tok in map_name for tok in (
            "prt_fild", "pay_fild", "mjolnir", "gef_fild", "ra_fild",
            "moc_fild", "cmd_fild", "iz_ac", "alu_fild", "ein_fild",
        ))
        respawn = str(raw.get("respawn_state", "") or "")
        dead = respawn == "dead"
        return bool(in_game and is_field and not dead)

    def _needs_directive(self, st: Any) -> str | None:
        """Conscious-level scan: does this bot need a high-level nudge?

        Returns a directive reason string, or None if the bot is fine (leave it alone).
        """
        raw = self._get(st, "raw", {}) or {}
        if not isinstance(raw, dict):
            raw = {}
        in_game = bool(raw.get("in_game", False))
        map_name = str(self._get(st, "map", "") or "").replace(".gat", "")
        respawn = str(raw.get("respawn_state", "") or "")
        dead = respawn == "dead"
        level = int(self._get(st, "base_level", 0) or 0)
        # A bot that is in-game but NOT on a field map and NOT in an academy
        # tutorial map, and not dead, is likely a non-farming bot drifting in a town
        # or stuck — direct it toward a farm (as INTENT, the per-bot brain routes it).
        if in_game and not dead and level >= 1 and map_name and \
           not self._is_farm_ready(st) and map_name not in ("iz_ac01_a", "iz_ac01"):
            return f"bot_not_on_farm_map:{map_name}"
        if dead:
            return "bot_dead"
        return None

    def tick(self, all_bot_states: dict[str, Any]) -> list[dict[str, Any]]:
        """Coarse-cadence orchestrator pass. Returns list of directive dicts issued.

        all_bot_states: bot_id -> state-snapshot dict. Runs independent of per-bot
        execution. Emits INTENT only via the coordinator/enqueue_fn when a bot clearly
        needs a nudge — never per-cycle spam.
        """
        with self._lock:
            now = time.time()
            if now - self._last_tick < self._tick_seconds:
                return []
            self._last_tick = now
            self._stats["ticks"] += 1
            directives: list[dict[str, Any]] = []
            for bot_id, st in (all_bot_states or {}).items():
                # Accept dict-like OR object-like snapshots (the live snapshot_cache
                # returns BotStateSnapshot objects — skipping objects would make the
                # orchestrator never actually fire on live states).
                if not (isinstance(st, dict) or hasattr(st, "__dict__")):
                    continue
                reason = self._needs_directive(st)
                if reason is None:
                    continue
                # Intent-level directive: from fleet registry (if any) look up the
                # bot's role to decide the right farm. Fall back to the default.
                target_map = _DEFAULT_TARGET
                if self._coordinator is not None:
                    try:
                        farmer = self._coordinator.get_farmer()
                        if farmer is not None and getattr(farmer, "map", ""):
                            target_map = farmer.map
                    except Exception:
                        target_map = _DEFAULT_TARGET
                cmd = f"move {target_map}"
                if reason == "bot_dead":
                    cmd = "ai auto"  # respawn intent
                # Issue via coordinator if present, else the enqueue fallback.
                issued = False
                if self._coordinator is not None and hasattr(self._coordinator, "issue_order"):
                    try:
                        issued = bool(self._coordinator.issue_order(
                            bot_id, cmd, f"[FleetOrchestrator] {reason}",
                            priority=3, ttl_seconds=30,
                        ))
                    except Exception as e:
                        logger.debug("fleet_orchestrator_order_failed: %s", e)
                        issued = False
                if not issued and self._enqueue_fn is not None:
                    try:
                        self._enqueue_fn(bot_id, cmd)
                        issued = True
                    except Exception as e:
                        logger.debug("fleet_orchestrator_enqueue_failed: %s", e)
                if issued:
                    directives.append({"bot_id": bot_id, "command": cmd, "reason": reason})
                    self._stats["orders"] += 1
                    self._stats["directives"] += 1
                    logger.info("fleet_orchestrator_directive bot=%s cmd=%s reason=%s",
                                bot_id, cmd, reason)
            return directives

    def set_cadence(self, seconds: float) -> None:
        with self._lock:
            self._tick_seconds = float(seconds)

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


def get_fleet_orchestrator(
    *, coordinator: Any | None = None, enqueue_fn: Callable | None = None
) -> FleetOrchestrator:
    """Provide a fleet orchestrator instance (optionally bound to the fleet registry)."""
    return FleetOrchestrator(coordinator=coordinator, enqueue_fn=enqueue_fn)
