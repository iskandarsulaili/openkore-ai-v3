"""TaskCommandTranslator — maps TaskScheduler semantic commands to real commands.

The TaskScheduler emits high-level task *intents* (return_town, buy_pots,
party, guild, attack, ...) that have no 1:1 OpenKore command. This module
translates them into executable commands where a safe mapping exists, and into
observable log actions where the capability belongs to another wired subsystem:

  - attack / hunt / grind      → owned by attackAuto + the PDCA combat monitor
  - party / guild              → owned by the fleet coordinator (god_mode /
                                 joiner_check) and the bridge party gates
  - sell / store / buy_ammo    → owned by sellAuto / buyAuto config
  - repair / quest / npc_talk  → need NPC coordinates the scheduler lacks
  - use_emergency_heal         → owned by the reflex heal-order pipeline

Safe real mappings (gated on signals):
  - return_town  → 'move prontera'  (level >= 6, only while on a fild map)
  - buy_pots     → 'buy 501 30'     (only while in a town map)

Emission is throttled per (bot, task) so unconditionally-pushed tasks
(grind_levels, hunt_current_map) cannot spam the action stream.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)

# ── Capabilities owned by other wired subsystems → log-only (never emit) ──
_OBSERVED_ONLY_COMMANDS = frozenset({
    "attack",
    "party",          # fleet coordinator owns partying; bare 'party' spams
    "guild",
    "skill_auto",
    "sell_loot",
    "store_items",
    "buy_ammo",
    "open_storage",
    "deposit_items",
    "repair_gear",
    "quest_auto",
    "npc_talk",
    "use_emergency_heal",  # reflex heal-order pipeline owns emergency healing
})

# ── Per-task emission cooldowns (seconds) — prevents spam from tasks the
#    scheduler pushes unconditionally every cycle ──
_EMIT_COOLDOWN: dict[str, float] = {
    "grind_levels": 600.0,
    "hunt_current_map": 600.0,
    "skill_rotation": 300.0,
    "party_check": 300.0,
    "guild_check": 600.0,
    "complete_quest_objective": 300.0,
    "job_change_prep": 600.0,
    "restock_pots": 900.0,
    "repair_equipment": 3600.0,
    "sell_loot": 600.0,
    "store_items": 600.0,
    "buy_consumables": 600.0,
    "emergency_heal": 30.0,
}
_DEFAULT_COOLDOWN = 120.0


def _parent_city(map_name: str) -> str:
    """Derive a field's parent town from the RO prefix graph (game_data)."""
    try:
        from ai_sidecar.game_data import parent_town as _pt
        return _pt(map_name)
    except Exception:
        return ""


class TaskCommandTranslator:
    """Translate scheduler task intents into safe, real actions."""

    def __init__(self) -> None:
        # (bot_id, task_name) -> last emission epoch
        self._last_emit: dict[tuple[str, str], float] = {}
        self._counters: dict[str, int] = {
            "translated": 0,
            "observed_only": 0,
            "throttled": 0,
            "real_commands": 0,
        }

    # ── Public API ────────────────────────────────────────────────────

    def translate(
        self,
        task: Any,
        signals: dict[str, Any],
        bot_id: str = "default",
    ) -> list[HeuristicAction]:
        """Translate one ScheduledTask into HeuristicActions.

        Returns [] when throttled (task emitted within its cooldown window).
        """
        name = getattr(task, "name", "unknown")
        now = time.time()

        # Throttle: unconditionally-pushed tasks must not spam
        cooldown = _EMIT_COOLDOWN.get(name, _DEFAULT_COOLDOWN)
        key = (bot_id, name)
        last = self._last_emit.get(key, 0.0)
        if now - last < cooldown:
            self._counters["throttled"] += 1
            return []
        self._last_emit[key] = now

        actions: list[HeuristicAction] = []
        commands = getattr(task, "commands", []) or []
        description = getattr(task, "description", name)
        domain = "planning"

        if not commands:
            # No command payload → log the intent (observability contract)
            actions.append(HeuristicAction(
                kind="log",
                command=f"task:{name}",
                confidence=0.6,
                reason=f"scheduler: {description}",
                domain=domain,
                metadata={"task": name, "source": "TaskScheduler"},
            ))
            self._counters["translated"] += 1
            return actions

        for cmd in commands:
            real = self._real_command(cmd, signals)
            if real:
                actions.append(HeuristicAction(
                    kind="command",
                    command=real,
                    confidence=0.85,
                    reason=f"scheduler: {description}",
                    domain=domain,
                    metadata={"task": name, "source": "TaskScheduler", "intent": cmd},
                ))
                self._counters["real_commands"] += 1
            else:
                # Capability owned elsewhere → observable intent only
                actions.append(HeuristicAction(
                    kind="log",
                    command=f"task:{name}:{cmd}",
                    confidence=0.5,
                    reason=f"scheduler: {description} (owned by wired subsystem)",
                    domain=domain,
                    metadata={"task": name, "source": "TaskScheduler", "intent": cmd},
                ))
                self._counters["observed_only"] += 1
            self._counters["translated"] += 1

        return actions

    # ── Internals ─────────────────────────────────────────────────────

    def _real_command(self, cmd: str, signals: dict[str, Any]) -> str | None:
        """Map a semantic command to a real OpenKore command, or None."""
        cmd = (cmd or "").strip().lower()

        if cmd in _OBSERVED_ONLY_COMMANDS:
            return None

        if cmd == "return_town":
            level = int(signals.get("base_level", 1) or 1)
            cur_map = str(signals.get("map", "") or "").lower()
            # Academy bots (level <= 5) stay put — cold-start owns their routing
            if level < 6:
                return None
            # Only head to town when actually out on a field
            if "fild" not in cur_map:
                return None
            # Only a REAL need justifies abandoning the hunting spot: the
            # scheduler also pushes sell_loot/repair on plain timers, but a
            # timer is not a need — without inventory_full we'd march the bot
            # to town every 10 min for nothing.
            if not signals.get("inventory_full"):
                return None
            # RULE.md: town = the field's parent city, derived from the core's
            # tables/cities.txt (map prefix graph), never a hardcoded literal.
            _town = _parent_city(cur_map)
            return f"move {_town}" if _town else None

        if cmd == "buy_pots":
            cur_map = str(signals.get("map", "") or "").lower()
            # Only buy while in town (never mid-field)
            if not cur_map or "fild" in cur_map or "int_land" in cur_map:
                return None
            # RULE.md: generic potion form (OpenKore resolves the best heal
            # item from its tables) — never a hardcoded server item id.
            return "buy potion 30"

        # Unknown semantic command → observe, don't emit
        return None

    def counters(self) -> dict[str, int]:
        return dict(self._counters)
