"""AutoStatBuilder — Intelligent stat allocation for OpenKore bots.

Reads the bot's current stats and available stat points from snapshots,
then allocates points to the highest-priority stat based on the class
archetype. Uses breakpoint multiples of 10 before switching to the next
priority stat. Thread-safe with RLock.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta
from threading import RLock
from typing import Any

from ai_sidecar.contracts.actions import ActionPriorityTier, ActionProposal
from ai_sidecar.game_engine import CLASS_ARCHETYPES

logger = logging.getLogger(__name__)

# ── Stat names used in OpenKore commands ──
STAT_NAMES = ("str", "agi", "vit", "int", "dex", "luk")

# ── Breakpoint: every N points in a stat before considering the next ──
BREAKPOINT_INTERVAL = 10


class AutoStatBuilder:
    """Thread-safe automatic stat point allocation.

    Reads CLASS_ARCHETYPES from the game engine to determine stat priorities
    per class, then allocates available stat_points to the highest-priority
    stat that hasn't reached its next breakpoint.

    Tracks allocation state per bot in ``runtime_state.stat_allocation_state``.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        # Per-bot stat allocation tracking: bot_id -> dict with "str": <allocated so far>, ...
        self._allocation_state: dict[str, dict[str, int]] = {}

    # ── Public API ──────────────────────────────────────────────────────

    def evaluate(
        self,
        bot_id: str,
        snapshot: Any,  # BotStateSnapshot, dict, or raw payload
    ) -> list[ActionProposal] | None:
        """Evaluate the snapshot and return stat-add ActionProposals.

        Returns ``None`` when there are no stat points to allocate.
        Returns a list of ActionProposal objects when points can be spent.
        Each proposal is a single ``stats_add <stat> 1`` command.
        """
        now = datetime.now(UTC)

        # 1. Extract current class name from snapshot
        class_name = self._extract_class_name(snapshot)
        if not class_name:
            logger.debug("stat_builder[%s]: no class name in snapshot", bot_id)
            return None

        # 2. Extract current stats from snapshot
        current_stats = self._extract_stats(snapshot)
        if not current_stats:
            logger.debug("stat_builder[%s]: no stats in snapshot", bot_id)
            return None

        # 3. Extract available stat points
        stat_points = self._extract_stat_points(snapshot)
        if not stat_points or stat_points <= 0:
            return None

        logger.info(
            "stat_builder[%s]: class=%s stats=%s points=%d",
            bot_id, class_name, current_stats, stat_points,
        )

        # 4. Get stat priority for this class
        priority = self._get_stat_priority(class_name)
        if not priority:
            logger.warning("stat_builder[%s]: no stat priority for class %s", bot_id, class_name)
            return None

        # 5. Get or initialise allocation state
        allocated = self._get_or_create_state(bot_id)
        if not allocated:
            allocated = {s: 0 for s in STAT_NAMES}
            with self._lock:
                self._allocation_state[bot_id] = allocated

        # 6. Build proposals — one point per point available
        proposals: list[ActionProposal] = []
        points_to_spend = min(stat_points, 50)  # Cap per evaluation to avoid flooding queue

        for _ in range(points_to_spend):
            target_stat = self._pick_stat(priority, current_stats, allocated)
            if target_stat is None:
                # All stats at breakpoints — allocate to highest priority regardless
                target_stat = priority[0]

            # Build a single-point command
            command = f"stats_add {target_stat} 1"
            action_id = f"stat_{target_stat}_{uuid.uuid4().hex[:8]}"

            proposal = ActionProposal(
                action_id=action_id,
                kind="command",
                command=command,
                priority_tier=ActionPriorityTier.strategic,
                source="planner",
                created_at=now,
                expires_at=now + timedelta(seconds=30),
                idempotency_key=f"stat-{bot_id}-{target_stat}-{uuid.uuid4().hex[:12]}",
                metadata={
                    "bot_id": bot_id,
                    "class": class_name,
                    "stat": target_stat,
                    "current_value": current_stats.get(target_stat, 0),
                    "allocated_so_far": allocated.get(target_stat, 0),
                    "stat_points_remaining": stat_points - len(proposals) - 1,
                },
            )
            proposals.append(proposal)

            # Track our allocation
            allocated[target_stat] = allocated.get(target_stat, 0) + 1

        with self._lock:
            self._allocation_state[bot_id] = allocated

        logger.info(
            "stat_builder[%s]: generated %d stat-add proposals for %s",
            bot_id, len(proposals), class_name,
        )
        return proposals if proposals else None

    def get_recommended_stats(
        self,
        bot_id: str,
        class_name: str,
    ) -> dict[str, int]:
        """Return the recommended target stat distribution for a class.

        Based on the class archetype's stat priority and standard
        Ragnarok Online build conventions.
        """
        class_key = class_name.lower().strip()
        archetype = CLASS_ARCHETYPES.get(class_key)
        if not archetype:
            logger.warning("stat_builder[%s]: unknown class '%s', using defaults", bot_id, class_name)
            return self._default_stats()

        priority = archetype.get("stat_priority", [])
        if not priority:
            return self._default_stats()

        # Standard RO build: primary stat ~99, secondary ~60-80, tertiary ~30-50
        targets: dict[str, int] = {}
        for idx, stat in enumerate(priority):
            if idx == 0:
                targets[stat] = 99
            elif idx == 1:
                targets[stat] = 60
            elif idx == 2:
                targets[stat] = 40
            else:
                targets[stat] = 20
        # Fill in stats not in priority
        for stat in STAT_NAMES:
            if stat not in targets:
                targets[stat] = 1

        return targets

    # ── Internal helpers ────────────────────────────────────────────────

    def _pick_stat(
        self,
        priority: list[str],
        current_stats: dict[str, int],
        allocated: dict[str, int],
    ) -> str | None:
        """Pick the best stat to allocate to.

        Checks each priority stat in order. If a stat's current level
        (including our pending allocation) is not at a breakpoint
        (multiple of BREAKPOINT_INTERVAL), picks that stat.
        Returns ``None`` if all priority stats are at breakpoints.
        """
        for stat in priority:
            base = current_stats.get(stat, 0)
            pending = allocated.get(stat, 0)
            effective = base + pending
            if effective % BREAKPOINT_INTERVAL != 0 or effective == 0:
                return stat
        return None

    def _get_stat_priority(self, class_name: str) -> list[str]:
        """Get the stat priority list for a class from CLASS_ARCHETYPES."""
        class_key = class_name.lower().strip()
        archetype = CLASS_ARCHETYPES.get(class_key)
        if archetype:
            return archetype.get("stat_priority", ["str", "dex", "agi"])
        logger.warning("stat_builder: no archetype for '%s'", class_name)
        return ["str", "dex", "agi"]

    def _extract_class_name(self, snapshot: Any) -> str | None:
        """Extract the job/class name from a snapshot."""
        if snapshot is None:
            return None

        # Dict-like access
        if isinstance(snapshot, dict):
            # Try progression block
            prog = snapshot.get("progression") or {}
            if isinstance(prog, dict):
                name = prog.get("job_name") or prog.get("class_name")
                if name:
                    return str(name).lower()
            # Try raw top-level class
            raw = snapshot.get("raw") or {}
            if isinstance(raw, dict):
                name = raw.get("job_name") or raw.get("class_name")
                if name:
                    return str(name).lower()
            # Top-level fallback
            return (
                str(snapshot.get("job_name", "")).lower()
                or str(snapshot.get("class_name", "")).lower()
                or None
            )

        # Object access
        prog = getattr(snapshot, "progression", None)
        if prog:
            name = getattr(prog, "job_name", None) or getattr(prog, "class_name", None)
            if name:
                return str(name).lower()
        raw = getattr(snapshot, "raw", None)
        if raw and isinstance(raw, dict):
            name = raw.get("job_name") or raw.get("class_name")
            if name:
                return str(name).lower()
        return (
            str(getattr(snapshot, "job_name", "")).lower()
            or str(getattr(snapshot, "class_name", "")).lower()
            or None
        )

    def _extract_stats(self, snapshot: Any) -> dict[str, int]:
        """Extract current stat values from a snapshot.

        Handles both dict-like and object snapshots. Looks in
        ``raw`` dict, top-level fields, and nested ``progression``.
        """
        stats: dict[str, int] = {}

        if snapshot is None:
            return stats

        raw: dict[str, Any] = {}

        if isinstance(snapshot, dict):
            # Primary: ``raw`` sub-dict (most common bridge format)
            raw_raw = snapshot.get("raw") or {}
            if isinstance(raw_raw, dict):
                raw.update(raw_raw)
            # Secondary: top-level dict
            for k in STAT_NAMES:
                v = snapshot.get(k, None)
                if v is not None:
                    try:
                        raw[k] = int(v)
                    except (ValueError, TypeError):
                        pass
            # Tertiary: nested "stats" dict
            nested_stats = snapshot.get("stats") or {}
            if isinstance(nested_stats, dict):
                for k, v in nested_stats.items():
                    k_lower = k.lower()
                    if k_lower in STAT_NAMES:
                        try:
                            raw[k_lower] = int(v)
                        except (ValueError, TypeError):
                            pass
        else:
            # Object access
            raw_obj = getattr(snapshot, "raw", None)
            if raw_obj and isinstance(raw_obj, dict):
                raw.update(raw_obj)
            for k in STAT_NAMES:
                v = getattr(snapshot, k, None)
                if v is not None:
                    try:
                        raw[k] = int(v)
                    except (ValueError, TypeError):
                        pass
            nested_stats = getattr(snapshot, "stats", None)
            if nested_stats and isinstance(nested_stats, dict):
                for k, v in nested_stats.items():
                    k_lower = k.lower() if isinstance(k, str) else k
                    if k_lower in STAT_NAMES:
                        try:
                            raw[k_lower] = int(v)
                        except (ValueError, TypeError):
                            pass

        # Normalise keys
        for k in STAT_NAMES:
            v = raw.get(k) or raw.get(k.upper()) or 0
            try:
                stats[k] = int(v)
            except (ValueError, TypeError):
                stats[k] = 0

        return stats

    def _extract_stat_points(self, snapshot: Any) -> int:
        """Extract available stat points from a snapshot."""
        if snapshot is None:
            return 0

        if isinstance(snapshot, dict):
            # Primary: progression block
            prog = snapshot.get("progression") or {}
            if isinstance(prog, dict):
                sp = prog.get("stat_points")
                if sp is not None:
                    return max(0, int(sp))
            # Top-level fallback
            sp = snapshot.get("stat_points")
            if sp is not None:
                return max(0, int(sp))
        else:
            prog = getattr(snapshot, "progression", None)
            if prog:
                sp = getattr(prog, "stat_points", None)
                if sp is not None:
                    return max(0, int(sp))
            sp = getattr(snapshot, "stat_points", None)
            if sp is not None:
                return max(0, int(sp))

        return 0

    def _get_or_create_state(self, bot_id: str) -> dict[str, int]:
        """Get or initialise the per-bot allocation tracking state."""
        with self._lock:
            if bot_id not in self._allocation_state:
                self._allocation_state[bot_id] = {s: 0 for s in STAT_NAMES}
            return dict(self._allocation_state[bot_id])

    @staticmethod
    def _default_stats() -> dict[str, int]:
        """Default stat distribution for unknown classes."""
        return {"str": 30, "agi": 30, "vit": 30, "int": 30, "dex": 30, "luk": 1}
