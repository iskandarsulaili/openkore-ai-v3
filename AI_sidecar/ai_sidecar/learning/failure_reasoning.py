"""
Failure Reasoning Engine — Unified failure capture, reasoning, and feedback loop.

Every failure (death, stuck, autobuy loop, reflex failure, etc.) is captured,
reasoned about, stored with server_id, shared across bots via P2P, and fed
back into decision-making via the PDCA loop and LLM planner context.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class FailureRecord:
    """A single failure event captured for reasoning and feedback."""

    id: str = ""
    server_id: str = "default"
    bot_id: str = ""
    category: str = "unknown"
    subcategory: str | None = None
    timestamp: float = 0.0
    context: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    lesson_learned: str = ""
    action_taken: str = ""
    action_effective: bool | None = None
    resolved: bool = False
    resolved_at: float | None = None
    recurrence_count: int = 1
    recurrence_key: str = ""
    applied_to_config: list[str] = field(default_factory=list)
    peer_shared: bool = False


# ---------------------------------------------------------------------------
# FailureReasoningEngine
# ---------------------------------------------------------------------------


class FailureReasoningEngine:
    """Unified failure reasoning engine.

    Captures failures from all subsystems, generates rule-based reasoning
    and lessons, stores them in the shared learning DB, escalates recurring
    issues, and shares knowledge via P2P.
    """

    VALID_CATEGORIES = frozenset({
        "autobuy_loop", "no_heal_items", "death", "reflex_failure",
        "validation_error", "latency_exceeded", "stuck", "overweight",
        "party_ghost", "llm_parse_error", "config_conflict",
        "movement_fail", "combat_fail", "economy_fail", "p2p_fail",
        "unknown",
    })

    def __init__(
        self,
        shared_db: Any = None,
        p2p_node: Any = None,
        server_adaptation: Any = None,
    ) -> None:
        self._lock = RLock()
        self._shared_db = shared_db
        self._p2p_node = p2p_node
        self._server_adaptation = server_adaptation
        self._init_db()

    def _init_db(self) -> None:
        """Ensure the failures table exists in the shared learning DB."""
        if self._shared_db is None:
            from ai_sidecar.learning.shared_learning_db import get_shared_learning_db
            self._shared_db = get_shared_learning_db()
        if hasattr(self._shared_db, "_ensure_failures_table"):
            self._shared_db._ensure_failures_table()

    # -- Public API ---------------------------------------------------------

    def capture_failure(
        self,
        category: str,
        subcategory: str | None = None,
        context: dict[str, Any] | None = None,
        bot_id: str = "default",
        server_id: str | None = None,
    ) -> str:
        """Capture a failure event, generate reasoning, store, and escalate if needed.

        Returns the failure ID.
        """
        if category not in self.VALID_CATEGORIES:
            logger.warning("failure_unknown_category: %s", category)
            category = "unknown"

        if server_id is None:
            server_id = self._get_server_id()

        now = time.time()
        ctx = context or {}

        # Build recurrence key from (server_id, category, subcategory, normalized context)
        recurrence_key = self._build_recurrence_key(server_id, category, subcategory, ctx)

        # Check for recurrence within 1 hour
        recurrence_count = 1
        if self._shared_db is not None and hasattr(self._shared_db, "increment_failure_recurrence"):
            existing_count = self._shared_db.increment_failure_recurrence(recurrence_key)
            if existing_count > 0:
                recurrence_count = existing_count

        # Generate a unique ID
        failure_id = self._generate_id(bot_id, category, now)

        record = FailureRecord(
            id=failure_id,
            server_id=server_id,
            bot_id=bot_id,
            category=category,
            subcategory=subcategory,
            timestamp=now,
            context=ctx,
            recurrence_count=recurrence_count,
            recurrence_key=recurrence_key,
        )

        # Generate reasoning and lesson
        record.reasoning = self._generate_reasoning(record)
        record.lesson_learned = self._generate_lesson(record)

        # Store in DB
        if self._shared_db is not None and hasattr(self._shared_db, "record_failure"):
            self._shared_db.record_failure({
                "id": record.id,
                "server_id": record.server_id,
                "bot_id": record.bot_id,
                "category": record.category,
                "subcategory": record.subcategory,
                "timestamp": record.timestamp,
                "context": json.dumps(record.context),
                "reasoning": record.reasoning,
                "lesson_learned": record.lesson_learned,
                "action_taken": record.action_taken,
                "action_effective": None,
                "resolved": 0,
                "resolved_at": None,
                "recurrence_count": record.recurrence_count,
                "recurrence_key": record.recurrence_key,
                "applied_to_config": json.dumps(record.applied_to_config),
                "peer_shared": 0,
            })

        logger.info(
            "failure_captured: id=%s category=%s subcategory=%s server=%s bot=%s count=%d",
            failure_id, category, subcategory, server_id, bot_id, recurrence_count,
        )

        # Escalate if recurring >= 3
        if recurrence_count >= 3:
            self._escalate(record)

        return failure_id

    def get_failures(
        self,
        server_id: str | None = None,
        category: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Query failures with optional filters."""
        if self._shared_db is not None and hasattr(self._shared_db, "get_failures"):
            return self._shared_db.get_failures(
                server_id=server_id, category=category, limit=limit,
            )
        return []

    def get_recurring_failures(
        self,
        server_id: str | None = None,
        min_count: int = 3,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Find failures with recurrence_count >= min_count."""
        if self._shared_db is not None and hasattr(self._shared_db, "get_recurring_failures"):
            return self._shared_db.get_recurring_failures(
                server_id=server_id, min_count=min_count, limit=limit,
            )
        return []

    def get_failure_summary(self, server_id: str | None = None) -> str:
        """Return a formatted summary for LLM context injection."""
        if self._shared_db is not None and hasattr(self._shared_db, "get_failure_summary"):
            return self._shared_db.get_failure_summary(server_id=server_id)
        return "No failure data available."

    def get_llm_context(self, server_id: str | None = None) -> str:
        """Return a concise failure context string for LLM planner prompts."""
        if server_id is None:
            server_id = self._get_server_id()

        failures = self.get_failures(server_id=server_id, limit=100)
        if not failures:
            return f"No recent failures. Server: {server_id}."

        total = len(failures)
        # Count by category
        cat_counts: dict[str, int] = {}
        for f in failures:
            cat = f.get("category", "unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        top_issues = sorted(cat_counts.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{cat}x{count}" for cat, count in top_issues)

        # Get most recent lesson
        recent_lesson = ""
        if failures:
            latest = max(failures, key=lambda f: f.get("timestamp", 0))
            recent_lesson = latest.get("lesson_learned", "")

        return (
            f"Recent failures: {total} total. "
            f"Top issues: {top_str}. "
            f"Lessons learned: {recent_lesson}. "
            f"Server: {server_id}."
        )

    def mark_resolved(self, failure_id: str, effective: bool = True) -> None:
        """Mark a failure as resolved."""
        if self._shared_db is not None and hasattr(self._shared_db, "mark_failure_resolved"):
            self._shared_db.mark_failure_resolved(failure_id, effective=effective)

    def share_via_p2p(self, record: FailureRecord) -> None:
        """Share a failure record via P2P if available."""
        if self._p2p_node is not None and hasattr(self._p2p_node, "broadcast_failure"):
            try:
                self._p2p_node.broadcast_failure({
                    "id": record.id,
                    "server_id": record.server_id,
                    "bot_id": record.bot_id,
                    "category": record.category,
                    "subcategory": record.subcategory,
                    "reasoning": record.reasoning,
                    "lesson_learned": record.lesson_learned,
                    "context": record.context,
                    "recurrence_count": record.recurrence_count,
                    "timestamp": record.timestamp,
                })
                record.peer_shared = True
                logger.info("failure_p2p_shared: id=%s", record.id)
            except Exception:
                logger.exception("failure_p2p_share_failed: id=%s", record.id)

    # -- Internal: Reasoning ------------------------------------------------

    def _generate_reasoning(self, record: FailureRecord) -> str:
        """Generate rule-based reasoning based on category + subcategory + context."""
        cat = record.category
        sub = record.subcategory
        ctx = record.context

        if cat == "autobuy_loop":
            count = ctx.get("count", 0)
            seconds = ctx.get("seconds", 0)
            zeny = ctx.get("zeny", 0)
            return (
                f"autobuy triggered {count} times in {seconds}s with {zeny} zeny "
                f"— no zeny to buy potions, no cooldown on autobuy call"
            )

        if cat == "no_heal_items":
            hp = ctx.get("hp", 0)
            max_hp = ctx.get("max_hp", 1)
            aggro = ctx.get("aggro", 0)
            zeny = ctx.get("zeny", 0)
            return (
                f"HP={hp}/{max_hp} with {aggro} aggro, no potions in inventory, "
                f"zeny={zeny} — cannot heal"
            )

        if cat == "death":
            # Delegate to DeathAnalyzer if available
            try:
                from ai_sidecar.learning.death_analysis import get_death_analyzer
                da = get_death_analyzer()
                from ai_sidecar.learning.death_analysis import DeathRecord as DRecord
                dr = DRecord(
                    timestamp=record.timestamp,
                    map_name=ctx.get("map", ""),
                    position=(ctx.get("x", 0), ctx.get("y", 0)),
                    monster_name=ctx.get("monster_name", "unknown"),
                    monster_id=ctx.get("monster_id", 0),
                    hp_before_death=ctx.get("hp", 0),
                    max_hp=ctx.get("max_hp", 1),
                    aggro_count=ctx.get("aggro", 0),
                    had_potions=bool(ctx.get("had_potions", False)),
                    was_casting=bool(ctx.get("was_casting", False)),
                    buffs_active=ctx.get("buffs", []),
                    seconds_since_last_heal=ctx.get("seconds_since_last_heal", 0),
                    cause_of_death=sub or "unknown",
                    lesson_learned="",
                )
                return da.analyze_death(dr)
            except Exception:
                pass
            return f"Death on {ctx.get('map', '?')} by {ctx.get('monster_name', '?')} — {sub or 'unknown cause'}"

        if cat == "reflex_failure":
            reflex_name = ctx.get("reflex_name", "unknown")
            return (
                f"bridge reflex {reflex_name} failed to reach bot {record.bot_id} "
                f"— action queue may be full or bot disconnected"
            )

        if cat == "validation_error":
            error_detail = ctx.get("error_detail", "unknown")
            return f"bridge sent malformed data: {error_detail}"

        if cat == "latency_exceeded":
            action_type = ctx.get("action_type", "unknown")
            return f"action arbiter latency budget exceeded for {action_type}"

        if cat == "stuck":
            map_name = ctx.get("map", "?")
            x = ctx.get("x", 0)
            y = ctx.get("y", 0)
            seconds = ctx.get("seconds", 0)
            return f"bot stuck on {map_name} at ({x},{y}) for {seconds}s"

        if cat == "overweight":
            weight_pct = ctx.get("weight_pct", 0)
            return f"weight {weight_pct}% > 90% threshold — cannot pick up items"

        if cat == "party_ghost":
            name = ctx.get("name", "unknown")
            return f"party member {name} has HP=0/1 — likely disconnected or invalid entry"

        if cat == "llm_parse_error":
            snippet = ctx.get("response_snippet", "")[:200]
            return f"LLM returned unparseable response: {snippet}"

        if cat == "config_conflict":
            key = ctx.get("key", "?")
            value = ctx.get("value", "?")
            other_key = ctx.get("other_key", "?")
            other_value = ctx.get("other_value", "?")
            return f"config {key} set to {value} conflicts with {other_key}={other_value}"

        if cat == "movement_fail":
            x1, y1 = ctx.get("x1", 0), ctx.get("y1", 0)
            x2, y2 = ctx.get("x2", 0), ctx.get("y2", 0)
            map_name = ctx.get("map", "?")
            return f"bot failed to move from ({x1},{y1}) to ({x2},{y2}) on {map_name}"

        if cat == "combat_fail":
            detail = ctx.get("detail", "unknown")
            return f"combat loop failed: {detail}"

        if cat == "economy_fail":
            detail = ctx.get("detail", "unknown")
            return f"economy action failed: {detail}"

        if cat == "p2p_fail":
            peer = ctx.get("peer", "unknown")
            detail = ctx.get("detail", "unknown")
            return f"P2P message to {peer} failed: {detail}"

        return f"Unknown failure: category={cat} subcategory={sub}"

    # -- Internal: Lessons --------------------------------------------------

    def _generate_lesson(self, record: FailureRecord) -> str:
        """Generate an actionable lesson based on category + subcategory."""
        cat = record.category
        sub = record.subcategory or ""
        ctx = record.context

        if cat == "autobuy_loop":
            if sub == "no_zeny":
                return (
                    "Add zeny check and cooldown to autobuy reflex. "
                    "Farm basic mobs for starting zeny first."
                )
            if sub == "npc_not_found":
                map_name = ctx.get("map", "?")
                return (
                    f"Verify NPC coordinates for {map_name}. "
                    f"Update npc_steps if server has custom NPC positions."
                )
            return "Add cooldown and zeny check to autobuy reflex."

        if cat == "no_heal_items":
            return (
                "Ensure buyAuto config has correct NPC coordinates and item names. "
                "Add fallback sit-to-regen when no potions."
            )

        if cat == "death":
            if sub == "overpulled":
                new_aggro = max(1, ctx.get("max_aggro", 5) - 1)
                return (
                    f"Reduce max_aggro to {new_aggro}. "
                    f"Ensure potion stock before engaging groups."
                )
            if sub == "no_potions":
                new_stock = min(50, ctx.get("min_potion_stock", 10) + 5)
                new_heal = max(0.3, ctx.get("heal_threshold", 0.6) - 0.1)
                return (
                    f"Increase min_potion_stock to {new_stock}. "
                    f"Lower heal_threshold to {new_heal:.1f}."
                )
            if sub == "boss_skill":
                new_hp = min(0.5, ctx.get("flee_hp_pct", 0.3) + 0.1)
                return (
                    f"Add boss-specific flee trigger at {new_hp:.0%} HP. "
                    f"Consider party play for this MVP."
                )
            if sub == "ambush":
                return (
                    "Increase situational awareness radius. "
                    "Pre-cast buffs before entering dangerous areas."
                )
            if sub == "heal_starvation":
                new_heal = max(0.3, ctx.get("heal_threshold", 0.6) - 0.1)
                new_flee = min(0.5, ctx.get("flee_hp_pct", 0.3) + 0.05)
                return (
                    f"Lower heal_threshold to {new_heal:.1f}. "
                    f"Ensure auto-heal triggers above {new_flee:.0%} HP."
                )
            if sub == "cast_lock":
                return (
                    "Add interrupt-on-danger reflex. "
                    "Consider VIT investment for cast interruption resistance."
                )
            if sub == "buff_drop":
                new_hp = min(0.7, ctx.get("heal_threshold", 0.6) + 0.1)
                return (
                    f"Add auto-buff trigger at {new_hp:.0%} HP. "
                    f"Ensure buffs are maintained during combat."
                )
            return f"Death by {ctx.get('monster_name', '?')} on {ctx.get('map', '?')} — investigate."

        if cat == "reflex_failure":
            return (
                f"Check action queue health. "
                f"Verify bot {record.bot_id} is still connected. "
                f"Increase queue capacity if needed."
            )

        if cat == "validation_error":
            field_name = ctx.get("field", "?")
            return (
                f"Fix bridge data serialization for {field_name}. "
                f"Add validation before sending."
            )

        if cat == "latency_exceeded":
            action_type = ctx.get("action_type", "?")
            return (
                f"Reduce action complexity. "
                f"Increase latency budget for {action_type}."
            )

        if cat == "stuck":
            return (
                "Add stuck detection with fly wing fallback. "
                "If no fly wings, walk to nearest portal."
            )

        if cat == "overweight":
            new_weight = max(50, ctx.get("weight_pct", 90) - 10)
            return (
                f"Add auto-sell trigger at {new_weight:.0f}% weight. "
                f"Return to town to sell."
            )

        if cat == "party_ghost":
            return (
                "Add party member validation — filter out entries with HP=0/1. "
                "Request party refresh."
            )

        if cat == "llm_parse_error":
            return (
                "Add retry with stricter JSON prompt. "
                "Fall back to rule-based decision after 3 failures."
            )

        if cat == "config_conflict":
            return (
                "Add config validation before applying. "
                "Detect conflicting keys."
            )

        if cat == "movement_fail":
            return (
                "Add alternative route calculation. "
                "Use fly wing if stuck for >30s."
            )

        if cat == "combat_fail":
            return (
                "Log combat state at time of failure. "
                "Restart combat loop if stuck."
            )

        if cat == "economy_fail":
            return (
                "Log market state. "
                "Fall back to NPC prices if market data unavailable."
            )

        if cat == "p2p_fail":
            return (
                "Remove unreachable peer. "
                "Retry with exponential backoff."
            )

        return "Investigate and add appropriate handling."

    # -- Internal: Escalation ----------------------------------------------

    def _escalate(self, record: FailureRecord) -> None:
        """Handle a recurring failure (count >= 3) — systemic issue."""
        logger.warning(
            "failure_escalated: id=%s category=%s subcategory=%s server=%s count=%d",
            record.id, record.category, record.subcategory, record.server_id,
            record.recurrence_count,
        )

        # Apply config adjustment
        config_changes = self._apply_config_adjustment(record)
        if config_changes:
            record.applied_to_config = config_changes
            logger.info(
                "failure_config_adjusted: id=%s changes=%s",
                record.id, config_changes,
            )

        # Share via P2P
        self.share_via_p2p(record)

    def _apply_config_adjustment(self, record: FailureRecord) -> list[str]:
        """Generate config change suggestions based on category.

        Returns list of config keys that should be changed.
        """
        cat = record.category
        sub = record.subcategory or ""

        if cat == "autobuy_loop" and sub == "no_zeny":
            return [
                "aiSidecar_autobuyCooldownMs 30000",
                "aiSidecar_autobuyMinZeny 500",
            ]

        if cat == "no_heal_items":
            return [
                'aiSidecar_fallbackHealItem "Red Potion"',
                "heal_threshold 0.4",
            ]

        if cat == "death" and sub == "overpulled":
            return [
                "teleportAuto_minAggressives 3",
                "teleportAuto_hp 30",
            ]

        if cat == "stuck":
            return [
                "route_randomWalk 2",
                "teleportAuto_useFlyWing 1",
            ]

        if cat == "overweight":
            return [
                "sellAuto 1",
                "storageAuto 1",
            ]

        if cat == "party_ghost":
            return [
                "partyAuto 0",
            ]

        if cat == "llm_parse_error":
            return [
                "llm_context_size 2000",
                "llm_prompt_template simple",
            ]

        return []

    # -- Internal: Helpers --------------------------------------------------

    def _build_recurrence_key(
        self,
        server_id: str,
        category: str,
        subcategory: str | None,
        context: dict[str, Any],
    ) -> str:
        """Build a hash key for deduplication of similar failures.

        Normalizes context by extracting only key fields relevant to the category.
        """
        # Normalize context to a stable subset
        norm_ctx: dict[str, Any] = {}
        if category == "autobuy_loop":
            norm_ctx["map"] = context.get("map", "")
        elif category == "no_heal_items":
            norm_ctx["map"] = context.get("map", "")
        elif category == "death":
            norm_ctx["monster_id"] = context.get("monster_id", 0)
            norm_ctx["map"] = context.get("map", "")
        elif category == "stuck":
            norm_ctx["map"] = context.get("map", "")
        elif category == "overweight":
            norm_ctx["map"] = context.get("map", "")
        elif category == "reflex_failure":
            norm_ctx["reflex_name"] = context.get("reflex_name", "")
        elif category == "llm_parse_error":
            pass  # All parse errors are similar
        elif category == "config_conflict":
            norm_ctx["key"] = context.get("key", "")
            norm_ctx["other_key"] = context.get("other_key", "")
        else:
            norm_ctx["map"] = context.get("map", "")

        raw = f"{server_id}|{category}|{subcategory or ''}|{json.dumps(norm_ctx, sort_keys=True)}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _generate_id(self, bot_id: str, category: str, now: float) -> str:
        """Generate a unique failure ID."""
        raw = f"{bot_id}|{category}|{now}|{id(self)}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def _get_server_id(self) -> str:
        """Get the server ID from server_adaptation or config."""
        if self._server_adaptation is not None:
            try:
                if hasattr(self._server_adaptation, "get_server_id"):
                    sid = self._server_adaptation.get_server_id()
                    if sid:
                        return sid
                profile = self._server_adaptation.get_profile()
                if profile and profile.server_name:
                    return profile.server_name
            except Exception:
                pass
        try:
            from ai_sidecar.config import settings
            return getattr(settings, "game_server_name", "default") or "default"
        except Exception:
            return "default"


# ---------------------------------------------------------------------------
# Global Singleton
# ---------------------------------------------------------------------------

_engine: FailureReasoningEngine | None = None
_engine_lock = RLock()


def get_failure_reasoning_engine() -> FailureReasoningEngine:
    """Return the global FailureReasoningEngine singleton (thread-safe)."""
    global _engine  # noqa: PLW0603
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = FailureReasoningEngine()
    return _engine
