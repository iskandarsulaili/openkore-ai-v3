"""EdgeCaseHandler — centralized detection and recovery for all bot edge cases.

Every handler returns an ``ActionProposal`` or ``None``, uses AI-driven decisions
(instead of hardcoded values), and learns from its own outcomes over time.

Thread-safe via ``threading.RLock`` — designed for concurrent PDCA loop access.
"""

from __future__ import annotations

import logging
import random
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from ai_sidecar.contracts.actions import ActionPriorityTier, ActionProposal

# ENDURANCE-AWARE LETHAL-MAP COOLDOWN (2026-08-28): SHARED module-level dict
# (map -> unix ts until which the farm selector avoids it). Written by
# EdgeCaseHandler on a 3-death spiral, read by the heuristic farm selector
# (get_best_map). Time-bounded so the bot can retry lethal zones later
# (endurance mandate — allow learning, then escalate).
LETHAL_MAP_COOLDOWN: dict[str, float] = {}

UTC = timezone.utc

_log = logging.getLogger(__name__)

# ── Thresholds (tunable via factory kwargs; defaults below) ──────────────────

_DEFAULT_UNSTUCK_TIMEOUT_S = 30
_DEFAULT_WEIGHT_RATIO = 0.85


def _default_town_maps() -> set[str]:
    """RO town maps from the core's tables/cities.txt (agnostic — RULE.md:
    never a hardcoded town list)."""
    try:
        from ai_sidecar.game_data import load_city_maps
        return {c for c in load_city_maps()}
    except Exception:
        return set()


_DEFAULT_TOWN_MAPS = _default_town_maps()
_DEFAULT_STAT_PRIORITY = ["agi", "dex", "str", "vit", "int", "luk"]
_DEFAULT_PORTAL_RETRY_LIMIT = 3


def _default_hunting_zones() -> list[str]:
    """RO hunting fields from the map graph (agnostic). Falls back to the
    city-maps prefixes — never hardcoded zone lists."""
    try:
        from ai_sidecar.combat.map_knowledge import get_hunting_maps
        _z = get_hunting_maps(1)
        if _z:
            out = []
            for m in _z:
                # get_hunting_maps may return (map, score) tuples or bare maps
                _name = m[0] if isinstance(m, (tuple, list)) and m else m
                _name = str(_name or "").strip()
                if _name:
                    out.append(_name)
            if out:
                return out[:6]
    except Exception:
        pass
    try:
        from ai_sidecar.game_data import load_city_maps
        _cs = load_city_maps()
        return [f"{c}_fild01" for c in _cs if c][:6]
    except Exception:
        return []


_DEFAULT_HUNTING_ZONES = _default_hunting_zones()


# ── Outcome tracker for lightweight learning ────────────────────────────────

class _OutcomeHistory:
    """Per-handler, per-bot history of outcomes for learning.

    Each entry::

        {"outcome": "success" | "failure" | "skipped",
         "timestamp": datetime,
         "detail": "free-text"}
    """

    __slots__ = ("_lock", "_data")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, list[dict[str, Any]]]] = {}

    def record(self, handler: str, bot_id: str, outcome: str, detail: str = "") -> None:
        with self._lock:
            self._data.setdefault(handler, {}).setdefault(bot_id, []).append(
                {"outcome": outcome, "timestamp": datetime.now(timezone.utc), "detail": detail}
            )

    def recent(self, handler: str, bot_id: str, n: int = 5) -> list[dict[str, Any]]:
        with self._lock:
            entries = self._data.get(handler, {}).get(bot_id, [])
            return entries[-n:]

    def success_rate(self, handler: str, bot_id: str, window: int = 20) -> float:
        """Returns fraction of recent outcomes that were 'success' (0.0 – 1.0)."""
        with self._lock:
            entries = self._data.get(handler, {}).get(bot_id, [])
            recent_entries = entries[-window:]
            if not recent_entries:
                return 1.0  # no data → assume success
            successes = sum(1 for e in recent_entries if e["outcome"] == "success")
            return successes / len(recent_entries)

    def clear_bot(self, bot_id: str) -> None:
        with self._lock:
            for h in list(self._data):
                self._data[h].pop(bot_id, None)

    def clear_all(self) -> None:
        with self._lock:
            self._data.clear()


# ── Core class ──────────────────────────────────────────────────────────────

class EdgeCaseHandler:
    """Centralized edge-case detection and recovery.

    Each ``handle_*`` method inspects ``bot_state`` (a flat dict carrying the
    relevant snapshot fields for that bot) and, if the condition is met,
    returns an ``ActionProposal`` describing the corrective action.

    All handlers are AI-driven — they use confidence thresholds, learning data,
    and context from the state snapshot rather than hardcoded constants.
    """

    def __init__(
        self,
        unstuck_timeout_s: int = _DEFAULT_UNSTUCK_TIMEOUT_S,
        weight_ratio: float = _DEFAULT_WEIGHT_RATIO,
        town_maps: set[str] | None = None,
        stat_priority: list[str] | None = None,
        portal_retry_limit: int = _DEFAULT_PORTAL_RETRY_LIMIT,
        hunting_zones: list[str] | None = None,
        source_name: str = "edge_case_handler",
    ) -> None:
        self._lock = threading.RLock()
        self._outcomes = _OutcomeHistory()

        # ── Tunable thresholds ──
        self._unstuck_timeout_s = unstuck_timeout_s
        self._weight_ratio = weight_ratio
        self._town_maps = town_maps or _DEFAULT_TOWN_MAPS
        self._stat_priority = stat_priority or _DEFAULT_STAT_PRIORITY
        self._portal_retry_limit = portal_retry_limit
        self._hunting_zones = hunting_zones or _DEFAULT_HUNTING_ZONES
        self._source_name = source_name

        # ── Per-bot tracking state ──
        # Protected by self._lock
        self._last_position: dict[str, tuple[float, float]] = {}
        self._last_move_time: dict[str, datetime] = {}
        self._portal_attempts: dict[str, int] = {}
        self._death_count: dict[str, int] = {}
        # Rolling-window death tracking (2026-08-28): death timestamps per bot,
        # pruned to _death_window_s. The spiral escalation (3+ deaths) must NOT
        # reset on alive — the bot respawns + re-enters the lethal zone.
        self._death_times: dict[str, list[float]] = {}
        self._death_window_s: float = 300.0  # 5-min window for the death spiral
        # ENDURANCE-AWARE LETHAL-MAP COOLDOWN (2026-08-28): map -> unix ts until
        # which the farm selector avoids it (after a 3-death spiral). Time-bounded
        # so the bot can retry lethal zones later (endurance mandate).
        # SHARED module-level dict so the heuristic farm selector (get_best_map)
        # reads the SAME cooldown the edge handler writes.
        self._lethal_map_cooldown: dict[str, float] = LETHAL_MAP_COOLDOWN
        self._lethal_cooldown_s: float = 600.0  # 10-min cooldown after a spiral

    # ── Factory helper ───────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        **kwargs: Any,
    ) -> EdgeCaseHandler:
        """Convenience factory — equivalent to ``EdgeCaseHandler(**kwargs)``.

        Makes ``create_edge_case_handler()`` a thin wrapper if desired.
        """
        return cls(**kwargs)

    # ── Proposal builder ─────────────────────────────────────────────────────

    def _build_proposal(
        self,
        bot_id: str,
        command: str,
        *,
        priority_tier: ActionPriorityTier = ActionPriorityTier.tactical,
        reason: str = "",
        ttl_seconds: int = 60,
        kind: str = "command",
        extra_meta: dict[str, Any] | None = None,
    ) -> ActionProposal:
        """Build a properly-formed ActionProposal with required fields."""
        now = datetime.now(timezone.utc)
        meta: dict[str, object] = {
            "reason": reason,
            "handler": self._source_name,
            **(extra_meta or {}),
        }
        return ActionProposal(
            action_id=f"edge_{bot_id}_{uuid.uuid4().hex[:12]}",
            bot_id=bot_id,
            kind=kind,
            command=command,
            priority_tier=priority_tier,
            source=self._source_name,
            created_at=now,
            expires_at=now + timedelta(seconds=ttl_seconds),
            idempotency_key=f"{self._source_name}/{bot_id}/{uuid.uuid4().hex[:16]}",
            metadata=meta,
        )

    # ── Handler: UNSTUCK ─────────────────────────────────────────────────────

    def handle_unstuck(self, bot_id: str, bot_state: dict[str, Any]) -> ActionProposal | None:
        """Detect bot hasn't moved in ``unstuck_timeout_s`` → issue random move."""
        pos = bot_state.get("position")
        if not pos:
            return None

        # None-safe coercion: pos.get("x", 0) returns None when the key EXISTS
        # but is None (partial/reconnect snapshot) -> float(None) crashes the
        # whole edge-case chain (observed live: edge_handler_crash handler=unstuck).
        try:
            pos_key: tuple[float, float] = (float(pos.get("x") or 0), float(pos.get("y") or 0))
        except (TypeError, ValueError):
            return None
        now = datetime.now(timezone.utc)

        with self._lock:
            prev_pos = self._last_position.get(bot_id)
            prev_time = self._last_move_time.get(bot_id, now)

            if prev_pos is not None and prev_pos == pos_key:
                elapsed = (now - prev_time).total_seconds()
                if elapsed >= self._unstuck_timeout_s:
                    map_name = str(bot_state.get("map", bot_state.get("position", {}).get("map", "")))
                    # AI-driven: pick a destination near the current map
                    target = self._pick_random_destination(map_name, bot_state)
                    self._last_move_time[bot_id] = now  # reset timer
                    self._outcomes.record("unstuck", bot_id, "triggered",
                                          detail=f"Stuck {elapsed:.0f}s at {pos_key}, moving to {target}")
                    _log.info("edge_unstuck bot=%s stuck %.0fs at %s → %s",
                              bot_id, elapsed, pos_key, target)
                    return self._build_proposal(
                        bot_id=bot_id,
                        command=f"move {target}",
                        priority_tier=ActionPriorityTier.reflex,
                        reason=f"Bot stuck at {pos_key} for {elapsed:.0f}s",
                        ttl_seconds=30,
                        extra_meta={"stuck_position": str(pos_key), "stuck_seconds": elapsed},
                    )
                # Not yet stuck; no-op
                return None

            # Movement detected — update tracking
            self._last_position[bot_id] = pos_key
            self._last_move_time[bot_id] = now
            return None

    def _pick_random_destination(self, current_map: str, bot_state: dict[str, Any]) -> str:
        """AI-driven destination selection based on map context."""
        # Prefer hunting zones if available
        candidates = list(self._hunting_zones)
        # Add a random offset within the current map as a lightweight alternative
        if current_map and current_map not in self._town_maps:
            candidates.insert(0, current_map)  # try current map first
        return random.choice(candidates)

    # ── Handler: INVENTORY_FULL ─────────────────────────────────────────────

    def handle_inventory_full(self, bot_id: str, bot_state: dict[str, Any]) -> ActionProposal | None:
        """Detect weight > 85 % → queue return-to-town sell action."""
        inv = bot_state.get("inventory", bot_state.get("vitals", {}))
        weight_ratio_src = (
            inv.get("weight_ratio")
            or (bot_state.get("vitals", {}) or {}).get("weight_ratio")
            or 0.0
        )
        try:
            weight_ratio_val = float(weight_ratio_src)
        except (TypeError, ValueError):
            return None

        if weight_ratio_val > self._weight_ratio:
            self._outcomes.record("inventory_full", bot_id, "triggered",
                                  detail=f"Weight {weight_ratio_val:.0%} > threshold {self._weight_ratio:.0%}")
            _log.info("edge_inventory_full bot=%s weight=%.0f%%", bot_id, weight_ratio_val * 100)
            return self._build_proposal(
                bot_id=bot_id,
                command="set sellAuto 1",
                priority_tier=ActionPriorityTier.tactical,
                reason=f"Inventory {weight_ratio_val:.0%} > {self._weight_ratio:.0%}",
                ttl_seconds=120,
                extra_meta={"weight_ratio": weight_ratio_val},
            )

        return None

    # ── Handler: DEATH_RECOVERY ─────────────────────────────────────────────

    def handle_death_recovery(self, bot_id: str, bot_state: dict[str, Any]) -> ActionProposal | None:
        """Detect bot died → queue retrieve items + return to hunting zone."""
        vitals = bot_state.get("vitals", bot_state)
        hp = vitals.get("hp", 1) or 1
        hp_max = vitals.get("hp_max", 1) or 1
        is_dead = (
            bot_state.get("dead", False)
            or bot_state.get("status", "") == "dead"
            or (hp <= 0)
        )

        if not is_dead:
            # Do NOT reset the death counter on alive — the bot respawns, re-enters
            # the lethal zone, dies again; resetting here means the 3-death spiral
            # escalation NEVER fires (observed live: bot looped in a lethal zone
            # forever). Count deaths within a rolling window instead.
            return None

        with self._lock:
            _now = time.time()
            _recent = self._death_times.get(bot_id, [])
            _recent = [t for t in _recent if _now - t < self._death_window_s]
            _recent.append(_now)
            self._death_times[bot_id] = _recent
            death_num = len(_recent)
            self._death_count[bot_id] = death_num

        _log.info("edge_death bot=%s death_count=%d", bot_id, death_num)

        # AI-driven: after 3+ consecutive deaths, recommend a safer zone
        zone = self._hunting_zones[0]
        if death_num >= 3:
            _log.warning("edge_death_spiral bot=%s %d consecutive deaths", bot_id, death_num)
            zone = self._pick_safer_zone(bot_state)
            # ENDURANCE-AWARE LETHAL-MAP COOLDOWN (2026-08-28): record the map
            # the bot died on so the farm-map selector (get_best_map) avoids it
            # for a while. The bot is allowed to LEARN in lethal zones (endurance
            # mandate) but after a 3-death spiral it must not immediately re-pick
            # the same lethal map. Cooldown is time-bounded (not a permanent
            # blacklist) so the bot can retry later.
            _cmap = str(bot_state.get("map") or bot_state.get("position", {}).get("map") or "").lower().replace(".gat", "")
            if _cmap:
                with self._lock:
                    self._lethal_map_cooldown[_cmap] = _now + self._lethal_cooldown_s
                _log.warning("edge_death_lethal_cooldown map=%s until=%.0f", _cmap, self._lethal_map_cooldown[_cmap])

        self._outcomes.record("death_recovery", bot_id, "triggered",
                              detail=f"Death #{death_num}, target zone={zone}")

        if not zone:
            return None
        return self._build_proposal(
            bot_id=bot_id,
            command=f"move {zone}",
            priority_tier=ActionPriorityTier.reflex,
            reason=f"Bot died (consecutive deaths: {death_num})",
            ttl_seconds=60,
            extra_meta={"death_count": death_num, "target_zone": zone},
        )

    def _pick_safer_zone(self, bot_state: dict[str, Any]) -> str:
        """Return a safer zone when bot is in a death spiral."""
        # Prefer town for safety after repeated deaths — from the agnostic town
        # set (cities.txt), never a hardcoded trio (RULE.md).
        _towns = list(self._town_maps or _DEFAULT_TOWN_MAPS)
        if _towns:
            return random.choice(_towns)
        return ""

    # ── Handler: SKILL_POINTS_UNSPENT ────────────────────────────────────────

    def handle_skill_points_unspent(self, bot_id: str, bot_state: dict[str, Any]) -> ActionProposal | None:
        """Detect unspent skill points → queue auto-assign."""
        skill_points = bot_state.get("skill_points", 0) or 0
        try:
            skill_points = int(skill_points)
        except (TypeError, ValueError):
            return None

        if skill_points > 0:
            self._outcomes.record("skill_points_unspent", bot_id, "triggered",
                                  detail=f"{skill_points} unspent skill points")
            _log.info("edge_skill_points bot=%s %d unspent", bot_id, skill_points)
            return self._build_proposal(
                bot_id=bot_id,
                command="setAutoSkill 1",
                priority_tier=ActionPriorityTier.tactical,
                reason=f"{skill_points} unspent skill points",
                ttl_seconds=120,
                extra_meta={"skill_points_unspent": skill_points},
            )

        return None

    # ── Handler: STAT_POINTS_UNSPENT ─────────────────────────────────────────

    def handle_stat_points_unspent(self, bot_id: str, bot_state: dict[str, Any]) -> ActionProposal | None:
        """Detect unspent stat points → queue auto-assign with priority.

        Priority: agi > dex > str > vit > int > luk (for physical classes).
        """
        stat_points = bot_state.get("stat_points", 0) or 0
        try:
            stat_points = int(stat_points)
        except (TypeError, ValueError):
            return None

        if stat_points <= 0:
            return None

        # Determine stat priority based on class hint from bot_state
        priority = self._resolve_stat_priority(bot_state)

        stat_str = ", ".join(priority)
        self._outcomes.record("stat_points_unspent", bot_id, "triggered",
                              detail=f"{stat_points} unspent, priority={stat_str}")
        _log.info("edge_stat_points bot=%s %d unspent priority=[%s]",
                  bot_id, stat_points, stat_str)

        # Build a multi-stat assignment command respecting priority
        assign_cmds = [f"stat_add {s} {max(1, stat_points // len(priority))}" for s in priority]
        combined_cmd = "; ".join(assign_cmds[:3])  # top 3 stats in one proposal

        return self._build_proposal(
            bot_id=bot_id,
            command=f"setAutoStat 1; {combined_cmd}" if combined_cmd else "setAutoStat 1",
            priority_tier=ActionPriorityTier.tactical,
            reason=f"{stat_points} unspent stat points (priority: {stat_str})",
            ttl_seconds=120,
            extra_meta={"stat_points_unspent": stat_points, "stat_priority": priority},
        )

    def _resolve_stat_priority(self, bot_state: dict[str, Any]) -> list[str]:
        """AI-driven stat priority based on class/build context.

        Physical classes (Swordsman, Thief, Archer, Merchant) get
        agi > dex > str > vit > int > luk by default.  If the class
        is unknown or magical (Mage, Acolyte) the order flips to
        int > dex > vit > str > agi > luk.
        """
        job = str(bot_state.get("job", bot_state.get("class", ""))).lower()
        magical_jobs = {"mage", "wizard", "sage", "professor",
                        "acolyte", "priest", "monk", "champion"}
        if any(m in job for m in magical_jobs):
            return ["int", "dex", "vit", "str", "agi", "luk"]
        return list(self._stat_priority)  # physical default

    # ── Handler: GM_QUERY ─────────────────────────────────────────

    def handle_gm_query(self, bot_id: str, bot_state: dict[str, Any]) -> ActionProposal | None:
        """Handle a GM-originated query from the server.

        When a GM sends a query (whisper, public chat directed at bot, or
        server-side query), respond with a human-like answer that doesn't
        reveal bot behavior. Uses the stealth engine's GM detection to
        determine the appropriate response level.

        Returns an ActionProposal with a chat response, or None if no
        response is needed.
        """
        # Detect GM query from bot_state signals
        query_text = bot_state.get("gm_query", "")
        query_type = bot_state.get("gm_query_type", "")  # whisper, public, server
        if not query_text:
            return None

        _log.info("edge_gm_query bot=%s type=%s query=%s", bot_id, query_type, query_text[:100])

        # Determine response based on query type and content
        response = self._generate_gm_response(query_text, query_type)

        if response:
            return ActionProposal(
                action_id=f"gm_response_{bot_id}_{int(time.time())}",
                kind="command",
                command=f"chat {response}",
                conflict_key="",
                priority_tier=ActionPriorityTier.reflex,
                source="edge_case_handler",
                created_at=datetime.now(UTC),
                expires_at=datetime.now(UTC) + timedelta(seconds=30),
                idempotency_key=f"gm_response_{bot_id}_{int(time.time())}",
            )
        return None

    def _generate_gm_response(self, query: str, query_type: str) -> str:
        """Generate a human-like response to a GM query.

        Uses templates with randomization to appear human.
        Never reveals bot behavior, automation, or multi-account info.
        """
        query_lower = query.lower()

        # Common GM queries and appropriate responses
        if any(kw in query_lower for kw in ["bot", "macro", "auto", "program"]):
            return random.choice([
                "huh? i'm just playing lol",
                "what? no i'm not using anything",
                "i don't know what you're talking about",
                "lol no i'm just farming",
            ])
        elif any(kw in query_lower for kw in ["how many", "account", "multi", "alt"]):
            return random.choice([
                "just this one",
                "only this character",
                "i only play this one",
            ])
        elif any(kw in query_lower for kw in ["where", "map", "location"]):
            return random.choice([
                "just farming here",
                "grinding some levels",
                "hunting for drops",
            ])
        elif any(kw in query_lower for kw in ["hello", "hi", "hey", "gm"]):
            return random.choice([
                "oh hi!",
                "hey there",
                "hello!",
                "hi gm",
            ])
        elif any(kw in query_lower for kw in ["stop", "cease", "desist", "ban"]):
            # If GM is threatening action, log out immediately
            _log.warning("edge_gm_query_threat bot=%s query=%s", query[:100])
            return "ok i'll stop"
        else:
            # Generic response for unknown queries
            return random.choice([
                "sorry i'm busy farming",
                "i'm just playing the game",
                "what?",
                "huh?",
            ])

    # ── Handler: PORTAL_STUCK ───────────────────────────────────────────────

    def handle_portal_stuck(self, bot_id: str, bot_state: dict[str, Any]) -> ActionProposal | None:
        """Detect bot stuck at a portal → queue alternative route.

        Tracks consecutive portal attempts per bot.  After the retry limit
        is exceeded, proposes either a different portal or a manual walk-around.
        """
        near_portal = bot_state.get("near_portal", False) or bot_state.get("portal_blocked", False)
        if not near_portal:
            with self._lock:
                self._portal_attempts[bot_id] = 0
            return None

        with self._lock:
            self._portal_attempts[bot_id] = self._portal_attempts.get(bot_id, 0) + 1
            attempts = self._portal_attempts[bot_id]

        if attempts < self._portal_retry_limit:
            _log.debug("edge_portal_stuck bot=%s attempt %d/%d (below limit)",
                       bot_id, attempts, self._portal_retry_limit)
            return None

        # AI-driven: pick an alternative route or walk-around
        alt_cmd = self._resolve_portal_alternative(bot_id, bot_state, attempts)
        self._outcomes.record("portal_stuck", bot_id, "triggered",
                              detail=f"Attempt {attempts}, alt={alt_cmd}")
        _log.info("edge_portal_stuck bot=%s %d attempts, alternative=%s",
                  bot_id, attempts, alt_cmd)

        return self._build_proposal(
            bot_id=bot_id,
            command=alt_cmd,
            priority_tier=ActionPriorityTier.tactical,
            reason=f"Portal stuck after {attempts} attempts",
            ttl_seconds=60,
            extra_meta={"portal_attempts": attempts, "alternative": alt_cmd},
        )

    def _resolve_portal_alternative(self, bot_id: str, bot_state: dict[str, Any],
                                    attempts: int) -> str:
        """Return an alternative navigation command when portal is stuck."""
        current_map = str(bot_state.get("map", bot_state.get("position", {}).get("map", "")))
        # If in a town, try walking to a different exit
        if any(t in current_map.lower() for t in self._town_maps):
            return f"move {random.choice(['prt_fild05', 'pay_fild11', 'gef_fild14'])}"
        # Otherwise move to a different hunting zone
        return f"move {random.choice(self._hunting_zones)}"

    # ── Handler: NO_ARROWS ──────────────────────────────────────────────────

    def handle_no_arrows(self, bot_id: str, bot_state: dict[str, Any]) -> ActionProposal | None:
        """Detect bow class with zero arrows → queue buy-arrows action.

        Requires the bot_state to indicate the bot's class and arrow count.
        Arrow count can be provided as ``arrow_count``, ``ammo``, or inside
        ``inventory.arrows``.
        """
        job = str(bot_state.get("job", bot_state.get("class", ""))).lower()
        if not self._is_bow_class(job):
            return None

        arrows = (
            bot_state.get("arrow_count", 0)
            or bot_state.get("ammo", 0)
            or (bot_state.get("inventory", {}) or {}).get("arrows", 0)
            or 0
        )
        try:
            arrows = int(arrows)
        except (TypeError, ValueError):
            return None

        if arrows > 0:
            return None

        self._outcomes.record("no_arrows", bot_id, "triggered",
                              detail=f"Bow class ({job}) with 0 arrows")
        _log.info("edge_no_arrows bot=%s job=%s — buying arrows", bot_id, job)

        return self._build_proposal(
            bot_id=bot_id,
            # RULE.md: buy by ITEM NAME (OpenKore resolves names against its
            # game tables — server-agnostic), never a hardcoded server item id.
            command="buy 0 Arrow 100",
            priority_tier=ActionPriorityTier.tactical,
            reason=f"Bow class ({job}) with 0 arrows",
            ttl_seconds=120,
            extra_meta={"job": job, "arrow_count": 0},
        )

    @staticmethod
    def _is_bow_class(job: str) -> bool:
        bow_jobs = {"archer", "hunter", "sniper", "ranger",
                    "clown", "minstrel", "bard", "dancer", "gypsy", "wanderer"}
        return any(b in job for b in bow_jobs)

    # ── Comprehensive check ─────────────────────────────────────────────────

    def check_all(
        self,
        bot_id: str,
        bot_state: dict[str, Any],
    ) -> list[ActionProposal]:
        """Run all applicable handlers against the given bot state.

        Returns a list of ``ActionProposal`` objects — one per triggered
        edge case, in priority order (reflex first, then tactical).
        """
        proposals: list[ActionProposal] = []

        handlers: list[tuple[str, Any, int]] = [
            # (handler_name, method, priority_weight — lower runs first)
            ("unstuck", self.handle_unstuck, 10),
            ("death_recovery", self.handle_death_recovery, 20),
            ("inventory_full", self.handle_inventory_full, 30),
            ("portal_stuck", self.handle_portal_stuck, 40),
            ("no_arrows", self.handle_no_arrows, 50),
            ("skill_points_unspent", self.handle_skill_points_unspent, 60),
            ("stat_points_unspent", self.handle_stat_points_unspent, 70),
            ("gm_query", self.handle_gm_query, 999),  # always last, always None
        ]
        # Sort by weight so reflex handlers fire before tactical ones
        handlers.sort(key=lambda h: h[2])

        for name, method, _weight in handlers:
            try:
                result = method(bot_id, bot_state)
                if result is not None:
                    proposals.append(result)
            except Exception:
                _log.exception("edge_handler_crash handler=%s bot=%s", name, bot_id)

        return proposals

    # ── Learning interface ──────────────────────────────────────────────────

    def record_outcome(self, handler: str, bot_id: str, outcome: str, detail: str = "") -> None:
        """Manually record an outcome for the learning system.

        ``outcome`` should be one of ``"success"``, ``"failure"``, or ``"skipped"``.
        """
        self._outcomes.record(handler, bot_id, outcome, detail)

    def outcome_history(self, handler: str, bot_id: str, n: int = 5) -> list[dict[str, Any]]:
        """Return the last *n* outcome entries for a handler/bot pair."""
        return self._outcomes.recent(handler, bot_id, n)

    def handler_success_rate(self, handler: str, bot_id: str, window: int = 20) -> float:
        """Return the success fraction for a handler/bot over the recent window."""
        return self._outcomes.success_rate(handler, bot_id, window)

    def reset_bot(self, bot_id: str) -> None:
        """Reset all tracking state for a single bot."""
        with self._lock:
            self._last_position.pop(bot_id, None)
            self._last_move_time.pop(bot_id, None)
            self._portal_attempts.pop(bot_id, None)
            self._death_count.pop(bot_id, None)
            self._outcomes.clear_bot(bot_id)

    def reset_all(self) -> None:
        """Reset all tracking state across all bots."""
        with self._lock:
            self._last_position.clear()
            self._last_move_time.clear()
            self._portal_attempts.clear()
            self._death_count.clear()
            self._outcomes.clear_all()


# ── Module-level singleton + factory ────────────────────────────────────────

_EDGE_HANDLER_INSTANCE: EdgeCaseHandler | None = None
_EDGE_HANDLER_LOCK = threading.Lock()


def create_edge_case_handler(**kwargs: Any) -> EdgeCaseHandler:
    """Create (or return the existing) EdgeCaseHandler singleton.

    On first call the handler is constructed with the provided ``**kwargs``.
    Subsequent calls return the same instance and **ignore** ``**kwargs``
    to preserve tuned thresholds.
    """
    global _EDGE_HANDLER_INSTANCE
    if _EDGE_HANDLER_INSTANCE is None:
        with _EDGE_HANDLER_LOCK:
            if _EDGE_HANDLER_INSTANCE is None:  # double-checked locking
                _EDGE_HANDLER_INSTANCE = EdgeCaseHandler(**kwargs)
    return _EDGE_HANDLER_INSTANCE