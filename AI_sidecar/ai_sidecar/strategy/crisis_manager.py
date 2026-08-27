"""
Crisis Manager — diagnoses problems, adapts strategy, recovers from failure.

A top player doesn't just die and respawn. They:
1. Diagnose what went wrong (why did I die? why am I stuck?)
2. Adapt strategy (change maps, change gear, change rotation)
3. Recover from failure (respawn, rebuff, re-equip, return to farm)
4. Learn from experience (don't repeat the same mistake)
5. Prevent recurring failures (add safeguards, change config)

This module wires into:
  - edge_case_handler.py: for edge case detection
  - failure_reasoning.py: for failure capture and reasoning
  - death_analysis.py: for post-mortem analysis
  - strategy_optimizer.py: for strategy adaptation
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class CrisisEvent:
    """A crisis event that needs diagnosis and recovery."""
    event_id: str
    bot_id: str
    crisis_type: str  # death, stuck, overweight, pvp_attack, gm_scan, economy_crash, party_wiped
    severity: int  # 1-10, 10=critical
    timestamp: float
    map_name: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    diagnosis: str = ""
    action_taken: str = ""
    resolved: bool = False
    resolved_at: float | None = None
    recurrence_count: int = 1
    recurrence_key: str = ""


@dataclass
class CrisisDiagnosis:
    """The diagnosis of a crisis event."""
    root_cause: str
    contributing_factors: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    config_changes: dict[str, Any] = field(default_factory=dict)
    map_blacklist: list[str] = field(default_factory=list)
    monster_blacklist: list[str] = field(default_factory=list)
    cooldown_seconds: int = 0
    lesson: str = ""


@dataclass
class CrisisRecovery:
    """A recovery plan for a crisis."""
    steps: list[str] = field(default_factory=list)
    estimated_time: int = 0  # seconds
    requires_town_visit: bool = False
    requires_restock: bool = False
    requires_regear: bool = False
    requires_leveling: bool = False
    priority: int = 5  # 1-10


# ---------------------------------------------------------------------------
# CrisisManager
# ---------------------------------------------------------------------------


class CrisisManager:
    """Centralized crisis management — diagnose, adapt, recover, learn.

    Wires into:
      - edge_case_handler.py: for edge case detection
      - failure_reasoning.py: for failure capture and reasoning
      - death_analysis.py: for post-mortem analysis
      - strategy_optimizer.py: for strategy adaptation
    """

    # Crisis types that can be diagnosed
    CRISIS_TYPES = frozenset({
        "death", "stuck", "overweight", "pvp_attack", "gm_scan",
        "economy_crash", "party_wiped", "autobuy_loop", "no_heal_items",
        "reflex_failure", "config_conflict", "movement_fail", "combat_fail",
    })

    def __init__(
        self,
        failure_reasoning: Any = None,
        death_analysis: Any = None,
        strategy_optimizer: Any = None,
        edge_case_handler: Any = None,
        enqueue_fn: Callable | None = None,
    ) -> None:
        self._lock = RLock()
        self._failure_reasoning = failure_reasoning
        self._death_analysis = death_analysis
        self._strategy_optimizer = strategy_optimizer
        self._edge_case_handler = edge_case_handler
        self._enqueue_fn = enqueue_fn

        # Crisis history
        self._crisis_history: dict[str, list[CrisisEvent]] = defaultdict(list)
        self._active_crises: dict[str, CrisisEvent] = {}  # bot_id -> active crisis

        # Learned lessons: recurrence_key -> lesson
        self._lessons: dict[str, str] = {}
        # Map blacklist: map_name -> expiry timestamp
        self._map_blacklist: dict[str, float] = {}
        # Monster blacklist: monster_name -> expiry timestamp
        self._monster_blacklist: dict[str, float] = {}
        # Config overrides: bot_id -> {config_key: value}
        self._config_overrides: dict[str, dict[str, Any]] = defaultdict(dict)

        # Cooldown tracking: bot_id -> crisis_type -> last timestamp
        self._cooldowns: dict[str, dict[str, float]] = defaultdict(dict)

        # Stats
        self._stats: dict[str, int] = {
            "crises_detected": 0,
            "crises_diagnosed": 0,
            "crises_resolved": 0,
            "crises_escalated": 0,
            "lessons_learned": 0,
            "maps_blacklisted": 0,
            "monsters_blacklisted": 0,
            "config_overrides": 0,
        }

    # ── Crisis Detection ──────────────────────────────────────────

    def detect_crisis(
        self,
        bot_id: str,
        signals: dict[str, Any],
    ) -> CrisisEvent | None:
        """Detect if a crisis is occurring based on signals.

        Args:
            bot_id: The bot to check.
            signals: Bot state signals (from snapshot).

        Returns:
            CrisisEvent if a crisis is detected, None otherwise.
        """
        with self._lock:
            # Check cooldown — don't re-detect same crisis type too fast
            now = time.time()

            # Death detection
            hp = signals.get("hp", 100)
            hp_max = signals.get("hp_max", 1)
            hp_ratio = signals.get("hp_ratio", 1.0)
            recent_death = signals.get("recent_death", False)
            map_name = signals.get("map", "")

            if recent_death or (hp == 0 and hp_max > 0):
                return self._create_crisis(
                    bot_id=bot_id,
                    crisis_type="death",
                    severity=8,
                    map_name=map_name,
                    context={"hp": hp, "hp_max": hp_max, "hp_ratio": hp_ratio},
                )

            # Stuck detection
            is_sitting = signals.get("is_sitting", False)
            aggro_count = signals.get("combat.aggro_count", 0)
            weight_ratio = signals.get("weight_ratio", 0.0)

            if is_sitting and aggro_count > 0:
                return self._create_crisis(
                    bot_id=bot_id,
                    crisis_type="stuck",
                    severity=6,
                    map_name=map_name,
                    context={"is_sitting": is_sitting, "aggro_count": aggro_count},
                )

            # Overweight detection
            if weight_ratio > 0.95:
                return self._create_crisis(
                    bot_id=bot_id,
                    crisis_type="overweight",
                    severity=4,
                    map_name=map_name,
                    context={"weight_ratio": weight_ratio},
                )

            # No heal items detection
            inventory = signals.get("inventory", {})
            items = inventory.get("items", []) if isinstance(inventory, dict) else []
            has_potions = any(
                "potion" in str(item.get("name", "")).lower() or
                "red" in str(item.get("name", "")).lower()
                for item in items
            )
            if hp_ratio < 0.5 and not has_potions:
                return self._create_crisis(
                    bot_id=bot_id,
                    crisis_type="no_heal_items",
                    severity=7,
                    map_name=map_name,
                    context={"hp_ratio": hp_ratio, "has_potions": has_potions},
                )

            return None

    def _create_crisis(
        self,
        bot_id: str,
        crisis_type: str,
        severity: int,
        map_name: str = "",
        context: dict[str, Any] | None = None,
    ) -> CrisisEvent:
        """Create a crisis event with deduplication."""
        now = time.time()

        # Check cooldown
        last_time = self._cooldowns[bot_id].get(crisis_type, 0)
        if now - last_time < 30:  # 30s cooldown per crisis type per bot
            return None  # Skip, already handling this

        self._cooldowns[bot_id][crisis_type] = now

        # Check for recurrence
        recurrence_key = f"{bot_id}:{crisis_type}"
        existing = self._crisis_history.get(recurrence_key, [])
        recurrence_count = 1
        if existing:
            recurrence_count = existing[-1].recurrence_count + 1

        event = CrisisEvent(
            event_id=f"crisis_{bot_id}_{crisis_type}_{int(now)}",
            bot_id=bot_id,
            crisis_type=crisis_type,
            severity=severity,
            timestamp=now,
            map_name=map_name,
            context=context or {},
            recurrence_count=recurrence_count,
            recurrence_key=recurrence_key,
        )

        self._crisis_history[recurrence_key].append(event)
        self._active_crises[bot_id] = event
        self._stats["crises_detected"] += 1

        logger.warning(
            "crisis_detected: bot=%s type=%s severity=%d map=%s recurrence=%d",
            bot_id, crisis_type, severity, map_name, recurrence_count,
        )

        return event

    # ── Crisis Diagnosis ───────────────────────────────────────────

    def diagnose(self, event: CrisisEvent) -> CrisisDiagnosis:
        """Diagnose a crisis event and produce a recovery plan."""
        self._stats["crises_diagnosed"] += 1

        if event.crisis_type == "death":
            return self._diagnose_death(event)
        elif event.crisis_type == "stuck":
            return self._diagnose_stuck(event)
        elif event.crisis_type == "overweight":
            return self._diagnose_overweight(event)
        elif event.crisis_type == "no_heal_items":
            return self._diagnose_no_heal(event)
        else:
            return CrisisDiagnosis(
                root_cause=f"Unknown crisis type: {event.crisis_type}",
                recommended_actions=["ai auto"],
                lesson="Unknown crisis encountered.",
            )

    def _diagnose_death(self, event: CrisisEvent) -> CrisisDiagnosis:
        """Diagnose a death event."""
        context = event.context
        hp_ratio = context.get("hp_ratio", 0)
        map_name = event.map_name
        recurrence = event.recurrence_count

        factors: list[str] = []
        actions: list[str] = []
        config_changes: dict[str, Any] = {}
        map_blacklist: list[str] = []
        monster_blacklist: list[str] = []
        lesson = ""

        # Check for overpulled
        aggro = context.get("aggro_count", 0)
        if aggro >= 5:
            factors.append(f"Overpulled: {aggro} monsters")
            config_changes["max_aggro"] = max(1, aggro - 2)
            actions.append(f"Reduce max_aggro to {config_changes['max_aggro']}")
            lesson = f"Died to overpull ({aggro} mobs). Reducing max_aggro."

        # Check for no potions
        has_potions = context.get("has_potions", True)
        if not has_potions:
            factors.append("No potions available")
            actions.append("Restock potions before returning to farm")
            lesson = "Died without potions. Ensure restock before farming."

        # Check for boss kill
        monster = context.get("monster_name", "")
        if monster and any(kw in monster.lower() for kw in ("mvp", "boss", "mini", "lord")):
            factors.append(f"Killed by boss: {monster}")
            monster_blacklist.append(monster)
            actions.append(f"Blacklist {monster} for {recurrence * 3600}s")
            lesson = f"Killed by {monster}. Blacklisting temporarily."

        # Recurrence escalation
        if recurrence >= 3:
            factors.append(f"Recurring deaths: {recurrence}x on {map_name}")
            map_blacklist.append(map_name)
            actions.append(f"Blacklist {map_name} for {recurrence * 3600}s")
            config_changes["lockMap"] = ""
            lesson = f"Died {recurrence}x on {map_name}. Blacklisting map."
            self._stats["crises_escalated"] += 1

        if recurrence >= 5:
            factors.append("Critical: 5+ deaths, switching to safe mode")
            actions.append("Switch to safe mode: low-level map, full potions")
            config_changes["safe_mode"] = True
            config_changes["lockMap"] = "prontera"
            lesson = "5+ deaths. Entering safe mode."

        if not factors:
            factors.append("Unknown death cause")
            actions.append("Respawn, rebuff, return to last safe map")
            lesson = "Death with unknown cause. Standard recovery."

        return CrisisDiagnosis(
            root_cause=f"Death on {map_name} (HP ratio was {hp_ratio:.0%})",
            contributing_factors=factors,
            recommended_actions=actions,
            config_changes=config_changes,
            map_blacklist=map_blacklist,
            monster_blacklist=monster_blacklist,
            cooldown_seconds=min(recurrence * 60, 600),  # 1-10 min cooldown
            lesson=lesson,
        )

    def _diagnose_stuck(self, event: CrisisEvent) -> CrisisDiagnosis:
        """Diagnose a stuck event."""
        context = event.context
        is_sitting = context.get("is_sitting", False)
        aggro = context.get("aggro_count", 0)
        recurrence = event.recurrence_count

        factors: list[str] = []
        actions: list[str] = []

        if is_sitting and aggro > 0:
            factors.append(f"Sitting with {aggro} aggro")
            actions.append("Stand up and teleport")
            actions.append("Use Fly Wing or emergency teleport")

        if recurrence >= 3:
            factors.append(f"Recurring stuck: {recurrence}x")
            actions.append("Change map to avoid stuck spot")
            self._stats["crises_escalated"] += 1

        return CrisisDiagnosis(
            root_cause="Bot is stuck (sitting with aggro)",
            contributing_factors=factors,
            recommended_actions=actions,
            cooldown_seconds=30,
            lesson=f"Stuck with {aggro} aggro. Teleporting out.",
        )

    def _diagnose_overweight(self, event: CrisisEvent) -> CrisisDiagnosis:
        """Diagnose an overweight event."""
        weight = event.context.get("weight_ratio", 0)
        return CrisisDiagnosis(
            root_cause=f"Inventory at {weight:.0%} weight capacity",
            contributing_factors=["Inventory full", "Need to sell/store"],
            recommended_actions=[
                "Return to town",
                "Sell junk items to NPC",
                "Store valuable items",
                "Restock potions while in town",
            ],
            requires_town_visit=True,
            requires_restock=True,
            lesson=f"Weight at {weight:.0%}. Return to town to sell/store.",
        )

    def _diagnose_no_heal(self, event: CrisisEvent) -> CrisisDiagnosis:
        """Diagnose a no-heal-items crisis."""
        hp_ratio = event.context.get("hp_ratio", 0)
        return CrisisDiagnosis(
            root_cause=f"No healing items with HP at {hp_ratio:.0%}",
            contributing_factors=["Out of potions", "Autobuy may have failed"],
            recommended_actions=[
                "Emergency teleport to town",
                "Buy potions from NPC",
                "Check autobuy config",
            ],
            requires_town_visit=True,
            requires_restock=True,
            lesson="No potions with low HP. Emergency town visit.",
        )

    # ── Crisis Recovery ────────────────────────────────────────────

    def create_recovery_plan(self, diagnosis: CrisisDiagnosis) -> CrisisRecovery:
        """Create a recovery plan from a diagnosis."""
        steps: list[str] = []
        estimated_time = 0
        requires_town = diagnosis.requires_town_visit
        requires_restock = diagnosis.requires_restock

        # Build recovery steps
        if diagnosis.map_blacklist:
            steps.append(f"Blacklist maps: {', '.join(diagnosis.map_blacklist)}")
            estimated_time += 5

        if diagnosis.monster_blacklist:
            steps.append(f"Blacklist monsters: {', '.join(diagnosis.monster_blacklist)}")
            estimated_time += 5

        if diagnosis.config_changes:
            steps.append(f"Apply config changes: {diagnosis.config_changes}")
            estimated_time += 10

        if requires_town:
            steps.append("Return to town")
            estimated_time += 60

        if requires_restock:
            steps.append("Restock potions and supplies")
            estimated_time += 30

        steps.append("Rebuff and return to farming")
        estimated_time += 30

        return CrisisRecovery(
            steps=steps,
            estimated_time=estimated_time,
            requires_town_visit=requires_town,
            requires_restock=requires_restock,
            priority=diagnosis.cooldown_seconds,
        )

    # ── Crisis Execution ───────────────────────────────────────────

    def execute_recovery(
        self,
        bot_id: str,
        event: CrisisEvent,
        diagnosis: CrisisDiagnosis,
        recovery: CrisisRecovery,
    ) -> bool:
        """Execute a recovery plan for a crisis.

        Returns True if recovery was initiated.
        """
        with self._lock:
            # Apply map blacklist
            for map_name in diagnosis.map_blacklist:
                expiry = time.time() + event.recurrence_count * 3600
                self._map_blacklist[map_name] = expiry
                self._stats["maps_blacklisted"] += 1
                logger.warning("crisis_map_blacklisted: %s for %ds", map_name, event.recurrence_count * 3600)

            # Apply monster blacklist
            for monster in diagnosis.monster_blacklist:
                expiry = time.time() + event.recurrence_count * 3600
                self._monster_blacklist[monster] = expiry
                self._stats["monsters_blacklisted"] += 1
                logger.warning("crisis_monster_blacklisted: %s for %ds", monster, event.recurrence_count * 3600)

            # Apply config overrides
            for key, value in diagnosis.config_changes.items():
                self._config_overrides[bot_id][key] = value
                self._stats["config_overrides"] += 1
                logger.warning("crisis_config_override: bot=%s %s=%s", bot_id, key, value)

            # Store lesson
            if diagnosis.lesson:
                key = event.recurrence_key
                self._lessons[key] = diagnosis.lesson
                self._stats["lessons_learned"] += 1

            # Execute recovery steps via enqueue
            if self._enqueue_fn:
                for step in recovery.steps[:3]:  # Execute top 3 steps
                    try:
                        self._enqueue_fn(bot_id, step)
                    except Exception:
                        pass

            # Mark resolved
            event.resolved = True
            event.resolved_at = time.time()
            event.diagnosis = diagnosis.root_cause
            event.action_taken = "; ".join(recovery.steps[:3])
            self._stats["crises_resolved"] += 1

            # Share lesson with failure reasoning engine
            if diagnosis.lesson and self._failure_reasoning is None:
                try:
                    from ai_sidecar.learning.failure_reasoning import get_failure_reasoning_engine
                    self._failure_reasoning = get_failure_reasoning_engine()
                except Exception:
                    pass

            if self._failure_reasoning and diagnosis.lesson:
                try:
                    self._failure_reasoning.capture_failure(
                        category=event.crisis_type,
                        context={
                            "bot_id": bot_id,
                            "map": event.map_name,
                            "diagnosis": diagnosis.root_cause,
                            "lesson": diagnosis.lesson,
                            "recurrence": event.recurrence_count,
                        },
                        bot_id=bot_id,
                    )
                except Exception:
                    pass

            # Remove from active crises
            if bot_id in self._active_crises:
                del self._active_crises[bot_id]

            return True

    # ── Crisis Prevention ─────────────────────────────────────────

    def is_map_blacklisted(self, map_name: str) -> bool:
        """Check if a map is blacklisted."""
        with self._lock:
            expiry = self._map_blacklist.get(map_name, 0)
            if expiry > time.time():
                return True
            if expiry > 0:  # Expired, clean up
                del self._map_blacklist[map_name]
            return False

    def is_monster_blacklisted(self, monster_name: str) -> bool:
        """Check if a monster is blacklisted."""
        with self._lock:
            expiry = self._monster_blacklist.get(monster_name, 0)
            if expiry > time.time():
                return True
            if expiry > 0:
                del self._monster_blacklist[monster_name]
            return False

    def get_config_overrides(self, bot_id: str) -> dict[str, Any]:
        """Get config overrides for a bot."""
        with self._lock:
            return dict(self._config_overrides.get(bot_id, {}))

    def get_lesson(self, recurrence_key: str) -> str:
        """Get the learned lesson for a recurrence key."""
        with self._lock:
            return self._lessons.get(recurrence_key, "")

    def get_recent_crises(self, bot_id: str, limit: int = 10) -> list[CrisisEvent]:
        """Get recent crises for a bot."""
        with self._lock:
            all_crises = []
            for key, events in self._crisis_history.items():
                if key.startswith(bot_id):
                    all_crises.extend(events)
            return sorted(all_crises, key=lambda e: e.timestamp, reverse=True)[:limit]

    # ── Summary / Context ──────────────────────────────────────────

    def get_crisis_summary(self) -> str:
        """Get a formatted summary of crisis management state."""
        with self._lock:
            lines = ["── Crisis Manager ──"]

            # Active crises
            if self._active_crises:
                lines.append(f"  Active crises: {len(self._active_crises)}")
                for bot_id, event in list(self._active_crises.items())[:3]:
                    lines.append(
                        f"    {bot_id}: {event.crisis_type} (sev={event.severity}) "
                        f"on {event.map_name}"
                    )
            else:
                lines.append("  No active crises.")

            # Blacklisted maps
            now = time.time()
            active_blacklists = {m: e for m, e in self._map_blacklist.items() if e > now}
            if active_blacklists:
                lines.append(f"  Blacklisted maps: {len(active_blacklists)}")
                for map_name, expiry in list(active_blacklists.items())[:3]:
                    remaining = int(expiry - now)
                    lines.append(f"    {map_name} ({remaining}s remaining)")

            # Lessons learned
            if self._lessons:
                lines.append(f"  Lessons learned: {len(self._lessons)}")
                for key, lesson in list(self._lessons.items())[:3]:
                    lines.append(f"    {key}: {lesson[:60]}...")

            # Stats
            lines.append(f"  Stats: {self._stats['crises_detected']} detected, "
                         f"{self._stats['crises_resolved']} resolved, "
                         f"{self._stats['lessons_learned']} lessons")

            return "\n".join(lines)

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def set_enqueue_fn(self, fn: Callable) -> None:
        """Wire the recovery-step execution hook (so execute_recovery can push commands to the bot)."""
        with self._lock:
            self._enqueue_fn = fn

    # ── Persistence ──

    def save_state(self) -> int:
        """Save all crisis manager state to persistent storage."""
        from ai_sidecar.persistence.strategy_state import StrategyStateDB
        with self._lock:
            data = {
                "crisis_history": {
                    k: [{
                        "event_id": e.event_id,
                        "bot_id": e.bot_id,
                        "crisis_type": e.crisis_type,
                        "severity": e.severity,
                        "timestamp": e.timestamp,
                        "map_name": e.map_name,
                        "context": e.context,
                        "diagnosis": e.diagnosis,
                        "action_taken": e.action_taken,
                        "resolved": e.resolved,
                        "resolved_at": e.resolved_at,
                        "recurrence_count": e.recurrence_count,
                        "recurrence_key": e.recurrence_key,
                    } for e in v]
                    for k, v in self._crisis_history.items()
                },
                "active_crises": {
                    k: {
                        "event_id": v.event_id,
                        "bot_id": v.bot_id,
                        "crisis_type": v.crisis_type,
                        "severity": v.severity,
                        "timestamp": v.timestamp,
                        "map_name": v.map_name,
                        "context": v.context,
                        "diagnosis": v.diagnosis,
                        "action_taken": v.action_taken,
                        "resolved": v.resolved,
                        "resolved_at": v.resolved_at,
                        "recurrence_count": v.recurrence_count,
                        "recurrence_key": v.recurrence_key,
                    }
                    for k, v in self._active_crises.items()
                },
                "lessons": dict(self._lessons),
                "map_blacklist": dict(self._map_blacklist),
                "monster_blacklist": dict(self._monster_blacklist),
                "config_overrides": {
                    k: dict(v) for k, v in self._config_overrides.items()
                },
                "cooldowns": {
                    k: dict(v) for k, v in self._cooldowns.items()
                },
                "stats": dict(self._stats),
            }
            return StrategyStateDB.save_crisis_manager(data)

    def load_state(self) -> bool:
        """Load crisis manager state from persistent storage."""
        from ai_sidecar.persistence.strategy_state import StrategyStateDB
        data = StrategyStateDB.load_crisis_manager()
        if data is None:
            return False
        with self._lock:
            self._crisis_history.clear()
            for bot_id, events in data.get("crisis_history", {}).items():
                for e_data in events:
                    self._crisis_history[bot_id].append(CrisisEvent(**e_data))
            self._active_crises.clear()
            for bot_id, e_data in data.get("active_crises", {}).items():
                self._active_crises[bot_id] = CrisisEvent(**e_data)
            self._lessons = dict(data.get("lessons", {}))
            self._map_blacklist = dict(data.get("map_blacklist", {}))
            self._monster_blacklist = dict(data.get("monster_blacklist", {}))
            self._config_overrides.clear()
            for bot_id, overrides in data.get("config_overrides", {}).items():
                self._config_overrides[bot_id] = defaultdict(dict, overrides)
            self._cooldowns.clear()
            for bot_id, cds in data.get("cooldowns", {}).items():
                self._cooldowns[bot_id] = dict(cds)
            saved_stats = data.get("stats", {})
            for k, v in saved_stats.items():
                if k in self._stats:
                    self._stats[k] = v
            logger.info("crisis_manager_state_loaded: %d crises, %d lessons",
                        sum(len(v) for v in self._crisis_history.values()), len(self._lessons))
            return True


# ── Global Singleton ──

_cm: CrisisManager | None = None
_cm_lock = RLock()


def get_crisis_manager() -> CrisisManager:
    global _cm
    with _cm_lock:
        if _cm is None:
            _cm = CrisisManager()
        return _cm
