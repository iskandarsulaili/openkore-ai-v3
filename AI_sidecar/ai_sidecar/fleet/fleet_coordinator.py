"""FleetCoordinator — real-time multi-bot shared state, role assignment,
knowledge sharing, and auto-coordination with ExperienceDatabase integration."""

from __future__ import annotations

import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Any

from ai_sidecar.experience_db import ExperienceDatabase, ExperienceEntry
from ai_sidecar.fleet.coordinator import (
    FleetCoordinatorService,
    BotFleetState,
    FleetMessage,
    RoleMetrics,
    RoleType,
)

logger = logging.getLogger(__name__)


@dataclass
class SharedGoal:
    """A team-wide goal that bots coordinate toward."""
    goal_id: str
    goal_type: str  # "hunt", "mvp", "quest", "level", "farm", "trade", "pvp", "gvg"
    params: dict[str, Any] = field(default_factory=dict)
    assigned_bots: list[str] = field(default_factory=list)
    priority: int = 5  # 1-10
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    status: str = "active"  # active | completed | failed | cancelled


@dataclass
class SharedKnowledge:
    """Learned patterns and knowledge shared across the fleet."""
    knowledge_type: str  # "hunting_spot", "danger_zone", "mvp_spawn", "safe_route", "price_trend"
    key: str  # Unique identifier within type
    value: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0  # 0.0 to 1.0
    reported_by: str = ""
    reported_at: float = field(default_factory=time.time)
    last_verified_at: float = field(default_factory=time.time)
    verification_count: int = 1


class FleetCoordinator:
    """Central coordinator for multi-bot swarms.

    Manages shared state across all bots in real-time:
    - Bot registry (who is online, their role, position, HP, map)
    - Blackboard (shared knowledge — MVP spawns, good hunting spots, danger zones)
    - Role assignments (who tanks, heals, DPS, crafts, vendors)
    - Shared goals (what the team should accomplish)
    - Performance tracking (per bot per role)
    - Auto role-switching (when a bot performs poorly, switch roles)

    Uses SQLite-backed ExperienceDatabase for cross-bot learning and crash recovery.
    Wraps the existing FleetCoordinatorService for low-level state management.
    """

    def __init__(
        self,
        experience_db: ExperienceDatabase,
        coordinator_service: FleetCoordinatorService | None = None,
        max_bots: int = 256,
        role_rotation_cooldown_s: int = 120,
        min_assignments_for_switch: int = 5,
        score_threshold_for_switch: float = 0.15,
    ):
        self._lock = threading.RLock()
        self._exp_db = experience_db
        self._coord = coordinator_service or FleetCoordinatorService(
            max_bots=max_bots,
            role_rotation_cooldown_s=role_rotation_cooldown_s,
        )
        self._min_assignments_for_switch = min_assignments_for_switch
        self._score_threshold_for_switch = score_threshold_for_switch

        # Shared goals
        self._goals: dict[str, SharedGoal] = {}
        self._goal_counter = 0

        # Shared knowledge
        self._knowledge: dict[str, SharedKnowledge] = {}

        # Auto-coordination state
        self._auto_reassign_enabled: bool = True
        self._last_performance_check: float = 0.0
        self._performance_check_interval_s: float = 60.0

    # ── Bot lifecycle ─────────────────────────────────────────────────

    def register_bot(self, bot_id: str, capabilities: list[str] | None = None) -> dict[str, Any]:
        """Register a bot with the fleet and assign an initial role.

        Args:
            bot_id: Unique bot identifier.
            capabilities: List of roles this bot can perform.

        Returns:
            Dict with bot state and assigned role.
        """
        available_roles = capabilities or []
        state = self._coord.register_bot(bot_id, available_roles=available_roles)

        # Assign best role based on capabilities and past performance
        assigned_role = self._pick_best_role(bot_id, available_roles)
        if assigned_role:
            self._coord.assign_role(bot_id, assigned_role, reason="initial_registration")

        return {
            "bot_id": bot_id,
            "assigned_role": assigned_role or RoleType.IDLE.value,
            "available_roles": available_roles,
            "is_online": True,
        }

    def unregister_bot(self, bot_id: str) -> None:
        """Remove a bot from the fleet."""
        self._coord.unregister_bot(bot_id)

    # ── State management ──────────────────────────────────────────────

    def update_bot_state(self, bot_id: str, state_dict: dict[str, Any]) -> dict[str, Any] | None:
        """Update a bot's live state and return the updated state.

        Acceptable keys in state_dict: position, map_name, hp, hp_max, sp, sp_max,
        level, job_level, zeny, weight, max_weight, status_message, active_objective.
        """
        # Filter to known fields on BotFleetState
        allowed_keys = {
            "position", "map_name", "hp", "hp_max", "sp", "sp_max",
            "level", "job_level", "zeny", "weight", "max_weight",
            "status_message", "active_objective",
        }
        kwargs = {k: v for k, v in state_dict.items() if k in allowed_keys}
        state = self._coord.update_bot_state(bot_id, **kwargs)
        if state is None:
            return None

        # If position was given as list/tuple, convert
        if "position" in kwargs and isinstance(kwargs["position"], (list, tuple)):
            state.position = tuple(kwargs["position"])

        return self._bot_to_dict(state)

    def get_bot_state(self, bot_id: str) -> dict[str, Any] | None:
        """Get a specific bot's state."""
        state = self._coord.get_bot(bot_id)
        if state is None:
            return None
        return self._bot_to_dict(state)

    def get_team_state(self) -> dict[str, Any]:
        """Get the state of all bots in the fleet."""
        return self._coord.fleet_status()

    def _bot_to_dict(self, b: BotFleetState) -> dict[str, Any]:
        return {
            "bot_id": b.bot_id,
            "position": list(b.position),
            "map_name": b.map_name,
            "hp": b.hp,
            "hp_max": b.hp_max,
            "hp_pct": b.hp_pct(),
            "sp": b.sp,
            "sp_max": b.sp_max,
            "sp_pct": b.sp_pct(),
            "level": b.level,
            "job_level": b.job_level,
            "zeny": b.zeny,
            "weight_pct": b.weight_pct(),
            "current_role": b.current_role,
            "available_roles": b.available_roles,
            "is_online": b.is_online,
            "last_seen_at": b.last_seen_at,
            "active_objective": b.active_objective,
            "status_message": b.status_message,
            "role_scores": {r: m.compute_score() for r, m in b.role_metrics.items()},
        }

    # ── Role assignment ───────────────────────────────────────────────

    def assign_role(self, bot_id: str, preferred_role: str) -> dict[str, Any]:
        """Assign the best role for a bot based on preference and past performance.

        Args:
            bot_id: The bot to assign a role to.
            preferred_role: The desired role.

        Returns:
            Dict with assigned_role, was_switched, and reasoning.
        """
        available = self._coord.get_bot(bot_id)
        if available is None:
            return {"assigned_role": None, "error": "bot_not_found"}
        avail_roles = available.available_roles

        # If preferred_role is available, use it
        if preferred_role in avail_roles:
            result = self._coord.assign_role(bot_id, preferred_role, reason="user_request")
            return {"assigned_role": result, "was_switched": result != available.current_role, "reason": "user_request"}

        # Otherwise pick the best available role
        best_role = self._pick_best_role(bot_id, avail_roles)
        if best_role:
            self._coord.assign_role(bot_id, best_role, reason="coordinator_pick")
            return {"assigned_role": best_role, "was_switched": True, "reason": "best_available"}

        # Fallback to first available
        if avail_roles:
            self._coord.assign_role(bot_id, avail_roles[0], reason="fallback")
            return {"assigned_role": avail_roles[0], "was_switched": True, "reason": "fallback"}

        return {"assigned_role": None, "error": "no_roles_available"}

    def _pick_best_role(self, bot_id: str, available: list[str]) -> str | None:
        """Pick best role by consulting ExperienceDatabase."""
        if not available:
            return None
        if len(available) == 1:
            return available[0]

        best_role = None
        best_score = -1.0

        # Query ExperienceDatabase for success rates per role for this bot
        for role in available:
            role_entries = self._exp_db.query(role=role, limit=100)
            if role_entries:
                sr = sum(1 for e in role_entries if e.success) / len(role_entries)
            else:
                sr = 0.5  # neutral default
            score = sr

            # Also check bot's own metrics
            bot = self._coord.get_bot(bot_id)
            if bot:
                m = bot.role_metrics.get(role)
                if m:
                    perf_score = m.compute_score()
                    score = max(score, perf_score)

            if score > best_score:
                best_score = score
                best_role = role

        return best_role

    # ── Experience / Knowledge sharing ────────────────────────────────

    def record_outcome(
        self,
        bot_id: str,
        context_type: str,
        action_taken: str,
        success: bool,
        reward: float = 0.0,
        *,
        map_name: str = "",
        monster_name: str = "",
        role: str = "",
        details: dict[str, Any] | None = None,
        damage: float = 0.0,
        healing: float = 0.0,
        zeny: float = 0.0,
        xp: float = 0.0,
        death: bool = False,
        response_time_s: float = 0.0,
    ) -> None:
        """Record an outcome in ExperienceDatabase for cross-bot learning,
        and also update the bot's role metrics for performance tracking."""
        # Record in ExperienceDatabase (cross-bot)
        entry = ExperienceEntry(
            bot_id=bot_id,
            timestamp=time.time(),
            context_type=context_type,
            map_name=map_name,
            monster_name=monster_name,
            role=role or "",
            action_taken=action_taken,
            success=success,
            reward=reward,
            details=details or {},
        )
        self._exp_db.record(entry)

        # Record in FleetCoordinatorService for per-bot role metrics
        self._coord.record_role_action(
            bot_id=bot_id,
            role=role,
            success=success,
            damage=damage,
            healing=healing,
            zeny=zeny,
            xp=xp,
            death=death,
            response_time_s=response_time_s,
        )

    def get_shared_knowledge(self, knowledge_type: str | None = None) -> list[dict[str, Any]]:
        """Get learned patterns — good hunting spots, danger zones, MVP spawns, etc.

        Args:
            knowledge_type: Optional filter by type (hunting_spot, danger_zone, mvp_spawn, etc.)

        Returns:
            List of shared knowledge entries.
        """
        with self._lock:
            results = list(self._knowledge.values())
            if knowledge_type:
                results = [k for k in results if k.knowledge_type == knowledge_type]

            # Also include ExperienceDatabase insights
            exp_stats = self._exp_db.stats()

            return [
                {
                    "knowledge_type": k.knowledge_type,
                    "key": k.key,
                    "value": k.value,
                    "confidence": k.confidence,
                    "reported_by": k.reported_by,
                    "reported_at": k.reported_at,
                    "last_verified_at": k.last_verified_at,
                    "verification_count": k.verification_count,
                }
                for k in results
            ] + [{
                "knowledge_type": "experience_db_stats",
                "key": "stats",
                "value": exp_stats,
                "confidence": 1.0,
                "reported_by": "system",
                "reported_at": time.time(),
                "last_verified_at": time.time(),
                "verification_count": 1,
            }]

    def add_shared_knowledge(
        self,
        knowledge_type: str,
        key: str,
        value: dict[str, Any],
        reported_by: str = "",
        confidence: float = 1.0,
    ) -> None:
        """Add or update a shared knowledge entry."""
        with self._lock:
            lookup_key = f"{knowledge_type}:{key}"
            existing = self._knowledge.get(lookup_key)
            if existing:
                existing.value.update(value)
                existing.confidence = (existing.confidence + confidence) / 2
                existing.verification_count += 1
                existing.last_verified_at = time.time()
            else:
                self._knowledge[lookup_key] = SharedKnowledge(
                    knowledge_type=knowledge_type,
                    key=key,
                    value=value,
                    confidence=confidence,
                    reported_by=reported_by,
                )

    # ── Goals ─────────────────────────────────────────────────────────

    def set_goal(self, goal_type: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Set a team goal that bots should coordinate toward.

        Args:
            goal_type: Type of goal (hunt, mvp, quest, level, farm, trade, pvp, gvg).
            params: Goal parameters (target, location, quantity, etc.)

        Returns:
            The created goal.
        """
        with self._lock:
            self._goal_counter += 1
            goal_id = f"goal_{int(time.time())}_{self._goal_counter}"
            goal = SharedGoal(
                goal_id=goal_id,
                goal_type=goal_type,
                params=params or {},
                priority=params.get("priority", 5) if params else 5,
            )
            self._goals[goal_id] = goal
            logger.info(
                "fleet_goal_set",
                extra={"event": "fleet_goal_set", "goal_id": goal_id, "goal_type": goal_type},
            )
            return {
                "goal_id": goal_id,
                "goal_type": goal_type,
                "params": goal.params,
                "priority": goal.priority,
                "status": goal.status,
                "created_at": goal.created_at,
            }

    def get_goals(self, status: str | None = "active") -> list[dict[str, Any]]:
        """Get all goals, optionally filtered by status."""
        with self._lock:
            results = list(self._goals.values())
            if status:
                results = [g for g in results if g.status == status]
            return [
                {
                    "goal_id": g.goal_id,
                    "goal_type": g.goal_type,
                    "params": g.params,
                    "assigned_bots": g.assigned_bots,
                    "priority": g.priority,
                    "created_at": g.created_at,
                    "completed_at": g.completed_at,
                    "status": g.status,
                }
                for g in sorted(results, key=lambda x: x.priority, reverse=True)
            ]

    def complete_goal(self, goal_id: str, success: bool = True) -> bool:
        """Mark a goal as completed or failed."""
        with self._lock:
            goal = self._goals.get(goal_id)
            if goal is None:
                return False
            goal.status = "completed" if success else "failed"
            goal.completed_at = time.time()
            return True

    # ── Auto-coordination ─────────────────────────────────────────────

    def auto_reassign(self, bot_id: str | None = None) -> list[dict[str, Any]]:
        """Check all bots (or a specific one) for role performance issues
        and reassign underperforming bots.

        Returns:
            List of reassignment actions taken.
        """
        with self._lock:
            actions: list[dict[str, Any]] = []
            bots_to_check: list[str] = []

            if bot_id:
                bot = self._coord.get_bot(bot_id)
                if bot:
                    bots_to_check = [bot_id]
            else:
                bots_to_check = [b.bot_id for b in self._coord.list_bots(online_only=True)]

            for bid in bots_to_check:
                recommendation = self._coord.recommend_role_change(bid)
                if recommendation.get("should_change"):
                    new_role = recommendation.get("recommended_role")
                    old_role = recommendation.get("current_role")
                    if new_role and new_role != old_role:
                        self._coord.assign_role(bid, new_role, reason="auto_reassign")
                        actions.append({
                            "bot_id": bid,
                            "from_role": old_role,
                            "to_role": new_role,
                            "reason": recommendation.get("reason", "better_performance"),
                            "improvement": recommendation.get("improvement", 0.0),
                        })
                        logger.info(
                            "fleet_auto_reassign",
                            extra={
                                "event": "fleet_auto_reassign",
                                "bot_id": bid,
                                "from_role": old_role,
                                "to_role": new_role,
                            },
                        )

            self._last_performance_check = time.time()
            return actions

    def periodic_performance_check(self) -> list[dict[str, Any]]:
        """Run auto-reassign if enough time has passed since last check."""
        now = time.time()
        if now - self._last_performance_check >= self._performance_check_interval_s:
            return self.auto_reassign()
        return []

    # ── Messaging ──────────────────────────────────────────────────────

    def send_message(
        self,
        sender_id: str,
        recipient_id: str,
        message_type: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a message between bots."""
        msg = FleetMessage(
            message_id=f"msg_{int(time.time() * 1000)}_{sender_id}",
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=message_type,
            payload=payload or {},
        )
        self._coord.send_message(msg)
        return {
            "message_id": msg.message_id,
            "sender_id": msg.sender_id,
            "recipient_id": msg.recipient_id,
            "message_type": msg.message_type,
            "sent_at": msg.sent_at,
        }

    def get_messages(self, bot_id: str, since: float = 0.0) -> list[dict[str, Any]]:
        """Get messages for a bot."""
        msgs = self._coord.get_messages_for(bot_id, since=since)
        return [
            {
                "message_id": m.message_id,
                "sender_id": m.sender_id,
                "recipient_id": m.recipient_id,
                "message_type": m.message_type,
                "payload": m.payload,
                "sent_at": m.sent_at,
            }
            for m in msgs
        ]

    # ── Role claim ────────────────────────────────────────────────────

    def claim_role(self, bot_id: str, role: str) -> dict[str, Any]:
        """Claim a role for a bot."""
        return self.assign_role(bot_id, role)

    # ── Discovery / Metadata ──────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Full fleet status including coordinator metadata."""
        fleet_status = self._coord.fleet_status()
        with self._lock:
            return {
                **fleet_status,
                "auto_reassign_enabled": self._auto_reassign_enabled,
                "goals_active": len([g for g in self._goals.values() if g.status == "active"]),
                "goals_total": len(self._goals),
                "shared_knowledge_entries": len(self._knowledge),
                "experience_db_entries": self._exp_db.size(),
                "min_assignments_for_switch": self._min_assignments_for_switch,
                "score_threshold_for_switch": self._score_threshold_for_switch,
                "performance_check_interval_s": self._performance_check_interval_s,
                "bots_checked": fleet_status.get("online_bots", 0),
            }

    def enable_auto_reassign(self, enabled: bool = True) -> None:
        """Enable or disable automatic role reassignment."""
        self._auto_reassign_enabled = enabled

    def _get_coord(self) -> FleetCoordinatorService:
        return self._coord
