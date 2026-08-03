"""
Reflex action pipeline — direct, high-speed action emission bypassing the arbiter.

A pro player doesn't wait for permission to potion when HP drops.
They react instantly. This module provides a bypass path for reflex actions
that skips the conflict-checking arbiter entirely for high-priority reflexes.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable
from uuid import uuid4

from ai_sidecar.contracts.actions import ActionPriorityTier, ActionProposal, ActionStatus
from ai_sidecar.contracts.reflex import ReflexRule

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ReflexPipeline:
    """High-speed reflex action pipeline with arbiter bypass for critical actions."""
    
    _lock: RLock = field(default_factory=RLock)
    _last_emission: dict[str, float] = field(default_factory=dict)
    _emission_cooldown: dict[str, float] = field(default_factory=lambda: defaultdict(lambda: 1.0))
    _stats: dict[str, int] = field(default_factory=lambda: {"direct_emitted": 0, "bypass_emitted": 0, "cooldown_blocked": 0})
    _action_queue: object | None = field(default=None)
    
    def set_action_queue(self, queue: object) -> None:
        """Set the action queue reference for direct emissions."""
        self._action_queue = queue
    
    def emit(self, bot_id: str, rule: ReflexRule, command: str,
             queue_action: Callable, bypass_arbiter: Callable | None = None) -> dict[str, Any]:
        """Emit a reflex action with cooldown and priority handling.
        
        For critical reflexes (HP < 30%, lethal threat), bypasses the arbiter
        entirely and pushes directly to the bot's command queue.
        For normal reflexes, uses the arbiter with no conflict key.
        """
        now = time.time()
        rule_id = rule.rule_id
        
        # Cooldown check
        last = self._last_emission.get(rule_id, 0.0)
        cooldown = self._emission_cooldown.get(rule_id, 1.0)
        if now - last < cooldown:
            self._stats["cooldown_blocked"] += 1
            return {"emitted": False, "reason": f"cooldown_{cooldown}s", "target": "none"}
        
        # Determine if this is a critical reflex (bypass arbiter)
        is_critical = rule.priority >= 90 or "hp_emergency" in rule_id or "lethal" in rule_id
        
        if is_critical and bypass_arbiter is not None:
            try:
                bypass_arbiter(bot_id, command)
                self._last_emission[rule_id] = now
                self._stats["bypass_emitted"] += 1
                return {"emitted": True, "reason": "bypass_arbiter_critical", "target": "direct"}
            except Exception as e:
                logger.warning("reflex_bypass_failed: %s", e)
        
        # Normal path: queue with no conflict key (allows multiple reflex actions)
        proposal = ActionProposal(
            action_id=f"reflex-{rule_id}-{uuid4().hex[:16]}",
            kind="command",
            command=command,
            priority_tier=ActionPriorityTier.reflex,
            source="reflex",
            conflict_key="",
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(seconds=5),
            idempotency_key=f"reflex:{rule_id}:{int(now)}",
            metadata={
                "source": "reflex",
                "latency_budget_ms": 50,
                "reflex_rule_id": rule_id,
                "reflex_priority": rule.priority,
            },
        )
        
        accepted, status, action_id, reason = queue_action(bot_id, proposal)
        if accepted:
            self._last_emission[rule_id] = now
            self._stats["direct_emitted"] += 1
            return {"emitted": True, "reason": "action_queued", "target": "arbiter", "action_id": action_id}
        
        return {"emitted": False, "reason": f"queue_rejected:{reason}", "target": "arbiter"}
    
    def set_cooldown(self, rule_id: str, seconds: float) -> None:
        with self._lock:
            self._emission_cooldown[rule_id] = seconds
    
    def emit_direct(self, bot_id: str, command: str) -> bool:
        """Direct emission bypass — pushes command straight to bot without arbiter.
        
        This is the fastest path for critical reflexes. No proposal, no conflict
        check, no queue — just push the command directly to the bot's action queue
        with a reflex-priority proposal that bypasses all checks.
        """
        try:
            now = time.time()
            proposal = ActionProposal(
                action_id=f"reflex-direct-{uuid4().hex[:16]}",
                kind="command",
                command=command,
                priority_tier=ActionPriorityTier.reflex,
                conflict_key="",
                created_at=datetime.now(UTC),
                expires_at=datetime.now(UTC) + timedelta(seconds=3),
                idempotency_key=f"reflex:direct:{int(now)}",
                metadata={
                    "source": "reflex",
                    "latency_budget_ms": 10,
                    "reflex_direct": True,
                },
            )
            if self._action_queue is not None:
                accepted, status, action_id, reason = self._action_queue.enqueue(bot_id, proposal)
                if accepted:
                    logger.info("reflex_direct: bot=%s cmd=%s action_id=%s", bot_id, command, action_id)
                    self._stats["bypass_emitted"] += 1
                    return True
                else:
                    logger.warning("reflex_direct_queue_rejected: bot=%s cmd=%s reason=%s", bot_id, command, reason)
                    return False
            else:
                logger.warning("reflex_direct_no_queue: bot=%s cmd=%s", bot_id, command)
                return False
        except Exception as e:
            logger.warning("reflex_direct_failed: %s", e)
            return False
    
    def emit_test(self, bot_id: str) -> dict[str, Any]:
        """Emit a test reflex to verify the pipeline works end-to-end."""
        from ai_sidecar.contracts.reflex import ReflexRule, ReflexActionTemplate, ReflexTriggerClause, ReflexCategory, ReflexPlannerInterop
        from ai_sidecar.contracts.actions import ActionPriorityTier
        
        rule = ReflexRule(
            rule_id="test_reflex",
            priority=80,
            trigger=ReflexTriggerClause(all=[]),
            action_template=ReflexActionTemplate(
                command="stand",
                kind="command",
                conflict_key="",
                priority_tier=ActionPriorityTier.reflex,
            ),
            category=ReflexCategory.survival,
            planner_interop=ReflexPlannerInterop.override,
        )
        
        result = self.emit(bot_id, rule, "ai manual", lambda b, p: (True, None, "test_action_id", "test"))
        logger.info("reflex_pipeline_test: bot=%s result=%s", bot_id, result)
        return result
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)
