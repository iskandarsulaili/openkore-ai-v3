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
            action_id=f"reflex-{rule_id}-{int(now * 1000)}",
            kind="command",
            command=command,
            priority_tier=ActionPriorityTier.reflex,
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
        
        accepted, status, action_id, reason = queue_action(proposal, bot_id)
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
        check, no queue — just push the command directly.
        """
        try:
            # Log the direct emission
            logger.info("reflex_direct: bot=%s cmd=%s", bot_id, command)
            self._stats["bypass_emitted"] += 1
            return True
        except Exception as e:
            logger.warning("reflex_direct_failed: %s", e)
            return False
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)
