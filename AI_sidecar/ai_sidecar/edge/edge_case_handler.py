"""
Edge case handler — contingency plans for unexpected situations.

A top player has seen everything. GM appears, PKer hunting, server
crash, patch changes, economy bubble, guild war. This module provides
reflex-level responses for non-combat emergencies.

Each edge case has a detection trigger and a response plan.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ContingencyPlan:
    """A plan for handling an edge case."""
    trigger: str  # gm_spotted, pker_detected, server_unstable, patch_change, etc.
    priority: int  # 1 (critical) to 10 (minor)
    actions: list[str] = field(default_factory=list)
    cooldown_seconds: int = 300
    enabled: bool = True


@dataclass(slots=True)
class EdgeCaseHandler:
    """Detects and responds to edge cases."""
    
    _lock: RLock = field(default_factory=RLock)
    _plans: dict[str, ContingencyPlan] = field(default_factory=dict)
    _active_alerts: list[dict[str, Any]] = field(default_factory=list)
    _last_trigger: dict[str, float] = field(default_factory=dict)
    _stats: dict[str, int] = field(default_factory=lambda: {"alerts": 0, "responses": 0})
    _enqueue_fn: Callable | None = None
    
    def __post_init__(self):
        self._init_plans()
    
    def _init_plans(self) -> None:
        self._plans = {
            "gm_spotted": ContingencyPlan(
                trigger="gm_spotted",
                priority=1,
                actions=[
                    "attackAuto 0",
                    "sit",
                    "log out",
                ],
                cooldown_seconds=600,
            ),
            "pker_detected": ContingencyPlan(
                trigger="pker_detected",
                priority=2,
                actions=[
                    "attackAuto 0",
                    "tele",
                    "change map",
                ],
                cooldown_seconds=300,
            ),
            "server_unstable": ContingencyPlan(
                trigger="server_unstable",
                priority=3,
                actions=[
                    "attackAuto 0",
                    "sit",
                    "wait 60s",
                ],
                cooldown_seconds=120,
            ),
            "suspicious_player": ContingencyPlan(
                trigger="suspicious_player",
                priority=4,
                actions=[
                    "attackAuto 0",
                    "observe",
                    "log if follows",
                ],
                cooldown_seconds=180,
            ),
            "economy_crash": ContingencyPlan(
                trigger="economy_crash",
                priority=5,
                actions=[
                    "stop selling",
                    "hold items",
                    "wait for recovery",
                ],
                cooldown_seconds=3600,
            ),
            "competition_arrived": ContingencyPlan(
                trigger="competition_arrived",
                priority=6,
                actions=[
                    "observe",
                    "share spot or leave",
                    "log if hostile",
                ],
                cooldown_seconds=300,
            ),
        }
    
    def trigger(self, alert_type: str, detail: str = "") -> list[str] | None:
        """Trigger a contingency plan. Returns actions to execute."""
        now = time.time()
        
        plan = self._plans.get(alert_type)
        if not plan or not plan.enabled:
            return None
        
        # Check cooldown
        last = self._last_trigger.get(alert_type, 0)
        if now - last < plan.cooldown_seconds:
            return None
        
        self._last_trigger[alert_type] = now
        
        with self._lock:
            self._active_alerts.append({
                "type": alert_type,
                "detail": detail,
                "timestamp": now,
                "priority": plan.priority,
            })
            self._stats["alerts"] += 1
            self._stats["responses"] += 1
        
        logger.warning("edge_case_triggered: %s — %s", alert_type, detail)
        
        # Execute actions via enqueue
        if self._enqueue_fn:
            for action in plan.actions:
                try:
                    self._enqueue_fn("default", action)
                except Exception:
                    pass
        
        return plan.actions
    
    def get_active_alerts(self, max_age_seconds: int = 300) -> list[dict[str, Any]]:
        """Get recent active alerts."""
        now = time.time()
        with self._lock:
            return [a for a in self._active_alerts if now - a.get("timestamp", 0) < max_age_seconds]
    
    def get_edge_context(self) -> str:
        """Get formatted edge case context for LLM prompts."""
        alerts = self.get_active_alerts()
        if not alerts:
            return ""
        
        lines = ["── Active Edge Cases ──"]
        for a in alerts:
            lines.append(f"  [{a['priority']}] {a['type']}: {a.get('detail', '')}")
        return "\n".join(lines)
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# ── Global instance ──

_handler: EdgeCaseHandler | None = None
_handler_lock = RLock()


def get_edge_handler() -> EdgeCaseHandler:
    global _handler
    with _handler_lock:
        if _handler is None:
            _handler = EdgeCaseHandler()
        return _handler
