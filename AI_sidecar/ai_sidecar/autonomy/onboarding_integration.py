
"""Single-call onboarding integration for PDCA loop.

Usage in pdca_loop.py:
    _total_actions += try_onboarding(context)
    
Returns number of actions queued (0 if not needed or error).
"""
from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular deps
_ONB_SERVICE = None


def try_onboarding(runtime: Any, bot_id: str | None, snapshot: Any = None) -> int:
    """Check if bot needs onboarding (new character setup).
    
    If yes, queue stat/skill/NPC/move actions and return count.
    If no onboarding needed, return 0 (cold start proceeds normally).
    """
    global _ONB_SERVICE
    if _ONB_SERVICE is None:
        from ai_sidecar.autonomy.onboarding_service import OnboardingService
        _ONB_SERVICE = OnboardingService()
    
    if not bot_id:
        return 0
    
    # Get latest snapshot (passed directly from pdca loop)
    _latest = snapshot
    if not _latest:
        return 0
    
    _prog = getattr(_latest, "progression", None)
    if not _prog:
        return 0
    
    # Check if onboarding is already complete
    if _ONB_SERVICE.is_complete(bot_id):
        return 0
    
    _onb_actions = _ONB_SERVICE.evaluate(bot_id, _prog)
    if not _onb_actions:
        return 0
    
    # Queue actions
    _aq = getattr(runtime, "action_queue", None)
    if _aq is None:
        return 0
    
    from ai_sidecar.contracts.actions import ActionProposal, ActionPriorityTier
    
    _count = 0
    for _oa in _onb_actions:
        _act_type = _oa.get("action", "")
        if _act_type == "stat_add":
            _cmd = f"stat_add {_oa.get('stat', 'str').lower()} {_oa.get('points', 1)}"
        elif _act_type == "skill_add":
            _cmd = f"skills add {_oa.get('skill_id', 'NV_BASIC')}"
        elif _act_type == "move":
            _cmd = f"move {_oa.get('target_map', 'prt_fild01')}"
        else:
            continue
        
        _prop = ActionProposal(
            action_id=f"onb_{bot_id}_{int(time.monotonic()*1000)}",
            kind="command",
            command=_cmd,
            priority_tier=ActionPriorityTier.reflex,
            source="bridge",
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(seconds=60),
            conflict_key=f"onb_{_act_type}_{bot_id}",
            idempotency_key=f"onb_{_act_type}_{bot_id}",
            metadata={
                "source": "onboarding",
                "reason": _oa.get("reason", ""),
                "action_type": _act_type,
                "bot_id": bot_id,
            },
        )
        _aq.enqueue(bot_id, _prop)
        _count += 1
        
        # Mark step as completed
        if bot_id not in _ONB_SERVICE._completed_steps:
            _ONB_SERVICE._completed_steps[bot_id] = set()
        _ONB_SERVICE._completed_steps[bot_id].add(_act_type)
    
    logger.info("onboarding_actions_queued: bot=%s count=%d", bot_id, _count)
    return _count
