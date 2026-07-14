"""
Combat API endpoints — threat-based target selection for the bridge.
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, Depends

from ai_sidecar.api.deps import get_runtime
from ai_sidecar.lifecycle import RuntimeState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v2/combat", tags=["combat"])


@router.post("/target")
def get_combat_target(
    runtime: RuntimeState = Depends(get_runtime),
):
    """Get the optimal combat target based on threat analysis."""
    try:
        targeting = getattr(runtime, "threat_targeting", None)
        if targeting is None:
            return {"ok": False, "error": "threat_targeting_not_initialized"}
        
        target = targeting.get_best_target()
        if target is None:
            return {"ok": True, "has_target": False}
        
        return {
            "ok": True,
            "has_target": True,
            "target": target,
        }
    except Exception as e:
        logger.warning("combat_target_failed: %s", e)
        return {"ok": False, "error": str(e)}
