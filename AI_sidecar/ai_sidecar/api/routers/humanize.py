"""Route Humanizer API — bridge-facing endpoint to humanize movement waypoints.

The bridge intercepts `move x y` commands and calls this endpoint before
execution. Returns slightly perturbed coordinates that look human-like
rather than the exact grid-aligned coordinates bots produce.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter

from ai_sidecar.anti_detection.route_humanizer import get_route_humanizer
from ai_sidecar.anti_detection.bridge_wiring import get_bridge_wiring

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/humanize", tags=["humanize"])


@router.post("/move")
def humanize_move(payload: dict[str, Any]) -> dict[str, Any]:
    """Humanize a movement waypoint.

    Accepts the bot's current position and target, returns
    a slightly perturbed target that mimics human movement.

    Request:
        {
            "bot_id": "master:username",
            "current_x": float,
            "current_y": float,
            "target_x": float,
            "target_y": float
        }

    Response:
        {
            "humanized_x": float,
            "humanized_y": float,
            "deviation": float,
            "humanized": True
        }

    If route_humanizer is disabled or unavailable, returns original coords.
    """
    bot_id = payload.get("bot_id", "default")
    current_x = float(payload.get("current_x", 0))
    current_y = float(payload.get("current_y", 0))
    target_x = float(payload.get("target_x", 0))
    target_y = float(payload.get("target_y", 0))

    try:
        rh = get_route_humanizer()
        noisy_x, noisy_y = rh.humanize_waypoint(
            bot_id, current_x, current_y, target_x, target_y,
        )
        # Calculate deviation distance for logging
        import math
        deviation = math.sqrt((noisy_x - target_x) ** 2 + (noisy_y - target_y) ** 2)
        return {
            "humanized_x": round(noisy_x, 1),
            "humanized_y": round(noisy_y, 1),
            "deviation": round(deviation, 2),
            "humanized": (noisy_x != target_x or noisy_y != target_y),
        }
    except Exception as e:
        logger.warning(f"Route humanize failed (returning original): {e}")
        return {
            "humanized_x": target_x,
            "humanized_y": target_y,
            "deviation": 0.0,
            "humanized": False,
            "error": str(e),
        }


@router.get("/status")
def humanize_status() -> dict[str, Any]:
    """Check if route humanizer is available and configured."""
    try:
        rh = get_route_humanizer()
        return {
            "available": True,
            "enabled": rh.config.enabled if hasattr(rh, 'config') else True,
            "deviation_strength": rh.config.deviation_strength if hasattr(rh, 'config') else 0.5,
        }
    except Exception as e:
        return {
            "available": False,
            "enabled": False,
            "error": str(e),
        }
