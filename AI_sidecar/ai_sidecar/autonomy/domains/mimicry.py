"""Mimicry domain — human-like behavior, anti-detection.

Extracted from heuristic_service.py lines 1978-1993 (randomized config
per bot), sitting detector 2066-2095, and anti-detection behaviors.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from ai_sidecar.autonomy.domains import BaseDomain
from ai_sidecar.autonomy.heuristic_service import HeuristicAction

logger = logging.getLogger(__name__)


class MimicryDomain(BaseDomain):
    name: str = "mimicry"
    priority: int = 60

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        service: Any,
    ) -> None:
        """Evaluate human-like behavior decisions.

        Handles: sitting detector (force stand), randomized pacing,
        avoidList management, human-like movement patterns.
        """
        bot_id = service._resolve_bot_id(signals)
        map_name = str(signals.get("map", "") or "").lower()

        # ── SITTING ON HUNTING MAP DETECTOR ──
        self._check_sitting(actions, signals, bot_id, map_name, service)

        # ── AVOID SYSTEM: disable on hunting maps ──
        self._apply_avoid_config(actions, bot_id, map_name, service)

    def _is_hunting_map(self, map_name: str) -> bool:
        return any(x in map_name.lower() for x in [
            "prt_fild", "pay_fild", "mjolnir", "gef_fild",
            "ra_fild", "moc_fild", "cmd_fild",
        ])

    def _check_sitting(
        self,
        actions: list[HeuristicAction],
        signals: dict[str, Any],
        bot_id: str,
        map_name: str,
        service: Any,
    ) -> None:
        """Force stand if bot has been sitting too long on hunting map."""
        if not self._is_hunting_map(map_name):
            service._sit_start_time[bot_id] = 0
            return

        _is_sitting = signals.get("is_sitting", False)
        if not _is_sitting:
            service._sit_start_time[bot_id] = 0
            return

        _now = __import__("time").time()
        _sit_start = service._sit_start_time.get(bot_id, _now)
        if not service._sit_start_time.get(bot_id):
            service._sit_start_time[bot_id] = _now
            return

        _duration = _now - _sit_start
        _hp = float(signals.get("hp_ratio", 1.0) or 1.0)

        # Force stand if HP > 50% OR sitting > 30s
        if _hp > 0.50 or _duration > 30:
            logger.info(
                "[sit_detector] %s: sitting on %s for %.0fs, HP=%.0f%% — forcing stand",
                bot_id, map_name, _duration, _hp * 100,
            )
            actions.append(HeuristicAction(
                kind="command", command="stand",
                confidence=0.99, domain="survival",
                reason=f"Sitting on hunting map for {_duration:.0f}s "
                       f"with HP={_hp:.0%} - forcing stand",
            ))
            actions.append(HeuristicAction(
                kind="command", command="ai auto",
                confidence=0.99, domain="survival",
                reason="Re-enable auto-attack after forced stand",
            ))
            service._sit_start_time[bot_id] = 0

    def _apply_avoid_config(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        map_name: str,
        service: Any,
    ) -> None:
        """Disable avoid system on hunting maps to prevent running from targets."""
        if not self._is_hunting_map(map_name):
            return

        service._set_config_once(
            actions, bot_id, "avoidList", "", "hunting",
            "Disable avoid system on hunting maps (prevents running from monsters)",
        )
        service._set_config_once(
            actions, bot_id, "avoidList_inLockOnly", "", "hunting",
            "Disable avoid system in lockMap (prevents running from monsters)",
        )


def create_domain() -> MimicryDomain:
    return MimicryDomain()
