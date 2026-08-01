"""Equipment domain — gear management and upgrades.

Extracted from heuristic_service.py lines 693-704 (get_optimal_weapon),
3285-3294 (equipment progression check).
"""

from __future__ import annotations

import logging
from typing import Any

from ai_sidecar.autonomy.domains import BaseDomain
from ai_sidecar.autonomy.heuristic_service import HeuristicAction

logger = logging.getLogger(__name__)


class EquipmentDomain(BaseDomain):
    name: str = "equipment"
    priority: int = 45

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        service: Any,
    ) -> None:
        """Evaluate equipment upgrade decisions.

        Checks if bot should upgrade weapon based on level and zeny.
        The actual buy happens in WEAPON_BUY state on next town visit.
        """
        bot_id = service._resolve_bot_id(signals)
        job_name = str(signals.get("job_name", "novice") or "novice").lower()
        base_level = int(signals.get("base_level", 1) or 1)
        zeny = int(signals.get("zeny", 0) or 0)
        hunt_duration = (
            __import__("time").time()
            - service._state_since.get(bot_id, __import__("time").time())
        )
        total_kills = int(signals.get("kills", 0) or 0)

        # Check equipment progression
        _prog = service._adaptive.equipment_progression.get(job_name, [])
        _best = None
        for _lvl, _wid, _desc in _prog:
            if base_level >= _lvl:
                _best = (_wid, _desc)

        if (
            _best
            and zeny >= 100
            and hunt_duration > 60
            and total_kills > 5
        ):
            # Log potential upgrade — actual buy handled by economy domain
            logger.info(
                "[equipment] %s: can upgrade to %s (%s) at level %d",
                bot_id, _best[0], _best[1], base_level,
            )

        # No-weapon detection on hunting map
        _atk_power = int(signals.get("attack_power", 0) or 0)
        _equip = signals.get("equipment", {}) or {}
        _has_weapon_equipped = any(
            "weapon" in k.lower()
            for k in (_equip.keys() if isinstance(_equip, dict) else [])
        )
        if not _has_weapon_equipped and _atk_power < 10 and zeny >= 100:
            actions.append(HeuristicAction(
                kind="log", command="ai_mode_auto",
                confidence=0.5, domain="planning",
                reason="No weapon detected - go buy one [log-only: config-audit owns AI mode]",
                metadata={"ai_mode": "auto"},
            ))


def create_domain() -> EquipmentDomain:
    return EquipmentDomain()
