"""Consumables domain — buffs, potions, food management.

Extracted from heuristic_service.py lines 3230-3239 (food/buff system),
2391-2417 (cold start potions/arrows), 2501-2519 (death recovery potions).
"""

from __future__ import annotations

import logging
from typing import Any

from ai_sidecar.autonomy.domains import BaseDomain
from ai_sidecar.autonomy.heuristic_service import HeuristicAction

logger = logging.getLogger(__name__)


class ConsumablesDomain(BaseDomain):
    name: str = "consumables"
    priority: int = 50

    # Stat -> food item ID mapping
    FOOD_ITEMS: dict[str, str] = {
        "str": "531",
        "agi": "532",
        "vit": "533",
        "int": "534",
        "dex": "535",
        "luk": "536",
    }

    # Job -> primary stat for food
    JOB_PRIMARY_STAT: dict[str, str] = {
        "archer": "dex", "hunter": "dex",
        "thief": "agi", "assassin": "agi",
        "swordman": "str", "knight": "str",
        "mage": "int", "wizard": "int",
        "acolyte": "int", "priest": "int",
        "merchant": "str", "blacksmith": "str",
    }

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        service: Any,
    ) -> None:
        """Evaluate consumable usage decisions.

        Handles: food buffs, emergency potion checks, Butterfly Wing return.
        """
        bot_id = service._resolve_bot_id(signals)
        map_name = str(signals.get("map", "") or "").lower()

        if not self._is_hunting_map(map_name):
            return

        zeny = int(signals.get("zeny", 0) or 0)
        job_name = str(signals.get("job_name", "novice") or "novice").lower()
        hp_ratio = float(signals.get("hp_ratio", 1.0) or 1.0)
        hunt_duration = (
            __import__("time").time()
            - service._state_since.get(bot_id, __import__("time").time())
        )
        _inv = signals.get("inventory_items", [])
        if isinstance(_inv, list):
            has_items = len(_inv) > 0
        else:
            has_items = int(_inv or 0) > 0

        # ── FOOD/BUFF: Use food if zeny > 1000 and hunting 5+ min ──
        if zeny > 1000 and hunt_duration > 300:
            _primary = self.JOB_PRIMARY_STAT.get(job_name, "str")
            _food_id = self.FOOD_ITEMS.get(_primary, "531")
            actions.append(HeuristicAction(
                kind="command", command=f"use {_food_id}",
                confidence=0.80, domain="economy",
                reason=f"Use {_primary} food (+4 {_primary.upper()}, 30 min)",
            ))

        # ── BUTTERFLY WING RETURN: low HP + no items + hunted 5+ min ──
        if hp_ratio < 0.3 and not has_items and hunt_duration > 300:
            actions.append(HeuristicAction(
                kind="command", command="use 602",
                confidence=0.95, domain="survival",
                reason=f"Low HP ({hp_ratio:.0%}) no items - Butterfly Wing to town",
            ))

        # ── SITTING TO REGEN ──
        if hp_ratio < 0.15:
            actions.append(HeuristicAction(
                kind="command", command="sit",
                confidence=0.99, domain="survival",
                reason=f"HP={hp_ratio:.0%} CRITICAL - emergency sit",
            ))
        elif hp_ratio < 0.2 and not has_items:
            actions.append(HeuristicAction(
                kind="command", command="sit",
                confidence=0.99, domain="survival",
                reason=f"HP={hp_ratio:.0%} no items - sitting to regen",
            ))
        elif hp_ratio < 0.3 and not has_items:
            actions.append(HeuristicAction(
                kind="command", command="sit",
                confidence=0.99, domain="survival",
                reason=f"HP={hp_ratio:.0%} no items - sitting to regen",
            ))

    def _is_hunting_map(self, map_name: str) -> bool:
        return any(x in map_name.lower() for x in [
            "prt_fild", "pay_fild", "mjolnir", "gef_fild",
            "ra_fild", "moc_fild", "cmd_fild",
        ])


def create_domain() -> ConsumablesDomain:
    return ConsumablesDomain()
