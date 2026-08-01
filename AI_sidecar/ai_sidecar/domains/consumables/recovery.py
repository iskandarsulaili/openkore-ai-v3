"""Potion usage at HP/SP thresholds for survival."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Recovery item definitions
_RECOVERY_ITEMS: dict[str, dict[str, Any]] = {
    "red_potion": {
        "item_id": "501", "name": "Red Potion",
        "heals_hp": 45, "weight": 3, "cost": 50, "min_level": 1,
    },
    "orange_potion": {
        "item_id": "502", "name": "Orange Potion",
        "heals_hp": 78, "weight": 3, "cost": 100, "min_level": 10,
    },
    "yellow_potion": {
        "item_id": "503", "name": "Yellow Potion",
        "heals_hp": 125, "weight": 3, "cost": 200, "min_level": 20,
    },
    "white_potion": {
        "item_id": "504", "name": "White Potion",
        "heals_hp": 187, "weight": 3, "cost": 500, "min_level": 35,
    },
    "blue_potion": {
        "item_id": "505", "name": "Blue Potion",
        "heals_hp": 0, "heals_sp": 50, "weight": 3, "cost": 1000, "min_level": 20,
    },
    "green_potion": {
        "item_id": "511", "name": "Green Potion",
        "heals_hp": 0, "cures": ["poison", "silence", "confusion"],
        "weight": 2, "cost": 100, "min_level": 1,
    },
    "panacea": {
        "item_id": "512", "name": "Panacea",
        "heals_hp": 0, "cures": ["all"],
        "weight": 2, "cost": 500, "min_level": 20,
    },
}

_DEFAULT_HP_THRESHOLD = 0.4
_DEFAULT_SP_THRESHOLD = 0.25
_EMERGENCY_HP_THRESHOLD = 0.2


class RecoveryManager:
    """Manage potion usage at HP/SP thresholds."""

    def __init__(self, db: Any = None) -> None:
        self._gk_db = db
        if self._gk_db is None:
            from ai_sidecar.game_knowledge_db import GameKnowledgeDB
            self._gk_db = GameKnowledgeDB()

    def assess_recovery(
        self,
        signals: dict[str, Any],
        bot_id: str,
    ) -> list[dict]:
        """Check HP/SP levels and recommend recovery actions.

        Returns list of action dicts.
        """
        actions: list[dict] = []
        hp = float(signals.get("hp", 100) or 100)
        max_hp = float(signals.get("max_hp", 100) or 100)
        sp = float(signals.get("sp", 100) or 100)
        max_sp = float(signals.get("max_sp", 100) or 100)
        inventory = signals.get("inventory", []) or []
        base_level = int(signals.get("base_level", 1) or 1)
        has_status_effects = bool(signals.get("status_effects", []) or [])

        hp_ratio = hp / max_hp if max_hp > 0 else 1.0
        sp_ratio = sp / max_sp if max_sp > 0 else 1.0

        # Check inventory for potions
        hp_potions = self._find_hp_potions(inventory, base_level)
        sp_potions = self._find_sp_potions(inventory)
        status_cures = self._find_status_cures(inventory)

        # Emergency HP recovery
        if hp_ratio < _EMERGENCY_HP_THRESHOLD:
            best_potion = self._get_best_hp_potion(hp_potions, max_hp - hp)
            if best_potion:
                actions.append({
                    "type": "use_recovery",
                    "priority": 10,
                    "reason": f"EMERGENCY: HP at {hp_ratio:.0%} — use {best_potion['name']}",
                    "item_id": best_potion["item_id"],
                    "item_name": best_potion["name"],
                    "is_emergency": True,
                })
                return actions

        if hp_ratio < _DEFAULT_HP_THRESHOLD:
            best_potion = self._get_best_hp_potion(hp_potions, max_hp - hp)
            if best_potion:
                actions.append({
                    "type": "use_recovery",
                    "priority": 8,
                    "reason": f"HP {hp_ratio:.0%} — use {best_potion['name']}",
                    "item_id": best_potion["item_id"],
                    "item_name": best_potion["name"],
                    "is_emergency": False,
                })

        if sp_ratio < _DEFAULT_SP_THRESHOLD:
            best_sp_potion = self._find_best_sp_potion(sp_potions)
            if best_sp_potion:
                actions.append({
                    "type": "use_recovery",
                    "priority": 7,
                    "reason": f"SP {sp_ratio:.0%} — use {best_sp_potion['name']}",
                    "item_id": best_sp_potion["item_id"],
                    "item_name": best_sp_potion["name"],
                    "is_emergency": False,
                })

        if has_status_effects and status_cures:
            cure = status_cures[0]
            status_names = signals.get("status_effects", [])
            actions.append({
                "type": "use_recovery",
                "priority": 9,
                "reason": f"Status effect(s): {status_names} — use {cure['name']}",
                "item_id": cure["item_id"],
                "item_name": cure["name"],
                "is_emergency": True,
            })

        return actions

    def get_use_item_command(self, item_id: str) -> str:
        return f"use {item_id}"

    def get_buy_potion_command(self, item_name: str, quantity: int = 10) -> str:
        return f"buy {item_name} {quantity}"

    def estimate_potion_needs(
        self,
        max_hp: float,
        max_sp: float,
        base_level: int,
        hunt_duration_minutes: int = 30,
    ) -> dict[str, int]:
        hp_potion_minutes = 2
        sp_potion_minutes = 5
        hp_needed = max(1, hunt_duration_minutes // hp_potion_minutes)
        sp_needed = max(1, hunt_duration_minutes // sp_potion_minutes)
        return {"hp_potions": hp_needed, "sp_potions": sp_needed}

    def _find_hp_potions(self, inventory: list[dict], base_level: int) -> list[dict]:
        potions: list[dict] = []
        for item in inventory:
            name = (item.get("name", "") or "").lower()
            amount = int(item.get("amount", 0) or 0)
            if amount <= 0:
                continue
            for pot_id, pot_def in _RECOVERY_ITEMS.items():
                if pot_def.get("heals_hp", 0) > 0 and pot_def["min_level"] <= base_level:
                    if pot_def["name"].lower() in name or pot_def["item_id"] == item.get("id", ""):
                        potions.append({
                            "item_id": pot_def["item_id"],
                            "name": pot_def["name"],
                            "heals": pot_def["heals_hp"],
                            "amount": amount,
                        })
                        break
        return potions

    def _find_sp_potions(self, inventory: list[dict]) -> list[dict]:
        potions: list[dict] = []
        for item in inventory:
            name = (item.get("name", "") or "").lower()
            amount = int(item.get("amount", 0) or 0)
            if amount <= 0:
                continue
            if "blue potion" in name:
                potions.append({
                    "item_id": "505", "name": "Blue Potion",
                    "heals": 50, "amount": amount,
                })
        return potions

    def _find_status_cures(self, inventory: list[dict]) -> list[dict]:
        cures: list[dict] = []
        for item in inventory:
            name = (item.get("name", "") or "").lower()
            amount = int(item.get("amount", 0) or 0)
            if amount <= 0:
                continue
            if "green potion" in name:
                cures.append({"item_id": "511", "name": "Green Potion", "amount": amount})
            elif "panacea" in name:
                cures.append({"item_id": "512", "name": "Panacea", "amount": amount})
        return cures

    def _get_best_hp_potion(self, potions: list[dict], heal_needed: float) -> dict | None:
        if not potions:
            return None
        sorted_potions = sorted(potions, key=lambda p: p["heals"])
        for pot in sorted_potions:
            if pot["heals"] >= heal_needed * 0.5:
                return pot
        return sorted_potions[-1]

    def _find_best_sp_potion(self, potions: list[dict]) -> dict | None:
        if not potions:
            return None
        return max(potions, key=lambda p: p["heals"])

    def cleanup_bot(self, bot_id: str) -> None:
        """Remove per-bot state on unregistration.

        RecoveryManager keeps no persistent per-bot dicts today (only the
        shared GameKnowledgeDB), but cleanup must be idempotent and defensive:
        any per-bot tracker attribute present is popped so a re-registered
        bot starts fresh.
        """
        for _attr in ("_last_heal", "_heal_timers", "_cooldowns", "_states", "_pending"):
            _holder = getattr(self, _attr, None)
            if isinstance(_holder, dict):
                _holder.pop(bot_id, None)
        logger.debug("[recovery] cleanup_bot %s", bot_id)
