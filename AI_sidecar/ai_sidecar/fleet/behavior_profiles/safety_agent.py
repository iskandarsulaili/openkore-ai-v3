"""SafetyAgent — emergency teleport, death recovery, weight, potions, repair."""

from __future__ import annotations

import time
from typing import Any

from ai_sidecar.fleet.behavior_profiles import BehaviorProfile

_POTION_PRIORITY = {
    "red_potion": {"hp_pct": 0.5, "restore": 45}, "orange_potion": {"hp_pct": 0.4, "restore": 105},
    "yellow_potion": {"hp_pct": 0.35, "restore": 175}, "white_potion": {"hp_pct": 0.3, "restore": 325},
    "blue_potion": {"hp_pct": 0.0, "sp_pct": 0.3, "restore": 50},
    "green_potion": {"status": "poisoned", "restore": 0},
    "panacea": {"status": "any", "restore": 0},
}


class SafetyAgent(BehaviorProfile):
    """Handles survival — emergency escape, death recovery, weight, consumables, repair."""

    def __init__(self, bot_id: str, experience_db=None):
        super().__init__(bot_id, experience_db)
        self._last_emergency = 0.0

    def emergency_check(self, hp_pct: float, sp_pct: float, is_attacked: bool,
                        weight_pct: float, broken_equip: bool) -> dict[str, Any] | None:
        now = time.time()
        if now - self._last_emergency < 5:
            return None

        if broken_equip:
            return {"action": "repair_equipment", "priority": "critical"}

        if hp_pct < 0.15 and is_attacked:
            self._last_emergency = now
            return {"action": "emergency_teleport", "method": "fly_wing",
                    "reason": "critical_hp"}
        if hp_pct < 0.08:
            self._last_emergency = now
            return {"action": "emergency_teleport", "method": "butterfly_wing",
                    "reason": "near_death"}
        if hp_pct < 0.3 and is_attacked and weight_pct > 0.9:
            return {"action": "drop_heavy_items", "reason": "overweight_under_attack"}
        return None

    def death_recovery(self, zeny: int) -> dict[str, Any]:
        return {"action": "respawn", "rebuff": True,
                "re-equip": True, "zeny_after_death": zeny}

    def weight_management(self, weight_pct: float, inventory: list[dict[str, Any]],
                          zeny: int) -> dict[str, Any]:
        if weight_pct < 0.7:
            return {"action": "no_action", "weight_pct": weight_pct}
        if weight_pct > 0.9:
            junk = [i for i in inventory if not i.get("equipped") and i.get("vendor_price", 0) > 0]
            junk.sort(key=lambda x: -x.get("weight", 1))
            return {"action": "vendor_or_store", "items_to_sell": junk[:10],
                    "weight_pct": weight_pct}
        return {"action": "monitor", "weight_pct": weight_pct}

    def potion_decision(self, hp_pct: float, sp_pct: float, status_effects: list[str],
                        inventory_counts: dict[str, int]) -> dict[str, Any]:
        for potion_name, cfg in _POTION_PRIORITY.items():
            if inventory_counts.get(potion_name, 0) <= 0:
                continue
            if status_effects and cfg.get("status"):
                if cfg["status"] == "any":
                    return {"action": "use_item", "item": potion_name, "reason": f"status:{status_effects}"}
                if any(cfg["status"] in se.lower() for se in status_effects):
                    return {"action": "use_item", "item": potion_name, "reason": cfg["status"]}
            if cfg.get("hp_pct", 0) and hp_pct <= cfg["hp_pct"]:
                return {"action": "use_item", "item": potion_name, "reason": f"hp<={cfg['hp_pct']:.0%}"}
            if cfg.get("sp_pct", 0) and sp_pct <= cfg["sp_pct"]:
                return {"action": "use_item", "item": potion_name, "reason": f"sp<={cfg['sp_pct']:.0%}"}
        return {"action": "no_potion_needed"}

    def potion_stock_check(self, inventory_counts: dict[str, int],
                           zeny: int) -> dict[str, Any]:
        need = {}
        for potion, cfg in _POTION_PRIORITY.items():
            if cfg.get("hp_pct", 0) or cfg.get("sp_pct", 0):
                if inventory_counts.get(potion, 0) < 50:
                    need[potion] = 50 - inventory_counts.get(potion, 0)
        if need and zeny > 1000:
            return {"action": "auto_buy_potions", "items": need, "total_cost": sum(
                self._potion_cost(p) * q for p, q in need.items())}
        return {"action": "stocked"}

    @staticmethod
    def _potion_cost(potion: str) -> int:
        return {"red_potion": 50, "orange_potion": 300, "yellow_potion": 900,
                "white_potion": 1200, "blue_potion": 3000,
                "green_potion": 80, "panacea": 2000}.get(potion, 100)

    def repair_check(self, broken_items: list[str], zeny: int) -> dict[str, Any]:
        if not broken_items:
            return {"action": "no_repair_needed"}
        return {"action": "visit_npc_repair", "items": broken_items,
                "repair_cost": len(broken_items) * 1000,
                "map_to_visit": "town"}

    def record_outcome(self, action: str, success: bool, hp_saved: float = 0.0) -> None:
        self._record_experience("survival", action, success, reward=hp_saved, hp_saved=hp_saved)
