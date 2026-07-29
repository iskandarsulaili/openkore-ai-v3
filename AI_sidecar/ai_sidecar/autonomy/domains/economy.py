"""Economy domain — sell, buy, storage, potion management, weight management.

Extracted from heuristic_service.py lines 1484-1616 (cold start buy/sell),
2602-2781 (SELL, BUY, WEAPON_BUY states), 3632-3694 (in-town buy).
"""

from __future__ import annotations

import logging
from typing import Any

from ai_sidecar.autonomy.domains import BaseDomain
from ai_sidecar.autonomy.heuristic_service import HeuristicAction, SELLABLE_JUNK

logger = logging.getLogger(__name__)


class EconomyDomain(BaseDomain):
    name: str = "economy"
    priority: int = 15

    POTION_COSTS: dict[int, int] = {501: 50, 502: 200, 504: 500}
    POTION_NAMES: dict[int, str] = {501: "Red", 502: "Orange", 504: "White"}
    POTION_TIERS: list[tuple[int, int, str, int]] = [
        (1, 501, "Red Potion", 45),
        (15, 502, "Orange Potion", 105),
        (30, 504, "White Potion", 250),
    ]
    NOVICE_WEIGHT_CAPACITY = 2000

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        service: Any,
    ) -> None:
        """Evaluate economy decisions.

        Handles: junk selling, potion buying, weapon buying,
        weight management, loot config, sellAuto config.
        """
        bot_id = service._resolve_bot_id(signals)
        state = service._get_state(signals, bot_id)
        map_name = str(signals.get("map", "") or "").lower()
        base_level = int(signals.get("base_level", 1) or 1)
        zeny = int(signals.get("zeny", 0) or 0)
        weight = float(signals.get("weight_ratio", 0) or 0)
        job_name = str(signals.get("job_name", "novice") or "novice").lower()
        inventory = signals.get("inventory_items", []) or []

        # ── SELLING (in town with weight) ──
        if state == "SELL":
            self._handle_sell(actions, bot_id, weight, inventory, service)

        # ── WEAPON BUY ──
        if state == "WEAPON_BUY":
            self._handle_weapon_buy(actions, bot_id, zeny, job_name, map_name, service)

        # ── BUY (potions) ──
        if state == "BUY":
            self._handle_buy(actions, bot_id, zeny, weight, base_level, service)

        # ── LOOTING CONFIG (hunting maps) ──
        if self._is_hunting_map(map_name):
            self._apply_loot_config(actions, bot_id, service)

        # ── SELL CONFIG (hunting maps) ──
        if self._is_hunting_map(map_name):
            self._apply_sell_config(actions, bot_id, service)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _is_hunting_map(self, map_name: str) -> bool:
        _m = map_name.lower()
        return any(x in _m for x in [
            "prt_fild", "pay_fild", "mjolnir", "gef_fild",
            "ra_fild", "moc_fild", "cmd_fild",
        ])

    def _is_town(self, map_name: str) -> bool:
        return any(x in map_name.lower() for x in [
            "prontera", "morocc", "geffen", "payon",
            "aldebaran", "alberta", "izlude",
        ])

    def _get_potion_id(self, base_level: int) -> int:
        if base_level < 15:
            return 501
        elif base_level < 30:
            return 502
        return 504

    def _get_potion_cost(self, potion_id: int) -> int:
        return self.POTION_COSTS.get(potion_id, 50)

    def _get_potion_max_buy(
        self, potion_id: int, zeny: int,
        weight: float, weight_capacity: int,
    ) -> int:
        _cost = self._get_potion_cost(potion_id)
        _max_by_zeny = zeny // _cost if _cost > 0 else 0
        _rem_weight = max(0.0, 1.0 - weight)
        _rem_units = _rem_weight * weight_capacity
        _max_by_weight = int(_rem_units // 1)
        _max_buy = min(_max_by_zeny, _max_by_weight, 30)
        return max(0, _max_buy)

    def _has_weapon(self, inventory: list) -> bool:
        return any(
            kw in str(item).lower()
            for item in inventory
            for kw in ["knife", "sword", "mace", "bow", "dagger", "rod"]
        )

    def _has_potions(self, inventory: list) -> bool:
        return any(
            kw in str(item).lower()
            for item in inventory
            for kw in ["potion", "red", "orange", "white"]
        )

    # ── STATE HANDLERS ──────────────────────────────────────────────────

    def _handle_sell(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        weight: float,
        inventory: list,
        service: Any,
    ) -> None:
        """Sell junk items when in town with inventory weight."""
        _now = __import__("time").time()
        _last_sell = service._last_sell_time.get(bot_id, 0)
        if _now - _last_sell < 60:
            return  # Cooldown
        service._last_sell_time[bot_id] = _now

        # Stand up
        actions.append(HeuristicAction(
            kind="command", command="stand",
            confidence=0.95, domain="economy",
            reason="Stand up before walking to Tool Dealer",
        ))
        # Walk to Tool Dealer
        actions.append(HeuristicAction(
            kind="command", command="move 290 221",
            confidence=0.95, domain="economy",
            reason=f"Weight {weight:.0%} - walk to Tool Dealer to sell junk",
        ))
        actions.append(HeuristicAction(
            kind="command", command="talknpc 290 221 c r1 n",
            confidence=0.90, domain="economy",
            reason="Open Tool Dealer and sell items (atomic dialog)",
        ))
        # Auto-sell known junk
        _junk_found = False
        for _item_entry in inventory:
            _item_str = str(_item_entry).lower().strip()
            for _junk_name, _junk_id in SELLABLE_JUNK.items():
                if _junk_name in _item_str:
                    if service._sell_config_once(bot_id, _junk_id, cooldown=120.0):
                        actions.append(HeuristicAction(
                            kind="command", command=f"sell {_junk_id} 0",
                            confidence=0.85, domain="economy",
                            reason=f"Sell {_junk_name} (item {_junk_id}) — junk",
                        ))
                        _junk_found = True
                    break
        if _junk_found:
            logger.info("[auto_sell] %s: queued sell commands for junk", bot_id)
        actions.append(HeuristicAction(
            kind="command", command="talk cont",
            confidence=0.80, domain="economy",
            reason="Complete sell transaction",
        ))

    def _handle_weapon_buy(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        zeny: int,
        job_name: str,
        map_name: str,
        service: Any,
    ) -> None:
        """Buy a weapon appropriate to the class."""
        if any(x in map_name for x in [
            "prt_fild", "pay_fild", "mjolnir", "gef_fild", "ra_fild",
        ]):
            # On hunting map - go through portal first
            actions.append(HeuristicAction(
                kind="command", command="move 22 203",
                confidence=0.99, domain="economy",
                reason="Go through portal to Prontera to buy weapon",
            ))
            return

        # Determine weapon by class
        _weapon_map = {
            "thief": "1301", "assassin": "1301",
            "sword": "1201", "knight": "1201",
            "mage": "1501", "wizard": "1501",
            "acolyte": "1501", "priest": "1501",
        }
        _weapon = "1701"  # Default bow
        for _k, _v in _weapon_map.items():
            if _k in job_name:
                _weapon = _v
                break

        # Walk to weapon shop
        actions.append(HeuristicAction(
            kind="command", command="move 160 133",
            confidence=0.95, domain="economy",
            reason=f"Zeny {zeny} - walk to Weapon Shop to buy weapon {_weapon}",
        ))
        actions.append(HeuristicAction(
            kind="command", command="talknpc 160 133 c r0 n",
            confidence=0.90, domain="economy",
            reason="Open Weapon Shop dialog",
        ))
        actions.append(HeuristicAction(
            kind="command", command=f"buy {_weapon} 1",
            confidence=0.85, domain="economy",
            reason=f"Buy weapon {_weapon} for class {job_name}",
        ))

    def _handle_buy(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        zeny: int,
        weight: float,
        base_level: int,
        service: Any,
    ) -> None:
        """Buy potions from Tool Dealer with weight awareness."""
        _now = __import__("time").time()
        _last_buy = service._last_buy_time.get(bot_id, 0)
        if _now - _last_buy < 60:
            return  # Cooldown
        service._last_buy_time[bot_id] = _now

        _potion_id = self._get_potion_id(base_level)
        _potion_cost = self._get_potion_cost(_potion_id)
        _potion_name = self.POTION_NAMES.get(_potion_id, "Red")

        _max_buy = self._get_potion_max_buy(
            _potion_id, zeny, weight, self.NOVICE_WEIGHT_CAPACITY,
        )
        _max_by_zeny = zeny // _potion_cost if _potion_cost > 0 else 0
        _max_buy = min(_max_buy, _max_by_zeny, 30)

        actions.append(HeuristicAction(
            kind="command", command="stand",
            confidence=0.95, domain="economy",
            reason="Stand up before walking to Tool Dealer",
        ))
        actions.append(HeuristicAction(
            kind="command", command="move 290 221",
            confidence=0.95, domain="economy",
            reason=f"Zeny {zeny} - walk to Tool Dealer to buy potions",
        ))
        actions.append(HeuristicAction(
            kind="command", command="talknpc 290 221",
            confidence=0.90, domain="economy",
            reason="Open Tool Dealer shop",
        ))
        actions.append(HeuristicAction(
            kind="command", command="talk resp 1",
            confidence=0.85, domain="economy",
            reason="Select buy option",
        ))
        if _max_buy > 0:
            actions.append(HeuristicAction(
                kind="command", command=f"buy {_potion_id} {_max_buy}",
                confidence=0.90, domain="economy",
                reason=f"Buy {_max_buy} {_potion_name} Potions "
                       f"(item {_potion_id}, {_potion_cost}z each, "
                       f"level={base_level}, weight={weight:.0%})",
            ))
        else:
            logger.info(
                "[economy] %s: can't buy potions at level %d — "
                "zeny=%d, cost=%d, weight=%.0f",
                bot_id, base_level, zeny, _potion_cost, weight,
            )
        actions.append(HeuristicAction(
            kind="command", command="talk any",
            confidence=0.80, domain="economy",
            reason="Complete buy dialog",
        ))

    def _apply_loot_config(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        service: Any,
    ) -> None:
        """Set loot config on hunting maps."""
        service._set_config_once(
            actions, bot_id, "itemsTakeAuto", "2", "economy",
            "Auto-take all dropped items",
        )
        service._set_config_once(
            actions, bot_id, "itemsGatherAuto", "2", "economy",
            "Auto-gather all items",
        )
        service._set_config_once(
            actions, bot_id, "itemsTakeAuto_party", "0", "economy",
            "Don't take party members' drops",
        )

    def _apply_sell_config(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        service: Any,
    ) -> None:
        """Set auto-sell config on hunting maps."""
        service._set_config_once(
            actions, bot_id, "sellAuto", "1", "economy",
            "Auto-sell loot when inventory full",
        )
        service._set_config_once(
            actions, bot_id, "sellAuto_npc", "prt_in 126 75", "economy",
            "Tool Dealer in Prontera",
        )
        service._set_config_once(
            actions, bot_id, "sellAuto_distance", "25", "economy",
            "Walk up to 25 cells to sell",
        )
        service._set_config_once(
            actions, bot_id, "sellAuto_maxWeight", "70", "economy",
            "Sell when weight > 70%",
        )
        service._set_config_once(
            actions, bot_id, "sellAuto_minZen", "0", "economy",
            "Sell even with 0 zeny",
        )
        service._set_config_once(
            actions, bot_id, "storageAuto", "1", "economy",
            "Auto-deposit at Kafra",
        )
        service._set_config_once(
            actions, bot_id, "storageAuto_distance", "5", "economy",
            "Stand next to Kafra",
        )


def create_domain() -> EconomyDomain:
    return EconomyDomain()
