"""Routing domain — map movement, portal navigation, lockMap management.

Extracted from heuristic_service.py lines 2783-2823 (TOWN_HUNT),
3065-3100 (second TOWN_HUNT), 3295-3315 (spawn circuit, portal exit).
"""

from __future__ import annotations

import logging
from typing import Any

from ai_sidecar.autonomy.domains import BaseDomain
from ai_sidecar.autonomy.heuristic_service import HeuristicAction
from ai_sidecar.autonomy.ro_mechanics import build_spawn_circuit

logger = logging.getLogger(__name__)


class RoutingDomain(BaseDomain):
    name: str = "routing"
    priority: int = 25

    # Portal coordinates from Prontera -> prt_fild05
    PRONTERA_PORTAL = (367, 205)
    PRT_FILD05_PORTAL = (22, 203)

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        service: Any,
    ) -> None:
        """Evaluate routing decisions.

        Handles: moving to hunting map, spawn circuit navigation,
        portal exit centering, lockMap setting.
        """
        bot_id = service._resolve_bot_id(signals)
        map_name = str(signals.get("map", "") or "").lower()
        base_level = int(signals.get("base_level", 1) or 1)
        state = service._get_state(signals, bot_id)

        # ── TOWN_HUNT: move from town to hunting map ──
        if state == "TOWN_HUNT":
            self._handle_town_hunt(actions, bot_id, map_name, base_level, signals, service)
            return

        # Check if on hunting map for spawn circuit / portal exit
        if self._is_hunting_map(map_name):
            self._check_portal_exit(actions, map_name, signals)
            self._check_spawn_circuit(actions, bot_id, map_name, signals, service)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _is_hunting_map(self, map_name: str) -> bool:
        return any(x in map_name.lower() for x in [
            "prt_fild", "pay_fild", "mjolnir", "gef_fild",
            "ra_fild", "moc_fild", "cmd_fild",
        ])

    # ── State handlers ──────────────────────────────────────────────────

    def _handle_town_hunt(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        map_name: str,
        base_level: int,
        signals: dict[str, Any],
        service: Any,
    ) -> None:
        """Stand up and move from town to hunting map."""
        target_map = service._adaptive.get_best_map(bot_id, base_level)
        if not target_map:
            if base_level >= 20:
                target_map = "pay_fild01"
            elif base_level >= 15:
                target_map = "prt_fild08"
            else:
                target_map = "prt_fild05"

        actions.append(HeuristicAction(
            kind="command", command="stand",
            confidence=0.99, domain="hunting",
            reason="Stand up before moving to hunting map",
        ))
        actions.append(HeuristicAction(
            kind="log", command="ai_mode_auto",
            confidence=0.5, domain="planning",
            reason="Enable auto-attack before moving to hunting map [log-only: config-audit owns AI mode]",
            metadata={"ai_mode": "auto"},
        ))

        # Set lockMap to hunting map
        service._set_config_once(
            actions, bot_id, "lockMap", target_map, "hunting",
            f"Lock to hunting map {target_map}",
        )
        service._set_config_once(
            actions, bot_id, "lockMap_randX", "100", "hunting",
            "Random walk radius X",
        )
        service._set_config_once(
            actions, bot_id, "lockMap_randY", "100", "hunting",
            "Random walk radius Y",
        )
        service._set_config_once(
            actions, bot_id, "route_randomWalk", "1", "hunting",
            "Walk within lockMap bounds",
        )

        actions.append(HeuristicAction(
            kind="command", command=f"move {target_map}",
            confidence=0.99, domain="hunting",
            reason=f"Level {base_level} - move to {target_map} for grinding",
        ))
        logger.info(
            "[routing] %s: TOWN_HUNT -> move to %s (level %d)",
            bot_id, target_map, base_level,
        )

    def _check_portal_exit(
        self,
        actions: list[HeuristicAction],
        map_name: str,
        signals: dict[str, Any],
    ) -> None:
        """If bot is at portal exit, move to center of hunting map.

        Without this, a bot that crosses into a farm field sits at the portal edge
        (where few mobs spawn) and either gets pulled back through the portal or
        spins 'AI restarted for target reselection' with no monster in range. Moving
        it to the field interior puts it where academy.txt spawns Porings/Lunatics.
        """
        _x = int(signals.get("x", 0) or 0)
        _y = int(signals.get("y", 0) or 0)
        if (
            abs(_x - 367) < 10 and abs(_y - 205) < 10
            and map_name == "prt_fild05"
        ):
            actions.append(HeuristicAction(
                kind="command", command="move 200 200",
                confidence=0.99, domain="hunting",
                reason="At portal exit - move to center of hunting map",
            ))
        # prt_fild08c (academy farm) portal exits: prontera portal at (170,378) and
        # izlude_c portal at (367,212). Move the bot to the field interior so it finds
        # the Porings/Lunatics/Fabre academy spawns instead of lingering at the edge.
        if map_name == "prt_fild08c" and (
            (abs(_x - 170) < 10 and abs(_y - 378) < 10)
            or (abs(_x - 367) < 10 and abs(_y - 212) < 10)
        ):
            actions.append(HeuristicAction(
                kind="command", command="move 200 200",
                confidence=0.99, domain="hunting",
                reason="At prt_fild08c portal exit - move to field interior to reach academy mob spawns",
            ))

    def _check_spawn_circuit(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        map_name: str,
        signals: dict[str, Any],
        service: Any,
    ) -> None:
        """Use spawn heatmap to build optimized walking path."""
        _spawn_heatmap = service._adaptive.spawn_heatmap.get(map_name, {})
        if not _spawn_heatmap or len(_spawn_heatmap) < 3:
            return
        _x = int(signals.get("x", 0) or 0)
        _y = int(signals.get("y", 0) or 0)
        _circuit = build_spawn_circuit(_spawn_heatmap, _x, _y, 5)
        if _circuit and len(_circuit) >= 2:
            _next_wp = _circuit[0]
            actions.append(HeuristicAction(
                kind="command",
                command=f"move {_next_wp[0]} {_next_wp[1]}",
                confidence=0.80, domain="hunting",
                reason=f"Spawn circuit: walk to hot zone "
                       f"({_next_wp[0]}, {_next_wp[1]})",
            ))


def create_domain() -> RoutingDomain:
    return RoutingDomain()
