"""Combat domain — attack decisions, mon_control, target selection, skill rotation.

Extracted from heuristic_service.py lines 1258-1281, 1826-1865, 3102-3200.
"""

from __future__ import annotations

import logging
from typing import Any

from ai_sidecar.autonomy.domains import BaseDomain
from ai_sidecar.autonomy.heuristic_service import HeuristicAction
from ai_sidecar.autonomy.ro_mechanics import (
    PER_MAP_MON_CONTROL,
    JOB_WEAPON_TYPE,
    get_best_skill,
    is_mvp,
    get_mvp_value,
)

logger = logging.getLogger(__name__)


class CombatDomain(BaseDomain):
    name: str = "combat"
    priority: int = 20

    # Per-class combat configs: (attack_distance, attack_max, teleport_min_aggro)
    CLASS_COMBAT_CONFIG: dict[str, tuple[int, int, int]] = {
        "swordman":  (5, 20, 8),
        "knight":    (5, 20, 8),
        "thief":     (3, 15, 6),
        "assassin":  (3, 15, 6),
        "acolyte":   (7, 25, 4),
        "priest":    (7, 25, 4),
        "archer":    (10, 30, 3),
        "hunter":    (10, 30, 3),
        "mage":      (8, 25, 2),
        "wizard":    (8, 25, 2),
    }

    # Monsters to ignore during Novice cold-start farming
    COLD_START_IGNORE: list[str] = [
        "Thief Bug Egg", "Pupa", "Thief Bug",
        "Lunatic", "Fabre", "Condor",
    ]

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        service: Any,
    ) -> None:
        """Evaluate combat decisions based on current signals.

        Handles: mon_control per map, class-appropriate attack config,
        skill rotation, MVP priority, aggressive escape.
        """
        bot_id = service._resolve_bot_id(signals)
        map_name = str(signals.get("map", "") or "").lower().replace(".gat", "")
        job_name = str(signals.get("job_name", "novice") or "novice").lower()
        base_level = int(signals.get("base_level", 1) or 1)
        hp_ratio = float(signals.get("hp_ratio", 1.0) or 1.0)

        # ── PER-MAP MON_CONTROL ──
        self._emit_mon_control(actions, bot_id, map_name, service)

        # ── CLASS-APPROPRIATE COMBAT CONFIG ──
        self._apply_combat_config(actions, bot_id, job_name, base_level, service)

        # ── SKILL ROTATION (DPS-optimized) ──
        self._apply_skill_rotation(signals, actions, service)

        # ── MVP AWARENESS ──
        self._check_mvp_nearby(signals, actions)

        # ── FLY WING ESCAPE when surrounded ──
        self._check_emergency_escape(signals, actions, hp_ratio)

        # ── ANTI-DETECTION: randomize combat pacing per bot ──
        self._apply_mimicry_config(actions, bot_id, service)

    def _emit_mon_control(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        map_name: str,
        service: Any,
    ) -> None:
        """Emit mon_control commands for the current map.

        Uses PER_MAP_MON_CONTROL table to set per-monster attack/ignore.
        Dedups via service._last_mon_control_map.
        """
        controls = PER_MAP_MON_CONTROL.get(map_name)
        if not controls:
            return
        _last_map = service._last_mon_control_map.get(bot_id, "")
        if _last_map == map_name:
            return
        service._last_mon_control_map[bot_id] = map_name
        logger.info(
            "[mon_control] %s: applying %d entries for %s",
            bot_id, len(controls), map_name,
        )
        for _monster, _attack, _lvl, _aggr in controls:
            actions.append(HeuristicAction(
                kind="command",
                command=f"mon_control {_monster}\t{_attack} {_lvl} {_aggr}",
                confidence=0.95, domain="hunting",
                reason=f"Per-map mon_control: {_monster} -> attack={_attack} on {map_name}",
            ))

    def _apply_combat_config(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        job_name: str,
        base_level: int,
        service: Any,
    ) -> None:
        """Set class-appropriate attack config with dedup."""
        config = self.CLASS_COMBAT_CONFIG.get(
            job_name, (3, 15, 8)
        )
        atk_dist, atk_max, tele_min_agg = config

        service._set_config_once(
            actions, bot_id, "attackDistance", str(atk_dist), "hunting",
            f"Class-appropriate attack distance for {job_name}",
        )
        service._set_config_once(
            actions, bot_id, "attackMaxDistance", str(atk_max), "hunting",
            "Set max chase distance",
        )
        _aa = "2" if base_level < 10 else "3"
        service._set_config_once(
            actions, bot_id, "attackAuto", _aa, "hunting",
            f"attackAuto={_aa} (level {base_level})",
        )
        service._set_config_once(
            actions, bot_id, "attackAuto_followTarget", "1", "hunting",
            "Chase fleeing monsters",
        )
        service._set_config_once(
            actions, bot_id, "attackAuto_noMove", "0", "hunting",
            "Allow movement during combat",
        )
        service._set_config_once(
            actions, bot_id, "attackAuto_inLockOnly", "1", "hunting",
            "Only attack monsters in lockMap area",
        )
        service._set_config_once(
            actions, bot_id, "attackAuto_onlyWhenSafe", "0", "hunting",
            "Attack even if not safe",
        )
        service._set_config_once(
            actions, bot_id, "attackAuto_startOnSight", "1", "hunting",
            "Attack monsters as soon as they appear",
        )
        service._set_config_once(
            actions, bot_id, "attackAuto_unstuck", "1", "hunting",
            "Don't give up mid-fight",
        )
        # Per-class teleport threshold
        service._set_config_once(
            actions, bot_id, "teleportAuto_minAggressives",
            str(tele_min_agg), "hunting",
            f"Per-class teleport at {tele_min_agg}+ mobs ({job_name})",
        )

    def _apply_skill_rotation(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        service: Any,
    ) -> None:
        """Evaluate best DPS skill and queue attack_skill command."""
        _current_sp = int(signals.get("sp", 0) or 0)
        _max_sp = int(signals.get("max_sp", 100) or 100)
        _agi = int(signals.get("agi", 1) or 1)
        _dex = int(signals.get("dex", 1) or 1)
        _player_hp = int(signals.get("hp", 100) or 100)
        _attack_power = int(signals.get("attack_power", 25) or 25)
        _known_skills = signals.get("skills", []) or []
        _skill_levels = signals.get("skill_levels", {}) or {}
        _monster_element = str(signals.get("monster_element", "Neutral") or "Neutral")
        _monster_def = int(signals.get("monster_def", 0) or 0)
        _monster_size = str(signals.get("monster_size", "Medium") or "Medium")
        _monster_race = str(signals.get("monster_race", "Brute") or "Brute")
        _aggro_count = int(signals.get("aggressives", 0) or 0)
        _job_name = str(signals.get("job_name", "novice") or "novice").lower()
        _weapon_type = JOB_WEAPON_TYPE.get(_job_name, "dagger")

        _best_skill = get_best_skill(
            _known_skills, _skill_levels, _attack_power, _weapon_type,
            _monster_def, _monster_size, _monster_element, _monster_race,
            _current_sp, _max_sp, _agi, _dex, _aggro_count, _player_hp,
        )
        if _best_skill:
            actions.append(HeuristicAction(
                kind="command", command=f"attack_skill {_best_skill}",
                confidence=0.90, domain="combat",
                reason=f"DPS skill: {_best_skill} (best DPS vs {_monster_element} monster)",
            ))

    def _check_mvp_nearby(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
    ) -> None:
        """If an MVP is nearby, prioritize attacking it."""
        _nearby = signals.get("monsters", []) or []
        for _nm in _nearby:
            _nm_name = (
                _nm.get("name", "") if isinstance(_nm, dict) else str(_nm)
            )
            if is_mvp(_nm_name):
                _mvp_value = get_mvp_value(_nm_name)
                actions.append(HeuristicAction(
                    kind="command", command=f"attack {_nm_name}",
                    confidence=0.99, domain="hunting",
                    reason=f"MVP {_nm_name} nearby! (drop value ~{_mvp_value:,}z)",
                ))
                break

    def _check_emergency_escape(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        hp_ratio: float,
    ) -> None:
        """Use Fly Wing if surrounded by 3+ mobs at low HP."""
        _aggro = int(signals.get("aggressives", 0) or 0)
        if _aggro >= 3 and hp_ratio < 0.5:
            actions.append(HeuristicAction(
                kind="command", command="use 601",
                confidence=0.99, domain="survival",
                reason=f"Surrounded by {_aggro} mobs at {hp_ratio:.0%} HP - Fly Wing escape",
            ))

    def _apply_mimicry_config(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        service: Any,
    ) -> None:
        """Per-bot randomized config for human-like combat behavior."""
        _seed = hash(bot_id) & 0xFFFFFFFF
        _rand = __import__("random").Random(_seed)
        _step = _rand.randint(2, 5)
        _walk = _rand.choice(["1", "2"])
        _pause = _rand.randint(0, 2)
        service._set_config_once(
            actions, bot_id, "route_randomWalk", "1", "hunting",
            "Enable random walk for human-like movement",
        )
        service._set_config_once(
            actions, bot_id, "route_randomWalk_inLockOnly", "1", "hunting",
            "Random walk only in lockMap",
        )
        service._set_config_once(
            actions, bot_id, "route_randomWalk_maxRouteTime",
            str(_step), "hunting",
            f"Random walk step {_step} (per-bot variation)",
        )
        service._set_config_once(
            actions, bot_id, "route_randomWalk_maxWalkTime",
            _walk, "hunting",
            f"Random walk time {_walk}s (per-bot variation)",
        )
        service._set_config_once(
            actions, bot_id, "attackAuto_pause",
            str(_pause), "hunting",
            f"Attack pause {_pause}s (per-bot variation)",
        )


def create_domain() -> CombatDomain:
    return CombatDomain()
