"""Progression domain — leveling, job change, stat allocation, skill training.

Extracted from heuristic_service.py lines 1484-1832 (cold start pipeline),
2825-3002 (JOB_CHANGE, STATS, SKILLS states), 3449-3481 (in-hunt stat allocation).
"""

from __future__ import annotations

import logging
from typing import Any

from ai_sidecar.autonomy.domains import BaseDomain
from ai_sidecar.autonomy.heuristic_service import (
    HeuristicAction,
    _class_stat_allocation,
    JOB_CHANGE_NPCS,
    JOB_CHANGE_2_1,
    JOB_2_1_CLASSES,
    JOB_CHANGE_TALK,
    CLASS_SKILL_TRAINING,
    CLASS_HUNTING_GROUNDS,
    NOVICE_WEIGHT_CAPACITY,
)
from ai_sidecar.autonomy.ro_mechanics import PER_MAP_MON_CONTROL

logger = logging.getLogger(__name__)


class ProgressionDomain(BaseDomain):
    name: str = "progression"
    priority: int = 30

    # Post-job-change hunt maps per class
    JOB_HUNT_MAPS: dict[str, str] = {
        "acolyte": "pay_fild01",
        "mage": "pay_fild01",
        "swordman": "prt_fild05",
        "hunter": "pay_fild01",
        "thief": "mjolnir_04",
        "merchant": "prt_fild05",
    }

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        service: Any,
    ) -> None:
        """Evaluate progression decisions.

        Handles: cold start pipeline, job change, stat allocation, skill training.
        """
        bot_id = service._resolve_bot_id(signals)
        state = service._get_state(signals, bot_id)

        # ── COLD START PIPELINE ──
        if state == "COLD_START":
            self._handle_cold_start(actions, signals, bot_id, service)

        # ── JOB CHANGE ──
        if state == "JOB_CHANGE":
            self._handle_job_change(actions, signals, bot_id, service)

        # ── STAT ALLOCATION ──
        if state == "STATS":
            self._handle_stats(actions, signals, bot_id, service)

        # ── SKILL TRAINING ──
        if state == "SKILLS":
            self._handle_skills(actions, signals, bot_id, service)

    # ── COLD START ──────────────────────────────────────────────────────

    def _handle_cold_start(
        self,
        actions: list[HeuristicAction],
        signals: dict[str, Any],
        bot_id: str,
        service: Any,
    ) -> None:
        """Execute cold start sequence (steps 0-8)."""
        _cs_key = bot_id.split(":")[-1].split("/")[-1] if ":" in bot_id else bot_id
        _cs_step = service._cold_start_step.get(_cs_key, 0)
        map_name = str(signals.get("map", "") or "")
        zeny = int(signals.get("zeny", 0) or 0)
        base_level = int(signals.get("base_level", 1) or 1)
        weight = float(signals.get("weight_ratio", 0) or 0)
        inventory = signals.get("inventory_items", []) or []

        _in_town = any(x in map_name for x in [
            "prontera", "morocc", "geffen", "payon",
            "aldebaran", "alberta", "izlude",
        ])
        _in_hunting = any(x in map_name for x in [
            "prt_fild", "pay_fild", "mjolnir", "gef_fild",
            "ra_fild", "moc_fild", "cmd_fild",
        ])
        _has_weapon = any(
            kw in str(item).lower()
            for item in inventory
            for kw in ["knife", "sword", "mace", "bow", "dagger", "rod"]
        )
        _has_potions = any(
            kw in str(item).lower()
            for item in inventory
            for kw in ["potion", "red", "orange", "white"]
        )

        if _cs_step == 0:
            self._cold_step0(actions, _cs_key, _has_weapon, _has_potions,
                            _in_hunting, _in_town, map_name, service)
        elif _cs_step == 1:
            self._cold_step1(actions, signals, _cs_key, zeny, _in_town,
                            _in_hunting, map_name, service)
        elif _cs_step == 2:
            self._cold_step2(actions, _cs_key, _has_weapon, zeny, base_level, service)
        elif _cs_step == 3:
            self._cold_step3(actions, _cs_key, _has_potions, zeny,
                            base_level, weight, inventory, service)
        elif _cs_step == 4:
            self._cold_step4(actions, _cs_key, map_name, _in_hunting,
                            base_level, service)
        elif _cs_step == 5:
            self._cold_step5(actions, signals, _cs_key, base_level,
                            _in_hunting, _in_town, service)
        elif _cs_step == 6:
            self._cold_step6(actions, signals, _cs_key, bot_id, base_level,
                            _in_town, service)
        elif _cs_step == 7:
            self._cold_step7(actions, signals, _cs_key, bot_id, _in_town, service)
        elif _cs_step == 8:
            self._cold_step8(actions, signals, _cs_key, _in_hunting, service)

    def _cold_step0(
        self, actions, _cs_key, _has_weapon, _has_potions,
        _in_hunting, _in_town, map_name, service,
    ):
        if _has_weapon and _has_potions:
            service._cold_start_step[_cs_key] = 4
            return
        if _in_hunting:
            service._cold_start_step[_cs_key] = 1
        elif not _in_town:
            actions.append(HeuristicAction(
                kind="log", command="ai_mode_manual",
                confidence=0.5, domain="planning",
                reason="Cold start - disable AI for portal walk [log-only: config-audit owns AI mode]",
                metadata={"ai_mode": "manual"},
            ))
            actions.append(HeuristicAction(
                kind="command", command="move prontera",
                confidence=0.99, domain="economy",
                reason="Cold start - walk to Prontera portal",
            ))
        else:
            service._cold_start_step[_cs_key] = 1

    def _cold_step1(
        self, actions, signals, _cs_key, zeny,
        _in_town, _in_hunting, map_name, service,
    ):
        if zeny >= 50:
            service._cold_start_step[_cs_key] = 2
            logger.info("[cold_start] step 1 -> 2 (farmed 50z)")
            return
        if _in_town:
            actions.append(HeuristicAction(
                kind="command", command="set lockMap prt_fild05",
                confidence=0.99, domain="economy",
                reason=f"Cold start step 1 - need {50 - zeny}z more",
            ))
            actions.append(HeuristicAction(
                kind="command", command="move prt_fild05",
                confidence=0.99, domain="economy",
                reason=f"Cold start step 1 - walk to prt_fild05, need {50 - zeny}z more",
            ))
        elif _in_hunting:
            actions.append(HeuristicAction(
                kind="log", command="ai_mode_auto",
                confidence=0.5, domain="planning",
                reason="Cold start step 1 - enable AI for farming Porings [log-only: config-audit owns AI mode]",
                metadata={"ai_mode": "auto"},
            ))
            actions.append(HeuristicAction(
                kind="command", command="set attackAuto 3",
                confidence=0.99, domain="economy",
                reason="Cold start step 1 - enable attack for farming",
            ))
            for _cs_ignore in [
                "Thief Bug Egg", "Pupa", "Thief Bug",
                "Lunatic", "Fabre", "Condor",
            ]:
                actions.append(HeuristicAction(
                    kind="command",
                    command=f"mon_control {_cs_ignore}\t-1 0 0",
                    confidence=0.95, domain="economy",
                    reason=f"Cold start step 1 - ignore {_cs_ignore}",
                ))

    def _cold_step2(
        self, actions, _cs_key, _has_weapon, zeny, base_level, service,
    ):
        if not _has_weapon and zeny >= 50:
            # RULE.md: the starter weapon comes from the AGNOSTIC gear planner
            # (best affordable weapon by stat/zeny). The generic 'buy weapon'
            # form (OpenKore resolves the best starter weapon from its tables)
            # is the fallback — never a hardcoded server item id.
            _cs2_w = ""
            try:
                from ai_sidecar.gear_progression_planner import get_gear_progression_planner
                _cs2_plan = get_gear_progression_planner().get_best_upgrade(
                    int(base_level or 1), zeny
                )
                if _cs2_plan is not None and _cs2_plan.slot_name == "weapon" and _cs2_plan.is_affordable:
                    _cs2_w = str(_cs2_plan.item_name or _cs2_plan.item_id or "")
            except Exception:
                _cs2_w = ""
            actions.append(HeuristicAction(
                kind="command", command=f"buy weapon 1" if not _cs2_w else f"buy {_cs2_w} 1",
                confidence=0.99, domain="economy",
                reason="Cold start step 2 - buy starter weapon" + (f" ({_cs2_w})" if _cs2_w else " (best affordable)"),
            ))
            actions.append(HeuristicAction(
                kind="command", command=f"equip weapon" if not _cs2_w else f"equip {_cs2_w}",
                confidence=0.99, domain="economy",
                reason="Cold start step 2 - equip starter weapon",
            ))
        else:
            service._cold_start_step[_cs_key] = 3
            logger.info("[cold_start] step 2 -> 3 (weapon confirmed)")

    def _cold_step3(
        self, actions, _cs_key, _has_potions, zeny,
        base_level, weight, inventory, service,
    ):
        if not _has_potions:
            _cs_potion_id = service._get_potion_id(base_level)
            _cs_potion_cost = service._get_potion_cost(_cs_potion_id)
            _cs_potion_name = {501: "Red", 502: "Orange", 504: "White"}.get(
                _cs_potion_id, "Red"
            )
            if zeny >= _cs_potion_cost:
                _cs_max_weight = int(
                    max(0.0, 1.0 - weight) * NOVICE_WEIGHT_CAPACITY
                )
                _cs_max_by_zeny = int(zeny / _cs_potion_cost)
                _cs_qty = min(_cs_max_weight, _cs_max_by_zeny, 10)
                if _cs_qty > 0:
                    actions.append(HeuristicAction(
                        kind="command",
                        command=f"buy {_cs_potion_id} {_cs_qty}",
                        confidence=0.99, domain="economy",
                        reason=f"Cold start - buy {_cs_qty} {_cs_potion_name} "
                               f"Potions (level {base_level})",
                    ))
        else:
            service._cold_start_step[_cs_key] = 4
            logger.info("[cold_start] step 3 -> 4 (potions confirmed)")

    def _cold_step4(
        self, actions, _cs_key, map_name,
        _in_hunting, base_level, service,
    ):
        if not _in_hunting or map_name == "prt_fild01":
            actions.append(HeuristicAction(
                kind="command", command="move prt_fild05",
                confidence=0.99, domain="hunting",
                reason="Cold start - return to hunting map",
            ))
        elif _in_hunting:
            if base_level >= 10:
                service._cold_start_step[_cs_key] = 5
            else:
                service._cold_start_step[_cs_key] = 4

    def _cold_step5(
        self, actions, signals, _cs_key, base_level,
        _in_hunting, _in_town, service,
    ):
        if base_level >= 10:
            service._cold_start_step[_cs_key] = 6
            logger.info("[cold_start] step 5 -> 6 (level 10 reached)")
            return
        if _in_hunting:
            actions.append(HeuristicAction(
                kind="log", command="ai_mode_auto",
                confidence=0.5, domain="planning",
                metadata={"ai_mode": "auto"},
                reason=f"Step 5 - farm to level 10 (currently {base_level})",
            ))
            actions.append(HeuristicAction(
                kind="command", command="set attackAuto 3",
                confidence=0.99, domain="progression",
                reason=f"Step 5 - keep attacking until level 10",
            ))
        elif _in_town:
            actions.append(HeuristicAction(
                kind="command", command="move prt_fild05",
                confidence=0.99, domain="progression",
                reason="Step 5 - return to hunting map to level",
            ))

    def _cold_step6(
        self, actions, signals, _cs_key, bot_id,
        base_level, _in_town, service,
    ):
        if service._team_jobs_assigned.get(_cs_key, False):
            service._cold_start_step[_cs_key] = 7
            logger.info("[cold_start] step 6 -> 7 (jobs assigned)")
            return
        service._team_levels[_cs_key] = base_level
        if not _in_town:
            actions.append(HeuristicAction(
                kind="command", command="move prontera",
                confidence=0.99, domain="progression",
                reason="Step 6 - return to town for job change",
            ))
            return
        _bot_profile = bot_id.split(":")[-1].split("/")[-1] if ":" in bot_id else bot_id
        _all_bots = signals.get("all_bots", []) or []
        _sorted_bots = sorted(_all_bots)
        _is_leader = bool(_sorted_bots) and _bot_profile == _sorted_bots[0]
        if _is_leader:
            _all_ready = all(
                service._team_levels.get(p, 0) >= 10
                for p in _all_bots if p != _bot_profile
            ) if _all_bots else False
            if _all_ready and base_level >= 10:
                # Try team synergy API, fallback to knowledge
                try:
                    import json, requests  # noqa: F401
                    _payload = {
                        "bots": [{
                            "bot_id": bot_id,
                            "profile_name": _bot_profile,
                            "base_level": base_level,
                            "current_job": str(signals.get("job_name", "Novice")),
                        }],
                    }
                    _resp = requests.post(
                        "http://127.0.0.1:18081/v1/conscious/team-synergy",
                        json=_payload, timeout=15.0,
                    )
                    if _resp.ok:
                        _data = _resp.json()
                        _assignments = _data.get("assignments", [])
                        for _a in _assignments:
                            _prof = _a.get("profile_name", "")
                            _job = _a.get("recommended_job", "Acolyte")
                            actions.append(HeuristicAction(
                                kind="command",
                                command=f"job_change {_prof} {_job}",
                                confidence=0.99, domain="progression",
                                reason=f"Team synergy: {_prof} -> {_job}",
                            ))
                            service._assigned_jobs[_prof] = _job
                        service._team_jobs_assigned[_cs_key] = True
                        logger.info("[team_synergy] jobs assigned via LLM")
                        return
                except Exception as _e:
                    logger.warning("[team_synergy] failed: %s — using fallback", _e)
                # Knowledge fallback
                _fallback_jobs = [
                    "Acolyte", "Mage", "Swordsman",
                    "Hunter", "Thief", "Merchant",
                ]
                for _i, _p in enumerate(_all_bots):
                    if _i < len(_fallback_jobs):
                        actions.append(HeuristicAction(
                            kind="command",
                            command=f"job_change {_p} {_fallback_jobs[_i]}",
                            confidence=0.95, domain="progression",
                            reason=f"Knowledge fallback: {_p} -> {_fallback_jobs[_i]}",
                        ))
                service._team_jobs_assigned[_cs_key] = True
        else:
            pass  # Follower waits for leader

    def _cold_step7(
        self, actions, signals, _cs_key, bot_id,
        _in_town, service,
    ):
        job_name = str(signals.get("job_name", "novice") or "novice").lower()
        if job_name != "novice":
            service._cold_start_step[_cs_key] = 8
            logger.info("[cold_start] step 7 -> 8 (job changed to %s)", job_name)
            return
        _bot_profile = bot_id.split(":")[-1].split("/")[-1] if ":" in bot_id else bot_id
        _assigned_job = service._assigned_jobs.get(_bot_profile, "").lower()
        if _assigned_job:
            _jc_data = JOB_CHANGE_2_1.get(_assigned_job)
            if _jc_data:
                _jc_map, _jc_x, _jc_y, _jc_talk = _jc_data
                if _in_town:
                    _tx = _jc_x + 1
                    _ty = _jc_y + 1
                    actions.append(HeuristicAction(
                        kind="command", command=f"move {_tx} {_ty}",
                        confidence=0.99, domain="progression",
                        reason=f"Step 7 - walk to {_assigned_job} job NPC",
                    ))
                    _talk_cmd = f"talknpc {_jc_x} {_jc_y}"
                    for _t in _jc_talk:
                        _talk_cmd += " " + _t.replace("talk @npc@", "").strip()
                    actions.append(HeuristicAction(
                        kind="command", command=_talk_cmd,
                        confidence=0.99, domain="progression",
                        reason=f"Step 7 - talk to {_assigned_job} job NPC",
                    ))
                else:
                    actions.append(HeuristicAction(
                        kind="command", command="move prontera",
                        confidence=0.99, domain="progression",
                        reason="Step 7 - go to Prontera for job change",
                    ))
        else:
            if not _in_town:
                actions.append(HeuristicAction(
                    kind="command", command="move prontera",
                    confidence=0.99, domain="progression",
                    reason="Step 7 - go to town, waiting for job assignment",
                ))

    def _cold_step8(
        self, actions, signals, _cs_key,
        _in_hunting, service,
    ):
        job_name = str(signals.get("job_name", "novice") or "novice").lower()
        _hunt_map = "prt_fild05"
        for _j, _m in self.JOB_HUNT_MAPS.items():
            if _j in job_name:
                _hunt_map = _m
                break
        if _in_hunting:
            actions.append(HeuristicAction(
                kind="log", command="ai_mode_auto",
                confidence=0.5, domain="planning",
                metadata={"ai_mode": "auto"},
                reason=f"Step 8 - farm {_hunt_map} as {job_name}",
            ))
            actions.append(HeuristicAction(
                kind="command", command="set attackAuto 3",
                confidence=0.99, domain="progression",
                reason=f"Step 8 - enable attack on {_hunt_map}",
            ))
        else:
            actions.append(HeuristicAction(
                kind="command", command=f"move {_hunt_map}",
                confidence=0.99, domain="progression",
                reason=f"Step 8 - move to {_hunt_map} for post-job farming",
            ))

    # ── JOB CHANGE ──────────────────────────────────────────────────────

    def _handle_job_change(
        self,
        actions: list[HeuristicAction],
        signals: dict[str, Any],
        bot_id: str,
        service: Any,
    ) -> None:
        """Execute job change: Novice -> class, or class -> 2-1."""
        job_name = str(signals.get("job_name", "novice") or "novice").lower()
        base_level = int(signals.get("base_level", 1) or 1)
        job_level = int(signals.get("job_level", 1) or 1)
        map_name = str(signals.get("map", "") or "").lower()

        _jc_target_class = ""
        _jc_npc_map = "prontera"
        _jc_npc_x = 160
        _jc_npc_y = 191
        _jc_talk_seq: list[str] = []
        _jc_is_2_1 = False

        if job_name == "novice":
            _jc_target_class = "archer"
            _jc_npc = JOB_CHANGE_NPCS.get("novice", ("prontera", 160, 191))
            _jc_npc_map, _jc_npc_x, _jc_npc_y = _jc_npc
            _jc_talk_seq = JOB_CHANGE_TALK.get("archer", [
                "talk continue", "talk resp 1", "talk resp 2", "talk resp 1",
            ])
            logger.info(
                "[job_change] %s: Novice Lv%d -> %s",
                bot_id, job_level, _jc_target_class,
            )
        else:
            _jc_is_2_1 = True
            _jc_2_1_data = JOB_CHANGE_2_1.get(job_name)
            if _jc_2_1_data:
                _jc_npc_map, _jc_npc_x, _jc_npc_y, _jc_talk_seq = _jc_2_1_data
            else:
                _jc_npc_map, _jc_npc_x, _jc_npc_y = ("prontera", 160, 191)
                _jc_talk_seq = ["talk continue", "talk resp 1", "talk resp 2", "talk resp 1"]
            _jc_target_class = JOB_2_1_CLASSES.get(job_name, job_name)
            logger.info(
                "[job_change] %s: %s Lv%d -> %s (2-1)",
                bot_id, job_name, job_level, _jc_target_class,
            )

        actions.append(HeuristicAction(
            kind="command", command="stand",
            confidence=0.95, domain="progression",
            reason="Stand up before walking to job change NPC",
        ))
        if map_name != _jc_npc_map:
            actions.append(HeuristicAction(
                kind="command", command=f"move {_jc_npc_map}",
                confidence=0.95, domain="progression",
                reason=f"Move to {_jc_npc_map} for job change to {_jc_target_class}",
            ))
        else:
            actions.append(HeuristicAction(
                kind="command", command=f"move {_jc_npc_x} {_jc_npc_y}",
                confidence=0.95, domain="progression",
                reason=f"Walk to job change NPC for {_jc_target_class}",
            ))
            for _idx, _cmd in enumerate(_jc_talk_seq):
                _conf = max(0.70, 0.95 - (_idx * 0.03))
                actions.append(HeuristicAction(
                    kind="command", command=_cmd,
                    confidence=_conf, domain="progression",
                    reason=f"Job change dialog step {_idx+1}: {_jc_target_class}",
                ))
            # Post-job-change cleanup
            service._last_mon_control_map[bot_id] = ""
            service._last_lockmap[bot_id] = ""
            service._cold_start_step[bot_id] = 4
            service._post_job_change_reset[bot_id] = True
            logger.info(
                "[job_change] %s: sequence sent for %s",
                bot_id, _jc_target_class,
            )

    # ── STATS ───────────────────────────────────────────────────────────

    def _handle_stats(
        self,
        actions: list[HeuristicAction],
        signals: dict[str, Any],
        bot_id: str,
        service: Any,
    ) -> None:
        """Allocate stat points using class-aware breakpoint logic."""
        _current = int(signals.get("stat_points", 0) or 0)
        if _current <= 0:
            return

        _job_name = str(signals.get("job_name", "novice") or "novice").lower()
        _stats = {
            "str": int(signals.get("str", 1) or 1),
            "agi": int(signals.get("agi", 1) or 1),
            "vit": int(signals.get("vit", 1) or 1),
            "int": int(signals.get("int", 1) or 1),
            "dex": int(signals.get("dex", 1) or 1),
            "luk": int(signals.get("luk", 1) or 1),
        }
        base_level = int(signals.get("base_level", 1) or 1)
        _allocations = _class_stat_allocation(
            _job_name, _stats, _current, service._adaptive, base_level,
        )
        for _stat_name, _points in _allocations:
            for _ in range(min(_points, _current)):
                actions.append(HeuristicAction(
                    kind="command", command=f"stat_add {_stat_name}",
                    confidence=0.95, domain="progression",
                    reason=f"Allocate 1 {_stat_name.upper()} "
                           f"({_job_name}, breakpoint-aware)",
                ))
                _current -= 1
                if _current <= 0:
                    break

    # ── SKILLS ──────────────────────────────────────────────────────────

    def _handle_skills(
        self,
        actions: list[HeuristicAction],
        signals: dict[str, Any],
        bot_id: str,
        service: Any,
    ) -> None:
        """Train skills using class-specific training priorities."""
        _sk_job = str(signals.get("job_name", "novice") or "novice").lower()
        _sk_known = set(signals.get("skills", []) if isinstance(signals.get("skills"), list) else [])
        _sk_points = int(signals.get("skill_points", 0) or 0)
        _sk_levels = signals.get("skill_levels", {}) or {}

        _sk_training = CLASS_SKILL_TRAINING.get(_sk_job, CLASS_SKILL_TRAINING["novice"])
        for _sk_id, _sk_target, _sk_desc in _sk_training:
            if _sk_points <= 0:
                break
            _current_lv = (
                _sk_levels.get(_sk_id, 0)
                if isinstance(_sk_levels, dict) else 0
            )
            if _current_lv >= _sk_target:
                continue
            _next = _current_lv + 1
            actions.append(HeuristicAction(
                kind="command", command=f"add {_sk_id}",
                confidence=0.90, domain="progression",
                reason=f"Learn/level {_sk_id} ({_sk_desc}) Lv{_next}/{_sk_target}",
            ))
            _sk_points -= 1


def create_domain() -> ProgressionDomain:
    return ProgressionDomain()
