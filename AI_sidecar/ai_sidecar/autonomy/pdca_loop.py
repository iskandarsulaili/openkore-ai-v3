"""PDCA autonomy loop — continuous Plan-Do-Check-Act cycle."""
from __future__ import annotations

import asyncio
import threading
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ai_sidecar.autonomy.plan_executor import PlanExecutor
from ai_sidecar.autonomy.progress_tracker import ProgressTracker
from ai_sidecar.contracts.common import ContractMeta
from ai_sidecar.contracts.crewai import CrewStrategizeRequest
from ai_sidecar.contracts.autonomy import GoalStackState
from ai_sidecar.planner.schemas import PlannerResponse, StrategicPlan, TacticalIntentBundle
from ai_sidecar.planner.schemas import PlanHorizon, PlannerPlanRequest
from ai_sidecar.contracts.state import BotStateSnapshot
from ai_sidecar.reflex.circuit_breaker import ReflexCircuitBreaker
from ai_sidecar.hunting_zone_manager import HuntingZoneManager
from ai_sidecar.cost_mode import CostModeManager
from ai_sidecar.anti_detection import AntiDetection
from ai_sidecar.game_engine import GameIntelligenceEngine
from ai_sidecar.fleet.swarm_ai import (
    SwarmTacticsEngine, SwarmReflexSystem, RoleDiscoveryEngine,
    FormationType, SkillCombo,
)
from ai_sidecar.autonomy.goal_decomposer import GoalDecomposer, GoalHorizon, CrossHorizonSynergy
from ai_sidecar.npc_discovery import NPCDiscoveryEngine
from ai_sidecar.server_adaptation import ServerAdaptationEngine
from ai_sidecar.p2p_knowledge import P2PKnowledgeNode, P2PNetworkManager

logger = logging.getLogger(__name__)

def _emit_heuristic_actions(runtime_state, horizon: str, bot_id: str | None = None) -> int:
    """Emit heuristic actions to the action queue.
    Uses the first registered bot if none specified.
    Returns number of actions queued."""
    import logging
    _log = logging.getLogger(__name__)
    try:
        hs = getattr(runtime_state, "heuristic_service", None)
        if hs is None:
            return 0
        # Resolve bot_id from runtime if not specified
        if not bot_id:
            br = getattr(runtime_state, "bot_registry", None)
            if br is not None:
                try:
                    bots = br.list_bots()
                    if bots:
                        bot_id = str(bots[0])
                except Exception:
                    pass
        if not bot_id:
            snapshots = getattr(runtime_state, "snapshot_cache", None)
            if snapshots is not None:
                try:
                    latest = snapshots.latest()
                    if latest and isinstance(latest, dict) and latest.get("bot_id"):
                        bot_id = str(latest["bot_id"])
                except Exception:
                    pass
        bot_id = bot_id or "default"
        
        # Build signals from available state
        signals = {
            "hp_ratio": 1.0, "sp_ratio": 1.0,
            "combat.aggro_count": 0, "map_known": False,
            "weight_ratio": 0.0, "horizon": horizon,
            "recent_death": False,
        }
        snapshots = getattr(runtime_state, "snapshot_cache", None)
        if snapshots is not None:
            try:
                # Read snapshot data regardless of whether we have candidate bot_ids
                latest = snapshots.latest()
                if latest is not None:
                    if isinstance(latest, dict):
                        v = latest.get("vitals") or {}
                        signals["hp_ratio"] = float(v.get("hp_ratio", 1.0))
                        signals["sp_ratio"] = float(v.get("sp_ratio", 1.0))
                        c = latest.get("combat") or {}
                        signals["combat.aggro_count"] = int(c.get("aggro_count", 0))
                        signals["map_known"] = bool(latest.get("map_known", False))
                        inv = latest.get("inventory") or {}
                        signals["weight_ratio"] = float(inv.get("weight_ratio", 0.0))
                    else:
                        v = getattr(latest, "vitals", None) or {}
                        signals["hp_ratio"] = float(getattr(v, "hp_ratio", 1.0) or 1.0)
                        signals["sp_ratio"] = float(getattr(v, "sp_ratio", 1.0) or 1.0)
                        c = getattr(latest, "combat", None) or {}
                        signals["combat.aggro_count"] = int(getattr(c, "aggro_count", 0) or 0)
                        signals["map_known"] = bool(getattr(latest, "map_known", False))
                        inv = getattr(latest, "inventory", None) or {}
                        signals["weight_ratio"] = float(getattr(inv, "weight_ratio", 0.0) or 0.0)
            except Exception:
                pass
        assessment = hs.assess(signals)
        if not assessment.actions:
            _log.info("heuristic_no_actions horizon=%s signals=%s", horizon, str(signals)[:200])
            return 0
        aq = getattr(runtime_state, "action_queue", None)
        if aq is None:
            return 0
        from datetime import UTC, datetime, timedelta as _td
        from ai_sidecar.contracts.actions import ActionProposal, ActionPriorityTier
        queued = 0
        for ha in assessment.actions:
            import time as _t
            _now = datetime.now(UTC)
            proposal = ActionProposal(
                action_id=f"heuristic_{horizon}_{ha.domain}_{_t.monotonic_ns()}",
                kind=ha.kind, command=ha.command or "ai auto",
                priority_tier=ActionPriorityTier.tactical,
                source="planner",
                created_at=_now,
                expires_at=_now + _td(seconds=30),
                idempotency_key=f"heuristic_{horizon}_{ha.domain}",
                metadata={"domain": ha.domain, "confidence": ha.confidence, "horizon": horizon, "reason": ha.reason},
            )
            try:
                aq.enqueue(bot_id, proposal)
                queued += 1
            except Exception:
                _log.exception("heuristic_action_push_failed")
        if queued:
            _log.info("heuristic_actions_emitted: %d for %s bot_id=%s", queued, horizon, bot_id)
        return queued
    except Exception:
        _log.exception("heuristic_action_emission_failed")
    return 0


def _emit_game_engine_actions(runtime_state, horizon: str, bot_id: str | None = None, map_name: str = "") -> int:
    """Emit actions from the game engine + hunting zone manager.
    
    This is the core of the "no hardcoded" philosophy. Instead of hardcoding
    hunting maps in config, the game engine reads rAthena data and recommends
    the optimal hunting zone based on the bot's level, class, and equipment.
    
    Returns number of actions queued.
    """
    import logging
    _log = logging.getLogger(__name__)
    try:
        hzm = getattr(runtime_state, "hunting_zone_manager", None)
        if hzm is None:
            return 0
        
        # Resolve bot_id
        if not bot_id:
            br = getattr(runtime_state, "bot_registry", None)
            if br is not None:
                try:
                    bots = br.list_bots()
                    if bots:
                        bot_id = str(bots[0])
                except Exception:
                    pass
        if not bot_id:
            snapshots = getattr(runtime_state, "snapshot_cache", None)
            if snapshots is not None:
                try:
                    latest = snapshots.latest()
                    if latest and isinstance(latest, dict) and latest.get("bot_id"):
                        bot_id = str(latest["bot_id"])
                except Exception:
                    pass
        bot_id = bot_id or "default"
        
        # Get bot level from snapshot
        bot_level = 1
        bot_class = "novice"
        try:
            snapshots = getattr(runtime_state, "snapshot_cache", None)
            if snapshots is not None:
                latest = snapshots.get(bot_id) if bot_id and hasattr(snapshots, 'get') else snapshots.latest()
                if latest is not None:
                    if isinstance(latest, dict):
                        # Dict-style snapshot — try progression sub-object first
                        _prog = latest.get("progression") or latest.get("raw", {}).get("progression") or {}
                        bot_level = int(_prog.get("base_level", _prog.get("level", 0)) or 0)
                        if bot_level == 0:
                            bot_level = int(latest.get("base_level", latest.get("level", 1)) or 1)
                        bot_class = str(_prog.get("job_name", "") or latest.get("job_name", latest.get("class", "novice")) or "novice")
                        logger.info("SNAPSHOT_DEBUG_DICT: progression=%s base_level=%s bot_class=%s",
                                    dict(_prog), _prog.get("base_level"), bot_class)
                    else:
                        # Object-style snapshot — read from progression attribute
                        _prog = getattr(latest, "progression", None)
                        if _prog is not None:
                            bot_level = int(getattr(_prog, "base_level", 0) or 0)
                            bot_class = str(getattr(_prog, "job_name", "") or "novice")
                        if bot_level == 0:
                            bot_level = int(getattr(latest, "base_level", 1) or 1)
                            bot_class = str(getattr(latest, "job_name", "novice") or "novice")
                        logger.info("SNAPSHOT_DEBUG_OBJ: progression=%s base_level=%s bot_class=%s",
                                    getattr(_prog, '__dict__', {}) if _prog else None,
                                    getattr(_prog, 'base_level', 'MISSING') if _prog else 'NO_PROG',
                                    bot_class)
        except Exception:
            pass
        
        # Get existing zone assignments for multi-bot coordination
        existing_assignments = getattr(runtime_state, "zone_assignments", {})
        
        # Get game engine for advanced scoring
        game_engine = getattr(runtime_state, "game_engine", None)
        
        # Recommend hunting zone
        zones = hzm.recommend_zone(
            bot_level=bot_level,
            bot_class=bot_class,
            goal="leveling" if horizon in ("short_term", "medium_term") else "farming",
            game_engine=game_engine,
        )
        
        if not zones:
            _log.info("game_engine_no_zones: bot=%s level=%d - trying fallback", bot_id, bot_level)
            # Fallback zones: dynamically computed from knowledge data + level range
            zones = hzm._fallback_zones(bot_level) if hasattr(hzm, '_fallback_zones') else []
            if not zones:
                _log.info("game_engine_no_zones_all_empty: bot=%s level=%d", bot_id, bot_level)
                return 0
        
        best_zone = zones[0]
        
        # Check if we're already on the right map
        if map_name and best_zone.map_name in map_name:
            # Already on the right map — emit ai auto
            aq = getattr(runtime_state, "action_queue", None)
            if aq is None:
                return 0
            from datetime import UTC, datetime, timedelta
            from ai_sidecar.contracts.actions import ActionProposal, ActionPriorityTier
            import hashlib as _hashlib
            _short_id = _hashlib.md5(f"{bot_id}_game_engine_{horizon}_{time.time()}".encode()).hexdigest()[:16]
            proposal = ActionProposal(
                action_id=f"ge_{horizon}_{_short_id}",
                kind="command",
                command="ai auto",
                priority_tier=ActionPriorityTier.tactical,
                source="planner",
                created_at=datetime.now(UTC),
                expires_at=datetime.now(UTC) + timedelta(seconds=60),
                idempotency_key=f"ge_{horizon}_{_short_id}",
                metadata={"goal": "grind", "objective": f"Hunt {best_zone.primary_monster} on {best_zone.map_name}",
                          "horizon": horizon, "bot_id": bot_id, "source": "game_engine"},
            )
            aq.enqueue(bot_id, proposal)
            _log.info(
                "game_engine_action: bot=%s zone=%s monster=%s score=%.2f exp_hp=%.2f danger=%.2f zeny=%.0f",
                bot_id, best_zone.map_name, best_zone.primary_monster,
                best_zone.score, best_zone.exp_per_hp, best_zone.danger_score, best_zone.zeny_per_kill,
            )
            # Broadcast hunting zone to P2P network
            _p2p = getattr(runtime_state, "p2p_node", None)
            if _p2p is not None:
                try:
                    _p2p.broadcast_hunting_zone(
                        map_name=best_zone.map_name,
                        monster_name=best_zone.primary_monster,
                        score=best_zone.score,
                        exp_per_hp=best_zone.exp_per_hp,
                        danger_score=best_zone.danger_score,
                        zeny_per_kill=best_zone.zeny_per_kill,
                    )
                except Exception:
                    pass
            return 1
        
        # Need to move to the hunting zone
        aq = getattr(runtime_state, "action_queue", None)
        if aq is None:
            return 0
        from datetime import UTC, datetime, timedelta
        from ai_sidecar.contracts.actions import ActionProposal, ActionPriorityTier
        import hashlib as _hashlib
        _short_id = _hashlib.md5(f"{bot_id}_game_engine_move_{horizon}_{time.time()}".encode()).hexdigest()[:16]
        proposal = ActionProposal(
            action_id=f"ge_move_{horizon}_{_short_id}",
            kind="command",
            command=f"move {best_zone.map_name}",
            priority_tier=ActionPriorityTier.tactical,
            source="planner",
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(seconds=120),
            idempotency_key=f"ge_move_{horizon}_{_short_id}",
            metadata={"goal": "travel", "objective": f"Move to {best_zone.map_name} for {best_zone.primary_monster}",
                      "horizon": horizon, "bot_id": bot_id, "source": "game_engine",
                      "target_map": best_zone.map_name, "reason": best_zone.reason},
        )
        aq.enqueue(bot_id, proposal)
        _log.info(
            "game_engine_move: bot=%s target=%s monster=%s score=%.2f reason=%s",
            bot_id, best_zone.map_name, best_zone.primary_monster,
            best_zone.score, best_zone.reason,
        )
        return 1
    except Exception:
        _log.exception("game_engine_action_emission_failed")
    return 0


def _emit_swarm_actions(runtime_state, horizon: str, bot_id: str | None = None) -> int:
    """Emit swarm AI actions — formations, skill combos, role discovery.
    
    Wires the previously dead swarm_ai.py (753 lines) into the action pipeline.
    Selects formation based on party composition, threat level, and AoE risk.
    Coordinates skill combos between bots.
    """
    import logging
    _log = logging.getLogger(__name__)
    try:
        swarm = getattr(runtime_state, "swarm_tactics", None)
        if swarm is None:
            return 0
        
        # Resolve bot_id
        if not bot_id:
            br = getattr(runtime_state, "bot_registry", None)
            if br is not None:
                try:
                    bots = br.list_bots()
                    if bots:
                        bot_id = str(bots[0])
                except Exception:
                    pass
        bot_id = bot_id or "default"
        
        # Get snapshot for state
        snapshots = getattr(runtime_state, "snapshot_cache", None)
        snapshot = None
        if snapshots is not None:
            try:
                snapshot = snapshots.latest()
            except Exception:
                pass
        
        # Get bot roles from role discovery
        roles = {}
        role_discovery = getattr(runtime_state, "role_discovery", None)
        if role_discovery is not None and snapshot is not None:
            try:
                roles = role_discovery.discover_roles(snapshot)
            except Exception:
                pass
        
        # Build bot list for swarm API — roles is a list from discover_roles(), not a dict
        _roles_list = roles if isinstance(roles, list) else (list(roles.values()) if isinstance(roles, dict) else ["idle"])
        _bots_list = [{"role": r, "hp_pct": 1.0, "bot_id": f"{bot_id}_{i}"} for i, r in enumerate(_roles_list[:12])]
        
        # Select formation based on party size and threat
        party_size = 1
        threat_level = 0
        aoe_risk = False
        team_hp_avg = 1.0
        try:
            if snapshot is not None:
                if isinstance(snapshot, dict):
                    party_size = int(snapshot.get("party_size", 1) or 1)
                    c = snapshot.get("combat", {}) or {}
                    threat_level = int(c.get("aggro_count", 0) or 0)
                    vitals = snapshot.get("vitals", {}) or {}
                    hp = float(vitals.get("hp_ratio", 1.0) or 1.0)
                    team_hp_avg = hp
                else:
                    party_size = int(getattr(snapshot, "party_size", 1) or 1)
                    c = getattr(snapshot, "combat", None) or {}
                    threat_level = int(getattr(c, "aggro_count", 0) or 0)
                    vitals = getattr(snapshot, "vitals", None) or {}
                    hp = float(getattr(vitals, "hp_ratio", 1.0) or 1.0)
                    team_hp_avg = hp
        except Exception:
            pass
        
        # Use party_id = bot_id, pass bots list with roles
        formation = swarm.select_formation(
            party_id=bot_id,
            bots=_bots_list,
            target_count=threat_level,
            threat_level=threat_level,
            aoe_risk=aoe_risk,
            team_hp_avg=team_hp_avg,
        )
        
        # WIRE ALL SWARM SUBSYSTEMS:
        # 1. Formation positioning
        formation_positions = swarm.get_formation_positions(
            formation=formation,
            bots=_bots_list,
            anchor_x=0, anchor_y=0,  # Relative to target
        )
        
        # 2. Movement suggestion for this bot
        move_cmd = swarm.suggest_movement(
            party_id=bot_id,
            bot_id=bot_id,
            formation=formation,
            positions=formation_positions,
        )
        
        # 3. Aggro management
        tank_id = None
        for b in _bots_list:
            if b["role"] == "tank":
                tank_id = b["bot_id"]
                break
        aggro_cmd = swarm.manage_aggro(
            party_id=bot_id,
            bots=_bots_list,
            tank_id=tank_id,
        )
        
        # 4. Heal suggestion
        heal_target = swarm.suggest_heal_target(
            party_id=bot_id,
            bots=_bots_list,
        )
        
        # 5. Select skill combo based on roles and target
        combo = swarm.select_combo(
            party_id=bot_id,
            bots=_bots_list,
            target_type="boss" if threat_level >= 4 else "normal",
        )
        
        # 6. Skill combo execution with step advancement
        combo_cmd = None
        combo_step = 0
        if combo is not None:
            # Track combo state across cycles using runtime state
            _combo_state = getattr(runtime_state, "swarm_combo_state", {})
            _party_combo = _combo_state.get(bot_id, {})
            _active_combo_name = _party_combo.get("name", "")
            _active_step = _party_combo.get("step", 0)
            
            if _active_combo_name == combo.name:
                # Continuing the same combo — advance step
                combo_step = _active_step + 1
                if swarm.is_combo_complete(party_id=bot_id, combo=combo):
                    swarm.complete_combo(party_id=bot_id, combo=combo)
                    _combo_state[bot_id] = {}
                    combo_step = 0
                    combo_cmd = None
                    _log.info("swarm_combo_complete: bot=%s combo=%s", bot_id, combo.name)
                else:
                    # Execute current step
                    combo_cmd = swarm.execute_combo_step(
                        party_id=bot_id, combo=combo, step_index=combo_step,
                        bots={b["bot_id"]: b for b in _bots_list},
                    )
                    _combo_state[bot_id] = {"name": combo.name, "step": combo_step}
            else:
                # New combo — start at step 0
                combo_cmd = swarm.execute_combo_step(
                    party_id=bot_id, combo=combo, step_index=0,
                    bots={b["bot_id"]: b for b in _bots_list},
                )
                _combo_state[bot_id] = {"name": combo.name, "step": 0}
            # Prune old combo states (keep last 100 bots)
            if len(_combo_state) > 100:
                _keys = list(_combo_state.keys())
                for _k in _keys[:-50]:
                    del _combo_state[_k]
            runtime_state.swarm_combo_state = _combo_state
        
        # 7. Swarm reflex assessment
        reflex_cmd = None
        _swarm_reflex = getattr(runtime_state, "swarm_reflex", None)
        if _swarm_reflex is not None:
            _signals = {"mvp_spotted": "", "nearby_hostiles": threat_level, "pvp_attacked": ""}
            _reflex_result = _swarm_reflex.assess(party_id=bot_id, bots=_bots_list, signals=_signals)
            if _reflex_result:
                reflex_cmd = _reflex_result.get("action", "")
                _log.info("swarm_reflex: bot=%s response=%s", bot_id, _reflex_result.get("response", ""))
        
        # 8. Cross-horizon synergy check with real long-term goal
        _gd = getattr(runtime_state, "goal_decomposer", None)
        if _gd is not None:
            _synergy = getattr(runtime_state, "synergy_engine", None)
            if _synergy is not None:
                # Track long-term goal across PDCA cycles via runtime state
                _long_term_goals = getattr(runtime_state, "long_term_goals", {})
                _long_term = _long_term_goals.get(bot_id, "")
                # Update long-term goal from goal decomposer's progress
                _gd_progress = _gd.progress(bot_id=bot_id)
                if _gd_progress.get("parent"):
                    _long_term = str(_gd_progress["parent"])
                    _long_term_goals[bot_id] = _long_term
                    # Prune old entries (keep last 100 bots)
                    if len(_long_term_goals) > 100:
                        _keys = list(_long_term_goals.keys())
                        for _k in _keys[:-50]:
                            del _long_term_goals[_k]
                    runtime_state.long_term_goals = _long_term_goals
                _conflicts = _synergy.detect_conflicts(
                    short_term_goal=formation.value, long_term_goal=_long_term
                )
                if _conflicts:
                    _log.info("goal_conflict: bot=%s short=%s long=%s conflicts=%s",
                              bot_id, formation.value, _long_term, _conflicts)
        
        # Emit formation command
        aq = getattr(runtime_state, "action_queue", None)
        if aq is None:
            return 0
        
        from datetime import UTC, datetime, timedelta
        from ai_sidecar.contracts.actions import ActionProposal, ActionPriorityTier
        import hashlib as _hashlib
        
        queued = 0
        
        # Emit formation action
        # Skip if bot is in a town (no hunting needed) — game engine will route out
        _in_town = False
        try:
            _snap = getattr(runtime_state, "snapshot_cache", None)
            if _snap is not None and bot_id:
                _s = _snap.get(bot_id) if hasattr(_snap, 'get') else _snap.latest()
                if _s is not None:
                    _map = str(getattr(getattr(_s, 'position', None), 'map', '') or '')
                    if _map and any(t in _map.lower() for t in ['prontera', 'morocc', 'payon', 'geffen', 'aldebaran', 'yuno', 'xmas', 'amatsu']):
                        _in_town = True
        except Exception:
            pass
        
        # Don't emit swarm actions in town — game engine will route bot to hunting zone
        if _in_town:
            _log.info("swarm_skipped_town: bot=%s map=%s", bot_id, _map if '_map' in dir() else '?')
            return 0
        
        _short_id = _hashlib.md5(f"{bot_id}_swarm_fmt_{horizon}_{time.time()}".encode()).hexdigest()[:16]
        
        # Determine the actual command to execute
        cmd = "ai auto"
        cmd_meta = f"Formation: {formation.value}"
        
        # Priority: combo > aggro > heal > movement > formation
        if combo_cmd and combo is not None:
            cmd = combo_cmd
            cmd_meta = f"Combo: {combo.name} step 1"
        elif aggro_cmd:
            cmd = aggro_cmd
            cmd_meta = f"Aggro: {aggro_cmd}"
        elif heal_target:
            cmd = f"ai auto"  # Healer will handle via reflex
            cmd_meta = f"Heal target: {heal_target}"
        elif move_cmd:
            cmd = move_cmd
            cmd_meta = f"Move to formation: {formation.value}"
        
        proposal = ActionProposal(
            action_id=f"swarm_fmt_{horizon}_{_short_id}",
            kind="command",
            command=cmd,
            priority_tier=ActionPriorityTier.tactical,
            source="fleet",
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(seconds=60),
            idempotency_key=f"swarm_fmt_{horizon}_{_short_id}",
            metadata={
                "goal": "swarm_formation",
                "objective": f"Formation: {formation.value}",
                "horizon": horizon, "bot_id": bot_id, "source": "swarm_ai",
                "formation": formation.value,
                "combo": combo.name if combo else "none",
            },
        )
        aq.enqueue(bot_id, proposal)
        queued += 1
        
        _log.info(
            "swarm_action: bot=%s formation=%s combo=%s party=%d threat=%d",
            bot_id, formation.value, combo.name if combo else "none",
            party_size, threat_level,
        )
        return queued
    except Exception:
        _log.exception("swarm_action_emission_failed")
    return 0


def _emit_vendor_actions(runtime_state, horizon: str, bot_id: str | None = None) -> int:
    """Emit vendor/storage actions when inventory is full.
    
    Uses game engine's valuate_item() to determine what to sell vs keep.
    Routes bot to nearest town for selling/storage.
    """
    import logging
    _log = logging.getLogger(__name__)
    try:
        # Check weight ratio from snapshot
        snapshots = getattr(runtime_state, "snapshot_cache", None)
        if snapshots is None:
            return 0
        
        try:
            latest = snapshots.latest()
        except Exception:
            return 0
        if latest is None:
            return 0
        
        weight_ratio = 0.0
        map_name = ""
        if isinstance(latest, dict):
            inv = latest.get("inventory", {}) or {}
            weight_ratio = float(inv.get("weight_ratio", 0.0) or 0.0)
            map_name = str(latest.get("map", latest.get("position", {}).get("map", "")) or "")
        else:
            inv = getattr(latest, "inventory", None) or {}
            weight_ratio = float(getattr(inv, "weight_ratio", 0.0) or 0.0)
            pos = getattr(latest, "position", None)
            map_name = str(getattr(pos, "map", "") if pos else "")
        
        # Only act when near full (>80% weight)
        if weight_ratio < 0.80:
            return 0
        
        # Resolve bot_id
        if not bot_id:
            if isinstance(latest, dict) and latest.get("bot_id"):
                bot_id = str(latest["bot_id"])
        bot_id = bot_id or "default"
        
        # Determine nearest town from map name using NPC discovery
        npc_disc = getattr(runtime_state, "npc_discovery", None)
        if npc_disc is not None:
            town_map = npc_disc.get_nearest_town_for_map(map_name)
        else:
            # Fallback if NPC discovery not available
            town_map = "prontera"
            if "payon" in map_name.lower() or "pay_" in map_name.lower():
                town_map = "payon"
            elif "morocc" in map_name.lower() or "moc_" in map_name.lower():
                town_map = "morocc"
            elif "geffen" in map_name.lower() or "gef_" in map_name.lower():
                town_map = "geffen"
            elif "aldebaran" in map_name.lower() or "alde_" in map_name.lower():
                town_map = "aldebaran"
            elif "yuno" in map_name.lower():
                town_map = "yuno"
            elif "xmas" in map_name.lower():
                town_map = "xmas"
            elif "amatsu" in map_name.lower() or "ama_" in map_name.lower():
                town_map = "amatsu"
        
        # Check if already in town
        if map_name and town_map in map_name.lower():
            # In town — discover NPC positions dynamically
            # Try to find vendor NPC, fall back to storage NPC
            npc_cmd = None
            if npc_disc is not None:
                npc_cmd = npc_disc.get_command_for_service(latest, map_name, "vendor")
                if not npc_cmd:
                    npc_cmd = npc_disc.get_command_for_service(latest, map_name, "storage")
            
            # If NPC discovered, use talknpc; otherwise just ai auto
            cmd = npc_cmd if npc_cmd else "ai auto"
            
            aq = getattr(runtime_state, "action_queue", None)
            if aq is None:
                return 0
            from datetime import UTC, datetime, timedelta
            from ai_sidecar.contracts.actions import ActionProposal, ActionPriorityTier
            import hashlib as _hashlib
            _short_id = _hashlib.md5(f"{bot_id}_vendor_{horizon}_{time.time()}".encode()).hexdigest()[:16]
            proposal = ActionProposal(
                action_id=f"vendor_{horizon}_{_short_id}",
                kind="command",
                command=cmd,
                priority_tier=ActionPriorityTier.tactical,
                source="planner",
                created_at=datetime.now(UTC),
                expires_at=datetime.now(UTC) + timedelta(seconds=60),
                idempotency_key=f"vendor_{horizon}_{_short_id}",
                metadata={
                    "goal": "economy",
                    "objective": f"Sell items in {town_map}, weight={weight_ratio:.0%}",
                    "horizon": horizon, "bot_id": bot_id, "source": "vendor_ai",
                    "needs_vendor": True, "town": town_map,
                },
            )
            aq.enqueue(bot_id, proposal)
            _log.info("vendor_action: bot=%s town=%s weight=%.0f%% cmd=%s", bot_id, town_map, weight_ratio * 100, cmd)
            return 1
        
        # Not in town — route to nearest town
        aq = getattr(runtime_state, "action_queue", None)
        if aq is None:
            return 0
        from datetime import UTC, datetime, timedelta
        from ai_sidecar.contracts.actions import ActionProposal, ActionPriorityTier
        import hashlib as _hashlib
        _short_id = _hashlib.md5(f"{bot_id}_vendor_move_{horizon}_{time.time()}".encode()).hexdigest()[:16]
        proposal = ActionProposal(
            action_id=f"vendor_move_{horizon}_{_short_id}",
            kind="command",
            command=f"move {town_map}",
            priority_tier=ActionPriorityTier.strategic,
            source="planner",
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(seconds=120),
            idempotency_key=f"vendor_move_{horizon}_{_short_id}",
            metadata={
                "goal": "economy",
                "objective": f"Return to {town_map} to sell, weight={weight_ratio:.0%}",
                "horizon": horizon, "bot_id": bot_id, "source": "vendor_ai",
                "target_map": town_map, "needs_vendor": True,
            },
        )
        aq.enqueue(bot_id, proposal)
        _log.info("vendor_move: bot=%s target=%s weight=%.0f%%", bot_id, town_map, weight_ratio * 100)
        return 1
    except Exception:
        _log.exception("vendor_action_emission_failed")
    return 0



# ── Class skill definitions for _emit_skill_actions ───────────────────────────────────────────
# Keyed by canonical class name, each entry has attack and buff lists of (skill_name, min_base_level).
CLASS_SKILLS: dict[str, dict[str, list[tuple[str, int]]]] = {
    "acolyte": {"attack": [("basic_attack", 1)], "buffs": [("heal", 1), ("cure", 1), ("increase_agility", 15), ("bless", 20)]},
    "alchemist": {"attack": [("acid_demonstration", 40)], "buffs": [("increase_agility", 1), ("bless", 1), ("fire_ins", 30), ("learning_potion", 50)]},
    "arch_bishop": {"attack": [("magnus_exorcismus", 50), ("adventus", 90)], "buffs": [("heal", 1), ("cure", 1), ("increase_agility", 15), ("bless", 20), ("highness_heal", 40), ("resurection", 30), ("assumptio", 70), ("epiclesis", 100)]},
    "arch_mage": {"attack": [("fire_bolt", 1), ("cold_bolt", 1), ("lightning_bolt", 1), ("frost_diver", 15), ("fire_ball", 30), ("fire_wall", 40), ("frost_nova", 50), ("storm_gust", 60), ("meteor_storm", 70), ("heaven_drive", 80), ("crimson_arrow", 100)], "buffs": [("increase_agility", 1), ("energy_coat", 40), ("mystical_amplification", 70)]},
    "archer": {"attack": [("double_strafing", 1), ("arrow_shower", 20)], "buffs": [("increase_agility", 1)]},
    "assassin": {"attack": [("double_attack", 1), ("sonic_blow", 30), ("grimtooth", 50)], "buffs": [("hiding", 15), ("venom_dust", 40)]},
    "assassin_cross": {"attack": [("double_attack", 1), ("sonic_blow", 30), ("grimtooth", 50)], "buffs": [("hiding", 15), ("venom_dust", 40), ("enchant_poison", 60)]},
    "bard": {"attack": [("double_strafing", 1), ("arrow_shower", 20), ("musical_instrument", 50)], "buffs": [("increase_agility", 1), ("bless", 1)]},
    "biolo": {"attack": [("cart_cannon", 1), ("acid_terror", 70), ("hell_plant", 80), ("bio_explosion", 100)], "buffs": [("increase_agility", 1), ("bless", 1), ("fire_ins", 30), ("learning_potion", 50), ("sphere_mine", 60)]},
    "blacksmith": {"attack": [("mammonite", 30), ("cart_revolution", 40), ("hammer_fall", 50)], "buffs": [("increase_agility", 1), ("weapon_perfection", 40), ("overthrust", 60)]},
    "cardinal": {"attack": [("magnus_exorcismus", 50)], "buffs": [("heal", 1), ("cure", 1), ("increase_agility", 15), ("bless", 20), ("highness_heal", 40), ("resurection", 30), ("assumptio", 70), ("epiclesis", 100)]},
    "champion": {"attack": [("triple_attack", 1), ("chain_combo", 30), ("finger_offensive", 50), ("asura", 70)], "buffs": [("increase_agility", 1), ("bless", 1), ("enchant_rage", 60)]},
    "clown": {"attack": [("double_strafing", 1), ("arrow_shower", 20), ("musical_instrument", 50), ("metallic_sound", 70)], "buffs": [("increase_agility", 1), ("bless", 1)]},
    "creator": {"attack": [("acid_demonstration", 40), ("acid_terror", 70)], "buffs": [("increase_agility", 1), ("bless", 1), ("fire_ins", 30), ("learning_potion", 50)]},
    "dancer": {"attack": [("double_strafing", 1), ("arrow_shower", 20), ("arrow_vulcan", 50)], "buffs": [("increase_agility", 1), ("bless", 1)]},
    "dragon_knight": {"attack": [("bowling_bash", 1), ("brandish_spear", 30), ("pierce", 50), ("spiral_pierce", 70), ("rune_mastery", 90), ("dragon_breath", 100)], "buffs": [("provoke", 1), ("aura_blade", 60), ("enchant_blade", 80)]},
    "elemental_master": {"attack": [("fire_bolt", 1), ("cold_bolt", 1), ("lightning_bolt", 1), ("diamond_dust", 80), ("lightning_cloud", 80), ("violet_force", 80), ("elemental_kill", 100)], "buffs": [("increase_agility", 1), ("energy_coat", 30), ("fiber_lock", 60), ("memorize", 70)]},
    "genetic": {"attack": [("cart_cannon", 1), ("acid_terror", 70), ("hell_plant", 80)], "buffs": [("increase_agility", 1), ("bless", 1), ("fire_ins", 30), ("learning_potion", 50), ("sphere_mine", 60)]},
    "guillotine_cross": {"attack": [("double_attack", 1), ("sonic_blow", 30), ("grimtooth", 50), ("cross_impact", 80)], "buffs": [("hiding", 15), ("venom_dust", 40), ("enchant_poison", 60), ("new_poison_research", 70)]},
    "gunslinger": {"attack": [("chain_action", 1), ("rapid_shower", 30), ("tracking", 50)], "buffs": [("increase_agility", 1), ("bless", 1)]},
    "gypsy": {"attack": [("double_strafing", 1), ("arrow_shower", 20), ("arrow_vulcan", 50), ("ravenous_wolf", 70)], "buffs": [("increase_agility", 1), ("bless", 1)]},
    "high_priest": {"attack": [("magnus_exorcismus", 50)], "buffs": [("heal", 1), ("cure", 1), ("increase_agility", 15), ("bless", 20), ("highness_heal", 40), ("resurection", 30), ("assumptio", 70)]},
    "high_wizard": {"attack": [("fire_bolt", 1), ("cold_bolt", 1), ("lightning_bolt", 1), ("frost_diver", 15), ("fire_ball", 30), ("fire_wall", 40), ("frost_nova", 50), ("storm_gust", 60), ("meteor_storm", 70), ("heaven_drive", 80)], "buffs": [("increase_agility", 1), ("energy_coat", 40)]},
    "hunter": {"attack": [("double_strafing", 1), ("arrow_shower", 20), ("blitz_beat", 40), ("beast_tamer", 60)], "buffs": [("increase_agility", 1)]},
    "kagerou": {"attack": [("throw_shuriken", 1), ("throw_kunai", 20), ("fire_blossom", 35), ("crimson_seal", 60)], "buffs": [("increase_agility", 1)]},
    "knight": {"attack": [("bowling_bash", 1), ("brandish_spear", 30), ("pierce", 50)], "buffs": [("provoke", 1)]},
    "lord_knight": {"attack": [("bowling_bash", 1), ("brandish_spear", 30), ("pierce", 50), ("spiral_pierce", 70)], "buffs": [("provoke", 1), ("aura_blade", 60)]},
    "mage": {"attack": [("fire_bolt", 1), ("cold_bolt", 1), ("lightning_bolt", 1), ("frost_diver", 15), ("stone_curse", 30)], "buffs": []},
    "meister": {"attack": [("mammonite", 30), ("cart_revolution", 40), ("hammer_fall", 50), ("mighty_push", 80)], "buffs": [("increase_agility", 1), ("weapon_perfection", 40), ("overthrust", 60)]},
    "merchant": {"attack": [("mammonite", 30), ("cart_revolution", 40)], "buffs": [("increase_agility", 1)]},
    "minstrel": {"attack": [("double_strafing", 1), ("arrow_shower", 20), ("musical_instrument", 50), ("metallic_sound", 70), ("sound_blend", 90)], "buffs": [("increase_agility", 1), ("bless", 1), ("lyrical", 80)]},
    "monk": {"attack": [("triple_attack", 1), ("chain_combo", 30), ("finger_offensive", 50)], "buffs": [("increase_agility", 1), ("bless", 1)]},
    "ninja": {"attack": [("throw_shuriken", 1), ("throw_kunai", 20), ("fire_blossom", 35)], "buffs": [("increase_agility", 1)]},
    "novice": {"attack": [("basic_attack", 1)], "buffs": []},
    "oboro": {"attack": [("throw_shuriken", 1), ("throw_kunai", 20), ("fire_blossom", 35), ("dragon_seal", 60)], "buffs": [("increase_agility", 1)]},
    "paladin": {"attack": [("holy_cross", 1), ("shield_boomerang", 30), ("grand_cross", 50)], "buffs": [("provoke", 1), ("increase_agility", 1)]},
    "priest": {"attack": [("magnus_exorcismus", 50)], "buffs": [("heal", 1), ("cure", 1), ("increase_agility", 15), ("bless", 20), ("highness_heal", 40), ("resurection", 30)]},
    "professor": {"attack": [("fire_bolt", 1), ("cold_bolt", 1), ("lightning_bolt", 1), ("dispel", 50)], "buffs": [("increase_agility", 1), ("energy_coat", 30), ("fiber_lock", 60), ("memorize", 70)]},
    "ranger": {"attack": [("aimed_bolt", 1), ("arrow_shower", 20), ("blitz_beat", 40), ("beast_tamer", 60), ("true_sight", 70), ("cambias_volley", 90)], "buffs": [("increase_agility", 1), ("wind_walk", 70)]},
    "rebel": {"attack": [("chain_action", 1), ("rapid_shower", 30), ("tracking", 50), ("void_man", 80)], "buffs": [("increase_agility", 1), ("bless", 1)]},
    "rogue": {"attack": [("double_attack", 1), ("steal", 10), ("backstab", 30)], "buffs": [("hiding", 15), ("stalk", 40)]},
    "royal_guard": {"attack": [("holy_cross", 1), ("shield_boomerang", 30), ("grand_cross", 50), ("cannon_spear", 80)], "buffs": [("provoke", 1), ("increase_agility", 1)]},
    "rune_knight": {"attack": [("bowling_bash", 1), ("brandish_spear", 30), ("pierce", 50), ("spiral_pierce", 70), ("rune_mastery", 90)], "buffs": [("provoke", 1), ("aura_blade", 60), ("enchant_blade", 80)]},
    "sage": {"attack": [("fire_bolt", 1), ("cold_bolt", 1), ("lightning_bolt", 1)], "buffs": [("increase_agility", 1), ("energy_coat", 30)]},
    "shadow_chaser": {"attack": [("double_attack", 1), ("steal", 10), ("backstab", 30), ("triangle_shot", 60), ("masquerade", 90)], "buffs": [("hiding", 15), ("stalk", 40), ("preserve", 70)]},
    "shadow_cross": {"attack": [("double_attack", 1), ("sonic_blow", 30), ("grimtooth", 50), ("cross_impact", 80), ("shadow_eternal", 100)], "buffs": [("hiding", 15), ("venom_dust", 40), ("enchant_poison", 60), ("new_poison_research", 70)]},
    "sniper": {"attack": [("double_strafing", 1), ("arrow_shower", 20), ("blitz_beat", 40), ("beast_tamer", 60), ("true_sight", 70)], "buffs": [("increase_agility", 1)]},
    "sorcerer": {"attack": [("fire_bolt", 1), ("cold_bolt", 1), ("lightning_bolt", 1), ("diamond_dust", 80), ("lightning_cloud", 80), ("violet_force", 80)], "buffs": [("increase_agility", 1), ("energy_coat", 30), ("fiber_lock", 60), ("memorize", 70)]},
    "soul_ascetic": {"attack": [("fire_bolt", 20), ("cold_bolt", 20), ("lightning_bolt", 20), ("eska", 50), ("esma", 60), ("es_soul", 90)], "buffs": [("ka_ahi", 30), ("ka_na", 30), ("estun", 40)]},
    "soul_linker": {"attack": [("fire_bolt", 20), ("cold_bolt", 20), ("lightning_bolt", 20), ("eska", 50)], "buffs": [("ka_ahi", 30), ("ka_na", 30), ("estun", 40), ("esma", 60)]},
    "soul_reaper": {"attack": [("fire_bolt", 20), ("cold_bolt", 20), ("lightning_bolt", 20), ("eska", 50), ("esma", 60)], "buffs": [("ka_ahi", 30), ("ka_na", 30), ("estun", 40)]},
    "stalker": {"attack": [("double_attack", 1), ("steal", 10), ("backstab", 30), ("triangle_shot", 60)], "buffs": [("hiding", 15), ("stalk", 40), ("preserve", 70)]},
    "star_gladiator": {"attack": [("basic_attack", 1), ("star_strike", 50)], "buffs": [("increase_agility", 1), ("bless", 1), ("wrath_crocodile", 30), ("fusion_crocodile", 30)]},
    "super_novice": {"attack": [("basic_attack", 1)], "buffs": [("increase_agility", 40), ("bless", 40)]},
    "sura": {"attack": [("triple_attack", 1), ("chain_combo", 30), ("finger_offensive", 50), ("asura", 70), ("fist_ancient", 90)], "buffs": [("increase_agility", 1), ("bless", 1), ("enchant_rage", 60)]},
    "swordman": {"attack": [("bash", 1), ("magnum_break", 25)], "buffs": [("provoke", 1)]},
    "taekwon": {"attack": [("flying_side_kick", 1), ("whirlwind_kick", 30)], "buffs": [("increase_agility", 1)]},
    "thief": {"attack": [("double_attack", 1), ("steal", 10)], "buffs": [("hiding", 15)]},
    "wanderer": {"attack": [("double_strafing", 1), ("arrow_shower", 20), ("arrow_vulcan", 50), ("ravenous_wolf", 70), ("dazzler", 90)], "buffs": [("increase_agility", 1), ("bless", 1)]},
    "whitesmith": {"attack": [("mammonite", 30), ("cart_revolution", 40), ("hammer_fall", 50)], "buffs": [("increase_agility", 1), ("weapon_perfection", 40), ("overthrust", 60)]},
    "windhawk": {"attack": [("aimed_bolt", 1), ("arrow_shower", 20), ("blitz_beat", 40), ("beast_tamer", 60), ("true_sight", 70), ("cambias_volley", 90)], "buffs": [("increase_agility", 1), ("wind_walk", 70)]},
    "wizard": {"attack": [("fire_bolt", 1), ("cold_bolt", 1), ("lightning_bolt", 1), ("frost_diver", 15), ("fire_ball", 30), ("fire_wall", 40), ("frost_nova", 50), ("storm_gust", 60), ("meteor_storm", 70)], "buffs": [("increase_agility", 1), ("energy_coat", 40)]},
}

# Buff names that can be checked against snapshot active_buffs.
# Maps skill name → canonical active_buff string.
_BUFF_ALIASES: dict[str, str] = {
    "increase_agility": "agi_up",
    "bless": "bless",
    "provoke": "provoke",
    "hiding": "hiding",
    "venom_dust": "venom_dust",
    "enchant_poison": "enchant_poison",
    "weapon_perfection": "weapon_perfection",
    "overthrust": "overthrust",
    "energy_coat": "energy_coat",
    "aura_blade": "aura_blade",
    "assumptio": "assumptio",
    "enchant_rage": "enchant_rage",
    "wind_walk": "wind_walk",
    "ka_ahi": "ka_ahi",
    "ka_na": "ka_na",
    "estun": "estun",
    "esma": "esma",
    "preserve": "preserve",
    "mystical_amplification": "mystical_amplification",
    "enchant_blade": "enchant_blade",
    "wrath_crocodile": "wrath_crocodile",
    "fusion_crocodile": "fusion_crocodile",
    "memorize": "memorize",
    "fiber_lock": "fiber_lock",
    "lyrical": "lyrical",
    "stalk": "stalk",
    "new_poison_research": "new_poison_research",
    "sphere_mine": "sphere_mine",
    "cure": "cure",
    "heal": "heal",
    "highness_heal": "highness_heal",
    "resurection": "resurection",
    "epiclesis": "epiclesis",
    "fire_ins": "fire_ins",
    "learning_potion": "learning_potion",
}

# Map OpenKore job_name strings → canonical CLASS_SKILLS keys.
_CLASS_ALIASES: dict[str, str] = {
    "novice": "novice",
    "super_novice": "super_novice",
    "swordman": "swordman",
    "knight": "knight",
    "lord_knight": "lord_knight",
    "rune_knight": "rune_knight",
    "dragon_knight": "dragon_knight",
    "paladin": "paladin",
    "royal_guard": "royal_guard",
    "mage": "mage",
    "wizard": "wizard",
    "high_wizard": "high_wizard",
    "arch_mage": "arch_mage",
    "sage": "sage",
    "professor": "professor",
    "sorcerer": "sorcerer",
    "elemental_master": "elemental_master",
    "acolyte": "acolyte",
    "priest": "priest",
    "high_priest": "high_priest",
    "arch_bishop": "arch_bishop",
    "cardinal": "cardinal",
    "monk": "monk",
    "champion": "champion",
    "sura": "sura",
    "archer": "archer",
    "hunter": "hunter",
    "sniper": "sniper",
    "ranger": "ranger",
    "windhawk": "windhawk",
    "bard": "bard",
    "clown": "clown",
    "minstrel": "minstrel",
    "dancer": "dancer",
    "gypsy": "gypsy",
    "wanderer": "wanderer",
    "thief": "thief",
    "assassin": "assassin",
    "assassin_cross": "assassin_cross",
    "guillotine_cross": "guillotine_cross",
    "shadow_cross": "shadow_cross",
    "rogue": "rogue",
    "stalker": "stalker",
    "shadow_chaser": "shadow_chaser",
    "merchant": "merchant",
    "blacksmith": "blacksmith",
    "whitesmith": "whitesmith",
    "meister": "meister",
    "alchemist": "alchemist",
    "creator": "creator",
    "genetic": "genetic",
    "biolo": "biolo",
    "soul_linker": "soul_linker",
    "soul_reaper": "soul_reaper",
    "soul_ascetic": "soul_ascetic",
    "star_gladiator": "star_gladiator",
    "taekwon": "taekwon",
    "gunslinger": "gunslinger",
    "rebel": "rebel",
    "ninja": "ninja",
    "kagerou": "kagerou",
    "oboro": "oboro",
}

def _resolve_bot_class(bot_class_raw: str) -> str:
    """Normalise raw class name to a canonical CLASS_SKILLS key."""
    raw = bot_class_raw.lower().strip().replace(" ", "_").replace("-", "_")
    if raw in _CLASS_ALIASES:
        return _CLASS_ALIASES[raw]
    # Substring fallback
    for alias_key, canonical in _CLASS_ALIASES.items():
        if alias_key in raw or raw in alias_key:
            return canonical
    return "novice"


def _level_gate(skill_name: str, min_base_level: int, bot_level: int) -> bool:
    """Return True if bot level >= required level for skill."""
    return bot_level >= min_base_level


def _is_buff_active(buff_name: str, active_buffs: list[str]) -> bool:
    """Return True if a buff with this name or alias is already active."""
    buff_lower = buff_name.lower().strip()
    alias = _BUFF_ALIASES.get(buff_lower, buff_lower)
    for active in active_buffs:
        act = active.lower().strip()
        if act == buff_lower or act == alias or alias in act or buff_lower in act:
            return True
    return False


def _extract_active_buffs(snapshot) -> list[str]:
    """Extract active buff names from a snapshot (handles dict + object)."""
    if isinstance(snapshot, dict):
        return list(snapshot.get("active_buffs", []) or [])
    raw = getattr(snapshot, "raw", None) or {}
    if isinstance(raw, dict):
        return list(raw.get("active_buffs", []) or [])
    return []


def _extract_bot_level(snapshot) -> int:
    """Extract bot base level from a snapshot (handles dict + object)."""
    if isinstance(snapshot, dict):
        v = snapshot.get("vitals") or {}
        level = v.get("level", 0) or 0
        if level > 0:
            return int(level)
        prog = snapshot.get("progression") or {}
        return int(prog.get("base_level", 1) or 1)
    # Object mode
    v = getattr(snapshot, "vitals", None)
    if v is not None:
        level = getattr(v, "level", 0) or 0
        if level > 0:
            return int(level)
    prog = getattr(snapshot, "progression", None)
    if prog is not None:
        return int(getattr(prog, "base_level", 1) or 1)
    return 1


def _emit_skill_actions(runtime_state, horizon: str, bot_id: str | None = None) -> int:
    """Emit skill rotation actions with SP management, level gating, buff reuse prevention,
    and rotation tracking.

    Returns number of actions queued."""
    import logging
    _log = logging.getLogger(__name__)

    try:
        game_engine = getattr(runtime_state, "game_engine", None)
        if game_engine is None:
            return 0

        # ── Snapshot ──
        snapshots = getattr(runtime_state, "snapshot_cache", None)
        if snapshots is None:
            return 0
        try:
            latest = snapshots.latest()
        except Exception:
            return 0
        if latest is None:
            return 0

        aq = getattr(runtime_state, "action_queue", None)
        if aq is None:
            return 0

        # ── Resolve bot_id ──
        if not bot_id:
            if isinstance(latest, dict):
                bot_id = str(latest.get("bot_id", "default"))
            else:
                bot_id = getattr(latest, "bot_id", "default") or "default"
        bot_id = bot_id or "default"

        # ── Extract bot info (dict + object handling) ──
        bot_class = "novice"
        bot_level = 1
        sp_ratio = 1.0
        active_buffs: list[str] = []
        mob_element = "Neutral"
        mob_race = "Formless"
        mob_size = "Medium"

        if isinstance(latest, dict):
            bot_class = str(latest.get("job_name", latest.get("class", "novice")) or "novice")
            bot_level = _extract_bot_level(latest)
            v = latest.get("vitals") or {}
            sp_ratio = float(v.get("sp_ratio", 1.0) or 1.0)
            target = latest.get("target", {}) or {}
            mob_element = str(target.get("element", "Neutral") or "Neutral")
            mob_race = str(target.get("race", "Formless") or "Formless")
            mob_size = str(target.get("size", "Medium") or "Medium")
            active_buffs = _extract_active_buffs(latest)
        else:
            bot_class = str(getattr(latest, "job_name", "novice") or "novice")
            bot_level = _extract_bot_level(latest)
            vt = getattr(latest, "vitals", None)
            sp_ratio = float(getattr(vt, "sp_ratio", 1.0) or 1.0) if vt else 1.0
            target = getattr(latest, "target", None) or {}
            mob_element = str(getattr(target, "element", "Neutral") or "Neutral")
            mob_race = str(getattr(target, "race", "Formless") or "Formless")
            mob_size = str(getattr(target, "size", "Medium") or "Medium")
            active_buffs = _extract_active_buffs(latest)

        # ── SP management: low SP → basic attack only ──
        if sp_ratio < 0.3:
            _log.info(
                "skill_action[%s]: sp_ratio=%.2f < 0.3 → basic attack only",
                bot_id, sp_ratio,
            )
            from datetime import UTC, datetime, timedelta
            from ai_sidecar.contracts.actions import ActionProposal, ActionPriorityTier
            import hashlib as _hashlib
            import time as _time
            _short_id = _hashlib.md5(f"{bot_id}_ata_{horizon}_{_time.time()}".encode()).hexdigest()[:16]
            proposal = ActionProposal(
                action_id=f"atk_{horizon}_{_short_id}",
                kind="command",
                command="attack_skill basic_attack",
                priority_tier=ActionPriorityTier.tactical,
                source="planner",
                created_at=datetime.now(UTC),
                expires_at=datetime.now(UTC) + timedelta(seconds=60),
                idempotency_key=f"atk_{horizon}_{_short_id}",
                metadata={
                    "goal": "combat",
                    "objective": "Basic attack (low SP)",
                    "horizon": horizon,
                    "bot_id": bot_id,
                    "source": "skill_ai",
                    "low_sp": True,
                    "sp_ratio": sp_ratio,
                },
            )
            aq.enqueue(bot_id, proposal)
            return 1

        # ── Resolve canonical class key ──
        canonical_class = _resolve_bot_class(bot_class)
        class_data = CLASS_SKILLS.get(canonical_class)
        if class_data is None:
            # Try substring matching against all keys
            for key, data in CLASS_SKILLS.items():
                if key in bot_class.lower().replace(" ", "_"):
                    class_data = data
                    canonical_class = key
                    break
        if class_data is None:
            _log.warning("skill_action[%s]: unknown class '%s', skipping", bot_id, bot_class)
            return 0

        # ── Build eligible skill list (level-gated) ──
        eligible_attacks: list[str] = []
        eligible_buffs: list[str] = []

        for skill_name, min_level in class_data.get("attack", []):
            if _level_gate(skill_name, min_level, bot_level):
                eligible_attacks.append(skill_name)

        for skill_name, min_level in class_data.get("buffs", []):
            if _level_gate(skill_name, min_level, bot_level):
                eligible_buffs.append(skill_name)

        # ── Get element recommendation from game engine ──
        skills = game_engine.recommend_skills_for_mob(
            job_name=bot_class,
            mob_element=mob_element,
            mob_race=mob_race,
            mob_size=mob_size,
        )

        # ── Initialise rotation state per bot ──
        rotation_state: dict = getattr(runtime_state, "skill_rotation_state", None)
        if rotation_state is None:
            rotation_state = {}
            object.__setattr__(runtime_state, "skill_rotation_state", rotation_state)

        bot_rotation = rotation_state.setdefault(bot_id, {
            "attack_index": 0,
            "buff_index": 0,
            "last_buff_ts": 0.0,
            "total_pick_count": 0,
        })
        # Prune old rotation state entries (keep last 100 bots)
        if len(rotation_state) > 100:
            _keys = list(rotation_state.keys())
            for _k in _keys[:-50]:
                del rotation_state[_k]

        now = time.time()

        # ── Decide: buff or attack? ──
        # Emit a buff roughly every 5 cycles if any buffs are available and not already active
        should_buff = (
            bool(eligible_buffs)
            and bot_rotation["total_pick_count"] % 5 == 0
            and (now - bot_rotation.get("last_buff_ts", 0.0)) > 30.0
        )

        from datetime import UTC, datetime, timedelta
        from ai_sidecar.contracts.actions import ActionProposal, ActionPriorityTier
        import hashlib as _hashlib
        import time as _time

        actions_queued = 0

        if should_buff:
            # Filter out buffs already active
            fresh_buffs = [
                name for name in eligible_buffs
                if not _is_buff_active(name, active_buffs)
            ]
            if fresh_buffs:
                b_idx = bot_rotation["buff_index"] % len(fresh_buffs)
                buff_cmd = fresh_buffs[b_idx]
                bot_rotation["buff_index"] = (bot_rotation["buff_index"] + 1) % len(fresh_buffs)
                bot_rotation["last_buff_ts"] = now
                bot_rotation["total_pick_count"] += 1

                _short_id = _hashlib.md5(f"{bot_id}_buff_{horizon}_{_time.time()}".encode()).hexdigest()[:16]
                proposal = ActionProposal(
                    action_id=f"skill_{horizon}_{_short_id}",
                    kind="command",
                    command=f"ss {buff_cmd}",
                    priority_tier=ActionPriorityTier.tactical,
                    source="planner",
                    created_at=datetime.now(UTC),
                    expires_at=datetime.now(UTC) + timedelta(seconds=60),
                    idempotency_key=f"skill_{horizon}_{_short_id}",
                    metadata={
                        "goal": "combat",
                        "objective": f"Buff with {buff_cmd}",
                        "horizon": horizon,
                        "bot_id": bot_id,
                        "source": "skill_ai",
                        "skill_type": "buff",
                        "skill_name": buff_cmd,
                        "rotation_index": bot_rotation["attack_index"],
                        "bot_level": bot_level,
                        "sp_ratio": sp_ratio,
                    },
                )
                aq.enqueue(bot_id, proposal)
                _log.info(
                    "skill_action[%s]: buff=%s level=%d sp_ratio=%.2f (rotation=%d)",
                    bot_id, buff_cmd, bot_level, sp_ratio,
                    bot_rotation["attack_index"],
                )
                actions_queued = 1

        # ── Emit attack skill ──
        if eligible_attacks:
            a_idx = bot_rotation["attack_index"] % len(eligible_attacks)
            attack_cmd = eligible_attacks[a_idx]
            bot_rotation["attack_index"] = (bot_rotation["attack_index"] + 1) % len(eligible_attacks)
            bot_rotation["total_pick_count"] += 1

            # If skills recommendation found a good element, prefer that
            recommended_element = "Neutral"
            damage_mult = 1.0
            if skills:
                recommended_element = skills[0].get("recommended_element", "Neutral")
                damage_mult = skills[0].get("damage_multiplier", 1.0)

            # Build objective string
            if attack_cmd == "basic_attack":
                command = f"attack_skill {attack_cmd}"
                obj = f"Basic attack vs {mob_element}"
            else:
                command = f"ss {attack_cmd}"
                obj = f"Use {attack_cmd} vs {mob_element}"

            _short_id = _hashlib.md5(f"{bot_id}_atk_{horizon}_{_time.time()}_{a_idx}".encode()).hexdigest()[:16]
            proposal = ActionProposal(
                action_id=f"skill_{horizon}_{_short_id}",
                kind="command",
                command=command,
                priority_tier=ActionPriorityTier.tactical,
                source="planner",
                created_at=datetime.now(UTC),
                expires_at=datetime.now(UTC) + timedelta(seconds=60),
                idempotency_key=f"skill_{horizon}_{_short_id}",
                metadata={
                    "goal": "combat",
                    "objective": obj,
                    "horizon": horizon,
                    "bot_id": bot_id,
                    "source": "skill_ai",
                    "skill_type": "attack",
                    "skill_name": attack_cmd,
                    "rotation_index": bot_rotation["attack_index"],
                    "bot_level": bot_level,
                    "bot_class": canonical_class,
                    "sp_ratio": sp_ratio,
                    "recommended_element": recommended_element,
                    "damage_multiplier": damage_mult,
                    "mob_element": mob_element,
                },
            )
            aq.enqueue(bot_id, proposal)
            _log.info(
                "skill_action[%s]: cmd=%s element=%s mult=%.0f%% level=%d sp_ratio=%.2f (rotation=%d)",
                bot_id, command, recommended_element, damage_mult * 100,
                bot_level, sp_ratio, bot_rotation["attack_index"],
            )
            actions_queued += 1

        return actions_queued

    except Exception:
        _log.exception("skill_action_emission_failed")
        return 0
_STARTUP_GATE_MIN_EVENTS = 2
_STARTUP_GATE_MAX_CREW_FAILURES = 2


class Horizon(Enum):
    SHORT_TERM = "short_term"      # 5s  — tactical movement, combat
    MEDIUM_TERM = "medium_term"    # 30s — zone clearing, quest step
    LONG_TERM = "long_term"        # 120s — map transition, gear upgrade


@dataclass
class PDCAResult:
    """Outcome of a single PDCA cycle iteration."""

    horizon: Horizon
    plan_id: str | None
    actions_queued: int
    progress_pct: float
    stuck: bool
    re_planned: bool
    cycle_ms: float
    force_replan: bool = False
    replan_reasons: list[str] = field(default_factory=list)
    objective: str = ""
    selected_goal: str = ""
    error: str | None = None


@dataclass
class PDCAConfig:
    """Configuration for the PDCA loop."""

    short_term_interval_s: float = 5.0
    medium_term_interval_s: float = 30.0
    long_term_interval_s: float = 120.0
    max_stuck_cycles: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_reset_s: float = 60.0
    plan_timeout_s: float = 30.0
    max_actions_per_cycle: int = 5


def _extract_command_from_goal(goal: str | None, objective: str | None = None) -> str:
    """Convert a PDCA goal key to a valid OpenKore command."""
    if not goal:
        return "ai auto"
    g = goal.lower()
    # Strip enum wrapper and quotes
    for ch in ".:'\"":
        g = g.replace(ch, " ")
    g = g.strip()
    # Extract last meaningful word
    parts = [p for p in g.split() if p and p not in ("goal", "key", "category", "goalcategory")]
    keyword = parts[-1] if parts else "auto"
    
    # Map to real OpenKore commands
    cmd_map = {
        "survival": "ai auto",
        "survival_pressure": "ai auto",
        "recovery": "sit",
        "grind": "ai auto",
        "economy": "ai auto",
        "job_advancement": "ai auto",
        "quest": "ai auto",
        "idle": "stand",
        "move": "move",
        "attack": "attack",
        "sit": "sit",
        "stand": "stand",
        "auto": "ai auto",
        "manual": "ai manual",
    }
    
    # Dynamic map routing: if objective mentions a hunting map, route there
    if objective and keyword in ("leveling", "grind", "hunt", "advancement", "survival"):
        for map_code in ("prt_fild", "pay_fild", "moc_fild", "gef_fild", "alde_fild", "mjolnir"):
            if map_code in objective.lower():
                import re as _re
                _maps = _re.findall(map_code.replace("_", ".") + "[0-9]+", objective.lower())
                if _maps:
                    return f"move {_maps[0]}"
        # If in Prontera with any common goal, route to nearby field
        if "prontera" in objective.lower() and keyword in ("survival", "job_advancement", "advancement", "idle", "economy"):
            return "move prt_fild08"
    
    # If goal is survival with no specific routing, still send ai auto
    if keyword == "survival":
        return "ai auto"
    
    return cmd_map.get(keyword, "ai auto")


class PDCALoop:
    """Continuous Plan-Do-Check-Act autonomy loop.

    Runs three nested horizons:
      SHORT_TERM  — every 5s,  tactical decisions (move, attack, loot)
      MEDIUM_TERM — every 30s, tactical bundles (clear zone, quest step)
      LONG_TERM   — every 120s, strategic plans (map change, gear upgrade)
    """

    def __init__(
        self,
        runtime_state: Any,  # RuntimeState from lifecycle
        config: PDCAConfig | None = None,
    ) -> None:
        self._runtime = runtime_state
        self._config = config or PDCAConfig()
        self._plan_executor = PlanExecutor(runtime_state)
        self._progress_tracker = ProgressTracker(runtime_state)
        self._circuit_breaker = ReflexCircuitBreaker()
        self._breaker_bot_id = "pdca"
        self._breaker_key = "queue.default"
        self._breaker_family = "queue"
        self._default_bot_id = "default"
        self._last_bot_id: str | None = None
        self._startup_gate_defaults = {
            "grace_s": max(20.0, self._policy_float("reconnect_grace_s", 20.0)),
            "min_events": _STARTUP_GATE_MIN_EVENTS,
            "max_crewai_failures": _STARTUP_GATE_MAX_CREW_FAILURES,
        }

        # Per-horizon state
        self._active_plan: dict[Horizon, StrategicPlan | TacticalIntentBundle | None] = {
            h: None for h in Horizon
        }
        self._last_plan_time: dict[Horizon, float] = {h: 0.0 for h in Horizon}
        self._stuck_counter: dict[Horizon, int] = {h: 0 for h in Horizon}
        self._objective_rotation_index: dict[Horizon, int] = {h: 0 for h in Horizon}
        self._last_objective_switch_at: dict[Horizon, float] = {h: 0.0 for h in Horizon}
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._cycle_count: int = 0

    # ── Public API ──────────────────────────────────────────────

    @property
    def running(self) -> bool:
        return self._running

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    def start(self) -> None:
        """Start the PDCA loop in a background thread with its own event loop.
        Prevents LLM API calls from blocking the uvicorn event loop."""
        if self._running:
            logger.warning("PDCALoop already running")
            return
        self._running = True
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_in_thread, daemon=True)
        self._thread.start()
        logger.info("PDCALoop started in background thread")

    def _run_in_thread(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run_loop())
        except Exception:
            logger.exception("PDCALoop thread crashed")
        finally:
            self._loop.close()
            self._running = False

    async def stop(self) -> None:
        """Stop the PDCA loop gracefully. Cleans up P2P nodes."""
        self._running = False
        if hasattr(self, '_thread') and self._thread is not None:
            self._thread.join(timeout=3)
        # Clean up P2P nodes
        _p2p = getattr(self._runtime, "p2p_node", None)
        if _p2p is not None:
            try:
                _p2p.stop_server()
            except Exception:
                pass
        _p2p_mgr = getattr(self._runtime, "p2p_manager", None)
        if _p2p_mgr is not None:
            try:
                _p2p_mgr.stop_all_servers()
            except Exception:
                pass
        logger.info("PDCALoop stopped")

    async def get_status(self) -> dict[str, Any]:
        """Return current loop status as a dict."""
        return {
            "running": self._running,
            "cycle_count": self._cycle_count,
            "circuit_breaker_tripped": self._circuit_breaker_tripped(),
            "horizons": {
                h.value: {
                    "has_active_plan": self._active_plan[h] is not None,
                    "stuck_cycles": self._stuck_counter[h],
                    "last_plan_seconds_ago": time.time() - self._last_plan_time[h]
                    if self._last_plan_time[h] > 0
                    else None,
                }
                for h in Horizon
            },
        }

    # ── Internal loop ───────────────────────────────────────────

    async def _run_loop(self) -> None:
        """Main async loop — runs until stopped."""
        logger.info("PDCALoop _run_loop entered")
        while self._running:
            try:
                now = time.time()
                for horizon in Horizon:
                    if self._circuit_breaker_tripped():
                        logger.warning("Circuit breaker tripped — skipping all horizons")
                        await asyncio.sleep(1.0)
                        continue

                    interval = self._interval_for(horizon)
                    if now - self._last_plan_time[horizon] >= interval:
                        result = await self._run_one_cycle(horizon)
                        self._cycle_count += 1
                        self._last_plan_time[horizon] = time.time()

                        if result.error:
                            self._circuit_breaker.record_failure(
                                bot_id=self._breaker_bot_id,
                                key=self._breaker_key,
                                family=self._breaker_family,
                                reason=result.error,
                            )
                            logger.error("PDCA cycle error [%s]: %s", horizon.value, result.error)
                        else:
                            self._circuit_breaker.record_success(
                                bot_id=self._breaker_bot_id,
                                key=self._breaker_key,
                                family=self._breaker_family,
                            )

                        # Log cycle result
                        logger.info(
                            "PDCA [%s] plan=%s actions=%d progress=%.1f%% stuck=%s replan=%s force=%s goal=%s objective=%s reasons=%s cycle_ms=%.1f",
                            horizon.value,
                            result.plan_id,
                            result.actions_queued,
                            result.progress_pct * 100,
                            result.stuck,
                            result.re_planned,
                            result.force_replan,
                            result.selected_goal,
                            result.objective,
                            ",".join(result.replan_reasons),
                            result.cycle_ms,
                        )

                # Sleep a short interval before re-checking horizons
                await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                logger.info("PDCALoop cancelled")
                break
            except Exception:
                logger.exception("PDCALoop unhandled error")
                await asyncio.sleep(5.0)

    async def _run_one_cycle(self, horizon: Horizon) -> PDCAResult:
        """Execute one PDCA cycle for the given horizon."""
        # Resolve current bot_id for cost gate
        _cycle_bot_id = self._resolve_cost_gate_bot_id()
        
        # ── Ensure all sub-services are initialized before any path ──
        if self._runtime is not None:
            try:
                from ai_sidecar.config import settings as _settings
                # Initialize hunting zone manager if not present
                _hzm = getattr(self._runtime, "hunting_zone_manager", None)
                if _hzm is None:
                    _hzm = HuntingZoneManager(
                        getattr(_settings, "game_engine_knowledge_path", "knowledge/knowledge.json")
                    )
                    self._runtime.hunting_zone_manager = _hzm
                # Initialize game engine if not present
                _ge = getattr(self._runtime, "game_engine", None)
                if _ge is None:
                    try:
                        _rathena_path = getattr(_settings, "rathena_path", None)
                        if not _rathena_path:
                            from pathlib import Path
                            _rathena_path = str(Path(__file__).parent.parent.parent.parent / "knowledge" / "rathena_db")
                        _ge = GameIntelligenceEngine(
                            getattr(_settings, "game_engine_knowledge_path", "knowledge/knowledge.json"),
                            rathena_path=_rathena_path,
                        )
                        self._runtime.game_engine = _ge
                        logger.info("game_engine_initialized: %d monsters, %d mob skills",
                                    len(_ge._monsters), len(_ge._mob_skills))
                    except Exception as e:
                        logger.warning("game_engine_init_failed: %s", e)
                # Initialize swarm tactics if not present
                _swarm = getattr(self._runtime, "swarm_tactics", None)
                if _swarm is None:
                    try:
                        _swarm = SwarmTacticsEngine()
                        self._runtime.swarm_tactics = _swarm
                    except Exception:
                        pass
                # Initialize anti-detection if not present
                _ad = getattr(self._runtime, "anti_detection", None)
                if _ad is None:
                    _ad = AntiDetection(enabled=getattr(_settings, "anti_detection_enabled", True))
                    self._runtime.anti_detection = _ad
                # Initialize role discovery if not present
                _role_disc = getattr(self._runtime, "role_discovery", None)
                if _role_disc is None:
                    try:
                        from ai_sidecar.fleet.swarm_ai import RoleDiscoveryEngine
                        _role_disc = RoleDiscoveryEngine()
                        self._runtime.role_discovery = _role_disc
                    except Exception:
                        pass
            except Exception:
                pass
        
        # ── Cost gate with 3 modes ──────────────────────────
        if self._runtime is not None:
            _ct = getattr(self._runtime, "cost_tracker", None)
            _settings_available = True
            try:
                from ai_sidecar.config import settings as _settings
                _tier = getattr(_settings, "llm_cost_tier", "standard")
                _cost_mode_str = getattr(_settings, "cost_mode", "standard")
                
                # Initialize cost mode manager if not present
                _cost_mode = getattr(self._runtime, "cost_mode_manager", None)
                if _cost_mode is None:
                    _cost_mode = CostModeManager(_cost_mode_str)
                    self._runtime.cost_mode_manager = _cost_mode
                    try:
                        _role_disc = RoleDiscoveryEngine()
                        self._runtime.role_discovery = _role_disc
                    except Exception:
                        pass
                
                # Initialize goal decomposer if not present
                _gd = getattr(self._runtime, "goal_decomposer", None)
                if _gd is None:
                    try:
                        _gd = GoalDecomposer()
                        self._runtime.goal_decomposer = _gd
                    except Exception:
                        pass
                
                # Initialize NPC discovery if not present
                _nd = getattr(self._runtime, "npc_discovery", None)
                if _nd is None:
                    try:
                        _nd = NPCDiscoveryEngine()
                        self._runtime.npc_discovery = _nd
                    except Exception:
                        pass
                
                # Initialize server adaptation if not present
                _sa = getattr(self._runtime, "server_adaptation", None)
                if _sa is None:
                    try:
                        _sa = ServerAdaptationEngine(
                            getattr(_settings, "game_engine_knowledge_path", "knowledge/knowledge.json")
                        )
                        self._runtime.server_adaptation = _sa
                    except Exception:
                        pass
                
                # Initialize P2P knowledge node if not present
                _p2p = getattr(self._runtime, "p2p_node", None)
                if _p2p is None:
                    try:
                        _p2p = P2PKnowledgeNode(
                            bot_id=_cycle_bot_id,
                            listen_port=18090 + abs(hash(_cycle_bot_id)) % 1000,
                            server_id=_cycle_bot_id.split(":")[0] if ":" in _cycle_bot_id else "default",
                        )
                        # Wire to experience DB
                        _exp_db = getattr(self._runtime, "experience_db", None)
                        if _exp_db is not None:
                            _p2p.set_experience_db(_exp_db)
                        _p2p_npc = getattr(self._runtime, "npc_discovery", None)
                        if _p2p_npc is not None:
                            _p2p.set_npc_discovery(_p2p_npc)
                        _p2p_sa = getattr(self._runtime, "server_adaptation", None)
                        if _p2p_sa is not None:
                            _p2p.set_server_adaptation(_p2p_sa)
                        self._runtime.p2p_node = _p2p
                        # Start P2P HTTP server to receive messages from peers
                        _p2p_started = _p2p.start_server()
                        if _p2p_started:
                            logger.info("p2p_node_ready: bot=%s port=%d server_id=%s",
                                       _cycle_bot_id, _p2p._listen_port, _p2p._server_id)
                        # Register with P2P network manager
                        _p2p_mgr = getattr(self._runtime, "p2p_manager", None)
                        if _p2p_mgr is None:
                            _p2p_mgr = P2PNetworkManager()
                            self._runtime.p2p_manager = _p2p_mgr
                        _p2p_mgr.register_node(_cycle_bot_id, _p2p)
                        # Connect to all known peers
                        _p2p_mgr.connect_all()
                    except Exception:
                        pass
                
                # Initialize swarm reflex if not present
                _sr = getattr(self._runtime, "swarm_reflex", None)
                if _sr is None:
                    try:
                        from ai_sidecar.fleet.swarm_ai import SwarmReflexSystem
                        _sr = SwarmReflexSystem()
                        self._runtime.swarm_reflex = _sr
                    except Exception:
                        pass
                
                # Initialize synergy engine if not present
                _se = getattr(self._runtime, "synergy_engine", None)
                if _se is None:
                    try:
                        _se = CrossHorizonSynergy()
                        self._runtime.synergy_engine = _se
                    except Exception:
                        pass
                
                # Initialize swarm goal coordinator if not present
                _sgc = getattr(self._runtime, "swarm_coordinator", None)
                if _sgc is None:
                    try:
                        from ai_sidecar.autonomy.goal_decomposer import SwarmGoalCoordinator
                        _sgc = SwarmGoalCoordinator()
                        self._runtime.swarm_coordinator = _sgc
                        # Wire into HZM for multi-bot zone coordination
                        _hzm_existing = getattr(self._runtime, "hunting_zone_manager", None)
                        if _hzm_existing is not None:
                            _hzm_existing.set_coordinator(_sgc)
                        # Wire into goal decomposer
                        _gd_existing = getattr(self._runtime, "goal_decomposer", None)
                        if _gd_existing is not None:
                            _gd_existing.set_swarm_coordinator(_sgc)
                    except Exception:
                        pass
                
                # ── NEW: Initialize Role Manager ──
                _role_mgr = getattr(self._runtime, "role_manager", None)
                if _role_mgr is None:
                    try:
                        from ai_sidecar.fleet.role_manager import RoleManager
                        _role_mgr = RoleManager()
                        self._runtime.role_manager = _role_mgr
                        logger.info("role_manager_initialized")
                    except Exception as e:
                        logger.warning("role_manager_init_failed: %s", e)
                
                # ── NEW: Initialize Experience DB ──
                _exp_db = getattr(self._runtime, "experience_db", None)
                if _exp_db is None:
                    try:
                        from ai_sidecar.experience_db import ExperienceDB
                        _db_path = getattr(_settings, "experience_db_path", "data/experience.db")
                        _exp_db = ExperienceDB(db_path=_db_path)
                        self._runtime.experience_db = _exp_db
                        logger.info("experience_db_initialized: path=%s", _db_path)
                    except Exception as e:
                        logger.warning("experience_db_init_failed: %s", e)
                
                # ── NEW: Initialize Fleet Learning System ──
                _fleet_learn = getattr(self._runtime, "fleet_learning", None)
                if _fleet_learn is None:
                    try:
                        from ai_sidecar.fleet.self_learning import FleetLearningSystem
                        _fleet_learn = FleetLearningSystem()
                        self._runtime.fleet_learning = _fleet_learn
                        logger.info("fleet_learning_initialized")
                    except Exception as e:
                        logger.warning("fleet_learning_init_failed: %s", e)
                
                # ── NEW: Initialize Goal Stack ──
                _goal_stack = getattr(self._runtime, "goal_stack", None)
                if _goal_stack is None:
                    try:
                        from ai_sidecar.autonomy.goal_stack import GoalStackComputation
                        _goal_stack = GoalStackComputation()
                        self._runtime.goal_stack = _goal_stack
                        logger.info("goal_stack_initialized")
                    except Exception as e:
                        logger.warning("goal_stack_init_failed: %s", e)
                
                # ── NEW: Initialize Memory Retrieval ──
                _mem = getattr(self._runtime, "memory_retrieval", None)
                if _mem is None:
                    try:
                        from ai_sidecar.memory.retrieval import MemoryRetrievalService, InMemoryMemoryProvider
                        _mem = MemoryRetrievalService(
                            provider=InMemoryMemoryProvider(max_entries=5000)
                        )
                        self._runtime.memory_retrieval = _mem
                        logger.info("memory_retrieval_initialized: provider=InMemoryMemoryProvider")
                    except Exception as e:
                        logger.debug("memory_init_skipped: %s", e)
                
                # ── NEW: Initialize Reflex Rule Engine ──
                _reflex_engine = getattr(self._runtime, "reflex_engine", None)
                if _reflex_engine is None:
                    try:
                        from ai_sidecar.reflex.rule_engine import ReflexRuleEngine
                        from pathlib import Path
                        _reflex_engine = ReflexRuleEngine(
                            workspace_root=Path("."),
                            contract_version="v1",
                            action_ttl_seconds=60,
                        )
                        self._runtime.reflex_engine = _reflex_engine
                        # Load reflex rules from YAML — use __file__-relative path for portability
                        _here = Path(__file__).resolve().parent.parent
                        _yaml_path = _here / "reflex" / "reflex_rules.yaml"
                        _rules_loaded = _reflex_engine.load_rules_from_yaml(_yaml_path)
                        logger.info("reflex_engine_initialized: %d rules loaded", _rules_loaded)
                    except Exception as e:
                        logger.warning("reflex_engine_init_failed: %s", e)
                
                # ── NEW: Initialize Reflex Action Emitter ──
                _reflex_emitter = getattr(self._runtime, "reflex_emitter", None)
                if _reflex_emitter is None:
                    try:
                        from ai_sidecar.reflex.action_emitter import ActionEmitter
                        from pathlib import Path
                        _reflex_emitter = ActionEmitter(
                            workspace_root=Path("."),
                            contract_version="v1",
                            action_ttl_seconds=60,
                        )
                        self._runtime.reflex_emitter = _reflex_emitter
                        logger.info("reflex_emitter_initialized")
                    except Exception as e:
                        logger.warning("reflex_emitter_init_failed: %s", e)
                
                # ── NEW: Initialize NPC Dialog ──
                _npc_dialog = getattr(self._runtime, "npc_dialog", None)
                if _npc_dialog is None:
                    try:
                        from ai_sidecar.npc_dialog import NPCDialogEngine
                        # Wire LLM adapter for NPC response decisions
                        _llm_adapter = None
                        _mr = getattr(self._runtime, "model_router", None)
                        if _mr is not None and hasattr(_mr, 'generate_text'):
                            _llm_adapter = _mr.generate_text
                        _npc_dialog = NPCDialogEngine(llm_adapter=_llm_adapter)
                        self._runtime.npc_dialog = _npc_dialog
                        logger.info("npc_dialog_initialized: llm=%s", "wired" if _llm_adapter else "none")
                    except Exception as e:
                        logger.warning("npc_dialog_init_failed: %s", e)
                
                # ── NEW: Initialize Macro Intelligence ──
                _macro_ai = getattr(self._runtime, "macro_intelligence", None)
                if _macro_ai is None:
                    try:
                        from ai_sidecar.autonomy.macro_intelligence import MacroIntelligence
                        from pathlib import Path
                        _knowledge_path = str(Path(__file__).parent.parent.parent.parent / "knowledge" / "knowledge.json")
                        _macro_ai = MacroIntelligence(knowledge_path=_knowledge_path)
                        self._runtime.macro_intelligence = _macro_ai
                        _pattern_count = len(_macro_ai.get_all_patterns())
                        logger.info("macro_intelligence_initialized: %d patterns loaded", _pattern_count)
                    except Exception as e:
                        logger.warning("macro_intelligence_init_failed: %s", e)
                
                # ── NEW: Initialize Combat Optimizer ──
                _combat_opt = getattr(self._runtime, "combat_optimizer", None)
                if _combat_opt is None:
                    try:
                        from ai_sidecar.crewai.agents.combat_agent import CombatOptimizer
                        from pathlib import Path
                        _knowledge_path = Path(__file__).parent.parent.parent.parent / "knowledge" / "knowledge.json"
                        _combat_opt = CombatOptimizer(knowledge_path=_knowledge_path)
                        self._runtime.combat_optimizer = _combat_opt
                        logger.info("combat_optimizer_initialized")
                    except Exception as e:
                        logger.warning("combat_opt_init_failed: %s", e)
                
                # ── NEW: Initialize Quest Automation ──
                _quest_auto = getattr(self._runtime, "quest_automation", None)
                if _quest_auto is None:
                    try:
                        from ai_sidecar.autonomy.quest_automation import QuestAutomation
                        _quest_auto = QuestAutomation()
                        self._runtime.quest_automation = _quest_auto
                        logger.info("quest_automation_initialized")
                    except Exception as e:
                        logger.warning("quest_auto_init_failed: %s", e)
                
                # ── NEW: Initialize High-Frequency Reflex ──
                _hf_reflex = getattr(self._runtime, "highfreq_reflex", None)
                if _hf_reflex is None:
                    try:
                        from ai_sidecar.reflex.highfreq_reflex import HighFreqReflex
                        _hf_reflex = HighFreqReflex()
                        self._runtime.highfreq_reflex = _hf_reflex
                        logger.info("highfreq_reflex_initialized")
                    except Exception as e:
                        logger.warning("highfreq_reflex_init_failed: %s", e)
                
                # ── NEW: Initialize Map Knowledge ──
                _map_know = getattr(self._runtime, "map_knowledge", None)
                if _map_know is None:
                    try:
                        from ai_sidecar.map_knowledge import MapKnowledge
                        from pathlib import Path
                        _map_know = MapKnowledge(
                            knowledge_path=Path(__file__).parent.parent.parent.parent / "knowledge" / "knowledge.json"
                        )
                        self._runtime.map_knowledge = _map_know
                        logger.info("map_knowledge_initialized: %d maps", _map_know.counters()["maps"])
                    except Exception as e:
                        logger.warning("map_knowledge_init_failed: %s", e)
                
                # ── NEW: Initialize Movement Optimizer ──
                _mov_opt = getattr(self._runtime, "movement_optimizer", None)
                if _mov_opt is None:
                    try:
                        from ai_sidecar.movement_optimizer import MovementOptimizer
                        _mov_opt = MovementOptimizer()
                        self._runtime.movement_optimizer = _mov_opt
                        logger.info("movement_optimizer_initialized")
                    except Exception as e:
                        logger.warning("movement_opt_init_failed: %s", e)
                
                # ── NEW: Initialize Combat Tactics ──
                _combat_tac = getattr(self._runtime, "combat_tactics", None)
                if _combat_tac is None:
                    try:
                        from ai_sidecar.combat_tactics import CombatTactics
                        _combat_tac = CombatTactics()
                        self._runtime.combat_tactics = _combat_tac
                        logger.info("combat_tactics_initialized")
                    except Exception as e:
                        logger.warning("combat_tactics_init_failed: %s", e)
                
                # ── NEW: Initialize Party Skill Coordinator ──
                _party_skill = getattr(self._runtime, "party_skill_coordinator", None)
                if _party_skill is None:
                    try:
                        from ai_sidecar.party_skill_coordinator import PartySkillCoordinator
                        _party_skill = PartySkillCoordinator()
                        self._runtime.party_skill_coordinator = _party_skill
                        logger.info("party_skill_coordinator_initialized")
                    except Exception as e:
                        logger.warning("party_skill_init_failed: %s", e)
                
                # ── NEW: Initialize Economy Optimizer ──
                _econ_opt = getattr(self._runtime, "economy_optimizer", None)
                if _econ_opt is None:
                    try:
                        from ai_sidecar.economy_optimizer import EconomyOptimizer
                        _econ_opt = EconomyOptimizer()
                        self._runtime.economy_optimizer = _econ_opt
                        logger.info("economy_optimizer_initialized")
                    except Exception as e:
                        logger.warning("economy_opt_init_failed: %s", e)
                
                # ── NEW: Initialize Gear Progression ──
                _gear_prog = getattr(self._runtime, "gear_progression", None)
                if _gear_prog is None:
                    try:
                        from ai_sidecar.gear_progression import GearProgression
                        _gear_prog = GearProgression()
                        self._runtime.gear_progression = _gear_prog
                        logger.info("gear_progression_initialized")
                    except Exception as e:
                        logger.warning("gear_progression_init_failed: %s", e)
                
                # ── NEW: Initialize MVP Tactics ──
                _mvp_tac = getattr(self._runtime, "mvp_tactics", None)
                if _mvp_tac is None:
                    try:
                        from ai_sidecar.crewai.agents.combat_agent import CombatOptimizer
                        _mvp_tac = CombatOptimizer(knowledge_path=None)
                        self._runtime.mvp_tactics = _mvp_tac
                        logger.info("mvp_tactics_initialized")
                    except Exception as e:
                        logger.warning("mvp_tactics_init_failed: %s", e)
                
                # ── NEW: Initialize Server Awareness ──
                _srv_aware = getattr(self._runtime, "server_awareness", None)
                if _srv_aware is None:
                    try:
                        from ai_sidecar.server_awareness import ServerAwareness
                        _srv_aware = ServerAwareness()
                        self._runtime.server_awareness = _srv_aware
                        logger.info("server_awareness_initialized")
                    except Exception as e:
                        logger.warning("server_awareness_init_failed: %s", e)
                
                # ── NEW: Initialize Social Intelligence ──
                _soc_intel = getattr(self._runtime, "social_intelligence", None)
                if _soc_intel is None:
                    try:
                        from ai_sidecar.social_intelligence import SocialIntelligenceV2 as SocialIntelligence
                        _soc_intel = SocialIntelligence()
                        self._runtime.social_intelligence = _soc_intel
                        logger.info("social_intelligence_initialized")
                    except Exception as e:
                        logger.warning("social_intelligence_init_failed: %s", e)
                
                # ── NEW: Initialize Predictive Planner ──
                _pred_plan = getattr(self._runtime, "predictive_planner", None)
                if _pred_plan is None:
                    try:
                        from ai_sidecar.predictive_planner import PredictivePlanner
                        _pred_plan = PredictivePlanner()
                        self._runtime.predictive_planner = _pred_plan
                        logger.info("predictive_planner_initialized")
                    except Exception as e:
                        logger.warning("predictive_planner_init_failed: %s", e)
                
                # ── NEW: Initialize Build Manager ──
                _build_mgr = getattr(self._runtime, "build_manager", None)
                if _build_mgr is None:
                    try:
                        from ai_sidecar.build_manager import BuildManager
                        _build_mgr = BuildManager()
                        self._runtime.build_manager = _build_mgr
                        logger.info("build_manager_initialized")
                    except Exception as e:
                        logger.warning("build_manager_init_failed: %s", e)
                
                # ── NEW: Initialize Risk Assessment ──
                _risk = getattr(self._runtime, "risk_assessment", None)
                if _risk is None:
                    try:
                        from ai_sidecar.risk_assessment import RiskAssessment
                        _risk = RiskAssessment()
                        self._runtime.risk_assessment = _risk
                        logger.info("risk_assessment_initialized")
                    except Exception as e:
                        logger.warning("risk_assessment_init_failed: %s", e)
                
                # ── NEW: Initialize Server Profiler ──
                _srv_prof = getattr(self._runtime, "server_profiler", None)
                if _srv_prof is None:
                    try:
                        from ai_sidecar.server_profiler import ServerProfiler
                        _srv_prof = ServerProfiler()
                        self._runtime.server_profiler = _srv_prof
                        logger.info("server_profiler_initialized")
                    except Exception as e:
                        logger.warning("server_profiler_init_failed: %s", e)
                
                # ── NEW: Initialize Knowledge Graph ──
                _kg = getattr(self._runtime, "knowledge_graph", None)
                if _kg is None:
                    try:
                        from ai_sidecar.knowledge_graph import KnowledgeGraph
                        from pathlib import Path
                        _kg = KnowledgeGraph(
                            knowledge_path=Path(__file__).parent.parent.parent.parent / "knowledge" / "knowledge.json"
                        )
                        self._runtime.knowledge_graph = _kg
                        logger.info("knowledge_graph_initialized: %s", _kg.counters())
                    except Exception as e:
                        logger.warning("knowledge_graph_init_failed: %s", e)
                
                # ── NEW: Initialize Combat Instinct ──
                _ci = getattr(self._runtime, "combat_instinct", None)
                if _ci is None:
                    try:
                        from ai_sidecar.combat_instinct import CombatInstinctEngine
                        _ci = CombatInstinctEngine()
                        self._runtime.combat_instinct = _ci
                        logger.info("combat_instinct_initialized")
                    except Exception as e:
                        logger.warning("combat_instinct_init_failed: %s", e)
                
                # ── NEW: Initialize Party Intelligence ──
                _pi = getattr(self._runtime, "party_intelligence", None)
                if _pi is None:
                    try:
                        from ai_sidecar.party_intelligence import PartyIntelligence
                        _pi = PartyIntelligence()
                        self._runtime.party_intelligence = _pi
                        logger.info("party_intelligence_initialized")
                    except Exception as e:
                        logger.warning("party_intelligence_init_failed: %s", e)
                
                # ── NEW: Initialize Market Intelligence ──
                _mi = getattr(self._runtime, "market_intelligence", None)
                if _mi is None:
                    try:
                        from ai_sidecar.market_intelligence import MarketIntelligence
                        _mi = MarketIntelligence()
                        self._runtime.market_intelligence = _mi
                        logger.info("market_intelligence_initialized")
                    except Exception as e:
                        logger.warning("market_intelligence_init_failed: %s", e)
                
                # ── NEW: Initialize WoE Intelligence ──
                _wi = getattr(self._runtime, "woe_intelligence", None)
                if _wi is None:
                    try:
                        from ai_sidecar.woe_intelligence import WoEIntelligence
                        _wi = WoEIntelligence()
                        self._runtime.woe_intelligence = _wi
                        logger.info("woe_intelligence_initialized")
                    except Exception as e:
                        logger.warning("woe_intelligence_init_failed: %s", e)
                
                # ── NEW: Initialize Navigation Intuition ──
                _ni = getattr(self._runtime, "navigation_intuition", None)
                if _ni is None:
                    try:
                        from ai_sidecar.navigation_intuition import NavigationIntuition
                        _ni = NavigationIntuition()
                        self._runtime.navigation_intuition = _ni
                        logger.info("navigation_intuition_initialized")
                    except Exception as e:
                        logger.warning("navigation_intuition_init_failed: %s", e)
                
                # ── NEW: Initialize Mechanical Intuition ──
                _mech = getattr(self._runtime, "mechanical_intuition", None)
                if _mech is None:
                    try:
                        from ai_sidecar.mechanical_intuition import MechanicalIntuition
                        _mech = MechanicalIntuition()
                        self._runtime.mechanical_intuition = _mech
                        logger.info("mechanical_intuition_initialized")
                    except Exception as e:
                        logger.warning("mechanical_intuition_init_failed: %s", e)
                
                # ── NEW: Initialize Opportunity Cost Optimizer ──
                _oc = getattr(self._runtime, "opportunity_cost", None)
                if _oc is None:
                    try:
                        from ai_sidecar.opportunity_cost import OpportunityCostOptimizer
                        _oc = OpportunityCostOptimizer()
                        self._runtime.opportunity_cost = _oc
                        logger.info("opportunity_cost_initialized")
                    except Exception as e:
                        logger.warning("opportunity_cost_init_failed: %s", e)
                
                # ── NEW: Initialize Meta Prediction ──
                _mp = getattr(self._runtime, "meta_prediction", None)
                if _mp is None:
                    try:
                        from ai_sidecar.meta_prediction import MetaPrediction
                        _mp = MetaPrediction()
                        self._runtime.meta_prediction = _mp
                        logger.info("meta_prediction_initialized")
                    except Exception as e:
                        logger.warning("meta_prediction_init_failed: %s", e)
                
                # ── NEW: Initialize Reflex Pipeline ──
                _rp = getattr(self._runtime, "reflex_pipeline", None)
                if _rp is None:
                    try:
                        from ai_sidecar.reflex.reflex_pipeline import ReflexPipeline
                        _rp = ReflexPipeline()
                        self._runtime.reflex_pipeline = _rp
                        logger.info("reflex_pipeline_initialized")
                    except Exception as e:
                        logger.warning("reflex_pipeline_init_failed: %s", e)
                
                # Wire action queue into reflex pipeline for direct emissions
                _rp = getattr(self._runtime, "reflex_pipeline", None)
                if _rp is not None:
                    _aq = getattr(self._runtime, "action_queue", None)
                    if _aq is not None:
                        _rp.set_action_queue(_aq)
                
                # Get heuristic confidence
                _hc = 0.0
                _hs = getattr(self._runtime, "heuristic_service", None)
                if _hs is not None:
                    _signals = {}
                    try:
                        _snap = getattr(self._runtime, "snapshot_cache", None)
                        if _snap:
                            _latest = _snap.latest()
                            if _latest and isinstance(_latest, dict):
                                v = _latest.get("vitals") or {}
                                _signals["hp_ratio"] = float(v.get("hp_ratio", 1.0))
                                _signals["sp_ratio"] = float(v.get("sp_ratio", 1.0))
                                c = _latest.get("combat") or {}
                                _signals["combat.aggro_count"] = int(c.get("aggro_count", 0))
                                _signals["map_known"] = bool(_latest.get("map_known", False))
                                inv = _latest.get("inventory") or {}
                                _signals["weight_ratio"] = float(inv.get("weight_ratio", 0.0))
                                _signals["horizon"] = horizon.value
                    except Exception:
                        pass
                    _hc = getattr(_hs, "confidence_for", lambda h, *a, **kw: 0.0)(horizon.value, signals=_signals, bot_id=_cycle_bot_id)
                
                # Read map name for routing
                _map_name = ""
                try:
                    _snap = getattr(self._runtime, "snapshot_cache", None)
                    if _snap:
                        _latest = _snap.latest()
                        if _latest:
                            if isinstance(_latest, dict):
                                _map_name = str(_latest.get("map", _latest.get("position", {}).get("map", "")))
                            else:
                                _map_name = str(getattr(getattr(_latest, "position", None), "map", ""))
                except Exception:
                    pass
                
                # Cost mode decision: should we use LLM?
                # Replaced by conscious trigger evaluation — LLM fires on demand
                _use_llm = False
                _trigger_reason = "conservative:no_triggers"
                _trigger_ctx: dict[str, object] = {}
                
                try:
                    # Read snapshot for trigger evaluation — PER BOT, not global
                    _conscious_snap = None
                    _conscious_deaths = 0
                    _snap_cons = getattr(self._runtime, "snapshot_cache", None)
                    if _snap_cons is not None:
                        try:
                            _conscious_snap = _snap_cons.get(_cycle_bot_id)
                        except Exception:
                            pass
                    if _conscious_snap is not None:
                        if isinstance(_conscious_snap, dict):
                            _current_hp = int(_conscious_snap.get("vitals", {}).get("hp", 1) or 1)
                            _prev_hp = getattr(self, "_prev_hp", {})
                            _bot_prev_hp = _prev_hp.get(_cycle_bot_id, _current_hp)
                            if _bot_prev_hp > 0 and _current_hp == 0:
                                _conscious_deaths = 1
                            _prev_hp[_cycle_bot_id] = _current_hp
                            object.__setattr__(self, "_prev_hp", _prev_hp)
                            # ── HIGH-FREQUENCY REFLEX CHECK ──
                            _hf_reflex = getattr(self._runtime, "highfreq_reflex", None)
                            if _hf_reflex is not None:
                                _vitals = _conscious_snap.get("vitals", {})
                                _hp = int(_vitals.get("hp", 1) or 1)
                                _max_hp = int(_vitals.get("max_hp", 1) or 1)
                                _sp = int(_vitals.get("sp", 0) or 0)
                                _max_sp = int(_vitals.get("max_sp", 1) or 1)
                                _aggro = int(_conscious_snap.get("combat", {}).get("aggro_count", 0))
                                _is_dead = _hp <= 0
                                _map = str(_conscious_snap.get("map", "") or "")
                                _is_town = any(t in _map.lower() for t in ["prontera", "morocc", "payon", "geffen", "aldebaran", "yuno"]) \
                                          and not any(f in _map.lower() for f in ["fild", "dun", "cave", "forest", "field"])
                                _inv = _conscious_snap.get("inventory", {}) or {}
                                _items = _inv.get("items", []) or _inv.get("item_list", []) or []
                                _has_pots = any("red_pot" in str(i.get("name", "")).lower() for i in _items if isinstance(i, dict))
                                _reflex_action = _hf_reflex.check_and_act(
                                    _cycle_bot_id, _hp, _max_hp, _sp, _max_sp,
                                    _aggro, _is_dead, _is_town, _has_pots, _map,
                                    reflex_pipeline=getattr(self._runtime, "reflex_pipeline", None),
                                )
                                if _reflex_action:
                                    logger.info("highfreq_reflex_action: bot=%s cmd=%s", _cycle_bot_id, _reflex_action)
                                    # Inject directly into action queue via reflex pipeline
                                    _rp = getattr(self._runtime, "reflex_pipeline", None)
                                    if _rp is not None:
                                        from ai_sidecar.contracts.reflex import ReflexRule, ReflexActionTemplate, ReflexTriggerClause, ReflexCategory, ReflexPlannerInterop
                                        from ai_sidecar.contracts.actions import ActionPriorityTier
                                        _rule = ReflexRule(
                                            rule_id="highfreq_reflex",
                                            priority=80,
                                            trigger=ReflexTriggerClause(all=[]),
                                            action_template=ReflexActionTemplate(
                                                command=_reflex_action,
                                                kind="command",
                                                conflict_key="",
                                                priority_tier=ActionPriorityTier.reflex,
                                            ),
                                            category=ReflexCategory.survival,
                                            planner_interop=ReflexPlannerInterop.override,
                                        )
                                        _aq = getattr(self._runtime, "action_queue", None)
                                        if _aq is not None:
                                            _rp.emit(_cycle_bot_id, _rule, _reflex_action, _aq.enqueue)
                    
                    _should_wake, _trigger_reason, _trigger_ctx = self._evaluate_conscious_triggers(
                        horizon=horizon,
                        bot_id=_cycle_bot_id,
                        snapshot=_conscious_snap,
                        recent_deaths=_conscious_deaths,
                    )
                    _use_llm = _should_wake
                    
                    # Two-tier conscious brain:
                    # - Tactical tier: every 30s, lighter evaluation, faster model
                    # - Strategic tier: every 2.5min, full evaluation, full model
                    # Both tiers feed into the same LLM but with different context depth
                    _tactical_trigger = False
                    if horizon == Horizon.SHORT_TERM:
                        _tactical_cycle = getattr(self, "_tactical_cycle", 0) + 1
                        object.__setattr__(self, "_tactical_cycle", _tactical_cycle)
                        if _tactical_cycle % 6 == 0:  # Every 30s (6 × 5s cycles)
                            _tactical_trigger = True
                            # Check for tactical-level triggers (HP trend, combat effectiveness)
                            try:
                                if _conscious_snap and isinstance(_conscious_snap, dict):
                                    _vitals = _conscious_snap.get("vitals", {})
                                    _hp_trend = _vitals.get("hp_trend", 0)
                                    if _hp_trend < -0.1:  # HP dropping fast
                                        _tactical_trigger = True
                                        _trigger_reason = "tactical:hp_dropping"
                            except Exception:
                                pass
                    
                    if _tactical_trigger and not _use_llm:
                        _use_llm = True
                        logger.info("conscious_tactical_trigger: bot=%s", _cycle_bot_id)
                    
                    if _use_llm:
                        logger.info(
                            "conscious_trigger: bot=%s reason=%s ctx=%s",
                            _cycle_bot_id, _trigger_reason, _trigger_ctx,
                        )
                        # Store trigger context for LLM path to consume — PER BOT
                        _trigger_store = getattr(self, "_last_trigger_context", {})
                        if not isinstance(_trigger_store, dict):
                            _trigger_store = {}
                        _trigger_store[_cycle_bot_id] = {
                            "reason": _trigger_reason,
                            "context": _trigger_ctx,
                            "timestamp": time.time(),
                        }
                        object.__setattr__(self, "_last_trigger_context", _trigger_store)
                except Exception:
                    logger.exception("conscious_trigger_eval_failed")
                
                # Update budget limits from cost mode (used as cap, not gate)
                if _ct is not None:
                    _daily_budget = _cost_mode.get_daily_budget_tokens()
                    _hourly_limit = _cost_mode.get_llm_calls_per_hour_limit()
                    _allowed, _reason = _ct.check(
                        daily_budget_tokens=_daily_budget,
                        max_calls_per_hour=_hourly_limit,
                        tier=_tier, bot_id=_cycle_bot_id,
                    )
                    if not _allowed and _use_llm:
                        logger.info("conscious_budget_exceeded: %s — skipping LLM, using fallback", _reason)
                        _use_llm = False
                if not _use_llm:
                    # Emit game engine + heuristic + swarm + vendor + skill actions
                    # Emit for ALL registered bots, not just the resolved one
                    _all_bot_ids: list[str] = []
                    try:
                        _br = getattr(self._runtime, "bot_registry", None)
                        if _br is not None:
                            _all_bot_ids = [str(b.bot_id) for b in _br.list() if hasattr(b, "bot_id") and b.bot_id]
                    except Exception:
                        pass
                    if not _all_bot_ids:
                        _all_bot_ids = [_cycle_bot_id]
                    _total_actions = 0
                    for _bid in _all_bot_ids:
                        _actions_queued_ge = _emit_game_engine_actions(
                            self._runtime, horizon.value, bot_id=_bid, map_name=_map_name
                        )
                        _actions_queued_hs = 0
                        try:
                            if _hs is not None:
                                _actions_queued_hs = _emit_heuristic_actions(self._runtime, horizon.value, bot_id=_bid)
                        except Exception:
                            pass
                        _actions_queued_swarm = _emit_swarm_actions(
                            self._runtime, horizon.value, bot_id=_bid
                        )
                        _actions_queued_vendor = _emit_vendor_actions(
                            self._runtime, horizon.value, bot_id=_bid
                        )
                        _actions_queued_skill = _emit_skill_actions(
                            self._runtime, horizon.value, bot_id=_bid
                        )
                        _total_actions += _actions_queued_ge + _actions_queued_hs + _actions_queued_swarm + _actions_queued_vendor + _actions_queued_skill
                    logger.info(
                        "cost_gate[%s]: mode=%s use_llm=False heuristic=%.2f total=%d bots=%d",
                        horizon.value, _cost_mode.mode.value, _hc,
                        _total_actions, len(_all_bot_ids),
                    )
                    return PDCAResult(horizon=horizon, plan_id="", actions_queued=_total_actions, progress_pct=0.0, stuck=False, re_planned=False,
                                      force_replan=False, selected_goal="cost_gated", objective=f"Cost mode {_cost_mode.mode.value}",
                                      replan_reasons=[], cycle_ms=0.0, error="")
                
                # Check daily/hourly budget (for LLM path)
                if _ct is not None:
                    _allowed, _reason = _ct.check(
                        daily_budget_tokens=getattr(_settings, "llm_daily_budget_tokens", 100000),
                        max_calls_per_hour=getattr(_settings, "llm_max_calls_per_hour", 30),
                        tier=_tier, bot_id=_cycle_bot_id,
                    )
                    if not _allowed:
                        logger.info("cost_gate[%s]: %s", horizon.value, _reason)
                        _actions_queued_budget = _emit_game_engine_actions(
                            self._runtime, horizon.value, bot_id=_cycle_bot_id, map_name=_map_name
                        )
                        return PDCAResult(horizon=horizon, plan_id="", actions_queued=_actions_queued_budget, progress_pct=0.0, stuck=False, re_planned=False,
                                          force_replan=False, selected_goal="budget_gated", objective=f"Budget exceeded: {_reason}",
                                          replan_reasons=[_reason], cycle_ms=0.0, error="")
            except Exception:
                logger.exception("cost_gate_check_failed")
        
        # ── FALLBACK: always emit game engine + swarm + vendor + skill actions into queue
        # This runs in ALL modes before the LLM path, so bots always have actions.
        # If the LLM path succeeds, these are just extra queued actions.
        # If the LLM path fails, the bot still has meaningful actions.
        # Emit for ALL registered bots, not just the resolved one
        _all_bot_ids: list[str] = []
        try:
            _br = getattr(self._runtime, "bot_registry", None)
            if _br is not None:
                _all_bot_ids = [str(b.bot_id) for b in _br.list() if hasattr(b, "bot_id") and b.bot_id]
        except Exception:
            pass
        if not _all_bot_ids:
            _all_bot_ids = [_cycle_bot_id]
        # Read map_name from snapshot for game engine routing — per bot_id
        _fb_map_name = ""
        _fallback_total = 0
        for _bid in _all_bot_ids:
            _fb_map_name = ""
            _bot_class = "novice"
            _bot_level = 1
            try:
                _fb_cache = getattr(self._runtime, "snapshot_cache", None)
                if _fb_cache is not None:
                    _fb_snap = _fb_cache.get(_bid)
                    if _fb_snap is not None:
                        if isinstance(_fb_snap, dict):
                            _fb_map_name = str(_fb_snap.get("map", _fb_snap.get("position", {}).get("map", "")) or "")
                            _bot_class = str(_fb_snap.get("job_name", _fb_snap.get("class", "novice")) or "novice")
                            _prog = _fb_snap.get("progression", {})
                            _bot_level = int(_prog.get("base_level", _prog.get("level", 0)) or 0)
                            if _bot_level == 0:
                                _bot_level = int(_fb_snap.get("base_level", _fb_snap.get("level", 1)) or 1)
                        else:
                            _fb_map_name = str(getattr(getattr(_fb_snap, "position", None), "map", "") or "")
                            _prog = getattr(_fb_snap, "progression", None)
                            if _prog is not None:
                                _bot_level = int(getattr(_prog, "base_level", 0) or 0)
                                _bot_class = str(getattr(_prog, "job_name", "novice") or "novice")
                            if _bot_level == 0:
                                _bot_level = int(getattr(_fb_snap, "base_level", 1) or 1)
                                _bot_class = str(getattr(_fb_snap, "job_name", "novice") or "novice")
                        
                        # ── Auto-register bot with role manager ──
                        try:
                            _rm = getattr(self._runtime, "role_manager", None)
                            if _rm is not None:
                                _rm.register_bot(bot_id=_bid, class_name=_bot_class, level=_bot_level)
                        except Exception:
                            pass
                        
                        # ── Record experience snapshot to DB ──
                        try:
                            _exp_db = getattr(self._runtime, "experience_db", None)
                            if _exp_db is not None:
                                if isinstance(_fb_snap, dict):
                                    _prog_data = _fb_snap.get("progression", {}) or {}
                                    _vitals = _fb_snap.get("vitals", {}) or {}
                                    _inv = _fb_snap.get("inventory", {}) or {}
                                    from ai_sidecar.experience_db import ExpSnapshot
                                    _exp_snap = ExpSnapshot(
                                        bot_id=_bid,
                                        base_level=_bot_level,
                                        job_level=int(_prog_data.get("job_level", 1) or 1),
                                        base_exp=int(_prog_data.get("base_exp", 0) or 0),
                                        job_exp=int(_prog_data.get("job_exp", 0) or 0),
                                        zeny=int(_prog_data.get("zeny", 0) or 0),
                                        map_name="",
                                    )
                                    _exp_db.record_exp_snapshot(_exp_snap)
                                    # Log leveling speed
                                    _ls = _exp_db.get_leveling_speed(bot_id=_bid)
                                    if _ls.get("time_at_current_level_min", 0) > 0:
                                        _plateau = _exp_db.get_plateau_warnings(bot_id=_bid)
                                        if _plateau:
                                            logger.info("exp_plateau: bot=%s level=%d mins=%d",
                                                       _bid, _ls.get("current_base_level", 0),
                                                       _ls.get("time_at_current_level_min", 0))
                        except Exception:
                            pass
                        
                        # ── Store episodic memory of this cycle ──
                        try:
                            _mem = getattr(self._runtime, "memory_retrieval", None)
                            if _mem is not None:
                                _mem.capture_snapshot(
                                    bot_id=_bid,
                                    tick_id=f"pdca_{horizon.value}_{time.monotonic_ns()}",
                                    summary=f"Level {_bot_level} {_bot_class} on {_fb_map_name}",
                                    payload={"map": _fb_map_name, "level": _bot_level, "class": _bot_class},
                                )
                        except Exception:
                            pass
            except Exception:
                pass
            _fallback_ge = _emit_game_engine_actions(
                self._runtime, horizon.value, bot_id=_bid, map_name=_fb_map_name
            )
            _fallback_hs = _emit_heuristic_actions(self._runtime, horizon.value, bot_id=_bid)
            _fallback_swarm = _emit_swarm_actions(self._runtime, horizon.value, bot_id=_bid)
            _fallback_vendor = _emit_vendor_actions(self._runtime, horizon.value, bot_id=_bid)
            _fallback_skill = _emit_skill_actions(self._runtime, horizon.value, bot_id=_bid)
            
            # ── Party coordination: signals for LLM to decide ──
            # The LLM CrewAI agents receive party-role signals through the
            # enriched state and plan context. PartyCoordinator.assess()
            # provides reflex-level coordination for urgent situations.
            try:
                _pc = getattr(self._runtime, "party_coordinator", None)
                if _pc is not None and _fb_map_name:
                    _signals = {"map_name": _fb_map_name, "hp_ratio": 1.0, "sp_ratio": 1.0,
                                "weight_ratio": 0.0, "in_combat": False}
                    _coord_action = _pc.assess(_signals, _bid)
                    if _coord_action is not None and _coord_action.confidence >= 0.5:
                        from ai_sidecar.contracts.actions import ActionProposal, ActionPriorityTier
                        from datetime import UTC, datetime, timedelta
                        import hashlib as _hashlib
                        _short_id = _hashlib.md5(f"{_bid}_party_{horizon.value}_{time.monotonic_ns()}".encode()).hexdigest()[:16]
                        _proposal = ActionProposal(
                            action_id=f"party_{horizon.value}_{_short_id}",
                            kind=_coord_action.kind, command=_coord_action.command,
                            priority_tier=ActionPriorityTier.tactical, source="planner",
                            created_at=datetime.now(UTC), expires_at=datetime.now(UTC) + timedelta(seconds=30),
                            idempotency_key=f"party_{_bid}_{horizon.value}",
                            metadata={"goal": "party", "objective": _coord_action.reason,
                                      "horizon": horizon.value, "bot_id": _bid, "source": "party_ai"},
                        )
                        _aq_party = getattr(self._runtime, "action_queue", None)
                        if _aq_party is not None:
                            _aq_party.enqueue(_bid, _proposal)
                            _fallback_skill += 1
                            logger.info("party_coordination: bot=%s action=%s reason=%s",
                                      _bid, _coord_action.kind, _coord_action.reason)
            except Exception:
                pass
            
            # ── Progression decisions handled by LLM CrewAI agents ──
            # All economy, survival, progression decisions flow through
            # the LLM planner. The context_assembler sends zeny, inventory,
            # stats, skills, and all bot state to the LLM.
            
            _fallback_total += _fallback_ge + _fallback_hs + _fallback_swarm + _fallback_vendor + _fallback_skill
        if _fallback_total > 0:
            logger.info(
                "fallback_emitters: total=%d bots=%d", _fallback_total, len(_all_bot_ids),
            )
        
        start = time.monotonic()
        plan_id: str | None = self._artifact_id(self._active_plan[horizon])
        actions_queued = 0
        re_planned = False
        force_replan = False
        objective = ""
        selected_goal = ""
        replan_reasons: list[str] = []

        try:
            # ── CHECK phase ──────────────────────────────────────
            latest_snapshot = self._get_latest_snapshot()
            progress = self._progress_tracker.evaluate(
                horizon=horizon,
                active_plan=self._active_plan[horizon],
                snapshot=latest_snapshot,
            )

            stuck = progress.stuck_cycles >= self._config.max_stuck_cycles
            # ── Web research trigger: if stuck, research the problem ──
            if stuck and hasattr(self._runtime, "web_research") and self._runtime.web_research is not None:
                _wr = self._runtime.web_research
                _bid = decision_meta.bot_id if "decision_meta" in dir() else _cycle_bot_id
                _map = str(getattr(getattr(latest_snapshot, "position", None), "map", "")) if latest_snapshot else ""
                if _wr.needs_research(_bid, f"stuck_{horizon.value}", cooldown_s=600):
                    _ctx = {"map": _map, "horizon": horizon.value, "bot_id": _bid}
                    try:
                        _task = asyncio.create_task(_wr.research("stuck_map", context=_ctx))
                        logger.info("web_research_triggered: bot=%s map=%s horizon=%s", _bid, _map, horizon.value)
                    except RuntimeError:
                        logger.warning("web_research_skipped: event_loop_stopping")
            replan_reasons = self._collect_replan_reasons(
                horizon=horizon,
                progress=progress,
                snapshot=latest_snapshot,
            )
            force_replan = bool(replan_reasons)

            decision_meta = ContractMeta(source="pdca_loop", bot_id=self._resolve_bot_id(latest_snapshot))
            goal_state = self._select_goal_state(
                meta=decision_meta,
                horizon=horizon,
                replan_reasons=replan_reasons,
            )
            
            # ── Push plan actions to queue ──────────────────────
            if goal_state is not None:
                try:
                    _bot = decision_meta.bot_id
                    _goal = str(getattr(goal_state, "selected_goal", "") or "")
                    _obj = str(getattr(getattr(goal_state, "selected_goal", None), "objective", "") or "")
                    if _goal and _obj:
                        from datetime import UTC, datetime, timedelta
                        from ai_sidecar.contracts.actions import ActionProposal, ActionPriorityTier
                        _aq = getattr(self._runtime, "action_queue", None)
                        if _aq is not None:
                            import hashlib as _hashlib
                            _short_id = _hashlib.md5(f"{_bot}_{horizon.value}_{_goal}_{time.monotonic_ns()}".encode()).hexdigest()[:16]
                            _cmd = _extract_command_from_goal(_goal, _obj)
                            
                            # Query goal decomposer for next sub-goal
                            _gd = getattr(self._runtime, "goal_decomposer", None)
                            if _gd is not None:
                                try:
                                    # Decompose the selected goal into sub-goals
                                    from ai_sidecar.contracts.autonomy import GoalDirective
                                    _directive = GoalDirective(
                                        goal_key=goal_state.selected_goal.goal_key,
                                        objective=goal_state.selected_goal.objective,
                                    )
                                    # Resolve bot_level from snapshot for goal decomposer
                                    _bot_level = 1
                                    if latest_snapshot is not None:
                                        try:
                                            if isinstance(latest_snapshot, dict):
                                                _bot_level = int(latest_snapshot.get("base_level", latest_snapshot.get("level", 1)) or 1)
                                            else:
                                                _bot_level = int(getattr(latest_snapshot, "base_level", 1) or 1)
                                        except Exception:
                                            pass
                                    _gd.decompose(bot_id=_bot, selected=_directive, bot_level=_bot_level)
                                    # Get next actionable sub-goal
                                    _next = _gd.next_action(bot_id=_bot)
                                    if _next is not None:
                                        _cmd = _next.metadata.get("action", _cmd)
                                        _obj = _next.objective
                                        logger.info("pdca_goal_decomposed: bot=%s sub_goal=%s action=%s",
                                                   _bot, _next.id, _cmd)
                                except Exception:
                                    logger.exception("pdca_goal_decomposition_failed")
                            
                            # Query ExperienceDatabase for best action
                            try:
                                if hasattr(self._runtime, "experience_db") and self._runtime.experience_db is not None:
                                    _exp = self._runtime.experience_db
                                    _map_name = ""
                                    if latest_snapshot is not None:
                                        _pos_snap = latest_snapshot.position if hasattr(latest_snapshot, "position") else None
                                        _map_name = getattr(_pos_snap, "map", "") if _pos_snap else ""
                                    _best_cmd, _best_rate = _exp.best_action(context_type="combat", map_name=_map_name)
                                    if _best_cmd:
                                        _cmd = _best_cmd
                                        logger.info("pdca_exp_best_action: bot=%s map=%s cmd=%s rate=%.2f", _bot, _map_name, _cmd, _best_rate)
                                    elif _map_name and "prontera" in _map_name.lower() and not "fild" in _map_name.lower():
                                        # Bot in town with no learned path — heuristic/reflex will handle
                                        _cmd = "ai auto"
                            except Exception:
                                pass
                            proposal = ActionProposal(
                                action_id=f"pdca_{horizon.value}_{_short_id}",
                                kind="command",
                                command=_cmd,
                                priority_tier=ActionPriorityTier.tactical if horizon.value in ("short_term", "tactical") else ActionPriorityTier.strategic,
                                source="planner",
                                created_at=datetime.now(UTC),
                                expires_at=datetime.now(UTC) + timedelta(seconds=60),
                                idempotency_key=f"pdca_{horizon.value}_{_short_id}",
                                metadata={"goal": _goal[:100], "objective": _obj[:100], "horizon": horizon.value, "bot_id": _bot},
                            )
                            _aq.enqueue(_bot, proposal)
                            logger.info("pdca_action_queued: bot=%s goal=%s horizon=%s", _bot, _goal, horizon.value)
                except Exception:
                    logger.exception("pdca_action_enqueue_failed")
            
            # ── Shadow mode: compare LLM decision vs heuristic ──
            if goal_state is not None and hasattr(self._runtime, "ml_shadow") and self._runtime.ml_shadow is not None:
                try:
                    _hs = getattr(self._runtime, "heuristic_service", None)
                    if _hs is not None:
                        _sigs = {}
                        if latest_snapshot and isinstance(latest_snapshot, dict):
                            v = latest_snapshot.get("vitals") or {}
                            _sigs["hp_ratio"] = float(v.get("hp_ratio", 1.0))
                            c = latest_snapshot.get("combat") or {}
                            _sigs["combat.aggro_count"] = int(c.get("aggro_count", 0))
                            _sigs["map_known"] = bool(latest_snapshot.get("map_known", False))
                        _sigs["bot_id"] = decision_meta.bot_id
                        _sigs["horizon"] = horizon.value
                        _h_assessment = _hs.assess(_sigs)
                        _llm_goal = str(getattr(goal_state, "selected_goal", "") or "")
                        _llm_obj = str(getattr(getattr(goal_state, "selected_goal", None), "objective", "") or "")
                        _h_action = _h_assessment.actions[0].command if _h_assessment.actions else "none"
                        self._runtime.ml_shadow.compare(
                            bot_id=decision_meta.bot_id, trace_id=f"pdca_{horizon.value}",
                            family=__import__("ai_sidecar.contracts.ml_subconscious", fromlist=["ModelFamily"]).ModelFamily.heuristic_decision,
                            model_version="heuristic_v1",
                            planner_choice={"goal": _llm_goal, "objective": _llm_obj},
                            recommendation={"action": _h_action, "top_domain": _h_assessment.top_domain},
                            confidence=_h_assessment.confidence,
                        )
                except Exception:
                    logger.exception("shadow_mode_compare_failed")
            
            if goal_state is not None:
                selected_goal = goal_state.selected_goal.goal_key.value
                objective = goal_state.selected_goal.objective
                goal_metadata = goal_state.selected_goal.metadata if isinstance(goal_state.selected_goal.metadata, dict) else {}
                if bool(goal_metadata.get("mission_force_replan")):
                    if "mission_agent_replan" not in replan_reasons:
                        replan_reasons.append("mission_agent_replan")
                    force_replan = True

            startup_gate = self._evaluate_startup_gate(
                bot_id=decision_meta.bot_id,
                horizon=horizon,
                snapshot=latest_snapshot,
                goal_state=goal_state,
                replan_reasons=replan_reasons,
            )
            if not bool(startup_gate.get("gate_open", False)):
                logger.info(
                    "pdca_startup_gate_blocked",
                    extra={
                        "event": "pdca_startup_gate_blocked",
                        "bot_id": decision_meta.bot_id,
                        "horizon": horizon.value,
                        "reason": startup_gate.get("reason"),
                        "mode": startup_gate.get("mode"),
                        "snapshot_ready": startup_gate.get("snapshot_ready"),
                        "history_ready": startup_gate.get("history_ready"),
                        "continuity_goal_state_present": startup_gate.get("continuity_goal_state_present"),
                        "recent_event_count": startup_gate.get("recent_event_count"),
                    },
                )
                return PDCAResult(
                    horizon=horizon,
                    plan_id=plan_id,
                    actions_queued=0,
                    progress_pct=progress.progress_pct,
                    stuck=stuck,
                    re_planned=False,
                    force_replan=force_replan,
                    replan_reasons=replan_reasons,
                    objective=objective,
                    selected_goal=selected_goal,
                    cycle_ms=(time.monotonic() - start) * 1000,
                    error=None,
                )

            # ── PLAN phase ───────────────────────────────────────
            if self._active_plan[horizon] is None or force_replan:
                objective_override = self._select_objective(
                    horizon=horizon,
                    snapshot=latest_snapshot,
                    replan_reasons=replan_reasons,
                    goal_state=goal_state,
                )
                objective = objective_override or self._objective_for(horizon=horizon, snapshot=latest_snapshot)
                plan = await self._generate_plan(
                    horizon,
                    latest_snapshot,
                    force_replan=force_replan and self._active_plan[horizon] is not None,
                    objective_override=objective_override,
                    startup_gate=startup_gate,
                )
                if plan:
                    self._active_plan[horizon] = plan
                    plan_id = self._artifact_id(plan) or f"plan_{int(time.time())}"
                    re_planned = True
                    self._stuck_counter[horizon] = 0
                else:
                    logger.warning(
                        "pdca_plan_generation_unavailable",
                        extra={
                            "event": "pdca_plan_generation_unavailable",
                            "bot_id": decision_meta.bot_id,
                            "horizon": horizon.value,
                            "mode": startup_gate.get("mode"),
                            "reason": startup_gate.get("reason"),
                            "selected_goal": selected_goal,
                            "objective": objective,
                        },
                    )
            else:
                objective = objective or self._objective_for(horizon=horizon, snapshot=latest_snapshot)

            # ── DO phase ─────────────────────────────────────────
            if self._active_plan[horizon] is not None or goal_state is not None:
                actions_queued = await self._plan_executor.execute(
                    plan=self._active_plan[horizon],
                    horizon=horizon,
                    max_actions=self._config.max_actions_per_cycle,
                    goal_state=goal_state,
                )

            # ── ACT phase ────────────────────────────────────────
            if force_replan and re_planned:
                self._stuck_counter[horizon] = 0
                logger.info(
                    "Re-planned [%s] after %d stuck cycles reasons=%s",
                    horizon.value,
                    progress.stuck_cycles,
                    ",".join(replan_reasons),
                )
            elif progress.stuck_cycles > 0:
                self._stuck_counter[horizon] = progress.stuck_cycles

            plan_id = plan_id or self._artifact_id(self._active_plan[horizon])

            return PDCAResult(
                horizon=horizon,
                plan_id=plan_id,
                actions_queued=actions_queued,
                progress_pct=progress.progress_pct,
                stuck=stuck,
                re_planned=re_planned,
                force_replan=force_replan,
                replan_reasons=replan_reasons,
                objective=objective,
                selected_goal=selected_goal,
                cycle_ms=(time.monotonic() - start) * 1000,
            )

        except Exception as e:
            logger.exception("PDCA cycle failed [%s]", horizon.value)
            return PDCAResult(
                horizon=horizon,
                plan_id=plan_id,
                actions_queued=actions_queued,
                progress_pct=0.0,
                stuck=False,
                re_planned=re_planned,
                force_replan=force_replan,
                replan_reasons=replan_reasons,
                objective=objective,
                selected_goal=selected_goal,
                cycle_ms=(time.monotonic() - start) * 1000,
                error=str(e),
            )

    def _artifact_id(self, artifact: StrategicPlan | TacticalIntentBundle | None) -> str | None:
        if artifact is None:
            return None
        return (
            getattr(artifact, "plan_id", None)
            or getattr(artifact, "bundle_id", None)
            or getattr(artifact, "id", None)
        )

    def _interval_for(self, horizon: Horizon) -> float:
        if horizon == Horizon.SHORT_TERM:
            return self._config.short_term_interval_s
        elif horizon == Horizon.MEDIUM_TERM:
            return self._config.medium_term_interval_s
        return self._config.long_term_interval_s

    def _get_latest_snapshot(self) -> BotStateSnapshot | None:
        """Get the latest snapshot from the runtime's snapshot cache."""
        try:
            bot_id = self._resolve_bot_id()
            cache = getattr(self._runtime, "snapshot_cache", None)
            if cache and hasattr(cache, "get"):
                snapshot = cache.get(bot_id)
                if snapshot is not None:
                    self._last_bot_id = bot_id
                    return snapshot
        except Exception:
            logger.exception("Failed to get latest snapshot")
        return None

    async def _generate_plan(
        self,
        horizon: Horizon,
        snapshot: BotStateSnapshot | None,
        *,
        force_replan: bool = False,
        objective_override: str | None = None,
        startup_gate: dict[str, object] | None = None,
    ) -> StrategicPlan | TacticalIntentBundle | None:
        """Generate a plan using planner or crewAI depending on horizon."""
        try:
            bot_id = self._resolve_bot_id(snapshot)
            objective = objective_override or self._objective_for(horizon=horizon, snapshot=snapshot)
            if horizon == Horizon.LONG_TERM:
                # Use crewAI strategize for long-term strategic plans
                crewai_reason = "crewai_unusable"
                startup_mode = str((startup_gate or {}).get("mode") or "pending")
                startup_reason = str((startup_gate or {}).get("reason") or "")
                if startup_mode not in {"conscious", "degraded"}:
                    logger.warning(
                        "pdca_long_term_startup_gate_unexpected_mode",
                        extra={
                            "event": "pdca_long_term_startup_gate_unexpected_mode",
                            "bot_id": bot_id,
                            "mode": startup_mode,
                            "reason": startup_reason,
                        },
                    )
                try:
                    result = await self._runtime.crewai_strategize(
                        CrewStrategizeRequest(
                            meta=ContractMeta(source="pdca_loop", bot_id=bot_id),
                            objective=objective,
                            horizon=PlanHorizon.strategic,
                            force_replan=force_replan,
                            max_steps=12,
                            context_overrides=self._context_overrides(snapshot),
                        )
                    )
                    crew_ok = bool(getattr(result, "ok", False))
                    crew_message = str(getattr(result, "message", "") or "")
                    crewai_errors = [str(item) for item in list(getattr(result, "errors", []) or [])]
                    crew_signals = [crew_message, *crewai_errors]
                    crew_degraded_signal = any(
                        token in signal.lower()
                        for signal in crew_signals
                        for token in ("crewai_disabled", "crewai_unavailable", "crewai_pipeline_disabled")
                    )

                    planner_response = getattr(result, "planner_response", None)
                    if crew_ok and not crew_degraded_signal and planner_response is not None:
                        artifact = planner_response.strategic_plan or planner_response.tactical_bundle
                        if artifact is not None:
                            self._record_startup_gate_success(bot_id=bot_id)
                            return artifact
                    crewai_reason = ",".join([crewai_reason, *crewai_errors]).strip(",") or "crewai_unusable"
                    self._record_startup_gate_failure(bot_id=bot_id, reason=crewai_reason)
                except Exception as exc:
                    crewai_reason = f"crewai_exception:{type(exc).__name__}"
                    self._record_startup_gate_failure(bot_id=bot_id, reason=crewai_reason)

                planner_fn = getattr(self._runtime, "planner_plan", None)
                if callable(planner_fn):
                    fallback_state = self._startup_gate_status(bot_id=bot_id)
                    fallback_mode = str(fallback_state.get("mode") or "warmup")
                    fallback_reason = str(fallback_state.get("reason") or "")
                    if fallback_mode != "degraded":
                        logger.error(
                            "pdca_long_term_conscious_required_before_fallback",
                            extra={
                                "event": "pdca_long_term_conscious_required_before_fallback",
                                "bot_id": bot_id,
                                "reason": crewai_reason,
                                "startup_mode": fallback_mode,
                                "startup_reason": fallback_reason,
                            },
                        )
                        return None
                    logger.info(
                        "pdca_long_term_fallback_to_planner",
                        extra={
                            "event": "pdca_long_term_fallback_to_planner",
                            "bot_id": bot_id,
                            "objective": objective,
                            "reason": crewai_reason,
                            "startup_mode": fallback_mode,
                            "startup_reason": fallback_reason,
                        },
                    )
                    fallback = await planner_fn(
                        PlannerPlanRequest(
                            meta=ContractMeta(source="pdca_loop", bot_id=bot_id),
                            objective=objective,
                            horizon=PlanHorizon.strategic,
                            force_replan=force_replan,
                            max_steps=12,
                        )
                    )
                    if fallback and getattr(fallback, "ok", False):
                        artifact = fallback.strategic_plan or fallback.tactical_bundle
                        if artifact is not None:
                            return artifact
                return None
            elif horizon == Horizon.MEDIUM_TERM:
                # Use planner for medium-term tactical bundles
                result = await self._runtime.planner_plan(
                    PlannerPlanRequest(
                        meta=ContractMeta(source="pdca_loop", bot_id=bot_id),
                        objective=objective,
                        horizon=PlanHorizon.tactical,
                        force_replan=force_replan,
                        max_steps=8,
                        context_overrides=self._context_overrides(snapshot),
                    )
                )
                if result and getattr(result, "ok", False) and result.tactical_bundle is not None:
                    return result.tactical_bundle
                return None
            else:
                # SHORT_TERM: use planner for immediate actions
                result = await self._runtime.planner_plan(
                    PlannerPlanRequest(
                        meta=ContractMeta(source="pdca_loop", bot_id=bot_id),
                        objective=objective,
                        horizon=PlanHorizon.tactical,
                        force_replan=force_replan,
                        max_steps=4,
                        context_overrides=self._context_overrides(snapshot),
                    )
                )
                if result and getattr(result, "ok", False) and result.tactical_bundle is not None:
                    return result.tactical_bundle
                return None
        except Exception:
            logger.exception("Plan generation failed [%s]", horizon.value)
            return None

    def _circuit_breaker_tripped(self) -> bool:
        allowed, _state = self._circuit_breaker.allow(
            bot_id=self._breaker_bot_id,
            key=self._breaker_key,
            family=self._breaker_family,
        )
        return not allowed

    def _resolve_bot_id(self, snapshot: BotStateSnapshot | dict[str, object] | None = None) -> str:
        if snapshot is not None:
            if isinstance(snapshot, dict):
                bot_id = snapshot.get("bot_id", snapshot.get("id", ""))
                if bot_id:
                    self._last_bot_id = str(bot_id)
                    return self._last_bot_id
            else:
                if getattr(snapshot, "meta", None) is not None:
                    bot_id = getattr(snapshot.meta, "bot_id", None)
                    if bot_id:
                        self._last_bot_id = str(bot_id)
                        return self._last_bot_id

        if self._last_bot_id:
            return self._last_bot_id

        # Try to find any registered bot from the runtime
        try:
            if hasattr(self._runtime, "list_bots"):
                bots = self._runtime.list_bots()
                if bots:
                    first = bots[0]
                    bot_id = first.get("bot_id") if isinstance(first, dict) else getattr(first, "bot_id", None)
                    if bot_id:
                        self._last_bot_id = str(bot_id)
                        return self._last_bot_id
        except Exception:
            logger.exception("Failed to resolve active bot id for PDCA loop")

        # Try snapshot cache as last resort
        try:
            cache = getattr(self._runtime, "snapshot_cache", None)
            if cache and hasattr(cache, "bot_ids"):
                ids = cache.bot_ids()
                if ids:
                    self._last_bot_id = ids[0]
                    return self._last_bot_id
        except Exception:
            pass

        return self._default_bot_id

    def _resolve_cost_gate_bot_id(self) -> str:
        try:
            if not hasattr(self, '_bot_rotation_index'):
                self._bot_rotation_index = 0
            if hasattr(self._runtime, "list_bots"):
                bots = self._runtime.list_bots()
                if bots:
                    # Filter out stale bots (not seen in last 5 min)
                    import datetime as _dt
                    _cutoff = _dt.datetime.now(_dt.UTC) - _dt.timedelta(minutes=5)
                    _fresh = []
                    for b in bots:
                        _last = b.get("last_seen_at") if isinstance(b, dict) else getattr(b, "last_seen_at", None)
                        if _last and isinstance(_last, str):
                            try:
                                _ts = _dt.datetime.fromisoformat(_last.replace("Z", "+00:00"))
                                if _ts >= _cutoff:
                                    _fresh.append(b)
                            except: pass
                    if not _fresh:
                        _fresh = bots
                    # Rotate through fresh bots
                    self._bot_rotation_index = (self._bot_rotation_index + 1) % max(len(_fresh), 1)
                    bid = _fresh[self._bot_rotation_index]
                    bot_id = bid.get("bot_id") if isinstance(bid, dict) else getattr(bid, "bot_id", None)
                    if bot_id:
                        return str(bot_id)
        except Exception:
            pass
        return self._default_bot_id

    def _objective_for(self, *, horizon: Horizon, snapshot: BotStateSnapshot | dict[str, object] | None) -> str:
        if isinstance(snapshot, dict):
            current_map = str(snapshot.get("map", snapshot.get("position", {}).get("map", "unknown")) or "unknown")
        else:
            current_map = getattr(getattr(snapshot, "position", None), "map", None) or "unknown"
        if horizon == Horizon.LONG_TERM:
            return f"advance long-term progression: grind and level up safely from {current_map}"
        if horizon == Horizon.MEDIUM_TERM:
            return f"progress tactical objective: farm and move toward targets from {current_map}"
        return f"execute immediate tactical actions: resume grinding safely on {current_map}"

    def _collect_replan_reasons(
        self,
        *,
        horizon: Horizon,
        progress: Any,
        snapshot: BotStateSnapshot | None,
    ) -> list[str]:
        reasons: list[str] = []

        progress_reasons = list(getattr(progress, "reasons", []) or [])
        for item in progress_reasons:
            if item and item not in reasons:
                reasons.append(str(item))

        if bool(getattr(progress, "force_replan_hint", False)) is False and progress.stuck_cycles >= self._config.max_stuck_cycles:
            reasons.append("stuck_cycles")

        if snapshot is not None:
            if self._snapshot_disconnected(snapshot):
                reasons.append("disconnect_recovery")
            reconnect_age_s = self._snapshot_reconnect_age_s(snapshot)
            if reconnect_age_s is not None and reconnect_age_s >= self._policy_float("reconnect_grace_s", 20.0):
                reasons.append("reconnect_stale")
            if self._overweight_ratio(snapshot) >= 0.90:
                reasons.append("inventory_overweight_pressure")

        fleet_status = self._fleet_status()
        fleet_central_enabled = bool(fleet_status.get("central_enabled", True))
        if fleet_central_enabled and bool(fleet_status.get("stale", False)):
            reasons.append("fleet_central_stale")
        if fleet_central_enabled and bool(fleet_status.get("central_available", True)) is False:
            reasons.append("fleet_central_unavailable")

        # Limit trigger aggression for long-term horizon to hard stale/failure reasons.
        if horizon == Horizon.LONG_TERM:
            hard = {
                "stale_progress",
                "objective_aged_out",
                "death_loop_detected",
                "disconnect_recovery",
                "reconnect_stale",
                "fleet_central_stale",
                "fleet_central_unavailable",
                "stuck_cycles",
            }
            reasons = [item for item in reasons if item in hard]

        deduped: list[str] = []
        seen: set[str] = set()
        for item in reasons:
            key = str(item).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        return deduped

    def _select_objective(
        self,
        *,
        horizon: Horizon,
        snapshot: BotStateSnapshot | None,
        replan_reasons: list[str],
        goal_state: GoalStackState | None = None,
    ) -> str | None:
        if goal_state is not None:
            objective = str(goal_state.selected_goal.objective or "").strip()
            if objective:
                return objective

        if horizon == Horizon.LONG_TERM:
            return None

        ranked = self._ranked_objectives()
        if not ranked:
            return None

        preferred: str | None = None
        if "inventory_overweight_pressure" in replan_reasons:
            preferred = "economy"
        elif "death_loop_detected" in replan_reasons or "disconnect_recovery" in replan_reasons or "reconnect_stale" in replan_reasons:
            preferred = "recovery"
        elif "objective_aged_out" in replan_reasons and "quest" in ranked:
            preferred = "quest"
        elif "stale_progress" in replan_reasons or "map_dwell_no_gain" in replan_reasons or "route_churn_no_position_gain" in replan_reasons:
            preferred = "grind"

        choice = self._pick_ranked_objective(horizon=horizon, ranked=ranked, preferred=preferred, force_rotate=bool(replan_reasons))
        if choice is None:
            return None

        current_map = getattr(getattr(snapshot, "position", None), "map", None) or "unknown"
        if choice == "recovery":
            return f"recover safely and re-establish operational posture on {current_map}"
        if choice == "economy":
            return f"stabilize inventory and economy pressure safely from {current_map}"
        if choice == "quest":
            return f"advance active quest objectives safely near {current_map}"
        return f"resume efficient grind and loot progression safely on {current_map}"

    def _select_goal_state(
        self,
        *,
        meta: ContractMeta,
        horizon: Horizon,
        replan_reasons: list[str],
    ) -> GoalStackState | None:
        if not hasattr(self._runtime, "autonomy_decide"):
            return self._fallback_goal_state(meta=meta, horizon=horizon)
        try:
            decided = self._runtime.autonomy_decide(
                meta=meta,
                horizon=horizon.value,
                replan_reasons=replan_reasons,
            )
            if decided is not None:
                return decided
            return self._fallback_goal_state(meta=meta, horizon=horizon)
        except Exception:
            logger.exception(
                "pdca_goal_decision_failed",
                extra={
                    "event": "pdca_goal_decision_failed",
                    "bot_id": meta.bot_id,
                    "horizon": horizon.value,
                },
            )
            return self._fallback_goal_state(meta=meta, horizon=horizon)

    def _fallback_goal_state(self, *, meta: ContractMeta, horizon: Horizon) -> GoalStackState | None:
        latest_fn = getattr(self._runtime, "latest_goal_state", None)
        if not callable(latest_fn):
            return None
        try:
            restored = latest_fn(bot_id=meta.bot_id)
        except Exception:
            logger.exception(
                "pdca_goal_state_restore_failed",
                extra={
                    "event": "pdca_goal_state_restore_failed",
                    "bot_id": meta.bot_id,
                    "horizon": horizon.value,
                },
            )
            return None
        if restored is None:
            return None
        logger.info(
            "pdca_goal_state_restored_from_runtime_cache",
            extra={
                "event": "pdca_goal_state_restored_from_runtime_cache",
                "bot_id": meta.bot_id,
                "horizon": horizon.value,
                "restored_horizon": restored.horizon,
                "decision_version": restored.decision_version,
                "tick_id": restored.tick_id,
                "selected_goal": restored.selected_goal.goal_key.value,
            },
        )
        return restored

    def _startup_gate_status(self, *, bot_id: str) -> dict[str, object]:
        status_fn = getattr(self._runtime, "startup_gate_status", None)
        if callable(status_fn):
            try:
                status = status_fn(bot_id=bot_id)
                if isinstance(status, dict):
                    return status
            except Exception:
                logger.exception(
                    "pdca_startup_gate_status_failed",
                    extra={
                        "event": "pdca_startup_gate_status_failed",
                        "bot_id": bot_id,
                    },
                )

        return {
            "bot_id": bot_id,
            "gate_open": True,
            "mode": "degraded",
            "reason": "startup_gate_runtime_unavailable",
            "failure_count": 0,
            "last_error": "",
            "elapsed_s": 0.0,
            "grace_s": float(self._startup_gate_defaults["grace_s"]),
            "min_events": int(self._startup_gate_defaults["min_events"]),
            "snapshot_ready": True,
            "history_ready": True,
            "continuity_goal_state_present": True,
            "recent_event_count": int(self._startup_gate_defaults["min_events"]),
        }

    def _update_startup_gate(
        self,
        *,
        bot_id: str,
        gate_open: bool,
        mode: str,
        reason: str,
        failure_count: int,
        last_error: str,
        grace_s: float,
        min_events: int,
        major_reasons: list[str] | None = None,
    ) -> dict[str, object]:
        update_fn = getattr(self._runtime, "update_startup_gate", None)
        if callable(update_fn):
            try:
                status = update_fn(
                    bot_id=bot_id,
                    gate_open=gate_open,
                    mode=mode,
                    reason=reason,
                    failure_count=failure_count,
                    last_error=last_error,
                    grace_s=grace_s,
                    min_events=min_events,
                    major_reasons=major_reasons,
                )
                if isinstance(status, dict):
                    return status
            except Exception:
                logger.exception(
                    "pdca_startup_gate_update_failed",
                    extra={
                        "event": "pdca_startup_gate_update_failed",
                        "bot_id": bot_id,
                        "mode": mode,
                        "reason": reason,
                    },
                )
        return self._startup_gate_status(bot_id=bot_id)

    def _evaluate_startup_gate(
        self,
        *,
        bot_id: str,
        horizon: Horizon,
        snapshot: BotStateSnapshot | None,
        goal_state: GoalStackState | None,
        replan_reasons: list[str],
    ) -> dict[str, object]:
        status = self._startup_gate_status(bot_id=bot_id)
        snapshot_ready = bool(snapshot is not None and str(getattr(snapshot, "tick_id", "") or "").strip())
        if snapshot_ready:
            map_name = str(getattr(getattr(snapshot, "position", None), "map", "") or "").strip()
            snapshot_ready = bool(map_name)
        bot_ready = bool(snapshot is not None and snapshot_ready and not self._snapshot_disconnected(snapshot))

        continuity_goal_state_present = bool(goal_state is not None or status.get("continuity_goal_state_present", False))
        history_ready = bool(continuity_goal_state_present or status.get("history_ready", False))
        # Also consider gate ready if we've had snapshot activity for > 30s even without position data
        elapsed_s = max(0.0, float(status.get("elapsed_s") or 0.0))
        if not snapshot_ready and bool(snapshot is not None) and elapsed_s > 30.0:
            snapshot_ready = True
        # Force gate open after 30s regardless of history — bots need to act
        if not history_ready and elapsed_s > 30.0:
            history_ready = True
        minimum_readiness = bool(bot_ready and history_ready)

        fleet_status = self._fleet_status()
        fleet_enabled = bool(fleet_status.get("central_enabled", True))
        fleet_stale = bool(fleet_status.get("stale", False))
        fleet_available = bool(fleet_status.get("central_available", False))

        planner_degraded = False
        planner_reason = ""
        planner_fn = getattr(self._runtime, "planner_status", None)
        if callable(planner_fn):
            try:
                planner = planner_fn(bot_id=bot_id)
                planner_healthy = bool(getattr(planner, "planner_healthy", False))
                planner_updated_at = getattr(planner, "updated_at", None)
                stale_seconds: float | None = None
                if isinstance(planner_updated_at, datetime):
                    if planner_updated_at.tzinfo is None:
                        planner_updated_at = planner_updated_at.replace(tzinfo=UTC)
                    stale_seconds = max(0.0, (datetime.now(UTC) - planner_updated_at.astimezone(UTC)).total_seconds())
                stale_threshold = max(float(getattr(self._runtime, "planner_stale_threshold_s", 60.0) or 60.0), 1.0)
                planner_stale = (not planner_healthy) or stale_seconds is None or stale_seconds > stale_threshold
                if planner_stale:
                    planner_degraded = True
                    planner_reason = "planner_stale"
            except Exception:
                planner_degraded = True
                planner_reason = "planner_status_unavailable"

        crew_degraded = False
        crew_reason = ""
        crew_status_fn = getattr(self._runtime, "crewai_status", None)
        if callable(crew_status_fn):
            try:
                crew_status = crew_status_fn()
                crew_available = bool(getattr(crew_status, "crew_available", False))
                crew_enabled = bool(getattr(crew_status, "crewai_enabled", True))
                if not crew_enabled:
                    crew_degraded = True
                    crew_reason = "crewai_disabled"
                elif not crew_available:
                    crew_degraded = True
                    crew_reason = "crewai_unavailable"
            except Exception:
                crew_degraded = True
                crew_reason = "crewai_status_unavailable"

        startup_failures = int(status.get("failure_count", 0) or 0)
        startup_last_error = str(status.get("last_error") or "").strip()

        degraded_reasons: list[str] = []
        if fleet_enabled and fleet_stale:
            degraded_reasons.append("fleet_central_stale")
        if fleet_enabled and not fleet_available:
            degraded_reasons.append("fleet_central_unavailable")
        if planner_degraded and planner_reason:
            degraded_reasons.append(planner_reason)
        if crew_degraded and crew_reason:
            degraded_reasons.append(crew_reason)
        if startup_failures > 0:
            degraded_reasons.append("crewai_failures_present")
        if startup_last_error:
            degraded_reasons.append(startup_last_error)

        deduped_reasons: list[str] = []
        seen: set[str] = set()
        for item in degraded_reasons:
            key = str(item).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped_reasons.append(key)

        grace_s = float(status.get("grace_s", self._startup_gate_defaults["grace_s"]) or self._startup_gate_defaults["grace_s"])
        min_events = int(status.get("min_events", self._startup_gate_defaults["min_events"]) or self._startup_gate_defaults["min_events"])

        if not minimum_readiness:
            wait_reasons: list[str] = []
            if not snapshot_ready:
                wait_reasons.append("snapshot_unavailable")
            if snapshot_ready and not bot_ready:
                wait_reasons.append("bot_not_ready")
            if not history_ready:
                wait_reasons.append("history_unavailable")
            reason = f"startup_gate_waiting_minimum_live_state:{','.join(wait_reasons) or 'minimum_state_unavailable'}"
            self._update_startup_gate(
                bot_id=bot_id,
                gate_open=False,
                mode="warmup",
                reason=reason,
                failure_count=int(status.get("failure_count", 0) or 0),
                last_error=str(status.get("last_error") or ""),
                grace_s=grace_s,
                min_events=min_events,
                major_reasons=wait_reasons,
            )
            return self._startup_gate_status(bot_id=bot_id)

        mode = "degraded" if deduped_reasons else "conscious"
        reason = (
            f"startup_gate_open_degraded_optional_subsystems:{','.join(deduped_reasons)}"
            if deduped_reasons
            else "startup_gate_open_minimum_live_state_ready"
        )
        opened = self._update_startup_gate(
            bot_id=bot_id,
            gate_open=True,
            mode=mode,
            reason=reason,
            failure_count=int(status.get("failure_count", 0) or 0),
            last_error=str(status.get("last_error") or ""),
            grace_s=grace_s,
            min_events=min_events,
            major_reasons=deduped_reasons,
        )
        logger.info(
            "pdca_startup_gate_ready",
            extra={
                "event": "pdca_startup_gate_ready",
                "bot_id": bot_id,
                "horizon": horizon.value,
                "mode": mode,
                "reason": reason,
                "snapshot_ready": snapshot_ready,
                "bot_ready": bot_ready,
                "history_ready": history_ready,
                "continuity_goal_state_present": continuity_goal_state_present,
                "degraded_reasons": deduped_reasons,
                "replan_reasons": list(replan_reasons),
            },
        )
        return opened

    def _record_startup_gate_success(self, *, bot_id: str) -> None:
        status = self._startup_gate_status(bot_id=bot_id)
        self._update_startup_gate(
            bot_id=bot_id,
            gate_open=bool(status.get("gate_open", False)),
            mode=str(status.get("mode") or "conscious"),
            reason=str(status.get("reason") or "startup_gate_open_state_history_ready"),
            failure_count=0,
            last_error="",
            grace_s=float(status.get("grace_s", self._startup_gate_defaults["grace_s"]) or self._startup_gate_defaults["grace_s"]),
            min_events=int(status.get("min_events", self._startup_gate_defaults["min_events"]) or self._startup_gate_defaults["min_events"]),
        )

    def _record_startup_gate_failure(self, *, bot_id: str, reason: str) -> None:
        status = self._startup_gate_status(bot_id=bot_id)
        failures = int(status.get("failure_count", 0) or 0) + 1
        elapsed_s = float(status.get("elapsed_s", 0.0) or 0.0)
        grace_s = float(status.get("grace_s", self._startup_gate_defaults["grace_s"]) or self._startup_gate_defaults["grace_s"])
        max_failures = int(self._startup_gate_defaults["max_crewai_failures"])
        normalized_reason = str(reason or "crewai_unusable").strip()
        reason_lower = normalized_reason.lower()
        immediate_tokens = ("crewai_disabled", "crewai_unavailable", "crewai_pipeline_disabled")
        degrade = failures >= max_failures or elapsed_s >= grace_s or any(token in reason_lower for token in immediate_tokens)

        mode = "degraded" if degrade else str(status.get("mode") or "conscious")
        gate_reason = "startup_gate_degraded_after_bounded_crewai_failures" if degrade else str(status.get("reason") or "startup_gate_open_minimum_live_state_ready")
        major_reasons = [
            str(item).strip()
            for item in list(status.get("major_reasons") or [])
            if str(item).strip()
        ]
        if normalized_reason:
            major_reasons.append(normalized_reason)
        self._update_startup_gate(
            bot_id=bot_id,
            gate_open=bool(status.get("gate_open", True)),
            mode=mode,
            reason=gate_reason,
            failure_count=failures,
            last_error=normalized_reason,
            grace_s=grace_s,
            min_events=int(status.get("min_events", self._startup_gate_defaults["min_events"]) or self._startup_gate_defaults["min_events"]),
            major_reasons=major_reasons,
        )
        if degrade:
            logger.warning(
                "pdca_startup_gate_degraded",
                extra={
                    "event": "pdca_startup_gate_degraded",
                    "bot_id": bot_id,
                    "failure_count": failures,
                    "elapsed_s": elapsed_s,
                    "grace_s": grace_s,
                    "reason": reason,
                },
            )

    def _pick_ranked_objective(
        self,
        *,
        horizon: Horizon,
        ranked: list[str],
        preferred: str | None,
        force_rotate: bool,
    ) -> str | None:
        if not ranked:
            return None

        now = time.time()
        current_index = int(self._objective_rotation_index.get(horizon, 0))
        last_switch = float(self._last_objective_switch_at.get(horizon, 0.0))
        cooldown_s = self._policy_float("objective_rotation_cooldown_s", 20.0)

        if preferred in ranked:
            current_index = ranked.index(preferred)
            self._objective_rotation_index[horizon] = current_index
            self._last_objective_switch_at[horizon] = now
            return ranked[current_index]

        if last_switch <= 0.0:
            self._objective_rotation_index[horizon] = current_index
            self._last_objective_switch_at[horizon] = now
            return ranked[current_index]

        if force_rotate or (now - last_switch) >= cooldown_s:
            current_index = (current_index + 1) % len(ranked)
            self._objective_rotation_index[horizon] = current_index
            self._last_objective_switch_at[horizon] = now

        return ranked[current_index]

    def _ranked_objectives(self) -> list[str]:
        policy = getattr(self._runtime, "autonomy_policy", {})
        ranked = []
        if isinstance(policy, dict):
            raw = policy.get("ranked_objectives")
            if isinstance(raw, list):
                ranked = [str(item).strip().lower() for item in raw if str(item).strip()]
            elif isinstance(raw, str):
                ranked = [item.strip().lower() for item in raw.split(",") if item.strip()]

        if not ranked:
            ranked = ["grind", "recovery", "economy", "quest"]
        return ranked

    def _policy_float(self, key: str, default: float) -> float:
        policy = getattr(self._runtime, "autonomy_policy", {})
        if isinstance(policy, dict):
            try:
                return float(policy.get(key, default))
            except (TypeError, ValueError):
                return default
        return default

    def _fleet_status(self) -> dict[str, object]:
        runtime_status_fn = getattr(self._runtime, "_fleet_status", None)
        if callable(runtime_status_fn):
            try:
                data = runtime_status_fn()
                if isinstance(data, dict):
                    return data
            except Exception:
                logger.exception("Failed to read runtime fleet status for PDCA")

        state = getattr(self._runtime, "fleet_constraint_state", None)
        if state is not None and hasattr(state, "status"):
            try:
                data = state.status()
                if isinstance(data, dict):
                    return data
            except Exception:
                logger.exception("Failed to read fleet status for PDCA")

        fleet_client = getattr(self._runtime, "fleet_sync_client", None)
        central_enabled = bool(getattr(fleet_client, "enabled", True))
        return {
            "mode": "local",
            "central_enabled": central_enabled,
            "central_available": False,
            "stale": (True if central_enabled else False),
            "last_sync_at": None,
            "doctrine_version": "local",
            "last_error": "fleet_constraint_state_unavailable",
        }

    def _snapshot_disconnected(self, snapshot: BotStateSnapshot | dict[str, object]) -> bool:
        raw = getattr(snapshot, "raw", {})
        if not isinstance(raw, dict):
            # Handle dict snapshots
            if isinstance(snapshot, dict):
                raw = snapshot.get("raw", {})
                if not isinstance(raw, dict):
                    raw = {}
            else:
                return False
        if raw.get("in_game") is False:
            return True
        status = str(raw.get("status") or raw.get("state") or raw.get("net_state") or "").strip().lower()
        return status in {
            "offline",
            "disconnected",
            "disconnect",
            "reconnecting",
            "connecting",
            "not_connected",
        }

    def _evaluate_conscious_triggers(
        self,
        *,
        horizon: Horizon,
        bot_id: str,
        snapshot: Any,
        recent_deaths: int,
    ) -> tuple[bool, str, dict[str, object]]:
        """Evaluate whether the conscious brain (LLM) should activate.
        
        EVERY cue matters — a missed trigger is a disaster.
        
        Trigger categories:
        - ANOMALY: death, disconnect, stuck, HP crash, zone mismatch
        - PROACTIVE: buy potions, sell/store, level up, job milestone,
                     zeny threshold, new map, card/rare drop
        - MILESTONE: stat/skill points
        - KAIZEN: strategic review + improvement review
        
        Returns: (should_activate, trigger_reason, trigger_context)
        """
        now = time.time()
        trigger_reasons: list[str] = []
        context: dict[str, object] = {}
        
        # ── PRUNE TRACKING STATE (prevent memory leaks) ──
        for _track_attr in ["_conscious_cycle_counts", "_conscious_tracked_state", "_stuck_start_times", "_prev_hp", "_last_trigger_context"]:
            _d = getattr(self, _track_attr, {})
            if len(_d) > 100:
                _keys = list(_d.keys())[:-50]
                for _k in _keys:
                    del _d[_k]
                object.__setattr__(self, _track_attr, _d)
        
        # ── EXTRACT SNAPSHOT DATA ──
        hp = 1; max_hp = 1; sp = 0; max_sp = 1
        is_dead = False; map_name = ""; is_town = False; is_new_map = False
        zeny = 0; has_pot = False; weight = 0.0
        stat_pts = 0; skill_pts = 0
        base_level = 1; job_level = 1; job_name = "novice"
        x = 0; y = 0; in_game = True; status = ""
        items_list: list[dict] = []
        try:
            if isinstance(snapshot, dict):
                hp = int(snapshot.get("vitals", {}).get("hp", 1) or 1)
                max_hp = int(snapshot.get("vitals", {}).get("max_hp", 1) or 1)
                sp = int(snapshot.get("vitals", {}).get("sp", 0) or 0)
                max_sp = int(snapshot.get("vitals", {}).get("max_sp", 1) or 1)
                is_dead = hp <= 0 or snapshot.get("status") == "dead"
                map_name = str(snapshot.get("map", snapshot.get("position", {}).get("map", "")) or "")
                is_town = any(t in map_name.lower() for t in ["prontera", "morocc", "payon", "geffen", "aldebaran", "yuno"]) \
                          and not any(f in map_name.lower() for f in ["fild", "dun", "cave", "forest", "field"])
                zeny = int(snapshot.get("progression", {}).get("zeny", 0) or 0)
                items_data = snapshot.get("inventory", {}) or {}
                items_list = items_data.get("items", []) or items_data.get("item_list", []) or []
                has_pot = any("red_pot" in str(i.get("name", "")).lower() for i in items_list if isinstance(i, dict))
                weight = float(items_data.get("weight_ratio", 0.0) or 0.0)
                stat_pts = int(snapshot.get("progression", {}).get("stat_points", 0) or 0)
                skill_pts = int(snapshot.get("progression", {}).get("skill_points", 0) or 0)
                base_level = int(snapshot.get("progression", {}).get("base_level", 1) or 1)
                job_level = int(snapshot.get("progression", {}).get("job_level", 1) or 1)
                job_name = str(snapshot.get("progression", {}).get("job_name", "novice") or "novice")
                pos = snapshot.get("position", {}) or {}
                x = int(pos.get("x", 0) or 0)
                y = int(pos.get("y", 0) or 0)
                raw = snapshot.get("raw", {}) or {}
                in_game = raw.get("in_game", True)
                status = str(raw.get("status", raw.get("state", raw.get("net_state", ""))) or "")
        except Exception:
            pass
        
        # Track cross-cycle state for delta detection
        _tracked = getattr(self, "_conscious_tracked_state", {})
        _prev = _tracked.get(bot_id, {})
        _now_state = {
            "map": map_name, "x": x, "y": y, "hp": hp, "base_level": base_level,
            "job_level": job_level, "zeny": zeny, "in_game": in_game,
            "stat_pts": stat_pts, "skill_pts": skill_pts,
        }
        _tracked[bot_id] = _now_state
        object.__setattr__(self, "_conscious_tracked_state", _tracked)
        
        # ── ⚠️  A N O M A L Y   D E T E C T I O N ⚠️ ──
        
        # A1: Death
        if recent_deaths > 0:
            trigger_reasons.append(f"anomaly:death_x{recent_deaths}")
            context["deaths"] = recent_deaths
        
        # A2: Disconnected or reconnecting
        if not in_game or status in ("offline", "disconnected", "disconnect", "reconnecting"):
            trigger_reasons.append(f"anomaly:disconnected({status})")
            context["status"] = status
        
        # A3: Position stuck (same x,y for >30s using real time)
        if _prev.get("x") == x and _prev.get("y") == y and _prev.get("map") == map_name:
            _stuck_start = getattr(self, "_stuck_start_times", {}).get(bot_id, now)
            _stuck_start_times = getattr(self, "_stuck_start_times", {})
            if bot_id not in _stuck_start_times:
                _stuck_start_times[bot_id] = now
            object.__setattr__(self, "_stuck_start_times", _stuck_start_times)
            _stuck_elapsed = now - _stuck_start_times.get(bot_id, now)
            if _stuck_elapsed >= 30.0 and is_town is False:
                trigger_reasons.append(f"anomaly:stuck_{int(_stuck_elapsed)}s")
                context["stuck_seconds"] = int(_stuck_elapsed)
        else:
            _stuck_start_times = getattr(self, "_stuck_start_times", {})
            _stuck_start_times[bot_id] = now
            object.__setattr__(self, "_stuck_start_times", _stuck_start_times)
        
        # A4: Frequent low-HP teleports (HP was >50%, now <20% without death)
        if _prev.get("hp", 1) > 0.5 * max_hp and hp < 0.2 * max_hp and not is_dead and hp > 0:
            trigger_reasons.append("anomaly:hp_crash")
            context["hp_drop"] = f"{_prev.get('hp',0)}→{hp}"
        
        # A5: action queue check — verify bot has recent activity
        _active_bots = getattr(self, "_active_bot_ids", set())
        _active_bots.add(bot_id)
        if len(_active_bots) > 50:
            _active_bots.clear()
        object.__setattr__(self, "_active_bot_ids", _active_bots)
        
        # ── 🎯  P R O A C T I V E   D E T E C T I O N 🎯 ──
        
        # P1: No potions + in town + has zeny = buy opportunity
        if is_town and not has_pot and zeny >= 500 and weight < 0.8:
            trigger_reasons.append("proactive:buy_potions")
            context["can_buy"] = True
        
        # P2: Overflowing inventory
        if weight >= 0.85:
            trigger_reasons.append("proactive:sell_or_store")
            context["weight_ratio"] = weight
        
        # P3: Level up detected (base_level changed)
        if _prev.get("base_level", 1) < base_level:
            trigger_reasons.append(f"proactive:level_up_lv{base_level}")
            context["new_level"] = base_level
        
        # P4: Job level milestone (10, 50, 70, 99)
        for milestone in [10, 50, 70, 99]:
            if _prev.get("job_level", 1) < milestone <= job_level:
                trigger_reasons.append(f"proactive:job_milestone_lv{job_level}")
                context["job_level"] = job_level
                context["next_job"] = job_name
        
        # P5: Zeny threshold crossed
        for threshold in [1000, 5000, 10000, 50000, 100000]:
            if _prev.get("zeny", 0) < threshold <= zeny:
                trigger_reasons.append(f"proactive:zeny_{threshold}")
                context["zeny"] = zeny
        
        # P6: New map discovered
        if _prev.get("map", "") != map_name and map_name:
            trigger_reasons.append(f"proactive:new_map_{map_name}")
            context["map"] = map_name
        
        # P7: Rare item in inventory (card, equipment with slots)
        if items_list:
            for _item in items_list:
                if isinstance(_item, dict):
                    _iname = str(_item.get("name", "")).lower()
                    if "card" in _iname:
                        trigger_reasons.append("proactive:card_drop")
                        context["item"] = _iname
                        break
                    _slots = int(_item.get("slots", 0) or 0)
                    if _slots > 0:
                        trigger_reasons.append("proactive:slotted_item")
                        context["item"] = _iname
                        break
        
        # ── 📊  M I L E S T O N E   D E T E C T I O N 📊 ──
        if stat_pts > 0 or skill_pts > 0:
            trigger_reasons.append(f"milestone:points_avail(s{stat_pts}/sk{skill_pts})")
            context["stat_points"] = stat_pts
            context["skill_points"] = skill_pts
        
        # ── 🔄  K A I Z E N   P E R I O D I C   R E V I E W 🔄 ──
        try:
            if horizon == Horizon.SHORT_TERM:
                _cycle_counts = getattr(self, "_conscious_cycle_counts", {})
                _count = _cycle_counts.get(bot_id, 0) + 1
                _cycle_counts[bot_id] = _count
                object.__setattr__(self, "_conscious_cycle_counts", _cycle_counts)
                
                if _count % 30 == 0:
                    trigger_reasons.append("strategic:periodic_review")
                    context["review_interval"] = "30_cycles"
                
                if _count % 60 == 0:
                    trigger_reasons.append("kaizen:kaizen_review")
                    context["kaizen"] = True
                # ── AUTO-TRAIN ML MODELS on kaizen cycle ──
                try:
                    _ml_harness = getattr(self._runtime, "ml_training", None)
                    if _ml_harness is not None:
                        from ai_sidecar.contracts.ml_subconscious import ModelFamily
                        _trained_any = False
                        for _family in ModelFamily:
                            _version, _samples, _metrics, _ab = _ml_harness.train(
                                family=_family, bot_id=bot_id,
                                incremental=True, max_samples=500,
                            )
                            if _version:
                                _trained_any = True
                                logger.info(
                                    "ml_auto_trained: family=%s version=%s samples=%d metrics=%s",
                                    _family.value, _version, _samples, _metrics,
                                )
                        if _trained_any:
                            context["ml_trained"] = True
                except Exception:
                    logger.exception("ml_auto_train_failed")
                # ── CHECK MACRO PATTERNS on kaizen cycle ──
                try:
                    _macro_ai = getattr(self._runtime, "macro_intelligence", None)
                    if _macro_ai is not None and isinstance(snapshot, dict):
                        _patterns = _macro_ai.get_patterns_for_context(bot_state=snapshot)
                        if _patterns:
                            _pattern_ids = [p.pattern_id for p in _patterns[:5]]
                            trigger_reasons.append(f"kaizen:macro_patterns_{len(_patterns)}")
                            context["macro_patterns"] = _pattern_ids
                except Exception:
                    pass
                # ── CHECK COMBAT OPTIMIZER on kaizen cycle ──
                try:
                    _combat_opt = getattr(self._runtime, "combat_optimizer", None)
                    if _combat_opt is not None and isinstance(snapshot, dict):
                        _monster_name = str(snapshot.get("target", snapshot.get("monster", "")) or "")
                        if _monster_name:
                            _is_mvp = _combat_opt.is_mvp(_monster_name)
                            _threat = _combat_opt.assess_threat(_monster_name, int(snapshot.get("progression", {}).get("base_level", 1) or 1))
                            _element_adv = _combat_opt.get_element_advantage(
                                str(snapshot.get("monster_element", "neutral"))
                            )
                            if _is_mvp:
                                trigger_reasons.append("combat:mvp_detected")
                                context["mvp"] = _monster_name
                            if _threat > 0.7:
                                trigger_reasons.append(f"combat:high_threat_{_threat:.1f}")
                                context["threat"] = _threat
                            if _element_adv < 0.5:
                                trigger_reasons.append("combat:element_disadvantage")
                                context["element_adv"] = _element_adv
                except Exception:
                    pass
                # ── CHECK QUEST AUTOMATION on kaizen cycle ──
                try:
                    _quest_auto = getattr(self._runtime, "quest_automation", None)
                    if _quest_auto is not None and isinstance(snapshot, dict):
                        _base_lv = int(snapshot.get("progression", {}).get("base_level", 1) or 1)
                        _job_lv = int(snapshot.get("progression", {}).get("job_level", 1) or 1)
                        _job = str(snapshot.get("progression", {}).get("job_name", "novice") or "novice")
                        _zeny = int(snapshot.get("progression", {}).get("zeny", 0) or 0)
                        _available = _quest_auto.get_available_quests(
                            bot_id, _base_lv, _job_lv, _job, _zeny
                        )
                        if _available:
                            trigger_reasons.append(f"quest:available_{len(_available)}")
                            context["available_quests"] = _available[:3]
                except Exception:
                    pass
                # ── CHECK SERVER AWARENESS on kaizen cycle ──
                try:
                    _srv = getattr(self._runtime, "server_awareness", None)
                    if _srv is not None:
                        _state = _srv.get_server_state(str(snapshot.get("map", "")) if isinstance(snapshot, dict) else "")
                        if _state.get("is_woe"):
                            trigger_reasons.append("server:woe_active")
                            context["woe"] = True
                        if _state.get("risk_level") == "high":
                            trigger_reasons.append("server:high_risk_window")
                            context["risk"] = "high"
                        if _state.get("player_density", 0) > 5:
                            trigger_reasons.append(f"server:crowded_{_state['player_density']}")
                            context["density"] = _state["player_density"]
                        if _srv.is_lagging():
                            trigger_reasons.append("server:lagging")
                            context["lag"] = True
                except Exception:
                    pass
                # ── CHECK SOCIAL INTELLIGENCE on kaizen cycle ──
                try:
                    _soc = getattr(self._runtime, "social_intelligence", None)
                    if _soc is not None:
                        _strategies = _soc.get_learned_strategies()
                        if _strategies:
                            trigger_reasons.append(f"social:learned_{len(_strategies)}")
                            context["learned_strategies"] = _strategies[:3]
                except Exception:
                    pass
                # ── CHECK PREDICTIVE PLANNER on kaizen cycle ──
                try:
                    _pred = getattr(self._runtime, "predictive_planner", None)
                    if _pred is not None and isinstance(snapshot, dict):
                        _level = int(snapshot.get("progression", {}).get("base_level", 1) or 1)
                        _zeny = int(snapshot.get("progression", {}).get("zeny", 0) or 0)
                        _inv = snapshot.get("inventory", {}) or {}
                        _items = _inv.get("items", []) or _inv.get("item_list", []) or []
                        _potion_count = sum(1 for i in _items if isinstance(i, dict) and "red_pot" in str(i.get("name", "")).lower())
                        _potion_pred = _pred.predict_potion_needs(_potion_count, _level, "")
                        if _potion_pred.get("critical"):
                            trigger_reasons.append("predict:potion_critical")
                            context["potion_minutes"] = round(_potion_pred["minutes_remaining"], 1)
                        elif _potion_pred.get("should_buy_soon"):
                            trigger_reasons.append("predict:potion_low")
                            context["potion_minutes"] = round(_potion_pred["minutes_remaining"], 1)
                        _gear_pred = _pred.predict_gear_upgrade(_level, 0, _zeny)
                        if _gear_pred:
                            trigger_reasons.append(f"predict:gear_upgrade_lv{_gear_pred['recommend_level']}")
                            context["gear_upgrade"] = _gear_pred
                except Exception:
                    pass
                # ── CHECK BUILD MANAGER on kaizen cycle ──
                try:
                    _bm = getattr(self._runtime, "build_manager", None)
                    if _bm is not None and isinstance(snapshot, dict):
                        _job = str(snapshot.get("progression", {}).get("job_name", "novice") or "novice")
                        _builds = _bm.get_available_builds(_job)
                        if _builds:
                            trigger_reasons.append(f"build:available_{len(_builds)}")
                            context["available_builds"] = [b["name"] for b in _builds[:3]]
                        # Check if bot has an active build
                        _active = _bm._active_builds.get(bot_id)
                        if _active is None and _builds:
                            trigger_reasons.append("build:needs_selection")
                            context["build_options"] = [b["name"] for b in _builds[:3]]
                        # Check stat allocation guidance
                        _stat_pts = int(snapshot.get("progression", {}).get("stat_points", 0) or 0)
                        if _stat_pts > 0 and _active:
                            _next_stat = _bm.get_next_stat(bot_id, {"STR": 0, "AGI": 0, "VIT": 0, "INT": 0, "DEX": 0, "LUK": 0})
                            if _next_stat:
                                context["recommended_stat"] = _next_stat
                                context["build_name"] = _active.get("name", "unknown")
                except Exception:
                    pass
                # ── CHECK RISK ASSESSMENT on kaizen cycle ──
                try:
                    _ra = getattr(self._runtime, "risk_assessment", None)
                    if _ra is not None and isinstance(snapshot, dict):
                        _hp = int(snapshot.get("vitals", {}).get("hp", 1) or 1)
                        _max_hp = int(snapshot.get("vitals", {}).get("max_hp", 1) or 1)
                        _sp = int(snapshot.get("vitals", {}).get("sp", 0) or 0)
                        _max_sp = int(snapshot.get("vitals", {}).get("max_sp", 1) or 1)
                        _level = int(snapshot.get("progression", {}).get("base_level", 1) or 1)
                        _zeny = int(snapshot.get("progression", {}).get("zeny", 0) or 0)
                        _target = str(snapshot.get("target", snapshot.get("monster", "")) or "")
                        _is_mvp = False
                        _kg = getattr(self._runtime, "knowledge_graph", None)
                        if _kg is not None and _target:
                            _is_mvp = _kg._monsters.get(_target.lower(), {}).get("Modes.Mvp", False)
                        _risk_ctx = {
                            "hp_pct": _hp / max(_max_hp, 1),
                            "sp_pct": _sp / max(_max_sp, 1),
                            "level": _level,
                            "target_level": _level + 5,
                            "zeny": _zeny,
                            "is_mvp": _is_mvp,
                            "has_escape": True,
                            "player_density": 0,
                            "is_woe": False,
                            "risk_window": "low",
                        }
                        _assessment = _ra.assess("general_farming", _risk_ctx)
                        if _assessment.get("recommendation") in ("avoid", "cautious"):
                            trigger_reasons.append(f"risk:{_assessment['recommendation']}")
                            context["risk"] = _assessment
                except Exception:
                    pass
                # ── CHECK SERVER PROFILER on kaizen cycle ──
                try:
                    _sp = getattr(self._runtime, "server_profiler", None)
                    if _sp is not None:
                        _personality = _sp.get_server_personality()
                        if _personality.get("strictness") == "strict":
                            trigger_reasons.append("server:strict_gm")
                            context["gm_risk"] = _personality["gm_risk"]
                except Exception:
                    pass
                # ── CHECK KNOWLEDGE GRAPH on kaizen cycle ──
                try:
                    _kg = getattr(self._runtime, "knowledge_graph", None)
                    if _kg is not None and isinstance(snapshot, dict):
                        _level = int(snapshot.get("progression", {}).get("base_level", 1) or 1)
                        _spots = _kg.find_farming_spot(_level)
                        if _spots:
                            context["recommended_spots"] = _spots[:3]
                except Exception:
                    pass
                # ── CHECK COMBAT INSTINCT on kaizen cycle ──
                try:
                    _ci = getattr(self._runtime, "combat_instinct", None)
                    if _ci is not None and isinstance(snapshot, dict):
                        _hp = int(snapshot.get("vitals", {}).get("hp", 1) or 1)
                        _max_hp = int(snapshot.get("vitals", {}).get("max_hp", 1) or 1)
                        _hp_drop = int(snapshot.get("vitals", {}).get("hp_drop", 0) or 0)
                        if _hp_drop > _max_hp * 0.2:
                            _analysis = _ci.analyze_damage(
                                bot_id, _hp_drop, _hp, _max_hp,
                                nearby_monsters=snapshot.get("monsters", [])
                            )
                            if _analysis.get("threat_level") in ("high", "critical"):
                                trigger_reasons.append(f"combat:{_analysis['cause']}")
                                context["combat_analysis"] = _analysis
                except Exception:
                    pass
                # ── CHECK PARTY INTELLIGENCE on kaizen cycle ──
                try:
                    _pi = getattr(self._runtime, "party_intelligence", None)
                    if _pi is not None:
                        _party_health = _pi.assess_party_health()
                        if _party_health.get("status") == "critical":
                            trigger_reasons.append("party:critical_health")
                            context["party_health"] = _party_health
                except Exception:
                    pass
                # ── CHECK MARKET INTELLIGENCE on kaizen cycle ──
                try:
                    _mi = getattr(self._runtime, "market_intelligence", None)
                    if _mi is not None:
                        _arb = _mi.find_arbitrage()
                        if _arb:
                            trigger_reasons.append(f"market:arbitrage_{len(_arb)}")
                            context["arbitrage"] = _arb[:3]
                        if _mi.is_woe_time():
                            trigger_reasons.append("market:woe_price_spike")
                except Exception:
                    pass
                # ── CHECK WOE INTELLIGENCE on kaizen cycle ──
                try:
                    _wi = getattr(self._runtime, "woe_intelligence", None)
                    if _wi is not None:
                        _woe = _wi.get_woe_status()
                        if _woe.get("active"):
                            trigger_reasons.append("woe:active")
                            context["woe"] = _woe
                        elif _woe.get("recommendation") == "prepare":
                            trigger_reasons.append("woe:preparing")
                            context["woe"] = _woe
                except Exception:
                    pass
                # ── CHECK NAVIGATION INTUITION on kaizen cycle ──
                try:
                    _ni = getattr(self._runtime, "navigation_intuition", None)
                    if _ni is not None and isinstance(snapshot, dict):
                        _map = str(snapshot.get("map", "") or "")
                        _target = str(snapshot.get("target_map", "") or "")
                        if _map and _target and _map != _target:
                            _route = _ni.find_route(_map, _target)
                            if len(_route) > 3:
                                trigger_reasons.append(f"nav:long_route_{len(_route)}_maps")
                                context["route"] = _route
                            _travel_time = _ni.estimate_travel_time(_map, _target)
                            if _travel_time > 120:
                                context["travel_time_s"] = _travel_time
                except Exception:
                    pass
                # ── CHECK MECHANICAL INTUITION on kaizen cycle ──
                try:
                    _mech = getattr(self._runtime, "mechanical_intuition", None)
                    if _mech is not None and isinstance(snapshot, dict):
                        _class = str(snapshot.get("progression", {}).get("job_name", "novice") or "novice")
                        _stats = {
                            "STR": int(snapshot.get("progression", {}).get("str", 0) or 0),
                            "AGI": int(snapshot.get("progression", {}).get("agi", 0) or 0),
                            "VIT": int(snapshot.get("progression", {}).get("vit", 0) or 0),
                            "INT": int(snapshot.get("progression", {}).get("int", 0) or 0),
                            "DEX": int(snapshot.get("progression", {}).get("dex", 0) or 0),
                            "LUK": int(snapshot.get("progression", {}).get("luk", 0) or 0),
                        }
                        _evals = _mech.evaluate_stats(_class, _stats)
                        _wasted = [e for e in _evals if e.get("wasted", 0) > 0]
                        if _wasted:
                            trigger_reasons.append(f"mech:wasted_stats_{len(_wasted)}")
                            context["stat_warnings"] = _wasted[:3]
                        _next_stat = _mech.get_next_stat_recommendation(_class, _stats)
                        if _next_stat:
                            context["recommended_stat"] = _next_stat
                except Exception:
                    pass
                # ── CHECK OPPORTUNITY COST on kaizen cycle ──
                try:
                    _oc = getattr(self._runtime, "opportunity_cost", None)
                    if _oc is not None and isinstance(snapshot, dict):
                        _weight = float(snapshot.get("inventory", {}).get("weight_ratio", 0.0) or 0.0)
                        _pots = int(snapshot.get("inventory", {}).get("potion_count", 0) or 0)
                        _return = _oc.should_return_to_town(bot_id, _weight, _pots)
                        if _return.get("should_return"):
                            trigger_reasons.append(f"uptime:{_return['reason']}")
                            context["return_advice"] = _return
                        _uptime = _oc.get_uptime(bot_id)
                        if _uptime.get("uptime_pct", 100) < 50:
                            trigger_reasons.append("uptime:low_efficiency")
                            context["uptime"] = _uptime
                except Exception:
                    pass
                # ── CHECK META PREDICTION on kaizen cycle ──
                try:
                    _mp = getattr(self._runtime, "meta_prediction", None)
                    if _mp is not None and isinstance(snapshot, dict):
                        _class = str(snapshot.get("progression", {}).get("job_name", "novice") or "novice")
                        _meta = _mp.predict_meta_shift()
                        if _meta.get("meta") == "woe_season":
                            trigger_reasons.append("meta:woe_season")
                            context["meta"] = _meta
                        _build_rec = _mp.get_build_recommendation(_class, 1)
                        if _build_rec.get("is_avoided"):
                            trigger_reasons.append(f"meta:{_class}_off_meta")
                            context["meta_advice"] = _build_rec
                except Exception:
                    pass
        except Exception:
            pass
        
        # Make decision
        should_activate = len(trigger_reasons) > 0
        primary_reason = trigger_reasons[0] if trigger_reasons else "conservative:no_triggers"
        
        return should_activate, primary_reason, {
            "triggers": trigger_reasons,
            **context,
        }

    def _snapshot_reconnect_age_s(self, snapshot: BotStateSnapshot | dict[str, object]) -> float | None:
        if isinstance(snapshot, dict):
            raw = snapshot.get("raw", {})
            if not isinstance(raw, dict):
                raw = {}
        else:
            raw = getattr(snapshot, "raw", {})
            if not isinstance(raw, dict):
                return None
        for key in ("reconnect_age_s", "disconnect_age_s", "offline_age_s"):
            value = raw.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    continue
        return None

    def _overweight_ratio(self, snapshot: BotStateSnapshot | dict[str, object]) -> float:
        if isinstance(snapshot, dict):
            inv = snapshot.get("inventory", {}) or {}
            weight = int(inv.get("weight", 0) or 0)
            weight_max = int(inv.get("weight_max", 1) or 1)
        else:
            vitals = getattr(snapshot, "vitals", None)
            weight = getattr(vitals, "weight", None)
            weight_max = getattr(vitals, "weight_max", None)
        if not isinstance(weight, int) or not isinstance(weight_max, int) or weight_max <= 0:
            return 0.0
        return max(0.0, min(2.0, float(weight) / float(weight_max)))

    def _context_overrides(self, snapshot: BotStateSnapshot | None) -> dict[str, object]:
        if snapshot is None:
            return {}
        result: dict[str, object] = {}
        # Handle both BotStateSnapshot objects and plain dicts
        if isinstance(snapshot, dict):
            result["map"] = snapshot.get("map", None)
            result["tick_id"] = snapshot.get("tick_id", None)
            _bot_id = str(snapshot.get("bot_id", snapshot.get("id", "")) or "")
        else:
            result["map"] = getattr(getattr(snapshot, "position", None), "map", None)
            result["tick_id"] = getattr(snapshot, "tick_id", None)
            _bot_id = getattr(snapshot, "bot_id", None) or getattr(snapshot, "id", None) or ""
        # Inject trigger context if available — PER BOT
        _trigger_store = getattr(self, "_last_trigger_context", None)
        if _trigger_store is not None and isinstance(_trigger_store, dict):
            _bot_trigger = _trigger_store.get(_bot_id) if isinstance(_trigger_store, dict) else None
            if _bot_trigger is not None and isinstance(_bot_trigger, dict):
                _age = time.time() - float(_bot_trigger.get("timestamp", 0))
                if _age < 60.0:  # Only inject triggers less than 60s old
                    result["trigger_reason"] = str(_bot_trigger.get("reason", ""))
                    result["trigger_context"] = dict(_bot_trigger.get("context", {}))
        return result
