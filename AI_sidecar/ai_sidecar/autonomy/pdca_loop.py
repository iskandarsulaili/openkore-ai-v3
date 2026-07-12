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
                latest = snapshots.latest()
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
            _log.info("game_engine_no_zones: bot=%s level=%d - using fallback", bot_id, bot_level)
            # Fallback: use fallback zones for this level
            zones = hzm._fallback_zones(bot_level) if hasattr(hzm, '_fallback_zones') else []
            if not zones:
                _log.info("game_engine_no_zones_fallback_empty: bot=%s level=%d", bot_id, bot_level)
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
            priority_tier=ActionPriorityTier.strategic,
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


def _emit_skill_actions(runtime_state, horizon: str, bot_id: str | None = None) -> int:
    """Emit skill rotation actions based on game engine recommendations.
    
    Uses game engine's recommend_skills_for_mob() to determine optimal
    skills against current targets. Replaces empty attackSkillSlot blocks.
    """
    import logging
    _log = logging.getLogger(__name__)
    try:
        game_engine = getattr(runtime_state, "game_engine", None)
        if game_engine is None:
            return 0
        
        # Get snapshot for current mob info
        snapshots = getattr(runtime_state, "snapshot_cache", None)
        if snapshots is None:
            return 0
        try:
            latest = snapshots.latest()
        except Exception:
            return 0
        if latest is None:
            return 0
        
        # Get current target info
        mob_element = "Neutral"
        mob_race = "Formless"
        mob_size = "Medium"
        bot_class = "novice"
        if isinstance(latest, dict):
            target = latest.get("target", {}) or {}
            mob_element = str(target.get("element", "Neutral") or "Neutral")
            mob_race = str(target.get("race", "Formless") or "Formless")
            mob_size = str(target.get("size", "Medium") or "Medium")
            bot_class = str(latest.get("job_name", latest.get("class", "novice")) or "novice")
        else:
            target = getattr(latest, "target", None) or {}
            mob_element = str(getattr(target, "element", "Neutral") or "Neutral")
            mob_race = str(getattr(target, "race", "Formless") or "Formless")
            mob_size = str(getattr(target, "size", "Medium") or "Medium")
            bot_class = str(getattr(latest, "job_name", "novice") or "novice")
        
        # Get skill recommendation
        skills = game_engine.recommend_skills_for_mob(
            job_name=bot_class,
            mob_element=mob_element,
            mob_race=mob_race,
            mob_size=mob_size,
        )
        
        if not skills:
            return 0
        
        # Resolve bot_id
        if not bot_id:
            if isinstance(latest, dict) and latest.get("bot_id"):
                bot_id = str(latest["bot_id"])
        bot_id = bot_id or "default"
        
        best = skills[0]
        recommended_element = best.get("recommended_element", "Neutral")
        
        # Map element to actual skill command based on class
        # Element converters: every class can use element via converters/endow
        # Magic classes: direct element spells
        # Physical classes: weapon element via converters (ss endow, etc.)
        element_to_skill = {
            "Holy": "ss heal",  # Heal vs Undead, or aspersio
            "Fire": "ss fire_bolt",
            "Water": "ss cold_bolt",
            "Wind": "ss lightning_bolt",
            "Earth": "ss stone_curse",
            "Poison": "ss venom_dust",
            "Ghost": "ss magnus",
            "Undead": "ss turn_undead",
            "Dark": "ss grimtooth",
        }
        
        # Class-specific skill mapping
        class_to_skill = {
            "mage": "ss fire_bolt",
            "wizard": "ss fire_bolt",
            "high_wizard": "ss fire_bolt",
            "arch_mage": "ss fire_bolt",
            "acolyte": "ss heal",
            "priest": "ss heal",
            "high_priest": "ss heal",
            "arch_bishop": "ss heal",
            "cardinal": "ss heal",
            "swordman": "ss magnum_break",
            "knight": "ss bowling_bash",
            "lord_knight": "ss bowling_bash",
            "rune_knight": "ss bowling_bash",
            "dragon_knight": "ss bowling_bash",
            "thief": "ss double_attack",
            "assassin": "ss sonic_blow",
            "assassin_cross": "ss sonic_blow",
            "guillotine_cross": "ss sonic_blow",
            "shadow_cross": "ss sonic_blow",
            "archer": "ss double_strafing",
            "hunter": "ss double_strafing",
            "sniper": "ss double_strafing",
            "ranger": "ss aimed_bolt",
            "windhawk": "ss aimed_bolt",
            "merchant": "ss mammonite",
            "blacksmith": "ss mammonite",
            "whitesmith": "ss mammonite",
            "meister": "ss mammonite",
            "alchemist": "ss acid_demonstration",
            "creator": "ss acid_demonstration",
            "genetic": "ss cart_cannon",
            "biolo": "ss cart_cannon",
        }
        
        # Determine best skill command
        skill_cmd = None
        bot_class_lower = bot_class.lower().replace(" ", "_").replace("-", "_")
        
        # Try element-based skill first (for magic classes)
        if bot_class_lower in ("mage", "wizard", "high_wizard", "arch_mage", "soul_linker", "soul_reaper", "soul_ascetic", "sage", "professor", "sorcerer", "elemental_master"):
            if recommended_element in element_to_skill:
                skill_cmd = element_to_skill[recommended_element]
        
        # Fall back to class default skill
        if not skill_cmd:
            for key, cmd in class_to_skill.items():
                if key in bot_class_lower:
                    skill_cmd = cmd
                    break
        
        # Final fallback: ai auto
        if not skill_cmd:
            skill_cmd = "ai auto"
        
        aq = getattr(runtime_state, "action_queue", None)
        if aq is None:
            return 0
        
        from datetime import UTC, datetime, timedelta
        from ai_sidecar.contracts.actions import ActionProposal, ActionPriorityTier
        import hashlib as _hashlib
        _short_id = _hashlib.md5(f"{bot_id}_skill_{horizon}_{time.time()}".encode()).hexdigest()[:16]
        proposal = ActionProposal(
            action_id=f"skill_{horizon}_{_short_id}",
            kind="command",
            command=skill_cmd,
            priority_tier=ActionPriorityTier.tactical,
            source="planner",
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(seconds=60),
            idempotency_key=f"skill_{horizon}_{_short_id}",
            metadata={
                "goal": "combat",
                "objective": f"Use {skill_cmd} vs {mob_element} ({best['damage_multiplier']:.0%})",
                "horizon": horizon, "bot_id": bot_id, "source": "skill_ai",
                "recommended_element": recommended_element,
                "damage_multiplier": best["damage_multiplier"],
                "mob_element": mob_element,
            },
        )
        aq.enqueue(bot_id, proposal)
        _log.info(
            "skill_action: bot=%s cmd=%s element=%s mult=%.0f%% vs %s",
            bot_id, skill_cmd, recommended_element, best["damage_multiplier"] * 100, mob_element,
        )
        return 1
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
                
                # Initialize hunting zone manager if not present
                _hzm = getattr(self._runtime, "hunting_zone_manager", None)
                if _hzm is None:
                    _hzm = HuntingZoneManager(
                        getattr(_settings, "game_engine_knowledge_path", "knowledge/knowledge.json")
                    )
                    self._runtime.hunting_zone_manager = _hzm
                
                # Initialize anti-detection if not present
                _ad = getattr(self._runtime, "anti_detection", None)
                if _ad is None:
                    _ad = AntiDetection(enabled=getattr(_settings, "anti_detection_enabled", True))
                    self._runtime.anti_detection = _ad
                
                # Initialize game engine if not present
                _ge = getattr(self._runtime, "game_engine", None)
                if _ge is None:
                    try:
                        _ge = GameIntelligenceEngine(
                            getattr(_settings, "game_engine_knowledge_path", "knowledge/knowledge.json")
                        )
                        self._runtime.game_engine = _ge
                    except Exception:
                        pass
                
                # Initialize swarm tactics if not present
                _swarm = getattr(self._runtime, "swarm_tactics", None)
                if _swarm is None:
                    try:
                        _swarm = SwarmTacticsEngine()
                        self._runtime.swarm_tactics = _swarm
                    except Exception:
                        pass
                
                # Initialize role discovery if not present
                _role_disc = getattr(self._runtime, "role_discovery", None)
                if _role_disc is None:
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
                
                # Check if this is a novel situation (first time seeing this map/state)
                _is_novel = False
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
                _use_llm = _cost_mode.should_use_llm(
                    horizon=horizon.value,
                    heuristic_confidence=_hc,
                    bot_id=_cycle_bot_id,
                    is_novel_situation=_is_novel,
                )
                
                # Update budget limits from cost mode
                if _ct is not None:
                    _daily_budget = _cost_mode.get_daily_budget_tokens()
                    _hourly_limit = _cost_mode.get_llm_calls_per_hour_limit()
                    _allowed, _reason = _ct.check(
                        daily_budget_tokens=_daily_budget,
                        max_calls_per_hour=_hourly_limit,
                        tier=_tier, bot_id=_cycle_bot_id,
                    )
                    if not _allowed:
                        logger.info("cost_gate[%s]: %s", horizon.value, _reason)
                        # Even when budget exceeded, emit game engine actions
                        _actions_queued_budget = _emit_game_engine_actions(
                            self._runtime, horizon.value, bot_id=_cycle_bot_id, map_name=_map_name
                        )
                        return PDCAResult(horizon=horizon, plan_id="", actions_queued=_actions_queued_budget, progress_pct=0.0, stuck=False, re_planned=False,
                                          force_replan=False, selected_goal="budget_gated", objective=f"Budget exceeded: {_reason}",
                                          replan_reasons=[_reason], cycle_ms=0.0, error="")
                
                if not _use_llm:
                    # Emit game engine + heuristic + swarm + vendor + skill actions
                    # Emit for ALL registered bots, not just the resolved one
                    _all_bot_ids: list[str] = []
                    try:
                        _br = getattr(self._runtime, "bot_registry", None)
                        if _br is not None:
                            _all_bot_ids = [str(b.get("bot_id","")) for b in _br.list_bots() if isinstance(b, dict) and b.get("bot_id")]
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
                _all_bot_ids = [str(b.get("bot_id","")) for b in _br.list_bots() if isinstance(b, dict) and b.get("bot_id")]
        except Exception:
            pass
        if not _all_bot_ids:
            _all_bot_ids = [_cycle_bot_id]
        # Read map_name from snapshot for game engine routing
        _fb_map_name = ""
        try:
            _fb_snap = self._get_latest_snapshot()
            if _fb_snap is not None:
                if isinstance(_fb_snap, dict):
                    _fb_map_name = str(_fb_snap.get("map", _fb_snap.get("position", {}).get("map", "")) or "")
                else:
                    _fb_map_name = str(getattr(getattr(_fb_snap, "position", None), "map", "") or "")
        except Exception:
            pass
        _fallback_total = 0
        for _bid in _all_bot_ids:
            _fallback_ge = _emit_game_engine_actions(
                self._runtime, horizon.value, bot_id=_bid, map_name=_fb_map_name
            )
            _fallback_hs = _emit_heuristic_actions(self._runtime, horizon.value, bot_id=_bid)
            _fallback_swarm = _emit_swarm_actions(self._runtime, horizon.value, bot_id=_bid)
            _fallback_vendor = _emit_vendor_actions(self._runtime, horizon.value, bot_id=_bid)
            _fallback_skill = _emit_skill_actions(self._runtime, horizon.value, bot_id=_bid)
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

    def _resolve_bot_id(self, snapshot: BotStateSnapshot | None = None) -> str:
        if snapshot is not None and getattr(snapshot, "meta", None) is not None:
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

    def _objective_for(self, *, horizon: Horizon, snapshot: BotStateSnapshot | None) -> str:
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

    def _snapshot_disconnected(self, snapshot: BotStateSnapshot) -> bool:
        raw = getattr(snapshot, "raw", {})
        if not isinstance(raw, dict):
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

    def _snapshot_reconnect_age_s(self, snapshot: BotStateSnapshot) -> float | None:
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

    def _overweight_ratio(self, snapshot: BotStateSnapshot) -> float:
        vitals = getattr(snapshot, "vitals", None)
        weight = getattr(vitals, "weight", None)
        weight_max = getattr(vitals, "weight_max", None)
        if not isinstance(weight, int) or not isinstance(weight_max, int) or weight_max <= 0:
            return 0.0
        return max(0.0, min(2.0, float(weight) / float(weight_max)))

    def _context_overrides(self, snapshot: BotStateSnapshot | None) -> dict[str, object]:
        if snapshot is None:
            return {}
        return {
            "map": getattr(getattr(snapshot, "position", None), "map", None),
            "tick_id": getattr(snapshot, "tick_id", None),
        }
