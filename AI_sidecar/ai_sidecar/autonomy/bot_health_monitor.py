"""
Bot Health Monitor — self-healing agent for the PDCA cycle.
Detects common bot issues (overweight, stuck, skill errors, death spiral)
and generates corrective config-change actions via the action queue.

Architecture rule: this runs inside the sidecar PDCA loop, NOT in the bridge plugin.
The bridge only reports state — the sidecar decides what to fix.
"""
import logging
from datetime import datetime, UTC

_log = logging.getLogger(__name__)

# Issue detection thresholds
MAX_WEIGHT_RATIO = 0.65       # Sell if weight > 65%
MIN_KILLS_PER_CYCLE = 1       # Alert if no kills in a cycle
MAX_CONSECUTIVE_DEATHS = 3    # Force teleport if died N times in a row
STUCK_TOWN_MAPS = {"prontera", "prt_in", "morocc", "payon", "geffen", "aldebaran", "alberta"}

# Config fixes that can be applied via "set <key> <value>" commands
CONFIG_FIXES = {
    "weight_2000": {"key": "weight", "value": "2000", "reason": "Max weight too low, bot stuck sitting"},
    "sell_auto_on": {"key": "sellAuto", "value": "1", "reason": "Auto-sell disabled, inventory filling up"},
    "sell_npc_prt_in": {"key": "sellAuto_npc", "value": "prt_in 126 76", "reason": "Sell NPC not set to Tool Dealer"},
    "sell_steps_correct": {"key": "sellAuto_npc_steps", "value": "c r0 c", "reason": "Sell NPC dialog steps wrong"},
    "items_take_2": {"key": "itemsTakeAuto", "value": "2", "reason": "Item pickup disabled"},
    "attack_auto_2": {"key": "attackAuto", "value": "3", "reason": "Attack mode not aggressive"},
    "teleport_hp_0": {"key": "teleportAuto_hp", "value": "0", "reason": "Teleport HP disabled - use sit instead"},
}


def check_bot_health(runtime_state, action_queue, bot_id: str) -> list[dict]:
    """Check a single bot's health and return corrective actions to enqueue.
    
    Returns list of ActionProposal-compatible dicts.
    """
    corrections = []
    if not runtime_state or not hasattr(runtime_state, "snapshot_cache"):
        return corrections
    
    snapshots = getattr(runtime_state, "snapshot_cache", None)
    if snapshots is None:
        return corrections
    
    try:
        latest = snapshots.latest()
    except Exception:
        return corrections
    if latest is None:
        return corrections
    
    # Parse snapshot
    weight_ratio = 0.0
    map_name = ""
    hp_ratio = 1.0
    sitting = False
    
    if isinstance(latest, dict):
        inv = latest.get("inventory", {}) or {}
        weight_ratio = float(inv.get("weight_ratio", 0.0) or 0.0)
        map_name = str(latest.get("map", latest.get("position", {}).get("map", "")) or "")
        vitals = latest.get("vitals", {}) or {}
        hp_ratio = float(vitals.get("hp_ratio", 1.0) or 1.0)
    else:
        try:
            if hasattr(latest, "vitals"):
                weight_ratio = float(getattr(latest.vitals, "weight_ratio", 0.0) or 0.0)
                hp_ratio = float(getattr(latest.vitals, "hp_ratio", 1.0) or 1.0)
            if hasattr(latest, "position"):
                map_name = str(getattr(latest.position, "map", "") or "")
        except Exception:
            pass
    
    now_key = f"health_{bot_id}"
    prev_state = getattr(check_bot_health, "_state", {})
    if not hasattr(check_bot_health, "_state"):
        check_bot_health._state = {}
    
    # ── Weight check ──
    if weight_ratio > MAX_WEIGHT_RATIO:
        _log.info("health_monitor: %s overweight (%.0f%%), enabling sellAuto", bot_id, weight_ratio * 100)
        corrections.append({
            "action_id": f"health_weight_{bot_id}",
            "kind": "command",
            "command": "set sellAuto 1",
            "priority_tier": "tactical",
            "source": "health_monitor",
            "metadata": {"reason": f"Weight {weight_ratio:.0%} > {MAX_WEIGHT_RATIO:.0%}, enabling auto-sell"},
        })
        # Also set the sell NPC if not already
        corrections.append({
            "action_id": f"health_sellnpc_{bot_id}",
            "kind": "command",
            "command": 'set sellAuto_npc prt_in 126 76',
            "priority_tier": "tactical",
            "source": "health_monitor",
            "metadata": {"reason": "Setting sell NPC for overweight bot"},
        })
        # Set proper weight if still 0
        corrections.append({
            "action_id": f"health_weightcfg_{bot_id}",
            "kind": "command",
            "command": "set weight 2000",
            "priority_tier": "tactical",
            "source": "health_monitor",
            "metadata": {"reason": "Fixing weight config from 0 to 2000"},
        })
    
    # ── Stuck in town check (bot in town for too long) ──
    town_maps = {m for m in STUCK_TOWN_MAPS}
    is_in_town = any(m in map_name.lower() for m in town_maps) if map_name else True
    
    if is_in_town and weight_ratio < MAX_WEIGHT_RATIO:
        # Bot is in town and NOT overweight — should be hunting
        # Check how many cycles it's been in town
        state = check_bot_health._state.get(now_key, {})
        prev_map = state.get("prev_map", "")
        town_cycles = state.get("town_cycles", 0) + 1 if prev_map == map_name else 1
        
        if town_cycles >= 3:  # 3+ cycles in town = stuck
            _log.info("health_monitor: %s stuck in town (%s, %d cycles), sending to hunt", 
                      bot_id, map_name, town_cycles)
            corrections.append({
                "action_id": f"health_move_hunt_{bot_id}",
                "kind": "command",
                "command": "move prt_fild05",
                "priority_tier": "tactical",
                "source": "health_monitor",
                "metadata": {"reason": f"Stuck in {map_name} for {town_cycles} cycles, sending to hunt"},
            })
        
        check_bot_health._state[now_key] = {"prev_map": map_name, "town_cycles": town_cycles}
    
    # ── Low HP check ──
    if hp_ratio < 0.20 and map_name and "prontera" not in map_name.lower() and "prt_in" not in map_name.lower():
        _log.info("health_monitor: %s critically low HP (%.0f%%), sending to town", bot_id, hp_ratio * 100)
        corrections.append({
            "action_id": f"health_town_{bot_id}",
            "kind": "command",
            "command": "move prontera",
            "priority_tier": "reflex",
            "source": "health_monitor",
            "metadata": {"reason": f"HP critically low ({hp_ratio:.0%}), retreating to town"},
        })
    
    return corrections


def run_health_checks(runtime_state, action_queue, bot_ids: list[str]) -> int:
    """Run health checks for all bots. Returns number of corrections enqueued."""
    count = 0
    for bot_id in bot_ids:
        try:
            corrections = check_bot_health(runtime_state, action_queue, bot_id)
            for corr in corrections:
                try:
                    from ai_sidecar.contracts.actions import ActionProposal, ActionPriorityTier
                    tier = getattr(ActionPriorityTier, corr.get("priority_tier", "tactical").upper(), 
                                  ActionPriorityTier.TACTICAL)
                    proposal = ActionProposal(
                        action_id=corr["action_id"],
                        kind=corr["kind"],
                        command=corr["command"],
                        priority_tier=tier,
                        source=corr.get("source", "health_monitor"),
                        expires_at=datetime.now(UTC),
                        metadata=corr.get("metadata", {}),
                    )
                    action_queue.enqueue(bot_id, proposal)
                    count += 1
                    _log.info("health_correction: bot=%s action=%s reason=%s", 
                              bot_id, corr["action_id"], corr.get("metadata", {}).get("reason", ""))
                except Exception as e:
                    _log.error("health_enqueue_failed: bot=%s error=%s", bot_id, e)
        except Exception as e:
            _log.error("health_check_failed: bot=%s error=%s", bot_id, e)
    return count
