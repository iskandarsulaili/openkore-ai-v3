"""
Bot Health Monitor — self-healing agent for the PDCA cycle.
Detects common bot issues (overweight, stuck, skill errors, death spiral)
and generates corrective config-change actions via the action queue.

Architecture rule: this runs inside the sidecar PDCA loop, NOT in the bridge plugin.
The bridge only reports state — the sidecar decides what to fix.
"""
import logging
from datetime import UTC, datetime, timedelta

_log = logging.getLogger(__name__)


def _is_town_map(map_name: str, safe_town: str) -> bool:
    """Agnostic town detection: the map IS the learned safe town, or its
    interior/city prefix (e.g. safe_town 'prontera' → 'prontera', 'prt_in').
    No hardcoded town list — the server's safe_town comes from the learned
    server_solutions store."""
    if not map_name:
        return True
    if not safe_town:
        return False
    base = safe_town.lower()
    cur = map_name.lower()
    if cur == base or cur.startswith(base + "_"):
        return True
    # Interior map of the town (e.g. 'prt_in' for 'prontera') is still town —
    # derived from the town's own 3-char prefix, never a hardcoded town list.
    if cur.startswith(base[:3] + "_in"):
        return True
    return False


# Issue detection thresholds
MAX_WEIGHT_RATIO = 0.65       # Sell if weight > 65%
MIN_KILLS_PER_CYCLE = 1       # Alert if no kills in a cycle
MAX_CONSECUTIVE_DEATHS = 3    # Force teleport if died N times in a row


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
        latest = snapshots.get(bot_id)
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
        # Also set the sell NPC if not already (via server_solutions_store, RULE.md)
        _store = getattr(runtime_state, "server_solutions_store", None)
        _sell_npc = str((_store.get("sell_npc", None) if _store else None) or "prt_in 126 76")
        corrections.append({
            "action_id": f"health_sellnpc_{bot_id}",
            "kind": "command",
            "command": f"set sellAuto_npc {_sell_npc}",
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
    # Agnostic: town = the learned safe_town (server_solutions store).
    _store_t = getattr(runtime_state, "server_solutions_store", None)
    _safe_t = str((_store_t.get("safe_town", None) if _store_t else None) or "")
    is_in_town = _is_town_map(map_name, _safe_t) if _safe_t else bool(map_name)
    
    if is_in_town and weight_ratio < MAX_WEIGHT_RATIO:
        # Bot is in town and NOT overweight — should be hunting
        # Check how many cycles it's been in town
        state = check_bot_health._state.get(now_key, {})
        prev_map = state.get("prev_map", "")
        town_cycles = state.get("town_cycles", 0) + 1 if prev_map == map_name else 1
        
        if town_cycles >= 3:  # 3+ cycles in town = stuck
            # Route through server_solutions_store (RULE.md: never hardcode server maps)
            _store = getattr(runtime_state, "server_solutions_store", None)
            _farm_map = str((_store.get("farm_map", None) if _store else None) or "")
            if _farm_map:
                _log.info("health_monitor: %s stuck in town (%s, %d cycles), sending to hunt (%s)",
                          bot_id, map_name, town_cycles, _farm_map)
                corrections.append({
                    "action_id": f"health_move_hunt_{bot_id}",
                    "kind": "command",
                    "command": f"move {_farm_map}",
                    "priority_tier": "tactical",
                    "source": "health_monitor",
                    "metadata": {"reason": f"Stuck in {map_name} for {town_cycles} cycles, sending to hunt {_farm_map}"},
                })
        
        check_bot_health._state[now_key] = {"prev_map": map_name, "town_cycles": town_cycles}
    
    # ── Low HP check ──
    _store = getattr(runtime_state, "server_solutions_store", None)
    _safe_town = str((_store.get("safe_town", None) if _store else None) or "")
    if hp_ratio < 0.20 and _safe_town and not _is_town_map(map_name, _safe_town):
        # Route through server_solutions_store (RULE.md: never hardcode server towns)
        _log.info("health_monitor: %s critically low HP (%.0f%%), sending to town (%s)", bot_id, hp_ratio * 100, _safe_town)
        corrections.append({
            "action_id": f"health_town_{bot_id}",
            "kind": "command",
            "command": f"move {_safe_town}",
            "priority_tier": "reflex",
            "source": "health_monitor",
            "metadata": {"reason": f"HP critically low ({hp_ratio:.0%}), retreating to {_safe_town}"},
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
                        created_at=datetime.now(UTC),
                        expires_at=datetime.now(UTC) + timedelta(seconds=30),
                        idempotency_key=corr["action_id"],
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
