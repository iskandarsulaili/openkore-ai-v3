"""Action Filter — reduces 72+ domain actions to the top meaningful commands.

95% of the 72 actions per tick are internal logging (kind="log").
Only 3-5 are real commands the bridge should execute.

This filter:
1. Drops all log actions (they're internal tracking)
2. Deduplicates by command text (last confidence wins)
3. Sorts by confidence (highest first)
4. Returns top N non-log, non-internal actions
"""
from __future__ import annotations
from typing import Any
import logging

from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)

# Action types that should NEVER go to the bridge
INTERNAL_ACTIONS = {"log", "debug", "trace", "metric"}

# Commands that are internal tracking (bridge doesn't need these)
INTERNAL_COMMAND_PREFIXES = (
    "goal=", "lifecycle_phase=", "quests_near_complete=", "equipment=",
    "cold_start", "inventory_policy", "world_state", "farming_cycle",
    "resource_pool", "loadout", "durability", "danger_multiplier",
    "idle", "market_shops", "social", "mvp_available", "woe_active",
    "event_active", "spawn_rotation",
)

# Commands that ARE real bot instructions
REAL_COMMAND_PREFIXES = (
    "attack", "move", "sit", "stand", "warp", "flywing", "butterfly",
    "retreat", "teleport", "airship", "kafra", "store", "buy", "sell",
    "navigate", "mon_control", "attackAuto", "itemsTakeAuto",
    "equip", "slot", "upgrade", "repair", "use_item", "use_skill",
    "skill_cast", "party", "combo", "loot", "talknpc", "quest",
    "team_", "reply", "transfer_zeny",
)


def filter_actions(actions: list[HeuristicAction], max_commands: int = 5) -> list[HeuristicAction]:
    """Filter domain actions to only the most meaningful bridge commands.
    
    Args:
        actions: All actions from domain modules
        max_commands: Max non-log actions to return
    
    Returns:
        Filtered actions: log actions + top N real commands
    """
    if not actions:
        return []
    
    logs = []
    commands = []
    
    for action in actions:
        if not action:
            continue
        
        # Separate log actions from commands
        if action.kind in INTERNAL_ACTIONS:
            logs.append(action)
            continue
        
        if action.command and any(action.command.startswith(p) for p in INTERNAL_COMMAND_PREFIXES):
            logs.append(action)
            continue
        
        if action.kind == "command" and action.command:
            commands.append(action)
        else:
            logs.append(action)
    
    # Deduplicate commands by command text (keep highest confidence)
    seen: dict[str, HeuristicAction] = {}
    for cmd in commands:
        text = cmd.command
        if text in seen:
            if cmd.confidence > seen[text].confidence:
                seen[text] = cmd
        else:
            seen[text] = cmd
    
    # Sort by confidence descending
    unique_commands = sorted(seen.values(), key=lambda a: -a.confidence)
    
    # Return top N commands + ALL logs (logs are for internal tracking)
    top_commands = unique_commands[:max_commands]
    
    # Log what we're sending
    if top_commands:
        logger.info(f"Bridge commands ({len(top_commands)}): {[c.command[:50] for c in top_commands]}")
    
    return logs + top_commands


class BridgeActionLogger:
    """Logs every action sent to the bridge for verification.
    
    This is how we PROVE the AI decisions are reaching OpenKore.
    Each action is logged with a unique ID so we can trace it end-to-end.
    """
    
    def __init__(self):
        self._action_log: list[dict] = []
        self._max_log = 1000
    
    def log_action(self, action: HeuristicAction, source: str = "domain") -> None:
        """Log an action being sent to the bridge."""
        entry = {
            "id": len(self._action_log),
            "kind": action.kind,
            "command": action.command[:80] if action.command else "",
            "confidence": action.confidence,
            "domain": action.domain,
            "source": source,
            "ts": __import__("time").time(),
        }
        self._action_log.append(entry)
        if len(self._action_log) > self._max_log:
            self._action_log = self._action_log[-self._max_log:]
        
        logger.debug(f"BRIDGE_ACTION [{entry['id']}] {action.command[:60]} (conf={action.confidence:.2f})")
    
    def get_recent_actions(self, n: int = 20) -> list[dict]:
        return self._action_log[-n:]
    
    def get_stats(self) -> dict:
        if not self._action_log:
            return {"total": 0, "by_domain": {}}
        by_domain: dict[str, int] = {}
        for e in self._action_log:
            d = e["domain"]
            by_domain[d] = by_domain.get(d, 0) + 1
        return {
            "total": len(self._action_log),
            "by_domain": dict(sorted(by_domain.items(), key=lambda x: -x[1])[:10]),
            "last_5": [e["command"] for e in self._action_log[-5:]],
        }


# Singleton
_filter_logger: BridgeActionLogger | None = None

def get_filter_logger() -> BridgeActionLogger:
    global _filter_logger
    if _filter_logger is None:
        _filter_logger = BridgeActionLogger()
    return _filter_logger
