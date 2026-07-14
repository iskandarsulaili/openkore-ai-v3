"""
Macro Intelligence Engine — AI-driven macro knowledge and generation.
====================================================================
Instead of hardcoding OpenKore macros, this system:
1. Stores KNOWLEDGE about community macro patterns (triggers, actions, sequences)
2. The LLM (CrewAI agents + planner) decides WHEN to generate a macro
3. Generates macro sequences dynamically based on bot state and goals
4. Emits macros through the action pipeline (not as Perl macros.txt files)

The AI treats macros as ACTION SEQUENCES — ordered lists of commands that
accomplish a specific goal (buy/sell, refine, complete quest, etc.).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1. MACRO KNOWLEDGE — patterns extracted from community OpenKore macros
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MacroAction:
    """A single step in a macro sequence."""
    command: str
    description: str = ""
    timeout_seconds: float = 10.0
    requires_confirmation: bool = False  # NPC dialog response, etc.
    conflict_key: str = ""


@dataclass
class MacroPattern:
    """A known macro pattern with triggers and action sequences.
    
    The AI uses these patterns as TEMPLATES — it fills in the actual
    values (item names, NPC locations, map names) based on current state.
    """
    pattern_id: str
    category: str  # survival, economy, combat, navigation, quest, social
    description: str
    triggers: list[dict[str, Any]]  # Conditions that suggest this pattern
    action_sequence: list[MacroAction]
    required_items: list[str] = field(default_factory=list)
    required_zeny: int = 0
    required_level_range: tuple[int, int] = (1, 999)
    cooldown_seconds: float = 0.0
    exclusive: bool = False  # Can run alongside other macros


# ── Community-derived macro patterns ──────────────────────────────────
# These are extracted from OpenKore macro repositories, wiki, and forums.
# The AI uses them as KNOWLEDGE — not as hardcoded scripts.

MACRO_PATTERNS: dict[str, MacroPattern] = {
    # ── SURVIVAL ──
    "auto_potion_hp": MacroPattern(
        pattern_id="auto_potion_hp", category="survival",
        description="Auto-use HP potion when HP drops below threshold. Most common OpenKore macro.",
        triggers=[{"type": "vitals.hp_ratio", "op": "lte", "value": 0.50}],
        action_sequence=[MacroAction(command="use red_potion", description="Use HP potion", timeout_seconds=2.0)],
        cooldown_seconds=2.0,
    ),
    "auto_potion_sp": MacroPattern(
        pattern_id="auto_potion_sp", category="survival",
        description="Auto-use SP potion when SP drops below threshold.",
        triggers=[{"type": "vitals.sp_ratio", "op": "lte", "value": 0.30}],
        action_sequence=[MacroAction(command="use blue_potion", description="Use SP potion", timeout_seconds=2.0)],
        cooldown_seconds=3.0,
    ),
    "emergency_teleport": MacroPattern(
        pattern_id="emergency_teleport", category="survival",
        description="Teleport to savepoint when HP critically low. Classic escape macro.",
        triggers=[{"type": "vitals.hp_ratio", "op": "lte", "value": 0.15}],
        action_sequence=[MacroAction(command="ai manual", description="Disable AI"),
                         MacroAction(command="tele", description="Teleport to savepoint", timeout_seconds=5.0),
                         MacroAction(command="ai auto", description="Re-enable AI")],
        cooldown_seconds=10.0, exclusive=True,
    ),
    "death_respawn_return": MacroPattern(
        pattern_id="death_respawn_return", category="survival",
        description="After death, respawn, rebuff, and return to hunting zone.",
        triggers=[{"type": "event.event_type", "op": "eq", "value": "player_died"}],
        action_sequence=[MacroAction(command="ai auto", description="Respawn with auto AI"),
                         MacroAction(command="move prt_fild08", description="Return to hunting zone", timeout_seconds=30.0)],
        cooldown_seconds=15.0, exclusive=True,
    ),
    "auto_buff_on_respawn": MacroPattern(
        pattern_id="auto_buff_on_respawn", category="survival",
        description="Re-apply buffs (bless, agi up) after death or teleport.",
        triggers=[{"type": "event.event_type", "op": "eq", "value": "player_respawned"}],
        action_sequence=[MacroAction(command="ss increase_agility", description="Cast Agi Up"),
                         MacroAction(command="ss bless", description="Cast Bless", timeout_seconds=3.0)],
        cooldown_seconds=30.0,
    ),

    # ── ECONOMY ──
    "auto_sell_junk": MacroPattern(
        pattern_id="auto_sell_junk", category="economy",
        description="Sell junk items to NPC when inventory is full. Common farming macro.",
        triggers=[{"type": "inventory.overweight_ratio", "op": "gte", "value": 0.85}],
        action_sequence=[MacroAction(command="move prontera", description="Return to town", timeout_seconds=30.0),
                         MacroAction(command="talknpc 181 186", description="Talk to tool dealer", timeout_seconds=5.0),
                         MacroAction(command="response 0", description="Open shop menu"),
                         MacroAction(command="response 2", description="Select sell option", requires_confirmation=True)],
        required_zeny=0, cooldown_seconds=60.0,
    ),
    "auto_storage_deposit": MacroPattern(
        pattern_id="auto_storage_deposit", category="economy",
        description="Deposit loot into Kafra storage when inventory is full.",
        triggers=[{"type": "inventory.item_count", "op": "gte", "value": 90}],
        action_sequence=[MacroAction(command="move prontera", description="Return to town", timeout_seconds=30.0),
                         MacroAction(command="talknpc 146 128", description="Talk to Kafra", timeout_seconds=5.0),
                         MacroAction(command="response 0", description="Open storage")],
        cooldown_seconds=120.0,
    ),
    "auto_buy_supplies": MacroPattern(
        pattern_id="auto_buy_supplies", category="economy",
        description="Buy potions/arrows when stock is low.",
        triggers=[{"type": "inventory.item_count", "op": "lte", "value": 5}],
        action_sequence=[MacroAction(command="move prontera", description="Return to town", timeout_seconds=30.0),
                         MacroAction(command="buy 1 red_potion 30", description="Buy 30 HP pots"),
                         MacroAction(command="ai auto", description="Return to hunting")],
        required_zeny=1000, cooldown_seconds=120.0,
    ),

    # ── COMBAT ──
    "mob_swarm_escape": MacroPattern(
        pattern_id="mob_swarm_escape", category="combat",
        description="Escape when surrounded by too many monsters. Prevents death.",
        triggers=[{"type": "combat.aggro_count", "op": "gte", "value": 5}],
        action_sequence=[MacroAction(command="ai manual", description="Stop combat"),
                         MacroAction(command="move 0 0", description="Move away"),
                         MacroAction(command="ai auto", description="Resume auto")],
        cooldown_seconds=15.0,
    ),
    "skill_rotation_burst": MacroPattern(
        pattern_id="skill_rotation_burst", category="combat",
        description="Execute burst skill rotation on a single target (boss/MVP).",
        triggers=[{"type": "combat.target_is_boss", "op": "eq", "value": True}],
        action_sequence=[MacroAction(command="ss magnum_break", description="Opening burst skill"),
                         MacroAction(command="ss bowling_bash", description="Main damage skill", timeout_seconds=2.0)],
        cooldown_seconds=10.0, exclusive=True,
    ),

    # ── NAVIGATION ──
    "route_stuck_recovery": MacroPattern(
        pattern_id="route_stuck_recovery", category="navigation",
        description="Recover from a stuck route by resetting AI.",
        triggers=[{"type": "navigation.stuck_score", "op": "gte", "value": 0.85}],
        action_sequence=[MacroAction(command="ai auto", description="Reset AI and recalculate route")],
        cooldown_seconds=5.0,
    ),
    "portal_travel": MacroPattern(
        pattern_id="portal_travel", category="navigation",
        description="Use warp portal NPC to travel between towns.",
        triggers=[{"type": "goal.target_out_of_range", "op": "eq", "value": True}],
        action_sequence=[MacroAction(command="move prontera", description="Go to warp town", timeout_seconds=30.0),
                         MacroAction(command="talknpc 165 190", description="Talk to warp NPC", timeout_seconds=5.0),
                         MacroAction(command="response 0", description="Select destination")],
        cooldown_seconds=120.0, exclusive=True,
    ),

    # ── QUEST / PROGRESSION ──
    "job_change_routine": MacroPattern(
        pattern_id="job_change_routine", category="quest",
        description="Navigate to job change NPC and initiate job advancement.",
        triggers=[{"type": "progression.job_change_ready", "op": "eq", "value": True}],
        action_sequence=[MacroAction(command="move prontera", description="Go to job change town", timeout_seconds=30.0),
                         MacroAction(command="talknpc 175 130", description="Talk to job change NPC", timeout_seconds=5.0),
                         MacroAction(command="response 0", description="Start job change")],
        cooldown_seconds=300.0, exclusive=True,
    ),
    "auto_stat_alloc": MacroPattern(
        pattern_id="auto_stat_alloc", category="quest",
        description="Allocate stat points when leveling up. Follows class archetype priorities.",
        triggers=[{"type": "progression.stat_points_available", "op": "gt", "value": 0}],
        action_sequence=[MacroAction(command="stat_add str 1", description="Allocate stat point", timeout_seconds=1.0)],
        cooldown_seconds=1.0,
    ),

    # ── SOCIAL ──
    "auto_party_invite": MacroPattern(
        pattern_id="auto_party_invite", category="social",
        description="Auto-invite nearby complement-role bots to party.",
        triggers=[{"type": "social.nearby_complement", "op": "eq", "value": True}],
        action_sequence=[MacroAction(command="party invite BOT_ID", description="Invite to party")],
        cooldown_seconds=30.0,
    ),
    "auto_party_buff": MacroPattern(
        pattern_id="auto_party_buff", category="social",
        description="Cast party-wide buffs when party members are nearby.",
        triggers=[{"type": "social.party_nearby", "op": "eq", "value": True},
                  {"type": "combat.is_in_combat", "op": "eq", "value": False}],
        action_sequence=[MacroAction(command="ss increase_agility", description="Party Agi Up"),
                         MacroAction(command="ss bless", description="Party Bless")],
        cooldown_seconds=60.0,
    ),
}


class MacroIntelligence:
    """AI-driven macro knowledge and generation.
    
    NOT a macro execution engine — the AI (LLM) decides when to generate
    macros based on bot state. This system provides:
    1. Knowledge about community macro patterns
    2. Dynamic sequence generation based on context
    3. Integration with the action pipeline
    
    The key insight: instead of writing Perl macros in macros.txt,
    the AI generates action sequences on-the-fly and queues them
    through the existing action pipeline.
    """
    
    def __init__(self, knowledge_path: str | None = None):
        self._lock = RLock()
        self._patterns: dict[str, MacroPattern] = dict(MACRO_PATTERNS)
        self._last_triggered: dict[str, float] = {}  # pattern_id -> timestamp
        self._load_custom_patterns(knowledge_path)
    
    def _load_custom_patterns(self, path: str | None) -> None:
        """Load custom macro patterns from a JSON file (optional)."""
        if not path:
            return
        p = Path(path)
        if not p.exists():
            return
        try:
            with open(p) as f:
                data = json.load(f)
            for item in data.get("patterns", []):
                pattern = MacroPattern(
                    pattern_id=item["pattern_id"],
                    category=item.get("category", "custom"),
                    description=item.get("description", ""),
                    triggers=item.get("triggers", []),
                    action_sequence=[MacroAction(**a) for a in item.get("action_sequence", [])],
                    required_items=item.get("required_items", []),
                    required_zeny=item.get("required_zeny", 0),
                    required_level_range=tuple(item.get("required_level_range", (1, 999))),
                    cooldown_seconds=item.get("cooldown_seconds", 0),
                    exclusive=item.get("exclusive", False),
                )
                self._patterns[pattern.pattern_id] = pattern
            logger.info("macro_patterns_loaded: %d custom patterns from %s", len(data.get("patterns", [])), path)
        except Exception as e:
            logger.warning("macro_patterns_load_failed: %s", e)
    
    def get_patterns_for_context(self, *, bot_state: dict[str, Any]) -> list[MacroPattern]:
        """Return macro patterns that match the current bot state.
        
        The AI uses this to find candidate macro patterns. The LLM makes
        the final decision about which pattern to execute.
        """
        with self._lock:
            candidates: list[MacroPattern] = []
            now = time.time()
            
            for pattern in self._patterns.values():
                # Check cooldown
                last = self._last_triggered.get(pattern.pattern_id, 0)
                if now - last < pattern.cooldown_seconds:
                    continue
                
                # Check triggers
                matches = True
                for trigger in pattern.triggers:
                    trigger_type = trigger.get("type", "")
                    op = trigger.get("op", "eq")
                    value = trigger.get("value")
                    
                    # Extract actual value from bot state
                    actual = self._resolve_state_path(bot_state, trigger_type)
                    if actual is None:
                        matches = False
                        break
                    
                    if op == "eq" and actual != value:
                        matches = False
                        break
                    elif op == "lte" and not (isinstance(actual, (int, float)) and actual <= value):
                        matches = False
                        break
                    elif op == "gte" and not (isinstance(actual, (int, float)) and actual >= value):
                        matches = False
                        break
                    elif op == "gt" and not (isinstance(actual, (int, float)) and actual > value):
                        matches = False
                        break
                
                if matches:
                    candidates.append(pattern)
            
            return candidates
    
    def _resolve_state_path(self, state: dict[str, Any], path: str) -> Any:
        """Resolve a dot-separated path into the state dict.
        e.g. 'vitals.hp_ratio' -> state['vitals']['hp_ratio']
        """
        parts = path.split(".")
        current = state
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
            if current is None:
                return None
        return current
    
    def generate_sequence(self, pattern: MacroPattern, *, bot_id: str,
                          context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Generate an action sequence from a macro pattern.
        
        Fills in template values from context (e.g., actual item names,
        NPC positions, map names).
        Returns a list of action proposals ready for the action queue.
        """
        from ai_sidecar.contracts.actions import ActionProposal, ActionPriorityTier
        from datetime import UTC, datetime, timedelta
        import hashlib
        
        ctx = context or {}
        sequence: list[dict[str, Any]] = []
        now = datetime.now(UTC)
        
        for i, step in enumerate(pattern.action_sequence):
            cmd = step.command
            # Fill in template values
            for key, val in ctx.items():
                cmd = cmd.replace(f"{{{key}}}", str(val))
            
            proposal = ActionProposal(
                action_id=f"macro_{pattern.pattern_id}_{i}_{hashlib.md5(f'{bot_id}_{time.monotonic_ns()}'.encode()).hexdigest()[:8]}",
                kind="command",
                command=cmd,
                priority_tier=ActionPriorityTier.tactical if not pattern.exclusive else ActionPriorityTier.strategic,
                source="planner",
                created_at=now,
                expires_at=now + timedelta(seconds=step.timeout_seconds),
                idempotency_key=f"macro_{pattern.pattern_id}_{bot_id}_{i}",
                metadata={
                    "goal": pattern.category,
                    "objective": step.description,
                    "bot_id": bot_id,
                    "source": "macro_ai",
                    "pattern_id": pattern.pattern_id,
                    "step": i,
                    "total_steps": len(pattern.action_sequence),
                    "sequence_id": f"{pattern.pattern_id}_{int(time.time())}",
                },
            )
            sequence.append({
                "action": proposal,
                "timeout": step.timeout_seconds,
                "requires_confirmation": step.requires_confirmation,
                "conflict_key": step.conflict_key or f"macro.{pattern.pattern_id}",
            })
        
        # Mark as triggered for cooldown
        with self._lock:
            self._last_triggered[pattern.pattern_id] = time.time()
        
        return sequence
    
    def get_all_patterns(self) -> dict[str, MacroPattern]:
        """Return all known macro patterns (for AI context)."""
        return dict(self._patterns)
    
    def get_patterns_by_category(self, category: str) -> list[MacroPattern]:
        """Return patterns matching a category."""
        return [p for p in self._patterns.values() if p.category == category]
    
    def add_custom_pattern(self, pattern: MacroPattern) -> None:
        """Add a new pattern dynamically (AI-generated patterns)."""
        with self._lock:
            self._patterns[pattern.pattern_id] = pattern
            logger.info("macro_pattern_added: %s (%s)", pattern.pattern_id, pattern.category)
    
    def stats(self) -> dict[str, Any]:
        """Return system statistics."""
        with self._lock:
            return {
                "total_patterns": len(self._patterns),
                "categories": list({p.category for p in self._patterns.values()}),
                "active_cooldowns": sum(1 for t in self._last_triggered.values()
                                       if time.time() - t < 60),
            }
