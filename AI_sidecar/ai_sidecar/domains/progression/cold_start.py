"""Cold start — automatic character creation and onboarding.

Provides:
  - ColdStartManager: checks if account has characters, creates one if none exist
  - Character creation with configured job class
  - Stat allocation plan from stat_planner.py
  - Character slot selection (up to 12 slots)
  - Delete-recreate if enabled
  - Post-creation verification
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)


# ── Character Slot Configuration ───────────────────────────────────────────

# Maximum number of character slots available
MAX_CHARACTER_SLOTS = 12

# Default character creation parameters
DEFAULT_START_MAP = "prontera"
DEFAULT_START_X = 156
DEFAULT_START_Y = 165


# ── Job class definitions ─────────────────────────────────────────────────

@dataclass
class JobClassDef:
    """Definition of a job class for character creation."""
    name: str
    display_name: str
    base_str: int = 1
    base_agi: int = 1
    base_vit: int = 1
    base_int: int = 1
    base_dex: int = 1
    base_luk: int = 1
    start_weapon: str = ""
    start_armor: str = ""

    def stat_allocation_string(self) -> str:
        """Generate stat allocation command string."""
        parts = []
        if self.base_str > 1:
            parts.append(f"str {self.base_str}")
        if self.base_agi > 1:
            parts.append(f"agi {self.base_agi}")
        if self.base_vit > 1:
            parts.append(f"vit {self.base_vit}")
        if self.base_int > 1:
            parts.append(f"int {self.base_int}")
        if self.base_dex > 1:
            parts.append(f"dex {self.base_dex}")
        if self.base_luk > 1:
            parts.append(f"luk {self.base_luk}")
        return " ".join(parts)


# ── Job class presets ──────────────────────────────────────────────────────

JOB_CLASSES: dict[str, JobClassDef] = {
    "novice": JobClassDef(
        name="novice", display_name="Novice",
        base_str=1, base_agi=1, base_vit=1, base_int=1, base_dex=1, base_luk=1,
    ),
    "swordman": JobClassDef(
        name="swordman", display_name="Swordman",
        base_str=9, base_agi=9, base_vit=1, base_int=1, base_dex=9, base_luk=1,
        start_weapon="1201", start_armor="2301",
    ),
    "mage": JobClassDef(
        name="mage", display_name="Mage",
        base_str=1, base_agi=1, base_vit=1, base_int=9, base_dex=9, base_luk=1,
        start_weapon="1601", start_armor="2301",
    ),
    "archer": JobClassDef(
        name="archer", display_name="Archer",
        base_str=1, base_agi=9, base_vit=1, base_int=1, base_dex=9, base_luk=5,
        start_weapon="1701", start_armor="2301",
    ),
    "acolyte": JobClassDef(
        name="acolyte", display_name="Acolyte",
        base_str=5, base_agi=1, base_vit=5, base_int=9, base_dex=5, base_luk=1,
        start_weapon="1501", start_armor="2301",
    ),
    "thief": JobClassDef(
        name="thief", display_name="Thief",
        base_str=5, base_agi=9, base_vit=1, base_int=1, base_dex=9, base_luk=1,
        start_weapon="1301", start_armor="2301",
    ),
    "merchant": JobClassDef(
        name="merchant", display_name="Merchant",
        base_str=9, base_agi=1, base_vit=5, base_int=1, base_dex=5, base_luk=1,
        start_weapon="1201", start_armor="2301",
    ),
}


# ── Stat Planner Integration ──────────────────────────────────────────────

@dataclass
class StatPlan:
    """A stat allocation plan for a character.

    Defines target stats at various level milestones.
    """
    name: str
    job_class: str
    milestones: list[dict[str, Any]] = field(default_factory=list)
    # milestone format: {"level": 10, "stats": {"str": 10, "agi": 20, ...}}

    def get_stats_for_level(self, level: int) -> dict[str, int]:
        """Get the recommended stat allocation for a given level."""
        best: dict[str, int] = {}
        for milestone in self.milestones:
            if milestone.get("level", 0) <= level:
                best = milestone.get("stats", {})
        return best

    def get_stat_commands(self, current_stats: dict[str, int], target_stats: dict[str, int]) -> list[str]:
        """Generate stat_add commands to reach target stats from current."""
        commands: list[str] = []
        for stat, target in target_stats.items():
            current = current_stats.get(stat, 1)
            if target > current:
                commands.append(f"stat_add {stat} {target - current}")
        return commands


# ── Default stat plans ─────────────────────────────────────────────────────

DEFAULT_STAT_PLANS: dict[str, StatPlan] = {
    "swordman": StatPlan(
        name="swordman_str_agi",
        job_class="swordman",
        milestones=[
            {"level": 10, "stats": {"str": 20, "agi": 15, "dex": 15, "vit": 10}},
            {"level": 20, "stats": {"str": 30, "agi": 25, "dex": 20, "vit": 15}},
            {"level": 30, "stats": {"str": 40, "agi": 35, "dex": 25, "vit": 20}},
            {"level": 40, "stats": {"str": 50, "agi": 45, "dex": 30, "vit": 25}},
            {"level": 50, "stats": {"str": 60, "agi": 50, "dex": 35, "vit": 30}},
            {"level": 60, "stats": {"str": 70, "agi": 55, "dex": 40, "vit": 35}},
            {"level": 70, "stats": {"str": 80, "agi": 60, "dex": 45, "vit": 40}},
            {"level": 80, "stats": {"str": 85, "agi": 65, "dex": 50, "vit": 45}},
            {"level": 90, "stats": {"str": 90, "agi": 70, "dex": 55, "vit": 50}},
            {"level": 99, "stats": {"str": 99, "agi": 75, "dex": 60, "vit": 55}},
        ],
    ),
    "mage": StatPlan(
        name="mage_int_dex",
        job_class="mage",
        milestones=[
            {"level": 10, "stats": {"int": 20, "dex": 15, "vit": 10}},
            {"level": 20, "stats": {"int": 30, "dex": 20, "vit": 15}},
            {"level": 30, "stats": {"int": 40, "dex": 25, "vit": 20}},
            {"level": 40, "stats": {"int": 50, "dex": 30, "vit": 25}},
            {"level": 50, "stats": {"int": 60, "dex": 35, "vit": 30}},
            {"level": 60, "stats": {"int": 70, "dex": 40, "vit": 35}},
            {"level": 70, "stats": {"int": 80, "dex": 45, "vit": 40}},
            {"level": 80, "stats": {"int": 85, "dex": 50, "vit": 45}},
            {"level": 90, "stats": {"int": 90, "dex": 55, "vit": 50}},
            {"level": 99, "stats": {"int": 99, "dex": 60, "vit": 55}},
        ],
    ),
    "archer": StatPlan(
        name="archer_dex_agi",
        job_class="archer",
        milestones=[
            {"level": 10, "stats": {"dex": 20, "agi": 15, "luk": 10}},
            {"level": 20, "stats": {"dex": 30, "agi": 20, "luk": 15}},
            {"level": 30, "stats": {"dex": 40, "agi": 25, "luk": 20}},
            {"level": 40, "stats": {"dex": 50, "agi": 30, "luk": 25}},
            {"level": 50, "stats": {"dex": 60, "agi": 35, "luk": 30}},
            {"level": 60, "stats": {"dex": 70, "agi": 40, "luk": 35}},
            {"level": 70, "stats": {"dex": 80, "agi": 45, "luk": 40}},
            {"level": 80, "stats": {"dex": 85, "agi": 50, "luk": 45}},
            {"level": 90, "stats": {"dex": 90, "agi": 55, "luk": 50}},
            {"level": 99, "stats": {"dex": 99, "agi": 60, "luk": 55}},
        ],
    ),
    "acolyte": StatPlan(
        name="acolyte_int_dex",
        job_class="acolyte",
        milestones=[
            {"level": 10, "stats": {"int": 20, "dex": 15, "vit": 10}},
            {"level": 20, "stats": {"int": 30, "dex": 20, "vit": 15}},
            {"level": 30, "stats": {"int": 40, "dex": 25, "vit": 20}},
            {"level": 40, "stats": {"int": 50, "dex": 30, "vit": 25}},
            {"level": 50, "stats": {"int": 60, "dex": 35, "vit": 30}},
            {"level": 60, "stats": {"int": 70, "dex": 40, "vit": 35}},
            {"level": 70, "stats": {"int": 80, "dex": 45, "vit": 40}},
            {"level": 80, "stats": {"int": 85, "dex": 50, "vit": 45}},
            {"level": 90, "stats": {"int": 90, "dex": 55, "vit": 50}},
            {"level": 99, "stats": {"int": 99, "dex": 60, "vit": 55}},
        ],
    ),
    "thief": StatPlan(
        name="thief_agi_dex",
        job_class="thief",
        milestones=[
            {"level": 10, "stats": {"agi": 20, "dex": 15, "str": 10}},
            {"level": 20, "stats": {"agi": 30, "dex": 20, "str": 15}},
            {"level": 30, "stats": {"agi": 40, "dex": 25, "str": 20}},
            {"level": 40, "stats": {"agi": 50, "dex": 30, "str": 25}},
            {"level": 50, "stats": {"agi": 60, "dex": 35, "str": 30}},
            {"level": 60, "stats": {"agi": 70, "dex": 40, "str": 35}},
            {"level": 70, "stats": {"agi": 80, "dex": 45, "str": 40}},
            {"level": 80, "stats": {"agi": 85, "dex": 50, "str": 45}},
            {"level": 90, "stats": {"agi": 90, "dex": 55, "str": 50}},
            {"level": 99, "stats": {"agi": 99, "dex": 60, "str": 55}},
        ],
    ),
    "merchant": StatPlan(
        name="merchant_str_vit",
        job_class="merchant",
        milestones=[
            {"level": 10, "stats": {"str": 20, "vit": 15, "dex": 10}},
            {"level": 20, "stats": {"str": 30, "vit": 20, "dex": 15}},
            {"level": 30, "stats": {"str": 40, "vit": 25, "dex": 20}},
            {"level": 40, "stats": {"str": 50, "vit": 30, "dex": 25}},
            {"level": 50, "stats": {"str": 60, "vit": 35, "dex": 30}},
            {"level": 60, "stats": {"str": 70, "vit": 40, "dex": 35}},
            {"level": 70, "stats": {"str": 80, "vit": 45, "dex": 40}},
            {"level": 80, "stats": {"str": 85, "vit": 50, "dex": 45}},
            {"level": 90, "stats": {"str": 90, "vit": 55, "dex": 50}},
            {"level": 99, "stats": {"str": 99, "vit": 60, "dex": 55}},
        ],
    ),
}


# ── Cold Start Manager ────────────────────────────────────────────────────

@dataclass
class ColdStartConfig:
    """Configuration for cold start behavior."""
    job_class: str = "swordman"
    character_name_prefix: str = "Bot"
    start_map: str = DEFAULT_START_MAP
    start_x: int = DEFAULT_START_X
    start_y: int = DEFAULT_START_Y
    enable_delete_recreate: bool = False
    max_creation_retries: int = 3
    verify_after_creation: bool = True
    preferred_slot: int = 0  # 0 = auto-select first empty slot


class ColdStartManager:
    """Manages automatic character creation and onboarding.

    Workflow:
    1. Check if the account has characters
    2. If no characters exist, create one with the configured job class
    3. Allocate stats according to the stat plan
    4. Handle character slot selection (up to 12 slots)
    5. Handle delete-recreate if enabled
    6. Verify character exists after creation before proceeding
    """

    def __init__(self, config: ColdStartConfig | None = None) -> None:
        self._config = config or ColdStartConfig()
        self._creation_attempts: dict[str, int] = {}  # bot_id -> attempt count
        self._connection_blocked_until: dict[str, float] = {}  # bot_id -> timestamp
        self._last_creation_cycle: dict[str, int] = {}  # bot_id -> cycle counter

    def _is_connection_blocked(self, signals: dict) -> bool:
        """Check if the connection is blocked by a transient server issue.

        Detects 'Dual login prohibited', 'Timeout on Character Select Server',
        and similar transient failures that character creation can't solve.
        """
        _blocked = False
        for _key in ["error", "raw_message", "message", "last_error"]:
            _val = signals.get(_key, "")
            if isinstance(_val, str) and ("Dual login" in _val or "Timeout on Character Select" in _val):
                _blocked = True
                break
        if not _blocked:
            # Also check lifecycle state
            _phase = str(signals.get("lifecycle_phase", ""))
            if "TIMEOUT" in _phase.upper() or "SERVER_BLOCKED" in _phase.upper():
                _blocked = True
        return _blocked

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        bot_id: str,
    ) -> None:
        """Evaluate cold start state and emit character creation actions.

        Called each cycle. Emits actions only when character creation is needed.
        Once a character exists and is verified, no further actions are emitted.
        """
        # CRITICAL GATE: skip if bot is already in-game. In-game, @main::chars
        # is empty (the bridge only populates it at char-select), so
        # characters=[] would make us think no character exists and emit
        # unnecessary relog/char_create commands on a live bot.
        # base_level > 0 catches all in-game bots (level 1+); at char-select
        # the bridge has no $char so base_level is 0.
        _map_known = bool(signals.get("map_known", False))
        _bl = int(signals.get("base_level", 0) or 0)
        # ROBUST IN-GAME GATE: also skip when the snapshot carries a real in-game
        # map (the bridge sets `map` to the current map name, and `in_game` to a
        # truthy flag) or a non-empty char_name. A live bot (izlude, level 1)
        # must NEVER have char-creation fired against it just because the
        # `characters` list is empty (it's only populated at char-select).
        _map_name = str(signals.get("map", "") or "").strip()
        _in_game_flag = bool(signals.get("in_game", False))
        if _map_known or _bl > 0 or _in_game_flag or _map_name:
            return
        # Check if we already have a character
        characters = self._get_characters(signals)
        if self._has_valid_character(characters):
            # Character exists — emit stat allocation if needed
            self._emit_stat_allocation(signals, actions, bot_id)
            return
        logger.debug("[cold_start] %s: no valid character in signals; chars=%s", bot_id, str(characters)[:200])
        logger.info("[cold_start] %s: characters_signal=%s", bot_id, str(characters)[:150])

        # Check if connection is blocked by server (Dual login, timeout, etc.)
        import time as _time
        _now = _time.time()
        _blocked_until = self._connection_blocked_until.get(bot_id, 0)
        if _now < _blocked_until:
            # Still in cooldown — don't attempt creation
            _remaining = int(_blocked_until - _now)
            actions.append(HeuristicAction(
                kind="log", command=f"cold_start_wait:{_remaining}s",
                confidence=0.95, domain="cold_start",
                reason=f"Server connection blocked, waiting {_remaining}s",
            ))
            return
        if self._is_connection_blocked(signals):
            # Connection blocked — set cooldown and skip creation
            _cooldown = 120  # 2 minutes
            self._connection_blocked_until[bot_id] = _now + _cooldown
            actions.append(HeuristicAction(
                kind="command", command="relog",
                confidence=0.95, domain="cold_start",
                reason="Connection blocked by server, reconnecting in 5s",
            ))
            logger.info("[cold_start] %s: Server blocked, reconnecting in 5s", bot_id)
            return

        # No valid character — need to create one
        attempt_count = self._creation_attempts.get(bot_id, 0)
        if attempt_count >= self._config.max_creation_retries:
            # Max retries reached — wait and retry connection instead of giving up forever
            _now = _time.time()
            _blocked_until = self._connection_blocked_until.get(bot_id, 0)
            if _now < _blocked_until:
                # Still in cooldown
                _remaining = int(_blocked_until - _now)
                actions.append(HeuristicAction(
                    kind="log", command=f"cold_start_wait:{_remaining}s",
                    confidence=0.95, domain="cold_start",
                    reason=f"Cooldown before reconnect retry: {_remaining}s",
                ))
                return
            # Cooldown expired — reset retries and emit relog
            self._creation_attempts[bot_id] = 0
            self._connection_blocked_until[bot_id] = _now + 120
            actions.append(HeuristicAction(
                kind="command", command="relog",
                confidence=0.95, domain="cold_start",
                reason="Max creation retries reached, reconnecting",
            ))
            logger.info("[cold_start] %s: Reset retries, emitting relog (cooldown 120s)", bot_id)
            return

        # Determine which slot to use
        slot = self._find_empty_slot(characters)
        if slot is None:
            if self._config.enable_delete_recreate:
                # Delete the lowest-level character and recreate
                slot = self._find_delete_slot(characters)
                if slot is not None:
                    self._emit_delete_character(actions, bot_id, slot, characters)
                    return
            logger.warning(
                "[cold_start] %s: All %d slots full and delete-recreate disabled",
                bot_id, MAX_CHARACTER_SLOTS,
            )
            return

        # Create the character
        self._creation_attempts[bot_id] = attempt_count + 1
        self._emit_create_character(actions, bot_id, slot, signals)

        # If verification is enabled, emit a verify action
        if self._config.verify_after_creation:
            self._emit_verify_character(actions, bot_id)

    # ── Character detection ───────────────────────────────────────────

    def _get_characters(self, signals: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract character list from signals."""
        characters = signals.get("characters", signals.get("character_list", []))
        if isinstance(characters, list):
            return characters
        if isinstance(characters, dict):
            return list(characters.values())
        return []

    def _has_valid_character(self, characters: list[dict[str, Any]]) -> bool:
        """Check if there's at least one valid character on the account."""
        for char in characters:
            if isinstance(char, dict):
                name = str(char.get("name", char.get("char_name", "")) or "")
                if name and not name.startswith("UNKNOWN"):
                    return True
        return False

    def _find_empty_slot(self, characters: list[dict[str, Any]]) -> int | None:
        """Find the first empty character slot (0-indexed, up to 12)."""
        occupied_slots: set[int] = set()
        for char in characters:
            if isinstance(char, dict):
                slot = int(char.get("slot", char.get("char_slot", -1)) or -1)
                if slot >= 0:
                    occupied_slots.add(slot)

        for slot in range(MAX_CHARACTER_SLOTS):
            if slot not in occupied_slots:
                return slot
        return None

    def _find_delete_slot(self, characters: list[dict[str, Any]]) -> int | None:
        """Find the best slot to delete (lowest-level character)."""
        best_slot: int | None = None
        best_level: int = 999

        for char in characters:
            if isinstance(char, dict):
                slot = int(char.get("slot", char.get("char_slot", -1)) or -1)
                level = int(char.get("level", char.get("base_level", 99)) or 99)
                if slot >= 0 and level < best_level:
                    best_level = level
                    best_slot = slot

        return best_slot

    # ── Action emitters ───────────────────────────────────────────────

    def _emit_create_character(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        slot: int,
        signals: dict[str, Any],
    ) -> None:
        """Emit character creation command."""
        job_class = self._config.job_class
        job_def = JOB_CLASSES.get(job_class, JOB_CLASSES["swordman"])

        # Generate character name
        char_name = self._generate_character_name(bot_id, slot, signals)

        # Build stat allocation string
        stat_str = job_def.stat_allocation_string()

        logger.info(
            "[cold_start] %s: Creating character '%s' (slot %d, class: %s, stats: %s)",
            bot_id, char_name, slot, job_class, stat_str or "default",
        )

        # Create character command — OpenKore uses char_create via the character select screen
        # PACKETVER 20250604 server uses char_create 0x0A39 format (matches rathena-ai-world):
        #   char_create <slot> "<name>" [hairstyle] [haircolor] [job] [sex]
        # Job: novice|summoner   Sex: M|F  (defaults: novice, F)
        _job_str = "novice"
        _sex_str = "M"
        create_cmd = f"char_create {slot} \"{char_name}\" 0 1 {_job_str} {_sex_str}"

        actions.append(HeuristicAction(
            kind="command",
            command=create_cmd,
            confidence=0.95,
            domain="progression",
            reason=f"ColdStart: create {job_class} character '{char_name}' in slot {slot}",
            metadata={
                "slot": slot,
                "char_name": char_name,
                "job_class": job_class,
                "stat_allocation": stat_str,
            },
        ))

        # Select the character after creation
        # REMOVED: char_select emitted inline — bridge's char_create handler
        # calls sendCharLogin(slot) after successful creation, which auto-enters
        # the game. Separate char_select action interferes.

    def _emit_delete_character(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        slot: int,
        characters: list[dict[str, Any]],
    ) -> None:
        """Emit character deletion command (for delete-recreate)."""
        char_name = "unknown"
        for char in characters:
            if isinstance(char, dict) and int(char.get("slot", -1)) == slot:
                char_name = str(char.get("name", "unknown"))
                break

        logger.warning(
            "[cold_start] %s: Deleting character '%s' (slot %d) for recreate",
            bot_id, char_name, slot,
        )

        actions.append(HeuristicAction(
            kind="command",
            command=f"char_delete {slot}",
            confidence=0.95,
            domain="progression",
            reason=f"ColdStart: delete '{char_name}' in slot {slot} for recreate",
            metadata={"slot": slot, "char_name": char_name},
        ))

    def _emit_verify_character(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
    ) -> None:
        """Emit character verification action."""
        actions.append(HeuristicAction(
            kind="command",
            command="char_list",
            confidence=0.90,
            domain="progression",
            reason="ColdStart: verify character exists after creation",
            metadata={"verify": True},
        ))

    def _emit_stat_allocation(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        bot_id: str,
    ) -> None:
        """Emit stat allocation commands based on the stat plan."""
        job_class = self._config.job_class

        base_level = int(signals.get("base_level", 1) or 1)
        status_points = int(signals.get("status_points", 0) or 0)

        if status_points < 5:
            return  # Not enough points to bother

        # Get current stats from signals
        current_stats: dict[str, int] = {}
        for stat in ["str", "agi", "vit", "int", "dex", "luk"]:
            current_stats[stat] = int(signals.get(f"stat_{stat}", signals.get(stat, 1)) or 1)

        # Primary: data-driven StatBreakpointPlanner (per-job breakpoint
        # builds). Fallback: embedded DEFAULT_STAT_PLANS milestones.
        target_stats: dict[str, int] | None = None
        stat_plan_name = "stat_breakpoint_planner"
        try:
            from ai_sidecar.domains.planning.stat_planner import StatBreakpointPlanner
            _sbp = getattr(self, "_stat_breakpoint_planner", None)
            if _sbp is None:
                _sbp = StatBreakpointPlanner()
                self._stat_breakpoint_planner = _sbp
            target_stats = _sbp.get_target_stats(job_class)
        except Exception:
            target_stats = None
        if not target_stats:
            stat_plan = DEFAULT_STAT_PLANS.get(job_class)
            if not stat_plan:
                return
            target_stats = stat_plan.get_stats_for_level(base_level)
            stat_plan_name = stat_plan.name

        # Generate stat commands
        commands = []
        for stat, target in (target_stats or {}).items():
            current = current_stats.get(stat, 1)
            if target > current:
                commands.append(f"stat_add {stat} {target - current}")
        if not commands:
            return

        for cmd in commands:
            actions.append(HeuristicAction(
                kind="command",
                command=cmd,
                confidence=0.90,
                domain="progression",
                reason=f"ColdStart: stat allocation ({cmd}) for {job_class} at level {base_level}",
                metadata={
                    "stat_plan": stat_plan_name,
                    "level": base_level,
                    "status_points": status_points,
                },
            ))

    # ── Name generation ────────────────────────────────────────────────

    def _generate_character_name(
        self,
        bot_id: str,
        slot: int,
        signals: dict[str, Any],
    ) -> str:
        """Generate a character name from bot_id and slot."""
        prefix = self._config.character_name_prefix
        # Use bot_id as base, sanitize it (RO allows letters/digits/underscore only)
        base = bot_id.replace("_", "").replace("-", "").replace(" ", "").replace(":", "")
        # Truncate to fit within RO's 23-char limit (NAME_LENGTH = 23+1 null)
        max_name_len = 23 - len(prefix) - 2  # -2 for slot suffix
        if max_name_len < 4:
            max_name_len = 4
        base = base[:max_name_len]
        return f"{prefix}{base}{slot:02d}"

    # ── Configuration ──────────────────────────────────────────────────

    @property
    def config(self) -> ColdStartConfig:
        return self._config

    def set_config(self, config: ColdStartConfig) -> None:
        self._config = config

    def reset_attempts(self, bot_id: str) -> None:
        """Reset creation attempt counter for a bot."""
        self._creation_attempts.pop(bot_id, None)


# ── Global Singleton ───────────────────────────────────────────────────────

_manager: ColdStartManager | None = None


def get_cold_start_manager() -> ColdStartManager:
    """Get the global ColdStartManager singleton."""
    global _manager
    if _manager is None:
        _manager = ColdStartManager()
    return _manager


def create_cold_start_manager(config: ColdStartConfig | None = None) -> ColdStartManager:
    """Factory for a new ColdStartManager with optional custom config."""
    return ColdStartManager(config=config)
