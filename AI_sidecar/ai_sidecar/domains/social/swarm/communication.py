"""Swarm communication — inter-bot messaging via shared state files.

Since bots cannot talk to each other directly (no ZMQ, no network),
each bot writes its state to a JSON file and reads other bots' states.
The party leader reads all states, makes decisions, and writes
a decision file that followers consume.

File layout:
  data/swarm_state_{bot_name}.json   — each bot writes its own state
  data/swarm_decision.json           — leader writes decisions, followers read
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
#  Data structures
# ────────────────────────────────────────────────────────────────

@dataclass
class BotSwarmState:
    """State that a single bot publishes to the swarm."""
    bot_name: str
    map_name: str = ""
    x: int = 0
    y: int = 0
    level: int = 1
    job_level: int = 1
    job: str = "novice"
    hp: int = 1
    hp_max: int = 1
    sp: int = 0
    sp_max: int = 1
    in_party: bool = False
    is_leader: bool = False
    target_monster: str = ""
    combat_active: bool = False
    role: str = "idle"          # tank / healer / dps / support
    formation_position: str = ""  # assigned slot name
    hp_pct: float = 1.0
    sp_pct: float = 1.0
    weight_pct: float = 0.0
    zeny: int = 0
    party_member_count: int = 0
    status: str = "idle"        # idle / hunting / returning / trading / dead
    current_hunt_map: str = ""
    skill_names: list[str] = field(default_factory=list)
    buffs_active: list[str] = field(default_factory=list)
    has_blessing: bool = False
    has_agi: bool = False
    vote_hunt_map: str = ""     # which map this bot wants to hunt
    vote_retreat: bool = False  # does this bot think we should retreat
    vote_confidence: float = 0.0
    wants_to_leave_party: bool = False
    acolyte_can_buff: bool = False  # does this bot have Blessing / Increase AGI
    timestamp: float = field(default_factory=time.time)
    bot_id: str = ""

    def __post_init__(self) -> None:
        if not self.bot_id:
            self.bot_id = self.bot_name
        # Recalculate hp_pct/sp_pct from hp/hp_max when hp differs from defaults
        # This ensures callers that set hp/hp_max get consistent pcts
        # Callers that explicitly set hp_pct keep their value when hp=hp_max=1
        if self.hp > 0 and self.hp_max > 0 and (self.hp != 1 or self.hp_max != 1):
            self.hp_pct = self.hp / max(1, self.hp_max)
        if self.sp > 0 and self.sp_max > 0 and (self.sp != 0 or self.sp_max != 1):
            self.sp_pct = self.sp / max(1, self.sp_max)


@dataclass
class SwarmDecision:
    """Decision written by the party leader for all bots to follow."""
    decision_id: str = ""
    leader_name: str = ""
    timestamp: float = field(default_factory=time.time)
    formation: str = "vanguard"       # line / box / spread / protect / wedge
    hunt_map: str = ""                # map all bots should hunt on
    target_monster: str = ""          # primary monster to kill
    retreat: bool = False             # global retreat signal
    focus_fire_monster_id: str = ""   # all bots attack this monster ID
    spread_targets: bool = False      # each bot picks a different target
    kite_mode: bool = False           # ranged bots kite, melee tanks
    formation_positions: dict[str, dict[str, int]] = field(default_factory=dict)
    member_range: int = 15            # max cells between members for shared exp
    party_auto_share: bool = True     # experience sharing on/off
    acolyte_buffs: bool = True        # acolyte should buff party
    buff_targets: list[str] = field(default_factory=list)  # who to buff
    migration_order: list[str] = field(default_factory=list)  # bot migration order
    consensus_threshold: float = 0.66  # fraction needed for consensus
    reason: str = ""
    version: int = 1


# ────────────────────────────────────────────────────────────────
#  File-based communication
# ────────────────────────────────────────────────────────────────

class SwarmFileStore:
    """Read/write swarm state and decision files from disk.

    Each bot writes its state atomically to avoid partial reads.
    """

    def __init__(self, data_dir: str | Path = "data/swarm") -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        logger.info("SwarmFileStore initialized at %s", self._data_dir.resolve())

    # ── Per-bot state files ─────────────────────────────────────

    @property
    def _state_glob(self) -> str:
        return str(self._data_dir / "swarm_state_*.json")

    def _state_path(self, bot_name: str) -> Path:
        return self._data_dir / f"swarm_state_{bot_name}.json"

    def write_bot_state(self, state: BotSwarmState) -> None:
        """Atomically write a bot's state to its state file."""
        path = self._state_path(state.bot_name)
        tmp = path.with_suffix(".tmp")
        try:
            with self._lock:
                state.timestamp = time.time()
                data = asdict(state)
                # Ensure serializable types
                data["timestamp"] = state.timestamp
                tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
                tmp.replace(path)
        except OSError as exc:
            logger.error("Failed to write swarm state for %s: %s", state.bot_name, exc)

    def read_bot_state(self, bot_name: str) -> BotSwarmState | None:
        """Read a single bot's state file. Returns None if missing/stale."""
        path = self._state_path(bot_name)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            age = time.time() - data.get("timestamp", 0)
            if age > 120:  # stale after 2 minutes
                logger.debug("Swarm state for %s is stale (%ds old)", bot_name, age)
                return None
            return BotSwarmState(**data)
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            logger.warning("Failed to read swarm state for %s: %s", bot_name, exc)
            return None

    def read_all_bot_states(self) -> dict[str, BotSwarmState]:
        """Read all non-stale bot states from the data directory."""
        states: dict[str, BotSwarmState] = {}
        try:
            for fpath in sorted(self._data_dir.glob("swarm_state_*.json")):
                bot_name = fpath.stem.replace("swarm_state_", "")
                state = self.read_bot_state(bot_name)
                if state is not None:
                    states[bot_name] = state
        except OSError as exc:
            logger.error("Failed to list swarm states: %s", exc)
        return states

    def list_known_bots(self) -> list[str]:
        """Return names of all bots that have recently written state."""
        return list(self.read_all_bot_states().keys())

    # ── Decision file (leader writes, followers read) ───────────

    @property
    def _decision_path(self) -> Path:
        return self._data_dir / "swarm_decision.json"

    def write_decision(self, decision: SwarmDecision) -> None:
        """Write the latest swarm decision (by party leader)."""
        path = self._decision_path
        tmp = path.with_suffix(".tmp")
        try:
            with self._lock:
                decision.timestamp = time.time()
                decision.version += 1
                tmp.write_text(
                    json.dumps(asdict(decision), indent=2, default=str),
                    encoding="utf-8",
                )
                tmp.replace(path)
        except OSError as exc:
            logger.error("Failed to write swarm decision: %s", exc)

    def read_decision(self) -> SwarmDecision | None:
        """Read the latest swarm decision. Returns None if missing/stale."""
        path = self._decision_path
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            age = time.time() - data.get("timestamp", 0)
            if age > 180:  # decisions stale after 3 minutes
                logger.debug("Swarm decision is stale (%ds old)", age)
                return None
            return SwarmDecision(**data)
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            logger.warning("Failed to read swarm decision: %s", exc)
            return None

    def clear_decision(self) -> None:
        """Delete the decision file (used when leader disbands)."""
        try:
            self._decision_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Failed to clear decision: %s", exc)

    # ── Convenience ─────────────────────────────────────────────

    def get_leader_name(self) -> str | None:
        """Return the highest-level bot name from available states."""
        states = self.read_all_bot_states()
        if not states:
            return None
        # Party leader is the highest-level bot
        return max(states, key=lambda n: states[n].level)

    def collect_bot_state_for_leader(self, bot_name: str, signals: dict[str, Any]) -> BotSwarmState:
        """Build a BotSwarmState from the bridge signals and the bot's PDCA context."""
        hp = int(signals.get("hp", signals.get("actor_hp", 1)) or 1)
        hp_max = int(signals.get("hp_max", signals.get("hp_max", 1)) or 1)
        sp = int(signals.get("sp", 0) or 0)
        sp_max = int(signals.get("sp_max", 1) or 1)

        skills_raw: list = signals.get("skills", []) or []
        skill_names: list[str] = []
        if isinstance(skills_raw, dict):
            skill_names = list(skills_raw.keys())
        elif isinstance(skills_raw, list):
            for s in skills_raw:
                if isinstance(s, dict):
                    skill_names.append(str(s.get("name", "")))
                elif isinstance(s, str):
                    skill_names.append(s)

        buffs: list[str] = list(signals.get("buffs", signals.get("active_buffs", [])) or [])
        has_blessing = any("bless" in b.lower() for b in buffs)
        has_agi = any("agi" in b.lower() or "increase_agi" in b.lower() for b in buffs)

        acolyte_can_buff = any(
            kw in sn.lower() for sn in skill_names
            for kw in ["blessing", "increase_agi", "agi_up"]
        )

        return BotSwarmState(
            bot_name=bot_name,
            map_name=str(signals.get("map", "") or ""),
            x=int(signals.get("x", 0) or 0),
            y=int(signals.get("y", 0) or 0),
            level=int(signals.get("base_level", 1) or 1),
            job_level=int(signals.get("job_level", 1) or 1),
            job=str(signals.get("job", signals.get("job_name", "novice")) or "novice"),
            hp=hp,
            hp_max=hp_max,
            sp=sp,
            sp_max=sp_max,
            in_party=bool(signals.get("in_party", False)),
            is_leader=bool(signals.get("is_leader", False)),
            target_monster=str(signals.get("target_monster", "") or ""),
            combat_active=bool(signals.get("in_combat", signals.get("combat_active", False))),
            role=str(signals.get("role", "idle") or "idle"),
            hp_pct=hp / max(1, hp_max),
            sp_pct=sp / max(1, sp_max),
            weight_pct=float(signals.get("weight_pct", 0.0)),
            zeny=int(signals.get("zeny", 0) or 0),
            party_member_count=int(signals.get("party_member_count", 0)),
            status=str(signals.get("status", "idle") or "idle"),
            current_hunt_map=str(signals.get("current_hunt_map", signals.get("hunt_map", "")) or ""),
            skill_names=skill_names,
            buffs_active=buffs,
            has_blessing=has_blessing,
            has_agi=has_agi,
            acolyte_can_buff=acolyte_can_buff,
            timestamp=time.time(),
            bot_id=bot_name,
        )
