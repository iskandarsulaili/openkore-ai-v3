"""
Skill Chain Execution — executes skill rotations, not just single skills.

A pro player doesn't use one skill — they use rotations. Fire Bolt → Cold Bolt →
Lightning Bolt for maximum DPS. This module executes multi-step skill chains
until interrupted by a higher-priority reflex.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class SkillChainStep:
    """A single step in a skill chain."""
    skill_name: str
    command_template: str = ""
    cooldown_ms: int = 0
    sp_cost: int = 0
    cast_time_ms: int = 0
    delay_ms: int = 0


@dataclass
class SkillChain:
    """A named skill rotation chain."""
    name: str
    steps: list[SkillChainStep] = field(default_factory=list)
    priority: int = 50
    loop: bool = True
    interruptible: bool = True
    min_sp: int = 0
    target_condition: str = ""


@dataclass
class ChainState:
    """Current state of a skill chain execution."""
    chain_name: str = ""
    current_step: int = 0
    started_at: float = 0.0
    last_step_at: float = 0.0
    is_active: bool = False
    is_paused: bool = False
    pause_reason: str = ""


class SkillChainExecutor:
    """Executes multi-step skill chains with cooldown-aware timing."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._chains: dict[str, SkillChain] = {}
        self._state: ChainState = ChainState()
        self._enqueue_fn: Callable | None = None
        self._load_chains()

    def _load_chains(self) -> None:
        """Load default skill chains."""
        # Mage DPS chain
        self._chains["mage_fire_combo"] = SkillChain(
            name="mage_fire_combo",
            steps=[
                SkillChainStep("Fire Bolt", "skill Fire Bolt 3", cooldown_ms=500, sp_cost=15, cast_time_ms=1500, delay_ms=500),
                SkillChainStep("Cold Bolt", "skill Cold Bolt 5", cooldown_ms=500, sp_cost=15, cast_time_ms=1500, delay_ms=500),
                SkillChainStep("Lightning Bolt", "skill Lightning Bolt 5", cooldown_ms=500, sp_cost=20, cast_time_ms=1500, delay_ms=500),
            ],
            priority=80,
            loop=True,
            interruptible=True,
            min_sp=50,
        )

        # Mage AoE chain
        self._chains["mage_aoe_chain"] = SkillChain(
            name="mage_aoe_chain",
            steps=[
                SkillChainStep("Fire Ball", "skill Fire Ball 5", cooldown_ms=2000, sp_cost=25, cast_time_ms=2000, delay_ms=1000),
                SkillChainStep("Thunderstorm", "skill Thunderstorm 5", cooldown_ms=3000, sp_cost=35, cast_time_ms=3000, delay_ms=1500),
                SkillChainStep("Lord of Vermilion", "skill Lord of Vermilion 5", cooldown_ms=5000, sp_cost=85, cast_time_ms=5000, delay_ms=3000),
            ],
            priority=70,
            loop=True,
            interruptible=True,
            min_sp=80,
        )

        # Mage boss chain
        self._chains["mage_boss_chain"] = SkillChain(
            name="mage_boss_chain",
            steps=[
                SkillChainStep("Storm Gust", "skill Storm Gust 10", cooldown_ms=5000, sp_cost=80, cast_time_ms=5000, delay_ms=3000),
                SkillChainStep("Meteor Storm", "skill Meteor Storm 10", cooldown_ms=5000, sp_cost=90, cast_time_ms=6000, delay_ms=3000),
                SkillChainStep("Heaven's Drive", "skill Heaven's Drive 5", cooldown_ms=5000, sp_cost=45, cast_time_ms=4000, delay_ms=2000),
            ],
            priority=90,
            loop=True,
            interruptible=False,
            min_sp=100,
        )

        # Archer DPS chain
        self._chains["archer_dps_chain"] = SkillChain(
            name="archer_dps_chain",
            steps=[
                SkillChainStep("Double Strafe", "skill Double Strafe 10", cooldown_ms=200, sp_cost=8, cast_time_ms=0, delay_ms=200),
                SkillChainStep("Double Strafe", "skill Double Strafe 10", cooldown_ms=200, sp_cost=8, cast_time_ms=0, delay_ms=200),
                SkillChainStep("Double Strafe", "skill Double Strafe 10", cooldown_ms=200, sp_cost=8, cast_time_ms=0, delay_ms=200),
            ],
            priority=80,
            loop=True,
            interruptible=True,
            min_sp=24,
        )

        # Archer AoE chain
        self._chains["archer_aoe_chain"] = SkillChain(
            name="archer_aoe_chain",
            steps=[
                SkillChainStep("Arrow Shower", "skill Arrow Shower 5", cooldown_ms=2000, sp_cost=15, cast_time_ms=0, delay_ms=1000),
                SkillChainStep("Arrow Shower", "skill Arrow Shower 5", cooldown_ms=2000, sp_cost=15, cast_time_ms=0, delay_ms=1000),
            ],
            priority=70,
            loop=True,
            interruptible=True,
            min_sp=30,
        )

        # Swordman melee chain
        self._chains["swordman_melee_chain"] = SkillChain(
            name="swordman_melee_chain",
            steps=[
                SkillChainStep("Bash", "skill Bash 10", cooldown_ms=500, sp_cost=5, cast_time_ms=0, delay_ms=500),
                SkillChainStep("Magnum Break", "skill Magnum Break 5", cooldown_ms=3000, sp_cost=12, cast_time_ms=0, delay_ms=1000),
                SkillChainStep("Bowling Bash", "skill Bowling Bash 5", cooldown_ms=2000, sp_cost=15, cast_time_ms=0, delay_ms=1000),
            ],
            priority=80,
            loop=True,
            interruptible=True,
            min_sp=20,
        )

        # Thief burst chain
        self._chains["thief_burst_chain"] = SkillChain(
            name="thief_burst_chain",
            steps=[
                SkillChainStep("Sonic Blow", "skill Sonic Blow 10", cooldown_ms=2000, sp_cost=20, cast_time_ms=0, delay_ms=1000),
                SkillChainStep("Double Attack", "attack", cooldown_ms=0, sp_cost=0, cast_time_ms=0, delay_ms=0),
                SkillChainStep("Sonic Blow", "skill Sonic Blow 10", cooldown_ms=2000, sp_cost=20, cast_time_ms=0, delay_ms=1000),
            ],
            priority=80,
            loop=True,
            interruptible=True,
            min_sp=40,
        )

        # Acolyte undead chain
        self._chains["acolyte_undead_chain"] = SkillChain(
            name="acolyte_undead_chain",
            steps=[
                SkillChainStep("Holy Light", "skill Holy Light 5", cooldown_ms=500, sp_cost=15, cast_time_ms=1500, delay_ms=500),
                SkillChainStep("Turn Undead", "skill Turn Undead 5", cooldown_ms=3000, sp_cost=20, cast_time_ms=2000, delay_ms=1000),
                SkillChainStep("Holy Light", "skill Holy Light 5", cooldown_ms=500, sp_cost=15, cast_time_ms=1500, delay_ms=500),
            ],
            priority=85,
            loop=True,
            interruptible=True,
            min_sp=30,
        )

    # ── Public API ──

    def start_chain(self, chain_name: str) -> bool:
        """Start executing a skill chain."""
        with self._lock:
            chain = self._chains.get(chain_name)
            if not chain:
                return False
            self._state = ChainState(
                chain_name=chain_name,
                current_step=0,
                started_at=time.time(),
                last_step_at=time.time(),
                is_active=True,
            )
            logger.info("skill_chain_started: %s", chain_name)
            return True

    def stop_chain(self) -> None:
        """Stop the current skill chain."""
        with self._lock:
            self._state.is_active = False
            self._state.is_paused = False

    def pause_chain(self, reason: str = "") -> None:
        """Pause the current skill chain."""
        with self._lock:
            self._state.is_paused = True
            self._state.pause_reason = reason

    def resume_chain(self) -> None:
        """Resume the current skill chain."""
        with self._lock:
            self._state.is_paused = False
            self._state.pause_reason = ""

    def get_next_step(self, current_sp: int = 0, cooldowns: dict[str, int] | None = None) -> SkillChainStep | None:
        """Get the next step in the current chain."""
        cooldowns = cooldowns or {}
        with self._lock:
            if not self._state.is_active or self._state.is_paused:
                return None

            chain = self._chains.get(self._state.chain_name)
            if not chain:
                return None

            # Check SP
            if current_sp < chain.min_sp:
                self.pause_chain("low_sp")
                return None

            # Get current step
            step = chain.steps[self._state.current_step]

            # Check cooldown
            remaining = cooldowns.get(step.skill_name, 0)
            if remaining > 0:
                # Try next step
                next_idx = (self._state.current_step + 1) % len(chain.steps)
                if next_idx != self._state.current_step:
                    next_step = chain.steps[next_idx]
                    next_remaining = cooldowns.get(next_step.skill_name, 0)
                    if next_remaining == 0:
                        self._state.current_step = next_idx
                        self._state.last_step_at = time.time()
                        return next_step
                return None

            # Advance to next step
            self._state.current_step = (self._state.current_step + 1) % len(chain.steps)
            self._state.last_step_at = time.time()
            return step

    def execute_next(self, current_sp: int = 0, cooldowns: dict[str, int] | None = None) -> bool:
        """Execute the next step in the current chain."""
        with self._lock:
            step = self.get_next_step(current_sp, cooldowns)
            if not step or not self._enqueue_fn:
                return False
            self._enqueue_fn("self", step.command_template)
            return True

    def get_chain(self, name: str) -> SkillChain | None:
        with self._lock:
            return self._chains.get(name)

    def register_chain(self, chain: SkillChain) -> None:
        with self._lock:
            self._chains[chain.name] = chain

    def get_all_chains(self) -> list[SkillChain]:
        with self._lock:
            return list(self._chains.values())

    def get_best_chain(self, job_class: str, target_element: str = "neutral",
                       aggro_count: int = 0, is_boss: bool = False) -> SkillChain | None:
        """Get the best chain for the current situation."""
        with self._lock:
            candidates: list[SkillChain] = []
            for chain in self._chains.values():
                # Match by job class
                if job_class in chain.name:
                    candidates.append(chain)

            if not candidates:
                return None

            # Sort by priority
            candidates.sort(key=lambda c: -c.priority)

            # Boss chains get priority for bosses
            if is_boss:
                boss_chains = [c for c in candidates if "boss" in c.name]
                if boss_chains:
                    return boss_chains[0]

            # AoE chains for groups
            if aggro_count > 3:
                aoe_chains = [c for c in candidates if "aoe" in c.name]
                if aoe_chains:
                    return aoe_chains[0]

            return candidates[0]

    def get_state(self) -> ChainState:
        with self._lock:
            return self._state

    def is_chain_active(self) -> bool:
        with self._lock:
            return self._state.is_active and not self._state.is_paused

    def set_enqueue_fn(self, fn: Callable) -> None:
        with self._lock:
            self._enqueue_fn = fn

    def get_chain_summary(self) -> str:
        with self._lock:
            lines = [f"── Skill Chain Executor ──"]
            lines.append(f"Chains loaded: {len(self._chains)}")
            lines.append(f"Active chain: {self._state.chain_name if self._state.is_active else 'none'}")
            lines.append(f"Step: {self._state.current_step}")
            lines.append(f"Paused: {self._state.is_paused} ({self._state.pause_reason})")
            for name, chain in sorted(self._chains.items()):
                lines.append(f"  {name}: {len(chain.steps)} steps, priority={chain.priority}, loop={chain.loop}")
            return "\n".join(lines)


# ── Global Singleton ──

_chain_executor: SkillChainExecutor | None = None
_chain_executor_lock = RLock()


def get_skill_chain_executor() -> SkillChainExecutor:
    global _chain_executor
    with _chain_executor_lock:
        if _chain_executor is None:
            _chain_executor = SkillChainExecutor()
        return _chain_executor
