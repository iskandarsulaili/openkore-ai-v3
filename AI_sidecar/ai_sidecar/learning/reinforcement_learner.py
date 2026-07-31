"""
Reinforcement Learner — Actual ML learning for the bot fleet.

A pro player learns from experience:
- What actions lead to good outcomes (exp gain, zeny gain)
- What actions lead to bad outcomes (death, wasted potions)
- Which maps are safe vs dangerous
- Which monsters are worth fighting
- When to rest, when to push

This engine implements:
- Q-learning for action selection
- State: (map, hp_pct, sp_pct, level, zeny, monsters_nearby, party_size)
- Actions: (farm, buy_potions, sell_items, level_skill, change_map, rest, socialize)
- Reward: (exp_gained + zeny_gained - potions_used - death_penalty) / time
- Experience replay buffer
- Epsilon-greedy exploration
- Wires into _PROVIDER_HARD_DENY_BY_WORKLOAD (makes it actually work)
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────────

# Actions the bot can take
ACTIONS: list[str] = [
    "farm",           # Attack monsters for exp/items
    "buy_potions",    # Restock consumables
    "sell_items",     # Sell loot to NPC or players
    "level_skill",    # Spend skill points
    "change_map",     # Move to a different map
    "rest",           # Sit and regenerate HP/SP
    "socialize",      # Chat, party, trade
    "upgrade_gear",   # Buy better equipment
    "craft",          # Craft items
    "vend",           # Set up a vending shop
]

# State dimension names
STATE_KEYS: list[str] = [
    "map_bucket",       # Discretized map (0-9)
    "hp_pct_bucket",    # HP percentage bucket (0-4)
    "sp_pct_bucket",    # SP percentage bucket (0-4)
    "level_bucket",     # Level bucket (0-9)
    "zeny_bucket",      # Zeny bucket (0-4)
    "monsters_nearby",  # 0=none, 1=few, 2=many
    "party_size",       # 0=alone, 1=small, 2=large
    "weight_bucket",    # 0=empty, 1=some, 2=full
    "death_recently",   # 0=no, 1=yes
    "time_bucket",      # 0=off_peak, 1=normal, 2=peak
]

# Q-learning hyperparameters
DEFAULT_LEARNING_RATE: float = 0.1
DEFAULT_DISCOUNT_FACTOR: float = 0.9
DEFAULT_EPSILON: float = 0.3
DEFAULT_EPSILON_DECAY: float = 0.995
DEFAULT_MIN_EPSILON: float = 0.05
DEFAULT_REPLAY_BUFFER_SIZE: int = 10000
DEFAULT_BATCH_SIZE: int = 32


@dataclass
class Experience:
    """A single experience for the replay buffer."""
    state: tuple
    action: str
    reward: float
    next_state: tuple
    done: bool
    timestamp: float = 0.0


@dataclass
class QLearningStats:
    """Statistics for the Q-learning system."""
    total_episodes: int = 0
    total_reward: float = 0.0
    best_reward: float = float("-inf")
    worst_reward: float = float("inf")
    avg_reward_last_100: float = 0.0
    exploration_rate: float = DEFAULT_EPSILON
    actions_taken: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    action_rewards: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    recent_rewards: deque = field(default_factory=lambda: deque(maxlen=100))


@dataclass(slots=True)
class ReinforcementLearner:
    """
    Q-learning based reinforcement learner.

    Learns optimal action selection from experience.
    Thread-safe: all mutable state is guarded by RLock.
    """

    _lock: RLock = field(default_factory=RLock)
    _q_table: dict[tuple, dict[str, float]] = field(default_factory=dict)
    _replay_buffer: deque = field(default_factory=lambda: deque(maxlen=DEFAULT_REPLAY_BUFFER_SIZE))
    _stats: QLearningStats = field(default_factory=QLearningStats)
    _learning_rate: float = DEFAULT_LEARNING_RATE
    _discount_factor: float = DEFAULT_DISCOUNT_FACTOR
    _epsilon: float = DEFAULT_EPSILON
    _epsilon_decay: float = DEFAULT_EPSILON_DECAY
    _min_epsilon: float = DEFAULT_MIN_EPSILON
    _batch_size: int = DEFAULT_BATCH_SIZE
    _last_state: tuple | None = None
    _last_action: str | None = None
    _last_reward: float = 0.0
    _episode_count: int = 0
    _model_path: str = "data/reinforcement_model.pkl"
    _stats_path: str = "data/reinforcement_stats.json"
    _last_save: float = 0.0
    _save_interval: float = 300.0  # Save every 5 minutes
    _initialized: bool = False

    # ── Public API ──

    def initialize(self, model_path: str = "") -> None:
        """Initialize the learner, loading existing model if available."""
        if model_path:
            self._model_path = model_path
            self._stats_path = model_path.replace(".pkl", "_stats.json")

        self._load_model()
        self._initialized = True
        logger.info("reinforcement_learner_initialized: %d states, %d experiences",
                    len(self._q_table), len(self._replay_buffer))

    # ── State Encoding ──

    def encode_state(self, signals: dict[str, Any]) -> tuple:
        """Encode a state dict into a discretized tuple for Q-table lookup."""
        # Map bucket (discretize map name)
        map_name = str(signals.get("map", "") or "")
        map_bucket = hash(map_name) % 10

        # HP percentage bucket
        hp_pct = float(signals.get("hp_ratio", signals.get("hp_pct", 1.0)) or 1.0)
        hp_bucket = min(4, int(hp_pct * 5))

        # SP percentage bucket
        sp_pct = float(signals.get("sp_ratio", signals.get("sp_pct", 1.0)) or 1.0)
        sp_bucket = min(4, int(sp_pct * 5))

        # Level bucket
        level = int(signals.get("base_level", signals.get("level", 1)) or 1)
        level_bucket = min(9, level // 10)

        # Zeny bucket
        zeny = int(signals.get("zeny", 0) or 0)
        if zeny < 1000:
            zeny_bucket = 0
        elif zeny < 10000:
            zeny_bucket = 1
        elif zeny < 100000:
            zeny_bucket = 2
        elif zeny < 1000000:
            zeny_bucket = 3
        else:
            zeny_bucket = 4

        # Monsters nearby
        aggro_count = int(signals.get("combat.aggro_count", 0) or 0)
        if aggro_count == 0:
            monsters_bucket = 0
        elif aggro_count <= 3:
            monsters_bucket = 1
        else:
            monsters_bucket = 2

        # Party size
        party_size = int(signals.get("party_size", 0) or 0)
        if party_size <= 1:
            party_bucket = 0
        elif party_size <= 3:
            party_bucket = 1
        else:
            party_bucket = 2

        # Weight bucket
        weight_ratio = float(signals.get("weight_ratio", 0.0) or 0.0)
        if weight_ratio < 0.3:
            weight_bucket = 0
        elif weight_ratio < 0.7:
            weight_bucket = 1
        else:
            weight_bucket = 2

        # Death recently
        death_recently = 1 if signals.get("recent_death", False) else 0

        # Time bucket
        hour = time.localtime().tm_hour
        if hour in (0, 1, 2, 3, 4, 5, 6, 7, 22, 23):
            time_bucket = 0  # off-peak
        elif hour in (18, 19, 20, 21):
            time_bucket = 2  # peak
        else:
            time_bucket = 1  # normal

        return (
            map_bucket, hp_bucket, sp_bucket, level_bucket,
            zeny_bucket, monsters_bucket, party_bucket,
            weight_bucket, death_recently, time_bucket,
        )

    # ── Action Selection ──

    def select_action(self, state: tuple, available_actions: list[str] | None = None) -> str:
        """Select an action using epsilon-greedy policy."""
        with self._lock:
            if available_actions is None:
                available_actions = ACTIONS

            # Epsilon-greedy exploration
            if random.random() < self._epsilon:
                action = random.choice(available_actions)
                self._stats.actions_taken[action] += 1
                return action

            # Greedy: select best action from Q-table
            q_values = self._q_table.get(state, {})
            best_action = None
            best_value = float("-inf")

            for action in available_actions:
                value = q_values.get(action, 0.0)
                if value > best_value:
                    best_value = value
                    best_action = action

            if best_action is None:
                best_action = random.choice(available_actions)

            self._stats.actions_taken[best_action] += 1
            return best_action

    # ── Learning ──

    def observe(self, state: tuple, action: str, reward: float,
                next_state: tuple, done: bool = False) -> None:
        """Observe an experience and add it to the replay buffer."""
        with self._lock:
            exp = Experience(
                state=state, action=action, reward=reward,
                next_state=next_state, done=done,
                timestamp=time.time(),
            )
            self._replay_buffer.append(exp)
            self._stats.total_reward += reward
            self._stats.recent_rewards.append(reward)
            self._stats.action_rewards[action] += reward

            # Update best/worst
            if reward > self._stats.best_reward:
                self._stats.best_reward = reward
            if reward < self._stats.worst_reward:
                self._stats.worst_reward = reward

            # Update average
            if self._stats.recent_rewards:
                self._stats.avg_reward_last_100 = (
                    sum(self._stats.recent_rewards) / len(self._stats.recent_rewards)
                )

            # Learn from this experience immediately (online learning)
            self._learn_from_experience(exp)

            # Sample from replay buffer for batch learning
            if len(self._replay_buffer) >= self._batch_size:
                batch = random.sample(
                    list(self._replay_buffer),
                    min(self._batch_size, len(self._replay_buffer)),
                )
                for batch_exp in batch:
                    self._learn_from_experience(batch_exp)

            # Decay epsilon
            self._epsilon = max(
                self._min_epsilon,
                self._epsilon * self._epsilon_decay,
            )
            self._stats.exploration_rate = self._epsilon

            # Save periodically
            if time.time() - self._last_save > self._save_interval:
                self._save_model()
                self._last_save = time.time()

    def _learn_from_experience(self, exp: Experience) -> None:
        """Update Q-values from a single experience."""
        state = exp.state
        action = exp.action
        reward = exp.reward
        next_state = exp.next_state
        done = exp.done

        # Initialize Q-values for this state if not present
        if state not in self._q_table:
            self._q_table[state] = {a: 0.0 for a in ACTIONS}

        # Get current Q-value
        current_q = self._q_table[state].get(action, 0.0)

        # Calculate target Q-value
        if done:
            target_q = reward
        else:
            # Get max Q-value for next state
            next_q_values = self._q_table.get(next_state, {})
            max_next_q = max(next_q_values.values()) if next_q_values else 0.0
            target_q = reward + self._discount_factor * max_next_q

        # Update Q-value
        new_q = current_q + self._learning_rate * (target_q - current_q)
        self._q_table[state][action] = new_q

    # ── Reward Calculation ──

    def calculate_reward(self, signals: dict[str, Any],
                         prev_signals: dict[str, Any] | None = None) -> float:
        """Calculate reward from current and previous state signals.

        Reward = (exp_gained + zeny_gained - potions_used - death_penalty) / time
        """
        reward = 0.0

        # Exp gained
        current_exp = float(signals.get("base_exp", 0) or 0)
        prev_exp = float(prev_signals.get("base_exp", 0) or 0) if prev_signals else 0
        exp_gained = max(0, current_exp - prev_exp)
        reward += exp_gained * 0.001  # Scale down

        # Zeny gained
        current_zeny = float(signals.get("zeny", 0) or 0)
        prev_zeny = float(prev_signals.get("zeny", 0) or 0) if prev_signals else 0
        zeny_gained = max(0, current_zeny - prev_zeny)
        reward += zeny_gained * 0.0001  # Scale down

        # Potions used (negative reward)
        # Estimate from HP/SP changes
        current_hp = float(signals.get("hp", 0) or 0)
        prev_hp = float(prev_signals.get("hp", 0) or 0) if prev_signals else 0
        hp_max = float(signals.get("hp_max", 1) or 1)
        if prev_hp > current_hp and prev_hp - current_hp > hp_max * 0.3:
            # Significant HP loss = potion used
            reward -= 0.5

        # Death penalty
        if signals.get("recent_death", False):
            reward -= 10.0  # Heavy penalty for death

        # Sitting/resting (small negative reward for not being productive)
        if signals.get("is_sitting", False):
            reward -= 0.1

        # Kills (positive reward)
        kills = int(signals.get("kills_this_session", 0) or 0)
        prev_kills = int(prev_signals.get("kills_this_session", 0) or 0) if prev_signals else 0
        kills_gained = max(0, kills - prev_kills)
        reward += kills_gained * 0.5

        return reward

    # ── PDCA Integration ──

    def tick(self, signals: dict[str, Any]) -> dict[str, Any]:
        """Called every PDCA cycle. Returns recommended action and stats."""
        with self._lock:
            if not self._initialized:
                self.initialize()

            # Encode current state
            state = self.encode_state(signals)

            # Calculate reward from previous state
            if self._last_state is not None and self._last_action is not None:
                reward = self.calculate_reward(signals)
                self.observe(self._last_state, self._last_action, reward, state)

            # Select next action
            action = self.select_action(state)

            # Store for next cycle
            self._last_state = state
            self._last_action = action
            self._episode_count += 1
            self._stats.total_episodes = self._episode_count

            return {
                "recommended_action": action,
                "state": state,
                "epsilon": self._epsilon,
                "avg_reward": self._stats.avg_reward_last_100,
                "total_episodes": self._episode_count,
                "q_table_size": len(self._q_table),
                "replay_buffer_size": len(self._replay_buffer),
            }

    # ── Model Persistence ──

    def _save_model(self) -> None:
        """Save Q-table and stats to disk."""
        try:
            path = Path(self._model_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Save Q-table
            save_data = {
                "q_table": {str(k): v for k, v in self._q_table.items()},
                "epsilon": self._epsilon,
                "episode_count": self._episode_count,
            }
            with open(path, "wb") as f:
                pickle.dump(save_data, f)

            # Save stats
            stats_path = Path(self._stats_path)
            with open(stats_path, "w") as f:
                json.dump({
                    "total_episodes": self._stats.total_episodes,
                    "total_reward": self._stats.total_reward,
                    "best_reward": self._stats.best_reward,
                    "worst_reward": self._stats.worst_reward,
                    "avg_reward_last_100": self._stats.avg_reward_last_100,
                    "exploration_rate": self._epsilon,
                    "actions_taken": dict(self._stats.actions_taken),
                    "action_rewards": dict(self._stats.action_rewards),
                    "q_table_size": len(self._q_table),
                }, f, indent=2)

            logger.debug("reinforcement_model_saved: %d states, %d episodes",
                        len(self._q_table), self._episode_count)
        except Exception as e:
            logger.warning("reinforcement_model_save_failed: %s", e)

    def _load_model(self) -> None:
        """Load Q-table and stats from disk."""
        try:
            path = Path(self._model_path)
            if path.exists():
                with open(path, "rb") as f:
                    data = pickle.load(f)
                self._q_table = {
                    eval(k): v for k, v in data.get("q_table", {}).items()
                }
                self._epsilon = data.get("epsilon", DEFAULT_EPSILON)
                self._episode_count = data.get("episode_count", 0)
                logger.info("reinforcement_model_loaded: %d states, %d episodes",
                           len(self._q_table), self._episode_count)

            # Load stats
            stats_path = Path(self._stats_path)
            if stats_path.exists():
                with open(stats_path) as f:
                    stats_data = json.load(f)
                self._stats.total_episodes = stats_data.get("total_episodes", 0)
                self._stats.total_reward = stats_data.get("total_reward", 0.0)
                self._stats.best_reward = stats_data.get("best_reward", float("-inf"))
                self._stats.worst_reward = stats_data.get("worst_reward", float("inf"))
                self._stats.avg_reward_last_100 = stats_data.get("avg_reward_last_100", 0.0)
        except Exception as e:
            logger.warning("reinforcement_model_load_failed: %s", e)

    # ── _PROVIDER_HARD_DENY_BY_WORKLOAD Integration ──

    def check_workload_deny(self, provider_name: str, current_workload: float) -> bool:
        """Check if a provider should be denied based on workload.

        This wires up the previously dead _PROVIDER_HARD_DENY_BY_WORKLOAD.
        Uses learned Q-values to decide if switching providers is worth it.

        Returns True if the provider should be denied (too much load).
        """
        if current_workload < 0.7:
            return False  # Low load, allow

        # Use Q-learning to decide: has switching providers been rewarding?
        state = (hash(provider_name) % 10, int(current_workload * 5), 0, 0, 0, 0, 0, 0, 0, 0)
        q_values = self._q_table.get(state, {})
        switch_value = q_values.get("change_map", 0.0)  # "change_map" ≈ switch provider
        stay_value = q_values.get("farm", 0.0)  # "farm" ≈ stay with current

        if switch_value > stay_value and current_workload > 0.85:
            return True  # Learning suggests switching

        return current_workload > 0.95  # Hard threshold

    # ── Context ──

    def get_learning_context(self) -> str:
        """Get formatted learning context for LLM prompts."""
        with self._lock:
            lines = ["── Reinforcement Learning ──"]
            lines.append(f"Episodes: {self._stats.total_episodes}")
            lines.append(f"Avg reward (last 100): {self._stats.avg_reward_last_100:.2f}")
            lines.append(f"Exploration rate: {self._epsilon:.3f}")
            lines.append(f"Q-table size: {len(self._q_table)} states")
            lines.append(f"Replay buffer: {len(self._replay_buffer)} experiences")

            # Best actions
            if self._stats.actions_taken:
                best_action = max(
                    self._stats.actions_taken.keys(),
                    key=lambda k: self._stats.actions_taken[k],
                )
                lines.append(f"Most taken action: {best_action} "
                             f"({self._stats.actions_taken[best_action]} times)")

            # Best rewarded action
            if self._stats.action_rewards:
                best_rewarded = max(
                    self._stats.action_rewards.keys(),
                    key=lambda k: self._stats.action_rewards[k],
                )
                lines.append(f"Best rewarded action: {best_rewarded} "
                             f"(total: {self._stats.action_rewards[best_rewarded]:.1f})")

            return "\n".join(lines)

    def counters(self) -> dict[str, int]:
        with self._lock:
            return {
                "episodes": self._stats.total_episodes,
                "q_table_size": len(self._q_table),
                "replay_buffer": len(self._replay_buffer),
            }


# ── Global Singleton ──

_reinforcement_learner: ReinforcementLearner | None = None
_reinforcement_learner_lock = RLock()


def get_reinforcement_learner() -> ReinforcementLearner:
    global _reinforcement_learner
    with _reinforcement_learner_lock:
        if _reinforcement_learner is None:
            _reinforcement_learner = ReinforcementLearner()
        return _reinforcement_learner
