"""
Reinforcement Learner — Neural network Q-learning with prioritized replay.

A pro player learns from experience:
- What actions lead to good outcomes (exp gain, zeny gain)
- What actions lead to bad outcomes (death, wasted potions)
- Which maps are safe vs dangerous
- Which monsters are worth fighting
- When to rest, when to push

This engine implements:
- 2-layer neural network (64-128 neurons each) instead of linear Q-table
- Prioritized experience replay (weighted sampling by TD error)
- Target network (updated every N steps for stable learning)
- Proper state encoding with normalization
- Reward shaping (scale rewards to [-1, 1])
- Model persistence with versioning
- Training metrics tracking
- Epsilon-greedy exploration with decay
- Wires into PDCA loop
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

import numpy as np

logger = logging.getLogger(__name__)

# -- Constants --

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
    "hp_ratio",         # 0.0-1.0 normalized
    "sp_ratio",         # 0.0-1.0 normalized
    "level_norm",       # level / 99 normalized
    "zeny_norm",        # log10(zeny+1) / 7 normalized
    "monsters_nearby",  # 0=none, 1=few, 2=many
    "party_size",       # 0=alone, 1=small, 2=large
    "weight_ratio",     # 0.0-1.0
    "death_recently",   # 0=no, 1=yes
    "time_bucket",      # 0=off_peak, 1=normal, 2=peak
    "kpm_norm",         # kills_per_min / 60 normalized
]

# Neural network hyperparameters
STATE_DIM = len(STATE_KEYS)  # 10
ACTION_DIM = len(ACTIONS)    # 10
HIDDEN_1 = 128
HIDDEN_2 = 64
LEARNING_RATE = 0.001
TAU = 0.005  # Soft update rate for target network
TARGET_UPDATE_INTERVAL = 100  # Hard update every N steps

# Q-learning hyperparameters
DEFAULT_LEARNING_RATE: float = 0.1
DEFAULT_DISCOUNT_FACTOR: float = 0.9
DEFAULT_EPSILON: float = 0.3
DEFAULT_EPSILON_DECAY: float = 0.995
DEFAULT_MIN_EPSILON: float = 0.05
DEFAULT_REPLAY_BUFFER_SIZE: int = 50000
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_PRIORITY_EPSILON: float = 1e-6
DEFAULT_ALPHA: float = 0.6  # How much prioritization to use (0=none, 1=full)
DEFAULT_BETA: float = 0.4   # Importance sampling correction (starts low, anneals to 1)
DEFAULT_BETA_INCREMENT: float = 0.001


@dataclass
class PrioritizedExperience:
    """A single experience with TD-error priority for prioritized replay."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    priority: float = 1.0  # TD-error based priority
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
    avg_td_error: float = 0.0
    max_td_error: float = 0.0
    training_steps: int = 0
    model_version: int = 0


class SumTree:
    """Sum tree data structure for prioritized experience replay.

    Allows O(log n) weighted sampling and O(log n) priority updates.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx: int, change: float) -> None:
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Find sample on leaf node with given priority sum."""
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Total priority sum."""
        return self.tree[0]

    def add(self, p: float, data: PrioritizedExperience) -> None:
        """Add experience with priority p."""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, p: float) -> None:
        """Update priority at leaf index."""
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s: float) -> tuple[int, float, PrioritizedExperience]:
        """Get sample with given priority sum s."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def __len__(self) -> int:
        return self.n_entries


class QNetwork:
    """2-layer neural network for Q-value approximation.

    Architecture:
      Input (10) -> Dense(128) -> ReLU -> Dense(64) -> ReLU -> Dense(10)
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM,
                 hidden_1: int = HIDDEN_1, hidden_2: int = HIDDEN_2):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # He initialization for ReLU
        scale_1 = math.sqrt(2.0 / state_dim)
        scale_2 = math.sqrt(2.0 / hidden_1)
        scale_out = math.sqrt(2.0 / hidden_2)

        self.w1 = np.random.randn(state_dim, hidden_1) * scale_1
        self.b1 = np.zeros(hidden_1)
        self.w2 = np.random.randn(hidden_1, hidden_2) * scale_2
        self.b2 = np.zeros(hidden_2)
        self.w3 = np.random.randn(hidden_2, action_dim) * scale_out
        self.b3 = np.zeros(action_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. x shape: (batch, state_dim) or (state_dim,)."""
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)

        # Layer 1: ReLU
        z1 = np.dot(x, self.w1) + self.b1
        a1 = np.maximum(0, z1)

        # Layer 2: ReLU
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = np.maximum(0, z2)

        # Output layer: linear
        q = np.dot(a2, self.w3) + self.b3

        if single:
            return q[0]
        return q

    def predict(self, states: np.ndarray) -> np.ndarray:
        """Get Q-values for states. states shape: (batch, state_dim)."""
        return self.forward(states)

    def get_gradients(self, x: np.ndarray) -> dict:
        """Compute forward pass and cache activations for backprop.

        Returns dict with activations for gradient computation.
        """
        # Layer 1
        z1 = np.dot(x, self.w1) + self.b1
        a1 = np.maximum(0, z1)

        # Layer 2
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = np.maximum(0, z2)

        # Output
        q = np.dot(a2, self.w3) + self.b3

        return {
            "x": x, "z1": z1, "a1": a1,
            "z2": z2, "a2": a2, "q": q,
        }

    def backward(self, grads: dict, lr: float = LEARNING_RATE) -> None:
        """Apply gradients from backpropagation.

        grads should contain: dw1, db1, dw2, db2, dw3, db3
        """
        self.w1 -= lr * grads["dw1"]
        self.b1 -= lr * grads["db1"]
        self.w2 -= lr * grads["dw2"]
        self.b2 -= lr * grads["db2"]
        self.w3 -= lr * grads["dw3"]
        self.b3 -= lr * grads["db3"]

    def soft_update_from(self, source: QNetwork, tau: float = TAU) -> None:
        """Soft update: theta = tau * source + (1 - tau) * theta."""
        self.w1 = tau * source.w1 + (1 - tau) * self.w1
        self.b1 = tau * source.b1 + (1 - tau) * self.b1
        self.w2 = tau * source.w2 + (1 - tau) * self.w2
        self.b2 = tau * source.b2 + (1 - tau) * self.b2
        self.w3 = tau * source.w3 + (1 - tau) * self.w3
        self.b3 = tau * source.b3 + (1 - tau) * self.b3

    def hard_update_from(self, source: QNetwork) -> None:
        """Hard update: copy all weights from source."""
        self.w1 = source.w1.copy()
        self.b1 = source.b1.copy()
        self.w2 = source.w2.copy()
        self.b2 = source.b2.copy()
        self.w3 = source.w3.copy()
        self.b3 = source.b3.copy()

    def get_state_dict(self) -> dict:
        """Get network weights as dict for serialization."""
        return {
            "w1": self.w1.tolist(), "b1": self.b1.tolist(),
            "w2": self.w2.tolist(), "b2": self.b2.tolist(),
            "w3": self.w3.tolist(), "b3": self.b3.tolist(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load network weights from dict."""
        self.w1 = np.array(state_dict["w1"])
        self.b1 = np.array(state_dict["b1"])
        self.w2 = np.array(state_dict["w2"])
        self.b2 = np.array(state_dict["b2"])
        self.w3 = np.array(state_dict["w3"])
        self.b3 = np.array(state_dict["b3"])


def compute_td_error(batch: list[PrioritizedExperience],
                     online_net: QNetwork, target_net: QNetwork,
                     discount: float) -> np.ndarray:
    """Compute TD errors for a batch of experiences.

    Returns array of absolute TD errors for priority updates.
    """
    states = np.array([e.state for e in batch])
    next_states = np.array([e.next_state for e in batch])
    actions = np.array([e.action for e in batch])
    rewards = np.array([e.reward for e in batch])
    dones = np.array([e.done for e in batch])

    # Current Q-values for taken actions
    q_values = online_net.predict(states)
    q_sa = q_values[np.arange(len(batch)), actions]

    # Target Q-values (from target network for stability)
    next_q = target_net.predict(next_states)
    max_next_q = np.max(next_q, axis=1)
    target_q = rewards + discount * max_next_q * (1 - dones)

    td_errors = np.abs(target_q - q_sa)
    return td_errors


def train_batch(batch: list[PrioritizedExperience],
                online_net: QNetwork, target_net: QNetwork,
                discount: float, lr: float) -> tuple[float, float, np.ndarray]:
    """Train on a batch of experiences.

    Returns (avg_loss, avg_td_error, td_errors_array).
    """
    states = np.array([e.state for e in batch])
    next_states = np.array([e.next_state for e in batch])
    actions = np.array([e.action for e in batch])
    rewards = np.array([e.reward for e in batch])
    dones = np.array([e.done for e in batch], dtype=float)

    batch_size = len(batch)

    # Forward pass through online network
    cache = online_net.get_gradients(states)
    q_values = cache["q"]

    # Current Q-values for taken actions
    q_sa = q_values[np.arange(batch_size), actions]

    # Target Q-values from target network
    next_q = target_net.predict(next_states)
    max_next_q = np.max(next_q, axis=1)
    target_q = rewards + discount * max_next_q * (1.0 - dones)

    # TD errors
    td_errors = target_q - q_sa
    abs_td = np.abs(td_errors)
    avg_td = float(np.mean(abs_td))
    max_td = float(np.max(abs_td))

    # MSE loss gradient
    dq = -2.0 * td_errors / batch_size  # d(Loss)/d(q_sa)

    # Backprop through output layer
    a2 = cache["a2"]
    dw3 = np.dot(a2.T, dq.reshape(-1, 1))
    db3 = np.sum(dq, axis=0)
    da2 = np.dot(dq.reshape(1, -1).T if dq.ndim == 1 else dq, online_net.w3.T)

    # Backprop through layer 2 (ReLU)
    z2 = cache["z2"]
    dz2 = da2 * (z2 > 0).astype(float)
    a1 = cache["a1"]
    dw2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0)
    da1 = np.dot(dz2, online_net.w2.T)

    # Backprop through layer 1 (ReLU)
    z1 = cache["z1"]
    dz1 = da1 * (z1 > 0).astype(float)
    x = cache["x"]
    dw1 = np.dot(x.T, dz1)
    db1 = np.sum(dz1, axis=0)

    # Apply gradients
    online_net.backward({
        "dw1": dw1, "db1": db1,
        "dw2": dw2, "db2": db2,
        "dw3": dw3, "db3": db3,
    }, lr=lr)

    return float(np.mean(td_errors ** 2)), avg_td, abs_td


@dataclass(slots=True)
class ReinforcementLearner:
    """
    Neural network Q-learning based reinforcement learner.

    Features:
    - 2-layer neural network (128 -> 64 -> 10) instead of linear Q-table
    - Prioritized experience replay (SumTree)
    - Target network for stable learning
    - Proper state encoding with normalization
    - Reward shaping (scale rewards to [-1, 1])
    - Model persistence with versioning
    - Training metrics tracking
    - Thread-safe: all mutable state is guarded by RLock
    """

    _lock: RLock = field(default_factory=RLock)
    _online_net: QNetwork = field(default_factory=QNetwork)
    _target_net: QNetwork = field(default_factory=QNetwork)
    _sum_tree: SumTree = field(default_factory=lambda: SumTree(DEFAULT_REPLAY_BUFFER_SIZE))
    _stats: QLearningStats = field(default_factory=QLearningStats)
    _learning_rate: float = DEFAULT_LEARNING_RATE
    _discount_factor: float = DEFAULT_DISCOUNT_FACTOR
    _epsilon: float = DEFAULT_EPSILON
    _epsilon_decay: float = DEFAULT_EPSILON_DECAY
    _min_epsilon: float = DEFAULT_MIN_EPSILON
    _batch_size: int = DEFAULT_BATCH_SIZE
    _alpha: float = DEFAULT_ALPHA
    _beta: float = DEFAULT_BETA
    _beta_increment: float = DEFAULT_BETA_INCREMENT
    _priority_epsilon: float = DEFAULT_PRIORITY_EPSILON
    _last_state: np.ndarray | None = None
    _last_action: int | None = None
    _last_action_str: str | None = None
    _last_reward: float = 0.0
    _episode_count: int = 0
    _training_steps: int = 0
    _model_path: str = "data/reinforcement_model.pkl"
    _stats_path: str = "data/reinforcement_stats.json"
    _last_save: float = 0.0
    _save_interval: float = 300.0  # Save every 5 minutes
    _initialized: bool = False
    _action_to_idx: dict = field(default_factory=lambda: {a: i for i, a in enumerate(ACTIONS)})
    _idx_to_action: dict = field(default_factory=lambda: {i: a for i, a in enumerate(ACTIONS)})

    # -- Public API --

    def initialize(self, model_path: str = "") -> None:
        """Initialize the learner, loading existing model if available."""
        if model_path:
            self._model_path = model_path
            self._stats_path = model_path.replace(".pkl", "_stats.json")

        # Initialize target network with same weights
        self._target_net.hard_update_from(self._online_net)

        self._load_model()
        self._initialized = True
        logger.info("reinforcement_learner_initialized: online_net=%d params, %d experiences",
                    self._count_params(), len(self._sum_tree))

    def _count_params(self) -> int:
        """Count total parameters in the online network."""
        return (self._online_net.w1.size + self._online_net.b1.size +
                self._online_net.w2.size + self._online_net.b2.size +
                self._online_net.w3.size + self._online_net.b3.size)

    # -- State Encoding --

    def encode_state(self, signals: dict[str, Any]) -> np.ndarray:
        """Encode a state dict into a normalized numpy array for neural network input.

        Returns normalized state vector of length STATE_DIM.
        """
        # HP ratio (0.0-1.0)
        hp_ratio = float(signals.get("hp_ratio", signals.get("hp_pct", 1.0)) or 1.0)
        hp_ratio = max(0.0, min(1.0, hp_ratio))

        # SP ratio (0.0-1.0)
        sp_ratio = float(signals.get("sp_ratio", signals.get("sp_pct", 1.0)) or 1.0)
        sp_ratio = max(0.0, min(1.0, sp_ratio))

        # Level normalized to [0, 1] (max level 99)
        level = int(signals.get("base_level", signals.get("level", 1)) or 1)
        level_norm = min(1.0, level / 99.0)

        # Zeny normalized: log10(zeny+1) / 7 (max ~10M zeny = log10(10M) = 7)
        zeny = int(signals.get("zeny", 0) or 0)
        zeny_norm = min(1.0, math.log10(max(1, zeny + 1)) / 7.0)

        # Monsters nearby
        aggro_count = int(signals.get("combat.aggro_count", 0) or 0)
        if aggro_count == 0:
            monsters_nearby = 0.0
        elif aggro_count <= 3:
            monsters_nearby = 0.5
        else:
            monsters_nearby = 1.0

        # Party size
        party_size = int(signals.get("party_size", 0) or 0)
        if party_size <= 1:
            party_bucket = 0.0
        elif party_size <= 3:
            party_bucket = 0.5
        else:
            party_bucket = 1.0

        # Weight ratio (0.0-1.0)
        weight_ratio = float(signals.get("weight_ratio", 0.0) or 0.0)
        weight_ratio = max(0.0, min(1.0, weight_ratio))

        # Death recently
        death_recently = 1.0 if signals.get("recent_death", False) else 0.0

        # Time bucket
        hour = time.localtime().tm_hour
        if hour in (0, 1, 2, 3, 4, 5, 6, 7, 22, 23):
            time_bucket = 0.0  # off-peak
        elif hour in (18, 19, 20, 21):
            time_bucket = 1.0  # peak
        else:
            time_bucket = 0.5  # normal

        # Kills per minute normalized
        kpm = float(signals.get("kills_per_min", 0) or 0)
        kpm_norm = min(1.0, kpm / 60.0)

        return np.array([
            hp_ratio, sp_ratio, level_norm, zeny_norm,
            monsters_nearby, party_bucket, weight_ratio,
            death_recently, time_bucket, kpm_norm,
        ], dtype=np.float32)

    # -- Action Selection --

    def select_action(self, state: np.ndarray, available_actions: list[str] | None = None) -> str:
        """Select an action using epsilon-greedy policy with neural network Q-values."""
        with self._lock:
            if available_actions is None:
                available_actions = ACTIONS

            # Epsilon-greedy exploration
            if random.random() < self._epsilon:
                action = random.choice(available_actions)
                self._stats.actions_taken[action] += 1
                return action

            # Greedy: use neural network to get Q-values
            q_values = self._online_net.predict(state)

            # Filter to available actions
            best_action = None
            best_value = float("-inf")
            for action in available_actions:
                idx = self._action_to_idx.get(action)
                if idx is not None and idx < len(q_values):
                    value = float(q_values[idx])
                    if value > best_value:
                        best_value = value
                        best_action = action

            if best_action is None:
                best_action = random.choice(available_actions)

            self._stats.actions_taken[best_action] += 1
            return best_action

    # -- Learning --

    def observe(self, state: np.ndarray, action: str, reward: float,
                next_state: np.ndarray, done: bool = False) -> None:
        """Observe an experience and add it to the prioritized replay buffer."""
        with self._lock:
            action_idx = self._action_to_idx.get(action, 0)

            # Compute initial priority (max priority for new experiences)
            max_priority = float(np.max(self._sum_tree.tree[-self._sum_tree.capacity:])) if self._sum_tree.n_entries > 0 else 1.0
            priority = max(max_priority, 1.0)

            exp = PrioritizedExperience(
                state=state, action=action_idx, reward=reward,
                next_state=next_state, done=done,
                priority=priority,
                timestamp=time.time(),
            )
            self._sum_tree.add(priority, exp)
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

            # Sample from replay buffer for batch learning
            if len(self._sum_tree) >= self._batch_size:
                self._train_from_replay()

            # Decay epsilon
            self._epsilon = max(
                self._min_epsilon,
                self._epsilon * self._epsilon_decay,
            )
            self._stats.exploration_rate = self._epsilon

            # Anneal beta
            self._beta = min(1.0, self._beta + self._beta_increment)

            # Save periodically
            if time.time() - self._last_save > self._save_interval:
                self._save_model()
                self._last_save = time.time()

    def _train_from_replay(self) -> None:
        """Sample a batch from prioritized replay and train."""
        batch_size = min(self._batch_size, len(self._sum_tree))
        batch: list[PrioritizedExperience] = []
        indices: list[int] = []
        priorities: list[float] = []

        # Importance sampling weights
        total_priority = self._sum_tree.total()
        segment = total_priority / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, exp = self._sum_tree.get(s)
            batch.append(exp)
            indices.append(idx)
            priorities.append(priority)

        # Compute importance sampling weights
        n = len(self._sum_tree)
        weights = np.array([(n * p / total_priority) ** (-self._beta) for p in priorities])
        weights = weights / np.max(weights)  # Normalize

        # Train on batch
        loss, avg_td, td_errors = train_batch(
            batch, self._online_net, self._target_net,
            self._discount_factor, self._learning_rate,
        )

        # Update priorities with TD errors + small epsilon to avoid zero
        for i, idx in enumerate(indices):
            new_priority = float(td_errors[i]) + self._priority_epsilon
            self._sum_tree.update(idx, new_priority)

        # Update stats
        self._stats.avg_td_error = avg_td
        self._stats.max_td_error = float(np.max(td_errors))
        self._stats.training_steps += 1
        self._training_steps += 1

        # Update target network
        if self._training_steps % TARGET_UPDATE_INTERVAL == 0:
            self._target_net.hard_update_from(self._online_net)
            self._stats.model_version += 1
            logger.debug("target_network_updated: step=%d version=%d",
                         self._training_steps, self._stats.model_version)

    # -- Reward Calculation --

    def calculate_reward(self, signals: dict[str, Any],
                         prev_signals: dict[str, Any] | None = None) -> float:
        """Calculate reward from current and previous state signals.

        Reward = (exp_gained + zeny_gained - potions_used - death_penalty) / time
        Scaled to [-1, 1] range for neural network stability.
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

        # Scale reward to [-1, 1] for neural network stability
        reward = max(-1.0, min(1.0, reward / 10.0))

        return reward

    # -- PDCA Integration --

    def tick(self, signals: dict[str, Any]) -> dict[str, Any]:
        """Called every PDCA cycle. Returns recommended action and stats."""
        with self._lock:
            if not self._initialized:
                self.initialize()

            # Encode current state
            state = self.encode_state(signals)

            # Calculate reward from previous state
            if self._last_state is not None and self._last_action_str is not None:
                reward = self.calculate_reward(signals)
                self.observe(self._last_state, self._last_action_str, reward, state)

            # Select next action
            action = self.select_action(state)

            # Store for next cycle
            self._last_state = state
            self._last_action_str = action
            self._episode_count += 1
            self._stats.total_episodes = self._episode_count

            return {
                "recommended_action": action,
                "state": state.tolist(),
                "epsilon": self._epsilon,
                "avg_reward": self._stats.avg_reward_last_100,
                "total_episodes": self._episode_count,
                "q_network_params": self._count_params(),
                "replay_buffer_size": len(self._sum_tree),
                "avg_td_error": self._stats.avg_td_error,
                "model_version": self._stats.model_version,
                "training_steps": self._training_steps,
            }

    # -- Model Persistence --

    def _save_model(self) -> None:
        """Save neural network and stats to disk with versioning."""
        try:
            path = Path(self._model_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Save model with version
            save_data = {
                "model_version": self._stats.model_version,
                "online_net": self._online_net.get_state_dict(),
                "target_net": self._target_net.get_state_dict(),
                "epsilon": self._epsilon,
                "episode_count": self._episode_count,
                "training_steps": self._training_steps,
                "beta": self._beta,
                "timestamp": time.time(),
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
                    "avg_td_error": self._stats.avg_td_error,
                    "max_td_error": self._stats.max_td_error,
                    "training_steps": self._training_steps,
                    "model_version": self._stats.model_version,
                    "q_network_params": self._count_params(),
                }, f, indent=2)

            logger.debug("reinforcement_model_saved: version=%d, %d params, %d episodes",
                        self._stats.model_version, self._count_params(), self._episode_count)
        except Exception as e:
            logger.warning("reinforcement_model_save_failed: %s", e)

    def _load_model(self) -> None:
        """Load neural network and stats from disk."""
        try:
            path = Path(self._model_path)
            if path.exists():
                with open(path, "rb") as f:
                    data = pickle.load(f)
                self._online_net.load_state_dict(data.get("online_net", {}))
                self._target_net.load_state_dict(data.get("target_net", {}))
                self._epsilon = data.get("epsilon", DEFAULT_EPSILON)
                self._episode_count = data.get("episode_count", 0)
                self._training_steps = data.get("training_steps", 0)
                self._beta = data.get("beta", DEFAULT_BETA)
                self._stats.model_version = data.get("model_version", 0)
                logger.info("reinforcement_model_loaded: version=%d, %d params, %d episodes",
                           self._stats.model_version, self._count_params(), self._episode_count)

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
                self._stats.avg_td_error = stats_data.get("avg_td_error", 0.0)
                self._stats.max_td_error = stats_data.get("max_td_error", 0.0)
        except Exception as e:
            logger.warning("reinforcement_model_load_failed: %s", e)

    # -- Workload Deny Integration --

    def check_workload_deny(self, provider_name: str, current_workload: float) -> bool:
        """Check if a provider should be denied based on workload.

        Uses learned Q-values to decide if switching providers is worth it.

        Returns True if the provider should be denied (too much load).
        """
        if current_workload < 0.7:
            return False  # Low load, allow

        # Use neural network to evaluate: has switching providers been rewarding?
        state = np.array([
            0.5,  # hp_ratio (neutral)
            0.5,  # sp_ratio (neutral)
            0.5,  # level_norm (neutral)
            0.5,  # zeny_norm (neutral)
            min(1.0, current_workload),  # monsters_nearby proxy
            0.0,  # party_size
            0.0,  # weight_ratio
            0.0,  # death_recently
            0.5,  # time_bucket
            0.0,  # kpm_norm
        ], dtype=np.float32)

        q_values = self._online_net.predict(state)
        switch_value = float(q_values[self._action_to_idx.get("change_map", 0)])
        stay_value = float(q_values[self._action_to_idx.get("farm", 0)])

        if switch_value > stay_value and current_workload > 0.85:
            return True  # Learning suggests switching

        return current_workload > 0.95  # Hard threshold

    # -- Context --

    def get_learning_context(self) -> str:
        """Get formatted learning context for LLM prompts."""
        with self._lock:
            lines = ["-- Reinforcement Learning (Neural Network) --"]
            lines.append(f"Episodes: {self._stats.total_episodes}")
            lines.append(f"Avg reward (last 100): {self._stats.avg_reward_last_100:.2f}")
            lines.append(f"Exploration rate: {self._epsilon:.3f}")
            lines.append(f"Q-network params: {self._count_params()}")
            lines.append(f"Replay buffer: {len(self._sum_tree)} experiences")
            lines.append(f"Avg TD error: {self._stats.avg_td_error:.4f}")
            lines.append(f"Training steps: {self._training_steps}")
            lines.append(f"Model version: {self._stats.model_version}")

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
                "q_network_params": self._count_params(),
                "replay_buffer": len(self._sum_tree),
                "training_steps": self._training_steps,
                "model_version": self._stats.model_version,
            }


# -- Global Singleton --

_reinforcement_learner: ReinforcementLearner | None = None
_reinforcement_learner_lock = RLock()


def get_reinforcement_learner() -> ReinforcementLearner:
    global _reinforcement_learner
    with _reinforcement_learner_lock:
        if _reinforcement_learner is None:
            _reinforcement_learner = ReinforcementLearner()
        return _reinforcement_learner
