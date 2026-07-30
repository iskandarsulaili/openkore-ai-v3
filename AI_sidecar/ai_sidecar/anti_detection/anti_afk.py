"""
Anti-AFK — random social/activity patterns to avoid idle detection.
==================================================================

Detection signal: bots that stand perfectly still for long periods, never
chat, never use emotes, never inspect other players. Humans fidget.

This module generates random "human fidget" actions:
1. Random emote usage (newer RO clients have an emote UI — /em, /bang, /hmm, etc.)
2. Random /who queries (players periodically check who's online)
3. Random equipment inspect of nearby players
4. Random chat messages (idle chatter, responses to environment)
5. Random walking to nearby spots (don't camp same exact tile)
6. Random camera zoom changes (if supported by client)
7. Random sitting/standing (players sit to AFK or regen)

All actions are configurable per profile and respect session fatigue
(fewer social actions when "tired").
"""

from __future__ import annotations

import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from ai_sidecar.anti_detection.behavior_engine import (
    BehaviorEngine,
    BehaviorProfileType,
    get_behavior_engine,
)

logger = logging.getLogger(__name__)

# ── RO Emotes (by ID) ────────────────────────────────────────────────────────
# Common RO client emotes. These are sent as packet commands.
# The actual emote ID depends on the server/version.

_EMOTES: dict[str, int] = {
    "hi": 1,
    "wave": 2,
    "hmm": 3,
    "heh": 4,
    "wah": 5,
    "ag": 6,
    "swt": 7,
    "sry": 8,
    "omg": 9,
    "..." : 10,
    "kya": 11,
    "hrr": 12,
    "heh2": 13,
    "lol": 14,
    "wah2": 15,
    "bignod": 16,
    "no": 17,
    "ok": 18,
    "love": 19,
    "shy": 20,
    "good": 21,
    "hmp": 22,
    "comfort": 23,
    "yawn": 24,
    "smile": 25,
    "tongue": 26,
    "glare": 27,
    "shock": 28,
    "..." : 29,
    "panic": 30,
    "whine": 31,
    "sigh": 32,
    "pff": 33,
    "surprised": 34,
    "worried": 35,
    "sad": 36,
    "cry": 37,
    "sweat": 38,
    "awkward": 39,
    "dodge": 40,
    "...": 41,
    "spit": 42,
    "ah": 43,
    "bold": 44,
    "...": 45,
    "pant": 46,
    "mmm": 47,
    "rumb": 48,
    "mlol": 49,
    "drool": 50,
    "...": 51,
    "ap": 52,
    "rr": 53,
    "chu": 54,
    "chu2": 55,
    "chu3": 56,
    "despair": 57,
    "...": 58,
    "chu4": 59,
    "hah": 60,
    "fsh": 61,
    "paku": 62,
    "maou": 63,
    "...": 64,
    "fury": 65,
    "...": 66,
    "puk": 67,
    "puk2": 68,
    "hua": 69,
}

# ── Idle chat phrases ────────────────────────────────────────────────────────
# Short, common RO idle chatter. Deliberately generic.

_IDLE_CHAT_PHRASES = [
    "brb",
    "afk",
    "gogogo",
    "nice",
    "lol",
    "gg",
    "ty",
    "np",
    "w8",
    "mb",
    "nt",
    "...",
    "hmm",
    "gotta go",
    "be right back",
    "sec",
    "almost there",
    "need a break",
    "getting coffee",
    "anyone need help",
    "grats",
    "nice drop",
    "gl hf",
]

# ── Action templates (anti-AFK behaviors) ────────────────────────────────────

_ANTI_AFK_ACTIONS = [
    "emote",
    "who_query",
    "inspect_player",
    "walk_to_offset",
    "sit_stand",
    "idle_chat",
    "camera_adjust",
]


@dataclass
class AntiAfkConfig:
    """Configuration for anti-AFK behavior generation."""
    enabled: bool = True

    # How often to perform an anti-AFK action
    min_interval_seconds: int = 120    # At least every 2 minutes
    max_interval_seconds: int = 600    # At most every 10 minutes

    # Action probabilities (must sum to ~1.0)
    emote_weight: float = 0.25
    who_query_weight: float = 0.20
    inspect_weight: float = 0.15
    walk_offset_weight: float = 0.15
    sit_stand_weight: float = 0.10
    idle_chat_weight: float = 0.10
    camera_adjust_weight: float = 0.05

    # Emote selection
    emote_use_random: bool = True
    emote_favorites: list[int] = field(default_factory=lambda: [1, 3, 6, 14, 18, 24])

    # Inspect settings
    inspect_max_distance: int = 15  # Max range to find players to inspect

    # Walk settings
    walk_offset_cells: list[int] = field(default_factory=lambda: [3, 8])
    walk_stay_duration_s: list[int] = field(default_factory=lambda: [5, 15])

    # Fatigue: reduce social actions over long sessions
    fatigue_reduction_enabled: bool = True
    fatigue_reduction_start_minutes: int = 90
    fatigue_max_reduction: float = 0.7  # Reduce to 30% of normal at max fatigue


@dataclass
class AntiAfkAction:
    """A generated anti-AFK action that can be dispatched to the bridge."""
    action_type: str            # emote, who_query, inspect_player, etc.
    parameters: dict[str, Any] = field(default_factory=dict)
    delay_before_ms: int = 0    # How long to wait before executing
    duration_ms: int = 0        # How long the action takes


class AntiAfkEngine:
    """Generates random social/activity patterns to avoid AFK detection.

    Usage::

        afk = AntiAfkEngine()
        action = afk.get_next_action("bot1", {"players_nearby": ["player1", "player2"]})
        if action:
            bridge_client.send_emote(action.parameters["emote_id"])
    """

    def __init__(
        self,
        engine: BehaviorEngine | None = None,
        config: AntiAfkConfig | None = None,
    ) -> None:
        self._lock = RLock()
        self._engine = engine or get_behavior_engine()
        self._config = config or AntiAfkConfig()
        self._last_action_time: float = 0.0
        self._next_action_time: float = time.time() + self._random_interval()
        self._consecutive_actions: int = 0

    # ── Public API ───────────────────────────────────────────────────────────

    def get_next_action(
        self, bot_id: str, context: dict[str, Any] | None = None
    ) -> AntiAfkAction | None:
        """Get the next anti-AFK action, or None if it's not time yet.

        Args:
            bot_id: Bot identifier.
            context: Context including 'players_nearby' list, 'map', etc.

        Returns:
            AntiAfkAction if due, None otherwise.
        """
        if not self._config.enabled:
            return None

        now = time.time()
        if now < self._next_action_time:
            return None

        # Check fatigue
        if self._is_fatigued():
            # Skip or reduce action frequency when tired
            if random.random() < self._fatigue_skip_chance():
                self._next_action_time = now + self._random_interval()
                return None

        # Select and generate action
        action = self._select_action(context or {})
        if action is None:
            # No suitable action right now — reschedule
            self._next_action_time = now + self._random_interval()
            return None

        # Update state
        self._last_action_time = now
        self._consecutive_actions += 1
        # Next action in a random interval (longer if we just did one)
        next_delay = self._random_interval() * (1.0 + self._consecutive_actions * 0.2)
        self._next_action_time = now + min(next_delay, 900)  # Cap at 15 min

        return action

    def force_action(
        self, bot_id: str, context: dict[str, Any] | None = None
    ) -> AntiAfkAction | None:
        """Force an anti-AFK action immediately (called by GM detection override)."""
        action = self._select_action(context or {})
        if action:
            self._last_action_time = time.time()
            self._next_action_time = time.time() + self._random_interval()
        return action

    def get_next_action_time(self) -> float:
        """Get the timestamp of the next scheduled anti-AFK action."""
        return self._next_action_time

    def set_config(self, config: AntiAfkConfig) -> None:
        """Update configuration at runtime."""
        with self._lock:
            self._config = config

    # ── Action selection ─────────────────────────────────────────────────────

    def _select_action(self, context: dict[str, Any]) -> AntiAfkAction | None:
        """Select a random anti-AFK action based on configured weights."""
        r = random.random()
        cumulative = 0.0

        weights = [
            (self._config.emote_weight, self._gen_emote),
            (self._config.who_query_weight, self._gen_who_query),
            (self._config.inspect_weight, self._gen_inspect),
            (self._config.walk_offset_weight, self._gen_walk_offset),
            (self._config.sit_stand_weight, self._gen_sit_stand),
            (self._config.idle_chat_weight, self._gen_idle_chat),
            (self._config.camera_adjust_weight, self._gen_camera_adjust),
        ]

        for weight, generator in weights:
            cumulative += weight
            if r <= cumulative:
                return generator(context)

        return self._gen_emote(context)  # Fallback

    # ── Action generators ────────────────────────────────────────────────────

    def _gen_emote(self, context: dict[str, Any]) -> AntiAfkAction:
        """Generate a random emote action."""
        if self._config.emote_use_random:
            emote_id = random.choice(list(_EMOTES.values()))
        else:
            emote_id = random.choice(self._config.emote_favorites)

        return AntiAfkAction(
            action_type="emote",
            parameters={"emote_id": emote_id},
            delay_before_ms=random.randint(50, 300),
            duration_ms=random.randint(1000, 3000),
        )

    def _gen_who_query(self, context: dict[str, Any]) -> AntiAfkAction:
        """Generate a /who query action."""
        search_terms = ["", "a", "b", "c", "g", "m", "p", "r", "s", "t"]
        term = random.choice(search_terms)
        return AntiAfkAction(
            action_type="who_query",
            parameters={"search_term": term},
            delay_before_ms=random.randint(100, 400),
            duration_ms=random.randint(500, 1500),
        )

    def _gen_inspect(self, context: dict[str, Any]) -> AntiAfkAction:
        """Generate a player inspect action."""
        players_nearby = context.get("players_nearby", [])
        if not players_nearby:
            # Fall back to emote
            return self._gen_emote(context)

        target = random.choice(players_nearby)
        return AntiAfkAction(
            action_type="inspect_player",
            parameters={"player_name": target},
            delay_before_ms=random.randint(200, 600),
            duration_ms=random.randint(2000, 5000),
        )

    def _gen_walk_offset(self, context: dict[str, Any]) -> AntiAfkAction:
        """Generate a short walk to a nearby offset (human fidget)."""
        min_cells, max_cells = self._config.walk_offset_cells
        offset_x = random.randint(min_cells, max_cells) * random.choice([-1, 1])
        offset_y = random.randint(min_cells, max_cells) * random.choice([-1, 1])
        stay_duration = random.randint(*self._config.walk_stay_duration_s)

        return AntiAfkAction(
            action_type="walk_to_offset",
            parameters={
                "dx": offset_x,
                "dy": offset_y,
                "stay_duration_s": stay_duration,
            },
            delay_before_ms=random.randint(100, 400),
            duration_ms=stay_duration * 1000,
        )

    def _gen_sit_stand(self, context: dict[str, Any]) -> AntiAfkAction:
        """Generate a sit/stand action (players sit when idle)."""
        action = random.choice(["sit", "stand"])
        return AntiAfkAction(
            action_type="sit_stand",
            parameters={"action": action},
            delay_before_ms=random.randint(50, 200),
            duration_ms=random.randint(500, 2000),
        )

    def _gen_idle_chat(self, context: dict[str, Any]) -> AntiAfkAction:
        """Generate an idle chat message."""
        message = random.choice(_IDLE_CHAT_PHRASES)
        return AntiAfkAction(
            action_type="idle_chat",
            parameters={"message": message},
            delay_before_ms=random.randint(200, 800),
            duration_ms=random.randint(1000, 3000),
        )

    def _gen_camera_adjust(self, context: dict[str, Any]) -> AntiAfkAction:
        """Generate a camera adjustment action."""
        # Simulate the player adjusting their camera view
        return AntiAfkAction(
            action_type="camera_adjust",
            parameters={
                "zoom": random.uniform(0.5, 1.5),
                "rotation": random.randint(0, 360),
            },
            delay_before_ms=random.randint(50, 200),
            duration_ms=random.randint(500, 1500),
        )

    # ── Fatigue helpers ──────────────────────────────────────────────────────

    def _is_fatigued(self) -> bool:
        """Check if session fatigue is active."""
        if not self._config.fatigue_reduction_enabled:
            return False
        elapsed_minutes = (time.time() - self._last_action_time) / 60.0
        return elapsed_minutes > self._config.fatigue_reduction_start_minutes

    def _fatigue_skip_chance(self) -> float:
        """Probability of skipping an anti-AFK action due to fatigue."""
        elapsed_minutes = (time.time() - self._last_action_time) / 60.0
        if elapsed_minutes <= self._config.fatigue_reduction_start_minutes:
            return 0.0
        fatigue_hours = (
            elapsed_minutes - self._config.fatigue_reduction_start_minutes
        ) / 60.0
        skip = min(
            self._config.fatigue_max_reduction,
            fatigue_hours * 0.1,
        )
        return skip

    def _random_interval(self) -> float:
        """Get a random interval between anti-AFK actions."""
        return random.uniform(
            self._config.min_interval_seconds,
            self._config.max_interval_seconds,
        )


# ── Global singleton ─────────────────────────────────────────────────────────

_afk_engine: AntiAfkEngine | None = None
_afk_engine_lock = RLock()


def get_anti_afk_engine(
    engine: BehaviorEngine | None = None,
) -> AntiAfkEngine:
    """Get or create the global AntiAfkEngine singleton."""
    global _afk_engine
    with _afk_engine_lock:
        if _afk_engine is None:
            _afk_engine = AntiAfkEngine(engine=engine)
        return _afk_engine
