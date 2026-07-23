"""
Bot Personality Profiles — each bot acts like a different person.

Instead of identical behavior across all bots, each bot has a unique
personality profile that affects chat frequency, typing style, social
behavior, movement style, and combat style.

Personalities are data-driven from observed player behavior patterns.

Extended with persistent state: interaction history, reputation tracking,
relationship management, persistent storage, and backstory generation.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. InteractionHistory — tracks all player interactions with timestamps
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Interaction:
    """A single interaction between the bot and another player."""
    player_name: str
    interaction_type: str  # chat, whisper, party_invite, party_join, party_leave, trade, trade_complete, pvp, heal, buff, kill_steal, etc.
    message: str = ""
    timestamp: float = 0.0
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    @property
    def datetime_str(self) -> str:
        return datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Interaction:
        return cls(**data)


class InteractionHistory:
    """Tracks all player interactions with timestamps and context.

    Maintains a rolling window of recent interactions per player,
    plus a full chronological log for analysis.
    """

    def __init__(self, max_per_player: int = 100, max_total: int = 10000):
        self._lock = RLock()
        self._max_per_player = max_per_player
        self._max_total = max_total
        # Per-player interaction lists
        self._by_player: dict[str, list[Interaction]] = defaultdict(list)
        # Full chronological log (for backstory generation)
        self._chronological: list[Interaction] = []
        # Summary stats
        self._stats: dict[str, Any] = defaultdict(lambda: defaultdict(int))

    def add_interaction(self, player_name: str, interaction_type: str,
                        message: str = "", context: dict[str, Any] | None = None) -> Interaction:
        """Record a new interaction."""
        interaction = Interaction(
            player_name=player_name,
            interaction_type=interaction_type,
            message=message,
            context=context or {},
        )
        with self._lock:
            # Add to per-player list
            player_list = self._by_player[player_name]
            player_list.append(interaction)
            # Trim per-player if over limit
            if len(player_list) > self._max_per_player:
                self._by_player[player_name] = player_list[-self._max_per_player:]

            # Add to chronological log
            self._chronological.append(interaction)
            # Trim total if over limit
            if len(self._chronological) > self._max_total:
                self._chronological = self._chronological[-self._max_total:]

            # Update stats
            self._stats[player_name][interaction_type] += 1

        return interaction

    def get_interactions_with(self, player_name: str,
                               limit: int = 50) -> list[Interaction]:
        """Get recent interactions with a specific player."""
        with self._lock:
            player_list = self._by_player.get(player_name, [])
            return player_list[-limit:] if player_list else []

    def get_interactions_by_type(self, interaction_type: str,
                                  limit: int = 50) -> list[Interaction]:
        """Get recent interactions of a specific type."""
        with self._lock:
            result = [i for i in self._chronological if i.interaction_type == interaction_type]
            return result[-limit:] if result else []

    def get_all_interactions(self, limit: int = 100) -> list[Interaction]:
        """Get the most recent interactions across all players."""
        with self._lock:
            return self._chronological[-limit:] if self._chronological else []

    def get_player_summary(self, player_name: str) -> dict[str, Any]:
        """Get a summary of interactions with a player."""
        with self._lock:
            stats = dict(self._stats.get(player_name, {}))
            total = sum(stats.values())
            last_interaction = self._by_player[player_name][-1] if self._by_player.get(player_name) else None
            return {
                "player_name": player_name,
                "total_interactions": total,
                "by_type": stats,
                "last_interaction_type": last_interaction.interaction_type if last_interaction else None,
                "last_interaction_time": last_interaction.timestamp if last_interaction else None,
                "last_interaction_message": last_interaction.message if last_interaction else "",
            }

    def get_all_players(self) -> list[str]:
        """Get all players we've interacted with."""
        with self._lock:
            return list(self._by_player.keys())

    def get_total_interaction_count(self) -> int:
        """Get total number of interactions recorded."""
        with self._lock:
            return len(self._chronological)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON persistence."""
        with self._lock:
            return {
                "by_player": {
                    player: [i.to_dict() for i in interactions]
                    for player, interactions in self._by_player.items()
                },
                "chronological": [i.to_dict() for i in self._chronological],
                "stats": dict(self._stats),
            }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InteractionHistory:
        """Deserialize from dict."""
        history = cls()
        with history._lock:
            for player, interactions in data.get("by_player", {}).items():
                history._by_player[player] = [Interaction.from_dict(i) for i in interactions]
            history._chronological = [Interaction.from_dict(i) for i in data.get("chronological", [])]
            history._stats = defaultdict(lambda: defaultdict(int), {
                k: defaultdict(int, v) for k, v in data.get("stats", {}).items()
            })
        return history


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ReputationTracker — per-player reputation that changes based on interactions
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ReputationEntry:
    """Reputation data for a single player."""
    player_name: str
    score: float = 0.0  # -100 to +100
    first_encounter: float = 0.0
    last_encounter: float = 0.0
    encounter_count: int = 0
    positive_interactions: int = 0
    negative_interactions: int = 0
    tags: list[str] = field(default_factory=list)  # e.g. "helpful", "scammer", "generous", "hostile"

    @property
    def sentiment(self) -> str:
        """Get the sentiment label for this reputation."""
        if self.score >= 50:
            return "beloved"
        elif self.score >= 20:
            return "friendly"
        elif self.score >= 5:
            return "neutral_positive"
        elif self.score >= -5:
            return "neutral"
        elif self.score >= -20:
            return "neutral_negative"
        elif self.score >= -50:
            return "unfriendly"
        else:
            return "hostile"

    @property
    def trust_level(self) -> str:
        """How much the bot trusts this player."""
        if self.score >= 30:
            return "trusted"
        elif self.score >= 10:
            return "likely_trusted"
        elif self.score >= -10:
            return "uncertain"
        elif self.score >= -30:
            return "suspicious"
        else:
            return "distrusted"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["sentiment"] = self.sentiment
        d["trust_level"] = self.trust_level
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReputationEntry:
        # Strip computed properties that aren't dataclass fields
        clean = {k: v for k, v in data.items() if k not in ("sentiment", "trust_level")}
        return cls(**clean)


# Interaction type -> reputation score delta
REPUTATION_DELTAS: dict[str, float] = {
    # Positive interactions
    "greeting": 1.0,
    "chat": 0.5,
    "whisper": 0.5,
    "party_invite": 2.0,
    "party_join": 1.0,
    "party_help": 3.0,
    "trade": 1.0,
    "trade_complete": 2.0,
    "heal": 5.0,
    "buff": 3.0,
    "resurrect": 8.0,
    "gift": 10.0,
    "compliment": 4.0,
    "shared_drop": 3.0,
    "party_invite_accepted": 2.0,
    "trade_accepted": 2.0,

    # Negative interactions
    "insult": -5.0,
    "scam": -20.0,
    "kill_steal": -10.0,
    "pvp_attack": -8.0,
    "steal": -15.0,
    "spam": -3.0,
    "party_invite_declined": -0.5,
    "trade_declined": -0.5,
    "ignore": -1.0,
    "harassment": -15.0,
    "griefing": -12.0,
    "scam_attempt": -25.0,
}


class ReputationTracker:
    """Tracks per-player reputation that evolves based on interactions.

    Reputation scores range from -100 (hostile) to +100 (beloved).
    Scores decay slowly over time toward neutral if no new interactions occur.
    """

    def __init__(self, decay_rate: float = 0.1, decay_interval_hours: float = 24.0):
        self._lock = RLock()
        self._reputations: dict[str, ReputationEntry] = {}
        self._decay_rate = decay_rate  # points lost per interval
        self._decay_interval = decay_interval_hours * 3600  # convert to seconds
        self._last_decay_check: float = time.time()

    def get_reputation(self, player_name: str) -> ReputationEntry:
        """Get the reputation entry for a player (creates if new)."""
        self._apply_decay()
        with self._lock:
            if player_name not in self._reputations:
                self._reputations[player_name] = ReputationEntry(
                    player_name=player_name,
                    first_encounter=time.time(),
                    last_encounter=time.time(),
                )
            return self._reputations[player_name]

    def record_interaction(self, player_name: str, interaction_type: str) -> float:
        """Record an interaction and update reputation. Returns the new score."""
        delta = REPUTATION_DELTAS.get(interaction_type, 0.0)
        self._apply_decay()

        with self._lock:
            entry = self.get_reputation(player_name)
            entry.last_encounter = time.time()
            entry.encounter_count += 1

            if delta > 0:
                entry.positive_interactions += 1
            elif delta < 0:
                entry.negative_interactions += 1

            # Apply delta with bounds
            entry.score = max(-100.0, min(100.0, entry.score + delta))

            # Auto-tag based on interaction type
            self._auto_tag(entry, interaction_type, delta)

            logger.debug("reputation_update: player=%s type=%s delta=%.1f score=%.1f",
                        player_name, interaction_type, delta, entry.score)
            return entry.score

    def _auto_tag(self, entry: ReputationEntry, interaction_type: str, delta: float) -> None:
        """Automatically assign tags based on interaction patterns."""
        if delta >= 5 and "generous" not in entry.tags:
            entry.tags.append("generous")
        if delta <= -10 and "hostile" not in entry.tags:
            entry.tags.append("hostile")
        if interaction_type == "scam" and "scammer" not in entry.tags:
            entry.tags.append("scammer")
        if interaction_type == "heal" and "helpful" not in entry.tags:
            entry.tags.append("helpful")
        if interaction_type == "kill_steal" and "kser" not in entry.tags:
            entry.tags.append("kser")

    def _apply_decay(self) -> None:
        """Slowly decay reputation scores toward neutral over time."""
        now = time.time()
        if now - self._last_decay_check < self._decay_interval:
            return

        with self._lock:
            self._last_decay_check = now
            for entry in self._reputations.values():
                if entry.score > 0:
                    entry.score = max(0, entry.score - self._decay_rate)
                elif entry.score < 0:
                    entry.score = min(0, entry.score + self._decay_rate)

    def get_all_reputations(self) -> dict[str, ReputationEntry]:
        """Get all reputation entries."""
        self._apply_decay()
        with self._lock:
            return dict(self._reputations)

    def get_trusted_players(self, min_score: float = 10.0) -> list[str]:
        """Get players with reputation above a threshold."""
        self._apply_decay()
        with self._lock:
            return [name for name, entry in self._reputations.items()
                    if entry.score >= min_score]

    def get_distrusted_players(self, max_score: float = -10.0) -> list[str]:
        """Get players with reputation below a threshold."""
        self._apply_decay()
        with self._lock:
            return [name for name, entry in self._reputations.items()
                    if entry.score <= max_score]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON persistence."""
        with self._lock:
            return {
                "reputations": {name: entry.to_dict() for name, entry in self._reputations.items()},
                "last_decay_check": self._last_decay_check,
            }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReputationTracker:
        """Deserialize from dict."""
        tracker = cls()
        with tracker._lock:
            tracker._reputations = {
                name: ReputationEntry.from_dict(entry)
                for name, entry in data.get("reputations", {}).items()
            }
            tracker._last_decay_check = data.get("last_decay_check", time.time())
        return tracker


# ═══════════════════════════════════════════════════════════════════════════════
# 3. RelationshipManager — friends, enemies, trade partners with levels
# ═══════════════════════════════════════════════════════════════════════════════

class RelationshipLevel:
    """Relationship level constants."""
    HATED = -3
    ENEMY = -2
    DISLIKED = -1
    NEUTRAL = 0
    ACQUAINTANCE = 1
    FRIEND = 2
    CLOSE_FRIEND = 3
    BEST_FRIEND = 4

    LABELS = {
        HATED: "hated",
        ENEMY: "enemy",
        DISLIKED: "disliked",
        NEUTRAL: "neutral",
        ACQUAINTANCE: "acquaintance",
        FRIEND: "friend",
        CLOSE_FRIEND: "close_friend",
        BEST_FRIEND: "best_friend",
    }

    @classmethod
    def label(cls, level: int) -> str:
        return cls.LABELS.get(level, "unknown")


@dataclass
class Relationship:
    """A relationship between the bot and another player."""
    player_name: str
    level: int = RelationshipLevel.NEUTRAL
    relationship_type: str = "neutral"  # friend, enemy, trade_partner, party_member, neutral
    affinity: float = 0.0  # -100 to +100, raw affinity score
    first_met: float = 0.0
    last_interaction: float = 0.0
    interaction_count: int = 0
    times_parted: int = 0
    times_traded: int = 0
    times_pvped: int = 0
    notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.first_met == 0.0:
            self.first_met = time.time()
        if self.last_interaction == 0.0:
            self.last_interaction = time.time()

    @property
    def level_label(self) -> str:
        return RelationshipLevel.label(self.level)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["level_label"] = self.level_label
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Relationship:
        # Strip computed properties that aren't dataclass fields
        clean = {k: v for k, v in data.items() if k not in ("level_label",)}
        return cls(**clean)


class RelationshipManager:
    """Manages relationships between the bot and other players.

    Tracks friends, enemies, trade partners, and party members with
    relationship levels that evolve over time.
    """

    def __init__(self):
        self._lock = RLock()
        self._relationships: dict[str, Relationship] = {}

    def get_relationship(self, player_name: str) -> Relationship:
        """Get the relationship with a player (creates if new)."""
        with self._lock:
            if player_name not in self._relationships:
                self._relationships[player_name] = Relationship(player_name=player_name)
            return self._relationships[player_name]

    def update_relationship(self, player_name: str, interaction_type: str) -> Relationship:
        """Update a relationship based on an interaction type."""
        with self._lock:
            rel = self.get_relationship(player_name)
            rel.last_interaction = time.time()
            rel.interaction_count += 1

            # Determine relationship type and affinity change
            type_changes = {
                "friend": ("friend", 5.0),
                "party_invite_accepted": ("party_member", 3.0),
                "party_help": ("party_member", 4.0),
                "trade_complete": ("trade_partner", 3.0),
                "trade": ("trade_partner", 1.0),
                "gift": ("friend", 8.0),
                "heal": ("friend", 4.0),
                "buff": ("friend", 2.0),
                "resurrect": ("friend", 6.0),
                "compliment": ("friend", 3.0),
                "shared_drop": ("friend", 3.0),
                "insult": ("enemy", -5.0),
                "pvp_attack": ("enemy", -8.0),
                "kill_steal": ("enemy", -10.0),
                "scam": ("enemy", -20.0),
                "steal": ("enemy", -15.0),
                "harassment": ("enemy", -15.0),
                "griefing": ("enemy", -12.0),
                "scam_attempt": ("enemy", -25.0),
            }

            if interaction_type in type_changes:
                new_type, affinity_delta = type_changes[interaction_type]
                rel.relationship_type = new_type
                rel.affinity = max(-100.0, min(100.0, rel.affinity + affinity_delta))

                # Track specific counters
                if interaction_type == "trade_complete":
                    rel.times_traded += 1
                elif interaction_type == "pvp_attack":
                    rel.times_pvped += 1

            # Update relationship level based on affinity
            rel.level = self._affinity_to_level(rel.affinity)

            return rel

    def _affinity_to_level(self, affinity: float) -> int:
        """Convert raw affinity to relationship level."""
        if affinity >= 80:
            return RelationshipLevel.BEST_FRIEND
        elif affinity >= 50:
            return RelationshipLevel.CLOSE_FRIEND
        elif affinity >= 20:
            return RelationshipLevel.FRIEND
        elif affinity >= 5:
            return RelationshipLevel.ACQUAINTANCE
        elif affinity >= -5:
            return RelationshipLevel.NEUTRAL
        elif affinity >= -20:
            return RelationshipLevel.DISLIKED
        elif affinity >= -50:
            return RelationshipLevel.ENEMY
        else:
            return RelationshipLevel.HATED

    def get_friends(self, min_level: int = RelationshipLevel.FRIEND) -> list[Relationship]:
        """Get all friends above a certain level."""
        with self._lock:
            return [r for r in self._relationships.values() if r.level >= min_level]

    def get_enemies(self, max_level: int = RelationshipLevel.ENEMY) -> list[Relationship]:
        """Get all enemies below a certain level."""
        with self._lock:
            return [r for r in self._relationships.values() if r.level <= max_level]

    def get_trade_partners(self) -> list[Relationship]:
        """Get all trade partners."""
        with self._lock:
            return [r for r in self._relationships.values()
                    if r.relationship_type == "trade_partner" and r.times_traded > 0]

    def get_party_members(self) -> list[Relationship]:
        """Get all party members."""
        with self._lock:
            return [r for r in self._relationships.values()
                    if r.relationship_type == "party_member"]

    def add_note(self, player_name: str, note: str) -> None:
        """Add a note about a player."""
        with self._lock:
            rel = self.get_relationship(player_name)
            rel.notes.append(note)

    def get_all_relationships(self) -> dict[str, Relationship]:
        """Get all relationships."""
        with self._lock:
            return dict(self._relationships)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON persistence."""
        with self._lock:
            return {
                "relationships": {name: rel.to_dict() for name, rel in self._relationships.items()},
            }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RelationshipManager:
        """Deserialize from dict."""
        mgr = cls()
        with mgr._lock:
            mgr._relationships = {
                name: Relationship.from_dict(rel)
                for name, rel in data.get("relationships", {}).items()
            }
        return mgr


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PersistentStorage — save/load from JSON file
# ═══════════════════════════════════════════════════════════════════════════════

class PersistentStorage:
    """Save/load personality state to/from a JSON file.

    All data persists across bot restarts. Uses atomic writes to prevent
    corruption from crashes during save.
    """

    def __init__(self, file_path: str | Path | None = None):
        self._lock = RLock()
        self._file_path = Path(file_path) if file_path else self._default_path()
        self._auto_save_interval: float = 300.0  # 5 minutes
        self._last_auto_save: float = time.time()
        self._dirty: bool = False

    @staticmethod
    def _default_path() -> Path:
        """Get the default data directory for personality state."""
        # Try common data directories
        candidates = [
            Path("data/personality_state.json"),
            Path.home() / ".openkore" / "personality_state.json",
            Path("personality_state.json"),
        ]
        for path in candidates:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                return path
            except (OSError, PermissionError):
                continue
        return candidates[0]

    def set_file_path(self, file_path: str | Path) -> None:
        """Set a custom file path for persistence."""
        with self._lock:
            self._file_path = Path(file_path)
            self._file_path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, data: dict[str, Any]) -> bool:
        """Save data to JSON file atomically."""
        with self._lock:
            try:
                self._file_path.parent.mkdir(parents=True, exist_ok=True)

                # Atomic write: write to temp file, then rename
                temp_path = self._file_path.with_suffix(".json.tmp")
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str)
                temp_path.rename(self._file_path)

                self._last_auto_save = time.time()
                self._dirty = False
                logger.info("personality_state_saved: path=%s size=%d",
                           self._file_path, len(json.dumps(data)))
                return True
            except (OSError, PermissionError, json.JSONEncodeError) as e:
                logger.error("personality_state_save_failed: path=%s error=%s",
                            self._file_path, e)
                return False

    def load(self) -> dict[str, Any] | None:
        """Load data from JSON file. Returns None if file doesn't exist."""
        with self._lock:
            if not self._file_path.exists():
                logger.info("personality_state_not_found: path=%s", self._file_path)
                return None

            try:
                with open(self._file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info("personality_state_loaded: path=%s", self._file_path)
                return data
            except (json.JSONDecodeError, OSError) as e:
                logger.error("personality_state_load_failed: path=%s error=%s",
                            self._file_path, e)
                return None

    def auto_save(self, data: dict[str, Any], force: bool = False) -> bool:
        """Auto-save if enough time has passed since last save."""
        if force:
            return self.save(data)

        now = time.time()
        if now - self._last_auto_save >= self._auto_save_interval:
            return self.save(data)
        return True

    def mark_dirty(self) -> None:
        """Mark the state as dirty (needs saving)."""
        with self._lock:
            self._dirty = True

    @property
    def is_dirty(self) -> bool:
        with self._lock:
            return self._dirty

    @property
    def file_path(self) -> Path:
        return self._file_path


# ═══════════════════════════════════════════════════════════════════════════════
# 5. BackstoryGenerator — generates consistent backstory from personality + history
# ═══════════════════════════════════════════════════════════════════════════════

BACKSTORY_TEMPLATES: dict[str, dict[str, Any]] = {
    "talkative": {
        "archetypes": ["The Social Butterfly", "The Town Crier", "The Gossip"],
        "origins": [
            "Started playing MMOs to socialize, found the grind was just background noise for conversation.",
            "Was a forum moderator in another life, now brings that energy to every public channel.",
            "Grew up in a large family where silence was suspicious — old habits die hard.",
        ],
        "personality_quirks": [
            "Has a habit of narrating their own actions out loud.",
            "Types with the enthusiasm of someone who just discovered the enter key.",
            "Treats every party like a podcast episode.",
        ],
        "secrets": [
            "Actually reads every message twice before sending, despite the casual tone.",
            "Keeps a mental list of who owes them a response.",
            "Has a second, completely silent account for when they need a break from talking.",
        ],
    },
    "quiet": {
        "archetypes": ["The Silent Guardian", "The Observer", "The Lone Wolf"],
        "origins": [
            "Learned early that talking less means fewer mistakes.",
            "Was burned by a guild betrayal years ago and never fully recovered socially.",
            "Prefers to let actions speak — words are cheap in a world where anyone can lie.",
        ],
        "personality_quirks": [
            "Responds with single words or emotes when forced to talk.",
            "Has a surprisingly elaborate inner monologue that never sees chat.",
            "Types perfectly when they do speak — no typos, no slang.",
        ],
        "secrets": [
            "Has a private blog where they write detailed stories about their adventures.",
            "Actually enjoys company but fears being a burden.",
            "Knows more about the server's drama than the town crier types, just doesn't share.",
        ],
    },
    "noob": {
        "archetypes": ["The Eternal Newbie", "The Curious Explorer", "The Accidental Tourist"],
        "origins": [
            "This is their first MMO and everything is still magical and confusing.",
            "Played years ago and everything has changed since they returned.",
            "Was carried through their previous games by friends who've since quit.",
        ],
        "personality_quirks": [
            "Asks for directions in maps they've visited fifty times.",
            "Types slowly because they're still learning the keyboard layout for gaming.",
            "Gets genuinely excited about common drops that veterans ignore.",
        ],
        "secrets": [
            "Is actually much better at the game than they let on — the noob act is partly for social warmth.",
            "Has a max-level character on another server but enjoys the fresh start experience.",
            "Knows more about the game's lore than most veterans, just not the mechanics.",
        ],
    },
    "efficient": {
        "archetypes": ["The Optimizer", "The Efficiency Demon", "The Spreadsheet Warrior"],
        "origins": [
            "Came from competitive gaming where every second counts.",
            "Has limited playtime and maximizes every minute of it.",
            "Treats the game as a system to be solved, not a world to be lived in.",
        ],
        "personality_quirks": [
            "Times their grinding sessions with military precision.",
            "Has a sixth sense for when a party member is about to afk.",
            "Gets visibly annoyed by inefficient pathing — even in other players.",
        ],
        "secrets": [
            "Actually has a second monitor with a detailed spreadsheet open at all times.",
            "Takes occasional 'inefficient' breaks to watch the sunset in-game — their guilty pleasure.",
            "Has a soft spot for noobs and secretly helps them optimize their builds.",
        ],
    },
    "social": {
        "archetypes": ["The Party Leader", "The Guild Heart", "The Social Coordinator"],
        "origins": [
            "Discovered that MMOs are 10% game and 90% people — optimized for the 90%.",
            "Was a team captain in school sports, now channels that into party coordination.",
            "Believes the best loot is the friends made along the way (but also wants the loot).",
        ],
        "personality_quirks": [
            "Uses party chat like a group therapy session.",
            "Remembers everyone's birthday and in-game achievements.",
            "Has a mental map of who synergizes well with whom in a party.",
        ],
        "secrets": [
            "Keeps a private ranking of who they'd save first in a wipe.",
            "Has a 'solo mode' alt character for when they need a break from people.",
            "Actually prefers solo play but is addicted to the satisfaction of a well-run party.",
        ],
    },
}


class BackstoryGenerator:
    """Generates consistent backstories from personality profiles and interaction history.

    Backstories are composed of:
    - Archetype (based on personality)
    - Origin story (how they started playing)
    - Personality quirks (unique behavioral traits)
    - Secrets (things the bot 'knows' about itself)
    - Recent history (derived from actual interactions)
    """

    def __init__(self):
        self._lock = RLock()
        self._generated_backstories: dict[str, dict[str, Any]] = {}

    def generate_backstory(self, bot_id: str, personality_name: str,
                           interaction_history: InteractionHistory | None = None,
                           reputation_tracker: ReputationTracker | None = None,
                           relationship_manager: RelationshipManager | None = None) -> dict[str, Any]:
        """Generate or retrieve a consistent backstory for a bot."""
        with self._lock:
            # Return cached backstory if it exists (consistency across restarts)
            if bot_id in self._generated_backstories:
                return self._update_recent_history(
                    self._generated_backstories[bot_id],
                    interaction_history,
                    reputation_tracker,
                    relationship_manager,
                )

            # Get template for this personality
            template = BACKSTORY_TEMPLATES.get(personality_name, BACKSTORY_TEMPLATES["quiet"])

            backstory = {
                "bot_id": bot_id,
                "personality": personality_name,
                "archetype": random.choice(template["archetypes"]),
                "origin": random.choice(template["origins"]),
                "quirks": random.sample(template["personality_quirks"],
                                        k=min(2, len(template["personality_quirks"]))),
                "secrets": random.sample(template["secrets"],
                                         k=min(1, len(template["secrets"]))),
                "recent_history": [],
                "known_players": [],
                "generated_at": time.time(),
            }

            self._generated_backstories[bot_id] = backstory
            logger.info("backstory_generated: bot=%s archetype=%s", bot_id, backstory["archetype"])

            return self._update_recent_history(
                backstory, interaction_history, reputation_tracker, relationship_manager
            )

    def _update_recent_history(
        self,
        backstory: dict[str, Any],
        interaction_history: InteractionHistory | None,
        reputation_tracker: ReputationTracker | None,
        relationship_manager: RelationshipManager | None,
    ) -> dict[str, Any]:
        """Update the backstory with recent interaction history."""
        recent_history = []
        known_players = []

        if interaction_history:
            recent = interaction_history.get_all_interactions(limit=10)
            for interaction in recent:
                recent_history.append(
                    f"{interaction.datetime_str}: {interaction.interaction_type} "
                    f"with {interaction.player_name}"
                    + (f" — \"{interaction.message[:60]}\"" if interaction.message else "")
                )

            known_players = interaction_history.get_all_players()

        if reputation_tracker and known_players:
            # Add reputation context
            for player in known_players[:5]:  # Top 5
                entry = reputation_tracker.get_reputation(player)
                recent_history.append(
                    f"Reputation with {player}: {entry.score:.0f} ({entry.sentiment})"
                )

        if relationship_manager and known_players:
            for player in known_players[:3]:
                rel = relationship_manager.get_relationship(player)
                recent_history.append(
                    f"Relationship with {player}: {rel.level_label} "
                    f"(affinity: {rel.affinity:.0f})"
                )

        backstory["recent_history"] = recent_history[-15:]  # Keep last 15 entries
        backstory["known_players"] = known_players
        return backstory

    def get_backstory(self, bot_id: str) -> dict[str, Any] | None:
        """Get the cached backstory for a bot."""
        with self._lock:
            return self._generated_backstories.get(bot_id)

    def set_backstory(self, bot_id: str, backstory: dict[str, Any]) -> None:
        """Set a backstory (e.g., when loading from persistent storage)."""
        with self._lock:
            self._generated_backstories[bot_id] = backstory

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON persistence."""
        with self._lock:
            return {
                "backstories": self._generated_backstories,
            }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BackstoryGenerator:
        """Deserialize from dict."""
        gen = cls()
        with gen._lock:
            gen._generated_backstories = data.get("backstories", {})
        return gen


# ═══════════════════════════════════════════════════════════════════════════════
# Original PersonalityProfile and PERSONALITIES (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class PersonalityProfile:
    """A unique personality profile for a bot."""
    name: str
    description: str

    # Chat behavior
    chat_frequency_per_hour: tuple[float, float] = (0, 2)  # min, max messages/hour
    typo_rate: float = 0.1  # 0.0-1.0, probability of typo per message
    capitalization: str = "normal"  # normal, lowercase, uppercase, random
    emoji_usage: float = 0.0  # 0.0-1.0
    slang_words: list[str] = field(default_factory=list)

    # Social behavior
    social_initiative: float = 0.0  # 0.0-1.0, probability of starting conversation
    social_responsiveness: float = 0.5  # 0.0-1.0, probability of responding
    party_preference: float = 0.3  # 0.0-1.0, probability of accepting party invite
    trade_preference: float = 0.3  # 0.0-1.0, probability of accepting trade

    # Movement style
    movement_quality: float = 0.9  # 0.0-1.0, 1.0 = optimal pathing
    wander_chance: float = 0.0  # 0.0-1.0, probability of taking scenic route
    afk_chance: float = 0.0  # 0.0-1.0, probability of standing still

    # Combat style
    combat_aggression: float = 0.7  # 0.0-1.0, 1.0 = attack everything
    flee_threshold: float = 0.2  # HP% below which to flee
    skill_spam_rate: float = 0.5  # 0.0-1.0, probability of using skills vs auto-attack
    target_switching: float = 0.3  # 0.0-1.0, probability of switching targets

    # Mistake patterns
    wrong_target_chance: float = 0.0
    pathing_error_chance: float = 0.0
    cast_cancel_chance: float = 0.0
    wrong_item_use_chance: float = 0.0

    # Time patterns
    active_hours: tuple[int, int] = (0, 23)  # 24h format
    break_frequency_minutes: tuple[int, int] = (0, 0)  # min, max minutes between breaks
    break_duration_minutes: tuple[int, int] = (0, 0)  # min, max break duration


# ── Pre-defined Personality Profiles ──────────────────────────────────────

PERSONALITIES: dict[str, PersonalityProfile] = {
    "talkative": PersonalityProfile(
        name="Talkative",
        description="Friendly, chatty player who types in public chat",
        chat_frequency_per_hour=(5, 15),
        typo_rate=0.15,
        capitalization="normal",
        emoji_usage=0.3,
        slang_words=["lol", "gg", "brb", "afk", "omw", "ty", "np"],
        social_initiative=0.3,
        social_responsiveness=0.8,
        party_preference=0.6,
        trade_preference=0.5,
        movement_quality=0.85,
        wander_chance=0.1,
        afk_chance=0.05,
        combat_aggression=0.6,
        flee_threshold=0.25,
        skill_spam_rate=0.4,
        target_switching=0.2,
        wrong_target_chance=0.05,
        pathing_error_chance=0.05,
        cast_cancel_chance=0.02,
        wrong_item_use_chance=0.01,
        active_hours=(8, 23),
        break_frequency_minutes=(45, 90),
        break_duration_minutes=(2, 8),
    ),
    "quiet": PersonalityProfile(
        name="Quiet",
        description="Reserved player who only responds to whispers",
        chat_frequency_per_hour=(0, 2),
        typo_rate=0.05,
        capitalization="normal",
        emoji_usage=0.05,
        slang_words=[],
        social_initiative=0.0,
        social_responsiveness=0.3,
        party_preference=0.2,
        trade_preference=0.2,
        movement_quality=0.95,
        wander_chance=0.02,
        afk_chance=0.02,
        combat_aggression=0.8,
        flee_threshold=0.15,
        skill_spam_rate=0.6,
        target_switching=0.1,
        wrong_target_chance=0.02,
        pathing_error_chance=0.02,
        cast_cancel_chance=0.01,
        wrong_item_use_chance=0.0,
        active_hours=(6, 22),
        break_frequency_minutes=(60, 120),
        break_duration_minutes=(3, 10),
    ),
    "noob": PersonalityProfile(
        name="Noob",
        description="Inexperienced player who makes obvious mistakes",
        chat_frequency_per_hour=(2, 8),
        typo_rate=0.3,
        capitalization="lowercase",
        emoji_usage=0.1,
        slang_words=["noob", "pls", "help", "where", "how"],
        social_initiative=0.1,
        social_responsiveness=0.6,
        party_preference=0.7,
        trade_preference=0.4,
        movement_quality=0.6,
        wander_chance=0.2,
        afk_chance=0.1,
        combat_aggression=0.4,
        flee_threshold=0.4,
        skill_spam_rate=0.3,
        target_switching=0.5,
        wrong_target_chance=0.15,
        pathing_error_chance=0.15,
        cast_cancel_chance=0.1,
        wrong_item_use_chance=0.05,
        active_hours=(10, 22),
        break_frequency_minutes=(30, 60),
        break_duration_minutes=(5, 15),
    ),
    "efficient": PersonalityProfile(
        name="Efficient",
        description="Optimized farmer who minimizes downtime",
        chat_frequency_per_hour=(0, 1),
        typo_rate=0.02,
        capitalization="normal",
        emoji_usage=0.0,
        slang_words=[],
        social_initiative=0.0,
        social_responsiveness=0.1,
        party_preference=0.1,
        trade_preference=0.1,
        movement_quality=0.98,
        wander_chance=0.0,
        afk_chance=0.0,
        combat_aggression=0.9,
        flee_threshold=0.1,
        skill_spam_rate=0.8,
        target_switching=0.05,
        wrong_target_chance=0.01,
        pathing_error_chance=0.01,
        cast_cancel_chance=0.0,
        wrong_item_use_chance=0.0,
        active_hours=(0, 23),
        break_frequency_minutes=(120, 180),
        break_duration_minutes=(1, 3),
    ),
    "social": PersonalityProfile(
        name="Social",
        description="Party-oriented player who prefers group play",
        chat_frequency_per_hour=(3, 10),
        typo_rate=0.1,
        capitalization="normal",
        emoji_usage=0.2,
        slang_words=["lol", "nice", "ty", "np", "wb", "brb"],
        social_initiative=0.2,
        social_responsiveness=0.9,
        party_preference=0.9,
        trade_preference=0.6,
        movement_quality=0.8,
        wander_chance=0.05,
        afk_chance=0.03,
        combat_aggression=0.5,
        flee_threshold=0.3,
        skill_spam_rate=0.5,
        target_switching=0.3,
        wrong_target_chance=0.05,
        pathing_error_chance=0.05,
        cast_cancel_chance=0.02,
        wrong_item_use_chance=0.01,
        active_hours=(10, 2),  # Late night player
        break_frequency_minutes=(30, 60),
        break_duration_minutes=(2, 5),
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Extended PersonalityEngine — wired with all persistent state subsystems
# ═══════════════════════════════════════════════════════════════════════════════

class PersonalityEngine:
    """Manages bot personality profiles and generates behavior modifiers.

    Extended with persistent state:
    - InteractionHistory: tracks all player interactions
    - ReputationTracker: per-player reputation that evolves
    - RelationshipManager: friends, enemies, trade partners
    - PersistentStorage: save/load from JSON
    - BackstoryGenerator: consistent backstories from personality + history
    """

    def __init__(self, storage_path: str | Path | None = None):
        self._lock = RLock()
        self._bot_personalities: dict[str, str] = {}  # bot_id -> personality name
        self._bot_break_until: dict[str, float] = {}  # bot_id -> break end time
        self._bot_last_chat: dict[str, float] = {}  # bot_id -> last chat time
        self._bot_chat_count: dict[str, int] = {}  # bot_id -> messages this hour
        self._bot_chat_hour: dict[str, int] = {}  # bot_id -> hour of last reset
        self._stats: dict[str, int] = defaultdict(int)

        # ── New persistent state subsystems ──
        self.interaction_history = InteractionHistory()
        self.reputation_tracker = ReputationTracker()
        self.relationship_manager = RelationshipManager()
        self.backstory_generator = BackstoryGenerator()
        self.storage = PersistentStorage(storage_path)

        # Auto-load on init
        self._load_state()

    # ── Personality assignment (original) ──

    def assign_personality(self, bot_id: str, personality_name: str) -> None:
        """Assign a personality to a bot."""
        with self._lock:
            if personality_name in PERSONALITIES:
                self._bot_personalities[bot_id] = personality_name
                self._stats["assignments"] += 1
                logger.info("personality_assigned: bot=%s personality=%s",
                           bot_id, personality_name)
                self.storage.mark_dirty()
            else:
                logger.warning("personality_not_found: bot=%s personality=%s",
                               bot_id, personality_name)

    def get_personality(self, bot_id: str) -> PersonalityProfile:
        """Get the personality profile for a bot."""
        with self._lock:
            name = self._bot_personalities.get(bot_id, "quiet")
            return PERSONALITIES.get(name, PERSONALITIES["quiet"])

    def get_behavior_modifier(self, bot_id: str,
                               context: dict[str, Any]) -> dict[str, Any]:
        """Get behavior modifiers for a bot based on its personality.

        Returns a dict with all behavior adjustments for this tick.
        """
        profile = self.get_personality(bot_id)
        now = time.time()

        # Check if on break
        with self._lock:
            break_until = self._bot_break_until.get(bot_id, 0)
            if now < break_until:
                return {
                    "on_break": True,
                    "break_remaining_s": int(break_until - now),
                    "movement_quality": 0.0,
                    "chat_allowed": False,
                    "combat_allowed": False,
                }

        # Check if should take a break
        min_break, max_break = profile.break_frequency_minutes
        if min_break > 0 and max_break > 0:
            with self._lock:
                last_break = self._bot_break_until.get(bot_id, 0)
            # Only take a break if we've had one before (skip initial break)
            if last_break > 0 and now - last_break > random.uniform(min_break * 60, max_break * 60):
                break_duration = random.uniform(
                    profile.break_duration_minutes[0] * 60,
                    profile.break_duration_minutes[1] * 60,
                )
                with self._lock:
                    self._bot_break_until[bot_id] = now + break_duration
                    self._stats["breaks"] += 1
                return {
                    "on_break": True,
                    "break_remaining_s": int(break_duration),
                    "movement_quality": 0.0,
                    "chat_allowed": False,
                    "combat_allowed": False,
                }
            # Set initial break marker so future calls can trigger breaks
            if last_break == 0:
                with self._lock:
                    self._bot_break_until[bot_id] = now

        # Chat frequency
        with self._lock:
            current_hour = int(time.localtime().tm_hour)
            last_hour = self._bot_chat_hour.get(bot_id, -1)
            if current_hour != last_hour:
                self._bot_chat_hour[bot_id] = current_hour
                self._bot_chat_count[bot_id] = 0

            chat_count = self._bot_chat_count.get(bot_id, 0)
            min_chat, max_chat = profile.chat_frequency_per_hour
            chat_allowed = chat_count < max_chat and random.random() < (min_chat / max_chat) if max_chat > 0 else False

        # Generate modifiers
        return {
            "on_break": False,
            "chat_allowed": chat_allowed,
            "typo_rate": profile.typo_rate,
            "capitalization": profile.capitalization,
            "emoji_usage": profile.emoji_usage,
            "slang_words": profile.slang_words,
            "social_initiative": profile.social_initiative,
            "social_responsiveness": profile.social_responsiveness,
            "party_preference": profile.party_preference,
            "trade_preference": profile.trade_preference,
            "movement_quality": profile.movement_quality,
            "wander_chance": profile.wander_chance,
            "combat_aggression": profile.combat_aggression,
            "flee_threshold": profile.flee_threshold,
            "skill_spam_rate": profile.skill_spam_rate,
            "target_switching": profile.target_switching,
            "wrong_target_chance": profile.wrong_target_chance,
            "pathing_error_chance": profile.pathing_error_chance,
            "cast_cancel_chance": profile.cast_cancel_chance,
            "wrong_item_use_chance": profile.wrong_item_use_chance,
        }

    def record_chat(self, bot_id: str) -> None:
        """Record that a bot sent a chat message."""
        with self._lock:
            self._bot_chat_count[bot_id] = self._bot_chat_count.get(bot_id, 0) + 1
            self._bot_last_chat[bot_id] = time.time()
            self._stats["chats"] += 1

    def get_stats(self) -> dict[str, int]:
        """Get personality engine statistics."""
        with self._lock:
            return dict(self._stats)

    # ── New: Interaction recording (wires all subsystems) ──

    def record_interaction(self, bot_id: str, player_name: str,
                           interaction_type: str, message: str = "",
                           context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Record a social interaction across all subsystems.

        This is the primary entry point for recording any player interaction.
        It updates interaction history, reputation, and relationships atomically.
        """
        # Record in interaction history
        interaction = self.interaction_history.add_interaction(
            player_name=player_name,
            interaction_type=interaction_type,
            message=message,
            context=context or {},
        )

        # Update reputation
        new_score = self.reputation_tracker.record_interaction(
            player_name=player_name,
            interaction_type=interaction_type,
        )

        # Update relationship
        relationship = self.relationship_manager.update_relationship(
            player_name=player_name,
            interaction_type=interaction_type,
        )

        # Mark state as dirty for auto-save
        self.storage.mark_dirty()

        # Auto-save periodically
        self._auto_save()

        return {
            "interaction": interaction.to_dict(),
            "reputation_score": new_score,
            "reputation_sentiment": self.reputation_tracker.get_reputation(player_name).sentiment,
            "relationship_level": relationship.level_label,
            "relationship_affinity": relationship.affinity,
        }

    def get_player_context(self, player_name: str) -> dict[str, Any]:
        """Get full context about a player across all subsystems."""
        return {
            "interaction_summary": self.interaction_history.get_player_summary(player_name),
            "reputation": self.reputation_tracker.get_reputation(player_name).to_dict(),
            "relationship": self.relationship_manager.get_relationship(player_name).to_dict(),
        }

    def get_backstory(self, bot_id: str) -> dict[str, Any]:
        """Get or generate a backstory for a bot."""
        return self.backstory_generator.generate_backstory(
            bot_id=bot_id,
            personality_name=self._bot_personalities.get(bot_id, "quiet"),
            interaction_history=self.interaction_history,
            reputation_tracker=self.reputation_tracker,
            relationship_manager=self.relationship_manager,
        )

    # ── Persistence ──

    def save_state(self, force: bool = False) -> bool:
        """Save all state to persistent storage."""
        data = self._serialize_state()
        return self.storage.save(data)

    def _auto_save(self) -> None:
        """Auto-save if interval has elapsed."""
        data = self._serialize_state()
        self.storage.auto_save(data)

    def _serialize_state(self) -> dict[str, Any]:
        """Serialize all state to a single dict."""
        with self._lock:
            return {
                "version": 2,
                "saved_at": time.time(),
                "bot_personalities": dict(self._bot_personalities),
                "bot_break_until": dict(self._bot_break_until),
                "bot_last_chat": dict(self._bot_last_chat),
                "bot_chat_count": dict(self._bot_chat_count),
                "bot_chat_hour": dict(self._bot_chat_hour),
                "stats": dict(self._stats),
                "interaction_history": self.interaction_history.to_dict(),
                "reputation_tracker": self.reputation_tracker.to_dict(),
                "relationship_manager": self.relationship_manager.to_dict(),
                "backstory_generator": self.backstory_generator.to_dict(),
            }

    def _load_state(self) -> bool:
        """Load all state from persistent storage."""
        data = self.storage.load()
        if data is None:
            return False

        try:
            with self._lock:
                self._bot_personalities = data.get("bot_personalities", {})
                self._bot_break_until = {k: float(v) for k, v in data.get("bot_break_until", {}).items()}
                self._bot_last_chat = {k: float(v) for k, v in data.get("bot_last_chat", {}).items()}
                self._bot_chat_count = {k: int(v) for k, v in data.get("bot_chat_count", {}).items()}
                self._bot_chat_hour = {k: int(v) for k, v in data.get("bot_chat_hour", {}).items()}
                self._stats = defaultdict(int, data.get("stats", {}))

                # Load new subsystems
                if "interaction_history" in data:
                    self.interaction_history = InteractionHistory.from_dict(data["interaction_history"])
                if "reputation_tracker" in data:
                    self.reputation_tracker = ReputationTracker.from_dict(data["reputation_tracker"])
                if "relationship_manager" in data:
                    self.relationship_manager = RelationshipManager.from_dict(data["relationship_manager"])
                if "backstory_generator" in data:
                    self.backstory_generator = BackstoryGenerator.from_dict(data["backstory_generator"])

            logger.info("personality_state_loaded: bots=%d interactions=%d reputations=%d relationships=%d",
                       len(self._bot_personalities),
                       self.interaction_history.get_total_interaction_count(),
                       len(self.reputation_tracker.get_all_reputations()),
                       len(self.relationship_manager.get_all_relationships()))
            return True
        except (KeyError, ValueError, TypeError) as e:
            logger.error("personality_state_load_error: %s", e)
            return False

    def set_storage_path(self, file_path: str | Path) -> None:
        """Set a custom storage path and reload state."""
        self.storage.set_file_path(file_path)
        self._load_state()


# Global singleton
_engine: PersonalityEngine | None = None

def get_personality_engine() -> PersonalityEngine:
    """Get the global PersonalityEngine instance."""
    global _engine
    if _engine is None:
        _engine = PersonalityEngine()
    return _engine
