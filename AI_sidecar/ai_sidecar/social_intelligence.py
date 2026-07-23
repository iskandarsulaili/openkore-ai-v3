"""
Social intelligence v2 — stealth-first player interaction handler.

Default mode: ABSOLUTE SILENCE. Never speak unless explicitly decided.
Handles: buff requests, trade, party invites, whispers, public chat.
AI decides to ignore/accept/reject based on context and risk assessment.
Responses include realistic human imperfections: typos, delayed typing, casual language.
"""

from __future__ import annotations

import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

# Realistic human-like response templates
RESPONSES = {
    "greeting": ["hey", "hi", "yo", "sup", "hello"],
    "farewell": ["cya", "later", "gtg", "bye"],
    "buff_accept": ["sure", "ok", "np", "k", "one sec"],
    "buff_decline": ["sry no sp", "cant rn", "maybe later", "afk"],
    "trade_accept": ["sure wat u got", "ok show me", "k"],
    "trade_decline": ["no thx", "not selling", "keeping it", "sry"],
    "party_accept": ["sure", "ok inv", "k"],
    "party_decline": ["busy rn", "sry in party", "maybe later"],
    "unknown": ["?", "huh", "what", "??", "not sure wat u mean"],
    "afk": ["brb", "afk sec", "one min"],
}

# Typing delay simulation (seconds)
MIN_TYPING_DELAY = 1.0
MAX_TYPING_DELAY = 4.0

# Typo probability
TYPO_PROBABILITY = 0.15

# ── Keyboard proximity map for realistic typos ──
# QWERTY keyboard layout: adjacent keys that are commonly mistyped
_KEYBOARD_PROXIMITY: dict[str, str] = {
    # Row 1
    "q": "w", "w": "qe", "e": "wr", "r": "et", "t": "ry", "y": "tu",
    "u": "yi", "i": "uo", "o": "ip", "p": "o",
    # Row 2
    "a": "s", "s": "ad", "d": "sf", "f": "dg", "g": "fh", "h": "gj",
    "j": "hk", "k": "jl", "l": "k",
    # Row 3
    "z": "x", "x": "zc", "c": "xv", "v": "cb", "b": "vn", "n": "bm",
    "m": "n",
    # Common fat-finger substitutions
    "i": "o", "o": "p", "e": "r", "n": "m", "s": "a",
}

# ── RO-themed name generator components ──
_NAME_PREFIXES = [
    "xX", "Dark", "Shadow", "Light", "Fire", "Ice", "Storm", "Thunder",
    "Night", "Silver", "Golden", "Crimson", "Azure", "Emerald", "Crystal",
    "Mystic", "Arcane", "Frost", "Blaze", "Phantom", "Soul", "Spirit",
    "Dragon", "Wolf", "Phoenix", "Demon", "Angel", "Chaos", "Omega",
    "Ultima", "Super", "Mega", "Hyper", "Neo", "Proto", "Cyber",
]

_NAME_SUFFIXES = [
    "Xx", "Kun", "Chan", "San", "Sama", "Senpai", "Dono",
    "Slayer", "Killer", "Hunter", "Mage", "Lord", "King", "Queen",
    "Master", "Blade", "Heart", "Soul", "Star", "Wind", "Flame",
    "Bolt", "Strike", "Fury", "Wrath", "Bliss", "Bane",
    "ROG", "POM", "FTW", "LOL", "OMG", "XD",
    "01", "69", "420", "1337", "007", "x", "z",
]

_NAME_CORES = [
    "poring", "pupa", "lunatic", "fabre", "chonchon", "condor", "wilow",
    "drops", "poporing", "hammer", "shield", "sword", "blade", "arrow",
    "bow", "staff", "robe", "boots", "ring", "cape", "wing",
    "flame", "frost", "thunder", "gale", "quake", "tide", "void",
    "nova", "comet", "nebula", "solar", "lunar", "stellar",
    "kappa", "taco", "noodle", "ramen", "pizza", "burger",
    "zero", "hero", "zero", "omega", "alpha", "beta", "delta",
    "knight", "mage", "thief", "archer", "acolyte", "merchant",
    "ninja", "samurai", "monk", "bard", "dancer", "rogue",
    "cat", "dog", "bird", "fish", "bear", "fox", "owl",
]


def _keyboard_typo(text: str) -> str:
    """Add a realistic keyboard-proximity typo to text.

    Instead of a simple character swap (which was the old ''.join(chars) approach),
    this replaces a random character with an adjacent key on a QWERTY keyboard,
    producing much more realistic typos like 'teh' for 'the' or 'adn' for 'and'.
    """
    if len(text) < 2:
        return text

    # Pick a random position
    idx = random.randint(0, len(text) - 1)
    char = text[idx].lower()

    # Check if we have a proximity replacement
    if char in _KEYBOARD_PROXIMITY:
        replacements = _KEYBOARD_PROXIMITY[char]
        replacement = random.choice(replacements)
        # Preserve case
        if text[idx].isupper():
            replacement = replacement.upper()
        return text[:idx] + replacement + text[idx + 1:]

    # Fallback: double a character (common typo: "helloo")
    if random.random() < 0.3:
        return text[:idx] + text[idx] + text[idx:]

    # Fallback: omit a character (common typo: "helo")
    if random.random() < 0.3 and len(text) > 3:
        return text[:idx] + text[idx + 1:]

    return text


def _add_typo(text: str) -> str:
    """Add a realistic typo to text using keyboard proximity."""
    if random.random() > TYPO_PROBABILITY:
        return text
    if len(text) < 3:
        return text
    return _keyboard_typo(text)


def _humanize(text: str) -> str:
    """Add human-like imperfections to text."""
    text = _add_typo(text)
    # Random capitalization
    if random.random() < 0.1:
        text = text.lower()
    # Random punctuation
    if random.random() < 0.2:
        text += random.choice(["", ".", "!", "..", "..."])
    return text


def generate_player_name() -> str:
    """Generate a realistic RO-style player name.

    Produces names like 'xXDarkSlayerXx', 'ShadowPoring', 'FireKnight1337',
    or 'MysticBlade' — the kind of names real players use on RO servers.
    """
    style = random.random()

    if style < 0.25:
        # xX_Name_Xx style
        prefix = random.choice(_NAME_PREFIXES)
        core = random.choice(_NAME_CORES)
        suffix = random.choice(_NAME_SUFFIXES)
        name = f"xX{prefix}{core.capitalize()}{suffix}Xx"

    elif style < 0.50:
        # PrefixCore style
        prefix = random.choice(_NAME_PREFIXES)
        core = random.choice(_NAME_CORES)
        name = f"{prefix}{core.capitalize()}"

    elif style < 0.75:
        # CoreSuffix style
        core = random.choice(_NAME_CORES)
        suffix = random.choice(_NAME_SUFFIXES)
        name = f"{core.capitalize()}{suffix}"

    else:
        # Simple name with number
        core = random.choice(_NAME_CORES)
        number = random.randint(1, 9999)
        name = f"{core.capitalize()}{number}"

    # Trim to max 24 chars (RO name limit)
    return name[:24]


@dataclass(slots=True)
class SocialIntelligenceV2:
    """Stealth-first social interaction handler."""

    _lock: RLock = field(default_factory=RLock)
    _whisper_history: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    _reputation: dict[str, float] = field(default_factory=dict)  # player -> reputation score
    _blocked_players: set[str] = field(default_factory=set)
    _last_response_time: dict[str, float] = field(default_factory=dict)
    _stats: dict[str, int] = field(default_factory=lambda: {"greetings": 0, "replies": 0, "ignored": 0, "blocked": 0})

    def process_public_chat(self, sender: str, message: str, context: dict[str, Any]) -> str | None:
        """Process a public chat message. Returns a response or None (stay silent)."""
        msg_lower = message.lower()

        # Check if we're being addressed directly
        addressed = any(name in msg_lower for name in ["bot", "kicap", sender[:4]])
        if not addressed:
            return None  # Not talking to us — stay silent

        return self._handle_interaction(sender, message, context, "public")

    def process_whisper(self, sender: str, message: str, context: dict[str, Any]) -> str | None:
        """Process a whisper. Returns a response or None."""
        return self._handle_interaction(sender, message, context, "whisper")

    def _handle_interaction(self, sender: str, message: str, context: dict[str, Any], channel: str) -> str | None:
        """Handle any player interaction. Returns response or None."""
        msg_lower = message.lower()

        with self._lock:
            # Track history
            self._whisper_history[sender].append({
                "message": message,
                "channel": channel,
                "timestamp": time.time(),
            })
            self._whisper_history[sender] = self._whisper_history[sender][-20:]

        # Check if blocked
        if sender in self._blocked_players:
            with self._lock:
                self._stats["ignored"] += 1
            return None

        # Check cooldown (don't spam)
        last_time = self._last_response_time.get(sender, 0)
        if time.time() - last_time < 10:
            return None

        # Check reputation
        rep = self._reputation.get(sender, 0.5)
        if rep < 0.2:
            with self._lock:
                self._stats["ignored"] += 1
            return None

        # Determine interaction type
        response = None

        # Buff request
        if any(kw in msg_lower for kw in ["buff", "heal", "bless", "agi", "boost"]):
            if rep > 0.3 and random.random() < 0.7:
                response = random.choice(RESPONSES["buff_accept"])
            else:
                response = random.choice(RESPONSES["buff_decline"])

        # Trade request
        elif any(kw in msg_lower for kw in ["buy", "sell", "trade", "price", "how much"]):
            if rep > 0.5 and random.random() < 0.5:
                response = random.choice(RESPONSES["trade_accept"])
            else:
                response = random.choice(RESPONSES["trade_decline"])

        # Party invite
        elif any(kw in msg_lower for kw in ["party", "pt", "invite", "join"]):
            if rep > 0.4 and random.random() < 0.6:
                response = random.choice(RESPONSES["party_accept"])
            else:
                response = random.choice(RESPONSES["party_decline"])

        # Greeting
        elif any(g in msg_lower for g in ["hello", "hi", "hey", "sup", "yo"]):
            response = random.choice(RESPONSES["greeting"])

        # Farewell
        elif any(f in msg_lower for f in ["bye", "cya", "later", "gtg"]):
            response = random.choice(RESPONSES["farewell"])

        # Unknown
        else:
            if random.random() < 0.3:  # 30% chance to respond to unknown
                response = random.choice(RESPONSES["unknown"])

        if response is not None:
            response = _humanize(response)
            with self._lock:
                self._last_response_time[sender] = time.time()
                self._stats["replies"] += 1
                # Simulate typing delay
                delay = random.uniform(MIN_TYPING_DELAY, MAX_TYPING_DELAY)
                logger.info("social_response: to=%s msg=%s delay=%.1fs", sender, response, delay)

        return response

    def update_reputation(self, player: str, delta: float) -> None:
        """Update a player's reputation score."""
        with self._lock:
            current = self._reputation.get(player, 0.5)
            self._reputation[player] = max(0.0, min(1.0, current + delta))

    def block_player(self, player: str) -> None:
        """Block a player (stop responding to them)."""
        with self._lock:
            self._blocked_players.add(player)
            self._stats["blocked"] += 1

    def get_greeting(self) -> str:
        """Get a greeting for login."""
        with self._lock:
            self._stats["greetings"] += 1
        return _humanize(random.choice(RESPONSES["greeting"]))

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)
