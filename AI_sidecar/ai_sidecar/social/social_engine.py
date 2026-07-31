"""
Social Engine — Complete social intelligence for the bot fleet.

A pro player knows:
- Who to trust and who to avoid (GM detection, PKer profiling)
- How to build reputation (helpful responses, reliable party member)
- When to chat and what to say (human-like patterns)
- The social contract: don't KS, don't ninja loot, don't bot in obvious places
- How to interact with guilds, parties, and trade partners

This engine wires into:
- player_profiler.py (existing player profiling)
- ChatMessageEvent / ChatStreamIngestRequest (chat event system)
- social_intelligence.py (existing social modules)
- conversation_engine.py (existing conversation system)
"""

from __future__ import annotations

import logging
import random
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────────────────────


@dataclass
class ChatMessage:
    """A chat message observed in-game."""
    channel: str  # public, whisper, party, guild, trade
    sender: str
    message: str
    timestamp: float
    map_name: str = ""
    target: str = ""


@dataclass
class Relationship:
    """Tracked relationship with another player."""
    player_name: str
    relationship_type: str = "neutral"  # friend, neutral, rival, trusted, blocked
    trust_score: int = 50  # 0-100
    interaction_count: int = 0
    last_interaction: float = 0.0
    first_seen: float = 0.0
    is_gm: bool = False
    is_pker: bool = False
    is_bot_reporter: bool = False
    is_trade_partner: bool = False
    is_party_member: bool = False
    is_guild_member: bool = False
    notes: str = ""


@dataclass
class SocialEvent:
    """A social event that happened."""
    event_type: str  # party_invite, trade_request, whisper, guild_chat, etc.
    actor: str
    target: str = ""
    detail: str = ""
    timestamp: float = 0.0
    response: str = ""  # accepted, declined, ignored


@dataclass
class ChatTemplate:
    """A template for generating human-like chat responses."""
    pattern: str  # regex pattern to match
    responses: list[str] = field(default_factory=list)
    cooldown_seconds: int = 60
    priority: int = 5  # 1-10, higher = more likely to respond
    requires_relationship: str = "any"  # any, friend, trusted, neutral
    last_used: float = 0.0


@dataclass(slots=True)
class SocialEngine:
    """
    Complete social intelligence engine.

    Tracks relationships, analyzes chat, manages reputation,
    and ensures human-like social behavior.
    """

    _lock: RLock = field(default_factory=RLock)
    _relationships: dict[str, Relationship] = field(default_factory=dict)
    _chat_history: deque = field(default_factory=lambda: deque(maxlen=500))
    _social_events: deque = field(default_factory=lambda: deque(maxlen=200))
    _chat_templates: list[ChatTemplate] = field(default_factory=list)
    _recent_responses: dict[str, float] = field(default_factory=dict)  # player -> last response time
    _guild_messages_5m: int = 0
    _party_messages_5m: int = 0
    _private_messages_5m: int = 0
    _total_chat_messages: int = 0
    _last_chat_time: float = 0.0
    _stats: dict[str, int] = field(default_factory=lambda: {
        "messages_analyzed": 0, "relationships_tracked": 0,
        "responses_generated": 0, "gms_detected": 0,
        "pkers_detected": 0, "friends_made": 0,
    })
    _player_profiler: Any = None  # PlayerProfiler instance
    _conversation_engine: Any = None  # ConversationEngine instance
    _enqueue_fn: Callable | None = None  # Function to enqueue chat commands
    _last_cleanup: float = 0.0

    # ── Configuration ──

    GM_KEYWORDS: list[str] = field(default_factory=lambda: [
        "gm", "game master", "admin", "staff", "moderator",
        "helper", "support", "dev", "developer",
    ])
    SUSPICIOUS_PATTERNS: list[str] = field(default_factory=lambda: [
        r"bot", r"report", r"hack", r"macro", r"auto",
        r"cheat", r"ban", r"gm\s+check",
    ])
    CHAT_COOLDOWN_MIN: float = 3.0  # minimum seconds between chat messages
    CHAT_COOLDOWN_MAX: float = 15.0  # maximum seconds between chat messages
    MAX_CHAT_PER_HOUR: int = 30  # don't chat more than this per hour
    RESPONSE_CHANCE: float = 0.3  # 30% chance to respond to a direct message
    GREETING_CHANCE: float = 0.1  # 10% chance to greet someone entering map

    # ── Initialization ──

    def __post_init__(self) -> None:
        self._init_chat_templates()

    def _init_chat_templates(self) -> None:
        """Initialize default chat response templates."""
        self._chat_templates = [
            # Greetings
            ChatTemplate(
                pattern=r"(hi|hello|hey|sup|yo)\b",
                responses=[
                    "hey {sender}!",
                    "hello {sender}",
                    "hi there!",
                    "yo {sender}",
                    "hey, how's it going?",
                ],
                cooldown_seconds=120,
                priority=3,
            ),
            # Party invites
            ChatTemplate(
                pattern=r"(party|pt|group)\s*(invite|inv|pls|please|?)\b",
                responses=[
                    "sure, inv me",
                    "yeah i can pt",
                    "ok, one sec",
                ],
                cooldown_seconds=60,
                priority=7,
            ),
            # Trade
            ChatTemplate(
                pattern=r"(trade|buy|sell|wts|wtb|wtt)\b",
                responses=[
                    "what are you selling?",
                    "how much?",
                    "not interested rn",
                    "sure, what's your offer?",
                ],
                cooldown_seconds=60,
                priority=6,
            ),
            # Thanks
            ChatTemplate(
                pattern=r"(thx|ty|thanks|thank you|appreciate)\b",
                responses=[
                    "np!",
                    "anytime",
                    "yw",
                    "no problem",
                ],
                cooldown_seconds=120,
                priority=4,
            ),
            # Questions about level/class
            ChatTemplate(
                pattern=r"(what\s*(class|job|lvl|level)|how\s*far)\b",
                responses=[
                    "i'm a {class}",
                    "lvl {level} {class}",
                    "still leveling lol",
                ],
                cooldown_seconds=180,
                priority=5,
            ),
            # Goodbye
            ChatTemplate(
                pattern=r"(bye|cya|g2g|gtg|brb|afk)\b",
                responses=[
                    "cya!",
                    "later!",
                    "gl hf",
                    "see you around",
                ],
                cooldown_seconds=300,
                priority=2,
            ),
            # Map/spot questions
            ChatTemplate(
                pattern=r"(where|spot|map|good\s*(spot|place|map)|exp\s*(spot|map))\b",
                responses=[
                    "try {map}",
                    "i've been farming {map}",
                    "not sure, i just got here",
                ],
                cooldown_seconds=180,
                priority=5,
            ),
            # WOE/Guild
            ChatTemplate(
                pattern=r"(woe|guild|castle|war|emp)\b",
                responses=[
                    "gl in woe!",
                    "who are you guys with?",
                    "i'm not in a guild rn",
                ],
                cooldown_seconds=300,
                priority=4,
            ),
        ]

    # ── Public API ──

    def set_player_profiler(self, profiler: Any) -> None:
        """Wire PlayerProfiler instance."""
        self._player_profiler = profiler

    def set_conversation_engine(self, engine: Any) -> None:
        """Wire ConversationEngine instance."""
        self._conversation_engine = engine

    def set_enqueue_fn(self, fn: Callable) -> None:
        """Set function to enqueue chat commands."""
        self._enqueue_fn = fn

    # ── Chat Processing ──

    def process_chat(self, channel: str, sender: str, message: str,
                     map_name: str = "", target: str = "") -> dict[str, Any] | None:
        """Process an incoming chat message. Returns response if one should be sent."""
        with self._lock:
            now = time.time()
            msg = ChatMessage(
                channel=channel, sender=sender, message=message,
                timestamp=now, map_name=map_name, target=target,
            )
            self._chat_history.append(msg)
            self._total_chat_messages += 1
            self._last_chat_time = now
            self._stats["messages_analyzed"] += 1

            # Update channel counters (rolling 5 min)
            if now - getattr(self, '_last_channel_reset', 0) > 300:
                self._guild_messages_5m = 0
                self._party_messages_5m = 0
                self._private_messages_5m = 0
                object.__setattr__(self, '_last_channel_reset', now)

            if channel == "guild":
                self._guild_messages_5m += 1
            elif channel in ("party", "party_chat"):
                self._party_messages_5m += 1
            elif channel in ("whisper", "pm"):
                self._private_messages_5m += 1

            # Update relationship
            self._update_relationship(sender, channel, message)

            # Check for GM detection
            self._check_gm_detection(sender, message)

            # Check for suspicious content (reports, accusations)
            self._check_suspicious_content(sender, message)

            # Determine if we should respond
            return self._should_respond(msg)

    def _update_relationship(self, sender: str, channel: str, message: str) -> None:
        """Update relationship tracking for a player."""
        if sender in ("", "System", "Server"):
            return

        now = time.time()
        if sender not in self._relationships:
            self._relationships[sender] = Relationship(
                player_name=sender,
                first_seen=now,
            )
            self._stats["relationships_tracked"] += 1

        rel = self._relationships[sender]
        rel.last_interaction = now
        rel.interaction_count += 1

        # Update relationship type based on channel
        if channel == "whisper" or channel == "pm":
            rel.trust_score = min(100, rel.trust_score + 2)
        elif channel == "party" or channel == "party_chat":
            rel.is_party_member = True
            rel.trust_score = min(100, rel.trust_score + 1)
        elif channel == "guild":
            rel.is_guild_member = True
            rel.trust_score = min(100, rel.trust_score + 1)

        # Check for positive interactions
        positive_words = ["thx", "ty", "thanks", "help", "nice", "good", "tyvm"]
        if any(w in message.lower() for w in positive_words):
            rel.trust_score = min(100, rel.trust_score + 3)

        # Check for negative interactions
        negative_words = ["bot", "report", "hack", "cheat", "noob", "stupid", "ks"]
        if any(w in message.lower() for w in negative_words):
            rel.trust_score = max(0, rel.trust_score - 5)

        # Auto-classify relationship
        if rel.trust_score >= 80:
            rel.relationship_type = "trusted"
        elif rel.trust_score >= 60:
            rel.relationship_type = "friend"
        elif rel.trust_score >= 40:
            rel.relationship_type = "neutral"
        elif rel.trust_score >= 20:
            rel.relationship_type = "rival"
        else:
            rel.relationship_type = "blocked"

    def _check_gm_detection(self, sender: str, message: str) -> bool:
        """Check if a player might be a GM."""
        is_gm = False
        for keyword in self.GM_KEYWORDS:
            if keyword in sender.lower():
                is_gm = True
                break

        if is_gm:
            rel = self._relationships.get(sender)
            if rel:
                rel.is_gm = True
                rel.trust_score = 0
                rel.relationship_type = "blocked"
            self._stats["gms_detected"] += 1
            logger.warning("gm_detected: %s (message: %s)", sender, message[:100])
            return True

        return False

    def _check_suspicious_content(self, sender: str, message: str) -> bool:
        """Check if message contains suspicious content (reports, accusations)."""
        for pattern in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                rel = self._relationships.get(sender)
                if rel:
                    rel.is_bot_reporter = True
                    rel.trust_score = max(0, rel.trust_score - 20)
                    rel.relationship_type = "blocked"
                logger.warning("suspicious_content: %s said '%s'", sender, message[:100])
                return True
        return False

    def _should_respond(self, msg: ChatMessage) -> dict[str, Any] | None:
        """Determine if and how to respond to a chat message."""
        now = time.time()

        # Don't respond to ourselves
        if msg.sender == "" or msg.sender == "System":
            return None

        # Rate limiting: don't chat too much
        if self._total_chat_messages > self.MAX_CHAT_PER_HOUR:
            return None

        # Cooldown between messages
        if now - self._last_chat_time < self.CHAT_COOLDOWN_MIN:
            return None

        # Check if we've responded to this player recently
        last_resp = self._recent_responses.get(msg.sender, 0)
        if now - last_resp < 60:
            return None

        # Check relationship
        rel = self._relationships.get(msg.sender)
        if rel and rel.relationship_type == "blocked":
            return None

        # Direct messages (whisper) have higher response chance
        if msg.channel in ("whisper", "pm"):
            if random.random() < self.RESPONSE_CHANCE + 0.3:
                return self._generate_response(msg)
            return None

        # Public chat: check templates
        for template in self._chat_templates:
            if now - template.last_used < template.cooldown_seconds:
                continue
            if re.search(template.pattern, msg.message, re.IGNORECASE):
                if random.random() < self.RESPONSE_CHANCE * (template.priority / 5):
                    template.last_used = now
                    return self._generate_response(msg, template)

        # Random social greeting (low chance)
        if random.random() < self.GREETING_CHANCE:
            return self._generate_greeting(msg)

        return None

    def _generate_response(self, msg: ChatMessage,
                           template: ChatTemplate | None = None) -> dict[str, Any]:
        """Generate a response to a chat message."""
        now = time.time()

        # Use conversation engine if available
        if self._conversation_engine is not None and msg.channel in ("whisper", "pm"):
            try:
                response = self._conversation_engine.generate_response(
                    sender=msg.sender,
                    message=msg.message,
                    context={
                        "relationship": self._relationships.get(msg.sender),
                        "channel": msg.channel,
                    },
                )
                if response:
                    self._recent_responses[msg.sender] = now
                    self._stats["responses_generated"] += 1
                    return {
                        "channel": msg.channel,
                        "target": msg.sender,
                        "message": response,
                        "delay_ms": random.randint(1500, 4000),
                    }
            except Exception:
                pass

        # Fallback: use templates
        if template and template.responses:
            response = random.choice(template.responses)
            response = response.replace("{sender}", msg.sender)
            response = response.replace("{map}", msg.map_name or "this map")
            response = response.replace("{class}", "novice")
            response = response.replace("{level}", "1")

            self._recent_responses[msg.sender] = now
            self._stats["responses_generated"] += 1

            # Variable delay to simulate human typing
            delay_ms = random.randint(1000, 4000) + len(response) * 50

            return {
                "channel": msg.channel,
                "target": msg.sender if msg.channel in ("whisper", "pm") else "",
                "message": response,
                "delay_ms": delay_ms,
            }

        return None

    def _generate_greeting(self, msg: ChatMessage) -> dict[str, Any]:
        """Generate a greeting response."""
        greetings = [
            f"hey {msg.sender}!",
            f"hi {msg.sender}",
            f"hello!",
            f"sup {msg.sender}",
        ]
        response = random.choice(greetings)

        self._stats["responses_generated"] += 1
        return {
            "channel": msg.channel,
            "target": msg.sender if msg.channel in ("whisper", "pm") else "",
            "message": response,
            "delay_ms": random.randint(2000, 5000),
        }

    # ── Social Event Processing ──

    def process_event(self, event_type: str, actor: str, target: str = "",
                      detail: str = "") -> dict[str, Any] | None:
        """Process a social event (party invite, trade request, etc.)."""
        with self._lock:
            now = time.time()
            event = SocialEvent(
                event_type=event_type, actor=actor, target=target,
                detail=detail, timestamp=now,
            )
            self._social_events.append(event)

            # Check relationship
            rel = self._relationships.get(actor)

            # Party invites
            if event_type == "party_invite":
                if rel and rel.relationship_type == "blocked":
                    event.response = "declined"
                    return {"action": "decline", "reason": "blocked_player"}
                if rel and rel.trust_score >= 40:
                    event.response = "accepted"
                    self._stats["friends_made"] += 1
                    return {"action": "accept", "reason": "trusted_player"}
                # Random chance to accept
                if random.random() < 0.4:
                    event.response = "accepted"
                    return {"action": "accept", "reason": "random_accept"}
                event.response = "declined"
                return {"action": "decline", "reason": "unknown_player"}

            # Trade requests
            if event_type == "trade_request":
                if rel and rel.trust_score >= 50:
                    event.response = "accepted"
                    return {"action": "accept", "reason": "trusted_trader"}
                if random.random() < 0.3:
                    event.response = "accepted"
                    return {"action": "accept", "reason": "random_trade"}
                event.response = "declined"
                return {"action": "decline", "reason": "not_interested"}

            # Friend requests
            if event_type == "friend_request":
                if rel and rel.trust_score >= 60:
                    event.response = "accepted"
                    return {"action": "accept", "reason": "trusted_player"}
                if random.random() < 0.2:
                    event.response = "accepted"
                    return {"action": "accept", "reason": "random_accept"}
                event.response = "declined"
                return {"action": "decline", "reason": "dont_know_you"}

            return None

    # ── Social Contract ──

    def check_social_contract(self, map_name: str, nearby_players: list[dict],
                               current_activity: str) -> dict[str, Any]:
        """Check if current behavior violates the social contract.

        Returns a dict with:
        - violation: bool
        - reason: str
        - suggested_action: str
        """
        with self._lock:
            result = {"violation": False, "reason": "", "suggested_action": ""}

            # Don't bot in popular spots
            popular_maps = ["prt_fild01", "pay_fild01", "moc_fild01",
                           "gef_fild01", "alde_dun01"]
            if map_name in popular_maps and len(nearby_players) > 3:
                result["violation"] = True
                result["reason"] = "Too many players on popular map"
                result["suggested_action"] = "change_map"
                return result

            # Don't KS (kill steal)
            if current_activity == "attacking" and len(nearby_players) > 0:
                for player in nearby_players:
                    if player.get("is_attacking", False):
                        result["violation"] = True
                        result["reason"] = "Potential KS - player is attacking nearby"
                        result["suggested_action"] = "move_away"
                        return result

            # Don't bot in town
            town_maps = ["prontera", "morocc", "payon", "geffen",
                        "aldebaran", "alberta", "izlude"]
            if any(t in map_name.lower() for t in town_maps):
                if current_activity in ("farming", "attacking", "looting"):
                    result["violation"] = True
                    result["reason"] = "Botting in town is suspicious"
                    result["suggested_action"] = "stop_activity"
                    return result

            return result

    # ── Reputation Building ──

    def get_reputation_context(self) -> str:
        """Get formatted reputation context for LLM prompts."""
        with self._lock:
            lines = ["── Social Intelligence ──"]
            lines.append(f"Total messages analyzed: {self._total_chat_messages}")
            lines.append(f"Relationships tracked: {len(self._relationships)}")

            # Trusted players
            trusted = [r for r in self._relationships.values()
                       if r.relationship_type == "trusted"]
            if trusted:
                lines.append(f"Trusted: {', '.join(r.player_name for r in trusted[:5])}")

            # Threats
            threats = [r for r in self._relationships.values()
                       if r.relationship_type in ("rival", "blocked")]
            if threats:
                lines.append(f"Threats: {', '.join(f'{r.player_name}({r.relationship_type})' for r in threats[:5])}")

            # GMs
            gms = [r for r in self._relationships.values() if r.is_gm]
            if gms:
                lines.append(f"GMs detected: {', '.join(r.player_name for r in gms)}")

            # Bot reporters
            reporters = [r for r in self._relationships.values() if r.is_bot_reporter]
            if reporters:
                lines.append(f"Bot reporters: {', '.join(r.player_name for r in reporters)}")

            # Recent chat activity
            lines.append(f"Chat activity (5m): guild={self._guild_messages_5m}, "
                         f"party={self._party_messages_5m}, pm={self._private_messages_5m}")

            return "\n".join(lines)

    # ── Cycle Tick ──

    def tick(self) -> dict[str, Any]:
        """Called every PDCA cycle to update social state."""
        now = time.time()
        result = {
            "messages_analyzed": self._total_chat_messages,
            "relationships": len(self._relationships),
            "gms_detected": self._stats["gms_detected"],
            "pkers_detected": self._stats["pkers_detected"],
            "friends_made": self._stats["friends_made"],
        }

        # Cleanup stale relationships every 10 minutes
        if now - self._last_cleanup > 600:
            self._cleanup()
            self._last_cleanup = now

        return result

    def _cleanup(self) -> None:
        """Remove stale data."""
        with self._lock:
            now = time.time()
            # Remove relationships not seen in 7 days
            stale = [k for k, v in self._relationships.items()
                     if now - v.last_interaction > 604800]
            for k in stale:
                del self._relationships[k]

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# ── Global Singleton ──

_social_engine: SocialEngine | None = None
_social_engine_lock = RLock()


def get_social_engine() -> SocialEngine:
    global _social_engine
    with _social_engine_lock:
        if _social_engine is None:
            _social_engine = SocialEngine()
        return _social_engine
