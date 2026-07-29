"""SocialInteractionEngine — basic player interaction intelligence.

Provides scripted responses to common player interactions in Ragnarok Online:
- Whisper handling with keyword-driven auto-replies
- Buff acknowledgement when buffed by priests
- Item evaluation for dropped items nearby
- Party invite acceptance/decline based on level
- Trade request handling

Thread-safe via RLock. Response history per player avoids spam repetition.
"""
from __future__ import annotations

import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

# Cooldown per interaction type per player (seconds)
_COOLDOWN_SECONDS = 30

# Max history entries kept per player per interaction type
_MAX_HISTORY_PER_PLAYER = 50

# Default level threshold for party invites (max difference to accept)
_DEFAULT_LEVEL_THRESHOLD = 15

# Minimum value threshold (zeny) for auto-pickup during ITEM_EVAL
_DEFAULT_PICKUP_VALUE_THRESHOLD = 1000


# ── Dataclasses ────────────────────────────────────────────────────────────


@dataclass
class WhisperIntent:
    """Parsed intent from a whisper message."""
    raw: str
    lower: str
    is_buff_request: bool = False
    is_price_query: bool = False
    is_party_query: bool = False
    is_greeting: bool = False
    is_warp_request: bool = False
    is_buff_ack: bool = False  # Other player acknowledged our buff
    confidence: float = 0.0


@dataclass
class InteractionRecord:
    """Record of a single interaction with a player."""
    interaction_type: str  # whisper, buff_ack, party_invite, trade, item_eval
    detail: str
    timestamp: float
    response: str | None = None


@dataclass
class PlayerHistory:
    """Per-player interaction history to avoid spam."""
    player_name: str
    interactions: list[InteractionRecord] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    total_interactions: int = 0


@dataclass
class ItemDrop:
    """Information about a dropped item on the ground."""
    item_name: str
    item_id: int | None
    x: int
    y: int
    dropped_by: str  # Player name who dropped it
    timestamp: float


@dataclass
class PartyInvite:
    """Incoming party invite details."""
    leader_name: str
    leader_level: int
    leader_map: str
    leader_job: str | None = None


@dataclass
class TradeRequest:
    """Incoming trade request details."""
    player_name: str
    items_offered: list[dict[str, Any]] = field(default_factory=list)
    zeny_offered: int = 0
    items_wanted: list[dict[str, Any]] = field(default_factory=list)
    zeny_wanted: int = 0


# ── SocialInteractionEngine ────────────────────────────────────────────────


@dataclass(slots=True)
class SocialInteractionEngine:
    """Handles basic player-to-player social interactions in Ragnarok Online.

    Provides scripted responses to common interactions. Does NOT implement
    anti-detection or GM evasion — those are handled by other modules.

    Thread-safe via RLock. All public methods acquire the lock.
    """

    _lock: RLock = field(default_factory=RLock)

    # ── Per-player interaction history ──
    _history: dict[str, PlayerHistory] = field(default_factory=dict)

    # ── Interaction cooldown tracking: {player_name: {type: last_time}} ──
    _cooldowns: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(dict))

    # ── Drop tracking ──
    _recent_drops: list[ItemDrop] = field(default_factory=list)

    # ── Pending invites/requests ──
    _pending_party_invites: dict[str, PartyInvite] = field(default_factory=dict)
    _pending_trades: dict[str, TradeRequest] = field(default_factory=dict)

    # ── Stats ──
    _stats: dict[str, int] = field(default_factory=lambda: {
        "whispers_handled": 0,
        "buff_acks_sent": 0,
        "items_evaluated": 0,
        "items_picked_up": 0,
        "party_invites_accepted": 0,
        "party_invites_declined": 0,
        "trades_handled": 0,
        "trades_accepted": 0,
        "default_responses": 0,
    })

    # ── Configuration overrides ──
    level_threshold: int = _DEFAULT_LEVEL_THRESHOLD
    pickup_value_threshold: int = _DEFAULT_PICKUP_VALUE_THRESHOLD
    cooldown_seconds: int = _COOLDOWN_SECONDS

    # ── Optional callbacks for external services ──
    _get_market_price: Callable[[str], int | None] | None = None
    _cast_buff_skill: Callable[[str], bool] | None = None
    _use_butterfly_wing: Callable[[], bool] | None = None
    _pickup_item: Callable[[str, int, int], bool] | None = None
    _send_whisper: Callable[[str, str], bool] | None = None
    _party_accept: Callable[[str], bool] | None = None
    _party_decline: Callable[[str], bool] | None = None
    _trade_accept: Callable[[str], bool] | None = None
    _trade_decline: Callable[[str], bool] | None = None
    _get_player_level: Callable[[str], int | None] | None = None
    _get_self_level: Callable[[], int] | None = None
    _get_inventory: Callable[[], list[dict[str, Any]]] | None = None

    _enqueue_fn: Callable | None = None

    # ── Helpers ──────────────────────────────────────────────────────────

    def _now(self) -> float:
        return time.time()

    def _record_interaction(
        self,
        player_name: str,
        interaction_type: str,
        detail: str,
        response: str | None = None,
    ) -> None:
        """Record an interaction in the player's history."""
        now = self._now()
        record = InteractionRecord(
            interaction_type=interaction_type,
            detail=detail,
            timestamp=now,
            response=response,
        )
        if player_name not in self._history:
            self._history[player_name] = PlayerHistory(
                player_name=player_name,
                first_seen=now,
            )
        ph = self._history[player_name]
        ph.interactions.append(record)
        ph.last_seen = now
        ph.total_interactions += 1
        # Trim to max size
        if len(ph.interactions) > _MAX_HISTORY_PER_PLAYER:
            ph.interactions = ph.interactions[-_MAX_HISTORY_PER_PLAYER:]

    def _is_on_cooldown(self, player_name: str, interaction_type: str) -> bool:
        """Check if we're on cooldown for this interaction type with this player."""
        now = self._now()
        player_cooldowns = self._cooldowns.get(player_name, {})
        last_time = player_cooldowns.get(interaction_type, 0.0)
        return (now - last_time) < self.cooldown_seconds

    def _set_cooldown(self, player_name: str, interaction_type: str) -> None:
        """Set the cooldown timestamp for this interaction type."""
        self._cooldowns[player_name][interaction_type] = self._now()

    def _spam_check(self, player_name: str, interaction_type: str) -> bool:
        """Check if this interaction would be spam given recent history.
        
        Returns True if the interaction should proceed, False if it should
        be suppressed (spam prevention).
        """
        if self._is_on_cooldown(player_name, interaction_type):
            logger.debug(
                "social_spam_suppressed: player=%s type=%s (cooldown)",
                player_name, interaction_type,
            )
            return False
        self._set_cooldown(player_name, interaction_type)
        return True

    def _log(self, msg: str) -> None:
        if self._enqueue_fn:
            self._enqueue_fn("default", f"social_log {msg}")

    # ── Whisper Intent Parsing ───────────────────────────────────────────

    @staticmethod
    def parse_whisper(text: str) -> WhisperIntent:
        """Parse a whisper message and classify its intent.

        Args:
            text: The raw whisper text from another player.

        Returns:
            A WhisperIntent with classification flags.
        """
        lower = text.lower().strip()
        intent = WhisperIntent(raw=text, lower=lower)

        # Buff acknowledgement (someone saying "ty" after we buffed them)
        thanks_keywords = ["ty", "thx", "thanks", "thank you", "<3", "tq"]
        buff_refs = ["buff", "bless", "agi"]
        if any(t in lower for t in thanks_keywords) and any(b in lower for b in buff_refs):
            intent.is_buff_ack = True
            intent.confidence = max(intent.confidence, 0.8)

        # Buff request — only if NOT a buff acknowledgement (avoids collision on "ty for buff")
        if not intent.is_buff_ack:
            buff_keywords = ["buff", "bless", "agi", "blessing", "improve",
                             "magnificat", "gloria", "kyrie", "aspersio"]
            if any(kw in lower for kw in buff_keywords):
                intent.is_buff_request = True
                intent.confidence = max(intent.confidence, 0.7)
                # Phrases like "buff pls", "buff please", "need buff"
                if "pls" in lower or "please" in lower or "need" in lower or "buff" in lower:
                    intent.confidence = max(intent.confidence, 0.9)

        # Price query — must contain a price-related keyword
        price_keywords = ["price", "cost", "how much", "worth", "sell", "buy"]
        if any(kw in lower for kw in price_keywords):
            intent.is_price_query = True
            intent.confidence = max(intent.confidence, 0.6)
            # More specific: "price?" or "how much is X"
            if lower in ("price", "price?", "how much") or "price" in lower:
                intent.confidence = max(intent.confidence, 0.85)

        # Party query
        party_keywords = ["party", "group", "team", "pt", "part"]
        if any(kw in lower for kw in party_keywords):
            intent.is_party_query = True
            intent.confidence = max(intent.confidence, 0.7)

        # Greeting
        greeting_keywords = ["hi", "hello", "hey", "sup", "yo", "howdy",
                             "greetings", "good morning", "good evening"]
        if any(lower.startswith(kw) or lower == kw for kw in greeting_keywords):
            intent.is_greeting = True
            intent.confidence = max(intent.confidence, 0.8)

        # Warp request
        warp_keywords = ["warp", "teleport", "fly wing", "butterfly", "port"]
        if any(kw in lower for kw in warp_keywords):
            intent.is_warp_request = True
            intent.confidence = max(intent.confidence, 0.7)

        # Buff acknowledgement (someone saying "ty" after we buffed them)
        thanks_keywords = ["ty", "thx", "thanks", "thank you", "<3", "tq"]
        buff_refs = ["buff", "bless", "agi"]
        if any(t in lower for t in thanks_keywords) and any(b in lower for b in buff_refs):
            intent.is_buff_ack = True
            intent.confidence = max(intent.confidence, 0.8)

        return intent

    # ── WHISPER_HANDLER ──────────────────────────────────────────────────

    def handle_whisper(self, sender: str, text: str) -> str | None:
        """Handle an incoming whisper from another player.

        Parses the message, selects an appropriate response, and returns
        the response string. Returns None if no response should be sent
        (spam suppression or cooldown).

        Args:
            sender: Name of the player who whispered.
            text: The whisper message text.

        Returns:
            Response string to send, or None to stay silent.
        """
        with self._lock:
            if not self._spam_check(sender, "whisper"):
                return None

            intent = self.parse_whisper(text)
            response: str | None = None

            # Priority order: most specific intents first

            if intent.is_buff_request:
                response = self._handle_buff_request(sender, text)

            elif intent.is_price_query:
                response = self._handle_price_query(sender, text)

            elif intent.is_party_query:
                response = self._handle_party_query(sender, text)

            elif intent.is_greeting:
                response = "hello :)"

            elif intent.is_warp_request:
                response = self._handle_warp_request()

            elif intent.is_buff_ack:
                response = self._handle_buff_ack(sender)

            else:
                # DEFAULT_RESPONSE
                response = self._default_whisper_response(text)

            # Record the interaction
            self._record_interaction(
                player_name=sender,
                interaction_type="whisper",
                detail=text[:100],
                response=response,
            )
            self._stats["whispers_handled"] += 1
            if response is None:
                self._stats["default_responses"] += 1

            logger.info(
                "social_whisper: from=%s text='%s' intent=%s response='%s'",
                sender, text[:60],
                self._summarize_intent(intent),
                (response or "(silent)")[:60],
            )

            return response

    def _summarize_intent(self, intent: WhisperIntent) -> str:
        flags = []
        if intent.is_buff_request:
            flags.append("buff")
        if intent.is_price_query:
            flags.append("price")
        if intent.is_party_query:
            flags.append("party")
        if intent.is_greeting:
            flags.append("greet")
        if intent.is_warp_request:
            flags.append("warp")
        if intent.is_buff_ack:
            flags.append("buff_ack")
        return "+".join(flags) if flags else "default"

    def _handle_buff_request(self, sender: str, text: str) -> str:
        """Handle a buff request whisper.
        
        Responds 'sure' and fires the cast_buff_skill callback if available.
        """
        # Determine which buff was requested
        lower = text.lower()
        buff_skill = "blessing"  # Default blessing
        if "agi" in lower:
            buff_skill = "increase_agility"
        elif "bless" in lower:
            buff_skill = "blessing"
        elif "kyrie" in lower:
            buff_skill = "kyrie_eleison"
        elif "gloria" in lower:
            buff_skill = "gloria"
        elif "magnificat" in lower:
            buff_skill = "magnificat"
        elif "aspersio" in lower:
            buff_skill = "aspersio"
        elif "improve" in lower:
            buff_skill = "improve_concentration"

        # Attempt to cast the buff
        if self._cast_buff_skill is not None:
            try:
                self._cast_buff_skill(buff_skill)
            except Exception as exc:
                logger.warning("social_cast_buff_failed: skill=%s error=%s", buff_skill, exc)

        return f"sure, casting {buff_skill.replace('_', ' ')}"

    def _handle_price_query(self, sender: str, text: str) -> str:
        """Handle a price query whisper.
        
        Extracts item name from the text and looks up market price.
        Falls back to a generic response if the item can't be identified.
        """
        # Try to extract item name from the text
        lower = text.lower()
        # Remove common price query prefixes
        for prefix in ["price ", "price?", "how much ", "how much is ",
                        "how much for ", "cost of ", "price of "]:
            if lower.startswith(prefix):
                item_name = text[len(prefix):].strip()
                break
            elif prefix.rstrip() in lower:
                item_name = text.strip()
                break
        else:
            # If no prefix matched, use the whole text minus '?' and common words
            item_name = text.strip().rstrip("?").strip()

        # Skip pure queries like "price?", "how much" — no item to look up
        generic_queries = {"price", "how much", "cost", "worth", "sell", "buy"}
        if item_name.lower().strip() in generic_queries:
            return "which item are you asking about?"

        if self._get_market_price is not None:
            try:
                price = self._get_market_price(item_name)
                if price is not None and price > 0:
                    return f"{item_name} is around {price}z in the market"
                else:
                    # Item not found in market data — give a vague answer
                    return f"not sure about {item_name}, check the shops"
            except Exception as exc:
                logger.warning("social_price_lookup_failed: item=%s error=%s", item_name, exc)

        return "check the market vendors in town"

    def _handle_party_query(self, sender: str, text: str) -> str:
        """Handle a party invite/query whisper.
        
        Checks level difference and decides whether to invite or decline.
        """
        sender_level = None
        if self._get_player_level is not None:
            try:
                sender_level = self._get_player_level(sender)
            except Exception:
                pass

        self_level = None
        if self._get_self_level is not None:
            try:
                self_level = self._get_self_level()
            except Exception:
                pass

        # If we have level data, make an informed decision
        if sender_level is not None and self_level is not None:
            level_diff = abs(sender_level - self_level)
            if level_diff <= self.level_threshold:
                return f"sure, invite me! ({sender_level}/{self_level})"
            else:
                return (
                    f"sorry, level gap too big ({sender_level} vs {self_level}). "
                    f"try someone closer to your level"
                )

        # No level data — accept by default to be social
        return "sure, invite me!"

    def _handle_warp_request(self) -> str:
        """Handle a warp request.
        
        Attempts to use Butterfly Wing if callback is available, otherwise
        politely declines.
        """
        if self._use_butterfly_wing is not None:
            try:
                success = self._use_butterfly_wing()
                if success:
                    return "using butterfly wing"
            except Exception as exc:
                logger.warning("social_butterfly_wing_failed: %s", exc)

        return "sorry, no warp skill"

    def _handle_buff_ack(self, sender: str) -> str:
        """Handle a buff acknowledgement from another player."""
        self._stats["buff_acks_sent"] += 1
        response = random.choice(["np!", "you're welcome!", "anytime!", "enjoy!", "👍"])
        self._record_interaction(
            player_name=sender,
            interaction_type="buff_ack",
            detail="ty for buff",
            response=response,
        )
        return response

    def _default_whisper_response(self, text: str) -> str:
        """DEFAULT_RESPONSE for unrecognized whispers — polite 'sorry?'"""
        # Add variety so it doesn't look like a bot
        return random.choice([
            "sorry?",
            "hmm?",
            "what was that?",
            "i didn't catch that",
            "say again?",
            "hm?",
        ])

    # ── BUFF_ACK ──────────────────────────────────────────────────────────

    def handle_buff_received(self, caster_name: str, buff_name: str) -> str | None:
        """Called when a priest/ally casts a buff on this bot.

        Args:
            caster_name: The player who cast the buff.
            buff_name: Name of the buff skill cast.

        Returns:
            A thank-you response string, or None if suppressed via cooldown.
        """
        with self._lock:
            interaction_key = f"buff_received_{caster_name}"
            if not self._spam_check(caster_name, interaction_key):
                return None

            self._stats["buff_acks_sent"] += 1
            response = "ty"

            self._record_interaction(
                player_name=caster_name,
                interaction_type="buff_received",
                detail=f"received {buff_name}",
                response=response,
            )
            logger.info(
                "social_buff_ack: caster=%s buff=%s response='%s'",
                caster_name, buff_name, response,
            )
            return response

    # ── ITEM_EVAL ─────────────────────────────────────────────────────────

    def handle_drop_nearby(
        self,
        item_name: str,
        item_id: int | None,
        x: int,
        y: int,
        dropper: str,
    ) -> str | None:
        """Evaluate a dropped item and decide whether to pick it up.

        Args:
            item_name: Name of the dropped item.
            item_id: Optional server-side item ID.
            x, y: Map coordinates of the drop.
            dropper: Player who dropped the item.

        Returns:
            "pickup" if item is worth picking up, "ignore" if not, or None
            if suppressed by cooldown.
        """
        with self._lock:
            interaction_key = f"drop_{dropper}"
            if not self._spam_check(dropper, interaction_key):
                return None

            self._stats["items_evaluated"] += 1

            drop = ItemDrop(
                item_name=item_name,
                item_id=item_id,
                x=x, y=y,
                dropped_by=dropper,
                timestamp=self._now(),
            )
            self._recent_drops.append(drop)
            # Trim drop list
            if len(self._recent_drops) > 100:
                self._recent_drops = self._recent_drops[-100:]

            # Evaluate value
            estimated_value = self._estimate_item_value(item_name, item_id)

            if estimated_value >= self.pickup_value_threshold:
                self._stats["items_picked_up"] += 1
                if self._pickup_item is not None:
                    try:
                        self._pickup_item(item_name, x, y)
                    except Exception as exc:
                        logger.warning(
                            "social_pickup_failed: item=%s error=%s",
                            item_name, exc,
                        )
                self._record_interaction(
                    player_name=dropper,
                    interaction_type="item_pickup",
                    detail=f"picked up {item_name} (value={estimated_value})",
                )
                logger.info(
                    "social_item_pickup: item=%s value=%d dropper=%s",
                    item_name, estimated_value, dropper,
                )
                return "pickup"

            self._record_interaction(
                player_name=dropper,
                interaction_type="item_ignore",
                detail=f"ignored {item_name} (value={estimated_value})",
            )
            logger.debug(
                "social_item_ignore: item=%s value=%d (below threshold %d)",
                item_name, estimated_value, self.pickup_value_threshold,
            )
            return "ignore"

    def _estimate_item_value(self, item_name: str, item_id: int | None) -> int:
        """Estimate the value of an item, using market data if available."""
        if self._get_market_price is not None:
            try:
                price = self._get_market_price(item_name)
                if price is not None:
                    return price
            except Exception:
                pass

        # Fallback: heuristic based on item name
        lower = item_name.lower()
        # Cards are always valuable
        if "card" in lower:
            return 50000
        # Equipment is usually worth something
        equip_keywords = ["sword", "staff", "bow", "mace", "knife", "dagger",
                          "shield", "armor", "boots", "muffler", "robe",
                          "hat", "helm", "goggles", "ring", "earring",
                          "necklace", "clip", "brooch", "belt"]
        if any(kw in lower for kw in equip_keywords):
            return 5000
        # Consumables
        if any(kw in lower for kw in ["potion", "fruit", "berry", "food"]):
            return 500
        # Materials
        if any(kw in lower for kw in ["elunium", "oridecon", "rough", "dust",
                                       "hammer", "anvil", "shard", "gemstone"]):
            return 2000
        # Default low value
        return 100

    # ── PARTY_INVITE ─────────────────────────────────────────────────────

    def handle_party_invite(
        self,
        leader_name: str,
        leader_level: int,
        leader_map: str,
        leader_job: str | None = None,
    ) -> bool:
        """Handle an incoming party invite.

        Args:
            leader_name: Name of the party leader.
            leader_level: Level of the party leader.
            leader_map: Map the leader is on.
            leader_job: Optional job class of the leader.

        Returns:
            True if the invite was accepted, False if declined or suppressed.
        """
        with self._lock:
            interaction_key = f"party_invite_{leader_name}"
            if not self._spam_check(leader_name, interaction_key):
                return False

            self_level = 0
            if self._get_self_level is not None:
                try:
                    self_level = self._get_self_level() or 0
                except Exception:
                    pass

            invite = PartyInvite(
                leader_name=leader_name,
                leader_level=leader_level,
                leader_map=leader_map,
                leader_job=leader_job,
            )
            self._pending_party_invites[leader_name] = invite

            # Decision logic: accept if level-appropriate
            level_diff = abs(leader_level - self_level) if self_level > 0 else 0

            if level_diff <= self.level_threshold:
                # Accept invite
                self._stats["party_invites_accepted"] += 1
                if self._party_accept is not None:
                    try:
                        self._party_accept(leader_name)
                    except Exception as exc:
                        logger.warning(
                            "social_party_accept_failed: leader=%s error=%s",
                            leader_name, exc,
                        )
                        return False

                self._record_interaction(
                    player_name=leader_name,
                    interaction_type="party_invite",
                    detail=f"accepted from {leader_name} (lvl {leader_level}, gap {level_diff})",
                    response="accepted",
                )
                logger.info(
                    "social_party_accepted: leader=%s lvl=%d self_lvl=%d gap=%d",
                    leader_name, leader_level, self_level, level_diff,
                )
                return True
            else:
                # Decline — level gap too large
                self._stats["party_invites_declined"] += 1
                if self._party_decline is not None:
                    try:
                        self._party_decline(leader_name)
                    except Exception as exc:
                        logger.warning(
                            "social_party_decline_failed: leader=%s error=%s",
                            leader_name, exc,
                        )

                self._record_interaction(
                    player_name=leader_name,
                    interaction_type="party_invite",
                    detail=f"declined from {leader_name} (lvl {leader_level}, gap {level_diff})",
                    response="declined",
                )
                logger.info(
                    "social_party_declined: leader=%s lvl=%d self_lvl=%d gap=%d",
                    leader_name, leader_level, self_level, level_diff,
                )
                return False

    # ── TRADE_REQUEST ────────────────────────────────────────────────────

    def handle_trade_request(self, player_name: str) -> str | None:
        """Handle an incoming trade request from another player.

        Args:
            player_name: The player requesting a trade.

        Returns:
            A message describing available items for sale, or None if suppressed.
        """
        with self._lock:
            interaction_key = f"trade_{player_name}"
            if not self._spam_check(player_name, interaction_key):
                return None

            self._stats["trades_handled"] += 1

            # Gather items available for trade from inventory
            sale_items: list[dict[str, Any]] = []
            if self._get_inventory is not None:
                try:
                    inventory = self._get_inventory() or []
                    for item in inventory:
                        if isinstance(item, dict) and item.get("sellable", False):
                            sale_items.append({
                                "name": item.get("name", "unknown"),
                                "qty": item.get("quantity", 1),
                                "price": item.get("estimated_price", 0),
                            })
                except Exception as exc:
                    logger.warning("social_inventory_lookup_failed: %s", exc)

            # Build response
            if sale_items:
                # Present up to 5 items for sale
                items_str = ", ".join(
                    f"{i['name']} x{i['qty']} @{i['price']}z"
                    for i in sale_items[:5]
                )
                if len(sale_items) > 5:
                    items_str += f" (and {len(sale_items) - 5} more)"
                response = f"i have: {items_str}"
            else:
                response = "nothing for sale right now, sorry"

            self._record_interaction(
                player_name=player_name,
                interaction_type="trade_request",
                detail=f"trade from {player_name}",
                response=response,
            )
            logger.info(
                "social_trade_request: from=%s items_available=%d",
                player_name, len(sale_items),
            )
            return response

    # ── Utility ──────────────────────────────────────────────────────────

    def get_player_history(self, player_name: str) -> PlayerHistory | None:
        """Get the interaction history for a specific player.

        Args:
            player_name: Name of the player.

        Returns:
            PlayerHistory or None if no interactions recorded.
        """
        with self._lock:
            return self._history.get(player_name)

    def get_recent_interactions(
        self,
        limit: int = 20,
        player_name: str | None = None,
    ) -> list[InteractionRecord]:
        """Get recent interactions, optionally filtered by player.

        Args:
            limit: Maximum number of records to return.
            player_name: Optional filter by player name.

        Returns:
            List of recent InteractionRecords (newest first).
        """
        with self._lock:
            if player_name:
                ph = self._history.get(player_name)
                if ph is None:
                    return []
                records = list(reversed(ph.interactions))
            else:
                # Collect all records across all players
                all_records: list[InteractionRecord] = []
                for ph in self._history.values():
                    all_records.extend(ph.interactions)
                records = sorted(
                    all_records, key=lambda r: r.timestamp, reverse=True,
                )
            return records[:limit]

    def get_stats(self) -> dict[str, int]:
        """Get interaction statistics.

        Returns:
            Copy of the internal stats dict.
        """
        with self._lock:
            return dict(self._stats)

    def get_total_players_interacted(self) -> int:
        """Get the number of unique players interacted with.

        Returns:
            Count of unique players.
        """
        with self._lock:
            return len(self._history)

    def reset_stats(self) -> None:
        """Reset all interaction statistics."""
        with self._lock:
            for key in self._stats:
                self._stats[key] = 0
            logger.info("social_stats_reset")

    def expire_old_drops(self, max_age_seconds: int = 300) -> None:
        """Remove drop entries older than max_age_seconds.

        Args:
            max_age_seconds: Maximum age of drops to keep (default 5 min).
        """
        now = self._now()
        with self._lock:
            self._recent_drops = [
                d for d in self._recent_drops
                if now - d.timestamp <= max_age_seconds
            ]


# ── Factory function ──────────────────────────────────────────────────────

_SocialInteractionEngine: SocialInteractionEngine | None = None
_engine_lock = RLock()


def get_social_interaction_engine() -> SocialInteractionEngine:
    """Factory function — returns the global SocialInteractionEngine singleton.

    Thread-safe. Creates the engine on first call.
    """
    global _SocialInteractionEngine
    with _engine_lock:
        if _SocialInteractionEngine is None:
            _SocialInteractionEngine = SocialInteractionEngine()
            logger.info("SocialInteractionEngine created")
        return _SocialInteractionEngine
