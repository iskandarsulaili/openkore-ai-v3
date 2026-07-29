"""Social Reputation System — whisper responses, avoid KS, rotate spots.

On a bot-allowed server, bots still need basic social awareness:
- Don't KS (kill-steal) other players' monsters
- Respond to whispers with pre-written responses
- Rotate farming spots to avoid over-farming
- Track which players are friendly/hostile
"""
from __future__ import annotations
from typing import Any
import logging
from datetime import datetime, timedelta
from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)


class ReputationTracker:
    """Tracks social interactions with other players.
    
    Fields per player: first_seen, last_seen, interactions, reputation_score.
    """
    
    def __init__(self):
        self._players: dict[str, dict] = {}
    
    def record_interaction(self, player_name: str, interaction_type: str) -> None:
        """Record an interaction with a player."""
        now = datetime.now()
        if player_name not in self._players:
            self._players[player_name] = {
                "first_seen": now,
                "interactions": [],
                "reputation": 0,
            }
        p = self._players[player_name]
        p["last_seen"] = now
        p["interactions"].append({"type": interaction_type, "time": now})
        p["reputation"] = self._calculate_reputation(p["interactions"])
    
    def _calculate_reputation(self, interactions: list[dict]) -> int:
        """Calculate reputation score from interactions."""
        score = 0
        for i in interactions:
            t = i["type"]
            if t == "shared_spot":
                score += 2
            elif t == "whisper_friendly":
                score += 1
            elif t == "party_invite_accept":
                score += 3
            elif t == "ks_detected":
                score -= 5
            elif t == "hostile_whisper":
                score -= 3
            elif t == "reported":
                score -= 10
        return max(-20, min(20, score))
    
    def get_reputation(self, player_name: str) -> int:
        p = self._players.get(player_name)
        return p["reputation"] if p else 0
    
    def is_friendly(self, player_name: str) -> bool:
        return self.get_reputation(player_name) >= 3
    
    def is_hostile(self, player_name: str) -> bool:
        return self.get_reputation(player_name) <= -5


class WhisperResponder:
    """Responds to whispers with pre-written human-like responses.
    
    Response categories:
    - Trade: "Trading bot, send offer" 
    - Party: "In a party, sorry"
    - GM: "I'm just playing" (vague, non-committal)
    - Spam: (ignore)
    """
    
    RESPONSES = {
        "party": [
            "sorry im in a party already",
            "already in a group thanks",
            "maybe later im farming atm",
        ],
        "trade": [
            "check my shop in town",
            "im just farming sry",
            "not selling rn",
        ],
        "gm": [
            "yeah im just playing",
            "nothing special just farming",
            "sure ill let you know if i see something",
        ],
        "default": [
            "huh?",
            "im busy rn talk later",
            "what?",
        ],
    }
    
    @staticmethod
    def get_response(whisper_text: str) -> str:
        """Get a pre-written response for a whisper."""
        text_lower = whisper_text.lower()
        
        if any(w in text_lower for w in ["party", "group", "team", "join"]):
            responses = WhisperResponder.RESPONSES["party"]
        elif any(w in text_lower for w in ["buy", "sell", "trade", "price", "shop", "zeny"]):
            responses = WhisperResponder.RESPONSES["trade"]
        elif any(w in text_lower for w in ["gm", "admin", "mod", "staff", "question"]):
            responses = WhisperResponder.RESPONSES["gm"]
        else:
            responses = WhisperResponder.RESPONSES["default"]
        
        import random
        return random.choice(responses)


class KSAvoidance:
    """Detects and avoids kill-stealing.
    
    If another player is already attacking a monster, don't attack it.
    If a player is farming in the same spot, move to a different spot.
    """
    
    def __init__(self):
        self._recent_attacks: dict[str, datetime] = {}  # monster_id -> time
        self._player_zones: dict[str, list] = {}  # map -> [(x, y, radius)]
    
    def is_monster_claimed(self, monster_id: str, timeout_seconds: int = 15) -> bool:
        """Check if another player is already attacking this monster."""
        if monster_id in self._recent_attacks:
            age = (datetime.now() - self._recent_attacks[monster_id]).total_seconds()
            return age < timeout_seconds
        return False
    
    def register_player_attack(self, monster_id: str) -> None:
        """Register that another player attacked this monster."""
        self._recent_attacks[monster_id] = datetime.now()
    
    def is_in_occupied_zone(self, map_name: str, x: int, y: int, zone_radius: int = 10) -> bool:
        """Check if a player is farming in the same spot."""
        zones = self._player_zones.get(map_name, [])
        for zx, zy, zr in zones:
            dist = ((x - zx) ** 2 + (y - zy) ** 2) ** 0.5
            if dist < (zone_radius + zr):
                return True
        return False


class FarmingSpotRotator:
    """Rotates farming spots periodically.
    
    Instead of staying on one map forever (which looks suspicious and
    triggers overdue penalties), rotate between 2-3 spots.
    """
    
    def __init__(self):
        self._current_spot_index: dict[str, int] = {}  # bot_id -> index
        self._spot_enter_time: dict[str, datetime] = {}  # bot_id -> time
    
    def get_next_spot(self, bot_id: str, available_spots: list[str], 
                      current_map: str, minutes_on_map: float) -> str | None:
        """Get the next spot to farm.
        
        Rotates if:
        - Been on current map > 30 minutes (overdue penalty)
        - Current map not in available spots (shouldn't be here)
        - Available spots have better options
        """
        if not available_spots:
            return None
        
        if bot_id not in self._current_spot_index:
            self._current_spot_index[bot_id] = 0
            self._spot_enter_time[bot_id] = datetime.now()
        
        # Check if we should rotate
        if minutes_on_map > 30 and current_map in available_spots:
            # Rotate to next spot
            idx = self._current_spot_index[bot_id]
            idx = (idx + 1) % len(available_spots)
            self._current_spot_index[bot_id] = idx
            self._spot_enter_time[bot_id] = datetime.now()
            return available_spots[idx]
        
        return current_map if current_map in available_spots else available_spots[0]


class SocialReputationDomain:
    """Domain that combines all social awareness features."""
    
    def __init__(self):
        self.reputation = ReputationTracker()
        self.whisper = WhisperResponder()
        self.ks = KSAvoidance()
        self.rotator = FarmingSpotRotator()
    
    def assess(self, signals: dict[str, Any], actions: list[HeuristicAction], bot_id: str) -> None:
        current_map = str(signals.get("map", "") or "")
        
        # Check for whisper responses
        whisper = signals.get("whisper", {}) or {}
        if isinstance(whisper, dict) and whisper.get("text"):
            response = self.whisper.get_response(whisper.get("text", ""))
            actions.append(HeuristicAction(
                kind="command",
                command=f"reply {whisper.get('from', '')} {response}",
                confidence=0.7,
                reason=f"Social: responded to whisper from {whisper.get('from', 'unknown')}",
                domain="social",
            ))
        
        # Check for KS detection
        other_players = signals.get("other_players", []) or []
        if isinstance(other_players, list) and len(other_players) > 0:
            # If other players are nearby, avoid their target
            if not getattr(self, '_ks_adjacent', False):
                actions.append(HeuristicAction(
                    kind="command",
                    command="mon_control * 1 0 1",
                    confidence=0.5,
                    reason=f"Social: {len(other_players)} players nearby — avoid KS",
                    domain="social",
                ))
                self._ks_adjacent = True
        else:
            self._ks_adjacent = False
        
        # Log social awareness
        actions.append(HeuristicAction(
            kind="log",
            command=f"social map={current_map} players={len(other_players)}",
            confidence=0.5,
            reason=f"Social awareness on {current_map}",
            domain="social",
        ))
