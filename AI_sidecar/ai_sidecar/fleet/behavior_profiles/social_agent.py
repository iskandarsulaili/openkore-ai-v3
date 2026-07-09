"""SocialAgent — party, guild, auto-response, friend list management."""

from __future__ import annotations

import re
import time
from typing import Any

from ai_sidecar.fleet.behavior_profiles import BehaviorProfile


class SocialAgent(BehaviorProfile):
    """Handles RO social mechanics — party, guild, whispers, friends."""

    def __init__(self, bot_id: str, experience_db=None):
        super().__init__(bot_id, experience_db)
        self._auto_replies: dict[str, tuple[str, float]] = {}
        self._ignored_players: set[str] = set()
        self._friend_list: set[str] = set()

    def party_invite_decision(self, inviter: str, inviter_level: int,
                              my_level: int, my_role: str) -> dict[str, Any]:
        if self._signals.get("in_party", False):
            return {"action": "decline", "reason": "already_in_party"}
        level_diff = abs(inviter_level - my_level)
        if level_diff > 20:
            return {"action": "decline", "reason": "level_gap_too_large"}
        return {"action": "accept_invite", "inviter": inviter,
                "my_role": my_role}

    def party_management(self, party_members: list[dict[str, Any]],
                         desired_composition: dict[str, int]) -> dict[str, Any]:
        current_roles = {m.get("role", "") for m in party_members}
        missing = [r for r, needed in desired_composition.items()
                   if r not in current_roles and needed > 0]
        if missing:
            return {"action": "recruit_for_role", "missing_roles": missing,
                    "message": f"Looking for {', '.join(missing)}"}
        return {"action": "party_ready", "composition": current_roles}

    def guild_decision(self, guild_name: str, guild_level: int,
                       guild_members: int) -> dict[str, Any]:
        if guild_level < 3 or guild_members < 5:
            return {"action": "decline", "reason": "guild_too_small"}
        if self._signals.get("guild_id", ""):
            return {"action": "already_in_guild"}
        return {"action": "accept_guild_invite", "guild": guild_name}

    def auto_response(self, whisper: str, sender: str) -> dict[str, Any]:
        if sender in self._ignored_players:
            return {"action": "ignore"}
        now = time.time()
        if sender in self._auto_replies:
            _, last_time = self._auto_replies[sender]
            if now - last_time < 10:
                return {"action": "cooldown", "reason": "rate_limit"}
        lower = whisper.lower()
        if re.search(r"\b(buy|sell|trade|price|wts|wtb)\b", lower):
            self._auto_replies[sender] = ("trade_reply", now)
            return {"action": "reply", "message": "I'm automated. Use /vendor to browse my shop."}
        if re.search(r"\b(party|invite|group|plz)\b", lower):
            return {"action": "reply", "message": "Auto-party is enabled. Sending invite."}
        if re.search(r"\b(heal|buff|bless|agi)\b", lower):
            return {"action": "reply", "message": "Buff request noted. Casting if in range."}
        best, score = self.best_action("social")
        if best and score > 0.5:
            return {"action": "dynamic_reply", "reply_template": best}
        return {"action": "no_reply", "reason": "no_trigger_detected"}

    def friend_management(self, friends_online: list[str],
                          friends_offline: list[str],
                          pending_requests: list[str]) -> dict[str, Any]:
        actions = []
        for requester in pending_requests:
            if requester not in self._friend_list:
                self._friend_list.add(requester)
                actions.append({"action": "accept_friend", "player": requester})
        if not friends_online and self._signals.get("map_is_safe", False):
            return {"action": "no_online_friends", "friend_count": len(self._friend_list)}
        return {"action": "friends_available", "online": friends_online[:5],
                "total": len(self._friend_list)}

    def auto_greet(self, nearby_players: list[str], my_map: str) -> dict[str, Any]:
        if nearby_players and self._signals.get("greet_enabled", True):
            return {"action": "greet", "message": "Hello! I'm an automated adventurer.",
                    "target_count": len(nearby_players)}
        return {"action": "no_greet"}

    def record_outcome(self, action: str, success: bool) -> None:
        self._record_experience("social", action, success, reward=1.0 if success else 0.0)
