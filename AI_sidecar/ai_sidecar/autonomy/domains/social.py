"""Social domain — party, guild, chat decisions.

Extracted from heuristic_service.py lines 2118-2256 (party check),
3004-3063 (PARTY state), 3504-3555 (in-town party).
"""

from __future__ import annotations

import logging
from typing import Any

from ai_sidecar.autonomy.domains import BaseDomain
from ai_sidecar.autonomy.heuristic_service import HeuristicAction

logger = logging.getLogger(__name__)


class SocialDomain(BaseDomain):
    name: str = "social"
    priority: int = 35

    TOWN_MAPS: tuple[str, ...] = (
        "prontera", "morocc", "geffen", "payon", "alberta",
        "izlude", "aldebaran", "comodo", "umbala", "niflheim",
        "louyang", "einbroch", "lighthalzen", "rachel", "veins",
        "juno", "yuno",
    )

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        service: Any,
    ) -> None:
        """Evaluate social/party decisions.

        Handles: party creation (leader), party join/auto-accept (joiner),
        party leave (level < 40), stale party detection.
        """
        bot_id = service._resolve_bot_id(signals)
        state = service._get_state(signals, bot_id)
        base_level = int(signals.get("base_level", 1) or 1)

        if state in ("COLD_START", "DEAD"):
            return

        # ── DIRECT PARTY CHECK (always runs) ──
        _party_in = signals.get("in_party", False)
        _party_members = signals.get("party_members", []) or []
        _all_bots = signals.get("all_bots", []) or []

        # Death/respawn flicker guard: cache party state for 120s
        _now_t = __import__("time").time()
        _last_seen = service._last_party_seen.get(bot_id, 0)

        if (
            (not _party_in or not _all_bots)
            and _last_seen > 0
            and _now_t - _last_seen < 120
        ):
            _party_in = True
            _party_members = service._last_party_members.get(bot_id, [])
            _all_bots = service._all_bots_cache.get(bot_id, [])

        if _party_in and _all_bots:
            service._last_party_seen[bot_id] = _now_t
            service._last_party_members[bot_id] = _party_members
            service._all_bots_cache[bot_id] = _all_bots

        _bot_profile = bot_id.split(":")[-1].split("/")[-1] if ":" in bot_id else bot_id
        _sorted_bots = sorted(_all_bots)
        _is_leader = bool(_sorted_bots) and _bot_profile == _sorted_bots[0]
        _expected_count = len(_all_bots)
        _actual_count = len(_party_members)
        _party_incomplete = (_actual_count + 1) < _expected_count

        # ── PARTY LEAVE (level < 40: solo is faster) ──
        self._check_party_leave(actions, bot_id, base_level, state, service, party_in=_party_in)

        # ── LEADER party creation (level >= 40) ──
        if _is_leader and _party_incomplete and base_level >= 40:
            self._leader_party_check(actions, bot_id, _party_in, _party_members, _all_bots, _bot_profile)

        # ── JOINER party management ──
        if not _is_leader and _all_bots and base_level >= 40:
            self._joiner_party_check(actions, bot_id, _party_in, _party_members, _all_bots, _sorted_bots, signals, service)

    def _check_party_leave(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        base_level: int,
        state: str,
        service: Any,
        party_in: bool = False,
    ) -> None:
        """Force leave party if level < 40 (solo is faster).

        Only fires when the bot is ACTUALLY in a party — issuing
        'party leave' to a bot with no party produces
        "You're not in a party." error spam every cycle.
        """
        if base_level >= 40:
            return
        if not party_in:
            return  # not in a party — nothing to leave, no spam
        _now = __import__("time").time()
        _last_leave = service._last_party_leave.get(bot_id, 0)
        if _now - _last_leave > 30:
            service._last_party_leave[bot_id] = _now
            actions.append(HeuristicAction(
                kind="command", command="party leave",
                confidence=0.99, domain="social",
                reason=f"Level {base_level} < 40 - force leave party (solo is faster)",
            ))

    def _leader_party_check(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        party_in: bool,
        party_members: list,
        all_bots: list,
        bot_profile: str,
    ) -> None:
        """Leader: create or complete party."""
        _now = __import__("time").time()
        if party_in and len(party_members) > 0:
            # Already have a party - request missing members
            for _other_bot in all_bots:
                if _other_bot != bot_profile:
                    _char_name = _other_bot
                    _already_in = any(
                        _char_name.lower() in m.lower()
                        for m in party_members
                    )
                    if not _already_in:
                        actions.append(HeuristicAction(
                            kind="command",
                            command=f"party request {_char_name}",
                            confidence=0.95, domain="social",
                            reason=f"Party check - request {_other_bot} ({_char_name})",
                        ))
                    elif len(party_members) < 3:
                        # Stale check - retry
                        actions.append(HeuristicAction(
                            kind="command",
                            command=f"party request {_char_name}",
                            confidence=0.80, domain="social",
                            reason=f"Party check - retry {_other_bot} ({_char_name})",
                        ))
        else:
            # Not in party - create new one
            _ts = int(__import__("time").time())
            actions.append(HeuristicAction(
                kind="command", command="move prontera",
                confidence=0.99, domain="social",
                reason="Party check - move to town for party formation",
            ))
            actions.append(HeuristicAction(
                kind="command", command=f"party create AI{_ts}",
                confidence=0.95, domain="social",
                reason="Leader creates party",
            ))
            for _other_bot in all_bots:
                if _other_bot != bot_profile:
                    actions.append(HeuristicAction(
                        kind="command",
                        command=f"party request {_other_bot}",
                        confidence=0.95, domain="social",
                        reason=f"Request {_other_bot} to join",
                    ))
            actions.append(HeuristicAction(
                kind="command", command="party share exp",
                confidence=0.90, domain="social",
                reason="Share experience in party",
            ))

    def _joiner_party_check(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        party_in: bool,
        party_members: list,
        all_bots: list,
        sorted_bots: list,
        signals: dict[str, Any],
        service: Any,
    ) -> None:
        """Joiner: leave stale/wrong party, auto-accept, move to town."""
        _leader_char = (
            (getattr(service, '_profile_to_char', {}) or {}).get(
                sorted_bots[0], sorted_bots[0]
            )
            if sorted_bots else ""
        )
        _in_wrong_party = (
            party_in
            and len(party_members) == 1
            and _leader_char
            and _leader_char not in party_members
        )
        map_name = str(signals.get("map", "") or "").lower()
        _stuck_in_town = (
            not party_in
            and map_name
            and any(t in map_name for t in self.TOWN_MAPS)
        )

        if party_in and _in_wrong_party or _stuck_in_town and all_bots:
            if party_in:
                actions.append(HeuristicAction(
                    kind="command", command="party leave",
                    confidence=0.99, domain="social",
                    reason="Party check - leave stale/wrong party",
                ))
            actions.append(HeuristicAction(
                kind="command", command="set partyAuto 2",
                confidence=0.99, domain="social",
                reason="Set auto-accept party invites",
            ))
            actions.append(HeuristicAction(
                kind="command", command="move prontera",
                confidence=0.95, domain="social",
                reason="Move to town for party invitation",
            ))
            actions.append(HeuristicAction(
                kind="command", command="ai auto",
                confidence=0.95, domain="social",
                reason="Continue after party check",
            ))


def create_domain() -> SocialDomain:
    return SocialDomain()
