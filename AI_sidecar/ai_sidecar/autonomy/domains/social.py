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
            # OBSERVABILITY ONLY — the fleet coordinator is the party actor
            # (sidecar emission of `party leave` on a solo bot produced
            # "You're not in a party." spam every cycle). Log the intent so
            # the state is observable; never execute party commands here.
            actions.append(HeuristicAction(
                kind="log", command="party_leave_requested",
                confidence=0.99, domain="social",
                reason=f"Level {base_level} < 40 - force leave party (solo is faster)",
                metadata={"party_action": "leave", "party_in": party_in},
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
        """Leader: create or complete party (OBSERVABILITY ONLY).

        The fleet coordinator is the party actor — raw `party create /
        request / share` emissions from this domain produced the frozen
        party-request spam that the bridge gates had to block. Keep the
        analysis (who is missing, whether a party exists) observable as
        log intents; formation commands belong to the coordinator.
        """
        _now = __import__("time").time()
        missing = [
            _other_bot for _other_bot in all_bots
            if _other_bot != bot_profile
            and not any(
                str(_other_bot).lower() in m.lower() for m in party_members
            )
        ]
        if party_in and len(party_members) > 0:
            # Already have a party - report missing members
            for _other_bot in missing:
                actions.append(HeuristicAction(
                    kind="log", command="party_member_missing",
                    confidence=0.95, domain="social",
                    reason=f"Party check - {_other_bot} not in party",
                    metadata={"party_action": "request_pending", "member": _other_bot},
                ))
            if len(party_members) < 3:
                actions.append(HeuristicAction(
                    kind="log", command="party_incomplete",
                    confidence=0.80, domain="social",
                    reason=f"Party check - {len(party_members)}/{max(3, len(all_bots))} members",
                    metadata={"party_action": "retry_pending", "members": len(party_members)},
                ))
        else:
            # Not in party - report formation need (coordinator acts)
            _ts = int(__import__("time").time())
            actions.append(HeuristicAction(
                kind="log", command="party_formation_needed",
                confidence=0.99, domain="social",
                reason="Party check - leader would create party",
                metadata={"party_action": "create_pending", "suffix": f"AI{_ts}"},
            ))
            for _other_bot in missing:
                actions.append(HeuristicAction(
                    kind="log", command="party_invite_pending",
                    confidence=0.95, domain="social",
                    reason=f"Request {_other_bot} to join",
                    metadata={"party_action": "invite_pending", "member": _other_bot},
                ))
            actions.append(HeuristicAction(
                kind="log", command="party_share_pending",
                confidence=0.90, domain="social",
                reason="Share experience in party",
                metadata={"party_action": "share_pending"},
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
            # OBSERVABILITY ONLY — same doctrine as leader/leave paths:
            # report the intent; the fleet coordinator executes party
            # changes. `set partyAuto`/`ai auto` config flips also fought
            # the config-audit stickiness, so they're log-only here.
            if party_in:
                actions.append(HeuristicAction(
                    kind="log", command="party_stale_leave_pending",
                    confidence=0.99, domain="social",
                    reason="Party check - leave stale/wrong party",
                    metadata={"party_action": "leave_pending"},
                ))
            actions.append(HeuristicAction(
                kind="log", command="party_autojoin_pending",
                confidence=0.99, domain="social",
                reason="Set auto-accept party invites",
                metadata={"party_action": "autojoin_pending"},
            ))
            actions.append(HeuristicAction(
                kind="log", command="party_town_wait_pending",
                confidence=0.95, domain="social",
                reason="Move to town for party invitation",
                metadata={"party_action": "town_pending"},
            ))
            actions.append(HeuristicAction(
                kind="log", command="party_continue_pending",
                confidence=0.95, domain="social",
                reason="Continue after party check",
                metadata={"party_action": "continue_pending"},
            ))


def create_domain() -> SocialDomain:
    return SocialDomain()
