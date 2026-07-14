"""
Party Skill Protocol — coordinates skills between party members.

A pro player in a party knows: "I'm about to cast Storm Gust, don't scatter
the mobs." "I'm tanking, heal me in 3 seconds." "Boss is phasing, save skills."
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class SkillAnnouncement:
    """An announcement that a skill is about to be used."""
    skill_name: str
    caster_name: str
    cast_time_ms: int = 0
    remaining_ms: int = 0
    target_id: int = 0
    is_aoe: bool = False
    aoe_radius: int = 0
    aoe_x: int = 0
    aoe_y: int = 0
    timestamp: float = 0.0
    expires_at: float = 0.0


@dataclass
class PartyRequest:
    """A request from a party member."""
    request_type: str  # heal, buff, tank, dps, rescue
    requester_name: str
    urgency: int = 5  # 1-10
    timestamp: float = 0.0
    expires_at: float = 0.0
    description: str = ""


class PartySkillProtocol:
    """Coordinates skills between party members."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._announcements: list[SkillAnnouncement] = []
        self._requests: list[PartyRequest] = []
        self._max_announcements: int = 50
        self._max_requests: int = 50
        self._enqueue_fn: Callable | None = None
        self._party_leader: bool = False
        self._my_role: str = "dps"  # tank, healer, dps, support

    # ── Public API ──

    def announce_skill(self, skill_name: str, cast_time_ms: int = 0,
                       is_aoe: bool = False, aoe_radius: int = 0,
                       aoe_x: int = 0, aoe_y: int = 0,
                       target_id: int = 0) -> None:
        """Announce a skill about to be used."""
        with self._lock:
            now = time.time()
            announcement = SkillAnnouncement(
                skill_name=skill_name,
                caster_name="self",
                cast_time_ms=cast_time_ms,
                remaining_ms=cast_time_ms,
                target_id=target_id,
                is_aoe=is_aoe,
                aoe_radius=aoe_radius,
                aoe_x=aoe_x,
                aoe_y=aoe_y,
                timestamp=now,
                expires_at=now + 5.0,  # 5s expiry
            )
            self._announcements.append(announcement)
            if len(self._announcements) > self._max_announcements:
                self._announcements.pop(0)

            # Send party chat announcement
            if self._enqueue_fn:
                if is_aoe:
                    msg = f"AoE incoming: {skill_name} at ({aoe_x},{aoe_y}) radius {aoe_radius}"
                else:
                    msg = f"Casting {skill_name} on target {target_id}"
                self._enqueue_fn("self", f"p {msg}")

    def request_heal(self, requester_name: str = "self", urgency: int = 5) -> None:
        """Request healing from party healer."""
        with self._lock:
            now = time.time()
            request = PartyRequest(
                request_type="heal",
                requester_name=requester_name,
                urgency=urgency,
                timestamp=now,
                expires_at=now + 10.0,
                description=f"HP critical" if urgency >= 8 else f"Need heal",
            )
            self._requests.append(request)
            if len(self._requests) > self._max_requests:
                self._requests.pop(0)
            if self._enqueue_fn and urgency >= 7:
                self._enqueue_fn("self", f"p Need heal! ({requester_name})")

    def request_buff(self, buff_name: str, requester_name: str = "self") -> None:
        """Request a buff from party support."""
        with self._lock:
            now = time.time()
            request = PartyRequest(
                request_type="buff",
                requester_name=requester_name,
                urgency=3,
                timestamp=now,
                expires_at=now + 30.0,
                description=f"Need {buff_name}",
            )
            self._requests.append(request)
            if len(self._requests) > self._max_requests:
                self._requests.pop(0)

    def request_rescue(self, requester_name: str = "self") -> None:
        """Request rescue (aggro too high)."""
        with self._lock:
            now = time.time()
            request = PartyRequest(
                request_type="rescue",
                requester_name=requester_name,
                urgency=10,
                timestamp=now,
                expires_at=now + 15.0,
                description=f"Overwhelmed! Need help!",
            )
            self._requests.append(request)
            if len(self._requests) > self._max_requests:
                self._requests.pop(0)
            if self._enqueue_fn:
                self._enqueue_fn("self", f"p Help! Overwhelmed! ({requester_name})")

    def get_active_announcements(self) -> list[SkillAnnouncement]:
        """Get active skill announcements (not expired)."""
        with self._lock:
            now = time.time()
            active = [a for a in self._announcements if a.expires_at > now]
            self._announcements = [a for a in self._announcements if a.expires_at > now]
            return active

    def get_pending_requests(self, request_type: str | None = None) -> list[PartyRequest]:
        """Get pending party requests."""
        with self._lock:
            now = time.time()
            active = [r for r in self._requests if r.expires_at > now]
            self._requests = [r for r in self._requests if r.expires_at > now]
            if request_type:
                return [r for r in active if r.request_type == request_type]
            return active

    def get_urgent_requests(self, min_urgency: int = 7) -> list[PartyRequest]:
        """Get urgent party requests."""
        with self._lock:
            now = time.time()
            return [r for r in self._requests if r.expires_at > now and r.urgency >= min_urgency]

    def should_save_cooldowns(self, boss_hp_pct: float = 1.0) -> bool:
        """Check if we should save cooldowns (boss phasing soon)."""
        with self._lock:
            return boss_hp_pct < 0.3  # Save skills for final phase

    def should_prepare_for_phase(self, boss_hp_pct: float = 1.0) -> bool:
        """Check if we should prepare for a boss phase change."""
        with self._lock:
            return 0.45 < boss_hp_pct < 0.55  # Phase change at 50%

    def set_role(self, role: str) -> None:
        with self._lock:
            self._my_role = role

    def get_role(self) -> str:
        with self._lock:
            return self._my_role

    def set_party_leader(self, is_leader: bool) -> None:
        with self._lock:
            self._party_leader = is_leader

    def is_party_leader(self) -> bool:
        with self._lock:
            return self._party_leader

    def set_enqueue_fn(self, fn: Callable) -> None:
        with self._lock:
            self._enqueue_fn = fn

    def get_protocol_summary(self) -> str:
        with self._lock:
            lines = [f"── Party Skill Protocol ──"]
            lines.append(f"Role: {self._my_role}")
            lines.append(f"Party leader: {self._party_leader}")
            lines.append(f"Active announcements: {len(self.get_active_announcements())}")
            lines.append(f"Pending requests: {len(self.get_pending_requests())}")
            urgent = self.get_urgent_requests()
            if urgent:
                lines.append(f"Urgent: {', '.join(f'{r.request_type}({r.urgency})' for r in urgent[:3])}")
            return "\n".join(lines)

    def reset(self) -> None:
        with self._lock:
            self._announcements.clear()
            self._requests.clear()


# ── Global Singleton ──

_party_protocol: PartySkillProtocol | None = None
_party_protocol_lock = RLock()


def get_party_skill_protocol() -> PartySkillProtocol:
    global _party_protocol
    with _party_protocol_lock:
        if _party_protocol is None:
            _party_protocol = PartySkillProtocol()
        return _party_protocol
