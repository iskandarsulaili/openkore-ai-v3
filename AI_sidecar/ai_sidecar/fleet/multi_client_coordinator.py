"""
Multi-Client Combat Protocol — real-time UDP multicast between bot instances
for coordinated combat. Each bot broadcasts its state every 200ms and listens
for others, adjusting target selection, skill timing, and positioning.
"""

from __future__ import annotations

import json
import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)

MULTICAST_GROUP = "239.255.0.1"
MULTICAST_PORT = 18091
BROADCAST_INTERVAL = 0.2  # 200ms


@dataclass
class BotCombatState:
    """Combat state broadcast by a bot."""
    bot_id: str
    x: int = 0
    y: int = 0
    hp_pct: float = 1.0
    sp_pct: float = 1.0
    target_id: int = 0
    target_hp_pct: float = 1.0
    aggro_count: int = 0
    is_casting: bool = False
    casting_skill: str = ""
    role: str = "dps"  # tank, healer, dps, support
    current_skill: str = ""
    skill_cooldowns: dict[str, int] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class CombatDirective:
    """A directive from the multi-client coordinator."""
    directive_type: str  # attack_target, hold_position, retreat, heal_target, use_skill
    target_id: int = 0
    target_x: int = 0
    target_y: int = 0
    skill_name: str = ""
    urgency: int = 5  # 1-10
    issued_by: str = ""


class MultiClientCombatCoordinator:
    """Coordinates combat across multiple bot instances via UDP multicast."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._my_state: BotCombatState = BotCombatState(bot_id="unknown")
        self._peer_states: dict[str, BotCombatState] = {}
        self._directives: list[CombatDirective] = []
        self._running: bool = False
        self._broadcast_thread: threading.Thread | None = None
        self._listen_thread: threading.Thread | None = None
        self._sock: socket.socket | None = None
        self._enqueue_fn: Callable | None = None
        self._my_role: str = "dps"
        self._party_target: int = 0
        self._last_broadcast: float = 0.0

    # ── Public API ──

    def start(self, bot_id: str, role: str = "dps") -> None:
        """Start the multicast coordinator."""
        with self._lock:
            if self._running:
                return
            self._my_state.bot_id = bot_id
            self._my_state.role = role
            self._my_role = role
            self._running = True

            # Set up UDP multicast socket
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)

            # Bind and join multicast group
            try:
                self._sock.bind(("", MULTICAST_PORT))
                mreq = struct.pack("4sl", socket.inet_aton(MULTICAST_GROUP), socket.INADDR_ANY)
                self._sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
                self._sock.settimeout(1.0)
            except OSError as e:
                logger.warning("multicast_bind_failed: %s (will use broadcast-only mode)", e)

            # Start threads
            self._broadcast_thread = threading.Thread(target=self._broadcast_loop, daemon=True)
            self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
            self._broadcast_thread.start()
            self._listen_thread.start()
            logger.info("multi_client_coordinator_started: %s role=%s", bot_id, role)

    def stop(self) -> None:
        with self._lock:
            self._running = False
            if self._sock:
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None
            logger.info("multi_client_coordinator_stopped")

    def update_state(self, x: int = 0, y: int = 0, hp_pct: float = 1.0, sp_pct: float = 1.0,
                     target_id: int = 0, target_hp_pct: float = 1.0, aggro_count: int = 0,
                     is_casting: bool = False, casting_skill: str = "",
                     current_skill: str = "", skill_cooldowns: dict[str, int] | None = None) -> None:
        """Update this bot's combat state for broadcast."""
        with self._lock:
            self._my_state.x = x
            self._my_state.y = y
            self._my_state.hp_pct = hp_pct
            self._my_state.sp_pct = sp_pct
            self._my_state.target_id = target_id
            self._my_state.target_hp_pct = target_hp_pct
            self._my_state.aggro_count = aggro_count
            self._my_state.is_casting = is_casting
            self._my_state.casting_skill = casting_skill
            self._my_state.current_skill = current_skill
            self._my_state.skill_cooldowns = skill_cooldowns or {}
            self._my_state.timestamp = time.time()

    def get_peer_states(self) -> dict[str, BotCombatState]:
        """Get combat states of all peer bots."""
        with self._lock:
            now = time.time()
            # Filter stale states (> 2s old)
            return {bid: s for bid, s in self._peer_states.items() if now - s.timestamp < 2.0}

    def get_coordinated_target(self) -> int:
        """Get the target the group should focus on."""
        with self._lock:
            peers = self.get_peer_states()
            if not peers:
                return self._my_state.target_id

            # Count targets being attacked
            target_counts: dict[int, int] = {}
            for state in peers.values():
                if state.target_id > 0:
                    target_counts[state.target_id] = target_counts.get(state.target_id, 0) + 1

            # If multiple bots are attacking the same target, focus it
            for tid, count in sorted(target_counts.items(), key=lambda x: -x[1]):
                if count >= 2:
                    return tid

            return self._my_state.target_id

    def should_tank(self) -> bool:
        """Check if this bot should be tanking based on role and HP."""
        with self._lock:
            if self._my_role != "tank":
                return False
            peers = self.get_peer_states()
            # If no other tank is alive, we must tank
            other_tanks = [s for s in peers.values() if s.role == "tank" and s.hp_pct > 0.3]
            return len(other_tanks) == 0

    def should_heal(self, target_bot: str = "") -> bool:
        """Check if a bot needs healing."""
        with self._lock:
            if self._my_role != "healer":
                return False
            peers = self.get_peer_states()
            for bid, state in peers.items():
                if target_bot and bid != target_bot:
                    continue
                if state.hp_pct < 0.5:
                    return True
            return False

    def get_heal_target(self) -> str | None:
        """Get the bot that most needs healing."""
        with self._lock:
            peers = self.get_peer_states()
            lowest_hp = 1.0
            target = None
            for bid, state in peers.items():
                if state.hp_pct < lowest_hp:
                    lowest_hp = state.hp_pct
                    target = bid
            return target if lowest_hp < 0.6 else None

    def get_coordination_summary(self) -> str:
        with self._lock:
            lines = [f"── Multi-Client Combat ──"]
            lines.append(f"Role: {self._my_role}")
            lines.append(f"Peers: {len(self.get_peer_states())}")
            for bid, state in self.get_peer_states().items():
                lines.append(f"  {bid}: {state.role} HP={state.hp_pct:.0%} target={state.target_id}")
            return "\n".join(lines)

    def set_enqueue_fn(self, fn: Callable) -> None:
        with self._lock:
            self._enqueue_fn = fn

    # ── Internal ──

    def _broadcast_loop(self) -> None:
        """Broadcast this bot's state periodically."""
        while self._running:
            try:
                with self._lock:
                    data = json.dumps(self._my_state.__dict__, default=str).encode("utf-8")
                    if self._sock:
                        self._sock.sendto(data, (MULTICAST_GROUP, MULTICAST_PORT))
                time.sleep(BROADCAST_INTERVAL)
            except Exception as e:
                logger.warning("multicast_broadcast_error: %s", e)
                time.sleep(1.0)

    def _listen_loop(self) -> None:
        """Listen for peer bot broadcasts."""
        while self._running:
            try:
                if not self._sock:
                    time.sleep(1.0)
                    continue
                data, addr = self._sock.recvfrom(4096)
                state_dict = json.loads(data.decode("utf-8"))
                state = BotCombatState(**state_dict)
                if state.bot_id != self._my_state.bot_id:
                    with self._lock:
                        self._peer_states[state.bot_id] = state
            except socket.timeout:
                continue
            except Exception as e:
                logger.warning("multicast_listen_error: %s", e)
                time.sleep(0.5)

    def reset(self) -> None:
        self.stop()
        with self._lock:
            self._peer_states.clear()
            self._directives.clear()


# ── Global Singleton ──

_mc_coordinator: MultiClientCombatCoordinator | None = None
_mc_coordinator_lock = RLock()


def get_multi_client_coordinator() -> MultiClientCombatCoordinator:
    global _mc_coordinator
    with _mc_coordinator_lock:
        if _mc_coordinator is None:
            _mc_coordinator = MultiClientCombatCoordinator()
        return _mc_coordinator
