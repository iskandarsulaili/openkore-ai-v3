"""Game Sense — map dead/hot detection, GM detection, abort awareness, party efficiency.

A pro RO player reads the game like a sixth sense:
- Knows when a map is dead (no kills) vs hot (good spawns)
- Spots GMs instantly and switches to human-like behavior
- Knows when party EXP share is actually beneficial vs solo
- Reacts to server events (MVP alarms, announcements)
- Adapts monster targeting based on monster-specific AI quirks
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────

# Kills per minute thresholds
DEAD_MAP_KPM = 5.0       # Below this for 10 min = dead map
HOT_MAP_KPM = 20.0        # Above this = hot map
DEAD_MAP_DURATION = 600   # 10 minutes in seconds
KPM_WINDOW = 60           # 1 minute rolling window

# GM name patterns (case-insensitive)
GM_NAME_PATTERNS: list[str] = [
    "gm-", "gm_", "game master", "game_master",
    "admin", "admin-", "admin_",
    "rookie", "helper", "support",
    "event", "event_", "event-",
]

# Server announcement patterns that trigger abort awareness
ABORT_ANNOUNCEMENTS: dict[str, str] = {
    "mvp": "MVP has been killed",
    "woe_start": "War of Emperium has started",
    "woe_end": "War of Emperium has ended",
    "maintenance": "Server will be down for maintenance",
    "event_start": "Event has started",
    "boss_spawn": "has appeared",
}

# EXP share efficiency table: party_size -> (exp_multiplier, share_per_member)
# RO formula: party of N gets (1.0 + 0.2 * (N-1)) multiplier, split N ways
PARTY_EXP_EFFICIENCY: dict[int, tuple[float, float]] = {
    1: (1.0, 1.0),       # Solo: 100% each
    2: (1.2, 0.6),       # 120% / 2 = 60% each
    3: (1.3, 0.433),     # 130% / 3 = 43% each
    4: (1.4, 0.35),      # 140% / 4 = 35% each
    5: (1.5, 0.30),      # 150% / 5 = 30% each
    6: (1.6, 0.267),     # 160% / 6 = 27% each
    7: (1.7, 0.243),     # 170% / 7 = 24% each
    8: (1.8, 0.225),     # 180% / 8 = 22.5% each
}

# Classes that benefit from party vs solo
SOLO_FRIENDLY_CLASSES: set[str] = {
    "wizard", "high_wizard", "warlock",
    "hunter", "sniper", "ranger",
    "assassin", "assassin_cross", "guillotine_cross",
    "monk", "champion", "sura",
}

PARTY_REQUIRED_CLASSES: set[str] = {
    "priest", "high_priest", "arch_bishop",
    "acolyte",
}


# ── Data models ───────────────────────────────────────────────────────────

@dataclass
class MapProductivity:
    """Tracks kill rate and productivity for a single map."""
    map_name: str = ""
    total_kills: int = 0
    kill_timestamps: list[float] = field(default_factory=list)
    last_check_time: float = 0.0
    is_dead: bool = False
    dead_since: float = 0.0
    is_hot: bool = False
    hot_since: float = 0.0
    player_count: int = 0
    competitor_count: int = 0  # Other players killing YOUR mobs
    last_switch_time: float = 0.0  # Cooldown to prevent map hopping

    @property
    def kills_per_minute(self) -> float:
        """Calculate kills per minute over the rolling window."""
        now = time.time()
        cutoff = now - KPM_WINDOW
        recent = [t for t in self.kill_timestamps if t > cutoff]
        if not recent:
            return 0.0
        # Prune old timestamps
        self.kill_timestamps = recent
        elapsed = min(KPM_WINDOW, now - min(recent)) if recent else 1
        return len(recent) / (elapsed / 60.0) if elapsed > 0 else 0.0

    def record_kill(self) -> None:
        """Record a kill at the current time."""
        now = time.time()
        self.kill_timestamps.append(now)
        self.total_kills += 1
        # Prune old entries to keep memory bounded
        cutoff = now - KPM_WINDOW * 2
        self.kill_timestamps = [t for t in self.kill_timestamps if t > cutoff]

    def assess_productivity(self) -> str:
        """Returns 'dead', 'normal', or 'hot' based on current KPM."""
        kpm = self.kills_per_minute
        now = time.time()

        if kpm < DEAD_MAP_KPM:
            if not self.is_dead:
                self.is_dead = True
                self.dead_since = now
                return "dead"
            # Already dead — check duration
            if now - self.dead_since > DEAD_MAP_DURATION:
                return "dead_long"
            return "dead"
        elif kpm > HOT_MAP_KPM:
            if not self.is_hot:
                self.is_hot = True
                self.hot_since = now
            return "hot"
        else:
            self.is_dead = False
            self.is_hot = False
            return "normal"


@dataclass
class GMDetectionState:
    """Tracks GM detection state."""
    gm_detected: bool = False
    gm_name: str = ""
    detected_at: float = 0.0
    last_scan_time: float = 0.0
    human_mode_active: bool = False
    human_mode_until: float = 0.0

    def is_in_human_mode(self) -> bool:
        """Check if we're still in human-like mode after GM detection."""
        if not self.human_mode_active:
            return False
        if time.time() > self.human_mode_until:
            self.human_mode_active = False
            return False
        return True

    def activate_human_mode(self, duration: float = 300.0) -> None:
        """Switch to human-like behavior for `duration` seconds."""
        self.human_mode_active = True
        self.human_mode_until = time.time() + duration
        logger.warning("[GM] Human mode activated for %.0f seconds", duration)


@dataclass
class AbortAwarenessState:
    """Tracks abort awareness triggers."""
    mvp_kill_alarm: bool = False
    mvp_name: str = ""
    mvp_kill_time: float = 0.0
    server_announcement: str = ""
    announcement_time: float = 0.0
    low_hp_evacuating: bool = False
    evacuate_until: float = 0.0

    def should_evacuate(self, hp_pct: float) -> bool:
        """Check if we should evacuate (MVP alarm + low HP)."""
        if self.mvp_kill_alarm and hp_pct < 0.50:
            return True
        if self.low_hp_evacuating and time.time() < self.evacuate_until:
            return True
        return False

    def trigger_evacuation(self, duration: float = 30.0) -> None:
        """Trigger evacuation mode."""
        self.low_hp_evacuating = True
        self.evacuate_until = time.time() + duration


# ── Game Sense Engine ─────────────────────────────────────────────────────

class GameSenseEngine:
    """God-tier game sense — map productivity, GM detection, abort awareness.

    Features:
      - Per-map kill rate tracking with rolling KPM window
      - Dead map detection (<5 KPM for 10 min → recommend move)
      - Hot map detection (>20 KPM → stay)
      - Competitor detection (other players killing your mobs)
      - GM name pattern detection with human-mode switch
      - MVP kill alarm → low HP evacuation
      - Server announcement parsing
      - Party efficiency calculator (solo vs party EXP analysis)
      - Monster-specific AI awareness (flees, calls friends, teleports, aggro night)
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._map_productivity: dict[str, MapProductivity] = {}
        self._gm_state: GMDetectionState = GMDetectionState()
        self._abort_state: AbortAwarenessState = AbortAwarenessState()
        self._last_announcement_check: float = 0.0
        self._last_map_switch: dict[str, float] = {}  # bot_id -> last switch time

    # ── Public API ────────────────────────────────────────────────────

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        bot_id: str,
    ) -> None:
        """Run game sense assessment — call every PDCA cycle."""
        if not signals:
            return

        with self._lock:
            map_name = str(signals.get("map", "") or "").lower().replace(".gat", "")
            if not map_name:
                return

            # 1. Map productivity tracking
            self._track_kills(signals, map_name)
            self._track_competitors(signals, map_name)

            # 2. GM detection
            self._detect_gm(signals, actions, bot_id)

            # 3. Abort awareness
            self._check_abort_triggers(signals, actions, bot_id)

            # 4. Party efficiency awareness
            self._assess_party_efficiency(signals, actions, bot_id)

            # 5. Map dead/hot recommendations
            self._recommend_map_action(signals, actions, bot_id, map_name)

    def get_map_productivity(self, map_name: str) -> MapProductivity | None:
        """Get productivity data for a specific map."""
        with self._lock:
            return self._map_productivity.get(map_name)

    def get_all_map_productivity(self) -> dict[str, MapProductivity]:
        """Get all tracked map productivity data."""
        with self._lock:
            return dict(self._map_productivity)

    def is_gm_detected(self) -> bool:
        """Check if a GM was recently detected."""
        with self._lock:
            return self._gm_state.is_in_human_mode()

    def get_gm_state(self) -> GMDetectionState:
        """Get current GM detection state."""
        with self._lock:
            return self._gm_state

    def get_abort_state(self) -> AbortAwarenessState:
        """Get current abort awareness state."""
        with self._lock:
            return self._abort_state

    def should_join_party(
        self,
        my_class: str,
        party_size: int,
        my_level: int,
        party_level_range: int = 10,
    ) -> tuple[bool, str]:
        """Determine if joining a party is beneficial.

        Returns (should_join, reason).

        RO EXP mechanics:
        - Party of 2: 120% EXP split 2 ways = 60% each vs 100% solo
        - Party of 3: 130% EXP split 3 ways = 43% each
        - Party of 4: 140% EXP split 4 ways = 35% each
        - Party of 5: 150% EXP split 5 ways = 30% each
        - Party of 6: 160% EXP split 6 ways = 27% each

        But with faster kills (shared DPS + buffs), effective EXP/hour can be higher.
        """
        my_class_lower = my_class.lower()

        # Priests should ALWAYS join parties (they can't solo efficiently)
        if my_class_lower in PARTY_REQUIRED_CLASSES:
            return (True, "Priest/acolyte — always party for efficiency")

        # Solo-friendly classes: check if party is worth it
        if my_class_lower in SOLO_FRIENDLY_CLASSES:
            if party_size <= 1:
                return (False, "Solo class — better alone")
            if party_size == 2:
                # 60% each — only worth it if party member doubles kill speed
                return (True, "Party of 2 — 60% each, worth it with synergy")
            if party_size >= 3:
                # 43% or less — usually not worth it for solo classes
                return (False, f"Party of {party_size} — {PARTY_EXP_EFFICIENCY.get(party_size, (0, 0))[1]:.0%} each, better solo")

        # Support/utility classes: party is usually better
        if party_size >= 2:
            return (True, f"Party of {party_size} — support class benefits from party")

        # Default: party is good
        if party_size >= 2:
            return (True, f"Party of {party_size} — shared EXP + faster kills")

        return (False, "No party or solo is better")

    # ── Internal: Map productivity ────────────────────────────────────

    def _track_kills(self, signals: dict[str, Any], map_name: str) -> None:
        """Track kills on the current map."""
        kills = signals.get("kills", 0) or 0
        if map_name not in self._map_productivity:
            self._map_productivity[map_name] = MapProductivity(map_name=map_name)

        mp = self._map_productivity[map_name]
        # Check if kills increased since last check
        if kills > mp.total_kills:
            diff = kills - mp.total_kills
            for _ in range(diff):
                mp.record_kill()
        mp.last_check_time = time.time()

    def _track_competitors(self, signals: dict[str, Any], map_name: str) -> None:
        """Track other players who might be competing for mobs."""
        players = signals.get("players", []) or []
        my_name = str(signals.get("name", "") or "")

        if map_name in self._map_productivity:
            mp = self._map_productivity[map_name]
            # Count non-party, non-self players on the map
            party_members = set()
            party = signals.get("party", {}) or {}
            if isinstance(party, dict):
                for m in party.get("members", []):
                    if isinstance(m, dict):
                        party_members.add(m.get("name", ""))
                    elif isinstance(m, str):
                        party_members.add(m)

            competitors = 0
            for p in players:
                if isinstance(p, dict):
                    p_name = str(p.get("name", "") or "")
                    if p_name and p_name != my_name and p_name not in party_members:
                        competitors += 1
            mp.competitor_count = competitors

    def _recommend_map_action(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        bot_id: str,
        map_name: str,
    ) -> None:
        """Recommend map changes based on productivity."""
        mp = self._map_productivity.get(map_name)
        if not mp:
            return

        status = mp.assess_productivity()
        now = time.time()

        # Prevent map hopping — only recommend switch every 60s
        last_switch = self._last_map_switch.get(bot_id, 0)
        if now - last_switch < 60:
            return

        if status == "dead_long":
            # Map has been dead for 10+ minutes — recommend moving
            actions.append(HeuristicAction(
                kind="command",
                command="map_change dead",
                confidence=0.85,
                domain="hunting",
                reason=f"Map {map_name} dead for 10+ min (KPM={mp.kills_per_minute:.1f}) — moving to better spot",
                metadata={"map": map_name, "kpm": mp.kills_per_minute, "action": "move"},
            ))
            self._last_map_switch[bot_id] = now

        elif mp.competitor_count >= 3:
            # Too many competitors — switch channels or move
            actions.append(HeuristicAction(
                kind="command",
                command="map_change competitors",
                confidence=0.80,
                domain="hunting",
                reason=f"{mp.competitor_count} competitors on {map_name} — switching channels",
                metadata={"map": map_name, "competitors": mp.competitor_count, "action": "channel_switch"},
            ))
            self._last_map_switch[bot_id] = now

        elif status == "hot":
            # Hot map — stay and enjoy
            actions.append(HeuristicAction(
                kind="log",
                command=f"map_hot {map_name}",
                confidence=0.95,
                domain="hunting",
                reason=f"Map {map_name} is HOT (KPM={mp.kills_per_minute:.1f}) — staying",
                metadata={"map": map_name, "kpm": mp.kills_per_minute, "action": "stay"},
            ))

    # ── Internal: GM detection ─────────────────────────────────────────

    def _detect_gm(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        bot_id: str,
    ) -> None:
        """Detect Game Masters on the map and switch to human mode."""
        players = signals.get("players", []) or []
        now = time.time()

        # Rate-limit GM scans to every 5 seconds
        if now - self._gm_state.last_scan_time < 5:
            return
        self._gm_state.last_scan_time = now

        for p in players:
            if not isinstance(p, dict):
                continue
            p_name = str(p.get("name", "") or "").lower()
            p_job = str(p.get("job", "") or "").lower()

            # Check GM name patterns
            is_gm = False
            for pattern in GM_NAME_PATTERNS:
                if pattern in p_name:
                    is_gm = True
                    break

            # Check for GM-specific job names
            if p_job in ("game_master", "gm", "admin"):
                is_gm = True

            if is_gm:
                self._gm_state.gm_detected = True
                self._gm_state.gm_name = p_name
                self._gm_state.detected_at = now
                self._gm_state.activate_human_mode(duration=300.0)

                actions.append(HeuristicAction(
                    kind="command",
                    command="gm_detected",
                    confidence=0.99,
                    domain="survival",
                    reason=f"GM detected: {p_name} — switching to human-like mode",
                    metadata={"gm_name": p_name, "action": "human_mode"},
                ))
                logger.warning("[GM DETECTED] %s on map — activating human mode", p_name)
                return

        # If no GM detected and human mode expired, log it
        if self._gm_state.gm_detected and not self._gm_state.is_in_human_mode():
            self._gm_state.gm_detected = False
            logger.info("[GM] All clear — resuming normal operation")

    # ── Internal: Abort awareness ─────────────────────────────────────

    def _check_abort_triggers(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        bot_id: str,
    ) -> None:
        """Check for abort triggers (MVP kill alarm, server announcements)."""
        now = time.time()
        hp_pct = signals.get("hp_ratio", 1.0) or 1.0

        # Check server announcements
        announcements = signals.get("announcements", signals.get("server_messages", [])) or []
        if isinstance(announcements, list):
            for msg in announcements:
                msg_lower = str(msg).lower() if isinstance(msg, str) else ""
                for trigger_type, keyword in ABORT_ANNOUNCEMENTS.items():
                    if keyword.lower() in msg_lower:
                        self._abort_state.server_announcement = str(msg)
                        self._abort_state.announcement_time = now

                        if trigger_type == "mvp":
                            self._abort_state.mvp_kill_alarm = True
                            # Extract MVP name if possible
                            parts = str(msg).split()
                            for i, part in enumerate(parts):
                                if part.lower() == "mvp" and i + 1 < len(parts):
                                    self._abort_state.mvp_name = parts[i + 1]
                                    break

                            if hp_pct < 0.50:
                                self._abort_state.trigger_evacuation(30.0)
                                actions.append(HeuristicAction(
                                    kind="command",
                                    command="evacuate",
                                    confidence=0.99,
                                    domain="survival",
                                    reason=f"MVP {self._abort_state.mvp_name} killed — low HP ({hp_pct:.0%}), evacuating!",
                                    metadata={"trigger": "mvp_kill", "hp_pct": hp_pct},
                                ))
                            break

        # Check if we should evacuate
        if self._abort_state.should_evacuate(hp_pct):
            actions.append(HeuristicAction(
                kind="command",
                command="use Butterfly Wing",  # RULE.md: by name (universal RO item)
                confidence=0.99,
                domain="survival",
                reason=f"Abort triggered — MVP alarm + low HP ({hp_pct:.0%}), using Butterfly Wing",
                metadata={"trigger": "abort_evacuation", "hp_pct": hp_pct},
            ))

    # ── Internal: Party efficiency ─────────────────────────────────────

    def _assess_party_efficiency(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        bot_id: str,
    ) -> None:
        """Assess whether current party setup is efficient."""
        in_party = signals.get("in_party", False)
        if not in_party:
            return

        party = signals.get("party", {}) or {}
        party_members = party.get("members", []) if isinstance(party, dict) else []
        party_size = len(party_members) + 1  # +1 for self

        my_class = str(signals.get("job_name", "novice") or "novice")
        my_level = signals.get("base_level", 1) or 1

        should_join, reason = self.should_join_party(my_class, party_size, my_level)

        if not should_join and party_size > 1:
            actions.append(HeuristicAction(
                kind="log",
                command="party_efficiency_warning",
                confidence=0.70,
                domain="social",
                reason=f"Party of {party_size} — {reason}",
                metadata={"party_size": party_size, "my_class": my_class, "efficiency": reason},
            ))

        # Log EXP efficiency
        eff = PARTY_EXP_EFFICIENCY.get(party_size)
        if eff:
            mult, share = eff
            actions.append(HeuristicAction(
                kind="log",
                command="party_exp_info",
                confidence=0.60,
                domain="social",
                reason=f"Party of {party_size}: {mult:.1f}x total, {share:.1%} each",
                metadata={"party_size": party_size, "multiplier": mult, "share": share},
            ))


# ── Singleton factory ─────────────────────────────────────────────────────

_game_sense_engine: GameSenseEngine | None = None


def get_game_sense_engine() -> GameSenseEngine:
    """Get or create the singleton GameSenseEngine."""
    global _game_sense_engine
    if _game_sense_engine is None:
        _game_sense_engine = GameSenseEngine()
    return _game_sense_engine
