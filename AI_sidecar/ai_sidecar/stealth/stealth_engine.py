"""
Stealth Engine — Anti-detection system for the bot fleet.

A pro player avoids detection by:
- Varying behavior patterns (not perfectly consistent)
- Managing session times (log in/out at human-like intervals)
- Rotating farming maps (don't farm same map for 24 hours)
- Diversifying activities (mix farming, shopping, socializing, resting)
- Detecting GMs and acting human or logging out
- Avoiding reports (don't KS, don't bot in popular spots)
- Following the social contract

This engine wires into:
- behavior_engine.py (existing human-like imperfections)
- anti_detection/ (existing anti-detection modules)
- player_profiler.py (GM detection)
- social_engine.py (social contract awareness)
"""

from __future__ import annotations

import logging
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────────────────────


@dataclass
class SessionProfile:
    """Profile for a single game session."""
    bot_id: str
    start_time: float
    current_duration_minutes: float = 0.0
    maps_visited: list[str] = field(default_factory=list)
    activities: list[str] = field(default_factory=list)
    chat_messages_sent: int = 0
    deaths: int = 0
    kills: int = 0
    is_suspicious: bool = False
    suspicion_reason: str = ""


@dataclass
class MapRotationEntry:
    """Track how long a bot has been on a map."""
    map_name: str
    bot_id: str
    entered_at: float
    duration_minutes: float = 0.0
    kills_on_map: int = 0
    is_overstaying: bool = False


@dataclass
class GMAlert:
    """Alert about a potential GM sighting."""
    player_name: str
    map_name: str
    first_seen: float
    last_seen: float
    confidence: float  # 0.0-1.0
    action_taken: str = ""  # none, logged_out, acted_human, changed_map


@dataclass(slots=True)
class StealthEngine:
    """
    Complete anti-detection system.

    Manages session profiles, map rotation, activity diversity,
    GM detection, and report avoidance.
    """

    _lock: RLock = field(default_factory=RLock)
    _sessions: dict[str, SessionProfile] = field(default_factory=dict)
    _map_rotation: dict[str, MapRotationEntry] = field(default_factory=dict)
    _gm_alerts: deque = field(default_factory=lambda: deque(maxlen=50))
    _activity_log: deque = field(default_factory=lambda: deque(maxlen=200))
    _map_history: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    _stats: dict[str, int] = field(default_factory=lambda: {
        "sessions_tracked": 0, "gm_alerts": 0, "map_rotations": 0,
        "activity_switches": 0, "suspicious_events": 0,
        "logouts_triggered": 0,
    })
    _behavior_engine: Any = None  # BehaviorEngine instance
    _player_profiler: Any = None  # PlayerProfiler instance
    _social_engine: Any = None  # SocialEngine instance
    _last_cleanup: float = 0.0

    # ── Configuration ──

    MAX_SESSION_MINUTES: int = 240  # 4 hours max session
    MIN_SESSION_MINUTES: int = 30  # 30 min minimum session
    SESSION_VARIANCE: float = 0.3  # 30% variance in session length
    MAX_MAP_STAY_MINUTES: int = 120  # 2 hours max on same map
    MAP_ROTATION_COOLDOWN: float = 14400  # 4 hours before returning to a map
    MIN_ACTIVITY_DIVERSITY: int = 3  # At least 3 different activities per session
    ACTIVITY_SWITCH_INTERVAL: tuple = (15, 45)  # Switch activities every 15-45 min
    SUSPICIOUS_MAP_THRESHOLD: int = 5  # Players on same map = suspicious
    GM_LOGOUT_CONFIDENCE: float = 0.7  # Log out if GM confidence > this
    REPORT_AVOIDANCE_MAPS: list[str] = field(default_factory=lambda: [
        "prt_fild01", "pay_fild01", "moc_fild01", "gef_fild01",
    ])

    # ── Public API ──

    def set_behavior_engine(self, engine: Any) -> None:
        """Wire BehaviorEngine instance."""
        self._behavior_engine = engine

    def set_player_profiler(self, profiler: Any) -> None:
        """Wire PlayerProfiler instance."""
        self._player_profiler = profiler

    def set_social_engine(self, engine: Any) -> None:
        """Wire SocialEngine instance."""
        self._social_engine = engine

    # ── Session Management ──

    def start_session(self, bot_id: str) -> SessionProfile:
        """Start tracking a new game session."""
        with self._lock:
            session = SessionProfile(
                bot_id=bot_id,
                start_time=time.time(),
            )
            self._sessions[bot_id] = session
            self._stats["sessions_tracked"] += 1
            logger.info("session_started: bot=%s", bot_id)
            return session

    def end_session(self, bot_id: str) -> dict[str, Any]:
        """End a game session and return summary."""
        with self._lock:
            session = self._sessions.pop(bot_id, None)
            if session is None:
                return {"error": "no_active_session"}

            duration = (time.time() - session.start_time) / 60
            summary = {
                "bot_id": bot_id,
                "duration_minutes": duration,
                "maps_visited": len(session.maps_visited),
                "activities": len(session.activities),
                "chat_messages": session.chat_messages_sent,
                "deaths": session.deaths,
                "kills": session.kills,
                "is_suspicious": session.is_suspicious,
            }
            logger.info("session_ended: bot=%s duration=%.1fmin maps=%d activities=%d",
                       bot_id, duration, len(session.maps_visited), len(session.activities))
            return summary

    def get_session_duration(self, bot_id: str) -> float:
        """Get current session duration in minutes."""
        with self._lock:
            session = self._sessions.get(bot_id)
            if session is None:
                return 0.0
            return (time.time() - session.start_time) / 60

    def should_end_session(self, bot_id: str) -> bool:
        """Check if the current session should end."""
        with self._lock:
            session = self._sessions.get(bot_id)
            if session is None:
                return False

            duration = (time.time() - session.start_time) / 60

            # Hard max
            if duration > self.MAX_SESSION_MINUTES:
                logger.info("session_end_reason: max_duration_reached bot=%s duration=%.1fmin",
                           bot_id, duration)
                return True

            # Variable session length (human-like)
            target_duration = self.MIN_SESSION_MINUTES + random.uniform(
                0, self.MAX_SESSION_MINUTES - self.MIN_SESSION_MINUTES
            )
            target_duration *= (1 + random.uniform(-self.SESSION_VARIANCE, self.SESSION_VARIANCE))

            if duration > target_duration:
                # Random chance to end (don't always end at exact time)
                if random.random() < 0.3:
                    logger.info("session_end_reason: variable_target bot=%s duration=%.1fmin target=%.1fmin",
                               bot_id, duration, target_duration)
                    return True

            return False

    # ── Map Rotation ──

    def record_map_entry(self, bot_id: str, map_name: str) -> None:
        """Record that a bot entered a map."""
        with self._lock:
            key = f"{bot_id}:{map_name}"
            self._map_rotation[key] = MapRotationEntry(
                map_name=map_name,
                bot_id=bot_id,
                entered_at=time.time(),
            )

            # Update session
            session = self._sessions.get(bot_id)
            if session and map_name not in session.maps_visited:
                session.maps_visited.append(map_name)

            # Update map history
            if map_name not in self._map_history[bot_id]:
                self._map_history[bot_id][map_name] = time.time()

    def get_map_stay_duration(self, bot_id: str, map_name: str) -> float:
        """Get how long a bot has been on a map in minutes."""
        with self._lock:
            key = f"{bot_id}:{map_name}"
            entry = self._map_rotation.get(key)
            if entry is None:
                return 0.0
            return (time.time() - entry.entered_at) / 60

    def should_rotate_map(self, bot_id: str, map_name: str) -> bool:
        """Check if the bot should rotate to a different map."""
        with self._lock:
            duration = self.get_map_stay_duration(bot_id, map_name)

            # Hard max
            if duration > self.MAX_MAP_STAY_MINUTES:
                logger.info("map_rotate_reason: max_stay bot=%s map=%s duration=%.1fmin",
                           bot_id, map_name, duration)
                self._stats["map_rotations"] += 1
                return True

            # Variable rotation (human-like)
            target_duration = self.MAX_MAP_STAY_MINUTES * random.uniform(0.4, 0.8)
            if duration > target_duration and random.random() < 0.2:
                logger.info("map_rotate_reason: variable_target bot=%s map=%s duration=%.1fmin",
                           bot_id, map_name, duration)
                self._stats["map_rotations"] += 1
                return True

            return False

    def get_rotation_suggestion(self, bot_id: str, current_map: str,
                                 available_maps: list[str]) -> str | None:
        """Suggest a map to rotate to."""
        with self._lock:
            now = time.time()

            # Filter out maps visited recently
            candidates = []
            for m in available_maps:
                if m == current_map:
                    continue
                last_visit = self._map_history.get(bot_id, {}).get(m, 0)
                if now - last_visit > self.MAP_ROTATION_COOLDOWN:
                    candidates.append(m)

            if not candidates:
                return None

            # Pick a random candidate (human-like variety)
            return random.choice(candidates)

    # ── Activity Diversity ──

    def record_activity(self, bot_id: str, activity: str) -> None:
        """Record an activity for diversity tracking."""
        with self._lock:
            now = time.time()
            self._activity_log.append({
                "bot_id": bot_id,
                "activity": activity,
                "timestamp": now,
            })

            session = self._sessions.get(bot_id)
            if session and activity not in session.activities:
                session.activities.append(activity)

    def get_activity_diversity(self, bot_id: str) -> int:
        """Get the number of different activities in the current session."""
        with self._lock:
            session = self._sessions.get(bot_id)
            if session is None:
                return 0
            return len(session.activities)

    def should_switch_activity(self, bot_id: str, current_activity: str) -> bool:
        """Check if the bot should switch to a different activity."""
        with self._lock:
            # Get time since last activity switch
            recent = [a for a in self._activity_log
                     if a["bot_id"] == bot_id and a["activity"] == current_activity]
            if not recent:
                return False

            last_switch = recent[-1]["timestamp"]
            elapsed = (time.time() - last_switch) / 60

            # Check if we've been doing this too long
            min_interval, max_interval = self.ACTIVITY_SWITCH_INTERVAL
            if elapsed > max_interval:
                self._stats["activity_switches"] += 1
                return True

            if elapsed > min_interval and random.random() < 0.15:
                self._stats["activity_switches"] += 1
                return True

            return False

    def suggest_activity_switch(self, bot_id: str, current_activity: str) -> str | None:
        """Suggest a different activity."""
        session = self._sessions.get(bot_id)
        if session is None:
            return None

        # Ensure minimum diversity
        diversity = self.get_activity_diversity(bot_id)
        if diversity < self.MIN_ACTIVITY_DIVERSITY:
            # Try an activity not yet done this session
            all_activities = ["farm", "socialize", "rest", "shop", "craft", "explore"]
            for act in all_activities:
                if act not in session.activities and act != current_activity:
                    return act

        # Random switch
        alternatives = [a for a in ["farm", "socialize", "rest", "shop"]
                       if a != current_activity]
        return random.choice(alternatives) if alternatives else None

    # ── GM Detection ──

    def check_gm_presence(self, bot_id: str, map_name: str,
                           nearby_players: list[dict]) -> dict[str, Any]:
        """Check if GMs are nearby and determine response."""
        with self._lock:
            result = {
                "gm_detected": False,
                "confidence": 0.0,
                "suggested_action": "none",
                "gm_names": [],
            }

            for player in nearby_players:
                name = player.get("name", "")
                if not name:
                    continue

                # Check for GM-like names
                is_gm = False
                for keyword in ["gm", "game master", "admin", "staff",
                                "moderator", "helper", "support"]:
                    if keyword in name.lower():
                        is_gm = True
                        break

                if is_gm:
                    result["gm_detected"] = True
                    result["confidence"] = 0.8
                    result["gm_names"].append(name)

                    # Create GM alert
                    alert = GMAlert(
                        player_name=name,
                        map_name=map_name,
                        first_seen=time.time(),
                        last_seen=time.time(),
                        confidence=0.8,
                    )
                    self._gm_alerts.append(alert)
                    self._stats["gm_alerts"] += 1

                    # Determine action based on confidence
                    if result["confidence"] >= self.GM_LOGOUT_CONFIDENCE:
                        result["suggested_action"] = "logout"
                        self._stats["logouts_triggered"] += 1
                    else:
                        result["suggested_action"] = "act_human"

            return result

    # ── Report Avoidance ──

    def check_report_risk(self, map_name: str, nearby_players: list[dict],
                           current_activity: str) -> dict[str, Any]:
        """Check if current behavior risks getting reported."""
        with self._lock:
            result = {
                "high_risk": False,
                "risk_score": 0.0,
                "reasons": [],
                "suggested_action": "none",
            }

            # Check if on a high-risk map
            if map_name in self.REPORT_AVOIDANCE_MAPS:
                result["risk_score"] += 0.3
                result["reasons"].append("High-risk map for reports")

            # Check player density
            if len(nearby_players) > self.SUSPICIOUS_MAP_THRESHOLD:
                result["risk_score"] += 0.3
                result["reasons"].append(f"Too many players ({len(nearby_players)})")

            # Check for known bot reporters
            if self._social_engine is not None:
                try:
                    for player in nearby_players:
                        name = player.get("name", "")
                        if name:
                            rel = self._social_engine._relationships.get(name)
                            if rel and rel.is_bot_reporter:
                                result["risk_score"] += 0.5
                                result["reasons"].append(f"Known bot reporter nearby: {name}")
                except Exception:
                    pass

            # Check activity
            if current_activity in ("farming", "attacking") and len(nearby_players) > 2:
                result["risk_score"] += 0.2
                result["reasons"].append("Farming near other players")

            # Determine action
            if result["risk_score"] >= 0.7:
                result["high_risk"] = True
                result["suggested_action"] = "change_map"
                self._stats["suspicious_events"] += 1
            elif result["risk_score"] >= 0.4:
                result["suggested_action"] = "act_human"

            return result

    # ── Human-like Behavior ──

    def get_human_likeness_score(self, bot_id: str) -> float:
        """Calculate how human-like the bot's behavior is (0.0-1.0)."""
        with self._lock:
            session = self._sessions.get(bot_id)
            if session is None:
                return 0.5

            score = 0.5  # Start at neutral

            # Session duration (humans don't play 24/7)
            duration = (time.time() - session.start_time) / 60
            if duration < 30:
                score += 0.1  # Short sessions are human-like
            elif duration > 180:
                score -= 0.2  # Long sessions are suspicious

            # Activity diversity
            diversity = len(session.activities)
            if diversity >= 3:
                score += 0.2  # Varied activities = human-like
            elif diversity <= 1:
                score -= 0.2  # Single activity = bot-like

            # Map changes
            if len(session.maps_visited) >= 2:
                score += 0.1  # Map changes = human-like

            # Chat activity
            if 5 <= session.chat_messages_sent <= 50:
                score += 0.1  # Some chat = human-like
            elif session.chat_messages_sent > 100:
                score -= 0.1  # Too much chat = suspicious

            # Deaths
            if session.deaths > 5:
                score -= 0.1  # Too many deaths = bad play

            return max(0.0, min(1.0, score))

    # ── Context ──

    def get_stealth_context(self) -> str:
        """Get formatted stealth context for LLM prompts."""
        with self._lock:
            lines = ["── Stealth & Anti-Detection ──"]

            # Active sessions
            for bot_id, session in self._sessions.items():
                duration = (time.time() - session.start_time) / 60
                diversity = len(session.activities)
                human_score = self.get_human_likeness_score(bot_id)
                lines.append(f"  {bot_id}: {duration:.0f}min, "
                             f"{diversity} activities, "
                             f"human_score={human_score:.2f}")

            # Recent GM alerts
            recent_gm = [a for a in self._gm_alerts
                        if time.time() - a.last_seen < 3600]
            if recent_gm:
                lines.append(f"  Recent GM alerts: {len(recent_gm)}")
                for alert in recent_gm[-3:]:
                    lines.append(f"    {alert.player_name} on {alert.map_name} "
                                 f"(confidence: {alert.confidence:.0%})")

            # Map rotation status
            overstaying = [e for e in self._map_rotation.values()
                          if e.is_overstaying]
            if overstaying:
                lines.append(f"  Overstaying maps: {len(overstaying)}")

            return "\n".join(lines)

    # ── Cycle Tick ──

    def tick(self, bot_id: str, signals: dict[str, Any]) -> dict[str, Any]:
        """Called every PDCA cycle. Returns stealth recommendations."""
        now = time.time()
        result = {
            "session_ok": True,
            "map_ok": True,
            "activity_ok": True,
            "gm_risk": 0.0,
            "report_risk": 0.0,
            "human_likeness": 0.5,
            "suggested_action": "none",
        }

        # Ensure session exists
        if bot_id not in self._sessions:
            self.start_session(bot_id)

        # Check session
        if self.should_end_session(bot_id):
            result["session_ok"] = False
            result["suggested_action"] = "logout"

        # Check map rotation
        map_name = str(signals.get("map", "") or "")
        if map_name:
            if self.should_rotate_map(bot_id, map_name):
                result["map_ok"] = False
                if result["suggested_action"] == "none":
                    result["suggested_action"] = "change_map"

        # Check activity diversity
        current_activity = str(signals.get("current_activity", "farm") or "farm")
        if self.should_switch_activity(bot_id, current_activity):
            result["activity_ok"] = False
            if result["suggested_action"] == "none":
                result["suggested_action"] = "switch_activity"

        # Check GM presence
        nearby_players = signals.get("nearby_players", []) or []
        gm_check = self.check_gm_presence(bot_id, map_name, nearby_players)
        if gm_check["gm_detected"]:
            result["gm_risk"] = gm_check["confidence"]
            if gm_check["suggested_action"] != "none":
                result["suggested_action"] = gm_check["suggested_action"]

        # Check report risk
        report_check = self.check_report_risk(map_name, nearby_players, current_activity)
        result["report_risk"] = report_check["risk_score"]
        if report_check["high_risk"] and result["suggested_action"] == "none":
            result["suggested_action"] = report_check["suggested_action"]

        # Calculate human-likeness
        result["human_likeness"] = self.get_human_likeness_score(bot_id)

        # Cleanup
        if now - self._last_cleanup > 600:
            self._cleanup()
            self._last_cleanup = now

        return result

    def _cleanup(self) -> None:
        """Remove stale data."""
        with self._lock:
            now = time.time()
            # Remove stale sessions (>24h)
            stale = [k for k, v in self._sessions.items()
                    if now - v.start_time > 86400]
            for k in stale:
                del self._sessions[k]

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# ── Global Singleton ──

_stealth_engine: StealthEngine | None = None
_stealth_engine_lock = RLock()


def get_stealth_engine() -> StealthEngine:
    global _stealth_engine
    with _stealth_engine_lock:
        if _stealth_engine is None:
            _stealth_engine = StealthEngine()
        return _stealth_engine
