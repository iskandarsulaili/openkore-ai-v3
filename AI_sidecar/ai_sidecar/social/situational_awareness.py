"""
Situational awareness — reads the server's mood and adjusts behavior.

A smart player knows when the server is tense after a ban wave, when
the economy is crashing, when to lay low. This module monitors server
signals and adjusts bot behavior dynamically.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ServerSignal:
    """A signal about server mood or events."""
    signal_type: str  # ban_wave, gm_active, economy_crash, player_exodus, new_patch, event_active
    severity: int  # 1-10
    description: str
    timestamp: float = 0.0
    expires_at: float = 0.0


@dataclass(slots=True)
class SituationalAwareness:
    """Reads server mood and adjusts behavior."""
    
    _lock: RLock = field(default_factory=RLock)
    _signals: deque = field(default_factory=lambda: deque(maxlen=50))
    _mood: str = "normal"  # normal, tense, dangerous, relaxed, chaotic
    _mood_confidence: float = 0.0
    _stats: dict[str, int] = field(default_factory=lambda: {"signals": 0, "mood_changes": 0})
    
    def record_signal(self, signal_type: str, severity: int, description: str, duration_minutes: int = 60) -> None:
        """Record a server signal."""
        now = time.time()
        signal = ServerSignal(
            signal_type=signal_type,
            severity=severity,
            description=description,
            timestamp=now,
            expires_at=now + duration_minutes * 60,
        )
        with self._lock:
            self._signals.append(signal)
            self._stats["signals"] += 1
            self._recalculate_mood()
        logger.info("situational_signal: %s (severity=%d) — %s", signal_type, severity, description)
    
    def _recalculate_mood(self) -> None:
        """Recalculate server mood based on active signals."""
        now = time.time()
        active = [s for s in self._signals if s.expires_at > now]
        
        if not active:
            self._mood = "normal"
            self._mood_confidence = 1.0
            return
        
        # Calculate weighted severity
        total_severity = sum(s.severity for s in active)
        avg_severity = total_severity / len(active)
        
        # Check for critical signals
        has_ban_wave = any(s.signal_type == "ban_wave" for s in active)
        has_gm_active = any(s.signal_type == "gm_active" for s in active)
        has_economy_crash = any(s.signal_type == "economy_crash" for s in active)
        has_new_patch = any(s.signal_type == "new_patch" for s in active)
        
        old_mood = self._mood
        
        if has_ban_wave:
            self._mood = "dangerous"
        elif has_gm_active and avg_severity > 5:
            self._mood = "tense"
        elif has_economy_crash:
            self._mood = "chaotic"
        elif has_new_patch:
            self._mood = "relaxed"  # GMs busy with patch, safe to farm
        elif avg_severity < 3:
            self._mood = "relaxed"
        else:
            self._mood = "normal"
        
        self._mood_confidence = min(1.0, len(active) * 0.2)
        
        if old_mood != self._mood:
            self._stats["mood_changes"] += 1
            logger.info("situational_mood_changed: %s → %s", old_mood, self._mood)
    
    def get_mood(self) -> str:
        with self._lock:
            return self._mood
    
    def should_lay_low(self) -> bool:
        """Should the bot reduce activity due to server mood?"""
        with self._lock:
            return self._mood in ("tense", "dangerous")
    
    def get_risk_multiplier(self) -> float:
        """Get risk multiplier based on mood. Higher = more dangerous."""
        multipliers = {"relaxed": 0.5, "normal": 1.0, "tense": 1.5, "dangerous": 3.0, "chaotic": 2.0}
        with self._lock:
            return multipliers.get(self._mood, 1.0)
    
    def get_situational_context(self) -> str:
        """Get formatted context for LLM prompts."""
        with self._lock:
            now = time.time()
            active = [s for s in self._signals if s.expires_at > now]
            
            lines = ["── Server Mood ──"]
            lines.append(f"  Mood: {self._mood} (confidence: {self._mood_confidence:.0%})")
            lines.append(f"  Lay low: {'YES' if self.should_lay_low() else 'no'}")
            lines.append(f"  Risk multiplier: {self.get_risk_multiplier():.1f}x")
            
            if active:
                lines.append(f"  Active signals ({len(active)}):")
                for s in sorted(active, key=lambda x: -x.severity)[:5]:
                    remaining = max(0, (s.expires_at - now) / 60)
                    lines.append(f"    [{s.severity}] {s.signal_type}: {s.description} ({remaining:.0f}min remaining)")
            
            return "\n".join(lines)
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# Global instance
_situational: SituationalAwareness | None = None
_situational_lock = RLock()


def get_situational() -> SituationalAwareness:
    global _situational
    with _situational_lock:
        if _situational is None:
            _situational = SituationalAwareness()
        return _situational
