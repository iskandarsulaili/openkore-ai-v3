"""
Anti-Detection System — gives each bot a unique "human-like" behavior profile.

This module generates per-bot behavior profiles that make each bot look like
a different human player. Profiles affect: command pacing, reaction time,
movement patterns, heal timing, and sit duration.

The system is config-driven via the sidecar's config push mechanism.
No hardcoded values — all parameters are randomized per bot at profile init.

Design principles:
1. Each bot gets a UNIQUE profile (seeded from bot name)
2. Profiles are STABLE across restarts (same bot = same profile)
3. All timings include random jitter within human-normal ranges
4. No two bots ever have identical behavior patterns
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

# Human reaction time ranges (ms)
# Source: RO player behavior studies, general HCI research
_HUMAN_REACTION_MIN = 150
_HUMAN_REACTION_MAX = 500

# Command pacing ranges (ms) — how fast a human issues commands
# Fast player: 200-400ms between actions
# Average player: 300-600ms
# Slow/cautious player: 500-1000ms
_CMD_PACING_RANGES = [
    (200, 400),   # Fast/aggressive
    (300, 600),   # Average/balanced
    (400, 800),   # Cautious
    (500, 1000),  # Slow/lazy
    (250, 500),   # Twitchy/inconsistent
]

# Heal reaction delay (ms) — delay before using a potion after HP drops
# Humans don't heal instantly — they notice, then act
_HEAL_REACTION_RANGES = [
    (100, 300),   # Quick healer (pro)
    (200, 500),   # Average healer
    (300, 800),   # Slow healer
    (150, 400),   # Inconsistent healer
]

# Sit duration ranges (seconds) — how long a human sits to regen
# Humans don't sit for exactly the same time every time
_SIT_DURATION_RANGES = [
    (3, 8),    # Quick rest
    (5, 12),   # Normal rest
    (8, 20),   # Long rest (AFK-ish)
    (4, 10),   # Inconsistent
]

# Movement variation levels
# 0 = straight line (bot-like, avoid)
# 1 = slight variation
# 2 = human-like erratic
_MOVEMENT_VARIATIONS = [1, 2]

# Profile personality names
_PROFILE_NAMES = [
    "aggressive", "cautious", "balanced",
    "lazy", "twitchy", "methodical",
    "reactive", "patient", "impulsive",
    "steady",
]


@dataclass
class BehaviorProfile:
    """Per-bot behavior profile for anti-detection."""
    
    # Bot identifier
    bot_id: str
    
    # Command pacing (ms)
    cmd_min_delay_ms: int
    cmd_max_delay_ms: int
    
    # Reaction time (ms)
    reaction_time_ms: int
    
    # Heal reaction delay (ms)
    heal_reaction_ms: int
    
    # Sit duration range (seconds)
    sit_min_seconds: int
    sit_max_seconds: int
    
    # Movement variation (0=straight, 1=slight, 2=erratic)
    movement_variation: int
    
    # Attack delay (ms)
    attack_delay_ms: int
    
    # Profile personality name
    profile_name: str
    
    # When profile was created
    created_at: float = field(default_factory=time.time)
    
    def get_cmd_delay_ms(self) -> int:
        """Get a random command delay within this bot's pacing range."""
        return self.cmd_min_delay_ms + random.randint(
            0, self.cmd_max_delay_ms - self.cmd_min_delay_ms
        )
    
    def get_heal_delay_ms(self) -> int:
        """Get a random heal reaction delay."""
        return self.heal_reaction_ms + random.randint(0, 200)
    
    def get_sit_duration(self) -> int:
        """Get a random sit duration in seconds."""
        return self.sit_min_seconds + random.randint(
            0, self.sit_max_seconds - self.sit_min_seconds
        )
    
    def get_attack_delay_ms(self) -> int:
        """Get a random attack delay."""
        return self.attack_delay_ms + random.randint(0, 150)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for config push."""
        return {
            "bot_id": self.bot_id,
            "cmd_min_delay_ms": self.cmd_min_delay_ms,
            "cmd_max_delay_ms": self.cmd_max_delay_ms,
            "reaction_time_ms": self.reaction_time_ms,
            "heal_reaction_ms": self.heal_reaction_ms,
            "sit_min_seconds": self.sit_min_seconds,
            "sit_max_seconds": self.sit_max_seconds,
            "movement_variation": self.movement_variation,
            "attack_delay_ms": self.attack_delay_ms,
            "profile_name": self.profile_name,
        }


class AntiDetectionSystem:
    """Manages per-bot behavior profiles for anti-detection."""
    
    def __init__(self) -> None:
        self._lock = RLock()
        self._profiles: dict[str, BehaviorProfile] = {}
        self._last_push: dict[str, str] = {}
        self._last_push_time: float = 0.0
    
    def get_or_create_profile(self, bot_id: str) -> BehaviorProfile:
        """Get existing profile or create a new one for this bot.
        
        Profiles are deterministic per bot name — same bot always gets
        the same profile, even across restarts.
        """
        with self._lock:
            if bot_id in self._profiles:
                return self._profiles[bot_id]
            
            # Create deterministic seed from bot name
            seed_bytes = bot_id.encode("utf-8")
            seed = int(hashlib.sha256(seed_bytes).hexdigest()[:8], 16)
            rng = random.Random(seed)
            
            # Select pacing range
            pacing_idx = rng.randint(0, len(_CMD_PACING_RANGES) - 1)
            cmd_min, cmd_max = _CMD_PACING_RANGES[pacing_idx]
            
            # Select heal reaction range
            heal_idx = rng.randint(0, len(_HEAL_REACTION_RANGES) - 1)
            heal_min, heal_max = _HEAL_REACTION_RANGES[heal_idx]
            
            # Select sit duration range
            sit_idx = rng.randint(0, len(_SIT_DURATION_RANGES) - 1)
            sit_min, sit_max = _SIT_DURATION_RANGES[sit_idx]
            
            # Select movement variation
            movement = rng.choice(_MOVEMENT_VARIATIONS)
            
            # Select profile name
            name = rng.choice(_PROFILE_NAMES)
            
            profile = BehaviorProfile(
                bot_id=bot_id,
                cmd_min_delay_ms=cmd_min,
                cmd_max_delay_ms=cmd_max,
                reaction_time_ms=rng.randint(_HUMAN_REACTION_MIN, _HUMAN_REACTION_MAX),
                heal_reaction_ms=rng.randint(heal_min, heal_max),
                sit_min_seconds=sit_min,
                sit_max_seconds=sit_max,
                movement_variation=movement,
                attack_delay_ms=rng.randint(50, 250),
                profile_name=name,
            )
            
            self._profiles[bot_id] = profile
            logger.info(
                "anti_detection_profile: bot=%s profile=%s cmd=%d-%dms reaction=%dms "
                "heal=%dms sit=%d-%ds movement=%d",
                bot_id, name, cmd_min, cmd_max,
                profile.reaction_time_ms, profile.heal_reaction_ms,
                sit_min, sit_max, movement,
            )
            
            return profile
    
    def get_profile(self, bot_id: str) -> BehaviorProfile | None:
        """Get existing profile, or None if not yet created."""
        with self._lock:
            return self._profiles.get(bot_id)
    
    def get_config_push(self, bot_id: str) -> dict[str, str] | None:
        """Compute config push for a bot's anti-detection settings.
        
        Returns a dict of config key->value to push to the bridge, or None
        if nothing changed.
        """
        profile = self.get_or_create_profile(bot_id)
        
        push = {
            "aiSidecar_antiDetectionEnabled": "1",
            "aiSidecar_cmdMinDelayMs": str(profile.cmd_min_delay_ms),
            "aiSidecar_cmdMaxDelayMs": str(profile.cmd_max_delay_ms),
            "aiSidecar_healReactionMs": str(profile.heal_reaction_ms),
            "aiSidecar_sitMinSeconds": str(profile.sit_min_seconds),
            "aiSidecar_sitMaxSeconds": str(profile.sit_max_seconds),
            "aiSidecar_movementVariation": str(profile.movement_variation),
            "aiSidecar_attackDelayMs": str(profile.attack_delay_ms),
        }
        
        # Only push if changed
        key = f"{bot_id}:{push}"
        now = time.time()
        if key == self._last_push.get(bot_id) and now - self._last_push_time < 30:
            return None
        
        self._last_push[bot_id] = key
        self._last_push_time = now
        return push
    
    def get_all_profiles(self) -> dict[str, dict[str, Any]]:
        """Get all profiles as dicts (for telemetry)."""
        with self._lock:
            return {
                bid: p.to_dict()
                for bid, p in self._profiles.items()
            }


# Global singleton
_system: AntiDetectionSystem | None = None
_system_lock = RLock()


def get_anti_detection_system() -> AntiDetectionSystem:
    """Get the global anti-detection system singleton."""
    global _system
    with _system_lock:
        if _system is None:
            _system = AntiDetectionSystem()
        return _system
