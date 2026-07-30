"""
Bridge Wiring — connects BehaviorEngine output to actual OpenKore command dispatch.
==============================================================================

The problem: the BehaviorEngine (behavior_engine.py) produces behavior modifier
dicts via get_behavior_modifier(), but these modifiers were never actually
*applied* to command execution. The bridge (aiSidecarBridge.pl) has its own
hardcoded ANTI_DETECTION_MIN_DELAY_MS/MAX_DELAY_MS and per-bot profiles, but
they're disconnected from the sidecar's BehaviorEngine.

This module closes the gap:
1. Polls the BehaviorEngine for the current behavior modifier
2. Translates modifier values into concrete command pacing parameters
3. Pushes these to the bridge via config push (aiSidecar_* config keys)
4. Provides a simple API for command dispatch to query "how long should I wait?"

Config push keys sent to the bridge:
  - aiSidecar_antiDetectionEnabled: "1"/"0"
  - aiSidecar_cmdMinDelayMs: minimum command delay (ms)
  - aiSidecar_cmdMaxDelayMs: maximum command delay (ms)
  - aiSidecar_healReactionMs: heal reaction delay (ms)
  - aiSidecar_reactionTimeMs: base reaction time (ms)
  - aiSidecar_fatigueMultiplier: current fatigue scaling factor
"""

from __future__ import annotations

import logging
import random
import time
from threading import RLock
from typing import Any

from ai_sidecar.anti_detection.behavior_engine import (
    BehaviorEngine,
    BehaviorProfileType,
    get_behavior_engine,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_DEFAULT_MIN_DELAY_MS = 150
_DEFAULT_MAX_DELAY_MS = 600
_PUSH_COOLDOWN_S = 5.0  # Don't push config more often than this


class BridgeWiring:
    """Connects the BehaviorEngine to the Perl bridge config system.

    Usage::

        wiring = BridgeWiring()
        delay_ms = wiring.get_command_delay("bot1", {"action_kind": "attack"})
        # Use delay_ms to sleep before sending command

        push = wiring.get_config_push("bot1")
        if push:
            # Send push config keys to the bridge
            bridge_client.send_config(push)
    """

    def __init__(
        self,
        behavior_engine: BehaviorEngine | None = None,
    ) -> None:
        self._lock = RLock()
        self._engine: BehaviorEngine = (
            behavior_engine or get_behavior_engine()
        )
        self._last_push_time: float = 0.0
        self._last_push_payload: dict[str, str] = {}

    # ── Public API ───────────────────────────────────────────────────────────

    def get_command_delay(
        self, bot_id: str, context: dict[str, Any] | None = None
    ) -> int:
        """Get the recommended delay (ms) before sending the next command.

        This is the **primary integration point** — command dispatch code
        calls this before each command to inject human-like timing jitter.

        Combines:
        - BehaviorEngine reaction time delay
        - Fatigue multiplier (longer sessions = slower)
        - Profile-based min/max jitter
        - Skill/macro timing jitter (±15% for skill casts)
        """
        modifier = self._engine.get_behavior_modifier(bot_id, context or {})
        base_delay = modifier.get("delay_ms", 0)

        ctx = context or {}
        action_kind = ctx.get("action_kind", "")

        # Base reaction delay range from engine
        min_ms = max(50, base_delay)
        max_ms = max(min_ms + 50, base_delay + 200)

        # Session fatigue scaling
        state = self._get_bot_state(bot_id)
        fatigue_mult = self._engine._get_fatigue_multiplier(state)
        max_ms = int(max_ms * fatigue_mult)

        # ±15% jitter for skill/macro timing (humans press buttons slightly
        # late or early — never with perfect precision)
        if action_kind in ("skill", "cast", "macro"):
            jitter = random.uniform(-0.15, 0.15)
            delay = base_delay * (1.0 + jitter)
            delay = max(min_ms, min(max_ms, delay))
            logger.debug(
                "bridge_wiring: skill_jitter bot=%s base=%dms jitter=%.1f%% final=%dms",
                bot_id, base_delay, jitter * 100, int(delay),
            )
            return int(delay)

        # Standard command delay
        delay = random.randint(min_ms, max_ms)
        return delay

    def get_heal_delay(self, bot_id: str) -> int:
        """Get the heal reaction delay — delay before using a potion after HP drops.

        Humans don't heal instantly — they notice HP drop, then act.
        """
        modifier = self._engine.get_behavior_modifier(bot_id)
        base = modifier.get("delay_ms", 200)
        # Heal reaction: base + 100-400ms jitter
        return base + random.randint(100, 400)

    def get_attack_delay(self, bot_id: str) -> int:
        """Get attack delay — slight delay before attacking a new target."""
        modifier = self._engine.get_behavior_modifier(bot_id)
        base = modifier.get("delay_ms", 150)
        return base + random.randint(50, 200)

    def get_movement_delay(self, bot_id: str) -> int:
        """Get movement command delay — jitter between movement waypoints."""
        modifier = self._engine.get_behavior_modifier(bot_id)
        base = modifier.get("delay_ms", 100)
        deviation = modifier.get("movement_deviation", {})
        if deviation.get("enabled"):
            base += deviation.get("delay_ms", 0)
        return max(50, base)

    def get_config_push(self, bot_id: str) -> dict[str, str] | None:
        """Compute the config push payload for the bridge.

        Returns a dict of aiSidecar_* config key/value pairs, or None
        if nothing has changed since the last push (to avoid thrashing).

        The bridge receives these as OpenKore config keys and applies
        them to command dispatch.
        """
        now = time.time()
        with self._lock:
            if now - self._last_push_time < _PUSH_COOLDOWN_S:
                return None

        modifier = self._engine.get_behavior_modifier(bot_id)
        delay_ms = modifier.get("delay_ms", 200)
        likeness = modifier.get("human_likeness", 0.5)
        profile = modifier.get("behavior_profile", "active")

        # Build push payload from behavior modifier
        push: dict[str, str] = {
            "aiSidecar_antiDetectionEnabled": "1",
            "aiSidecar_cmdMinDelayMs": str(max(50, int(delay_ms * 0.6))),
            "aiSidecar_cmdMaxDelayMs": str(max(100, int(delay_ms * 1.4))),
            "aiSidecar_reactionTimeMs": str(int(delay_ms)),
            "aiSidecar_humanLikeness": str(round(likeness, 3)),
            "aiSidecar_behaviorProfile": profile,
        }

        # Add fatigue info if available
        state = self._get_bot_state(bot_id)
        fatigue_mult = self._engine._get_fatigue_multiplier(state)
        push["aiSidecar_fatigueMultiplier"] = str(round(fatigue_mult, 2))

        # Only push if changed
        payload_key = f"{bot_id}:{push}"
        with self._lock:
            if payload_key == self._last_push_payload.get(bot_id):
                return None
            self._last_push_payload[bot_id] = payload_key
            self._last_push_time = now

        logger.debug(
            "bridge_wiring: config_push bot=%s profile=%s delay=%dms likeness=%.2f fatigue=%.1f",
            bot_id, profile, delay_ms, likeness, fatigue_mult,
        )
        return push

    def get_current_modifier(self, bot_id: str) -> dict[str, Any]:
        """Get the raw behavior modifier for diagnostics and telemetry."""
        return self._engine.get_behavior_modifier(bot_id)

    def on_gm_detected(self, bot_id: str) -> None:
        """Handle GM detection alert — switch to WATCHING profile.

        Called by game_sense module when a GM character is detected.
        Switches to near-perfect play (minimal mistakes, fast reactions).
        """
        self._engine.set_profile("watching")
        logger.warning(
            "bridge_wiring: GM_DETECTED bot=%s switching to WATCHING profile",
            bot_id,
        )

    def on_gm_clear(self, bot_id: str) -> None:
        """Clear GM detection — resume normal profile cycling."""
        self._engine.set_profile("")
        logger.info(
            "bridge_wiring: GM_CLEARED bot=%s resuming profile cycling",
            bot_id,
        )

    def skill_recast_jitter(self, base_cooldown_ms: int) -> int:
        """Apply ±15% jitter to a skill's re-cast timer.

        Bots execute macros with exact precision, which is a detection signal.
        Humans press buttons slightly late, sometimes early.
        Returns the jittered cooldown in ms.
        """
        if base_cooldown_ms <= 0:
            return 0
        jitter = random.uniform(-0.15, 0.15)
        return max(100, int(base_cooldown_ms * (1.0 + jitter)))

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _get_bot_state(self, bot_id: str) -> dict[str, Any]:
        """Access internal bot state from the BehaviorEngine."""
        # Access the engine's bot state via its internal method
        return self._engine._ensure_bot_state(bot_id)


# ── Global singleton ─────────────────────────────────────────────────────────

_wiring: BridgeWiring | None = None
_wiring_lock = RLock()


def get_bridge_wiring(
    engine: BehaviorEngine | None = None,
) -> BridgeWiring:
    """Get or create the global BridgeWiring singleton."""
    global _wiring
    with _wiring_lock:
        if _wiring is None:
            _wiring = BridgeWiring(behavior_engine=engine)
        return _wiring
