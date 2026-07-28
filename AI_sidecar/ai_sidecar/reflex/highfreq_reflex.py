"""
High-frequency reflex monitor — checks vitals every 50ms, bypasses PDCA cycle.

This runs as a separate asyncio task alongside PDCA, not inside it.
When HP drops below threshold, it injects directly into the action queue
via the bridge — no 5-second PDCA wait.

The LLM reviews reflex effectiveness on kaizen cycles and can adjust
thresholds dynamically based on observed survival patterns.

Healing is dynamically optimized — uses knowledge.json to select the
best potion for the bot's current level, HP deficit, and zeny.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Default reflex thresholds (LLM can override these via kaizen review)
DEFAULT_THRESHOLDS: dict[str, float] = {
    "heal_potion_hp_pct": 0.50,      # Use healing item at 50% HP
    "emergency_potion_hp_pct": 0.30, # Use emergency heal at 30% HP
    "escape_teleport_hp_pct": 0.15,  # Teleport at 15% HP
    "sit_rest_hp_pct": 0.40,         # Sit to rest at 40% HP (out of combat)
    "sit_rest_sp_pct": 0.20,         # Sit to rest at 20% SP (out of combat)
    "aggro_escape_count": 3,         # Escape if 3+ mobs aggro
    "mvp_escape_hp_pct": 0.40,      # Escape from MVP at 40% HP
}


@dataclass(slots=True)
class HighFreqReflex:
    """High-frequency (50ms) vital sign monitor with direct action injection."""
    
    enqueue_fn: Callable[[str, dict[str, Any]], bool] | None = None
    _lock: RLock = field(default_factory=RLock)
    _running: bool = False
    _task: asyncio.Task | None = None
    _thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    _last_hp: dict[str, int] = field(default_factory=dict)
    _last_action_time: dict[str, float] = field(default_factory=dict)
    _cooldown_until: dict[str, float] = field(default_factory=dict)
    _stats: dict[str, int] = field(default_factory=lambda: {"checks": 0, "actions": 0, "misses": 0})
    _llm_adjustments: list[dict[str, Any]] = field(default_factory=list)
    
    # Cooldowns to prevent spam (seconds)
    POTION_COOLDOWN = 2.0
    TELEPORT_COOLDOWN = 10.0
    SIT_COOLDOWN = 5.0
    
    # Reference to healing optimizer (set at init time)
    healing_optimizer: object | None = None
    
    def start(self) -> None:
        """Start the high-frequency monitor in a background asyncio task."""
        with self._lock:
            if self._running:
                return
            self._running = True
        
        async def _monitor_loop():
            logger.info("highfreq_reflex_started: interval=50ms")
            while True:
                try:
                    with self._lock:
                        if not self._running:
                            break
                    await self._tick()
                    await asyncio.sleep(0.05)  # 50ms
                except asyncio.CancelledError:
                    break
                except Exception:
                    await asyncio.sleep(0.1)  # Back off on error
        
        try:
            loop = asyncio.get_event_loop()
            self._task = loop.create_task(_monitor_loop())
        except RuntimeError:
            # No event loop running — create one
            self._task = asyncio.ensure_future(_monitor_loop())
    
    def stop(self) -> None:
        """Stop the high-frequency monitor."""
        with self._lock:
            self._running = False
        if self._task is not None:
            self._task.cancel()
            self._task = None
    
    def update_thresholds(self, thresholds: dict[str, float], reason: str = "manual") -> None:
        """Update reflex thresholds. Called by LLM kaizen review."""
        with self._lock:
            for key, value in thresholds.items():
                if key in self._thresholds:
                    old = self._thresholds[key]
                    self._thresholds[key] = float(value)
                    self._llm_adjustments.append({
                        "key": key,
                        "old": old,
                        "new": float(value),
                        "reason": reason,
                        "timestamp": time.time(),
                    })
                    logger.info("highfreq_reflex_threshold: %s %.2f→%.2f (%s)", key, old, float(value), reason)
    
    def get_thresholds(self) -> dict[str, float]:
        with self._lock:
            return dict(self._thresholds)
    
    def get_adjustments(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._llm_adjustments[-20:])
    
    def get_stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)
    
    async def _tick(self) -> None:
        """Single tick of the high-frequency monitor."""
        with self._lock:
            self._stats["checks"] += 1
    
    def _get_heal_command(self, hp: int, max_hp: int, sp: int, max_sp: int,
                          zeny: int, level: int) -> str | None:
        """Get the best healing command using the healing optimizer.
        
        Returns 'use <Item Name>' or None if no suitable heal found.
        Falls back to reasonable defaults if optimizer isn't loaded.
        """
        if self.healing_optimizer is not None:
            try:
                result = self.healing_optimizer.select_healing_command(
                    hp=hp, max_hp=max_hp, sp=sp, max_sp=max_sp,
                    zeny=zeny, level=level, prefer_hp=True,
                )
                if result:
                    return result
            except Exception as e:
                logger.warning("highfreq_reflex_heal_opt_failed: %s", e)
        
        # Fallback: use level-appropriate defaults
        # COLD_START buys Red Potion (501) — align fallback with what we actually buy
        if level < 30:
            return "use Red Potion"
        elif level < 60:
            return "use Orange Potion"
        elif level < 90:
            return "use White Potion"
        else:
            return "use White Potion"  # Best general-purpose heal
    
    def _get_emergency_heal_command(self, hp: int, max_hp: int, sp: int, max_sp: int,
                                    zeny: int, level: int) -> str | None:
        """Get the best emergency heal command."""
        if self.healing_optimizer is not None:
            try:
                result = self.healing_optimizer.select_healing_command(
                    hp=hp, max_hp=max_hp, sp=sp, max_sp=max_sp,
                    zeny=zeny, level=level, prefer_hp=True,
                )
                if result:
                    return result
            except Exception as e:
                logger.warning("highfreq_reflex_emergency_opt_failed: %s", e)
        
        # Fallback: high-level emergency heal
        if level < 40:
            return "use Orange Potion"
        elif level < 80:
            return "use White Potion"
        else:
            return "use White Potion"
    
    def check_and_act(self, bot_id: str, hp: int, max_hp: int, sp: int, max_sp: int,
                      aggro_count: int, is_dead: bool, is_town: bool,
                      has_potions: bool, current_map: str,
                      zeny: int = 0, level: int = 1,
                      reflex_pipeline: object | None = None) -> str | None:
        """Check vitals and return an action command if needed.
        
        Called from PDCA loop's snapshot processing — NOT from the async task.
        Uses HealingOptimizer to dynamically select the best healing item
        based on the bot's level, HP deficit, and zeny.
        
        If reflex_pipeline is provided, emits through the pipeline instead of
        returning a string — bypassing the arbiter for critical actions.
        """
        now = time.time()
        hp_pct = hp / max_hp if max_hp > 0 else 1.0
        sp_pct = sp / max_sp if max_sp > 0 else 1.0
        
        with self._lock:
            thresholds = dict(self._thresholds)
            cooldown = dict(self._cooldown_until)
        
        # Check cooldown
        if cooldown.get(bot_id, 0) > now:
            return None
        
        # Dead — no actions possible
        if is_dead:
            return None
        
        # Track HP changes for death detection
        prev_hp = self._last_hp.get(bot_id, hp)
        self._last_hp[bot_id] = hp
        
        # ── EMERGENCY: Escape at 15% HP ──
        # Multi-tier escape for all character levels:
        # TIER 1: Sit to regen (works for everyone, no items needed)
        # TIER 2: Run away (move in random direction)
        # TIER 3: Accept death (don't spam teleport that will fail)
        # NOTE: Level 1 novices don't have Teleport or Fly Wings.
        # Sending "ai manual" as escape was causing 100% failure rate.
        if hp_pct <= thresholds.get("escape_teleport_hp_pct", 0.15) and not is_town:
            with self._lock:
                self._cooldown_until[bot_id] = now + self.TELEPORT_COOLDOWN
                self._stats["actions"] += 1
            logger.info("highfreq_reflex: bot=%s escape hp=%.0f%% aggro=%d", bot_id, hp_pct * 100, aggro_count)
            # TIER 1: Sit to regen (always available, no items needed)
            if aggro_count <= 2:
                cmd = "sit"
            # TIER 2: Run away if heavily aggroed
            elif aggro_count <= 5:
                import random
                dirs = ['n', 's', 'e', 'w', 'nw', 'ne', 'sw', 'se']
                cmd = f"move {random.choice(dirs)}"
            # TIER 3: Accept death — don't spam failing teleport
            else:
                cmd = "sit"  # Sit and accept death gracefully
                logger.info("highfreq_reflex: bot=%s accepting_death hp=%.0f%% aggro=%d", bot_id, hp_pct * 100, aggro_count)
            if reflex_pipeline is not None:
                reflex_pipeline.emit_direct(bot_id, cmd)
                return None
            return cmd
        
        # ── EMERGENCY: Healing item at 30% HP ──
        if hp_pct <= thresholds.get("emergency_potion_hp_pct", 0.30) and has_potions:
            with self._lock:
                self._cooldown_until[bot_id] = now + self.POTION_COOLDOWN
                self._stats["actions"] += 1
            cmd = self._get_emergency_heal_command(hp, max_hp, sp, max_sp, zeny, level)  # returns None if no heal available — don't toggle AI mode
            logger.info("highfreq_reflex: bot=%s emergency_heal=%s hp=%.0f%%", bot_id, cmd, hp_pct * 100)
            if reflex_pipeline is not None:
                reflex_pipeline.emit_direct(bot_id, cmd)
                return None
            return cmd
        
        # ── SURVIVAL: Healing item at 50% HP ──
        if hp_pct <= thresholds.get("heal_potion_hp_pct", 0.50):
            if has_potions:
                with self._lock:
                    self._cooldown_until[bot_id] = now + self.POTION_COOLDOWN
                    self._stats["actions"] += 1
                cmd = self._get_heal_command(hp, max_hp, sp, max_sp, zeny, level)
                logger.info("highfreq_reflex: bot=%s heal=%s hp=%.0f%%", bot_id, cmd, hp_pct * 100)
                if reflex_pipeline is not None:
                    reflex_pipeline.emit_direct(bot_id, cmd)
                    return None
                return cmd
            else:
                # No potions available — sit to regen instead of spamming
                with self._lock:
                    self._cooldown_until[bot_id] = now + self.SIT_COOLDOWN
                    self._stats["actions"] += 1
                logger.info("highfreq_reflex: bot=%s no_potions_sit hp=%.0f%%", bot_id, hp_pct * 100)
                cmd = "sit"
                if reflex_pipeline is not None:
                    reflex_pipeline.emit_direct(bot_id, cmd)
                    return None
                return cmd
        
        # ── SURVIVAL: Sit to rest (out of combat) ──
        if aggro_count == 0 and not is_town:
            if hp_pct <= thresholds.get("sit_rest_hp_pct", 0.40) or sp_pct <= thresholds.get("sit_rest_sp_pct", 0.20):
                with self._lock:
                    self._cooldown_until[bot_id] = now + self.SIT_COOLDOWN
                    self._stats["actions"] += 1
                logger.info("highfreq_reflex: bot=%s sit_rest hp=%.0f%% sp=%.0f%%", bot_id, hp_pct * 100, sp_pct * 100)
                cmd = "sit"
                if reflex_pipeline is not None:
                    reflex_pipeline.emit_direct(bot_id, cmd)
                    return None
                return cmd
        
        # ── SP recovery: Use SP healing item when below threshold ──
        if sp_pct <= thresholds.get("sit_rest_sp_pct", 0.20) and has_potions and not is_town and aggro_count == 0:
            with self._lock:
                self._cooldown_until[bot_id] = now + self.POTION_COOLDOWN
                self._stats["actions"] += 1
            cmd = self._get_heal_command(hp, max_hp, sp, max_sp, zeny, level) or "sit"
            logger.info("highfreq_reflex: bot=%s sp_heal=%s sp=%.0f%%", bot_id, cmd, sp_pct * 100)
            if reflex_pipeline is not None:
                reflex_pipeline.emit_direct(bot_id, cmd)
                return None
            return cmd
        
        return None
