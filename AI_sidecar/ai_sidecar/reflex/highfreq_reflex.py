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
    "heal_potion_hp_pct": 0.70,         # Pro RO: heal at 70% HP (not 50%)
    "emergency_potion_hp_pct": 0.50,    # Pro RO: emergency heal at 50% (not 30%)
    "escape_teleport_hp_pct": 0.30,     # Pro RO: flee at 30% HP (not 15% — 15% is obituary)
    "sit_rest_hp_pct": 0.60,            # Sit to rest at 60% HP (out of combat)
    "sit_rest_sp_pct": 0.30,            # Sit to rest at 30% SP (out of combat)
    "aggro_escape_count": 2,            # Pro RO: flee when 2+ mobs aggro (not 3+)
    "mvp_escape_hp_pct": 0.50,          # Escape MVP at 50% HP
}


@dataclass(slots=True)
class HighFreqReflex:
    """High-frequency (50ms) vital sign monitor with direct action injection.
    
    Runs as an independent async task alongside PDCA, not inside it.
    Accesses snapshot cache to get per-bot vitals every 50ms and injects
    survival actions (heal, escape, sit) directly into the action queue
    — no 5-second PDCA wait.
    """
    
    enqueue_fn: Callable[[str, dict[str, Any]], bool] | None = None
    snapshot_cache: Any = None
    bot_registry: Any = None
    reflex_pipeline: Any = None
    integration_bus: Any = None
    _lock: RLock = field(default_factory=RLock)
    _running: bool = False
    _task: asyncio.Task | None = None
    _thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    _last_hp: dict[str, int] = field(default_factory=dict)
    _hp_history: dict[str, list[tuple[float, int]]] = field(default_factory=dict)  # (timestamp, hp) pairs for trend
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
            return list(self._llm_adjustments)
    
    def stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def get_stats(self) -> dict[str, int]:
        """Public stats accessor (alias of stats()) for consistent API surface."""
        with self._lock:
            return dict(self._stats)
    
    async def _tick(self) -> None:
        """Single tick of the high-frequency monitor.
        
        Reads all active bots' snapshots from cache and injects
        survival actions via check_and_act if thresholds are exceeded.
        """
        with self._lock:
            self._stats["checks"] += 1
        
        # Get active bot IDs from registry or fallback to known IDs
        bot_ids: list[str] = []
        _registry = self.bot_registry
        if _registry is not None and hasattr(_registry, 'active_bots'):
            try:
                bot_ids = _registry.active_bots()
            except Exception:
                pass
        elif _registry is not None and hasattr(_registry, '_get_active_bots'):
            try:
                bot_ids = _registry._get_active_bots()
            except Exception:
                pass
        elif _registry is not None and hasattr(_registry, 'get_all'):
            try:
                bot_ids = _registry.get_all()
            except Exception:
                pass
        
        if not bot_ids:
            return
        
        _snap_cache = self.snapshot_cache
        _rflx_pipe = self.reflex_pipeline
        _enqueue = self.enqueue_fn
        
        for bot_id in bot_ids:
            try:
                # Get latest snapshot
                snapshot = None
                if _snap_cache is not None:
                    if hasattr(_snap_cache, 'get'):
                        snapshot = _snap_cache.get(bot_id) if hasattr(_snap_cache, '__call__') else \
                                   getattr(_snap_cache, 'get', lambda x: None)(bot_id)
                    elif hasattr(_snap_cache, '__getitem__'):
                        try:
                            snapshot = _snap_cache[bot_id]
                        except (KeyError, IndexError):
                            pass
                
                if snapshot is None:
                    continue
                
                # Extract vitals from snapshot
                hp = 100; max_hp = 100; sp = 50; max_sp = 80
                is_dead = False; is_town = False; has_potions = False
                zeny = 0; level = 1; aggro_count = 0; current_map = ""
                
                if isinstance(snapshot, dict):
                    hp = int(snapshot.get("hp", snapshot.get("vitals", {}).get("hp", 100)) or 100)
                    max_hp = int(snapshot.get("hp_max", snapshot.get("vitals", {}).get("hp_max", 1)) or 1)
                    sp = int(snapshot.get("sp", snapshot.get("vitals", {}).get("sp", 50)) or 50)
                    max_sp = int(snapshot.get("max_sp", snapshot.get("vitals", {}).get("max_sp", 80)) or 80)
                    is_dead = bool(snapshot.get("is_dead", snapshot.get("vitals", {}).get("is_dead", False)))
                    is_town = bool(snapshot.get("is_town", snapshot.get("vitals", {}).get("is_town", True)))
                    zeny = int(snapshot.get("zeny", snapshot.get("inventory", {}).get("zeny", 0)) or 0)
                    level = int(snapshot.get("base_level", snapshot.get("progression", {}).get("base_level", 1)) or 1)
                    aggro_count = int(snapshot.get("combat", {}).get("aggro_count", snapshot.get("aggro_count", 0)))
                    current_map = str(snapshot.get("map", snapshot.get("position", {}).get("map", "")) or "")
                    inv_items = snapshot.get("inventory_items", snapshot.get("inventory", {}).get("items", []))
                    if isinstance(inv_items, list):
                        for item in inv_items:
                            if isinstance(item, dict) and item.get("name", "") and "potion" in str(item.get("name", "")).lower():
                                has_potions = True
                                break
                else:
                    # BotStateSnapshot object
                    v = getattr(snapshot, "vitals", None)
                    if v:
                        hp = int(getattr(v, "hp", 100) or 100)
                        max_hp = int(getattr(v, "hp_max", 1) or 1)
                        sp = int(getattr(v, "sp", 50) or 50)
                        max_sp = int(getattr(v, "max_sp", 80) or 80)
                    prog = getattr(snapshot, "progression", None)
                    if prog:
                        level = int(getattr(prog, "base_level", 1) or 1)
                    inv_items = getattr(snapshot, "inventory_items", []) or []
                    for item in inv_items:
                        name = getattr(item, "name", "") if not isinstance(item, dict) else item.get("name", "")
                        if name and "potion" in name.lower():
                            has_potions = True
                            break
                    pos = getattr(snapshot, "position", None) or {}
                    if hasattr(pos, "map"):
                        current_map = str(pos.map)
                hp_pct = hp / max_hp if max_hp > 0 else 1.0
                
                # Check if in town (safe zones)
                _town_maps = ["prontera", "izlude", "morocc", "geffen", "payon", 
                             "alberta", "aldebaran", "comodo", "yuno", "amatsu",
                             "gonryun", "umbala", "niflheim", "lighthalzen",
                             "einbroch", "einbech", "hugel", "rachel", "veins"]
                _is_town = any(m in current_map.lower() for m in _town_maps) if current_map else True
                
                # Call check_and_act with extracted data
                action = self.check_and_act(
                    bot_id=bot_id,
                    hp=hp, max_hp=max_hp,
                    sp=sp, max_sp=max_sp,
                    aggro_count=aggro_count,
                    is_dead=is_dead,
                    is_town=_is_town,
                    has_potions=has_potions,
                    current_map=current_map,
                    zeny=zeny, level=level,
                    reflex_pipeline=_rflx_pipe,
                )
                
                if action and _enqueue:
                    _enqueue(bot_id, {
                        "action": action,
                        "source": "highfreq_reflex",
                        "bot_id": bot_id,
                    })
            except Exception:
                continue
    
    def _get_heal_command(self, hp: int, max_hp: int, sp: int, max_sp: int,
                          zeny: int, level: int) -> str | None:
        """Get the best healing command using the healing optimizer.
        
        Returns 'use <Item Name>' or None if no suitable heal found.
        Falls back to reasonable defaults if optimizer isn't loaded.
        Returns None when no potions are available — caller handles sit/return.
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
        # NOTE: This returns a command even if the item isn't in inventory.
        # The caller (check_and_act) gates on has_potions before calling this.
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
        
        # ── PREDICTIVE DANGER: HP trend analysis across 50ms ticks ──
        # Track HP changes over recent ticks to detect rapid drops
        _now_ts = now
        _history = self._hp_history.setdefault(bot_id, [])
        _history.append((_now_ts, hp))
        # Keep last 20 ticks (1 second of 50ms data)
        if len(_history) > 20:
            _history.pop(0)
        
        _predictive_flee = False
        _predictive_reason = ""
        if len(_history) >= 5:
            # Calculate HP slope over last 5 ticks (250ms window)
            _h5 = _history[-5:]
            _hp_deltas = [_h5[i+1][1] - _h5[i][1] for i in range(len(_h5)-1)]
            _avg_drop_per_tick = -sum(d for d in _hp_deltas if d < 0) / len(_hp_deltas)
            _drop_rate_per_second = _avg_drop_per_tick * 20  # 20 ticks/sec = 50ms
            # If dropping faster than 25% HP/sec, flee predictively
            if _avg_drop_per_tick > 0 and hp_pct < 0.70:
                _time_to_30pct = ((hp_pct - 0.30) * max_hp) / max(_avg_drop_per_tick * 20, 1)
                if _time_to_30pct < 2.0:  # Will hit 30% in under 2 seconds
                    _predictive_flee = True
                    _predictive_reason = f"HP dropping {_drop_rate_per_second:.0f}%/sec, will hit 30% in {_time_to_30pct:.1f}s"
        
        # ── EMERGENCY: Healing item takes priority over fleeing ──
        # A Pro player heals when potions exist and HP is recoverable
        # (> critical). Running away with potions in hand is a mistake.
        # Escape only fires when: no potions available, OR HP is critical
        # (≤ 15% — a heal won't land before death), OR predictive flee.
        _critical_hp = hp_pct <= thresholds.get("critical_hp_pct", 0.15)
        _heal_cmd: str | None = None
        if has_potions and hp_pct <= thresholds.get("emergency_potion_hp_pct", 0.30) and not _critical_hp:
            _heal_cmd = self._get_emergency_heal_command(hp, max_hp, sp, max_sp, zeny, level)
            if _heal_cmd:
                with self._lock:
                    self._cooldown_until[bot_id] = now + self.POTION_COOLDOWN
                    self._stats["actions"] += 1
                logger.info("highfreq_reflex: bot=%s emergency_heal=%s hp=%.0f%%", bot_id, _heal_cmd, hp_pct * 100)
                if reflex_pipeline is not None:
                    reflex_pipeline.emit_direct(bot_id, _heal_cmd)
                    return None
                return _heal_cmd

        # ── EMERGENCY: Escape at threshold ──
        # Multi-tier escape for all character levels:
        # TIER 0: 'ai manual' (level ≥ 10 — real escape, disengages auto-AI
        #         so the bot can flee/teleport; verified working at level 10+)
        # TIER 1: Sit to regen (works for everyone, no items needed)
        # TIER 2: Run away (move in random direction)
        # TIER 3: Accept death (don't spam teleport that will fail)
        # NOTE: Level 1 novices don't have Teleport or Fly Wings.
        # Sending "ai manual" as escape for level-1 was causing 100% failure rate.
        if ((hp_pct <= thresholds.get("escape_teleport_hp_pct", 0.30) and _heal_cmd is None) or _predictive_flee or _critical_hp) and not is_town:
            with self._lock:
                self._cooldown_until[bot_id] = now + self.TELEPORT_COOLDOWN
                self._stats["actions"] += 1
            logger.info("highfreq_reflex: bot=%s escape hp=%.0f%% aggro=%d%s", bot_id, hp_pct * 100, aggro_count, f' PREDICTIVE:{_predictive_reason}' if _predictive_flee else '')
            # TIER 0: Real escape for capable levels
            if level >= 10:
                # Never 'ai manual' (freezes auto-attack forever). attackAuto 0 stops
                # combat so the bot can flee/teleport while staying navigable;
                # auto-attack re-enables on arrival at safety.
                cmd = "attackAuto 0"
            # TIER 1: Sit to regen (always available, no items needed)
            elif aggro_count <= 2:
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
