"""
Dynamic healing optimizer — selects the best healing item based on context.

A pro player doesn't always use Red Potion. They choose:
- Red Potion (heals 45-65) for low level, cheap
- Orange Potion (heals 100-140) for mid level
- White Potion (heals 300-400) for high level
- Blue Potion (heals 40-60 SP) when SP is low
- Yggdrasil Berry (full heal) for emergencies
- Condensed/Concentrated variants for efficiency

This module queries knowledge.json for ALL healing items and selects
the optimal one based on the bot's current state.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

# Path to knowledge.json (resolved relative to this file)
_KNOWLEDGE_PATH = Path(__file__).resolve().parent.parent.parent.parent / "knowledge" / "knowledge.json"


@dataclass
class HealingItem:
    """A healing item with parsed stats."""
    id: int
    name: str
    aegis_name: str
    item_type: str
    buy: int
    weight: int
    heal_hp_min: int
    heal_hp_max: int
    heal_sp_min: int
    heal_sp_max: int
    heal_percent_hp: float  # 0.0 if not percentage-based
    heal_percent_sp: float
    level_required: int  # Some items have level requirements


@dataclass(slots=True)
class HealingOptimizer:
    """Selects the best healing item for a bot's current state."""
    
    _lock: RLock = field(default_factory=RLock)
    _healing_items: list[HealingItem] = field(default_factory=list)
    _loaded: bool = False
    _load_error: str = ""
    _stats: dict[str, int] = field(default_factory=lambda: {"lookups": 0, "cache_hits": 0})
    _cache: dict[str, str] = field(default_factory=dict)  # cache_key -> item_name
    
    def load(self) -> bool:
        """Load healing items from knowledge.json."""
        if self._loaded:
            return True
        try:
            path = _KNOWLEDGE_PATH
            if not path.exists():
                self._load_error = f"knowledge.json not found at {path}"
                logger.warning("healing_optimizer_no_knowledge: %s", self._load_error)
                self._loaded = True  # Don't retry every cycle
                return False
            
            with open(path) as f:
                data = json.load(f)
            
            items = data.get("items", {}).get("all", [])
            if not items:
                items = data.get("items", {}).get("items", [])
            if not items:
                # Try flat structure
                for key in ("items", "item_db", "all_items"):
                    if isinstance(data.get(key), list):
                        items = data[key]
                        break
            
            count = 0
            for item in items:
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("Type", "") or "")
                name = str(item.get("Name", "") or "")
                aegis = str(item.get("AegisName", "") or "")
                
                # Only process Healing type items
                if item_type not in ("Healing", "Usable", "Usable_Delayed", "DelayConsume"):
                    continue
                
                script = str(item.get("Script", "") or "")
                if not script or "itemheal" not in script.lower():
                    continue
                
                parsed = self._parse_itemheal(script)
                if parsed is None:
                    continue
                
                heal_hp_min, heal_hp_max, heal_sp_min, heal_sp_max, percent_hp, percent_sp = parsed
                
                # Skip items that heal nothing
                if heal_hp_min <= 0 and heal_sp_min <= 0 and percent_hp <= 0 and percent_sp <= 0:
                    continue
                
                buy = int(item.get("Buy", 0) or 0)
                weight = int(item.get("Weight", 0) or 0)
                
                self._healing_items.append(HealingItem(
                    id=int(item.get("Id", 0) or 0),
                    name=name,
                    aegis_name=aegis,
                    item_type=item_type,
                    buy=buy,
                    weight=weight,
                    heal_hp_min=heal_hp_min,
                    heal_hp_max=heal_hp_max,
                    heal_sp_min=heal_sp_min,
                    heal_sp_max=heal_sp_max,
                    heal_percent_hp=percent_hp,
                    heal_percent_sp=percent_sp,
                    level_required=0,
                ))
                count += 1
            
            # Sort by healing efficiency (HP per zeny)
            self._healing_items.sort(key=lambda h: (
                -(h.heal_hp_max / max(h.buy, 1)),
                -h.heal_hp_max,
            ))
            
            logger.info("healing_optimizer_loaded: %d healing items from %s", count, path)
            self._loaded = True
            return count > 0
            
        except Exception as e:
            self._load_error = str(e)
            logger.warning("healing_optimizer_load_failed: %s", e)
            self._loaded = True
            return False
    
    def select_healing_command(
        self,
        *,
        hp: int,
        max_hp: int,
        sp: int,
        max_sp: int,
        zeny: int,
        level: int,
        prefer_hp: bool = True,
    ) -> str | None:
        """Select the best healing item and return the command to use it.
        
        Returns None if no suitable healing item is available.
        """
        self._lock.acquire()
        try:
            if not self._loaded:
                self.load()
            
            if not self._healing_items:
                self._lock.release()
                return None
            
            self._stats["lookups"] += 1
            
            hp_deficit = max_hp - hp
            sp_deficit = max_sp - sp
            hp_ratio = hp / max(max_hp, 1)
            sp_ratio = sp / max(max_sp, 1)
            
            # Check cache first (same state = same recommendation)
            cache_key = f"{hp_deficit // 50},{sp_deficit // 30},{zeny // 1000},{level}"
            cached = self._cache.get(cache_key)
            if cached:
                self._stats["cache_hits"] += 1
                self._lock.release()
                return f"use {cached}"
            
            best_item: HealingItem | None = None
            best_score = -1.0
            
            for item in self._healing_items:
                # Skip items that are too expensive
                if item.buy > 0 and item.buy > zeny * 0.3:
                    continue
                
                # Calculate effective heal
                avg_hp_heal = (item.heal_hp_min + item.heal_hp_max) / 2
                avg_sp_heal = (item.heal_sp_min + item.heal_sp_max) / 2
                pct_hp_heal = item.heal_percent_hp * max_hp
                pct_sp_heal = item.heal_percent_sp * max_sp
                
                total_hp_heal = avg_hp_heal + pct_hp_heal
                total_sp_heal = avg_sp_heal + pct_sp_heal
                
                # Score based on what we need
                if prefer_hp and hp_ratio < 0.5:
                    # Need HP — score by heal amount vs deficit (don't over-heal)
                    if total_hp_heal > 0:
                        over_heal = max(0, total_hp_heal - hp_deficit)
                        efficiency = total_hp_heal / max(item.buy, 1) if item.buy > 0 else total_hp_heal
                        score = total_hp_heal * efficiency - over_heal * 2
                    else:
                        score = 0
                elif sp_ratio < 0.3:
                    # Need SP — score by SP heal
                    score = total_sp_heal / max(item.buy, 1) if item.buy > 0 else total_sp_heal
                else:
                    # Maintenance — score by efficiency
                    score = (total_hp_heal + total_sp_heal) / max(item.buy, 1) if item.buy > 0 else (total_hp_heal + total_sp_heal)
                
                if score > best_score:
                    best_score = score
                    best_item = item
            
            self._lock.release()
            
            if best_item is None:
                return None
            
            # Cache the result
            self._cache[cache_key] = best_item.name
            # Prune cache if too large
            if len(self._cache) > 1000:
                self._cache.clear()
            
            return f"use {best_item.name}"
        
        except Exception as e:
            self._lock.release()
            logger.warning("healing_optimizer_select_failed: %s", e)
            return None
    
    def select_sp_healing_command(
        self, *, hp: int, max_hp: int, sp: int, max_sp: int, zeny: int, level: int
    ) -> str | None:
        """Select the best SP healing item."""
        return self.select_healing_command(
            hp=hp, max_hp=max_hp, sp=sp, max_sp=max_sp,
            zeny=zeny, level=level, prefer_hp=False,
        )
    
    def _parse_itemheal(self, script: str) -> tuple[int, int, int, int, float, float] | None:
        """Parse 'itemheal rand(45,65),0;' into (hp_min, hp_max, sp_min, sp_max, hp_pct, sp_pct)."""
        try:
            # Match: itemheal rand(45,65),0; or itemheal rand(100,200),rand(20,30);
            # Or percentage-based: itemheal 300,0;
            # Remove whitespace
            script = script.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
            
            # Match itemheal(...) pattern
            match = re.search(r'itemheal\(?([^;)]*)\)?', script)
            if not match:
                return None
            
            args_str = match.group(1)
            # Split by comma
            parts = [p.strip() for p in args_str.split(",")]
            
            hp_min, hp_max = 0, 0
            sp_min, sp_max = 0, 0
            hp_pct, sp_pct = 0.0, 0.0
            
            if len(parts) >= 1:
                hp_part = parts[0]
                # Check if it's a rand() expression
                rand_match = re.match(r'rand\((\d+),(\d+)\)', hp_part)
                if rand_match:
                    hp_min = int(rand_match.group(1))
                    hp_max = int(rand_match.group(2))
                else:
                    try:
                        val = float(hp_part)
                        if val < 100:  # Arbitrary threshold: small values = flat heal
                            hp_min = hp_max = int(val)
                        else:
                            hp_pct = val / 100.0
                    except ValueError:
                        pass
            
            if len(parts) >= 2:
                sp_part = parts[1]
                rand_match = re.match(r'rand\((\d+),(\d+)\)', sp_part)
                if rand_match:
                    sp_min = int(rand_match.group(1))
                    sp_max = int(rand_match.group(2))
                else:
                    try:
                        val = float(sp_part)
                        if val < 100:
                            sp_min = sp_max = int(val)
                        else:
                            sp_pct = val / 100.0
                    except ValueError:
                        pass
            
            return (hp_min, hp_max, sp_min, sp_max, hp_pct, sp_pct)
            
        except Exception:
            return None
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# Global instance
_optimizer: HealingOptimizer | None = None
_optimizer_lock = RLock()


def get_healing_optimizer() -> HealingOptimizer:
    """Get or create the global healing optimizer instance."""
    global _optimizer
    with _optimizer_lock:
        if _optimizer is None:
            _optimizer = HealingOptimizer()
            _optimizer.load()
        return _optimizer


def select_best_heal_command(
    hp: int, max_hp: int, sp: int, max_sp: int, zeny: int, level: int
) -> str | None:
    """Convenience function to select the best healing command."""
    opt = get_healing_optimizer()
    return opt.select_healing_command(
        hp=hp, max_hp=max_hp, sp=sp, max_sp=max_sp,
        zeny=zeny, level=level,
    )
